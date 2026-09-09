"""
visualize.py
============
Visualization utilities for BigEarthNet 19-class LULC segmentation.

Uses fixed BigEarthNet class names and colors (matching the reference
diagnostic script) so every visualization is consistent.

Key design:
  - GT and prediction masks are colorized via a uint8 lookup table,
    NOT via matplotlib colormaps with vmin/vmax (avoids the -1 clipping bug).
  - ignore_index=255 pixels render as dark grey, not black, so they are
    visually distinct from dark-colored classes like water.
  - Legend shows only the classes actually present in the patch.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# BigEarthNet CLC → 19-class canonical mapping
# ---------------------------------------------------------------------------

# Raw CLC ID → class name (matches reference script exactly)
CLC_TO_CLASS_NAME: Dict[int, str] = {
    111: "Urban fabric",
    112: "Urban fabric",
    121: "Industrial or commercial units",
    211: "Arable land",
    212: "Arable land",
    213: "Arable land",
    221: "Permanent crops",
    222: "Permanent crops",
    223: "Permanent crops",
    231: "Pastures",
    241: "Permanent crops",
    242: "Complex cultivation patterns",
    243: "Land principally occupied by agriculture",
    244: "Agro-forestry areas",
    311: "Broad-leaved forest",
    312: "Coniferous forest",
    313: "Mixed forest",
    321: "Natural grassland and sparsely vegetated areas",
    322: "Moors, heathland and sclerophyllous vegetation",
    323: "Moors, heathland and sclerophyllous vegetation",
    324: "Transitional woodland, shrub",
    331: "Beaches, dunes, sands",
    333: "Natural grassland and sparsely vegetated areas",
    411: "Inland wetlands",
    412: "Inland wetlands",
    421: "Coastal wetlands",
    422: "Coastal wetlands",
    511: "Inland waters",
    512: "Inland waters",
    521: "Marine waters",
    522: "Marine waters",
    523: "Marine waters",
}

# Canonical 19-class list — index = train ID (0-indexed)
BIGEARTHNET_CLASS_NAMES: List[str] = [
    "Urban fabric",                                           # 0
    "Industrial or commercial units",                         # 1
    "Arable land",                                            # 2
    "Permanent crops",                                        # 3
    "Pastures",                                               # 4
    "Complex cultivation patterns",                           # 5
    "Land principally occupied by agriculture",               # 6
    "Agro-forestry areas",                                    # 7
    "Broad-leaved forest",                                    # 8
    "Coniferous forest",                                      # 9
    "Mixed forest",                                           # 10
    "Natural grassland and sparsely vegetated areas",         # 11
    "Moors, heathland and sclerophyllous vegetation",         # 12
    "Transitional woodland, shrub",                           # 13
    "Beaches, dunes, sands",                                  # 14
    "Inland wetlands",                                        # 15
    "Coastal wetlands",                                       # 16
    "Inland waters",                                          # 17
    "Marine waters",                                          # 18
]

# Fixed colors per class — same hex values as reference script
BIGEARTHNET_COLORS: List[str] = [
    "#e41a1c",  # 0  Urban fabric
    "#984ea3",  # 1  Industrial or commercial units
    "#ffd92f",  # 2  Arable land
    "#ffad33",  # 3  Permanent crops
    "#78c679",  # 4  Pastures
    "#a6d854",  # 5  Complex cultivation patterns
    "#fdae61",  # 6  Land principally occupied by agriculture
    "#66c2a5",  # 7  Agro-forestry areas
    "#006d2c",  # 8  Broad-leaved forest
    "#238b45",  # 9  Coniferous forest
    "#31a354",  # 10 Mixed forest
    "#8c96c6",  # 11 Natural grassland
    "#9e9ac8",  # 12 Moors, heathland
    "#8856a7",  # 13 Transitional woodland, shrub
    "#fdd49e",  # 14 Beaches, dunes, sands
    "#74c476",  # 15 Inland wetlands
    "#41ae76",  # 16 Coastal wetlands
    "#2171b5",  # 17 Inland waters
    "#6baed6",  # 18 Marine waters
]

# CLC raw ID → train ID  (derived from the two dicts above)
CLC_TO_TRAIN_ID: Dict[int, int] = {
    raw: BIGEARTHNET_CLASS_NAMES.index(name)
    for raw, name in CLC_TO_CLASS_NAME.items()
    if name in BIGEARTHNET_CLASS_NAMES
}


# ---------------------------------------------------------------------------
# Color utilities
# ---------------------------------------------------------------------------

def hex_to_rgb(hex_color: str) -> tuple:
    """'#rrggbb' → (R, G, B) floats in [0, 1]."""
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def build_color_map(
    class_colors: List[str],
    ignore_index: int = 255,
    ignore_color: tuple = (30, 30, 30),   # dark grey for ignored pixels
) -> np.ndarray:
    """
    Build a [256, 3] uint8 lookup table: train_id → RGB.

    Indices 0..N-1 → class colors.
    ignore_index   → dark grey (distinguishable from dark classes like water).
    """
    cmap = np.zeros((256, 3), dtype=np.uint8)
    for i, hex_c in enumerate(class_colors):
        r, g, b = hex_to_rgb(hex_c)
        cmap[i] = [int(r * 255), int(g * 255), int(b * 255)]
    cmap[ignore_index % 256] = list(ignore_color)
    return cmap


def mask_to_rgb(mask: np.ndarray, color_map: np.ndarray) -> np.ndarray:
    """
    Integer mask [H, W] → RGB image [H, W, 3] via direct lookup.

    Uses the color_map array as a lookup table — no matplotlib normalization,
    so there is no risk of -1 or out-of-range values corrupting the colors.
    """
    return color_map[np.clip(mask.astype(np.int32), 0, 255)]


# ---------------------------------------------------------------------------
# S2 → display RGB
# ---------------------------------------------------------------------------

def s2_to_rgb(
    s2: torch.Tensor,
    norm_mean: Optional[List] = None,
    norm_std: Optional[List] = None,
) -> np.ndarray:
    """
    Normalized S2 tensor [12, H, W] → display RGB [H, W, 3] float32.

    Uses B04 (index 3) = red, B03 (index 2) = green, B02 (index 1) = blue.
    Denormalizes if mean/std provided. Stretches to typical reflectance
    range [0, 0.3] and applies sqrt gamma for visibility.
    """
    arr = s2.cpu().numpy().astype(np.float32)  # [12, H, W]

    if norm_mean is not None and norm_std is not None:
        mean = np.array(norm_mean, dtype=np.float32).reshape(-1, 1, 1)
        std  = np.array(norm_std,  dtype=np.float32).reshape(-1, 1, 1)
        arr  = arr * std + mean   # undo z-normalization → reflectance

    # Natural colour: R=B04, G=B03, B=B02
    rgb = np.stack([arr[3], arr[2], arr[1]], axis=-1)  # [H, W, 3]
    rgb = np.clip(rgb, 0.0, 0.3) / 0.3                # stretch to [0,1]
    rgb = np.power(np.clip(rgb, 0, 1), 0.5)            # sqrt gamma
    return rgb.astype(np.float32)


# ---------------------------------------------------------------------------
# 4-panel qualitative sample plot
# ---------------------------------------------------------------------------

def plot_qualitative_sample(
    s2: torch.Tensor,
    gt_mask: torch.Tensor,
    pred_mask: torch.Tensor,
    class_names: List[str],
    class_colors: List[str],
    ignore_index: int = 255,
    save_path: Optional[str] = None,
    norm_mean: Optional[List] = None,
    norm_std: Optional[List] = None,
) -> None:
    """
    Save a 4-panel PNG per patch:
        Panel 1 – S2 natural-colour composite
        Panel 2 – Ground truth  (class colors; ignored pixels = dark grey)
        Panel 3 – Prediction    (same colorization)
        Panel 4 – Error map     (white=correct, red=wrong, dark grey=ignored)

    Colorization uses direct uint8 lookup — NOT matplotlib imshow with
    vmin/vmax, which would silently clip out-of-range values to the first
    or last color and corrupt the display.

    Args:
        s2:           [12, H, W] normalized S2 tensor.
        gt_mask:      [H, W] int64 tensor with train IDs (0–18) or ignore_index.
        pred_mask:    [H, W] int64 tensor with model predictions.
        class_names:  19 class name strings.
        class_colors: 19 hex color strings (same index order as class_names).
        ignore_index: Pixels with this value are shown as dark grey.
        save_path:    Output PNG path. If None the figure is shown interactively.
        norm_mean/std: S2 normalization stats for denormalization.
    """
    color_map = build_color_map(class_colors, ignore_index)

    rgb      = s2_to_rgb(s2, norm_mean, norm_std)         # [H,W,3] float32
    gt_np    = gt_mask.numpy().astype(np.int32)
    pred_np  = pred_mask.numpy().astype(np.int32)

    # Direct lookup — correct colors guaranteed regardless of value range
    gt_rgb   = mask_to_rgb(gt_np,   color_map)            # [H,W,3] uint8
    pred_rgb = mask_to_rgb(pred_np, color_map)            # [H,W,3] uint8

    # Error map
    valid   = gt_np != ignore_index
    correct = (gt_np == pred_np) & valid
    error   = np.full((*gt_np.shape, 3), 30, dtype=np.uint8)  # dark grey base
    error[valid & correct]  = [255, 255, 255]   # correct  → white
    error[valid & ~correct] = [220,  50,  50]   # wrong    → red

    # ---- figure ------------------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(22, 6))

    for ax, img, title in zip(
        axes,
        [rgb, gt_rgb, pred_rgb, error],
        ["S2 RGB", "Ground Truth", "Prediction", "Error Map"],
    ):
        ax.imshow(img)
        ax.set_title(title, fontsize=12)
        ax.axis("off")

    # Legend: only classes present in this patch + nodata entry
    present_ids = set(int(v) for v in np.unique(gt_np[valid]))
    legend_patches = []
    for i, (name, color) in enumerate(zip(class_names, class_colors)):
        if i in present_ids:
            r, g, b = hex_to_rgb(color)
            legend_patches.append(
                mpatches.Patch(facecolor=(r, g, b), edgecolor="black",
                               label=f"{i}: {name}")
            )
    legend_patches.append(
        mpatches.Patch(facecolor=(30/255, 30/255, 30/255), edgecolor="black",
                       label="No data / Ignored")
    )

    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=min(len(legend_patches), 5),
        fontsize=8,
        frameon=True,
        bbox_to_anchor=(0.5, -0.04),
    )

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        logger.info(f"Qualitative sample saved: {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Standalone reference-map viewer (mirrors reference diagnostic script)
# ---------------------------------------------------------------------------

def plot_reference_map(
    tif_path: str,
    raw_to_train: Optional[Dict[int, int]] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Visualize a raw reference TIFF with BigEarthNet colors.

    Reads the raw CLC IDs, converts them to train IDs, and displays
    with the same colors as plot_qualitative_sample.
    Pixels with value 0 (nodata) are shown as light grey.

    Args:
        tif_path:     Path to reference TIFF.
        raw_to_train: Override CLC→train_id mapping (default: CLC_TO_TRAIN_ID).
        save_path:    Where to save. Interactive display if None.
    """
    import rasterio

    mapping = raw_to_train or CLC_TO_TRAIN_ID
    color_map = build_color_map(BIGEARTHNET_COLORS, ignore_index=255)
    # Nodata (value 0 in TIFF) → light grey in the color map
    color_map[0] = [200, 200, 200]

    with rasterio.open(tif_path) as src:
        raw = src.read(1).astype(np.int32)

    # Convert CLC IDs → train IDs; unmapped → 255 (shown as dark grey)
    out = np.full(raw.shape, 255, dtype=np.uint8)
    for raw_id, train_id in mapping.items():
        out[raw == raw_id] = train_id

    rgb = mask_to_rgb(out, color_map)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(rgb)
    ax.set_title(os.path.basename(tif_path), fontsize=9)
    ax.axis("off")

    # Legend: present classes only
    present = set(int(v) for v in np.unique(out) if v != 255)
    patches = [
        mpatches.Patch(
            facecolor=tuple(c / 255 for c in color_map[i]),
            edgecolor="black",
            label=f"{i}: {BIGEARTHNET_CLASS_NAMES[i]}",
        )
        for i in sorted(present)
    ]
    patches.append(
        mpatches.Patch(facecolor=(200/255,)*3, edgecolor="black", label="No data")
    )
    ax.legend(handles=patches, bbox_to_anchor=(1.02, 1),
              loc="upper left", fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: str,
) -> None:
    """Normalized confusion matrix heatmap (rows = true, cols = predicted)."""
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm  = np.divide(cm.astype(float), row_sums,
                         out=np.zeros_like(cm, dtype=float),
                         where=row_sums > 0)

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    n = len(class_names)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([str(i) for i in range(n)], rotation=45,
                       ha="right", fontsize=8)
    ax.set_yticklabels([str(i) for i in range(n)], fontsize=8)
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")
    ax.set_title("Normalized Confusion Matrix (recall per row)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.info(f"Confusion matrix saved: {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------

def plot_training_curves(history_csv: str, save_path: str) -> None:
    """Plot train/val loss and val mIoU over epochs from training_history CSV."""
    import pandas as pd

    df = pd.read_csv(history_csv)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(df["epoch"], df["train_loss"], label="Train Loss", color="tab:blue")
    axes[0].plot(df["epoch"], df["val_loss"],   label="Val Loss",   color="tab:orange")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df["epoch"], df["val_miou"], label="Val mIoU", color="tab:green")
    if "val_dice" in df.columns:
        axes[1].plot(df["epoch"], df["val_dice"], label="Val Dice",
                     color="tab:red", linestyle="--")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Validation Metrics")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Training Progress")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.info(f"Training curves saved: {save_path}")
    plt.close(fig)
