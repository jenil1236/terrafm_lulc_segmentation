"""
visualize.py
============
Visualization utilities for BigEarthNet 19-class LULC segmentation.

Key design decisions:
  - GT and prediction are colorized via direct uint8 LUT (no matplotlib
    vmin/vmax, which caused the -1 clipping / all-red bug).
  - nodata pixels (ignore_index=255) are shown as light grey #d9d9d9 so
    they are clearly distinguishable from real classes.
  - plot_reference_map() reads a raw 120x120 reference TIFF, converts
    CLC IDs → train IDs, and displays exactly like the reference script.
  - All 19 classes always appear in the full legend; per-patch legend
    shows only present classes.
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
# BigEarthNet CLC → 19-class mapping  (matches reference script exactly)
# ---------------------------------------------------------------------------

CLC_TO_CLASS_NAME: Dict[int, str] = {
    111: "Urban fabric",
    112: "Urban fabric",
    121: "Industrial or commercial units",
    122: "Unlabeled",
    123: "Unlabeled",
    131: "Unlabeled",
    132: "Unlabeled",
    133: "Unlabeled",
    141: "Unlabeled",
    142: "Unlabeled",
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
    332: "Unlabeled",
    333: "Natural grassland and sparsely vegetated areas",
    334: "Unlabeled",
    335: "Unlabeled",
    411: "Inland wetlands",
    412: "Inland wetlands",
    421: "Coastal wetlands",
    422: "Coastal wetlands",
    423: "Unlabeled",
    511: "Inland waters",
    512: "Inland waters",
    521: "Marine waters",
    522: "Marine waters",
    523: "Marine waters",
    999: "Unlabeled",
}

# Canonical 19-class list — index IS the train ID
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

# Fixed colors — same hex values as reference script
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

NODATA_COLOR: str = "#d9d9d9"   # light grey for nodata/ignored pixels

# CLC raw ID → train ID
# "Unlabeled" CLC codes are intentionally excluded — they map to
# ignore_index=255 at preprocessing time, never to a train ID.
CLC_TO_TRAIN_ID: Dict[int, int] = {
    raw: BIGEARTHNET_CLASS_NAMES.index(name)
    for raw, name in CLC_TO_CLASS_NAME.items()
    if name in BIGEARTHNET_CLASS_NAMES and name != "Unlabeled"
}


# ---------------------------------------------------------------------------
# Color utilities
# ---------------------------------------------------------------------------

def hex_to_rgb(hex_color: str) -> tuple:
    """'#rrggbb' → (R, G, B) floats in [0, 1]."""
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def _hex_to_uint8(hex_color: str) -> list:
    h = hex_color.lstrip("#")
    return [int(h[i:i + 2], 16) for i in (0, 2, 4)]


def build_color_map(
    class_colors: List[str],
    ignore_index: int = 255,
) -> np.ndarray:
    """
    Build a [256, 3] uint8 lookup table: train_id → RGB.

    This is ONLY ever called with already-remapped train IDs (0–18) or
    ignore_index (255). Raw CLC IDs like 311, 511 never reach this function —
    they have already been converted to 0–18 or 255 by read_reference_mask()
    in preprocessing.py before any visualization happens.

    Indices 0..N-1  → class colors.
    ignore_index    → NODATA_COLOR (light grey).
    All others      → NODATA_COLOR (safe default for any unexpected value).
    """
    nd = _hex_to_uint8(NODATA_COLOR)
    cmap = np.tile(np.array(nd, dtype=np.uint8), (256, 1))
    for i, hex_c in enumerate(class_colors):
        if i < 256:
            cmap[i] = _hex_to_uint8(hex_c)
    return cmap


def mask_to_rgb(mask: np.ndarray, color_map: np.ndarray) -> np.ndarray:
    """
    Integer mask [H, W] → RGB [H, W, 3] uint8 via direct lookup.
    No matplotlib normalization — value 9 → color_map[9] exactly.
    """
    return color_map[np.clip(mask.astype(np.int32), 0, 255)]


# ---------------------------------------------------------------------------
# S2 natural-colour display
# ---------------------------------------------------------------------------

def s2_to_rgb(
    s2: torch.Tensor,
    norm_mean: Optional[List] = None,
    norm_std: Optional[List] = None,
) -> np.ndarray:
    """
    Normalized S2 tensor [12, H, W] → display RGB [H, W, 3] float32.

    Uses B04 (index 3) = red, B03 (index 2) = green, B02 (index 1) = blue.
    Denormalizes if stats provided. Stretches reflectance [0, 0.3] → [0, 1]
    and applies sqrt gamma so dark areas are visible.
    """
    arr = s2.cpu().numpy().astype(np.float32)

    if norm_mean is not None and norm_std is not None:
        mean = np.array(norm_mean[:12], dtype=np.float32).reshape(-1, 1, 1)
        std  = np.array(norm_std[:12],  dtype=np.float32).reshape(-1, 1, 1)
        arr  = arr * std + mean

    rgb = np.stack([arr[3], arr[2], arr[1]], axis=-1)  # R=B04, G=B03, B=B02
    rgb = np.clip(rgb, 0.0, 0.3) / 0.3
    rgb = np.power(np.clip(rgb, 0, 1), 0.5)
    return rgb.astype(np.float32)


# ---------------------------------------------------------------------------
# Reference-map viewer  (mirrors reference script, works on raw 120×120 TIFFs)
# ---------------------------------------------------------------------------

def plot_reference_map(
    tif_path: str,
    raw_to_train: Optional[Dict[int, int]] = None,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
) -> None:
    """
    Display a raw reference TIFF (120×120, uint16 CLC IDs) with
    BigEarthNet colors.  Mirrors the reference diagnostic script exactly.

    Conversion pipeline:
        raw uint16 CLC ID  →  train ID (0-18) via raw_to_train
        pixels not in mapping  →  shown as NODATA_COLOR (light grey)

    nodata value (0 in the TIFF) is treated as background and shown grey.
    All 19 classes appear in the legend; present classes are marked with *.

    Args:
        tif_path:     Path to reference TIFF.
        raw_to_train: CLC→train_id mapping. Defaults to CLC_TO_TRAIN_ID.
        save_path:    Save PNG here. Interactive display if None.
        title:        Figure title. Defaults to filename.
    """
    import rasterio

    mapping   = raw_to_train if raw_to_train is not None else CLC_TO_TRAIN_ID
    color_map = build_color_map(BIGEARTHNET_COLORS, ignore_index=255)

    with rasterio.open(tif_path) as src:
        raw    = src.read(1).astype(np.int32)
        nodata = src.nodata

    # Convert CLC IDs → train IDs; unmapped (incl nodata) → 255
    seg = np.full(raw.shape, 255, dtype=np.uint8)
    for raw_id, train_id in mapping.items():
        seg[raw == raw_id] = train_id
    if nodata is not None:
        seg[raw == int(nodata)] = 255

    rgb = mask_to_rgb(seg, color_map)   # [H, W, 3] uint8

    # Which classes are present
    present = set(int(v) for v in np.unique(seg) if v != 255)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(rgb, interpolation="nearest")
    ax.set_title(title or os.path.basename(tif_path), fontsize=9)
    ax.axis("off")

    # Full 19-class legend; present classes marked with *
    patches = []
    for i, (name, color) in enumerate(zip(BIGEARTHNET_CLASS_NAMES, BIGEARTHNET_COLORS)):
        marker = "* " if i in present else "  "
        patches.append(
            mpatches.Patch(
                facecolor=hex_to_rgb(color),
                edgecolor="black",
                linewidth=0.5,
                label=f"{marker}{i}: {name}",
            )
        )

    fig.legend(
        handles=patches,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=7.5,
        frameon=True,
        title="* = present in this patch",
        title_fontsize=7,
    )

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        logger.info(f"Reference map saved: {save_path}")
    else:
        plt.show()
    plt.close(fig)


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
    4-panel PNG for one patch: S2 RGB | Ground Truth | Prediction | Error Map.

    Panel colors use direct uint8 LUT lookup — no matplotlib normalization.
    Legend shows all 19 classes; present classes in ground truth are marked *.
    nodata/ignored pixels are shown as light grey in GT and prediction panels.

    Error map:
        white      = correct prediction (gt == pred, both valid)
        red        = wrong prediction   (gt != pred, both valid)
        light grey = no data / ignored  (gt == ignore_index)
    """
    color_map = build_color_map(class_colors, ignore_index)

    rgb     = s2_to_rgb(s2, norm_mean, norm_std)
    gt_np   = gt_mask.numpy().astype(np.int32)
    pred_np = pred_mask.numpy().astype(np.int32)

    gt_rgb   = mask_to_rgb(gt_np,   color_map)   # [H,W,3] uint8
    pred_rgb = mask_to_rgb(pred_np, color_map)

    # Error map
    nd_rgb  = _hex_to_uint8(NODATA_COLOR)
    valid   = gt_np != ignore_index
    correct = (gt_np == pred_np) & valid
    error   = np.tile(np.array(nd_rgb, dtype=np.uint8), (*gt_np.shape, 1))
    error[valid & correct]  = [255, 255, 255]
    error[valid & ~correct] = [220,  50,  50]

    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    for ax, img, ttl in zip(
        axes,
        [rgb, gt_rgb, pred_rgb, error],
        ["S2 RGB", "Ground Truth", "Prediction", "Error Map"],
    ):
        ax.imshow(img, interpolation="nearest")
        ax.set_title(ttl, fontsize=12)
        ax.axis("off")

    # Legend — all 19 classes; present in GT marked with *
    present = set(int(v) for v in np.unique(gt_np[valid]))
    patches = []
    for i, (name, color) in enumerate(zip(class_names, class_colors)):
        marker = "* " if i in present else "  "
        patches.append(
            mpatches.Patch(
                facecolor=hex_to_rgb(color),
                edgecolor="black",
                linewidth=0.5,
                label=f"{marker}{i}: {name}",
            )
        )
    patches.append(
        mpatches.Patch(
            facecolor=hex_to_rgb(NODATA_COLOR),
            edgecolor="black",
            linewidth=0.5,
            label="No data / Ignored",
        )
    )

    fig.legend(
        handles=patches,
        loc="lower center",
        ncol=5,
        fontsize=7.5,
        frameon=True,
        bbox_to_anchor=(0.5, -0.18),
        title="* = present in ground truth",
        title_fontsize=7,
    )

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        logger.info(f"Sample saved: {save_path}")
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
    """Row-normalized confusion matrix heatmap (rows=true, cols=predicted)."""
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm  = np.divide(cm.astype(float), row_sums,
                         out=np.zeros_like(cm, dtype=float), where=row_sums > 0)

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    n = len(class_names)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels([str(i) for i in range(n)], rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels([str(i) for i in range(n)], fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Normalized Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.info(f"Confusion matrix saved: {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------

def plot_training_curves(history_csv: str, save_path: str) -> None:
    """Train/val loss and val mIoU over epochs."""
    import pandas as pd
    df = pd.read_csv(history_csv)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(df["epoch"], df["train_loss"], label="Train Loss", color="tab:blue")
    axes[0].plot(df["epoch"], df["val_loss"],   label="Val Loss",   color="tab:orange")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(df["epoch"], df["val_miou"], label="Val mIoU", color="tab:green")
    if "val_dice" in df.columns:
        axes[1].plot(df["epoch"], df["val_dice"], label="Val Dice",
                     color="tab:red", linestyle="--")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Score")
    axes[1].set_title("Validation Metrics"); axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.suptitle("Training Progress")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.info(f"Training curves saved: {save_path}")
    plt.close(fig)
