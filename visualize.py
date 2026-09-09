"""
visualize.py
============
Visualization utilities for LULC segmentation results.

What this file does:
    - Plot S2 RGB composite + ground truth + prediction + error map
    - Plot confusion matrix heatmap
    - Plot training/validation loss and mIoU curves

Uses fixed BigEarthNet 19-class colors matching the reference code,
so every visualization is consistent across the entire project.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# BigEarthNet CLC → 19-class mapping
# Mirrors the reference code exactly so raw TIFF values are always
# converted the same way regardless of which module calls the mapping.
# ---------------------------------------------------------------------------

# Raw CLC ID → BigEarthNet class name
CLC_TO_CLASS_NAME: dict = {
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

# BigEarthNet 19-class names in canonical order (index = train ID)
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

# Train ID → class name lookup
TRAIN_ID_TO_NAME: dict = {i: n for i, n in enumerate(BIGEARTHNET_CLASS_NAMES)}

# CLC raw ID → train ID (built from above two dicts)
CLC_TO_TRAIN_ID: dict = {
    raw: BIGEARTHNET_CLASS_NAMES.index(name)
    for raw, name in CLC_TO_CLASS_NAME.items()
    if name in BIGEARTHNET_CLASS_NAMES
}

# Fixed colors per class (index matches BIGEARTHNET_CLASS_NAMES)
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


# ---------------------------------------------------------------------------
# Color utilities
# ---------------------------------------------------------------------------

def hex_to_rgb(hex_color: str) -> tuple:
    """Convert '#rrggbb' → (R, G, B) floats in [0, 1]."""
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))


def build_color_map(
    class_colors: List[str],
    ignore_index: int = 255,
    ignore_color: tuple = (20, 20, 20),   # very dark grey for ignored pixels
) -> np.ndarray:
    """
    Build a [256, 3] uint8 lookup table mapping train IDs → RGB.

    Index 0..N-1  → class colors
    ignore_index  → dark grey (distinguishable from pure black water/shadow)
    """
    cmap = np.zeros((256, 3), dtype=np.uint8)
    for i, hex_c in enumerate(class_colors):
        r, g, b = hex_to_rgb(hex_c)
        cmap[i] = [int(r * 255), int(g * 255), int(b * 255)]
    cmap[ignore_index % 256] = ignore_color
    return cmap


def mask_to_rgb(mask: np.ndarray, color_map: np.ndarray) -> np.ndarray:
    """Integer mask [H, W] → RGB image [H, W, 3] via color_map lookup."""
    return color_map[np.clip(mask.astype(np.int32), 0, 255)]


def build_listed_cmap(class_colors: List[str]) -> ListedColormap:
    """Build a matplotlib ListedColormap for imshow/colorbar use."""
    return ListedColormap(class_colors)


# ---------------------------------------------------------------------------
# S2 band → display RGB
# ---------------------------------------------------------------------------

def s2_to_rgb(
    s2: torch.Tensor,
    norm_mean: Optional[List] = None,
    norm_std: Optional[List] = None,
) -> np.ndarray:
    """
    Convert normalized S2 tensor [12, H, W] → display RGB [H, W, 3].

    Uses B04 (index 3) = red, B03 (index 2) = green, B02 (index 1) = blue.
    Denormalizes if mean/std provided, stretches reflectance to [0, 1]
    and applies sqrt gamma so dark areas remain visible.
    """
    arr = s2.cpu().numpy().astype(np.float32)  # [12, H, W]

    if norm_mean is not None and norm_std is not None:
        mean = np.array(norm_mean, dtype=np.float32).reshape(-1, 1, 1)
        std  = np.array(norm_std,  dtype=np.float32).reshape(-1, 1, 1)
        arr  = arr * std + mean  # back to reflectance

    # B04=idx3 (R), B03=idx2 (G), B02=idx1 (B)
    rgb = np.stack([arr[3], arr[2], arr[1]], axis=-1)  # [H, W, 3]
    rgb = np.clip(rgb, 0.0, 0.3) / 0.3                # stretch to visible range
    rgb = np.power(np.clip(rgb, 0, 1), 0.5)            # sqrt gamma
    return rgb.astype(np.float32)


# ---------------------------------------------------------------------------
# Qualitative sample plot  (4-panel)
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
    Save a 4-panel PNG:
        Panel 1 – S2 natural-color composite
        Panel 2 – Ground truth (colorized by class)
        Panel 3 – Model prediction (same colorization)
        Panel 4 – Error map  (white=correct, red=wrong, dark grey=ignored)

    Ground truth and prediction use a matplotlib ListedColormap so colors
    match the reference code exactly. The legend shows only classes that
    actually appear in this patch (not all 19) to keep it readable.

    Args:
        s2:           [12, H, W] float32 S2 tensor (normalized).
        gt_mask:      [H, W] int64 ground-truth tensor (train IDs or ignore).
        pred_mask:    [H, W] int64 prediction tensor.
        class_names:  19 class name strings.
        class_colors: 19 hex color strings (same index order as class_names).
        ignore_index: Value to treat as "no data" (default 255).
        save_path:    Where to write the PNG.
        norm_mean/std: S2 normalization stats for denormalization.
    """
    # Build color assets
    cmap_list  = build_listed_cmap(class_colors)          # for imshow
    color_map  = build_color_map(class_colors, ignore_index)

    rgb      = s2_to_rgb(s2, norm_mean, norm_std)
    gt_np    = gt_mask.numpy().astype(np.int32)
    pred_np  = pred_mask.numpy().astype(np.int32)

    # Replace ignore pixels with NaN-equivalent for imshow (show as grey)
    def _to_display(arr):
        """Return float array with ignore pixels set to -1 (outside cmap range)."""
        out = arr.astype(np.float32)
        out[arr == ignore_index] = -1
        return out

    gt_display   = _to_display(gt_np)
    pred_display = _to_display(pred_np)

    # Error map
    valid   = gt_np != ignore_index
    correct = (gt_np == pred_np) & valid
    error   = np.zeros((*gt_np.shape, 3), dtype=np.uint8)
    error[valid & correct]  = [255, 255, 255]   # correct  → white
    error[valid & ~correct] = [220,  50,  50]   # wrong    → red
    error[~valid]           = [20,   20,  20]   # ignored  → dark grey

    # ---- Plot --------------------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(22, 6))

    axes[0].imshow(rgb)
    axes[0].set_title("S2 RGB", fontsize=12)
    axes[0].axis("off")

    im_gt = axes[1].imshow(gt_display, cmap=cmap_list, vmin=0, vmax=len(class_names) - 1,
                           interpolation="nearest")
    axes[1].set_title("Ground Truth", fontsize=12)
    axes[1].axis("off")

    axes[2].imshow(pred_display, cmap=cmap_list, vmin=0, vmax=len(class_names) - 1,
                   interpolation="nearest")
    axes[2].set_title("Prediction", fontsize=12)
    axes[2].axis("off")

    axes[3].imshow(error)
    axes[3].set_title("Error Map", fontsize=12)
    axes[3].axis("off")

    # Legend: only classes present in this patch
    present_ids = set(np.unique(gt_np[valid]).tolist())
    legend_patches = []
    for i, (name, color) in enumerate(zip(class_names, class_colors)):
        if i in present_ids:
            r, g, b = hex_to_rgb(color)
            legend_patches.append(
                mpatches.Patch(facecolor=(r, g, b), edgecolor="black",
                               label=f"{i}: {name}")
            )
    # Always add ignored entry
    legend_patches.append(
        mpatches.Patch(facecolor=(20/255, 20/255, 20/255), edgecolor="black",
                       label="255: No data / Ignored")
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
    plt.close(fig)


# ---------------------------------------------------------------------------
# Standalone reference-map viewer  (mirrors reference code exactly)
# ---------------------------------------------------------------------------

def plot_reference_map(
    tif_path: str,
    raw_to_train: Optional[dict] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Visualize a raw reference TIFF using the BigEarthNet colormap.

    Mirrors the reference diagnostic code so the output looks identical.
    If raw_to_train is None, CLC_TO_TRAIN_ID is used.

    Args:
        tif_path:     Path to the reference TIFF file.
        raw_to_train: Optional override mapping raw IDs → train IDs.
        save_path:    Where to save the PNG (shown interactively if None).
    """
    import rasterio

    mapping = raw_to_train or CLC_TO_TRAIN_ID

    with rasterio.open(tif_path) as src:
        data   = src.read(1)
        nodata = src.nodata

    # Convert raw CLC IDs → train IDs
    segmentation = np.full(data.shape, len(BIGEARTHNET_CLASS_NAMES), dtype=np.int32)
    for raw_id, train_id in mapping.items():
        segmentation[data == raw_id] = train_id
    if nodata is not None:
        segmentation[data == int(nodata)] = len(BIGEARTHNET_CLASS_NAMES)  # background

    # Add a background/nodata color
    all_colors = BIGEARTHNET_COLORS + ["#d9d9d9"]   # grey for nodata
    cmap = ListedColormap(all_colors)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(segmentation, cmap=cmap,
              vmin=0, vmax=len(all_colors) - 1,
              interpolation="nearest")
    ax.set_title(os.path.basename(tif_path), fontsize=10)
    ax.axis("off")

    legend_items = [
        mpatches.Patch(facecolor=BIGEARTHNET_COLORS[i], edgecolor="black",
                       label=f"{i}: {name}")
        for i, name in enumerate(BIGEARTHNET_CLASS_NAMES)
        if i in np.unique(segmentation)
    ]
    legend_items.append(
        mpatches.Patch(facecolor="#d9d9d9", edgecolor="black", label="No data")
    )
    ax.legend(handles=legend_items, bbox_to_anchor=(1.02, 1),
              loc="upper left", fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        logger.info(f"Reference map plot saved: {save_path}")
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
    ax.set_xticks(range(n));  ax.set_yticks(range(n))
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
    """Plot train/val loss and val mIoU over epochs."""
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

from __future__ import annotations

import logging
import os
from typing import List, Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Color utilities
# ---------------------------------------------------------------------------

def hex_to_rgb(hex_color: str) -> tuple:
    """Convert hex color string to (R, G, B) tuple in [0, 1]."""
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))


def build_color_map(class_colors: List[str], ignore_index: int = 255) -> np.ndarray:
    """
    Build a [256, 3] uint8 color lookup table.

    Index 0..N-1 map to class colors.
    Index ignore_index maps to black (0, 0, 0).
    """
    color_map = np.zeros((256, 3), dtype=np.uint8)
    for i, hex_c in enumerate(class_colors):
        r, g, b = hex_to_rgb(hex_c)
        color_map[i] = [int(r*255), int(g*255), int(b*255)]
    color_map[ignore_index % 256] = [0, 0, 0]
    return color_map


def mask_to_rgb(mask: np.ndarray, color_map: np.ndarray) -> np.ndarray:
    """
    Convert an integer mask [H, W] to an RGB image [H, W, 3].

    Args:
        mask:      [H, W] integer array with values 0..N-1 or ignore_index.
        color_map: [256, 3] uint8 color lookup.

    Returns:
        RGB image [H, W, 3] uint8.
    """
    mask_clipped = np.clip(mask.astype(np.int32), 0, 255)
    return color_map[mask_clipped]


# ---------------------------------------------------------------------------
# S2 band visualization helper
# ---------------------------------------------------------------------------

def s2_to_rgb(s2: torch.Tensor, norm_mean: Optional[List] = None,
              norm_std: Optional[List] = None) -> np.ndarray:
    """
    Convert a normalized S2 tensor to a display-ready RGB image.

    Uses bands B04 (red, index 3), B03 (green, index 2), B02 (blue, index 1).

    Steps:
        1. Denormalize using training-set mean/std if provided.
        2. Extract RGB channels.
        3. Clip to [0, 1] reflectance.
        4. Apply gamma correction (γ=2.2→display) for visibility.

    Returns:
        [H, W, 3] float32 array in [0, 1].
    """
    arr = s2.cpu().numpy().astype(np.float32)  # [12, H, W]

    if norm_mean is not None and norm_std is not None:
        mean = np.array(norm_mean, dtype=np.float32).reshape(12, 1, 1)
        std  = np.array(norm_std,  dtype=np.float32).reshape(12, 1, 1)
        arr = arr * std + mean  # denormalize to reflectance

    # B04=index3 (R), B03=index2 (G), B02=index1 (B)
    rgb = np.stack([arr[3], arr[2], arr[1]], axis=-1)  # [H, W, 3]

    # Clip and stretch to visible range (reflectance 0–0.3 is typical)
    rgb = np.clip(rgb, 0.0, 0.3) / 0.3
    # Mild gamma for display
    rgb = np.power(np.clip(rgb, 0, 1), 0.5)
    return rgb.astype(np.float32)


# ---------------------------------------------------------------------------
# Qualitative sample plot
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
    Plot a 4-panel figure: RGB composite | Ground truth | Prediction | Error map.

    Error map highlights:
        - Correct pixels: white
        - Wrong predictions: red
        - Ignored pixels: black

    Args:
        s2:           [12, H, W] tensor.
        gt_mask:      [H, W] integer tensor.
        pred_mask:    [H, W] integer tensor.
        class_names:  List of class name strings.
        class_colors: Hex color strings.
        ignore_index: Label to treat as ignore.
        save_path:    Where to save the figure.
        norm_mean/std: Used to denormalize for display.
    """
    color_map = build_color_map(class_colors, ignore_index)

    rgb = s2_to_rgb(s2, norm_mean, norm_std)
    gt_np   = gt_mask.numpy().astype(np.int32)
    pred_np = pred_mask.numpy().astype(np.int32)

    gt_rgb   = mask_to_rgb(gt_np,   color_map)
    pred_rgb = mask_to_rgb(pred_np, color_map)

    # Error map
    valid = gt_np != ignore_index
    correct = (gt_np == pred_np) & valid
    error_map = np.zeros((*gt_np.shape, 3), dtype=np.uint8)
    error_map[valid & correct]  = [255, 255, 255]   # correct: white
    error_map[valid & ~correct] = [220,  50,  50]   # wrong:   red
    # ignored pixels stay black (0, 0, 0)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    titles = ["S2 RGB", "Ground Truth", "Prediction", "Error Map"]
    images = [rgb, gt_rgb, pred_rgb, error_map]

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=12)
        ax.axis("off")

    # Legend
    legend_patches = []
    for i, (name, color) in enumerate(zip(class_names, class_colors)):
        r, g, b = hex_to_rgb(color)
        legend_patches.append(
            mpatches.Patch(facecolor=(r, g, b), label=f"{i}: {name}")
        )
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=min(len(class_names), 10),
        fontsize=7,
        frameon=True,
        bbox_to_anchor=(0.5, -0.05),
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        logger.info(f"Qualitative figure saved: {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Confusion matrix plot
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: str,
) -> None:
    """
    Plot and save a normalized confusion matrix heatmap.

    Rows = ground truth, columns = predicted.
    Values are normalized by row (recall per class).
    """
    # Normalize by row (true class total)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(
        cm.astype(float), row_sums,
        out=np.zeros_like(cm, dtype=float),
        where=row_sums > 0,
    )

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    n = len(class_names)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    short = [f"{i}" for i in range(n)]
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(short, fontsize=8)
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

def plot_training_curves(
    history_csv: str,
    save_path: str,
) -> None:
    """
    Plot training loss, validation loss, and validation mIoU over epochs.

    Args:
        history_csv: Path to training_history.csv.
        save_path:   Where to save the PNG.
    """
    import pandas as pd
    df = pd.read_csv(history_csv)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    axes[0].plot(df["epoch"], df["train_loss"], label="Train Loss", color="tab:blue")
    axes[0].plot(df["epoch"], df["val_loss"],   label="Val Loss",   color="tab:orange")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # mIoU plot
    axes[1].plot(df["epoch"], df["val_miou"], label="Val mIoU", color="tab:green")
    if "val_dice" in df.columns:
        axes[1].plot(df["epoch"], df["val_dice"], label="Val Dice", color="tab:red", linestyle="--")
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
