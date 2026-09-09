"""
visualize.py
============
Visualization utilities for LULC segmentation results.

What this file does:
    Provides functions to:
      - Plot S2 RGB composite + ground truth + prediction + error map
      - Plot confusion matrix heatmap
      - Plot training/validation loss and mIoU curves
      - Save class color legends

What goes in:
    - Tensors from dataset or model output
    - Metric results from metrics.py

What comes out:
    - PNG files saved to results directory

How it connects:
    evaluate.py and the Colab notebook call these functions.

DESIGN PRINCIPLE:
    Color assignments are deterministic and saved to config.
    Every visualization uses the same class→color mapping so that
    ground-truth, prediction, and legend are always consistent.
"""

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
