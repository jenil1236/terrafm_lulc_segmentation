"""
metrics.py
==========
Semantic segmentation evaluation metrics.

What this file does:
    Computes mIoU, per-class IoU, Dice/F1, pixel accuracy, precision,
    recall, and a full confusion matrix. Handles ignore_index correctly
    (ignored pixels never contribute to any metric).

What goes in:
    - Predicted class tensors [B, H, W] or accumulated confusion matrix
    - Ground-truth tensors    [B, H, W]

What comes out:
    - Dict of scalar metrics
    - Per-class metrics table (as a pandas DataFrame)
    - Confusion matrix (numpy array)

How it connects:
    train.py and evaluate.py call SegmentationMetrics.update() per batch,
    then .compute() at the end of an epoch.

WHY mIoU and not just pixel accuracy:
    Pixel accuracy is dominated by frequent classes (e.g., a background
    class covering 80% of pixels → accuracy=80% by predicting only background).
    mIoU averages the overlap ratio per class, giving equal weight to rare
    and common classes. It is the standard metric for semantic segmentation
    and the best single number to track during training.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from config import CFG

logger = logging.getLogger(__name__)


class SegmentationMetrics:
    """
    Online metric accumulator for semantic segmentation.

    Usage:
        metrics = SegmentationMetrics(num_classes=19)
        for preds, targets in loader:
            metrics.update(preds, targets)
        results = metrics.compute()
        metrics.reset()

    Args:
        num_classes:  Number of valid classes (excluding ignore_index).
        ignore_index: Value in targets to ignore.
        class_names:  Optional list of human-readable class names.
    """

    def __init__(
        self,
        num_classes: int = 19,
        ignore_index: int = 255,
        class_names: Optional[List[str]] = None,
    ):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        self._conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def reset(self) -> None:
        """Clear accumulated statistics."""
        self._conf_matrix[:] = 0

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Accumulate predictions into the confusion matrix.

        Args:
            preds:   [B, H, W] or [H, W] int64 predicted class IDs.
            targets: [B, H, W] or [H, W] int64 ground-truth class IDs.
        """
        preds = preds.cpu().numpy().astype(np.int64).ravel()
        targets = targets.cpu().numpy().astype(np.int64).ravel()

        # Mask out ignore pixels
        valid = targets != self.ignore_index
        preds = preds[valid]
        targets = targets[valid]

        # Clamp predictions to valid range (model might rarely predict OOB)
        preds = np.clip(preds, 0, self.num_classes - 1)
        targets = np.clip(targets, 0, self.num_classes - 1)

        # Accumulate into confusion matrix
        # conf[i, j] = number of pixels with true class i predicted as class j
        np.add.at(
            self._conf_matrix,
            (targets, preds),
            1,
        )

    def compute(self) -> Dict:
        """
        Compute all metrics from the accumulated confusion matrix.

        Returns:
            dict with keys:
                mean_iou, mean_dice, pixel_acc,
                per_class_iou, per_class_dice, per_class_precision,
                per_class_recall, per_class_pixel_count,
                confusion_matrix
        """
        cm = self._conf_matrix.astype(np.float64)
        n = self.num_classes

        # Diagonal = true positives per class
        tp = np.diag(cm)

        # Per-class: total predicted as class i (column sum)
        pred_sum = cm.sum(axis=0)
        # Per-class: total true class i pixels (row sum)
        true_sum = cm.sum(axis=1)

        # IoU = TP / (TP + FP + FN) = TP / (pred_sum + true_sum - TP)
        union = pred_sum + true_sum - tp
        iou = np.where(union > 0, tp / np.maximum(union, 1e-10), np.nan)

        # Dice / F1 = 2TP / (2TP + FP + FN) = 2TP / (pred_sum + true_sum)
        denom = pred_sum + true_sum
        dice = np.where(denom > 0, 2.0 * tp / np.maximum(denom, 1e-10), np.nan)

        # Precision = TP / (TP + FP) = TP / pred_sum
        precision = np.where(pred_sum > 0, tp / np.maximum(pred_sum, 1e-10), np.nan)

        # Recall = TP / (TP + FN) = TP / true_sum
        recall = np.where(true_sum > 0, tp / np.maximum(true_sum, 1e-10), np.nan)

        # Pixel accuracy = sum(TP) / total valid pixels
        total_pixels = cm.sum()
        pixel_acc = tp.sum() / max(total_pixels, 1)

        # Mean IoU: average over classes that actually appear in ground truth
        valid_classes = true_sum > 0
        mean_iou = float(np.nanmean(iou[valid_classes])) if valid_classes.any() else 0.0
        mean_dice = float(np.nanmean(dice[valid_classes])) if valid_classes.any() else 0.0

        # Frequency-weighted IoU
        freq = true_sum / max(total_pixels, 1)
        fw_iou = float(np.nansum(freq * np.nan_to_num(iou)))

        return {
            "mean_iou": mean_iou,
            "mean_dice": mean_dice,
            "pixel_accuracy": float(pixel_acc),
            "freq_weighted_iou": fw_iou,
            "per_class_iou": iou.tolist(),
            "per_class_dice": dice.tolist(),
            "per_class_precision": precision.tolist(),
            "per_class_recall": recall.tolist(),
            "per_class_pixel_count": true_sum.astype(int).tolist(),
            "confusion_matrix": self._conf_matrix.copy(),
            "num_valid_classes": int(valid_classes.sum()),
        }

    def per_class_table(self) -> pd.DataFrame:
        """
        Return per-class metrics as a nicely formatted DataFrame.

        Columns: Class | IoU | Dice | Precision | Recall | Pixel Count
        """
        results = self.compute()
        rows = []
        for i, name in enumerate(self.class_names):
            rows.append({
                "Class ID": i,
                "Class Name": name,
                "IoU": f"{results['per_class_iou'][i]:.4f}" if not np.isnan(results['per_class_iou'][i]) else "N/A",
                "Dice": f"{results['per_class_dice'][i]:.4f}" if not np.isnan(results['per_class_dice'][i]) else "N/A",
                "Precision": f"{results['per_class_precision'][i]:.4f}" if not np.isnan(results['per_class_precision'][i]) else "N/A",
                "Recall": f"{results['per_class_recall'][i]:.4f}" if not np.isnan(results['per_class_recall'][i]) else "N/A",
                "Pixel Count": results["per_class_pixel_count"][i],
            })
        df = pd.DataFrame(rows)
        return df

    @property
    def confusion_matrix(self) -> np.ndarray:
        return self._conf_matrix.copy()
