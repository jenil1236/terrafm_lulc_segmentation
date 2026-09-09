"""
losses.py
=========
Loss functions for 19-class LULC segmentation with class imbalance handling.

What this file does:
    Provides CE+Dice combined loss (and optional Focal+Dice variant),
    class-weight computation from pixel frequency statistics, and
    safe handling of the ignore_index.

What goes in:
    - Logits [B, C, H, W] and targets [B, H, W]
    - Class pixel frequency statistics

What comes out:
    - Scalar loss tensor for backpropagation

How it connects:
    train.py instantiates SegmentationLoss and calls it each step.

LOSS CHOICE RATIONALE:
-------------------------------------------------------------------
Pure Cross-Entropy:
  - Standard, numerically stable.
  - Pixel accuracy dominated by frequent classes.
  - Rare classes get very small gradients.

CE + Dice:
  - CE handles per-pixel classification probability.
  - Dice directly optimizes the overlap metric (IoU proxy).
  - Together they balance per-pixel and set-level objectives.
  - Dice loss naturally handles class imbalance by normalizing
    by the predicted + ground truth counts per class.
  CHOSEN as default (loss_type = "ce_dice").

Focal + Dice:
  - Focal loss down-weights easy examples (frequent correct predictions).
  - Useful if CE+Dice still under-performs on rare classes.
  - Fallback option (loss_type = "focal_dice").

Class weighting:
  - Inverse frequency weighting with capping.
  - We cap at max_class_weight=10 to prevent extreme weights for very
    rare classes, which can destabilize training (large gradient spikes).
  - A gentler approach: sqrt(inverse-frequency) is also provided.
-------------------------------------------------------------------
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import CFG

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Class weight computation
# ---------------------------------------------------------------------------

def compute_class_weights(
    pixel_counts: Dict[int, int],
    num_classes: int = 19,
    max_weight: float = 10.0,
    method: str = "inv_freq",
) -> torch.Tensor:
    """
    Compute per-class loss weights from pixel count statistics.

    Args:
        pixel_counts: dict {class_id: pixel_count} from training set.
        num_classes:  Total number of classes.
        max_weight:   Cap weight to prevent instability for rare classes.
        method:       "inv_freq" or "inv_sqrt_freq".

    Returns:
        weights: torch.Tensor [num_classes] float32.

    WHY we cap weights:
        Inverse frequency of a class with 100 pixels vs 1M pixels gives
        weight=10000. Backpropagating with such extreme weights destabilizes
        training (exploding gradients even with clipping). Capping at 10
        preserves the signal without the instability.
    """
    counts = torch.zeros(num_classes, dtype=torch.float64)
    for cls_id, cnt in pixel_counts.items():
        if 0 <= cls_id < num_classes:
            counts[cls_id] = float(cnt)

    # Replace zeros with 1 to avoid division by zero for unseen classes
    counts = counts.clamp(min=1.0)
    total = counts.sum()

    if method == "inv_sqrt_freq":
        # sqrt dampens extreme ratios
        freq = counts / total
        weights = 1.0 / torch.sqrt(freq)
    else:
        # Standard inverse-frequency
        freq = counts / total
        weights = 1.0 / freq

    # Normalize so the median weight = 1 (maintains LR scale)
    median_w = weights.median()
    weights = weights / median_w.clamp(min=1e-6)

    # Cap
    weights = weights.clamp(max=max_weight)

    logger.info(
        f"Class weights (method={method}, cap={max_weight}):\n"
        + "\n".join(
            f"  class {i}: count={int(counts[i])}, weight={weights[i]:.3f}"
            for i in range(num_classes)
        )
    )
    return weights.float()


# ---------------------------------------------------------------------------
# Dice loss
# ---------------------------------------------------------------------------

class DiceLoss(nn.Module):
    """
    Soft Dice loss for multi-class segmentation.

    Operates on softmax probabilities, not logits.
    Ignores pixels with label == ignore_index.

    Dice = 2 * |P ∩ T| / (|P| + |T|)
    Loss = 1 - mean(Dice per class)

    Args:
        smooth:       Small constant for numerical stability.
        ignore_index: Class label to ignore.
    """

    def __init__(self, smooth: float = 1.0, ignore_index: int = 255):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  [B, C, H, W] raw logits.
            targets: [B, H, W]    integer class labels.

        Returns:
            Scalar Dice loss.
        """
        num_classes = logits.shape[1]
        B, C, H, W = logits.shape

        probs = F.softmax(logits, dim=1)  # [B, C, H, W]

        # Build valid pixel mask
        valid = (targets != self.ignore_index)  # [B, H, W]

        # One-hot encode targets, setting ignore pixels to background
        targets_clamped = targets.clone()
        targets_clamped[~valid] = 0  # temporarily, will be masked out
        one_hot = F.one_hot(targets_clamped, num_classes=num_classes)  # [B, H, W, C]
        one_hot = one_hot.permute(0, 3, 1, 2).float()  # [B, C, H, W]

        # Zero out ignore pixels
        valid_expanded = valid.unsqueeze(1).float()  # [B, 1, H, W]
        probs_masked = probs * valid_expanded
        one_hot_masked = one_hot * valid_expanded

        # Compute Dice per class
        intersection = (probs_masked * one_hot_masked).sum(dim=(0, 2, 3))  # [C]
        pred_sum = probs_masked.sum(dim=(0, 2, 3))
        true_sum = one_hot_masked.sum(dim=(0, 2, 3))

        dice_per_class = (2.0 * intersection + self.smooth) / (pred_sum + true_sum + self.smooth)
        dice_loss = 1.0 - dice_per_class.mean()
        return dice_loss


# ---------------------------------------------------------------------------
# Focal loss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """
    Focal loss for multi-class segmentation.
    Down-weights easy (well-classified) pixels, focuses on hard ones.

    FL(p) = -alpha * (1 - p)^gamma * log(p)

    Args:
        gamma:        Focusing parameter (2.0 is standard).
        weight:       Per-class weight tensor [C].
        ignore_index: Class label to ignore.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        weight: Optional[torch.Tensor] = None,
        ignore_index: int = 255,
    ):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight)
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)  # [B, C, H, W]
        probs = torch.exp(log_probs)

        # Standard CE: -log(p_y) per pixel
        ce_loss = F.nll_loss(
            log_probs, targets,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction="none",
        )  # [B, H, W]

        # Gather p_y for valid pixels
        valid = targets != self.ignore_index
        targets_clamped = targets.clone()
        targets_clamped[~valid] = 0
        p_t = probs.gather(dim=1, index=targets_clamped.unsqueeze(1)).squeeze(1)

        focal_factor = (1.0 - p_t) ** self.gamma
        focal_loss = focal_factor * ce_loss
        return focal_loss[valid].mean() if valid.any() else focal_loss.mean()


# ---------------------------------------------------------------------------
# Combined segmentation loss
# ---------------------------------------------------------------------------

class SegmentationLoss(nn.Module):
    """
    Combined loss for 19-class LULC segmentation.

    Modes:
        "ce":          Cross-Entropy only.
        "ce_dice":     α·CE + β·Dice  (default, recommended).
        "focal_dice":  α·Focal + β·Dice (for severe class imbalance).

    Args:
        loss_type:    One of "ce", "ce_dice", "focal_dice".
        class_weights: Per-class weight tensor [C] (from training statistics).
        ce_weight:    Weight of CE/Focal component.
        dice_weight:  Weight of Dice component.
        focal_gamma:  Gamma for Focal loss.
        ignore_index: Label value to ignore.
    """

    def __init__(
        self,
        loss_type: str = "ce_dice",
        class_weights: Optional[torch.Tensor] = None,
        ce_weight: float = 0.5,
        dice_weight: float = 0.5,
        focal_gamma: float = 2.0,
        ignore_index: int = 255,
    ):
        super().__init__()
        self.loss_type = loss_type
        self.ce_w = ce_weight
        self.dice_w = dice_weight
        self.ignore_index = ignore_index

        # CE loss
        self.ce_loss = nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=ignore_index,
            reduction="mean",
            label_smoothing=0.05,  # mild label smoothing helps with noisy masks
        )

        # Dice loss
        self.dice_loss = DiceLoss(smooth=1.0, ignore_index=ignore_index)

        # Focal loss (optional)
        self.focal_loss = FocalLoss(
            gamma=focal_gamma,
            weight=class_weights,
            ignore_index=ignore_index,
        ) if "focal" in loss_type else None

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            logits:  [B, C, H, W] raw model output.
            targets: [B, H, W]    integer targets (0..C-1 or ignore_index).

        Returns:
            Combined scalar loss.
        """
        if logits.shape[-2:] != targets.shape[-2:]:
            logits = F.interpolate(
                logits, size=targets.shape[-2:],
                mode="bilinear", align_corners=False
            )

        if self.loss_type == "ce":
            return self.ce_loss(logits, targets)

        elif self.loss_type == "ce_dice":
            ce = self.ce_loss(logits, targets)
            dice = self.dice_loss(logits, targets)
            return self.ce_w * ce + self.dice_w * dice

        elif self.loss_type == "focal_dice":
            focal = self.focal_loss(logits, targets)
            dice = self.dice_loss(logits, targets)
            return self.ce_w * focal + self.dice_w * dice

        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")
