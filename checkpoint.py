"""
checkpoint.py
=============
Robust checkpoint save/load for Colab-safe training.

What this file does:
    Saves and restores full training state (model, optimizer, scheduler,
    scaler, epoch, best metric, config, class mapping, norm stats) so that
    a Colab session can be resumed after a crash or timeout.

What goes in:
    - All stateful objects from train.py

What comes out:
    - checkpoint_epoch_N.pth  (periodic saves)
    - best_model.pth          (saved when val mIoU improves)
    - last_checkpoint.pth     (always the most recent epoch)

How it connects:
    train.py calls save_checkpoint() after every epoch.
    To resume: python train.py (with CFG.resume_from set).
"""

from __future__ import annotations

import logging
import os
import shutil
from typing import Any, Dict, List, Optional
import torch

from config import CFG

logger = logging.getLogger(__name__)


def save_checkpoint(
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: Optional[torch.cuda.amp.GradScaler],
    best_val_miou: float,
    val_miou: float,
    norm_mean: List[float],
    norm_std: List[float],
    raw_to_train: Dict[int, int],
    is_best: bool,
    checkpoint_dir: str,
    keep_last_n: int = 3,
    extra: Optional[Dict] = None,
) -> str:
    """
    Save a training checkpoint.

    Always saves:
        last_checkpoint.pth

    Periodically saves (every CFG.checkpoint_frequency epochs):
        checkpoint_epoch_{N}.pth

    When val mIoU improves:
        best_model.pth

    Args:
        epoch:           Current epoch (0-indexed).
        model:           The nn.Module.
        optimizer:       Optimizer instance.
        scheduler:       LR scheduler instance.
        scaler:          AMP GradScaler (or None).
        best_val_miou:   Best validation mIoU seen so far.
        val_miou:        Current epoch's validation mIoU.
        norm_mean/std:   Preprocessing statistics.
        raw_to_train:    Class mapping dict.
        is_best:         True if this epoch achieved a new best mIoU.
        checkpoint_dir:  Directory to save checkpoints.
        keep_last_n:     Number of periodic checkpoints to retain.

    Returns:
        Path to the saved checkpoint file.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "best_val_miou": best_val_miou,
        "val_miou": val_miou,
        "norm_mean": norm_mean,
        "norm_std": norm_std,
        "raw_to_train": {str(k): v for k, v in raw_to_train.items()},
        "config": {
            "model_size": CFG.model_size,
            "num_classes": CFG.num_classes,
            "image_size": CFG.image_size,
            "ignore_index": CFG.ignore_index,
            "freeze_stage": CFG.freeze_stage,
            "seed": CFG.seed,
        },
    }
    if extra:
        state.update(extra)

    # Always save as last_checkpoint.pth
    last_path = os.path.join(checkpoint_dir, "last_checkpoint.pth")
    torch.save(state, last_path)

    # Periodic checkpoint
    if (epoch + 1) % CFG.checkpoint_frequency == 0:
        epoch_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1:03d}.pth")
        shutil.copy2(last_path, epoch_path)
        logger.info(f"Checkpoint saved: {epoch_path}")
        _cleanup_old_checkpoints(checkpoint_dir, keep_last_n)

    # Best model
    if is_best:
        best_path = os.path.join(checkpoint_dir, "best_model.pth")
        shutil.copy2(last_path, best_path)
        logger.info(f"New best model saved: val_mIoU={val_miou:.4f} → {best_path}")

    return last_path


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Load a checkpoint and restore all training state.

    Args:
        path:      Path to the checkpoint .pth file.
        model:     Model to restore weights into.
        optimizer: Optimizer to restore state (None = weights only).
        scheduler: LR scheduler to restore.
        scaler:    AMP GradScaler to restore.
        device:    Device to map tensors to.

    Returns:
        Dict with restored metadata:
            epoch, best_val_miou, norm_mean, norm_std, raw_to_train

    IMPORTANT: When resuming, do NOT restart epoch counter, optimizer state,
    or scheduler state. This function restores all of them correctly.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    logger.info(f"Resuming from checkpoint: {path}")
    state = torch.load(path, map_location=device)

    # Restore model weights
    model.load_state_dict(state["model_state_dict"])

    # Restore optimizer (preserves momentum, learning rate state)
    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])

    # Restore scheduler (preserves step count, last_epoch)
    if scheduler is not None and state.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(state["scheduler_state_dict"])

    # Restore AMP scaler (preserves scale factor)
    if scaler is not None and state.get("scaler_state_dict") is not None:
        scaler.load_state_dict(state["scaler_state_dict"])

    epoch = state.get("epoch", 0)
    best_val_miou = state.get("best_val_miou", 0.0)
    norm_mean = state.get("norm_mean", CFG.norm_mean)
    norm_std = state.get("norm_std", CFG.norm_std)
    raw_to_train = {
        int(k): v for k, v in state.get("raw_to_train", {}).items()
    }

    logger.info(
        f"Resumed: epoch={epoch+1}, best_val_mIoU={best_val_miou:.4f}"
    )
    return {
        "epoch": epoch,
        "best_val_miou": best_val_miou,
        "norm_mean": norm_mean,
        "norm_std": norm_std,
        "raw_to_train": raw_to_train,
    }


def _cleanup_old_checkpoints(checkpoint_dir: str, keep_last_n: int) -> None:
    """Remove old periodic checkpoint files, keeping only the last N."""
    import glob, re
    pattern = os.path.join(checkpoint_dir, "checkpoint_epoch_*.pth")
    files = sorted(glob.glob(pattern))
    if len(files) > keep_last_n:
        for f in files[:-keep_last_n]:
            os.remove(f)
            logger.debug(f"Removed old checkpoint: {f}")
