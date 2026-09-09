"""
train.py
========
Main training script for TerraFM S1+S2 LULC segmentation.

What goes in:
    Split CSVs (train.csv, val.csv) with columns s1_name, patch_id, reference_map_id
    S2/S1 normalization statistics, class mapping, class pixel counts

What comes out:
    best_model.pth, last_checkpoint.pth, training_history_<phase>.csv

Training phases:
    Phase 0 – overfit 64 samples (smoke test, mandatory)
    Phase 1 – pilot on ~2000 samples
    Phase 2 – full training:
        Epochs 1-5:  encoder frozen (Stage 0) — decoder warms up
        Epoch 6+:    last 4 ViT blocks + patch embed unfrozen (Stage 1)
        Early stopping patience = 8
"""

from __future__ import annotations

import csv
import logging
import math
import os
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from checkpoint import load_checkpoint, save_checkpoint
from config import CFG
from dataset import build_dataloaders
from losses import SegmentationLoss, compute_class_weights
from metrics import SegmentationMetrics
from model import TerraFMLULC, build_model
from utils import get_gpu_memory_gb, set_seed

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

def build_optimizer(model: TerraFMLULC) -> AdamW:
    """
    AdamW with differential LR:
        encoder (pretrained) → encoder_lr (5e-5)
        decoder (random init) → decoder_lr (5e-4)
    When encoder is fully frozen no encoder group is added.
    """
    enc_params = [p for p in model.encoder.parameters() if p.requires_grad]
    dec_params = list(model.decoder.parameters())

    if enc_params:
        groups = [
            {"params": enc_params, "lr": CFG.encoder_lr, "name": "encoder"},
            {"params": dec_params, "lr": CFG.decoder_lr, "name": "decoder"},
        ]
        logger.info(f"Optimizer: enc_lr={CFG.encoder_lr}  dec_lr={CFG.decoder_lr}")
    else:
        groups = [{"params": dec_params, "lr": CFG.decoder_lr, "name": "decoder"}]
        logger.info(f"Optimizer: encoder frozen  dec_lr={CFG.decoder_lr}")

    return AdamW(groups, weight_decay=CFG.weight_decay, betas=(0.9, 0.999), eps=1e-8)


# ---------------------------------------------------------------------------
# Scheduler (cosine with linear warmup)
# ---------------------------------------------------------------------------

def build_scheduler(optimizer: AdamW, total_steps: int, warmup_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(warmup_steps, 1)
        progress    = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        cosine_val  = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(CFG.min_lr / CFG.decoder_lr, cosine_val)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(
    train_csv: str,
    val_csv: str,
    s2_mean: List[float],
    s2_std: List[float],
    s1_mean: List[float],
    s1_std: List[float],
    raw_to_train: Dict[int, int],
    class_pixel_counts: Dict[int, int],
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    freeze_stage: int = 0,
    phase_name: str = "phase2",
    resume_from: Optional[str] = None,
) -> str:
    """
    Run the full training loop for one phase.

    Args:
        train_csv / val_csv:   Split CSVs with s1_name, patch_id, reference_map_id.
        s2_mean/std:           S2 normalisation stats (12 values each).
        s1_mean/std:           S1 normalisation stats ( 2 values each).
        raw_to_train:          Raw class ID → train ID mapping.
        class_pixel_counts:    Training pixel counts per class (for loss weighting).
        epochs:                Override CFG.epochs.
        batch_size:            Override CFG.batch_size.
        freeze_stage:          Initial encoder freeze stage.
        phase_name:            Label used for log filenames.
        resume_from:           Path to a checkpoint to resume from.

    Returns:
        Path to best_model.pth.
    """
    set_seed(CFG.seed)
    CFG.ensure_dirs()

    n_epochs = epochs    or CFG.epochs
    bs       = batch_size or CFG.batch_size
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}  |  Phase: {phase_name}")

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------
    train_loader, val_loader, _ = build_dataloaders(
        train_csv=train_csv,
        val_csv=val_csv,
        test_csv=val_csv,          # placeholder; only train+val used here
        s2_mean=s2_mean, s2_std=s2_std,
        s1_mean=s1_mean, s1_std=s1_std,
        raw_to_train=raw_to_train,
        batch_size=bs,
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = build_model(freeze_stage=freeze_stage)
    model.to(device)

    if CFG.gradient_checkpointing:
        if hasattr(model.encoder.backbone, "set_grad_checkpointing"):
            model.encoder.backbone.set_grad_checkpointing(True)
            logger.info("Gradient checkpointing enabled.")

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    class_weights = None
    if CFG.use_class_weights and class_pixel_counts:
        class_weights = compute_class_weights(
            class_pixel_counts,
            num_classes=CFG.num_classes,
            max_weight=CFG.max_class_weight,
        ).to(device)

    criterion = SegmentationLoss(
        loss_type=CFG.loss_type,
        class_weights=class_weights,
        ce_weight=CFG.ce_weight,
        dice_weight=CFG.dice_weight,
        focal_gamma=CFG.focal_gamma,
        ignore_index=CFG.ignore_index,
    )

    # ------------------------------------------------------------------
    # Optimizer + scheduler
    # ------------------------------------------------------------------
    optimizer    = build_optimizer(model)
    total_steps  = (len(train_loader) // CFG.grad_accum_steps) * n_epochs
    warmup_steps = int(total_steps * CFG.warmup_ratio)
    scheduler    = build_scheduler(optimizer, total_steps, warmup_steps)

    # ------------------------------------------------------------------
    # AMP  (FP16 for T4)
    # ------------------------------------------------------------------
    use_amp   = device.type == "cuda"
    amp_dtype = torch.float16 if CFG.amp_dtype == "fp16" else torch.bfloat16
    scaler    = GradScaler(enabled=(use_amp and CFG.amp_dtype == "fp16"))

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------
    start_epoch    = 0
    best_val_miou  = 0.0
    resume_path    = resume_from or CFG.resume_from
    if resume_path and os.path.exists(resume_path):
        state = load_checkpoint(
            resume_path, model, optimizer, scheduler, scaler,
            device=str(device),
        )
        start_epoch   = state["epoch"] + 1
        best_val_miou = state["best_val_miou"]
        s2_mean       = state.get("s2_mean", s2_mean)
        s2_std        = state.get("s2_std",  s2_std)
        s1_mean       = state.get("s1_mean", s1_mean)
        s1_std        = state.get("s1_std",  s1_std)
        raw_to_train  = state.get("raw_to_train", raw_to_train)

    # ------------------------------------------------------------------
    # History CSV
    # ------------------------------------------------------------------
    history_path = os.path.join(CFG.output_dir, f"training_history_{phase_name}.csv")
    fields = [
        "epoch", "train_loss", "val_loss", "val_miou", "val_dice",
        "val_pixel_acc", "lr_enc", "lr_dec", "gpu_gb", "epoch_s",
    ]
    if not os.path.exists(history_path):
        with open(history_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    train_m = SegmentationMetrics(CFG.num_classes, CFG.ignore_index, CFG.class_names)
    val_m   = SegmentationMetrics(CFG.num_classes, CFG.ignore_index, CFG.class_names)

    best_ckpt_path  = os.path.join(CFG.checkpoint_dir, "best_model.pth")
    patience_count  = 0

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for epoch in range(start_epoch, start_epoch + n_epochs):
        t0 = time.time()

        # Staged unfreeze: after 5 warm-up epochs, open last blocks
        if (phase_name == "phase2"
                and epoch == start_epoch + 5
                and CFG.freeze_stage == 0):
            logger.info("Epoch 6: switching to Stage 1 (partial unfreeze)")
            model.set_freeze_stage(1)
            optimizer    = build_optimizer(model)
            remain_steps = (
                (len(train_loader) // CFG.grad_accum_steps) * (n_epochs - 5)
            )
            scheduler = build_scheduler(optimizer, remain_steps, warmup_steps=0)

        # ---- Train --------------------------------------------------
        model.train()
        train_m.reset()
        running_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Ep {epoch+1:03d}/{start_epoch+n_epochs} [train]",
            leave=False,
        )
        for step, (fused, mask) in pbar:
            fused = fused.to(device, non_blocking=True)
            mask  = mask.to(device, non_blocking=True)

            with autocast(enabled=use_amp, dtype=amp_dtype):
                logits = model(fused)
                loss   = criterion(logits, mask) / CFG.grad_accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % CFG.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            running_loss += loss.item() * CFG.grad_accum_steps
            with torch.no_grad():
                train_m.update(logits.argmax(dim=1), mask)
            pbar.set_postfix({"loss": f"{loss.item() * CFG.grad_accum_steps:.4f}"})

        avg_train_loss = running_loss / max(len(train_loader), 1)

        # ---- Validate -----------------------------------------------
        model.eval()
        val_m.reset()
        val_loss_acc = 0.0

        with torch.no_grad():
            for fused, mask in tqdm(val_loader,
                                    desc=f"Ep {epoch+1:03d} [val]", leave=False):
                fused = fused.to(device, non_blocking=True)
                mask  = mask.to(device, non_blocking=True)
                with autocast(enabled=use_amp, dtype=amp_dtype):
                    logits = model(fused)
                    vloss  = criterion(logits, mask)
                val_loss_acc += vloss.item()
                val_m.update(logits.argmax(dim=1), mask)

        avg_val_loss = val_loss_acc / max(len(val_loader), 1)
        vr           = val_m.compute()
        val_miou     = vr["mean_iou"]
        val_dice     = vr["mean_dice"]
        val_acc      = vr["pixel_accuracy"]

        lrs     = {pg["name"]: pg["lr"] for pg in optimizer.param_groups if "name" in pg}
        lr_enc  = lrs.get("encoder", 0.0)
        lr_dec  = lrs.get("decoder", CFG.decoder_lr)
        gpu_gb  = get_gpu_memory_gb()
        ep_s    = time.time() - t0
        is_best = val_miou > best_val_miou

        print(
            f"\nEpoch {epoch+1:03d}/{start_epoch+n_epochs:03d}  "
            f"{'← BEST' if is_best else ''}\n"
            f"  Train Loss: {avg_train_loss:.4f}  |  Val Loss: {avg_val_loss:.4f}\n"
            f"  Val mIoU:   {val_miou:.4f}  |  Val Dice: {val_dice:.4f}  "
            f"|  Acc: {val_acc:.4f}\n"
            f"  LR enc/dec: {lr_enc:.2e} / {lr_dec:.2e}  |  "
            f"GPU: {gpu_gb:.1f} GB  |  "
            f"Time: {int(ep_s//60)}m {int(ep_s%60):02d}s"
        )

        if is_best:
            best_val_miou  = val_miou
            patience_count = 0
        else:
            patience_count += 1

        save_checkpoint(
            epoch=epoch, model=model,
            optimizer=optimizer, scheduler=scheduler, scaler=scaler,
            best_val_miou=best_val_miou, val_miou=val_miou,
            norm_mean=s2_mean, norm_std=s2_std,  # kept for compat
            raw_to_train=raw_to_train,
            is_best=is_best,
            checkpoint_dir=CFG.checkpoint_dir,
            extra={
                "s2_mean": s2_mean, "s2_std": s2_std,
                "s1_mean": s1_mean, "s1_std": s1_std,
            },
        )

        with open(history_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writerow({
                "epoch": epoch + 1,
                "train_loss": f"{avg_train_loss:.6f}",
                "val_loss":   f"{avg_val_loss:.6f}",
                "val_miou":   f"{val_miou:.6f}",
                "val_dice":   f"{val_dice:.6f}",
                "val_pixel_acc": f"{val_acc:.6f}",
                "lr_enc":     f"{lr_enc:.2e}",
                "lr_dec":     f"{lr_dec:.2e}",
                "gpu_gb":     f"{gpu_gb:.2f}",
                "epoch_s":    f"{ep_s:.1f}",
            })

        if patience_count >= CFG.early_stopping_patience:
            print(f"\nEarly stopping after {CFG.early_stopping_patience} epochs "
                  f"without improvement.")
            break

    print(f"\n{'='*60}")
    print(f"Done. Best val mIoU: {best_val_miou:.4f}")
    print(f"Best checkpoint: {best_ckpt_path}")
    return best_ckpt_path


# ---------------------------------------------------------------------------
# Phase 0 smoke test
# ---------------------------------------------------------------------------

def run_smoke_test(
    train_csv: str,
    val_csv: str,
    s2_mean: List[float],
    s2_std: List[float],
    s1_mean: List[float],
    s1_std: List[float],
    raw_to_train: Dict[int, int],
) -> None:
    """
    Overfit a tiny subset (64 samples) to verify the full pipeline.

    Goal: training loss < 1.0 within 30 epochs.
    If not reached, there is a bug in data loading, class mapping, or the model.
    This test is MANDATORY before committing to full training.
    """
    import tempfile
    import pandas as pd

    df_train = pd.read_csv(train_csv).head(CFG.phase0_samples)
    df_val   = pd.read_csv(val_csv).head(16)

    with tempfile.TemporaryDirectory() as tmp:
        t_csv = os.path.join(tmp, "tiny_train.csv")
        v_csv = os.path.join(tmp, "tiny_val.csv")
        df_train.to_csv(t_csv, index=False)
        df_val.to_csv(v_csv,   index=False)

        print("=" * 60)
        print("Phase 0: Smoke test")
        print(f"  {len(df_train)} train / {len(df_val)} val samples")
        print("  Target: train loss < 1.0 within 30 epochs")
        print("=" * 60)

        train(
            train_csv=t_csv, val_csv=v_csv,
            s2_mean=s2_mean, s2_std=s2_std,
            s1_mean=s1_mean, s1_std=s1_std,
            raw_to_train=raw_to_train,
            class_pixel_counts={},
            epochs=CFG.phase0_epochs,
            batch_size=CFG.phase0_batch_size,
            freeze_stage=0,
            phase_name="phase0",
        )
