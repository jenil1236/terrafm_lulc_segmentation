"""
evaluate.py
===========
Full evaluation on the test set (S1+S2 model).

What goes in:
    - best_model.pth or terrafm_lulc_model.pth
    - test.csv  (s1_name, patch_id, reference_map_id)

What comes out:
    results/
    ├── metrics.json
    ├── per_class_metrics.csv
    ├── confusion_matrix.csv
    ├── confusion_matrix.png
    └── qualitative_predictions/sample_NNN.png
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import CFG
from dataset import S1S2LULCDataset, load_records_from_csv
from metrics import SegmentationMetrics
from model import load_final_model, build_model
from visualize import plot_confusion_matrix, plot_qualitative_sample

logger = logging.getLogger(__name__)


def evaluate(
    model_path: str,
    test_csv: str,
    s2_mean: List[float],
    s2_std: List[float],
    s1_mean: List[float],
    s1_std: List[float],
    raw_to_train: Dict[int, int],
    class_names: Optional[List[str]] = None,
    num_qualitative: int = 8,
) -> Dict:
    """
    Run evaluation on the test set and save all results.

    Args:
        model_path:      Path to best_model.pth or terrafm_lulc_model.pth.
        test_csv:        Path to test split CSV.
        s2_mean/std:     S2 normalization stats.
        s1_mean/std:     S1 normalization stats.
        raw_to_train:    Class ID remapping.
        class_names:     Human-readable class names.
        num_qualitative: Number of sample visualizations to save.

    Returns:
        Dict of metric values.
    """
    CFG.ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cn     = class_names or CFG.class_names

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    try:
        model, meta = load_final_model(model_path, device=str(device))
        # Override norm stats from the saved artifact (authoritative)
        s2_mean = meta["s2_mean"]
        s2_std  = meta["s2_std"]
        s1_mean = meta["s1_mean"]
        s1_std  = meta["s1_std"]
    except Exception:
        from checkpoint import load_checkpoint
        model = build_model(freeze_stage=2)
        load_checkpoint(model_path, model, device=str(device))
        model.to(device)
        model.eval()

    # ------------------------------------------------------------------
    # Test dataset
    # ------------------------------------------------------------------
    records = load_records_from_csv(test_csv)
    test_ds = S1S2LULCDataset(
        records=records,
        s2_root=CFG.s2_dir,
        s1_root=CFG.s1_dir,
        ref_root=CFG.ref_dir,
        s2_mean=s2_mean, s2_std=s2_std,
        s1_mean=s1_mean, s1_std=s1_std,
        raw_to_train=raw_to_train,
        augment=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=CFG.batch_size,
        shuffle=False, num_workers=CFG.num_workers,
    )

    # ------------------------------------------------------------------
    # Inference loop
    # ------------------------------------------------------------------
    seg_m    = SegmentationMetrics(CFG.num_classes, CFG.ignore_index, cn)
    use_amp  = device.type == "cuda"
    amp_dtype = torch.float16 if CFG.amp_dtype == "fp16" else torch.bfloat16
    qual_samples: List[dict] = []

    model.eval()
    with torch.no_grad():
        for fused, mask in tqdm(test_loader, desc="Evaluating test set"):
            fused = fused.to(device)
            mask  = mask.to(device)
            with autocast(enabled=use_amp, dtype=amp_dtype):
                logits = model(fused)
            preds = logits.argmax(dim=1)
            seg_m.update(preds, mask)

            if len(qual_samples) < num_qualitative:
                for i in range(fused.shape[0]):
                    if len(qual_samples) >= num_qualitative:
                        break
                    # Save only the S2 channels (first 12) for visualization
                    qual_samples.append({
                        "s2":   fused[i, :12].cpu(),
                        "mask": mask[i].cpu(),
                        "pred": preds[i].cpu(),
                    })

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    results       = seg_m.compute()
    scalar_metrics = {
        "mean_iou":        results["mean_iou"],
        "mean_dice":       results["mean_dice"],
        "pixel_accuracy":  results["pixel_accuracy"],
        "freq_weighted_iou": results["freq_weighted_iou"],
        "num_valid_classes": results["num_valid_classes"],
    }

    metrics_path = os.path.join(CFG.results_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(scalar_metrics, f, indent=2)

    df_cls = seg_m.per_class_table()
    df_cls.to_csv(os.path.join(CFG.results_dir, "per_class_metrics.csv"), index=False)

    cm = results["confusion_matrix"]
    pd.DataFrame(
        cm,
        index=[f"{i}:{n}" for i, n in enumerate(cn)],
        columns=[f"{i}:{n}" for i, n in enumerate(cn)],
    ).to_csv(os.path.join(CFG.results_dir, "confusion_matrix.csv"))
    plot_confusion_matrix(cm, cn, os.path.join(CFG.results_dir, "confusion_matrix.png"))

    print("\n" + "=" * 60)
    print("TEST SET RESULTS")
    print("=" * 60)
    print(f"  Mean IoU:       {results['mean_iou']:.4f}")
    print(f"  Mean Dice:      {results['mean_dice']:.4f}")
    print(f"  Pixel Accuracy: {results['pixel_accuracy']:.4f}")
    print(f"  FW-IoU:         {results['freq_weighted_iou']:.4f}")
    print("\n" + df_cls[["Class Name", "IoU", "Pixel Count"]].to_string(index=False))

    # ------------------------------------------------------------------
    # Qualitative samples
    # ------------------------------------------------------------------
    qual_dir = os.path.join(CFG.results_dir, "qualitative_predictions")
    os.makedirs(qual_dir, exist_ok=True)
    for idx, s in enumerate(qual_samples):
        plot_qualitative_sample(
            s2=s["s2"], gt_mask=s["mask"], pred_mask=s["pred"],
            class_names=cn, class_colors=CFG.class_colors,
            ignore_index=CFG.ignore_index,
            save_path=os.path.join(qual_dir, f"sample_{idx:03d}.png"),
            norm_mean=s2_mean, norm_std=s2_std,
        )

    print(f"\nSaved to {CFG.results_dir}")
    return results
