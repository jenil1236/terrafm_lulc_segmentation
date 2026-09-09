"""
prepare_dataset.py
==================
Dataset preparation, validation, and split generation.

What this file does:
    1. Reads file_clean.csv (s1_name, patch_id, reference_map_id).
    2. Validates that all S1/S2 patch folders and reference TIFFs exist.
    3. Extracts geographic tile codes from patch_id for leakage-safe splitting.
    4. Creates deterministic train/val/test splits → train.csv, val.csv, test.csv.
       Each split CSV keeps all three columns (s1_name, patch_id, reference_map_id).
    5. Computes S2 and S1 normalization statistics from the TRAINING SET only.
    6. Discovers class IDs from reference maps → class_mapping.json.
    7. Computes class pixel frequency stats from training reference maps.
    8. Saves a dataset_report.txt.

What goes in:
    /content/data/file_clean.csv  (output of clean_dataset.py)
    /content/data/S1/, S2/, selected_reference_map/

What comes out:
    outputs/splits/train.csv, val.csv, test.csv
    outputs/norm_stats.json
    outputs/class_mapping.json
    outputs/class_stats.json
    outputs/dataset_report.txt

How it connects:
    Run once before training. All subsequent scripts read the split CSVs
    and norm_stats.json.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm

from config import CFG
from preprocessing import (
    compute_s1_normalization_stats,
    compute_s2_normalization_stats,
    discover_class_ids,
    discover_s1_bands,
    discover_s2_bands,
)
from utils import set_seed

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1.  Load and validate the cleaned CSV
# ---------------------------------------------------------------------------

def load_clean_csv(csv_path: str) -> pd.DataFrame:
    """
    Load file_clean.csv and verify required columns.

    Expected columns: s1_name, patch_id, reference_map_id

    Returns a validated DataFrame.
    """
    df = pd.read_csv(csv_path, dtype=str).dropna(
        subset=["s1_name", "patch_id", "reference_map_id"]
    )
    df = df.apply(lambda col: col.str.strip() if col.dtype == object else col)

    required = {"s1_name", "patch_id", "reference_map_id"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV '{csv_path}' is missing columns: {missing}. "
            f"Run clean_dataset.py first to generate file_clean.csv."
        )

    logger.info(f"Loaded {len(df)} rows from {csv_path}")
    return df.reset_index(drop=True)


def quick_validate(
    df: pd.DataFrame,
    s2_root: str,
    s1_root: str,
    ref_root: str,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Fast existence check (no TIFF read – that was done by clean_dataset.py).

    Drops rows where the S2 folder, S1 folder, or reference TIFF is missing.
    Returns (clean_df, report_dict).
    """
    s2_root_p  = Path(s2_root)
    s1_root_p  = Path(s1_root)
    ref_root_p = Path(ref_root)

    valid_mask = []
    for _, row in df.iterrows():
        s2_ok  = (s2_root_p  / row["patch_id"]).is_dir()
        s1_ok  = (s1_root_p  / row["s1_name"]).is_dir()
        ref_ok = (
            (ref_root_p / f"{row['reference_map_id']}.tif").exists() or
            (ref_root_p / f"{row['reference_map_id']}.tiff").exists()
        )
        valid_mask.append(s2_ok and s1_ok and ref_ok)

    clean_df = df[valid_mask].reset_index(drop=True)
    n_dropped = len(df) - len(clean_df)
    if n_dropped > 0:
        logger.warning(
            f"Dropped {n_dropped} rows with missing files during quick validation. "
            "Re-run clean_dataset.py if this is unexpected."
        )
    report = {
        "total_rows":   len(df),
        "valid_rows":   len(clean_df),
        "dropped_rows": n_dropped,
    }
    return clean_df, report


# ---------------------------------------------------------------------------
# 2.  Geographic tile extraction for leakage-safe splitting
# ---------------------------------------------------------------------------

def extract_tile_code(patch_id: str, regex: str = r"T\d{2}[A-Z]{3}") -> Optional[str]:
    """
    Extract the Sentinel-2 MGRS tile code from a patch_id string.

    Example:
        "S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_37_88"
        → "T33UUP"

    Returns None if no match is found (random split fallback will be used).
    """
    m = re.search(regex, patch_id)
    return m.group(0) if m else None


# ---------------------------------------------------------------------------
# 3.  Train/val/test splitting
# ---------------------------------------------------------------------------

def create_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.80,
    val_ratio:   float = 0.10,
    seed:        int   = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create geographically safe train/val/test splits.

    Strategy:
        1. Try to extract MGRS tile codes from patch_id.
        2. If found: split by tile (no tile appears in both train and test).
        3. If not found: random patch-level split with a warning.

    WHY tile-based splitting matters:
        Adjacent patches within the same Sentinel-2 tile share similar
        spectral properties and atmospheric conditions. A random split
        allows the model to see patches from the same tile in both train
        and test, giving overly optimistic evaluation scores.

    All three output CSVs contain the full triplet columns:
        s1_name, patch_id, reference_map_id
    so that dataset.py can load them directly without joining.
    """
    set_seed(seed)
    rng = np.random.default_rng(seed)

    # Try tile extraction
    df = df.copy()
    df["_tile"] = df["patch_id"].apply(
        lambda p: extract_tile_code(p, CFG.tile_regex)
    )
    n_with_tile = df["_tile"].notna().sum()
    logger.info(f"Tile codes extracted for {n_with_tile}/{len(df)} patches")

    if n_with_tile > len(df) * 0.5:
        train_df, val_df, test_df = _tile_split(df, "_tile", train_ratio, val_ratio, rng)
    else:
        logger.warning(
            "Fewer than 50% of patches have extractable tile codes. "
            "Falling back to random split. "
            "This may introduce geographic data leakage."
        )
        train_df, val_df, test_df = _random_split(df, train_ratio, val_ratio, rng)

    # Drop helper column
    for d in (train_df, val_df, test_df):
        d.drop(columns=["_tile"], errors="ignore", inplace=True)

    return train_df, val_df, test_df


def _tile_split(
    df, tile_col, train_ratio, val_ratio, rng
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tiles = df[tile_col].dropna().unique()
    rng.shuffle(tiles)
    n         = len(tiles)
    n_train   = max(1, int(n * train_ratio))
    n_val     = max(1, int(n * val_ratio))

    train_tiles = set(tiles[:n_train])
    val_tiles   = set(tiles[n_train:n_train + n_val])
    test_tiles  = set(tiles[n_train + n_val:])

    # Patches without a tile code go to training by default
    no_tile    = df[df[tile_col].isna()]
    train_df   = pd.concat(
        [df[df[tile_col].isin(train_tiles)], no_tile], ignore_index=True
    )
    val_df     = df[df[tile_col].isin(val_tiles)].reset_index(drop=True)
    test_df    = df[df[tile_col].isin(test_tiles)].reset_index(drop=True)

    logger.info(
        f"Tile split ({n} tiles): "
        f"train={len(train_df)} / val={len(val_df)} / test={len(test_df)}"
    )
    return train_df, val_df, test_df


def _random_split(
    df, train_ratio, val_ratio, rng
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    idx     = rng.permutation(len(df))
    n_train = int(len(idx) * train_ratio)
    n_val   = int(len(idx) * val_ratio)

    train_df = df.iloc[idx[:n_train]].reset_index(drop=True)
    val_df   = df.iloc[idx[n_train:n_train + n_val]].reset_index(drop=True)
    test_df  = df.iloc[idx[n_train + n_val:]].reset_index(drop=True)

    logger.info(
        f"Random split: train={len(train_df)} / val={len(val_df)} / test={len(test_df)}"
    )
    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# 4.  Class statistics
# ---------------------------------------------------------------------------

def compute_class_stats(
    df: pd.DataFrame,
    raw_to_train: Dict[int, int],
    ref_root: str,
    num_classes: int = 19,
    max_samples: int = 5000,
    seed: int = 42,
) -> Tuple[Dict[int, int], Dict[int, int]]:
    """
    Count pixels and images per class from the TRAINING split only.

    Uses reference_map_id from the DataFrame to locate TIFFs.
    """
    ref_root_p = Path(ref_root)
    rng        = np.random.default_rng(seed)

    rows = df.to_dict("records")
    if len(rows) > max_samples:
        idx  = rng.choice(len(rows), size=max_samples, replace=False)
        rows = [rows[i] for i in idx]

    pixel_counts = {i: 0 for i in range(num_classes)}
    image_counts = {i: 0 for i in range(num_classes)}

    for row in tqdm(rows, desc="Computing class stats"):
        ref_stem = row["reference_map_id"]
        ref_path = next(
            (p for p in [ref_root_p / f"{ref_stem}.tif",
                         ref_root_p / f"{ref_stem}.tiff"]
             if p.exists()),
            None,
        )
        if ref_path is None:
            continue
        try:
            with rasterio.open(ref_path) as src:
                data   = src.read(1)
                nodata = src.nodata
        except Exception as e:
            logger.warning(f"Could not read {ref_path}: {e}")
            continue

        for raw_id, train_id in raw_to_train.items():
            mask = data == raw_id
            if nodata is not None:
                mask &= data != nodata
            cnt = int(mask.sum())
            if cnt > 0:
                pixel_counts[train_id] += cnt
                image_counts[train_id] += 1

    return pixel_counts, image_counts


# ---------------------------------------------------------------------------
# 5.  Main pipeline
# ---------------------------------------------------------------------------

def prepare_all(
    csv_file:   Optional[str] = None,
    s2_root:    Optional[str] = None,
    s1_root:    Optional[str] = None,
    ref_root:   Optional[str] = None,
    output_dir: Optional[str] = None,
) -> Dict:
    """
    Run the full preparation pipeline.
    Call this ONCE (after clean_dataset.py has run).

    Returns a dict with all paths and stats needed by train.py.
    """
    CFG.ensure_dirs()
    set_seed(CFG.seed)

    csv_file   = csv_file   or CFG.csv_file     # should point to file_clean.csv
    s2_root    = s2_root    or CFG.s2_dir
    s1_root    = s1_root    or CFG.s1_dir
    ref_root   = ref_root   or CFG.ref_dir
    out_dir    = output_dir or CFG.output_dir

    # ---- 1. Load cleaned CSV
    print("Loading cleaned CSV...")
    df = load_clean_csv(csv_file)

    # ---- 2. Quick existence check
    print("Validating file existence...")
    df, val_report = quick_validate(df, s2_root, s1_root, ref_root)
    _print_validation_report(val_report, len(df))

    if len(df) == 0:
        raise RuntimeError(
            "No valid rows remain after validation. "
            "Run clean_dataset.py first, then re-run prepare_all()."
        )

    # ---- 3. Discover class mapping
    print("Discovering class IDs from reference maps...")
    ref_paths = []
    for row in df.itertuples():
        for ext in (".tif", ".tiff"):
            p = Path(ref_root) / f"{row.reference_map_id}{ext}"
            if p.exists():
                ref_paths.append(str(p))
                break

    raw_to_train = discover_class_ids(ref_paths)
    CFG.raw_to_train = raw_to_train

    os.makedirs(out_dir, exist_ok=True)
    with open(CFG.class_mapping_file, "w") as f:
        json.dump({"raw_to_train": {str(k): v for k, v in raw_to_train.items()}}, f, indent=2)
    print(f"Class mapping ({len(raw_to_train)} classes) saved: {CFG.class_mapping_file}")

    # ---- 4. Create splits
    print("Creating train/val/test splits...")
    train_df, val_df, test_df = create_splits(
        df, train_ratio=CFG.train_ratio, val_ratio=CFG.val_ratio, seed=CFG.split_seed
    )

    os.makedirs(CFG.split_dir, exist_ok=True)
    cols      = ["s1_name", "patch_id", "reference_map_id"]
    train_csv = os.path.join(CFG.split_dir, "train.csv")
    val_csv   = os.path.join(CFG.split_dir, "val.csv")
    test_csv  = os.path.join(CFG.split_dir, "test.csv")
    train_df[cols].to_csv(train_csv, index=False)
    val_df[cols].to_csv(val_csv,     index=False)
    test_df[cols].to_csv(test_csv,   index=False)
    print(f"Splits: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    # ---- 5. S2 normalization stats (training set only)
    print("Computing S2 normalization statistics (training set)...")
    s2_dirs = [str(Path(s2_root) / pid) for pid in train_df["patch_id"]]
    s2_mean, s2_std = compute_s2_normalization_stats(
        s2_dirs, max_samples=5000, seed=CFG.seed
    )
    CFG.s2_norm_mean = s2_mean
    CFG.s2_norm_std  = s2_std

    # ---- 6. S1 normalization stats (training set only)
    print("Computing S1 normalization statistics (training set)...")
    s1_dirs = [str(Path(s1_root) / name) for name in train_df["s1_name"]]
    s1_mean, s1_std = compute_s1_normalization_stats(
        s1_dirs, max_samples=5000, seed=CFG.seed
    )
    CFG.s1_norm_mean = s1_mean
    CFG.s1_norm_std  = s1_std

    norm_stats = {
        "s2_mean": s2_mean, "s2_std": s2_std,
        "s1_mean": s1_mean, "s1_std": s1_std,
    }
    with open(CFG.norm_stats_file, "w") as f:
        json.dump(norm_stats, f, indent=2)
    print(f"Norm stats saved: {CFG.norm_stats_file}")

    # ---- 7. Class pixel statistics (training set only)
    print("Computing class frequency statistics...")
    pixel_counts, image_counts = compute_class_stats(
        train_df, raw_to_train, ref_root,
        num_classes=CFG.num_classes, seed=CFG.seed,
    )
    class_stats_path = os.path.join(out_dir, "class_stats.json")
    with open(class_stats_path, "w") as f:
        json.dump({"pixel_counts": pixel_counts, "image_counts": image_counts}, f, indent=2)
    print(f"Class stats saved: {class_stats_path}")
    _print_class_stats(pixel_counts, image_counts)

    # ---- 8. Dataset report
    report_path = os.path.join(out_dir, "dataset_report.txt")
    _save_report(
        report_path, val_report, s2_mean, s2_std, s1_mean, s1_std,
        pixel_counts, image_counts, len(train_df), len(val_df), len(test_df),
    )

    return {
        "train_csv":        train_csv,
        "val_csv":          val_csv,
        "test_csv":         test_csv,
        "norm_stats_file":  CFG.norm_stats_file,
        "class_stats_file": class_stats_path,
        "s2_mean":          s2_mean,
        "s2_std":           s2_std,
        "s1_mean":          s1_mean,
        "s1_std":           s1_std,
        "raw_to_train":     raw_to_train,
        "pixel_counts":     pixel_counts,
        "image_counts":     image_counts,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_validation_report(report: Dict, n_valid: int) -> None:
    print("\n" + "=" * 50)
    print("DATASET VALIDATION REPORT")
    print("=" * 50)
    print(f"  Total rows in CSV: {report['total_rows']}")
    print(f"  Valid rows:        {n_valid}")
    print(f"  Dropped rows:      {report['dropped_rows']}")
    print("=" * 50)


def _print_class_stats(pixel_counts: Dict, image_counts: Dict) -> None:
    total = max(sum(pixel_counts.values()), 1)
    print("\n  Class pixel distribution:")
    print(f"  {'Class':>6} | {'Pixels':>12} | {'%':>7} | {'Images':>8}")
    print("  " + "-" * 44)
    for cls in sorted(pixel_counts):
        cnt  = pixel_counts[cls]
        pct  = 100.0 * cnt / total
        imgs = image_counts.get(cls, 0)
        print(f"  {cls:>6} | {cnt:>12,d} | {pct:>6.2f}% | {imgs:>8,d}")


def _save_report(
    path, val_report, s2_mean, s2_std, s1_mean, s1_std,
    pixel_counts, image_counts, n_train, n_val, n_test,
) -> None:
    with open(path, "w") as f:
        f.write("DATASET PREPARATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total rows:  {val_report['total_rows']}\n")
        f.write(f"Valid rows:  {val_report['valid_rows']}\n")
        f.write(f"Train / Val / Test: {n_train} / {n_val} / {n_test}\n\n")
        f.write("S2 Normalization (per-channel mean / std):\n")
        for i, (m, s) in enumerate(zip(s2_mean, s2_std)):
            f.write(f"  S2 Band {i:02d}: mean={m:.6f}, std={s:.6f}\n")
        f.write("\nS1 Normalization (per-channel mean / std, in dB):\n")
        for i, (m, s) in enumerate(zip(s1_mean, s1_std)):
            f.write(f"  S1 Band {i:02d}: mean={m:.4f} dB, std={s:.4f} dB\n")
        f.write("\nClass pixel counts:\n")
        total = max(sum(pixel_counts.values()), 1)
        for cls in sorted(pixel_counts):
            cnt = pixel_counts[cls]
            pct = 100.0 * cnt / total
            f.write(f"  Class {cls:2d}: {cnt:>12,d} pixels ({pct:.2f}%)\n")
    print(f"Dataset report: {path}")
