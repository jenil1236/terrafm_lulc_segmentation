"""
clean_dataset.py
================
Dataset cleaning script — run this FIRST before any other step.

What this file does:
    Reads file.csv, validates every S1 patch folder, S2 patch folder,
    and reference map TIFF for corruption or missing files.  Corrupted
    or unreadable entries are:
        1. Logged to  corrupted_patches.txt  (with reason and count).
        2. Removed from file.csv  →  file_clean.csv  is written.
        3. Their on-disk folders/files are deleted from S1/ and S2/
           and the reference map TIFF is deleted from reference_maps_selected/.

Usage (from Colab or terminal):
    python clean_dataset.py

    # or with explicit paths:
    python clean_dataset.py \
        --csv  /content/data/file.csv \
        --s1   /content/data/S1 \
        --s2   /content/data/S2 \
        --ref  /content/data/reference_maps_selected \
        --out_csv    /content/data/file_clean.csv \
        --log        /content/outputs/corrupted_patches.txt

What goes in:
    file.csv with columns:  s1_name, patch_id, reference_map_id
    S1/  directory tree
    S2/  directory tree
    reference_maps_selected/ directory

What comes out:
    file_clean.csv           – cleaned CSV (use this for all training)
    corrupted_patches.txt    – log of every removed row + reason + total count

Corruption checks performed per row:
    S2:  - folder exists
         - exactly 12 TIFF files present
         - every TIFF is readable by rasterio (no truncation / bad header)
         - no TIFF returns all-NaN or all-zero data
    S1:  - folder exists
         - at least 1 TIFF file present (expect 2: VV + VH)
         - every TIFF is readable by rasterio
         - no TIFF returns all-NaN or all-zero data
    REF: - file exists (tries .tif and .tiff extensions)
         - file is readable by rasterio
         - contains at least one valid (non-nodata) pixel
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm

# Allow running standalone without importing CFG to keep this script
# self-contained (it is run before the main pipeline is set up).
DEFAULT_CSV = "/content/data/file.csv"
DEFAULT_S1  = "/content/data/S1"
DEFAULT_S2  = "/content/data/S2"
DEFAULT_REF = "/content/data/reference_maps_selected"
DEFAULT_OUT_CSV = "/content/data/file_clean.csv"
DEFAULT_LOG = "/content/outputs/corrupted_patches.txt"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Individual file / folder validators
# ---------------------------------------------------------------------------

def _check_tiff(path: Path) -> Optional[str]:
    """
    Try to open and read the first band of a TIFF file.

    Returns:
        None if healthy.
        Error string if corrupted / unreadable / all-zero / all-NaN.
    """
    if not path.exists():
        return f"file not found: {path}"
    try:
        with rasterio.open(path) as src:
            data = src.read(1)
    except Exception as e:
        return f"rasterio open error ({path.name}): {e}"

    if data is None or data.size == 0:
        return f"empty array: {path.name}"

    # Cast to float to handle any dtype safely
    arr = data.astype(np.float32)

    # All-NaN after float cast → corrupt
    if np.all(np.isnan(arr)):
        return f"all-NaN values: {path.name}"

    # All-zero is suspicious for reflectance / SAR data
    valid = arr[~np.isnan(arr)]
    if valid.size > 0 and np.all(valid == 0):
        return f"all-zero values: {path.name}"

    return None  # healthy


def check_s2_patch(s2_dir: Path) -> Optional[str]:
    """
    Validate an S2 patch directory.

    Returns None if healthy, otherwise a reason string.
    """
    if not s2_dir.exists():
        return f"S2 folder missing: {s2_dir}"

    tiffs = sorted(
        list(s2_dir.glob("*.tif")) + list(s2_dir.glob("*.tiff"))
    )
    if len(tiffs) != 12:
        return (
            f"S2 folder '{s2_dir.name}' has {len(tiffs)} TIFFs "
            f"(expected 12)"
        )

    for tiff in tiffs:
        err = _check_tiff(tiff)
        if err:
            return f"S2 TIFF corrupt – {err}"

    return None


def check_s1_patch(s1_dir: Path) -> Optional[str]:
    """
    Validate an S1 patch directory.

    Returns None if healthy, otherwise a reason string.
    """
    if not s1_dir.exists():
        return f"S1 folder missing: {s1_dir}"

    tiffs = sorted(
        list(s1_dir.glob("*.tif")) + list(s1_dir.glob("*.tiff"))
    )
    if len(tiffs) == 0:
        return f"S1 folder '{s1_dir.name}' has no TIFF files"

    if len(tiffs) < 2:
        logger.warning(
            f"S1 folder '{s1_dir.name}' has only {len(tiffs)} TIFF "
            f"(expected 2: VV + VH). Proceeding if readable."
        )

    for tiff in tiffs:
        err = _check_tiff(tiff)
        if err:
            return f"S1 TIFF corrupt – {err}"

    return None


def check_reference(ref_dir: Path, ref_stem: str) -> Tuple[Optional[str], Optional[Path]]:
    """
    Find and validate a reference map TIFF.

    Tries: <ref_stem>.tif, <ref_stem>.tiff

    Returns:
        (error_string_or_None, resolved_path_or_None)
    """
    candidates = [
        ref_dir / f"{ref_stem}.tif",
        ref_dir / f"{ref_stem}.tiff",
    ]
    ref_path = next((p for p in candidates if p.exists()), None)

    if ref_path is None:
        return (
            f"reference map not found: {ref_stem}(.tif/.tiff) in {ref_dir}",
            None,
        )

    try:
        with rasterio.open(ref_path) as src:
            data = src.read(1)
            nodata = src.nodata
    except Exception as e:
        return (f"reference TIFF open error ({ref_path.name}): {e}", ref_path)

    # Check that at least one valid (non-nodata) pixel exists
    valid_mask = (data != nodata) if nodata is not None else np.ones_like(data, dtype=bool)
    if not valid_mask.any():
        return (f"reference TIFF has no valid pixels: {ref_path.name}", ref_path)

    return None, ref_path


# ---------------------------------------------------------------------------
# Delete helpers
# ---------------------------------------------------------------------------

def _safe_delete_folder(folder: Path) -> None:
    """Delete a directory tree if it exists, logging the action."""
    if folder.exists() and folder.is_dir():
        shutil.rmtree(folder)
        logger.info(f"  Deleted folder: {folder}")


def _safe_delete_file(path: Path) -> None:
    """Delete a single file if it exists, logging the action."""
    if path.exists() and path.is_file():
        path.unlink()
        logger.info(f"  Deleted file:   {path}")


# ---------------------------------------------------------------------------
# Main cleaning routine
# ---------------------------------------------------------------------------

def clean_dataset(
    csv_path: str,
    s1_root: str,
    s2_root: str,
    ref_root: str,
    out_csv: str,
    log_path: str,
) -> None:
    """
    Main entry point. Reads file.csv, checks every row, removes corrupt
    entries, saves cleaned CSV and corruption log.
    """
    # Make sure log output dir exists
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    s1_root_p  = Path(s1_root)
    s2_root_p  = Path(s2_root)
    ref_root_p = Path(ref_root)

    # ------------------------------------------------------------------
    # Load CSV
    # Expected columns: s1_name, patch_id, reference_map_id
    # ------------------------------------------------------------------
    logger.info(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"  Rows in CSV: {len(df)}")

    required_cols = {"s1_name", "patch_id", "reference_map_id"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"file.csv is missing required columns: {missing_cols}. "
            f"Found columns: {list(df.columns)}"
        )

    # ------------------------------------------------------------------
    # Validate each row
    # ------------------------------------------------------------------
    corrupted_rows: List[dict] = []   # rows to remove
    clean_rows:     List[int]  = []   # indices to keep

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Validating patches"):
        s1_name  = str(row["s1_name"]).strip()
        patch_id = str(row["patch_id"]).strip()
        ref_stem = str(row["reference_map_id"]).strip()

        errors: List[str] = []

        # ---- S2 check
        s2_dir = s2_root_p / patch_id
        s2_err = check_s2_patch(s2_dir)
        if s2_err:
            errors.append(s2_err)

        # ---- S1 check
        s1_dir = s1_root_p / s1_name
        s1_err = check_s1_patch(s1_dir)
        if s1_err:
            errors.append(s1_err)

        # ---- Reference map check
        ref_err, ref_path = check_reference(ref_root_p, ref_stem)
        if ref_err:
            errors.append(ref_err)

        if errors:
            corrupted_rows.append({
                "s1_name": s1_name,
                "patch_id": patch_id,
                "reference_map_id": ref_stem,
                "errors": " | ".join(errors),
                "s2_dir": str(s2_dir),
                "s1_dir": str(s1_dir),
                "ref_path": str(ref_path) if ref_path else "",
            })
        else:
            clean_rows.append(idx)

    n_total     = len(df)
    n_corrupted = len(corrupted_rows)
    n_clean     = len(clean_rows)

    # ------------------------------------------------------------------
    # Write corruption log
    # ------------------------------------------------------------------
    with open(log_path, "w") as log_f:
        log_f.write("CORRUPTED PATCH LOG\n")
        log_f.write("=" * 70 + "\n")
        log_f.write(f"Total rows in original CSV:  {n_total}\n")
        log_f.write(f"Corrupted / removed rows:    {n_corrupted}\n")
        log_f.write(f"Clean rows retained:         {n_clean}\n")
        log_f.write("=" * 70 + "\n\n")

        for entry in corrupted_rows:
            log_f.write(f"PATCH: {entry['patch_id']}\n")
            log_f.write(f"  S1 name:   {entry['s1_name']}\n")
            log_f.write(f"  Ref stem:  {entry['reference_map_id']}\n")
            log_f.write(f"  Errors:    {entry['errors']}\n\n")

    logger.info(
        f"\nCleaning complete:\n"
        f"  Total rows:     {n_total}\n"
        f"  Corrupted:      {n_corrupted}\n"
        f"  Clean:          {n_clean}\n"
        f"  Log:            {log_path}"
    )

    # ------------------------------------------------------------------
    # Delete corrupted folders/files from disk
    # ------------------------------------------------------------------
    if n_corrupted > 0:
        logger.info("\nDeleting corrupted patch data from disk...")
        for entry in corrupted_rows:
            # Delete S2 folder
            _safe_delete_folder(Path(entry["s2_dir"]))
            # Delete S1 folder
            _safe_delete_folder(Path(entry["s1_dir"]))
            # Delete reference map file
            ref_p = Path(entry["ref_path"])
            if ref_p.suffix in (".tif", ".tiff"):
                _safe_delete_file(ref_p)
            else:
                # Try both extensions if ref_path was empty
                _safe_delete_file(
                    ref_root_p / f"{entry['reference_map_id']}.tif"
                )
                _safe_delete_file(
                    ref_root_p / f"{entry['reference_map_id']}.tiff"
                )
        logger.info("Disk cleanup complete.")

    # ------------------------------------------------------------------
    # Write cleaned CSV
    # ------------------------------------------------------------------
    clean_df = df.iloc[clean_rows].reset_index(drop=True)
    clean_df.to_csv(out_csv, index=False)
    logger.info(f"Cleaned CSV written: {out_csv}  ({len(clean_df)} rows)")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description="Validate and clean the dataset CSV, removing corrupted patches."
    )
    p.add_argument("--csv",     default=DEFAULT_CSV,     help="Path to original file.csv")
    p.add_argument("--s1",      default=DEFAULT_S1,      help="Root S1 directory")
    p.add_argument("--s2",      default=DEFAULT_S2,      help="Root S2 directory")
    p.add_argument("--ref",     default=DEFAULT_REF,     help="Root reference map directory")
    p.add_argument("--out_csv", default=DEFAULT_OUT_CSV, help="Output cleaned CSV path")
    p.add_argument("--log",     default=DEFAULT_LOG,     help="Output corruption log path")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    clean_dataset(
        csv_path=args.csv,
        s1_root=args.s1,
        s2_root=args.s2,
        ref_root=args.ref,
        out_csv=args.out_csv,
        log_path=args.log,
    )
