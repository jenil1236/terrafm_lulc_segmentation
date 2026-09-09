"""
preprocessing.py
================
Geospatial preprocessing for Sentinel-2 and Sentinel-1 patches.

What this file does:
    - Reads 12 S2 TIFFs from an S2 patch folder → [12, H, W] float32 tensor
    - Reads 2  S1 TIFFs from an S1 patch folder → [ 2, H, W] float32 tensor
    - Aligns all bands to a common 10 m grid at 224×224
    - Converts S2 DN → surface reflectance;  S1 linear → dB
    - Normalises each modality with training-set statistics
    - Reads and reprojects reference masks (nearest-neighbour, categorical)
    - Computes per-channel normalization statistics over the training set

What goes in:
    S2 patch dir, S1 patch dir, norm stats, target size

What comes out:
    s2_tensor:  [12, 224, 224] float32
    s1_tensor:  [ 2, 224, 224] float32
    (the caller in dataset.py concatenates them → [14, 224, 224])
    out_transform, out_crs   for GeoTIFF output during inference

How it connects:
    Called by dataset.py (training) and inference.py (prediction).

INTERPOLATION CHOICES:
    Reflectance / SAR (continuous)  → bilinear resampling
    Categorical reference masks     → nearest-neighbour (never bilinear)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import reproject
import torch

from config import CFG

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Band discovery helpers
# ---------------------------------------------------------------------------

def _natural_sort_key(path: Path) -> list:
    import re
    parts = re.split(r"(\d+)", path.stem)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def discover_s2_bands(patch_dir: str) -> List[Path]:
    """
    Return sorted list of 12 S2 TIFF paths in a patch directory.

    Raises ValueError if not exactly 12 TIFFs are found.
    """
    patch_path = Path(patch_dir)
    tiffs = sorted(
        list(patch_path.glob("*.tif")) + list(patch_path.glob("*.tiff")),
        key=_natural_sort_key,
    )
    if len(tiffs) != CFG.s2_num_channels:
        raise ValueError(
            f"Expected {CFG.s2_num_channels} S2 TIFFs in '{patch_dir}', "
            f"found {len(tiffs)}: {[p.name for p in tiffs]}"
        )
    return tiffs


def discover_s1_bands(patch_dir: str) -> List[Path]:
    """
    Return sorted list of S1 TIFF paths (expected: 2, for VV and VH).

    Raises ValueError if no TIFFs are found.
    """
    patch_path = Path(patch_dir)
    tiffs = sorted(
        list(patch_path.glob("*.tif")) + list(patch_path.glob("*.tiff")),
        key=_natural_sort_key,
    )
    if len(tiffs) == 0:
        raise ValueError(f"No S1 TIFFs found in '{patch_dir}'")
    if len(tiffs) != CFG.s1_num_channels:
        logger.warning(
            f"S1 patch '{patch_dir}' has {len(tiffs)} TIFFs "
            f"(expected {CFG.s1_num_channels}). Using first {CFG.s1_num_channels}."
        )
    return tiffs[: CFG.s1_num_channels]


# ---------------------------------------------------------------------------
# Low-level TIFF reader with resampling
# ---------------------------------------------------------------------------

def _read_and_align_band(
    band_path: Path,
    dst_transform,
    dst_crs,
    dst_width: int,
    dst_height: int,
    resampling: Resampling = Resampling.bilinear,
) -> np.ndarray:
    """
    Read one TIFF band and reproject/resample to the target grid.

    Returns float32 array [dst_height, dst_width].
    NoData pixels are set to NaN.
    """
    dest = np.full((dst_height, dst_width), fill_value=np.nan, dtype=np.float32)
    with rasterio.open(band_path) as src:
        nodata_val = src.nodata
        reproject(
            source=rasterio.band(src, 1),
            destination=dest,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=resampling,
            src_nodata=nodata_val,
            dst_nodata=np.nan,
        )
    if nodata_val is not None:
        dest[dest == nodata_val] = np.nan
    return dest


def _output_transform(
    ref_path: Path, target_size: int
) -> Tuple[object, object, int, int]:
    """
    Compute an output Affine transform and CRS for target_size × target_size
    that preserves the spatial extent of the reference band.
    Also returns original (height, width) for reference.
    """
    with rasterio.open(ref_path) as src:
        crs = src.crs
        t   = src.transform
        h, w = src.height, src.width

    left   = t.c
    top    = t.f
    right  = left + t.a * w
    bottom = top  + t.e * h
    out_t  = from_bounds(left, bottom, right, top, target_size, target_size)
    return out_t, crs, h, w


# ---------------------------------------------------------------------------
# Sentinel-2 preprocessing
# ---------------------------------------------------------------------------

def preprocess_s2_patch(
    patch_dir: str,
    norm_mean: List[float],
    norm_std: List[float],
    target_size: int = 224,
) -> Tuple[torch.Tensor, object, object]:
    """
    Full preprocessing pipeline for one Sentinel-2 patch.

    Pipeline:
        discover bands
        → reproject each to 10 m reference grid at target_size × target_size
            (bilinear – appropriate for continuous reflectance)
        → divide by reflectance_scale (10000) → [0, 1] reflectance
        → nan_to_num + clip to [-0.1, 1.2]
        → (x - mean) / std  channel-wise normalization
        → torch.Tensor [12, target_size, target_size]

    Returns:
        tensor:    [12, target_size, target_size] float32
        transform: rasterio Affine for the output grid
        crs:       rasterio CRS for the output grid
    """
    band_paths = discover_s2_bands(patch_dir)

    # Use B02 (index 1, native 10 m) as the spatial reference
    out_transform, out_crs, _, _ = _output_transform(band_paths[1], target_size)

    stacked = np.zeros((CFG.s2_num_channels, target_size, target_size), dtype=np.float32)
    for i, bp in enumerate(band_paths):
        stacked[i] = _read_and_align_band(
            bp, out_transform, out_crs, target_size, target_size,
            resampling=Resampling.bilinear,
        )

    # DN → reflectance
    stacked /= CFG.reflectance_scale

    # NoData / invalid handling
    stacked = np.nan_to_num(stacked, nan=0.0, posinf=1.0, neginf=0.0)
    stacked = np.clip(stacked, -0.1, 1.2)

    # Channel-wise z-normalisation
    mean = np.array(norm_mean, dtype=np.float32).reshape(12, 1, 1)
    std  = np.array(norm_std,  dtype=np.float32).reshape(12, 1, 1)
    std  = np.where(std < 1e-6, 1.0, std)
    stacked = (stacked - mean) / std

    return torch.from_numpy(stacked).float(), out_transform, out_crs


# ---------------------------------------------------------------------------
# Sentinel-1 preprocessing
# ---------------------------------------------------------------------------

def preprocess_s1_patch(
    patch_dir: str,
    norm_mean: List[float],
    norm_std: List[float],
    target_size: int = 224,
    ref_transform=None,
    ref_crs=None,
) -> torch.Tensor:
    """
    Full preprocessing pipeline for one Sentinel-1 patch.

    Pipeline:
        discover bands (VV, VH)
        → reproject each to match the S2 output grid
            (bilinear – appropriate for continuous SAR power values)
        → convert linear power → dB:  10 * log10(max(x, 1e-10))
            WHY dB: SAR backscatter spans several orders of magnitude in
            linear scale. Log (dB) compresses the range and makes the
            distribution more Gaussian – matching the assumption of
            z-normalisation better than raw linear values.
        → clip to [-30, 0] dB  (physically valid range for Sentinel-1 GRD)
        → nan_to_num + fill with min dB for NoData pixels
        → (x - mean) / std  channel-wise normalisation
        → torch.Tensor [2, target_size, target_size]

    Args:
        patch_dir:     S1 patch directory (contains VV.tif and VH.tif).
        norm_mean:     Per-channel mean for S1 (2 values).
        norm_std:      Per-channel std  for S1 (2 values).
        target_size:   Output spatial size.
        ref_transform: Affine transform from the S2 preprocessing (use same grid).
        ref_crs:       CRS from the S2 preprocessing.

    Returns:
        tensor: [2, target_size, target_size] float32
    """
    band_paths = discover_s1_bands(patch_dir)

    # If S2 transform is provided, co-register S1 exactly to the S2 grid.
    # Otherwise use the S1 reference band's own transform.
    if ref_transform is not None and ref_crs is not None:
        out_transform = ref_transform
        out_crs       = ref_crs
    else:
        out_transform, out_crs, _, _ = _output_transform(band_paths[0], target_size)

    stacked = np.zeros((CFG.s1_num_channels, target_size, target_size), dtype=np.float32)
    for i, bp in enumerate(band_paths):
        stacked[i] = _read_and_align_band(
            bp, out_transform, out_crs, target_size, target_size,
            resampling=Resampling.bilinear,
        )

    # Convert linear power to dB
    # NaN / negative values become the floor clip value (-30 dB) after log
    stacked = np.nan_to_num(stacked, nan=1e-10, posinf=1.0, neginf=1e-10)
    stacked = np.clip(stacked, 1e-10, None)            # avoid log(0)
    stacked = 10.0 * np.log10(stacked)                 # linear → dB

    # Physically valid Sentinel-1 GRD range: roughly [-30, 0] dB
    stacked = np.clip(stacked, -30.0, 0.0)

    # z-normalisation
    mean = np.array(norm_mean, dtype=np.float32).reshape(2, 1, 1)
    std  = np.array(norm_std,  dtype=np.float32).reshape(2, 1, 1)
    std  = np.where(std < 1e-6, 1.0, std)
    stacked = (stacked - mean) / std

    return torch.from_numpy(stacked).float()


# ---------------------------------------------------------------------------
# Reference mask reader
# ---------------------------------------------------------------------------

def read_reference_mask(
    mask_path: str,
    target_transform,
    target_crs,
    target_width: int,
    target_height: int,
    raw_to_train: Dict[int, int],
    ignore_index: int = 255,
) -> np.ndarray:
    """
    Read and remap a reference TIFF to contiguous training class IDs.

    Uses nearest-neighbour reprojection (mandatory for categorical data –
    bilinear would produce fractional class IDs that corrupt the labels).

    Returns:
        uint8 [H, W] array: values in {0..N-1, ignore_index}.
    """
    with rasterio.open(mask_path) as src:
        dest = np.zeros((target_height, target_width), dtype=np.uint8)
        reproject(
            source=rasterio.band(src, 1),
            destination=dest,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=target_transform,
            dst_crs=target_crs,
            resampling=Resampling.nearest,
            src_nodata=src.nodata,
            dst_nodata=ignore_index,
        )

    # Remap raw IDs → contiguous training IDs
    output = np.full_like(dest, fill_value=ignore_index, dtype=np.uint8)
    for raw_id, train_id in raw_to_train.items():
        output[dest == raw_id] = train_id

    return output


# ---------------------------------------------------------------------------
# Normalization statistics computation (training set only)
# ---------------------------------------------------------------------------

def compute_s2_normalization_stats(
    patch_dirs: List[str],
    max_samples: int = 5000,
    seed: int = 42,
) -> Tuple[List[float], List[float]]:
    """
    Compute per-channel mean and std for S2 from the TRAINING set only.
    Uses Welford's online algorithm to keep memory low.

    Returns (mean_list, std_list) each of length 12.
    """
    rng = np.random.default_rng(seed)
    if len(patch_dirs) > max_samples:
        idx = rng.choice(len(patch_dirs), size=max_samples, replace=False)
        patch_dirs = [patch_dirs[i] for i in idx]

    n    = 0
    mean = np.zeros(CFG.s2_num_channels, dtype=np.float64)
    M2   = np.zeros(CFG.s2_num_channels, dtype=np.float64)

    for patch_dir in patch_dirs:
        try:
            band_paths = discover_s2_bands(patch_dir)
        except ValueError as e:
            logger.warning(f"Skipping S2 {patch_dir}: {e}")
            continue

        for c, bp in enumerate(band_paths):
            try:
                with rasterio.open(bp) as src:
                    data    = src.read(1).astype(np.float64)
                    nodata  = src.nodata
                    if nodata is not None:
                        data = data[data != nodata]
                    data = data / CFG.reflectance_scale
                    data = data[(data >= -0.1) & (data <= 1.2)]
                    if data.size == 0:
                        continue
                    n_c   = data.size
                    delta = data - mean[c]
                    mean[c] += delta.sum() / (n + n_c)
                    M2[c]   += ((data - mean[c]) * delta).sum()
                    if c == 0:
                        n += n_c
            except Exception as e:
                logger.warning(f"  S2 band {c} error in {patch_dir}: {e}")

    std = np.sqrt(M2 / max(n - 1, 1))
    return mean.tolist(), std.tolist()


def compute_s1_normalization_stats(
    patch_dirs: List[str],
    max_samples: int = 5000,
    seed: int = 42,
) -> Tuple[List[float], List[float]]:
    """
    Compute per-channel mean and std for S1 (in dB) from the TRAINING set only.
    Values are converted to dB before statistics are computed, matching
    the preprocessing applied at training/inference time.

    Returns (mean_list, std_list) each of length 2.
    """
    rng = np.random.default_rng(seed)
    if len(patch_dirs) > max_samples:
        idx = rng.choice(len(patch_dirs), size=max_samples, replace=False)
        patch_dirs = [patch_dirs[i] for i in idx]

    n    = 0
    mean = np.zeros(CFG.s1_num_channels, dtype=np.float64)
    M2   = np.zeros(CFG.s1_num_channels, dtype=np.float64)

    for patch_dir in patch_dirs:
        try:
            band_paths = discover_s1_bands(patch_dir)
        except ValueError as e:
            logger.warning(f"Skipping S1 {patch_dir}: {e}")
            continue

        for c, bp in enumerate(band_paths):
            try:
                with rasterio.open(bp) as src:
                    data   = src.read(1).astype(np.float64)
                    nodata = src.nodata
                    if nodata is not None:
                        data = data[data != nodata]
                    data = data[data > 0]              # remove zeros
                    data = 10.0 * np.log10(data)       # linear → dB
                    data = data[(data >= -30.0) & (data <= 0.0)]
                    if data.size == 0:
                        continue
                    n_c   = data.size
                    delta = data - mean[c]
                    mean[c] += delta.sum() / (n + n_c)
                    M2[c]   += ((data - mean[c]) * delta).sum()
                    if c == 0:
                        n += n_c
            except Exception as e:
                logger.warning(f"  S1 band {c} error in {patch_dir}: {e}")

    std = np.sqrt(M2 / max(n - 1, 1))
    return mean.tolist(), std.tolist()


# ---------------------------------------------------------------------------
# Class ID discovery
# ---------------------------------------------------------------------------

def discover_class_ids(reference_paths: List[str]) -> Dict[int, int]:
    """
    Scan a sample of reference TIFFs and build raw_id → train_id mapping.

    Returns dict mapping raw IDs (sorted) → contiguous 0..N-1.
    """
    unique_ids: set = set()
    sample = reference_paths[: min(500, len(reference_paths))]

    for path in sample:
        try:
            with rasterio.open(path) as src:
                data   = src.read(1)
                nodata = src.nodata
                valid  = data if nodata is None else data[data != nodata]
                unique_ids.update(np.unique(valid).tolist())
        except Exception as e:
            logger.warning(f"Could not read {path}: {e}")

    sorted_ids = sorted(unique_ids)
    if len(sorted_ids) > CFG.num_classes:
        logger.warning(
            f"Found {len(sorted_ids)} unique class IDs but "
            f"num_classes={CFG.num_classes}. Extra IDs → ignore_index."
        )
    return {raw: train for train, raw in enumerate(sorted_ids[: CFG.num_classes])}
