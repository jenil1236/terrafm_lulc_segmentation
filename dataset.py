"""
dataset.py
==========
PyTorch Dataset and DataLoader factory for S1+S2 LULC segmentation.

What this file does:
    S1S2LULCDataset loads one (s1+s2 fused tensor, mask) pair per
    __getitem__ call using the triplet index from file_clean.csv.
    build_dataloaders() returns train/val/test DataLoaders.

What goes in:
    - Cleaned CSV (file_clean.csv) with columns: s1_name, patch_id, reference_map_id
    - S2 patch directories, S1 patch directories, reference TIFF files
    - Normalization statistics (separate for S2 and S1)
    - Class mapping dict

What comes out:
    - Each item: (fused_tensor [14, H, W], mask [H, W])
      fused_tensor = cat([s2 [12, H, W], s1 [2, H, W]], dim=0)

How it connects:
    train.py calls build_dataloaders().
    dataset.py calls preprocessing.py for all TIFF reading.

NO CACHING:
    Every (S1, S2, mask) triplet is loaded fresh each epoch.
    With random geometric augmentation the same files are never processed
    identically twice across epochs. Caching 50k preprocessed samples
    would require ~120 GB on the Colab VM, which is impractical.
    Live TIFF loading with 2 DataLoader workers is fast enough for T4.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from config import CFG
from preprocessing import (
    discover_s2_bands,
    preprocess_s1_patch,
    preprocess_s2_patch,
    read_reference_mask,
)

logger = logging.getLogger(__name__)


class S1S2LULCDataset(Dataset):
    """
    Map-style dataset that yields fused S1+S2 tensors and LULC masks.

    Each item:
        fused:  torch.Tensor [14, H, W] float32  (12 S2 + 2 S1 channels)
        mask:   torch.Tensor  [H, W]    int64    (0..18 or ignore_index=255)

    Args:
        records:      List of dicts with keys s1_name, patch_id, reference_map_id.
        s2_root:      Root directory for S2 patches.
        s1_root:      Root directory for S1 patches.
        ref_root:     Root directory for reference TIFFs.
        s2_mean/std:  Per-channel S2 normalisation stats (12 values each).
        s1_mean/std:  Per-channel S1 normalisation stats (2 values each).
        raw_to_train: Raw class ID → contiguous train ID mapping.
        augment:      Whether to apply geometric augmentations.
        target_size:  Spatial size of output tensors (224 for TerraFM).
    """

    def __init__(
        self,
        records: List[Dict[str, str]],
        s2_root: str,
        s1_root: str,
        ref_root: str,
        s2_mean: List[float],
        s2_std: List[float],
        s1_mean: List[float],
        s1_std: List[float],
        raw_to_train: Dict[int, int],
        augment: bool = False,
        target_size: int = 224,
    ):
        self.records      = records
        self.s2_root      = Path(s2_root)
        self.s1_root      = Path(s1_root)
        self.ref_root     = Path(ref_root)
        self.s2_mean      = s2_mean
        self.s2_std       = s2_std
        self.s1_mean      = s1_mean
        self.s1_std       = s1_std
        self.raw_to_train = raw_to_train
        self.augment      = augment
        self.target_size  = target_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        rec      = self.records[idx]
        s1_name  = rec["s1_name"]
        patch_id = rec["patch_id"]
        ref_stem = rec["reference_map_id"]

        # ------------------------------------------------------------------
        # S2 preprocessing → [12, H, W]
        # ------------------------------------------------------------------
        s2_dir = str(self.s2_root / patch_id)
        try:
            s2_tensor, out_transform, out_crs = preprocess_s2_patch(
                s2_dir,
                norm_mean=self.s2_mean,
                norm_std=self.s2_std,
                target_size=self.target_size,
            )
        except Exception as e:
            logger.error(f"S2 load failed for '{patch_id}': {e}")
            return self._zero_sample()

        # ------------------------------------------------------------------
        # S1 preprocessing → [2, H, W]
        # Co-registered to the same grid as S2 (ref_transform / ref_crs).
        # ------------------------------------------------------------------
        s1_dir = str(self.s1_root / s1_name)
        try:
            s1_tensor = preprocess_s1_patch(
                s1_dir,
                norm_mean=self.s1_mean,
                norm_std=self.s1_std,
                target_size=self.target_size,
                ref_transform=out_transform,
                ref_crs=out_crs,
            )
        except Exception as e:
            logger.error(f"S1 load failed for '{s1_name}': {e}")
            return self._zero_sample()

        # ------------------------------------------------------------------
        # Reference mask → [H, W]
        # ------------------------------------------------------------------
        ref_path = self._find_ref_path(ref_stem)
        if ref_path is None:
            logger.warning(f"Reference map not found for '{ref_stem}'")
            mask_tensor = torch.full(
                (self.target_size, self.target_size),
                CFG.ignore_index, dtype=torch.long,
            )
        else:
            try:
                mask_np = read_reference_mask(
                    str(ref_path),
                    target_transform=out_transform,
                    target_crs=out_crs,
                    target_width=self.target_size,
                    target_height=self.target_size,
                    raw_to_train=self.raw_to_train,
                    ignore_index=CFG.ignore_index,
                )
                mask_tensor = torch.from_numpy(mask_np.astype(np.int64))
            except Exception as e:
                logger.error(f"Mask load failed for '{ref_stem}': {e}")
                mask_tensor = torch.full(
                    (self.target_size, self.target_size),
                    CFG.ignore_index, dtype=torch.long,
                )

        # ------------------------------------------------------------------
        # Fuse S2 + S1 along channel dimension → [14, H, W]
        # Channel order: [S2_B01, …, S2_B12, S1_VV, S1_VH]
        # ------------------------------------------------------------------
        fused = torch.cat([s2_tensor, s1_tensor], dim=0)   # [14, H, W]

        # ------------------------------------------------------------------
        # Geometric augmentation (same transform applied to both modalities
        # and the mask – spectral relationships are preserved)
        # ------------------------------------------------------------------
        if self.augment:
            fused, mask_tensor = _augment(fused, mask_tensor)

        return fused, mask_tensor

    def _find_ref_path(self, ref_stem: str) -> Optional[Path]:
        for ext in (".tif", ".tiff"):
            p = self.ref_root / f"{ref_stem}{ext}"
            if p.exists():
                return p
        return None

    def _zero_sample(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return an all-zero fused tensor and an all-ignore mask.
        Used as a safe fallback when a patch fails to load, so that
        training does not crash. The loss on this sample will be 0
        (all pixels are ignored) so gradients are unaffected.
        """
        fused = torch.zeros(CFG.total_in_channels, self.target_size, self.target_size)
        mask  = torch.full(
            (self.target_size, self.target_size),
            CFG.ignore_index, dtype=torch.long,
        )
        return fused, mask


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def _augment(
    fused: torch.Tensor, mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply geometric augmentations to the fused [C, H, W] tensor AND mask.

    Only physically meaningful transforms for satellite imagery:
      - Horizontal flip
      - Vertical flip
      - 90° rotation  (preserves geospatial structure)

    The SAME random decision is applied to every channel and the mask.
    Never apply spectral/colour transforms – that corrupts physical meaning.
    """
    if torch.rand(1).item() < CFG.aug_hflip_prob:
        fused = torch.flip(fused, dims=[2])
        mask  = torch.flip(mask,  dims=[1])

    if torch.rand(1).item() < CFG.aug_vflip_prob:
        fused = torch.flip(fused, dims=[1])
        mask  = torch.flip(mask,  dims=[0])

    if torch.rand(1).item() < CFG.aug_rotate90_prob:
        k     = torch.randint(1, 4, (1,)).item()
        fused = torch.rot90(fused, k=k, dims=[1, 2])
        mask  = torch.rot90(mask,  k=k, dims=[0, 1])

    return fused, mask


# ---------------------------------------------------------------------------
# CSV loading helper
# ---------------------------------------------------------------------------

def load_records_from_csv(csv_path: str) -> List[Dict[str, str]]:
    """
    Read a split CSV and return a list of record dicts.

    The CSV must have columns: s1_name, patch_id, reference_map_id.
    (This is the format written by prepare_dataset.py's split step.)
    """
    df = pd.read_csv(csv_path)
    required = {"s1_name", "patch_id", "reference_map_id"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Split CSV '{csv_path}' is missing columns: {missing}. "
            f"Found: {list(df.columns)}"
        )
    return df[["s1_name", "patch_id", "reference_map_id"]].astype(str).to_dict("records")


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def build_dataloaders(
    train_csv: str,
    val_csv: str,
    test_csv: str,
    s2_mean: List[float],
    s2_std: List[float],
    s1_mean: List[float],
    s1_std: List[float],
    raw_to_train: Dict[int, int],
    batch_size: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train, val, and test DataLoaders from split CSVs.

    Args:
        train/val/test_csv: Paths to split CSVs (s1_name, patch_id, reference_map_id).
        s2_mean/std:        S2 normalisation stats.
        s1_mean/std:        S1 normalisation stats.
        raw_to_train:       Class ID remapping dict.
        batch_size:         Override CFG.batch_size if set.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    bs = batch_size or CFG.batch_size

    train_records = load_records_from_csv(train_csv)
    val_records   = load_records_from_csv(val_csv)
    test_records  = load_records_from_csv(test_csv)

    logger.info(
        f"Dataset sizes: train={len(train_records)}, "
        f"val={len(val_records)}, test={len(test_records)}"
    )

    def _make_ds(records, augment):
        return S1S2LULCDataset(
            records=records,
            s2_root=CFG.s2_dir,
            s1_root=CFG.s1_dir,
            ref_root=CFG.ref_dir,
            s2_mean=s2_mean, s2_std=s2_std,
            s1_mean=s1_mean, s1_std=s1_std,
            raw_to_train=raw_to_train,
            augment=augment,
            target_size=CFG.image_size,
        )

    train_ds = _make_ds(train_records, augment=CFG.use_augmentation)
    val_ds   = _make_ds(val_records,   augment=False)
    test_ds  = _make_ds(test_records,  augment=False)

    pf = CFG.prefetch_factor if CFG.num_workers > 0 else None

    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True,
        num_workers=CFG.num_workers, pin_memory=CFG.pin_memory,
        prefetch_factor=pf, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False,
        num_workers=CFG.num_workers, pin_memory=CFG.pin_memory,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=bs, shuffle=False,
        num_workers=CFG.num_workers, pin_memory=CFG.pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
