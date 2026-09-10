"""
config.py
=========
Central configuration for the TerraFM LULC segmentation project.

What this file does:
    Defines every hyperparameter, path, class mapping, and preprocessing
    constant used across the entire pipeline. A single Config object is
    imported by every other module.

What goes in:  Nothing (source of truth).
What comes out: CFG singleton imported everywhere.
How it connects: Every module does `from config import CFG`.
"""

from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# ARCHITECTURAL NOTES
# ---------------------------------------------------------------------------
# CONFIRMED from official HuggingFace model card and GitHub repo:
#   - TerraFM is ViT-based, trained on Sentinel-1 SAR + Sentinel-2 optical
#   - Uses modality-specific patch embeddings + cross-attention fusion
#   - ViT-B = 768-dim / 12 blocks;  ViT-L = 1024-dim / 24 blocks
#   - Expected input: 224×224 spatial size
#   - S2: 12 channels (L2A band order).  S1: 2 channels (VV, VH).
#
# INPUT MODE (this project):
#   use_s1 = True  →  S1 (2ch) + S2 (12ch) fused → 14-channel input tensor.
#   The encoder receives a single [B, 14, 224, 224] tensor; the extra 2
#   channels beyond the original S2 patch-embedding weights are handled by
#   expanding the first projection layer (see terrafm_encoder.py).
# ---------------------------------------------------------------------------


@dataclass
class Config:
    # ------------------------------------------------------------------
    # Paths – override these in the notebook for your Colab environment
    # ------------------------------------------------------------------
    data_root: str = "/content/data"
    s2_dir: str = "/content/data/S2"
    s1_dir: str = "/content/data/S1"
    ref_dir: str = "/content/data/reference_maps_selected"
    csv_file: str = "/content/data/file.csv"
    output_dir: str = "/content/outputs"
    checkpoint_dir: str = "/content/outputs/checkpoints"
    results_dir: str = "/content/outputs/results"
    split_dir: str = "/content/outputs/splits"
    weights_dir: str = "/content/outputs/weights"

    # ------------------------------------------------------------------
    # TerraFM model
    # ------------------------------------------------------------------
    terrafm_hub_id: str = "MBZUAI/TerraFM"
    model_size: str = "base"   # "base" (T4-safe) or "large" (needs A100)

    # S1 + S2 fusion is now always enabled.
    # S1 contributes 2 channels (VV + VH); S2 contributes 12.
    # Total input channels fed to the encoder = 14.
    use_s1: bool = True
    s2_num_channels: int = 12
    s1_num_channels: int = 2
    # Total channels = s2_num_channels + s1_num_channels = 14
    @property
    def total_in_channels(self) -> int:
        return self.s2_num_channels + (self.s1_num_channels if self.use_s1 else 0)

    # CONFIRMED: 224×224 is the pre-training spatial size
    image_size: int = 224
    patch_size: int = 16
    num_patches_per_side: int = 14   # = image_size // patch_size

    # ------------------------------------------------------------------
    # CSV column names (matches the actual file.csv format)
    #   s1_name,patch_id,reference_map_id
    # ------------------------------------------------------------------
    csv_s1_col: str = "s1_name"
    csv_s2_col: str = "patch_id"           # S2 patch folder name
    csv_ref_col: str = "reference_map_id"  # reference TIFF name (no extension)
    patch_id_col: str = "patch_id"         # used as the unique row key

    # ------------------------------------------------------------------
    # Sentinel-2 band configuration
    # ESA Sentinel-2 L2A standard band order (natural sort of TIFFs):
    #   B01(60m) B02(10m) B03(10m) B04(10m) B05(20m) B06(20m)
    #   B07(20m) B08(10m) B8A(20m) B09(60m) B11(20m) B12(20m)
    # ------------------------------------------------------------------
    s2_band_names: List[str] = field(default_factory=lambda: [
        "B01", "B02", "B03", "B04", "B05", "B06",
        "B07", "B08", "B8A", "B09", "B11", "B12",
    ])
    s2_band_resolutions: List[int] = field(default_factory=lambda: [
        60, 10, 10, 10, 20, 20, 20, 10, 20, 60, 20, 20
    ])
    target_resolution_m: int = 10

    # ------------------------------------------------------------------
    # Sentinel-1 band configuration
    # S1 GRD products provide VV and VH polarisation bands.
    # Natural sort of TIFFs in the S1 patch folder → [VH, VV] or [VV, VH]
    # depending on naming. The order is consistent across all patches
    # (same acquisition product), so relative ordering is preserved.
    # ------------------------------------------------------------------
    s1_band_names: List[str] = field(default_factory=lambda: ["VV", "VH"])

    # ------------------------------------------------------------------
    # Normalization
    # S2: divide raw int16 DN by 10000 → [0,1] surface reflectance.
    # S1: convert to dB (10*log10(linear)) then standardize.
    #     Raw S1 GRD values are in linear power scale [0, ~1].
    # Training-set statistics are computed by prepare_dataset.py and
    # stored here; defaults are placeholders.
    # ------------------------------------------------------------------
    reflectance_scale: float = 10000.0   # S2 DN → reflectance

    # S2 normalization (12 channels) – filled by prepare_dataset.py
    s2_norm_mean: List[float] = field(default_factory=lambda: [0.0] * 12)
    s2_norm_std:  List[float] = field(default_factory=lambda: [1.0] * 12)

    # S1 normalization (2 channels) – filled by prepare_dataset.py
    s1_norm_mean: List[float] = field(default_factory=lambda: [0.0] * 2)
    s1_norm_std:  List[float] = field(default_factory=lambda: [1.0] * 2)

    norm_stats_file: str = "/content/outputs/norm_stats.json"

    # ------------------------------------------------------------------
    # Class mapping (19 classes + ignore index 255)
    # ------------------------------------------------------------------
    num_classes: int = 19
    ignore_index: int = 255
    raw_to_train: Dict[int, int] = field(default_factory=dict)

    # BigEarthNet 19-class names (index 0–18, matching class_to_number order)
    class_names: List[str] = field(default_factory=lambda: [
        "Urban fabric",                                                   # 0
        "Industrial or commercial units",                                 # 1
        "Arable land",                                                    # 2
        "Permanent crops",                                                # 3
        "Pastures",                                                       # 4
        "Complex cultivation patterns",                                   # 5
        "Land principally occupied by agriculture",                       # 6
        "Agro-forestry areas",                                            # 7
        "Broad-leaved forest",                                            # 8
        "Coniferous forest",                                              # 9
        "Mixed forest",                                                   # 10
        "Natural grassland and sparsely vegetated areas",                 # 11
        "Moors, heathland and sclerophyllous vegetation",                 # 12
        "Transitional woodland, shrub",                                   # 13
        "Beaches, dunes, sands",                                          # 14
        "Inland wetlands",                                                # 15
        "Coastal wetlands",                                               # 16
        "Inland waters",                                                  # 17
        "Marine waters",                                                  # 18
    ])

    # Fixed colors per BigEarthNet class (same order as class_names above)
    class_colors: List[str] = field(default_factory=lambda: [
        "#e41a1c",  # 0  Urban fabric
        "#984ea3",  # 1  Industrial or commercial units
        "#ffd92f",  # 2  Arable land
        "#ffad33",  # 3  Permanent crops
        "#78c679",  # 4  Pastures
        "#a6d854",  # 5  Complex cultivation patterns
        "#fdae61",  # 6  Land principally occupied by agriculture
        "#66c2a5",  # 7  Agro-forestry areas
        "#006d2c",  # 8  Broad-leaved forest
        "#238b45",  # 9  Coniferous forest
        "#31a354",  # 10 Mixed forest
        "#8c96c6",  # 11 Natural grassland
        "#9e9ac8",  # 12 Moors, heathland
        "#8856a7",  # 13 Transitional woodland, shrub
        "#fdd49e",  # 14 Beaches, dunes, sands
        "#74c476",  # 15 Inland wetlands
        "#41ae76",  # 16 Coastal wetlands
        "#2171b5",  # 17 Inland waters
        "#6baed6",  # 18 Marine waters
    ])
    class_mapping_file: str = "/content/outputs/class_mapping.json"

    # ------------------------------------------------------------------
    # Train/val/test split
    # There is NO pre-existing split column in file.csv.
    # We use tile-ID extracted from patch names for geographic splitting,
    # falling back to random if no tile pattern is detected.
    # ------------------------------------------------------------------
    train_ratio: float = 0.80
    val_ratio:   float = 0.10
    test_ratio:  float = 0.10
    split_seed:  int   = 42
    # Geographic grouping: try to extract Sentinel-2 tile code (e.g. "T33UUP")
    # from patch_id using a regex. If extraction succeeds, split by tile.
    tile_regex: str = r"T\d{2}[A-Z]{3}"   # matches e.g. T33UUP

    # ------------------------------------------------------------------
    # Training phases
    # ------------------------------------------------------------------
    phase0_samples:    int = 64
    phase0_epochs:     int = 30
    phase0_batch_size: int = 4

    phase1_samples:    int = 2000
    phase1_epochs:     int = 10
    phase1_batch_size: int = 4

    # Phase 2 (full training)
    # 50k samples / batch 4 = 12.5k steps/epoch.
    # Pretrained encoder → convergence faster than training from scratch.
    epochs:           int = 35
    batch_size:       int = 4
    grad_accum_steps: int = 4   # effective batch = 16

    # ------------------------------------------------------------------
    # Optimizer & scheduler
    # ------------------------------------------------------------------
    optimizer:     str   = "adamw"
    encoder_lr:    float = 5e-5   # pretrained → smaller LR
    decoder_lr:    float = 5e-4   # random init → larger LR
    weight_decay:  float = 0.05
    warmup_ratio:  float = 0.05
    min_lr:        float = 1e-6
    scheduler:     str   = "cosine"
    grad_clip:     float = 1.0

    # ------------------------------------------------------------------
    # Mixed precision (FP16 for T4; BF16 needs Ampere+)
    # ------------------------------------------------------------------
    amp_dtype: str = "fp16"

    # ------------------------------------------------------------------
    # Freeze / unfreeze strategy
    # ------------------------------------------------------------------
    freeze_stage:          int   = 0
    unfreeze_last_n_blocks: int  = 4
    lora_rank:             int   = 8
    lora_alpha:            float = 16.0
    lora_dropout:          float = 0.1

    # ------------------------------------------------------------------
    # Early stopping / checkpointing
    # ------------------------------------------------------------------
    early_stopping_patience: int = 8
    val_frequency:           int = 1
    checkpoint_frequency:    int = 1

    # ------------------------------------------------------------------
    # Decoder (UPerNet)
    # ------------------------------------------------------------------
    decoder_type:     str   = "upernet"
    decoder_channels: int   = 256
    decoder_dropout:  float = 0.1
    vit_feature_indices_base:  List[int] = field(default_factory=lambda: [2, 5, 8, 11])
    vit_feature_indices_large: List[int] = field(default_factory=lambda: [5, 11, 17, 23])

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    loss_type:         str   = "ce_dice"
    dice_weight:       float = 0.5
    ce_weight:         float = 0.5
    focal_gamma:       float = 2.0
    use_class_weights: bool  = True
    max_class_weight:  float = 10.0

    # ------------------------------------------------------------------
    # Data augmentation (geometric only – safe for multispectral)
    # ------------------------------------------------------------------
    use_augmentation:   bool  = True
    aug_hflip_prob:     float = 0.5
    aug_vflip_prob:     float = 0.5
    aug_rotate90_prob:  float = 0.5

    # ------------------------------------------------------------------
    # DataLoader
    # No caching: each (S1+S2+mask) triplet is unique per epoch because
    # augmentation randomness means the same files are never loaded twice
    # in identical form. Live TIFF loading is simpler and avoids the
    # ~120 GB storage requirement that caching 50k preprocessed patches
    # would require on the Colab VM.
    # ------------------------------------------------------------------
    num_workers:    int  = 2
    pin_memory:     bool = True
    prefetch_factor: int = 2

    # ------------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------------
    seed: int = 42

    # ------------------------------------------------------------------
    # Gradient checkpointing (optional memory saving, ~20% slower)
    # ------------------------------------------------------------------
    gradient_checkpointing: bool = False

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    inference_overlap:    float = 0.25
    inference_batch_size: int   = 1

    # ------------------------------------------------------------------
    # Resume training
    # ------------------------------------------------------------------
    resume_from: Optional[str] = None

    # ------------------------------------------------------------------
    # Final export path
    # ------------------------------------------------------------------
    final_model_path: str = "/content/outputs/terrafm_lulc_model.pth"

    # ------------------------------------------------------------------
    # Helper properties
    # ------------------------------------------------------------------
    @property
    def vit_embed_dim(self) -> int:
        # TerraFM-B patch embedding outputs 2304 (= 768 × 3 fused modalities)
        # Standard ViT-B is 768 — TerraFM is NOT a standard ViT-B here.
        # The encoder probes the actual checkpoint at load time and overrides
        # its own embed_dim, but model.py reads this property to size the decoder.
        return 2304 if self.model_size == "base" else 3072

    @property
    def vit_num_blocks(self) -> int:
        return 12 if self.model_size == "base" else 24

    @property
    def vit_feature_indices(self) -> List[int]:
        return (self.vit_feature_indices_base if self.model_size == "base"
                else self.vit_feature_indices_large)

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.grad_accum_steps

    def ensure_dirs(self) -> None:
        for d in [
            self.output_dir, self.checkpoint_dir, self.results_dir,
            self.split_dir, self.weights_dir,
            os.path.join(self.results_dir, "qualitative_predictions"),
        ]:
            os.makedirs(d, exist_ok=True)


CFG = Config()
