"""
model.py
========
Complete TerraFM + UPerNet segmentation model for S1+S2 LULC.

What this file does:
    Combines TerraFMEncoder (14-ch input) + UPerNetDecoder into a single
    nn.Module.  Provides build_model(), save_final_model(), load_final_model().

What goes in:   [B, 14, 224, 224]  (fused S1+S2 tensor)
What comes out: [B, 19, 224, 224]  logits  /  [B, 224, 224] predictions

FULL ARCHITECTURE FLOW:
-------------------------------------------------------------------
Input:    [B, 14, 224, 224]  (12 S2 + 2 S1, normalised)
              ↓
TerraFM-B patch embed (inflated to 14ch) → 196 tokens
          12 transformer blocks
              ↓
Extract   blocks [2, 5, 8, 11] → 4× [B, 196, 768]
              ↓
Reshape   each → [B, 768, 14, 14]
              ↓
UPerNet   lateral 1×1 → [B, 256, 14, 14] ×4
          PPM global context
          FPN merge
          upsample → [B, 256, 224, 224]
          1×1 head → [B, 19, 224, 224]
              ↓
Output:   [B, 19, 224, 224]  logits
Loss:     CE+Dice (ignore_index=255)
Pred:     argmax(dim=1) → [B, 224, 224]
-------------------------------------------------------------------
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import CFG
from terrafm_encoder import TerraFMEncoder
from segmentation_decoder import UPerNetDecoder

logger = logging.getLogger(__name__)


class TerraFMLULC(nn.Module):
    """
    End-to-end TerraFM LULC segmentation model (S1+S2 input).

    Args:
        model_size:   "base" or "large".
        num_classes:  Output classes (default 19).
        freeze_stage: 0=freeze encoder, 1=partial, 2=full.
    """

    def __init__(
        self,
        model_size: str = "base",
        num_classes: int = 19,
        freeze_stage: int = 0,
    ):
        super().__init__()
        self.model_size  = model_size
        self.num_classes = num_classes

        # Build encoder first — it loads TerraFM via the official terrafm.py
        # and sets embed_dim = 768 (ViT-B) or 1024 (ViT-L).
        self.encoder = TerraFMEncoder(
            model_size=model_size,
            freeze_stage=freeze_stage,
        )

        # Read the true embed_dim from the encoder after it has loaded
        # the checkpoint, so the decoder is sized correctly.
        embed_dim = self.encoder.embed_dim

        self.decoder = UPerNetDecoder(
            in_channels=embed_dim,
            num_scales=len(CFG.vit_feature_indices),
            decoder_channels=CFG.decoder_channels,
            num_classes=num_classes,
            dropout=CFG.decoder_dropout,
        )

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fused: [B, 14, H, W] float32 (normalised S2+S1 tensor).
        Returns:
            logits: [B, num_classes, H, W] float32.
        """
        H, W = fused.shape[-2:]
        if H != CFG.image_size or W != CFG.image_size:
            logger.warning(
                f"Input {H}×{W} ≠ expected {CFG.image_size}×{CFG.image_size}. "
                "Preprocessing should resize to 224×224 before passing to encoder."
            )

        features = self.encoder(fused)               # 4× [B, C, 14, 14]
        logits   = self.decoder(features, output_size=H)  # [B, 19, H, W]
        return logits

    def predict(self, fused: torch.Tensor) -> torch.Tensor:
        """Forward + argmax → [B, H, W] class IDs."""
        return self.forward(fused).argmax(dim=1)

    def set_freeze_stage(self, stage: int) -> None:
        self.encoder.set_freeze_stage(stage)

    def count_parameters(self) -> Dict[str, int]:
        enc       = sum(p.numel() for p in self.encoder.parameters())
        enc_train = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        dec       = sum(p.numel() for p in self.decoder.parameters())
        return {
            "encoder_total":     enc,
            "encoder_trainable": enc_train,
            "decoder_total":     dec,
            "total":             enc + dec,
            "total_trainable":   enc_train + dec,
        }


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(
    model_size: Optional[str] = None,
    num_classes: Optional[int] = None,
    freeze_stage: Optional[int] = None,
) -> TerraFMLULC:
    ms = model_size  or CFG.model_size
    nc = num_classes or CFG.num_classes
    fs = freeze_stage if freeze_stage is not None else CFG.freeze_stage

    model  = TerraFMLULC(model_size=ms, num_classes=nc, freeze_stage=fs)
    counts = model.count_parameters()
    logger.info(
        f"Model: TerraFM-{ms.upper()} + UPerNet  |  "
        f"encoder {counts['encoder_total']/1e6:.1f}M "
        f"(trainable {counts['encoder_trainable']/1e6:.1f}M)  |  "
        f"decoder {counts['decoder_total']/1e6:.1f}M  |  "
        f"total trainable {counts['total_trainable']/1e6:.1f}M"
    )
    return model


# ---------------------------------------------------------------------------
# Final model artifact: save / load
# ---------------------------------------------------------------------------

def save_final_model(
    model: TerraFMLULC,
    path: str,
    s2_mean: List[float],
    s2_std: List[float],
    s1_mean: List[float],
    s1_std: List[float],
    raw_to_train: Dict[int, int],
    class_names: List[str],
    class_colors: List[str],
) -> None:
    """
    Save a single self-contained model file for inference.

    Contains: architecture config + state_dict + all preprocessing metadata.
    The inference script can load this one file and run without any other
    config file.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    artifact = {
        "arch": {
            "model_size":         model.model_size,
            "num_classes":        model.num_classes,
            "image_size":         CFG.image_size,
            "patch_size":         CFG.patch_size,
            "total_in_channels":  CFG.total_in_channels,
            "s2_num_channels":    CFG.s2_num_channels,
            "s1_num_channels":    CFG.s1_num_channels,
            "decoder_channels":   CFG.decoder_channels,
            "decoder_dropout":    CFG.decoder_dropout,
            "vit_feature_indices": CFG.vit_feature_indices,
        },
        "state_dict":       model.state_dict(),
        "s2_mean":          s2_mean,
        "s2_std":           s2_std,
        "s1_mean":          s1_mean,
        "s1_std":           s1_std,
        "reflectance_scale": CFG.reflectance_scale,
        "raw_to_train":     {str(k): v for k, v in raw_to_train.items()},
        "class_names":      class_names,
        "class_colors":     class_colors,
        "ignore_index":     CFG.ignore_index,
    }
    torch.save(artifact, path)
    mb = os.path.getsize(path) / 1e6
    logger.info(f"Final model saved: {path}  ({mb:.1f} MB)")


def load_final_model(path: str, device: str = "cpu") -> Tuple[TerraFMLULC, dict]:
    """
    Load a model saved by save_final_model().

    Returns:
        (model, metadata)
        metadata keys: s2_mean, s2_std, s1_mean, s1_std,
                       raw_to_train, class_names, class_colors,
                       ignore_index, image_size, reflectance_scale
    """
    artifact = torch.load(path, map_location=device)
    arch     = artifact["arch"]

    model = TerraFMLULC(
        model_size=arch["model_size"],
        num_classes=arch["num_classes"],
        freeze_stage=2,   # fully unfrozen for inference
    )
    model.load_state_dict(artifact["state_dict"])
    model.to(device)
    model.eval()

    metadata = {
        "s2_mean":          artifact["s2_mean"],
        "s2_std":           artifact["s2_std"],
        "s1_mean":          artifact["s1_mean"],
        "s1_std":           artifact["s1_std"],
        "reflectance_scale": artifact["reflectance_scale"],
        "raw_to_train":     {int(k): v for k, v in artifact["raw_to_train"].items()},
        "class_names":      artifact["class_names"],
        "class_colors":     artifact["class_colors"],
        "ignore_index":     artifact["ignore_index"],
        "image_size":       arch["image_size"],
    }
    logger.info(
        f"Loaded: TerraFM-{arch['model_size'].upper()} + UPerNet, "
        f"{arch['num_classes']} classes, from {path}"
    )
    return model, metadata
