"""
segmentation_decoder.py
=======================
UPerNet segmentation decoder compatible with TerraFM's ViT feature representation.

What this file does:
    Implements UPerNet (Unified Perceptual Parsing Network) decoder that takes
    the 4-scale feature maps from TerraFM and produces dense 19-class logits.

What goes in:
    - 4 feature maps from TerraFM encoder, each [B, embed_dim, 14, 14]

What comes out:
    - Logit tensor [B, 19, H_out, W_out] (H_out = W_out = image_size = 224)

How it connects:
    model.py combines TerraFMEncoder + UPerNetDecoder into TerraFMLULC.

DECODER CHOICE RATIONALE:
-------------------------------------------------------------------
Candidate      | Pros                         | Cons              | T4 Cost | Decision
---------------|------------------------------|-------------------|---------|----------
Simple U-Net   | Simple                       | Needs CNN encoder | Low     | ✗ (ViT mismatch)
FPN            | Multi-scale, lightweight     | Less context      | Low     | OK fallback
UPerNet        | Best for ViT, multi-scale,   | Slightly heavier  | Medium  | ✓ CHOSEN
               | PPM captures global context  |                   |         |
SegFormer MLP  | Very lightweight, ViT-native | Less global pool  | Low     | OK fallback
DeepLab ASPP   | Good for boundary            | Needs stride conv | Medium  | Less suited
Mask2Former    | SOTA                         | Heavy, complex    | High    | ✗ T4 too tight

CHOSEN: UPerNet
WHY:
  1. UPerNet was specifically designed for multi-level vision transformer features
     (the original paper proposes it for parsing, widely adopted for segmentation).
  2. The Pyramid Pooling Module (PPM) in UPerNet's head adds global context,
     which helps LULC classes that depend on landscape-level patterns.
  3. It handles the flat 14×14 ViT token maps naturally: each scale is bilinearly
     upsampled to 1/4 of the input size (56×56 for 224/4), merged via FPN-style
     lateral connections, then further upsampled to full resolution.
  4. Computationally moderate: adds ~15-20M params, fits T4 comfortably.
  5. Same decoder used by BEiT-B for semantic segmentation tasks.
-------------------------------------------------------------------
"""

from __future__ import annotations

import logging
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import CFG

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ConvBNReLU(nn.Sequential):
    """Conv2d → BatchNorm2d → ReLU (standard building block)."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, padding: int = 1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class PPM(nn.Module):
    """
    Pyramid Pooling Module from PSPNet.
    Pools features at 4 scales (1×1, 2×2, 3×3, 6×6), upsample back and
    concatenate. Captures global context at multiple granularities.
    """

    def __init__(self, in_channels: int, pool_channels: int = 128):
        super().__init__()
        self.pool_sizes = [1, 2, 3, 6]
        self.pools = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(ps),
                ConvBNReLU(in_channels, pool_channels, kernel=1, padding=0),
            )
            for ps in self.pool_sizes
        ])
        self.bottleneck = ConvBNReLU(
            in_channels + pool_channels * len(self.pool_sizes),
            in_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2:]
        parts = [x]
        for pool in self.pools:
            p = pool(x)
            p = F.interpolate(p, size=(H, W), mode="bilinear", align_corners=False)
            parts.append(p)
        return self.bottleneck(torch.cat(parts, dim=1))


# ---------------------------------------------------------------------------
# UPerNet Decoder
# ---------------------------------------------------------------------------

class UPerNetDecoder(nn.Module):
    """
    UPerNet decoder for ViT-style multi-scale features.

    Architecture:
        4× [B, C, 14, 14]  (C = 768 for ViT-B, 1024 for ViT-L)
            ↓
        Lateral projections → each to decoder_channels (256)
            ↓
        FPN top-down pathway (upsample + add)
            ↓
        PPM on deepest feature
            ↓
        Fuse all 4 scales at 1/4 resolution (56×56 for 224/4)
            ↓
        Head conv + dropout + upsample to full resolution (224×224)
            ↓
        [B, num_classes, 224, 224]

    Args:
        in_channels:    embed_dim of the encoder (768 or 1024).
        num_scales:     Number of feature scales (default 4).
        decoder_channels: Width of all intermediate layers (default 256).
        num_classes:    Number of output classes.
        dropout:        Dropout rate before the final classification layer.
    """

    def __init__(
        self,
        in_channels: int,
        num_scales: int = 4,
        decoder_channels: int = 256,
        num_classes: int = 19,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_scales = num_scales
        self.decoder_channels = decoder_channels

        # Lateral projections: reduce embed_dim to decoder_channels
        self.lateral_convs = nn.ModuleList([
            ConvBNReLU(in_channels, decoder_channels, kernel=1, padding=0)
            for _ in range(num_scales)
        ])

        # FPN top-down output convolutions
        self.fpn_convs = nn.ModuleList([
            ConvBNReLU(decoder_channels, decoder_channels)
            for _ in range(num_scales)
        ])

        # PPM on the deepest (most semantic) scale
        self.ppm = PPM(decoder_channels, pool_channels=decoder_channels // 4)

        # Fusion: concatenate all FPN scales → reduce
        self.fuse = ConvBNReLU(decoder_channels * num_scales, decoder_channels)

        # Segmentation head
        self.head = nn.Sequential(
            ConvBNReLU(decoder_channels, decoder_channels),
            nn.Dropout2d(dropout),
            nn.Conv2d(decoder_channels, num_classes, kernel_size=1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize decoder weights (encoder uses pretrained weights)."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, features: List[torch.Tensor], output_size: int = 224) -> torch.Tensor:
        """
        Args:
            features:    List of 4 tensors from TerraFM, each [B, C, 14, 14].
                         Ordered shallow → deep.
            output_size: Target spatial size of the output logits.

        Returns:
            logits: [B, num_classes, output_size, output_size]
        """
        # Step 1: Lateral projections – uniform channel width
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]

        # Step 2: Apply PPM to the deepest feature for global context
        laterals[-1] = self.ppm(laterals[-1])

        # Step 3: FPN top-down pathway
        # Start from deepest and progressively upsample + add shallower features
        fpn_outs = [None] * self.num_scales
        fpn_outs[-1] = self.fpn_convs[-1](laterals[-1])

        for i in range(self.num_scales - 2, -1, -1):
            upsampled = F.interpolate(
                fpn_outs[i + 1],
                size=laterals[i].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            fpn_outs[i] = self.fpn_convs[i](laterals[i] + upsampled)

        # Step 4: Upsample all FPN outputs to the same spatial size as
        # the shallowest feature (largest spatial map = 14×14 here).
        # Then concatenate and fuse.
        target_hw = fpn_outs[0].shape[-2:]
        aligned = [
            F.interpolate(f, size=target_hw, mode="bilinear", align_corners=False)
            for f in fpn_outs
        ]
        fused = self.fuse(torch.cat(aligned, dim=1))  # [B, C, 14, 14]

        # Step 5: Upsample to full output resolution (224×224)
        fused = F.interpolate(
            fused, size=(output_size, output_size),
            mode="bilinear", align_corners=False,
        )

        # Step 6: Segmentation head → [B, num_classes, H, W]
        logits = self.head(fused)
        return logits
