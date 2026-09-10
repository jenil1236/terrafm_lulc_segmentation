"""
terrafm_encoder.py
==================
TerraFM pretrained encoder wrapper for S1+S2 semantic segmentation.

What this file does:
    Loads TerraFM-B/L from the HuggingFace snapshot using the official
    terrafm.py class definition (TerraFM / terrafm_base / terrafm_large).
    Handles S1+S2 fused input by routing each modality through its correct
    patch embedding branch, then fusing before the transformer.
    Extracts multi-scale spatial features from intermediate ViT blocks.

What goes in:
    - Fused tensor [B, 14, 224, 224]  (first 12 = S2, last 2 = S1)

What comes out:
    - List of 4 feature maps, each [B, embed_dim, 14, 14]
      (embed_dim = 768 for TerraFM-B, 1024 for TerraFM-L)

CONFIRMED from official terrafm.py:
    - TerraFM(embed_dim=768) → standard ViT-B transformer at 768-dim
    - PatchEmbed routes by channel count:
        C == 2  → conv2d_s1  → 2304-dim tokens (NO projection)
        is_l2a  → conv2d_s2_l2a → 2304-dim → TokenProjection → 768-dim
        else    → conv2d_s2_l1c → 2304-dim → TokenProjection → 768-dim
    - Our fused [14, H, W] tensor is split: s2=[12ch], s1=[2ch]
    - We embed each separately, add them, then run the transformer.
    - get_intermediate_layers(x, n) returns the LAST n blocks' outputs.
"""

from __future__ import annotations

import importlib.util
import logging
import os
from functools import partial
from typing import List, Optional

import torch
import torch.nn as nn

from config import CFG

logger = logging.getLogger(__name__)


class TerraFMEncoder(nn.Module):
    """
    TerraFM backbone wrapper for fused S1+S2 input [B, 14, H, W].

    Splits the fused tensor into S2 (ch 0:12) and S1 (ch 12:14),
    embeds each through the correct PatchEmbed branch, sums them,
    then runs the full ViT transformer.

    Args:
        model_size:   "base" (embed_dim=768) or "large" (embed_dim=1024).
        freeze_stage: 0=freeze all, 1=partial, 2=unfreeze all.
    """

    def __init__(self, model_size: str = "base", freeze_stage: int = 0):
        super().__init__()
        self.model_size       = model_size
        self.embed_dim        = 768 if model_size == "base" else 1024
        self.num_blocks       = 12  if model_size == "base" else 24
        self.feature_indices  = (
            CFG.vit_feature_indices_base  if model_size == "base"
            else CFG.vit_feature_indices_large
        )
        self.patch_size       = CFG.patch_size
        self.image_size       = CFG.image_size
        self.num_patches_side = self.image_size // self.patch_size  # 14

        # Load official TerraFM model
        self.backbone = self._load_terrafm()

        # Apply initial freeze
        self.set_freeze_stage(freeze_stage)

        logger.info("TerraFM: using custom forward with S1+S2 split")

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_terrafm(self) -> nn.Module:
        logger.info(f"Loading TerraFM-{self.model_size} from {CFG.terrafm_hub_id}")
        backbone = self._load_from_snapshot()
        if backbone is None:
            raise RuntimeError(
                "Failed to load TerraFM.\n"
                f"  Hub ID: {CFG.terrafm_hub_id}\n"
                "  Ensure HuggingFace Hub is reachable."
            )
        return backbone

    def _load_from_snapshot(self) -> Optional[nn.Module]:
        """
        Download (or use cached) snapshot, import terrafm.py,
        instantiate TerraFM via the terrafm_base / terrafm_large factory,
        load weights with strict=False.
        """
        try:
            import glob
            from huggingface_hub import snapshot_download

            local_dir = snapshot_download(
                repo_id=CFG.terrafm_hub_id,
                cache_dir=CFG.weights_dir,
            )
            logger.info(f"Snapshot: {local_dir}")
            logger.info(f"Files: {sorted(os.listdir(local_dir))}")

            # Import terrafm.py dynamically
            terrafm_py = os.path.join(local_dir, "terrafm.py")
            if not os.path.exists(terrafm_py):
                raise FileNotFoundError(f"terrafm.py not found in {local_dir}")

            spec   = importlib.util.spec_from_file_location("terrafm_official", terrafm_py)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Instantiate using the official factory function
            # terrafm_base() → TerraFM(embed_dim=768, depth=12, num_heads=12, ...)
            if self.model_size == "base":
                model = module.terrafm_base(patch_size=self.patch_size)
            else:
                model = module.terrafm_large(patch_size=self.patch_size)

            logger.info(f"Instantiated via terrafm_{self.model_size}()")

            # Find checkpoint
            ckpts = sorted(
                glob.glob(os.path.join(local_dir, "*.pth")) +
                glob.glob(os.path.join(local_dir, "*.pt"))
            )
            if not ckpts:
                raise FileNotFoundError(f"No .pth/.pt in {local_dir}")

            # Pick the right checkpoint for the model size
            ckpt_path = ckpts[0]
            for c in ckpts:
                if self.model_size == "base" and "B" in os.path.basename(c).upper():
                    ckpt_path = c
                    break
                if self.model_size == "large" and "L" in os.path.basename(c).upper():
                    ckpt_path = c
                    break

            logger.info(f"Loading checkpoint: {ckpt_path}")
            state = torch.load(ckpt_path, map_location="cpu")
            state = state.get("model", state.get("state_dict", state))

            result = model.load_state_dict(state, strict=False)
            self._log_load_result(result, len(state))
            self._log_model_info(model)
            return model

        except Exception as e:
            logger.error(f"Snapshot load failed: {e}")
            return None

    def _log_load_result(self, result, total_keys: int) -> None:
        missing    = result.missing_keys
        unexpected = result.unexpected_keys
        if missing:
            logger.warning(f"Missing keys ({len(missing)}): {missing[:5]}")
        if unexpected:
            logger.warning(f"Unexpected keys ({len(unexpected)}): {unexpected[:5]}")
        ratio = len(missing) / max(total_keys, 1)
        if ratio > 0.10:
            raise RuntimeError(
                f"Too many missing keys: {len(missing)}/{total_keys} "
                f"({ratio*100:.1f}%). Wrong checkpoint or model_size."
            )
        logger.info(f"Weights: {total_keys - len(missing)}/{total_keys} matched.")

    def _log_model_info(self, model: nn.Module) -> None:
        total = sum(p.numel() for p in model.parameters()) / 1e6
        logger.info(f"  TerraFM params: {total:.1f}M")

    # ------------------------------------------------------------------
    # Freeze / unfreeze
    # ------------------------------------------------------------------

    def set_freeze_stage(self, stage: int) -> None:
        """
        Stage 0: Freeze everything (decoder-only training).
        Stage 1: Freeze all except last N blocks + patch embed.
        Stage 2: Unfreeze everything.
        """
        if stage == 0:
            for p in self.backbone.parameters():
                p.requires_grad = False
            logger.info("TerraFM: ALL frozen (Stage 0)")

        elif stage == 1:
            for p in self.backbone.parameters():
                p.requires_grad = False
            blocks = self.backbone.blocks
            n = CFG.unfreeze_last_n_blocks
            for blk in blocks[-n:]:
                for p in blk.parameters():
                    p.requires_grad = True
            # Unfreeze patch embed so S1/S2 conv weights can adapt
            for p in self.backbone.patch_embed.parameters():
                p.requires_grad = True
            # Unfreeze final norm
            for p in self.backbone.norm.parameters():
                p.requires_grad = True
            logger.info(f"TerraFM: last {n}/{len(blocks)} blocks + patch_embed unfrozen (Stage 1)")

        elif stage == 2:
            for p in self.backbone.parameters():
                p.requires_grad = True
            logger.info("TerraFM: ALL unfrozen (Stage 2)")

        trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        logger.info(f"  Trainable encoder params: {trainable/1e6:.1f}M")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, fused: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract multi-scale spatial features.

        Args:
            fused: [B, 14, H, W]  — channels 0:12 = S2 L2A, channels 12:14 = S1

        Strategy (derived from terrafm.py PatchEmbed.forward logic):
            1. Split fused → s2 [B,12,H,W] and s1 [B,2,H,W]
            2. Embed S2 via patch_embed(s2, is_l2a=True) → [B, N, embed_dim]
               (goes through conv2d_s2_l2a → TokenProjection → 768-dim)
            3. Embed S1 via patch_embed(s1) → [B, N, 2304-dim]
               S1 does NOT go through TokenProjection in the original code.
               We add a learned linear to project S1 tokens to embed_dim.
            4. Sum S2 and S1 token embeddings → [B, N, embed_dim]
            5. Add CLS token + positional encoding
            6. Run transformer blocks
            7. Return intermediate block outputs reshaped to [B, C, 14, 14]

        Returns:
            List of 4 tensors, each [B, 768, 14, 14].
        """
        B, C, H, W = fused.shape
        assert C == 14, f"Expected 14 channels, got {C}"

        s2 = fused[:, :12, :, :]   # S2 L2A channels
        s1 = fused[:, 12:, :, :]   # S1 VV+VH channels

        # --- S2 embedding: [B, N, 2304] → TokenProjection → [B, N, 768] ---
        x_s2 = self.backbone.patch_embed(s2, is_l2a=True)   # [B, N, embed_dim]

        # --- S1 embedding: [B, 2, H, W] → conv2d_s1 → [B, N, 2304] ---
        # The original forward returns 2304-dim for S1 (no projection).
        # We project it to embed_dim using a dedicated linear layer.
        x_s1_raw = self._embed_s1(s1)   # [B, N, embed_dim]

        # --- Fuse: sum the two modality embeddings ---
        x = x_s2 + x_s1_raw   # [B, N, embed_dim]

        # --- Add CLS token and positional encoding (from backbone) ---
        cls_tokens = self.backbone.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.backbone.interpolate_pos_encoding(x, W, H)
        x = self.backbone.pos_drop(x)

        # --- Run transformer blocks, collecting intermediate outputs ---
        # feature_indices are 0-indexed block positions (e.g. [2, 5, 8, 11])
        target_set = set(self.feature_indices)
        collected  = {}

        for i, blk in enumerate(self.backbone.blocks):
            x = blk(x)
            if i in target_set:
                collected[i] = x

        # Normalize the last collected output only
        # (applying norm to all is optional; we normalize the final one)
        last_idx = max(self.feature_indices)
        if last_idx in collected:
            collected[last_idx] = self.backbone.norm(collected[last_idx])

        # Return in order of feature_indices
        features = [collected[i] for i in self.feature_indices]
        return [self._to_spatial(f, B) for f in features]

    def _embed_s1(self, s1: torch.Tensor) -> torch.Tensor:
        """
        Embed S1 tokens using conv2d_s1, then project to embed_dim.

        The official PatchEmbed.forward() for S1 returns 2304-dim tokens
        (attn_dim = embed_dim * 3) without calling TokenProjection.
        We project to embed_dim using a lazily-created linear layer so
        the first call initialises it to the correct size.
        """
        # conv2d_s1: [B, 2, H, W] → [B, attn_dim, H/p, W/p] → [B, N, attn_dim]
        pe = self.backbone.patch_embed
        x  = pe.conv2d_s1(s1).flatten(2).transpose(1, 2)   # [B, N, 2304]
        x  = x + pe.s1_embed                                 # add modality embed

        # Project 2304 → embed_dim (768) using the TokenProjection already
        # present in the patch embed (it was designed for this purpose)
        x = pe.projection(x)   # [B, N, embed_dim]
        return x

    def _to_spatial(self, tokens: torch.Tensor, B: int) -> torch.Tensor:
        """
        Reshape block output [B, N+1, C] → spatial map [B, C, H, W].
        Strips the CLS token (position 0).
        """
        if tokens.dim() == 2:
            tokens = tokens.unsqueeze(0)
        # Strip CLS token
        patch_tokens = tokens[:, 1:, :]   # [B, N, C]
        N = patch_tokens.shape[1]
        expected = self.num_patches_side ** 2   # 196
        if N != expected:
            logger.warning(f"Token count {N} ≠ {expected}; taking first {expected}.")
            patch_tokens = patch_tokens[:, :expected, :]
        feat = patch_tokens.permute(0, 2, 1).reshape(
            B, self.embed_dim, self.num_patches_side, self.num_patches_side
        )
        return feat.contiguous()
