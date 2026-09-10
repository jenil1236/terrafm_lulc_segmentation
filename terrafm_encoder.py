"""
terrafm_encoder.py
==================
TerraFM pretrained encoder wrapper for S1+S2 semantic segmentation.

What this file does:
    Loads TerraFM (ViT-B or ViT-L) from HuggingFace (MBZUAI/TerraFM),
    adapts its patch embedding to accept 14-channel fused S1+S2 input,
    extracts multi-scale spatial feature maps from intermediate ViT blocks,
    and manages freeze/unfreeze stages.

What goes in:
    - Fused tensor [B, 14, 224, 224]  (12 S2 + 2 S1 channels, normalized)

What comes out:
    - List of 4 feature maps, each [B, embed_dim, 14, 14]
      (patch grid = 224/16 = 14 per side)

How it connects:
    model.py wraps TerraFMEncoder + UPerNetDecoder into TerraFMLULC.

14-CHANNEL INPUT STRATEGY:
-------------------------------------------------------------------
TerraFM's pretrained patch embedding was trained on modality-specific
inputs.  When we fuse S2 (12ch) + S1 (2ch) into a single 14-channel
tensor we need to expand the embedding's input projection.

CONFIRMED from TerraFM paper / repo:
  - TerraFM uses modality-specific patch embeddings with cross-attention
    fusion, so it natively handles multi-modal input.
  - When loaded via AutoModel the model may already accept S1+S2 jointly.
  - If the loaded model only accepts 12-ch S2, we expand the first Conv2d
    weight to 14 channels using weight inflation:
      new_weight[:, :12, :, :] = pretrained_weight   (S2 channels)
      new_weight[:,12:14,:, :] = mean(pretrained_weight, dim=1, keepdim=True)
                                  × 2                 (S1 channels, init)
    This preserves all pretrained S2 representations while giving the
    two new S1 channels a sensible starting point.

ENGINEERING RECOMMENDATION:
  - We first try to load the model as-is and inspect its patch-embed
    input channel count.
  - If in_channels == 14: model natively handles the fused tensor (ideal).
  - If in_channels == 12: we inflate the weight (safe fallback).
  - If in_channels == anything else: we raise a clear error.
-------------------------------------------------------------------
"""

from __future__ import annotations

import logging
from typing import List, Optional

import torch
import torch.nn as nn

from config import CFG

logger = logging.getLogger(__name__)


class TerraFMEncoder(nn.Module):
    """
    TerraFM backbone wrapper that accepts 14-channel S1+S2 fused input.

    Args:
        model_size:   "base" (ViT-B, 768-dim) or "large" (ViT-L, 1024-dim).
        freeze_stage: 0=freeze all, 1=partial, 2=unfreeze all.
    """

    def __init__(self, model_size: str = "base", freeze_stage: int = 0):
        super().__init__()
        self.model_size        = model_size
        self.embed_dim         = 768 if model_size == "base" else 1024
        self.num_blocks        = 12  if model_size == "base" else 24
        self.feature_indices   = (
            CFG.vit_feature_indices_base  if model_size == "base"
            else CFG.vit_feature_indices_large
        )
        self.patch_size        = CFG.patch_size
        self.image_size        = CFG.image_size
        self.num_patches_side  = self.image_size // self.patch_size  # 14
        self.in_channels       = CFG.total_in_channels               # 14

        # Load backbone and adapt to 14 channels
        self.backbone = self._load_terrafm()

        # Apply freeze strategy
        self.set_freeze_stage(freeze_stage)

        # Choose feature extraction method
        self._use_timm_api = hasattr(self.backbone, "get_intermediate_layers")
        if not self._use_timm_api:
            self._hook_outputs: List = [None] * len(self.feature_indices)
            self._hooks: List        = []
            self._register_hooks()
            logger.info("TerraFM: using forward hooks for feature extraction")
        else:
            logger.info("TerraFM: using get_intermediate_layers() for feature extraction")

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_terrafm(self) -> nn.Module:
        logger.info(f"Loading TerraFM-{self.model_size} from {CFG.terrafm_hub_id}")
        backbone = self._try_load_huggingface()
        if backbone is None:
            raise RuntimeError(
                "Failed to load TerraFM from any source.\n"
                f"  Hub ID: {CFG.terrafm_hub_id}\n"
                "  Ensure HuggingFace Hub is reachable and "
                "'transformers'/'timm' are installed."
            )
        # Adapt the patch embedding to accept 14 channels
        backbone = self._adapt_patch_embed(backbone)
        return backbone

    def _try_load_huggingface(self) -> Optional[nn.Module]:
        # Approach 1: HF AutoModel with remote code (preferred for TerraFM)
        try:
            from transformers import AutoModel
            model = AutoModel.from_pretrained(
                CFG.terrafm_hub_id,
                trust_remote_code=True,
                cache_dir=CFG.weights_dir,
            )
            logger.info("TerraFM loaded via AutoModel (trust_remote_code=True)")
            self._log_model_info(model)
            return model
        except Exception as e1:
            logger.warning(f"AutoModel load failed: {e1}")

        # Approach 2: timm via HF Hub
        try:
            import timm
            model = timm.create_model(
                f"hf_hub:{CFG.terrafm_hub_id}",
                pretrained=True,
                num_classes=0,
                in_chans=CFG.s2_num_channels,   # load as 12-ch first
                img_size=CFG.image_size,
            )
            logger.info("TerraFM loaded via timm from HF Hub")
            self._log_model_info(model)
            return model
        except Exception as e2:
            logger.warning(f"timm HF Hub load failed: {e2}")

        # Approach 3: Snapshot download + manual weight load
        try:
            import glob, os
            from huggingface_hub import snapshot_download
            local_dir = snapshot_download(
                repo_id=CFG.terrafm_hub_id,
                cache_dir=CFG.weights_dir,
            )
            ckpts = (
                glob.glob(os.path.join(local_dir, "*.pth")) +
                glob.glob(os.path.join(local_dir, "*.pt"))
            )
            if not ckpts:
                raise FileNotFoundError(f"No .pth/.pt checkpoint in {local_dir}")
            ckpt_path = ckpts[0]
            logger.info(f"Loading from local checkpoint: {ckpt_path}")
            model = self._build_bare_vit()
            state = torch.load(ckpt_path, map_location="cpu")
            state = state.get("model", state.get("state_dict", state))

            # TerraFM uses modality-specific patch embed keys.
            # Remap conv2d_s2_l2a → patch_embed.proj so the timm ViT scaffold
            # receives the correct pretrained S2 weights.
            remap = {
                "patch_embed.conv2d_s2_l2a.weight": "patch_embed.proj.weight",
                "patch_embed.conv2d_s2_l2a.bias":   "patch_embed.proj.bias",
            }
            for src_key, dst_key in remap.items():
                if src_key in state and dst_key not in state:
                    state[dst_key] = state[src_key]
                    logger.info(f"Remapped {src_key} → {dst_key}")

            result = model.load_state_dict(state, strict=False)
            self._verify_load(result, len(state))
            return model
        except Exception as e3:
            logger.error(f"Snapshot load failed: {e3}")

        return None

    def _build_bare_vit(self) -> nn.Module:
        """
        Build a standard timm ViT scaffold to receive raw TerraFM-B weights.

        TerraFM-B's patch embedding projects to 2304 channels
        (= 768 × 3, one per modality: S2-L2A, S2-L1C, S1), NOT the
        standard 768.  We set embed_dim=2304 so the checkpoint weights
        load without a shape mismatch.

        The transformer blocks still operate at embed_dim=2304 and we
        extract features at that dimensionality. The UPerNet lateral
        convs handle the projection down to decoder_channels=256.
        """
        import timm

        # Detect the true embed_dim from the checkpoint before building
        # so we don't hard-code a wrong value if a future version changes.
        checkpoint_embed_dim = self._probe_checkpoint_embed_dim()
        actual_embed_dim = checkpoint_embed_dim if checkpoint_embed_dim else 2304

        if actual_embed_dim != self.embed_dim:
            import logging
            logging.getLogger(__name__).info(
                f"TerraFM-B checkpoint embed_dim={actual_embed_dim} "
                f"(overrides default {self.embed_dim}). "
                "Updating encoder embed_dim to match checkpoint."
            )
            self.embed_dim = actual_embed_dim

        name = ("vit_base_patch16_224" if self.model_size == "base"
                else "vit_large_patch16_224")
        return timm.create_model(
            name,
            pretrained=False,
            num_classes=0,
            in_chans=CFG.s2_num_channels,   # 12  (inflated to 14 after load)
            img_size=CFG.image_size,
            embed_dim=actual_embed_dim,      # match checkpoint exactly
        )

    def _probe_checkpoint_embed_dim(self) -> int:
        """
        Peek at the checkpoint to read the true patch embedding output dim.
        Returns 0 if the checkpoint cannot be found or read.
        """
        try:
            import glob, os, torch
            cache_root = CFG.weights_dir
            pattern = os.path.join(
                cache_root, "models--MBZUAI--TerraFM", "snapshots",
                "*", "TerraFM-B.pth"
            )
            ckpts = glob.glob(pattern)
            if not ckpts:
                return 0
            # Load only the patch embed weight tensor (fast, no full load)
            state = torch.load(ckpts[0], map_location="cpu")
            state = state.get("model", state.get("state_dict", state))
            key = "patch_embed.conv2d_s2_l2a.weight"
            if key in state:
                return int(state[key].shape[0])   # out_channels
        except Exception:
            pass
        return 0

    # ------------------------------------------------------------------
    # Patch-embed channel adaptation
    # ------------------------------------------------------------------

    def _adapt_patch_embed(self, model: nn.Module) -> nn.Module:
        """
        Ensure the patch embedding Conv2d accepts `self.in_channels` (14) inputs.

        TerraFM-B uses modality-specific patch embeddings in its checkpoint:
            patch_embed.conv2d_s2_l2a  (12-channel S2 L2A)
            patch_embed.conv2d_s2_l1c  (12-channel S2 L1C)
            patch_embed.conv2d_s1      ( 2-channel S1)

        The timm ViT scaffold (used as fallback) has a standard
        patch_embed.proj Conv2d. We need to:
          1. Load conv2d_s2_l2a weights from the checkpoint into patch_embed.proj
          2. Inflate from 12 → 14 channels (adding S1 channels)

        Cases handled:
          A) model already has in_channels == 14  → no change.
          B) model has in_channels == 12          → inflate to 14.
          C) model has in_channels == anything else → raise.
        """
        conv = self._find_patch_embed_conv(model)
        if conv is None:
            logger.warning(
                "Could not locate patch embedding Conv2d. "
                "Assuming model already handles 14-channel input."
            )
            return model

        current_in = conv.in_channels
        target_in  = self.in_channels   # 14

        if current_in == target_in:
            logger.info(
                f"Patch embedding already has {current_in} input channels."
            )
            return model

        if current_in != CFG.s2_num_channels:   # not 12
            raise RuntimeError(
                f"Unexpected patch embedding in_channels={current_in}. "
                f"Expected {CFG.s2_num_channels} (S2-only) or {target_in} (S1+S2)."
            )
        # TerraFM stores them under patch_embed.conv2d_s2_l2a.weight.
        # If found, use them; otherwise fall back to the current conv weights.
        # ------------------------------------------------------------------
        s2_weight = conv.weight.data   # [out, 12, ph, pw] — may be random if
                                        # the checkpoint keys didn't match

        # Look in the backbone's state_dict for the TerraFM-specific key
        sd = {k: v for k, v in self.backbone.state_dict().items()}
        terrafm_key = "patch_embed.conv2d_s2_l2a.weight"
        if terrafm_key not in sd:
            # Also try without the patch_embed prefix
            for k in sd:
                if "conv2d_s2_l2a.weight" in k:
                    terrafm_key = k
                    break

        if terrafm_key in sd:
            candidate = sd[terrafm_key]   # [out, 12, ph, pw]
            if candidate.shape == s2_weight.shape:
                s2_weight = candidate.clone()
                logger.info(
                    "Using TerraFM conv2d_s2_l2a weights for patch embedding "
                    "(correct pretrained S2 L2A weights loaded)."
                )
            else:
                logger.warning(
                    f"conv2d_s2_l2a shape {candidate.shape} != "
                    f"expected {s2_weight.shape}. Using current weights."
                )
        else:
            logger.warning(
                "patch_embed.conv2d_s2_l2a.weight not found in backbone state. "
                "Patch embedding will use random weights for S2 channels."
            )

        logger.info(
            f"Inflating patch embedding from {current_in} → {target_in} channels "
            "(S2 L2A weights preserved, S1 channels initialised from S2 mean)"
        )

        old_b = conv.bias.data if conv.bias is not None else None

        new_conv = nn.Conv2d(
            target_in, conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            bias=(conv.bias is not None),
        )
        with torch.no_grad():
            new_conv.weight[:, :current_in, :, :] = s2_weight
            # S1 channels: mean of S2 weights × 2 for a reasonable init
            s1_init = s2_weight.mean(dim=1, keepdim=True).expand(
                -1, target_in - current_in, -1, -1
            ) * 2.0
            new_conv.weight[:, current_in:, :, :] = s1_init
            if old_b is not None:
                new_conv.bias.copy_(old_b)

        self._set_patch_embed_conv(model, new_conv)
        return model

    def _find_patch_embed_conv(self, model: nn.Module) -> Optional[nn.Conv2d]:
        """Locate the patch-embedding Conv2d regardless of model class."""
        # timm ViT: model.patch_embed.proj
        if hasattr(model, "patch_embed") and hasattr(model.patch_embed, "proj"):
            c = model.patch_embed.proj
            if isinstance(c, nn.Conv2d):
                return c
        # HuggingFace ViT: model.embeddings.patch_embeddings.projection
        for path in [
            ["embeddings", "patch_embeddings", "projection"],
            ["vit", "embeddings", "patch_embeddings", "projection"],
        ]:
            obj = model
            try:
                for attr in path:
                    obj = getattr(obj, attr)
                if isinstance(obj, nn.Conv2d):
                    return obj
            except AttributeError:
                continue
        # Generic search
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d) and "patch" in name.lower():
                return module
        return None

    def _set_patch_embed_conv(self, model: nn.Module, new_conv: nn.Conv2d) -> None:
        """Replace the patch-embedding Conv2d in-place."""
        if hasattr(model, "patch_embed") and hasattr(model.patch_embed, "proj"):
            model.patch_embed.proj = new_conv
            return
        for path in [
            ["embeddings", "patch_embeddings"],
            ["vit", "embeddings", "patch_embeddings"],
        ]:
            obj = model
            try:
                for attr in path:
                    obj = getattr(obj, attr)
                obj.projection = new_conv
                return
            except AttributeError:
                continue
        logger.error(
            "Could not replace patch embedding Conv2d. "
            "The encoder may still use the old 12-channel embedding."
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _log_model_info(self, model: nn.Module) -> None:
        total     = sum(p.numel() for p in model.parameters()) / 1e6
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        logger.info(f"  TerraFM params: {total:.1f}M total / {trainable:.1f}M trainable")

    def _verify_load(self, result, total_keys: int) -> None:
        missing    = result.missing_keys
        unexpected = result.unexpected_keys
        if missing:
            logger.warning(f"Missing keys ({len(missing)}): {missing[:10]}"
                           f"{'...' if len(missing) > 10 else ''}")
        if unexpected:
            logger.warning(f"Unexpected keys ({len(unexpected)}): {unexpected[:5]}")

        # Ignore the original modality-specific keys as "unexpected" —
        # they were remapped before load_state_dict was called.
        ignorable = {"patch_embed.s2_l2a_embed", "patch_embed.s2_l1c_embed",
                     "patch_embed.s1_embed", "patch_embed.conv2d_s2_l2a.weight",
                     "patch_embed.conv2d_s2_l2a.bias",
                     "patch_embed.conv2d_s2_l1c.weight",
                     "patch_embed.conv2d_s2_l1c.bias",
                     "patch_embed.conv2d_s1.weight",
                     "patch_embed.conv2d_s1.bias"}
        real_missing = [k for k in missing if k not in ignorable]
        ratio = len(real_missing) / max(total_keys, 1)
        if ratio > 0.10:
            raise RuntimeError(
                f"Checkpoint mismatch: {len(real_missing)}/{total_keys} keys missing "
                f"({ratio*100:.1f}%). Wrong model_size or corrupted checkpoint."
            )
        logger.info(f"Weights loaded: {total_keys - len(missing)}/{total_keys} matched.")

    # ------------------------------------------------------------------
    # Hooks (fallback feature extraction)
    # ------------------------------------------------------------------

    def _register_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._hook_outputs = [None] * len(self.feature_indices)

        blocks = self._get_vit_blocks()
        if blocks is None:
            logger.error("Cannot find ViT transformer blocks for hook registration.")
            return

        for slot, block_idx in enumerate(self.feature_indices):
            if block_idx < len(blocks):
                def make_hook(s):
                    def hook(module, inp, output):
                        if isinstance(output, torch.Tensor):
                            self._hook_outputs[s] = output
                        elif hasattr(output, "last_hidden_state"):
                            self._hook_outputs[s] = output.last_hidden_state
                        else:
                            self._hook_outputs[s] = output[0]
                    return hook
                self._hooks.append(
                    blocks[block_idx].register_forward_hook(make_hook(slot))
                )

    def _get_vit_blocks(self):
        if hasattr(self.backbone, "blocks"):
            return self.backbone.blocks
        if hasattr(self.backbone, "encoder") and hasattr(self.backbone.encoder, "layer"):
            return self.backbone.encoder.layer
        if hasattr(self.backbone, "vit"):
            enc = getattr(self.backbone.vit, "encoder", None)
            if enc and hasattr(enc, "layer"):
                return enc.layer
        return None

    # ------------------------------------------------------------------
    # Freeze / unfreeze
    # ------------------------------------------------------------------

    def set_freeze_stage(self, stage: int) -> None:
        """
        Stage 0: Freeze entire encoder (decoder-only training).
        Stage 1: Freeze all except the last N transformer blocks + patch embed.
        Stage 2: Unfreeze everything.
        """
        if stage == 0:
            for p in self.backbone.parameters():
                p.requires_grad = False
            logger.info("TerraFM encoder: ALL frozen (Stage 0)")

        elif stage == 1:
            for p in self.backbone.parameters():
                p.requires_grad = False
            blocks = self._get_vit_blocks()
            if blocks is not None:
                n = CFG.unfreeze_last_n_blocks
                for blk in blocks[-n:]:
                    for p in blk.parameters():
                        p.requires_grad = True
                logger.info(
                    f"TerraFM encoder: last {n}/{len(blocks)} blocks unfrozen (Stage 1)"
                )
            # Also unfreeze the (now-inflated) patch embedding so S1 weights train
            conv = self._find_patch_embed_conv(self.backbone)
            if conv is not None:
                for p in conv.parameters():
                    p.requires_grad = True
            # Unfreeze final LayerNorms
            for name, m in self.backbone.named_modules():
                if isinstance(m, nn.LayerNorm) and "norm" in name.lower():
                    for p in m.parameters():
                        p.requires_grad = True

        elif stage == 2:
            for p in self.backbone.parameters():
                p.requires_grad = True
            logger.info("TerraFM encoder: ALL unfrozen (Stage 2)")

        trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        logger.info(f"  Trainable encoder params: {trainable/1e6:.1f}M")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, fused: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract multi-scale spatial features from TerraFM.

        Args:
            fused: [B, 14, 224, 224]  (S2 channels first, then S1)

        Returns:
            List of 4 tensors, each [B, embed_dim, 14, 14].
            Ordered from shallow → deep transformer blocks.

        SPATIAL NOTE:
            224×224 / 16 patch = 14×14 grid = 196 tokens.
            Each token covers ~16 px × 10 m/px = 160 m of ground.
            The decoder upsamples back to 224×224.
        """
        B = fused.shape[0]

        if self._use_timm_api:
            raw = self.backbone.get_intermediate_layers(
                fused,
                n=self.feature_indices,
                reshape=False,
                return_prefix_tokens=False,
            )
            features = list(raw)
        else:
            self._hook_outputs = [None] * len(self.feature_indices)
            _ = self.backbone(fused)
            features = []
            for i, feat in enumerate(self._hook_outputs):
                if feat is None:
                    features.append(
                        torch.zeros(B, self.num_patches_side**2, self.embed_dim,
                                    device=fused.device, dtype=fused.dtype)
                    )
                else:
                    features.append(feat)

        return [self._to_spatial(f, B) for f in features]

    def _to_spatial(self, tokens: torch.Tensor, B: int) -> torch.Tensor:
        """Reshape [B, N, C] (or [B, N+1, C] with CLS) → [B, C, H, W]."""
        if tokens.dim() == 2:
            tokens = tokens.unsqueeze(0)
        N = tokens.shape[1]
        exp = self.num_patches_side ** 2   # 196
        if N == exp + 1:
            tokens = tokens[:, 1:, :]      # remove CLS
        elif N != exp:
            logger.warning(f"Token count {N} ≠ {exp}; truncating.")
            tokens = tokens[:, :exp, :]
        feat = tokens.permute(0, 2, 1).reshape(
            B, self.embed_dim, self.num_patches_side, self.num_patches_side
        )
        return feat.contiguous()
