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
      (patch grid = 224/16 = 14 per side; embed_dim = 768 for ViT-B)

How it connects:
    model.py wraps TerraFMEncoder + UPerNetDecoder into TerraFMLULC.

14-CHANNEL INPUT STRATEGY:
-------------------------------------------------------------------
TerraFM's pretrained patch embedding was trained on modality-specific
inputs.  When we fuse S2 (12ch) + S1 (2ch) into a single 14-channel
tensor we need to expand the embedding's input projection.

CONFIRMED from TerraFM checkpoint keys:
  - TerraFM-B transformer embed_dim = 768   (blocks.0.norm1.weight shape [768])
  - patch_embed.conv2d_s2_l2a.weight shape [2304, 12, 16, 16]
    → out_channels = 2304 is the patch embed output (NOT the transformer dim)
    → the official class projects 2304 → 768 internally
  - The official terrafm.py class must be loaded from the snapshot so that
    all modality-specific projections and cross-attention fusion are intact.

LOADING STRATEGY:
  1. Download snapshot (or use cached copy).
  2. Import terrafm.py from snapshot via importlib.
  3. Instantiate the official TerraFM class.
  4. Load checkpoint into that class with strict=False.
  5. Detect actual embed_dim from blocks.0.norm1.weight shape (= 768).
  6. Inflate patch embedding input channels from 12 → 14.
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
        # Default embed_dim; will be overridden by checkpoint detection at load time
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
        # ----------------------------------------------------------------
        # APPROACH 1: Use terrafm.py from snapshot (authoritative)
        # ----------------------------------------------------------------
        try:
            import glob
            import os
            import importlib.util
            from huggingface_hub import snapshot_download

            local_dir = snapshot_download(
                repo_id=CFG.terrafm_hub_id,
                cache_dir=CFG.weights_dir,
            )
            logger.info(f"TerraFM snapshot: {local_dir}")
            logger.info(f"Files: {sorted(os.listdir(local_dir))}")

            # Find terrafm.py
            terrafm_py = os.path.join(local_dir, "terrafm.py")
            if not os.path.exists(terrafm_py):
                raise FileNotFoundError(f"terrafm.py not in {local_dir}")

            # Import it dynamically
            spec = importlib.util.spec_from_file_location("terrafm_official", terrafm_py)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Find checkpoint (.pth or .pt)
            ckpts = sorted(
                glob.glob(os.path.join(local_dir, "*.pth")) +
                glob.glob(os.path.join(local_dir, "*.pt"))
            )
            if not ckpts:
                raise FileNotFoundError(f"No .pth/.pt in {local_dir}")
            ckpt_path = ckpts[0]
            logger.info(f"Checkpoint: {ckpt_path}")

            state = torch.load(ckpt_path, map_location="cpu")
            state = state.get("model", state.get("state_dict", state))

            # Detect true transformer embed_dim from block norm weights
            # (= 768 for TerraFM-B, NOT 2304 which is the patch embed output)
            for k, v in state.items():
                if "blocks.0.norm1.weight" in k:
                    self.embed_dim = int(v.shape[0])
                    logger.info(f"Detected transformer embed_dim={self.embed_dim} "
                                f"from {k} shape {list(v.shape)}")
                    break

            # Instantiate official model from terrafm.py
            model = self._instantiate_from_module(module, state)

            result = model.load_state_dict(state, strict=False)
            self._verify_load(result, len(state))
            logger.info("TerraFM loaded via official terrafm.py")
            self._log_model_info(model)
            return model

        except Exception as e1:
            logger.warning(f"terrafm.py approach failed: {e1}")

        # ----------------------------------------------------------------
        # APPROACH 2: AutoModel with trust_remote_code
        # ----------------------------------------------------------------
        try:
            from transformers import AutoModel
            model = AutoModel.from_pretrained(
                CFG.terrafm_hub_id,
                trust_remote_code=True,
                cache_dir=CFG.weights_dir,
            )
            self._detect_embed_dim_from_model(model)
            logger.info("TerraFM loaded via AutoModel")
            self._log_model_info(model)
            return model
        except Exception as e2:
            logger.warning(f"AutoModel failed: {e2}")

        return None

    def _instantiate_from_module(self, module, state: dict) -> nn.Module:
        """Try all common factory patterns found in terrafm.py."""
        # Try known class names and factory functions
        for name in [
            "TerraFM", "TerraFMBase", "TerraFMLarge",
            "build_terrafm", "build_model", "create_model",
            "terrafm_base", "terrafm_large",
        ]:
            obj = getattr(module, name, None)
            if obj is None:
                continue
            for kwargs in [
                {"size": "base"}, {"model_size": "base"}, {"variant": "base"},
                {"pretrained": False}, {},
            ]:
                try:
                    if isinstance(obj, type) and issubclass(obj, nn.Module):
                        m = obj(**kwargs) if kwargs else obj()
                    else:
                        m = obj(**kwargs) if kwargs else obj()
                    if isinstance(m, nn.Module):
                        logger.info(f"Instantiated via {name}({kwargs})")
                        return m
                except Exception:
                    continue

        # Fallback: scan for any nn.Module subclass in the module
        for name in sorted(dir(module)):
            if name.startswith("_"):
                continue
            obj = getattr(module, name)
            if (
                isinstance(obj, type)
                and issubclass(obj, nn.Module)
                and obj is not nn.Module
            ):
                try:
                    m = obj()
                    if isinstance(m, nn.Module):
                        logger.info(f"Instantiated via {name}()")
                        return m
                except Exception:
                    pass

        raise RuntimeError(
            f"Cannot instantiate model from terrafm.py. "
            f"Names: {[n for n in dir(module) if not n.startswith('_')]}"
        )

    def _detect_embed_dim_from_model(self, model: nn.Module) -> None:
        """Read actual embed_dim from loaded model parameters."""
        for name, param in model.named_parameters():
            if "blocks.0.norm1.weight" in name or "layer.0" in name:
                self.embed_dim = int(param.shape[0])
                logger.info(
                    f"Detected embed_dim={self.embed_dim} from {name}"
                )
                return

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

        # TerraFM stores pretrained S2 L2A weights under conv2d_s2_l2a.weight.
        # Use them if available; otherwise fall back to current conv weights.
        s2_weight = conv.weight.data   # [out, 12, ph, pw]

        sd = dict(model.state_dict())
        terrafm_key = "patch_embed.conv2d_s2_l2a.weight"
        if terrafm_key not in sd:
            for k in sd:
                if "conv2d_s2_l2a.weight" in k:
                    terrafm_key = k
                    break

        if terrafm_key in sd:
            candidate = sd[terrafm_key]
            if candidate.shape == s2_weight.shape:
                s2_weight = candidate.clone()
                logger.info(
                    "Using TerraFM conv2d_s2_l2a weights for patch embedding "
                    "(correct pretrained S2 L2A weights loaded)."
                )
            else:
                logger.warning(
                    f"conv2d_s2_l2a shape {list(candidate.shape)} != "
                    f"expected {list(s2_weight.shape)}. Using current weights."
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
            # S1 channels: mean of S2 weights × 2 for a sensible init
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
        # Generic search: first Conv2d with "patch" in its module path
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
            logger.warning(
                f"Missing keys ({len(missing)}): {missing[:10]}"
                f"{'...' if len(missing) > 10 else ''}"
            )
        if unexpected:
            logger.warning(
                f"Unexpected keys ({len(unexpected)}): {unexpected[:5]}"
            )

        # Modality-specific patch embed keys are expected to be "unexpected"
        # when loading into an official TerraFM class — that class handles
        # them internally.  Exclude them from the mismatch ratio check.
        ignorable = {
            "patch_embed.s2_l2a_embed",
            "patch_embed.s2_l1c_embed",
            "patch_embed.s1_embed",
            "patch_embed.conv2d_s2_l2a.weight",
            "patch_embed.conv2d_s2_l2a.bias",
            "patch_embed.conv2d_s2_l1c.weight",
            "patch_embed.conv2d_s2_l1c.bias",
            "patch_embed.conv2d_s1.weight",
            "patch_embed.conv2d_s1.bias",
        }
        real_missing = [k for k in missing if k not in ignorable]
        ratio = len(real_missing) / max(total_keys, 1)
        if ratio > 0.10:
            raise RuntimeError(
                f"Checkpoint mismatch: {len(real_missing)}/{total_keys} keys missing "
                f"({ratio*100:.1f}%). Wrong model_size or corrupted checkpoint."
            )
        logger.info(
            f"Weights loaded: {total_keys - len(missing)}/{total_keys} matched."
        )

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

        trainable = sum(
            p.numel() for p in self.backbone.parameters() if p.requires_grad
        )
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
            embed_dim = 768 for ViT-B (confirmed from checkpoint).

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
                        torch.zeros(
                            B,
                            self.num_patches_side ** 2,
                            self.embed_dim,
                            device=fused.device,
                            dtype=fused.dtype,
                        )
                    )
                else:
                    features.append(feat)

        return [self._to_spatial(f, B) for f in features]

    def _to_spatial(self, tokens: torch.Tensor, B: int) -> torch.Tensor:
        """Reshape [B, N, C] (or [B, N+1, C] with CLS token) → [B, C, H, W]."""
        if tokens.dim() == 2:
            tokens = tokens.unsqueeze(0)
        N = tokens.shape[1]
        exp = self.num_patches_side ** 2   # 196
        if N == exp + 1:
            tokens = tokens[:, 1:, :]      # strip CLS token
        elif N != exp:
            logger.warning(f"Token count {N} ≠ {exp}; truncating.")
            tokens = tokens[:, :exp, :]
        feat = tokens.permute(0, 2, 1).reshape(
            B, self.embed_dim, self.num_patches_side, self.num_patches_side
        )
        return feat.contiguous()
