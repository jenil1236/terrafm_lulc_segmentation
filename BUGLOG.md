# Bug Log — TerraFM LULC Segmentation

---

## BUG 1 — Ground truth mask all black (ignore_index everywhere)
**File:** `preprocessing.py` → `read_reference_mask()`  
**Symptom:** Ground truth panel completely black in visualizations.  
**Root cause:** Destination array allocated as `dtype=np.uint8`. Raw CLC IDs (311, 312, 511 etc.) are uint16 and exceed 255. When rasterio wrote them into a uint8 array they wrapped modulo 256 (e.g. 311 % 256 = 55). No wrapped value matched any key in `raw_to_train`, so all pixels stayed at ignore_index=255 (black).  
**Fix:** Changed destination array to `dtype=np.uint32` so raw CLC IDs up to 65534 are preserved before remapping.

---

## BUG 2 — Train loss = NaN every other epoch
**File:** `losses.py` → `DiceLoss.forward()` and `SegmentationLoss.forward()`  
**Symptom:** Train loss shows NaN intermittently; GradScaler scale keeps halving.  
**Root cause:** Dice loss computes softmax + division in FP16 (AMP). Early in training the decoder produces large random logits → softmax outputs overflow to inf in FP16 → division produces NaN → GradScaler marks the step as invalid and skips it → `running_loss / steps` becomes 0/0 = NaN.  
**Fix:** Added `logits = logits.float()` at the top of both `DiceLoss.forward()` and `SegmentationLoss.forward()` to force float32 for the loss computation regardless of AMP dtype. Also added a NaN guard that returns zero loss and logs a warning if a non-finite loss slips through.

---

## BUG 3 — TerraFM patch embedding weights not loaded (random init)
**File:** `terrafm_encoder.py` → `_try_load_huggingface()` and `_adapt_patch_embed()`  
**Symptom:** Log shows `Missing keys: ['patch_embed.proj.weight', 'patch_embed.proj.bias']` and `Unexpected keys: ['patch_embed.conv2d_s2_l2a.weight', ...]`. Despite "162/164 keys matched", the patch embedding was randomly initialized.  
**Root cause:** TerraFM-B uses modality-specific patch embeddings (`conv2d_s2_l2a`, `conv2d_s2_l1c`, `conv2d_s1`) instead of the standard timm `patch_embed.proj`. The fallback bare ViT scaffold has `patch_embed.proj`, so the checkpoint key names didn't match and those weights were silently skipped.  
**Fix 1:** In `_try_load_huggingface()`, before calling `load_state_dict`, remap `patch_embed.conv2d_s2_l2a.weight → patch_embed.proj.weight` so the timm scaffold receives the correct S2 L2A pretrained weights.  
**Fix 2:** In `_adapt_patch_embed()`, after loading, search the backbone state_dict for `conv2d_s2_l2a.weight` and use it as the S2 base for the 12→14 channel inflation, instead of whatever random weights happen to be in `conv.weight.data`.

---

## BUG 4 — Visualization colors wrong (all red instead of class colors)
**File:** `visualize.py` → `plot_qualitative_sample()`  
**Symptom:** GT and prediction panels showed entirely red pixels instead of class-specific colors.  
**Root cause:** Used `matplotlib.imshow` with `ListedColormap` and set ignored pixels to value `-1`. Matplotlib clips values below `vmin=0` to the first colormap color (index 0 = red = Urban fabric), so all 255-valued ignored pixels rendered red.  
**Fix:** Replaced `imshow + ListedColormap` with direct uint8 array lookup via `mask_to_rgb(mask, color_map)`. The color_map is a `[256, 3]` numpy array; `color_map[pixel_value]` returns the exact RGB for that train ID with no matplotlib normalization involved.

---

## BUG 5 — visualize.py had duplicate function definitions
**File:** `visualize.py`  
**Symptom:** Two `plot_qualitative_sample` functions in the same file; the second (old buggy one) shadowed the fixed one.  
**Root cause:** Partial file edit left both old and new implementations in the file.  
**Fix:** Rewrote the entire file from scratch to eliminate duplication.

---

## BUG 6 — Wrong dataset paths in notebook
**File:** `TerraFM_LULC_Colab.ipynb` → Cell 3  
**Symptom:** `FileNotFoundError` when loading S2/S1/reference data.  
**Root cause:** Notebook used `/content/data/S2` etc., but actual extraction structure after unzipping `dataset.zip` is `/content/content/dataset/images/S2` (zip contains a `content/dataset/images/` prefix).  
**Fix:** Updated all data paths in Cell 3 to match the actual extraction structure.

---

## BUG 7 — class_names showing `class_0`, `class_1` in legend
**File:** `config.py` default + `TerraFM_LULC_Colab.ipynb` Cell 3  
**Symptom:** Legend in visualization showed `class_0` through `class_17` instead of real names.  
**Root cause:** `CFG.class_names` defaulted to `[f"class_{i}" for i in range(19)]`. The updated BigEarthNet names were defined in `visualize.py` but never assigned to `CFG` in the notebook.  
**Fix:** Added to Cell 3 of the notebook:
```python
from visualize import BIGEARTHNET_CLASS_NAMES, BIGEARTHNET_COLORS
CFG.class_names  = BIGEARTHNET_CLASS_NAMES
CFG.class_colors = BIGEARTHNET_COLORS
```
