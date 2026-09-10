# TerraFM LULC Segmentation (S1 + S2)

19-class Sentinel-2 + Sentinel-1 land-use / land-cover semantic segmentation  
using **TerraFM-Base** (pretrained encoder) + **UPerNet** decoder.  
Single **NVIDIA T4 (16 GB VRAM)** on Google Colab Free — no Drive mount required.

---

## Architecture Overview

```
Input:  [B, 14, 224, 224]
         └─ 12 S2 channels (B01…B12, normalized reflectance)
         └─  2 S1 channels (VV, VH, normalized dB)
              ↓
TerraFM-B encoder
  - Patch embedding (16×16) inflated from 12 → 14 input channels
  - 12 transformer blocks, 768-dim hidden
  - Feature extraction from blocks [2, 5, 8, 11] → 4× [B, 768, 14, 14]
              ↓
UPerNet decoder
  - Lateral 1×1 projections → 256-dim
  - Pyramid Pooling Module (global context)
  - FPN top-down merge
  - Bilinear upsample → [B, 256, 224, 224]
  - 1×1 classification head
              ↓
Output: [B, 19, 224, 224]  logits
Pred:   argmax → [B, 224, 224]  class IDs (0–18 valid, 255 = ignore)
```

---

## Project File Structure

```
segmentation/
├── clean_dataset.py        ← RUN FIRST: validate + remove corrupted patches
├── config.py               ← All settings (edit paths here)
├── preprocessing.py        ← S2/S1 TIFF reading, resampling, normalization
├── dataset.py              ← S1S2LULCDataset + DataLoader factory (no caching)
├── terrafm_encoder.py      ← TerraFM loading, 14-ch patch embed inflation
├── segmentation_decoder.py ← UPerNet with PPM
├── model.py                ← End-to-end model + save/load artifact
├── losses.py               ← CE+Dice, Focal+Dice, class weights
├── metrics.py              ← mIoU, Dice, confusion matrix (ignore-safe)
├── checkpoint.py           ← Colab-safe save/resume
├── train.py                ← Training loop (all phases)
├── evaluate.py             ← Test-set evaluation + result files
├── inference.py            ← Single-patch and large-image inference
├── visualize.py            ← Training curves, confusion matrix, sample plots
├── prepare_dataset.py      ← Splits + norm stats + class stats
├── utils.py                ← Seed, logging, GPU memory, config I/O
├── requirements.txt        ← Python dependencies
├── TerraFM_LULC_Colab.ipynb ← 15-cell Colab workflow
└── README_PROJECT.md       ← This file
```

---

## file.csv Format

The dataset index CSV has **no split column**. It looks like:

```
s1_name,patch_id,reference_map_id
S1B_IW_GRDH_..._33UUP_37_88,S2A_MSIL2A_..._33UUP_37_88,S2A_MSIL2A_..._33UUP_37_88_reference_map
S1B_IW_GRDH_..._33UUP_37_89,S2A_MSIL2A_..._33UUP_37_89,S2A_MSIL2A_..._33UUP_37_89_reference_map
...
```

- `s1_name` → folder name inside `S1/`
- `patch_id` → folder name inside `S2/`
- `reference_map_id` → file stem in `reference_maps_selected/` (`.tif` appended)

---

## Step-by-Step Workflow

### Step 1 — Clean the dataset (run once)

```bash
python clean_dataset.py \
    --csv  /content/data/file.csv \
    --s1   /content/data/S1 \
    --s2   /content/data/S2 \
    --ref  /content/data/reference_maps_selected \
    --out_csv  /content/data/file_clean.csv \
    --log  /content/outputs/corrupted_patches.txt
```

This script:
- Opens every S2 TIFF (12 per patch), every S1 TIFF (2 per patch), and every reference TIFF
- Checks for missing files, wrong band count, unreadable headers, all-zero / all-NaN data
- Logs corrupted rows to `corrupted_patches.txt` with counts and reasons
- **Deletes the corrupted folders/files from disk** (S1 folder, S2 folder, reference TIFF)
- Writes a clean `file_clean.csv` with only valid rows

### Step 2 — Prepare dataset (run once)

After cleaning, run `prepare_all()` or Cell 6 in the notebook:
- Reads `file_clean.csv`
- Extracts MGRS tile codes from `patch_id` for geographic splitting
- Creates `train.csv`, `val.csv`, `test.csv` (all three columns retained)
- Computes S2 normalization stats from training set (reflectance scale)
- Computes S1 normalization stats in dB from training set
- Discovers class IDs from reference maps
- Computes class pixel frequencies

### Step 3 — Run the Colab notebook

Open `TerraFM_LULC_Colab.ipynb` and run cells in order:

| Cell | Action |
|------|--------|
| 1 | Install dependencies (including `gdown`) |
| 2 | Download dataset from Google Drive with `gdown` |
| 3 | Set paths, configure CFG |
| 4 | Verify TerraFM Hub access |
| 5 | Run `clean_dataset.py` |
| 6 | Run `prepare_all()` → splits + stats |
| 7 | Reload stats (after restart) |
| 8 | Visualize examples |
| 9 | **Phase 0 smoke test** (mandatory) |
| 10 | Phase 1 pilot (~2000 samples) |
| 11 | **Phase 2 full training** |
| 12 | Evaluate on test set |
| 13 | Plot training curves |
| 14 | Export `terrafm_lulc_model.pth` |
| 15 | Inference on new S1+S2 patch |

---

## S1 Input Details

Sentinel-1 GRD products provide VV and VH polarisation bands. Each S1 patch folder
contains 2 TIFFs, naturally sorted → [VV, VH] (or [VH, VV] depending on naming —
the relative order is consistent across all patches from the same dataset).

**Preprocessing pipeline for S1:**
```
raw linear power (float32)
    → clip to [1e-10, ∞) to avoid log(0)
    → 10 × log10(x)   →  dB scale  [-30, 0] dB typical for S1 GRD
    → clip to [-30, 0]
    → (x - mean_dB) / std_dB    channel-wise z-normalization
    → torch.Tensor [2, 224, 224]
```

**Why dB?** SAR backscatter spans several orders of magnitude in linear scale.
Converting to dB compresses the range and produces a more Gaussian distribution,
which matches the z-normalization assumption much better.

---

## 14-Channel Patch Embedding Inflation

TerraFM's pretrained patch embedding Conv2d was trained on 12-channel S2 input.
To accept the fused 14-channel tensor we inflate the weight:

```
new_weight[:, :12, :, :] = pretrained_weight    (S2 channels preserved)
new_weight[:, 12:14, :, :] = mean(pretrained_weight, dim=1) × 2.0
```

The S1 channels are initialised at twice the mean S2 weight, which gives them a
reasonable starting gradient magnitude. Fine-tuning quickly corrects this.

---

## No Caching — Why?

Each `__getitem__` call loads S1 + S2 + reference mask fresh from TIFF files.
There is no `.npz` cache. Reasons:

1. **Uniqueness**: random geometric augmentations (flip, rotate) mean the same
   files are never processed identically across epochs.
2. **Storage**: caching 50k preprocessed patches at `[14, 224, 224] × 4 bytes ≈ 2.8 MB`
   per patch = **140 GB** — far exceeds Colab's VM disk.
3. **Simplicity**: with 2 DataLoader workers, live TIFF loading is fast enough
   that the T4 GPU is never waiting for data.

---

## Key Hyperparameters

| Parameter | Value | Reason |
|---|---|---|
| Input channels | 14 | 12 S2 + 2 S1 |
| `model_size` | `base` | ViT-B; Large needs A100 |
| `image_size` | 224 | TerraFM pre-training size |
| `batch_size` | 4 | T4-safe with 14-ch input + UPerNet |
| `grad_accum_steps` | 4 | Effective batch = 16 |
| `encoder_lr` | 5e-5 | Pretrained → slow updates |
| `decoder_lr` | 5e-4 | Random init → fast updates |
| `epochs` | 35 | With early stopping (patience=8) |
| Freeze strategy | Stage 0 → Stage 1 at epoch 6 | Decoder warms up first |
| `loss_type` | `ce_dice` | CE + Dice (0.5 each) |
| `amp_dtype` | `fp16` | T4 is FP16 (Turing arch) |
| S1 preprocessing | linear → dB | Compresses SAR dynamic range |

---

## Geographic Splitting

The code extracts MGRS tile codes (e.g. `T33UUP`) from `patch_id` using a regex.
If ≥50% of patches have extractable tiles, it splits by tile: no tile appears in
both training and test sets. This prevents the model from seeing near-duplicate
spectral conditions during evaluation.

If tile extraction fails (unusual naming), it falls back to random split with a warning.

---

## Output Files

```
outputs/
├── checkpoints/
│   ├── best_model.pth          ← Best val mIoU checkpoint
│   ├── last_checkpoint.pth     ← Resume from here after crash
│   └── checkpoint_epoch_NNN.pth
├── splits/
│   ├── train.csv  val.csv  test.csv   (all have s1_name, patch_id, reference_map_id)
├── results/
│   ├── metrics.json
│   ├── per_class_metrics.csv
│   ├── confusion_matrix.png
│   ├── training_curves.png
│   └── qualitative_predictions/
├── norm_stats.json             ← s2_mean, s2_std, s1_mean, s1_std
├── class_mapping.json
├── class_stats.json
├── dataset_report.txt
├── corrupted_patches.txt       ← Log from clean_dataset.py
├── training_history_phase2.csv
└── terrafm_lulc_model.pth      ← Final self-contained inference artifact
```

---

## Resuming After a Colab Crash

Set `CFG.resume_from` before running Cell 11:

```python
CFG.resume_from = '/content/outputs/checkpoints/last_checkpoint.pth'
```

The checkpoint restores: model weights, optimizer state, scheduler step,
scaler state, epoch count, best mIoU, and normalization statistics.

---

## Inference on a New Patch

```python
from inference import patch_inference

outputs = patch_inference(
    model_path = 'terrafm_lulc_model.pth',
    s2_patch_dir = '/path/to/S2/PATCH_ID/',
    s1_patch_dir = '/path/to/S1/S1_NAME/',
    output_dir   = './predictions/',
)
# outputs['mask_tif'] → GeoTIFF with class IDs
# outputs['vis_png']  → colorized visualization
```

---

## What Indicates Success?

| Phase | Target |
|-------|--------|
| Phase 0 smoke test | Train loss < 1.0 within 30 epochs |
| Phase 1 pilot | Val mIoU > 0.20 after 10 epochs (frozen encoder) |
| Phase 2 (epoch 10+) | Val mIoU > 0.35 after encoder partially unfrozen |
| Phase 2 final | Val mIoU > 0.50 is a strong result for 19-class LULC |
| Per-class IoU | > 0.30 for common classes; rare classes lower |

---

## Teammate Guide

You don't need to read the TerraFM paper. Here is what matters:

- **`config.py`** — change paths and hyperparameters here only
- **`clean_dataset.py`** — run before anything else
- **`prepare_dataset.py`** — run once to create splits and statistics  
- **`TerraFM_LULC_Colab.ipynb`** — follow cells 1→15 for the full workflow
- **`terrafm_lulc_model.pth`** — the final deliverable; pass to `inference.py`

The model takes a 14-channel image (12 spectral bands from Sentinel-2 + 2 SAR
bands from Sentinel-1), feeds it through a pretrained vision transformer
(TerraFM), and uses the extracted features to predict a land-cover class for
every pixel.

---

## Correct Dataset Paths

After extracting `dataset.zip`, the folder structure is:

```
/content/
└── content/
    └── dataset/
        ├── images/
        │   ├── S1/
        │   ├── S2/
        │   └── reference_maps_selected/
        └── file_clean.csv
```

These paths are already set correctly in Cell 3 of the notebook.

---

## Notebook Cell Order (Updated)

| Cell | Action | Re-run every session? |
|------|--------|-----------------------|
| 1 | Install dependencies | ✅ Yes |
| 2 | Clone repo from GitHub | ✅ Yes |
| 3 | Configure paths + CFG + BigEarthNet names | ✅ Yes |
| 4 | Download dataset with gdown | Only if not extracted |
| 5 | Verify reference map (visual) | Optional |
| 6 | Verify S2+S1 tensor shapes | Optional |
| 7 | Prepare splits + norm stats + class stats | **Once only** |
| 8 | Reload saved stats | ✅ Yes (skip Cell 7 after first run) |
| 9 | Visualize training examples | Optional |
| 10 | Phase 0 smoke test | Before first full training |
| 11 | Phase 1 pilot | Optional |
| 12 | Phase 2 full training | Set RESUME_FROM to continue |
| 13 | Evaluate test set | After training |
| 14 | Plot training curves | After training |
| 15 | Export final model | After training |
| 16 | Inference on new patch | After export |

---

## Session Resume Guide

### What to download before session ends

| Phase | File | Why needed |
|-------|------|------------|
| After Cell 7 | `outputs/splits/train.csv`, `val.csv`, `test.csv` | Reproducible split — avoid re-running 10 min prep |
| After Cell 7 | `outputs/norm_stats.json` | S2+S1 stats must be identical across sessions |
| After Cell 7 | `outputs/class_mapping.json` + `class_stats.json` | Class IDs + loss weights |
| During Phase 2 | `outputs/checkpoints/last_checkpoint.pth` | Resume training from exact epoch |
| After training | `outputs/checkpoints/best_model.pth` | Best weights |
| After Cell 15 | `outputs/terrafm_lulc_model.pth` | Self-contained inference artifact |

### How to resume in a new session

1. Run Cells 1, 2, 3 (always required — installs, clones, sets paths)
2. Upload your saved files to `/content/outputs/` via the Colab file browser
3. Skip Cell 7 → run **Cell 8** to reload stats from uploaded files
4. Phase 2 resume — in Cell 12 set:
   ```python
   RESUME_FROM = '/content/outputs/checkpoints/last_checkpoint.pth'
   ```
   Then run Cell 12. Training continues from the last saved epoch, with the same optimizer and scheduler state.

---

## Known Bugs Fixed

See `BUGLOG.md` for full details.

| # | Bug | Fix |
|---|-----|-----|
| 1 | GT mask all black | uint8 overflow for CLC IDs > 255 → changed to uint32 |
| 2 | Train loss = NaN | FP16 overflow in Dice loss → cast logits to float32 before loss |
| 3 | Patch embedding not loaded | TerraFM uses `conv2d_s2_l2a` key, not standard `proj` → key remapping added |
| 4 | Wrong colors in visualization | matplotlib vmin/vmax clipped -1 to red → replaced with direct uint8 LUT |
| 5 | Duplicate functions in visualize.py | Full rewrite of the file |
| 6 | Wrong data paths in notebook | Updated to match actual zip extraction structure |
| 7 | Legend showed `class_0` etc. | Added BigEarthNet names assignment in Cell 3 |
