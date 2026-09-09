I want you to act as a **senior remote-sensing deep-learning engineer and PyTorch researcher** and build a complete, production-quality training pipeline for **19-class Sentinel-2 LULC semantic segmentation** using **TerraFM as the pretrained encoder**.

This is not a request for a toy example. I want a complete project that I can run in **Google Colab Free using a T4 GPU**, understand easily, train reliably, resume from checkpoints, evaluate properly, and finally use for inference on new Sentinel-2 TIFF patches.

Do not give me a generic segmentation example. Design everything around the exact dataset structure, TerraFM implementation, hardware constraints, and requirements described below.

---

# 1. First: verify TerraFM before writing the architecture

The pretrained model I want to use is:

**https://huggingface.co/MBZUAI/TerraFM**

and the official TerraFM repository/model implementation associated with it.

Before generating code, inspect the official TerraFM implementation/documentation and determine:

1. Exact TerraFM architecture.
2. Exact TerraFM-Base model construction.
3. Exact TerraFM-Large model construction.
4. Expected Sentinel-2 input channels.
5. Exact expected channel ordering.
6. Expected input spatial size.
7. Expected normalization/scaling.
8. Whether the official implementation expects L1C, L2A, or both.
9. What TerraFM's forward pass returns.
10. Which internal tensor/features should be used as the encoder output for downstream semantic segmentation.
11. Whether intermediate ViT features can/should be extracted.
12. Whether TerraFM already contains a task-specific decoder, or whether I need to add my own segmentation decoder.
13. Which pretrained checkpoint file should be downloaded.
14. How the official weights should be loaded.
15. Whether any layers should remain frozen.
16. Whether the official model supports Sentinel-1 and Sentinel-2 jointly, and whether Sentinel-2-only inference is supported.
17. Whether the 12-channel Sentinel-2-only input shown in the official example is exactly the configuration we should use.

**Do not invent TerraFM APIs.**

Use the actual official implementation/API from the current repository/model page.

If the official implementation differs from assumptions in this prompt, follow the official implementation and clearly explain the difference before writing the final code.

The official TerraFM resources indicate that TerraFM is a ViT-based multisensor Earth-observation foundation model trained on large-scale Sentinel-1/2 data and provide TerraFM-Base/Large models; the official usage example demonstrates a 12-channel, 224×224 Sentinel-2 input. Verify all details against the actual implementation rather than relying on this statement alone.

---

# 2. My actual task

I have approximately:

**50,000 Sentinel-2 and Sentinel-1 patches**

Each patch has:

* Sentinel-2 multispectral imagery
* Sentinel-1 SAR imagery
* a corresponding pixel-level reference map.

The reference map contains a class ID for every pixel.

The task is:

> **19-class semantic segmentation of Sentinel-2 imagery.**

This is a closed-set semantic segmentation problem.

For every valid pixel, the trained model should predict exactly one of the 19 LULC classes.

I am NOT trying to make the LULC model promptable.

SAM 3 will remain a separate model for general promptable segmentation.

The model we are building in this project is specifically optimized for:

**Sentinel-2 → 19-class LULC segmentation**

---

# 3. Hardware constraint

I will initially use:

**Google Colab Free**
**NVIDIA T4**
**16 GB VRAM**

Therefore the architecture and training procedure MUST realistically fit within approximately 16 GB VRAM.

Do not design a training configuration that requires:

* A100
* H100
* 40+ GB VRAM
* distributed training
* multi-GPU
* extremely large batch sizes.

The solution should be optimized for a single T4.

If TerraFM-L cannot realistically be fine-tuned on T4, use:

**TerraFM-Base**

as the default model.

Explain why.

I still want the architecture to be structured so that TerraFM-L can potentially be used later by changing one configuration value if more GPU memory becomes available.

---

# 4. Exact dataset structure

My data will look approximately like this:

```text
content/
└── data/
    ├── S1/
    │   ├── PATCH_ID_001/
    │   │   ├── band1.tif
    │   │   └── band2.tif
    │   │
    │   ├── PATCH_ID_002/
    │   │   ├── band1.tif
    │   │   └── band2.tif
    │   │
    │   └── ...
    │
    ├── S2/
    │   ├── PATCH_ID_001/
    │   │   ├── band1.tif
    │   │   ├── band2.tif
    │   │   ├── ...
    │   │   └── band12.tif
    │   │
    │   ├── PATCH_ID_002/
    │   │   ├── band1.tif
    │   │   ├── band2.tif
    │   │   ├── ...
    │   │   └── band12.tif
    │   │
    │   └── ...
    │
    ├── file.csv
    │
    └── selected_reference_map/
        ├── PATCH_ID_001.tif
        ├── PATCH_ID_002.tif
        └── ...
```

The actual TIFF filenames may differ.

Therefore:

**Do not hard-code assumptions about filenames.**

The project must have a robust dataset-discovery layer that:

1. Reads the S2 patch directories.
2. Identifies the 12 Sentinel-2 TIFFs.
3. Reads the S1 patch directory if S1 is needed.
4. Uses `file.csv` to determine valid S1-S2 pairings.
5. Finds the matching reference map using the Sentinel-2 patch ID.
6. Validates that all required files exist.
7. Produces a clean dataset index.

However:

### Important

For the primary LULC model, determine from the official TerraFM implementation whether we should use:

**S2 only**

or

**S1 + S2**.

Do not automatically include S1 simply because it exists.

I want you to decide whether S1 provides a meaningful advantage for my 19-class LULC task and whether using S1 would complicate training unnecessarily on a T4.

If TerraFM supports S2-only properly, make **S2-only the default** unless there is a strong technical reason to use S1.

The code should nevertheless be modular enough that S1 can be added later.

---

# 5. Understand my Sentinel-2 bands correctly

Each S2 patch contains 12 TIFF files.

The 12 bands may not all have the same native spatial resolution.

Some bands are 10 m.
Some are 20 m.
Some may have lower native spatial resolution depending on the Sentinel-2 configuration.

I need proper preprocessing.

Do NOT simply resize every TIFF independently to the final image size without first considering its geospatial resolution.

Design the preprocessing pipeline so that:

1. The bands are identified correctly.
2. Their native resolutions are determined.
3. Their CRS/geotransform are checked.
4. Their spatial extents are checked.
5. All required bands are aligned to a common grid.
6. All bands are resampled to the chosen target resolution.
7. The resulting channels are stacked in the correct TerraFM order.
8. The output tensor has a consistent spatial meaning.

Prefer a physically meaningful geospatial resampling strategy rather than arbitrary image resizing.

Explain which interpolation method should be used and why.

For example:

* continuous reflectance → bilinear
* categorical labels → nearest-neighbor.

Never use bilinear interpolation on the class-ID reference map.

---

# 6. Reference-map processing

Each reference TIFF contains integer class IDs per pixel.

For example:

```text
0
1
2
3
...
18
```

or potentially the original raw CLC IDs.

The code must NOT assume blindly that the values are already 0–18.

Build a configurable class-mapping system.

The pipeline should:

1. Read the reference TIFF.
2. Inspect unique class values.
3. Convert raw class IDs into contiguous training IDs.
4. Produce class IDs:

```text
0 ... 18
```

for the 19 valid classes.

5. Reserve a dedicated `IGNORE_INDEX`, such as 255, for:

   * unlabeled
   * NoData
   * invalid pixels.

The final training target should therefore be:

```text
H × W
```

where every valid pixel contains one integer from:

```text
0 ... 18
```

and ignored pixels contain the ignore index.

Do not turn this into polygons.

Do not turn this into bounding boxes.

Do not convert it into SAM-style instance annotations.

This is a standard semantic segmentation target.

---

# 7. Model architecture I want

The architecture should be:

```text
12-band Sentinel-2 input
        ↓
TerraFM pretrained encoder
        ↓
multiscale / spatial feature extraction
        ↓
semantic segmentation decoder
        ↓
19-class segmentation head
        ↓
H × W × 19 logits
```

However, I want YOU to decide the exact decoder.

Compare appropriate decoder choices such as:

* simple U-Net decoder
* FPN
* UPerNet
* DeepLab-style decoder
* SegFormer-style decoder
* another transformer-compatible decoder.

Then choose ONE.

The decoder must be:

* appropriate for ViT encoder features,
* computationally reasonable on T4,
* good for dense segmentation,
* simple enough for my teammate to understand,
* easy to save and reload,
* compatible with TerraFM's feature representation.

I prefer a clean architecture over unnecessary complexity.

---

# 8. Critical question: how to use TerraFM features correctly

This is one of the most important parts.

TerraFM is a ViT-based foundation model.

A ViT does not naturally produce a conventional CNN feature pyramid.

Therefore determine how to obtain spatial feature maps from TerraFM.

Possible approaches might include:

* final patch-token features
* intermediate transformer block outputs
* several intermediate layers
* reshaping patch tokens into 2D feature maps
* projection layers
* feature pyramid construction
* decoder using multi-level transformer features.

DO NOT guess.

Inspect the official implementation and determine what is technically appropriate.

Then explain:

```text
TerraFM internal representation
        ↓
how it becomes spatial feature maps
        ↓
segmentation decoder
```

I want the final code to make this explicit and well documented.

---

# 9. Input image size

The official TerraFM model specification indicates 224×224 input.

Determine whether we should train with:

**224×224**

or whether there is a technically valid way to use larger spatial sizes.

Because my original Sentinel-2 patches may be 120×120 or otherwise small, determine how we should handle this.

Possible approaches:

### Option A

Resample the entire patch to 224×224.

### Option B

Pad to 224×224.

### Option C

Resize while maintaining the correct geospatial aspect ratio.

### Option D

Use overlapping crops/tiles.

Choose the method that is most appropriate for TerraFM and explain the consequences for:

* spatial resolution
* segmentation accuracy
* memory
* interpolation
* boundary quality.

Do not unnecessarily upscale just because 224×224 is convenient.

---

# 10. Important issue: geospatial resolution vs neural-network image resolution

Make sure the final model does not accidentally confuse:

**physical Sentinel-2 spatial resolution**

with

**neural-network tensor dimensions**.

For example, if a patch represents approximately 1.2 km × 1.2 km at 10 m:

```text
120 × 120 pixels
```

and we resize it to:

```text
224 × 224
```

we have NOT created new spatial information.

Explain this clearly in the documentation.

I care about preserving the true spatial information.

---

# 11. TerraFM pretrained weights

Use the official pretrained TerraFM weights from:

**https://huggingface.co/MBZUAI/TerraFM**

Do not train TerraFM from scratch.

The code should:

1. Download/load the correct pretrained model.
2. Verify the checkpoint.
3. Load the pretrained weights correctly.
4. Clearly report missing/unexpected keys.
5. Fail loudly if weights do not match instead of silently ignoring major mismatches.

Store the downloaded pretrained weights in a persistent Colab location where practical.

Do not repeatedly download the weights every epoch.

---

# 12. Freeze/unfreeze strategy

I specifically want you to decide what to freeze.

Do not simply say "freeze the encoder" or "fine-tune everything."

Analyze TerraFM's size, pretrained nature, and T4 memory constraints.

I want a staged strategy.

For example:

### Stage 0 — smoke test

Freeze TerraFM completely.

Train only the decoder.

Purpose:

* verify data pipeline
* verify TerraFM forward pass
* verify segmentation decoder
* verify loss
* verify labels.

### Stage 1

Keep most TerraFM blocks frozen.

Unfreeze a small number of final transformer blocks.

Train decoder + final blocks.

### Stage 2

If validation mIoU improves enough, optionally use:

* adapters
* LoRA
* partial encoder fine-tuning.

I want YOU to determine whether an adapter is actually necessary.

If it is necessary, select a sensible method and explain:

* where adapters are inserted,
* why,
* adapter rank,
* dropout,
* which modules are targeted,
* whether T4 memory is reduced,
* whether it is better than simply unfreezing the final blocks.

Do not use an adapter just because it sounds sophisticated.

If partial fine-tuning is more appropriate than LoRA for TerraFM-B on a T4, choose partial fine-tuning.

---

# 13. Hyperparameter selection

I want a carefully justified initial hyperparameter configuration.

Do not give random numbers.

Choose values appropriate for:

* TerraFM-B
* segmentation decoder
* 50,000 samples
* 19 classes
* T4
* mixed precision.

At minimum determine:

```text
optimizer
learning rate
encoder learning rate
decoder learning rate
weight decay
batch size
gradient accumulation
epochs
warmup ratio
scheduler
minimum learning rate
gradient clipping
precision
dropout
loss
validation frequency
checkpoint frequency
early stopping patience
```

I want differential learning rates if appropriate.

For example:

```text
decoder LR > encoder LR
```

because the decoder is newly initialized.

But only use this if justified.

---

# 14. Loss function

Choose the most appropriate loss for this dataset.

Possible components:

* Cross Entropy
* weighted Cross Entropy
* Dice loss
* Focal loss
* Lovász-Softmax loss
* combination of CE + Dice
* other appropriate segmentation loss.

The dataset likely has class imbalance.

Therefore:

1. Calculate class frequencies from the training set.
2. Report them.
3. Determine reasonable class weights.
4. Prevent extremely rare classes from receiving unstable weights.
5. Use an appropriate combined loss if beneficial.

Do not blindly use inverse-frequency weights.

Explain why your chosen loss is appropriate.

Also ensure:

**ignore-index pixels do not contribute to the loss.**

---

# 15. Train/validation/test split

Do NOT randomly split 50,000 patches if there is spatial correlation.

This is a geospatial dataset.

Adjacent patches can be highly correlated.

Therefore design a leakage-resistant split.

The CSV and patch metadata may provide enough information to identify geographic/tile groups.

The code should determine whether there are:

* geographic IDs
* Sentinel tile IDs
* scene IDs
* countries
* acquisition dates
* other grouping fields.

If official train/validation/test assignments already exist in the data metadata, use them.

Otherwise implement a group-based split.

The split must guarantee:

> no geographic group used in test appears in training.

Prefer:

```text
Train: ~80-90%
Validation: ~5-10%
Test: ~5-10%
```

but prioritize geographic independence over exact percentages.

I want the code to save the resulting split to:

```text
train.csv
val.csv
test.csv
```

so that every future run uses exactly the same split.

The split generation must be deterministic using a fixed seed.

---

# 16. Prevent data leakage

The following must be strictly fit using training data only:

* normalization statistics
* class statistics
* class weights
* any learned preprocessing.

Do not calculate normalization statistics using validation/test imagery.

Do not tune hyperparameters using the test set.

The test set should be used only for final evaluation.

---

# 17. Sentinel-2 normalization

Determine the exact scaling of my TIFF values.

The code should inspect:

* dtype
* min
* max
* NoData
* metadata.

Do not blindly divide by 10000 unless the actual TIFF storage convention confirms this.

Determine whether the data are:

* raw DN
* scaled reflectance
* already normalized.

If they are scaled Sentinel-2 L2A values, use the correct reflectance conversion.

Then determine the normalization expected by TerraFM.

This is extremely important.

I want the preprocessing code to have a clear function such as conceptually:

```text
read_band
→ convert_to_reflectance
→ handle_nodata
→ clip_valid_range_if_justified
→ align_resolution
→ normalize
→ stack_channels
→ tensor
```

The code comments should explain WHY each stage is performed.

---

# 18. NoData and invalid pixels

Handle:

* TIFF NoData
* invalid reflectance
* missing files
* masked pixels
* invalid reference labels.

Do not replace invalid label pixels with a real semantic class.

Use the ignore index.

For image pixels, choose a sensible strategy such as:

* zero after normalization,
* masked values,
* another strategy justified by TerraFM.

Document it.

---

# 19. Data augmentation

Use augmentations appropriate for satellite imagery.

Consider:

* horizontal flip
* vertical flip
* 90° rotation
* random crop where appropriate.

Avoid arbitrary natural-image transformations such as:

* strong hue shift
* strong RGB color jitter.

Remember that multispectral channels represent physical measurements.

Do not break their spectral relationships.

If using geometric augmentation, apply EXACTLY the same transform to:

* all Sentinel-2 bands
* S1 bands if used
* reference mask.

Use nearest-neighbor for masks.

---

# 20. Efficient TIFF loading

This is extremely important on Colab.

My dataset is composed of many TIFF files.

Do not design a pipeline that repeatedly performs expensive preprocessing during every epoch if we can avoid it.

Design an efficient strategy.

Consider:

### Option A

Preprocess all input patches once into a cached format.

### Option B

Cache resized/aligned patches.

### Option C

Precompute normalization.

### Option D

Convert to a faster tensor format.

Recommend the best solution for approximately 50,000 patches while considering Colab storage constraints.

The pipeline should avoid making the T4 wait for slow CPU TIFF operations as much as possible.

Explain the storage tradeoff.

---

# 21. Dataset validation before training

Before training, create a validation/preflight stage.

It should verify:

* number of samples
* missing S2 files
* missing reference maps
* duplicate patch IDs
* number of channels
* channel ordering
* image dimensions
* CRS
* resolution
* class IDs
* percentage of ignored pixels
* percentage of each class
* NaN/Inf values
* extreme reflectance values.

Generate a concise dataset report.

Also visualize several examples:

```text
S2 composite
reference mask
class legend
```

and ideally one panel showing selected spectral bands.

Save these diagnostics.

---

# 22. Training stages

Implement the training strategy as clear phases.

### Phase 0 — 20 to 100 samples

Purpose:

* check pipeline
* overfit a tiny dataset
* confirm the model can almost perfectly fit a tiny batch.

This is mandatory.

Explain why a tiny-data overfitting test is important.

### Phase 1 — 1,000 to 2,000 samples

Purpose:

* validate real training behavior
* test TerraFM + decoder
* benchmark T4 speed and memory.

### Phase 2 — full training set

Use all training samples.

Track validation mIoU every epoch.

### Phase 3 — optional encoder adaptation

If the frozen encoder underperforms:

* unfreeze selected TerraFM layers or use adapter.

Explain the decision rule.

---

# 23. Epochs and early stopping

Do NOT arbitrarily say "train for 100 epochs."

Give a sensible starting range.

For example, determine whether something like:

```text
20–40 epochs
```

is appropriate, but base it on:

* 50k samples
* batch size
* trainable parameter count
* pretrained encoder
* validation convergence.

Use:

* early stopping
* best checkpoint selection
* LR scheduler.

The best model should be selected using:

**validation mIoU**

not training loss.

---

# 24. Batch size and gradient accumulation

Because I have a T4:

Determine the largest stable batch size that is likely to fit.

Prefer something such as:

```text
batch_size = 4
```

or

```text
batch_size = 8
```

if appropriate.

Use gradient accumulation if necessary.

Report:

```text
physical batch size
gradient accumulation
effective batch size
```

and explain why.

The code should automatically detect CUDA memory problems where practical and make the batch size configurable.

---

# 25. Mixed precision

Use mixed precision appropriately.

Prefer BF16 if the T4 supports it correctly; otherwise use FP16.

Verify T4 compatibility rather than assuming.

Use:

* autocast
* gradient scaling when needed.

Explain this in the training code.

---

# 26. Optimizer and scheduler

Use AdamW unless the architecture strongly suggests something else.

Use separate parameter groups if appropriate:

```text
TerraFM encoder
segmentation decoder
```

with potentially different learning rates.

Use warmup.

Use cosine decay or another appropriate schedule.

Document the reasoning.

---

# 27. Evaluation metrics

I want rigorous semantic-segmentation evaluation.

At minimum compute:

### Primary

* Mean IoU
* per-class IoU

### Secondary

* Dice/F1
* pixel accuracy
* precision
* recall
* confusion matrix.

Also report:

* macro metrics
* frequency-weighted metrics where appropriate.

Do NOT allow ignored pixels to affect the metrics.

Report metrics both:

1. across all classes,
2. per individual class.

Create a table such as:

```text
Class | IoU | Dice | Precision | Recall | Pixel Count
```

Sort or display according to the class mapping.

---

# 28. Evaluation outputs

After testing, save:

```text
results/
├── metrics.json
├── per_class_metrics.csv
├── confusion_matrix.csv
├── confusion_matrix.png
├── training_curves.png
├── validation_curves.png
└── qualitative_predictions/
```

Generate qualitative examples showing:

```text
Input RGB/composite
Ground truth
Prediction
Error map
```

for randomly selected test patches.

Also specifically inspect examples containing rare classes.

---

# 29. Checkpointing

This is important because Colab sessions can terminate.

Implement robust checkpoints.

Save at least:

```text
checkpoint_epoch_X.pth
best_model.pth
last_checkpoint.pth
```

A checkpoint should contain:

* model state_dict
* optimizer state
* scheduler state
* epoch
* best validation mIoU
* scaler state
* configuration
* class mapping
* normalization statistics
* random seed.

The user should be able to resume training after a Colab restart.

---

# 30. Critical requirement: final output must be ONE easily usable model file

At the end of training I want a final model artifact that is easy to use for inference.

Ideally:

```text
terrafm_lulc_model.pth
```

or another single file.

It should contain enough information to reconstruct:

* TerraFM encoder
* segmentation decoder
* segmentation head
* configuration
* class mapping
* preprocessing configuration.

Do NOT require me to manually reconstruct the architecture from memory.

The inference script should load this single trained model file and run prediction.

If storing the full model object is unsafe or not portable, then save a checkpoint that contains:

```text
architecture configuration
+
state_dict
+
preprocessing metadata
+
class mapping
```

and have one model factory reconstruct the architecture automatically.

Explain which approach you choose and why.

---

# 31. Inference code

Provide a complete inference script.

Input:

```text
path to a Sentinel-2 patch directory
```

containing the 12 TIFF bands.

The script should:

1. discover the bands,
2. preprocess exactly like training,
3. load the single trained model file,
4. run inference,
5. produce per-pixel predictions,
6. convert class IDs back to their class names,
7. save the predicted mask as a GeoTIFF.

The output GeoTIFF should preserve the appropriate geospatial metadata:

* CRS
* transform
* spatial extent.

Also produce:

```text
prediction_visualization.png
```

with a readable color/class legend.

---

# 32. Inference on a full larger Sentinel-2 image

Also explain how the trained model should be used if I later have a large Sentinel-2 scene rather than one 120×120 patch.

Implement or at least provide a well-designed inference path supporting:

```text
large raster
→ sliding windows / tiles
→ overlapping prediction
→ stitching
→ final LULC GeoTIFF
```

Handle tile boundaries properly.

Use overlap if appropriate.

Explain how overlapping predictions are combined.

---

# 33. Model output details

Be very explicit.

During training:

```text
Input:
[B, 12, H, W]

Output:
[B, 19, H_out, W_out]
```

The decoder should upsample to the reference-mask resolution.

Then:

```text
prediction = argmax(logits, dim=1)
```

resulting in:

```text
[B, H, W]
```

where:

```text
0 ... 18 = LULC classes
```

and:

```text
255 = ignore
```

if appropriate.

Make sure the model output and reference-mask dimensions always match before loss calculation.

---

# 34. Class imbalance analysis

Before training:

1. Calculate pixel frequency of all 19 classes.
2. Calculate the number of images containing each class.
3. Identify extremely rare classes.
4. Show these statistics.

Then select a sensible loss weighting strategy.

Also consider whether a class-balanced sampler is useful.

Do not use aggressive oversampling that destroys geographic diversity.

Explain the chosen sampling approach.

---

# 35. Reproducibility

Everything should be reproducible.

Set:

* random seed
* numpy seed
* PyTorch seed
* CUDA seed where appropriate.

Save:

```text
config.yaml
```

or an equivalent configuration file containing:

* paths
* model
* band order
* image size
* normalization
* batch size
* LR
* epochs
* loss
* class mapping
* random seed.

The exact same configuration should be usable for training and inference.

---

# 36. Project structure

I want the project organized into clean Python files, NOT one massive notebook.

Design something like:

```text
project/
│
├── config.py
├── dataset.py
├── preprocessing.py
├── terrafm_encoder.py
├── segmentation_decoder.py
├── model.py
├── losses.py
├── metrics.py
├── train.py
├── evaluate.py
├── inference.py
├── utils.py
├── checkpoint.py
├── visualize.py
├── prepare_dataset.py
├── requirements.txt
└── README.md
```

You may modify this structure if you have a better organization.

Every file should have:

* a clear purpose,
* module-level documentation,
* useful comments,
* simple function names,
* type hints where practical,
* clear error messages.

Avoid unnecessarily complicated abstractions.

My teammate should be able to read the code and understand the entire pipeline without being an expert in the original TerraFM paper.

---

# 37. Google Colab notebook

In addition to the Python files, create a simple Colab workflow.

The notebook should conceptually perform:

### Cell 1

Install dependencies.

### Cell 2

Mount Google Drive.

### Cell 3

Set project paths.

### Cell 4

Download/load TerraFM pretrained weights.

### Cell 5

Run dataset validation.

### Cell 6

Create deterministic train/val/test splits.

### Cell 7

Calculate class statistics.

### Cell 8

Visualize training examples.

### Cell 9

Run tiny-dataset overfit test.

### Cell 10

Run small 1,000–2,000 sample experiment.

### Cell 11

Train full model.

### Cell 12

Evaluate best checkpoint.

### Cell 13

Generate qualitative results.

### Cell 14

Export final single model file.

### Cell 15

Run inference on a new patch.

The notebook should call the Python modules rather than contain all the implementation itself.

---

# 38. Performance monitoring

During training print a clean progress line such as:

```text
Epoch 03/30
Train Loss: 0.842
Val Loss: 0.791
Val mIoU: 0.613
Val Dice: 0.731
LR: 1.00e-4
GPU: 13.2 GB
Time: 18m 42s
```

Use a progress bar where appropriate.

Track:

* train loss
* validation loss
* validation mIoU
* learning rate
* epoch time
* GPU memory.

Save these to:

```text
training_history.csv
```

---

# 39. Checkpoint resume

The training script should support:

```text
resume_from = "path/to/last_checkpoint.pth"
```

and continue correctly.

Do not restart:

* optimizer
* scheduler
* epoch count
* best metric

when resuming.

---

# 40. T4 memory safety

Design specifically for 16 GB VRAM.

Use:

* AMP
* gradient accumulation
* frozen layers when appropriate
* gradient checkpointing if beneficial and actually necessary.

Do not activate gradient checkpointing automatically if it significantly slows training without being necessary.

Explain which memory-saving strategies are enabled by default and which are optional.

---

# 41. TerraFM-B vs TerraFM-L

I want the code to support:

```text
model_size = "base"
```

and optionally:

```text
model_size = "large"
```

but make Base the default for T4.

The architecture should not require rewriting the entire project to switch between them.

Explain expected:

* memory
* speed
* accuracy potential
* practicality.

---

# 42. Do NOT silently make architectural assumptions

There are several things I explicitly want you to verify rather than assume:

* exact TerraFM channels,
* exact channel order,
* exact normalization,
* exact input size,
* exact pretrained checkpoint,
* exact feature extraction method,
* exact output feature dimensions,
* whether intermediate ViT layers are exposed,
* whether TerraFM's S1/S2 fusion is required,
* whether 12-channel S2-only mode is officially supported,
* whether the official implementation allows arbitrary image sizes,
* how its patch embedding works.

If something cannot be confirmed from the official implementation, say so and design the safest solution.

---

# 43. Prefer a stable implementation over a clever one

My priorities are:

1. Correctness
2. Reproducibility
3. Good LULC accuracy
4. T4 compatibility
5. Easy inference
6. Easy maintenance
7. Training speed.

Do NOT optimize for architectural novelty.

A clean TerraFM-B + robust segmentation decoder that trains reliably is better than an unnecessarily sophisticated architecture that barely fits into memory.

---

# 44. Important: compare decoder choices before coding

Before generating the final implementation, provide a brief architectural decision section:

```text
Decoder Candidate | Advantages | Disadvantages | T4 Cost | Final decision
```

Then choose exactly one decoder.

Explain why.

---

# 45. Important: decide whether adapters are necessary

Similarly provide:

```text
Strategy | Memory | Accuracy potential | Complexity | Recommendation
```

for:

* frozen encoder
* last-N-block fine-tuning
* LoRA
* adapters
* full fine-tuning.

Then choose the preferred strategy for T4.

My expectation is that you may choose something like:

```text
Stage 1:
Frozen TerraFM + train decoder

Stage 2:
Unfreeze last few TerraFM blocks

Stage 3:
Optional lightweight adapter/LoRA only if Stage 2 is insufficient
```

But DO NOT assume this.

Make the decision based on the actual TerraFM architecture and T4 memory.

---

# 46. Training strategy I expect

A likely overall flow is:

```text
                 Sentinel-2 TIFFs
                       │
                       ▼
               Dataset validation
                       │
                       ▼
              Geographic split
                       │
                       ▼
             Train/Val/Test CSV
                       │
                       ▼
              Band preprocessing
                       │
                       ▼
             12-channel tensor
                       │
                       ▼
                  TerraFM-B
                pretrained encoder
                       │
                       ▼
             segmentation decoder
                       │
                       ▼
                 19-class logits
                       │
                       ▼
             CE/Dice-type loss
                       │
                       ▼
              validation mIoU
                       │
                       ▼
             best checkpoint
                       │
                       ▼
           final single model file
                       │
                       ▼
                 inference
                       │
                       ▼
             GeoTIFF LULC map
```

Refine this based on your technical analysis.

---

# 47. Research-quality evaluation

I want the final result to be suitable for a serious ML/remote-sensing project.

Therefore explain:

* why geographic splitting matters,
* why mIoU matters,
* why overall accuracy alone is insufficient,
* why per-class IoU matters,
* how class imbalance affects evaluation,
* what constitutes overfitting,
* how to identify problematic classes.

Also identify common confusion patterns that I should look for.

---

# 48. Qualitative visualization

Create visualizations where each LULC class has a consistent color.

Use the same color mapping across:

* ground truth
* prediction
* legends
* future inference.

The color map should be saved as metadata/configuration so it is reproducible.

Do not use random colors per run.

---

# 49. Final deliverables

I want the final answer to provide:

## A. Architecture explanation

Very clearly:

```text
Input
→ preprocessing
→ TerraFM encoder
→ decoder
→ 19-class head
→ output
```

with dimensions at each major step.

## B. Complete project structure

Show the directory tree.

## C. Complete Python files

Provide all required code.

Do not omit important sections with phrases such as:

> "implement similarly"

or

> "code omitted for brevity."

I want complete runnable code.

## D. Colab workflow

Explain exactly which cells/commands to run and in what order.

## E. Hyperparameter table

Show every important hyperparameter and the chosen value.

## F. Training procedure

Explain Phase 0, Phase 1, Phase 2, and optional Phase 3.

## G. Evaluation

Provide the exact evaluation workflow.

## H. Inference

Provide complete inference code.

## I. Checkpointing

Explain exactly which file is the best checkpoint and how to resume.

## J. Final model artifact

Ensure the workflow produces ONE easy-to-use trained model file such as:

```text
terrafm_lulc_model.pth
```

which the inference script can load directly.

---

# 50. Code quality requirements

The code must follow these principles:

* Python 3
* PyTorch
* rasterio for GeoTIFF handling
* NumPy
* pandas
* scikit-learn only where genuinely useful
* matplotlib for visualizations
* tqdm for progress
* Hugging Face utilities only where appropriate
* TerraFM official implementation wherever possible.

Avoid unnecessary dependencies.

Use:

* clear variable names,
* functions with one responsibility,
* docstrings,
* comments explaining WHY, not just WHAT,
* explicit error checking.

Do not hide errors.

Do not catch broad exceptions unless there is a strong reason.

---

# 51. Documentation requirements

Every major file should explain:

```text
What this file does
What goes in
What comes out
How it connects to the other files
```

The README should explain the complete project in simple language.

A teammate who knows basic PyTorch but has not read the TerraFM paper should be able to understand the project.

---

# 52. Important scientific caveat

My reference maps are pixel-level masks, but they may contain:

* boundary noise,
* class ambiguity,
* spatial mismatch,
* unlabeled areas,
* rare classes.

Do not assume 100% clean labels.

Make the training and evaluation pipeline robust to this.

---

# 53. Practical target

I am NOT asking you to guarantee an accuracy number.

I want a pipeline that gives me the best reasonable chance of achieving strong segmentation performance on my 19 classes with the data I have.

The model should be evaluated empirically.

Do not claim:

> "TerraFM will achieve X% accuracy."

Instead explain what would indicate a successful run.

---

# 54. FINAL OUTPUT FORMAT FOR YOUR RESPONSE

Structure your response exactly like this:

## Part 1 — Final architecture choice

Tell me:

* TerraFM-B or TerraFM-L
* S2 only or S1 + S2
* decoder choice
* freeze/unfreeze strategy
* adapter/LoRA or not
* why.

## Part 2 — Data pipeline

Explain:

```text
raw TIFF
→ validation
→ spatial alignment
→ reflectance conversion
→ normalization
→ augmentation
→ tensor
```

## Part 3 — Label pipeline

Explain:

```text
reference TIFF
→ class mapping
→ ignore index
→ final mask
```

## Part 4 — Train/val/test split

Explain exactly how leakage is prevented.

## Part 5 — Hyperparameters

Give the final recommended values in a table.

## Part 6 — Project structure

Show the full directory structure.

## Part 7 — Complete code

Provide every required Python file in full.

## Part 8 — Google Colab instructions

Give a step-by-step execution order.

## Part 9 — Training procedure

Explain:

* tiny overfit test
* pilot training
* full training
* optional fine-tuning stage.

## Part 10 — Evaluation

Explain:

* metrics
* per-class results
* confusion matrix
* qualitative analysis.

## Part 11 — Checkpoints

Explain:

* best model
* latest model
* resume training
* final export.

## Part 12 — Inference

Provide complete inference code and explain:

```text
new S2 patch
→ preprocess
→ model
→ 19-class mask
→ GeoTIFF
→ visualization
```

## Part 13 — Large-image inference

Explain sliding-window inference and stitching.

## Part 14 — Final exact steps

End with a very simple numbered list:

```text
1.
2.
3.
4.
...
```

This final list should tell me exactly what I need to do from an empty Google Colab session until I have:

```text
terrafm_lulc_model.pth
```

and can run inference on a new Sentinel-2 patch.

---

# 55. Most important instruction

Before producing code, reason through the architecture carefully.

Do not blindly implement:

```text
TerraFM + generic U-Net
```

without checking whether that is technically compatible with TerraFM's ViT feature representation.

The final implementation must use TerraFM's pretrained representations in a scientifically sensible way.

Verify the official TerraFM repository/model card first.

Then choose the decoder and feature-extraction mechanism.

The goal is not merely to produce code that runs.

The goal is to produce a **correct, trainable, memory-conscious, reproducible TerraFM-based 19-class Sentinel-2 semantic segmentation system for a single T4 GPU.**

Also distinguish clearly between:

* what is confirmed by the official TerraFM implementation,
* what is your engineering recommendation,
* and what is a tunable hyperparameter that should be experimentally validated.
