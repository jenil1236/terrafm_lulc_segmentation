"""
inference.py
============
Single-patch and large-image inference for S1+S2 LULC model.

What goes in:
    - terrafm_lulc_model.pth
    - S2 patch directory (12 TIFFs)
    - S1 patch directory (2 TIFFs: VV + VH)

What comes out:
    - <patch_id>_prediction.tif   : GeoTIFF with class IDs
    - <patch_id>_prediction_vis.png

Sliding-window large-image inference:
    Tiles a large raster with overlap, averages softmax probabilities
    using a cosine taper, takes argmax → final GeoTIFF.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from torch.cuda.amp import autocast
from tqdm import tqdm

from config import CFG
from model import load_final_model
from preprocessing import (
    preprocess_s1_patch,
    preprocess_s2_patch,
)
from visualize import build_color_map, mask_to_rgb

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single-patch inference
# ---------------------------------------------------------------------------

def patch_inference(
    model_path: str,
    s2_patch_dir: str,
    s1_patch_dir: str,
    output_dir: str = ".",
    device: Optional[str] = None,
) -> Dict[str, str]:
    """
    Predict LULC for one S2+S1 patch pair.

    Args:
        model_path:    Path to terrafm_lulc_model.pth.
        s2_patch_dir:  Directory containing 12 S2 TIFFs.
        s1_patch_dir:  Directory containing 2 S1 TIFFs (VV, VH).
        output_dir:    Where to save results.
        device:        "cuda" / "cpu" / None (auto).

    Returns:
        {"mask_tif": path, "vis_png": path}
    """
    os.makedirs(output_dir, exist_ok=True)
    dev        = device or ("cuda" if torch.cuda.is_available() else "cpu")
    device_obj = torch.device(dev)

    model, meta = load_final_model(model_path, device=dev)
    model.eval()

    s2_mean    = meta["s2_mean"]
    s2_std     = meta["s2_std"]
    s1_mean    = meta["s1_mean"]
    s1_std     = meta["s1_std"]
    img_size   = meta["image_size"]
    cn         = meta["class_names"]
    cc         = meta["class_colors"]
    ign        = meta["ignore_index"]

    # Preprocess S2
    s2_tensor, out_transform, out_crs = preprocess_s2_patch(
        s2_patch_dir, norm_mean=s2_mean, norm_std=s2_std, target_size=img_size,
    )

    # Preprocess S1 – co-registered to the S2 grid
    s1_tensor = preprocess_s1_patch(
        s1_patch_dir, norm_mean=s1_mean, norm_std=s1_std,
        target_size=img_size, ref_transform=out_transform, ref_crs=out_crs,
    )

    # Fuse → [1, 14, H, W]
    fused = torch.cat([s2_tensor, s1_tensor], dim=0).unsqueeze(0).to(device_obj)

    use_amp   = device_obj.type == "cuda"
    amp_dtype = torch.float16 if CFG.amp_dtype == "fp16" else torch.bfloat16

    with torch.no_grad():
        with autocast(enabled=use_amp, dtype=amp_dtype):
            logits = model(fused)
        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    name    = Path(s2_patch_dir).name
    tif_out = os.path.join(output_dir, f"{name}_prediction.tif")
    png_out = os.path.join(output_dir, f"{name}_prediction_vis.png")

    _save_geotiff(pred, out_transform, out_crs, tif_out, nodata=ign)
    _save_vis(pred, cn, cc, ign, png_out)

    print(f"Saved:\n  GeoTIFF: {tif_out}\n  Visual:  {png_out}")
    return {"mask_tif": tif_out, "vis_png": png_out}


# ---------------------------------------------------------------------------
# Large-image sliding-window inference
# ---------------------------------------------------------------------------

def large_image_inference(
    model_path: str,
    s2_band_paths: List[str],
    s1_band_paths: List[str],
    output_dir: str = ".",
    tile_size: int = 224,
    overlap_frac: float = 0.25,
    device: Optional[str] = None,
) -> str:
    """
    Sliding-window inference on a large Sentinel-2 + Sentinel-1 scene.

    Args:
        model_path:    Path to terrafm_lulc_model.pth.
        s2_band_paths: 12 single-band S2 TIFFs OR 1 path to a 12-band TIFF.
        s1_band_paths: 2  single-band S1 TIFFs  OR 1 path to a  2-band TIFF.
        output_dir:    Output directory.
        tile_size:     Tile spatial size (224 matches TerraFM).
        overlap_frac:  Overlap fraction (0.25 → 56 px overlap for 224-tile).
        device:        "cuda" / "cpu" / None.

    Returns:
        Path to output GeoTIFF.
    """
    os.makedirs(output_dir, exist_ok=True)
    dev        = device or ("cuda" if torch.cuda.is_available() else "cpu")
    device_obj = torch.device(dev)

    model, meta = load_final_model(model_path, device=dev)
    model.eval()

    s2_mean = meta["s2_mean"];  s2_std = meta["s2_std"]
    s1_mean = meta["s1_mean"];  s1_std = meta["s1_std"]
    cn      = meta["class_names"]
    cc      = meta["class_colors"]
    ign     = meta["ignore_index"]
    n_cls   = len(cn)

    # Read large rasters
    s2_arr, transform, crs, H, W = _read_bands(
        s2_band_paths, s2_mean, s2_std, CFG.reflectance_scale, is_s2=True
    )
    s1_arr, _, _, _, _ = _read_bands(
        s1_band_paths, s1_mean, s1_std, reflectance_scale=None, is_s2=False
    )

    # Fuse [14, H, W]
    fused_arr = np.concatenate([s2_arr, s1_arr], axis=0)

    prob_acc   = np.zeros((n_cls, H, W), dtype=np.float32)
    weight_acc = np.zeros((H, W), dtype=np.float32)
    stride     = max(1, int(tile_size * (1 - overlap_frac)))
    taper      = _cosine_taper(tile_size)
    use_amp    = device_obj.type == "cuda"
    amp_dtype  = torch.float16 if CFG.amp_dtype == "fp16" else torch.bfloat16

    rows = sorted(set(list(range(0, max(1, H - tile_size + 1), stride)) + [max(0, H - tile_size)]))
    cols = sorted(set(list(range(0, max(1, W - tile_size + 1), stride)) + [max(0, W - tile_size)]))

    with torch.no_grad():
        for r0 in tqdm(rows, desc="Sliding window"):
            r1 = min(r0 + tile_size, H)
            for c0 in cols:
                c1 = min(c0 + tile_size, W)
                th, tw = r1 - r0, c1 - c0

                tile = fused_arr[:, r0:r1, c0:c1]
                if th < tile_size or tw < tile_size:
                    pad = np.zeros((fused_arr.shape[0], tile_size, tile_size), dtype=np.float32)
                    pad[:, :th, :tw] = tile
                    tile = pad

                t = torch.from_numpy(tile).unsqueeze(0).to(device_obj)
                with autocast(enabled=use_amp, dtype=amp_dtype):
                    logits = model(t)
                probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()[:, :th, :tw]
                w = taper[:th, :tw]
                prob_acc[:, r0:r1, c0:c1]  += probs * w[np.newaxis]
                weight_acc[r0:r1, c0:c1]   += w

    weight_acc = np.where(weight_acc == 0, 1.0, weight_acc)
    pred = (prob_acc / weight_acc[np.newaxis]).argmax(axis=0).astype(np.uint8)

    tif_out = os.path.join(output_dir, "large_image_prediction.tif")
    png_out = os.path.join(output_dir, "large_image_prediction_vis.png")
    _save_geotiff(pred, transform, crs, tif_out, nodata=ign)
    _save_vis(pred, cn, cc, ign, png_out)

    print(f"Large-image prediction:\n  GeoTIFF: {tif_out}\n  Visual:  {png_out}")
    return tif_out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cosine_taper(size: int) -> np.ndarray:
    h = np.hanning(size)
    return np.outer(h, h).astype(np.float32)


def _read_bands(
    paths: List[str],
    mean: List[float],
    std: List[float],
    reflectance_scale: Optional[float],
    is_s2: bool,
) -> Tuple[np.ndarray, object, object, int, int]:
    """Read 1 multi-band or N single-band TIFFs into a normalised array."""
    if len(paths) == 1:
        with rasterio.open(paths[0]) as src:
            transform = src.transform
            crs       = src.crs
            arr       = src.read().astype(np.float32)
    else:
        with rasterio.open(paths[0]) as ref:
            transform = ref.transform
            crs       = ref.crs
            H, W      = ref.height, ref.width
        arr = np.zeros((len(paths), H, W), dtype=np.float32)
        for i, p in enumerate(paths):
            with rasterio.open(p) as src:
                d = src.read(1).astype(np.float32)
                nd = src.nodata
                if nd is not None:
                    d[d == nd] = 0.0
                arr[i] = d

    H, W = arr.shape[1], arr.shape[2]

    if is_s2 and reflectance_scale:
        arr = arr / reflectance_scale
        arr = np.clip(arr, -0.1, 1.2)
    else:
        # S1: linear → dB
        arr = np.clip(arr, 1e-10, None)
        arr = 10.0 * np.log10(arr)
        arr = np.clip(arr, -30.0, 0.0)

    mean_arr = np.array(mean, dtype=np.float32).reshape(-1, 1, 1)
    std_arr  = np.array(std,  dtype=np.float32).reshape(-1, 1, 1)
    std_arr  = np.where(std_arr < 1e-6, 1.0, std_arr)
    arr      = (arr - mean_arr) / std_arr
    return arr, transform, crs, H, W


def _save_geotiff(pred, transform, crs, path, nodata=255):
    H, W = pred.shape
    with rasterio.open(
        path, "w", driver="GTiff",
        height=H, width=W, count=1, dtype=rasterio.uint8,
        crs=crs, transform=transform, nodata=nodata, compress="deflate",
    ) as dst:
        dst.write(pred, 1)


def _save_vis(pred, class_names, class_colors, ignore_index, path):
    cmap = build_color_map(class_colors, ignore_index)
    rgb  = mask_to_rgb(pred, cmap)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(rgb); ax.axis("off"); ax.set_title("LULC Prediction", fontsize=14)
    patches = [
        mpatches.Patch(facecolor=tuple(c/255 for c in cmap[i]), label=f"{i}: {n}")
        for i, n in enumerate(class_names)
    ]
    fig.legend(handles=patches, loc="lower center",
               ncol=min(len(class_names), 5), fontsize=8,
               bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
