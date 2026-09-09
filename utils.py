"""
utils.py
========
General utility functions used across the project.

What this file does:
    - Seed setting for reproducibility
    - GPU memory reporting
    - Logging setup
    - Config save/load (YAML)
    - Miscellaneous helpers

How it connects:
    Imported by train.py, prepare_dataset.py, and the Colab notebook.
"""

from __future__ import annotations

import json
import logging
import os
import random
from typing import Any, Dict

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set all random seeds for full reproducibility.

    Sets: Python random, NumPy, PyTorch CPU, PyTorch CUDA.
    Also sets deterministic CUDA operations (may slightly reduce speed).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Makes CUDA ops deterministic (required for full reproducibility)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # set True to speed up fixed-size inputs


def get_gpu_memory_gb() -> float:
    """Return current GPU memory usage in GB, or 0.0 if no CUDA device."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        return round(allocated, 2)
    return 0.0


def get_gpu_memory_reserved_gb() -> float:
    """Return reserved (cached) GPU memory in GB."""
    if torch.cuda.is_available():
        return round(torch.cuda.memory_reserved() / 1e9, 2)
    return 0.0


def setup_logging(log_level: str = "INFO", log_file: str = None) -> None:
    """
    Configure root logger with console + optional file handler.

    Args:
        log_level: Logging level ("DEBUG", "INFO", "WARNING", "ERROR").
        log_file:  Optional path to write logs to a file.
    """
    handlers = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )


def save_config(config, path: str) -> None:
    """
    Save the Config dataclass to a JSON file.

    Args:
        config: CFG instance from config.py.
        path:   Output .json file path.
    """
    import dataclasses
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    d = dataclasses.asdict(config)
    # Remove non-serializable values
    for k, v in list(d.items()):
        if not isinstance(v, (str, int, float, bool, list, dict, type(None))):
            d[k] = str(v)
    with open(path, "w") as f:
        json.dump(d, f, indent=2)


def load_norm_stats(path: str):
    """Load normalization statistics from JSON."""
    with open(path) as f:
        data = json.load(f)
    return data["mean"], data["std"]


def load_class_mapping(path: str) -> Dict[int, int]:
    """Load raw→train class ID mapping from JSON."""
    with open(path) as f:
        data = json.load(f)
    return {int(k): int(v) for k, v in data["raw_to_train"].items()}


def load_class_stats(path: str) -> Dict:
    """Load class pixel/image counts from JSON."""
    with open(path) as f:
        data = json.load(f)
    # Keys are strings in JSON; convert to int
    return {
        "pixel_counts": {int(k): v for k, v in data["pixel_counts"].items()},
        "image_counts": {int(k): v for k, v in data["image_counts"].items()},
    }


def check_amp_support() -> str:
    """
    Detect whether FP16 or BF16 AMP is supported on the current GPU.

    T4 supports FP16 (Turing arch, compute capability 7.5).
    BF16 requires Ampere (compute capability 8.0+, e.g. A100, A10).

    Returns:
        "fp16", "bf16", or "none".
    """
    if not torch.cuda.is_available():
        return "none"
    major, minor = torch.cuda.get_device_capability()
    if major >= 8:
        return "bf16"   # Ampere+ supports BF16 natively
    elif major >= 7:
        return "fp16"   # Turing/Volta: FP16 with GradScaler
    return "none"


def print_model_summary(model: torch.nn.Module) -> None:
    """Print a compact model parameter count summary."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    print(f"\nModel Summary:")
    print(f"  Total params:     {total/1e6:.2f}M")
    print(f"  Trainable params: {trainable/1e6:.2f}M")
    print(f"  Frozen params:    {frozen/1e6:.2f}M")
