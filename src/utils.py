"""
utils.py

Purpose
-------
General utilities used across the project:
- Reproducibility helpers (set_seed).
- YAML config loading with attribute-style access (DotDict).
- Simple logger that writes to console and (optionally) a file.
- Output directory helpers.

Design Notes
------------
Keep utils small, explicit, and well-documented so it’s easy to learn from.
"""

from __future__ import annotations

import os
import json
import random
import logging
import pathlib
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    """
    Set random seeds for Python, NumPy, and PyTorch to improve reproducibility.
    This doesn’t guarantee bit-exact runs across all GPUs, but it stabilizes results.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # safe if no GPU present
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class DotDict:
    """
    Minimal dot-access wrapper around a nested dict so you can write:
        cfg.train.epochs
    instead of:
        cfg['train']['epochs']
    """
    _data: Dict[str, Any]

    def __getattr__(self, item: str) -> Any:
        val = self._data[item]
        if isinstance(val, dict):
            return DotDict(val)
        return val

    def to_dict(self) -> Dict[str, Any]:
        return _deep_to_plain(self._data)


def _deep_to_plain(obj: Any) -> Any:
    """Recursively convert DotDict/dicts to plain Python types for JSON/YAML dumps."""
    if isinstance(obj, DotDict):
        return {k: _deep_to_plain(v) for k, v in obj._data.items()}
    if isinstance(obj, dict):
        return {k: _deep_to_plain(v) for k, v in obj.items()}
    return obj


def load_config(path: str | os.PathLike) -> DotDict:
    """
    Load a YAML configuration file and return a DotDict.

    Parameters
    ----------
    path : str or PathLike
        Path to a .yaml file.
    """
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return DotDict(raw)


def ensure_dir(path: str | os.PathLike) -> str:
    """
    Create a directory (and parents) if it does not exist, and return its absolute path.
    """
    p = pathlib.Path(path).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return str(p)


def build_output_tree(base_dir: str) -> Dict[str, str]:
    """
    Create the standard output tree (logs/, checkpoints/, preds/) and return paths.
    Useful for keeping experiments tidy and predictable.
    """
    logs = ensure_dir(os.path.join(base_dir, "logs"))
    ckpt = ensure_dir(os.path.join(base_dir, "checkpoints"))
    preds = ensure_dir(os.path.join(base_dir, "preds"))
    return {"logs": logs, "ckpt": ckpt, "preds": preds}


def get_logger(name: str, log_file: str | None = None) -> logging.Logger:
    """
    Return a configured logger that logs to stdout and, if provided, to a file.

    Usage
    -----
    logger = get_logger("train", "outputs/logs/run1.log")
    logger.info("hello")
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def save_json(obj: Dict[str, Any], path: str) -> None:
    """
    Save a dict to pretty-printed JSON for quick experiment notes.
    """
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
