"""
Purpose
-------
Config-driven model builder for segmentation using segmentation_models_pytorch (SMP).
We default to a light, reliable baseline: UNet with a ResNet-34 encoder.

Why SMP?
- Battle-tested encoders/decoders
- Easy to swap architectures (UNet, FPN, DeepLabV3+, etc.)
- Pretrained encoders (ImageNet) help when data is limited
"""

from __future__ import annotations
from typing import Any
import torch
import segmentation_models_pytorch as smp


def _build_model(cfg: Any) -> torch.nn.Module:
    """
    Build a segmentation model based on the YAML config.

    Expected cfg fields
    --------
    cfg.model.arch: str             # "UNet", "FPN", "DeepLabV3Plus", etc.
    cfg.model.encoder: str          # e.g., "resnet-34"
    cfg.model.encoder_weights: str or None
    cfg.model.in_channels: int      # Usually 3
    cfg.model.classes: int          # 1 for binary segmentation

    Returns
    -------
    torch.nn.Module
    """
    arch = cfg.model.arch.lower()

    common_kwargs = dict(
        encoder_name=cfg.model.encoder,
        encoder_weights=cfg.model.encoder_weights,
        in_channels=cfg.model.in_channels,
        classes=cfg.model.classes
    )

    if arch == "unet":
        model = smp.Unet(**common_kwargs)
    elif arch == "fpn":
        model = smp.FPN(**common_kwargs)
    elif arch in ("deeplabv3", "deeplabv3plus", "deeplabv3+"):
        model = smp.DeepLabV3Plus(**common_kwargs)
    else:
        raise ValueError(f"Unsupported architecture: {cfg.model.arch}")

    return model
