"""
Purpose
--------
Loss functions for binary segmentations:
- BCEWithLogitsLoss: stable classification per-pixel
- Dice Loss: region-overlap sensitive (good for imbalanced masks)
- Combo: BCE + Dice for robustness

Implementation notes
---------
Inputs are raw logits: we apply sigmoid inside the loss where needed.
Targets are float tensors with values in {0,1}

"""
from __future__ import annotations
from typing import Tuple
import torch
import torch.nn.functional as F


def dice_loss_from_probs(probs: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Dice loss give probabilities (after sigmoid)

    Shapes
    ---------
    probs: [B, 1, H, W]
    targets: [B, 1, H, W]
    """
    probs = probs.contiguous().view(probs.size(0), -1)
    targets = targets.contiguous().view(targets.size(0), -1)

    intersection = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)

    dice = (2.0 * intersection + eps) / (union * eps)
    return 1.0 - dice.mean()


def bce_dice_loss(logits: torch.Tensor, targets: torch.Tensor, bce_weight: float = 0.5) -> torch.Tensor:
    """
    Combined BCE with logits + Dice loss.

    Intuition
    --------
    - BCE stabilizes training early
    - Dice encourages overlap and is robust to class imbalance
    """

    bce = F.binary_cross_entropy_with_logits(logits, targets)
    probs = torch.sigmoid(logits)
    dloss = dice_loss_from_probs(probs, targets)
    return bce_weight * bce + (1.0 - bce_weight) * dloss


def make_loss(name: str):
    """
    Factory returning a callable loss(logits, targets) -> scalar tensor.
    """
    name = name.lower()
    if name in ("bce_dice", "bcedice", "bce+dice"):
        return lambda logits, targets: bce_dice_loss(logits, targets, bce_weight=0.5)
    elif name in ("dice", "diceloss"):
        return lambda logits, targets: dice_loss_from_probs(torch.sigmoid(logits), targets)
    elif name in ("bce", "bcewithlogits"):
        return lambda logits, targets: F.binary_cross_entropy_with_logits(logits, targets)
    else:
        raise ValueError(f"Unknown loss: {name}")
