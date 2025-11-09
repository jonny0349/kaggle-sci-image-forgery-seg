"""
Purpose
---------
Batch-safe Dice and IoU metrics for binary segmentation
We compute on probabilities (after sigmoid) with a configurable threshold

Notes
---------
- These are "soft-thresholds" metrics for validation: leaderboard metric may differ. 
- We threshold to binary masks for overlap computation. 
"""
from __future__ import annotations
from typing import Tuple
import torch


@torch.no_grad()
def compute_confusion(preds: torch.Tensor, targets: torch.Tensor, thr: float = 0.5) -> Tuple[torch.Tensor, ...]:
    """
    Compute TP, FP, FN per-batch given probabilities and targets.

    Inputs
    --------
    preds: [B, 1, H, W] probabilities in [0,1]
    targets: [B, 1, H, W] {0,1}

    Returns
    --------
    (tp, fp, fn) each of shape [B]
    """
    preds_b = (preds >= thr).to(torch.float32)
    targets_b = (targets > 0.5).to(torch.float32)

    # Flatten per-sample
    preds_b = preds_b.view(preds_b.size(0), -1)
    targets_b = targets_b.view(targets_b.size(0), -1)

    tp = (preds_b * targets_b).sum(dim=1)
    fp = (preds_b * (1.0 - targets_b)).sum(dim=1)
    fn = ((1.0 - preds_b) * targets_b).sum(dim=1)
    return tp, fp, fn


@torch.no_grad()
def dice_coef(preds: torch.Tensor, targets: torch.Tensor, thr: float = 0.5, eps: float = 1e-6) -> torch.Tensor:
    """
    Dice coefficient averaged over the batch
    """
    tp, fp, fn = compute_confusion(preds, targets, thr=thr)
    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    return dice.mean()


@torch.no_grad()
def iou_coef(preds: torch.Tensor, targets: torch.Tensor, thr: float = 0.5, eps: float = 1e-6) -> torch.Tensor:
    """
    IoU (Jaccard) averaged over the batch
    """
    tp, fp, fn = compute_confusion(preds, targets, thr=thr)
    iou = (tp + eps) / (tp + fp + fn + eps)
    return iou.mean()
