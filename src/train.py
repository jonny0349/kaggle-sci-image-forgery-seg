"""
train.py

Purpose
-------
Minimal, readable training loop for binary segmentation:
- Loads config (YAML)
- Builds DataLoaders (albumentations)
- Builds model, loss, optimizer, scheduler
- Trains with AMP (mixed precision) if enabled
- Logs to console and TensorBoard
- Saves best checkpoints by validation Dice

Usage
-----
python src/train.py --cfg configs/baseline.yaml
"""

from __future__ import annotations
import os
import argparse
from typing import Dict

import torch
from torch import optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

from src.utils import load_config, set_seed, build_output_tree, get_logger, ensure_dir, save_json
from src.data import make_dataloaders
from src.model import build_model
from src.losses import make_loss
from src.metrics import dice_coef, iou_coef


def create_scheduler(optimizer, cfg):
    """
    Cosine annealing with a short warmup, implemented with PyTorch schedulers.
    - Warmup: linear increase to base LR over 'warmup_epochs'
    - Then cosine anneal to min_lr over (max_epochs - warmup_epochs)
    """
    warmup_epochs = int(cfg.scheduler.warmup_epochs)
    max_epochs = int(cfg.scheduler.max_epochs)
    min_lr = float(cfg.scheduler.min_lr)

    if warmup_epochs > 0:
        # LambdaLR for warmup
        def warmup_lambda(epoch):
            return (epoch + 1) / max(1, warmup_epochs)
        warmup = optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=warmup_lambda)
        # Main cosine
        cosine = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, max_epochs - warmup_epochs), eta_min=min_lr)
        return warmup, cosine
    else:
        cosine = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, max_epochs), eta_min=min_lr)
        return None, cosine


def train_one_epoch(model, loader, loss_fn, optimizer, scaler, device, amp):
    model.train()
    running_loss = 0.0

    for imgs, masks, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if amp:
            with autocast():
                logits = model(imgs)
                loss = loss_fn(logits, masks)
            if torch.isnan(loss):
                continue
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs)
            loss = loss_fn(logits, masks)
            if torch.isnan(loss):
                continue
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * imgs.size(0)

    return running_loss / len(loader.dataset)


@torch.no_grad()
def validate_one_epoch(model, loader, device, thr=0.5):
    model.eval()
    dices, ious = [], []
    for imgs, masks, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        logits = model(imgs)
        probs = torch.sigmoid(logits)
        dices.append(dice_coef(probs, masks, thr=thr).item())
        ious.append(iou_coef(probs, masks, thr=thr).item())
    return sum(dices) / len(dices), sum(ious) / len(ious)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True,
                        help="Path to YAML config")
    args = parser.parse_args()

    # Load config and prep reproducibility
    cfg = load_config(args.cfg)
    set_seed(int(cfg.project.seed))

    # Prepare outputs and logging
    out_dir = ensure_dir(cfg.project.output_dir)
    paths = build_output_tree(out_dir)
    writer = SummaryWriter(log_dir=paths["logs"])
    logger = get_logger("train", os.path.join(paths["logs"], "train.log"))

    # Save a snapshot of config for traceability
    save_json(cfg.to_dict(), os.path.join(out_dir, "config_snapshot.json"))

    # Data
    loaders = make_dataloaders(cfg)

    # Device
    device = cfg.project.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning(
            "CUDA requested but not available. Falling back to CPU.")
        device = "cpu"

    # Model & loss
    model = build_model(cfg).to(device)
    loss_fn = make_loss(cfg.loss.name)

    # Optimizer
    if cfg.optimizer.name.lower() == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=float(
            cfg.optimizer.lr), weight_decay=float(cfg.optimizer.weight_decay))
    else:
        raise ValueError(f"Unsupported optimizer: {cfg.optimizer.name}")

    # Scheduler(s)
    warmup, cosine = create_scheduler(optimizer, cfg)

    # AMP scaler
    amp = bool(cfg.train.mixed_precision) and (device == "cuda")
    scaler = GradScaler(enabled=amp)

    # Checkpointing
    best_metric = -1.0
    best_path = None
    metric_name = cfg.checkpoint.metric
    metric_mode = cfg.checkpoint.mode  # "max" expected

    # Training loop
    logger.info("Starting training...")
    total_epochs = int(cfg.train.epochs)
    global_epoch = 0

    for epoch in range(total_epochs):
        # Step warmup or cosine appropriately
        if warmup is not None and epoch < int(cfg.scheduler.warmup_epochs):
            warmup.step()
        else:
            cosine.step()

        # Train
        train_loss = train_one_epoch(
            model, loaders["train"], loss_fn, optimizer, scaler, device, amp)

        # Validate
        val_dice, val_iou = validate_one_epoch(
            model, loaders["val"], device, thr=0.5)

        # Logs
        lr_now = optimizer.param_groups[0]["lr"]
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("val/dice", val_dice, epoch)
        writer.add_scalar("val/iou", val_iou, epoch)
        writer.add_scalar("lr", lr_now, epoch)

        logger.info(
            f"Epoch {epoch+1}/{total_epochs} | loss={train_loss:.4f} | dice={val_dice:.4f} | iou={val_iou:.4f} | lr={lr_now:.6f}")

        # Checkpointing on chosen metric (default: dice, maximize)
        current_metric = val_dice if metric_name.lower() == "dice" else val_iou
        is_better = current_metric > best_metric if metric_mode == "max" else current_metric < best_metric

        if is_better:
            best_metric = current_metric
            best_path = os.path.join(paths["ckpt"], f"best_{metric_name}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_metric": best_metric,
                },
                best_path,
            )
            logger.info(
                f"Saved new best checkpoint: {best_path} (best {metric_name}={best_metric:.4f})")

    writer.close()
    logger.info(
        f"Training finished. Best {metric_name}={best_metric:.4f}. Checkpoints in {paths['ckpt']}")


if __name__ == "__main__":
    main()
