"""
Purpose
---------
Run inference on a folder of images and save predicted binary masks (and optional overlays).

Usage
--------
Python -m src/infer \
    --cfg configs/baseline.yaml \
    --checkpoint outputs/checpoints/best_dice.pt\
    --input_dir data/raw/val/images \
    --output dir outputs/preds/val \
    --thr 0.5 \
    --save overlay

Notes
--------
- Uses the same val transform as training (resize + normalize)
- Handles PNG/JPG images.
- Overlays are saved as <stem>_overlay.png for quick visual QA
"""

from __future__ import annotations
import os
import argparse
from glob import glob
import cv2
import numpy as np
import torch
from src.utils import load_config, ensure_dir, get_logger
from src.model import build_model
# Reuse our val transforms
from src.data import _build_val_transforms  # type: ignore


def pick_device(requested: str) -> str:
    requested = str(requested).strip().lower()
    if requested in ("auto", "auto_gpu", ""):
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        return "cpu"
    if requested == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        return "cpu"
    return requested


def read_image_rgb(path: str):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_mask_png(mask01: np.ndarray, out_path: str):
    """
    mask01: HxW float/bool in {0,1} -> saved as 0/255 PNG
    """
    mask255 = (mask01.astype(np.uint8)) * 255
    cv2.imwrite(out_path, mask255)


def make_overlay(img_rgb: np.ndarray, mask01: np.ndarray, alpha: float = 0.5):
    """
    Return an RGB overlay (red mask on top of the original image)
    """
    overlay = img_rgb.copy()
    red = np.zeros_like(img_rgb)
    red[..., 0] = 255
    m3 = np.repeat(mask01[..., None].astype(np.uint8), 3, axis=2)
    overlay = np.where(m3 == 1, (alpha * red + (1 - alpha)
                       * overlay).astype(np.uint8), overlay)
    return overlay


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, type=str,
                    help="Path to YAML config")
    ap.add_argument("--checkpoint", required=True,
                    type=str, help="Path to .pt checkpoint")
    ap.add_argument("--input_dir", required=True,
                    type=str, help="Folder with images")
    ap.add_argument("--output_dir", required=True, type=str,
                    help="Where to save predicted masks")
    ap.add_argument("--thr", type=float, default=0.5,
                    help="Probability threshold for binarization")
    ap.add_argument("--save_overlay", action="store_true",
                    help="Also save <stem>_overlay.png")
    args = ap.parse_args()

    cfg = load_config(args.cfg)
    device = pick_device(cfg.project.device)
    torch_device = torch.device(device)

    ensure_dir(args.output_dir)
    logger = get_logger("infer")

    print("CWD:", os.getcwd())
    print("Input dir:", os.path.abspath(args.input_dir))
    print("Output dir:", os.path.abspath(args.output_dir))
    print("Checkpoint:", os.path.abspath(args.checkpoint))

    # Build model and load weights
    model = build_model(cfg).to(torch_device)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state_dict = ckpt["state_dict"] if isinstance(
        ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # Val transforms (resize + normalize + ToTensor)
    tfm = _build_val_transforms(cfg)

    # Collect images
    exts = (".png", ".jpeg", ".jpg", ".tif", ".tiff", ".bmp")
    paths = [p for p in glob(os.path.join(args.input_dir, "*"))
             if os.path.splitext(p)[1].lower() in exts]
    if len(paths) == 0:
        logger.warning(f"No images found under: {args.input_dir}")
        return

    print("Found images:", len(paths))

    logger.info(
        f"Device: {device} | Images: {len(paths)} | Saving to: {args.output_dir}")

    with torch.no_grad():
        for ip in paths:
            stem = os.path.splitext(os.path.basename(ip))[0]
            img_rgb = read_image_rgb(ip)
            ih, iw = img_rgb.shape[:2]

            # Albumentations expects dict; returns CHW tensor
            transformed = tfm(image=img_rgb)
            timg = transformed["image"].unsqueeze(
                0).to(torch_device)  # [1,3,H,W]

            # Forward
            logits = model(timg)
            probs = torch.sigmoid(logits)[0, 0]  # [H,W]
            mask = (probs.cpu().numpy() >= float(args.thr)).astype(np.uint8)

            # Resize mask back to original image size (nearest)
            if mask.shape != (ih, iw):
                mask = cv2.resize(
                    mask, (iw, ih), interpolation=cv2.INTER_NEAREST)

            # Save mask
            out_mask = os.path.join(args.output_dir, f"{stem}.png")
            save_mask_png(mask, out_mask)

            # Optional overlay
            if args.save_overlay:
                overlay = make_overlay(img_rgb, mask, alpha=0.5)
                out_ov = os.path.join(args.output_dir, f"{stem}_overlay.png")
                cv2.imwrite(out_ov, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    logger.info("Inference complete.")


if __name__ == '__main__':
    main()
