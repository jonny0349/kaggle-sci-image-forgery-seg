from __future__ import annotations

import os
from dataclasses import dataclass
from glob import glob
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2


# -----------------------
# Helpers
# -----------------------

def _is_abs(p: str) -> bool:
    return os.path.isabs(p)


def resolve_path(root: str, p: str) -> str:
    """
    Join root + p unless p is already absolute.
    Prevents bugs like /kaggle/working/data/raw + /kaggle/working/data/raw/train/images.
    """
    if p is None:
        return root
    return p if _is_abs(p) else os.path.join(root, p)


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def read_image_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def mask_to_hw(mask: np.ndarray) -> np.ndarray:
    """
    Normalize masks to HxW, handling:
    - HxW
    - 1xHxW
    - HxWx1
    - 3xHxW / 4xHxW
    - HxWx3 / HxWx4

    Returns:
      HxW array (same dtype as input)
    """
    m = np.asarray(mask)

    if m.ndim == 2:
        return m

    if m.ndim == 3:
        # [1,H,W]
        if m.shape[0] == 1:
            return m[0]
        # [H,W,1]
        if m.shape[-1] == 1:
            return m[..., 0]
        # [C,H,W] where C in {3,4}
        if m.shape[0] in (3, 4) and m.shape[1] > 8 and m.shape[2] > 8:
            return m.max(axis=0)
        # [H,W,C] where C in {3,4}
        if m.shape[-1] in (3, 4) and m.shape[0] > 8 and m.shape[1] > 8:
            return m.max(axis=-1)

    m = np.squeeze(m)
    if m.ndim != 2:
        raise ValueError(f"Mask not 2D after normalization. Shape={m.shape}")
    return m


def mask_to_binary01(mask_hw: np.ndarray) -> np.ndarray:
    """
    Convert mask HxW to binary {0,1} uint8.
    """
    m = mask_hw
    # Some datasets store floats, ints, or encoded channels — we treat >0 as foreground.
    m01 = (m > 0).astype(np.uint8)
    return m01


def build_stem_to_image_map(images_dir: str, recursive: bool = True) -> Dict[str, str]:
    """
    Build {stem: absolute_image_path} supporting nested folders (Kaggle uses authentic/forged).
    """
    exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
    pattern = os.path.join(
        images_dir, "**", "*") if recursive else os.path.join(images_dir, "*")
    paths = [p for p in glob(pattern, recursive=recursive)
             if os.path.splitext(p)[1].lower() in exts]

    m: Dict[str, str] = {}
    for p in paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        # If duplicates exist, keep the first (rare). You can change policy later.
        if stem not in m:
            m[stem] = p
    return m


def build_stem_to_mask_map(masks_dir: str) -> Dict[str, str]:
    """
    Build {stem: absolute_mask_path} for .npy masks.
    """
    paths = glob(os.path.join(masks_dir, "*.npy"))
    m: Dict[str, str] = {}
    for p in paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        if stem not in m:
            m[stem] = p
    return m


# -----------------------
# Transforms
# -----------------------

def _build_train_transforms(cfg):
    h = int(cfg.data.resize_h)
    w = int(cfg.data.resize_w)

    # Keep this conservative to avoid albumentations version mismatches.
    return A.Compose(
        [
            A.Resize(h, w),
            A.HorizontalFlip(p=float(getattr(cfg.augment, "hflip_p", 0.5))),
            A.RandomBrightnessContrast(
                p=float(getattr(cfg.augment, "bc_p", 0.2))),
            A.Normalize(),
            ToTensorV2(),
        ],
        is_check_shapes=True,
    )


def _build_val_transforms(cfg):
    h = int(cfg.data.resize_h)
    w = int(cfg.data.resize_w)
    return A.Compose(
        [
            A.Resize(h, w),
            A.Normalize(),
            ToTensorV2(),
        ],
        is_check_shapes=True,
    )


# -----------------------
# Dataset
# -----------------------

class ImageMaskDataset(Dataset):
    """
    Image + mask dataset with robust stem matching and mask normalization.
    Supports:
      - flat folders: images/*.png and masks/*.npy
      - nested images (Kaggle): train_images/authentic|forged/*.png
    """

    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        ids: Optional[List[str]] = None,
        transform=None,
        recursive_images: bool = True,
    ):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.recursive_images = recursive_images

        if not os.path.isdir(images_dir):
            raise RuntimeError(f"Images directory not found: {images_dir}")
        if not os.path.isdir(masks_dir):
            raise RuntimeError(f"Masks directory not found: {masks_dir}")

        self.stem_to_img = build_stem_to_image_map(
            images_dir, recursive=recursive_images)
        self.stem_to_msk = build_stem_to_mask_map(masks_dir)

        if ids is None:
            common = sorted(set(self.stem_to_img.keys()) &
                            set(self.stem_to_msk.keys()))
            self.ids = common
        else:
            self.ids = [s for s in ids if (
                s in self.stem_to_img and s in self.stem_to_msk)]

        if len(self.ids) == 0:
            raise RuntimeError(
                f"No (image, mask) pairs found under {images_dir} and {masks_dir}.\n"
                f"Tip: if images are nested (authentic/forged), recursive scan must be enabled."
            )

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        stem = self.ids[idx]
        img_path = self.stem_to_img[stem]
        msk_path = self.stem_to_msk[stem]

        img = read_image_rgb(img_path)  # HxWx3 RGB
        mask = np.load(msk_path)

        mask = mask_to_hw(mask)
        mask01 = mask_to_binary01(mask)  # HxW {0,1}

        # If mask doesn't match image, fix here (before transforms)
        ih, iw = img.shape[:2]
        if mask01.shape != (ih, iw):
            mask01 = cv2.resize(
                mask01, (iw, ih), interpolation=cv2.INTER_NEAREST)

        if self.transform is not None:
            out = self.transform(image=img, mask=mask01)
            img_t = out["image"]  # [3,H,W]
            mask_t = out["mask"]  # [H,W]
        else:
            img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            mask_t = torch.from_numpy(mask01).long()

        # Ensure mask is [1,H,W] float32 for BCEWithLogits
        mask_t = mask_t.unsqueeze(0).float()

        return img_t, mask_t, stem


# -----------------------
# Dataloaders + layout detection
# -----------------------

def _detect_layout(root: str) -> str:
    """
    Auto-detect dataset layout.
    Returns 'raw' or 'kaggle'.
    """
    if os.path.isdir(os.path.join(root, "train_images")) and os.path.isdir(os.path.join(root, "train_masks")):
        return "kaggle"
    if os.path.isdir(os.path.join(root, "raw", "train", "images")):
        return "raw"
    return "raw"


def make_dataloaders(cfg, pin_memory: bool = False):
    """
    Creates train/val DataLoaders.
    Works on both:
      - cfg.data.root pointing to Kaggle input root (train_images/train_masks)
      - cfg.data.root pointing to a local project data directory containing raw/train/images ...
    """
    root = str(cfg.data.root)
    layout = getattr(cfg.data, "layout", "auto")
    if layout == "auto":
        layout = _detect_layout(root)

    if layout == "kaggle":
        train_images = resolve_path(root, "train_images")
        train_masks = resolve_path(root, "train_masks")
        # You will create a true val split next; for now we allow val to be provided by cfg if exists.
        val_images = resolve_path(root, getattr(
            cfg.data, "val_images", "train_images"))
        val_masks = resolve_path(root, getattr(
            cfg.data, "val_masks", "train_masks"))
        recursive = True
    else:
        train_images = resolve_path(root, "raw/train/images")
        train_masks = resolve_path(root, "raw/train/masks")
        val_images = resolve_path(root, "raw/val/images")
        val_masks = resolve_path(root, "raw/val/masks")
        recursive = False

    train_tfms = _build_train_transforms(cfg)
    val_tfms = _build_val_transforms(cfg)

    train_ds = ImageMaskDataset(
        train_images, train_masks, transform=train_tfms, recursive_images=recursive)
    val_ds = ImageMaskDataset(val_images, val_masks,
                              transform=val_tfms, recursive_images=recursive)

    bs = int(cfg.train.batch_size)
    nw = int(cfg.train.num_workers)

    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        num_workers=nw,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return {"train": train_loader, "val": val_loader}
