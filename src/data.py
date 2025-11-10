"""
data.py

Purpose
-------
- Dataset classes that read images and binary masks from disk.
- Albumentations augmentation pipelines for train/val.
- PyTorch DataLoaders with sane defaults.

Assumptions
-----------
- Directory layout is controlled via configs/baseline.yaml:
    data.root/train/images/*.png
    data.root/train/masks/*.png
    data.root/val/images/*.png
    data.root/val/masks/*.png
  with same filename stems between image and mask (e.g., 0001.png).

- Masks: single-channel; any non-zero is treated as 1 (forged region).
"""

from __future__ import annotations

import os
from typing import Callable, Optional, List

import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader


def _build_train_transforms(cfg) -> A.BasicTransform:
    """
    Version-safe train transforms:
      - Resize
      - Horizontal/Vertical flip
      - Rotate (not Affine / ShiftScaleRotate to avoid API drift)
      - RandomBrightnessContrast
      - Normalize + ToTensorV2
    """
    # read either dot-access or dict-style
    resize = getattr(cfg.augment.train, "resize", [768, 768])
    h, w = resize
    rotate_limit = cfg.augment.train.get("rotate_limit", 10)
    rotate_p = cfg.augment.train.get("rotate_p", 0.25)

    return A.Compose([
        A.Resize(h, w, interpolation=cv2.INTER_LINEAR),

        A.HorizontalFlip(p=cfg.augment.train.get("hflip_p", 0.5)),
        A.VerticalFlip(p=cfg.augment.train.get("vflip_p", 0.0)),

        # Rotate is widely supported and supports border_mode
        A.Rotate(limit=rotate_limit,
                 border_mode=cv2.BORDER_REFLECT_101,
                 p=rotate_p),

        A.RandomBrightnessContrast(
            p=cfg.augment.train.get("brightness_contrast_p", 0.15)),

        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])


def _build_val_transforms(cfg) -> A.BasicTransform:
    """
    Version-safe val transforms:
      - Resize
      - Normalize + ToTensorV2
    """
    resize = getattr(cfg.augment.val, "resize", [768, 768])
    h, w = resize
    return A.Compose([
        A.Resize(h, w, interpolation=cv2.INTER_LINEAR),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])


class ImageMaskDataset(Dataset):
    """
    Generic image+mask dataset.

    Parameters
    ----------
    images_dir : str
        Folder containing image files.
    masks_dir : str
        Folder containing mask files.
    img_ext : str
        Image extension (e.g., ".png", ".jpg").
    mask_ext : str
        Mask extension (e.g., ".png", ".npy").
    mask_suffix : str
        Optional suffix between stem and extension for mask files (e.g., "_mask").
    transform : albumentations transform
        Applied jointly to image and mask.
    background_is_zero : bool
        If True, non-zero mask pixels become 1 (binary).
    """

    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        img_ext: str = ".png",
        mask_ext: str = ".png",
        mask_suffix: str = "",
        transform: Optional[Callable] = None,
        background_is_zero: bool = True
    ) -> None:
        super().__init__()
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.img_ext = img_ext.lower()
        self.mask_ext = mask_ext.lower()
        self.mask_suffix = mask_suffix
        self.transform = transform
        self.background_is_zero = background_is_zero

        self.ids = self._collect_ids()

    def _mask_path(self, stem: str) -> str:
        return os.path.join(self.masks_dir, f"{stem}{self.mask_suffix}{self.mask_ext}")

    def _collect_ids(self) -> List[str]:
        if not os.path.isdir(self.images_dir):
            raise RuntimeError(
                f"Images directory not found: {self.images_dir}")
        if not os.path.isdir(self.masks_dir):
            raise RuntimeError(f"Masks directory not found: {self.masks_dir}")

        ids: List[str] = []
        for f in os.listdir(self.images_dir):
            if f.lower().endswith(self.img_ext):
                stem = os.path.splitext(f)[0]
                if os.path.exists(self._mask_path(stem)):
                    ids.append(stem)
        ids.sort()
        if len(ids) == 0:
            raise RuntimeError(
                f"No (image, mask) pairs found under {self.images_dir} and {self.masks_dir}."
            )
        return ids

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        stem = self.ids[idx]
        img_path = os.path.join(self.images_dir, stem + self.img_ext)
        msk_path = self._mask_path(stem)

        # Read image (BGR -> RGB)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Read mask
        if self.mask_ext == ".npy":
            mask = np.load(msk_path)
            # Accept common shapes/dtypes and sanitize
            if mask.ndim == 3:
                # If HxWxC, take first channel
                mask = mask[..., 0]
            mask = mask.astype(np.float32)
        else:
            mask = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(f"Could not read mask: {msk_path}")
            mask = mask.astype(np.float32)

        # Binarize:
        # If background_is_zero=True, then any non-zero -> 1
        # Else invert (treat zeros as foreground)
        if self.background_is_zero:
            mask = (mask > 0.5).astype(np.uint8)
        else:
            mask = (mask <= 0.5).astype(np.uint8)

        # After reading `img` (RGB) and `mask` (float/uint8) and binarizing...

        # --- Ensure mask shape matches image shape BEFORE Albumentations ---
        ih, iw = img.shape[:2]
        mh, mw = mask.shape[:2]
        if (mh != ih) or (mw != iw):
            # Resize mask to image size using NEAREST to preserve labels
            mask = cv2.resize(mask, (iw, ih), interpolation=cv2.INTER_NEAREST)

        # Albumentations expects dicts:
        if self.transform is not None:
            transformed = self.transform(image=img, mask=mask)
            img = transformed["image"]
            mask = transformed["mask"].unsqueeze(0).float()  # [1, H, W]
        else:
            img = np.transpose(img, (2, 0, 1))
            mask = np.expand_dims(mask, 0).astype(np.float32)

        return img, mask, stem


def make_dataloaders(cfg, pin_memory: boo = True):
    """
    Construct train/val DataLoaders based on the config.

    Returns
    -------
    dict with keys: train, val
    """
    train_tfms = _build_train_transforms(cfg)
    val_tfms = _build_val_transforms(cfg)

    train_ds = ImageMaskDataset(
        images_dir=os.path.join(cfg.data.root, cfg.data.train_images_dir),
        masks_dir=os.path.join(cfg.data.root, cfg.data.train_masks_dir),
        img_ext=cfg.data.img_ext,
        mask_ext=cfg.data.mask_ext,
        mask_suffix=cfg.data.get("mask_suffix", ""),
        transform=train_tfms,
        background_is_zero=cfg.data.background_is_zero
    )

    val_ds = ImageMaskDataset(
        images_dir=os.path.join(cfg.data.root, cfg.data.val_images_dir),
        masks_dir=os.path.join(cfg.data.root, cfg.data.val_masks_dir),
        img_ext=cfg.data.img_ext,
        mask_ext=cfg.data.mask_ext,
        mask_suffix=cfg.data.get("mask_suffix", ""),
        transform=val_tfms,
        background_is_zero=cfg.data.background_is_zero
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=max(1, cfg.train.batch_size // 2),
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return {"train": train_loader, "val": val_loader}
