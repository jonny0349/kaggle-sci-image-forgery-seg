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
    """Create the training augmentation pipeline."""
    h, w = cfg.augment.train["resize"]
    return A.Compose([
        A.Resize(h, w, interpolation=cv2.INTER_LINEAR),
        A.HorizontalFlip(p=cfg.augment.train.get("hflip_p", 0.5)),
        A.VerticalFlip(p=cfg.augment.train.get("vflip_p", 0.0)),
        A.ShiftScaleRotate(
            shift_limit=0.02, scale_limit=0.1,
            rotate_limit=cfg.augment.train.get("rotate_limit", 10),
            border_mode=cv2.BORDER_REFLECT_101,
            p=cfg.augment.train.get("rotate_p", 0.25)
        ),
        A.RandomBrightnessContrast(
            p=cfg.augment.train.get("brightness_contrast_p", 0.15)),
        A.CoarseDropout(max_holes=4, max_height=32, max_width=32, p=0.1),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])


def _build_val_transforms(cfg) -> A.BasicTransform:
    """Validation pipeline: deterministic resize + normalization."""
    h, w = cfg.augment.val["resize"]
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
        Folder containing corresponding mask files.
    img_ext : str
        Image extension (e.g., ".png", ".jpg").
    mask_ext : str
        Mask extension (e.g., ".png").
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
        transform: Optional[Callable] = None,
        background_is_zero: bool = True
    ) -> None:
        super().__init__()
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.transform = transform
        self.background_is_zero = background_is_zero
        self.ids = self._collect_ids()

    def _collect_ids(self) -> List[str]:
        ids = []
        if not os.path.isdir(self.images_dir):
            raise RuntimeError(
                f"Images directory not found: {self.images_dir}")
        if not os.path.isdir(self.masks_dir):
            raise RuntimeError(f"Masks directory not found: {self.masks_dir}")

        for f in os.listdir(self.images_dir):
            if f.lower().endswith(self.img_ext):
                stem = os.path.splitext(f)[0]
                mask_path = os.path.join(self.masks_dir, stem + self.mask_ext)
                if os.path.exists(mask_path):
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
        msk_path = os.path.join(self.masks_dir, stem + self.mask_ext)

        # Read image (BGR -> RGB)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Read mask (grayscale)
        mask = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Could not read mask: {msk_path}")

        # Binarize: any non-zero becomes 1
        if self.background_is_zero:
            mask = (mask > 0).astype(np.uint8)
        else:
            # invert if dataset encodes background differently
            mask = (mask == 0).astype(np.uint8)

        if self.transform is not None:
            transformed = self.transform(image=img, mask=mask)
            img = transformed["image"]                     # tensor [3,H,W]
            mask = transformed["mask"].unsqueeze(0).float()  # tensor [1,H,W]
        else:
            # Fallback (rare): channel-first numpy arrays
            img = np.transpose(img, (2, 0, 1))
            mask = np.expand_dims(mask, 0).astype(np.float32)

        return img, mask, stem


def make_dataloaders(cfg):
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
        transform=train_tfms,
        background_is_zero=cfg.data.background_is_zero
    )

    val_ds = ImageMaskDataset(
        images_dir=os.path.join(cfg.data.root, cfg.data.val_images_dir),
        masks_dir=os.path.join(cfg.data.root, cfg.data.val_masks_dir),
        img_ext=cfg.data.img_ext,
        mask_ext=cfg.data.mask_ext,
        transform=val_tfms,
        background_is_zero=cfg.data.background_is_zero
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=max(1, cfg.train.batch_size // 2),
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return {"train": train_loader, "val": val_loader}
