"""
Dataset that returns (image, sample_id) for feature extraction.
Uses same transforms as Stage 1 (val/test = no augmentation).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from Stage1.data.dataset import IMAGENET_MEAN, IMAGENET_STD, get_transforms


class ImageSampleDataset(Dataset):
    """Returns (image_tensor, sample_id) for each row. No labels."""

    def __init__(self, csv_file, data_root="", img_size=224, split="val"):
        self.data = pd.read_csv(csv_file)
        self.data_root = Path(data_root) if data_root else None
        self.img_size = img_size
        self.image_col = "image_path" if "image_path" in self.data.columns else "img_path"
        # Use val/test transform (no augmentation) for deterministic features
        self.transform = get_transforms(split, img_size)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        sample_id = str(row["sample_id"])
        img_path = Path(row[self.image_col]).expanduser()
        if not img_path.is_absolute() and self.data_root is not None:
            img_path = self.data_root / img_path
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, sample_id
