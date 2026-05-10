import pandas as pd
from PIL import Image
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_transforms(split, img_size=299, augment=False):
    """
    Return a transform pipeline for the given split.

    augment=False  →  original baseline augmentation (preserves existing results).
    augment=True   →  stronger augmentation:
                       • RandomResizedCrop (scale 0.85–1.0) instead of plain Resize
                       • Stronger ColorJitter (brightness/contrast 0.4, saturation 0.3, hue 0.05)
                       • RandomErasing (p=0.5, 2–15 % area) after ToTensor
    """
    if split == "train":
        if augment:
            return transforms.Compose([
                transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(180),
                transforms.ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.3,
                    hue=0.05,
                ),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                transforms.RandomErasing(p=0.5, scale=(0.02, 0.15)),
            ])
        else:
            return transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(180),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                ),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
    else:  # val / test — never augmented
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])


class SkinLesionDataset(Dataset):
    """
    Dataset for skin lesions. Supports binary (y / is_malignant) or multiclass (y_class).
    """

    def __init__(self, csv_file, data_root="", split="train", img_size=299, augment=False):
        """
        Args:
            csv_file (str): path to stage1_{train,val,test}.csv or stage1_multiclass_*.csv
            data_root (str): root directory containing image folders
            split (str): train | val | test
            img_size (int): image size (299 for Xception)
        """
        assert split in ["train", "val", "test"]

        self.data = pd.read_csv(csv_file)
        self.data_root = Path(data_root) if data_root else None
        self.split = split
        self.transform = get_transforms(split, img_size, augment=augment)

        # Multiclass: use y_class if present
        if "y_class" in self.data.columns:
            self.label_col = "y_class"
            self.multiclass = True
        else:
            self.label_col = "y" if "y" in self.data.columns else "is_malignant"
            self.multiclass = False

        self.image_col = "image_path" if "image_path" in self.data.columns else "img_path"
        if self.image_col not in self.data.columns:
            raise ValueError("Missing image path column: expected 'image_path' or 'img_path'")
        if self.label_col not in self.data.columns:
            raise ValueError(f"Missing label column: expected '{self.label_col}'")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        img_path = Path(row[self.image_col]).expanduser()
        if not img_path.is_absolute() and self.data_root is not None:
            img_path = self.data_root / img_path
        label = int(row[self.label_col])

        # load image
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.multiclass:
            label = torch.tensor(label, dtype=torch.long)
        else:
            label = torch.tensor(label, dtype=torch.float32)

        return image, label
