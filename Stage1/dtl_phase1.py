"""
DTL Phase 1 – Domain Adaptation via Rotation Prediction.

Bridges the ImageNet → dermoscopy domain gap by fine-tuning the last layers
of a pre-trained backbone on unlabeled ISIC dermoscopy images.  A rotation
prediction proxy task (predict 0°/90°/180°/270° rotation) is used so that no
labels are required.

Usage
-----
    python -m Stage1.dtl_phase1 --config Stage1/configs/dtl_phase1_efficientnet.yaml
    python -m Stage1.dtl_phase1 --config Stage1/configs/dtl_phase1_resnet50.yaml

The script saves  <output_dir>/phase1.pt  – a state-dict that can be loaded
by Stage1/train.py via the  phase1_checkpoint  config key.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Stage1.models import build_model

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Dataset – self-supervised rotation prediction
# ---------------------------------------------------------------------------

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

ROTATIONS = [0, 90, 180, 270]  # degrees; label = index (0–3)


class RotationDataset(Dataset):
    """
    Loads JPEG images from a flat directory.  Each image is randomly rotated
    by 0°, 90°, 180°, or 270° and the rotation index becomes the label.
    The model therefore learns dermoscopy-domain features as a side-effect of
    solving this proxy task.
    """

    def __init__(
        self,
        image_dir: str | Path,
        img_size: int = 224,
        *,
        max_images: int | None = None,
        seed: int = 42,
    ) -> None:
        self.paths = sorted(Path(image_dir).glob("*.jpg"))
        if not self.paths:
            raise FileNotFoundError(
                f"No .jpg files found in {image_dir}. "
                "Check that the ISIC 2020 images are in that directory."
            )
        if max_images is not None and len(self.paths) > max_images:
            rng = random.Random(seed)
            self.paths = rng.sample(self.paths, max_images)
        self.img_size = img_size
        self.base_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert("RGB")
        label = random.randint(0, 3)
        img = transforms.functional.rotate(img, ROTATIONS[label])
        img = self.base_transform(img)
        return img, label


# ---------------------------------------------------------------------------
# Layer freezing
# ---------------------------------------------------------------------------

def freeze_early_layers(model: nn.Module, model_name: str) -> None:
    """
    Freeze the early (generic) layers and leave only the last feature-
    extraction layers trainable, following the DTL paper's strategy.

    EfficientNet-B0 : freeze features[0]–features[5], train features[6–8]
    ResNet-50       : freeze stem + layer1–layer3, train layer4
    """
    if model_name == "efficientnet_b0":
        freeze_up_to = {"features.0", "features.1", "features.2",
                        "features.3", "features.4", "features.5"}
        for name, param in model.named_parameters():
            top = ".".join(name.split(".")[:2])
            param.requires_grad = top not in freeze_up_to

    elif model_name == "resnet50":
        freeze_modules = {"conv1", "bn1", "layer1", "layer2", "layer3"}
        for name, param in model.named_parameters():
            top = name.split(".")[0]
            param.requires_grad = top not in freeze_modules

    else:
        raise ValueError(f"Unknown model_name for layer freezing: {model_name!r}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"  Trainable params : {trainable:,}")
    print(f"  Frozen params    : {frozen:,}")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="DTL Phase 1 – domain adaptation")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    try:
        import yaml
    except ImportError as e:
        raise SystemExit("Missing dependency: pyyaml.") from e

    cfg = yaml.safe_load(Path(args.config).read_text())

    seed       = int(cfg.get("seed", 42))
    img_size   = int(cfg.get("img_size", 224))
    batch_size = int(cfg.get("batch_size", 64))
    num_workers= int(cfg.get("num_workers", 0))
    lr         = float(cfg.get("lr", 1e-6))
    weight_decay = float(cfg.get("weight_decay", 1e-4))
    num_epochs = int(cfg.get("epochs", 40))
    patience   = int(cfg.get("patience", 8))
    model_name = cfg["model"]["name"]
    data_cfg   = cfg.get("data") or {}
    image_dir  = data_cfg["image_dir"]
    max_images = data_cfg.get("max_images")
    if max_images is not None:
        max_images = int(max_images)
    out_dir    = Path(cfg["output"]["dir"])

    set_seed(seed)
    device = get_device()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device : {device}")
    print(f"Model  : {model_name}")
    print(f"Images : {image_dir}")

    # Build model with a 4-class rotation head
    model = build_model(model_name, num_classes=4, pretrained=True)
    print("\nFreezing early layers...")
    freeze_early_layers(model, model_name)
    model.to(device)

    # Dataset & loader
    dataset = RotationDataset(
        image_dir,
        img_size=img_size,
        max_images=max_images,
        seed=seed,
    )
    print(f"\nLoaded {len(dataset):,} images for Phase 1 training.")
    if max_images is not None:
        print(f"(subset: max_images={max_images}, sampled deterministically with seed={seed})")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )

    scheduler_name = cfg.get("scheduler", "cosine")
    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=1e-8
        )
    else:
        scheduler = None

    best_loss   = float("inf")
    no_improve  = 0

    print(f"\nStarting Phase 1 training for {num_epochs} epochs "
          f"(patience={patience}, lr={lr:.2e})\n")

    for epoch in range(1, num_epochs + 1):
        loss, acc = train_one_epoch(model, loader, criterion, optimizer, device)
        if scheduler:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:3d}/{num_epochs} | loss={loss:.4f} | acc={acc:.4f} | lr={current_lr:.2e}")

        if loss < best_loss:
            best_loss  = loss
            no_improve = 0
            torch.save(model.state_dict(), out_dir / "phase1.pt")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs).")
                break

    print(f"\nPhase 1 complete. Best loss: {best_loss:.4f}")
    print(f"Saved adapted weights to: {out_dir / 'phase1.pt'}")
    print(
        "\nNext step: add  phase1_checkpoint: "
        f"{out_dir / 'phase1.pt'}  to your Stage 1 training YAML, "
        "then run train.py as normal."
    )


if __name__ == "__main__":
    main()
