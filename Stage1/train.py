from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Stage1.data.dataset import SkinLesionDataset
from Stage1.models.efficientnet import build_efficientnet_b0


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


def compute_pos_weight(df: pd.DataFrame, label_col: str) -> torch.Tensor:
    pos = float((df[label_col] == 1).sum())
    neg = float((df[label_col] == 0).sum())
    if pos == 0:
        return torch.tensor(1.0)
    return torch.tensor(neg / pos)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device).view(-1, 1)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_logits = []
    all_labels = []
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu().view(-1, 1))
    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()
    y_true = labels.numpy().reshape(-1)
    y_prob = probs.numpy().reshape(-1)
    y_pred = preds.numpy().reshape(-1)
    acc = accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return logits, labels, {"acc": acc, "auc": auc, "precision": precision, "recall": recall, "f1": f1}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    try:
        import yaml
    except ImportError as e:
        raise SystemExit("Missing dependency: pyyaml. Install it to use YAML configs.") from e

    cfg = yaml.safe_load(Path(args.config).read_text())

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    device = get_device()

    data_cfg = cfg["data"]
    train_csv = Path(data_cfg["train_csv"])
    val_csv = Path(data_cfg["val_csv"])
    data_root = data_cfg.get("data_root", "")

    img_size = int(cfg.get("img_size", 224))
    batch_size = int(cfg.get("batch_size", 16))
    num_workers = int(cfg.get("num_workers", 0))

    train_ds = SkinLesionDataset(train_csv, data_root=data_root, split="train", img_size=img_size)
    val_ds = SkinLesionDataset(val_csv, data_root=data_root, split="val", img_size=img_size)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    model = build_efficientnet_b0(num_classes=1, pretrained=cfg["model"].get("pretrained", True))
    model.to(device)

    # Class imbalance handling
    train_df = pd.read_csv(train_csv)
    label_col = "y" if "y" in train_df.columns else "is_malignant"
    pos_weight = compute_pos_weight(train_df, label_col).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.get("lr", 3e-4)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
    )

    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    best_auc = -1.0
    history = []

    for epoch in range(1, int(cfg.get("epochs", 10)) + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        _, _, val_metrics = evaluate(model, val_loader, device)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
        )

        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            torch.save(model.state_dict(), out_dir / "best.pt")

        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f} "
            f"val_acc={val_metrics['acc']:.4f} val_auc={val_metrics['auc']:.4f}"
        )

    (out_dir / "history.json").write_text(json.dumps(history, indent=2))


if __name__ == "__main__":
    main()
