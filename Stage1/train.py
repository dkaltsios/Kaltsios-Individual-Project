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
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Stage1.data.dataset import SkinLesionDataset
from Stage1.models import build_model


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


def compute_class_weights(df: pd.DataFrame, label_col: str, num_classes: int) -> torch.Tensor:
    """Inverse frequency weights for CrossEntropyLoss."""
    counts = df[label_col].value_counts().reindex(range(num_classes), fill_value=0).astype(float)
    total = counts.sum()
    if total == 0:
        return torch.ones(num_classes)
    weights = total / (num_classes * counts.clip(1))
    return torch.tensor(weights.values, dtype=torch.float32)


def _to_json_safe(obj):
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_json_safe(x) for x in obj]
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def train_one_epoch(model, loader, criterion, optimizer, device, multiclass=False):
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        if not multiclass:
            labels = labels.view(-1, 1)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


def evaluate_binary(model, loader, device):
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
    try:
        pr_auc = average_precision_score(y_true, y_prob)
    except ValueError:
        pr_auc = float("nan")
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return logits, labels, {"acc": acc, "auc": auc, "pr_auc": pr_auc, "precision": precision, "recall": recall, "f1": f1}


def evaluate_multiclass(model, loader, device, num_classes, class_names=None):
    model.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())
    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    probs = torch.softmax(logits, dim=1)
    preds = logits.argmax(dim=1)
    y_true = labels.numpy()
    y_prob = probs.numpy()
    y_pred = preds.numpy()

    acc = accuracy_score(y_true, y_pred)
    precision_per_class = precision_score(
        y_true, y_pred, average=None, zero_division=0, labels=np.arange(num_classes)
    )
    recall_per_class = recall_score(
        y_true, y_pred, average=None, zero_division=0, labels=np.arange(num_classes)
    )
    f1_per_class = f1_score(
        y_true, y_pred, average=None, zero_division=0, labels=np.arange(num_classes)
    )
    precision_macro = float(precision_per_class.mean())
    recall_macro = float(recall_per_class.mean())
    f1_macro = float(f1_per_class.mean())

    try:
        roc_auc_macro = roc_auc_score(
            y_true, y_prob, multi_class="ovr", average="macro"
        )
    except ValueError:
        roc_auc_macro = float("nan")
    try:
        pr_auc_per_class = []
        for c in range(num_classes):
            y_c = (y_true == c).astype(int)
            if y_c.sum() == 0:
                pr_auc_per_class.append(float("nan"))
            else:
                pr_auc_per_class.append(
                    float(average_precision_score(y_c, y_prob[:, c]))
                )
        pr_auc_macro = float(np.nanmean(pr_auc_per_class))
    except Exception:
        pr_auc_per_class = [float("nan")] * num_classes
        pr_auc_macro = float("nan")

    names = class_names or [str(i) for i in range(num_classes)]
    per_class = {}
    for i, name in enumerate(names):
        per_class[name] = {
            "precision": float(precision_per_class[i]),
            "recall": float(recall_per_class[i]),
            "f1": float(f1_per_class[i]),
            "pr_auc": pr_auc_per_class[i] if i < len(pr_auc_per_class) else float("nan"),
        }
    metrics = {
        "acc": acc,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "roc_auc_macro": roc_auc_macro,
        "pr_auc_macro": pr_auc_macro,
        "per_class": per_class,
    }
    return logits, labels, metrics


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

    model_name = cfg["model"].get("name", "efficientnet_b0")
    num_classes = int(cfg.get("num_classes", 1))
    model = build_model(
        model_name,
        num_classes=num_classes,
        pretrained=cfg["model"].get("pretrained", True),
    )
    model.to(device)

    train_df = pd.read_csv(train_csv)
    label_col = "y_class" if "y_class" in train_df.columns else ("y" if "y" in train_df.columns else "is_malignant")
    multiclass = num_classes > 1

    if multiclass:
        class_weights = compute_class_weights(train_df, label_col, num_classes).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        label_mapping_path = data_cfg.get("label_mapping")
        if label_mapping_path:
            with open(Path(label_mapping_path)) as f:
                label_mapping = json.load(f)
            class_names = label_mapping.get("class_names", list(range(num_classes)))
        else:
            class_names = list(range(num_classes))
    else:
        pos_weight = compute_pos_weight(train_df, label_col).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        class_names = None

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.get("lr", 3e-4)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
    )

    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    if multiclass and label_mapping_path:
        import shutil
        shutil.copy(Path(label_mapping_path), out_dir / "label_mapping_multiclass.json")

    best_metric = -1.0
    history = []

    for epoch in range(1, int(cfg.get("epochs", 10)) + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, multiclass=multiclass)
        if multiclass:
            _, _, val_metrics = evaluate_multiclass(
                model, val_loader, device, num_classes, class_names
            )
            current = val_metrics["pr_auc_macro"]
            if np.isnan(current):
                current = val_metrics["f1_macro"]
        else:
            _, _, val_metrics = evaluate_binary(model, val_loader, device)
            current = val_metrics["auc"]

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
        )

        if current > best_metric:
            best_metric = current
            torch.save(model.state_dict(), out_dir / "best.pt")

        if multiclass:
            print(
                f"Epoch {epoch}: train_loss={train_loss:.4f} "
                f"val_acc={val_metrics['acc']:.4f} val_f1_macro={val_metrics['f1_macro']:.4f} "
                f"val_pr_auc_macro={val_metrics['pr_auc_macro']:.4f}"
            )
        else:
            print(
                f"Epoch {epoch}: train_loss={train_loss:.4f} "
                f"val_acc={val_metrics['acc']:.4f} val_auc={val_metrics['auc']:.4f} val_pr_auc={val_metrics['pr_auc']:.4f}"
            )

    (out_dir / "history.json").write_text(
        json.dumps(_to_json_safe(history), indent=2)
    )


if __name__ == "__main__":
    main()
