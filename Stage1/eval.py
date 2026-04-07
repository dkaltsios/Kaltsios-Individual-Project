from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
)
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Stage1.data.dataset import SkinLesionDataset
from Stage1.models import build_model


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
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
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")
    try:
        pr_auc = average_precision_score(y_true, y_prob)
    except ValueError:
        pr_auc = float("nan")
    metrics = {
        "acc": accuracy_score(y_true, y_pred),
        "auc": auc,
        "pr_auc": pr_auc,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    return y_true, y_prob, y_pred, metrics


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
    return y_true, y_prob, y_pred, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--ckpt", required=True, help="Path to model checkpoint")
    parser.add_argument(
        "--split",
        choices=("test", "val"),
        default="test",
        help="Evaluate on test or val CSV from config (writes test_* or val_* prediction files).",
    )
    args = parser.parse_args()

    try:
        import yaml
    except ImportError as e:
        raise SystemExit("Missing dependency: pyyaml. Install it to use YAML configs.") from e

    cfg = yaml.safe_load(Path(args.config).read_text())

    device = get_device()

    data_cfg = cfg["data"]
    if args.split == "val":
        val_path = data_cfg.get("val_csv")
        if not val_path:
            raise SystemExit("val split requires data.val_csv in the YAML config.")
        split_csv = Path(val_path)
    else:
        split_csv = Path(data_cfg.get("test_csv", "Dataset/stage1_test.csv"))
    data_root = data_cfg.get("data_root", "")
    num_classes = int(cfg.get("num_classes", 1))
    label_mapping_path = data_cfg.get("label_mapping")

    img_size = int(cfg.get("img_size", 224))
    batch_size = int(cfg.get("batch_size", 16))
    num_workers = int(cfg.get("num_workers", 0))

    ds_split = "val" if args.split == "val" else "test"
    eval_ds = SkinLesionDataset(split_csv, data_root=data_root, split=ds_split, img_size=img_size)
    eval_loader = DataLoader(
        eval_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    model_name = cfg["model"].get("name", "efficientnet_b0")
    model = build_model(model_name, num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.to(device)

    if num_classes > 1:
        class_names = None
        if label_mapping_path and Path(label_mapping_path).exists():
            with open(Path(label_mapping_path)) as f:
                label_mapping = json.load(f)
            class_names = label_mapping.get("class_names", list(range(num_classes)))
        if class_names is None:
            class_names = [str(i) for i in range(num_classes)]
        y_true, y_prob, y_pred, metrics = evaluate_multiclass(
            model, eval_loader, device, num_classes, class_names
        )
        # JSON-safe metrics (replace nan with None)
        def _to_json_safe(obj):
            if isinstance(obj, dict):
                return {k: _to_json_safe(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_to_json_safe(x) for x in obj]
            if isinstance(obj, float) and np.isnan(obj):
                return None
            return obj
        metrics_save = _to_json_safe(metrics)
    else:
        y_true, y_prob, y_pred, metrics = evaluate_binary(model, eval_loader, device)
        metrics_save = metrics

    print(metrics_save)

    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_prefix = "val" if args.split == "val" else "test"
    (out_dir / f"{out_prefix}_metrics.json").write_text(json.dumps(metrics_save, indent=2))

    sample_ids = eval_ds.data["sample_id"].astype(str).to_numpy()

    if num_classes > 1 and class_names:
        pred_names = [class_names[int(p)] for p in y_pred]
        true_names = [class_names[int(t)] for t in y_true]
        preds_df = pd.DataFrame({
            "sample_id": sample_ids,
            "y_true": y_true,
            "y_pred": y_pred,
            "true_class": true_names,
            "predicted_class": pred_names,
        })
        for c, name in enumerate(class_names):
            preds_df[f"prob_{name}"] = y_prob[:, c]
    else:
        preds_df = pd.DataFrame(
            {"sample_id": sample_ids, "y_true": y_true, "y_prob": y_prob, "y_pred": y_pred}
        )
    preds_df.to_csv(out_dir / f"{out_prefix}_predictions.csv", index=False)


if __name__ == "__main__":
    main()
