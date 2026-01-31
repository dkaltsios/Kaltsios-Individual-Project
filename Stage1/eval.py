from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Stage1.data.dataset import SkinLesionDataset
from Stage1.models.efficientnet import build_efficientnet_b0


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


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
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")
    metrics = {
        "acc": accuracy_score(y_true, y_pred),
        "auc": auc,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    return y_true, y_prob, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--ckpt", required=True, help="Path to model checkpoint")
    args = parser.parse_args()

    try:
        import yaml
    except ImportError as e:
        raise SystemExit("Missing dependency: pyyaml. Install it to use YAML configs.") from e

    cfg = yaml.safe_load(Path(args.config).read_text())

    device = get_device()

    data_cfg = cfg["data"]
    test_csv = Path(data_cfg.get("test_csv", "Dataset/stage1_test.csv"))
    data_root = data_cfg.get("data_root", "")

    img_size = int(cfg.get("img_size", 224))
    batch_size = int(cfg.get("batch_size", 16))
    num_workers = int(cfg.get("num_workers", 0))

    test_ds = SkinLesionDataset(test_csv, data_root=data_root, split="test", img_size=img_size)
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    model = build_efficientnet_b0(num_classes=1, pretrained=False)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.to(device)

    y_true, y_prob, metrics = evaluate(model, test_loader, device)
    print(metrics)

    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2))

    preds_df = pd.DataFrame({"y_true": y_true, "y_prob": y_prob})
    preds_df.to_csv(out_dir / "test_predictions.csv", index=False)


if __name__ == "__main__":
    main()
