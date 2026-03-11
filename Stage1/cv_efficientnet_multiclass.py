"""
Cross-validation for Stage 1 EfficientNet (multiclass).

Uses StratifiedGroupKFold on train+val samples (grouped by patient_global),
keeping the existing test set untouched.

For each fold:
- Builds train/val subsets from a combined stage1_multiclass train+val CSV
- Trains EfficientNet with the same hyperparameters as efficientnet_b0_multiclass.yaml
- Evaluates multiclass metrics on the fold's validation split

Writes a JSON summary with per-fold and mean metrics to:
Stage1/cv_results/efficientnet_b0_multiclass_cv.json
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Stage1.data.dataset import SkinLesionDataset
from Stage1.models import build_model
from Stage1.train import (
    set_seed,
    get_device,
    compute_class_weights,
    train_one_epoch,
    evaluate_multiclass,
)


def main():
    import argparse
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="Stage1/configs/efficientnet_b0_multiclass.yaml",
        help="Path to EfficientNet multiclass YAML config",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of StratifiedGroupKFold folds",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())

    seed = int(cfg.get("seed", 42))
    n_splits = int(args.folds)

    set_seed(seed)
    device = get_device()

    data_cfg = cfg["data"]
    train_csv = Path(data_cfg["train_csv"])
    val_csv = Path(data_cfg["val_csv"])

    # Load train+val CSVs and build combined DataFrame
    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)
    df_cv = pd.concat([df_train, df_val], axis=0).reset_index(drop=True)

    if "y_class" not in df_cv.columns:
        raise ValueError("Expected 'y_class' column in stage1_multiclass train/val CSVs.")
    if "patient_global" not in df_cv.columns:
        raise ValueError("Expected 'patient_global' column for grouped CV.")

    y = df_cv["y_class"].astype(int).to_numpy()
    groups = df_cv["patient_global"].astype(str).to_numpy()

    num_classes = int(cfg.get("num_classes", 7))

    # Build a temporary combined CSV for the dataset
    combined_csv = PROJECT_ROOT / "Dataset" / "stage1_multiclass_trainval_cv.csv"
    df_cv.to_csv(combined_csv, index=False)

    img_size = int(cfg.get("img_size", 224))
    batch_size = int(cfg.get("batch_size", 16))
    num_workers = int(cfg.get("num_workers", 0))

    full_dataset = SkinLesionDataset(
        combined_csv, data_root=data_cfg.get("data_root", ""), split="train", img_size=img_size
    )

    # Load label mapping for class names if available
    label_mapping_path = data_cfg.get("label_mapping")
    if label_mapping_path and Path(label_mapping_path).exists():
        label_mapping = json.loads(Path(label_mapping_path).read_text())
        class_names = label_mapping.get("class_names", [str(i) for i in range(num_classes)])
    else:
        class_names = [str(i) for i in range(num_classes)]

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    fold_results: list[dict] = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(df_cv, y, groups)):
        print(f"\n=== Fold {fold_idx + 1}/{n_splits} ===")
        ds_train = Subset(full_dataset, train_idx)
        ds_val = Subset(full_dataset, val_idx)

        train_loader = DataLoader(
            ds_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False,
        )
        val_loader = DataLoader(
            ds_val,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
        )

        # Build model and optimizer fresh for each fold
        model_name = cfg["model"].get("name", "efficientnet_b0")
        model = build_model(
            model_name,
            num_classes=num_classes,
            pretrained=cfg["model"].get("pretrained", True),
        )
        model.to(device)

        # Class weights from df_cv rows for this fold's train indices
        df_fold_train = df_cv.iloc[train_idx].reset_index(drop=True)
        class_weights = compute_class_weights(df_fold_train, "y_class", num_classes).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(cfg.get("lr", 3e-4)),
            weight_decay=float(cfg.get("weight_decay", 1e-4)),
        )

        epochs = int(cfg.get("epochs", 15))
        best_f1 = -1.0
        best_metrics = None

        for epoch in range(1, epochs + 1):
            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, device, multiclass=True
            )
            _, _, val_metrics = evaluate_multiclass(
                model, val_loader, device, num_classes, class_names
            )

            f1_macro = val_metrics["f1_macro"]
            if f1_macro > best_f1:
                best_f1 = f1_macro
                best_metrics = val_metrics

            print(
                f"Fold {fold_idx}, Epoch {epoch}: "
                f"train_loss={train_loss:.4f}, "
                f"val_acc={val_metrics['acc']:.4f}, "
                f"val_f1_macro={val_metrics['f1_macro']:.4f}, "
                f"val_pr_auc_macro={val_metrics['pr_auc_macro']:.4f}"
            )

        fold_results.append(
            {
                "fold": fold_idx,
                "n_train": int(len(ds_train)),
                "n_val": int(len(ds_val)),
                "best_val_metrics": best_metrics,
            }
        )

    def mean_std(key: str):
        vals = [fr["best_val_metrics"][key] for fr in fold_results]
        return float(np.mean(vals)), float(np.std(vals))

    summary = {
        "config": {
            "config_path": str(Path(args.config)),
            "n_splits": n_splits,
            "seed": seed,
        },
        "metrics_mean_std": {
            key: {"mean": mean_std(key)[0], "std": mean_std(key)[1]}
            for key in ["acc", "precision_macro", "recall_macro", "f1_macro", "roc_auc_macro", "pr_auc_macro"]
        },
        "folds": fold_results,
    }

    out_dir = PROJECT_ROOT / "Stage1" / "cv_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "efficientnet_b0_multiclass_cv.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved EfficientNet CV summary to {out_path}")


if __name__ == "__main__":
    main()

