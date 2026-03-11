# Cross-validation pipeline for Stage 2 Random Forest model.
# Uses train and validation to define CV pool
# Does not use test set
# Runs a StratifiedGroupKFold with n_splits=5
# At each fold it fits a preprocessor on the train rows, trains a RF and evaluates on val
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Stage2.data.preprocess import TabularPreprocessor
from Stage2.train import evaluate_multiclass


def main():
    import argparse
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="Stage2/configs/random_forest_multiclass.yaml",
        help="Path to multiclass RF YAML config",
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

    data_cfg = cfg["data"]
    stage2_path = Path(data_cfg["stage2_csv"])
    stage1_train_path = Path(data_cfg["stage1_train_csv"])
    stage1_val_path = Path(data_cfg["stage1_val_csv"])
    stage1_test_path = Path(data_cfg["stage1_test_csv"])

    # Load full multiclass metadata
    df_stage2 = pd.read_csv(stage2_path)
    df_stage2["sample_id"] = df_stage2["sample_id"].astype(str)

    label_col = "y_class"
    if label_col not in df_stage2.columns:
        raise ValueError(f"Expected '{label_col}' in {stage2_path}")

    # Determine allowed IDs for CV (train+val) and test IDs from Stage1 splits
    train_ids = set(pd.read_csv(stage1_train_path)["sample_id"].astype(str))
    val_ids = set(pd.read_csv(stage1_val_path)["sample_id"].astype(str))
    test_ids = set(pd.read_csv(stage1_test_path)["sample_id"].astype(str))

    allowed_ids = train_ids | val_ids

    df_cv = df_stage2[df_stage2["sample_id"].isin(allowed_ids)].copy()
    df_cv = df_cv.reset_index(drop=True)

    if "patient_global" not in df_cv.columns:
        raise ValueError("Expected 'patient_global' column for grouped CV.")

    y = df_cv[label_col].astype(int).to_numpy()
    groups = df_cv["patient_global"].astype(str).to_numpy()

    # Load label mapping for class names, if available
    label_mapping_path = data_cfg.get("label_mapping")
    num_classes = int(cfg.get("num_classes", 7))
    if label_mapping_path and Path(label_mapping_path).exists():
        label_mapping = json.loads(Path(label_mapping_path).read_text())
        class_names = label_mapping.get("class_names", [str(i) for i in range(num_classes)])
    else:
        class_names = [str(i) for i in range(num_classes)]

    # Set up CV
    cv = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=seed
    )

    rf_cfg = cfg["model"]

    fold_results: list[dict] = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(df_cv, y, groups)):
        df_train = df_cv.iloc[train_idx].reset_index(drop=True)
        df_val = df_cv.iloc[val_idx].reset_index(drop=True)

        # Preprocessor on train only
        id_cols = ["sample_id", "dataset_id", "patient_global"]
        if "subtype" in df_train.columns:
            id_cols.append("subtype")

        preprocessor = TabularPreprocessor()
        preprocessor.fit(df_train, label_col=label_col, id_cols=id_cols)

        X_train = preprocessor.transform(df_train)
        y_train = df_train[label_col].astype(int).to_numpy()

        X_val = preprocessor.transform(df_val)
        y_val = df_val[label_col].astype(int).to_numpy()

        model = RandomForestClassifier(
            n_estimators=int(rf_cfg.get("n_estimators", 500)),
            max_depth=rf_cfg.get("max_depth", None),
            min_samples_split=int(rf_cfg.get("min_samples_split", 2)),
            min_samples_leaf=int(rf_cfg.get("min_samples_leaf", 1)),
            max_features=rf_cfg.get("max_features", "sqrt"),
            class_weight=rf_cfg.get("class_weight", "balanced"),
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        y_val_prob = model.predict_proba(X_val)

        metrics = evaluate_multiclass(
            y_val, y_val_prob, num_classes=num_classes, class_names=class_names
        )

        fold_results.append(
            {
                "fold": fold_idx,
                "n_train": int(len(df_train)),
                "n_val": int(len(df_val)),
                "metrics": metrics,
            }
        )
        print(
            f"Fold {fold_idx}: "
            f"acc={metrics['acc']:.4f}, "
            f"f1_macro={metrics['f1_macro']:.4f}, "
            f"pr_auc_macro={metrics['pr_auc_macro']:.4f}"
        )

    # Compute mean/std over folds for top-level metrics
    def mean_std(key: str):
        vals = [fr["metrics"][key] for fr in fold_results]
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
        "test_ids_count": len(test_ids),
    }

    out_dir = PROJECT_ROOT / "Stage2" / "cv_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "random_forest_multiclass_cv.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved CV summary to {out_path}")


if __name__ == "__main__":
    main()

