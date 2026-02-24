"""
Stage 4 eval: evaluate combined (image + metadata) binary classifier on test set.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import xgboost as xgb
from joblib import load
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Stage4.data.merge import load_merged_splits


def evaluate_probs(y_true: np.ndarray, y_prob: np.ndarray):
    y_pred = (y_prob >= 0.5).astype(int)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")
    try:
        pr_auc = average_precision_score(y_true, y_prob)
    except ValueError:
        pr_auc = float("nan")
    return {
        "acc": accuracy_score(y_true, y_pred),
        "auc": auc,
        "pr_auc": pr_auc,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Stage 4 YAML config")
    parser.add_argument("--ckpt", required=True, help="Path to model checkpoint (best.json or best.joblib)")
    args = parser.parse_args()

    import yaml
    cfg = yaml.safe_load(Path(args.config).read_text())

    out_dir = Path(cfg["output"]["dir"])
    features_dir = out_dir / "features"
    if not features_dir.exists():
        raise FileNotFoundError(f"Features not found: {features_dir}. Run extract_features.py first.")

    data_cfg = cfg["data"]
    output_cfg = cfg["output"]
    preprocessor_path = output_cfg.get("preprocessor_path")
    if preprocessor_path is None:
        preprocessor_path = out_dir / "preprocessor.joblib"
    else:
        preprocessor_path = Path(preprocessor_path)
    _, _, _, _, X_test, y_test, _ = load_merged_splits(
        features_dir=features_dir,
        stage2_csv=Path(data_cfg["stage2_csv"]),
        stage1_train_csv=Path(data_cfg["stage1_train_csv"]),
        stage1_val_csv=Path(data_cfg["stage1_val_csv"]),
        stage1_test_csv=Path(data_cfg["stage1_test_csv"]),
        preprocessor_path=preprocessor_path,
        label_col=data_cfg.get("label_col", "y"),
    )

    model_name = cfg["model"].get("name", "xgboost")
    ckpt = Path(args.ckpt)

    if model_name == "xgboost":
        booster = xgb.Booster()
        booster.load_model(str(ckpt))
        y_prob = booster.predict(xgb.DMatrix(X_test))
    elif model_name == "random_forest":
        model = load(ckpt)
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        raise ValueError(f"Unknown model: {model_name}")

    metrics = evaluate_probs(y_test, y_prob)
    print(metrics)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2))

    y_pred = (y_prob >= 0.5).astype(int)
    import pandas as pd
    pd.DataFrame({"y_true": y_test, "y_prob": y_prob, "y_pred": y_pred}).to_csv(
        out_dir / "test_predictions.csv", index=False
    )


if __name__ == "__main__":
    main()
