"""
Stage 4 eval: evaluate combined (image + metadata) binary classifier on test set.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
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

from Stage2.train import evaluate_multiclass
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


def _to_json_safe(obj):
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_json_safe(x) for x in obj]
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


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
    label_col = data_cfg.get("label_col", "y")
    num_classes = int(cfg.get("num_classes", 2))
    multiclass = label_col == "y_class" or num_classes > 2
    if multiclass and num_classes < 2:
        num_classes = 7

    _, _, _, _, X_test, y_test, _ = load_merged_splits(
        features_dir=features_dir,
        stage2_csv=Path(data_cfg["stage2_csv"]),
        stage1_train_csv=Path(data_cfg["stage1_train_csv"]),
        stage1_val_csv=Path(data_cfg["stage1_val_csv"]),
        stage1_test_csv=Path(data_cfg["stage1_test_csv"]),
        preprocessor_path=preprocessor_path,
        label_col=label_col,
    )

    model_name = cfg["model"].get("name", "xgboost")
    ckpt = Path(args.ckpt)

    if model_name == "xgboost":
        booster = xgb.Booster()
        booster.load_model(str(ckpt))
        y_prob = booster.predict(xgb.DMatrix(X_test))
        if not multiclass and isinstance(y_prob, np.ndarray) and y_prob.ndim == 2:
            y_prob = y_prob[:, 1]
    elif model_name == "random_forest":
        model = load(ckpt)
        y_prob = model.predict_proba(X_test)
        if not multiclass:
            y_prob = y_prob[:, 1]
    else:
        raise ValueError(f"Unknown model: {model_name}")

    if multiclass:
        class_names = None
        label_mapping_path = data_cfg.get("label_mapping") or str(out_dir / "label_mapping_multiclass.json")
        if label_mapping_path and Path(label_mapping_path).exists():
            with open(Path(label_mapping_path)) as f:
                label_mapping = json.load(f)
            class_names = label_mapping.get("class_names", list(range(num_classes)))
        if class_names is None:
            class_names = [str(i) for i in range(num_classes)]
        metrics_save = _to_json_safe(evaluate_multiclass(y_test, y_prob, num_classes, class_names))
        y_pred = y_prob.argmax(axis=1)
        pred_names = [class_names[int(p)] for p in y_pred]
        true_names = [class_names[int(t)] for t in y_test]
        preds_df = pd.DataFrame({
            "y_true": y_test,
            "y_pred": y_pred,
            "true_class": true_names,
            "predicted_class": pred_names,
        })
        for c, name in enumerate(class_names):
            preds_df[f"prob_{name}"] = y_prob[:, c]
    else:
        metrics_save = evaluate_probs(y_test, y_prob)
        preds_df = pd.DataFrame({
            "y_true": y_test,
            "y_prob": y_prob,
            "y_pred": (y_prob >= 0.5).astype(int),
        })

    print(metrics_save)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "test_metrics.json").write_text(json.dumps(metrics_save, indent=2))
    preds_df.to_csv(out_dir / "test_predictions.csv", index=False)


if __name__ == "__main__":
    main()
