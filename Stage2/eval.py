from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
)
import xgboost as xgb
from joblib import load

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Stage2.data.preprocess import TabularPreprocessor


def evaluate_probs(y_true: np.ndarray, y_prob: np.ndarray):
    """Binary: y_prob is 1-d."""
    y_pred = (y_prob >= 0.5).astype(int)
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


def evaluate_multiclass(y_true: np.ndarray, y_prob: np.ndarray, num_classes: int, class_names=None):
    """y_prob: (n, num_classes)."""
    y_pred = y_prob.argmax(axis=1)
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


def split_stage2_by_stage1(stage2_df, stage1_test_csv):
    test_ids = set(pd.read_csv(stage1_test_csv)["sample_id"].astype(str))

    stage2_df = stage2_df.copy()
    stage2_df["sample_id"] = stage2_df["sample_id"].astype(str)

    test_df = stage2_df[stage2_df["sample_id"].isin(test_ids)].reset_index(drop=True)
    return test_df


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

    data_cfg = cfg["data"]
    stage2_csv = Path(data_cfg["stage2_csv"])
    stage1_test_csv = Path(data_cfg["stage1_test_csv"])
    num_classes = int(cfg.get("num_classes", 7))
    label_mapping_path = data_cfg.get("label_mapping")

    stage2_df = pd.read_csv(stage2_csv)
    label_col = "y_class" if "y_class" in stage2_df.columns else ("y" if "y" in stage2_df.columns else "is_malignant")
    multiclass = label_col == "y_class"

    test_df = split_stage2_by_stage1(stage2_df, stage1_test_csv)

    out_dir = Path(cfg["output"]["dir"])
    preprocessor_path = Path(cfg["output"].get("preprocessor_path", out_dir / "preprocessor.joblib"))
    preprocessor = TabularPreprocessor.load(preprocessor_path)

    X_test = preprocessor.transform(test_df)
    y_test = test_df[label_col].astype(int).to_numpy()

    model_name = cfg["model"].get("name", "xgboost")

    if model_name == "xgboost":
        booster = xgb.Booster()
        booster.load_model(args.ckpt)
        dtest = xgb.DMatrix(X_test)
        y_prob = booster.predict(dtest)
        if multiclass and isinstance(y_prob, np.ndarray) and y_prob.ndim == 2:
            pass  # (n, num_class)
        elif not multiclass:
            y_prob = y_prob  # 1-d
    elif model_name == "random_forest":
        model = load(args.ckpt)
        y_prob = model.predict_proba(X_test)
        if not multiclass:
            y_prob = y_prob[:, 1]
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    if multiclass:
        class_names = None
        if label_mapping_path and Path(label_mapping_path).exists():
            with open(Path(label_mapping_path)) as f:
                label_mapping = json.load(f)
            class_names = label_mapping.get("class_names", list(range(num_classes)))
        if class_names is None:
            class_names = [str(i) for i in range(num_classes)]
        y_true, y_prob, y_pred, metrics = evaluate_multiclass(
            y_test, y_prob, num_classes, class_names
        )
        def _to_json_safe(obj):
            if isinstance(obj, dict):
                return {k: _to_json_safe(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_to_json_safe(x) for x in obj]
            if isinstance(obj, float) and np.isnan(obj):
                return None
            return obj
        metrics_save = _to_json_safe(metrics)
        pred_names = [class_names[int(p)] for p in y_pred]
        true_names = [class_names[int(t)] for t in y_true]
        preds_df = pd.DataFrame({
            "y_true": y_true,
            "y_pred": y_pred,
            "true_class": true_names,
            "predicted_class": pred_names,
        })
        for c, name in enumerate(class_names):
            preds_df[f"prob_{name}"] = y_prob[:, c]
    else:
        y_true, y_prob, y_pred, metrics = evaluate_probs(y_test, y_prob)
        metrics_save = metrics
        preds_df = pd.DataFrame({"y_true": y_true, "y_prob": y_prob, "y_pred": y_pred})

    print(metrics_save)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "test_metrics.json").write_text(json.dumps(metrics_save, indent=2))
    preds_df.to_csv(out_dir / "test_predictions.csv", index=False)


if __name__ == "__main__":
    main()
