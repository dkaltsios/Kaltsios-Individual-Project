from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
from joblib import load

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Stage2.data.preprocess import TabularPreprocessor


def evaluate_probs(y_true: np.ndarray, y_prob: np.ndarray):
    y_pred = (y_prob >= 0.5).astype(int)
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

    stage2_df = pd.read_csv(stage2_csv)
    label_col = "y" if "y" in stage2_df.columns else "is_malignant"

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
    elif model_name == "random_forest":
        model = load(args.ckpt)
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    y_true, y_prob, metrics = evaluate_probs(y_test, y_prob)
    print(metrics)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2))

    preds_df = pd.DataFrame({"y_true": y_true, "y_prob": y_prob})
    preds_df.to_csv(out_dir / "test_predictions.csv", index=False)


if __name__ == "__main__":
    main()
