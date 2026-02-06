from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
from joblib import dump
from sklearn.ensemble import RandomForestClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Stage2.data.preprocess import TabularPreprocessor


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def compute_scale_pos_weight(labels: np.ndarray) -> float:
    pos = float((labels == 1).sum())
    neg = float((labels == 0).sum())
    if pos == 0:
        return 1.0
    return neg / pos


def evaluate_probs(y_true: np.ndarray, y_prob: np.ndarray):
    y_pred = (y_prob >= 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return {"acc": acc, "auc": auc, "precision": precision, "recall": recall, "f1": f1}


def split_stage2_by_stage1(stage2_df, stage1_train_csv, stage1_val_csv):
    train_ids = set(pd.read_csv(stage1_train_csv)["sample_id"].astype(str))
    val_ids = set(pd.read_csv(stage1_val_csv)["sample_id"].astype(str))

    stage2_df = stage2_df.copy()
    stage2_df["sample_id"] = stage2_df["sample_id"].astype(str)

    train_df = stage2_df[stage2_df["sample_id"].isin(train_ids)].reset_index(drop=True)
    val_df = stage2_df[stage2_df["sample_id"].isin(val_ids)].reset_index(drop=True)
    return train_df, val_df


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

    data_cfg = cfg["data"]
    stage2_csv = Path(data_cfg["stage2_csv"])
    stage1_train_csv = Path(data_cfg["stage1_train_csv"])
    stage1_val_csv = Path(data_cfg["stage1_val_csv"])

    stage2_df = pd.read_csv(stage2_csv)
    label_col = "y" if "y" in stage2_df.columns else "is_malignant"
    id_cols = ["sample_id", "dataset_id", "patient_global"]

    train_df, val_df = split_stage2_by_stage1(stage2_df, stage1_train_csv, stage1_val_csv)

    preprocessor = TabularPreprocessor()
    preprocessor.fit(train_df, label_col=label_col, id_cols=id_cols)

    X_train = preprocessor.transform(train_df)
    y_train = train_df[label_col].astype(int).to_numpy()
    X_val = preprocessor.transform(val_df)
    y_val = val_df[label_col].astype(int).to_numpy()

    model_cfg = cfg["model"]
    model_name = model_cfg.get("name", "xgboost")

    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    preprocessor_path = Path(cfg["output"].get("preprocessor_path", out_dir / "preprocessor.joblib"))
    preprocessor.save(preprocessor_path)

    if model_name == "xgboost":
        scale_pos_weight = compute_scale_pos_weight(y_train)
        model = xgb.XGBClassifier(
            n_estimators=int(model_cfg.get("n_estimators", 500)),
            learning_rate=float(model_cfg.get("learning_rate", 0.05)),
            max_depth=int(model_cfg.get("max_depth", 5)),
            min_child_weight=float(model_cfg.get("min_child_weight", 1)),
            subsample=float(model_cfg.get("subsample", 0.8)),
            colsample_bytree=float(model_cfg.get("colsample_bytree", 0.8)),
            gamma=float(model_cfg.get("gamma", 0.0)),
            reg_alpha=float(model_cfg.get("reg_alpha", 0.0)),
            reg_lambda=float(model_cfg.get("reg_lambda", 1.0)),
            objective="binary:logistic",
            eval_metric="auc",
            scale_pos_weight=scale_pos_weight,
            random_state=seed,
            tree_method="hist",
        )

        eval_set = [(X_val, y_val)]
        early_stopping_rounds = model_cfg.get("early_stopping_rounds", 50)

        model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=True,
            early_stopping_rounds=int(early_stopping_rounds) if early_stopping_rounds else None,
        )

        y_val_prob = model.predict_proba(X_val)[:, 1]
        val_metrics = evaluate_probs(y_val, y_val_prob)
        print(
            f"val_acc={val_metrics['acc']:.4f} "
            f"val_auc={val_metrics['auc']:.4f}"
        )

        booster = model.get_booster()
        booster.save_model(str(out_dir / "best.json"))
        history = {
            "best_iteration": int(booster.best_iteration)
            if booster.best_iteration is not None
            else None,
            "evals_result": model.evals_result(),
            "val_metrics": val_metrics,
        }
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))
        return

    if model_name == "random_forest":
        model = RandomForestClassifier(
            n_estimators=int(model_cfg.get("n_estimators", 500)),
            max_depth=model_cfg.get("max_depth", None),
            min_samples_split=int(model_cfg.get("min_samples_split", 2)),
            min_samples_leaf=int(model_cfg.get("min_samples_leaf", 1)),
            max_features=model_cfg.get("max_features", "sqrt"),
            class_weight=model_cfg.get("class_weight", "balanced"),
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        y_val_prob = model.predict_proba(X_val)[:, 1]
        val_metrics = evaluate_probs(y_val, y_val_prob)
        print(
            f"val_acc={val_metrics['acc']:.4f} "
            f"val_auc={val_metrics['auc']:.4f}"
        )
        dump(model, out_dir / "best.joblib")
        history = {"val_metrics": val_metrics}
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))
        return

    raise ValueError(f"Unknown model name: {model_name}")


if __name__ == "__main__":
    main()
