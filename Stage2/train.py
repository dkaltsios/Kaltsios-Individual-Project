from __future__ import annotations

import argparse
import json
import random
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
    """Binary: y_prob is 1-d probability of positive class."""
    y_pred = (y_prob >= 0.5).astype(int)
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
    return {"acc": acc, "auc": auc, "pr_auc": pr_auc, "precision": precision, "recall": recall, "f1": f1}


def evaluate_multiclass(y_true: np.ndarray, y_prob: np.ndarray, num_classes: int, class_names=None):
    """y_prob: (n, num_classes). Returns per-class and macro metrics."""
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
    return {
        "acc": acc,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "roc_auc_macro": roc_auc_macro,
        "pr_auc_macro": pr_auc_macro,
        "per_class": per_class,
    }


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
    label_col = "y_class" if "y_class" in stage2_df.columns else ("y" if "y" in stage2_df.columns else "is_malignant")
    multiclass = label_col == "y_class"
    num_classes = int(cfg.get("num_classes", 7)) if multiclass else 2
    id_cols = ["sample_id", "dataset_id", "patient_global"]
    if multiclass and "subtype" in stage2_df.columns:
        id_cols = list(id_cols) + ["subtype"]  # keep subtype out of features

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
        if multiclass:
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
                objective="multi:softprob",
                num_class=num_classes,
                eval_metric="mlogloss",
                random_state=seed,
                tree_method="hist",
            )
        else:
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

        if multiclass:
            y_val_prob = model.predict_proba(X_val)
            label_mapping_path = data_cfg.get("label_mapping")
            class_names = None
            if label_mapping_path and Path(label_mapping_path).exists():
                with open(Path(label_mapping_path)) as f:
                    label_mapping = json.load(f)
                class_names = label_mapping.get("class_names", list(range(num_classes)))
            val_metrics = evaluate_multiclass(y_val, y_val_prob, num_classes, class_names)
            print(
                f"val_acc={val_metrics['acc']:.4f} val_f1_macro={val_metrics['f1_macro']:.4f} "
                f"val_pr_auc_macro={val_metrics['pr_auc_macro']:.4f}"
            )
        else:
            y_val_prob = model.predict_proba(X_val)[:, 1]
            val_metrics = evaluate_probs(y_val, y_val_prob)
            print(
                f"val_acc={val_metrics['acc']:.4f} "
                f"val_auc={val_metrics['auc']:.4f} val_pr_auc={val_metrics['pr_auc']:.4f}"
            )

        booster = model.get_booster()
        booster.save_model(str(out_dir / "best.json"))
        history = {
            "best_iteration": int(booster.best_iteration) if booster.best_iteration is not None else None,
            "evals_result": model.evals_result(),
            "val_metrics": val_metrics,
        }
        def _to_json_safe(obj):
            if isinstance(obj, dict):
                return {k: _to_json_safe(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_to_json_safe(x) for x in obj]
            if isinstance(obj, float) and np.isnan(obj):
                return None
            return obj
        (out_dir / "history.json").write_text(json.dumps(_to_json_safe(history), indent=2))
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
        if multiclass:
            y_val_prob = model.predict_proba(X_val)
            label_mapping_path = data_cfg.get("label_mapping")
            class_names = None
            if label_mapping_path and Path(label_mapping_path).exists():
                with open(Path(label_mapping_path)) as f:
                    label_mapping = json.load(f)
                class_names = label_mapping.get("class_names", list(range(num_classes)))
            val_metrics = evaluate_multiclass(y_val, y_val_prob, num_classes, class_names)
            print(
                f"val_acc={val_metrics['acc']:.4f} val_f1_macro={val_metrics['f1_macro']:.4f} "
                f"val_pr_auc_macro={val_metrics['pr_auc_macro']:.4f}"
            )
        else:
            y_val_prob = model.predict_proba(X_val)[:, 1]
            val_metrics = evaluate_probs(y_val, y_val_prob)
            print(
                f"val_acc={val_metrics['acc']:.4f} "
                f"val_auc={val_metrics['auc']:.4f} val_pr_auc={val_metrics['pr_auc']:.4f}"
            )
        dump(model, out_dir / "best.joblib")
        def _to_json_safe(obj):
            if isinstance(obj, dict):
                return {k: _to_json_safe(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_to_json_safe(x) for x in obj]
            if isinstance(obj, float) and np.isnan(obj):
                return None
            return obj
        history = {"val_metrics": val_metrics}
        (out_dir / "history.json").write_text(json.dumps(_to_json_safe(history), indent=2))
        return

    raise ValueError(f"Unknown model name: {model_name}")


if __name__ == "__main__":
    main()
