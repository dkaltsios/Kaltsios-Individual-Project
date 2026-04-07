"""
Stage 4 train: [image features + metadata] -> binary (malignant/benign) or multiclass (7 lesion types).
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys

import numpy as np
import xgboost as xgb
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
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

from Stage2.train import compute_scale_pos_weight, evaluate_probs, evaluate_multiclass
from Stage4.data.merge import load_merged_splits


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


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
    parser.add_argument("--config", required=True, help="Path to Stage 4 YAML config")
    args = parser.parse_args()

    import yaml
    cfg = yaml.safe_load(Path(args.config).read_text())

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    features_dir = out_dir / "features"
    if not features_dir.exists():
        raise FileNotFoundError(
            f"Run extract_features.py first to create {features_dir}. "
            "Example: python3 Stage4/extract_features.py --config Stage4/configs/stage4.yaml"
        )

    data_cfg = cfg["data"]
    output_cfg = cfg["output"]
    preprocessor_path = output_cfg.get("preprocessor_path")
    if preprocessor_path is not None:
        preprocessor_path = Path(preprocessor_path)
    label_col = data_cfg.get("label_col", "y")
    num_classes = int(cfg.get("num_classes", 2))
    multiclass = label_col == "y_class" or num_classes > 2
    if multiclass and num_classes < 2:
        num_classes = 7

    feature_sel_path = out_dir / "feature_selection.json"
    image_feature_indices = None
    if feature_sel_path.exists():
        sel = json.loads(feature_sel_path.read_text())
        image_feature_indices = sel["selected_indices"]
        print(
            f"Feature selection active: using {sel['n_selected']}/{sel['n_total']} "
            f"image features ({sel['keep_percent']}%)"
        )

    X_train, y_train, X_val, y_val, X_test, y_test, preprocessor = load_merged_splits(
        features_dir=features_dir,
        stage2_csv=Path(data_cfg["stage2_csv"]),
        stage1_train_csv=Path(data_cfg["stage1_train_csv"]),
        stage1_val_csv=Path(data_cfg["stage1_val_csv"]),
        stage1_test_csv=Path(data_cfg["stage1_test_csv"]),
        preprocessor_path=preprocessor_path,
        label_col=label_col,
        image_feature_indices=image_feature_indices,
    )
    if preprocessor_path is None or not Path(preprocessor_path).exists():
        preprocessor.save(out_dir / "preprocessor.joblib")

    class_names = None
    if multiclass:
        label_mapping_path = data_cfg.get("label_mapping")
        if label_mapping_path and Path(label_mapping_path).exists():
            with open(Path(label_mapping_path)) as f:
                label_mapping = json.load(f)
            class_names = label_mapping.get("class_names", list(range(num_classes)))
            (out_dir / "label_mapping_multiclass.json").write_text(json.dumps(label_mapping, indent=2))
        else:
            class_names = [str(i) for i in range(num_classes)]

    model_cfg = cfg["model"]
    model_name = model_cfg.get("name", "xgboost")
    booster = None

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
        early_stopping = model_cfg.get("early_stopping_rounds", 50)
        model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=True,
            early_stopping_rounds=int(early_stopping) if early_stopping else None,
        )
        booster = model.get_booster()
        booster.save_model(str(out_dir / "best.json"))
        y_val_prob = model.predict_proba(X_val)
        if not multiclass:
            y_val_prob = y_val_prob[:, 1]
    elif model_name == "softmax_regression":
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_val_sc = scaler.transform(X_val)
        dump(scaler, out_dir / "scaler.joblib")
        model = LogisticRegression(
            solver=model_cfg.get("solver", "lbfgs"),
            C=float(model_cfg.get("C", 1.0)),
            penalty=model_cfg.get("penalty", "l2"),
            class_weight=model_cfg.get("class_weight", "balanced"),
            max_iter=int(model_cfg.get("max_iter", 1000)),
            random_state=seed,
        )
        model.fit(X_train_sc, y_train)
        dump(model, out_dir / "best.joblib")
        y_val_prob = model.predict_proba(X_val_sc)
        if not multiclass:
            y_val_prob = y_val_prob[:, 1]
    elif model_name == "naive_bayes":
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_val_sc = scaler.transform(X_val)
        dump(scaler, out_dir / "scaler.joblib")
        model = GaussianNB(
            var_smoothing=float(model_cfg.get("var_smoothing", 1e-9)),
        )
        model.fit(X_train_sc, y_train)
        dump(model, out_dir / "best.joblib")
        y_val_prob = model.predict_proba(X_val_sc)
        if not multiclass:
            y_val_prob = y_val_prob[:, 1]
    else:
        raise ValueError(f"Unknown model: {model_name}")

    if multiclass:
        val_metrics = evaluate_multiclass(y_val, y_val_prob, num_classes, class_names)
        print(
            f"val_acc={val_metrics['acc']:.4f} val_f1_macro={val_metrics['f1_macro']:.4f} "
            f"val_pr_auc_macro={val_metrics['pr_auc_macro']:.4f}"
        )
    else:
        val_metrics = evaluate_probs(y_val, y_val_prob)
        print(
            f"val_acc={val_metrics['acc']:.4f} val_auc={val_metrics['auc']:.4f} "
            f"val_pr_auc={val_metrics['pr_auc']:.4f}"
        )

    history = {"val_metrics": val_metrics}
    if model_name == "xgboost" and booster is not None:
        history["best_iteration"] = int(booster.best_iteration) if booster.best_iteration is not None else None
        history["evals_result"] = model.evals_result()
    (out_dir / "history.json").write_text(json.dumps(_to_json_safe(history), indent=2))


if __name__ == "__main__":
    main()
