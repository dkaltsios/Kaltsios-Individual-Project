"""
Cross-validation for Stage 4 (multiclass).

Uses StratifiedGroupKFold on train+val samples (grouped by patient_global),
keeping the existing test set untouched.

For each fold:
- Loads pre-extracted image features from the features_dir
- Fits a TabularPreprocessor on train metadata
- Optionally applies Pareto feature selection indices
- Trains the specified model (xgboost, softmax_regression, naive_bayes)
- Evaluates multiclass metrics on the fold's validation split

Writes a JSON summary to Stage4/cv_results/<config_stem>_cv.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Stage2.data.preprocess import TabularPreprocessor
from Stage2.train import evaluate_multiclass


def _load_image_features(features_dir: Path, split_ids: list[str], image_feature_indices=None):
    all_ids, all_feats = [], []
    for split in ["train", "val"]:
        npz = np.load(features_dir / f"{split}.npz", allow_pickle=True)
        ids = [str(s) for s in npz["sample_ids"]]
        feats = npz["features"]
        if image_feature_indices is not None:
            feats = feats[:, image_feature_indices]
        all_ids.extend(ids)
        all_feats.append(feats)
    all_feats = np.concatenate(all_feats, axis=0)
    id_to_idx = {sid: i for i, sid in enumerate(all_ids)}
    indices = [id_to_idx[sid] for sid in split_ids]
    return all_feats[indices]


def _build_model(model_name, model_cfg, num_classes, seed):
    if model_name == "xgboost":
        return xgb.XGBClassifier(
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
        ), False
    elif model_name == "softmax_regression":
        return LogisticRegression(
            solver=model_cfg.get("solver", "lbfgs"),
            C=float(model_cfg.get("C", 1.0)),
            penalty=model_cfg.get("penalty", "l2"),
            class_weight=model_cfg.get("class_weight", "balanced"),
            max_iter=int(model_cfg.get("max_iter", 1000)),
            random_state=seed,
        ), True
    elif model_name == "naive_bayes":
        return GaussianNB(
            var_smoothing=float(model_cfg.get("var_smoothing", 1e-9)),
        ), True
    else:
        raise ValueError(f"Unknown model: {model_name}")


def main():
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Stage 4 YAML config")
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    config_stem = Path(args.config).stem

    seed = int(cfg.get("seed", 42))
    n_splits = args.folds
    num_classes = int(cfg.get("num_classes", 7))
    label_col = cfg["data"].get("label_col", "y_class")

    out_dir = Path(cfg["output"]["dir"])
    features_dir = out_dir / "features"
    if not features_dir.exists():
        raise FileNotFoundError(
            f"Features not found at {features_dir}. Run extract_features.py first."
        )

    feature_sel_path = out_dir / "feature_selection.json"
    image_feature_indices = None
    if feature_sel_path.exists():
        sel = json.loads(feature_sel_path.read_text())
        image_feature_indices = sel["selected_indices"]
        print(f"Feature selection active: {sel['n_selected']}/{sel['n_total']} features")

    data_cfg = cfg["data"]
    stage2 = pd.read_csv(data_cfg["stage2_csv"])
    stage2["sample_id"] = stage2["sample_id"].astype(str)

    train_ids = set(pd.read_csv(data_cfg["stage1_train_csv"])["sample_id"].astype(str))
    val_ids = set(pd.read_csv(data_cfg["stage1_val_csv"])["sample_id"].astype(str))
    allowed_ids = train_ids | val_ids

    df_cv = stage2[stage2["sample_id"].isin(allowed_ids)].copy().reset_index(drop=True)

    label_mapping_path = data_cfg.get("label_mapping")
    if label_mapping_path and Path(label_mapping_path).exists():
        label_mapping = json.loads(Path(label_mapping_path).read_text())
        class_names = label_mapping.get("class_names", [str(i) for i in range(num_classes)])
    else:
        class_names = [str(i) for i in range(num_classes)]

    y = df_cv[label_col].astype(int).to_numpy()
    groups = df_cv["patient_global"].astype(str).to_numpy()

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    model_cfg = cfg["model"]
    model_name = model_cfg.get("name", "xgboost")
    id_cols = ["sample_id", "dataset_id", "patient_global"]
    if "subtype" in df_cv.columns:
        id_cols.append("subtype")

    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(df_cv, y, groups)):
        print(f"\n=== Fold {fold_idx + 1}/{n_splits} ===")
        df_train = df_cv.iloc[train_idx].reset_index(drop=True)
        df_val = df_cv.iloc[val_idx].reset_index(drop=True)

        train_sids = df_train["sample_id"].tolist()
        val_sids = df_val["sample_id"].tolist()

        img_train = _load_image_features(features_dir, train_sids, image_feature_indices)
        img_val = _load_image_features(features_dir, val_sids, image_feature_indices)

        preprocessor = TabularPreprocessor()
        preprocessor.fit(df_train, label_col=label_col, id_cols=id_cols)

        X_meta_train = preprocessor.transform(df_train)
        X_meta_val = preprocessor.transform(df_val)

        X_train = np.concatenate([img_train, X_meta_train], axis=1).astype(np.float32)
        X_val = np.concatenate([img_val, X_meta_val], axis=1).astype(np.float32)
        y_train = df_train[label_col].astype(int).to_numpy()
        y_val = df_val[label_col].astype(int).to_numpy()

        model, needs_scaling = _build_model(model_name, model_cfg, num_classes, seed)

        if needs_scaling:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)

        model.fit(X_train, y_train)
        y_val_prob = model.predict_proba(X_val)

        metrics = evaluate_multiclass(y_val, y_val_prob, num_classes, class_names)
        fold_results.append({
            "fold": fold_idx,
            "n_train": len(df_train),
            "n_val": len(df_val),
            "metrics": metrics,
        })
        print(
            f"Fold {fold_idx}: acc={metrics['acc']:.4f}, "
            f"f1_macro={metrics['f1_macro']:.4f}, "
            f"pr_auc_macro={metrics['pr_auc_macro']:.4f}"
        )

    def mean_std(key):
        vals = [fr["metrics"][key] for fr in fold_results]
        return float(np.mean(vals)), float(np.std(vals))

    summary = {
        "config": {
            "config_path": str(Path(args.config)),
            "model_name": model_name,
            "feature_extractor": cfg.get("feature_extractor", "13feat"),
            "n_splits": n_splits,
            "seed": seed,
        },
        "metrics_mean_std": {
            key: {"mean": mean_std(key)[0], "std": mean_std(key)[1]}
            for key in ["acc", "precision_macro", "recall_macro", "f1_macro",
                        "roc_auc_macro", "pr_auc_macro"]
        },
        "folds": fold_results,
    }

    cv_dir = PROJECT_ROOT / "Stage4" / "cv_results"
    cv_dir.mkdir(parents=True, exist_ok=True)
    out_path = cv_dir / f"{config_stem}_cv.json"
    out_path.write_text(json.dumps(summary, indent=2))

    print(f"\nSaved CV summary to {out_path}")
    ms = summary["metrics_mean_std"]
    print(
        f"Mean: acc={ms['acc']['mean']:.4f}±{ms['acc']['std']:.4f}, "
        f"f1_macro={ms['f1_macro']['mean']:.4f}±{ms['f1_macro']['std']:.4f}, "
        f"pr_auc_macro={ms['pr_auc_macro']['mean']:.4f}±{ms['pr_auc_macro']['std']:.4f}"
    )


if __name__ == "__main__":
    main()
