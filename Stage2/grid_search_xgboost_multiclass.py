"""
Grid search for Stage 2 XGBoost (multiclass) using cross-validation.

For each hyperparameter combination defined in xgboost_multiclass_grid.yaml:
- Runs StratifiedGroupKFold CV on train+val data (grouped by patient_global)
- Fits a TabularPreprocessor on each fold's train split
- Trains an XGBClassifier
- Evaluates multiclass metrics on the fold's validation split

Writes results to Stage2/cv_results/xgboost_multiclass_grid.json
with per-parameter-set mean/std metrics across folds and the best setting
according to the chosen metric (default: f1_macro).
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Stage2.data.preprocess import TabularPreprocessor
from Stage2.train import evaluate_multiclass


def run_cv_for_params(
    df_cv: pd.DataFrame,
    label_col: str,
    num_classes: int,
    class_names: list[str],
    xgb_base_cfg: dict,
    params: dict,
    n_splits: int,
    seed: int,
):
    """Run StratifiedGroupKFold CV for a specific XGBoost hyperparameter combination."""
    y = df_cv[label_col].astype(int).to_numpy()
    groups = df_cv["patient_global"].astype(str).to_numpy()

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    fold_metrics: list[dict] = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(df_cv, y, groups)):
        df_train = df_cv.iloc[train_idx].reset_index(drop=True)
        df_val = df_cv.iloc[val_idx].reset_index(drop=True)

        id_cols = ["sample_id", "dataset_id", "patient_global"]
        if "subtype" in df_train.columns and "subtype" not in id_cols:
            id_cols.append("subtype")

        preprocessor = TabularPreprocessor()
        preprocessor.fit(df_train, label_col=label_col, id_cols=id_cols)

        X_train = preprocessor.transform(df_train)
        y_train = df_train[label_col].astype(int).to_numpy()

        X_val = preprocessor.transform(df_val)
        y_val = df_val[label_col].astype(int).to_numpy()

        xgb_kwargs = {
            "n_estimators": int(xgb_base_cfg.get("n_estimators", 500)),
            "learning_rate": float(params.get("learning_rate", xgb_base_cfg.get("learning_rate", 0.05))),
            "max_depth": int(params.get("max_depth", xgb_base_cfg.get("max_depth", 5))),
            "min_child_weight": float(params.get("min_child_weight", xgb_base_cfg.get("min_child_weight", 1))),
            "subsample": float(params.get("subsample", xgb_base_cfg.get("subsample", 0.8))),
            "colsample_bytree": float(params.get("colsample_bytree", xgb_base_cfg.get("colsample_bytree", 0.8))),
            "gamma": float(xgb_base_cfg.get("gamma", 0.0)),
            "reg_alpha": float(xgb_base_cfg.get("reg_alpha", 0.0)),
            "reg_lambda": float(xgb_base_cfg.get("reg_lambda", 1.0)),
            "objective": "multi:softprob",
            "num_class": num_classes,
            "eval_metric": "mlogloss",
            "random_state": seed,
            "tree_method": "hist",
        }

        model = xgb.XGBClassifier(**xgb_kwargs)
        model.fit(X_train, y_train)
        y_val_prob = model.predict_proba(X_val)

        metrics = evaluate_multiclass(y_val, y_val_prob, num_classes=num_classes, class_names=class_names)
        fold_metrics.append({"fold": fold_idx, "metrics": metrics})

    keys = ["acc", "precision_macro", "recall_macro", "f1_macro", "roc_auc_macro", "pr_auc_macro"]
    mean_std = {}
    for key in keys:
        vals = [fm["metrics"][key] for fm in fold_metrics]
        mean_std[key] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
        }

    return {
        "params": params,
        "metrics_mean_std": mean_std,
        "folds": fold_metrics,
    }


def main():
    import argparse
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="Stage2/configs/xgboost_multiclass_grid.yaml",
        help="Path to XGBoost multiclass grid YAML config",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())

    seed = int(cfg.get("seed", 42))
    num_classes = int(cfg.get("num_classes", 7))

    data_cfg = cfg["data"]
    stage2_path = Path(data_cfg["stage2_csv"])
    stage1_train_path = Path(data_cfg["stage1_train_csv"])
    stage1_val_path = Path(data_cfg["stage1_val_csv"])
    stage1_test_path = Path(data_cfg["stage1_test_csv"])

    df_stage2 = pd.read_csv(stage2_path)
    df_stage2["sample_id"] = df_stage2["sample_id"].astype(str)

    label_col = "y_class"
    if label_col not in df_stage2.columns:
        raise ValueError(f"Expected '{label_col}' in {stage2_path}")

    train_ids = set(pd.read_csv(stage1_train_path)["sample_id"].astype(str))
    val_ids = set(pd.read_csv(stage1_val_path)["sample_id"].astype(str))
    test_ids = set(pd.read_csv(stage1_test_path)["sample_id"].astype(str))

    allowed_ids = train_ids | val_ids
    df_cv = df_stage2[df_stage2["sample_id"].isin(allowed_ids)].copy().reset_index(drop=True)

    if "patient_global" not in df_cv.columns:
        raise ValueError("Expected 'patient_global' column for grouped CV.")

    label_mapping_path = data_cfg.get("label_mapping")
    if label_mapping_path and Path(label_mapping_path).exists():
        label_mapping = json.loads(Path(label_mapping_path).read_text())
        class_names = label_mapping.get("class_names", [str(i) for i in range(num_classes)])
    else:
        class_names = [str(i) for i in range(num_classes)]

    cv_cfg = cfg.get("cv", {})
    n_splits = int(cv_cfg.get("n_splits", 5))
    selection_metric = cv_cfg.get("metric", "f1_macro")

    xgb_cfg = cfg["model"]
    xgb_base_cfg = xgb_cfg.get("base", {})
    grid_cfg = xgb_cfg.get("param_grid", {})

    grid_keys = sorted(grid_cfg.keys())
    grid_values = [grid_cfg[k] for k in grid_keys]
    param_combinations = [
        dict(zip(grid_keys, values))
        for values in itertools.product(*grid_values)
    ]

    print(f"Running XGBoost grid search over {len(param_combinations)} combinations...")

    all_results = []
    for i, params in enumerate(param_combinations):
        print(f"\n[{i+1}/{len(param_combinations)}] Params: {params}")
        res = run_cv_for_params(
            df_cv=df_cv,
            label_col=label_col,
            num_classes=num_classes,
            class_names=class_names,
            xgb_base_cfg=xgb_base_cfg,
            params=params,
            n_splits=n_splits,
            seed=seed,
        )
        all_results.append(res)
        ms = res["metrics_mean_std"]
        print(
            f"  mean {selection_metric}={ms[selection_metric]['mean']:.4f} "
            f"(acc={ms['acc']['mean']:.4f}, pr_auc_macro={ms['pr_auc_macro']['mean']:.4f})"
        )

    best = max(
        all_results,
        key=lambda r: r["metrics_mean_std"][selection_metric]["mean"],
    )

    summary = {
        "config_path": str(Path(args.config)),
        "selection_metric": selection_metric,
        "results": all_results,
        "best": best,
        "test_ids_count": len(test_ids),
    }

    out_dir = PROJECT_ROOT / "Stage2" / "cv_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "xgboost_multiclass_grid.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved XGBoost grid-search results to {out_path}")
    print(
        f"Best params: {best['params']} "
        f"with mean {selection_metric}="
        f"{best['metrics_mean_std'][selection_metric]['mean']:.4f}"
    )


if __name__ == "__main__":
    main()

