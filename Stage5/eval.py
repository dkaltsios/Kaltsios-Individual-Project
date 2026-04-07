"""
Stage 5 eval: compare unsupervised clusters with known class labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from joblib import load
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, v_measure_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Stage4.data.merge import load_merged_splits


def purity_score(y_true: np.ndarray, y_cluster: np.ndarray) -> float:
    total = len(y_true)
    if total == 0:
        return 0.0
    clusters = np.unique(y_cluster)
    purity_sum = 0
    for c in clusters:
        idx = np.where(y_cluster == c)[0]
        if len(idx) == 0:
            continue
        labels, counts = np.unique(y_true[idx], return_counts=True)
        _ = labels
        purity_sum += int(counts.max())
    return float(purity_sum / total)


def best_map_accuracy(y_true: np.ndarray, y_cluster: np.ndarray, n_classes: int, n_clusters: int):
    conf = np.zeros((n_clusters, n_classes), dtype=np.int64)
    for c, y in zip(y_cluster, y_true):
        conf[int(c), int(y)] += 1
    row_ind, col_ind = linear_sum_assignment(-conf)
    mapping = {int(r): int(c) for r, c in zip(row_ind, col_ind)}
    y_mapped = np.array([mapping.get(int(c), -1) for c in y_cluster], dtype=int)
    acc = float((y_mapped == y_true).mean())
    return acc, mapping, conf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Stage 5 YAML config")
    parser.add_argument("--ckpt", required=True, help="Path to kmeans.joblib")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="Split to evaluate")
    args = parser.parse_args()

    import yaml

    cfg = yaml.safe_load(Path(args.config).read_text())
    out_dir = Path(cfg["output"]["dir"])
    data_cfg = cfg["data"]
    drop_contains = tuple(data_cfg.get("drop_columns_contains", []))

    preprocessor_path = cfg["output"].get("preprocessor_path")
    if preprocessor_path is None:
        preprocessor_path = out_dir / "preprocessor.joblib"
    else:
        preprocessor_path = Path(preprocessor_path)

    X_train, y_train, X_val, y_val, X_test, y_test, _ = load_merged_splits(
        features_dir=out_dir / "features",
        stage2_csv=Path(data_cfg["stage2_csv"]),
        stage1_train_csv=Path(data_cfg["stage1_train_csv"]),
        stage1_val_csv=Path(data_cfg["stage1_val_csv"]),
        stage1_test_csv=Path(data_cfg["stage1_test_csv"]),
        preprocessor_path=preprocessor_path,
        label_col=data_cfg.get("label_col", "y_class"),
        id_cols=("sample_id", "dataset_id", "patient_global", "y", "y_class"),
        image_feature_indices=None,
        drop_columns_contains=drop_contains,
    )

    if args.split == "train":
        X_eval, y_eval = X_train, y_train
    elif args.split == "val":
        X_eval, y_eval = X_val, y_val
    else:
        X_eval, y_eval = X_test, y_test

    scaler = load(out_dir / "scaler.joblib")
    kmeans = load(Path(args.ckpt))
    X_eval_sc = scaler.transform(X_eval)
    clusters = kmeans.predict(X_eval_sc)

    num_classes = int(cfg.get("num_classes", 7))
    n_clusters = int(kmeans.n_clusters)
    mapped_acc, mapping, conf = best_map_accuracy(y_eval, clusters, num_classes, n_clusters)
    metrics = {
        "split": args.split,
        "n_samples": int(len(y_eval)),
        "n_classes": num_classes,
        "n_clusters": n_clusters,
        "ari": float(adjusted_rand_score(y_eval, clusters)),
        "nmi": float(normalized_mutual_info_score(y_eval, clusters)),
        "v_measure": float(v_measure_score(y_eval, clusters)),
        "purity": purity_score(y_eval, clusters),
        "mapped_accuracy": mapped_acc,
        "cluster_to_class_mapping": {str(k): int(v) for k, v in mapping.items()},
        "confusion_cluster_x_class": conf.tolist(),
    }

    class_names = None
    label_mapping_path = data_cfg.get("label_mapping")
    if label_mapping_path and Path(label_mapping_path).exists():
        class_names = json.loads(Path(label_mapping_path).read_text()).get("class_names")

    mapped_pred = np.array([mapping.get(int(c), -1) for c in clusters], dtype=int)
    preds = {
        "y_true": y_eval,
        "cluster_id": clusters,
        "mapped_y_pred": mapped_pred,
    }
    preds_df = pd.DataFrame(preds)
    if class_names is not None:
        preds_df["true_class"] = [class_names[int(v)] for v in y_eval]
        preds_df["mapped_pred_class"] = [
            class_names[int(v)] if int(v) >= 0 and int(v) < len(class_names) else "unmapped"
            for v in mapped_pred
        ]

    (out_dir / f"{args.split}_metrics.json").write_text(json.dumps(metrics, indent=2))
    preds_df.to_csv(out_dir / f"{args.split}_predictions.csv", index=False)
    print(metrics)


if __name__ == "__main__":
    main()
