"""
Stage 5 train: unsupervised clustering on Stage4 merged features.
The target disease label is never used as an input feature.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys

import numpy as np
from joblib import dump
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Stage4.data.merge import load_merged_splits


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to Stage 5 YAML config")
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
            f"Run Stage4/extract_features.py first to create {features_dir}. "
            "Example: python3 Stage4/extract_features.py --config Stage5/configs/stage5_multiclass_kmeans.yaml"
        )

    data_cfg = cfg["data"]
    output_cfg = cfg["output"]
    model_cfg = cfg["model"]

    preprocessor_path = output_cfg.get("preprocessor_path")
    if preprocessor_path is not None:
        preprocessor_path = Path(preprocessor_path)

    label_col = data_cfg.get("label_col", "y_class")
    drop_contains = tuple(data_cfg.get("drop_columns_contains", []))

    load_out = load_merged_splits(
        features_dir=features_dir,
        stage2_csv=Path(data_cfg["stage2_csv"]),
        stage1_train_csv=Path(data_cfg["stage1_train_csv"]),
        stage1_val_csv=Path(data_cfg["stage1_val_csv"]),
        stage1_test_csv=Path(data_cfg["stage1_test_csv"]),
        preprocessor_path=preprocessor_path,
        label_col=label_col,
        id_cols=("sample_id", "dataset_id", "patient_global", "y", "y_class"),
        image_feature_indices=None,
        drop_columns_contains=drop_contains,
        return_feature_names=True,
    )
    X_train, y_train, X_val, y_val, X_test, y_test, preprocessor, feature_names = load_out
    _ = (y_train, y_val, y_test)  # labels are not used for fitting

    # Safety guard against target leakage into model inputs.
    explicit_blocklist = {
        label_col.lower(),
        "y",
        "y_class",
        "target",
        "label",
        "class_label",
        "disease_label",
    }
    leaked = []
    for name in feature_names:
        n = str(name).strip().lower()
        # Only block exact or suffix-style names (e.g. one-hot expansions like "y_class_3"),
        # avoiding false positives such as "mislabeled" in unrelated feature names.
        if n in explicit_blocklist or n.startswith(f"{label_col.lower()}_"):
            leaked.append(name)
    if leaked:
        leaked_preview = leaked[:20]
        raise ValueError(
            "Potential target leakage detected in Stage5 input features. "
            f"Found suspicious feature names: {leaked_preview}"
        )

    if preprocessor_path is None or not Path(preprocessor_path).exists():
        preprocessor.save(out_dir / "preprocessor.joblib")

    fit_on = model_cfg.get("fit_on", "train")
    if fit_on == "train":
        X_fit = X_train
    elif fit_on == "trainval":
        X_fit = np.concatenate([X_train, X_val], axis=0)
    elif fit_on == "all":
        X_fit = np.concatenate([X_train, X_val, X_test], axis=0)
    else:
        raise ValueError(f"Unknown fit_on: {fit_on}")

    scaler = StandardScaler()
    X_fit_sc = scaler.fit_transform(X_fit)
    dump(scaler, out_dir / "scaler.joblib")

    num_classes = int(cfg.get("num_classes", 7))
    n_clusters = int(model_cfg.get("n_clusters", num_classes))
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=seed,
        n_init=int(model_cfg.get("n_init", 20)),
        max_iter=int(model_cfg.get("max_iter", 300)),
    )
    kmeans.fit(X_fit_sc)
    dump(kmeans, out_dir / "kmeans.joblib")

    train_clusters = kmeans.predict(scaler.transform(X_train))
    val_clusters = kmeans.predict(scaler.transform(X_val))
    test_clusters = kmeans.predict(scaler.transform(X_test))

    history = {
        "seed": seed,
        "fit_on": fit_on,
        "n_clusters": n_clusters,
        "n_features": int(X_fit.shape[1]),
        "drop_columns_contains": list(drop_contains),
        "feature_names_preview": feature_names[:100],
        "train_cluster_counts": np.bincount(train_clusters, minlength=n_clusters).tolist(),
        "val_cluster_counts": np.bincount(val_clusters, minlength=n_clusters).tolist(),
        "test_cluster_counts": np.bincount(test_clusters, minlength=n_clusters).tolist(),
        "inertia": float(kmeans.inertia_),
    }
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))


if __name__ == "__main__":
    main()
