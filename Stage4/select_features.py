"""
Pareto feature selection for Stage 4.

Ranks all extracted image features by mutual information with the target
labels (fitted on the train split only) and keeps the top ``keep_percent``
(default 20 %, Pareto principle).

Outputs ``feature_selection.json`` in the output directory, which is
consumed by train.py and eval.py to subset image feature columns.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Stage 4 YAML config")
    args = parser.parse_args()

    import yaml

    cfg = yaml.safe_load(Path(args.config).read_text())

    seed = int(cfg.get("seed", 42))
    img_size = int(cfg.get("img_size", 128))
    keep_percent = float(cfg.get("feature_selection", {}).get("keep_percent", 20))

    out_dir = Path(cfg["output"]["dir"])
    features_dir = out_dir / "features"
    if not features_dir.exists():
        raise FileNotFoundError(
            f"Features not found at {features_dir}. Run extract_features.py first."
        )

    data_cfg = cfg["data"]
    label_col = data_cfg.get("label_col", "y")

    train_npz = np.load(features_dir / "train.npz", allow_pickle=True)
    sample_ids = [str(s) for s in train_npz["sample_ids"]]
    features = train_npz["features"]
    n_total = features.shape[1]

    stage2 = pd.read_csv(data_cfg["stage2_csv"])
    stage2["sample_id"] = stage2["sample_id"].astype(str)
    labels = (
        stage2.set_index("sample_id")
        .loc[sample_ids][label_col]
        .astype(int)
        .values
    )

    print(
        f"Computing mutual information for {n_total} features "
        f"on {len(labels)} train samples ..."
    )
    mi_scores = mutual_info_classif(
        features, labels, discrete_features=False, random_state=seed
    )

    n_keep = max(1, int(np.ceil(n_total * keep_percent / 100.0)))
    ranked_indices = np.argsort(mi_scores)[::-1]
    selected_indices = sorted(ranked_indices[:n_keep].tolist())

    feature_names = None
    extractor_name = cfg.get("feature_extractor", "13feat")
    if extractor_name == "pareto":
        from Stage4.data.image_features_pareto import get_feature_names

        feature_names = get_feature_names(img_size)

    selected_names = (
        [feature_names[i] for i in selected_indices]
        if feature_names
        else selected_indices
    )

    selection = {
        "n_total": n_total,
        "n_selected": n_keep,
        "keep_percent": keep_percent,
        "selected_indices": selected_indices,
        "selected_names": selected_names,
        "mi_scores": [float(s) for s in mi_scores],
        "ranking": ranked_indices.tolist(),
    }

    out_path = out_dir / "feature_selection.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(selection, indent=2))

    print(f"Kept {n_keep}/{n_total} features ({keep_percent}%)")
    print(f"Top 10 features by MI:")
    for rank, idx in enumerate(ranked_indices[:10]):
        name = feature_names[idx] if feature_names else f"feature_{idx}"
        print(f"  {rank + 1:3d}. {name:30s}  MI={mi_scores[idx]:.6f}")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
