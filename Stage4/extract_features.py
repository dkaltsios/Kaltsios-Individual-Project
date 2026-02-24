"""
Extract handcrafted image features (no CNN) for each sample.
Reads train/val/test CSVs, loads images, computes color + texture features, saves NPZ.
Stage 4 is standalone: no Stage 1 or Stage 2 required.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Stage4.data.image_features import extract_features_from_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Stage 4 config YAML")
    args = parser.parse_args()

    import yaml
    cfg = yaml.safe_load(Path(args.config).read_text())

    data_cfg = cfg["data"]
    data_root = Path(data_cfg.get("data_root", "."))
    img_size = int(cfg.get("img_size", 128))
    image_col = "image_path" if "image_path" in pd.read_csv(data_cfg["stage1_train_csv"]).columns else "img_path"

    out_dir = Path(cfg["output"]["dir"])
    features_dir = out_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        csv_key = f"stage1_{split}_csv"
        df = pd.read_csv(Path(data_cfg[csv_key]))
        sample_ids = df["sample_id"].astype(str).tolist()
        paths = df[image_col].tolist()

        feats_list = []
        for i, (sid, rel_path) in enumerate(zip(sample_ids, paths)):
            if i % 500 == 0 and i > 0:
                print(f"  {split}: {i}/{len(sample_ids)}")
            full_path = data_root / rel_path if not Path(rel_path).is_absolute() else Path(rel_path)
            feats = extract_features_from_path(full_path, img_size=img_size)
            feats_list.append(feats)

        features = np.stack(feats_list, axis=0).astype(np.float32)
        np.savez(
            features_dir / f"{split}.npz",
            sample_ids=np.array(sample_ids, dtype=object),
            features=features,
        )
        print(f"{split}: {len(sample_ids)} samples, features shape {features.shape}")

    print(f"Saved to {features_dir}")


if __name__ == "__main__":
    main()
