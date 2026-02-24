"""
Load and merge image features with metadata for binary malignant/benign.
When preprocessor_path is None, fits TabularPreprocessor on train metadata (Stage 4 standalone).
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union

import numpy as np
import pandas as pd

from Stage2.data.preprocess import TabularPreprocessor


def load_merged_splits(
    features_dir: Path,
    stage2_csv: Path,
    stage1_train_csv: Path,
    stage1_val_csv: Path,
    stage1_test_csv: Path,
    preprocessor_path: Union[Path, None] = None,
    label_col: str = "y",
    id_cols: Tuple[str, ...] = ("sample_id", "dataset_id", "patient_global"),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, TabularPreprocessor]:
    """
    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test, preprocessor
    If preprocessor_path is None, preprocessor is fitted on train metadata and should be saved by the caller.
    """
    stage2 = pd.read_csv(stage2_csv)
    stage2["sample_id"] = stage2["sample_id"].astype(str)
    id_cols_list = list(id_cols)

    def get_sample_ids_and_features(split: str):
        if split == "train":
            stage1_csv = stage1_train_csv
        elif split == "val":
            stage1_csv = stage1_val_csv
        else:
            stage1_csv = stage1_test_csv
        npz = np.load(features_dir / f"{split}.npz", allow_pickle=True)
        sample_ids = npz["sample_ids"].tolist()
        sample_ids = [str(s) for s in sample_ids]
        img_feats = npz["features"]
        return sample_ids, img_feats

    train_ids, img_train = get_sample_ids_and_features("train")
    val_ids, img_val = get_sample_ids_and_features("val")
    test_ids, img_test = get_sample_ids_and_features("test")

    if preprocessor_path is not None and Path(preprocessor_path).exists():
        preprocessor = TabularPreprocessor.load(Path(preprocessor_path))
    else:
        # Fit on train metadata (Stage 4 standalone)
        meta_train = stage2.set_index("sample_id").loc[train_ids].reset_index()
        preprocessor = TabularPreprocessor()
        preprocessor.fit(meta_train, label_col=label_col, id_cols=id_cols_list)

    def build_xy(sample_ids, img_feats):
        meta_df = stage2.set_index("sample_id").loc[sample_ids].reset_index()
        X_meta = preprocessor.transform(meta_df)
        y = meta_df[label_col].astype(int).to_numpy()
        X = np.concatenate([img_feats, X_meta], axis=1).astype(np.float32)
        return X, y

    X_train, y_train = build_xy(train_ids, img_train)
    X_val, y_val = build_xy(val_ids, img_val)
    X_test, y_test = build_xy(test_ids, img_test)

    return X_train, y_train, X_val, y_val, X_test, y_test, preprocessor
