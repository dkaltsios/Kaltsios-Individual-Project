from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class PreprocessorArtifacts:
    transformer: ColumnTransformer
    categorical_cols: list[str]
    numeric_cols: list[str]
    feature_names: list[str]


class TabularPreprocessor:
    def __init__(
        self,
        categorical_cols: Iterable[str] | None = None,
        numeric_cols: Iterable[str] | None = None,
    ):
        self._categorical_cols = list(categorical_cols) if categorical_cols else None
        self._numeric_cols = list(numeric_cols) if numeric_cols else None
        self.artifacts: PreprocessorArtifacts | None = None

    def _infer_columns(self, df: pd.DataFrame, label_col: str, id_cols: Iterable[str]):
        drop_cols = set(id_cols) | {label_col}
        candidates = [c for c in df.columns if c not in drop_cols]

        if self._categorical_cols is None or self._numeric_cols is None:
            cat_cols = [c for c in candidates if df[c].dtype == "object"]
            num_cols = [c for c in candidates if c not in cat_cols]
        else:
            cat_cols = self._categorical_cols
            num_cols = self._numeric_cols
        return cat_cols, num_cols

    def fit(self, df: pd.DataFrame, label_col: str, id_cols: Iterable[str]):
        cat_cols, num_cols = self._infer_columns(df, label_col, id_cols)

        # basic missing value handling
        df = df.copy()
        for c in cat_cols:
            df[c] = df[c].fillna("Unknown")
        for c in num_cols:
            df[c] = df[c].fillna(df[c].median())

        transformer = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
                ("num", StandardScaler(), num_cols),
            ],
            remainder="drop",
        )
        transformer.fit(df[cat_cols + num_cols])

        cat_feature_names = []
        if cat_cols:
            cat_feature_names = list(
                transformer.named_transformers_["cat"].get_feature_names_out(cat_cols)
            )
        feature_names = cat_feature_names + num_cols

        self.artifacts = PreprocessorArtifacts(
            transformer=transformer,
            categorical_cols=cat_cols,
            numeric_cols=num_cols,
            feature_names=feature_names,
        )
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self.artifacts is None:
            raise RuntimeError("Preprocessor must be fit before calling transform().")

        df = df.copy()
        for c in self.artifacts.categorical_cols:
            df[c] = df[c].fillna("Unknown")
        for c in self.artifacts.numeric_cols:
            df[c] = df[c].fillna(df[c].median())

        features = self.artifacts.categorical_cols + self.artifacts.numeric_cols
        X = self.artifacts.transformer.transform(df[features])
        return X.astype(np.float32)

    def save(self, path: Path) -> None:
        if self.artifacts is None:
            raise RuntimeError("Nothing to save. Fit the preprocessor first.")
        dump(self.artifacts, path)

    @staticmethod
    def load(path: Path) -> "TabularPreprocessor":
        artifacts = load(path)
        preprocessor = TabularPreprocessor()
        preprocessor.artifacts = artifacts
        return preprocessor
