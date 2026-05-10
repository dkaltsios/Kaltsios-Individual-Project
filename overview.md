# Skin Lesion Classification DL Framework — Project Overview

This document describes the **end-to-end multimodal pipeline** developed for multiclass skin lesion classification: convolutional image models, tabular metadata models, **late fusion** of their predictions, **early fusion** of handcrafted image features with metadata, and an optional **unsupervised clustering** baseline. The repository is organised in five stages; configuration is YAML-driven, and each stage writes metrics and predictions under its checkpoint directory.

---

## 1. Goal and task formulation

The supervised task is **multiclass disease classification** (for example seven classes: actinic keratosis, basal cell carcinoma, benign other, melanoma, nevus, squamous cell carcinoma, seborrheic keratosis). More specifically:

```json
{
  "num_classes": 7,
  "class_names": ["ak", "bcc", "benign_other", "melanoma", "nevus", "scc", "seborrheic_keratosis"]
}
```

The workflow uses predefined train/validation/test splits and YAML configuration files for reproducible runs.

---

## 2. High-level architecture

```text
                ┌──────────────┐
Images ───────▶ │   CNN Model  │ ──▶ Probabilities ──┐
                └──────────────┘                     │
                                                     ├──▶ Late Fusion ─▶ Final Prediction
                ┌──────────────┐                     │
Metadata ─────▶ │ Tabular ML   │ ──▶ Probabilities ──┘
                └──────────────┘

Images ───────▶ Feature Extraction ─┐
                                    ├──▶ Early Fusion ─▶ Tabular Model ─▶ Prediction
Metadata ───────────────────────────┘
```


| Stage | Role | Modality |
|-------|------|----------|
| **Stage 1** | CNN (EfficientNet / ResNet) on lesion images | Image |
| **Stage 2** | XGBoost and Random Forest on clinical/tabular metadata | Metadata |
| **Stage 3** | Late fusion: combine Stage 1 and Stage 2 **probability vectors** | Predictions |
| **Stage 4** | Early fusion: concatenate **handcrafted extracted features from image** + preprocessed metadata, then train a tabular classifier | Image + Metadata |
| **Stage 5** | K-Means on Stage 4–style merged features; labels used only for evaluation | Image + Metadata |

---

## 3. Stage 1 — Image models

**Data loading** uses PyTorch `Dataset` with ImageNet-style normalisation; training applies flips, rotation, and colour jitter; validation/test use resize and normalise only:

```python
def get_transforms(split, img_size=299):
    """
    Paper-style transforms.
    Xception expects 299x299.
    """
    if split == "train":
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(180),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:  # val / test
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
```

**Training** selects device (MPS / CUDA / CPU), builds the model from config, and uses inverse-frequency **class weights** for `CrossEntropyLoss` in multiclass runs:

```python
def compute_class_weights(df: pd.DataFrame, label_col: str, num_classes: int) -> torch.Tensor:
    """Inverse frequency weights for CrossEntropyLoss."""
    counts = df[label_col].value_counts().reindex(range(num_classes), fill_value=0).astype(float)
    total = counts.sum()
    if total == 0:
        return torch.ones(num_classes)
    weights = total / (num_classes * counts.clip(1))
    return torch.tensor(weights.values, dtype=torch.float32)
```

Evaluation scripts emit `test_predictions.csv` (and optionally validation) with `sample_id`, `y_true`, and `prob_*` columns used by Stage 3.

---

## 4. Stage 2 — Metadata models

Tabular columns are **inferred** (categorical vs numeric), with both `y` and `y_class` excluded from features to avoid label leakage when only one is the training target:

```python
    def _infer_columns(self, df: pd.DataFrame, label_col: str, id_cols: Iterable[str]):
        # Always exclude both target columns from candidate features.
        # This prevents multiclass runs (label_col=y_class) from leaking the binary
        # target "y", and vice-versa if both labels exist in a CSV.
        drop_cols = set(id_cols) | {label_col, "y", "y_class"}
        candidates = [c for c in df.columns if c not in drop_cols]

        if self._categorical_cols is None or self._numeric_cols is None:
            cat_cols = [c for c in candidates if df[c].dtype == "object"]
            num_cols = [c for c in candidates if c not in cat_cols]
        else:
            cat_cols = self._categorical_cols
            num_cols = self._numeric_cols
        return cat_cols, num_cols
```

Categorical features are one-hot encoded; numeric features are standardised. Checkpoints are written as JSON (XGBoost) or joblib (Random Forest), with the same prediction CSV schema as Stage 1 for fusion.

---

## 5. Stage 3 — Late fusion

Stage 3 merges prediction CSVs on `sample_id` and combines per-class probabilities with a **weighted linear pool** (weight `w` on Stage 1, `1-w` on Stage 2), renormalised to sum to 1:

```python
def weighted_linear_pool(p1: np.ndarray, p2: np.ndarray, w: float) -> np.ndarray:
    raw = w * p1 + (1.0 - w) * p2
    s = raw.sum(axis=1, keepdims=True)
    s = np.maximum(s, 1e-12)
    return raw / s
```

If validation predictions are supplied and `w` is omitted, the script can **grid-search** `w` on validation macro-F1. An alternative **stacking** path uses logistic regression on stacked validation probabilities.

---

## 6. Stage 4 — Early fusion (features + metadata)

**Handcrafted features** are extracted from images into NPZ files (`train.npz`, `val.npz`, `test.npz`) with aligned `sample_id` lists. The extractor is chosen in config (`13feat` vs `pareto`)*:

```python
def _get_extractor(name: str):
    if name == "13feat":
        from Stage4.data.image_features import extract_features_from_path
    elif name == "pareto":
        from Stage4.data.image_features_pareto import extract_features_from_path
    else:
        raise ValueError(
            f"Unknown feature_extractor: {name!r}. Choose '13feat' or 'pareto'."
        )
    return extract_features_from_path
```
*`13feat` and `pareto` correspond to two feature extraction strategies. The former uses a fixed set of 13 commonly used handcrafted image features reported in the literature, while the latter follows a Pareto-based approach: it starts from a larger pool of extracted features and selects the most informative subset based on performance.    

**Merging** concatenates image feature matrices with **transformed** metadata rows aligned by `sample_id` (same `TabularPreprocessor` as Stage 2, fitted on train when no saved preprocessor is passed):

```python
    def build_xy(sample_ids, img_feats):
        meta_df = stage2.set_index("sample_id").loc[sample_ids].reset_index()
        meta_df = _drop_requested_cols(meta_df)
        X_meta = preprocessor.transform(meta_df)
        y = meta_df[label_col].astype(int).to_numpy()
        X = np.concatenate([img_feats, X_meta], axis=1).astype(np.float32)
        return X, y
```

Downstream training supports models such as XGBoost, softmax regression, and naive Bayes; optional Pareto-based feature selection is also supported.

---

## 7. Stage 5 — Unsupervised clustering

Stage 5 reuses the merged feature pipeline but **does not** use disease labels as inputs—only for evaluation (e.g. ARI, NMI, V-measure, purity, Hungarian-mapped accuracy). Training fits `KMeans` on scaled features:

```python
"""
Stage 5 train: unsupervised clustering on merged multimodal features.
The target disease label is never used as an input feature.
"""
```

Feature matrices follow the same extraction and merge pattern as the early-fusion stage.

---

## 8. Outputs and reproducibility

Across supervised stages, evaluation typically writes:

- `test_metrics.json` — accuracy, F1, optionally AUC/AP where applicable  
- `test_predictions.csv` — identifiers, true labels, predicted class, per-class probabilities  

Stage 5 adds clustering-specific metrics in `test_metrics.json`. **Seeds** are set where relevant (NumPy, PyTorch, Python `random`) for reproducibility.

---

## 9. Experimental results

All numbers below come from the **test split** and summarise the runs currently available in this report snapshot; retraining can change values slightly.

### 9.1 Supervised models — overall test metrics

Macro-averaged metrics (multiclass: one-vs-rest ROC AUC and PR AUC where reported by the eval scripts):

| Stage | Model / setting | Accuracy | Macro F1 | Macro ROC AUC | Macro PR AUC |
|-------|-----------------|----------|----------|---------------|--------------|
| 1 | EfficientNet-B0 (multiclass) | 0.487 | 0.464 | 0.848 | 0.503 |
| 1 | ResNet-50 (multiclass) | 0.477 | 0.457 | 0.835 | 0.467 |
| 2 | XGBoost (metadata) | 0.474 | 0.403 | 0.817 | 0.476 |
| 2 | Random Forest (metadata) | 0.462 | 0.388 | 0.802 | 0.448 |
| 3 | Late fusion, `w = 0.5` (EfficientNet + XGBoost) | **0.611** | **0.583** | **0.919** | **0.650** |
| 3 | Late fusion, `w` tuned on validation (same backbones) | 0.539 | 0.502 | 0.869 | 0.552 |
| 4 | Early fusion (13 handcrafted features + metadata, config `stage4_multiclass`) | 0.466 | 0.391 | 0.814 | 0.465 |

**Takeaways.** The **image CNN** alone reaches moderate accuracy with fairly high **ROC AUC** (ranking quality), while **metadata-only** models are competitive on accuracy but weaker on minority classes in places. **Late fusion** with a fixed equal blend (`w = 0.5`) yields the strongest **accuracy and macro F1** in this table, showing that combining image and tabular probability vectors is effective. The **validation-tuned** fusion run stored next to it scores lower on this test snapshot—typical when validation selection does not transfer, or when checkpoints were produced at different times; always compare against the same prediction files when interpreting `w`.

**Early fusion** (Stage 4) under the saved `stage4_multiclass` run is slightly below the single-modality image model on these aggregates, which can happen with limited handcrafted features and a strong CNN baseline; Pareto / other Stage 4 configs may differ if you trained them.

### 9.2 Per-class F1 (illustrative)

Hardest classes in the saved runs are often **melanoma** and **benign_other** (class imbalance and overlap). Example **F1** from the EfficientNet-B0 test run: ak 0.658, bcc 0.564, benign_other 0.358, melanoma 0.327, nevus 0.453, scc 0.398, seborrheic_keratosis 0.489. The fused model (`weighted_pool`) improves melanoma F1 to about **0.504** on the same evaluation files.

### 9.3 Stage 3 fusion settings (saved metadata)

```json
{
  "method": "weighted_linear_pool",
  "w": 0.5,
  "tuned_on_val": false
}
```

The tuned fusion directory records validation macro-F1 **0.527** with a 21-step grid over `w`, objective macro-F1.

### 9.4 Stage 5 — Unsupervised clustering (test evaluation)

K-Means with **7** clusters on merged features (`n = 818` test samples), labels used only for metrics:

| Metric | Value |
|--------|------:|
| Adjusted Rand Index (ARI) | 0.095 |
| Normalized Mutual Information (NMI) | 0.194 |
| V-measure | 0.194 |
| Purity | 0.380 |
| Mapped accuracy (Hungarian) | 0.324 |

Unsupervised structure only partially aligns with disease labels, which is expected under high intra-class variability.

### 9.5 Cross-validation and grid search

It is useful to document **model selection** alongside test scores: **k-fold cross-validation** estimates performance on training data without using the test set; **grid search** tries many hyperparameter combinations and picks the best by a chosen validation metric (here typically **macro F1**).

**Scripts (entry points).**

| Stage | CV | Grid search |
|-------|-----|----------------|
| 1 | Yes | No |
| 2 | Yes | Yes |
| 4 | Yes | Yes |

Below: **5-fold** means and standard deviations from the cross-validation summaries (validation folds inside training data, not the final test split).

**Cross-validation summary (mean ± std)**

| Setup | What was cross-validated | Accuracy | Macro F1 | Macro ROC AUC | Macro PR AUC |
|-------|--------------------------|----------|----------|---------------|--------------|
| Stage 1 | CNN (ResNet-50) | 0.486 ± 0.009 | 0.459 ± 0.011 | 0.831 ± 0.009 | 0.481 ± 0.022 |
| Stage 2 | XGBoost, default config | 0.472 ± 0.025 | 0.396 ± 0.019 | 0.806 ± 0.011 | 0.459 ± 0.016 |
| Stage 2 | Random Forest, default config | 0.467 ± 0.015 | 0.382 ± 0.013 | 0.791 ± 0.009 | 0.428 ± 0.007 |
| Stage 4 | Early fusion, XGBoost, `13feat` | 0.480 ± 0.021 | 0.405 ± 0.017 | 0.808 ± 0.009 | 0.456 ± 0.015 |

Additional CV runs also exist for other Stage 4 setups (e.g. Pareto features, softmax, naive Bayes) using the same workflow with different model choices.

**Grid search — best configuration by macro F1 (5-fold mean ± std in each JSON)**

| Stage | Best hyperparameters (abbrev.) | Macro F1 (mean ± std) |
|-------|-------------------------------|------------------------|
| Stage 2 XGBoost | `max_depth` 3, `learning_rate` 0.05, `colsample_bytree` 0.9, `subsample` 0.7, `min_child_weight` 1 | 0.404 ± 0.021 |
| Stage 2 Random Forest | `max_depth` 10, `max_features` 0.5, `min_samples_leaf` 5 | 0.414 ± 0.020 |
| Stage 4 XGBoost (`13feat`) | `max_depth` 3, `learning_rate` 0.1, `colsample_bytree` 0.7, `subsample` 0.7 | 0.406 ± 0.023 |

CV scores are **not** directly comparable to the test table in §9.1 (different evaluation protocol and sometimes different configs than the final trained model). Together they show **stability across folds** (std) and **how hyperparameters were chosen** before final training and test evaluation.

---

## 10. Running the pipeline

All training/evaluation commands and experiment settings can be included in an appendix if required by the assessor.

---

*This overview is intended for coursework submission; it summarises design choices, model behaviour, and reported results as a standalone document.*
