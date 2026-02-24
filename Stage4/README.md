# Stage 4: Image + metadata fusion (binary: malignant vs benign)

**Standalone:** Stage 4 does not require Stage 1 or Stage 2 to be run. It extracts **handcrafted image features** (no CNN), combines them with **metadata**, and trains a binary classifier.

## What it does

1. **Image features (no CNN)**  
   For each image: color stats (RGB + HSV mean/std), texture (gray-level co-occurrence: contrast, homogeneity, energy, correlation), and per-channel histograms. Implemented in `Stage4/data/image_features.py` (uses `scikit-image` and NumPy).

2. **Metadata**  
   Same metadata as Stage 2 (age, sex, site, etc.). Stage 4 **fits its own** preprocessor on the train split (one-hot + scaling) and saves it under the Stage 4 output dir.

3. **Fusion**  
   Concatenate image feature vector + metadata feature vector per sample, then train XGBoost or Random Forest for **malignant (1) vs benign (0)**.

## Data

Uses the same CSVs as the rest of the project:

- `Dataset/stage1_train.csv`, `stage1_val.csv`, `stage1_test.csv` (columns: `sample_id`, `image_path`, `y`)
- `Dataset/combined_stage2.csv` (metadata + `y`, keyed by `sample_id`)

No Stage 1 or Stage 2 outputs are needed.

## Pipeline

```bash
# 1) Extract handcrafted image features (no GPU, no CNN)
python3 Stage4/extract_features.py --config Stage4/configs/stage4.yaml

# 2) Train fusion model (fits metadata preprocessor, trains classifier)
python3 Stage4/train.py --config Stage4/configs/stage4.yaml

# 3) Evaluate
python3 Stage4/eval.py --config Stage4/configs/stage4.yaml --ckpt Stage4/checkpoints/stage4_standalone/best.json
```

For Random Forest use `Stage4/configs/stage4_rf.yaml` and `--ckpt .../best.joblib`.

## Dependencies

- `numpy`, `pandas`, `Pillow`, `scikit-learn`, `scikit-image`, `xgboost`, `joblib`, `PyYAML`
- Optional: `scikit-image` for texture (GLCM); if missing, texture features are zeros and color/histogram features still run.

## Config

- `data`: `data_root`, `stage1_*_csv`, `stage2_csv`, `label_col`
- `output.dir`: Stage 4 output (features, preprocessor, model, metrics)
- `img_size`: image size for feature extraction (default 128)
- `model.name`: `xgboost` or `random_forest`

No `stage1` or `preprocessor_path` keys; Stage 4 is self-contained.
