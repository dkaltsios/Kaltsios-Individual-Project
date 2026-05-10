# Kaltsios Individual Project

Multimodal skin lesion classification and analysis pipeline: five training stages plus a Streamlit demo.

- `Stage1`: image models (EfficientNet/ResNet) for multiclass classification
- `Stage2`: metadata models (XGBoost / Random Forest)
- `Stage3`: late fusion of Stage1 + Stage2 predictions
- `Stage4`: early fusion (image handcrafted features + metadata)
- `Stage5`: unsupervised clustering baseline on Stage4-style merged features
- `demo/`: Streamlit app for interactive inference (CNN, metadata, late/early fusion, Grad-CAM, optional SHAP)

## Project Structure

- `Dataset/`: split CSVs and label mapping JSON
- `Stage1/`: image model train/eval + CV scripts
- `Stage2/`: metadata model train/eval + CV/grid-search scripts
- `Stage3/`: fusion script (`weighted_pool` or `stacking_lr`)
- `Stage4/`: feature extraction, early-fusion training/eval, feature selection, CV/grid search
- `Stage5/`: KMeans clustering train/eval on merged features
- `demo/`: Streamlit UI (`app.py`), inference helpers (`predict.py`), clinical-text parsing (`parse_metadata.py`), sample cases (`demo_clinical_cases.xlsx`)
- `demo_notebook.ipynb`: optional notebook walkthrough

## Stage 1 (Image Models)

### Train

`python3 Stage1/train.py --config Stage1/configs/efficientnet_b0_multiclass.yaml`

### Evaluate

`python3 Stage1/eval.py --config Stage1/configs/efficientnet_b0_multiclass.yaml --ckpt Stage1/checkpoints/efficientnet_b0_multiclass/best.pt --split test`

Use `--split val` to evaluate on validation data (requires `data.val_csv` in config).

## Stage 2 (Metadata Models)

### Train

`python3 Stage2/train.py --config Stage2/configs/xgboost_multiclass.yaml`

### Evaluate

`python3 Stage2/eval.py --config Stage2/configs/xgboost_multiclass.yaml --ckpt Stage2/checkpoints/xgboost_multiclass/best.json --split test`

Model choices from config:
- `xgboost` (checkpoint: `best.json`)
- `random_forest` (checkpoint: `best.joblib`)

## Stage 3 (Late Fusion)

Fusion entry point: `Stage3/fuse.py`

Supported methods:
- `weighted_pool`: weighted probability average
- `stacking_lr`: logistic regression stacker trained on validation predictions

### Example: weighted pool with fixed `w`

`python3 Stage3/fuse.py --stage1_preds Stage1/checkpoints/efficientnet_b0_multiclass/test_predictions.csv --stage2_preds Stage2/checkpoints/xgboost_multiclass/test_predictions.csv --out_dir Stage3/checkpoints/weighted_pool --method weighted_pool --w 0.5 --label_mapping Dataset/label_mapping_multiclass.json`

### Example: tune `w` on val, apply to test

1) `python3 Stage1/eval.py --config Stage1/configs/efficientnet_b0_multiclass.yaml --ckpt Stage1/checkpoints/efficientnet_b0_multiclass/best.pt --split val`
2) `python3 Stage2/eval.py --config Stage2/configs/xgboost_multiclass.yaml --ckpt Stage2/checkpoints/xgboost_multiclass/best.json --split val`
3) `python3 Stage3/fuse.py --stage1_preds Stage1/checkpoints/efficientnet_b0_multiclass/test_predictions.csv --stage2_preds Stage2/checkpoints/xgboost_multiclass/test_predictions.csv --val_stage1_preds Stage1/checkpoints/efficientnet_b0_multiclass/val_predictions.csv --val_stage2_preds Stage2/checkpoints/xgboost_multiclass/val_predictions.csv --out_dir Stage3/checkpoints/weighted_pool_tuned --method weighted_pool --label_mapping Dataset/label_mapping_multiclass.json`

If `--w` is omitted and validation predictions are provided, the script tunes `w` on validation.

## Stage 4 (Early Fusion: Image Features + Metadata)

Stage 4 merges:
- handcrafted image features extracted from Stage1 split images
- preprocessed Stage2 tabular metadata

### 1) Extract image features

`python3 Stage4/extract_features.py --config Stage4/configs/stage4_multiclass.yaml`

Feature extractors (from config key `feature_extractor`):
- `13feat` (compact handcrafted vector)
- `pareto` (high-dimensional feature vector for feature selection workflows)

### 2) (Optional) Feature selection

`python3 Stage4/select_features.py --config Stage4/configs/stage4_multiclass_pareto.yaml`

### 3) Train

`python3 Stage4/train.py --config Stage4/configs/stage4_multiclass.yaml`

Supported models:
- `xgboost`
- `softmax_regression`
- `naive_bayes`

### 4) Evaluate

`python3 Stage4/eval.py --config Stage4/configs/stage4_multiclass.yaml --ckpt Stage4/checkpoints/stage4_multiclass/best.json`

### CV and Grid Search

- `python3 Stage4/cv_stage4_multiclass.py --config <cv_config.yaml>`
- `python3 Stage4/grid_search_stage4_multiclass.py --config <grid_config.yaml>`

## Stage 5 (Unsupervised Clustering)

Stage 5 runs unsupervised clustering on merged Stage4-style features.
Target labels (`y_class`) are excluded from model inputs and used only for evaluation.

### Run

1) `python3 Stage4/extract_features.py --config Stage5/configs/stage5_multiclass_kmeans.yaml`
2) `python3 Stage5/train.py --config Stage5/configs/stage5_multiclass_kmeans.yaml`
3) `python3 Stage5/eval.py --config Stage5/configs/stage5_multiclass_kmeans.yaml --ckpt Stage5/checkpoints/stage5_multiclass_kmeans/kmeans.joblib --split test`

### Stage 5 metrics

`test_metrics.json` includes:
- `ari`
- `nmi`
- `v_measure`
- `purity`
- `mapped_accuracy` (cluster IDs mapped to labels using Hungarian matching)

## Demo (Streamlit)

Interactive demo: upload a lesion image, enter or paste clinical text (or use `demo/demo_clinical_cases.xlsx`), and compare **Stage 1–4** predictions with **Grad-CAM** (image) and **SHAP** (metadata) explainability. Late fusion supports **weighted pool** and **stacking** (logistic regression) where checkpoints exist.

### Run

From the **repository root**:

`streamlit run demo/app.py`

### Dependencies

Install the packages for the stages you trained (see `Stage1/requirements.txt`, `Stage2/requirements.txt`, and Stage 4 tooling as needed), plus:

`pip install streamlit shap matplotlib openai`

(`openai` is only required if you enable the GPT-backed clinical parser; see below.)

### Checkpoints

The app loads weights from the same paths used in training output directories, for example:

- `Stage1/checkpoints/…/best.pt` (CNN options listed in the UI)
- `Stage2/checkpoints/…/best.json` or `best.joblib` plus `preprocessor.joblib`
- `Stage3/checkpoints/stacking_lr/stacker.joblib` (for stacking late fusion)
- `Stage4/checkpoints/<run_name>/` (early fusion: `best.json` or `best.joblib`, preprocessor, optional scaler / feature selection)

Train or copy checkpoints into these locations before running the demo. (Large checkpoint files are typically gitignored.)

### Optional: GPT clinical text parsing

Free-text clinical notes can be parsed with **GPT-4o-mini** when `OPENAI_API_KEY` is set (environment variable, `demo/.env`, or Streamlit secrets). If the key is missing, the app falls back to a **regex-based parser**.

### Regenerate sample spreadsheet (optional)

`python3 demo/scripts/build_demo_clinical_cases_xlsx.py`  
(writes/updates `demo/demo_clinical_cases.xlsx` according to that script’s rules.)

## Common Outputs

For eval scripts, outputs are written under each stage checkpoint/output directory:

- `test_metrics.json` and `test_predictions.csv` for test split
- `val_metrics.json` and `val_predictions.csv` where val evaluation is supported
