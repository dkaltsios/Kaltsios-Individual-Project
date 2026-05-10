"""
demo/predict.py – single-sample inference wrappers for all pipeline stages.

Each loader is meant to be called once and cached via @st.cache_resource.
Each predictor takes lightweight inputs (bytes / dict) and returns a (7,) probability array.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CLASS_NAMES = ["ak", "bcc", "benign_other", "melanoma", "nevus", "scc", "seborrheic_keratosis"]
NUM_CLASSES = 7


# ── Stage 1 — CNN ─────────────────────────────────────────────────────────────

def load_cnn(model_name: str, ckpt_path: str, img_size: int = 224):
    """Load a Stage1 CNN from a .pt checkpoint.

    Returns (model, img_size, device).
    """
    import torch
    from Stage1.models import build_model

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = build_model(model_name, num_classes=NUM_CLASSES, pretrained=False)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model, img_size, device


def predict_cnn(
    model_bundle: Tuple,
    image_bytes,
    *,
    img_size: int | None = None,
) -> np.ndarray:
    """Run a single image (file-like / BytesIO) through the CNN.

    Returns a (7,) softmax probability array.
    """
    import torch
    from PIL import Image
    from Stage1.data.dataset import get_transforms

    model, default_size, device = model_bundle
    size = img_size if img_size is not None else default_size

    tf = get_transforms("val", img_size=size)
    img = Image.open(image_bytes).convert("RGB")
    tensor = tf(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    return probs.astype(np.float32)


def grad_cam(
    model_bundle,
    image_bytes,
    *,
    class_idx: int | None = None,
    img_size: int | None = None,
):
    """Compute a Grad-CAM heatmap for the top predicted class (or a given class_idx).

    Returns
    -------
    cam : np.ndarray, shape (H, W), values in [0, 1]
    orig_img : PIL.Image.Image  (RGB, original resolution)
    pred_idx : int
    """
    import torch
    from PIL import Image
    from Stage1.data.dataset import get_transforms

    model, default_size, device = model_bundle
    size = img_size if img_size is not None else default_size

    # Last convolutional block differs by architecture
    if hasattr(model, "layer4"):        # ResNet-50
        target_layer = model.layer4[-1]
    else:                               # EfficientNet-B0
        target_layer = model.features[-1]

    tf = get_transforms("val", img_size=size)
    orig_img = Image.open(image_bytes).convert("RGB")
    tensor = tf(orig_img).unsqueeze(0).to(device)

    activations: list = []
    gradients: list = []

    fh = target_layer.register_forward_hook(
        lambda m, i, o: activations.append(o)
    )
    bh = target_layer.register_full_backward_hook(
        lambda m, gi, go: gradients.append(go[0])
    )

    try:
        model.zero_grad()
        logits = model(tensor)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())
        logits[0, class_idx].backward()
    finally:
        fh.remove()
        bh.remove()

    acts  = activations[0].detach().squeeze(0)   # (C, H, W)
    grads = gradients[0].detach().squeeze(0)     # (C, H, W)

    weights = grads.mean(dim=(1, 2))             # (C,)
    cam = torch.relu((weights[:, None, None] * acts).sum(dim=0)).cpu().numpy()

    if cam.max() > 0:
        cam = cam / cam.max()

    return cam, orig_img, class_idx


def overlay_heatmap(
    cam: np.ndarray,
    orig_img,
    *,
    alpha: float = 0.45,
):
    """Blend a Grad-CAM heatmap onto the original image using the jet colormap.

    Returns a PIL.Image.Image (RGB).
    """
    from matplotlib import colormaps
    from PIL import Image

    cam_pil = Image.fromarray(np.uint8(255 * cam)).resize(orig_img.size, Image.BILINEAR)
    cam_np  = np.array(cam_pil) / 255.0

    cmap    = colormaps["jet"]
    heatmap = np.uint8(cmap(cam_np)[:, :, :3] * 255)
    orig_np = np.array(orig_img.convert("RGB"))
    blended = np.uint8(alpha * heatmap + (1.0 - alpha) * orig_np)
    return Image.fromarray(blended)


# ── Stage 2 — Tabular metadata model ──────────────────────────────────────────

def load_metadata_model(model_name: str, ckpt_path: str, preprocessor_path: str):
    """Load a Stage2 tabular model and its fitted preprocessor.

    Returns (model_name, model, preprocessor).
    """
    from Stage2.data.preprocess import TabularPreprocessor

    preprocessor = TabularPreprocessor.load(Path(preprocessor_path))

    if model_name == "xgboost":
        import xgboost as xgb

        booster = xgb.Booster()
        booster.load_model(ckpt_path)
        return ("xgboost", booster, preprocessor)

    if model_name == "random_forest":
        from joblib import load as jload

        clf = jload(ckpt_path)
        return ("random_forest", clf, preprocessor)

    raise ValueError(f"Unknown metadata model: {model_name!r}")


def predict_metadata(model_bundle: Tuple, metadata_dict: dict) -> np.ndarray:
    """Run a single metadata row through the tabular model.

    ``metadata_dict`` must contain at least the columns the preprocessor was
    fitted on (extra keys are silently ignored by the ColumnTransformer).
    Returns a (7,) probability array.
    """
    import pandas as pd

    model_name, model, preprocessor = model_bundle
    df = pd.DataFrame([metadata_dict])

    # Ensure dtypes match what the preprocessor was fitted on
    for c in preprocessor.artifacts.numeric_cols:
        df[c] = df[c].astype(float)
    for c in preprocessor.artifacts.categorical_cols:
        df[c] = df[c].astype(str)

    X = preprocessor.transform(df)

    if model_name == "xgboost":
        import xgboost as xgb

        probs = model.predict(xgb.DMatrix(X))
        if probs.ndim == 1:
            probs = probs.reshape(1, -1)
    else:
        probs = model.predict_proba(X)

    return probs[0].astype(np.float32)


# Human-readable labels for every feature the preprocessor can produce
_FEATURE_DISPLAY: dict[str, str] = {
    "sex": "Sex",
    "anatomical_site_clean": "Anatomical site",
    "age": "Age",
    "fitzpatrick": "Fitzpatrick scale",
    "lesion_size_mm": "Lesion size (mm)",
    "diameter_2": "Diameter 2 (mm)",
    "bleed": "Bleed",
    "hurt": "Hurt",
    "itch": "Itch",
    "changed": "Changed",
    "grew": "Grew",
    "elevation": "Elevation",
    "smoking": "Smoking",
    "alcohol_consumption": "Alcohol",
    "cancer_history": "Cancer history",
    "skin_cancer_history": "Skin cancer history",
    "pesticide": "Pesticide",
    "has_piped_water": "Piped water",
    "has_sewage_system": "Sewage system",
}


def _clean_feature_name(fname: str, cat_cols: list[str]) -> str:
    """Convert a raw preprocessor feature name into a human-readable label."""
    for col in sorted(cat_cols, key=len, reverse=True):
        if fname.startswith(col + "_"):
            val = fname[len(col) + 1:].replace("_", " ")
            label = _FEATURE_DISPLAY.get(col, col.replace("_", " ").title())
            return f"{label}: {val}"
    return _FEATURE_DISPLAY.get(fname, fname.replace("_", " ").title())


def shap_explanation(
    model_bundle,
    metadata_dict: dict,
    pred_idx: int,
    top_n: int = 10,
):
    """Compute SHAP feature contributions for the predicted class.

    Returns a pandas Series mapping cleaned feature names to their SHAP values,
    sorted by absolute value descending (top_n entries).
    """
    import pandas as pd
    import shap

    model_name, model, preprocessor = model_bundle
    df = pd.DataFrame([metadata_dict])

    for c in preprocessor.artifacts.numeric_cols:
        df[c] = df[c].astype(float)
    for c in preprocessor.artifacts.categorical_cols:
        df[c] = df[c].astype(str)

    X = preprocessor.transform(df)

    feature_names = preprocessor.artifacts.feature_names
    cat_cols      = preprocessor.artifacts.categorical_cols

    explainer = shap.TreeExplainer(model)
    raw = explainer.shap_values(X)

    # Handle both list (older SHAP) and ndarray (newer SHAP) returns
    if isinstance(raw, list):
        vals = raw[pred_idx][0]
    elif hasattr(raw, "ndim") and raw.ndim == 3:
        vals = raw[0, :, pred_idx]
    else:
        vals = raw[0]

    cleaned = [_clean_feature_name(f, cat_cols) for f in feature_names]
    series = pd.Series(vals, index=cleaned)
    return series.reindex(series.abs().sort_values(ascending=False).index).head(top_n)


# ── Stage 3 — Late fusion ──────────────────────────────────────────────────────

def late_fusion(
    p_cnn: np.ndarray,
    p_meta: np.ndarray,
    w: float = 0.5,
) -> np.ndarray:
    """Weighted linear pool of two (7,) probability vectors, renormalised."""
    raw = w * p_cnn + (1.0 - w) * p_meta
    total = raw.sum()
    return (raw / max(total, 1e-12)).astype(np.float32)


def load_stacking_lr(ckpt_path: str):
    """Load the stacking logistic-regression model saved in Stage3/checkpoints/stacking_lr."""
    from joblib import load as jload
    return jload(ckpt_path)


def late_fusion_stacking(
    stacker,
    p_cnn: np.ndarray,
    p_meta: np.ndarray,
) -> np.ndarray:
    """Run the stacking LR meta-learner on concatenated CNN + metadata probability vectors.

    Returns a (7,) probability array.
    """
    X = np.concatenate([p_cnn, p_meta]).reshape(1, -1)
    probs = stacker.predict_proba(X)[0].astype(np.float32)
    return probs


# ── Stage 4 — Early fusion ─────────────────────────────────────────────────────

def load_early_fusion_model(ckpt_dir: str):
    """Load a Stage4 early-fusion model from its checkpoint directory.

    Detects model type (XGBoost / sklearn) and optional scaler / Pareto
    feature-selection indices automatically from the files present.

    Returns (model_type, model, preprocessor, scaler_or_None, feat_indices_or_None).
    """
    import json
    from Stage2.data.preprocess import TabularPreprocessor

    ckpt = Path(ckpt_dir)
    preprocessor = TabularPreprocessor.load(ckpt / "preprocessor.joblib")

    # Pareto feature-selection indices (only in pareto variants)
    feat_indices = None
    sel_path = ckpt / "feature_selection.json"
    if sel_path.exists():
        sel = json.loads(sel_path.read_text())
        feat_indices = sel["selected_indices"]

    # Optional StandardScaler (softmax / Naive-Bayes variants)
    scaler = None
    scaler_path = ckpt / "scaler.joblib"
    if scaler_path.exists():
        from joblib import load as jload
        scaler = jload(scaler_path)

    # Load the model — XGBoost (.json) or sklearn (.joblib)
    xgb_path    = ckpt / "best.json"
    joblib_path = ckpt / "best.joblib"

    if xgb_path.exists():
        import xgboost as xgb
        booster = xgb.Booster()
        booster.load_model(str(xgb_path))
        return ("xgboost", booster, preprocessor, scaler, feat_indices)

    if joblib_path.exists():
        from joblib import load as jload
        model = jload(joblib_path)
        return ("sklearn", model, preprocessor, scaler, feat_indices)

    raise FileNotFoundError(f"No model file (best.json / best.joblib) found in {ckpt_dir}")


def predict_early_fusion(
    model_bundle: Tuple,
    image_bytes,
    metadata_dict: dict,
    *,
    img_size: int = 128,
) -> np.ndarray:
    """Extract handcrafted image features, concatenate with preprocessed
    metadata, and run through the Stage4 model.

    Handles XGBoost, Softmax Regression, and Naive Bayes variants, with
    optional Pareto feature selection and StandardScaler.

    Returns a (7,) probability array.
    """
    import pandas as pd
    from PIL import Image
    from Stage4.data.image_features import extract_features_from_array

    model_type, model, preprocessor, scaler, feat_indices = model_bundle

    img = Image.open(image_bytes).convert("RGB")
    rgb = np.array(img)
    img_feats = extract_features_from_array(rgb, img_size=img_size).reshape(1, -1)

    # Apply Pareto feature selection when present
    if feat_indices is not None:
        img_feats = img_feats[:, feat_indices]

    df = pd.DataFrame([metadata_dict])
    for c in preprocessor.artifacts.numeric_cols:
        df[c] = df[c].astype(float)
    for c in preprocessor.artifacts.categorical_cols:
        df[c] = df[c].astype(str)
    X_meta = preprocessor.transform(df)

    X = np.concatenate([img_feats, X_meta], axis=1).astype(np.float32)

    # Apply scaler for sklearn models that require it
    if scaler is not None:
        X = scaler.transform(X)

    if model_type == "xgboost":
        import xgboost as xgb
        probs = model.predict(xgb.DMatrix(X))
        if probs.ndim == 1:
            probs = probs.reshape(1, -1)
    else:
        probs = model.predict_proba(X)

    return probs[0].astype(np.float32)
