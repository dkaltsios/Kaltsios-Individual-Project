"""
Skin Lesion Classification — Multimodal Demo
Run from the project root:  streamlit run demo/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from demo.predict import (
    CLASS_NAMES,
    grad_cam,
    late_fusion,
    late_fusion_stacking,
    load_cnn,
    load_early_fusion_model,
    load_metadata_model,
    load_stacking_lr,
    overlay_heatmap,
    predict_cnn,
    predict_early_fusion,
    predict_metadata,
    shap_explanation,
)
import os

from demo.parse_metadata import EXAMPLE_TEXT, KNOWN_SITES, parse_clinical_text

# ── Constants ──────────────────────────────────────────────────────────────────

CKPTS = {
    "efficientnet_b0":        ROOT / "Stage1/checkpoints/efficientnet_b0_multiclass/best.pt",
    "resnet50":               ROOT / "Stage1/checkpoints/resnet50_multiclass/best.pt",
    "efficientnet_b0_aug":    ROOT / "Stage1/checkpoints/efficientnet_b0_multiclass_aug/best.pt",
    "resnet50_aug":           ROOT / "Stage1/checkpoints/resnet50_multiclass_aug/best.pt",
    "efficientnet_b0_aug_v2": ROOT / "Stage1/checkpoints/efficientnet_b0_multiclass_aug_v2/best.pt",
    "resnet50_aug_v2":        ROOT / "Stage1/checkpoints/resnet50_multiclass_aug_v2/best.pt",
    "xgboost":             ROOT / "Stage2/checkpoints/xgboost_multiclass/best.json",
    "xgboost_prep":        ROOT / "Stage2/checkpoints/xgboost_multiclass/preprocessor.joblib",
    "random_forest":       ROOT / "Stage2/checkpoints/random_forest_multiclass/best.joblib",
    "rf_prep":             ROOT / "Stage2/checkpoints/random_forest_multiclass/preprocessor.joblib",
}

LATE_FUSION_OPTIONS = {
    "Weighted Pool  (w = 0.5)":      "weighted_pool",
    "Stacking — Logistic Regression": "stacking_lr",
}

LATE_FUSION_CKPTS = {
    "stacking_lr": ROOT / "Stage3/checkpoints/stacking_lr/stacker.joblib",
}

EARLY_FUSION_CKPTS = {
    "stage4_multiclass":                ROOT / "Stage4/checkpoints/stage4_multiclass",
    "stage4_multiclass_nb":             ROOT / "Stage4/checkpoints/stage4_multiclass_nb",
    "stage4_multiclass_pareto":         ROOT / "Stage4/checkpoints/stage4_multiclass_pareto",
    "stage4_multiclass_pareto_nb":      ROOT / "Stage4/checkpoints/stage4_multiclass_pareto_nb",
    "stage4_multiclass_pareto_softmax": ROOT / "Stage4/checkpoints/stage4_multiclass_pareto_softmax",
    "stage4_multiclass_softmax":        ROOT / "Stage4/checkpoints/stage4_multiclass_softmax",
}

IMG_SIZE = {
    "efficientnet_b0":        224,
    "resnet50":               224,
    "efficientnet_b0_aug":    224,
    "resnet50_aug":           224,
    "efficientnet_b0_aug_v2": 224,
    "resnet50_aug_v2":        224,
}
CNN_OPTIONS = {
    "EfficientNet-B0":                     "efficientnet_b0",
    "EfficientNet-B0 (augmented v1)":      "efficientnet_b0_aug",
    "EfficientNet-B0 (augmented v2)":      "efficientnet_b0_aug_v2",
    "ResNet-50":                           "resnet50",
    "ResNet-50 (augmented v1)":            "resnet50_aug",
    "ResNet-50 (augmented v2)":            "resnet50_aug_v2",
}

# Changes applied in all augmented variants vs the baseline
_AUG_CHANGES_SHARED = [
    ("Random Resized Crop",   "Scale 85–100 % of the image before resizing to 224×224, "
                              "adding scale and position variation. "
                              "Baseline uses a plain resize."),
    ("Stronger Color Jitter", "Brightness ±0.4, contrast ±0.4, saturation ±0.3, hue ±0.05 "
                              "(baseline: brightness/contrast/saturation ±0.2, no hue)."),
    ("Random Erasing",        "With p = 0.5, a random rectangle covering 2–15 % of the image "
                              "is zeroed out after normalisation, discouraging shortcut learning "
                              "on artefacts such as ruler marks or hair."),
]
AUG_CHANGES = {
    "v1": _AUG_CHANGES_SHARED + [
        ("Weighted Sampler (hard)", "1 / count — fully balances all classes each epoch. "
                                    "Improved Melanoma and SCC recall but suppressed BCC recall."),
    ],
    "v2": _AUG_CHANGES_SHARED + [
        ("Weighted Sampler (soft)", "1 / √count — partial balance; minority classes "
                                    "are oversampled but majority classes (BCC) retain "
                                    "more exposure than in v1."),
        ("Cosine Annealing LR",     "Learning rate decays smoothly from 3×10⁻⁴ to 10⁻⁶ "
                                    "over 30 epochs, allowing fine-grained convergence "
                                    "in later epochs."),
        ("30 epochs",               "Double the baseline training budget, giving the model "
                                    "sufficient time to converge under harder augmentation."),
    ],
}
META_OPTIONS = {"XGBoost": "xgboost", "Random Forest": "random_forest"}
EARLY_FUSION_OPTIONS = {
    "Softmax Regression — All features":    "stage4_multiclass_softmax",
    "Softmax Regression — Pareto features": "stage4_multiclass_pareto_softmax",
    "XGBoost — Pareto features":            "stage4_multiclass_pareto",
    "XGBoost — All features":               "stage4_multiclass",
    "Naive Bayes — All features":           "stage4_multiclass_nb",
    "Naive Bayes — Pareto features":        "stage4_multiclass_pareto_nb",
}

DISPLAY_NAMES = {
    "ak":                   "Actinic Keratosis",
    "bcc":                  "Basal Cell Carcinoma",
    "benign_other":         "Benign Other",
    "melanoma":             "Melanoma",
    "nevus":                "Nevus",
    "scc":                  "Squamous Cell Carcinoma",
    "seborrheic_keratosis": "Seborrheic Keratosis",
}

MODALITY_OPTIONS = [
    "Unimodal — Image CNN",
    "Unimodal — Metadata only",
    "Multimodal — Late Fusion",
    "Multimodal — Early Fusion",
]

# ── Explanation helpers ────────────────────────────────────────────────────────

def _cnn_explanation_text(probs: np.ndarray) -> str:
    pred_idx   = int(probs.argmax())
    pred_name  = DISPLAY_NAMES[CLASS_NAMES[pred_idx]]
    confidence = float(probs[pred_idx])

    if confidence >= 0.80:
        level = "high"
    elif confidence >= 0.55:
        level = "moderate"
    else:
        level = "low — the model is uncertain between multiple classes"

    sorted_idx      = probs.argsort()[::-1]
    runner_up_name  = DISPLAY_NAMES[CLASS_NAMES[sorted_idx[1]]]
    runner_up_conf  = float(probs[sorted_idx[1]])

    return (
        f"The CNN predicted **{pred_name}** with {confidence:.1%} confidence "
        f"(**{level}** confidence). "
        f"The next most likely class was **{runner_up_name}** at {runner_up_conf:.1%}. "
        f"The Grad-CAM map above highlights the lesion regions that drove this decision."
    )


def _render_aug_info(model_key: str) -> None:
    """Render a compact description of augmentation changes for the augmented CNN variants."""
    version  = "v2" if model_key.endswith("_v2") else "v1"
    changes  = AUG_CHANGES[version]
    rows = "".join(
        f"<tr>"
        f"<td style='padding:4px 12px 4px 0;color:#60a5fa;font-weight:600;white-space:nowrap'>{name}</td>"
        f"<td style='padding:4px 0;color:#c9d1d9;font-size:0.85rem'>{desc}</td>"
        f"</tr>"
        for name, desc in changes
    )
    label = f"Augmentation changes vs baseline (v{version[-1]})"
    st.markdown(
        f"<div style='background:#161b22;border:1px solid #30363d;border-radius:8px;"
        f"padding:0.9rem 1.1rem;margin-top:0.6rem'>"
        f"<div style='font-size:0.78rem;font-weight:700;color:#8b949e;text-transform:uppercase;"
        f"letter-spacing:.06em;margin-bottom:0.6rem'>{label}</div>"
        f"<table style='border-collapse:collapse;width:100%'>{rows}</table>"
        f"</div>",
        unsafe_allow_html=True,
    )


def _render_shap_chart(shap_series: "pd.Series", pred_name: str) -> None:
    import matplotlib.pyplot as plt

    labels = list(shap_series.index)
    values = list(shap_series.values)
    colors = ["#22c55e" if v >= 0 else "#ef4444" for v in values]

    fig, ax = plt.subplots(figsize=(7, max(3, len(labels) * 0.42)))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    ax.barh(range(len(labels)), values, color=colors)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, color="#e6edf3", fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0, color="#8b949e", linewidth=0.8)
    ax.set_xlabel("SHAP value  (+ pushes toward · − pushes away)", color="#8b949e", fontsize=8)
    ax.tick_params(colors="#8b949e", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    ax.set_title(f"Feature contributions → {pred_name}", color="#e6edf3", fontsize=10, pad=8)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ── Cached loaders ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading CNN…")
def get_cnn(model_name: str):
    return load_cnn(model_name, str(CKPTS[model_name]), img_size=IMG_SIZE[model_name])

@st.cache_resource(show_spinner="Loading metadata model…")
def get_metadata_model(model_name: str):
    prep_key = "xgboost_prep" if model_name == "xgboost" else "rf_prep"
    return load_metadata_model(model_name, str(CKPTS[model_name]), str(CKPTS[prep_key]))

@st.cache_resource(show_spinner="Loading early-fusion model…")
def get_early_fusion_model(model_key: str):
    return load_early_fusion_model(str(EARLY_FUSION_CKPTS[model_key]))

@st.cache_resource(show_spinner="Loading stacking model…")
def get_stacking_lr():
    return load_stacking_lr(str(LATE_FUSION_CKPTS["stacking_lr"]))

# ── Styling ────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Multimodal Skin Lesion Classifier",
    page_icon=None,
    layout="wide",
)

st.markdown("""
<style>
/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 0 !important; max-width: 1280px; }

/* ── Hero header ── */
.hero {
    background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 60%, #60a5fa 100%);
    padding: 2.8rem 2rem 2.2rem;
    text-align: center;
    margin: -4rem -4rem 2.5rem -4rem;
}
.hero h1 {
    color: #ffffff;
    font-size: 2.4rem;
    font-weight: 700;
    letter-spacing: -0.5px;
    margin: 0 0 0.4rem 0;
}
.hero p {
    color: #bfdbfe;
    font-size: 1rem;
    margin: 0;
}

/* ── Section card header ── */
.card-title {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 1.05rem;
    font-weight: 600;
    color: #e6edf3;
    margin-bottom: 1.1rem;
}
.card-title .icon {
    font-size: 1.1rem;
}

/* ── Step badge (numbered circle) ── */
.step-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 26px;
    height: 26px;
    min-width: 26px;
    background: #2563eb;
    border-radius: 50%;
    color: #fff;
    font-size: 0.78rem;
    font-weight: 700;
    margin-right: 6px;
}

/* ── Result prediction block ── */
.pred-label {
    font-size: 1.5rem;
    font-weight: 700;
    color: #60a5fa;
    margin: 0.3rem 0 0.1rem 0;
}
.pred-confidence {
    font-size: 0.9rem;
    color: #8b949e;
    margin-bottom: 0.8rem;
}
.result-source {
    font-size: 0.78rem;
    color: #6e7681;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 600;
    margin-bottom: 0.6rem;
}

/* ── Metric override ── */
[data-testid="stMetricValue"] { color: #60a5fa !important; }
[data-testid="stMetricLabel"] { color: #8b949e !important; }

/* ── Primary button ── */
div[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(90deg, #1d4ed8, #2563eb) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.55rem 2.5rem !important;
    font-size: 1rem !important;
}
div[data-testid="stButton"] > button[kind="primary"]:hover {
    background: linear-gradient(90deg, #1e40af, #1d4ed8) !important;
}

/* ── Subtle horizontal rule ── */
hr { border-color: #30363d !important; margin: 1.4rem 0 !important; }


</style>
""", unsafe_allow_html=True)

# ── Resolve OpenAI API key (env var only — never stored in the app) ────────────

def _resolve_openai_key() -> str | None:
    # 1. Environment variable (set before launching streamlit)
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if key:
        return key
    # 2. .env file in the demo directory (or project root)
    for env_path in [
        Path(__file__).parent / ".env",
        ROOT / ".env",
    ]:
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if line.startswith("OPENAI_API_KEY"):
                    _, _, val = line.partition("=")
                    val = val.strip().strip('"').strip("'")
                    if val:
                        return val
    # 3. Streamlit secrets (optional — only if secrets.toml exists)
    try:
        key = st.secrets.get("OPENAI_API_KEY", "").strip()
        if key:
            return key
    except Exception:
        pass
    return None

_openai_key: str | None = _resolve_openai_key()

with st.sidebar:
    st.markdown("### Metadata parser")
    if _openai_key:
        st.success("GPT-4o-mini parser active")
    else:
        st.info("Regex parser active")
        st.caption(
            "Set the `OPENAI_API_KEY` in `demo/.env` to enable the GPT-4o-mini parser."
        )

# ── Hero header ────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero">
    <h1>Multimodal Skin Lesion Classifier</h1>
    <p>Advanced multimodal analysis for skin lesion classification</p>
</div>
""", unsafe_allow_html=True)

# ── Step 1 — Modality & model selection ───────────────────────────────────────

with st.container(border=True):
    st.markdown('<div class="card-title"><span class="step-badge">1</span> Select modality and model</div>',
                unsafe_allow_html=True)

    modality = st.radio(
        "Modality",
        MODALITY_OPTIONS,
        horizontal=True,
        label_visibility="collapsed",
    )

    use_image = modality in {
        "Unimodal — Image CNN",
        "Multimodal — Late Fusion",
        "Multimodal — Early Fusion",
    }
    use_meta = modality in {
        "Unimodal — Metadata only",
        "Multimodal — Late Fusion",
        "Multimodal — Early Fusion",
    }

    cnn_model_key      = None
    meta_model_key     = None
    fusion_w           = 0.5
    late_fusion_method = "weighted_pool"
    lf_label           = None
    early_fusion_key   = None
    ef_label           = None

    st.markdown("")
    sel1, sel2, sel3 = st.columns([1, 1, 1])

    if modality == "Unimodal — Image CNN":
        with sel1:
            cnn_label     = st.selectbox("CNN model", list(CNN_OPTIONS.keys()))
            cnn_model_key = CNN_OPTIONS[cnn_label]
        if "_aug" in cnn_model_key:
            _render_aug_info(cnn_model_key)

    elif modality == "Unimodal — Metadata only":
        with sel1:
            meta_label     = st.selectbox("Metadata model", list(META_OPTIONS.keys()))
            meta_model_key = META_OPTIONS[meta_label]

    elif modality == "Multimodal — Late Fusion":
        with sel1:
            cnn_label     = st.selectbox("CNN model", list(CNN_OPTIONS.keys()))
            cnn_model_key = CNN_OPTIONS[cnn_label]
        if "_aug" in cnn_model_key:
            _render_aug_info(cnn_model_key)
        with sel2:
            meta_label     = st.selectbox("Metadata model", list(META_OPTIONS.keys()))
            meta_model_key = META_OPTIONS[meta_label]
        with sel3:
            lf_label          = st.selectbox("Fusion method", list(LATE_FUSION_OPTIONS.keys()))
            late_fusion_method = LATE_FUSION_OPTIONS[lf_label]
        fusion_w = 0.5

    elif modality == "Multimodal — Early Fusion":
        with sel1:
            ef_label         = st.selectbox("Early fusion model", list(EARLY_FUSION_OPTIONS.keys()))
            early_fusion_key = EARLY_FUSION_OPTIONS[ef_label]

# ── Step 2 — Inputs ────────────────────────────────────────────────────────────

with st.container(border=True):
    st.markdown('<div class="card-title"><span class="step-badge">2</span> Provide inputs</div>',
                unsafe_allow_html=True)

    _spec = [1, 1] if (use_image and use_meta) else [1]
    _cols = st.columns(_spec)
    img_col  = _cols[0]
    meta_col = _cols[1] if len(_cols) > 1 else _cols[0]

    uploaded_file = None
    if use_image:
        with img_col:
            st.markdown("**Lesion image**")
            uploaded_file = st.file_uploader(
                "Upload image", type=["jpg", "jpeg", "png"],
                label_visibility="collapsed",
            )
            if uploaded_file is not None:
                st.image(uploaded_file, use_container_width=True)

    metadata_dict: dict = {}
    if use_meta:
        with meta_col:
            st.markdown("**Patient description**")

            clinical_text = st.text_area(
                "Describe the patient",
                value=EXAMPLE_TEXT,
                height=110,
                label_visibility="collapsed",
                help="Write a free-text clinical description. The system will extract the relevant fields automatically.",
            )

            # ── Parse text ────────────────────────────────────────────────
            parsed = parse_clinical_text(clinical_text, api_key=_openai_key)
            p = parsed.metadata

            if parsed.parser_used == "openai":
                st.markdown(
                    "<small style='color:#22c55e'>Parsed with GPT-4o-mini</small>",
                    unsafe_allow_html=True,
                )
            elif parsed.error:
                st.warning(f"OpenAI parse failed, using regex fallback. Error: {parsed.error}")
            elif not _openai_key:
                st.markdown(
                    "<small style='color:#8b949e'>Parsed with regex (add API key to demo/.env for GPT-4o-mini)</small>",
                    unsafe_allow_html=True,
                )

            # ── Expander: manual override form ─────────────────────────────
            with st.expander("Edit fields manually"):
                st.caption("Values below are pre-filled from your description. Change any field to override.")
                r1, r2 = st.columns(2)
                with r1:
                    age         = st.number_input("Age",             min_value=1,   max_value=110,   value=int(p["age"]))
                    lesion_size = st.number_input("Lesion size (mm)", min_value=0.0, max_value=200.0, value=float(p["lesion_size_mm"]), step=0.5)
                    diameter_2  = st.number_input("Diameter 2 (mm)",  min_value=0.0, max_value=200.0, value=float(p["diameter_2"]),     step=0.5)
                    fitzpatrick = st.selectbox("Fitzpatrick scale", [1, 2, 3, 4, 5, 6],
                                              index=[1,2,3,4,5,6].index(int(p["fitzpatrick"])))
                with r2:
                    sex  = st.selectbox("Sex", ["male", "female"],
                                        index=["male","female"].index(p["sex"]))
                    site = st.selectbox("Anatomical site", KNOWN_SITES,
                                        index=KNOWN_SITES.index(p["anatomical_site_clean"])
                                        if p["anatomical_site_clean"] in KNOWN_SITES else 0)

                st.markdown("**Symptoms**")
                s1, s2, s3 = st.columns(3)
                with s1:
                    bleed     = int(st.checkbox("Bleed",     value=bool(p["bleed"])))
                    hurt      = int(st.checkbox("Hurt",      value=bool(p["hurt"])))
                with s2:
                    itch      = int(st.checkbox("Itch",      value=bool(p["itch"])))
                    changed   = int(st.checkbox("Changed",   value=bool(p["changed"])))
                with s3:
                    grew      = int(st.checkbox("Grew",      value=bool(p["grew"])))
                    elevation = int(st.checkbox("Elevation", value=bool(p["elevation"])))

                st.markdown("**History & environment**")
                h1, h2, h3 = st.columns(3)
                with h1:
                    smoking = int(st.checkbox("Smoking", value=bool(p["smoking"])))
                    alcohol = int(st.checkbox("Alcohol", value=bool(p["alcohol_consumption"])))
                with h2:
                    cancer_hist      = int(st.checkbox("Cancer history",      value=bool(p["cancer_history"])))
                    skin_cancer_hist = int(st.checkbox("Skin cancer history", value=bool(p["skin_cancer_history"])))
                with h3:
                    pesticide   = int(st.checkbox("Pesticide exposure", value=bool(p["pesticide"])))
                    piped_water = int(st.checkbox("Piped water",        value=bool(p["has_piped_water"])))
                    sewage      = int(st.checkbox("Sewage system",      value=bool(p["has_sewage_system"])))

            metadata_dict = {
                "age": age, "sex": sex, "fitzpatrick": fitzpatrick,
                "lesion_size_mm": lesion_size, "diameter_2": diameter_2,
                "anatomical_site_clean": site,
                "bleed": bleed, "hurt": hurt, "itch": itch,
                "changed": changed, "grew": grew, "elevation": elevation,
                "smoking": smoking, "alcohol_consumption": alcohol,
                "cancer_history": cancer_hist, "skin_cancer_history": skin_cancer_hist,
                "pesticide": pesticide, "has_piped_water": piped_water,
                "has_sewage_system": sewage,
            }

# ── Predict button ─────────────────────────────────────────────────────────────

predict_ready = not (use_image and uploaded_file is None)

if not predict_ready:
    st.info("Upload a lesion image above to enable prediction.")

_, btn_col, _ = st.columns([2, 1, 2])
with btn_col:
    run = st.button("Run Analysis", type="primary", disabled=not predict_ready, use_container_width=True)

# ── Step 3 — Results ───────────────────────────────────────────────────────────

if run:
    cnn_probs = meta_probs = fused_probs = early_probs = None
    cam_overlay = None
    cam_orig    = None
    meta_shap   = None

    with st.spinner("Running pipeline…"):
        if cnn_model_key is not None:
            bundle = get_cnn(cnn_model_key)
            uploaded_file.seek(0)
            cnn_probs = predict_cnn(bundle, uploaded_file)
            uploaded_file.seek(0)
            _cam, cam_orig, _ = grad_cam(bundle, uploaded_file)
            cam_overlay = overlay_heatmap(_cam, cam_orig)

        if meta_model_key is not None:
            bundle = get_metadata_model(meta_model_key)
            meta_probs = predict_metadata(bundle, metadata_dict)
            if modality == "Unimodal — Metadata only":
                meta_shap = shap_explanation(bundle, metadata_dict, int(meta_probs.argmax()))

        if modality == "Multimodal — Late Fusion":
            if late_fusion_method == "stacking_lr":
                stacker = get_stacking_lr()
                fused_probs = late_fusion_stacking(stacker, cnn_probs, meta_probs)
            else:
                fused_probs = late_fusion(cnn_probs, meta_probs, w=fusion_w)

        if modality == "Multimodal — Early Fusion":
            bundle = get_early_fusion_model(early_fusion_key)
            uploaded_file.seek(0)
            early_probs = predict_early_fusion(bundle, uploaded_file, metadata_dict)

    st.markdown("---")
    with st.container(border=True):
        st.markdown('<div class="card-title"><span class="step-badge">3</span> Results</div>',
                    unsafe_allow_html=True)

        def _result_panel(source_label: str, probs: np.ndarray) -> None:
            pred_idx   = int(probs.argmax())
            pred_name  = DISPLAY_NAMES[CLASS_NAMES[pred_idx]]
            confidence = float(probs[pred_idx])

            st.markdown(f'<p class="result-source">{source_label}</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="pred-label">{pred_name}</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="pred-confidence">{confidence:.1%} confidence</p>', unsafe_allow_html=True)

            df = pd.DataFrame(
                {"Probability": probs},
                index=[DISPLAY_NAMES[c] for c in CLASS_NAMES],
            )
            st.bar_chart(df, height=230)

        if modality == "Unimodal — Image CNN":
            _result_panel(cnn_label, cnn_probs)

        elif modality == "Unimodal — Metadata only":
            _result_panel(meta_label, meta_probs)

        elif modality == "Multimodal — Late Fusion":
            c1, c2, c3 = st.columns(3)
            with c1:
                _result_panel(f"CNN — {cnn_label}", cnn_probs)
            with c2:
                _result_panel(f"Metadata — {meta_label}", meta_probs)
            with c3:
                _result_panel(f"Late Fusion — {lf_label}", fused_probs)
            if late_fusion_method == "stacking_lr":
                st.caption(
                    "Stacking LR: a Logistic Regression meta-learner trained on the "
                    "concatenated CNN and metadata probability vectors from the validation set."
                )
            else:
                st.caption(
                    "Weighted Pool: equal-weight average of the CNN and metadata "
                    "probability vectors (w = 0.5), matching the published evaluation."
                )

        elif modality == "Multimodal — Early Fusion":
            _result_panel(f"Early Fusion — {ef_label}", early_probs)

        # ── Explanation subsection ─────────────────────────────────────────
        if modality == "Unimodal — Image CNN" or meta_shap is not None:
            st.markdown("---")
            st.markdown(
                '<div class="card-title">Explanation</div>',
                unsafe_allow_html=True,
            )

        if modality == "Unimodal — Image CNN":
            st.markdown(_cnn_explanation_text(cnn_probs))

        if meta_shap is not None:
            pred_name = DISPLAY_NAMES[CLASS_NAMES[int(meta_probs.argmax())]]
            st.caption(
                "The chart shows which patient features contributed most to the predicted class. "
                "Green bars push the model **toward** the prediction; red bars push it **away**."
            )
            _render_shap_chart(meta_shap, pred_name)

        if cam_overlay is not None:
            st.markdown("---")
            st.markdown(
                '<div class="card-title">Grad-CAM Attention Map</div>',
                unsafe_allow_html=True,
            )
            st.caption(
                "Highlights the image regions that most influenced the CNN's prediction. "
                "Warm colours (red/yellow) indicate high attention; cool colours (blue) indicate low attention."
            )
            gc1, gc2 = st.columns(2)
            with gc1:
                st.markdown(
                    "<small style='color:#8b949e; font-weight:600; text-transform:uppercase; "
                    "letter-spacing:.05em'>Original</small>",
                    unsafe_allow_html=True,
                )
                st.image(cam_orig, use_container_width=True)
            with gc2:
                st.markdown(
                    "<small style='color:#8b949e; font-weight:600; text-transform:uppercase; "
                    "letter-spacing:.05em'>Grad-CAM overlay</small>",
                    unsafe_allow_html=True,
                )
                st.image(cam_overlay, use_container_width=True)
