"""
Comprehensive handcrafted image feature extraction for Pareto-based selection.

Produces exactly 118 scalar features per image:
  - GLCM (8):          Mean, Variance, Entropy, Contrast, Homogeneity, ASM, Dissimilarity, Correlation
  - HOG (3):           Mean, Std, Energy (summarised from full vector)
  - Gabor (24):        Energy + Magnitude x 3 frequencies x 4 orientations
  - LBP (4):           Mean, Std, Entropy, Energy (summarised from histogram)
  - Color (43):        4 stats x 3 channels x 3 spaces (RGB/HSV/LAB)
                       + 2 hist stats x 3 channels + intra-variance
  - Shape/Border (15): Area, Perimeter, Compactness, Circularity, Aspect Ratio,
                       Extent, Asymmetry H/V, 7 Hu Moments
  - Wavelet/DWT (12):  Energy + Std x 3 subbands (LH, HL, HH) x 2 levels
  - FFT (5):           Mean, Std, Energy, Skewness, Kurtosis
  - Edge (4):          Canny density, Sobel mean, Sobel std, Sobel energy
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Union

import numpy as np
from PIL import Image
from scipy.stats import kurtosis, skew

try:
    import pywt

    _HAS_PYWT = True
except ImportError:
    _HAS_PYWT = False

try:
    from skimage import color as skcolor
    from skimage import filters, measure, morphology, transform
    from skimage.feature import (
        canny,
        graycomatrix,
        graycoprops,
        hog,
        local_binary_pattern,
    )

    _HAS_SKIMAGE = True
except ImportError:
    _HAS_SKIMAGE = False

DEFAULT_IMG_SIZE = 128
N_FEATURES = 118

GLCM_DISTANCES = [1]
GLCM_ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

GABOR_FREQUENCIES = [0.1, 0.25, 0.4]
GABOR_ORIENTATIONS = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

LBP_RADIUS = 1
LBP_N_POINTS = 8

HOG_ORIENTATIONS = 9
HOG_PPC = (16, 16)
HOG_CPB = (2, 2)


def _load_and_resize(img_path: Union[str, Path], size: int) -> np.ndarray:
    path = Path(img_path).expanduser()
    img = Image.open(path).convert("RGB")
    arr = np.array(img)
    if size and (arr.shape[0] != size or arr.shape[1] != size):
        if _HAS_SKIMAGE:
            arr = transform.resize(
                arr, (size, size), preserve_range=True, anti_aliasing=True
            ).astype(np.uint8)
        else:
            img = img.resize((size, size), Image.Resampling.LANCZOS)
            arr = np.array(img)
    return arr


def _to_grayscale_uint8(rgb: np.ndarray) -> np.ndarray:
    gray = (
        rgb[:, :, 0].astype(np.float64) * 0.299
        + rgb[:, :, 1].astype(np.float64) * 0.587
        + rgb[:, :, 2].astype(np.float64) * 0.114
    )
    return np.clip(gray, 0, 255).astype(np.uint8)


def _safe_entropy(hist: np.ndarray) -> float:
    h = hist[hist > 0]
    return float(-np.sum(h * np.log2(h)))


# ── GLCM (8) ──────────────────────────────────────────────────────────


def _glcm_features(gray_u8: np.ndarray) -> np.ndarray:
    if not _HAS_SKIMAGE:
        return np.zeros(8, dtype=np.float32)
    try:
        glcm = graycomatrix(
            gray_u8,
            distances=GLCM_DISTANCES,
            angles=GLCM_ANGLES,
            levels=256,
            symmetric=True,
            normed=True,
        )
        contrast = graycoprops(glcm, "contrast").mean()
        homogeneity = graycoprops(glcm, "homogeneity").mean()
        asm = graycoprops(glcm, "ASM").mean()
        dissimilarity = graycoprops(glcm, "dissimilarity").mean()
        correlation = graycoprops(glcm, "correlation").mean()

        i_idx = np.arange(256, dtype=np.float64).reshape(-1, 1)
        means, variances, entropies = [], [], []
        for a in range(len(GLCM_ANGLES)):
            P = glcm[:, :, 0, a].astype(np.float64)
            mu = float(np.sum(i_idx * P))
            var = float(np.sum(((i_idx - mu) ** 2) * P))
            P_pos = P[P > 0]
            ent = float(-np.sum(P_pos * np.log2(P_pos)))
            means.append(mu)
            variances.append(var)
            entropies.append(ent)

        return np.array(
            [
                np.mean(means),
                np.mean(variances),
                np.mean(entropies),
                contrast,
                homogeneity,
                asm,
                dissimilarity,
                correlation,
            ],
            dtype=np.float32,
        )
    except Exception:
        return np.zeros(8, dtype=np.float32)


# ── HOG (3) ────────────────────────────────────────────────────────────


def _hog_features(gray_u8: np.ndarray) -> np.ndarray:
    if not _HAS_SKIMAGE:
        return np.zeros(3, dtype=np.float32)
    try:
        h = hog(
            gray_u8.astype(np.float64) / 255.0,
            orientations=HOG_ORIENTATIONS,
            pixels_per_cell=HOG_PPC,
            cells_per_block=HOG_CPB,
            block_norm="L2-Hys",
            feature_vector=True,
        )
        return np.array(
            [np.mean(h), np.std(h), np.sum(h**2)], dtype=np.float32
        )
    except Exception:
        return np.zeros(3, dtype=np.float32)


# ── Gabor (24) ─────────────────────────────────────────────────────────


def _gabor_features(gray_u8: np.ndarray) -> np.ndarray:
    n = len(GABOR_FREQUENCIES) * len(GABOR_ORIENTATIONS) * 2
    if not _HAS_SKIMAGE:
        return np.zeros(n, dtype=np.float32)
    gray_f = gray_u8.astype(np.float64) / 255.0
    feats = []
    for freq in GABOR_FREQUENCIES:
        for theta in GABOR_ORIENTATIONS:
            real, imag = filters.gabor(gray_f, frequency=freq, theta=theta)
            mag = np.sqrt(real**2 + imag**2)
            feats.extend([np.mean(mag**2), np.mean(mag)])
    return np.array(feats, dtype=np.float32)


# ── LBP (4) ───────────────────────────────────────────────────────────


def _lbp_features(gray_u8: np.ndarray) -> np.ndarray:
    if not _HAS_SKIMAGE:
        return np.zeros(4, dtype=np.float32)
    try:
        lbp = local_binary_pattern(gray_u8, LBP_N_POINTS, LBP_RADIUS, method="uniform")
        n_bins = LBP_N_POINTS + 2
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
        return np.array(
            [np.mean(hist), np.std(hist), _safe_entropy(hist), np.sum(hist**2)],
            dtype=np.float32,
        )
    except Exception:
        return np.zeros(4, dtype=np.float32)


# ── Color (43) ─────────────────────────────────────────────────────────


def _color_features(rgb: np.ndarray) -> np.ndarray:
    images = {"rgb": rgb.astype(np.float64)}
    if _HAS_SKIMAGE:
        images["hsv"] = skcolor.rgb2hsv(rgb.astype(np.float64) / 255.0)
        images["lab"] = skcolor.rgb2lab(rgb)
    else:
        images["hsv"] = rgb.astype(np.float64) / 255.0
        images["lab"] = rgb.astype(np.float64)

    feats = []

    # 4 stats x 3 channels x 3 spaces = 36
    for space in ["rgb", "hsv", "lab"]:
        arr = images[space]
        for c in range(3):
            ch = arr[:, :, c].ravel()
            feats.extend(
                [np.mean(ch), np.std(ch), float(skew(ch)), float(kurtosis(ch))]
            )

    # 2 hist stats (entropy, energy) x 3 RGB channels = 6
    for c in range(3):
        h, _ = np.histogram(
            rgb[:, :, c].ravel(), bins=32, range=(0, 256), density=True
        )
        feats.append(_safe_entropy(h))
        feats.append(float(np.sum(h**2)))

    # Intra-variance: mean per-pixel variance across RGB channels = 1
    pixel_var = np.var(rgb.astype(np.float64), axis=2)
    feats.append(float(np.mean(pixel_var)))

    return np.array(feats, dtype=np.float32)


# ── Shape / Border (15) ───────────────────────────────────────────────


def _segment_lesion(gray_u8: np.ndarray) -> np.ndarray:
    if not _HAS_SKIMAGE:
        return gray_u8 < 128
    try:
        blurred = filters.gaussian(gray_u8.astype(np.float64), sigma=2)
        thresh = filters.threshold_otsu(blurred)
        mask = blurred < thresh
        mask = morphology.remove_small_objects(mask, min_size=64)
        mask = morphology.remove_small_holes(mask, area_threshold=64)
        if mask.sum() == 0:
            mask = blurred < np.mean(blurred)
        return mask
    except Exception:
        return gray_u8 < 128


def _shape_features(gray_u8: np.ndarray) -> np.ndarray:
    try:
        mask = _segment_lesion(gray_u8)
        labeled = measure.label(mask.astype(np.uint8))
        regions = measure.regionprops(labeled)
        if not regions:
            return np.zeros(15, dtype=np.float32)

        r = max(regions, key=lambda x: x.area)

        area = float(r.area)
        perimeter = float(r.perimeter) if r.perimeter > 0 else 1e-6
        compactness = (4 * np.pi * area) / (perimeter**2)
        circularity = compactness
        minor = r.axis_minor_length if r.axis_minor_length > 0 else 1e-6
        aspect_ratio = r.axis_major_length / minor
        extent = float(r.extent)

        # Asymmetry: flip mask and compute overlap difference
        h, w = mask.shape
        cy, cx = int(r.centroid[0]), int(r.centroid[1])
        asym_h = float(np.sum(np.abs(
            mask[:cy, :].astype(np.float64)
            - np.flipud(mask[cy: 2 * cy, :].astype(np.float64))[: cy, :]
        ))) / max(area, 1)
        asym_v = float(np.sum(np.abs(
            mask[:, :cx].astype(np.float64)
            - np.fliplr(mask[:, cx: 2 * cx].astype(np.float64))[:, : cx]
        ))) / max(area, 1)

        # Hu moments (7)
        mu = measure.moments_central(mask.astype(np.float64))
        nu = measure.moments_normalized(mu)
        hu = measure.moments_hu(nu)
        hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)

        return np.array(
            [area, perimeter, compactness, circularity, aspect_ratio, extent,
             asym_h, asym_v, *hu_log],
            dtype=np.float32,
        )
    except Exception:
        return np.zeros(15, dtype=np.float32)


# ── Wavelet / DWT (12) ────────────────────────────────────────────────


def _wavelet_features(gray_u8: np.ndarray) -> np.ndarray:
    if not _HAS_PYWT:
        return np.zeros(12, dtype=np.float32)
    try:
        gray_f = gray_u8.astype(np.float64) / 255.0
        feats = []
        current = gray_f
        for _ in range(2):
            coeffs = pywt.dwt2(current, "haar")
            LL, (LH, HL, HH) = coeffs
            for sub in [LH, HL, HH]:
                feats.append(float(np.sum(sub**2)))
                feats.append(float(np.std(sub)))
            current = LL
        return np.array(feats, dtype=np.float32)
    except Exception:
        return np.zeros(12, dtype=np.float32)


# ── FFT (5) ───────────────────────────────────────────────────────────


def _fft_features(gray_u8: np.ndarray) -> np.ndarray:
    try:
        gray_f = gray_u8.astype(np.float64) / 255.0
        f_transform = np.fft.fft2(gray_f)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift).ravel()
        return np.array(
            [
                np.mean(magnitude),
                np.std(magnitude),
                np.sum(magnitude**2),
                float(skew(magnitude)),
                float(kurtosis(magnitude)),
            ],
            dtype=np.float32,
        )
    except Exception:
        return np.zeros(5, dtype=np.float32)


# ── Edge (4) ──────────────────────────────────────────────────────────


def _edge_features(gray_u8: np.ndarray) -> np.ndarray:
    if not _HAS_SKIMAGE:
        return np.zeros(4, dtype=np.float32)
    try:
        gray_f = gray_u8.astype(np.float64) / 255.0
        edges = canny(gray_f, sigma=1.0)
        canny_density = float(np.mean(edges))
        sobel_mag = filters.sobel(gray_f)
        sobel_mean = float(np.mean(sobel_mag))
        sobel_std = float(np.std(sobel_mag))
        sobel_energy = float(np.sum(sobel_mag**2))
        return np.array(
            [canny_density, sobel_mean, sobel_std, sobel_energy], dtype=np.float32
        )
    except Exception:
        return np.zeros(4, dtype=np.float32)


# ── Public API ─────────────────────────────────────────────────────────


def get_feature_names(img_size: int = DEFAULT_IMG_SIZE) -> List[str]:
    names: List[str] = []

    # GLCM (8)
    for prop in ["mean", "variance", "entropy", "contrast", "homogeneity",
                 "ASM", "dissimilarity", "correlation"]:
        names.append(f"glcm_{prop}")

    # HOG (3)
    names.extend(["hog_mean", "hog_std", "hog_energy"])

    # Gabor (24)
    for fi in range(len(GABOR_FREQUENCIES)):
        for oi in range(len(GABOR_ORIENTATIONS)):
            names.append(f"gabor_energy_f{fi}_o{oi}")
            names.append(f"gabor_magnitude_f{fi}_o{oi}")

    # LBP (4)
    names.extend(["lbp_mean", "lbp_std", "lbp_entropy", "lbp_energy"])

    # Color (43)
    for space in ["rgb", "hsv", "lab"]:
        channels = {"rgb": "rgb", "hsv": "hsv", "lab": "lab"}[space]
        for ch in channels:
            for stat in ["mean", "std", "skew", "kurt"]:
                names.append(f"color_{space}_{ch}_{stat}")
    for ch in "rgb":
        names.append(f"hist_{ch}_entropy")
        names.append(f"hist_{ch}_energy")
    names.append("color_intra_variance")

    # Shape (15)
    names.extend([
        "shape_area", "shape_perimeter", "shape_compactness", "shape_circularity",
        "shape_aspect_ratio", "shape_extent", "shape_asymmetry_h", "shape_asymmetry_v",
    ])
    for i in range(7):
        names.append(f"shape_hu_{i}")

    # Wavelet (12)
    for level in [1, 2]:
        for sub in ["LH", "HL", "HH"]:
            names.append(f"dwt_L{level}_{sub}_energy")
            names.append(f"dwt_L{level}_{sub}_std")

    # FFT (5)
    names.extend(["fft_mean", "fft_std", "fft_energy", "fft_skew", "fft_kurt"])

    # Edge (4)
    names.extend(["edge_canny_density", "edge_sobel_mean", "edge_sobel_std", "edge_sobel_energy"])

    return names


def feature_dim(img_size: int = DEFAULT_IMG_SIZE) -> int:
    return N_FEATURES


def extract_features_from_array(
    rgb: np.ndarray, img_size: int = DEFAULT_IMG_SIZE
) -> np.ndarray:
    if rgb.shape[0] != img_size or rgb.shape[1] != img_size:
        if _HAS_SKIMAGE:
            rgb = transform.resize(
                rgb, (img_size, img_size), preserve_range=True, anti_aliasing=True
            ).astype(np.uint8)
        else:
            pil = Image.fromarray(rgb)
            pil = pil.resize((img_size, img_size), Image.Resampling.LANCZOS)
            rgb = np.array(pil)

    gray = _to_grayscale_uint8(rgb)

    parts = [
        _glcm_features(gray),        # 8
        _hog_features(gray),          # 3
        _gabor_features(gray),        # 24
        _lbp_features(gray),          # 4
        _color_features(rgb),         # 43
        _shape_features(gray),        # 15
        _wavelet_features(gray),      # 12
        _fft_features(gray),          # 5
        _edge_features(gray),         # 4
    ]
    return np.concatenate(parts).astype(np.float32)


def extract_features_from_path(
    img_path: Union[str, Path], img_size: int = DEFAULT_IMG_SIZE
) -> np.ndarray:
    rgb = _load_and_resize(img_path, img_size)
    return extract_features_from_array(rgb, img_size=img_size)
