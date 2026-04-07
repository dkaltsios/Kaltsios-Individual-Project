"""
Handcrafted image feature extraction for Stage 4.

Produces a 13-dimensional feature vector per image:
  1-8.  GLCM: Mean, Variance, Entropy, Contrast, Homogeneity, ASM, Dissimilarity, Correlation
  9.    Gabor Energy
  10.   Gabor Magnitude
  11.   HOG
  12.   Color (mean RGB intensity)
  13.   LBP
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

try:
    from skimage import feature, filters, transform
    from skimage.feature import graycomatrix, graycoprops, local_binary_pattern, hog

    _HAS_SKIMAGE = True
except ImportError:
    _HAS_SKIMAGE = False

DEFAULT_IMG_SIZE = 128

GLCM_DISTANCES = [1]
GLCM_ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

GABOR_FREQUENCIES = [0.1, 0.25, 0.4]
GABOR_ORIENTATIONS = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

LBP_RADIUS = 1
LBP_N_POINTS = 8 * LBP_RADIUS
LBP_METHOD = "uniform"

HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (16, 16)
HOG_CELLS_PER_BLOCK = (2, 2)

N_FEATURES = 13


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


def _glcm_features(gray_u8: np.ndarray) -> np.ndarray:
    """
    8 GLCM features averaged over all distance/angle combinations:
    Mean, Variance, Entropy, Contrast, Homogeneity, ASM, Dissimilarity, Correlation.
    """
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
        for d in range(len(GLCM_DISTANCES)):
            for a in range(len(GLCM_ANGLES)):
                P = glcm[:, :, d, a].astype(np.float64)
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


def _gabor_features(gray_u8: np.ndarray) -> np.ndarray:
    """
    2 Gabor features: mean energy and mean magnitude across all
    (frequency, orientation) filter responses.
    """
    if not _HAS_SKIMAGE:
        return np.zeros(2, dtype=np.float32)

    gray_f = gray_u8.astype(np.float64) / 255.0
    energies, magnitudes = [], []
    for freq in GABOR_FREQUENCIES:
        for theta in GABOR_ORIENTATIONS:
            real, imag = filters.gabor(gray_f, frequency=freq, theta=theta)
            mag = np.sqrt(real**2 + imag**2)
            energies.append(np.mean(mag**2))
            magnitudes.append(np.mean(mag))

    return np.array(
        [np.mean(energies), np.mean(magnitudes)], dtype=np.float32
    )


def _hog_feature(gray_u8: np.ndarray) -> float:
    """Single HOG scalar: mean gradient magnitude across all cells."""
    if not _HAS_SKIMAGE:
        return 0.0
    try:
        h = hog(
            gray_u8.astype(np.float64) / 255.0,
            orientations=HOG_ORIENTATIONS,
            pixels_per_cell=HOG_PIXELS_PER_CELL,
            cells_per_block=HOG_CELLS_PER_BLOCK,
            block_norm="L2-Hys",
            feature_vector=True,
        )
        return float(np.mean(h))
    except Exception:
        return 0.0


def _color_feature(rgb: np.ndarray) -> float:
    """Mean RGB intensity across all pixels and channels."""
    return float(np.mean(rgb))


def _lbp_feature(gray_u8: np.ndarray) -> float:
    """Mean uniform-LBP value."""
    if not _HAS_SKIMAGE:
        return 0.0
    try:
        lbp = local_binary_pattern(gray_u8, LBP_N_POINTS, LBP_RADIUS, method=LBP_METHOD)
        return float(np.mean(lbp))
    except Exception:
        return 0.0


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

    glcm_f = _glcm_features(gray)          # (8,)
    gabor_f = _gabor_features(gray)         # (2,)
    hog_f = _hog_feature(gray)              # scalar
    color_f = _color_feature(rgb)           # scalar
    lbp_f = _lbp_feature(gray)             # scalar

    return np.array(
        [*glcm_f, *gabor_f, hog_f, color_f, lbp_f], dtype=np.float32
    )


def extract_features_from_path(
    img_path: Union[str, Path], img_size: int = DEFAULT_IMG_SIZE
) -> np.ndarray:
    rgb = _load_and_resize(img_path, img_size)
    return extract_features_from_array(rgb, img_size=img_size)
