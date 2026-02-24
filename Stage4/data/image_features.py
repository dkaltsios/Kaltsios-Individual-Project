"""
Handcrafted image feature extraction (no CNN).
Color + texture features for skin lesion images.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

try:
    from skimage import color, feature, transform
    _HAS_SKIMAGE = True
except ImportError:
    _HAS_SKIMAGE = False


# Default size for consistent feature extraction
DEFAULT_IMG_SIZE = 128
# GLCM distances and angles for texture
GLCM_DISTANCES = [1]
GLCM_ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]


def _load_and_resize(img_path: Union[str, Path], size: int) -> np.ndarray:
    """Load image, resize, return RGB array (H, W, 3) uint8."""
    path = Path(img_path).expanduser()
    img = Image.open(path).convert("RGB")
    arr = np.array(img)
    if size and (arr.shape[0] != size or arr.shape[1] != size):
        if _HAS_SKIMAGE:
            arr = transform.resize(arr, (size, size), preserve_range=True, anti_aliasing=True).astype(np.uint8)
        else:
            img = img.resize((size, size), Image.Resampling.LANCZOS)
            arr = np.array(img)
    return arr


def _color_features(rgb: np.ndarray) -> np.ndarray:
    """RGB and HSV channel statistics. Shape (18,) or similar."""
    feats = []
    # RGB mean, std per channel (6)
    for c in range(3):
        feats.append(np.mean(rgb[:, :, c]))
        feats.append(np.std(rgb[:, :, c]))
    # HSV mean, std (6) - useful for skin
    if _HAS_SKIMAGE:
        hsv = color.rgb2hsv(rgb.astype(np.float32) / 255.0)
        for c in range(3):
            feats.append(float(np.mean(hsv[:, :, c])))
            feats.append(float(np.std(hsv[:, :, c])))
    else:
        # Fallback: use R,G,B again
        for c in range(3):
            feats.append(np.mean(rgb[:, :, c]) / 255.0)
            feats.append(np.std(rgb[:, :, c]) / 255.0)
    return np.array(feats, dtype=np.float32)


def _texture_features(gray: np.ndarray) -> np.ndarray:
    """Gray-level co-occurrence matrix stats. Shape (4,) or (4*angles)."""
    if not _HAS_SKIMAGE:
        return np.zeros(4, dtype=np.float32)
    if gray.max() <= 1.0:
        gray = (gray * 255).astype(np.uint8)
    else:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    gray_u = gray
    try:
        glcm = feature.graycomatrix(
            gray_u,
            distances=GLCM_DISTANCES,
            angles=GLCM_ANGLES,
            levels=256,
            symmetric=True,
            normed=True,
        )
        contrast = feature.graycoprops(glcm, "contrast").ravel()
        homogeneity = feature.graycoprops(glcm, "homogeneity").ravel()
        energy = feature.graycoprops(glcm, "energy").ravel()
        correlation = feature.graycoprops(glcm, "correlation").ravel()
        # Mean over angles
        return np.array(
            [contrast.mean(), homogeneity.mean(), energy.mean(), correlation.mean()],
            dtype=np.float32,
        )
    except Exception:
        return np.zeros(4, dtype=np.float32)


def _histogram_features(rgb: np.ndarray, bins: int = 16) -> np.ndarray:
    """Per-channel histogram (normalized). Shape (bins*3,)."""
    feats = []
    for c in range(3):
        h, _ = np.histogram(rgb[:, :, c].ravel(), bins=bins, range=(0, 256), density=True)
        feats.append(h.astype(np.float32))
    return np.concatenate(feats, axis=0)


def extract_features_from_array(rgb: np.ndarray, img_size: int = DEFAULT_IMG_SIZE) -> np.ndarray:
    """
    Extract handcrafted features from an RGB image array (H, W, 3).
    Optionally resizes to img_size for consistency.
    """
    if rgb.shape[0] != img_size or rgb.shape[1] != img_size:
        if _HAS_SKIMAGE:
            rgb = transform.resize(rgb, (img_size, img_size), preserve_range=True, anti_aliasing=True).astype(np.uint8)
        else:
            from PIL import Image
            pil = Image.fromarray(rgb)
            pil = pil.resize((img_size, img_size), Image.Resampling.LANCZOS)
            rgb = np.array(pil)

    gray = rgb[:, :, 0] * 0.299 + rgb[:, :, 1] * 0.587 + rgb[:, :, 2] * 0.114
    if _HAS_SKIMAGE and gray.max() <= 1.0:
        gray = (gray * 255).astype(np.uint8)

    color_f = _color_features(rgb)
    texture_f = _texture_features(gray)
    hist_f = _histogram_features(rgb, bins=16)
    return np.concatenate([color_f, texture_f, hist_f], axis=0).astype(np.float32)


def extract_features_from_path(img_path: Union[str, Path], img_size: int = DEFAULT_IMG_SIZE) -> np.ndarray:
    """Load image from path and extract handcrafted feature vector."""
    rgb = _load_and_resize(img_path, img_size)
    return extract_features_from_array(rgb, img_size=img_size)
