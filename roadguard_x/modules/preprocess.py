"""
Frame preprocessing: Gaussian blur, CLAHE, and gamma correction.

Improves edge stability and robustness under varying illumination.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def gaussian_blur(
    image: np.ndarray, kernel_size: Tuple[int, int] = (5, 5), sigma: float = 0.0
) -> np.ndarray:
    """
    Apply Gaussian blur to reduce sensor noise before edge/segmentation steps.

    Args:
        image: BGR uint8 image.
        kernel_size: Odd kernel dimensions (width, height).
        sigma: If 0, computed from kernel size.

    Returns:
        Blurred BGR image.
    """
    kx, ky = kernel_size
    if kx % 2 == 0 or ky % 2 == 0:
        raise ValueError("Gaussian kernel dimensions must be odd.")
    return cv2.GaussianBlur(image, kernel_size, sigma)


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) on luminance.

    Operates in LAB space to avoid color shifts.

    Args:
        image: BGR uint8 image.
        clip_limit: CLAHE clip limit.
        tile_grid_size: Tile grid for local histogram equalization.

    Returns:
        Equalized BGR image.
    """
    try:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        l_eq = clahe.apply(l)
        merged = cv2.merge([l_eq, a, b])
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    except Exception:
        return image


def gamma_correction(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """
    Apply gamma correction using a lookup table (efficient, monotonic).

    Args:
        image: BGR uint8 image.
        gamma: Gamma value (>1 darkens, <1 brightens mid-tones).

    Returns:
        Gamma-corrected BGR image.
    """
    if gamma <= 0:
        raise ValueError("gamma must be positive.")
    inv_gamma = 1.0 / gamma
    table = (np.linspace(0, 1, 256) ** inv_gamma * 255).astype("uint8")
    return cv2.LUT(image, table)


def preprocess_frame(
    image: np.ndarray,
    *,
    blur_kernel: Tuple[int, int] = (5, 5),
    blur_sigma: float = 0.0,
    clahe_clip: float = 2.0,
    gamma: float = 1.0,
    use_clahe: bool = True,
    use_blur: bool = True,
) -> np.ndarray:
    """
    Full preprocessing pipeline: optional blur → CLAHE → gamma.

    Args:
        image: Raw BGR frame.
        blur_kernel: Gaussian kernel size.
        blur_sigma: Gaussian sigma.
        clahe_clip: CLAHE clip limit.
        gamma: Gamma factor (1.0 = identity).
        use_clahe: Whether to run CLAHE.
        use_blur: Whether to run Gaussian blur first.

    Returns:
        Preprocessed BGR frame.
    """
    out = image.copy()
    if use_blur:
        out = gaussian_blur(out, blur_kernel, blur_sigma)
    if use_clahe:
        out = apply_clahe(out, clip_limit=clahe_clip)
    if abs(gamma - 1.0) > 1e-6:
        out = gamma_correction(out, gamma)
    return out
