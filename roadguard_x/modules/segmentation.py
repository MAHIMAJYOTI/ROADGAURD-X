"""
Scene segmentation using K-means on color features and morphological cleanup.

Optional watershed refinement on distance transform (marker-free approximation).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np


@dataclass
class SegmentationResult:
    """K-means label map and cleaned binary mask (largest foreground cluster)."""

    labels: np.ndarray
    mask: np.ndarray
    num_clusters: int


def kmeans_segment(
    image: np.ndarray,
    k: int = 4,
    max_iter: int = 20,
    epsilon: float = 0.5,
) -> SegmentationResult:
    """
    Segment image into k regions using K-means on LAB color samples.

    Args:
        image: BGR uint8 image.
        k: Number of clusters (3 or 4 typical).
        max_iter, epsilon: K-means termination criteria.

    Returns:
        SegmentationResult with per-pixel labels and a binary road-like mask.
    """
    h, w = image.shape[:2]
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    data = lab.reshape((-1, 3)).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, epsilon)
    _, labels, centers = cv2.kmeans(
        data,
        k,
        None,
        criteria,
        attempts=2,
        flags=cv2.KMEANS_PP_CENTERS,
    )
    labels_flat = labels.flatten().astype(np.int32)
    label_map = labels_flat.reshape((h, w))

    # Choose foreground as cluster with lowest average L* (often road asphalt darker than sky)
    # Fallback: median L per cluster
    l_channel = lab[:, :, 0].reshape(-1)
    cluster_mean_l = []
    for c in range(k):
        idx = labels_flat == c
        cluster_mean_l.append(float(np.mean(l_channel[idx])) if np.any(idx) else 128.0)
    road_cluster = int(np.argmin(cluster_mean_l))

    mask = (label_map == road_cluster).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    return SegmentationResult(labels=label_map, mask=mask, num_clusters=k)


def watershed_refine(
    binary_mask: np.ndarray,
    *,
    min_dist: int = 5,
    fg_threshold: float = 0.35,
) -> np.ndarray:
    """
    Optional watershed on distance transform to split touching blobs.

    Uses Otsu on distance to build sure foreground markers.

    Args:
        binary_mask: Single-channel uint8 (0/255).
        min_dist: Minimum distance for peak local maxima.
        fg_threshold: Fraction of max distance for sure FG.

    Returns:
        Label image (int32) from watershed, or zeros if refinement skipped.
    """
    if binary_mask.size == 0 or np.max(binary_mask) == 0:
        return np.zeros_like(binary_mask, dtype=np.int32)

    dist = cv2.distanceTransform((binary_mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    if dist.max() < 1e-6:
        return np.zeros_like(binary_mask, dtype=np.int32)

    _, sure_fg = cv2.threshold(
        dist, fg_threshold * dist.max(), 255, cv2.THRESH_BINARY
    )
    sure_fg = np.uint8(sure_fg)

    kernel = np.ones((3, 3), np.uint8)
    unknown = cv2.subtract(cv2.dilate(binary_mask, kernel), sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    bgr = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
    cv2.watershed(bgr, markers)
    return markers
