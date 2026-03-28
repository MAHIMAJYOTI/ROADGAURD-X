"""
Lane detection using ROI masking, Canny edges, and probabilistic Hough lines.

Computes lane center offset relative to image center (explainable drift signal).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class LaneResult:
    """Structured output for lane analysis."""

    left_lines: List[Tuple[int, int, int, int]]
    right_lines: List[Tuple[int, int, int, int]]
    center_offset_px: float
    center_offset_norm: float
    roi_mask: np.ndarray
    edges: np.ndarray


def _roi_trapezoid(
    width: int, height: int, top_width_ratio: float = 0.45, bottom_margin: int = 40
) -> np.ndarray:
    """Build a trapezoidal ROI mask (road region)."""
    mask = np.zeros((height, width), dtype=np.uint8)
    top_y = int(height * 0.58)
    bottom_y = height - bottom_margin
    top_w = int(width * top_width_ratio)
    top_left = (width // 2 - top_w // 2, top_y)
    top_right = (width // 2 + top_w // 2, top_y)
    bottom_left = (0, bottom_y)
    bottom_right = (width - 1, bottom_y)
    pts = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.int32)
    cv2.fillConvexPoly(mask, pts, 255)
    return mask


def detect_lanes(
    image: np.ndarray,
    *,
    canny_low: int = 50,
    canny_high: int = 150,
    hough_threshold: int = 40,
    min_line_length: int = 40,
    max_line_gap: int = 20,
) -> LaneResult:
    """
    Detect lane lines via Canny + Hough; estimate center offset.

    Lines are classified as left/right by slope relative to image center.

    Args:
        image: BGR image (typically preprocessed).
        canny_low, canny_high: Canny thresholds.
        hough_threshold: Accumulator threshold for HoughLinesP.
        min_line_length, max_line_gap: Line segment parameters.

    Returns:
        LaneResult with lines, offset, and intermediate maps.
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    roi = _roi_trapezoid(w, h)
    masked = cv2.bitwise_and(gray, gray, mask=roi)
    edges = cv2.Canny(masked, canny_low, canny_high)
    lines_p = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )

    left: List[Tuple[int, int, int, int]] = []
    right: List[Tuple[int, int, int, int]] = []
    cx = w / 2.0

    if lines_p is not None:
        for line in lines_p:
            x1, y1, x2, y2 = line[0].astype(float)
            dx = x2 - x1
            if abs(dx) < 1e-3:
                continue
            slope = (y2 - y1) / dx
            # Ignore near-horizontal (noise)
            if abs(slope) < 0.35:
                continue
            mid_x = (x1 + x2) / 2.0
            seg = (int(x1), int(y1), int(x2), int(y2))
            # In image coords, road lane lines: left side negative slope, right positive
            if slope < 0 and mid_x < cx:
                left.append(seg)
            elif slope > 0 and mid_x > cx:
                right.append(seg)

    # Average bottom x for left/right lane boundaries (near bottom of ROI)
    bottom_y = int(h * 0.92)

    def _avg_bottom_x(segs: List[Tuple[int, int, int, int]]) -> Optional[float]:
        xs = []
        for x1, y1, x2, y2 in segs:
            # Interpolate x at bottom_y if segment spans it
            ymin, ymax = min(y1, y2), max(y1, y2)
            if ymin <= bottom_y <= ymax and y2 != y1:
                t = (bottom_y - y1) / (y2 - y1)
                xi = x1 + t * (x2 - x1)
                xs.append(xi)
        if not xs:
            return None
        return float(np.mean(xs))

    lx = _avg_bottom_x(left)
    rx = _avg_bottom_x(right)

    if lx is not None and rx is not None:
        lane_center = (lx + rx) / 2.0
    elif lx is not None:
        lane_center = lx + w * 0.12
    elif rx is not None:
        lane_center = rx - w * 0.12
    else:
        lane_center = cx

    offset_px = lane_center - cx
    offset_norm = offset_px / max(w, 1)

    return LaneResult(
        left_lines=left,
        right_lines=right,
        center_offset_px=float(offset_px),
        center_offset_norm=float(offset_norm),
        roi_mask=roi,
        edges=edges,
    )
