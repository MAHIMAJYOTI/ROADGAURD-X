"""
Spatial and temporal feature extraction for risk modeling.

Spatial: lane offset, object count, average object area, edge density, brightness.
Temporal: average speed, motion variance, direction consistency (from track buffer).
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import cv2
import numpy as np
from scipy.stats import circstd

from modules.lane import LaneResult
from modules.tracker import FrameTracks, ObjectTracker, TrackState

# Fixed order must match ml/train.py and risk_model.py
# scene_complexity_index (SCI): normalized 0–1 from edge density, object count, motion variance
FEATURE_NAMES: Tuple[str, ...] = (
    "lane_offset_norm",
    "object_count",
    "avg_object_area",
    "edge_density",
    "brightness",
    "avg_speed",
    "motion_variance",
    "direction_consistency",
    "scene_complexity_index",
)

# Upper-bound scale for SCI normalization (heuristic caps on raw components)
_SCI_DENOM = 0.5 * 1.0 + 0.3 * 25.0 + 0.2 * 100.0


def _edge_density(lane: LaneResult) -> float:
    roi_pixels = np.sum(lane.roi_mask > 0) + 1e-6
    edge_pixels = np.sum(lane.edges > 0)
    return float(edge_pixels / roi_pixels)


def _brightness(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray) / 255.0)


def compute_scene_complexity_index(
    edge_density: float, object_count: float, motion_variance: float
) -> float:
    """
    Scene Complexity Index (SCI): weighted mix, normalized to [0, 1].

    SCI = 0.5 * edge_density + 0.3 * object_count + 0.2 * motion_variance (raw),
    then divided by a fixed scale so typical ranges map to [0, 1].
    """
    raw = (
        0.5 * float(edge_density)
        + 0.3 * float(object_count)
        + 0.2 * float(motion_variance)
    )
    sci = float(raw / _SCI_DENOM)
    # Strict [0, 1] clamp (defensive against numerical edge cases)
    return float(max(0.0, min(1.0, sci)))


def compute_scene_stability(motion_variance: float) -> float:
    """Higher when motion variance is low (stable scene). Range (0, 1]."""
    return float(1.0 / (1.0 + float(motion_variance)))


def scene_complexity_label(sci: float) -> str:
    """HUD / human-readable bucket for SCI."""
    if sci < 1.0 / 3.0:
        return "LOW"
    if sci < 2.0 / 3.0:
        return "MEDIUM"
    return "HIGH"


def _circular_mean_sin_cos(angles: List[float]) -> Tuple[float, float]:
    if not angles:
        return 0.0, 0.0
    s = np.mean(np.sin(angles))
    c = np.mean(np.cos(angles))
    return float(s), float(c)


def direction_consistency_from_tracks(
    tracker: ObjectTracker, tracks: Sequence[TrackState]
) -> float:
    """
    Temporal direction stability: high when motion headings are steady (uses scipy circstd).

    Combines short history per track with mean resultant length for the current frame.
    """
    angles: List[float] = []
    for t in tracks:
        if t.speed < 0.5:
            continue
        dh = tracker.get_direction_history(t.track_id)
        if len(dh) >= 3:
            angles.extend(list(dh)[-8:])
        else:
            angles.append(t.direction_rad)

    if len(angles) < 2:
        return 1.0
    arr = np.asarray(angles, dtype=np.float64)
    arr = (arr + np.pi) % (2.0 * np.pi)
    dispersion = float(circstd(arr, high=2.0 * np.pi, low=0.0))
    s, c = _circular_mean_sin_cos(list(arr))
    r = float(np.hypot(s, c))
    # Blend alignment (r) with inverse dispersion
    consistency = 0.5 * r + 0.5 * (1.0 / (1.0 + dispersion))
    return float(np.clip(consistency, 0.0, 1.0))


def extract_spatial(
    image: np.ndarray,
    lane: LaneResult,
    tracks: Sequence[TrackState],
) -> Tuple[np.ndarray, dict]:
    """
    Extract spatial features from current frame and lane result.

    Returns:
        (8,) vector slice matching FEATURE_NAMES[0:5] extended with temporal zeros
        Actually we return full 8-vector by combining - see extract_combined.
    """
    n = len(tracks)
    if n == 0:
        avg_area = 0.0
    else:
        avg_area = float(np.mean([t.area for t in tracks]))

    spatial_dict = {
        "lane_offset_norm": abs(lane.center_offset_norm),
        "object_count": float(n),
        "avg_object_area": avg_area,
        "edge_density": _edge_density(lane),
        "brightness": _brightness(image),
    }
    return spatial_dict


def temporal_from_tracker(tracker: ObjectTracker, current_tracks: Sequence[TrackState]) -> dict:
    """
    Temporal statistics from rolling buffer and current tracks.
    """
    speeds: List[float] = [t.speed for t in current_tracks]
    avg_speed = float(np.mean(speeds)) if speeds else 0.0

    # Variance of speeds across buffer (aggregated per frame)
    frame_speeds: List[float] = []
    for ft in tracker.frame_buffer:
        if not ft.tracks:
            continue
        frame_speeds.append(float(np.mean([x.speed for x in ft.tracks])))
    motion_var = float(np.var(frame_speeds)) if len(frame_speeds) > 1 else 0.0

    dir_cons = direction_consistency_from_tracks(tracker, current_tracks)

    return {
        "avg_speed": avg_speed,
        "motion_variance": motion_var,
        "direction_consistency": dir_cons,
    }


def extract_combined(
    image: np.ndarray,
    lane: LaneResult,
    frame_tracks: FrameTracks,
    tracker: ObjectTracker,
) -> Tuple[np.ndarray, dict]:
    """
    Build full feature vector (9,) including SCI and a dict for explainability.

    Returns:
        features: shape (9,) float32
        feature_dict: name -> value (includes scene_stability for reporting)
    """
    sp = extract_spatial(image, lane, frame_tracks.tracks)
    tm = temporal_from_tracker(tracker, frame_tracks.tracks)
    feature_dict = {**sp, **tm}
    sci = compute_scene_complexity_index(
        feature_dict["edge_density"],
        feature_dict["object_count"],
        feature_dict["motion_variance"],
    )
    feature_dict["scene_complexity_index"] = max(0.0, min(1.0, float(sci)))
    feature_dict["scene_stability"] = compute_scene_stability(
        feature_dict["motion_variance"]
    )
    vec = np.array([feature_dict[name] for name in FEATURE_NAMES], dtype=np.float32)
    return vec, feature_dict
