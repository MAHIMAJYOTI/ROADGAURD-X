"""
Rule-augmented explanations: translate features + model output into human reasons.

Returns structured dict with risk, confidence, primary_cause, reasons, and RF contributions.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

# Thresholds tuned for normalized / scale-aware features (heuristic, explainable)
THRESH_LANE_DRIFT = 0.08
THRESH_HIGH_OBJECTS = 12.0
THRESH_LOW_VISIBILITY = 0.28
THRESH_RAPID_APPROACH = 18.0
THRESH_HIGH_EDGE = 0.42
THRESH_CHAOTIC_MOTION = 0.35

# Canonical short names for reports / HUD (aligned with FEATURE_NAMES)
FEATURE_SHORT_NAME: Dict[str, str] = {
    "lane_offset_norm": "lane_offset",
    "object_count": "object_density",
    "avg_object_area": "avg_object_area",
    "edge_density": "edge_density",
    "brightness": "brightness",
    "avg_speed": "avg_speed",
    "motion_variance": "motion_variance",
    "direction_consistency": "direction_consistency",
    "scene_complexity_index": "scene_complexity",
}

# Lower number = higher dominance for primary_cause selection
_CAUSE_PRIORITY: Dict[str, int] = {
    "lane_drift": 1,
    "rapid_approach": 2,
    "high_object_density": 3,
    "low_visibility": 4,
    "high_edge_density": 5,
    "unstable_motion": 6,
    "composite_risk": 7,
    "moderate_load": 8,
}

_REASON_TO_CODE: Dict[str, str] = {
    "Lane drift detected": "lane_drift",
    "Sustained lane departure from center": "lane_drift",
    "High object density": "high_object_density",
    "Low visibility": "low_visibility",
    "Rapid approaching object": "rapid_approach",
    "Cluttered scene (high edge density)": "high_edge_density",
    "Unstable motion patterns": "unstable_motion",
    "Elevated composite risk score": "composite_risk",
    "Moderate environmental load": "moderate_load",
}


def short_feature_name(full_name: str) -> str:
    """Map internal feature key to concise label for UI and JSON."""
    return FEATURE_SHORT_NAME.get(full_name, full_name)


def _feature_contributions_dict(
    top_contributions: Sequence[Tuple[str, float]],
) -> Dict[str, float]:
    """Top-3 RF importances with short keys, rounded to 2 decimals."""
    out: Dict[str, float] = {}
    for name, val in list(top_contributions)[:3]:
        key = short_feature_name(name)
        out[key] = round(float(val), 2)
    return out


def _pick_primary_cause(reasons: List[str]) -> str:
    """Choose dominant cause from triggered reasons using priority ordering."""
    if not reasons:
        return "composite_risk"
    best: Optional[tuple[int, str]] = None
    for r in reasons:
        code = _REASON_TO_CODE.get(r, "composite_risk")
        pr = _CAUSE_PRIORITY.get(code, 99)
        if best is None or pr < best[0]:
            best = (pr, code)
    return best[1] if best else "composite_risk"


def build_explanation(
    risk_label: str,
    feature_dict: Mapping[str, Any],
    confidence: float,
    top_contributions: Sequence[Tuple[str, float]],
    extra: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Produce explainable reasons from feature magnitudes and temporal drift context.

    Args:
        risk_label: Model output LOW | MEDIUM | HIGH.
        feature_dict: Scalar features (including SCI / stability for display if needed).
        confidence: Probability of the predicted class from predict_proba.
        top_contributions: Top-3 (feature_name, rf_importance) from the forest.
        extra: Optional keys, e.g. lane_drift_event (bool) for sustained temporal drift.

    Returns:
        Dict including risk, confidence, primary_cause, reasons, feature_contributions.
    """
    reasons: List[str] = []
    ex = dict(extra or {})

    lane_off = float(feature_dict.get("lane_offset_norm", 0.0))
    if abs(lane_off) > THRESH_LANE_DRIFT:
        reasons.append("Lane drift detected")

    if ex.get("lane_drift_event"):
        reasons.append("Sustained lane departure from center")

    nobj = float(feature_dict.get("object_count", 0.0))
    if nobj > THRESH_HIGH_OBJECTS:
        reasons.append("High object density")

    bright = float(feature_dict.get("brightness", 0.5))
    if bright < THRESH_LOW_VISIBILITY:
        reasons.append("Low visibility")

    avg_sp = float(feature_dict.get("avg_speed", 0.0))
    if avg_sp > THRESH_RAPID_APPROACH:
        reasons.append("Rapid approaching object")

    edge_d = float(feature_dict.get("edge_density", 0.0))
    if edge_d > THRESH_HIGH_EDGE:
        reasons.append("Cluttered scene (high edge density)")

    mvar = float(feature_dict.get("motion_variance", 0.0))
    dcons = float(feature_dict.get("direction_consistency", 1.0))
    if mvar > 25.0 and dcons < THRESH_CHAOTIC_MOTION:
        reasons.append("Unstable motion patterns")

    if risk_label == "HIGH" and not reasons:
        reasons.append("Elevated composite risk score")
    elif risk_label == "MEDIUM" and not reasons:
        reasons.append("Moderate environmental load")

    # De-duplicate preserving order
    seen = set()
    uniq: List[str] = []
    for r in reasons:
        if r not in seen:
            seen.add(r)
            uniq.append(r)

    primary = _pick_primary_cause(uniq)
    feature_contributions = _feature_contributions_dict(top_contributions)

    return {
        "risk": risk_label,
        "confidence": float(confidence),
        "primary_cause": primary,
        "reasons": uniq,
        "feature_contributions": feature_contributions,
    }
