"""
Session statistics and JSON report writer (output/report.json).
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

_RISK_CODE = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}


@dataclass
class SessionStats:
    """Aggregates over a video run."""

    total_frames: int = 0
    object_counts: List[float] = field(default_factory=list)
    risk_labels: List[str] = field(default_factory=list)
    lane_drift_events: int = 0
    lane_offset_samples: List[float] = field(default_factory=list)
    scene_stability_samples: List[float] = field(default_factory=list)
    scene_complexity_samples: List[float] = field(default_factory=list)
    # Per-frame top-3 RF feature short names (for global frequency summary)
    top_feature_hits: List[str] = field(default_factory=list)

    def record_frame(
        self,
        object_count: int,
        risk: str,
        lane_offset_norm: float,
        drift_event: bool,
        scene_stability: float,
        scene_complexity: float,
        top_feature_short_names: Sequence[str],
    ) -> None:
        self.total_frames += 1
        self.object_counts.append(float(object_count))
        self.risk_labels.append(risk)
        self.lane_offset_samples.append(abs(float(lane_offset_norm)))
        self.scene_stability_samples.append(float(scene_stability))
        self.scene_complexity_samples.append(float(scene_complexity))
        self.top_feature_hits.extend(list(top_feature_short_names))
        if drift_event:
            self.lane_drift_events += 1

    def risk_trend(self) -> str:
        """
        Compare mean ordinal risk of the last 5 frames vs the previous 5.

        LOW=0, MEDIUM=1, HIGH=2. Requires at least 10 frames; otherwise stable.
        """
        codes = [_RISK_CODE.get(r, 1) for r in self.risk_labels]
        if len(codes) < 10:
            return "stable"
        previous = codes[-10:-5]
        recent = codes[-5:]
        previous_mean = sum(previous) / 5.0
        recent_mean = sum(recent) / 5.0
        if recent_mean > previous_mean:
            return "increasing"
        if recent_mean < previous_mean:
            return "decreasing"
        return "stable"

    def top_features_global(self) -> List[str]:
        """Most frequent top-contributing features across the session (short names)."""
        if not self.top_feature_hits:
            return []
        c = Counter(self.top_feature_hits)
        return [name for name, _ in c.most_common(5)]

    def reset(self) -> None:
        """Clear session aggregates (keyboard reset during live run)."""
        self.total_frames = 0
        self.object_counts.clear()
        self.risk_labels.clear()
        self.lane_drift_events = 0
        self.lane_offset_samples.clear()
        self.scene_stability_samples.clear()
        self.scene_complexity_samples.clear()
        self.top_feature_hits.clear()

    def summary(self) -> Dict[str, Any]:
        dist = dict(Counter(self.risk_labels))
        avg_obj = (
            float(sum(self.object_counts) / len(self.object_counts))
            if self.object_counts
            else 0.0
        )
        avg_stab = (
            float(sum(self.scene_stability_samples) / len(self.scene_stability_samples))
            if self.scene_stability_samples
            else 0.0
        )
        avg_sci = (
            float(sum(self.scene_complexity_samples) / len(self.scene_complexity_samples))
            if self.scene_complexity_samples
            else 0.0
        )
        return {
            "total_frames": self.total_frames,
            "avg_objects": avg_obj,
            "risk_distribution": dist,
            "lane_drift_events": self.lane_drift_events,
            "avg_abs_lane_offset_norm": float(
                sum(self.lane_offset_samples) / len(self.lane_offset_samples)
            )
            if self.lane_offset_samples
            else 0.0,
            "scene_stability": avg_stab,
            "risk_trend": self.risk_trend(),
            "avg_scene_complexity": avg_sci,
            "top_features_global": self.top_features_global(),
        }


def write_report(
    path: Path,
    stats: SessionStats,
    meta: Optional[Dict[str, Any]] = None,
    summary_override: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Write report.json with session summary and optional metadata.

    Args:
        path: output/report.json
        stats: SessionStats instance
        meta: e.g. source, model path, feature schema
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "summary": summary_override if summary_override is not None else stats.summary(),
    }
    if meta:
        payload["meta"] = meta
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
