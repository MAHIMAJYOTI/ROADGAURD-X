"""Risk timeline chart generator using matplotlib."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


class TimelineChart:
    """Generate confidence/complexity timeline with risk zone shading."""

    def __init__(self, fps: float) -> None:
        """Initialize chart settings."""
        self.fps = float(max(fps, 1.0))

    def generate(
        self,
        frame_data_list: Sequence[Dict[str, Any]],
        lane_drift_frames: Sequence[int],
        output_path: Path,
    ) -> str:
        """Create and save the timeline chart image."""
        if not frame_data_list:
            return "Warning: timeline skipped (no frame data)."

        idx = np.array([int(d["frame_index"]) for d in frame_data_list], dtype=np.int32)
        conf = np.array([float(d["confidence"]) for d in frame_data_list], dtype=np.float32)
        sci = np.array(
            [float(d["scene_complexity_index"]) for d in frame_data_list], dtype=np.float32
        )
        risks = [str(d["risk"]) for d in frame_data_list]

        fig, ax = plt.subplots(figsize=(14, 5), dpi=150)
        self._shade_risk_zones(ax, idx, risks)
        ax.plot(idx, conf, color="#5b677a", linewidth=1.0, alpha=0.5, label="confidence")

        colors = []
        for r in risks:
            if r == "HIGH":
                colors.append("#ef4444")
            elif r == "MEDIUM":
                colors.append("#f59e0b")
            else:
                colors.append("#22c55e")
        ax.scatter(idx, conf, c=colors, s=8, alpha=0.9)
        ax.plot(
            idx,
            sci,
            color="#9ca3af",
            linestyle="--",
            linewidth=1.3,
            alpha=0.95,
            label="complexity",
        )

        for f in lane_drift_frames:
            ax.axvline(float(f), linestyle="--", linewidth=0.8, color="#60a5fa", alpha=0.35)

        ax.set_ylim(0.0, 1.0)
        ax.set_xlim(float(idx.min()), float(idx.max()) if idx.max() > 0 else 1.0)
        ax.set_xlabel("Frame Index")
        ax.set_ylabel("Score (0-1)")
        ax.set_title("RoadGuard-X Risk Timeline")
        ax.grid(True, alpha=0.22)

        secax = ax.secondary_xaxis(
            "top",
            functions=(lambda x: x / self.fps, lambda s: s * self.fps),
        )
        secax.set_xlabel("Time (seconds)")

        handles = [
            plt.Line2D([0], [0], color="#5b677a", linewidth=1.5, label="confidence"),
            plt.Line2D([0], [0], color="#9ca3af", linestyle="--", linewidth=1.5, label="complexity"),
            plt.Rectangle((0, 0), 1, 1, color="#fecaca", alpha=0.32, label="HIGH zone"),
            plt.Rectangle((0, 0), 1, 1, color="#fed7aa", alpha=0.25, label="MEDIUM zone"),
        ]
        ax.legend(handles=handles, loc="upper right", framealpha=0.9)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(str(output_path), dpi=150)
        plt.close(fig)
        return f"Timeline chart saved to {output_path}"

    def _shade_risk_zones(
        self, ax: plt.Axes, idx: np.ndarray, risks: Sequence[str]
    ) -> None:
        """Shade contiguous HIGH and MEDIUM ranges in the plot background."""
        for level, color, alpha in (("HIGH", "#fecaca", 0.32), ("MEDIUM", "#fed7aa", 0.25)):
            for x0, x1 in self._ranges_for_level(idx, risks, level):
                ax.axvspan(float(x0), float(x1), color=color, alpha=alpha, linewidth=0)

    def _ranges_for_level(
        self, idx: np.ndarray, risks: Sequence[str], level: str
    ) -> List[Tuple[int, int]]:
        """Return contiguous index ranges for a target risk level."""
        ranges: List[Tuple[int, int]] = []
        start = None
        prev = None
        for i, r in zip(idx.tolist(), risks):
            if r == level:
                if start is None:
                    start = i
                prev = i
            elif start is not None and prev is not None:
                ranges.append((start, prev))
                start, prev = None, None
        if start is not None and prev is not None:
            ranges.append((start, prev))
        return ranges
