"""Session summary image generator using OpenCV only."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence

import cv2
import numpy as np


class SummaryFrameGenerator:
    """Build a visual summary sheet from selected risky frames."""

    def __init__(self) -> None:
        """Initialize summary generator."""
        self.thumb_w = 400
        self.thumb_h = 225

    def generate(
        self,
        frames_data: Sequence[Dict[str, Any]],
        report_dict: Dict[str, Any],
        output_path: Path,
    ) -> str:
        """Create and save summary image with up to three thumbnails."""
        if not frames_data:
            return "Warning: summary skipped (no candidate frames)."

        picks = list(frames_data)[:3]
        canvas_w = 60 + self.thumb_w * 3 + 40
        canvas_h = 430
        canvas = np.full((canvas_h, canvas_w, 3), 22, dtype=np.uint8)

        self._draw_header(canvas, report_dict)
        self._draw_thumbnails(canvas, picks)
        self._draw_stats_row(canvas, report_dict)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(str(output_path), canvas)
        if not ok:
            return f"Warning: failed to save summary to {output_path}."
        return f"Summary saved to {output_path}"

    def _draw_header(self, canvas: np.ndarray, report_dict: Dict[str, Any]) -> None:
        """Draw title and top-level session details."""
        summary = report_dict.get("summary", {})
        dist = summary.get("risk_distribution", {})
        dist_txt = f"LOW {dist.get('LOW', 0)} | MED {dist.get('MEDIUM', 0)} | HIGH {dist.get('HIGH', 0)}"
        cv2.putText(
            canvas,
            "RoadGuard-X Session Summary",
            (24, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.95,
            (235, 238, 245),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            (26, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (170, 175, 185),
            1,
            cv2.LINE_AA,
        )
        total_frames = int(summary.get("total_frames", 0))
        cv2.putText(
            canvas,
            f"Frames: {total_frames}    Risk Distribution: {dist_txt}",
            (24, 98),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (210, 214, 222),
            1,
            cv2.LINE_AA,
        )

    def _draw_thumbnails(
        self, canvas: np.ndarray, picks: Sequence[Dict[str, Any]]
    ) -> None:
        """Draw thumbnails and per-frame annotation bars."""
        base_y = 120
        for i, item in enumerate(picks):
            x = 20 + i * (self.thumb_w + 20)
            frame = item["frame"]
            thumb = cv2.resize(frame, (self.thumb_w, self.thumb_h))
            canvas[base_y : base_y + self.thumb_h, x : x + self.thumb_w] = thumb
            self._draw_bottom_bar(canvas, x, base_y, item)

    def _draw_bottom_bar(
        self, canvas: np.ndarray, x: int, y: int, item: Dict[str, Any]
    ) -> None:
        """Draw semi-transparent metadata bar on thumbnail."""
        bar_h = 52
        y0 = y + self.thumb_h - bar_h
        overlay = canvas.copy()
        cv2.rectangle(
            overlay, (x, y0), (x + self.thumb_w, y + self.thumb_h), (10, 10, 10), -1
        )
        cv2.addWeighted(overlay, 0.58, canvas, 0.42, 0, canvas)

        risk = str(item.get("risk", "LOW"))
        conf = float(item.get("confidence", 0.0))
        cause = str(item.get("primary_cause", "unknown")).replace("_", " ")
        t_sec = float(item.get("time_sec", 0.0))

        risk_col = (70, 220, 90)
        if risk == "HIGH":
            risk_col = (60, 60, 255)
        elif risk == "MEDIUM":
            risk_col = (0, 180, 255)

        cv2.putText(
            canvas,
            f"{risk}  {int(conf * 100)}%",
            (x + 10, y0 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            risk_col,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"{cause[:28]} | t={t_sec:.1f}s",
            (x + 10, y0 + 41),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (220, 224, 230),
            1,
            cv2.LINE_AA,
        )

    def _draw_stats_row(self, canvas: np.ndarray, report_dict: Dict[str, Any]) -> None:
        """Draw compact bottom row of session aggregate stats."""
        summary = report_dict.get("summary", {})
        txt = (
            f"avg_objects={float(summary.get('avg_objects', 0.0)):.2f}    "
            f"lane_drift_events={int(summary.get('lane_drift_events', 0))}    "
            f"scene_stability={float(summary.get('scene_stability', 0.0)):.3f}    "
            f"risk_trend={summary.get('risk_trend', 'stable')}"
        )
        cv2.putText(
            canvas,
            txt,
            (24, 392),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (220, 224, 230),
            1,
            cv2.LINE_AA,
        )
