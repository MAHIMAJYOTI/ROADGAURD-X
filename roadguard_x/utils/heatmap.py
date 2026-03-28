"""Toggleable motion+edge heatmap overlay."""

from __future__ import annotations

import cv2
import numpy as np


class HeatmapOverlay:
    """Maintain a decayed heatmap accumulation and render blended output."""

    def __init__(self, decay: float = 0.85, blend: float = 0.5) -> None:
        """Initialize heatmap smoothing parameters."""
        self.decay = float(np.clip(decay, 0.0, 0.999))
        self.blend = float(np.clip(blend, 0.0, 1.0))
        self.buffer: np.ndarray | None = None

    def update(self, frame: np.ndarray, fg_mask: np.ndarray) -> None:
        """Update running heatmap from current frame and foreground mask."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edge = cv2.Canny(gray, 50, 150).astype(np.float32) / 255.0
        motion = fg_mask.astype(np.float32) / 255.0
        if motion.shape != edge.shape:
            motion = cv2.resize(motion, (edge.shape[1], edge.shape[0]))
        combined = 0.6 * edge + 0.4 * motion
        if self.buffer is None or self.buffer.shape != combined.shape:
            self.buffer = combined.copy()
            return
        self.buffer = self.buffer * self.decay + combined * (1.0 - self.decay)

    def render(self, frame: np.ndarray) -> np.ndarray:
        """Render heatmap-blended output frame."""
        if self.buffer is None:
            return frame
        norm = cv2.normalize(self.buffer, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heat = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        out = cv2.addWeighted(frame, 1.0 - self.blend, heat, self.blend, 0.0)
        cv2.putText(
            out,
            "HEATMAP ON",
            (16, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 220, 255),
            2,
            cv2.LINE_AA,
        )
        return out
