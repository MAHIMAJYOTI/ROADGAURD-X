"""
Moving-object tracking: MOG2 background subtraction, contours, and ID assignment.

Maintains temporal memory (trajectories, velocity, acceleration) in a deque buffer.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class TrackState:
    """Per-object kinematic state."""

    track_id: int
    centroid: Tuple[float, float]
    bbox: Tuple[int, int, int, int]
    area: float
    velocity: Tuple[float, float]
    speed: float
    direction_rad: float
    acceleration: float


@dataclass
class FrameTracks:
    """Tracks and raw foreground mask for one frame."""

    tracks: List[TrackState]
    fg_mask: np.ndarray
    contours: List[np.ndarray]


class ObjectTracker:
    """
    MOG2 + contour detection + greedy ID association by nearest centroid.

    Keeps a rolling buffer of per-frame track snapshots for temporal features.
    """

    def __init__(
        self,
        history: int = 500,
        var_threshold: float = 16.0,
        detect_shadows: bool = True,
        min_area: int = 400,
        max_assoc_dist: float = 80.0,
        buffer_size: int = 30,
    ) -> None:
        self._bg = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=detect_shadows,
        )
        self._min_area = min_area
        self._max_assoc_dist = max_assoc_dist
        self._next_id = 1
        self._prev_centroids: Dict[int, Tuple[float, float]] = {}
        self._prev_vel: Dict[int, Tuple[float, float]] = {}
        self._buffer_size = buffer_size
        self._frame_buffer: Deque[FrameTracks] = deque(maxlen=buffer_size)
        # Per-id deque of (cx, cy, speed) for variance / consistency
        self._history_speed: Dict[int, Deque[float]] = {}
        self._history_dir: Dict[int, Deque[float]] = {}

    @property
    def frame_buffer(self) -> Deque[FrameTracks]:
        return self._frame_buffer

    def reset(self) -> None:
        """Reset background model and associations (new video source)."""
        self._bg = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=16.0,
            detectShadows=True,
        )
        self._next_id = 1
        self._prev_centroids.clear()
        self._prev_vel.clear()
        self._frame_buffer.clear()
        self._history_speed.clear()
        self._history_dir.clear()

    def _assign_ids(
        self, detections: List[Tuple[Tuple[float, float], Tuple[int, int, int, int], float]]
    ) -> List[TrackState]:
        """Greedy match current detections to previous centroids."""
        used_ids = set()
        tracks: List[TrackState] = []

        for (cx, cy), bbox, area in detections:
            best_id: Optional[int] = None
            best_d = self._max_assoc_dist + 1.0
            for tid, (px, py) in self._prev_centroids.items():
                if tid in used_ids:
                    continue
                d = np.hypot(cx - px, cy - py)
                if d < best_d and d <= self._max_assoc_dist:
                    best_d = d
                    best_id = tid
            if best_id is None:
                best_id = self._next_id
                self._next_id += 1
            used_ids.add(best_id)

            px, py = self._prev_centroids.get(best_id, (cx, cy))
            vx, vy = (cx - px, cy - py)
            speed = float(np.hypot(vx, vy))
            direction = float(np.arctan2(vy, vx + 1e-9))
            pvx, pvy = self._prev_vel.get(best_id, (0.0, 0.0))
            accel = float(np.hypot(vx - pvx, vy - pvy))

            st = TrackState(
                track_id=best_id,
                centroid=(float(cx), float(cy)),
                bbox=bbox,
                area=float(area),
                velocity=(float(vx), float(vy)),
                speed=speed,
                direction_rad=direction,
                acceleration=accel,
            )
            tracks.append(st)

        new_prev_c: Dict[int, Tuple[float, float]] = {}
        new_prev_v: Dict[int, Tuple[float, float]] = {}
        for t in tracks:
            new_prev_c[t.track_id] = t.centroid
            new_prev_v[t.track_id] = t.velocity
            if t.track_id not in self._history_speed:
                self._history_speed[t.track_id] = deque(maxlen=self._buffer_size)
            if t.track_id not in self._history_dir:
                self._history_dir[t.track_id] = deque(maxlen=self._buffer_size)
            self._history_speed[t.track_id].append(t.speed)
            self._history_dir[t.track_id].append(t.direction_rad)

        self._prev_centroids = new_prev_c
        self._prev_vel = new_prev_v

        # Drop histories for vanished IDs
        active = {t.track_id for t in tracks}
        for tid in list(self._history_speed.keys()):
            if tid not in active:
                del self._history_speed[tid]
                del self._history_dir[tid]

        return tracks

    def update(self, frame: np.ndarray) -> FrameTracks:
        """
        Update background model, extract contours, assign IDs, buffer result.

        Args:
            frame: BGR uint8 frame.

        Returns:
            FrameTracks for this frame.
        """
        fg = self._bg.apply(frame)
        fg = cv2.medianBlur(fg, 5)
        _, thresh = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        detections: List[Tuple[Tuple[float, float], Tuple[int, int, int, int], float]] = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < self._min_area:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            m = cv2.moments(c)
            if m["m00"] < 1e-6:
                continue
            cx = m["m10"] / m["m00"]
            cy = m["m01"] / m["m00"]
            detections.append(((cx, cy), (x, y, bw, bh), area))

        tracks = self._assign_ids(detections)
        ft = FrameTracks(tracks=tracks, fg_mask=thresh, contours=contours)
        self._frame_buffer.append(ft)
        return ft

    def get_speed_history(self, track_id: int) -> Deque[float]:
        return self._history_speed.get(track_id, deque())

    def get_direction_history(self, track_id: int) -> Deque[float]:
        return self._history_dir.get(track_id, deque())
