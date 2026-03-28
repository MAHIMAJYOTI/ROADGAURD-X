"""
Professional OpenCV dashboard: semi-transparent panel, tracks, lane overlay, status strip.

Uses OpenCV drawing only (no external GUI frameworks).
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from modules.features import scene_complexity_label
from modules.lane import LaneResult
from modules.tracker import TrackState

DASHBOARD_WIDTH = 350


def risk_color_bgr(risk_label: str) -> Tuple[int, int, int]:
    """Product palette: LOW green, MEDIUM yellow, HIGH red (BGR)."""
    if risk_label == "HIGH":
        return (40, 40, 255)
    if risk_label == "MEDIUM":
        return (0, 220, 255)
    return (60, 200, 80)


def _humanize_cause(code: str) -> str:
    return code.replace("_", " ").strip().title()


def _motion_box_color(speed: float, max_speed: float) -> Tuple[int, int, int]:
    """BGR from calm (green/cyan) to intense (orange/red) by motion intensity."""
    denom = max(max_speed, 8.0)
    t = float(np.clip(speed / denom, 0.0, 1.0))
    b = int(40 + 80 * t)
    g = int(220 - 100 * t)
    r = int(80 + 175 * t)
    return (b, g, r)


def draw_lane_overlay(
    frame: np.ndarray,
    lane: LaneResult,
    color_left: Tuple[int, int, int] = (80, 180, 255),
    color_right: Tuple[int, int, int] = (180, 255, 80),
) -> None:
    """Draw Hough line segments and ROI outline."""
    for x1, y1, x2, y2 in lane.left_lines:
        cv2.line(frame, (x1, y1), (x2, y2), color_left, 2, cv2.LINE_AA)
    for x1, y1, x2, y2 in lane.right_lines:
        cv2.line(frame, (x1, y1), (x2, y2), color_right, 2, cv2.LINE_AA)
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.polylines(
        overlay,
        [np.array([[0, h - 40], [w, h - 40], [w, int(h * 0.58)], [0, int(h * 0.58)]])],
        True,
        (100, 100, 100),
        1,
    )
    cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)


def draw_tracks(
    frame: np.ndarray,
    tracks: Sequence[TrackState],
    arrow_scale: float = 3.0,
) -> None:
    """Bounding boxes and arrows colored by motion intensity."""
    max_sp = max((t.speed for t in tracks), default=1.0)
    for t in tracks:
        x, y, bw, bh = t.bbox
        color = _motion_box_color(t.speed, max_sp)
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)
        label = f"ID {t.track_id} v={t.speed:.0f}"
        cv2.putText(
            frame,
            label,
            (x, max(y - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
        cx, cy = int(t.centroid[0]), int(t.centroid[1])
        vx, vy = t.velocity
        tip = (int(cx + vx * arrow_scale), int(cy + vy * arrow_scale))
        cv2.arrowedLine(frame, (cx, cy), tip, (220, 220, 255), 2, tipLength=0.22)


def draw_left_dashboard(
    frame: np.ndarray,
    *,
    risk_label: str,
    confidence: float,
    primary_cause: str,
    complexity_sci: float,
    stability: float,
    risk_trend: str,
    top_factors: Sequence[str],
) -> None:
    """Semi-transparent left panel (~350px) with product-style metrics."""
    _, w = frame.shape[:2]
    panel_w = min(DASHBOARD_WIDTH, w)
    panel = frame[:, :panel_w].copy()
    dark = np.zeros_like(panel)
    dark[:] = (28, 32, 38)
    cv2.addWeighted(dark, 0.52, panel, 0.48, 0, panel)
    frame[:, :panel_w] = panel

    accent = risk_color_bgr(risk_label)
    cx_bucket = scene_complexity_label(complexity_sci)
    y = 28
    line_h = 26

    def line(txt: str, col: Tuple[int, int, int], size: float = 0.52, thick: int = 1) -> None:
        nonlocal y
        cv2.putText(
            frame,
            txt[:52],
            (16, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            size,
            col,
            thick,
            cv2.LINE_AA,
        )
        y += line_h

    line("RoadGuard-X", (240, 245, 250), 0.62, 2)
    y += 6
    line("---", (120, 125, 130), 0.45, 1)
    line(f"Risk: {risk_label} ({confidence:.2f})", accent, 0.58, 2)
    line(f"Cause: {_humanize_cause(primary_cause)}", (230, 235, 240), 0.52, 1)
    line(f"Complexity: {cx_bucket}", (220, 225, 230), 0.52, 1)
    line(f"Stability: {stability:.2f}", (220, 225, 230), 0.52, 1)
    line(f"Trend: {risk_trend.capitalize()}", (220, 225, 230), 0.52, 1)
    line("-----------------", (100, 105, 110), 0.45, 1)
    line("Top Factors:", (200, 205, 210), 0.5, 1)
    for tf in list(top_factors)[:3]:
        line(f"  * {tf}", (190, 210, 215), 0.48, 1)
    line("---", (120, 125, 130), 0.45, 1)


def draw_high_risk_flash(
    frame: np.ndarray,
    frame_index: int,
    panel_width: int,
) -> None:
    """Alternating flash for HIGH risk in the video region (right of dashboard)."""
    if (frame_index // 12) % 2 == 0:
        return
    h, w = frame.shape[:2]
    x0 = panel_width + 20
    cv2.putText(
        frame,
        "!! HIGH RISK !!",
        (x0, h // 2),
        cv2.FONT_HERSHEY_DUPLEX,
        1.1,
        (50, 50, 255),
        3,
        cv2.LINE_AA,
    )


def draw_status_strip(
    frame: np.ndarray,
    *,
    frame_index: int,
    time_sec: float,
    fps: float,
    mode_label: str,
    recording: bool,
    panel_width: int,
) -> None:
    """Bottom-right: frame, time, FPS, mode, recording indicator."""
    h, w = frame.shape[:2]
    lines = [
        f"Frame: {frame_index}",
        f"Time: {time_sec:.1f}s",
        f"FPS: {fps:.0f}",
        f"Mode: {mode_label}",
    ]
    if recording:
        lines.append("REC *")
    y = h - 16
    for txt in reversed(lines):
        tw = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)[0][0]
        x = max(panel_width + 8, w - tw - 14)
        cv2.putText(
            frame,
            txt,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (245, 245, 248),
            1,
            cv2.LINE_AA,
        )
        y -= 20


def draw_lane_departure_counter(
    frame: np.ndarray, count: int, just_triggered: bool
) -> None:
    """Draw top-right lane departure pill with trigger highlight."""
    h, w = frame.shape[:2]
    txt = f"Lane deps: {int(count)}" + (" (!)" if just_triggered else "")
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
    pad_x, pad_y = 12, 10
    x1 = max(8, w - tw - pad_x * 2 - 16)
    y1 = 12
    x2 = min(w - 8, x1 + tw + pad_x * 2)
    y2 = y1 + th + pad_y * 2

    overlay = frame.copy()
    if just_triggered:
        bg = (20, 20, 150)
        fg = (70, 70, 255)
    else:
        bg = (28, 32, 38)
        fg = (235, 235, 238)
    cv2.rectangle(overlay, (x1, y1), (x2, y2), bg, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (90, 95, 110), 1, cv2.LINE_AA)
    cv2.putText(
        frame,
        txt,
        (x1 + pad_x, y2 - pad_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        fg,
        1 if not just_triggered else 2,
        cv2.LINE_AA,
    )


def draw_segmentation_overlay(
    frame: np.ndarray,
    mask_small: np.ndarray,
    alpha: float = 0.22,
) -> None:
    """Blend K-means road mask (only right of dashboard to keep panel readable)."""
    if mask_small.size == 0:
        return
    h, w = frame.shape[:2]
    mask_up = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_NEAREST)
    color = np.zeros_like(frame)
    color[:, :, 1] = mask_up
    x0 = min(DASHBOARD_WIDTH, w)
    roi = frame[:, x0:]
    croi = color[:, x0:]
    cv2.addWeighted(croi, alpha, roi, 1.0 - alpha, 0, roi)


def compose_frame(
    frame: np.ndarray,
    lane: LaneResult,
    tracks: Sequence[TrackState],
    risk_label: str,
    reasons: Sequence[str],
    frame_index: int,
    confidence: float,
    primary_cause: str,
    scene_complexity_sci: float,
    top_factors: Sequence[str],
    stability: float,
    risk_trend: str,
    fps_display: float,
    time_sec: float,
    mode_label: str,
    recording: bool,
    seg_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Full composition: segmentation, lanes, tracks, dashboard, alerts, status."""
    out = frame
    if seg_mask is not None:
        draw_segmentation_overlay(out, seg_mask)
    draw_lane_overlay(out, lane)
    draw_tracks(out, tracks)
    draw_left_dashboard(
        out,
        risk_label=risk_label,
        confidence=confidence,
        primary_cause=primary_cause,
        complexity_sci=scene_complexity_sci,
        stability=stability,
        risk_trend=risk_trend,
        top_factors=top_factors,
    )
    if risk_label == "HIGH":
        draw_high_risk_flash(out, frame_index, DASHBOARD_WIDTH)
    draw_status_strip(
        out,
        frame_index=frame_index,
        time_sec=time_sec,
        fps=fps_display,
        mode_label=mode_label,
        recording=recording,
        panel_width=DASHBOARD_WIDTH,
    )
    # Compact rule-based reasons along bottom-left (below dashboard text area)
    h = out.shape[0]
    y = h - 110
    for i, line in enumerate(list(reasons)[:5]):
        cv2.putText(
            out,
            line[:88],
            (16, y + i * 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (210, 215, 220),
            1,
            cv2.LINE_AA,
        )
    return out
