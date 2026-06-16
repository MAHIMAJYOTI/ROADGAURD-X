"""
RoadGuard-X — Context-Aware Driving Intelligence System

CLI entry: real-time analysis with dashboard, optional capture, and reporting.

Usage (from this directory):
  pip install -r requirements.txt
  python main.py --source sample
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

# Ensure imports work when executed as `python main.py`
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from modules.explain import build_explanation, short_feature_name  # noqa: E402
from modules.features import extract_combined  # noqa: E402
from modules.lane import detect_lanes  # noqa: E402
from modules.preprocess import preprocess_frame  # noqa: E402
from modules.risk_model import RiskModel  # noqa: E402
from modules.segmentation import kmeans_segment  # noqa: E402
from modules.tracker import ObjectTracker  # noqa: E402
from utils.clip_writer import ClipWriter, get_video_writer, reencode_h264  # noqa: E402
from utils.data_logger import append_dataset_row, wipe_dataset  # noqa: E402
from utils.heatmap import HeatmapOverlay  # noqa: E402
from utils.hud import compose_frame, draw_lane_departure_counter  # noqa: E402
from utils.reporter import SessionStats, write_report  # noqa: E402
from utils.summary_frame import SummaryFrameGenerator  # noqa: E402
from utils.timeline_chart import TimelineChart  # noqa: E402

LANE_DRIFT_THRESHOLD = 0.08
LANE_OFFSET_BUFFER_LEN = 10
MIN_LANE_BUFFER_FOR_DRIFT = 5
STABILITY_SMOOTHING_LEN = 10
DATASET_CSV = _ROOT / "data" / "dataset.csv"
OUTPUT_VIDEO = _ROOT / "output" / "output.mp4"


def pick_sample_video() -> Path:
    """Randomly pick a bundled driving clip; fallback to sample.mp4."""
    candidates = [
        _ROOT / "samples" / "road_real_1.mp4",
        _ROOT / "samples" / "road_real_2.mp4",
        _ROOT / "samples" / "street.mp4",
        _ROOT / "samples" / "sample.mp4",
    ]
    existing = [p for p in candidates if p.exists()]
    if not existing:
        return _ROOT / "samples" / "sample.mp4"
    return random.choice(existing)


def open_capture(source: str, sample_path: Path) -> tuple[cv2.VideoCapture, float]:
    """Open webcam or file; return (capture, nominal_fps)."""
    if source == "webcam":
        if sys.platform == "win32":
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)
        fps = 30.0
        return cap, fps
    cap = cv2.VideoCapture(str(sample_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 1e-3:
        fps = 30.0
    return cap, fps


def main() -> None:
    parser = argparse.ArgumentParser(description="RoadGuard-X driving scene analysis.")
    parser.add_argument(
        "--source",
        choices=("webcam", "sample"),
        default="sample",
        help="Video source (webcam falls back to sample if unavailable).",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Process without display window (report still written).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after N frames (0 = until video ends or quit).",
    )
    parser.add_argument(
        "--collect-data",
        action="store_true",
        help="Append feature vectors + risk labels to data/dataset.csv",
    )
    parser.add_argument(
        "--overwrite-dataset",
        action="store_true",
        help="Remove existing dataset.csv before run (only with --collect-data).",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Record annotated output to output/output.mp4 (can toggle with v).",
    )
    parser.add_argument(
        "--save-clips",
        action="store_true",
        help="Save danger clips around HIGH-risk segments to output/clips/.",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Skip output/summary.png generation.",
    )
    parser.add_argument(
        "--no-timeline",
        action="store_true",
        help="Skip output/risk_timeline.png generation.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Analyze a specific video file (optional; overrides --source for capture).",
    )
    args = parser.parse_args()

    file_mode: bool
    if args.input is not None:
        sample_path = args.input.expanduser().resolve()
        if not sample_path.is_file():
            raise SystemExit(f"Input video not found: {sample_path}")
        cap, source_fps = open_capture("sample", sample_path)
        file_mode = True
    elif args.source == "sample":
        sample_path = pick_sample_video()
        if not sample_path.is_file():
            print(
                "Sample video not found. Please run first:\n"
                "    python generate_sample.py"
            )
            sys.exit(1)
        cap, source_fps = open_capture("sample", sample_path)
        file_mode = True
    else:
        sample_path = _ROOT / "samples" / "sample.mp4"
        cap, source_fps = open_capture("webcam", sample_path)
        file_mode = False

    if args.source == "webcam" and args.input is None and not cap.isOpened():
        print("Webcam not available; falling back to sample video.")
        cap.release()
        sample_path = pick_sample_video()
        cap, source_fps = open_capture("sample", sample_path)

    if not cap.isOpened():
        raise SystemExit(f"Could not open video source. Tried {sample_path}")
    (_ROOT / "output").mkdir(parents=True, exist_ok=True)

    if args.collect_data and args.overwrite_dataset:
        wipe_dataset(DATASET_CSV)
        print(f"Cleared {DATASET_CSV}")

    model = RiskModel()
    tracker = ObjectTracker(buffer_size=35)
    stats = SessionStats()
    lane_offsets_buffer: deque[float] = deque(maxlen=LANE_OFFSET_BUFFER_LEN)
    risk_history: deque[str] = deque(maxlen=20)
    stability_buffer: deque[float] = deque(maxlen=STABILITY_SMOOTHING_LEN)

    frame_idx = 0
    snap_counter = 0
    recording = bool(args.save_video)
    video_writer_available = True
    video_writer: cv2.VideoWriter | None = None
    heatmap_enabled = False
    lane_departure_count = 0
    lane_departure_flash_frames = 0
    prev_lane_offset_norm = 0.0

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    clip_enabled = bool(args.save_clips and ((not args.headless) or args.save_video))
    clip_writer = ClipWriter(
        clips_dir=_ROOT / "output" / "clips",
        fps=source_fps,
        frame_size=(frame_w, frame_h),
        enabled=clip_enabled,
    )
    summary_generator = SummaryFrameGenerator()
    timeline = TimelineChart(fps=source_fps)
    heatmap_overlay = HeatmapOverlay()
    timeline_rows: List[Dict[str, Any]] = []
    lane_drift_frames: List[int] = []
    high_frame_rows: List[Dict[str, Any]] = []
    medium_frame_rows: List[Dict[str, Any]] = []

    fps_smooth = source_fps
    prev_t = time.perf_counter()

    last_frame_snapshot: dict | None = None

    mode_label = "Real-Time Analysis"
    if args.input is not None:
        mode_label = f"File: {sample_path.name}"
    elif args.source == "sample":
        mode_label = f"Sample: {sample_path.name}"

    print(
        "RoadGuard-X  keys: q quit | s save frame | r reset session stats | "
        "v toggle recording | dataset " + ("ON" if args.collect_data else "off")
    )
    display_enabled = not args.headless

    while True:
        try:
            ok, frame = cap.read()
        except SystemError:
            break
        if not ok:
            break

        now = time.perf_counter()
        dt = max(now - prev_t, 1e-6)
        prev_t = now
        inst_fps = 1.0 / dt
        fps_smooth = 0.85 * fps_smooth + 0.15 * inst_fps

        frame_idx += 1
        proc = preprocess_frame(
            frame,
            blur_kernel=(5, 5),
            gamma=0.95,
            use_clahe=True,
        )
        lane = detect_lanes(proc)

        small = cv2.resize(proc, (240, 135))
        seg = kmeans_segment(small, k=3)

        ft = tracker.update(proc)
        feats, feat_dict = extract_combined(proc, lane, ft, tracker)

        raw_stability = float(feat_dict.get("scene_stability", 0.0))
        stability_buffer.append(raw_stability)
        smoothed_stability = float(np.mean(stability_buffer))
        feat_dict["scene_stability"] = smoothed_stability

        risk_label, confidence, top_contrib = model.predict(feats)
        risk_history.append(risk_label)

        lane_offsets_buffer.append(float(lane.center_offset_norm))
        if len(lane_offsets_buffer) >= MIN_LANE_BUFFER_FOR_DRIFT:
            temporal_drift = abs(float(np.mean(lane_offsets_buffer))) > LANE_DRIFT_THRESHOLD
        else:
            temporal_drift = False

        lane_dep_trigger = (
            abs(float(lane.center_offset_norm)) > 0.3 and abs(prev_lane_offset_norm) <= 0.3
        )
        prev_lane_offset_norm = float(lane.center_offset_norm)
        if lane_dep_trigger:
            lane_departure_count += 1
            lane_departure_flash_frames = 30
            lane_drift_frames.append(frame_idx)
        elif lane_departure_flash_frames > 0:
            lane_departure_flash_frames -= 1

        explanation = build_explanation(
            risk_label,
            feat_dict,
            confidence,
            top_contributions=top_contrib,
            extra={"lane_drift_event": temporal_drift},
        )

        sci = float(feat_dict.get("scene_complexity_index", 0.0))
        stab = float(feat_dict.get("scene_stability", 0.0))

        top_short = [short_feature_name(n) for n, _ in top_contrib]

        stats.record_frame(
            object_count=len(ft.tracks),
            risk=risk_label,
            lane_offset_norm=lane.center_offset_norm,
            drift_event=temporal_drift,
            scene_stability=stab,
            scene_complexity=sci,
            top_feature_short_names=top_short,
        )

        if args.collect_data:
            append_dataset_row(DATASET_CSV, feats, risk_label)

        contrib_bits = ", ".join(
            f"{k}={v}" for k, v in explanation["feature_contributions"].items()
        )
        reason_lines = [f"RF: {contrib_bits}"] + explanation["reasons"]

        last_frame_snapshot = {
            "risk": risk_label,
            "confidence": explanation["confidence"],
            "primary_cause": explanation["primary_cause"],
            "reasons": list(explanation["reasons"]),
            "feature_contributions": dict(explanation["feature_contributions"]),
        }

        time_sec = (
            (frame_idx - 1) / source_fps
            if file_mode
            else (frame_idx - 1) / max(fps_smooth, 1e-3)
        )

        display = compose_frame(
            proc.copy(),
            lane,
            ft.tracks,
            risk_label,
            reason_lines,
            frame_idx,
            confidence=explanation["confidence"],
            primary_cause=explanation["primary_cause"],
            scene_complexity_sci=sci,
            top_factors=top_short,
            stability=stab,
            risk_trend=stats.risk_trend(),
            fps_display=fps_smooth,
            time_sec=time_sec,
            mode_label=mode_label,
            recording=recording,
            seg_mask=seg.mask,
        )
        try:
            heatmap_overlay.update(proc, ft.fg_mask)
        except Exception as e:
            print(f"Warning: heatmap update failed: {e}")
        if heatmap_enabled:
            try:
                display = heatmap_overlay.render(display)
            except Exception as e:
                print(f"Warning: heatmap render failed: {e}")
        draw_lane_departure_counter(
            display,
            lane_departure_count,
            just_triggered=(lane_departure_flash_frames > 0),
        )

        clip_status = clip_writer.update(display, risk_label)
        if clip_status:
            print(clip_status)

        row = {
            "frame": display.copy(),
            "risk": risk_label,
            "confidence": float(explanation["confidence"]),
            "primary_cause": str(explanation["primary_cause"]),
            "frame_index": frame_idx,
            "time_sec": float(time_sec),
        }
        if risk_label == "HIGH":
            high_frame_rows.append(row)
        elif risk_label == "MEDIUM":
            medium_frame_rows.append(row)

        timeline_rows.append(
            {
                "frame_index": frame_idx,
                "risk": risk_label,
                "confidence": float(explanation["confidence"]),
                "lane_offset_norm": float(lane.center_offset_norm),
                "scene_complexity_index": float(sci),
            }
        )

        if recording and video_writer_available:
            if video_writer is None:
                oh, ow = display.shape[:2]
                write_fps = (
                    source_fps
                    if file_mode
                    else float(max(15.0, min(60.0, fps_smooth)))
                )
                try:
                    video_writer = get_video_writer(
                        OUTPUT_VIDEO,
                        write_fps,
                        (int(ow), int(oh)),
                    )
                except RuntimeError:
                    print("Warning: could not open VideoWriter; disabling recording.")
                    video_writer_available = False
                    recording = False
            if video_writer is not None and video_writer.isOpened():
                video_writer.write(display)

        if display_enabled:
            try:
                cv2.imshow("RoadGuard-X", display)
                key = cv2.waitKey(1) & 0xFF
            except cv2.error:
                # Some OpenCV builds (often headless wheels) do not ship GUI backends.
                # Fall back to headless processing so report/video generation still works.
                display_enabled = False
                print(
                    "OpenCV GUI (imshow) not available on this system. "
                    "Continuing in headless mode."
                )
                key = -1
            if key == ord("q"):
                break
            if key == ord("s"):
                snap_counter += 1
                out_img = _ROOT / "output" / f"frame_{frame_idx}_{snap_counter}.jpg"
                out_img.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out_img), display)
                print(f"Saved {out_img}")
            if key == ord("r"):
                stats.reset()
                lane_offsets_buffer.clear()
                stability_buffer.clear()
                risk_history.clear()
                print("Session stats reset.")
            if key == ord("v"):
                recording = not recording
                print(f"Recording {'ON' if recording else 'paused'}")
            if key == ord("h"):
                heatmap_enabled = not heatmap_enabled
                print(f"Heatmap {'ON' if heatmap_enabled else 'OFF'}")

        if args.max_frames and frame_idx >= args.max_frames:
            break

    cap.release()
    if video_writer is not None:
        video_writer.release()
        reencode_h264(OUTPUT_VIDEO)
    if display_enabled:
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass

    clip_flush = clip_writer.flush_remaining()
    if clip_flush:
        print(clip_flush)

    report_path = _ROOT / "output" / "report.json"
    summary_payload = stats.summary()
    summary_payload["danger_clips_saved"] = clip_writer.clip_count
    explanation_meta: dict | None = None
    if last_frame_snapshot:
        primary = str(last_frame_snapshot.get("primary_cause") or "composite_risk")
        reasons = list(last_frame_snapshot.get("reasons") or [])
        contribs = dict(last_frame_snapshot.get("feature_contributions") or {})
        conf_val = float(last_frame_snapshot.get("confidence") or 0.0)
        risk_val = str(last_frame_snapshot.get("risk") or "LOW")
        reason_text = (
            "; ".join(reasons) if reasons else "No rule triggers; model confidence is primary."
        )
        explanation_meta = {
            "risk": risk_val,
            "confidence": conf_val,
            "primary_cause": primary,
            "reasons": reasons,
            "feature_contributions": contribs,
            "top_features": list(contribs.keys()),
            "summary": (
                f"{risk_val} risk ({conf_val * 100:.1f}% confidence). "
                f"Primary cause: {primary.replace('_', ' ')}. {reason_text}"
            ),
        }

    report_meta = {
        "source": "file" if args.input is not None else args.source,
        "sample_file": sample_path.name if file_mode else None,
        "model": "models/risk_model.pkl",
        "frames_processed": frame_idx,
        "dataset_appended": args.collect_data,
        "video_saved": OUTPUT_VIDEO.exists() and OUTPUT_VIDEO.stat().st_size > 0,
        "last_frame": last_frame_snapshot,
        "explanation": explanation_meta,
    }
    write_report(
        report_path,
        stats,
        meta=report_meta,
        summary_override=summary_payload,
    )
    print(f"Wrote report to {report_path}")
    print(f"Danger clips saved: {clip_writer.clip_count}")

    if not args.no_timeline:
        try:
            timeline_status = timeline.generate(
                frame_data_list=timeline_rows,
                lane_drift_frames=lane_drift_frames,
                output_path=_ROOT / "output" / "risk_timeline.png",
            )
            print(timeline_status)
        except Exception as e:
            print(f"Warning: timeline chart generation failed: {e}")

    if not args.no_summary:
        try:
            picks = sorted(high_frame_rows, key=lambda x: x["confidence"], reverse=True)
            if len(picks) < 3:
                medium_sorted = sorted(
                    medium_frame_rows, key=lambda x: x["confidence"], reverse=True
                )
                picks.extend(medium_sorted[: max(0, 3 - len(picks))])
            summary_status = summary_generator.generate(
                frames_data=picks[:3],
                report_dict={"summary": summary_payload, "meta": report_meta},
                output_path=_ROOT / "output" / "summary.png",
            )
            print(summary_status)
        except Exception as e:
            print(f"Warning: summary generation failed: {e}")


if __name__ == "__main__":
    main()
