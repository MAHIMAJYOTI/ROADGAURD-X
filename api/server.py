"""
RoadGuard-X FastAPI gateway: upload video, run existing CLI pipeline, serve outputs.

Run from repository root:
  pip install -r api/requirements.txt
  uvicorn api.server:app --reload --host 0.0.0.0 --port 8000

Or from roadguard_x (adjust PYTHONPATH) — prefer running uvicorn from repo root.
"""

from __future__ import annotations

import json
import mimetypes
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

REPO_ROOT = Path(__file__).resolve().parent.parent
RG_ROOT = REPO_ROOT / "roadguard_x"
OUTPUT_DIR = RG_ROOT / "output"
REPORT_PATH = OUTPUT_DIR / "report.json"
OUTPUT_VIDEO = OUTPUT_DIR / "output.mp4"
SUMMARY_IMAGE = OUTPUT_DIR / "summary.png"
TIMELINE_IMAGE = OUTPUT_DIR / "risk_timeline.png"
CLIPS_DIR = OUTPUT_DIR / "clips"
# The CLI pipeline can be CPU-heavy on larger inputs (tracking + k-means per frame).
# Keep this generous for lower-end laptops to avoid false timeout errors.
SUBPROCESS_TIMEOUT_SEC = 420
IDLE_RESET_AFTER_DONE_SEC = 10.0
PIPELINE_TIMEOUT_MESSAGE = "Processing timeout (video too long or system busy)"
ALLOWED_VIDEO_EXT = frozenset({".mp4", ".avi", ".mov", ".mkv", ".webm"})

MODEL_LABEL = "Random Forest v2"
FEATURE_COUNT = 9

MODEL_META_PATH = RG_ROOT / "models" / "training_metadata.json"
UPLOAD_DIR = Path(__file__).resolve().parent / "uploads"

app = FastAPI(title="RoadGuard-X API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    # Allow the Next.js dev UI when opened via LAN IP (e.g. http://192.168.x.x:3000) so
    # fetch() and media URLs resolve to the same machine as the API.
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1|[0-9]{1,3}(\.[0-9]{1,3}){3})(:[0-9]+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def _resolve_output_file(requested_path: str) -> Path:
    """Resolve a requested media path within OUTPUT_DIR (path traversal safe)."""
    out_root = OUTPUT_DIR.resolve()
    candidate = (OUTPUT_DIR / requested_path).resolve()
    try:
        candidate.relative_to(out_root)
    except ValueError as e:
        raise HTTPException(status_code=404, detail="Not found") from e
    return candidate


def _guess_media_type(path: Path) -> str | None:
    """Force correct content-type for mp4 so browsers decode reliably."""
    if path.suffix.lower() == ".mp4":
        return "video/mp4"
    return mimetypes.guess_type(str(path))[0]


@app.get("/files/{requested_path:path}")
def serve_files(requested_path: str) -> FileResponse:
    """Serve output video under /files/* with correct content-type."""
    file_path = _resolve_output_file(requested_path)
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(
        path=str(file_path),
        media_type=_guess_media_type(file_path),
        filename=file_path.name,
    )


@app.get("/media/{requested_path:path}")
def serve_media(requested_path: str) -> FileResponse:
    """Serve artifacts under /media/* with correct content-type."""
    file_path = _resolve_output_file(requested_path)
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(
        path=str(file_path),
        media_type=_guess_media_type(file_path),
        filename=file_path.name,
    )

_state_lock = threading.Lock()

# Strict pipeline state machine:
# idle -> processing -> done|error -> idle (via timer)
_pipeline_status: str = "idle"
_error_message: str | None = None
_result_payload: dict | None = None
_completion_monotonic: float | None = None


def _reset_to_idle_unlocked() -> None:
    """Reset state for a new session (caller holds _state_lock)."""
    global _pipeline_status, _error_message, _result_payload, _completion_monotonic
    _pipeline_status = "idle"
    _error_message = None
    _result_payload = None
    _completion_monotonic = None


def _try_expire_unlocked() -> None:
    """Expire done/error state back to idle (caller holds _state_lock)."""
    global _pipeline_status, _completion_monotonic
    if _pipeline_status not in ("done", "error"):
        return
    if _completion_monotonic is None:
        return
    if time.monotonic() - _completion_monotonic >= IDLE_RESET_AFTER_DONE_SEC:
        _reset_to_idle_unlocked()


def cleanup_stale_uploads(max_age_sec: float = 3600.0, max_files: int = 32) -> None:
    """Remove old upload files to avoid filling disk."""
    try:
        files = [p for p in UPLOAD_DIR.iterdir() if p.is_file()]
    except OSError:
        return
    now = time.time()
    for p in files:
        try:
            if now - p.stat().st_mtime > max_age_sec:
                p.unlink(missing_ok=True)
        except OSError:
            continue
    try:
        remaining = sorted(
            (p for p in UPLOAD_DIR.iterdir() if p.is_file()),
            key=lambda p: p.stat().st_mtime,
        )
        while len(remaining) > max_files:
            oldest = remaining.pop(0)
            try:
                oldest.unlink(missing_ok=True)
            except OSError:
                break
    except OSError:
        pass


def _terminate_child_cleanly(proc: subprocess.Popen | None) -> None:
    """Best-effort stop of a timed-out pipeline child."""
    if proc is None:
        return
    try:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.wait(timeout=3)
                except OSError:
                    pass
    except OSError:
        try:
            proc.kill()
        except OSError:
            pass


def _run_pipeline_subprocess(input_video: Path) -> None:
    """Invoke existing main.py (unchanged CV pipeline)."""
    main_py = RG_ROOT / "main.py"
    if not main_py.is_file():
        raise RuntimeError(f"main.py not found at {main_py}")

    cmd = [
        sys.executable,
        str(main_py),
        "--input",
        str(input_video),
        "--headless",
        "--save-video",
        "--save-clips",
        "--max-frames",
        "0",
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(RG_ROOT),
            capture_output=True,
            text=True,
            timeout=SUBPROCESS_TIMEOUT_SEC,
        )
    except subprocess.TimeoutExpired as e:
        # subprocess.run may not expose .process depending on Python version/platform.
        try:
            _terminate_child_cleanly(getattr(e, "process", None))
        except Exception:
            pass
        # Best-effort logging for debugging stuck/slow runs.
        try:
            tail = ""
            if getattr(e, "stdout", None):
                tail += str(e.stdout)[-2000:]
            if getattr(e, "stderr", None):
                tail += str(e.stderr)[-2000:]
            if tail:
                print(f"[API] Processing timeout tail (last 2k chars):\n{tail}")
        except Exception:
            pass
        print(f"[API] Processing timeout after {SUBPROCESS_TIMEOUT_SEC}s")
        raise RuntimeError(PIPELINE_TIMEOUT_MESSAGE) from None
    except (subprocess.SubprocessError, OSError) as e:
        raise RuntimeError(f"Pipeline subprocess failed: {e}") from e
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip() or f"exit {proc.returncode}"
        raise RuntimeError(err)


def _artifact_payload() -> dict:
    """Build media URLs for optional generated artifacts."""
    payload: dict = {}
    if SUMMARY_IMAGE.is_file():
        payload["summary_image_url"] = "/media/summary.png"
    if TIMELINE_IMAGE.is_file():
        payload["timeline_image_url"] = "/media/risk_timeline.png"
    if CLIPS_DIR.is_dir():
        clips = sorted(p.name for p in CLIPS_DIR.glob("*.mp4") if p.is_file())
        payload["danger_clips"] = [f"/media/clips/{name}" for name in clips]
    else:
        payload["danger_clips"] = []
    return payload


@app.get("/health")
def health() -> dict:
    from roadguard_x.utils.clip_writer import resolve_ffmpeg_executable

    model_path = RG_ROOT / "models" / "risk_model.pkl"
    samples_dir = RG_ROOT / "samples"
    samples_ok = False
    if samples_dir.is_dir():
        try:
            samples_ok = any(p.is_file() and p.suffix.lower() in ALLOWED_VIDEO_EXT for p in samples_dir.iterdir())
        except OSError:
            samples_ok = False
    return {
        "status": "ok",
        "cwd": str(RG_ROOT.resolve()),
        "model_exists": model_path.is_file(),
        "samples_available": samples_ok,
        "ffmpeg_available": resolve_ffmpeg_executable() is not None,
    }


@app.get("/model-info")
def model_info() -> JSONResponse:
    """Return model training metadata for frontend feature-importance visualization."""
    fallback = {
        "timestamp": None,
        "n_samples": 0,
        "class_distribution": {"LOW": 0, "MEDIUM": 0, "HIGH": 0},
        "feature_importances": {},
        "sklearn_version": None,
    }
    if not MODEL_META_PATH.is_file():
        return JSONResponse(content=fallback)
    try:
        data = json.loads(MODEL_META_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return JSONResponse(content=fallback)
    if not isinstance(data, dict):
        return JSONResponse(content=fallback)
    payload = dict(fallback)
    payload.update(data)
    if not isinstance(payload.get("feature_importances"), dict):
        payload["feature_importances"] = {}
    return JSONResponse(content=payload)

def _build_done_payload(report: dict) -> dict:
    """Build a /report done payload for the frontend."""
    summary = report.get("summary") if isinstance(report, dict) else {}
    meta_in = report.get("meta") if isinstance(report, dict) else {}
    frames = summary.get("total_frames", meta_in.get("frames_processed"))
    try:
        frames = int(frames) if frames is not None else None
    except (TypeError, ValueError):
        frames = None

    video_url = "/files/output.mp4" if OUTPUT_VIDEO.is_file() else None
    payload = {
        "status": "done",
        "video_url": video_url,
        "report": report,
        "meta": {
            "frames": frames,
            "model": MODEL_LABEL,
            "features": FEATURE_COUNT,
            # keep key for UI extra line; actual training metadata is exposed via /model-info
            "training_data": "Synthetic (calibrated) + optional CSV collection",
        },
    }
    payload.update(_artifact_payload())
    return payload


def run_pipeline(saved_path: Path) -> None:
    """Background job: run subprocess, then update global state."""
    global _pipeline_status, _error_message, _result_payload

    print("[API] Processing started...")
    try:
        # Clear previous artifacts to avoid stale UI images.
        try:
            if SUMMARY_IMAGE.is_file():
                SUMMARY_IMAGE.unlink(missing_ok=True)
            if TIMELINE_IMAGE.is_file():
                TIMELINE_IMAGE.unlink(missing_ok=True)
            if CLIPS_DIR.is_dir():
                for p in CLIPS_DIR.glob("*.mp4"):
                    p.unlink(missing_ok=True)
        except OSError:
            pass

        _run_pipeline_subprocess(saved_path)

        if not REPORT_PATH.is_file():
            raise RuntimeError("Pipeline finished but report.json was not created.")

        try:
            report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid report JSON: {e}") from e

        payload = _build_done_payload(report)
        with _state_lock:
            _pipeline_status = "done"
            _error_message = None
            _result_payload = payload
        print("[API] Processing completed")
    except Exception as e:
        msg = str(e) or type(e).__name__
        print(f"[API] Processing failed: {msg}")
        err_body = {"status": "error", "message": msg}
        with _state_lock:
            _pipeline_status = "error"
            _error_message = msg
            _result_payload = err_body
    finally:
        try:
            saved_path.unlink(missing_ok=True)
        except OSError:
            pass


@app.get("/status")
def get_status() -> JSONResponse:
    """Pipeline state for polling."""
    with _state_lock:
        if _pipeline_status == "error":
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "message": _error_message or "Unknown error",
                },
            )
        return JSONResponse(content={"status": _pipeline_status})


@app.get("/report")
def get_report() -> JSONResponse:
    """Report payload for the last completed pipeline run."""
    with _state_lock:
        if _pipeline_status == "processing":
            return JSONResponse(content={"status": "processing"})
        if _pipeline_status == "error":
            payload = JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "message": _error_message or "Unknown error",
                }
            )
            _reset_to_idle_unlocked()
            return payload
        if _pipeline_status == "done" and _result_payload is not None:
            payload = JSONResponse(content=dict(_result_payload))
            _reset_to_idle_unlocked()
            return payload
        # idle
        return JSONResponse(content={"status": "idle"})


@app.post("/analyze")
async def analyze(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
) -> JSONResponse:
    """Upload a video and queue CLI processing in the background."""
    global _pipeline_status, _error_message, _result_payload

    if not file.filename:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": "Missing filename"},
        )

    base_name = os.path.basename(file.filename)
    suffix = Path(base_name).suffix.lower() or ".mp4"
    if suffix not in ALLOWED_VIDEO_EXT:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": "Unsupported video format."},
        )

    cleanup_stale_uploads()

    with _state_lock:
        if _pipeline_status == "processing":
            raise HTTPException(
                status_code=409,
                detail="Analysis already in progress. Wait or poll /status.",
            )
        if _pipeline_status != "idle":
            raise HTTPException(
                status_code=409,
                detail="Wait for the current session to finish (poll /status or /report).",
            )
        _pipeline_status = "processing"
        _error_message = None
        _result_payload = None

    ts = int(time.time() * 1000)
    short_id = uuid.uuid4().hex[:8]
    dest = (UPLOAD_DIR / f"{ts}_{short_id}{suffix}").resolve()
    try:
        dest.relative_to(UPLOAD_DIR.resolve())
    except ValueError:
        with _state_lock:
            _reset_to_idle_unlocked()
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": "Invalid upload path."},
        )

    try:
        content = await file.read()
        if not content:
            with _state_lock:
                _reset_to_idle_unlocked()
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "Empty file."},
            )
        dest.write_bytes(content)
    except OSError as e:
        with _state_lock:
            _pipeline_status = "error"
            _error_message = str(e)
            _result_payload = {"status": "error", "message": str(e)}
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Could not save upload: {e}"},
        )
    finally:
        await file.close()

    background_tasks.add_task(run_pipeline, dest)
    return JSONResponse(content={"status": "processing"})


@app.get("/download/video")
def download_video() -> FileResponse:
    if not OUTPUT_VIDEO.is_file():
        raise HTTPException(status_code=404, detail="No processed video.")
    return FileResponse(
        path=str(OUTPUT_VIDEO),
        media_type="video/mp4",
        filename="roadguard_output.mp4",
    )
