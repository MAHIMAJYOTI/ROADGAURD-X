"""Danger clip extraction utility."""

from __future__ import annotations

import functools
import os
import shutil
import subprocess
from collections import deque
from pathlib import Path
from typing import Deque, List, Tuple

import cv2
import numpy as np

_FFMPEG_MISSING_WARNED = False
_FFMPEG_ENCODE_FAIL_WARNED = False


@functools.lru_cache(maxsize=1)
def resolve_ffmpeg_executable() -> str | None:
    """
    Return path to ffmpeg, or None.

    On Windows, winget may install Gyan FFmpeg without putting `ffmpeg` on PATH for all
    processes; we probe PATH first, then common install locations.
    """
    for name in ("ffmpeg", "ffmpeg.exe"):
        w = shutil.which(name)
        if w:
            return w
    if os.name != "nt":
        return None
    local = os.environ.get("LOCALAPPDATA", "")
    program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
    candidates = [
        Path(local) / "Microsoft" / "WinGet" / "Links" / "ffmpeg.exe",
        Path(program_files) / "ffmpeg" / "bin" / "ffmpeg.exe",
        Path(r"C:\ffmpeg\bin\ffmpeg.exe"),
        Path(local) / "Programs" / "ffmpeg" / "bin" / "ffmpeg.exe",
    ]
    for p in candidates:
        if p.is_file():
            return str(p)
    wg_pkgs = Path(local) / "Microsoft" / "WinGet" / "Packages"
    if wg_pkgs.is_dir():
        try:
            for p in wg_pkgs.rglob("ffmpeg.exe"):
                if p.is_file():
                    return str(p)
        except OSError:
            pass
    return None


def get_video_writer(
    path: Path,
    fps: float,
    frame_size: Tuple[int, int],
) -> cv2.VideoWriter:
    """
    Create a VideoWriter using mp4v codec.

    Re-encode to H264 via ffmpeg after writing.
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), frame_size)
    if writer.isOpened():
        return writer
    raise RuntimeError(f"Could not open VideoWriter for {path}")


def reencode_h264(input_path: Path) -> bool:
    """Re-encode mp4v file to H264 using ffmpeg for browser compatibility. Returns True if OK."""
    global _FFMPEG_MISSING_WARNED, _FFMPEG_ENCODE_FAIL_WARNED
    exe = resolve_ffmpeg_executable()
    if not exe:
        if not _FFMPEG_MISSING_WARNED:
            print(
                "Warning: ffmpeg not found (not on PATH and no common Windows install). "
                "Videos stay MPEG-4 Part 2 (mp4v) and may not play in browsers. "
                "Install FFmpeg, then open a new terminal and run: ffmpeg -version"
            )
            _FFMPEG_MISSING_WARNED = True
        return False

    tmp = str(input_path).replace(".mp4", "_tmp.mp4")
    try:
        result = subprocess.run(
            [
                exe,
                "-y",
                "-i",
                str(input_path),
                "-vcodec",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-an",
                tmp,
            ],
            capture_output=True,
            timeout=120,
            text=True,
        )
        if result.returncode == 0:
            os.replace(tmp, str(input_path))
            return True
        if os.path.exists(tmp):
            os.remove(tmp)
        if not _FFMPEG_ENCODE_FAIL_WARNED:
            err = (result.stderr or result.stdout or "").strip()
            tail = err[-800:] if err else "(no stderr)"
            print(
                "Warning: ffmpeg re-encode failed; browser may not play this MP4. "
                f"Last stderr lines:\n{tail}"
            )
            _FFMPEG_ENCODE_FAIL_WARNED = True
        return False
    except subprocess.TimeoutExpired:
        if os.path.exists(tmp):
            os.remove(tmp)
        if not _FFMPEG_ENCODE_FAIL_WARNED:
            print("Warning: ffmpeg re-encode timed out; video may not play in browser.")
            _FFMPEG_ENCODE_FAIL_WARNED = True
        return False
    except OSError as e:
        if not _FFMPEG_MISSING_WARNED:
            print(f"Warning: could not run ffmpeg ({e!r}). Videos may not play in browser.")
            _FFMPEG_MISSING_WARNED = True
        return False


class ClipWriter:
    """Extract standalone clips around HIGH-risk segments."""

    def __init__(
        self,
        clips_dir: Path,
        fps: float,
        frame_size: Tuple[int, int],
        enabled: bool,
        context_seconds: float = 2.0,
    ) -> None:
        """Initialize clip state and output directory settings."""
        self.clips_dir = clips_dir
        self.fps = float(max(fps, 1.0))
        self.frame_size = (int(frame_size[0]), int(frame_size[1]))
        self.enabled = bool(enabled)
        self.context_seconds = float(max(context_seconds, 0.0))
        self.context_frames = int(round(self.fps * self.context_seconds))
        self.pre_buffer: Deque[np.ndarray] = deque(maxlen=max(1, self.context_frames))
        self.segment_frames: List[np.ndarray] = []
        self.post_frames: List[np.ndarray] = []
        self.state: str = "idle"  # idle | high | post
        self.clip_count: int = 0

    def update(self, frame: np.ndarray, risk_label: str) -> str | None:
        """Update extraction state with the current frame and risk label."""
        if not self.enabled:
            return None
        snap = frame.copy()
        self.pre_buffer.append(snap)
        is_high = risk_label == "HIGH"

        if self.state == "idle":
            if is_high:
                self.segment_frames = list(self.pre_buffer)
                self.state = "high"
            return None

        if self.state == "high":
            self.segment_frames.append(snap)
            if not is_high:
                self.state = "post"
                self.post_frames = []
            return None

        # post state
        if is_high:
            # Merge back into same segment to avoid splitting jittery transitions.
            self.segment_frames.extend(self.post_frames)
            self.segment_frames.append(snap)
            self.post_frames = []
            self.state = "high"
            return None

        self.post_frames.append(snap)
        if len(self.post_frames) >= self.context_frames:
            return self._finalize_segment()
        return None

    def flush_remaining(self) -> str | None:
        """Flush any active segment at end-of-run."""
        if not self.enabled:
            return None
        if self.state == "idle":
            return None
        return self._finalize_segment()

    def _finalize_segment(self) -> str | None:
        """Write active segment to disk and reset transient state."""
        frames = self.segment_frames + self.post_frames
        self.segment_frames = []
        self.post_frames = []
        self.state = "idle"

        if not frames:
            return "Warning: danger clip extraction had no frames to save."

        self.clips_dir.mkdir(parents=True, exist_ok=True)
        self.clip_count += 1
        out_path = self.clips_dir / f"clip_{self.clip_count:03d}.mp4"
        try:
            writer = get_video_writer(
                out_path, self.fps, (self.frame_size[0], self.frame_size[1])
            )
        except RuntimeError:
            self.clip_count -= 1
            return f"Warning: could not open clip writer for {out_path.name}."

        for fr in frames:
            if fr.shape[1] != self.frame_size[0] or fr.shape[0] != self.frame_size[1]:
                fr = cv2.resize(fr, self.frame_size)
            writer.write(fr)
        writer.release()
        reencode_h264(out_path)
        return f"Saved danger clip: {out_path}"
