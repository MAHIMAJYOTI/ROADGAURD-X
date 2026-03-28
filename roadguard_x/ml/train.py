"""
Train RandomForest risk model from synthetic data and/or data/dataset.csv.

Run from `roadguard_x/`:
  python -m ml.train
  python -m ml.train --from-csv
  python -m ml.train --generate-sample
  python -m ml.train --generate-real-samples
"""

from __future__ import annotations

import argparse
import csv
import pickle
import sys
from pathlib import Path
from typing import Tuple

import numpy as np

# Project root (parent of `ml/`)
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from modules.features import FEATURE_NAMES, _SCI_DENOM  # noqa: E402

_RISK_TO_INT = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}


def _label_from_rules(X: np.ndarray) -> np.ndarray:
    """
    Deterministic weak-labeler: maps feature rows to 0=LOW, 1=MEDIUM, 2=HIGH.
    """
    n = X.shape[0]
    y = np.zeros(n, dtype=np.int32)
    for i in range(n):
        lane = X[i, 0]
        nobj = X[i, 1]
        bright = X[i, 4]
        spd = X[i, 5]
        mvar = X[i, 6]
        dcons = X[i, 7]
        edge = X[i, 3]
        sci = X[i, 8]

        score = 0
        if lane > 0.1:
            score += 3
        if nobj > 12:
            score += 2
        if bright < 0.28:
            score += 2
        if spd > 18:
            score += 3
        if edge > 0.42:
            score += 1
        if mvar > 25 and dcons < 0.35:
            score += 2
        if sci > 0.65:
            score += 1

        if score >= 6:
            y[i] = 2
        elif score >= 2:
            y[i] = 1
        else:
            y[i] = 0
    return y


def build_synthetic_dataset(n_samples: int = 1500, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Generate random feature vectors and rule-based labels (reproducible)."""
    rng = np.random.default_rng(seed)
    X = np.zeros((n_samples, len(FEATURE_NAMES)), dtype=np.float64)
    X[:, 0] = rng.uniform(0.0, 0.22, n_samples)
    X[:, 1] = rng.uniform(0.0, 20.0, n_samples)
    X[:, 2] = rng.uniform(50.0, 9000.0, n_samples)
    X[:, 3] = rng.uniform(0.05, 0.55, n_samples)
    X[:, 4] = rng.uniform(0.12, 0.95, n_samples)
    X[:, 5] = rng.uniform(0.0, 35.0, n_samples)
    X[:, 6] = rng.uniform(0.0, 90.0, n_samples)
    X[:, 7] = rng.uniform(0.05, 1.0, n_samples)
    raw_sci = 0.5 * X[:, 3] + 0.3 * X[:, 1] + 0.2 * X[:, 6]
    X[:, 8] = np.clip(raw_sci / _SCI_DENOM, 0.0, 1.0)

    y = _label_from_rules(X)
    return X, y


def load_dataset_csv(csv_path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Load rows from dataset.csv; returns None if missing or invalid."""
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return None
    rows: list[list[float]] = []
    labels: list[int] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "risk" not in reader.fieldnames:
            return None
        for name in FEATURE_NAMES:
            if name not in reader.fieldnames:
                return None
        for row in reader:
            try:
                feats = [float(row[k]) for k in FEATURE_NAMES]
                rk = row["risk"].strip().upper()
                if rk not in _RISK_TO_INT:
                    continue
                rows.append(feats)
                labels.append(_RISK_TO_INT[rk])
            except (KeyError, ValueError):
                continue
    if len(rows) < 3:
        return None
    return np.asarray(rows, dtype=np.float64), np.asarray(labels, dtype=np.int32)


def train_and_save(
    model_path: Path,
    *,
    csv_path: Path | None = None,
    prefer_csv: bool = True,
    require_csv: bool = False,
) -> None:
    """
    Fit RandomForest on normalized features; persist scaler + classifier.

    If ``prefer_csv`` and ``dataset.csv`` loads successfully, uses those rows
    (optionally augmented with synthetic samples when the set is small).
    If ``require_csv`` is True, missing/invalid CSV aborts training.
    Otherwise falls back to synthetic data only.
    """
    X: np.ndarray
    y: np.ndarray
    used_real = False

    loaded = load_dataset_csv(csv_path) if (csv_path and prefer_csv) else None
    if require_csv and loaded is None:
        raise SystemExit(
            f"No valid dataset at {csv_path}. Collect data first or use default training."
        )
    if loaded is not None:
        X, y = loaded
        used_real = True
        if len(X) < 80:
            X_s, y_s = build_synthetic_dataset(n_samples=400, seed=43)
            X = np.vstack([X, X_s])
            y = np.concatenate([y, y_s])
    else:
        X, y = build_synthetic_dataset()

    scaler = StandardScaler()
    Xn = scaler.fit_transform(X)

    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        class_weight="balanced",
        random_state=42,
    )
    clf.fit(Xn, y)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "classifier": clf,
        "feature_names": FEATURE_NAMES,
        "scaler": scaler,
        "schema_version": 3,
        "trained_on_real": used_real,
    }
    with open(model_path, "wb") as f:
        pickle.dump(payload, f)


def generate_sample_video(out_path: Path, frames: int = 180, fps: float = 15.0) -> None:
    """Create a small synthetic driving-like clip for demos (< 5MB target)."""
    import cv2

    out_path.parent.mkdir(parents=True, exist_ok=True)
    w, h = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError("Could not open VideoWriter for sample generation.")

    for i in range(frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[: int(h * 0.55), :] = (220, 180, 140)
        cv2.rectangle(frame, (0, int(h * 0.55)), (w, h), (50, 50, 55), -1)
        shift = int(8 * np.sin(i * 0.08))
        mid = w // 2 + shift
        pts_left = np.array(
            [[mid - 160, h - 30], [mid - 40, int(h * 0.58)]], np.int32
        )
        pts_right = np.array(
            [[mid + 160, h - 30], [mid + 40, int(h * 0.58)]], np.int32
        )
        cv2.polylines(frame, [pts_left], False, (240, 240, 245), 3)
        cv2.polylines(frame, [pts_right], False, (240, 240, 245), 3)
        bx = int(w * 0.45 + 40 * np.sin(i * 0.12))
        by = int(h * 0.62 + (i % 40))
        cv2.rectangle(frame, (bx, by), (bx + 70, by + 50), (30, 80, 200), -1)
        glare = int(25 * np.sin(i * 0.2))
        frame[:, :] = np.clip(frame.astype(np.int16) + glare, 0, 255).astype(np.uint8)
        writer.write(frame)

    writer.release()


def generate_road_variant(
    out_path: Path,
    *,
    phase: float,
    road_tint: Tuple[int, int, int],
    frames: int = 90,
    fps: float = 15.0,
) -> None:
    """Short ~6s clip (90 frames @ 15fps); distinct look for multi-sample rotation."""
    import cv2

    out_path.parent.mkdir(parents=True, exist_ok=True)
    w, h = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError("Could not open VideoWriter for road variant.")

    for i in range(frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        sky = (200 + int(15 * np.sin(phase + i * 0.02)), 175, 130)
        frame[: int(h * 0.55), :] = sky
        cv2.rectangle(frame, (0, int(h * 0.55)), (w, h), road_tint, -1)
        shift = int(10 * np.sin(phase + i * 0.09))
        mid = w // 2 + shift
        pts_left = np.array(
            [[mid - 150, h - 28], [mid - 35, int(h * 0.56)]], np.int32
        )
        pts_right = np.array(
            [[mid + 150, h - 28], [mid + 35, int(h * 0.56)]], np.int32
        )
        cv2.polylines(frame, [pts_left], False, (240, 245, 250), 3)
        cv2.polylines(frame, [pts_right], False, (240, 245, 250), 3)
        bx = int(w * 0.42 + 45 * np.sin(phase * 0.5 + i * 0.11))
        by = int(h * 0.6 + (i * 0.35) % 45)
        cv2.rectangle(frame, (bx, by), (bx + 65, by + 48), (25, 70, 210), -1)
        writer.write(frame)

    writer.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RoadGuard-X risk RF model.")
    parser.add_argument(
        "--generate-sample",
        action="store_true",
        help="Write samples/sample.mp4 synthetic clip.",
    )
    parser.add_argument(
        "--generate-real-samples",
        action="store_true",
        help="Write samples/road_real_1.mp4 and road_real_2.mp4 (short clips).",
    )
    parser.add_argument(
        "--from-csv",
        action="store_true",
        help="Prefer data/dataset.csv when training (if present and valid).",
    )
    parser.add_argument(
        "--dataset-out",
        type=Path,
        default=_ROOT / "data" / "dataset.csv",
        help="CSV path for --from-csv.",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=_ROOT / "ml" / "models" / "risk_rf.pkl",
        help="Output path for pickled model.",
    )
    args = parser.parse_args()

    if args.generate_sample:
        sample_path = _ROOT / "samples" / "sample.mp4"
        generate_sample_video(sample_path)
        print(f"Wrote sample video to {sample_path}")

    if args.generate_real_samples:
        samples_dir = _ROOT / "samples"
        generate_road_variant(
            samples_dir / "road_real_1.mp4", phase=0.3, road_tint=(45, 48, 52)
        )
        generate_road_variant(
            samples_dir / "road_real_2.mp4", phase=1.7, road_tint=(42, 46, 50)
        )
        print(f"Wrote {samples_dir / 'road_real_1.mp4'} and road_real_2.mp4")

    # Default: try CSV if present; --from-csv requires a valid dataset file
    train_and_save(
        args.model_out,
        csv_path=args.dataset_out,
        prefer_csv=True,
        require_csv=args.from_csv,
    )
    size_kb = args.model_out.stat().st_size / 1024
    print(f"Saved model to {args.model_out} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
