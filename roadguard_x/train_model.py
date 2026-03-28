"""Train a calibrated synthetic RandomForest model for RoadGuard-X."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from modules.features import FEATURE_NAMES

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "risk_model.pkl"
META_PATH = MODELS_DIR / "training_metadata.json"

LABELS = ("LOW", "MEDIUM", "HIGH")
LOW, MEDIUM, HIGH = 0, 1, 2


def _gen_bucket(
    rng: np.random.Generator, count: int, bucket: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate one class bucket with realistic feature relationships."""
    X = np.zeros((count, len(FEATURE_NAMES)), dtype=np.float64)
    y = np.full(count, bucket, dtype=np.int32)

    if bucket == LOW:
        X[:, 0] = rng.uniform(0.00, 0.20, count)  # lane_offset_norm
        X[:, 1] = rng.integers(0, 3, count).astype(np.float64)  # object_count
        X[:, 2] = rng.uniform(0.05, 0.35, count)  # avg_object_area (normalized proxy)
        X[:, 3] = rng.uniform(0.25, 0.65, count)  # edge_density
        X[:, 4] = rng.uniform(0.40, 0.80, count)  # brightness
        X[:, 5] = rng.uniform(0.00, 0.30, count)  # avg_speed
        X[:, 6] = rng.uniform(0.00, 0.25, count)  # motion_variance
        X[:, 7] = rng.uniform(0.70, 1.00, count)  # direction_consistency
        X[:, 8] = rng.uniform(0.00, 0.35, count)  # scene_complexity_index
    elif bucket == MEDIUM:
        X[:, 0] = rng.uniform(0.15, 0.40, count)
        X[:, 1] = rng.integers(1, 5, count).astype(np.float64)
        X[:, 2] = rng.uniform(0.20, 0.55, count)
        X[:, 3] = rng.uniform(0.20, 0.75, count)
        X[:, 4] = rng.uniform(0.25, 0.85, count)
        X[:, 5] = rng.uniform(0.20, 0.50, count)
        X[:, 6] = rng.uniform(0.20, 0.60, count)
        X[:, 7] = rng.uniform(0.35, 0.90, count)
        X[:, 8] = rng.uniform(0.30, 0.60, count)
    else:
        X[:, 0] = rng.uniform(0.20, 0.80, count)
        X[:, 1] = rng.integers(3, 10, count).astype(np.float64)
        X[:, 2] = rng.uniform(0.35, 0.95, count)
        X[:, 3] = rng.uniform(0.35, 1.00, count)
        X[:, 4] = rng.uniform(0.10, 0.75, count)
        X[:, 5] = rng.uniform(0.35, 1.00, count)
        X[:, 6] = rng.uniform(0.45, 1.00, count)
        X[:, 7] = rng.uniform(0.05, 0.70, count)
        X[:, 8] = rng.uniform(0.55, 1.00, count)

        # Enforce "at least one elevated trigger" for HIGH risk.
        triggers = np.column_stack(
            [
                X[:, 0] > 0.35,
                X[:, 1] > 4.0,
                X[:, 5] > 0.5,
                X[:, 6] > 0.6,
                X[:, 8] > 0.6,
            ]
        )
        none = np.where(triggers.sum(axis=1) == 0)[0]
        for i in none:
            sel = int(rng.integers(0, 5))
            if sel == 0:
                X[i, 0] = rng.uniform(0.36, 0.65)
            elif sel == 1:
                X[i, 1] = rng.uniform(4.5, 8.0)
            elif sel == 2:
                X[i, 5] = rng.uniform(0.55, 1.0)
            elif sel == 3:
                X[i, 6] = rng.uniform(0.65, 1.0)
            else:
                X[i, 8] = rng.uniform(0.65, 1.0)

    return X, y


def build_synthetic_dataset(n_samples: int = 5000, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a noise-perturbed synthetic dataset with class priors."""
    rng = np.random.default_rng(seed)
    n_low = int(n_samples * 0.50)
    n_med = int(n_samples * 0.35)
    n_high = n_samples - n_low - n_med

    x_low, y_low = _gen_bucket(rng, n_low, LOW)
    x_med, y_med = _gen_bucket(rng, n_med, MEDIUM)
    x_high, y_high = _gen_bucket(rng, n_high, HIGH)

    X = np.vstack([x_low, x_med, x_high])
    y = np.concatenate([y_low, y_med, y_high])

    # Add Gaussian noise to avoid perfectly separable clusters.
    noise_scale = np.array([0.03, 0.35, 0.05, 0.04, 0.04, 0.05, 0.06, 0.05, 0.05], dtype=np.float64)
    X += rng.normal(0.0, noise_scale, size=X.shape)

    # Clip normalized features to [0,1], counts to [0,+).
    normalized_idx = [0, 2, 3, 4, 5, 6, 7, 8]
    X[:, normalized_idx] = np.clip(X[:, normalized_idx], 0.0, 1.0)
    X[:, 1] = np.clip(X[:, 1], 0.0, None)

    perm = rng.permutation(len(X))
    return X[perm], y[perm]


def main() -> None:
    """Train model, print diagnostics, and save model + metadata."""
    X, y = build_synthetic_dataset(n_samples=5000, seed=42)
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        class_weight="balanced",
        random_state=42,
    )
    clf.fit(X, y)

    y_pred = clf.predict(X)
    print("=== Classification report (train set) ===")
    print(classification_report(y, y_pred, target_names=LABELS, digits=4))

    importances = dict(
        sorted(
            ((name, float(val)) for name, val in zip(FEATURE_NAMES, clf.feature_importances_)),
            key=lambda x: x[1],
            reverse=True,
        )
    )
    print("=== Feature importances ===")
    for k, v in importances.items():
        print(f"{k:>24}: {v:.4f}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "classifier": clf,
        "feature_names": FEATURE_NAMES,
        "scaler": None,
        "schema_version": 4,
        "trained_on_real": False,
    }
    joblib.dump(payload, MODEL_PATH)

    counts = Counter(y.tolist())
    meta: Dict[str, object] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_samples": int(len(y)),
        "class_distribution": {
            "LOW": int(counts.get(LOW, 0)),
            "MEDIUM": int(counts.get(MEDIUM, 0)),
            "HIGH": int(counts.get(HIGH, 0)),
        },
        "feature_importances": importances,
        "sklearn_version": sklearn.__version__,
    }
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved metadata: {META_PATH}")


if __name__ == "__main__":
    main()
