"""
Append feature vectors and risk labels to data/dataset.csv for ML retraining.

CSV columns: FEATURE_NAMES..., risk (append-only; header written once).
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from modules.features import FEATURE_NAMES


def append_dataset_row(dataset_path: Path, features: np.ndarray, risk_label: str) -> None:
    """
    Append one row. Creates parent dirs. Writes header if file missing or empty.

    Args:
        dataset_path: e.g. data/dataset.csv
        features: 1-D vector matching FEATURE_NAMES order.
        risk_label: LOW | MEDIUM | HIGH
    """
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    feats = np.asarray(features, dtype=np.float64).flatten()
    if feats.size != len(FEATURE_NAMES):
        raise ValueError(
            f"Expected {len(FEATURE_NAMES)} features, got {feats.size}"
        )
    row = [f"{float(feats[i]):.8g}" for i in range(len(FEATURE_NAMES))] + [risk_label]

    write_header = not dataset_path.exists() or dataset_path.stat().st_size == 0

    with open(dataset_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(list(FEATURE_NAMES) + ["risk"])
        w.writerow(row)


def wipe_dataset(dataset_path: Path) -> None:
    """Remove dataset file (next run will recreate header)."""
    if dataset_path.exists():
        dataset_path.unlink()
