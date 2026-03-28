"""
Lightweight RandomForest risk classifier (LOW / MEDIUM / HIGH).

Loads the committed model artifact from `roadguard_x/models/risk_model.pkl` (joblib).
Falls back to legacy `roadguard_x/ml/models/risk_rf.pkl` if needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np

RISK_LABELS = ("LOW", "MEDIUM", "HIGH")


def _default_model_path() -> Path:
    root = Path(__file__).resolve().parent.parent
    preferred = root / "models" / "risk_model.pkl"
    legacy = root / "ml" / "models" / "risk_rf.pkl"
    if preferred.is_file():
        return preferred
    return legacy


class RiskModel:
    """Wrapper around trained RandomForestClassifier."""

    def __init__(self, model_path: Path | None = None) -> None:
        self._path = model_path or _default_model_path()
        payload = joblib.load(self._path)
        self._clf = payload["classifier"]
        self.feature_names: Tuple[str, ...] = tuple(payload.get("feature_names", ()))
        self._scaler: Optional[Any] = payload.get("scaler")

    def _top_feature_contributions(self) -> List[Tuple[str, float]]:
        """RandomForest global feature importances, top-3 by magnitude."""
        names: Sequence[str] = self.feature_names
        importances = self._clf.feature_importances_
        if not names or len(names) != len(importances):
            names = tuple(f"f{i}" for i in range(len(importances)))
        contributions = dict(zip(names, importances))
        top_contrib = sorted(contributions.items(), key=lambda x: x[1], reverse=True)[:3]
        return top_contrib

    def predict(
        self, features: np.ndarray
    ) -> Tuple[str, float, List[Tuple[str, float]]]:
        """
        Args:
            features: shape (n_features,) or (1, n_features).

        Returns:
            risk_label, confidence (probability of predicted class), top-3 (name, importance).
        """
        x = np.atleast_2d(features.astype(np.float64))
        if self._scaler is not None:
            x = self._scaler.transform(x)
        proba = self._clf.predict_proba(x)[0]
        pred = int(np.argmax(proba))
        label = RISK_LABELS[pred]
        confidence = float(proba[pred])
        top_contrib = self._top_feature_contributions()
        return label, confidence, top_contrib

    def predict_batch(
        self, rows: Iterable[np.ndarray]
    ) -> List[Tuple[str, float, List[Tuple[str, float]]]]:
        out = []
        for r in rows:
            out.append(self.predict(r))
        return out
