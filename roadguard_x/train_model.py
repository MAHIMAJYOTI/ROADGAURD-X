"""Train a calibrated synthetic RandomForest model for RoadGuard-X."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split

from modules.features import FEATURE_NAMES

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "output"
MODEL_PATH = MODELS_DIR / "risk_model.pkl"
META_PATH = MODELS_DIR / "training_metadata.json"
CONFUSION_MATRIX_PATH = OUTPUT_DIR / "confusion_matrix.png"
CLASSIFICATION_REPORT_PATH = OUTPUT_DIR / "classification_report.txt"

LABELS = ("LOW", "MEDIUM", "HIGH")
LOW, MEDIUM, HIGH = 0, 1, 2

RF_HYPERPARAMETERS = {
    "n_estimators": 100,
    "max_depth": 8,
    "class_weight": "balanced",
    "random_state": 42,
}

TRAIN_TEST_SPLIT = {
    "test_size": 0.2,
    "random_state": 42,
    "stratify": True,
}


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


def _evaluate_classifier(
    clf: RandomForestClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, object]:
    """Compute held-out test metrics and confusion matrix."""
    y_pred = clf.predict(X_test)
    accuracy = float(accuracy_score(y_test, y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        y_pred,
        average=None,
        labels=[LOW, MEDIUM, HIGH],
        zero_division=0,
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test,
        y_pred,
        average="macro",
        labels=[LOW, MEDIUM, HIGH],
        zero_division=0,
    )
    cm = confusion_matrix(y_test, y_pred, labels=[LOW, MEDIUM, HIGH])

    per_class: Dict[str, Dict[str, float]] = {}
    for idx, label in enumerate(LABELS):
        per_class[label] = {
            "precision": round(float(precision[idx]), 4),
            "recall": round(float(recall[idx]), 4),
            "f1": round(float(f1[idx]), 4),
        }

    return {
        "accuracy": round(accuracy, 4),
        "precision_macro": round(float(precision_macro), 4),
        "recall_macro": round(float(recall_macro), 4),
        "f1_macro": round(float(f1_macro), 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def _save_confusion_matrix_plot(cm: np.ndarray, path: Path) -> None:
    """Write a labeled confusion-matrix heatmap for portfolio / dashboard use."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(LABELS))
    ax.set(
        xticks=tick_marks,
        yticks=tick_marks,
        xticklabels=LABELS,
        yticklabels=LABELS,
        ylabel="True label",
        xlabel="Predicted label",
        title="RoadGuard-X Random Forest — Test Set Confusion Matrix",
    )
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(int(cm[i, j]), "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=12,
            )
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_classification_report(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    path: Path,
    *,
    n_train: int,
    n_test: int,
) -> None:
    """Persist sklearn classification report plus split summary."""
    path.parent.mkdir(parents=True, exist_ok=True)
    report = classification_report(y_test, y_pred, target_names=LABELS, digits=4)
    header = (
        "RoadGuard-X Random Forest — Held-out Test Set Evaluation\n"
        f"Train samples: {n_train}\n"
        f"Test samples: {n_test}\n"
        f"Split: {int((1 - TRAIN_TEST_SPLIT['test_size']) * 100)}% train / "
        f"{int(TRAIN_TEST_SPLIT['test_size'] * 100)}% test (stratified)\n"
        f"Hyperparameters: {json.dumps(RF_HYPERPARAMETERS)}\n"
        "\n"
    )
    path.write_text(header + report, encoding="utf-8")


def main() -> None:
    """Train model, evaluate on held-out test set, and save model + metadata."""
    X, y = build_synthetic_dataset(n_samples=5000, seed=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TRAIN_TEST_SPLIT["test_size"],
        random_state=TRAIN_TEST_SPLIT["random_state"],
        stratify=y,
    )

    clf = RandomForestClassifier(**RF_HYPERPARAMETERS)
    clf.fit(X_train, y_train)

    y_train_pred = clf.predict(X_train)
    print("=== Classification report (train set) ===")
    print(classification_report(y_train, y_train_pred, target_names=LABELS, digits=4))

    y_test_pred = clf.predict(X_test)
    print("=== Classification report (held-out test set) ===")
    print(classification_report(y_test, y_test_pred, target_names=LABELS, digits=4))

    evaluation = _evaluate_classifier(clf, X_test, y_test)
    cm = np.asarray(evaluation["confusion_matrix"], dtype=np.int64)
    _save_confusion_matrix_plot(cm, CONFUSION_MATRIX_PATH)
    _save_classification_report(
        y_test,
        y_test_pred,
        CLASSIFICATION_REPORT_PATH,
        n_train=len(y_train),
        n_test=len(y_test),
    )

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

    print("=== Test metrics (summary) ===")
    print(f"Accuracy:  {evaluation['accuracy']:.4f}")
    print(f"Precision: {evaluation['precision_macro']:.4f} (macro)")
    print(f"Recall:    {evaluation['recall_macro']:.4f} (macro)")
    print(f"F1-score:  {evaluation['f1_macro']:.4f} (macro)")

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
        "hyperparameters": dict(RF_HYPERPARAMETERS),
        "train_test_split": {
            "test_size": TRAIN_TEST_SPLIT["test_size"],
            "random_state": TRAIN_TEST_SPLIT["random_state"],
            "stratify": TRAIN_TEST_SPLIT["stratify"],
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
        },
        "evaluation": evaluation,
        "evaluation_artifacts": {
            "confusion_matrix_png": "output/confusion_matrix.png",
            "classification_report_txt": "output/classification_report.txt",
        },
    }
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved metadata: {META_PATH}")
    print(f"Saved confusion matrix: {CONFUSION_MATRIX_PATH}")
    print(f"Saved classification report: {CLASSIFICATION_REPORT_PATH}")


if __name__ == "__main__":
    main()
