# RoadGuard-X — Context-Aware Driving Intelligence System

**RoadGuard-X** is a **terminal-driven**, **offline** driving scene analyzer. It combines **classical computer vision**, a **lightweight scikit-learn RandomForest** model, and **temporal reasoning** over a short frame buffer to estimate **LOW / MEDIUM / HIGH** risk and produce **human-readable explanations** plus a **JSON session report**.

There are **no deep learning frameworks**, **no cloud APIs**, **no `.env` files**, and **no large pretrained weights**. After `pip install`, the system runs **fully offline**.

---

## 1. Project overview

Given a **webcam** or **sample video**, RoadGuard-X:

1. **Preprocesses** each frame (blur, CLAHE, gamma).
2. **Detects lanes** (ROI → Canny → Hough) and estimates **lane center offset**.
3. **Segments** a downscaled view with **K-means** (+ morphological cleanup; optional watershed helpers in code).
4. **Tracks moving objects** with **MOG2** background subtraction, **contours**, and **ID assignment** with centroid association.
5. Builds **spatial + temporal features** (lane offset, object statistics, edge density, brightness, motion variance, **scipy circular** direction stability).
6. Runs a **RandomForestClassifier** loaded from `ml/models/risk_rf.pkl`.
7. **Explains** risk via thresholded rules aligned with the feature schema.
8. **Overlays** a HUD (lanes, boxes, motion arrows, risk banner, reasons).
9. Writes **`output/report.json`** when the run ends.

---

## 2. Features

| Area | Capability |
|------|------------|
| **Lanes** | Trapezoid ROI, Canny edges, probabilistic Hough lines, normalized center offset |
| **Segmentation** | K-means (k=3–4) on LAB, open/close morphology, optional watershed refinement |
| **Tracking** | MOG2, contour filtering, greedy ID matching, per-object velocity / acceleration |
| **Temporal** | `deque` of recent `FrameTracks`, speed history, direction history, buffer-based variance |
| **ML** | `RandomForestClassifier` (8-D features), labels `LOW` / `MEDIUM` / `HIGH` |
| **Explainability** | Rule-based reasons (e.g. lane drift, low visibility, rapid motion) |
| **Reporting** | JSON: frame count, avg objects, risk distribution, lane drift events |

---

## 3. Architecture (text diagram)

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│  Video src  │────▶│ preprocess   │────▶│ lane        │────▶│ features     │
│ (web/sample)│     │ blur+CLAHE+γ │     │ ROI+Canny+  │     │ spatial+     │
└─────────────┘     └──────────────┘     │ Hough       │     │ temporal     │
       │                      │          └─────────────┘            │
       │                      │                 │                    │
       │                      ▼                 ▼                    ▼
       │               ┌──────────────┐  ┌─────────────┐    ┌──────────────┐
       └──────────────▶│ segmentation │  │ tracker     │───▶│ risk_model   │
                       │ K-means+morph│  │ MOG2+tracks │    │ RandomForest │
                       └──────────────┘  └─────────────┘    └──────┬───────┘
                                                                   │
                       ┌──────────────┐    ┌─────────────┐          │
                       │ hud          │◀───│ explain     │◀─────────┘
                       │ overlay      │    │ reasons     │
                       └──────────────┘    └─────────────┘
                              │
                              ▼
                       ┌──────────────┐
                       │ reporter     │──▶ output/report.json
                       └──────────────┘
```

---

## 4. Installation

**Requirements:** Python 3.10+ recommended (tested with 3.14 on Windows).

```bash
cd roadguard_x
python -m venv .venv
```

**Windows (PowerShell):**

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Linux / macOS:**

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

**Dependencies (only these):**

- `opencv-python`
- `numpy`
- `scikit-learn`
- `scipy`

If your global `pip install` hits permission errors on Windows, use a **virtual environment** (as above).

---

## 5. How to run

From the **`roadguard_x`** directory:

```bash
python main.py --source sample
```

- **`--source webcam`** — live camera; if it cannot be opened, the app **falls back** to `samples/sample.mp4`.
- **`--source sample`** — uses `samples/sample.mp4`.
- **`--headless`** — no OpenCV window; still processes video and writes `output/report.json`.
- **`--max-frames N`** — stop after `N` frames (useful for CI or quick tests).

### Interactive keys (when not `--headless`)

| Key | Action |
|-----|--------|
| **q** | Quit |
| **s** | Save current HUD frame to `output/frame_<frame>_<n>.jpg` |

---

## 6. Sample commands

```bash
# Default demo (sample video + on-screen HUD)
python main.py --source sample

# Headless batch (no display)
python main.py --source sample --headless --max-frames 200

# Retrain RF + regenerate bundled sample clip (optional)
python -m ml.train --generate-sample
```

The repository already includes **`ml/models/risk_rf.pkl`** (&lt; 1 MB) and **`samples/sample.mp4`** (&lt; 5 MB), so you do **not** need to retrain to run inference.

---

## 7. Output explanation

### Console

- Progress messages and path to **`output/report.json`**.

### `output/report.json`

Example shape:

```json
{
  "summary": {
    "total_frames": 60,
    "avg_objects": 1.27,
    "risk_distribution": { "LOW": 26, "MEDIUM": 33, "HIGH": 1 },
    "lane_drift_events": 20,
    "avg_abs_lane_offset_norm": 0.12
  },
  "meta": {
    "source": "sample",
    "model": "ml/models/risk_rf.pkl",
    "frames_processed": 60
  }
}
```

- **`risk_distribution`** — counts of predicted labels over processed frames.
- **`lane_drift_events`** — frames where \|lane offset\| exceeded an internal drift threshold.
- **`avg_abs_lane_offset_norm`** — mean absolute normalized offset (spatial stability signal).

---

## 8. Folder structure

```
roadguard_x/
├── main.py
├── modules/
│   ├── preprocess.py
│   ├── lane.py
│   ├── segmentation.py
│   ├── tracker.py
│   ├── features.py
│   ├── risk_model.py
│   └── explain.py
├── ml/
│   ├── train.py
│   └── models/
│       └── risk_rf.pkl
├── utils/
│   ├── hud.py
│   └── reporter.py
├── samples/
│   └── sample.mp4
├── output/
│   └── (report.json generated at runtime)
├── requirements.txt
└── README.md
```

---

## 9. Limitations

- **No semantic class names** (e.g. “car”, “person”): motion blobs are generic **foreground regions**.
- **Heuristic lane geometry**: Hough-based lanes are sensitive to texture, lighting, and ROI choice; offset is a **proxy**, not ground-truth localization.
- **MOG2 warm-up**: first frames may be noisy until the background model stabilizes.
- **Display**: interactive mode uses **OpenCV HighGUI** (`cv2.imshow`). On servers without a display, use **`--headless`**.
- **Synthetic / rule-aligned training**: `ml/train.py` uses a **synthetic** dataset with **rule-derived** labels so the project ships without external datasets; real-world calibration would need labeled data and validation.

---

## 10. License / ethics

This software is for **research and education**. It is **not** a certified advanced driver-assistance system (ADAS) and must not be used as the sole basis for safety-critical decisions.
