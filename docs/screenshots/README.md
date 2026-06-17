# Screenshots for README

All images render on the main [README](../../README.md).

| File | What it shows |
|------|----------------|
| `dashboard-hero.png` | Hero, tech pills, CV preview, metric cards |
| `dashboard-upload-zone.png` | Upload / drag-and-drop area |
| `dashboard-processed-output.png` | Processed video with HUD overlays |
| `dashboard-analysis-metrics.png` | Risk badge, confidence, session overview |
| `dashboard-explanation.png` | Session explanation and feature contributions |
| `dashboard-session-insights.png` | Session summary, timeline, danger clip |
| `dashboard-risk-distribution.png` | Danger clip and risk distribution |
| `dashboard-model-info.png` | Feature-importance chart |
| `dashboard-model-eval.png` | Held-out test metrics |
| `dashboard-confusion-matrix.png` | Confusion matrix and hyperparameters |
| `cli-hud.png` | Live CLI HUD on sample video |

**Aliases:** `dashboard-upload.png` → upload zone · `dashboard-results.png` → processed output

## Recapture

1. `python -m uvicorn api.server:app --port 8000`
2. `cd web && npm run dev` → upload `roadguard_x/samples/demo.mp4`
3. CLI HUD: `python roadguard_x/main.py --source sample`
