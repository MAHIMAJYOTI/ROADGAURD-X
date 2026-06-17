Bundled demo clip for RoadGuard-X
=================================

Place the short driving demo video here as:

  demo.mp4

This file is committed to the repo so anyone can run the CLI and localhost
dashboard without generating a synthetic sample.

Recommended: MP4, under ~20 MB, 10–30 seconds, front-facing road footage.

Current bundled clip: `demo.mp4` (driving scene, ~17 MB).

Used by:
  python main.py --source sample
  python main.py --input samples/demo.mp4 --headless
  Web UI upload on http://localhost:3000

Optional: run `python generate_sample.py` from roadguard_x/ to create
samples/street.mp4 locally if demo.mp4 is missing.
