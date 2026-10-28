#!/bin/bash
# Download body-retargeted LAFAN1 motions; use the separately supplied full Inspire asset.
# LAFAN1 (Ubisoft La Forge) is CC BY-NC-ND 4.0; the Unitree G1 description is BSD-3-Clause.
set -euo pipefail
cd "$(dirname "$0")/.."
uv run --locked --no-sync python tools/download_data.py "$@"
uv run --locked --no-sync python tools/preprocess_motion.py
