#!/bin/bash
# Create the local, locked CUDA 12 environment with uv.
set -euo pipefail
cd "$(dirname "$0")/.."
command -v uv >/dev/null || { echo "Install uv first: https://docs.astral.sh/uv/getting-started/installation/" >&2; exit 1; }
uv sync --locked
uv run --locked --no-sync python tools/check_runtime.py
