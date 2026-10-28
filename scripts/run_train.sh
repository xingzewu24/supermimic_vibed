#!/bin/bash
# Usage: [CPUS=0-23] ./scripts/run_train.sh <run_name> [Hydra options / key=value overrides...]
# Run in the foreground; use tmux or your scheduler for detached training.
set -euo pipefail
cd "$(dirname "$0")/.."
RUN=${1:-hoi_main}
if [ "$#" -gt 0 ]; then shift; fi
mkdir -p "logs/$RUN"
PIN=()
if [ -n "${CPUS:-}" ]; then PIN=(taskset -c "$CPUS"); fi
# Keep Hydra options before overrides; its argparse parser does not intermix them.
# --no-sync: the env comes from scripts/setup.sh; skips uv's per-run network check.
"${PIN[@]}" uv run --locked --no-sync python -u -m supermimic.train "$@" "run_name=$RUN" 2>&1 | tee -a "logs/$RUN/stdout.log"
