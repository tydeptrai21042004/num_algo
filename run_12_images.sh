#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ -f .venv/bin/activate ]]; then . .venv/bin/activate; fi
export FCAOIP_THREADS="${FCAOIP_THREADS:-1}"
python scripts/run_12_images.py "$@"
