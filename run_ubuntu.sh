#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
bash run_all.sh --profile smoke --resume "$@"
