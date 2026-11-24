#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if ! command -v ffmpeg >/dev/null || ! command -v curl >/dev/null; then sudo apt-get update; sudo apt-get install -y ffmpeg curl python3-venv; fi
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/verify_frozen_core.py
python -m pytest -q
