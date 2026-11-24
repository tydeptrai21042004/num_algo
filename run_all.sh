#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
profile=paper; resume=(); download=1
while (($#)); do case "$1" in --profile) profile="$2";shift 2;; --resume) resume=(--resume);shift;; --skip-download) download=0;shift;; *) echo "Unknown argument $1" >&2;exit 2;; esac;done
[[ "$profile" =~ ^(smoke|lite|paper|stress)$ ]] || exit 2
if [[ -f .venv/bin/activate ]]; then . .venv/bin/activate;fi
base="results/$profile"; mkdir -p "$base"
python scripts/verify_frozen_core.py
run(){ local name="$1";shift; python scripts/run_experiments.py --profile "$profile" --out "$base/$name" "${resume[@]}" "$@"; }
if [[ "$profile" == smoke ]]; then
    for g in mathematics hyperparameters calibration validation robustness60 ablation payload_robustness security compute video video_frames60;do run "$g" --group "$g" --split legacy;done
else
    if ((download));then python scripts/download_images.py --profile "$profile";python scripts/download_videos.py --profile "$profile";fi
    python scripts/verify_data_splits.py
    run mathematics --group mathematics
    run hyperparameters --group hyperparameters --split train --limit 12 --keys 1
    run calibration --group calibration --split train --keys 1
    run validation --group validation --split validation --calibration "$base/calibration"
    python scripts/freeze_configs.py --calibration "$base/calibration" --validation "$base/validation" --out "$base/frozen_config.json"
    run robustness60_legacy --group robustness60 --split legacy --frozen "$base/frozen_config.json"
    for g in robustness60 ablation payload_robustness security;do run "$g" --group "$g" --split test --frozen "$base/frozen_config.json";done
    run compute --group compute
    run video_development --group video --split development --methods FC-AOIP --video-mode codec
    run video_validation --group video --split validation --methods FC-AOIP --video-mode codec
    run video --group video --split test
    run video_frames60 --group video_frames60 --split test
fi
python scripts/summarize_results.py --root "$base" --out "results/${profile}_paper_ready"
python scripts/generate_figures.py --root "$base" --out "results/${profile}_paper_ready/figures"
