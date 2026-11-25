#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
args=()
while (($#)); do
 case "$1" in
 --profile) [[ "$2" == paper || "$2" == compact12 || "$2" == lite ]] || { echo 'Use paper, compact12 or lite'; exit 2; }; if [[ "$2" == lite ]]; then args+=(--video-profile lite); fi; shift 2;;
 *) args+=("$1"); shift;;
 esac
done
exec bash run_12_images.sh "${args[@]}"
