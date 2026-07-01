#!/usr/bin/env bash
# Download Label Studio project 27 (tab 34) with recommended settings.
# Prerequisites: copy .env.example → .env and fill in credentials.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-/home/khoa/miniconda3/envs/cardiovis-inference/bin/python}"

if [[ ! -f .env ]]; then
  echo "Missing .env — copy .env.example to .env and set LABEL_STUDIO_API_KEY + CF_ACCESS_*." >&2
  exit 1
fi
set -a
# shellcheck disable=SC1091
source .env
set +a

export LABEL_STUDIO_EXPORT_VIEW_ID="${LABEL_STUDIO_EXPORT_VIEW_ID:-34}"
export LABEL_STUDIO_EXPORT_ANNOTATED_ONLY="${LABEL_STUDIO_EXPORT_ANNOTATED_ONLY:-1}"
export LABEL_STUDIO_EXPORT_FINISHED_ONLY="${LABEL_STUDIO_EXPORT_FINISHED_ONLY:-1}"

exec "$PYTHON" download_frames_labels.py --project-id 27 "$@"
