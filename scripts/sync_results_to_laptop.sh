#!/usr/bin/env bash
# Push artifacts from GPU server to laptop (run ON the server).
# Set LAPTOP=user@host in .env or environment.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f .env ]]; then
  # shellcheck disable=SC1091
  source .env
fi

LAPTOP="${LAPTOP:?Set LAPTOP=user@your-laptop in .env}"
LOCAL_DIR="${LOCAL_DIR:-~/final_rings_for_paper}"

echo "Pushing ${ROOT} → ${LAPTOP}:${LOCAL_DIR}"

rsync -avz --progress \
  checkpoints/ "${LAPTOP}:${LOCAL_DIR}/checkpoints/" \
  mlflow.db "${LAPTOP}:${LOCAL_DIR}/" \
  logs/grok_grid/ "${LAPTOP}:${LOCAL_DIR}/logs/grok_grid/" \
  new_paper_rings/grok_grid_results.json "${LAPTOP}:${LOCAL_DIR}/new_paper_rings/" \
  2>/dev/null || true

[[ -d mlartifacts ]] && rsync -avz --progress mlartifacts/ "${LAPTOP}:${LOCAL_DIR}/mlartifacts/" || true

echo "Done."
