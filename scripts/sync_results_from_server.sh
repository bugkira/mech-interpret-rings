#!/usr/bin/env bash
# Pull final_rings_for_paper results from GPU server (incremental, no archive dump).
#
# Default: only grok-grid experiment data (~200 MB), NOT the 10 GB checkpoint archive
# that was rsync'd to the server once for seeding.
#
# Usage:
#   export SSHPASS='...'
#   export RSYNC_RSH="sshpass -e ssh -o StrictHostKeyChecking=no -p 47009"
#   ./scripts/sync_results_from_server.sh user@195.208.16.1
#
# Options (env):
#   SYNC_CHECKPOINTS=0     — skip checkpoints (MLflow + logs only, fastest)
#   SYNC_CHECKPOINTS=all   — full checkpoints/ tree (slow; one-time archive merge)
#
set -euo pipefail

REMOTE="${1:?Usage: $0 user@host [remote_dir]  (set RSYNC_RSH for sshpass/custom port)}"
REMOTE_DIR="${2:-~/final_rings_for_paper}"
LOCAL="$(cd "$(dirname "$0")/.." && pwd)"
CK_MODE="${SYNC_CHECKPOINTS:-grid}"  # grid | all | 0

echo "Pulling final_rings_for_paper from ${REMOTE}:${REMOTE_DIR} → ${LOCAL}"
if [[ -z "${RSYNC_RSH:-}" ]]; then
  echo "Hint: export RSYNC_RSH=\"sshpass -e ssh -o StrictHostKeyChecking=no -p PORT\""
fi

RSYNC=(rsync -avz --progress)
if [[ -n "${RSYNC_RSH:-}" ]]; then
  RSYNC+=(-e "$RSYNC_RSH")
fi

mkdir -p "${LOCAL}/logs/grok_grid" "${LOCAL}/new_paper_rings" "${LOCAL}/mlruns_final_rings"

echo "→ mlflow_server.db (experiment final_rings_for_paper)"
"${RSYNC[@]}" "${REMOTE}:${REMOTE_DIR}/mlflow.db" "${LOCAL}/mlflow_server.db"

# MLflow UI does not hot-reload SQLite — restart if running on :5001
if pgrep -f 'mlflow ui.*5001' >/dev/null 2>&1; then
  echo "→ restarting MLflow UI (port 5001) to pick up DB changes"
  pkill -f 'mlflow ui.*5001' 2>/dev/null || true
  sleep 2
  nohup "${LOCAL}/scripts/mlflow_final_rings_ui.sh" 5001 >"${LOCAL}/logs/mlflow_final_rings_ui.log" 2>&1 &
fi

echo "→ mlruns_final_rings/1/ (MLflow artifacts for this experiment only)"
"${RSYNC[@]}" "${REMOTE}:${REMOTE_DIR}/mlruns/1/" "${LOCAL}/mlruns_final_rings/1/"

echo "→ logs/grok_grid/"
"${RSYNC[@]}" "${REMOTE}:${REMOTE_DIR}/logs/grok_grid/" "${LOCAL}/logs/grok_grid/"

echo "→ grok_grid_results.json"
"${RSYNC[@]}" "${REMOTE}:${REMOTE_DIR}/new_paper_rings/grok_grid_results.json" "${LOCAL}/new_paper_rings/"

if [[ "$CK_MODE" == "0" ]]; then
  echo "→ checkpoints/ skipped (SYNC_CHECKPOINTS=0)"
elif [[ "$CK_MODE" == "all" ]]; then
  echo "→ checkpoints/ FULL (archive — may be ~10 GB first time)"
  mkdir -p "${LOCAL}/checkpoints"
  "${RSYNC[@]}" "${REMOTE}:${REMOTE_DIR}/checkpoints/" "${LOCAL}/checkpoints/"
else
  echo "→ checkpoints/ grok-grid tags only (zheng_wd01|wd1|wd2, ~500 MB)"
  mkdir -p "${LOCAL}/checkpoints"
  set +e
  "${RSYNC[@]}" \
    --include='*/' \
    --include='*_transformer_zheng_wd01_seed*/***' \
    --include='*_transformer_zheng_wd1_seed*/***' \
    --include='*_transformer_zheng_wd2_seed*/***' \
    --exclude='*' \
    "${REMOTE}:${REMOTE_DIR}/checkpoints/" "${LOCAL}/checkpoints/"
  rc=$?
  set -e
  if [[ $rc -ne 0 && $rc -ne 24 ]]; then
    exit $rc
  fi
  [[ $rc -eq 24 ]] && echo "  (rsync 24: some epoch checkpoints vanished mid-training — ok)"
fi

echo ""
echo "Done."
echo "MLflow UI (restart required after sync — UI caches SQLite):"
echo "  pkill -f 'mlflow ui.*5001'; ./scripts/mlflow_final_rings_ui.sh"
echo "  → http://127.0.0.1:5001/#/experiments/1"
echo "Local mlflow.db (port 5000) was NOT touched."
