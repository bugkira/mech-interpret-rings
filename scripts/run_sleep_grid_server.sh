#!/usr/bin/env bash
# Launch sleep grid on GPU server (uses .venv with torch cu124, NOT uv run)
set -euo pipefail
cd "$(dirname "$0")/.."
export MLFLOW_EXPERIMENT=final_rings_for_paper
export REQUIRE_ZHENG_PROTOCOL=1
export PYTHONUNBUFFERED=1
mkdir -p logs/sleep_grid

PYTHON=.venv/bin/python
if [[ ! -x "$PYTHON" ]]; then
  echo "Run scripts/setup_gpu_server.sh first" >&2
  exit 1
fi

# Link legacy checkpoints, summarize what we already have
"$PYTHON" scripts/sleep_grid.py --link-existing --summarize || true

PARALLEL="${1:-8}"
# Kill prior orchestrator + trainers if restarting
pkill -f 'scripts/sleep_grid.py --parallel' 2>/dev/null || true
sleep 2
pkill -f 'experiments/ring_transformer.py' 2>/dev/null || true
sleep 1

nohup "$PYTHON" -u scripts/sleep_grid.py \
  --parallel "$PARALLEL" \
  --device cuda \
  --skip-existing \
  > logs/sleep_grid/orchestrator.log 2>&1 &

echo "Sleep grid started PID=$! parallel=$PARALLEL"
echo "  tail -f logs/sleep_grid/orchestrator.log"
echo "  tail -f logs/sleep_grid/*.log"
