#!/usr/bin/env bash
# Parallel DAS on GPU server (after sleep grid). Uses .venv, NOT uv run.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
mkdir -p logs/das_parallel new_paper_rings

PYTHON=.venv/bin/python
PARALLEL="${1:-10}"
PHASE="${2:-multiseed-iia}"

if [[ ! -x "$PYTHON" ]]; then
  echo "Run scripts/setup_gpu_server.sh first" >&2
  exit 1
fi

"$PYTHON" scripts/sleep_grid.py --summarize

case "$PHASE" in
  multiseed-iia)
    nohup "$PYTHON" -u scripts/das_parallel.py \
      --phase multiseed-iia \
      --parallel "$PARALLEL" \
      --device cuda \
      --skip-existing \
      > logs/das_parallel/orchestrator_multiseed.log 2>&1 &
    ;;
  disjoint)
    nohup "$PYTHON" -u scripts/das_parallel.py \
      --phase disjoint \
      --parallel "$PARALLEL" \
      --device cuda \
      --skip-existing \
      --merge \
      > logs/das_parallel/orchestrator_disjoint.log 2>&1 &
    ;;
  large-eval)
    nohup "$PYTHON" -u scripts/das_parallel.py \
      --phase large-eval \
      --parallel 8 \
      --device cuda \
      --skip-existing \
      > logs/das_parallel/orchestrator_large_eval.log 2>&1 &
    ;;
  all)
    nohup "$PYTHON" -u scripts/das_parallel.py \
      --phase all \
      --parallel "$PARALLEL" \
      --device cuda \
      --skip-existing \
      --merge \
      > logs/das_parallel/orchestrator_all.log 2>&1 &
    ;;
  *)
    echo "Usage: $0 [parallel] [multiseed-iia|disjoint|large-eval|all]" >&2
    exit 1
    ;;
esac

echo "DAS grid started PID=$! phase=$PHASE parallel=$PARALLEL"
echo "  tail -f logs/das_parallel/orchestrator_${PHASE//-/_}.log"
