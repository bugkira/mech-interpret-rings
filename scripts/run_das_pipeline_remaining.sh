#!/usr/bin/env bash
# Run remaining DAS phases sequentially: disjoint → large-eval → significance.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
mkdir -p logs/das_parallel new_paper_rings

PYTHON=.venv/bin/python
LOG=logs/das_parallel/pipeline_remaining.log
PARALLEL="${1:-10}"

exec > >(tee -a "$LOG") 2>&1

echo "=== DAS pipeline start $(date -Iseconds) ==="

echo "--- phase: disjoint (parallel=$PARALLEL) ---"
"$PYTHON" -u scripts/das_parallel.py \
  --phase disjoint \
  --parallel "$PARALLEL" \
  --device cuda \
  --skip-existing \
  --merge

echo "--- phase: large-eval (parallel=8) ---"
"$PYTHON" -u scripts/das_parallel.py \
  --phase large-eval \
  --parallel 8 \
  --device cuda \
  --skip-existing

CKPT="$("$PYTHON" - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))
from scripts.das_parallel import _pick_grokked_ckpt, PRODUCT_RING
p = _pick_grokked_ckpt(PRODUCT_RING, wd=0.1) or _pick_grokked_ckpt(PRODUCT_RING)
if p is None:
    raise SystemExit("no grokked checkpoint for product ring")
print(p)
PY
)"

echo "--- phase: significance checkpoint=$CKPT ---"
"$PYTHON" scripts/das_significance.py \
  --ring tri2_f3_x_f3_x_f3 \
  --checkpoint "$CKPT" \
  --device cuda

echo "=== DAS pipeline DONE $(date -Iseconds) ==="
