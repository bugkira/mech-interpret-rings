# Non-commutative finite algebras — reproducibility code

PyTorch code for mechanistic interpretability of grokking on non-commutative finite algebras
(Zheng-style commutative replication, grokking grid, IIA / DAS probes).

Repository: https://github.com/bugkira/mech-interpret-rings

## Setup

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # if needed
uv sync
cp .env.example .env
uv run python scripts/register_mlflow_experiment.py
```

## Quick checks

```bash
uv run pytest tests/test_zheng_protocol.py tests/test_finite_rings.py tests/test_true_algebras.py -q
```

## Grokking grid (192 runs: 8 algebras × 3 λ × 8 seeds)

```bash
uv run python scripts/grok_grid.py --parallel 3 --device cuda --skip-existing
uv run python scripts/grok_grid.py --summarize
```

| Parameter | Value |
|-----------|-------|
| Algebras (8) | `f8_x_f8`, `f2_z6`, `mat2_f2_x_mat2_f2`, `lam_f2_k3t`, `lam_f3_k2`, `tri2_f3`, `tri3_f2`, `tri2_f3_x_f3_x_f3` |
| λ | 0.1, 1.0, 2.0 |
| Seeds | 42–49 |
| Protocol | `--zheng-protocol`, init std=0.02, AdamW weight decay on weights only |
| LR | 1e-3 (λ=0.1, 1.0); 5e-3 (λ=2.0) |
| Steps | 100k |

Training logs to MLflow experiment `final_rings_for_paper` (`sqlite:///mlflow.db`).

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Single-algebra training

```bash
uv run python experiments/ring_transformer.py --ring tri2_f3 --zheng-protocol \
  --mlflow-experiment final_rings_for_paper
```

## Commutative replication

```bash
uv run python scripts/zheng_commutative_repro.py --device cuda
```

## Notes

- Do **not** commit `checkpoints/`, `mlflow.db`, or `logs/` (see `.gitignore`).
- Hardware used for paper runs: laptop GPU (RTX A1000 6GB) for local eval; desktop RTX 3090 for the full 192-run grid.
