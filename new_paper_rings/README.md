# Workshop paper: non-commutative finite rings

PyTorch reproduction of *From Groups to Rings* (Zheng et al., 2026) on $T_n(\mathbb{F}_q)$, $M_n(\mathbb{F}_q)$, direct products, and exterior-algebra controls.

**Paper PDF:** `make pdf` (or `pdflatex main.tex` twice).

## Setup

All commands assume repository root `mech-interpret-rings`:

```bash
cd ..   # repo root
uv sync
```

**Hardware used in the paper runs**

| Role | Device | Notes |
|------|--------|-------|
| Local training / evaluation | NVIDIA RTX A1000 6GB Laptop | $\sim$20 GPU-hours total for reported main-text experiments |
| Full grokking grid (192 runs) | NVIDIA RTX 3090 | $\sim$3.5 h wall time at 8-way parallel training |

At $d{=}128$, batch 512, one 100k-step run takes $\sim$3.5–11 min depending on $|R|$.

**Logging:** MLflow backend `sqlite:///mlflow.db` (local) or `mlflow_server.db` (server bundle). Grokking grid experiment: `final_rings_for_paper`.

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Checkpoints, `mlflow.db`, and `runs/` are gitignored; regenerate via commands below or use the submission bundle `release/final_rings_for_paper/`.

Extended DAS artifact index: `DAS_ARCHIVE.md`.

---

## Reproduction index (paper table/figure → script → artifact)

Paths relative to repo root unless noted.

| Paper | Script | Config / artifact |
|-------|--------|-------------------|
| Tab. ring-roster | (structure only) | Wedderburn labels in Sec.~3.2 |
| Tab. grok-summary (a) | `ring_benchmark_matched_exposure.py`; `ring_transformer.py` | Six-ring NC benchmark; matched exp. + 100k steps |
| Tab. grok-summary (b) | `grok_grid.py` | $8×3×8$ grid; `grok_grid_results.json` |
| Tab. zheng-repro | `scripts/zheng_commutative_repro.py` | Rings `f8_x_f8`, `f2_z6`, `f3_z4`, extensions `lam_f2_k3t`, `lam_f3_k2`; `--zheng-protocol`; `zheng_commutative_repro_seed42.json` |
| Tab. nc-iia-summary (hierarchy + per-component) | `scripts/ring_iia_hierarchy.py`; `scripts/ring_transformer_iia.py` | Product: `grok_wd01_seed42`; $T_3$: `lr5e4_seed42`; IIA seed 44 |
| Fig. iia-hierarchy | `scripts/make_paper_iia_figure.py` | → `plots/alt_group_interp/article_runs/tri2_f3_x_f3_x_f3_paper_iia.png` |
| Tab. err-comp | `scripts/ring_iia_hierarchy.py` | `mean_error_preservation_ratio` in hierarchy JSON |
| Tab. wd-sweep | `experiments/ring_transformer.py` | `--ring tri2_f3_x_f3_x_f3`; tags `grok_wd01`, `v2`; seeds 42–46 |
| Tab. linear-das | `scripts/ring_das_iia.py`; `ring_das_disjoint_train.py`; `ring_das_large_eval.py`; `das_significance.py` | `tri2_f3_x_f3_x_f3_das_iia.json`; `*_das_disjoint_dense_seed42.json`; `*_das_significance_seed42.json`; `*_das_large_eval_seed*.json` |
| Fig. iia-trace | `scripts/ring_transformer_iia_trace.py` | `--ring tri3_f2 --lr 5e-4 --checkpoint-tag lr5e4` |
| Sec. training-traces | `scripts/verify_raw_iia_mechanism.py`; `diagnose_iia_pair_split.py` | Raw IIA = counterfactual accuracy sanity check |
| Tab. algebra-wm-iia | `scripts/ring_transformer_iia.py` | `--label-scheme lam_wm` on `lam_f3_k2` reference-recipe checkpoint |
| Tab. probe-fit | `scripts/ring_probe_fit_compare.py` | `probe_fit_compare/*.json` |
| Tab. classmean-probe | `scripts/ring_classmean_probe_compare.py` | `*classmean_probe_compare.json` |
| Tabs. tri2/tri2xf3 multiseed | `experiments/ring_transformer.py` | `--weight-decay 2 --lr 5e-3`; rings `tri2_f3`, `tri2_f3_x_f3`; seeds 42–47 |
| Tab. grok-grid-grok (legacy) | `scripts/grok_grid.py` | rows fragment → `generated/grok_grid_rows.tex` inside `tab:grok-summary` |
| Tab. das-multiseed | `scripts/das_parallel.py`; `das_multiseed_stats.py` | `zheng_wd2` ckpts seeds 42–47 vs random-init; `*_das_multiseed_stats.json` |

---

## Core experiments

**Product ring grokking** ($T_2(\mathbb{F}_3)\times\mathbb{F}_3\times\mathbb{F}_3$, $\lambda=0.1$):

```bash
uv run python experiments/ring_transformer.py --ring tri2_f3_x_f3_x_f3 \
  --checkpoint-tag grok_wd01 --weight-decay 0.1
```

**IIA hierarchy:**

```bash
uv run python scripts/ring_iia_hierarchy.py --ring tri2_f3_x_f3_x_f3 \
  --grokked-checkpoint checkpoints/tri2_f3_x_f3_x_f3_transformer_grok_wd01_seed42/best_model.pt
```

**DAS (axis-aligned + rotated):**

```bash
uv run python scripts/ring_das_iia.py --ring tri2_f3_x_f3_x_f3 \
  --checkpoint checkpoints/tri2_f3_x_f3_x_f3_transformer_grok_wd01_seed42/best_model.pt
uv run python scripts/ring_das_disjoint_train.py
```

**Commutative replication (Table zheng-repro):**

```bash
uv run python experiments/ring_transformer.py --ring f8_x_f8 --max-steps 100000 --checkpoint-tag zheng_repro --seed 42 --zheng-protocol
uv run python scripts/zheng_commutative_repro.py --device cuda
```

**Cross-ring benchmark (matched exposure):**

```bash
uv run python scripts/ring_benchmark_matched_exposure.py
```

**$T_3(\mathbb{F}_2)$ IIA trace:**

```bash
uv run python scripts/ring_transformer_iia_trace.py --ring tri3_f2 --lr 5e-4 \
  --checkpoint-tag lr5e4 --iia-interval 100
```

---

## Systematic grokking grid (192 runs)

Portable server bundle: `release/final_rings_for_paper/` (build: `bash scripts/build_final_rings_bundle.sh`).

Rings: `f8_x_f8`, `f2_z6`, `mat2_f2_x_mat2_f2`, `lam_f2_k3t`, `lam_f3_k2`, `tri2_f3`, `tri3_f2`, `tri2_f3_x_f3_x_f3`.

Weight decay: 0.1 / 1.0 / 2.0 (lr=$10^{-3}$ except $\lambda=2.0$ → lr=$5×10^{-3}$). Seeds: 42–49. All jobs require `--zheng-protocol`.

```bash
uv run python scripts/grok_grid.py --parallel 3 --device cuda --skip-existing
uv run python scripts/grok_grid.py --summarize   # → grok_grid_results.json + generated/grok_grid_rate.tex
uv run python scripts/grok_grid.py --link-existing
```

Sync from GPU server: `bash scripts/sync_results_from_server.sh user@host ~/final_rings_for_paper`

---

## Tests

```bash
uv run pytest tests/
```
