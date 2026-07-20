#!/usr/bin/env python3
"""Build *_das_multiseed_stats.json from per-seed ring_das_iia outputs."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "new_paper_rings"


def _mean_val(path: Path) -> float:
    d = json.loads(path.read_text(encoding="utf-8"))
    return float(sum(c["val_das_iia"] for c in d["components"]) / len(d["components"]))


def _seed_from_grokked(path: Path) -> int | None:
    m = re.search(r"_s(\d+)\.json$", path.name)
    return int(m.group(1)) if m else None


def _seed_from_randinit(path: Path) -> int:
    m = re.search(r"rs(\d+)\.json$", path.name)
    if m:
        return int(m.group(1))
    m = re.search(r"randinit_s(\d+)\.json$", path.name)
    if m:
        return int(m.group(1))
    raise ValueError(f"cannot parse random-init seed from {path.name}")


def welch_ttest(a: list[float], b: list[float]) -> dict:
    import scipy.stats as stats

    t, p = stats.ttest_ind(a, b, equal_var=False)
    return {"statistic": float(t), "pvalue": float(p), "equal_var": False, "alternative": "two-sided"}


def mann_whitney(a: list[float], b: list[float]) -> dict:
    import scipy.stats as stats

    u, p = stats.mannwhitneyu(a, b, alternative="greater")
    return {"statistic": float(u), "pvalue": float(p), "alternative": "greater"}


def cohens_d(a: list[float], b: list[float]) -> float:
    if len(a) < 2 or len(b) < 2:
        return math.nan
    va = st.pvariance(a)
    vb = st.pvariance(b)
    pooled = math.sqrt((va + vb) / 2)
    return (st.mean(a) - st.mean(b)) / pooled if pooled > 0 else math.inf


def build_stats(ring: str, grok_glob: str, rand_glob: str) -> dict:
    grok_paths = sorted(OUT.glob(grok_glob))
    rand_paths = sorted(OUT.glob(rand_glob))
    if not grok_paths:
        raise FileNotFoundError(f"no grokked files: {grok_glob}")
    if not rand_paths:
        raise FileNotFoundError(f"no random-init files: {rand_glob}")

    grok_vals: dict[int, float] = {}
    grok_acc: dict[int, float] = {}
    for p in grok_paths:
        d = json.loads(p.read_text(encoding="utf-8"))
        s = d.get("run_seed") or _seed_from_grokked(p)
        if s is None:
            continue
        grok_vals[int(s)] = _mean_val(p)
        acc = d.get("test_accuracy")
        if acc is not None:
            grok_acc[int(s)] = float(acc)

    rand_vals = [_mean_val(p) for p in rand_paths]
    rand_seeds = [_seed_from_randinit(p) for p in rand_paths]
    grok_list = list(grok_vals.values())

    payload = {
        "ring": ring,
        "metric": "mean_val_das_iia_across_components",
        "grokked": {
            "n": len(grok_vals),
            "seeds": sorted(grok_vals),
            "val_das": {str(k): v for k, v in sorted(grok_vals.items())},
            "mean": st.mean(grok_list),
            "std": st.pstdev(grok_list) if len(grok_list) > 1 else 0.0,
        },
        "random_init": {
            "n": len(rand_vals),
            "seeds": rand_seeds,
            "mean": st.mean(rand_vals),
            "std": st.pstdev(rand_vals) if len(rand_vals) > 1 else 0.0,
            "val_das": rand_vals,
        },
        "welch_ttest": welch_ttest(grok_list, rand_vals),
        "mann_whitney_u": mann_whitney(grok_list, rand_vals),
        "effect_size_cohens_d": cohens_d(grok_list, rand_vals),
    }
    if grok_acc:
        payload["grokked"]["train_best_test_acc"] = {str(k): v for k, v in sorted(grok_acc.items())}
        payload["grokked"]["train_best_test_acc_mean"] = st.mean(grok_acc.values())
    return payload


def main() -> int:
    p = argparse.ArgumentParser(description="Aggregate DAS multiseed stats (Welch + Mann–Whitney)")
    p.add_argument("--ring", required=True)
    p.add_argument(
        "--grok-glob",
        default=None,
        help="Glob under new_paper_rings/ for grokked runs (default: {ring}_das_iia_grokked_*_s*.json)",
    )
    p.add_argument(
        "--rand-glob",
        default=None,
        help="Glob for random-init (default: {ring}_das_iia_randinit_rs*.json)",
    )
    args = p.parse_args()
    grok_glob = args.grok_glob or f"{args.ring}_das_iia_grokked_*_s*.json"
    rand_glob = args.rand_glob or f"{args.ring}_das_iia_randinit_rs*.json"
    payload = build_stats(args.ring, grok_glob, rand_glob)
    out_path = OUT / f"{args.ring}_das_multiseed_stats.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"→ {out_path}")
    print(
        f"  grokked n={payload['grokked']['n']} mean={payload['grokked']['mean']:.3f} "
        f"random n={payload['random_init']['n']} mean={payload['random_init']['mean']:.3f} "
        f"Welch p={payload['welch_ttest']['pvalue']:.2e}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
