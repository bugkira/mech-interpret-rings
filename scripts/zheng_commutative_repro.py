#!/usr/bin/env python3
"""Direct replication of Zheng et al. commutative benchmarks (seed 42).

    uv run python scripts/zheng_commutative_repro.py
    uv run python scripts/zheng_commutative_repro.py --train-missing
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gf_grokking.ring_transformer_iia import (
    load_ring_transformer,
    run_ring_transformer_iia_analysis,
)
from gf_grokking.ring_wedderburn import LAM_WM_SCHEME, get_wedderburn_spec

OUT = ROOT / "new_paper_rings" / "zheng_commutative_repro_seed42.json"

BENCHMARKS = (
    {
        "ring": "f8_x_f8",
        "zheng_name": r"F_2[x]/(\Phi_7) \cong F_8 \times F_8",
        "checkpoint_tag": "zheng_repro",
        "zheng_protocol": True,
    },
    {
        "ring": "f2_z6",
        "zheng_name": r"F_2[\mathbb{Z}_6]",
        "checkpoint_tag": "zheng_repro",
        "zheng_protocol": True,
    },
    {
        "ring": "f3_z4",
        "zheng_name": r"F_3[\mathbb{Z}_4] \cong F_3 \times F_3 \times F_9",
        "checkpoint_tag": "ab_zheng",
        "zheng_protocol": True,
    },
    {
        "ring": "lam_f2_k3t",
        "zheng_name": r"\Lambda(\mathbb{F}_2^3)/\Lambda^{\geq 3}",
        "checkpoint_tag": "zheng_repro",
        "zheng_protocol": True,
        "extension": True,
        "label_scheme": LAM_WM_SCHEME,
    },
    {
        "ring": "lam_f3_k2",
        "zheng_name": r"\Lambda(\mathbb{F}_3^2)",
        "checkpoint_tag": "zheng_repro",
        "zheng_protocol": True,
        "extension": True,
        "label_scheme": LAM_WM_SCHEME,
    },
)


def _checkpoint_path(ring: str, tag: str = "zheng_repro") -> Path:
    return ROOT / "checkpoints" / f"{ring}_transformer_{tag}_seed42" / "best_model.pt"


def _train_if_missing(row: dict) -> None:
    ring = row["ring"]
    tag = row["checkpoint_tag"]
    ck = _checkpoint_path(ring, tag)
    if ck.exists():
        print(f"  checkpoint exists: {ck}")
        return
    print(f"  training {ring} (100k steps, tag={tag})...")
    cmd = [
        sys.executable,
        str(ROOT / "experiments" / "ring_transformer.py"),
        "--ring",
        ring,
        "--max-steps",
        "100000",
        "--checkpoint-tag",
        tag,
        "--seed",
        "42",
    ]
    if row.get("zheng_protocol", True):
        cmd.append("--zheng-protocol")
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def _mean(vals: list[float]) -> float:
    return float(np.mean(vals)) if vals else 0.0


def _run_iia(ring: str, checkpoint: Path, *, device: str, seed: int = 44) -> dict:
    model, ckpt, mult_table, wspec = load_ring_transformer(str(checkpoint), device)
    if ring in ("lam_f2_k3t", "lam_f3_k2"):
        wspec = get_wedderburn_spec(ring, label_scheme=LAM_WM_SCHEME)
    result = run_ring_transformer_iia_analysis(
        model,
        mult_table,
        wspec,
        checkpoint=str(checkpoint),
        device=device,
        max_pairs_per_component=150,
        seed=seed,
        probe_fit_mode="all",
    )
    per = {m.irrep: m for m in result.per_irrep if m.n_pairs > 0}
    return {
        "held_out_test": float(ckpt["test_acc"]),
        "full_table_acc": float(result.test_accuracy),
        "mean_raw_iia": float(result.mean_raw_iia),
        "mean_subspace_iia": _mean([m.subspace_iia for m in result.per_irrep if m.n_pairs > 0]),
        "mean_random_subspace_iia": _mean([m.random_subspace_iia for m in result.per_irrep if m.n_pairs > 0]),
        "mean_error_preservation_ratio": float(result.mean_error_preservation_ratio),
        "per_component": {
            name: {
                "raw_iia": per[name].raw_iia,
                "subspace_iia": per[name].subspace_iia,
                "random_subspace_iia": per[name].random_subspace_iia,
            }
            for name in per
        },
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-missing", action="store_true", help="Train rings without checkpoints")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    if args.train_missing:
        for row in BENCHMARKS:
            _train_if_missing(row)

    device = args.device
    results: list[dict] = []
    for row in BENCHMARKS:
        ring = row["ring"]
        ck = _checkpoint_path(ring, row["checkpoint_tag"])
        if not ck.exists():
            raise FileNotFoundError(f"missing checkpoint for {ring}: {ck}")
        print(f"IIA {ring} ...")
        metrics = _run_iia(ring, ck, device=device)
        results.append(
            {
                "ring": ring,
                "zheng_name": row["zheng_name"],
                "checkpoint": str(ck),
                "extension": row.get("extension", False),
                **metrics,
            }
        )

    payload = {
        "protocol": "zheng_commutative_direct_repro",
        "seed": 42,
        "iia_seed": 44,
        "training_note": "All rows use --zheng-protocol (init std=0.02, no bias decay); 100k steps, seed 42.",
        "zheng_targets": {
            "f8_x_f8": {"held_out_test": 1.0, "subspace_iia": 0.998, "random_subspace": 0.129},
            "f2_z6": {"raw_rad": 0.995, "raw_ss": 0.996, "error_preservation": 0.643},
            "f3_z4": {"held_out_test": 0.995, "raw_iia": 0.996},
        },
        "results": results,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {OUT}")
    for r in results:
        print(
            f"  {r['ring']}: held-out={r['held_out_test']:.4f} "
            f"raw={r['mean_raw_iia']:.3f} sub={r['mean_subspace_iia']:.3f} "
            f"rand={r['mean_random_subspace_iia']:.3f}"
        )


if __name__ == "__main__":
    main()
