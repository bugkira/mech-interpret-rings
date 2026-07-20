#!/usr/bin/env python3
"""2×2 ablation: {paper-tuned vs Zheng-default hparams} × {legacy vs --zheng-protocol}.

Default: train 4 models (2 rings × 2 hparam cells, all with --zheng-protocol).
Compare against existing legacy checkpoints from the paper.

    uv run python scripts/nc_zheng_protocol_ablation.py --train
    uv run python scripts/nc_zheng_protocol_ablation.py --summarize
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "new_paper_rings" / "nc_zheng_protocol_ablation_seed42.json"

EXPERIMENTS = (
    {
        "ring": "tri2_f3_x_f3_x_f3",
        "label": "TriFthreeProd",
        "hparam_cell": "tuned",
        "train_args": ["--weight-decay", "0.1"],
        "checkpoint_tag": "zheng_proto_wd01",
        "legacy_tag": "grok_wd01",
    },
    {
        "ring": "tri2_f3_x_f3_x_f3",
        "label": "TriFthreeProd",
        "hparam_cell": "zheng_default",
        "train_args": ["--weight-decay", "1.0", "--lr", "1e-3"],
        "checkpoint_tag": "zheng_proto_wd1",
        "legacy_tag": "v2",
    },
    {
        "ring": "tri3_f2",
        "label": "T3(F2)",
        "hparam_cell": "tuned",
        "train_args": ["--lr", "5e-4", "--weight-decay", "1.0"],
        "checkpoint_tag": "zheng_proto_lr5e4",
        "legacy_tag": "lr5e4",
    },
    {
        "ring": "tri3_f2",
        "label": "T3(F2)",
        "hparam_cell": "zheng_default",
        "train_args": ["--weight-decay", "1.0", "--lr", "1e-3"],
        "checkpoint_tag": "zheng_proto_wd1",
        "legacy_tag": "",
    },
)


def _ckpt(ring: str, tag: str | None, seed: int = 42) -> Path:
    if tag:
        name = f"{ring}_transformer_{tag}_seed{seed}"
    else:
        name = f"{ring}_transformer_seed{seed}"
    return ROOT / "checkpoints" / name / "best_model.pt"


def _load_metrics(path: Path) -> dict:
    if not path.exists():
        return {"exists": False, "path": str(path)}
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    return {
        "exists": True,
        "path": str(path),
        "best_test_acc": float(cfg.get("test_acc", ckpt.get("test_acc", 0.0))),
        "best_epoch": int(cfg.get("best_epoch", ckpt.get("epoch", -1))),
        "weight_decay": cfg.get("weight_decay"),
        "lr": cfg.get("lr"),
        "zheng_protocol": bool(cfg.get("zheng_protocol", False)),
    }


def _train_one(exp: dict, *, seed: int) -> None:
    cmd = [
        sys.executable,
        str(ROOT / "experiments" / "ring_transformer.py"),
        "--ring",
        exp["ring"],
        "--max-steps",
        "100000",
        "--seed",
        str(seed),
        "--checkpoint-tag",
        exp["checkpoint_tag"],
        "--zheng-protocol",
        *exp["train_args"],
    ]
    print("TRAIN", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def train_all(*, seed: int) -> None:
    for exp in EXPERIMENTS:
        zheng_ck = _ckpt(exp["ring"], exp["checkpoint_tag"], seed)
        if zheng_ck.exists():
            print(f"skip (exists): {zheng_ck}", flush=True)
            continue
        _train_one(exp, seed=seed)


def summarize(*, seed: int) -> dict:
    rows: list[dict] = []
    for exp in EXPERIMENTS:
        legacy_tag = exp["legacy_tag"]
        if legacy_tag is None:
            continue
        legacy = _load_metrics(_ckpt(exp["ring"], legacy_tag if legacy_tag != "" else None, seed))
        zheng = _load_metrics(_ckpt(exp["ring"], exp["checkpoint_tag"], seed))
        rows.append(
            {
                **exp,
                "legacy": legacy,
                "zheng_protocol_run": zheng,
                "delta_test": (
                    zheng["best_test_acc"] - legacy["best_test_acc"]
                    if legacy.get("exists") and zheng.get("exists")
                    else None
                ),
            }
        )
    payload = {
        "seed": seed,
        "description": "Paper rings: tuned vs Zheng-default hparams; legacy init vs --zheng-protocol",
        "rows": rows,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {OUT}")
    for r in rows:
        leg = r["legacy"]
        z = r["zheng_protocol_run"]
        print(
            f"  {r['ring']} [{r['hparam_cell']}]: "
            f"legacy={leg.get('best_test_acc', 'NA')} "
            f"zheng_proto={z.get('best_test_acc', 'NA')} "
            f"delta={r['delta_test']}"
        )
    return payload


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train", action="store_true")
    p.add_argument("--summarize", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.train:
        train_all(seed=args.seed)
    if args.summarize or not args.train:
        summarize(seed=args.seed)


if __name__ == "__main__":
    main()
