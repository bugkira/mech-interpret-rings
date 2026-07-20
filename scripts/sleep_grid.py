#!/usr/bin/env python3
"""Sleep grid: 8 rings × 3 weight decay × 8 seeds, mandatory --zheng-protocol.

    uv run python scripts/sleep_grid.py --parallel 6 --device cuda
    uv run python scripts/sleep_grid.py --summarize
    uv run python scripts/sleep_grid.py --link-existing
    uv run python scripts/sleep_grid.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from gf_grokking.mlflow_logger import FINAL_RINGS_EXPERIMENT

ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "new_paper_rings" / "sleep_grid_results.json"
OUT_TEX = ROOT / "new_paper_rings" / "generated" / "sleep_grid_grok_rate.tex"
OUT_TEX_ROWS = ROOT / "new_paper_rings" / "generated" / "sleep_grid_grok_rows.tex"
LOG_DIR = ROOT / "logs" / "sleep_grid"
TRAIN_SCRIPT = ROOT / "experiments" / "ring_transformer.py"

GROK_THRESHOLD = 0.95
MAX_STEPS = 100_000
ZHENG_INIT_STD = 0.02

RINGS: tuple[str, ...] = (
    "f8_x_f8",
    "f2_z6",
    "mat2_f2_x_mat2_f2",
    "lam_f2_k3t",
    "lam_f3_k2",
    "tri2_f3",
    "tri3_f2",
    "tri2_f3_x_f3_x_f3",
)

SEEDS: tuple[int, ...] = tuple(range(42, 50))

WD_CELLS: tuple[tuple[float, float, str], ...] = (
    (0.1, 1e-3, "zheng_wd01"),
    (1.0, 1e-3, "zheng_wd1"),
    (2.0, 5e-3, "zheng_wd2"),
)

# Legacy checkpoint_tag aliases (seed 42–46 @ wd=1.0, etc.)
LEGACY_TAGS: dict[tuple[str, float, int], str] = {
    ("f8_x_f8", 1.0, 42): "zheng_repro",
    ("f2_z6", 1.0, 42): "zheng_repro",
    ("lam_f2_k3t", 1.0, 42): "zheng_repro",
    ("lam_f3_k2", 1.0, 42): "zheng_repro",
    ("lam_f3_k2", 1.0, 43): "zheng_repro",
    ("lam_f3_k2", 1.0, 44): "zheng_repro",
    ("lam_f3_k2", 1.0, 45): "zheng_repro",
    ("lam_f3_k2", 1.0, 46): "zheng_repro",
    ("tri2_f3_x_f3_x_f3", 0.1, 42): "zheng_proto_wd01",
    ("tri2_f3_x_f3_x_f3", 1.0, 42): "zheng_proto_wd1",
    ("tri3_f2", 1.0, 42): "zheng_proto_wd1",
}

RING_LABELS: dict[str, str] = {
    "f8_x_f8": r"$\mathbb{F}_2[x]/(\Phi_7)$",
    "f2_z6": r"$\mathbb{F}_2[\mathbb{Z}_6]$",
    "mat2_f2_x_mat2_f2": r"$M_2(\mathbb{F}_2)^2$",
    "lam_f2_k3t": r"$\Lambda(\mathbb{F}_2^3)/\Lambda^{\ge 3}$",
    "lam_f3_k2": r"$\Lambda(\mathbb{F}_3^2)$",
    "tri2_f3": r"$T_2(\mathbb{F}_3)$",
    "tri3_f2": r"$T_3(\mathbb{F}_2)$",
    "tri2_f3_x_f3_x_f3": r"$T_2(\mathbb{F}_3)\times\mathbb{F}_3\times\mathbb{F}_3$",
}


@dataclass(frozen=True)
class GridJob:
    ring: str
    weight_decay: float
    lr: float
    seed: int
    checkpoint_tag: str

    @property
    def log_name(self) -> str:
        wd_s = f"{self.weight_decay:g}".replace(".", "p")
        return f"{self.ring}_wd{wd_s}_s{self.seed}.log"

    def candidate_tags(self) -> list[str]:
        tags = [self.checkpoint_tag]
        legacy = LEGACY_TAGS.get((self.ring, self.weight_decay, self.seed))
        if legacy and legacy not in tags:
            tags.append(legacy)
        return tags


def checkpoint_dir(ring: str, tag: str, seed: int) -> Path:
    return ROOT / "checkpoints" / f"{ring}_transformer_{tag}_seed{seed}"


def all_jobs(*, rings: tuple[str, ...] | None = None) -> list[GridJob]:
    ring_list = rings or RINGS
    jobs: list[GridJob] = []
    for ring in ring_list:
        for wd, lr, tag in WD_CELLS:
            for seed in SEEDS:
                jobs.append(GridJob(ring, wd, lr, seed, tag))
    return jobs


def _load_ckpt_metrics(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    cfg = ckpt.get("config", {})
    return {
        "checkpoint": str(path),
        "checkpoint_dir": str(path.parent),
        "checkpoint_tag": cfg.get("checkpoint_tag"),
        "best_test_acc": float(ckpt.get("best_test_acc", ckpt.get("test_acc", 0.0))),
        "best_epoch": int(ckpt.get("best_epoch", ckpt.get("epoch", 0))),
        "weight_decay": cfg.get("weight_decay"),
        "lr": cfg.get("lr"),
        "seed": cfg.get("seed"),
        "zheng_protocol": bool(cfg.get("zheng_protocol", False)),
        "zheng_init_std": cfg.get("zheng_init_std"),
        "max_steps": cfg.get("max_steps"),
    }


KNOWN_ZHENG_TAGS = frozenset(
    {
        "zheng_repro",
        "zheng_proto_wd01",
        "zheng_proto_wd1",
        "zheng_proto_wd2",
        "zheng_proto_lr5e4",
        "zheng_wd01",
        "zheng_wd1",
        "zheng_wd2",
    }
)


def _is_zheng_checkpoint_tag(tag: str) -> bool:
    return tag in KNOWN_ZHENG_TAGS or tag.startswith("zheng_")


def _config_compatible(metrics: dict, job: GridJob, *, matched_tag: str) -> bool:
    zp = metrics.get("zheng_protocol")
    zheng_tag = _is_zheng_checkpoint_tag(matched_tag)
    if zp is False and not zheng_tag:
        return False
    if zp is not True and not zheng_tag:
        legacy = LEGACY_TAGS.get((job.ring, job.weight_decay, job.seed))
        if legacy != matched_tag:
            return False
    if metrics.get("seed") != job.seed:
        return False
    if metrics.get("weight_decay") is not None and abs(float(metrics["weight_decay"]) - job.weight_decay) > 1e-6:
        return False
    if metrics.get("lr") is not None and abs(float(metrics["lr"]) - job.lr) > 1e-9:
        return False
    std = metrics.get("zheng_init_std")
    if std is not None and abs(float(std) - ZHENG_INIT_STD) > 1e-9:
        return False
    return True


def find_existing(job: GridJob) -> dict | None:
    for tag in job.candidate_tags():
        path = checkpoint_dir(job.ring, tag, job.seed) / "best_model.pt"
        metrics = _load_ckpt_metrics(path)
        if metrics and _config_compatible(metrics, job, matched_tag=tag):
            metrics["matched_tag"] = tag
            metrics["canonical_tag"] = job.checkpoint_tag
            metrics["legacy"] = tag != job.checkpoint_tag
            return metrics
    return None


def build_train_cmd(job: GridJob, *, device: str) -> list[str]:
    return [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--ring",
        job.ring,
        "--max-steps",
        str(MAX_STEPS),
        "--checkpoint-tag",
        job.checkpoint_tag,
        "--zheng-protocol",
        "--zheng-init-std",
        str(ZHENG_INIT_STD),
        "--weight-decay",
        str(job.weight_decay),
        "--lr",
        str(job.lr),
        "--seed",
        str(job.seed),
        "--num-workers",
        "0",
        "--mlflow-experiment",
        os.environ.get("MLFLOW_EXPERIMENT", FINAL_RINGS_EXPERIMENT),
        "--device",
        device,
    ]


def run_one(job: GridJob, *, device: str, skip_existing: bool, dry_run: bool) -> dict:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / job.log_name

    existing = find_existing(job)
    if skip_existing and existing is not None:
        return {
            "job": asdict(job),
            "status": "skipped",
            "metrics": existing,
            "log": str(log_path),
        }

    cmd = build_train_cmd(job, device=device)
    if dry_run:
        return {"job": asdict(job), "status": "dry_run", "cmd": cmd, "log": str(log_path)}

    env = {
        **os.environ,
        "REQUIRE_ZHENG_PROTOCOL": "1",
        "MLFLOW_EXPERIMENT": os.environ.get("MLFLOW_EXPERIMENT", FINAL_RINGS_EXPERIMENT),
    }
    t0 = time.perf_counter()
    with open(log_path, "w", encoding="utf-8") as log_f:
        proc = subprocess.run(cmd, cwd=ROOT, stdout=log_f, stderr=subprocess.STDOUT, env=env)
    elapsed = time.perf_counter() - t0

    metrics = find_existing(job)
    status = "ok" if proc.returncode == 0 and metrics is not None else "failed"
    return {
        "job": asdict(job),
        "status": status,
        "returncode": proc.returncode,
        "elapsed_seconds": elapsed,
        "metrics": metrics,
        "log": str(log_path),
    }


def link_existing(*, dry_run: bool = False) -> list[dict]:
    """Symlink canonical checkpoint dirs to legacy-compatible runs."""
    linked: list[dict] = []
    for job in all_jobs():
        existing = find_existing(job)
        if existing is None:
            continue
        if existing.get("matched_tag") == job.checkpoint_tag:
            continue
        src = Path(existing["checkpoint_dir"])
        dst = checkpoint_dir(job.ring, job.checkpoint_tag, job.seed)
        if dst.exists() or dst.is_symlink():
            linked.append({"job": asdict(job), "action": "exists", "dst": str(dst)})
            continue
        if dry_run:
            linked.append({"job": asdict(job), "action": "would_link", "src": str(src), "dst": str(dst)})
            continue
        dst.symlink_to(src.resolve())
        linked.append({"job": asdict(job), "action": "linked", "src": str(src), "dst": str(dst)})
    return linked


def summarize() -> dict:
    rows: list[dict] = []
    by_cell: dict[str, list[dict]] = {}

    for job in all_jobs():
        metrics = find_existing(job)
        row = {
            "ring": job.ring,
            "weight_decay": job.weight_decay,
            "lr": job.lr,
            "seed": job.seed,
            "checkpoint_tag": job.checkpoint_tag,
            "found": metrics is not None,
            "grok": bool(metrics and metrics["best_test_acc"] >= GROK_THRESHOLD),
        }
        if metrics:
            row.update(metrics)
        rows.append(row)
        key = f"{job.ring}|wd={job.weight_decay:g}"
        by_cell.setdefault(key, []).append(row)

    summary_cells: list[dict] = []
    for key, cell_rows in sorted(by_cell.items()):
        ring, wd_part = key.split("|wd=", 1)
        found = [r for r in cell_rows if r["found"]]
        grokked = [r for r in found if r.get("grok")]
        accs = [r["best_test_acc"] for r in found if r.get("best_test_acc") is not None]
        summary_cells.append(
            {
                "ring": ring,
                "weight_decay": float(wd_part),
                "n_seeds_total": len(cell_rows),
                "n_found": len(found),
                "n_grok": len(grokked),
                "grok_rate": len(grokked) / len(cell_rows) if cell_rows else 0.0,
                "best_test_acc_mean": sum(accs) / len(accs) if accs else None,
                "best_test_acc_min": min(accs) if accs else None,
                "best_test_acc_max": max(accs) if accs else None,
            }
        )

    payload = {
        "protocol": "sleep_grid_zheng",
        "rings": list(RINGS),
        "seeds": list(SEEDS),
        "weight_decay_values": [wd for wd, _, _ in WD_CELLS],
        "grok_threshold": GROK_THRESHOLD,
        "zheng_init_std": ZHENG_INIT_STD,
        "max_steps": MAX_STEPS,
        "n_jobs_total": len(rows),
        "n_found": sum(1 for r in rows if r["found"]),
        "cells": summary_cells,
        "runs": rows,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_tex(payload)
    return payload


def write_tex(payload: dict | None = None) -> Path:
    data = payload or json.loads(OUT_JSON.read_text(encoding="utf-8"))
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    cell_map = {(c["ring"], c["weight_decay"]): c for c in data["cells"]}
    row_lines = [
        "% Auto-generated fragment for Table~\\ref{tab:grok-summary}b (scripts/sleep_grid.py --summarize)",
        r"\begin{tabular}{lccc}",
        r"\hline",
        r"Ring & $\lambda{=}0.1$ & $\lambda{=}1.0$ & $\lambda{=}2.0$ \\",
        r"\hline",
    ]
    for ring in RINGS:
        label = RING_LABELS.get(ring, ring)
        cols: list[str] = []
        for wd in (0.1, 1.0, 2.0):
            c = cell_map.get((ring, wd))
            if c is None or c["n_found"] == 0:
                cols.append("---")
            elif c["n_found"] < c["n_seeds_total"]:
                pct = 100.0 * c["grok_rate"]
                mean = 100.0 * c["best_test_acc_mean"] if c["best_test_acc_mean"] is not None else 0.0
                cols.append(f"{pct:.0f}\\% ({c['n_found']}/8; ${mean:.0f}\\%$ mean)")
            else:
                pct = 100.0 * c["grok_rate"]
                mean = 100.0 * c["best_test_acc_mean"] if c["best_test_acc_mean"] is not None else 0.0
                cols.append(f"{pct:.0f}\\% (${mean:.0f}\\%$ mean)")
        row_lines.append(f"{label} & {' & '.join(cols)} \\\\")
    row_lines.extend([r"\hline", r"\end{tabular}", ""])
    OUT_TEX_ROWS.write_text("\n".join(row_lines), encoding="utf-8")

    lines = [
        "% Auto-generated by scripts/sleep_grid.py --summarize",
        r"% Legacy standalone table; main paper uses \input{generated/sleep_grid_grok_rows.tex} in tab:grok-summary.",
        r"\begin{table}[htbp]",
        r"\centering",
        (
            r"\caption{Grok-rate in the systematic grokking grid ($\ge$95\% held-out test) under the reference training recipe "
            r"(init std=0.02, bias-free AdamW decay; 100k steps; seeds 42--49). "
            f"Completed {data['n_found']}/{data['n_jobs_total']} runs at generation time."
            "}"
        ),
        r"\label{tab:sleep-grid-grok-legacy}",
        r"\small",
        *row_lines[1:-1],
        r"\multicolumn{4}{p{11cm}}{\footnotesize "
        r"$\lambda{=}2.0$ uses $\mathrm{lr}{=}5{\times}10^{-3}$; $\lambda{\in}\{0.1,1.0\}$ use $\mathrm{lr}{=}10^{-3}$. "
        r"Consolidated in main text Table~\ref{tab:grok-summary}.}",
        r"\end{table}",
        "",
    ]
    OUT_TEX.write_text("\n".join(lines), encoding="utf-8")
    return OUT_TEX


def run_grid(
    *,
    parallel: int,
    device: str,
    skip_existing: bool,
    dry_run: bool,
    rings: tuple[str, ...] | None,
) -> list[dict]:
    jobs = all_jobs(rings=rings)
    results: list[dict] = []
    if parallel <= 1:
        for job in jobs:
            results.append(run_one(job, device=device, skip_existing=skip_existing, dry_run=dry_run))
            print(f"[{len(results)}/{len(jobs)}] {job.ring} wd={job.weight_decay} s={job.seed} → {results[-1]['status']}")
    else:
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            futs = {
                pool.submit(run_one, job, device=device, skip_existing=skip_existing, dry_run=dry_run): job
                for job in jobs
            }
            done = 0
            for fut in as_completed(futs):
                job = futs[fut]
                res = fut.result()
                results.append(res)
                done += 1
                print(f"[{done}/{len(jobs)}] {job.ring} wd={job.weight_decay} s={job.seed} → {res['status']}")
    summarize()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Sleep grid: Zheng protocol + weight-decay sweep")
    parser.add_argument("--parallel", type=int, default=3, help="Concurrent training jobs (default 3; use 2 on laptops)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", action="store_false", dest="skip_existing")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize", action="store_true", help="Aggregate checkpoints → sleep_grid_results.json")
    parser.add_argument("--link-existing", action="store_true", help="Symlink legacy zheng checkpoints to canonical tags")
    parser.add_argument("--rings", nargs="*", default=None, help="Subset of ring ids")
    args = parser.parse_args()

    if args.summarize:
        payload = summarize()
        print(f"Wrote {OUT_JSON} ({payload['n_found']}/{payload['n_jobs_total']} found)")
        return

    if args.link_existing:
        linked = link_existing(dry_run=args.dry_run)
        print(json.dumps(linked, indent=2))
        summarize()
        return

    rings = tuple(args.rings) if args.rings else None
    jobs = all_jobs(rings=rings)
    print(f"Sleep grid: {len(jobs)} jobs, parallel={args.parallel}, device={args.device}")
    t0 = time.perf_counter()
    results = run_grid(
        parallel=args.parallel,
        device=args.device,
        skip_existing=args.skip_existing,
        dry_run=args.dry_run,
        rings=rings,
    )
    elapsed = time.perf_counter() - t0
    ok = sum(1 for r in results if r["status"] in ("ok", "skipped"))
    failed = sum(1 for r in results if r["status"] == "failed")
    print(f"Done in {elapsed / 3600:.2f} h: ok/skipped={ok}, failed={failed}")
    print(f"Summary → {OUT_JSON}")


if __name__ == "__main__":
    main()
