#!/usr/bin/env python3
"""Parallel DAS orchestrator — fan-out independent jobs (seeds / components).

Unlike ring_das_*.py (sequential component loops inside one process), this script
spawns one subprocess per job so RTX 3090 can run ~10 lightweight DAS fits at once.

    uv run python scripts/das_parallel.py --phase multiseed-iia --parallel 10 --device cuda
    uv run python scripts/das_parallel.py --phase disjoint --parallel 10 --merge
    uv run python scripts/das_parallel.py --phase large-eval --parallel 8
    uv run python scripts/das_parallel.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "new_paper_rings"
LOG_DIR = ROOT / "logs" / "das_parallel"
GROK_GRID_JSON = ROOT / "new_paper_rings" / "grok_grid_results.json"

GROKKED_SEEDS = tuple(range(42, 48))
RANDOM_SEEDS = tuple(range(43, 53))
DEFAULT_DAS_RINGS = ("tri2_f3", "tri2_f3_x_f3_x_f3")
PRODUCT_RING = "tri2_f3_x_f3_x_f3"


@dataclass(frozen=True)
class DasJob:
    phase: str
    log_name: str
    cmd: list[str]
    output_path: Path | None

    @property
    def exists(self) -> bool:
        return self.output_path is not None and self.output_path.exists()


def _ckpt(ring: str, tag: str, seed: int) -> Path:
    return ROOT / "checkpoints" / f"{ring}_transformer_{tag}_seed{seed}" / "best_model.pt"


def _pick_grokked_ckpt(ring: str, *, wd: float = 2.0) -> Path | None:
    """Best grokked checkpoint for DAS baseline (prefers grok-grid JSON)."""
    if GROK_GRID_JSON.exists():
        data = json.loads(GROK_GRID_JSON.read_text(encoding="utf-8"))
        grok = [
            r
            for r in data.get("runs", [])
            if r.get("ring") == ring
            and r.get("grok")
            and r.get("found")
            and abs(float(r.get("weight_decay", 0)) - wd) < 1e-6
        ]
        if grok:
            best = max(grok, key=lambda r: float(r.get("best_test_acc", 0)))
            p = Path(best["checkpoint"])
            if p.exists():
                return p
    for seed in GROKKED_SEEDS:
        for tag in ("zheng_wd2", "zheng_wd1", "zheng_wd01"):
            p = _ckpt(ring, tag, seed)
            if p.exists():
                return p
    return None


def _python() -> str:
    return sys.executable


def jobs_multiseed_iia(
    *,
    rings: tuple[str, ...],
    device: str,
    grokked_seeds: tuple[int, ...],
    random_seeds: tuple[int, ...],
    wd_tag: str,
) -> list[DasJob]:
    jobs: list[DasJob] = []
    for ring in rings:
        for seed in grokked_seeds:
            ckpt = _ckpt(ring, wd_tag, seed)
            if not ckpt.exists():
                ckpt = _pick_grokked_ckpt(ring)
            if ckpt is None or not ckpt.exists():
                continue
            tag = f"grokked_{wd_tag}_s{seed}"
            out = OUT_DIR / f"{ring}_das_iia_{tag}.json"
            jobs.append(
                DasJob(
                    phase="multiseed-iia",
                    log_name=f"{ring}_iia_grokked_s{seed}.log",
                    cmd=[
                        _python(),
                        str(ROOT / "scripts" / "ring_das_iia.py"),
                        "--ring",
                        ring,
                        "--checkpoint",
                        str(ckpt),
                        "--tag",
                        tag,
                        "--device",
                        device,
                    ],
                    output_path=out,
                )
            )
        base = _pick_grokked_ckpt(ring)
        if base is None:
            continue
        for rs in random_seeds:
            tag = f"randinit_rs{rs}"
            out = OUT_DIR / f"{ring}_das_iia_{tag}.json"
            jobs.append(
                DasJob(
                    phase="multiseed-iia",
                    log_name=f"{ring}_iia_randinit_rs{rs}.log",
                    cmd=[
                        _python(),
                        str(ROOT / "scripts" / "ring_das_iia.py"),
                        "--ring",
                        ring,
                        "--checkpoint",
                        str(base),
                        "--random-init",
                        "--run-seed",
                        str(rs),
                        "--tag",
                        tag,
                        "--device",
                        device,
                    ],
                    output_path=out,
                )
            )
    return jobs


def jobs_disjoint(
    *,
    ring: str,
    checkpoint: Path,
    components: tuple[str, ...] | None,
    device: str,
    seed: int,
) -> list[DasJob]:
    from gf_grokking.ring_wedderburn import get_wedderburn_spec

    spec = get_wedderburn_spec(ring)
    comps = components or tuple(spec.component_names)
    tag = f"_seed{seed}"
    jobs: list[DasJob] = []
    for comp in comps:
        out = OUT_DIR / f"{ring}_{comp}_das_disjoint_dense{tag}.json"
        jobs.append(
            DasJob(
                phase="disjoint",
                log_name=f"{ring}_disjoint_{comp}_s{seed}.log",
                cmd=[
                    _python(),
                    str(ROOT / "scripts" / "ring_das_disjoint_train.py"),
                    "--ring",
                    ring,
                    "--checkpoint",
                    str(checkpoint),
                    "--components",
                    comp,
                    "--seed",
                    str(seed),
                    "--device",
                    device,
                    "--skip-existing",
                ],
                output_path=out,
            )
        )
    return jobs


def jobs_large_eval(
    *,
    ring: str,
    checkpoint: Path,
    components: tuple[str, ...],
    das_seeds: tuple[int, ...],
    device: str,
) -> list[DasJob]:
    jobs: list[DasJob] = []
    for ds in das_seeds:
        tag = f"_seed{ds}"
        if len(components) == 1:
            comp_suffix = f"_{components[0]}"
            out = OUT_DIR / f"{ring}_das_large_eval{comp_suffix}{tag}.json"
        else:
            out = OUT_DIR / f"{ring}_das_large_eval{tag}.json"
        cmd = [
            _python(),
            str(ROOT / "scripts" / "ring_das_large_eval.py"),
            "--ring",
            ring,
            "--checkpoint",
            str(checkpoint),
            "--das-seed",
            str(ds),
            "--device",
            device,
        ]
        if len(components) == 1:
            cmd.extend(["--components", components[0]])
        jobs.append(
            DasJob(
                phase="large-eval",
                log_name=f"{ring}_large_eval_ds{ds}.log",
                cmd=cmd,
                output_path=out if out.exists() else None,
            )
        )
    return jobs


def merge_disjoint_aggregate(ring: str, seed: int = 42) -> Path:
    from gf_grokking.ring_wedderburn import get_wedderburn_spec

    spec = get_wedderburn_spec(ring)
    tag = f"_seed{seed}"
    rows: list[dict] = []
    ckpt = None
    test_acc = None
    for comp in spec.component_names:
        p = OUT_DIR / f"{ring}_{comp}_das_disjoint_dense{tag}.json"
        if not p.exists():
            raise FileNotFoundError(f"missing component JSON: {p}")
        row = json.loads(p.read_text(encoding="utf-8"))
        rows.append(row)
        ckpt = row.get("checkpoint", ckpt)
        test_acc = row.get("test_accuracy", test_acc)

    agg = {
        "ring": ring,
        "checkpoint": ckpt,
        "test_accuracy": test_acc,
        "protocol": "disjoint_element_dense_das",
        "pyvene_reference": "LowRankRotatedSpaceIntervention",
        "seed": seed,
        "element_seed": rows[0].get("element_seed", seed),
        "bases_per_source": rows[0].get("bases_per_source", 50),
        "element_train_fraction": rows[0].get("element_train_fraction", 0.7),
        "min_subspace_dim": rows[0].get("min_subspace_dim"),
        "rank_override": rows[0].get("rank_override"),
        "components": rows,
    }
    out = OUT_DIR / f"{ring}_das_disjoint_dense{tag}.json"
    out.write_text(json.dumps(agg, indent=2), encoding="utf-8")
    print(f"→ merged {out}")
    return out


def run_one(job: DasJob, *, skip_existing: bool, dry_run: bool) -> dict:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / job.log_name
    if skip_existing and job.exists:
        return {"job": job.log_name, "status": "skipped", "output": str(job.output_path)}
    if dry_run:
        return {"job": job.log_name, "status": "dry_run", "cmd": job.cmd}
    t0 = time.perf_counter()
    with open(log_path, "w", encoding="utf-8") as log_f:
        proc = subprocess.run(job.cmd, cwd=ROOT, stdout=log_f, stderr=subprocess.STDOUT)
    elapsed = time.perf_counter() - t0
    ok = proc.returncode == 0 and (job.output_path is None or job.output_path.exists())
    return {
        "job": job.log_name,
        "status": "ok" if ok else "failed",
        "returncode": proc.returncode,
        "elapsed_seconds": elapsed,
        "log": str(log_path),
    }


def run_jobs(
    jobs: list[DasJob],
    *,
    parallel: int,
    skip_existing: bool,
    dry_run: bool,
) -> list[dict]:
    results: list[dict] = []
    if parallel <= 1:
        for job in jobs:
            res = run_one(job, skip_existing=skip_existing, dry_run=dry_run)
            results.append(res)
            print(f"[{len(results)}/{len(jobs)}] {job.log_name} → {res['status']}")
        return results

    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futs = {pool.submit(run_one, job, skip_existing=skip_existing, dry_run=dry_run): job for job in jobs}
        done = 0
        for fut in as_completed(futs):
            job = futs[fut]
            res = fut.result()
            results.append(res)
            done += 1
            print(f"[{done}/{len(jobs)}] {job.log_name} → {res['status']}")
    return results


def main() -> int:
    p = argparse.ArgumentParser(description="Parallel DAS job orchestrator")
    p.add_argument(
        "--phase",
        choices=("multiseed-iia", "disjoint", "large-eval", "all"),
        default="multiseed-iia",
    )
    p.add_argument("--parallel", type=int, default=10)
    p.add_argument("--device", default="cuda")
    p.add_argument("--rings", nargs="*", default=list(DEFAULT_DAS_RINGS))
    p.add_argument("--wd-tag", default="zheng_wd2", help="Checkpoint tag for grokked multiseed")
    p.add_argument("--checkpoint", type=Path, default=None, help="Override checkpoint for disjoint/large-eval")
    p.add_argument("--das-seeds", type=int, nargs="*", default=list(range(42, 52)))
    p.add_argument("--components", nargs="*", default=None)
    p.add_argument("--skip-existing", action="store_true", default=True)
    p.add_argument("--no-skip-existing", action="store_false", dest="skip_existing")
    p.add_argument("--merge", action="store_true", help="After disjoint phase, merge component JSONs")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    rings = tuple(args.rings)
    jobs: list[DasJob] = []

    if args.phase in ("multiseed-iia", "all"):
        jobs.extend(
            jobs_multiseed_iia(
                rings=rings,
                device=args.device,
                grokked_seeds=GROKKED_SEEDS,
                random_seeds=RANDOM_SEEDS,
                wd_tag=args.wd_tag,
            )
        )

    ckpt = args.checkpoint
    if ckpt is None and args.phase in ("disjoint", "large-eval", "all"):
        ckpt = _pick_grokked_ckpt(PRODUCT_RING, wd=0.1) or _pick_grokked_ckpt(PRODUCT_RING)
    if ckpt is not None and not ckpt.is_absolute():
        ckpt = ROOT / ckpt

    if args.phase in ("disjoint", "all") and ckpt is not None:
        comps = tuple(args.components) if args.components else None
        jobs.extend(jobs_disjoint(ring=PRODUCT_RING, checkpoint=ckpt, components=comps, device=args.device, seed=42))

    if args.phase in ("large-eval", "all") and ckpt is not None:
        comps = tuple(args.components) if args.components else ("L_rad",)
        jobs.extend(
            jobs_large_eval(
                ring=PRODUCT_RING,
                checkpoint=ckpt,
                components=comps,
                das_seeds=tuple(args.das_seeds),
                device=args.device,
            )
        )

    if not jobs:
        print("No jobs (missing checkpoints?). Run grok_grid --summarize first.", file=sys.stderr)
        return 1

    print(f"DAS parallel: {len(jobs)} jobs, phase={args.phase}, parallel={args.parallel}")
    t0 = time.perf_counter()
    results = run_jobs(jobs, parallel=args.parallel, skip_existing=args.skip_existing, dry_run=args.dry_run)
    elapsed = time.perf_counter() - t0
    ok = sum(1 for r in results if r["status"] in ("ok", "skipped"))
    failed = sum(1 for r in results if r["status"] == "failed")
    print(f"Done in {elapsed/60:.1f} min: ok/skipped={ok}, failed={failed}")

    if args.merge and args.phase in ("disjoint", "all") and not args.dry_run:
        merge_disjoint_aggregate(PRODUCT_RING, seed=42)

    if args.phase in ("multiseed-iia", "all") and not args.dry_run and failed == 0:
        for ring in rings:
            try:
                subprocess.run(
                    [_python(), str(ROOT / "scripts" / "das_random_baseline_summary.py"), ring],
                    cwd=ROOT,
                    check=False,
                )
                subprocess.run(
                    [
                        _python(),
                        str(ROOT / "scripts" / "das_multiseed_stats.py"),
                        "--ring",
                        ring,
                        "--grok-glob",
                        f"{ring}_das_iia_grokked_*_s*.json",
                    ],
                    cwd=ROOT,
                    check=False,
                )
            except Exception as exc:
                print(f"post-aggregate {ring}: {exc}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
