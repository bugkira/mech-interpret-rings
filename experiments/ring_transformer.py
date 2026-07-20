#!/usr/bin/env python3
"""Transformer на умножении в конечных некоммутативных кольцах (Zheng protocol).

Примеры:
    uv run python experiments/ring_transformer.py --ring tri2_f3
    uv run python experiments/ring_transformer.py --ring mat2_f3 --smoke --max-steps 500
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch

from gf_grokking import mlflow_logger, viz
from gf_grokking.alt_group_data import create_group_transformer_loaders
from gf_grokking.finite_rings import RING_REGISTRY, build_mul_table, get_ring_spec
from gf_grokking.models.group_transformer import GroupTransformer
from gf_grokking.ring_transformer_iia import measure_ring_transformer_iia_row
from gf_grokking.ring_wedderburn import get_wedderburn_spec
from gf_grokking.zheng_protocol import (
    ZHENG_INIT_STD,
    build_adamw_optimizer,
    describe_optimizer_groups,
    init_group_transformer_zheng,
)
from gf_grokking.train import Float64CrossEntropyLoss, train_model
from gf_grokking.train_budget import (
    BUDGET_MATCHED_EXPOSURES,
    BUDGET_STEPS,
    default_train_fraction,
    num_epochs as budget_num_epochs,
    resolve_training_budget,
    steps_per_epoch as budget_steps_per_epoch,
    train_size as budget_train_size,
)

OUT_DIR = Path("new_paper_about_a5")
PLOTS_DIR = Path("plots/alt_group_interp/article_runs")

PAPER_D_MODEL = 128
PAPER_NHEAD = 4
PAPER_FFN = 512
PAPER_LAYERS = 1
PAPER_LR = 1e-3
PAPER_WD = 1.0
PAPER_BATCH = 512
PAPER_MAX_STEPS = 100_000
PAPER_TRAIN_FRAC_SMALL = 0.5
PAPER_TRAIN_FRAC_LARGE = 0.3


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_cuda(device: str) -> None:
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        return
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


def steps_to_epochs(max_steps: int, train_size: int, batch_size: int) -> int:
    return budget_num_epochs(max_steps, train_size, batch_size)


def main() -> int:
    parser = argparse.ArgumentParser(description="Ring Transformer (Nanda/Zheng protocol)")
    parser.add_argument(
        "--ring",
        type=str,
        required=True,
        choices=sorted(RING_REGISTRY),
        help="ID кольца из finite_rings",
    )
    parser.add_argument("--train-fraction", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=PAPER_BATCH)
    parser.add_argument("--max-steps", type=int, default=PAPER_MAX_STEPS)
    parser.add_argument(
        "--budget-mode",
        choices=[BUDGET_STEPS, BUDGET_MATCHED_EXPOSURES],
        default=BUDGET_STEPS,
        help="steps=Zheng fixed budget; matched_exposures=align train-pair passes to reference ring",
    )
    parser.add_argument(
        "--exposure-reference-ring",
        type=str,
        default="tri2_f3_x_f3_x_f3",
        help="Reference ring for matched_exposures (default: product ring @ 100k steps)",
    )
    parser.add_argument("--lr", type=float, default=PAPER_LR)
    parser.add_argument("--weight-decay", type=float, default=PAPER_WD)
    parser.add_argument("--d-model", type=int, default=PAPER_D_MODEL)
    parser.add_argument("--nhead", type=int, default=PAPER_NHEAD)
    parser.add_argument("--ffn-dim", type=int, default=PAPER_FFN)
    parser.add_argument("--num-layers", type=int, default=PAPER_LAYERS)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--layernorm", action="store_true")
    parser.add_argument(
        "--loss-fp64",
        action="store_true",
        help="Cross-entropy in float64 (logits.double()) to curb fp32 norm drift",
    )
    parser.add_argument(
        "--adam-eps",
        type=float,
        default=1e-8,
        help="AdamW epsilon (default 1e-8; try 1e-5..1e-4 to damp optimizer spikes)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=-1)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--override-lr", type=float, default=None, help="Fine-tune lr after --resume")
    parser.add_argument("--extra-epochs", type=int, default=None)
    parser.add_argument("--checkpoint-tag", type=str, default=None)
    parser.add_argument(
        "--zheng-protocol",
        action="store_true",
        help="Zheng/Nanda init (normal std=0.02) + AdamW without bias decay",
    )
    parser.add_argument(
        "--zheng-init-std",
        type=float,
        default=None,
        help="Init std when --zheng-protocol (default 0.02)",
    )
    parser.add_argument(
        "--mlflow-experiment",
        type=str,
        default=None,
        help="MLflow experiment name (default: MLFLOW_EXPERIMENT env or gf_grokking_rings)",
    )
    parser.add_argument("--iia-interval", type=int, default=0, help="Wedderburn IIA every N ep (0=off)")
    parser.add_argument("--iia-max-pairs", type=int, default=40)
    args = parser.parse_args()

    if os.environ.get("REQUIRE_ZHENG_PROTOCOL") == "1" and not args.zheng_protocol:
        raise SystemExit("REQUIRE_ZHENG_PROTOCOL=1 but --zheng-protocol not set")

    if args.zheng_init_std is not None and not args.zheng_protocol:
        raise SystemExit("--zheng-init-std requires --zheng-protocol")

    zheng_init_std = args.zheng_init_std if args.zheng_init_std is not None else ZHENG_INIT_STD

    resume_ckpt = None
    if args.resume:
        resume_ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        ckpt_cfg = resume_ckpt.get("config", {})
        args.ring = ckpt_cfg.get("ring_id", args.ring)
        args.train_fraction = ckpt_cfg.get("train_fraction", args.train_fraction)
        args.batch_size = ckpt_cfg.get("batch_size", args.batch_size)
        args.max_steps = ckpt_cfg.get("max_steps", args.max_steps)
        args.lr = ckpt_cfg.get("lr", args.lr)
        args.weight_decay = ckpt_cfg.get("weight_decay", args.weight_decay)
        args.d_model = ckpt_cfg.get("d_model", args.d_model)
        args.nhead = ckpt_cfg.get("nhead", args.nhead)
        args.ffn_dim = ckpt_cfg.get("ffn_dim", args.ffn_dim)
        args.num_layers = ckpt_cfg.get("num_layers", args.num_layers)
        args.dropout = ckpt_cfg.get("dropout", args.dropout)
        args.layernorm = ckpt_cfg.get("layernorm", args.layernorm)
        args.seed = ckpt_cfg.get("seed", args.seed)
        if args.checkpoint_tag is None and ckpt_cfg.get("checkpoint_tag"):
            args.checkpoint_tag = ckpt_cfg["checkpoint_tag"]
        if args.override_lr is not None:
            args.lr = args.override_lr

    if not args.resume:
        set_seed(args.seed)

    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    configure_cuda(device)

    spec = get_ring_spec(args.ring)
    n_elements = spec.n_elements
    train_fraction = (
        args.train_fraction
        if args.train_fraction is not None
        else default_train_fraction(n_elements)
    )
    total_pairs = n_elements * n_elements
    train_size = int(total_pairs * train_fraction)
    test_size = total_pairs - train_size

    budget = resolve_training_budget(
        args.ring,
        budget_mode=args.budget_mode,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        train_fraction=train_fraction,
        exposure_reference_ring=args.exposure_reference_ring,
    )
    num_epochs = int(budget["num_epochs"])
    args.max_steps = int(budget["max_steps"])

    if args.resume and args.extra_epochs is not None:
        num_epochs = args.extra_epochs
        spe = budget_steps_per_epoch(train_size, args.batch_size)
        resume_epoch = int(resume_ckpt.get("epoch", 0)) if resume_ckpt else 0
        args.max_steps = (resume_epoch + num_epochs) * spe
        budget = {
            **budget,
            "num_epochs": num_epochs,
            "max_steps": args.max_steps,
            "pair_exposures": args.max_steps / spe,
        }

    if args.smoke:
        args.max_steps = min(args.max_steps, 500)
        num_epochs = steps_to_epochs(args.max_steps, train_size, args.batch_size)
        args.eval_interval = 10
        args.log_interval = 10
        if args.iia_interval > 0:
            args.iia_interval = min(args.iia_interval, 25)

    mult_table, _ = build_mul_table(args.ring)
    wb_spec = get_wedderburn_spec(args.ring)
    per_epoch_iia: list[dict] = []
    ckpt_tag = f"_{args.checkpoint_tag}" if args.checkpoint_tag else ""
    iia_json_path = OUT_DIR / f"{args.ring}_transformer{ckpt_tag}_wedderburn_iia_trace_seed{args.seed}.json"

    print(
        f"{spec.name} (|A|={n_elements}): train={train_size}, test={test_size}, "
        f"batch={args.batch_size}, budget={args.budget_mode}, "
        f"max_steps={args.max_steps} → epochs={num_epochs}, "
        f"pair_exposures≈{budget['pair_exposures']:.0f}, "
        f"d_model={args.d_model}, layers={args.num_layers}, LN={args.layernorm}, device={device}"
    )

    train_loader, test_loader, _, _ = create_group_transformer_loaders(
        train_size=train_size,
        test_size=test_size,
        batch_size=args.batch_size,
        device=device,
        seed=args.seed,
        num_workers=args.num_workers,
        ring_id=args.ring,
    )

    model = GroupTransformer(
        num_elements=n_elements,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
        layernorm=args.layernorm,
    ).to(device)

    if args.zheng_protocol:
        init_group_transformer_zheng(model, std=zheng_init_std)

    optimizer = build_adamw_optimizer(
        model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        eps=args.adam_eps,
        exclude_bias_from_decay=args.zheng_protocol,
    )
    if args.zheng_protocol:
        og = describe_optimizer_groups(optimizer)
        print(
            f"Zheng protocol: init_std={zheng_init_std}, "
            f"AdamW decay={og['decay']} no_decay={og['no_decay']}"
        )
    elif not args.resume:
        print("Legacy protocol: PyTorch default init, AdamW wd on all params incl. biases")

    run_config = {
        "model": "group_transformer",
        "task": "ring_multiplication",
        "ring_id": args.ring,
        "ring_name": spec.name,
        "ring_type": spec.ring_type,
        "commutative": spec.commutative,
        "num_elements": n_elements,
        "train_fraction": train_fraction,
        "train_size": train_size,
        "test_size": test_size,
        "batch_size": args.batch_size,
        "max_steps": args.max_steps,
        "num_epochs": num_epochs,
        "budget_mode": args.budget_mode,
        "steps_per_epoch": budget["steps_per_epoch"],
        "pair_exposures": budget["pair_exposures"],
        "exposure_reference_ring": budget.get("exposure_reference_ring"),
        "reference_pair_exposures": budget.get("reference_pair_exposures"),
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "d_model": args.d_model,
        "nhead": args.nhead,
        "ffn_dim": args.ffn_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "layernorm": args.layernorm,
        "loss_fp64": args.loss_fp64,
        "adam_eps": args.adam_eps,
        "seed": args.seed,
        "eval_interval": args.eval_interval,
        "log_interval": args.log_interval,
        "num_workers": args.num_workers,
        "device": str(device),
        "input_encoding": "token_indices",
        "checkpoint_tag": args.checkpoint_tag,
        "resume_from": args.resume,
        "override_lr": args.override_lr,
        "paper_protocol": "zheng_from_groups_to_rings",
        "zheng_protocol": args.zheng_protocol,
        "zheng_init_std": zheng_init_std if args.zheng_protocol else None,
        "iia_interval": args.iia_interval,
        "iia_max_pairs": args.iia_max_pairs,
        "wedderburn_components": list(wb_spec.component_names),
    }
    if spec.encoding_version is not None:
        run_config["encoding_version"] = spec.encoding_version

    if args.smoke:
        experiment_name = mlflow_logger.SMOKE_EXPERIMENT
    elif args.mlflow_experiment:
        experiment_name = args.mlflow_experiment
    else:
        experiment_name = os.environ.get(
            "MLFLOW_EXPERIMENT",
            mlflow_logger.RING_EXPERIMENT,
        )
    run_prefix = "smoke_" if args.smoke else ""
    ckpt_tag = f"_{args.checkpoint_tag}" if args.checkpoint_tag else ""
    run_name = (
        f"{run_prefix}{args.ring}_transformer_d{args.d_model}"
        f"_seed{args.seed}{ckpt_tag}"
    )
    mlflow_logger.setup_experiment(
        experiment_name=experiment_name,
        run_name=run_name,
        tags={
            "model": "group_transformer",
            "ring": args.ring,
            "ring_type": spec.ring_type,
            "mode": "smoke" if args.smoke else "research",
        },
    )
    mlflow_logger.log_params(run_config)
    mlflow_logger.log_alt_group_dataset_info(
        spec.name, spec.n_elements, n_elements, train_size, test_size
    )

    run_id = mlflow_logger.active_run_id() or f"seed{args.seed}"
    base = "_smoke/" if args.smoke else ""
    run_tag = f"{args.ring}_transformer{ckpt_tag}_seed{args.seed}"
    checkpoint_dir = f"checkpoints/{base}{run_tag}"

    def after_epoch(epoch: int, test_acc: float, train_acc: float) -> bool:
        if args.iia_interval <= 0:
            return False
        if epoch % args.iia_interval != 0 and epoch != num_epochs:
            return False
        iia_row = measure_ring_transformer_iia_row(
            model,
            mult_table,
            wb_spec,
            None,
            device=str(device),
            max_pairs_per_component=args.iia_max_pairs,
            seed=args.seed + epoch,
        )
        row = {"epoch": epoch, "test_acc": float(test_acc), "train_acc": float(train_acc), **iia_row}
        per_epoch_iia.append(row)
        mlflow_logger.log_metrics(
            {
                "mean_raw_iia": iia_row["mean_raw_iia"],
                "mean_subspace_iia": iia_row["mean_subspace_iia"],
                **{k: v for k, v in iia_row.items() if k.startswith("iia_")},
                **{k: v for k, v in iia_row.items() if k.startswith(("logistic_probe_", "mlp_probe_", "mean_"))},
            },
            step=epoch,
        )
        payload = {
            "ring_id": args.ring,
            "ring_name": spec.name,
            "seed": args.seed,
            "mlflow_run_id": run_id,
            "run_config": run_config,
            "per_epoch": per_epoch_iia,
        }
        iia_json_path.parent.mkdir(parents=True, exist_ok=True)
        iia_json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(
            f"  [IIA ep={epoch}] test={test_acc:.4f} raw={iia_row['mean_raw_iia']:.3f} "
            f"sub={iia_row['mean_subspace_iia']:.3f}",
            flush=True,
        )
        return False

    criterion = Float64CrossEntropyLoss() if args.loss_fp64 else torch.nn.CrossEntropyLoss()

    result = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=num_epochs,
        eval_interval=args.eval_interval,
        log_interval=args.log_interval,
        tb_log_dir=f"runs/{base}{run_tag}",
        mlflow_log=True,
        checkpoint_dir=checkpoint_dir,
        config=run_config,
        checkpoint_keep_last=2 if args.smoke else 5,
        resume_from=args.resume,
        override_lr=args.override_lr,
        after_epoch=after_epoch if args.iia_interval > 0 else None,
    )

    best_test = max(result.test_acc_history) if result.test_acc_history else 0.0
    print(f"\n{'=' * 60}")
    print(f"Результат {spec.name} — RingTransformer")
    print(f"  Final Train Acc:  {result.final_train_acc:.4f}")
    print(f"  Final Test Acc:   {result.final_test_acc:.4f}")
    print(f"  Best Test Acc:    {best_test:.4f} (epoch {result.best_epoch})")
    print(f"  Elapsed:          {result.elapsed_seconds:.0f}s")
    if result.best_model_path:
        print(f"  Best Model:       {result.best_model_path}")
    print(f"{'=' * 60}")

    mlflow_logger.log_json(run_config, "config", "config.json")
    mlflow_logger.log_model(model, "model")
    mlflow_logger.log_metrics({
        "final_train_acc": result.final_train_acc,
        "final_test_acc": result.final_test_acc,
        "best_test_acc": best_test,
        "best_epoch": float(result.best_epoch),
        "elapsed_seconds": result.elapsed_seconds,
    })

    plot_dir = Path(f"plots/{base}{run_id}")
    plot_dir.mkdir(parents=True, exist_ok=True)
    curves_path = plot_dir / "curves.png"
    viz.plot_learning_curves(result, save_path=curves_path, show=False)
    mlflow_logger.log_artifact(curves_path, "plots")
    if result.best_model_path:
        mlflow_logger.log_artifact(result.best_model_path, "checkpoints")
    if result.last_model_path:
        mlflow_logger.log_artifact(result.last_model_path, "checkpoints")
    if per_epoch_iia:
        mlflow_logger.log_json({"per_epoch": per_epoch_iia}, "iia", "wedderburn_iia_trace.json")
        mlflow_logger.log_artifact(iia_json_path, "iia")

    mlflow_logger.end_run()
    print(f"\nПлоты: {plot_dir}/ (MLflow run {run_id})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
