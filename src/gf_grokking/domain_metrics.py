"""Доменно-специфичные метрики для умножения в GF(2^n).

Проверяем, насколько модель MLP(a,b)≈a·b уважает аксиомы поля:
  - коммутативность: pred(a,b) = pred(b,a)
  - ассоциативность: pred(pred(a,b), c) = pred(a, pred(b,c))
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn

from .gf_field import gf_mult_vectorized, split_pair_indices


@torch.no_grad()
def predict_pairs(
    model: nn.Module,
    a: torch.Tensor,
    b: torch.Tensor,
    n: int,
    device: torch.device | str,
    input_format: Literal["onehot", "indices"] = "onehot",
    batch_size: int = 4096,
) -> torch.Tensor:
    """Предсказания model(a,b) для пар [N]."""
    model.eval()
    preds: list[torch.Tensor] = []
    num = 1 << n
    for start in range(0, len(a), batch_size):
        ae = a[start : start + batch_size].to(device)
        be = b[start : start + batch_size].to(device)
        if input_format == "indices":
            x = torch.stack([ae, be], dim=1)
        else:
            from .data import encode_pairs_one_hot

            x = encode_pairs_one_hot(ae, be, num)
        logits = model(x)
        preds.append(logits.argmax(dim=1).cpu())
    return torch.cat(preds, dim=0)


def commutativity_metrics(
    model: nn.Module,
    a: torch.Tensor,
    b: torch.Tensor,
    table: torch.Tensor,
    n: int,
    device: torch.device | str,
    input_format: Literal["onehot", "indices"] = "onehot",
    batch_size: int = 4096,
) -> dict[str, float]:
    """Метрики коммутативности: pred(a,b) vs pred(b,a)."""
    pa = predict_pairs(model, a, b, n, device, input_format, batch_size)
    pb = predict_pairs(model, b, a, n, device, input_format, batch_size)
    labels = table[a, b].long()
    consistent = pa == pb
    both_correct = consistent & (pa == labels)
    return {
        "comm_consistency": consistent.float().mean().item(),
        "comm_both_correct": both_correct.float().mean().item(),
        "comm_inconsistent_count": int((~consistent).sum().item()),
        "pairs_evaluated": len(a),
    }


def associativity_metrics(
    model: nn.Module,
    table: torch.Tensor,
    n: int,
    device: torch.device | str,
    *,
    a: torch.Tensor | None = None,
    b: torch.Tensor | None = None,
    c: torch.Tensor | None = None,
    num_samples: int = 4096,
    seed: int = 0,
    input_format: Literal["onehot", "indices"] = "onehot",
    batch_size: int = 2048,
) -> dict[str, float]:
    """(ab)c vs a(bc): согласованность предсказаний и согласие с истиной."""
    num = 1 << n
    if a is None or b is None or c is None:
        gen = torch.Generator().manual_seed(seed)
        a = torch.randint(0, num, (num_samples,), generator=gen)
        b = torch.randint(0, num, (num_samples,), generator=gen)
        c = torch.randint(0, num, (num_samples,), generator=gen)

    ab = table[a, b].long()
    bc = table[b, c].long()
    true_left = table[ab, c].long()
    true_right = table[a, bc].long()
    assert (true_left == true_right).all(), "GF multiplication must be associative"

    pred_ab = predict_pairs(model, a, b, n, device, input_format, batch_size)
    pred_bc = predict_pairs(model, b, c, n, device, input_format, batch_size)
    pred_left = predict_pairs(model, ab, c, n, device, input_format, batch_size)
    pred_right = predict_pairs(model, a, bc, n, device, input_format, batch_size)

    model_assoc = pred_left == pred_right
    true_assoc = pred_left == true_left
    chain_correct = (pred_ab == ab) & (pred_bc == bc) & (pred_left == true_left)

    return {
        "assoc_model_consistency": model_assoc.float().mean().item(),
        "assoc_accuracy": true_assoc.float().mean().item(),
        "assoc_chain_correct": chain_correct.float().mean().item(),
        "triples_evaluated": len(a),
    }


def compute_domain_metrics(
    model: nn.Module,
    n: int,
    poly: int,
    train_size: int,
    test_size: int,
    seed: int,
    device: torch.device | str,
    *,
    input_format: Literal["onehot", "indices"] = "onehot",
    eval_batch_size: int = 4096,
    assoc_samples: int = 4096,
) -> dict:
    """Коммутативность на test-парах + ассоциативность на случайных тройках."""
    table = gf_mult_vectorized(n, poly)
    _, _, a_test, b_test = split_pair_indices(n, train_size, test_size, seed)

    comm = commutativity_metrics(
        model, a_test, b_test, table, n, device, input_format, eval_batch_size
    )
    assoc = associativity_metrics(
        model, table, n, device,
        num_samples=assoc_samples,
        seed=seed + 1,
        input_format=input_format,
        batch_size=eval_batch_size,
    )
    return {"commutativity": comm, "associativity": assoc}


def detect_input_format(model: nn.Module, n: int) -> Literal["onehot", "indices"]:
    """Эвристика: Transformer (Embedding) vs MLP (one-hot concat)."""
    if any(isinstance(m, nn.Embedding) for m in model.modules()):
        return "indices"
    return "onehot"


def flatten_for_logging(domain: dict) -> dict[str, float]:
    """Плоские скаляры для TensorBoard / MLflow."""
    flat: dict[str, float] = {}
    for key, val in domain.get("commutativity", {}).items():
        if isinstance(val, (int, float)):
            flat[f"comm_{key}"] = float(val)
    for key, val in domain.get("associativity", {}).items():
        if isinstance(val, (int, float)):
            flat[f"assoc_{key}"] = float(val)
    return flat


def plot_algebraic_metrics(domain: dict, output_path: str | Path) -> None:
    """Bar chart: comm/assoc метрики."""
    import matplotlib.pyplot as plt

    path = Path(output_path)
    labels = [
        "comm_consistency",
        "comm_both_correct",
        "assoc_model_consistency",
        "assoc_accuracy",
        "assoc_chain_correct",
    ]
    comm = domain.get("commutativity", {})
    assoc = domain.get("associativity", {})
    values = [
        comm.get("comm_consistency", 0.0),
        comm.get("comm_both_correct", 0.0),
        assoc.get("assoc_model_consistency", 0.0),
        assoc.get("assoc_accuracy", 0.0),
        assoc.get("assoc_chain_correct", 0.0),
    ]

    fig, ax = plt.subplots(figsize=(8, 4))
    x = range(len(labels))
    ax.bar(x, values, color="steelblue", alpha=0.85)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("fraction")
    ax.set_title("Algebraic property metrics")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
