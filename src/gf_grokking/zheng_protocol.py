"""Nanda / Zheng grokking training conventions (init + AdamW param groups)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .models.group_transformer import GroupTransformer

ZHENG_INIT_STD = 0.02


def zheng_init_std(d_model: int, *, fixed_std: float = ZHENG_INIT_STD) -> float:
    """TransformerLens-style scale: min(fixed_std, 1/sqrt(d_model)) is not used; Zheng uses 0.02."""
    return fixed_std


def init_group_transformer_zheng(
    model: GroupTransformer,
    *,
    std: float | None = None,
) -> None:
    """Normal_(0, std) on embedding + all Linear weights; zero biases."""
    scale = std if std is not None else zheng_init_std(model.d_model)
    nn.init.normal_(model.embedding.embedding.weight, std=scale)
    for mod in model.modules():
        if isinstance(mod, nn.Linear):
            nn.init.normal_(mod.weight, std=scale)
            if mod.bias is not None:
                nn.init.zeros_(mod.bias)


def split_decay_param_groups(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Weight matrices (ndim >= 2) vs biases / scalars (ndim < 2)."""
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2:
            decay.append(p)
        else:
            no_decay.append(p)
    return decay, no_decay


def build_adamw_optimizer(
    model: nn.Module,
    *,
    lr: float,
    weight_decay: float,
    eps: float = 1e-8,
    exclude_bias_from_decay: bool = True,
) -> torch.optim.AdamW:
    if not exclude_bias_from_decay:
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, eps=eps)
    decay, no_decay = split_decay_param_groups(model)
    groups: list[dict] = []
    if decay:
        groups.append({"params": decay, "weight_decay": weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return torch.optim.AdamW(groups, lr=lr, eps=eps)


def describe_optimizer_groups(optimizer: torch.optim.AdamW) -> dict[str, int]:
    out: dict[str, int] = {"decay": 0, "no_decay": 0}
    for g in optimizer.param_groups:
        key = "decay" if g.get("weight_decay", 0) > 0 else "no_decay"
        out[key] += len(g["params"])
    return out
