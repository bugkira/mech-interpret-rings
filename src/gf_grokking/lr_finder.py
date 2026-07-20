"""Learning rate range test (Leslie Smith) для MLP/SGD-оптимизаторов."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


@dataclass
class LRFinderResult:
    """Результат LR range test."""

    lrs: list[float]
    losses: list[float]
    suggested_lr: float
    steepest_lr: float
    min_loss_lr: float
    stopped_early: bool
    inconclusive: bool = False


def _set_lr(optimizer: Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def _smooth(values: list[float], beta: float = 0.9) -> list[float]:
    if not values:
        return []
    out = [values[0]]
    for v in values[1:]:
        out.append(beta * out[-1] + (1 - beta) * v)
    return out


def suggest_lr(lrs: list[float], losses: list[float], *, skip_start: int = 10, skip_end: int = 5) -> tuple[float, float, float, bool]:
    """Подбор LR: fastai steepest/10 или fallback для плоской кривой."""
    if len(lrs) < skip_start + skip_end + 2:
        raise ValueError("not enough LR finder points")

    smooth = _smooth(losses)
    grads = [0.0]
    for i in range(1, len(smooth)):
        grads.append((smooth[i] - smooth[i - 1]) / (lrs[i] / lrs[i - 1] + 1e-12))

    usable = range(skip_start, len(grads) - skip_end)
    steepest_idx = min(usable, key=lambda i: grads[i])
    steepest_lr = lrs[steepest_idx]
    min_idx = min(usable, key=lambda i: smooth[i])
    min_loss_lr = lrs[min_idx]

    usable_losses = [smooth[i] for i in usable]
    # Плоская кривая: до «обрыва» loss почти не меняется (типично для слабого сигнала).
    cutoff = skip_start + max(1, int((len(smooth) - skip_start - skip_end) * 0.85))
    pre_cliff = smooth[skip_start:cutoff]
    inconclusive = (max(pre_cliff) - min(pre_cliff)) < 0.06

    if inconclusive:
        # Плоская кривая: одношаговый loss не падает — берём консервативно ниже «обрыва».
        suggested = min_loss_lr / 5.0
    else:
        suggested = steepest_lr / 10.0

    return suggested, steepest_lr, min_loss_lr, inconclusive


@torch.no_grad()
def _save_state(model: nn.Module, optimizer: Optimizer) -> tuple[dict, dict]:
    model_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    optim_state = optimizer.state_dict()
    return model_state, optim_state


def _restore_state(
    model: nn.Module,
    optimizer: Optimizer,
    model_state: dict,
    optim_state: dict,
) -> None:
    model.load_state_dict(model_state)
    optimizer.load_state_dict(optim_state)


def lr_find(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: Optimizer,
    criterion: nn.Module,
    *,
    start_lr: float = 1e-7,
    end_lr: float = 10.0,
    num_iter: int = 200,
    diverge_factor: float = 4.0,
) -> LRFinderResult:
    """Экспоненциально наращивает LR по батчам, записывает loss.

    Веса и optimizer восстанавливаются после прогона.
    """
    if num_iter < 1:
        raise ValueError("num_iter must be >= 1")
    if start_lr <= 0 or end_lr <= start_lr:
        raise ValueError("require 0 < start_lr < end_lr")

    device = next(model.parameters()).device
    model_state, optim_state = _save_state(model, optimizer)
    model.train()

    lrs: list[float] = []
    losses: list[float] = []
    best_loss = float("inf")
    stopped_early = False
    lr_mult = (end_lr / start_lr) ** (1 / max(num_iter - 1, 1))

    current_lr = start_lr
    _set_lr(optimizer, current_lr)

    batch_iter = iter(train_loader)
    for step in range(num_iter):
        try:
            inputs, labels = next(batch_iter)
        except StopIteration:
            batch_iter = iter(train_loader)
            inputs, labels = next(batch_iter)

        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        loss_val = float(loss.item())
        lrs.append(current_lr)
        losses.append(loss_val)

        if loss_val < best_loss:
            best_loss = loss_val
        if step > 10 and loss_val > diverge_factor * best_loss:
            stopped_early = True
            break

        current_lr *= lr_mult
        _set_lr(optimizer, current_lr)

    suggested, steepest_lr, min_loss_lr, inconclusive = suggest_lr(lrs, losses)
    _restore_state(model, optimizer, model_state, optim_state)

    return LRFinderResult(
        lrs=lrs,
        losses=losses,
        suggested_lr=suggested,
        steepest_lr=steepest_lr,
        min_loss_lr=min_loss_lr,
        stopped_early=stopped_early,
        inconclusive=inconclusive,
    )
