"""Training loop для GF(2^n) арифметики с метриками и TensorBoard."""

import time
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .eval import algebraic_metrics, confusion_matrix, error_distribution


class Float64CrossEntropyLoss(nn.Module):
    """Cross-entropy in float64 to reduce fp32 gradient drift at 100% train acc."""

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits.double(), targets)


@dataclass
class TrainingResult:
    """Результаты тренировки."""

    train_loss_history: list[float]
    train_acc_history: list[float]
    test_loss_history: list[float]
    test_acc_history: list[float]
    elapsed_seconds: float
    final_train_acc: float
    final_test_acc: float
    best_epoch: int
    best_model_path: str | None = None
    last_model_path: str | None = None


def _build_checkpoint_payload(
    epoch: int,
    model: nn.Module,
    optimizer: Optimizer,
    train_acc: float,
    test_acc: float,
    config: dict[str, Any] | None,
    full: bool = True,
) -> dict[str, Any]:
    """Собирает payload чекпойнта.

    Args:
        full: если True — полный payload для resume (optimizer + grad + rng);
            если False — лёгкий снимок (только веса + метрики + config) для
            периодических чекпойнтов и анализа динамики.
    """
    payload: dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "train_acc": train_acc,
        "test_acc": test_acc,
        "config": config or {},
    }
    if not full:
        return payload

    grad_snapshot: dict[str, torch.Tensor] = {
        name: param.grad.detach().clone()
        for name, param in model.named_parameters()
        if param.grad is not None
    }
    payload["optimizer_state_dict"] = optimizer.state_dict()
    payload["grad_state_dict"] = grad_snapshot
    payload["rng_state"] = torch.get_rng_state()
    if torch.cuda.is_available():
        payload["cuda_rng_state"] = torch.cuda.get_rng_state_all()
    return payload


def load_resume_state(
    checkpoint_path: str | Path,
    model: nn.Module,
    optimizer: Optimizer,
) -> dict[str, Any]:
    """Восстанавливает model/optimizer/RNG из полного чекпойнта (last.pt / best_model.pt)."""
    ckpt = torch.load(checkpoint_path, map_location=next(model.parameters()).device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if "rng_state" in ckpt:
        rng = ckpt["rng_state"]
        if isinstance(rng, torch.Tensor):
            rng = rng.cpu().clone().to(torch.uint8)
        else:
            rng = torch.ByteTensor(rng)
        torch.set_rng_state(rng)
    if ckpt.get("cuda_rng_state") and torch.cuda.is_available():
        cuda_states = []
        for state in ckpt["cuda_rng_state"]:
            if isinstance(state, torch.Tensor):
                cuda_states.append(state.cpu().clone().to(torch.uint8))
            else:
                cuda_states.append(torch.ByteTensor(state))
        torch.cuda.set_rng_state_all(cuda_states)
    return ckpt


def _load_best_metrics(checkpoint_dir: Path) -> tuple[float, int]:
    """Читает best test acc/epoch из best_model.pt, если есть."""
    best_path = checkpoint_dir / "best_model.pt"
    if not best_path.exists():
        return 0.0, 0
    ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
    return float(ckpt.get("test_acc", 0.0)), int(ckpt.get("epoch", 0))


def _prune_checkpoints(checkpoint_dir: Path, keep_last: int) -> None:
    """Оставляет только keep_last последних периодических чекпойнтов."""
    if keep_last <= 0:
        return
    ckpts = sorted(
        checkpoint_dir.glob("checkpoint_epoch*.pt"),
        key=lambda p: int(p.stem.replace("checkpoint_epoch", "")),
    )
    for stale in ckpts[:-keep_last]:
        stale.unlink(missing_ok=True)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> tuple[float, float]:
    """Вычисляет loss + accuracy на датасете.

    Returns:
        (mean_loss, mean_accuracy)
    """
    device = next(model.parameters()).device
    use_cuda = device.type == "cuda"
    non_blocking = use_cuda
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=non_blocking)
        labels = labels.to(device, non_blocking=non_blocking)
        logits = model(inputs)
        loss = criterion(logits, labels)
        total_loss += loss.item() * inputs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: Optimizer,
    criterion: nn.Module = nn.CrossEntropyLoss(),
    num_epochs: int = 5000,
    eval_interval: int = 1,
    log_interval: int = 100,
    tb_log_dir: str | Path = "runs/gf_grokking",
    mlflow_log: bool = True,
    checkpoint_dir: str | Path | None = "checkpoints",
    save_best_model: bool = True,
    config: dict[str, Any] | None = None,
    field_n: int | None = None,
    checkpoint_keep_last: int | None = None,
    resume_from: str | Path | None = None,
    override_lr: float | None = None,
    after_epoch: Callable[[int, float, float], bool] | None = None,
) -> TrainingResult:
    """Основной training loop.

    Args:
        model: PyTorch-модель (train на device).
        train_loader: TRAIN DataLoader.
        test_loader: TEST DataLoader.
        optimizer: оптимизатор (Adam).
        criterion: loss function.
        num_epochs: количество эпох.
        eval_interval: полный train+test eval каждые N эпох (1 = каждую эпоху).
            На пропущенных эпохах обучение идёт, метрики не считаются.
            Последняя эпоха всегда с eval.
        log_interval: вывод каждые N эпох.
        tb_log_dir: директория TensorBoard логов.
        mlflow_log: логировать метрики в MLflow.
        checkpoint_dir: директория для чекпойнтов.
        save_best_model: сохранять лучшую модель по test accuracy.
        config: гиперпараметры для записи в чекпойнты.
        field_n: степень поля для доменных метрик (берётся из config["n"] если не задано).
        checkpoint_keep_last: оставлять только N последних периодических чекпойнтов
            (None — хранить все). best_model.pt и last.pt не удаляются.
        resume_from: путь к last.pt / best_model.pt для продолжения обучения.
        after_epoch: вызывается после каждой эпохи (epoch, test_acc, train_acc);
            если возвращает True — обучение останавливается (Optuna pruning).

    Returns:
        TrainingResult с историей метрик.
    """
    device = next(model.parameters()).device
    use_cuda = device.type == "cuda"
    non_blocking = use_cuda
    writer = SummaryWriter(log_dir=str(tb_log_dir))
    n = field_n if field_n is not None else (config or {}).get("n")

    train_loss_hist: list[float] = []
    train_acc_hist: list[float] = []
    test_loss_hist: list[float] = []
    test_acc_hist: list[float] = []

    best_acc = 0.0
    best_epoch = 0
    best_model_path: str | None = None
    last_model_path: str | None = None
    start_epoch = 1

    if checkpoint_dir:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if resume_from:
        resume_ckpt = load_resume_state(resume_from, model, optimizer)
        start_epoch = int(resume_ckpt["epoch"]) + 1
        if checkpoint_dir:
            best_acc, best_epoch = _load_best_metrics(checkpoint_dir)
        print(f"Resume from epoch {resume_ckpt['epoch']} → continue at {start_epoch}")

    # Переопределение lr после восстановления optimizer state (напр. resume с меньшим lr).
    if override_lr is not None:
        for group in optimizer.param_groups:
            group["lr"] = override_lr
        print(f"Override lr → {override_lr}")

    if eval_interval < 1:
        raise ValueError(f"eval_interval must be >= 1, got {eval_interval}")

    last_epoch = start_epoch + num_epochs - 1
    start = time.time()

    for epoch in range(start_epoch, start_epoch + num_epochs):
        model.train()

        for inputs, labels in train_loader:
            inputs = inputs.to(device, non_blocking=non_blocking)
            labels = labels.to(device, non_blocking=non_blocking)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        should_eval = epoch % eval_interval == 0 or epoch == last_epoch
        should_log = epoch % log_interval == 0 or epoch == last_epoch

        if not should_eval:
            continue

        # Train-метрики считаем в eval-режиме (dropout OFF) по финальным весам эпохи,
        # чтобы они были напрямую сопоставимы с test. Иначе при dropout > 0 train_acc
        # систематически занижается и может оказаться НИЖЕ test_acc — это артефакт
        # измерения (dropout активен + усреднение по меняющимся весам), а не утечка данных.
        train_epoch_loss, train_epoch_acc = evaluate(model, train_loader, criterion)
        test_epoch_loss, test_epoch_acc = evaluate(model, test_loader, criterion)

        if should_log:
            writer.add_scalar("Loss/train", train_epoch_loss, epoch)
            writer.add_scalar("Loss/test", test_epoch_loss, epoch)
            writer.add_scalar("Acc/train", train_epoch_acc, epoch)
            writer.add_scalar("Acc/test", test_epoch_acc, epoch)

        metrics: dict[str, float] = {
            "train_loss": train_epoch_loss,
            "train_acc": train_epoch_acc,
            "test_loss": test_epoch_loss,
            "test_acc": test_epoch_acc,
        }

        if should_log and n is not None and config and config.get("log_domain_metrics"):
            from .domain_metrics import compute_domain_metrics, detect_input_format, flatten_for_logging
            from .gf_field import get_irreducible_poly

            poly = config.get("poly", get_irreducible_poly(n))
            train_size = config.get("train_size", 0)
            test_size = config.get("test_size", 0)
            seed = config.get("seed", 0)
            if train_size > 0 and test_size > 0:
                domain = compute_domain_metrics(
                    model,
                    n=n,
                    poly=poly,
                    train_size=train_size,
                    test_size=test_size,
                    seed=seed,
                    device=device,
                    input_format=detect_input_format(model, n),
                    eval_batch_size=config.get("domain_eval_batch_size", 4096),
                    assoc_samples=config.get("domain_assoc_samples", 2048),
                )
                metrics.update(flatten_for_logging(domain))

        if should_log and n is not None:
            conf = confusion_matrix(model, test_loader, n)
            metrics.update(algebraic_metrics(conf, n))
            metrics.update(error_distribution(conf, n))
            for name, value in metrics.items():
                if name not in ("train_loss", "test_loss"):
                    writer.add_scalar(f"Domain/{name}", value, epoch)

        if mlflow_log:
            from . import mlflow_logger

            mlflow_logger.log_metrics(metrics, step=epoch)

        train_loss_hist.append(train_epoch_loss)
        train_acc_hist.append(train_epoch_acc)
        test_loss_hist.append(test_epoch_loss)
        test_acc_hist.append(test_epoch_acc)

        if test_epoch_acc > best_acc:
            if save_best_model and checkpoint_dir:
                best_model_path = str(checkpoint_dir / "best_model.pt")
                torch.save(
                    _build_checkpoint_payload(
                        epoch, model, optimizer,
                        train_epoch_acc, test_epoch_acc, config, full=True,
                    ),
                    best_model_path,
                )
            best_acc = test_epoch_acc
            best_epoch = epoch

        if checkpoint_dir and should_log:
            # Лёгкий периодический снимок (только веса + метрики + config).
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch{epoch}.pt"
            torch.save(
                _build_checkpoint_payload(
                    epoch, model, optimizer,
                    train_epoch_acc, test_epoch_acc, config, full=False,
                ),
                checkpoint_path,
            )
            # Полный payload для resume — перезаписываемый last.pt.
            last_model_path = str(checkpoint_dir / "last.pt")
            torch.save(
                _build_checkpoint_payload(
                    epoch, model, optimizer,
                    train_epoch_acc, test_epoch_acc, config, full=True,
                ),
                last_model_path,
            )
            if checkpoint_keep_last is not None:
                _prune_checkpoints(checkpoint_dir, checkpoint_keep_last)

        if should_log:
            print(
                f"Epoch {epoch:6d} | "
                f"Train Loss: {train_epoch_loss:.4f} | "
                f"Train Acc: {train_epoch_acc:.4f} | "
                f"Test Loss: {test_epoch_loss:.4f} | "
                f"Test Acc: {test_epoch_acc:.4f} | "
                f"Best Test Acc: {best_acc:.4f} (epoch {best_epoch})"
            )

        if after_epoch is not None and after_epoch(epoch, test_epoch_acc, train_epoch_acc):
            break

    writer.close()
    elapsed = time.time() - start

    return TrainingResult(
        train_loss_history=train_loss_hist,
        train_acc_history=train_acc_hist,
        test_loss_history=test_loss_hist,
        test_acc_history=test_acc_hist,
        elapsed_seconds=elapsed,
        final_train_acc=train_acc_hist[-1] if train_acc_hist else 0.0,
        final_test_acc=test_acc_hist[-1] if test_acc_hist else 0.0,
        best_epoch=best_epoch,
        best_model_path=best_model_path,
        last_model_path=last_model_path,
    )
