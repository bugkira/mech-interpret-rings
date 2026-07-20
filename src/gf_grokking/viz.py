"""Визуализация результатов обучения: learning curves, confusion matrices."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from .train import TrainingResult


def plot_learning_curves(
    result: TrainingResult,
    save_path: str | Path | None = None,
    show: bool = False,
) -> None:
    """Плотит кривые обучения (train/test loss + accuracy).

    Args:
        result: TrainingResult с историями метрик.
        save_path: Путь для сохранения (например, "plots/curves.png").
        show: Показывать график сразу.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(result.train_loss_history) + 1)

    # ---- Loss curves ----
    ax1.plot(epochs, result.train_loss_history, "b-", label="Train Loss", linewidth=2)
    ax1.plot(epochs, result.test_loss_history, "r-", label="Test Loss", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Learning Curves: Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ---- Accuracy curves ----
    ax2.plot(epochs, result.train_acc_history, "b-", label="Train Acc", linewidth=2)
    ax2.plot(epochs, result.test_acc_history, "r-", label="Test Acc", linewidth=2)
    ax2.axhline(
        y=max(result.test_acc_history),
        color="g",
        linestyle="--",
        label=f"Best Test: {max(result.test_acc_history):.4f}",
        linewidth=1.5,
    )
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Learning Curves: Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Learning curves saved to: {save_path}")

    if show:
        plt.show()

    plt.close()


def plot_confusion_matrix(
    confusion: torch.Tensor,
    n: int,
    save_path: str | Path | None = None,
    show: bool = False,
) -> None:
    """Плотит confusion matrix для GF(2^n).

    Args:
        confusion: [2^n, 2^n] confusion matrix.
        n: Степень поля.
        save_path: Путь для сохранения.
        show: Показывать график.
    """
    num_elements = 1 << n
    confusion_np = confusion.cpu().numpy()

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(confusion_np, cmap="Blues", interpolation="nearest")

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix: GF(2^{n})")

    # Colorbar
    plt.colorbar(im, ax=ax, label="Count")

    # Аннотации для диагонали
    for i in range(num_elements):
        ax.text(i, i, str(int(confusion_np[i, i])), ha="center", va="center", color="red", fontsize=8)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Confusion matrix saved to: {save_path}")

    if show:
        plt.show()

    plt.close()


def plot_error_analysis(
    confusion: torch.Tensor,
    n: int,
    save_path: str | Path | None = None,
    show: bool = False,
) -> None:
    """Анализ ошибок: распределение по классам.

    Args:
        confusion: [2^n, 2^n] confusion matrix.
        n: Степень поля.
        save_path: Путь для сохранения.
        show: Показывать график.
    """
    num_elements = 1 << n
    confusion_np = confusion.cpu().numpy()

    # Error rate per class
    row_sums = confusion_np.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # avoid division by zero
    error_rates = 1 - confusion_np.diagonal() / row_sums.flatten()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(num_elements), error_rates, color="red", alpha=0.7)
    ax.set_xlabel("Class (element of GF(2^n))")
    ax.set_ylabel("Error Rate")
    ax.set_title(f"Error Rate per Class: GF(2^{n})")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Error analysis saved to: {save_path}")

    if show:
        plt.show()

    plt.close()
