"""Evaluation utilities для GF(2^n) моделей.

Включает доменные метрики для алгебры полей Галуа:
- accuracy по подгруппам (степени элементов)
- анализ ошибок по структуре умножения
- zero divisor analysis
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def confusion_matrix(
    model: nn.Module,
    loader: DataLoader,
    n: int,
) -> torch.Tensor:
    """Вычисляет матрицу ошибок (confusion matrix) размером 2^n × 2^n.

    Строки — истина, столбцы — предсказание.

    Args:
        model: обученная модель.
        loader: DataLoader с (inputs, labels).
        n: степень поля (для размера матрицы).

    Returns:
        confusion: [2^n, 2^n] — counts matrix.
    """
    return confusion_matrix_num_classes(model, loader, 1 << n)


def confusion_matrix_num_classes(
    model: nn.Module,
    loader: DataLoader,
    num_classes: int,
) -> torch.Tensor:
    """Confusion matrix [num_classes, num_classes] для произвольного числа классов."""
    model.eval()
    num_elements = num_classes
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    cm = torch.zeros(num_elements, num_elements, dtype=torch.long, device=device)

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = model(inputs)
            preds = logits.argmax(dim=1)
            indices = num_elements * labels + preds
            cm += torch.bincount(
                indices, minlength=num_elements * num_elements
            ).reshape(num_elements, num_elements)

    return cm.cpu()


def per_class_accuracy(confusion: torch.Tensor) -> torch.Tensor:
    """Вычисляет accuracy по каждой истинной классовой метке.

    Args:
        confusion: [num_classes, num_classes] matrix.

    Returns:
        acc_per_class: [num_classes] — accuracy для каждого класса.
    """
    return confusion.diag() / confusion.sum(dim=1)


@torch.no_grad()
def analyze_failures(
    model: nn.Module,
    loader: DataLoader,
    n: int,
) -> torch.Tensor:
    """Находит примеры, где модель ошибается, декодируя входы из one-hot.

    Вход: [batch, 2 * 2^n] = concat(one_hot(a), one_hot(b)).

    Args:
        model: обученная модель.
        loader: DataLoader с (inputs, labels).
        n: степень поля.

    Returns:
        failed: [num_failures, 4] — (a_idx, b_idx, true_label, pred).
    """
    num_elements = 1 << n
    model.eval()
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    rows: list[torch.Tensor] = []

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        preds = model(inputs).argmax(dim=1)

        mask = preds != labels
        if not mask.any():
            continue

        a_idx = inputs[:, :num_elements].argmax(dim=1)
        b_idx = inputs[:, num_elements:].argmax(dim=1)
        rows.append(
            torch.stack(
                [a_idx[mask], b_idx[mask], labels[mask], preds[mask]], dim=1
            ).cpu()
        )

    if not rows:
        return torch.empty((0, 4), dtype=torch.long)
    return torch.cat(rows, dim=0).to(torch.long)


def algebraic_metrics(
    confusion: torch.Tensor,
    n: int,
) -> dict[str, float]:
    """Вычисляет доменные метрики для GF(2^n).

    Args:
        confusion: [2^n, 2^n] confusion matrix.
        n: степень поля.

    Returns:
        dict с метриками (всё выводится из confusion matrix, где строки — истинный
        label, столбцы — предсказание; пары (a, b) в ней не сохранены):
        - zero_accuracy: точность предсказания label=0 (результат-ноль)
        - identity_accuracy: точность предсказания label=1
        - overall_accuracy: общая accuracy (сумма диагонали / всего)
        - avg_nontrivial_accuracy: accuracy по label вне {0, 1}

    Note:
        Истинная точность на квадратах a*a требует пар (a, b) и здесь
        не вычислима — используйте analyze_failures для разбора по парам.
    """
    # Результат-ноль: строка label=0.
    zero_row = confusion[0, :]
    zero_accuracy = zero_row[0].item() / zero_row.sum().item() if zero_row.sum() > 0 else 0.0

    # Результат-единица: строка label=1.
    one_row = confusion[1, :]
    identity_accuracy = one_row[1].item() / one_row.sum().item() if one_row.sum() > 0 else 0.0

    diag = confusion.diag()
    overall_accuracy = diag.sum().item() / confusion.sum().item() if confusion.sum() > 0 else 0.0

    nontrivial = confusion[2:, 2:]
    avg_nontrivial = nontrivial.diag().sum().item() / nontrivial.sum().item() if nontrivial.sum() > 0 else 0.0

    return {
        "zero_accuracy": zero_accuracy,
        "identity_accuracy": identity_accuracy,
        "overall_accuracy": overall_accuracy,
        "avg_nontrivial_accuracy": avg_nontrivial,
    }


def error_distribution(
    confusion: torch.Tensor,
    n: int,
) -> dict[str, float]:
    """Анализирует распределение ошибок.

    Args:
        confusion: [2^n, 2^n] confusion matrix.
        n: степень поля.

    Returns:
        dict со статистикой ошибок:
        - total_errors: всего ошибок
        - off_by_one: ошибок, где pred = label ± 1
        - zero_errors: ошибок на умножении на 0
        - random_errors: ошибок, где pred случайный
    """
    num_elements = 1 << n
    total = confusion.sum().item()
    correct = confusion.diag().sum().item()
    total_errors = int(total - correct)
    
    # Ошибки на умножении на 0 (строка 0, кроме столбца 0)
    zero_errors = int(confusion[0, :].sum().item() - confusion[0, 0].item())
    
    # Off-by-one ошибки (предсказание = label ± 1)
    off_by_one = 0
    for i in range(num_elements):
        if i > 0:
            off_by_one += confusion[i, i-1].item()
        if i < num_elements - 1:
            off_by_one += confusion[i, i+1].item()
    off_by_one = int(off_by_one)
    
    return {
        "total_errors": total_errors,
        "zero_errors": zero_errors,
        "off_by_one_errors": off_by_one,
        "error_rate": total_errors / total if total > 0 else 0.0,
    }
