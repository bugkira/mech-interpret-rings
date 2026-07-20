"""Знакопеременные группы A_n: таблица умножения и разбиение пар."""

from __future__ import annotations

import numpy as np
import torch
from sympy.combinatorics import AlternatingGroup


def build_alternating_mult_table(n: int) -> tuple[np.ndarray, int]:
    """Строит таблицу умножения для A_n.

    Args:
        n: 5 → A_5 (60 элементов), 6 → A_6 (360 элементов).

    Returns:
        (mult_table, n_elements) — mult_table[i, j] = индекс произведения.
    """
    group = AlternatingGroup(n)
    elements = list(group.elements)
    n_elements = len(elements)
    perm_to_idx = {el: i for i, el in enumerate(elements)}

    mult_table = np.zeros((n_elements, n_elements), dtype=np.int64)
    for i, el1 in enumerate(elements):
        for j, el2 in enumerate(elements):
            mult_table[i, j] = perm_to_idx[el1 * el2]

    return mult_table, n_elements


def generate_alternating_group_data(
    n: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Генерирует полный OHE-датасет умножения в A_n.

    n=5 → A_5 (60 элементов), n=6 → A_6 (360 элементов).
    """
    mult_table, n_elements = build_alternating_mult_table(n)

    x_rows: list[np.ndarray] = []
    y_rows: list[int] = []
    eye = np.eye(n_elements, dtype=np.float32)
    for i in range(n_elements):
        for j in range(n_elements):
            x_rows.append(np.concatenate([eye[i], eye[j]]))
            y_rows.append(int(mult_table[i, j]))

    return np.stack(x_rows), np.array(y_rows, dtype=np.int64), n_elements


def alternating_group_order(n: int) -> int:
    """Порядок A_n без построения полной таблицы."""
    return len(AlternatingGroup(n).elements)


def split_pair_indices(
    num_elements: int,
    train_size: int,
    test_size: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Непересекающееся разбиение пар (a, b) на train и test."""
    total_pairs = num_elements * num_elements
    if train_size + test_size > total_pairs:
        raise ValueError(
            f"train_size + test_size ({train_size + test_size}) "
            f"превышает число пар ({total_pairs})"
        )

    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(total_pairs, generator=generator)

    flat_train = perm[:train_size]
    flat_test = perm[train_size : train_size + test_size]

    a_train = (flat_train // num_elements).long()
    b_train = (flat_train % num_elements).long()
    if test_size == 0:
        empty = torch.empty(0, dtype=torch.long)
        return a_train, b_train, empty, empty

    a_test = (flat_test // num_elements).long()
    b_test = (flat_test % num_elements).long()
    return a_train, b_train, a_test, b_test
