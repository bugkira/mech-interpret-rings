"""Character projectors для isotypic components регулярного представления A_n."""

from __future__ import annotations

import numpy as np
from sympy.combinatorics import AlternatingGroup

from .alternating_group import build_alternating_mult_table

PHI = (1 + 5**0.5) / 2
PSI = (1 - 5**0.5) / 2

# Вещественные irreps A5: trivial + 3 + 3' + 4 + 5
A5_IRREP_SPECS: dict[str, tuple[int, dict[str, float]]] = {
    "triv": (1, {"1A": 1, "2A": 1, "3A": 1, "5A": 1, "5B": 1}),
    "3": (3, {"1A": 3, "2A": -1, "3A": 0, "5A": PHI, "5B": PSI}),
    "3p": (3, {"1A": 3, "2A": -1, "3A": 0, "5A": PSI, "5B": PHI}),
    "4": (4, {"1A": 4, "2A": 0, "3A": 1, "5A": -1, "5B": -1}),
    "5": (5, {"1A": 5, "2A": 1, "3A": -1, "5A": 0, "5B": 0}),
}

A5_NONTRIV_NAMES = ("3", "3p", "4", "5")

# Вещественные irreps A6: trivial + 5 + 5 + 8 + 8 + 9 + 10
A6_IRREP_SPECS: dict[str, tuple[int, dict[str, float]]] = {
    "triv": (1, {"1a": 1, "2a": 1, "3a": 1, "3b": 1, "4a": 1, "5a": 1, "5b": 1}),
    "5a": (5, {"1a": 5, "2a": 1, "3a": 2, "3b": -1, "4a": -1, "5a": 0, "5b": 0}),
    "5b": (5, {"1a": 5, "2a": 1, "3a": -1, "3b": 2, "4a": -1, "5a": 0, "5b": 0}),
    "8a": (8, {"1a": 8, "2a": 0, "3a": -1, "3b": -1, "4a": 0, "5a": PSI, "5b": PHI}),
    "8b": (8, {"1a": 8, "2a": 0, "3a": -1, "3b": -1, "4a": 0, "5a": PHI, "5b": PSI}),
    "9": (9, {"1a": 9, "2a": 1, "3a": 0, "3b": 0, "4a": 1, "5a": -1, "5b": -1}),
    "10": (10, {"1a": 10, "2a": -2, "3a": 1, "3b": 1, "4a": 0, "5a": 0, "5b": 0}),
}

A6_NONTRIV_NAMES = ("5a", "5b", "8a", "8b", "9", "10")


def _a5_class_labels(n_elements: int) -> dict[int, str]:
    """Метки классов сопряжённости A5 по индексам элементов."""
    if n_elements != 60:
        raise ValueError("A5 class labels only implemented for |A5|=60")
    group = AlternatingGroup(5)
    elems = list(group.elements)
    idx = {g: i for i, g in enumerate(elems)}
    unseen = set(range(n_elements))
    classes: list[set[int]] = []
    while unseen:
        i = next(iter(unseen))
        x = elems[i]
        cls = {idx[g * x * g**-1] for g in elems}
        classes.append(cls)
        unseen -= cls

    info: list[tuple[set[int], int]] = []
    for c in classes:
        rep = elems[next(iter(c))]
        info.append((c, rep.order()))
    info.sort(key=lambda t: (t[1], len(t[0])))

    labels: dict[int, str] = {}
    five: list[set[int]] = []
    for c, order in info:
        if order == 1:
            lab = "1A"
        elif order == 2:
            lab = "2A"
        elif order == 3:
            lab = "3A"
        elif order == 5:
            five.append(c)
            continue
        else:
            raise AssertionError(order)
        for i in c:
            labels[i] = lab
    for n, c in enumerate(five):
        lab = "5A" if n == 0 else "5B"
        for i in c:
            labels[i] = lab
    return labels


def a5_character_projectors(
    mult_table: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Центральные character-проекторы P_χ для A5, shape [|G|, |G|]."""
    if mult_table is None:
        mult_table, n_elements = build_alternating_mult_table(5)
    else:
        n_elements = mult_table.shape[0]
    if n_elements != 60:
        raise ValueError("a5_character_projectors requires |A5|=60")

    labels = _a5_class_labels(n_elements)
    left_regs: list[np.ndarray] = []
    for h in range(n_elements):
        m = np.zeros((n_elements, n_elements), dtype=np.float64)
        for g in range(n_elements):
            m[mult_table[h, g], g] = 1.0
        left_regs.append(m)

    projectors: dict[str, np.ndarray] = {}
    for name, (dim, char_by_class) in A5_IRREP_SPECS.items():
        p = np.zeros((n_elements, n_elements), dtype=np.float64)
        for h in range(n_elements):
            p += char_by_class[labels[h]] * left_regs[h]
        p *= dim / n_elements
        projectors[name] = p
    return projectors


def _a6_class_labels(n_elements: int) -> dict[int, str]:
    """Метки классов сопряжённости A6 по индексам элементов."""
    if n_elements != 360:
        raise ValueError("A6 class labels only implemented for |A6|=360")
    group = AlternatingGroup(6)
    elems = list(group.elements)
    idx = {g: i for i, g in enumerate(elems)}
    unseen = set(range(n_elements))
    classes: list[set[int]] = []
    while unseen:
        i = next(iter(unseen))
        x = elems[i]
        cls = {idx[g * x * g**-1] for g in elems}
        classes.append(cls)
        unseen -= cls

    info: list[tuple[set[int], int]] = []
    for c in classes:
        rep = elems[next(iter(c))]
        info.append((c, rep.order()))
    info.sort(key=lambda t: (t[1], len(t[0]), min(t[0])))

    labels: dict[int, str] = {}
    pending: dict[int, list[set[int]]] = {}
    for c, order in info:
        pending.setdefault(order, []).append(c)

    class_names_by_order = {
        1: ("1a",),
        2: ("2a",),
        3: ("3a", "3b"),
        4: ("4a",),
        5: ("5a", "5b"),
    }
    for order, cls_list in pending.items():
        names = class_names_by_order[order]
        cls_list.sort(key=lambda s: min(s))
        for name, c in zip(names, cls_list):
            for i in c:
                labels[i] = name
    return labels


def a6_character_projectors(
    mult_table: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Центральные character-проекторы P_χ для A6, shape [|G|, |G|]."""
    if mult_table is None:
        mult_table, n_elements = build_alternating_mult_table(6)
    else:
        n_elements = mult_table.shape[0]
    if n_elements != 360:
        raise ValueError("a6_character_projectors requires |A6|=360")

    labels = _a6_class_labels(n_elements)
    left_regs: list[np.ndarray] = []
    for h in range(n_elements):
        m = np.zeros((n_elements, n_elements), dtype=np.float64)
        for g in range(n_elements):
            m[mult_table[h, g], g] = 1.0
        left_regs.append(m)

    projectors: dict[str, np.ndarray] = {}
    for name, (dim, char_by_class) in A6_IRREP_SPECS.items():
        p = np.zeros((n_elements, n_elements), dtype=np.float64)
        for h in range(n_elements):
            p += char_by_class[labels[h]] * left_regs[h]
        p *= dim / n_elements
        projectors[name] = p
    return projectors


def character_projectors(group_n: int, mult_table: np.ndarray | None = None) -> dict[str, np.ndarray]:
    if group_n == 5:
        return a5_character_projectors(mult_table)
    if group_n == 6:
        return a6_character_projectors(mult_table)
    raise ValueError(f"character projectors not implemented for group_n={group_n}")


def _projector_mat(p: np.ndarray) -> np.ndarray:
    if np.iscomplexobj(p):
        return p
    return p.astype(np.float64)


def _projector_conj_transpose(p: np.ndarray) -> np.ndarray:
    if np.iscomplexobj(p):
        return p.conj().T
    return p.T


def isotypic_energy_fractions(
    matrix: np.ndarray,
    projectors: dict[str, np.ndarray],
    *,
    names: tuple[str, ...] = A5_NONTRIV_NAMES,
) -> dict[str, float]:
    """Доля энергии ||P_χ X||_F^2 / ||X||_F^2 для каждого irrep."""
    x = matrix - matrix.mean(axis=0, keepdims=True)
    total = float(np.linalg.norm(x, "fro") ** 2)
    if total < 1e-15:
        return {n: 0.0 for n in names}
    out: dict[str, float] = {}
    for name in names:
        p = _projector_mat(projectors[name])
        px = p @ x
        if np.iscomplexobj(px):
            px = px.real
        out[name] = float(np.linalg.norm(px, "fro") ** 2 / total)
    return out


def isotypic_orthobasis(
    projectors: dict[str, np.ndarray],
    *,
    order: tuple[str, ...] = ("triv",) + A5_NONTRIV_NAMES,
) -> tuple[np.ndarray, list[tuple[str, int, int]]]:
    """Ортонормированный базис [|G|, |G|], сгруппированный по изотипическим блокам.

    Для каждого проектора P_χ (симметричный идемпотент) берутся его собственные
    векторы с собственным значением ≈1 (базис range P_χ). Столбцы Q упорядочены
    по order; в этом базисе Q^T C Q приближённо блочно-диагональна.

    Возвращает (Q, blocks), где blocks — список (name, start, end) индексов
    столбцов каждого блока.
    """
    cols: list[np.ndarray] = []
    blocks: list[tuple[str, int, int]] = []
    start = 0
    for name in order:
        p = projectors[name]
        vals, vecs = np.linalg.eigh(p)
        keep = vecs[:, vals > 0.5]
        # Повторная ортонормировка на всякий случай (numerically clean).
        q, _ = np.linalg.qr(keep)
        cols.append(q)
        end = start + q.shape[1]
        blocks.append((name, start, end))
        start = end
    return np.concatenate(cols, axis=1), blocks


def covariance_block_fraction(
    matrix: np.ndarray,
    projectors: dict[str, np.ndarray],
    *,
    names: tuple[str, ...] = A5_NONTRIV_NAMES,
) -> tuple[float, float]:
    """Доля Frobenius-нормы block vs off-block в C = X X^T по isotypic projectors."""
    x = matrix - matrix.mean(axis=0, keepdims=True)
    c = x @ x.T
    total = float(np.linalg.norm(c, "fro"))
    if total < 1e-15:
        return 0.0, 0.0
    block = np.zeros_like(
        c,
        dtype=np.complex128 if any(np.iscomplexobj(p) for p in projectors.values()) else np.float64,
    )
    for n in names:
        p = _projector_mat(projectors[n])
        block += p @ c @ _projector_conj_transpose(p)
    if np.iscomplexobj(block):
        block = block.real
    off = c - block
    return float(np.linalg.norm(block, "fro") / total), float(np.linalg.norm(off, "fro") / total)
