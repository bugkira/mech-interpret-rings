"""Character projectors и таблицы классов для конечных групп по Cayley-таблице."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .alternating_group import build_alternating_mult_table
from .irrep_projectors import (
    A5_IRREP_SPECS,
    A5_NONTRIV_NAMES,
    A6_IRREP_SPECS,
    A6_NONTRIV_NAMES,
    a5_character_projectors,
    a6_character_projectors,
)
from .psl2_group import build_psl2_mult_table

IrrepSpec = dict[str, tuple[int, dict[str, float | complex]]]


def conjugacy_data(mult_table: np.ndarray) -> tuple[int, np.ndarray, list[list[int]], list[str]]:
    """Идентификатор, инверсии, классы сопряжения и их метки."""
    n = mult_table.shape[0]
    identity = next(i for i in range(n) if np.all(mult_table[i] == np.arange(n)))
    inverse = np.zeros(n, dtype=np.int64)
    for g in range(n):
        inverse[g] = int(np.where(mult_table[g] == identity)[0][0])

    unseen = set(range(n))
    classes: list[list[int]] = []
    while unseen:
        g = next(iter(unseen))
        cls: set[int] = set()
        stack = [g]
        while stack:
            x = stack.pop()
            if x in cls:
                continue
            cls.add(x)
            for h in range(n):
                y = int(mult_table[mult_table[h, x], inverse[h]])
                if y not in cls:
                    stack.append(y)
        classes.append(sorted(cls))
        unseen -= cls

    def element_order(g: int) -> int:
        if g == identity:
            return 1
        cur = int(mult_table[g, g])
        k = 2
        while cur != identity and k <= n:
            cur = int(mult_table[cur, g])
            k += 1
        return k

    sig_count: dict[tuple[int, int], int] = {}
    labels: list[str] = []
    class_meta: list[tuple[int, int, int]] = []
    for cls in classes:
        order = element_order(cls[0])
        size = len(cls)
        key = (order, size)
        idx = sig_count.get(key, 0)
        sig_count[key] = idx + 1
        class_meta.append((order, size, cls[0]))
        labels.append(f"{order}_{size}" if idx == 0 else f"{order}_{size}_{idx}")
    order_idx = sorted(range(len(classes)), key=lambda i: class_meta[i])
    classes = [classes[i] for i in order_idx]
    labels = [labels[i] for i in order_idx]
    return identity, inverse, classes, labels


def left_regular_matrices(mult_table: np.ndarray) -> list[np.ndarray]:
    n = mult_table.shape[0]
    mats: list[np.ndarray] = []
    for h in range(n):
        m = np.zeros((n, n), dtype=np.float64)
        for g in range(n):
            m[int(mult_table[h, g]), g] = 1.0
        mats.append(m)
    return mats


def build_character_projectors(
    mult_table: np.ndarray,
    irrep_specs: IrrepSpec,
) -> dict[str, np.ndarray]:
    """Центральные character-projectors P_χ из таблицы классов и характеров."""
    n = mult_table.shape[0]
    _, _, classes, class_labels = conjugacy_data(mult_table)
    elem_label: dict[int, str] = {}
    for i, cls in enumerate(classes):
        for g in cls:
            elem_label[g] = class_labels[i]
    left_regs = left_regular_matrices(mult_table)

    projectors: dict[str, np.ndarray] = {}
    for name, (dim, char_by_class) in irrep_specs.items():
        p = np.zeros((n, n), dtype=np.complex128)
        for g in range(n):
            p += complex(char_by_class[elem_label[g]]) * left_regs[g]
        p *= dim / n
        projectors[name] = p.real if np.max(np.abs(p.imag)) < 1e-10 else p
    return projectors


def _sigma() -> complex:
    re, im = -0.5, math.sqrt(7) / 2.0
    return complex(re, im)


PSL2_Q7_SPECS: IrrepSpec = {
    "triv": (1, {
        "1_1": 1, "2_21": 1, "3_56": 1, "4_42": 1, "7_24": 1, "7_24_1": 1,
    }),
    "3a": (3, {
        "1_1": 3, "2_21": -1, "3_56": 0, "4_42": 1,
        "7_24": _sigma(), "7_24_1": _sigma().conjugate(),
    }),
    "3b": (3, {
        "1_1": 3, "2_21": -1, "3_56": 0, "4_42": 1,
        "7_24": _sigma().conjugate(), "7_24_1": _sigma(),
    }),
    "6": (6, {
        "1_1": 6, "2_21": 2, "3_56": 0, "4_42": 0, "7_24": -1, "7_24_1": -1,
    }),
    "7": (7, {
        "1_1": 7, "2_21": -1, "3_56": 1, "4_42": -1, "7_24": 0, "7_24_1": 0,
    }),
    "8": (8, {
        "1_1": 8, "2_21": 0, "3_56": -1, "4_42": 0, "7_24": 1, "7_24_1": 1,
    }),
}

PSL2_Q7_NONTRIV = ("3a", "3b", "6", "7", "8")


def _e(n: int, k: int) -> complex:
    """Primitive n-th root of unity exp(2πik/n)."""
    return complex(math.cos(2 * math.pi * k / n), math.sin(2 * math.pi * k / n))


# PSL(2,8): 9 классов; таблица из GAP (CharacterTable(PSL(2,8))).
# Наши метки: 1_1, 2_63, 3_56, 7_72*, 9_56* (сортировка по (order, size)).
# GAP-колонки: 1a, 2a, 3a, 9a, 9b, 9c, 7a, 7b, 7c.
_E9 = _e(9, 1)
_E7 = _e(7, 1)
PSL2_Q8_SPECS: IrrepSpec = {
    "triv": (1, {
        "1_1": 1, "2_63": 1, "3_56": 1,
        "7_72": 1, "7_72_1": 1, "7_72_2": 1,
        "9_56": 1, "9_56_1": 1, "9_56_2": 1,
    }),
    "7_sym": (7, {
        "1_1": 7, "2_63": -1, "3_56": -2,
        "7_72": 0, "7_72_1": 0, "7_72_2": 0,
        "9_56": 1, "9_56_1": 1, "9_56_2": 1,
    }),
    "7a": (7, {
        "1_1": 7, "2_63": -1, "3_56": 1,
        "7_72": 0, "7_72_1": 0, "7_72_2": 0,
        "9_56": -_E9**4 - _E9**5,
        "9_56_1": _E9**2 + _E9**4 + _E9**5 + _E9**7,
        "9_56_2": -_E9**2 - _E9**7,
    }),
    "7b": (7, {
        "1_1": 7, "2_63": -1, "3_56": 1,
        "7_72": 0, "7_72_1": 0, "7_72_2": 0,
        "9_56": -_E9**2 - _E9**7,
        "9_56_1": -_E9**4 - _E9**5,
        "9_56_2": _E9**2 + _E9**4 + _E9**5 + _E9**7,
    }),
    "7c": (7, {
        "1_1": 7, "2_63": -1, "3_56": 1,
        "7_72": 0, "7_72_1": 0, "7_72_2": 0,
        "9_56": _E9**2 + _E9**4 + _E9**5 + _E9**7,
        "9_56_1": -_E9**2 - _E9**7,
        "9_56_2": -_E9**4 - _E9**5,
    }),
    "8": (8, {
        "1_1": 8, "2_63": 0, "3_56": -1,
        "7_72": 1, "7_72_1": 1, "7_72_2": 1,
        "9_56": -1, "9_56_1": -1, "9_56_2": -1,
    }),
    "9a": (9, {
        "1_1": 9, "2_63": 1, "3_56": 0,
        "7_72": _E7**3 + _E7**4,
        "7_72_1": _E7 + _E7**6,
        "7_72_2": _E7**2 + _E7**5,
        "9_56": 0, "9_56_1": 0, "9_56_2": 0,
    }),
    "9b": (9, {
        "1_1": 9, "2_63": 1, "3_56": 0,
        "7_72": _E7**2 + _E7**5,
        "7_72_1": _E7**3 + _E7**4,
        "7_72_2": _E7 + _E7**6,
        "9_56": 0, "9_56_1": 0, "9_56_2": 0,
    }),
    "9c": (9, {
        "1_1": 9, "2_63": 1, "3_56": 0,
        "7_72": _E7 + _E7**6,
        "7_72_1": _E7**2 + _E7**5,
        "7_72_2": _E7**3 + _E7**4,
        "9_56": 0, "9_56_1": 0, "9_56_2": 0,
    }),
}

PSL2_Q8_NONTRIV = ("7_sym", "7a", "7b", "7c", "8", "9a", "9b", "9c")

# PSL(2,11): 8 классов; таблица из GAP (CharacterTable(PSL(2,11))).
# Наши метки: 1_1, 2_55, 3_110, 5_132*, 6_110, 11_60*.
# GAP-колонки: 1a, 11a, 11b, 2a, 3a, 6a, 5a, 5b.
_E11 = _e(11, 1)
_E5 = _e(5, 1)
_A11 = _E11 + _E11**3 + _E11**4 + _E11**5 + _E11**9
_A11c = _E11**2 + _E11**6 + _E11**7 + _E11**8 + _E11**10
_B5 = _E5**2 + _E5**3
_B5c = _E5 + _E5**4
PSL2_Q11_SPECS: IrrepSpec = {
    "triv": (1, {
        "1_1": 1, "2_55": 1, "3_110": 1, "5_132": 1, "5_132_1": 1,
        "6_110": 1, "11_60": 1, "11_60_1": 1,
    }),
    "5a": (5, {
        "1_1": 5, "2_55": 1, "3_110": -1, "5_132": 0, "5_132_1": 0,
        "6_110": 1, "11_60": _A11, "11_60_1": _A11c,
    }),
    "5b": (5, {
        "1_1": 5, "2_55": 1, "3_110": -1, "5_132": 0, "5_132_1": 0,
        "6_110": 1, "11_60": _A11c, "11_60_1": _A11,
    }),
    "10a": (10, {
        "1_1": 10, "2_55": -2, "3_110": 1, "5_132": 0, "5_132_1": 0,
        "6_110": 1, "11_60": -1, "11_60_1": -1,
    }),
    "10b": (10, {
        "1_1": 10, "2_55": 2, "3_110": 1, "5_132": 0, "5_132_1": 0,
        "6_110": -1, "11_60": -1, "11_60_1": -1,
    }),
    "11": (11, {
        "1_1": 11, "2_55": -1, "3_110": -1, "5_132": 1, "5_132_1": 1,
        "6_110": -1, "11_60": 0, "11_60_1": 0,
    }),
    "12a": (12, {
        "1_1": 12, "2_55": 0, "3_110": 0, "5_132": _B5, "5_132_1": _B5c,
        "6_110": 0, "11_60": 1, "11_60_1": 1,
    }),
    "12b": (12, {
        "1_1": 12, "2_55": 0, "3_110": 0, "5_132": _B5c, "5_132_1": _B5,
        "6_110": 0, "11_60": 1, "11_60_1": 1,
    }),
}

PSL2_Q11_NONTRIV = ("5a", "5b", "10a", "10b", "11", "12a", "12b")


def _remap_a5_specs(mult_table: np.ndarray) -> tuple[IrrepSpec, tuple[str, ...]]:
    atlas = ["1A", "2A", "3A", "5A", "5B"]
    _, _, _, our_labels = conjugacy_data(mult_table)
    mapping = dict(zip(our_labels, atlas, strict=True))
    specs: IrrepSpec = {}
    for name, (dim, chars) in A5_IRREP_SPECS.items():
        specs[name] = (
            dim,
            {our_lab: chars[atlas_lab] for our_lab, atlas_lab in mapping.items()},
        )
    return specs, A5_NONTRIV_NAMES


def _remap_a6_specs(mult_table: np.ndarray) -> tuple[IrrepSpec, tuple[str, ...]]:
    atlas = ["1a", "2a", "3a", "3b", "4a", "5a", "5b"]
    _, _, _, our_labels = conjugacy_data(mult_table)
    mapping = dict(zip(our_labels, atlas, strict=True))
    specs: IrrepSpec = {}
    for name, (dim, chars) in A6_IRREP_SPECS.items():
        specs[name] = (
            dim,
            {our_lab: chars[atlas_lab] for our_lab, atlas_lab in mapping.items()},
        )
    return specs, A6_NONTRIV_NAMES


def psl2_character_projectors(
    q: int,
    mult_table: np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], tuple[str, ...]]:
    """Character projectors для PSL(2,q) на нашей Cayley-таблице."""
    if mult_table is None:
        mult_table, n_elements = build_psl2_mult_table(q)
    else:
        n_elements = mult_table.shape[0]

    if q == 5 and n_elements == 60:
        specs, names = _remap_a5_specs(mult_table)
    elif q == 9 and n_elements == 360:
        specs, names = _remap_a6_specs(mult_table)
    elif q == 7 and n_elements == 168:
        specs, names = PSL2_Q7_SPECS, PSL2_Q7_NONTRIV
    elif q == 8 and n_elements == 504:
        specs, names = PSL2_Q8_SPECS, PSL2_Q8_NONTRIV
    elif q == 11 and n_elements == 660:
        specs, names = PSL2_Q11_SPECS, PSL2_Q11_NONTRIV
    else:
        raise ValueError(f"character projectors not implemented for PSL(2,{q}) with |G|={n_elements}")

    return build_character_projectors(mult_table, specs), names


def projector_metadata(q: int, mult_table: np.ndarray | None = None) -> dict[str, Any]:
    """Диагностика: классы и метки для PSL(2,q)."""
    if mult_table is None:
        mult_table, _ = build_psl2_mult_table(q)
    _, _, classes, labels = conjugacy_data(mult_table)
    return {
        "q": q,
        "n_elements": mult_table.shape[0],
        "classes": [{"label": labels[i], "size": len(cls)} for i, cls in enumerate(classes)],
    }
