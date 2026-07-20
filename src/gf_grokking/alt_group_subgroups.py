"""Сопряжённые подгруппы A_n для coset-анализа."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sympy.combinatorics import AlternatingGroup, Permutation
from sympy.combinatorics.perm_groups import PermutationGroup


@dataclass(frozen=True)
class SubgroupSpec:
    """Одна подгруппа H ≤ G (индексы элементов в таблице A_n)."""

    name: str
    family: str
    order: int
    index: int
    element_indices: frozenset[int]


@dataclass(frozen=True)
class SubgroupFamily:
    name: str
    order: int
    subgroups: tuple[SubgroupSpec, ...]


def _group_and_index_map(group_n: int) -> tuple[PermutationGroup, dict]:
    group = AlternatingGroup(group_n)
    elements = list(group.elements)
    perm_to_idx = {el: i for i, el in enumerate(elements)}
    return group, perm_to_idx


def _indices_from_subgroup(
    H: PermutationGroup,
    perm_to_idx: dict,
    *,
    family: str,
    name: str,
) -> SubgroupSpec:
    idxs = frozenset(perm_to_idx[h] for h in H.elements)
    G_order = len(perm_to_idx)
    return SubgroupSpec(
        name=name,
        family=family,
        order=H.order(),
        index=G_order // H.order(),
        element_indices=idxs,
    )


def _conjugate(H: PermutationGroup, g: Permutation) -> PermutationGroup:
    gens = [g * h * g**-1 for h in H.generators]
    return PermutationGroup(gens)


def _unique_conjugates(
    H: PermutationGroup,
    G: PermutationGroup,
    perm_to_idx: dict,
    *,
    family: str,
    name_prefix: str,
) -> tuple[SubgroupSpec, ...]:
    seen: list[frozenset[int]] = []
    specs: list[SubgroupSpec] = []
    for g in G.elements:
        Hg = _conjugate(H, g)
        idxs = frozenset(perm_to_idx[h] for h in Hg.elements)
        if idxs in seen:
            continue
        seen.append(idxs)
        specs.append(
            _indices_from_subgroup(
                Hg,
                perm_to_idx,
                family=family,
                name=f"{name_prefix}_{len(specs)}",
            )
        )
    return tuple(specs)


def a5_subgroup_families() -> tuple[SubgroupFamily, ...]:
    """Канонические семейства сопряжённых подгрупп A_5.

    A_4 (×5, индекс 5), D_5 (×6, индекс 6), D_3 (×10, индекс 10).
    """
    G, perm_to_idx = _group_and_index_map(5)

    a4 = G.stabilizer(4)
    c5 = Permutation(0, 1, 2, 3, 4)
    r = Permutation(1, 4)(2, 3)
    d5 = PermutationGroup([c5, r])
    c3 = Permutation(0, 1, 2)
    s = Permutation(0, 1)(3, 4)
    d3 = PermutationGroup([c3, s])

    families = (
        SubgroupFamily(
            "A4",
            12,
            _unique_conjugates(a4, G, perm_to_idx, family="A4", name_prefix="A4"),
        ),
        SubgroupFamily(
            "D5",
            10,
            _unique_conjugates(d5, G, perm_to_idx, family="D5", name_prefix="D5"),
        ),
        SubgroupFamily(
            "D3",
            6,
            _unique_conjugates(d3, G, perm_to_idx, family="D3", name_prefix="D3"),
        ),
    )
    return families


def a6_subgroup_families() -> tuple[SubgroupFamily, ...]:
    """Канонические семейства подгрупп A_6 для coset-анализа.

    A5 — точечные стабилизаторы (×6, индекс 6).
    A4 — стабилизаторы двух точек, действующие как A4 на оставшихся четырёх (×15).
    S4 — импримитивные стабилизаторы разбиений 6 точек на 3 пары (×15).
    D5 и D3 — нормализаторы циклических 5- и 3-подгрупп (×36 и ×60).
    """
    G, perm_to_idx = _group_and_index_map(6)
    p = Permutation

    a5 = G.stabilizer(5)
    a4 = PermutationGroup(
        [
            p([1, 2, 0, 3, 4, 5]),  # (0 1 2), fixes 3,4,5
            p([1, 0, 3, 2, 4, 5]),  # (0 1)(2 3)
        ]
    )
    s4 = PermutationGroup(
        [
            p([1, 0, 3, 2, 4, 5]),  # flip two pairs
            p([2, 3, 0, 1, 4, 5]),  # swap pair blocks
            p([2, 3, 4, 5, 0, 1]),  # cycle the three pair blocks
        ]
    )
    d5 = PermutationGroup(
        [
            p([1, 2, 3, 4, 0, 5]),  # (0 1 2 3 4)
            p([0, 4, 3, 2, 1, 5]),  # (1 4)(2 3)
        ]
    )
    d3 = PermutationGroup(
        [
            p([1, 2, 0, 3, 4, 5]),  # (0 1 2)
            p([1, 0, 2, 4, 3, 5]),  # (0 1)(3 4)
        ]
    )

    families = (
        SubgroupFamily(
            "A5",
            60,
            _unique_conjugates(a5, G, perm_to_idx, family="A5", name_prefix="A5"),
        ),
        SubgroupFamily(
            "A4",
            12,
            _unique_conjugates(a4, G, perm_to_idx, family="A4", name_prefix="A4"),
        ),
        SubgroupFamily(
            "S4",
            24,
            _unique_conjugates(s4, G, perm_to_idx, family="S4", name_prefix="S4"),
        ),
        SubgroupFamily(
            "D5",
            10,
            _unique_conjugates(d5, G, perm_to_idx, family="D5", name_prefix="D5"),
        ),
        SubgroupFamily(
            "D3",
            6,
            _unique_conjugates(d3, G, perm_to_idx, family="D3", name_prefix="D3"),
        ),
    )
    return families


def all_subgroup_specs(group_n: int) -> tuple[SubgroupSpec, ...]:
    if group_n == 5:
        families = a5_subgroup_families()
    elif group_n == 6:
        families = a6_subgroup_families()
    else:
        raise NotImplementedError(f"subgroup catalog for A_{group_n} not implemented yet")
    specs: list[SubgroupSpec] = []
    for fam in families:
        specs.extend(fam.subgroups)
    return tuple(specs)


def left_coset_labels(
    subgroup: SubgroupSpec,
    n_elements: int,
    *,
    mult_table: np.ndarray,
) -> np.ndarray:
    """Метка левого смежного класса gH для каждого g ∈ G (по индексу элемента)."""
    H = sorted(subgroup.element_indices)
    in_H = np.zeros(n_elements, dtype=bool)
    in_H[list(H)] = True
    labels = np.full(n_elements, -1, dtype=np.int64)
    next_id = 0
    for g in range(n_elements):
        if labels[g] >= 0:
            continue
        # левый косет: {g*h : h in H}
        coset = [mult_table[g, h] for h in H]
        for x in coset:
            labels[x] = next_id
        next_id += 1
    return labels


def right_coset_labels(
    subgroup: SubgroupSpec,
    n_elements: int,
    *,
    mult_table: np.ndarray,
) -> np.ndarray:
    """Метка правого смежного класса Hg."""
    H = sorted(subgroup.element_indices)
    labels = np.full(n_elements, -1, dtype=np.int64)
    next_id = 0
    for g in range(n_elements):
        if labels[g] >= 0:
            continue
        coset = [mult_table[h, g] for h in H]
        for x in coset:
            labels[x] = next_id
        next_id += 1
    return labels
