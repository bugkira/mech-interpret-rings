"""Subgroup families for PSL(2,q) on Cayley-table indices."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .alt_group_subgroups import SubgroupFamily, SubgroupSpec
from .group_characters import conjugacy_data
from .psl2_group import build_psl2_mult_table

_CACHE_DIR = Path(__file__).resolve().parents[2] / "new_paper_about_a5"


def _inverse(mult: np.ndarray) -> np.ndarray:
    n = mult.shape[0]
    identity = int(np.where(mult == np.arange(n))[0][0])
    inv = np.zeros(n, dtype=np.int64)
    for g in range(n):
        inv[g] = int(np.where(mult[g] == identity)[0][0])
    return inv


def _element_order(g: int, mult: np.ndarray, identity: int) -> int:
    if g == identity:
        return 1
    cur = int(mult[g, g])
    k = 2
    while cur != identity and k <= mult.shape[0]:
        cur = int(mult[cur, g])
        k += 1
    return k


def _closure(generators: list[int], mult: np.ndarray) -> frozenset[int]:
    n = mult.shape[0]
    identity = next(i for i in range(n) if np.all(mult[i] == np.arange(n)))
    elems = set(generators) | {identity}
    stack = list(generators)
    while stack:
        g = stack.pop()
        for h in list(elems):
            prod = int(mult[g, h])
            if prod not in elems:
                elems.add(prod)
                stack.append(prod)
    return frozenset(elems)


def _conjugate_subgroup(H: frozenset[int], x: int, mult: np.ndarray, inv: np.ndarray) -> frozenset[int]:
    out: set[int] = set()
    for h in H:
        out.add(int(mult[x, mult[h, inv[x]]]))
    return frozenset(out)


def _conjugacy_classes(H: frozenset[int], mult: np.ndarray, inv: np.ndarray) -> list[frozenset[int]]:
    n = mult.shape[0]
    seen: list[frozenset[int]] = []
    for x in range(n):
        Hx = _conjugate_subgroup(H, x, mult, inv)
        if Hx not in seen:
            seen.append(Hx)
    return seen


def find_subgroup_family(mult: np.ndarray, order: int, family: str, *, max_gens: int = 3) -> SubgroupFamily | None:
    n = mult.shape[0]
    identity = next(i for i in range(n) if np.all(mult[i] == np.arange(n)))
    inv = _inverse(mult)
    seen_sizes: set[frozenset[int]] = set()
    specs: list[SubgroupSpec] = []

    def try_add(H: frozenset[int]) -> None:
        if len(H) != order:
            return
        for cls in _conjugacy_classes(H, mult, inv):
            if cls in seen_sizes:
                continue
            seen_sizes.add(cls)
            specs.append(
                SubgroupSpec(
                    name=f"{family}_{len(specs)}",
                    family=family,
                    order=order,
                    index=n // order,
                    element_indices=cls,
                )
            )

    # cyclic
    for g in range(n):
        if g == identity:
            continue
        if _element_order(g, mult, identity) == order:
            try_add(_closure([g], mult))
        if specs:
            break

    # small generating sets
    if not specs:
        candidates = [g for g in range(n) if g != identity and _element_order(g, mult, identity) <= order]
        from itertools import combinations

        for r in range(2, max_gens + 1):
            for gens in combinations(candidates[:40], r):
                try_add(_closure(list(gens), mult))
            if specs:
                break

    if not specs:
        return None
    return SubgroupFamily(name=family, order=order, subgroups=tuple(specs))


# Representative orders per field (coset families). max_gens>1 only where needed.
PSL2_SUBGROUP_ORDERS: dict[int, dict[str, tuple[int, int]]] = {
    7: {"C8": (8, 1), "S4": (24, 3), "C21": (21, 1)},
    8: {"C9": (9, 1), "C56": (56, 2), "C7": (7, 1)},
    11: {"C11": (11, 1), "C55": (55, 2), "C60": (60, 2)},
}


def _supported_fields() -> tuple[int, ...]:
    return tuple(sorted(PSL2_SUBGROUP_ORDERS))


def psl2_subgroup_families(q: int, *, cache: bool = True) -> tuple[SubgroupFamily, ...]:
    if q not in PSL2_SUBGROUP_ORDERS:
        raise ValueError(
            f"subgroup export implemented for q in {_supported_fields()}, got {q}"
        )
    cache_path = _CACHE_DIR / f"psl2_q{q}_subgroups.json"
    if cache and cache_path.exists():
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        return tuple(
            SubgroupFamily(
                name=f["name"],
                order=f["order"],
                subgroups=tuple(
                    SubgroupSpec(
                        name=s["name"],
                        family=s["family"],
                        order=s["order"],
                        index=s["index"],
                        element_indices=frozenset(s["element_indices"]),
                    )
                    for s in f["subgroups"]
                ),
            )
            for f in data["families"]
        )

    mult, _ = build_psl2_mult_table(q)
    families: list[SubgroupFamily] = []
    for fam, (order, max_gens) in PSL2_SUBGROUP_ORDERS[q].items():
        sf = find_subgroup_family(mult, order, fam, max_gens=max_gens)
        if sf is not None:
            families.append(sf)
    if not families:
        raise RuntimeError(f"no subgroup families found for PSL(2,{q})")
    if cache:
        _write_cache(cache_path, tuple(families))
    return tuple(families)


def all_psl2_subgroup_specs(q: int) -> tuple[SubgroupSpec, ...]:
    out: list[SubgroupSpec] = []
    for fam in psl2_subgroup_families(q):
        out.extend(fam.subgroups)
    return tuple(out)


def _write_cache(path: Path, families: tuple[SubgroupFamily, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "families": [
            {
                "name": f.name,
                "order": f.order,
                "subgroups": [
                    {
                        "name": s.name,
                        "family": s.family,
                        "order": s.order,
                        "index": s.index,
                        "element_indices": sorted(s.element_indices),
                    }
                    for s in f.subgroups
                ],
            }
            for f in families
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
