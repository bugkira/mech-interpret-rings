"""Конечные неассоциативные магмы / алгебры для grokking."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .finite_rings import (
    _build_table,
    _mat2_decode,
    _mat2_encode,
    _mat2_mul,
    _tri3_f2_coords,
    _tri3_f2_encode,
    _tri3_f2_mul,
)

MAGMA_REGISTRY: frozenset[str] = frozenset({"jordan_tri2_f3", "jordan_tri3_f2"})


@dataclass(frozen=True)
class MagmaSpec:
    id: str
    name: str
    n_elements: int
    associative: bool = False
    has_identity: bool = False
    magma_type: str = ""


def _jordan_mul_mat2(i: int, j: int, p: int, upper_triangular: bool) -> int:
    """Jordan product A∘B = (AB + BA)/2 mod p."""
    ab = _mat2_mul(i, j, p, upper_triangular)
    ba = _mat2_mul(j, i, p, upper_triangular)
    if p == 2:
        return ab ^ ba  # (ab+ba)/2 = ab+ba in char 2
    inv2 = pow(2, -1, p)
    # decode ab, ba as matrices and add coefficient-wise - easier: rebuild via mul on basis
    # For small p, brute decode-add-encode:
    na = nb = nc = nd = 0
    for idx, coeff in ((ab, 1), (ba, 1)):
        a, b, c, d = _mat2_decode(idx, p)
        if upper_triangular:
            c = 0
        na = (na + coeff * a) % p
        nb = (nb + coeff * b) % p
        nc = (nc + coeff * c) % p
        nd = (nd + coeff * d) % p
    if upper_triangular:
        nc = 0
        return _mat2_encode((na * inv2) % p, (nb * inv2) % p, 0, (nd * inv2) % p, p)
    return _mat2_encode((na * inv2) % p, (nb * inv2) % p, (nc * inv2) % p, (nd * inv2) % p, p)


def _jordan_mul_tri3_f2(i: int, j: int) -> int:
    ab = _tri3_f2_mul(i, j)
    ba = _tri3_f2_mul(j, i)
    p = 2
    acc = [0] * 6
    for idx in (ab, ba):
        coords = _tri3_f2_coords(idx)
        for k in range(6):
            acc[k] ^= coords[k]
    return _tri3_f2_encode(*acc)


def assert_non_associative(table: np.ndarray) -> tuple[int, int, int]:
    """Вернуть (a,b,c) с (ab)c != a(bc) или raise."""
    n = table.shape[0]
    for a in range(n):
        for b in range(n):
            ab = table[a, b]
            for c in range(n):
                if table[ab, c] != table[a, table[b, c]]:
                    return a, b, c
    raise ValueError("magma is associative — not suitable for this experiment")


@lru_cache(maxsize=4)
def _jordan_tri2_f3_table() -> np.ndarray:
    p = 3

    def mul(i: int, j: int) -> int:
        return _jordan_mul_mat2(i, j, p, upper_triangular=True)

    table = _build_table(p**3, mul)
    assert_non_associative(table)
    return table


@lru_cache(maxsize=4)
def _jordan_tri3_f2_table() -> np.ndarray:
    def mul(i: int, j: int) -> int:
        return _jordan_mul_tri3_f2(i, j)

    table = _build_table(64, mul)
    assert_non_associative(table)
    return table


def get_magma_spec(magma_id: str) -> MagmaSpec:
    specs = {
        "jordan_tri2_f3": MagmaSpec(
            id="jordan_tri2_f3",
            name="Jordan T_2(F_3)",
            n_elements=27,
            associative=False,
            has_identity=False,
            magma_type="jordan_upper_tri",
        ),
        "jordan_tri3_f2": MagmaSpec(
            id="jordan_tri3_f2",
            name="Jordan T_3(F_2)",
            n_elements=64,
            associative=False,
            has_identity=True,
            magma_type="jordan_upper_tri",
        ),
    }
    if magma_id not in specs:
        raise ValueError(f"unknown magma_id {magma_id!r}; choose from {sorted(MAGMA_REGISTRY)}")
    return specs[magma_id]


def build_magma_table(magma_id: str) -> tuple[np.ndarray, MagmaSpec]:
    spec = get_magma_spec(magma_id)
    if magma_id == "jordan_tri2_f3":
        table = _jordan_tri2_f3_table()
    elif magma_id == "jordan_tri3_f2":
        table = _jordan_tri3_f2_table()
    else:
        raise ValueError(f"unknown magma_id {magma_id!r}")
    if table.shape[0] != spec.n_elements:
        raise RuntimeError(f"{magma_id}: size mismatch")
    return table, spec


def compare_to_associative_ring(magma_id: str) -> str:
    """Ring id same shape for baseline comparison."""
    return {"jordan_tri2_f3": "tri2_f3", "jordan_tri3_f2": "tri3_f2"}[magma_id]
