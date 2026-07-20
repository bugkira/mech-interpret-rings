"""Конечные некоммутативные кольца: таблицы умножения для grokking."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

import numpy as np

RING_REGISTRY: frozenset[str] = frozenset(
    {
        "mat2_f2",
        "mat2_f3",
        "quat_f3",
        "tri2_f2",
        "tri2_f3",
        "tri3_f2",
        "f3",
        "f3_x_f3",
        "mat2_f3_x_f3",
        "quat_f3_x_f3",
        "tri2_f3_x_f3",
        "tri2_f3_x_f3_x_f3",
        "mat2_f2_x_mat2_f2",
        "f2_x7",
        "ext_f2_k6",
        "ext_f2_k8",
        "ext_f3_k6",
        "ext_f3_k7",
        "lam_f3_k2",
        "lam_f2_k3t",
        "f2_s3",
        "f2_d4",
        "f3_s3",
        "f2_z6",
        "f2_z7",
        "f8",
        "f9",
        "f8_x_f8",
        "f3_x_f3_x_f9",
        "f3_z4",
    }
)


TRI2_ENCODING_VERSION = 2


@dataclass(frozen=True)
class RingSpec:
    id: str
    name: str
    n_elements: int
    commutative: bool = False
    prime: int = 0
    ring_type: str = ""
    encoding_version: int | None = None


def _mat2_encode(a: int, b: int, c: int, d: int, p: int) -> int:
    return a + p * b + (p * p) * c + (p**3) * d


def _mat2_decode(idx: int, p: int) -> tuple[int, int, int, int]:
    a = idx % p
    idx //= p
    b = idx % p
    idx //= p
    c = idx % p
    idx //= p
    d = idx % p
    return a, b, c, d


def _mat2_mul(
    i: int, j: int, p: int, upper_triangular: bool
) -> int:
    if upper_triangular:
        return _tri2_mul(i, j, p)
    a1, b1, c1, d1 = _mat2_decode(i, p)
    a2, b2, c2, d2 = _mat2_decode(j, p)
    # [[a1,b1],[c1,d1]] @ [[a2,b2],[c2,d2]]
    na = (a1 * a2 + b1 * c2) % p
    nb = (a1 * b2 + b1 * d2) % p
    nc = (c1 * a2 + d1 * c2) % p
    nd = (c1 * b2 + d1 * d2) % p
    return _mat2_encode(na, nb, nc, nd, p)


def _tri2_encode(a: int, b: int, d: int, p: int) -> int:
    """T_2(F_p): [[a,b],[0,d]], c ≡ 0; idx = a + p·b + p²·d."""
    return a + p * b + (p * p) * d


def _tri2_decode(idx: int, p: int) -> tuple[int, int, int]:
    a = idx % p
    idx //= p
    b = idx % p
    idx //= p
    d = idx % p
    return a, b, d


def _tri2_mul(i: int, j: int, p: int) -> int:
    a1, b1, d1 = _tri2_decode(i, p)
    a2, b2, d2 = _tri2_decode(j, p)
    na = (a1 * a2) % p
    nb = (a1 * b2 + b1 * d2) % p
    nd = (d1 * d2) % p
    return _tri2_encode(na, nb, nd, p)


def tri2_identity_index(p: int) -> int:
    """Единичная матрица I в T_2(F_p)."""
    return _tri2_encode(1, 0, 1, p)


def _quat_f3_encode(a: int, b: int, c: int, d: int) -> int:
    """H(F_3): q = a + bi + cj + dk, idx = a + 3b + 9c + 27d."""
    return a + 3 * b + 9 * c + 27 * d


def _quat_f3_decode(idx: int) -> tuple[int, int, int, int]:
    a = idx % 3
    idx //= 3
    b = idx % 3
    idx //= 3
    c = idx % 3
    idx //= 3
    d = idx % 3
    return a, b, c, d


def _quat_f3_mul(i: int, j: int) -> int:
    """Hamilton product in H(F_3): i²=j²=k²=ijk=-1."""
    a1, b1, c1, d1 = _quat_f3_decode(i)
    a2, b2, c2, d2 = _quat_f3_decode(j)
    p = 3
    na = (a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2) % p
    nb = (a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2) % p
    nc = (a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2) % p
    nd = (a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2) % p
    return _quat_f3_encode(na, nb, nc, nd)


@lru_cache(maxsize=4)
def _quat_f3_table() -> np.ndarray:
    n = 81

    def mul(i: int, j: int) -> int:
        return _quat_f3_mul(i, j)

    table = _build_table(n, mul)
    assert_associativity(table)
    return table


def quat_f3_identity_index() -> int:
    """Multiplicative identity 1 in H(F_3)."""
    return _quat_f3_encode(1, 0, 0, 0)


def _ext_parse_ring_id(ring_id: str) -> tuple[int, int]:
    """Return (prime p, k) for ext_f{p}_k{k}."""
    import re

    m = re.match(r"ext_f(\d+)_k(\d+)$", ring_id)
    if not m:
        raise ValueError(ring_id)
    return int(m.group(1)), int(m.group(2))


def _ext_n_elements(k: int, p: int) -> int:
    """|R| = zero + (±1)·{all monomials}; in char 2 signs collapse → 2^k+1."""
    if p == 2:
        return (1 << k) + 1
    return (1 << (k + 1))  # 2 * 2^k


def _ext_merge_sign(mi: int, mj: int) -> int:
    """Sign of e_mi ∧ e_mj when masks are disjoint (+1 or -1)."""
    sign = 1
    m = mj
    while m:
        low = m & -m
        j = low.bit_length() - 1
        below = mi & ((1 << j) - 1)
        if below.bit_count() % 2:
            sign = -sign
        m ^= low
    return sign


def _ext_decode(idx: int, p: int) -> tuple[int | None, int]:
    """(mask, sign) with sign ∈ {+1,-1}; idx=0 → (None,0) zero."""
    if idx == 0:
        return None, 0
    if p == 2:
        if idx == 1:
            return 0, 1
        return idx - 1, 1
    if idx == 1:
        return 0, 1
    mask = idx // 2
    sign = 1 if idx % 2 == 0 else -1
    return mask, sign


def _ext_encode(mask: int, sign: int, p: int) -> int:
    if sign == 0:
        return 0
    if mask == 0:
        return 1 if sign > 0 else 0
    if p == 2:
        return mask + 1
    return 2 * mask if sign > 0 else 2 * mask + 1


def _ext_mul(i: int, j: int, p: int) -> int:
    """Exterior/wedge on basis monomials; anticommutative (trivial in char 2)."""
    if i == 0 or j == 0:
        return 0
    mi, si = _ext_decode(i, p)
    mj, sj = _ext_decode(j, p)
    if mi is None:
        return _ext_encode(mj, si * sj, p)
    if mj is None:
        return _ext_encode(mi, si * sj, p)
    if mi & mj:
        return 0
    out_sign = si * sj * _ext_merge_sign(mi, mj)
    if p == 2:
        out_sign = 1
    return _ext_encode(mi | mj, out_sign, p)


@lru_cache(maxsize=8)
def _ext_table(k: int, p: int) -> np.ndarray:
    n = _ext_n_elements(k, p)
    out = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        for j in range(n):
            out[i, j] = _ext_mul(i, j, p)
    assert_associativity(out)
    return out


def ext_identity_index() -> int:
    """Multiplicative identity 1 ∈ Λ."""
    return 1


def _ext_f2_k(ring_id: str) -> int:
    return _ext_parse_ring_id(ring_id)[1]


def _tri3_f2_coords(idx: int) -> tuple[int, int, int, int, int, int]:
    """Upper 3x3 over F2: (a,b,c,d,e,f) for [[a,b,c],[0,d,e],[0,0,f]]."""
    a = idx & 1
    b = (idx >> 1) & 1
    c = (idx >> 2) & 1
    d = (idx >> 3) & 1
    e = (idx >> 4) & 1
    f = (idx >> 5) & 1
    return a, b, c, d, e, f


def _tri3_f2_encode(a: int, b: int, c: int, d: int, e: int, f: int) -> int:
    return a | (b << 1) | (c << 2) | (d << 3) | (e << 4) | (f << 5)


def _tri3_f2_mul(i: int, j: int) -> int:
    a1, b1, c1, d1, e1, f1 = _tri3_f2_coords(i)
    a2, b2, c2, d2, e2, f2 = _tri3_f2_coords(j)
    p = 2
    # M1 @ M2 mod 2
    na = (a1 * a2) % p
    nb = (a1 * b2 + b1 * d2) % p
    nc = (a1 * c2 + b1 * e2 + c1 * f2) % p
    nd = (d1 * d2) % p
    ne = (d1 * e2 + e1 * f2) % p
    nf = (f1 * f2) % p
    return _tri3_f2_encode(na, nb, nc, nd, ne, nf)


def _build_table(n: int, mul_fn: Callable[[int, int], int]) -> np.ndarray:
    table = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        for j in range(n):
            table[i, j] = mul_fn(i, j)
    return table


def assert_associativity(table: np.ndarray) -> None:
    n = table.shape[0]
    for i in range(n):
        for j in range(n):
            ij = table[i, j]
            for k in range(n):
                left = table[ij, k]
                right = table[i, table[j, k]]
                if left != right:
                    raise ValueError(
                        f"associativity failed: ({i}*{j})*{k}={left} vs {i}*({j}*{k})={right}"
                    )


@lru_cache(maxsize=4)
def _field_fp_table(p: int) -> np.ndarray:
    n = p

    def mul(i: int, j: int) -> int:
        return (i * j) % p

    table = _build_table(n, mul)
    assert_associativity(table)
    return table


@lru_cache(maxsize=4)
def _tri2_fp_table(p: int) -> np.ndarray:
    n = p**3

    def mul(i: int, j: int) -> int:
        return _tri2_mul(i, j, p)

    table = _build_table(n, mul)
    assert_associativity(table)
    return table


@lru_cache(maxsize=8)
def _mat2_fp_table(p: int, upper_triangular: bool) -> np.ndarray:
    if upper_triangular:
        return _tri2_fp_table(p)
    n = p**4

    def mul(i: int, j: int) -> int:
        return _mat2_mul(i, j, p, upper_triangular=False)

    table = _build_table(n, mul)
    assert_associativity(table)
    return table


def _f2_poly_coords(idx: int, degree: int = 7) -> tuple[int, ...]:
    return tuple((idx >> k) & 1 for k in range(degree))


def _f2_poly_encode(coeffs: tuple[int, ...]) -> int:
    return sum((c & 1) << i for i, c in enumerate(coeffs))


def _f2_poly_mul_mod(i: int, j: int, degree: int = 7) -> int:
    """F_2[x]/(x^degree): polynomial multiply."""
    a = _f2_poly_coords(i, degree)
    b = _f2_poly_coords(j, degree)
    prod = [0] * (2 * degree - 1)
    for ia, ca in enumerate(a):
        if not ca:
            continue
        for ib, cb in enumerate(b):
            if cb:
                prod[ia + ib] ^= 1
    out = [0] * degree
    for k, v in enumerate(prod):
        if v:
            if k < degree:
                out[k] ^= 1
    return _f2_poly_encode(tuple(out))


@lru_cache(maxsize=4)
def _f8_table() -> np.ndarray:
    """GF(8) ≅ F_2[x]/(x^3+x+1); used in F_2[x]/(Φ_7) ≅ F_8×F_8 replication."""
    import torch

    from .gf_field import gf_mult_vectorized

    return gf_mult_vectorized(3, 11).numpy().astype(np.int64)


def _fp_poly_coords(idx: int, degree: int, p: int) -> tuple[int, ...]:
    return tuple((idx // (p**k)) % p for k in range(degree))


def _fp_poly_encode(coeffs: tuple[int, ...], p: int) -> int:
    return sum((c % p) * (p**k) for k, c in enumerate(coeffs))


def _fp_poly_mul_mod(i: int, j: int, *, p: int, degree: int, irr_low: tuple[int, ...]) -> int:
    """Multiply in F_p[x]/(x^degree + sum irr_low[t] x^t). irr_low: constant..x^{degree-1} coeffs."""
    a = _fp_poly_coords(i, degree, p)
    b = _fp_poly_coords(j, degree, p)
    prod = [0] * (2 * degree - 1)
    for ia, ca in enumerate(a):
        if ca == 0:
            continue
        for ib, cb in enumerate(b):
            if cb == 0:
                continue
            prod[ia + ib] = (prod[ia + ib] + ca * cb) % p
    out = [0] * degree
    for k in range(2 * degree - 2, degree - 1, -1):
        v = prod[k]
        if v == 0:
            continue
        for t, c in enumerate(irr_low):
            out[k - degree + t] = (out[k - degree + t] - v * c) % p
    for k in range(degree):
        out[k] = (out[k] + prod[k]) % p
    return _fp_poly_encode(tuple(out), p)


@lru_cache(maxsize=4)
def _f9_table() -> np.ndarray:
    """GF(9) over F_3; irreducible x^2+x+2."""
    n = 9
    degree = 2
    irr_low = (2, 1)

    def mul(i: int, j: int) -> int:
        return _fp_poly_mul_mod(i, j, p=3, degree=degree, irr_low=irr_low)

    table = _build_table(n, mul)
    assert_associativity(table)
    return table


@lru_cache(maxsize=4)
def _f2_x7_table() -> np.ndarray:
    n = 128
    degree = 7

    def mul(i: int, j: int) -> int:
        return _f2_poly_mul_mod(i, j, degree)

    table = _build_table(n, mul)
    assert_associativity(table)
    return table


@lru_cache(maxsize=4)
def _tri3_f2_table() -> np.ndarray:
    n = 64

    def mul(i: int, j: int) -> int:
        return _tri3_f2_mul(i, j)

    table = _build_table(n, mul)
    assert_associativity(table)
    return table


def _product_table(table_a: np.ndarray, table_b: np.ndarray) -> np.ndarray:
    """Прямое произведение колец: (a,b)·(c,d) = (a·c, b·d)."""
    na, nb = table_a.shape[0], table_b.shape[0]
    n = na * nb
    out = np.zeros((n, n), dtype=np.int64)
    for ia in range(na):
        for ib in range(nb):
            i = ia * nb + ib
            for ja in range(na):
                for jb in range(nb):
                    j = ja * nb + jb
                    oa = int(table_a[ia, ja])
                    ob = int(table_b[ib, jb])
                    out[i, j] = oa * nb + ob
    assert_associativity(out)
    return out


def _split_product_index(idx: int, nb: int) -> tuple[int, int]:
    return idx // nb, idx % nb


@lru_cache(maxsize=16)
def _cached_product_table(ring_a: str, ring_b: str) -> np.ndarray:
    ta, _ = build_mul_table(ring_a)
    tb, _ = build_mul_table(ring_b)
    return _product_table(ta, tb)


def get_ring_spec(ring_id: str) -> RingSpec:
    if "_x_" in ring_id:
        left, right = ring_id.split("_x_", 1)
        sa, sb = get_ring_spec(left), get_ring_spec(right)
        return RingSpec(
            id=ring_id,
            name=f"{sa.name} × {sb.name}",
            n_elements=sa.n_elements * sb.n_elements,
            commutative=sa.commutative and sb.commutative,
            prime=0,
            ring_type="product",
        )
    specs = {
        "f3": RingSpec(
            id="f3",
            name="F_3",
            n_elements=3,
            commutative=True,
            prime=3,
            ring_type="field",
        ),
        "mat2_f2": RingSpec(
            id="mat2_f2",
            name="Mat_2(F_2)",
            n_elements=16,
            commutative=False,
            prime=2,
            ring_type="matrix_full",
        ),
        "mat2_f3": RingSpec(
            id="mat2_f3",
            name="Mat_2(F_3)",
            n_elements=81,
            commutative=False,
            prime=3,
            ring_type="matrix_full",
        ),
        "quat_f3": RingSpec(
            id="quat_f3",
            name="H(F_3)",
            n_elements=81,
            commutative=False,
            prime=3,
            ring_type="quaternion",
        ),
        "tri2_f2": RingSpec(
            id="tri2_f2",
            name="T_2(F_2)",
            n_elements=8,
            commutative=False,
            prime=2,
            ring_type="matrix_upper_tri",
            encoding_version=TRI2_ENCODING_VERSION,
        ),
        "tri2_f3": RingSpec(
            id="tri2_f3",
            name="T_2(F_3)",
            n_elements=27,
            commutative=False,
            prime=3,
            ring_type="matrix_upper_tri",
            encoding_version=TRI2_ENCODING_VERSION,
        ),
        "tri3_f2": RingSpec(
            id="tri3_f2",
            name="T_3(F_2)",
            n_elements=64,
            commutative=False,
            prime=2,
            ring_type="matrix_upper_tri",
        ),
        "f2_x7": RingSpec(
            id="f2_x7",
            name="F_2[x]/(x^7)",
            n_elements=128,
            commutative=True,
            prime=2,
            ring_type="polynomial_local",
        ),
        "ext_f2_k6": RingSpec(
            id="ext_f2_k6",
            name="Λ(F₂⁶)",
            n_elements=65,
            commutative=True,
            prime=2,
            ring_type="exterior",
        ),
        "ext_f2_k8": RingSpec(
            id="ext_f2_k8",
            name="Λ(F₂⁸)",
            n_elements=257,
            commutative=True,
            prime=2,
            ring_type="exterior",
        ),
        "ext_f3_k6": RingSpec(
            id="ext_f3_k6",
            name="Λ(F₃⁶) [basis monoid — not full algebra]",
            n_elements=128,
            commutative=False,
            prime=3,
            ring_type="exterior_monoid",
        ),
        "ext_f3_k7": RingSpec(
            id="ext_f3_k7",
            name="Λ(F₃⁷) [basis monoid — not full algebra]",
            n_elements=256,
            commutative=False,
            prime=3,
            ring_type="exterior_monoid",
        ),
        "lam_f3_k2": RingSpec(
            id="lam_f3_k2",
            name="Λ(F₃²) full",
            n_elements=81,
            commutative=False,
            prime=3,
            ring_type="exterior_full",
        ),
        "lam_f2_k3t": RingSpec(
            id="lam_f2_k3t",
            name="Λ(F₂³)/Λ^{≥3}",
            n_elements=128,
            # char 2: e_i e_j = e_j e_i (signs collapse); not an NC exterior stress test
            commutative=True,
            prime=2,
            ring_type="exterior_trunc",
        ),
        "f2_s3": RingSpec(
            id="f2_s3",
            name="F₂[S₃]",
            n_elements=64,
            commutative=False,
            prime=2,
            ring_type="group_ring",
        ),
        "f2_d4": RingSpec(
            id="f2_d4",
            name="F₂[D₄]",
            n_elements=256,
            commutative=False,
            prime=2,
            ring_type="group_ring",
        ),
        "f3_s3": RingSpec(
            id="f3_s3",
            name="F₃[S₃]",
            n_elements=729,
            commutative=False,
            prime=3,
            ring_type="group_ring",
        ),
        "f2_z6": RingSpec(
            id="f2_z6",
            name="F₂[ℤ₆]",
            n_elements=64,
            commutative=True,
            prime=2,
            ring_type="group_ring",
        ),
        "f2_z7": RingSpec(
            id="f2_z7",
            name="F₂[ℤ₇]",
            n_elements=128,
            commutative=True,
            prime=2,
            ring_type="group_ring",
        ),
        "f8": RingSpec(
            id="f8",
            name="GF(8)",
            n_elements=8,
            commutative=True,
            prime=2,
            ring_type="field_extension",
        ),
        "f9": RingSpec(
            id="f9",
            name="GF(9)",
            n_elements=9,
            commutative=True,
            prime=3,
            ring_type="field_extension",
        ),
        "f8_x_f8": RingSpec(
            id="f8_x_f8",
            name="GF(8)×GF(8)",
            n_elements=64,
            commutative=True,
            prime=2,
            ring_type="crt_product",
        ),
        "f3_x_f3_x_f9": RingSpec(
            id="f3_x_f3_x_f9",
            name="F₃×F₃×GF(9)",
            n_elements=81,
            commutative=True,
            prime=3,
            ring_type="crt_product",
        ),
        "f3_z4": RingSpec(
            id="f3_z4",
            name="F₃[ℤ₄]",
            n_elements=81,
            commutative=True,
            prime=3,
            ring_type="group_ring",
        ),
    }
    if ring_id not in specs:
        raise ValueError(f"unknown ring_id {ring_id!r}; choose from {sorted(RING_REGISTRY)}")
    return specs[ring_id]


def build_mul_table(ring_id: str) -> tuple[np.ndarray, RingSpec]:
    """Построить таблицу умножения кольца.

    Returns:
        (table [n,n] int64, RingSpec)
    """
    spec = get_ring_spec(ring_id)
    if "_x_" in ring_id:
        left, right = ring_id.split("_x_", 1)
        table = _cached_product_table(left, right)
    elif ring_id == "f3":
        table = _field_fp_table(3)
    elif ring_id == "mat2_f2":
        table = _mat2_fp_table(2, upper_triangular=False)
    elif ring_id == "mat2_f3":
        table = _mat2_fp_table(3, upper_triangular=False)
    elif ring_id == "quat_f3":
        table = _quat_f3_table()
    elif ring_id == "tri2_f2":
        table = _mat2_fp_table(2, upper_triangular=True)
    elif ring_id == "tri2_f3":
        table = _mat2_fp_table(3, upper_triangular=True)
    elif ring_id == "tri3_f2":
        table = _tri3_f2_table()
    elif ring_id == "f2_x7":
        table = _f2_x7_table()
    elif ring_id.startswith("ext_f"):
        p, k = _ext_parse_ring_id(ring_id)
        table = _ext_table(k, p)
    elif ring_id == "lam_f3_k2":
        from .true_algebras import lam_f3_k2_table

        table = lam_f3_k2_table()
    elif ring_id == "lam_f2_k3t":
        from .true_algebras import lam_f2_k3_trunc_table

        table = lam_f2_k3_trunc_table()
    elif ring_id == "f2_s3":
        from .true_algebras import f2_s3_table

        table = f2_s3_table()
    elif ring_id == "f2_d4":
        from .true_algebras import f2_d4_table

        table = f2_d4_table()
    elif ring_id == "f3_s3":
        from .true_algebras import f3_s3_table

        table = f3_s3_table()
    elif ring_id == "f2_z6":
        from .true_algebras import f2_z6_table

        table = f2_z6_table()
    elif ring_id == "f2_z7":
        from .true_algebras import f2_z7_table

        table = f2_z7_table()
    elif ring_id == "f8":
        table = _f8_table()
    elif ring_id == "f9":
        table = _f9_table()
    elif ring_id == "f3_z4":
        from .true_algebras import f3_z4_table

        table = f3_z4_table()
    else:
        raise ValueError(f"unknown ring_id {ring_id!r}")
    if table.shape[0] != spec.n_elements:
        raise RuntimeError(f"{ring_id}: table size {table.shape[0]} != spec {spec.n_elements}")
    return table, spec


def mat2_f3_identity_index() -> int:
    """Единичная матрица I в Mat_2(F_3)."""
    return _mat2_encode(1, 0, 0, 1, 3)
