"""Full finite algebras (not basis monoids): exterior, truncated exterior, group rings."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from .finite_rings import assert_associativity


def _coeff_decode(idx: int, n_basis: int, p: int) -> np.ndarray:
    out = np.zeros(n_basis, dtype=np.int64)
    x = idx
    for k in range(n_basis):
        out[k] = x % p
        x //= p
    return out


def _coeff_encode(coeffs: np.ndarray, p: int) -> int:
    idx = 0
    mult = 1
    for k in range(len(coeffs)):
        idx += int(coeffs[k]) * mult
        mult *= p
    return idx


def _wedge_sign_disjoint(mi: int, mj: int) -> int:
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


def _lam_basis_masks(k: int, *, max_degree: int | None = None) -> list[int]:
    """Basis monomial masks for Λ(F_p^k); optional truncation by total degree."""
    masks = [0]
    for r in range(1, k + 1):
        if max_degree is not None and r > max_degree:
            break
        for mask in range(1 << k):
            if mask.bit_count() == r:
                masks.append(mask)
    return masks


def _lam_mul_table(k: int, p: int, *, max_degree: int | None = None) -> np.ndarray:
    masks = _lam_basis_masks(k, max_degree=max_degree)
    n_basis = len(masks)
    mask_to_idx = {m: i for i, m in enumerate(masks)}
    n = p**n_basis

    def wedge_vec(mi: int, mj: int) -> np.ndarray:
        out = np.zeros(n_basis, dtype=np.int64)
        if mi == 0:
            out[mask_to_idx[mj]] = 1
            return out
        if mj == 0:
            out[mask_to_idx[mi]] = 1
            return out
        if mi & mj:
            return out
        if max_degree is not None and mi.bit_count() + mj.bit_count() > max_degree:
            return out
        sign = _wedge_sign_disjoint(mi, mj)
        if p == 2:
            sign = 1
        out_mask = mi | mj
        out[mask_to_idx[out_mask]] = sign % p
        return out

    table = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        ai = _coeff_decode(i, n_basis, p)
        for j in range(n):
            aj = _coeff_decode(j, n_basis, p)
            acc = np.zeros(n_basis, dtype=np.int64)
            for bi in range(n_basis):
                if ai[bi] == 0:
                    continue
                for bj in range(n_basis):
                    if aj[bj] == 0:
                        continue
                    coef = (ai[bi] * aj[bj]) % p
                    acc = (acc + coef * wedge_vec(masks[bi], masks[bj])) % p
            table[i, j] = _coeff_encode(acc, p)
    assert_associativity(table)
    return table


@lru_cache(maxsize=4)
def lam_f3_k2_table() -> np.ndarray:
    """Full exterior algebra Λ(F₃²), |R| = 3⁴ = 81."""
    return _lam_mul_table(2, 3)


@lru_cache(maxsize=4)
def lam_f2_k3_trunc_table() -> np.ndarray:
    """Λ(F₂³) / Λ^{≥3}, |R| = 2⁷ = 128."""
    return _lam_mul_table(3, 2, max_degree=2)


def _perm_compose(p: tuple[int, ...], q: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(p[q[i]] for i in range(len(p)))


def _s3_perms() -> tuple[tuple[int, ...], ...]:
    return (
        (0, 1, 2),
        (1, 0, 2),
        (2, 1, 0),
        (0, 2, 1),
        (1, 2, 0),
        (2, 0, 1),
    )


@lru_cache(maxsize=1)
def _s3_cayley() -> np.ndarray:
    perms = _s3_perms()
    n = len(perms)
    idx = {p: i for i, p in enumerate(perms)}
    cay = np.zeros((n, n), dtype=np.int64)
    for i, pi in enumerate(perms):
        for j, pj in enumerate(perms):
            cay[i, j] = idx[_perm_compose(pi, pj)]
    return cay


def _d4_mul(i: int, j: int) -> int:
    """D₄ = ⟨r,s | r⁴=1, s²=1, srs=r⁻¹⟩; indices 0..3 = r^a, 4..7 = s·r^a."""
    ai, si = (i % 4, i >= 4)
    aj, sj = (j % 4, j >= 4)
    if not si and not sj:
        return (ai + aj) % 4
    if not si and sj:
        return 4 + ((aj - ai) % 4)
    if si and not sj:
        return 4 + ((ai + aj) % 4)
    return (aj - ai) % 4


@lru_cache(maxsize=1)
def _d4_cayley() -> np.ndarray:
    n = 8
    cay = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        for j in range(n):
            cay[i, j] = _d4_mul(i, j)
    return cay


def _cyclic_cayley(n: int) -> np.ndarray:
    """Cayley table of ℤ_n under addition."""
    cay = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        for j in range(n):
            cay[i, j] = (i + j) % n
    return cay


@lru_cache(maxsize=1)
def _z4_cayley() -> np.ndarray:
    return _cyclic_cayley(4)


@lru_cache(maxsize=1)
def _z6_cayley() -> np.ndarray:
    return _cyclic_cayley(6)


@lru_cache(maxsize=1)
def _z7_cayley() -> np.ndarray:
    return _cyclic_cayley(7)


def _fp_group_ring_table(cayley: np.ndarray, p: int) -> np.ndarray:
    n_g = cayley.shape[0]
    n = p**n_g
    table = np.zeros((n, n), dtype=np.int64)
    for f in range(n):
        cf = _coeff_decode(f, n_g, p)
        for g in range(n):
            cg = _coeff_decode(g, n_g, p)
            out = np.zeros(n_g, dtype=np.int64)
            for x in range(n_g):
                if cf[x] == 0:
                    continue
                for y in range(n_g):
                    if cg[y] == 0:
                        continue
                    h = int(cayley[x, y])
                    out[h] = (out[h] + cf[x] * cg[y]) % p
            table[f, g] = _coeff_encode(out, p)
    assert_associativity(table)
    return table


@lru_cache(maxsize=1)
def f2_s3_table() -> np.ndarray:
    """Group ring F₂[S₃], |R| = 2⁶ = 64."""
    return _fp_group_ring_table(_s3_cayley(), 2)


@lru_cache(maxsize=1)
def f2_d4_table() -> np.ndarray:
    """Group ring F₂[D₄], |R| = 2⁸ = 256."""
    return _fp_group_ring_table(_d4_cayley(), 2)


@lru_cache(maxsize=1)
def f3_s3_table() -> np.ndarray:
    """Group ring F₃[S₃], |R| = 3⁶ = 729."""
    return _fp_group_ring_table(_s3_cayley(), 3)


@lru_cache(maxsize=1)
def f3_z4_table() -> np.ndarray:
    """Group ring F₃[ℤ₄], |R| = 3⁴ = 81. Zheng Table 3 semisimple benchmark."""
    return _fp_group_ring_table(_z4_cayley(), 3)


@lru_cache(maxsize=1)
def f2_z6_table() -> np.ndarray:
    """Group ring F₂[ℤ₆], |R| = 2⁶ = 64. Zheng Table 8: rf=50%, 4/10 grok."""
    return _fp_group_ring_table(_z6_cayley(), 2)


@lru_cache(maxsize=1)
def f2_z7_table() -> np.ndarray:
    """Group ring F₂[ℤ₇], |R| = 2⁷ = 128. Zheng Table 8: rf=0%, 10/10 grok."""
    return _fp_group_ring_table(_z7_cayley(), 2)


def lam_f3_k2_identity_index() -> int:
    coeffs = np.zeros(4, dtype=np.int64)
    coeffs[0] = 1
    return _coeff_encode(coeffs, 3)


def lam_f2_k3_trunc_identity_index() -> int:
    coeffs = np.zeros(7, dtype=np.int64)
    coeffs[0] = 1
    return _coeff_encode(coeffs, 2)


def f2_group_ring_identity_index() -> int:
    return 1
