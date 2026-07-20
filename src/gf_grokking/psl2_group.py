"""PSL(2, q): таблица умножения над конечным полем GF(q).

Поддерживаются простые степени ``q = p^k``. Для ``k = 1`` используется обычная
арифметика по модулю простого ``p``; для ``k > 1`` строится поле
``GF(p^k)`` в полиномиальном базисе по автоматически найденному неприводимому
многочлену степени ``k`` над ``F_p``.
"""

from __future__ import annotations

from itertools import product
from math import gcd

import numpy as np

from .alternating_group import split_pair_indices

SL2Matrix = tuple[int, int, int, int]

__all__ = [
    "SL2Matrix",
    "build_psl2_mult_table",
    "psl2_group_order",
    "split_pair_indices",
]


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    d = 3
    while d * d <= n:
        if n % d == 0:
            return False
        d += 2
    return True


def _prime_power(q: int) -> tuple[int, int]:
    """Return (p, k) for q = p^k, or raise for non-prime-powers."""
    if q < 2:
        raise ValueError(f"q must be >= 2, got {q}")
    for p in range(2, q + 1):
        if not _is_prime(p) or q % p != 0:
            continue
        n = q
        k = 0
        while n % p == 0:
            n //= p
            k += 1
        if n == 1:
            return p, k
        break
    raise ValueError(f"q must be a prime power p^k, got {q}")


def _decode(x: int, p: int, k: int) -> list[int]:
    coeffs: list[int] = []
    for _ in range(k):
        coeffs.append(x % p)
        x //= p
    return coeffs


def _encode(coeffs: list[int], p: int) -> int:
    out = 0
    base = 1
    for c in coeffs:
        out += (c % p) * base
        base *= p
    return out


def _trim(poly: list[int]) -> list[int]:
    while len(poly) > 1 and poly[-1] == 0:
        poly.pop()
    return poly


def _poly_mod(poly: list[int], divisor: list[int], p: int) -> list[int]:
    """Remainder of poly / divisor over F_p. divisor must be monic."""
    rem = [c % p for c in poly]
    divisor = _trim([c % p for c in divisor])
    deg_div = len(divisor) - 1
    if deg_div < 0 or divisor[-1] != 1:
        raise ValueError("divisor must be monic")
    while len(rem) - 1 >= deg_div and not (len(rem) == 1 and rem[0] == 0):
        shift = len(rem) - 1 - deg_div
        coeff = rem[-1] % p
        if coeff:
            for i in range(deg_div + 1):
                rem[shift + i] = (rem[shift + i] - coeff * divisor[i]) % p
        _trim(rem)
    return rem


def _is_irreducible(poly: list[int], p: int) -> bool:
    """Brute-force irreducibility test for small monic polynomials."""
    deg = len(poly) - 1
    if deg <= 0 or poly[-1] != 1:
        return False
    for d in range(1, deg // 2 + 1):
        for coeffs in product(range(p), repeat=d):
            divisor = list(coeffs) + [1]
            if _poly_mod(poly, divisor, p) == [0]:
                return False
    return True


def _find_irreducible_poly(p: int, k: int) -> list[int]:
    """Find a monic irreducible polynomial of degree k over F_p."""
    for coeffs in product(range(p), repeat=k):
        # Non-zero constant term is necessary for irreducibility for k > 1.
        if coeffs[0] == 0:
            continue
        poly = list(coeffs) + [1]
        if _is_irreducible(poly, p):
            return poly
    raise RuntimeError(f"no irreducible polynomial found for GF({p}^{k})")


def _field_tables(q: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return add, sub, mul, neg tables for GF(q)."""
    p, k = _prime_power(q)
    elems = np.arange(q, dtype=np.int64)

    if k == 1:
        add = (elems[:, None] + elems[None, :]) % p
        sub = (elems[:, None] - elems[None, :]) % p
        mul = (elems[:, None] * elems[None, :]) % p
        neg = (-elems) % p
        return add.astype(np.int64), sub.astype(np.int64), mul.astype(np.int64), neg.astype(np.int64)

    modulus = _find_irreducible_poly(p, k)
    add = np.zeros((q, q), dtype=np.int64)
    sub = np.zeros((q, q), dtype=np.int64)
    mul = np.zeros((q, q), dtype=np.int64)
    neg = np.zeros(q, dtype=np.int64)

    decoded = [_decode(x, p, k) for x in range(q)]
    for a in range(q):
        ca = decoded[a]
        neg[a] = _encode([(-x) % p for x in ca], p)
        for b in range(q):
            cb = decoded[b]
            add[a, b] = _encode([(x + y) % p for x, y in zip(ca, cb)], p)
            sub[a, b] = _encode([(x - y) % p for x, y in zip(ca, cb)], p)

            prod_coeffs = [0] * (2 * k - 1)
            for i, x in enumerate(ca):
                for j, y in enumerate(cb):
                    prod_coeffs[i + j] = (prod_coeffs[i + j] + x * y) % p

            for deg in range(2 * k - 2, k - 1, -1):
                coeff = prod_coeffs[deg] % p
                if coeff == 0:
                    continue
                shift = deg - k
                for i in range(k):
                    prod_coeffs[shift + i] = (
                        prod_coeffs[shift + i] - coeff * modulus[i]
                    ) % p
                prod_coeffs[deg] = 0
            mul[a, b] = _encode(prod_coeffs[:k], p)

    return add, sub, mul, neg


def _mat_mul(
    m1: SL2Matrix,
    m2: SL2Matrix,
    add: np.ndarray,
    mul: np.ndarray,
) -> SL2Matrix:
    a, b, c, d = m1
    e, f, g, h = m2
    return (
        int(add[mul[a, e], mul[b, g]]),
        int(add[mul[a, f], mul[b, h]]),
        int(add[mul[c, e], mul[d, g]]),
        int(add[mul[c, f], mul[d, h]]),
    )


def _mat_scalar(m: SL2Matrix, scalar: int, mul: np.ndarray) -> SL2Matrix:
    return tuple(int(mul[scalar, x]) for x in m)


def _canonical_psl(m: SL2Matrix, neg_one: int, mul: np.ndarray) -> SL2Matrix:
    if neg_one == 1:
        return m
    return min(m, _mat_scalar(m, neg_one, mul))


def _iter_sl2(
    q: int,
    add: np.ndarray,
    sub: np.ndarray,
    mul: np.ndarray,
) -> list[SL2Matrix]:
    mats: list[SL2Matrix] = []
    for a in range(q):
        for b in range(q):
            for c in range(q):
                for d in range(q):
                    det = int(sub[mul[a, d], mul[b, c]])
                    if det == 1:
                        mats.append((a, b, c, d))
    return mats


def psl2_group_order(q: int) -> int:
    """Порядок PSL(2, q) для простых степеней q = p^k."""
    _prime_power(q)
    if q == 2:
        return 6
    return q * (q * q - 1) // gcd(2, q - 1)


def build_psl2_mult_table(q: int) -> tuple[np.ndarray, int]:
    """Строит таблицу умножения PSL(2, q)."""
    if q < 3:
        raise ValueError(f"PSL(2, q) requires q >= 3, got {q}")

    add, sub, mul, neg = _field_tables(q)
    neg_one = int(neg[1])

    canon_to_idx: dict[SL2Matrix, int] = {}
    for m in _iter_sl2(q, add, sub, mul):
        canon = _canonical_psl(m, neg_one, mul)
        canon_to_idx.setdefault(canon, len(canon_to_idx))

    reps = sorted(canon_to_idx)
    n_elements = len(reps)
    expected = psl2_group_order(q)
    if n_elements != expected:
        raise RuntimeError(
            f"PSL(2,{q}) order mismatch: built {n_elements}, expected {expected}"
        )

    mult_table = np.zeros((n_elements, n_elements), dtype=np.int64)
    for i, m1 in enumerate(reps):
        for j, m2 in enumerate(reps):
            prod = _canonical_psl(_mat_mul(m1, m2, add, mul), neg_one, mul)
            mult_table[i, j] = canon_to_idx[prod]

    return mult_table, n_elements
