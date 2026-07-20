"""Paired significance tests for DAS vs axis / random baselines."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PairedProportionTest:
    """Paired comparison of two Bernoulli outcomes on the same items."""

    n: int
    p_a: float
    p_b: float
    risk_diff: float
    cohens_h: float
    mcnemar_b: int  # only A correct
    mcnemar_c: int  # only B correct
    mcnemar_p: float
    bootstrap_ci_low: float
    bootstrap_ci_high: float

    @property
    def significant_005(self) -> bool:
        return self.mcnemar_p < 0.05


def cohens_h(p1: float, p2: float) -> float:
    p1 = float(np.clip(p1, 0.0, 1.0))
    p2 = float(np.clip(p2, 0.0, 1.0))
    return float(2.0 * (math.asin(math.sqrt(p1)) - math.asin(math.sqrt(p2))))


def mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for discordant counts (b, c)."""
    n = int(b) + int(c)
    if n == 0:
        return 1.0
    k = min(int(b), int(c))
    p_one = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return float(min(1.0, 2.0 * p_one))


def bootstrap_risk_diff_ci(
    a_ok: np.ndarray,
    b_ok: np.ndarray,
    *,
    n_boot: int = 10_000,
    seed: int = 0,
) -> tuple[float, float]:
    a_ok = np.asarray(a_ok, dtype=bool)
    b_ok = np.asarray(b_ok, dtype=bool)
    if a_ok.shape != b_ok.shape:
        raise ValueError("a_ok and b_ok must have the same shape")
    n = len(a_ok)
    if n == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = a_ok[idx].mean() - b_ok[idx].mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(lo), float(hi)


def paired_proportion_test(
    a_ok: np.ndarray,
    b_ok: np.ndarray,
    *,
    bootstrap_seed: int = 0,
) -> PairedProportionTest:
    a_ok = np.asarray(a_ok, dtype=bool)
    b_ok = np.asarray(b_ok, dtype=bool)
    if a_ok.shape != b_ok.shape:
        raise ValueError("a_ok and b_ok must have the same shape")
    n = int(len(a_ok))
    p_a = float(a_ok.mean()) if n else float("nan")
    p_b = float(b_ok.mean()) if n else float("nan")
    only_a = int(np.sum(a_ok & ~b_ok))
    only_b = int(np.sum(~a_ok & b_ok))
    ci_lo, ci_hi = bootstrap_risk_diff_ci(a_ok, b_ok, seed=bootstrap_seed)
    return PairedProportionTest(
        n=n,
        p_a=p_a,
        p_b=p_b,
        risk_diff=p_a - p_b,
        cohens_h=cohens_h(p_a, p_b),
        mcnemar_b=only_a,
        mcnemar_c=only_b,
        mcnemar_p=mcnemar_exact_p(only_a, only_b),
        bootstrap_ci_low=ci_lo,
        bootstrap_ci_high=ci_hi,
    )
