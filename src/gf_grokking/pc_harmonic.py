"""Поиск дискретного log_g и гармоники k для PCA-осей.

Для PC_N ищем пару (g, k), где PC_N ≈ a + b·sin(2πk·log_g/order) + c·cos(...).
Среди всех пар с R² ≈ максимуму берём **минимальный k** — каноническое
представление гармоники на циклической группе (Z/orderZ)*.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd

import numpy as np


def harmonic_wavelength(order: int, k: int) -> int:
    """Длина волны sin(2πk·log/order) в координате log: order/gcd(k, order)."""
    return order // gcd(k, order)


def harmonic_n_tiles(order: int, k: int) -> int:
    """Сколько копий фундаментальной волны укладывается в log ∈ [0, order-1]."""
    return gcd(k, order)


@dataclass(frozen=True)
class HarmonicFit:
    g: int
    k: int
    r2: float
    rel_resid: float
    amplitude: float
    phase_rad: float

    @property
    def cycles(self) -> float:
        return float(self.k)


def sin_cos_fit(y: np.ndarray, phase: np.ndarray) -> tuple[float, float, float, float, np.ndarray]:
    """Фит y ≈ a + b·sin(φ) + c·cos(φ). Возвращает r2, rel_resid, amp, phase_offset, pred."""
    y = np.asarray(y, dtype=np.float64)
    phase = np.asarray(phase, dtype=np.float64)
    x = np.column_stack([np.ones_like(phase), np.sin(phase), np.cos(phase)])
    coef, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    pred = x @ coef
    var = float(np.var(y))
    r2 = float(1.0 - np.var(y - pred) / var) if var > 0 else 0.0
    rel_resid = float(np.std(y - pred) / np.sqrt(var)) if var > 0 else 1.0
    amp = float(np.hypot(coef[1], coef[2]))
    phase_off = float(np.arctan2(coef[2], coef[1]))
    return r2, rel_resid, amp, phase_off, pred


def best_k_for_log(
    y: np.ndarray,
    log_arr: np.ndarray,
    *,
    order: int,
    k_max: int | None = None,
) -> HarmonicFit | None:
    """Лучший k для фиксированного log_g (примитивный корень g)."""
    mask = np.isfinite(log_arr)
    if mask.sum() < order - 1:
        return None
    yf = y[mask]
    lf = log_arr[mask].astype(np.float64)
    k_hi = k_max if k_max is not None else order // 2
    best: HarmonicFit | None = None
    for k in range(1, k_hi + 1):
        phase = 2.0 * np.pi * k * lf / order
        r2, rel, amp, ph, _ = sin_cos_fit(yf, phase)
        cand = HarmonicFit(g=-1, k=k, r2=r2, rel_resid=rel, amplitude=amp, phase_rad=ph)
        if best is None or cand.r2 > best.r2 + 1e-12 or (
            abs(cand.r2 - best.r2) <= 1e-12 and cand.k < best.k
        ):
            best = cand
    return best


def log_array_for(g: int, mul: np.ndarray | object, num: int) -> np.ndarray:
    """Дискретный лог по основанию g; NaN для 0."""
    order = num - 1
    out = np.full(num, np.nan, dtype=np.float64)
    x = 1
    for step in range(order):
        out[x] = step
        if hasattr(mul, "shape"):
            x = int(mul[x, g])
        else:
            x = int(mul[x][g])
    return out


def is_primitive_log(log_arr: np.ndarray, order: int) -> bool:
    mask = np.isfinite(log_arr)
    return int(mask.sum()) == order


def scan_all_gk(
    y: np.ndarray,
    mul: np.ndarray | object,
    num: int,
    *,
    g_min: int = 2,
    g_max: int | None = None,
    k_max: int | None = None,
) -> list[HarmonicFit]:
    """Полный перебор (g, k): для каждого примитивного g — лучший k."""
    order = num - 1
    g_hi = g_max if g_max is not None else order
    rows: list[HarmonicFit] = []
    for g in range(g_min, g_hi + 1):
        log_arr = log_array_for(g, mul, num)
        if not is_primitive_log(log_arr, order):
            continue
        fit = best_k_for_log(y, log_arr, order=order, k_max=k_max)
        if fit is None:
            continue
        rows.append(
            HarmonicFit(
                g=g,
                k=fit.k,
                r2=fit.r2,
                rel_resid=fit.rel_resid,
                amplitude=fit.amplitude,
                phase_rad=fit.phase_rad,
            )
        )
    return rows


def min_k_representation(
    y: np.ndarray,
    mul: np.ndarray | object,
    num: int,
    *,
    g_min: int = 2,
    g_max: int | None = None,
    k_max: int | None = None,
    r2_tol: float = 1e-4,
) -> tuple[HarmonicFit, list[HarmonicFit]]:
    """Каноническая пара (g, k) с минимальным k среди почти оптимальных по R².

    Алгоритм:
    1. Для каждого примитивного g находим k*, максимизирующий R².
    2. Берём R²_max по всем g.
    3. Среди пар с R² >= R²_max - r2_tol выбираем минимальный k.
    4. При равном k — меньший g, затем выше R².

    Возвращает (лучший, все кандидаты с тем же минимальным k).
    """
    all_fits = scan_all_gk(
        y, mul, num, g_min=g_min, g_max=g_max, k_max=k_max,
    )
    if not all_fits:
        raise ValueError("no primitive log bases in range")

    r2_max = max(f.r2 for f in all_fits)
    near = [f for f in all_fits if f.r2 >= r2_max - r2_tol]
    k_min = min(f.k for f in near)
    tied = [f for f in near if f.k == k_min]
    tied.sort(key=lambda f: (f.g, -f.r2))
    best = tied[0]
    return best, tied


def min_k_for_pcs(
    pcs: np.ndarray,
    mul: np.ndarray | object,
    num: int,
    *,
    n_pcs: int | None = None,
    g_min: int = 2,
    g_max: int | None = None,
    k_max: int | None = None,
    r2_tol: float = 1e-4,
) -> list[dict[str, object]]:
    """Для каждой PCA-оси — канонический (g, k) с минимальным k."""
    n = n_pcs if n_pcs is not None else pcs.shape[1]
    n = min(n, pcs.shape[1])
    rows: list[dict[str, object]] = []
    for i in range(n):
        best, tied = min_k_representation(
            pcs[:, i],
            mul,
            num,
            g_min=g_min,
            g_max=g_max,
            k_max=k_max,
            r2_tol=r2_tol,
        )
        order = num - 1
        wl = harmonic_wavelength(order, best.k)
        rows.append(
            {
                "pc": i + 1,
                "g": best.g,
                "k": best.k,
                "r2": best.r2,
                "rel_resid": best.rel_resid,
                "amplitude": best.amplitude,
                "phase_rad": best.phase_rad,
                "cycles": best.cycles,
                "wavelength_log": wl,
                "n_tiles": harmonic_n_tiles(order, best.k),
                "n_equivalent_g": len(tied),
                "equivalent_g": [f.g for f in tied[:12]],
            }
        )
    return rows
