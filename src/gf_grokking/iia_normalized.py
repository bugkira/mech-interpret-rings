"""Normalized IIA aggregates (silent-patch fallacy fix).

Per component $i$ on a fixed checkpoint:
  Δ_i = subspace_iia_i - random_subspace_iia_i
  gain_i = (subspace_iia_i - random_subspace_iia_i) / (raw_iia_i - random_subspace_iia_i)

Means are unweighted over components with n_pairs > 0. Components with
|raw - rand| below eps get gain_i = NaN (excluded from mean gain).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def normalized_subspace_gain(
    subspace_iia: float,
    raw_iia: float,
    random_subspace_iia: float,
    *,
    eps: float = 1e-9,
) -> float:
    denom = raw_iia - random_subspace_iia
    if not math.isfinite(denom) or abs(denom) < eps:
        return float("nan")
    return float((subspace_iia - random_subspace_iia) / denom)


def enrich_per_component_normalized(
    per_component: list[dict[str, Any]],
    *,
    eps: float = 1e-9,
) -> dict[str, float]:
    """Add per-row delta/gain and return aggregate means."""
    deltas: list[float] = []
    gains: list[float] = []
    for row in per_component:
        if int(row.get("n_pairs", 0) or 0) <= 0:
            continue
        sub = float(row["subspace_iia"])
        raw = float(row["raw_iia"])
        rand = float(row["random_subspace_iia"])
        delta = sub - rand
        gain = normalized_subspace_gain(sub, raw, rand, eps=eps)
        row["subspace_delta"] = delta
        row["normalized_subspace_gain"] = gain
        deltas.append(delta)
        if math.isfinite(gain):
            gains.append(gain)
    return {
        "mean_subspace_delta": float(np.mean(deltas)) if deltas else float("nan"),
        "mean_normalized_subspace_gain": float(np.mean(gains)) if gains else float("nan"),
    }


def enrich_condition_normalized(condition: dict[str, Any], *, eps: float = 1e-9) -> None:
    """In-place: enrich per_component and add aggregate normalized fields."""
    per = condition.get("per_component")
    if not per:
        return
    agg = enrich_per_component_normalized(per, eps=eps)
    condition.update(agg)
