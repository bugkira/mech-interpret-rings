"""Training budget helpers: steps, epochs, and train-pair exposure."""

from __future__ import annotations

import math

PAPER_BATCH = 512
PAPER_MAX_STEPS = 100_000
PAPER_TRAIN_FRAC_SMALL = 0.5
PAPER_TRAIN_FRAC_LARGE = 0.3

BUDGET_STEPS = "steps"
BUDGET_MATCHED_EXPOSURES = "matched_exposures"


def default_train_fraction(n_elements: int) -> float:
    return PAPER_TRAIN_FRAC_SMALL if n_elements <= 64 else PAPER_TRAIN_FRAC_LARGE


def train_size(n_elements: int, train_fraction: float | None = None) -> int:
    frac = train_fraction if train_fraction is not None else default_train_fraction(n_elements)
    return int(n_elements * n_elements * frac)


def steps_per_epoch(train_size: int, batch_size: int) -> int:
    return max(1, math.ceil(train_size / batch_size))


def num_epochs(max_steps: int, train_size: int, batch_size: int) -> int:
    return max(1, math.ceil(max_steps / steps_per_epoch(train_size, batch_size)))


def pair_exposures(max_steps: int, train_size: int, batch_size: int) -> float:
    """Mean full passes over the training split (= epochs when one epoch covers train once)."""
    return max_steps / steps_per_epoch(train_size, batch_size)


def max_steps_for_pair_exposures(
    target_exposures: float,
    train_size: int,
    batch_size: int,
) -> int:
    return max(1, math.ceil(target_exposures * steps_per_epoch(train_size, batch_size)))


def reference_pair_exposures(
    reference_ring: str,
    *,
    max_steps: int = PAPER_MAX_STEPS,
    batch_size: int = PAPER_BATCH,
    train_fraction: float | None = None,
) -> float:
    from .finite_rings import get_ring_spec

    spec = get_ring_spec(reference_ring)
    ts = train_size(spec.n_elements, train_fraction)
    return pair_exposures(max_steps, ts, batch_size)


def resolve_training_budget(
    ring_id: str,
    *,
    budget_mode: str = BUDGET_STEPS,
    max_steps: int = PAPER_MAX_STEPS,
    batch_size: int = PAPER_BATCH,
    train_fraction: float | None = None,
    exposure_reference_ring: str = "tri2_f3_x_f3_x_f3",
    target_pair_exposures: float | None = None,
) -> dict[str, float | int | str]:
    from .finite_rings import get_ring_spec

    spec = get_ring_spec(ring_id)
    ts = train_size(spec.n_elements, train_fraction)
    spe = steps_per_epoch(ts, batch_size)

    if budget_mode == BUDGET_STEPS:
        resolved_steps = max_steps
        ref_exp = None
    elif budget_mode == BUDGET_MATCHED_EXPOSURES:
        ref_exp = (
            target_pair_exposures
            if target_pair_exposures is not None
            else reference_pair_exposures(
                exposure_reference_ring,
                max_steps=max_steps,
                batch_size=batch_size,
            )
        )
        resolved_steps = max_steps_for_pair_exposures(ref_exp, ts, batch_size)
    else:
        raise ValueError(f"unknown budget_mode {budget_mode!r}")

    epochs = num_epochs(resolved_steps, ts, batch_size)
    exposures = pair_exposures(resolved_steps, ts, batch_size)
    return {
        "budget_mode": budget_mode,
        "train_size": ts,
        "steps_per_epoch": spe,
        "max_steps": resolved_steps,
        "num_epochs": epochs,
        "pair_exposures": exposures,
        "reference_pair_exposures": ref_exp,
        "exposure_reference_ring": exposure_reference_ring if budget_mode == BUDGET_MATCHED_EXPOSURES else None,
    }
