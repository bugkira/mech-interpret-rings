#!/usr/bin/env python3
"""Verify reviewer-proposed true algebras vs legacy exterior monoid."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gf_grokking.finite_rings import build_mul_table, get_ring_spec  # noqa: E402
from gf_grokking.ring_wedderburn import get_wedderburn_spec  # noqa: E402
from gf_grokking.true_algebras import (  # noqa: E402
    _coeff_encode,
    lam_f3_k2_identity_index,
)

OUT = ROOT / "new_paper_rings" / "algebra_alternatives_verified.json"


def _check(ring_id: str, *, note: str) -> dict:
    table, spec = build_mul_table(ring_id)
    n = spec.n_elements
    e = None
    for i in range(n):
        if all(table[i, j] == j for j in range(n)) and all(table[j, i] == j for j in range(n)):
            e = i
            break
    noncomm = False
    for i in range(min(n, 32)):
        for j in range(i + 1, min(n, 32)):
            if table[i, j] != table[j, i]:
                noncomm = True
                break
        if noncomm:
            break
    wb = get_wedderburn_spec(ring_id)
    return {
        "ring_id": ring_id,
        "name": spec.name,
        "n_elements": n,
        "ring_type": spec.ring_type,
        "identity_index": e,
        "non_commutative": noncomm,
        "n_wedderburn_components": len(wb.component_names),
        "note": note,
    }


def main() -> int:
    rows = [
        _check(
            "lam_f3_k2",
            note="Alt 1: full Λ(F₃²), |R|=3⁴=81; true algebra (all F₃-linear combos).",
        ),
        _check(
            "lam_f2_k3t",
            note="Alt 2: Λ(F₂³)/Λ^{≥3}, |R|=2⁷=128; degree-3 wedges zero.",
        ),
        _check(
            "f2_s3",
            note="Alt 3A: group ring F₂[S₃], |R|=2⁶=64; Wedderburn-aligned labels TBD.",
        ),
        _check(
            "f2_d4",
            note="Alt 3B: group ring F₂[D₄], |R|=2⁸=256; Zheng grokking-failure benchmark.",
        ),
        _check(
            "ext_f3_k6",
            note="REJECTED for paper: basis ±e_S monoid (|R|=128), not Λ(F₃⁶) with |R|=3⁶⁴.",
        ),
    ]

    # Sanity: Λ(F₃²) anticommutation
    table, _ = build_mul_table("lam_f3_k2")
    e1 = _coeff_encode(__import__("numpy").array([0, 1, 0, 0]), 3)
    e2 = _coeff_encode(__import__("numpy").array([0, 0, 1, 0]), 3)
    anticomm = table[e1, e2] != table[e2, e1]

    payload = {
        "verified_alternatives": rows[:4],
        "rejected_legacy": rows[4],
        "lam_f3_k2_anticommutative": bool(anticomm),
        "lam_f3_k2_identity": int(lam_f3_k2_identity_index()),
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"\n→ {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
