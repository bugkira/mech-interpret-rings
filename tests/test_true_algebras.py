"""Tests for full exterior and group-ring algebras."""

from __future__ import annotations

import numpy as np

from gf_grokking.finite_rings import build_mul_table, get_ring_spec
from gf_grokking.iia import sample_matched_pairs
from gf_grokking.ring_wedderburn import (
    LAM_BASIS_SCHEME,
    LAM_WM_SCHEME,
    get_wedderburn_spec,
    labels_to_signatures,
)
from gf_grokking.true_algebras import (
    _coeff_decode,
    _coeff_encode,
    lam_f3_k2_identity_index,
)


def test_lam_f3_k2_size_and_noncommutative():
    table, spec = build_mul_table("lam_f3_k2")
    assert spec.n_elements == 81
    assert table.shape == (81, 81)
    # e1 * e2 = e12, e2 * e1 = -e12 in F3
    e1 = _coeff_encode(np.array([0, 1, 0, 0]), 3)
    e2 = _coeff_encode(np.array([0, 0, 1, 0]), 3)
    assert table[e1, e2] != table[e2, e1]


def test_lam_f2_k3_trunc_size():
    table, spec = build_mul_table("lam_f2_k3t")
    assert spec.n_elements == 128
    assert table.shape == (128, 128)
    # char 2: wedge signs collapse → multiplication table is commutative
    assert spec.commutative is True
    assert np.array_equal(table, table.T)
    # e1*e2*e3 = 0 by truncation
    e1 = _coeff_encode(np.array([0, 1, 0, 0, 0, 0, 0]), 2)
    e2 = _coeff_encode(np.array([0, 0, 1, 0, 0, 0, 0]), 2)
    e3 = _coeff_encode(np.array([0, 0, 0, 1, 0, 0, 0]), 2)
    e12 = table[e1, e2]
    assert table[e12, e3] == 0


def test_f2_group_rings_sizes():
    for ring_id, n in (("f2_s3", 64), ("f2_d4", 256), ("f2_z6", 64), ("f2_z7", 128)):
        table, spec = build_mul_table(ring_id)
        assert spec.n_elements == n
        assert table.shape == (n, n)
        assert table[1, 1] == 1  # multiplicative identity


def test_f2_z6_commutative():
    table, spec = build_mul_table("f2_z6")
    assert spec.commutative
    assert np.array_equal(table, table.T)


def test_f2_z7_commutative():
    table, spec = build_mul_table("f2_z7")
    assert spec.commutative
    assert np.array_equal(table, table.T)


def test_f3_s3_group_ring_size_and_noncommutative():
    table, spec = build_mul_table("f3_s3")
    assert spec.n_elements == 729
    assert table.shape == (729, 729)
    assert not np.array_equal(table, table.T)


def test_lam_f3_k2_wedderburn_spec():
    spec = get_wedderburn_spec("lam_f3_k2")
    assert "max_grade" in spec.component_names
    assert spec.n_elements == 81


def test_lam_wm_wedderburn_malcev_labels():
    spec = get_wedderburn_spec("lam_f3_k2", label_scheme=LAM_WM_SCHEME)
    assert spec.component_names == ("lam_field", "lam_rad")
    assert spec.labels["lam_field"].shape == (81,)
    assert spec.labels["lam_rad"].shape == (81, 3)
    assert abs(spec.chance["lam_field"] - 1 / 3) < 1e-9
    assert abs(spec.chance["lam_rad"] - 1 / 27) < 1e-9
    sigs = labels_to_signatures(spec)
    rng = np.random.default_rng(0)
    n_field = sample_matched_pairs(
        81, sigs, spec.component_names, "lam_field", max_pairs=50, rng=rng
    )
    n_rad = sample_matched_pairs(
        81, sigs, spec.component_names, "lam_rad", max_pairs=50, rng=rng
    )
    assert len(n_field) > 0
    assert len(n_rad) > 0


def test_lam_basis_labels_independent_for_iia():
    """Grade buckets are redundant; basis-coeff labels admit matched IIA pairs."""
    grade = get_wedderburn_spec("lam_f2_k3t")
    basis = get_wedderburn_spec("lam_f2_k3t", label_scheme=LAM_BASIS_SCHEME)
    assert basis.component_names == (
        "lam_1",
        "lam_e0",
        "lam_e1",
        "lam_e2",
        "lam_e01",
        "lam_e02",
        "lam_e12",
    )
    sigs_grade = labels_to_signatures(grade)
    sigs_basis = labels_to_signatures(basis)
    rng = np.random.default_rng(0)
    n_grade = sample_matched_pairs(
        128, sigs_grade, grade.component_names, "grade_1", max_pairs=50, rng=rng
    )
    n_basis = sample_matched_pairs(
        128, sigs_basis, basis.component_names, "lam_e1", max_pairs=50, rng=rng
    )
    assert len(n_grade) == 0
    assert len(n_basis) > 0


def test_ext_monoid_is_not_full_algebra():
    """Legacy ext_f3_k6: |R|=128 basis monoid, not 3^(2^6)."""
    _, spec = build_mul_table("ext_f3_k6")
    assert spec.n_elements == 128
    assert spec.n_elements != 3 ** (2**6)


def test_identity_lam_f3_k2():
    table, _ = build_mul_table("lam_f3_k2")
    e = lam_f3_k2_identity_index()
    for i in range(81):
        assert table[e, i] == i
        assert table[i, e] == i
