"""Tests for finite ring multiplication tables."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from gf_grokking.alt_group_data import create_group_transformer_loaders
from gf_grokking.finite_rings import (
    RING_REGISTRY,
    assert_associativity,
    build_mul_table,
    ext_identity_index,
    mat2_f3_identity_index,
    quat_f3_identity_index,
    tri2_identity_index,
)
from gf_grokking.iia import probe_subspace_basis_from_vector_labels
from gf_grokking.ring_transformer_iia import min_subspace_dim_for_probe
from gf_grokking.ring_wedderburn import get_wedderburn_spec, labels_to_signatures, ring_identity_index
from gf_grokking.models.group_transformer import GroupTransformer


def test_ring_registry_sizes():
    expected = {
        "mat2_f2": 16,
        "mat2_f3": 81,
        "quat_f3": 81,
        "tri2_f2": 8,
        "tri2_f3": 27,
        "tri3_f2": 64,
        "f3": 3,
        "f3_x_f3": 9,
        "mat2_f3_x_f3": 243,
        "quat_f3_x_f3": 243,
        "tri2_f3_x_f3": 81,
        "tri2_f3_x_f3_x_f3": 243,
        "mat2_f2_x_mat2_f2": 256,
        "f2_x7": 128,
        "ext_f2_k6": 65,
        "ext_f2_k8": 257,
        "ext_f3_k6": 128,
        "ext_f3_k7": 256,
    }
    for ring_id, n in expected.items():
        table, spec = build_mul_table(ring_id)
        assert ring_id in RING_REGISTRY
        assert table.shape == (n, n)
        assert spec.n_elements == n
        if ring_id not in ("f3", "f3_x_f3", "f2_x7", "ext_f2_k6", "ext_f2_k8"):
            assert spec.commutative is False


def test_associativity_all_rings():
    for ring_id in sorted(RING_REGISTRY):
        table, _ = build_mul_table(ring_id)
        assert_associativity(table)


def test_min_subspace_dim_for_vector_probe():
    labels = np.arange(16 * 4, dtype=np.int64).reshape(16, 4)
    assert min_subspace_dim_for_probe(labels) == 4
    assert min_subspace_dim_for_probe(np.array([0, 1, 0, 1])) == 1
    assert min_subspace_dim_for_probe(np.arange(3)) == 2
    assert min_subspace_dim_for_probe(np.arange(16)) == 2
    spec = get_wedderburn_spec("mat2_f2_x_mat2_f2", label_scheme="coords_joint")
    assert spec.component_names == ("L_mat2", "R_mat2")
    assert spec.labels["L_mat2"].shape == (256, 4)
    sigs = labels_to_signatures(spec)
    assert sigs["L_mat2"].shape == (256, 4)


def test_probe_subspace_basis_from_vector_labels():
    rng = np.random.default_rng(2)
    n, d, h = 64, 4, 32
    labels = rng.integers(0, 2, size=(n, d))
    hidden = rng.standard_normal((n, h))
    for j in range(d):
        hidden[:, j * 4 : (j + 1) * 4] += labels[:, j : j + 1] * 2.0
    result = probe_subspace_basis_from_vector_labels(
        hidden, labels, min_subspace_dim=2
    )
    assert result.k >= 2
    assert result.n_classes == 4
    assert result.probe_accuracy > 0.5
    assert result.basis.shape == (result.k, h)


def test_mat2_f3_identity():
    table, _ = build_mul_table("mat2_f3")
    e = mat2_f3_identity_index()
    n = table.shape[0]
    for i in range(n):
        assert table[e, i] == i
        assert table[i, e] == i


def test_quat_f3_identity_and_non_commutative():
    table, spec = build_mul_table("quat_f3")
    assert spec.n_elements == 81
    assert spec.ring_type == "quaternion"
    e = quat_f3_identity_index()
    n = table.shape[0]
    for i in range(n):
        assert table[e, i] == i
        assert table[i, e] == i
    assert ring_identity_index("quat_f3", table) == e
    found = any(table[i, j] != table[j, i] for i in range(n) for j in range(i + 1, n))
    assert found


def test_quat_f3_wedderburn_spec():
    spec = get_wedderburn_spec("quat_f3")
    assert spec.n_elements == 81
    assert spec.component_names == ("a", "b", "c", "d")
    wb_coords = get_wedderburn_spec("quat_f3", label_scheme="coords")
    assert wb_coords.component_names == ("a", "b", "c", "d")


@pytest.mark.parametrize("ring_id,prime", [("tri2_f2", 2), ("tri2_f3", 3)])
def test_tri2_identity(ring_id: str, prime: int):
    table, spec = build_mul_table(ring_id)
    n = table.shape[0]
    assert n == prime**3
    e = tri2_identity_index(prime)
    assert e < n
    assert table[e, :].tolist() == list(range(n))
    assert table[:, e].tolist() == list(range(n))
    assert ring_identity_index(ring_id, table) == e
    assert spec.encoding_version == 2


def test_tri2_f3_not_commutative():
    table, _ = build_mul_table("tri2_f3")
    found = False
    n = table.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if table[i, j] != table[j, i]:
                found = True
                break
        if found:
            break
    assert found


def test_ext_f2_k6_char2_commutative_limit():
    """In char 2, e_i∧e_j = -e_j∧e_i = e_j∧e_i → commutative on monomials."""
    table, spec = build_mul_table("ext_f2_k6")
    assert spec.commutative
    e = ext_identity_index()
    assert ring_identity_index("ext_f2_k6", table) == e
    e0, e1 = 2, 3  # masks 1 and 2
    assert table[e0, e0] == 0
    assert table[e0, e1] == table[e1, e0] == 4


def test_ext_f3_k6_anticommutative():
    table, spec = build_mul_table("ext_f3_k6")
    assert not spec.commutative
    assert spec.n_elements == 128
    e0, e1 = 2, 4  # +e_0, +e_1
    assert table[e0, e0] == 0
    assert table[e0, e1] != table[e1, e0]
    assert table[e0, e1] == 7  # -e_{01}
    assert table[e1, e0] == 6  # +e_{01}
    wb = get_wedderburn_spec("ext_f3_k6")
    assert "sign" in wb.component_names


def test_wedderburn_quat_f3_x_f3_product_spec():
    spec = get_wedderburn_spec("quat_f3_x_f3")
    assert spec.n_elements == 243
    assert len(spec.component_names) == 5  # L_a..L_d + R_field


def test_product_ring_identity():
    table, _ = build_mul_table("mat2_f3_x_f3")
    n = table.shape[0]
    e = None
    for i in range(n):
        if np.array_equal(table[i], np.arange(n)) and np.array_equal(table[:, i], np.arange(n)):
            e = i
            break
    assert e is not None


def test_wedderburn_product_spec():
    spec = get_wedderburn_spec("mat2_f2_x_mat2_f2")
    assert spec.n_elements == 256
    assert len(spec.component_names) == 6  # 3 + 3 from two M_2(F_2)


def test_ring_transformer_loader_smoke():
    ring_id = "tri2_f3"
    table, spec = build_mul_table(ring_id)
    train_loader, test_loader, mult_table, n = create_group_transformer_loaders(
        train_size=100,
        test_size=50,
        batch_size=32,
        seed=42,
        ring_id=ring_id,
        num_workers=0,
    )
    assert n == spec.n_elements
    assert mult_table.shape[0] == n
    tokens, labels = next(iter(train_loader))
    assert tokens.shape[1] == 2
    assert labels.max() < n

    model = GroupTransformer(num_elements=n, layernorm=False)
    logits = model(tokens)
    assert logits.shape == (tokens.shape[0], n)
