"""Wedderburn-компоненты конечных колец для IIA (Zheng et al.)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .finite_rings import (
    RING_REGISTRY,
    _mat2_decode,
    _quat_f3_decode,
    _tri2_decode,
    _tri3_f2_coords,
    build_mul_table,
    get_ring_spec,
)
from .iia import identity_element


@dataclass(frozen=True)
class WedderburnSpec:
    ring_id: str
    component_names: tuple[str, ...]
    labels: dict[str, np.ndarray]  # name -> [n] int labels per element
    chance: dict[str, float]  # 1 / |im pi_k|

    @property
    def n_elements(self) -> int:
        return int(next(iter(self.labels.values())).shape[0])


LabelScheme = str  # "scalar" | "coords" | "coords_joint" | "lam_basis"

SCALAR_SCHEME: LabelScheme = "scalar"
COORDS_SCHEME: LabelScheme = "coords"
COORDS_JOINT_SCHEME: LabelScheme = "coords_joint"
LAM_BASIS_SCHEME: LabelScheme = "lam_basis"
LAM_WM_SCHEME: LabelScheme = "lam_wm"


def _chance_from_labels(arr: np.ndarray) -> float:
    u = np.unique(arr)
    return 1.0 / len(u) if len(u) > 0 else 0.0


def _chance_from_label_array(arr: np.ndarray) -> float:
    if arr.ndim == 1:
        return _chance_from_labels(arr)
    rows = np.unique(arr, axis=0)
    return 1.0 / len(rows) if len(rows) > 0 else 0.0


def _mat2_coord_label_dict(n: int, p: int) -> dict[str, np.ndarray]:
    """Per-entry coordinates (a,b,c,d) for Mat_2(F_p) element index."""
    out = {k: np.zeros(n, dtype=np.int64) for k in ("a", "b", "c", "d")}
    for g in range(n):
        a, b, c, d = _mat2_decode(g, p)
        out["a"][g] = a
        out["b"][g] = b
        out["c"][g] = c
        out["d"][g] = d
    return out


def _mat2_joint_label_array(n: int, p: int) -> np.ndarray:
    """Stack (a,b,c,d) as [n, 4] for multi-output probes."""
    coords = _mat2_coord_label_dict(n, p)
    return np.column_stack([coords[k] for k in ("a", "b", "c", "d")])


def ring_identity_index(ring_id: str, mult_table: np.ndarray) -> int:
    return identity_element(mult_table)


def wedderburn_spec_tri3_f2() -> WedderburnSpec:
    """T_3(F_2): A/rad ≅ F_2^3 (диагональ), rad — строго верхнетреуг. (b,c,e)."""
    n = 64
    diag1 = np.zeros(n, dtype=np.int64)
    diag2 = np.zeros(n, dtype=np.int64)
    diag3 = np.zeros(n, dtype=np.int64)
    rad = np.zeros(n, dtype=np.int64)
    for g in range(n):
        a, b, c, d, e, f = _tri3_f2_coords(g)
        diag1[g] = a
        diag2[g] = d
        diag3[g] = f
        rad[g] = b + 2 * c + 4 * e
    labels = {"diag1": diag1, "diag2": diag2, "diag3": diag3, "rad": rad}
    chance = {k: _chance_from_labels(v) for k, v in labels.items()}
    return WedderburnSpec(
        ring_id="tri3_f2",
        component_names=("diag1", "diag2", "diag3", "rad"),
        labels=labels,
        chance=chance,
    )


def wedderburn_spec_tri2_f3() -> WedderburnSpec:
    """T_2(F_3): диагональ (a,d) ∈ F_3^2, rad — координата b."""
    n = 27
    p = 3
    diag1 = np.zeros(n, dtype=np.int64)
    diag2 = np.zeros(n, dtype=np.int64)
    rad = np.zeros(n, dtype=np.int64)
    for g in range(n):
        a, b, d = _tri2_decode(g, p)
        diag1[g] = a
        diag2[g] = d
        rad[g] = b
    labels = {"diag1": diag1, "diag2": diag2, "rad": rad}
    chance = {k: _chance_from_labels(v) for k, v in labels.items()}
    return WedderburnSpec(
        ring_id="tri2_f3",
        component_names=("diag1", "diag2", "rad"),
        labels=labels,
        chance=chance,
    )


def _quat_f3_coord_label_dict(n: int) -> dict[str, np.ndarray]:
    out = {k: np.zeros(n, dtype=np.int64) for k in ("a", "b", "c", "d")}
    for g in range(n):
        a, b, c, d = _quat_f3_decode(g)
        out["a"][g] = a
        out["b"][g] = b
        out["c"][g] = c
        out["d"][g] = d
    return out


def _quat_f3_joint_label_array(n: int) -> np.ndarray:
    coords = _quat_f3_coord_label_dict(n)
    return np.column_stack([coords[k] for k in ("a", "b", "c", "d")])


def wedderburn_spec_quat_f3() -> WedderburnSpec:
    """H(F_3): coordinate components (a,b,c,d) for IIA probes."""
    n = 81
    coords = _quat_f3_coord_label_dict(n)
    labels = {k: coords[k] for k in ("a", "b", "c", "d")}
    chance = {k: _chance_from_labels(v) for k, v in labels.items()}
    return WedderburnSpec(
        ring_id="quat_f3",
        component_names=("a", "b", "c", "d"),
        labels=labels,
        chance=chance,
    )


def wedderburn_spec_quat_f3_coords(*, joint: bool = False) -> WedderburnSpec:
    n = 81
    if joint:
        labels = {"quat": _quat_f3_joint_label_array(n)}
        names = ("quat",)
    else:
        coords = _quat_f3_coord_label_dict(n)
        labels = {k: coords[k] for k in ("a", "b", "c", "d")}
        names = ("a", "b", "c", "d")
    chance = {k: _chance_from_label_array(v) for k, v in labels.items()}
    return WedderburnSpec(
        ring_id="quat_f3",
        component_names=names,
        labels=labels,
        chance=chance,
    )


def wedderburn_spec_mat2_f3() -> WedderburnSpec:
    """Mat_2(F_3): semisimple; блок-факторы по диагонали (a,d) и off-diag trace slot c."""
    n = 81
    p = 3
    diag1 = np.zeros(n, dtype=np.int64)
    diag2 = np.zeros(n, dtype=np.int64)
    offdiag = np.zeros(n, dtype=np.int64)
    for g in range(n):
        a, b, c, d = _mat2_decode(g, p)
        diag1[g] = a
        diag2[g] = d
        offdiag[g] = b + 3 * c
    labels = {"diag1": diag1, "diag2": diag2, "offdiag": offdiag}
    chance = {k: _chance_from_labels(v) for k, v in labels.items()}
    return WedderburnSpec(
        ring_id="mat2_f3",
        component_names=("diag1", "diag2", "offdiag"),
        labels=labels,
        chance=chance,
    )


def wedderburn_spec_tri2_f2() -> WedderburnSpec:
    n = 8
    p = 2
    diag1 = np.zeros(n, dtype=np.int64)
    diag2 = np.zeros(n, dtype=np.int64)
    rad = np.zeros(n, dtype=np.int64)
    for g in range(n):
        a, b, d = _tri2_decode(g, p)
        diag1[g] = a
        diag2[g] = d
        rad[g] = b
    labels = {"diag1": diag1, "diag2": diag2, "rad": rad}
    chance = {k: _chance_from_labels(v) for k, v in labels.items()}
    return WedderburnSpec(
        ring_id="tri2_f2",
        component_names=("diag1", "diag2", "rad"),
        labels=labels,
        chance=chance,
    )


def wedderburn_spec_mat2_f2() -> WedderburnSpec:
    n = 16
    p = 2
    diag1 = np.zeros(n, dtype=np.int64)
    diag2 = np.zeros(n, dtype=np.int64)
    offdiag = np.zeros(n, dtype=np.int64)
    for g in range(n):
        a, b, c, d = _mat2_decode(g, p)
        diag1[g] = a
        diag2[g] = d
        offdiag[g] = b + 2 * c
    labels = {"diag1": diag1, "diag2": diag2, "offdiag": offdiag}
    chance = {k: _chance_from_labels(v) for k, v in labels.items()}
    return WedderburnSpec(
        ring_id="mat2_f2",
        component_names=("diag1", "diag2", "offdiag"),
        labels=labels,
        chance=chance,
    )


def wedderburn_spec_f3() -> WedderburnSpec:
    n = 3
    field = np.arange(n, dtype=np.int64)
    labels = {"field": field}
    chance = {"field": _chance_from_labels(field)}
    return WedderburnSpec(ring_id="f3", component_names=("field",), labels=labels, chance=chance)


def _wedderburn_product(left: str, right: str) -> WedderburnSpec:
    sa = get_wedderburn_spec(left)
    sb = get_wedderburn_spec(right)
    nb = get_ring_spec(right).n_elements
    n = sa.n_elements * nb
    labels: dict[str, np.ndarray] = {}
    names: list[str] = []
    for name in sa.component_names:
        key = f"L_{name}"
        names.append(key)
        arr = np.zeros(n, dtype=np.int64)
        for idx in range(n):
            ia = idx // nb
            arr[idx] = sa.labels[name][ia]
        labels[key] = arr
    for name in sb.component_names:
        key = f"R_{name}"
        names.append(key)
        arr = np.zeros(n, dtype=np.int64)
        for idx in range(n):
            ib = idx % nb
            arr[idx] = sb.labels[name][ib]
        labels[key] = arr
    chance = {k: _chance_from_labels(v) for k, v in labels.items()}
    ring_id = f"{left}_x_{right}"
    return WedderburnSpec(ring_id=ring_id, component_names=tuple(names), labels=labels, chance=chance)


def wedderburn_spec_f3_z4() -> WedderburnSpec:
    """F₃[ℤ₄] ≅ F₃×F₃×F₉: reuse CRT factor labels (f3_x_f3_x_f9)."""
    return _wedderburn_product("f3_x_f3", "f9")


def wedderburn_spec_f8() -> WedderburnSpec:
    n = 8
    field = np.arange(n, dtype=np.int64)
    labels = {"field": field}
    chance = {"field": _chance_from_labels(field)}
    return WedderburnSpec(ring_id="f8", component_names=("field",), labels=labels, chance=chance)


def wedderburn_spec_f9() -> WedderburnSpec:
    n = 9
    field = np.arange(n, dtype=np.int64)
    labels = {"field": field}
    chance = {"field": _chance_from_labels(field)}
    return WedderburnSpec(ring_id="f9", component_names=("field",), labels=labels, chance=chance)


def wedderburn_spec_f2_z6() -> WedderburnSpec:
    """F₂[ℤ₆]: Z₂ (radical) and ℤ₃ (semisimple) CRT factors — Zheng Table 10."""
    from .true_algebras import _coeff_decode

    n = 64
    rad = np.zeros(n, dtype=np.int64)
    ss = np.zeros(n, dtype=np.int64)
    for elem in range(n):
        coeffs = _coeff_decode(elem, 6, 2)
        rad[elem] = int(coeffs[0] + 2 * coeffs[3])
        ss[elem] = int(coeffs[0] + 2 * coeffs[1] + 4 * coeffs[2])
    labels = {"rad": rad, "ss": ss}
    chance = {k: _chance_from_labels(v) for k, v in labels.items()}
    return WedderburnSpec(ring_id="f2_z6", component_names=("rad", "ss"), labels=labels, chance=chance)


def wedderburn_spec_f2_x7() -> WedderburnSpec:
    """F_2[x]/(x^7): coefficient-bit pseudo-labels ONLY.

    NOT Wedderburn factors — convolution couples coefficients under multiply.
    Raw IIA on these labels is misleading; use random-subspace or skip raw IIA.
    See new_paper_rings/REVIEW_GAPS.md §1.
    """
    n = 128
    labels: dict[str, np.ndarray] = {}
    for k in range(7):
        arr = np.zeros(n, dtype=np.int64)
        for g in range(n):
            arr[g] = (g >> k) & 1
        labels[f"c{k}"] = arr
    chance = {k: _chance_from_labels(v) for k, v in labels.items()}
    return WedderburnSpec(
        ring_id="f2_x7",
        component_names=tuple(f"c{k}" for k in range(7)),
        labels=labels,
        chance=chance,
    )


def wedderburn_spec_mat2_f2_coords(*, joint: bool = False) -> WedderburnSpec:
    n, p = 16, 2
    if joint:
        labels = {"mat2": _mat2_joint_label_array(n, p)}
        names = ("mat2",)
    else:
        coords = _mat2_coord_label_dict(n, p)
        labels = {k: coords[k] for k in ("a", "b", "c", "d")}
        names = ("a", "b", "c", "d")
    chance = {k: _chance_from_label_array(v) for k, v in labels.items()}
    return WedderburnSpec(
        ring_id="mat2_f2",
        component_names=names,
        labels=labels,
        chance=chance,
    )


def wedderburn_spec_mat2_f3_coords(*, joint: bool = False) -> WedderburnSpec:
    n, p = 81, 3
    if joint:
        labels = {"mat2": _mat2_joint_label_array(n, p)}
        names = ("mat2",)
    else:
        coords = _mat2_coord_label_dict(n, p)
        labels = {k: coords[k] for k in ("a", "b", "c", "d")}
        names = ("a", "b", "c", "d")
    chance = {k: _chance_from_label_array(v) for k, v in labels.items()}
    return WedderburnSpec(
        ring_id="mat2_f3",
        component_names=names,
        labels=labels,
        chance=chance,
    )


def _wedderburn_product_coords(left: str, right: str, *, joint: bool) -> WedderburnSpec:
    sa = get_wedderburn_spec(left, label_scheme=COORDS_SCHEME if not joint else COORDS_JOINT_SCHEME)
    sb = get_wedderburn_spec(right, label_scheme=COORDS_SCHEME if not joint else COORDS_JOINT_SCHEME)
    nb = get_ring_spec(right).n_elements
    n = sa.n_elements * nb
    labels: dict[str, np.ndarray] = {}
    names: list[str] = []

    def _lift_left(name: str, arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 1:
            out = np.zeros(n, dtype=np.int64)
            for idx in range(n):
                out[idx] = arr[idx // nb]
            return out
        out = np.zeros((n, arr.shape[1]), dtype=np.int64)
        for idx in range(n):
            out[idx] = arr[idx // nb]
        return out

    def _lift_right(name: str, arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 1:
            out = np.zeros(n, dtype=np.int64)
            for idx in range(n):
                out[idx] = arr[idx % nb]
            return out
        out = np.zeros((n, arr.shape[1]), dtype=np.int64)
        for idx in range(n):
            out[idx] = arr[idx % nb]
        return out

    for name in sa.component_names:
        key = f"L_{name}"
        names.append(key)
        labels[key] = _lift_left(name, sa.labels[name])
    for name in sb.component_names:
        key = f"R_{name}"
        names.append(key)
        labels[key] = _lift_right(name, sb.labels[name])
    chance = {k: _chance_from_label_array(v) for k, v in labels.items()}
    ring_id = f"{left}_x_{right}"
    return WedderburnSpec(ring_id=ring_id, component_names=tuple(names), labels=labels, chance=chance)


def wedderburn_spec_ext(k: int, p: int) -> WedderburnSpec:
    """Λ(F_p^k) basis *monoid* (legacy); not the full exterior algebra."""
    from .finite_rings import _ext_decode, _ext_n_elements

    ring_id = f"ext_f{p}_k{k}"
    n = _ext_n_elements(k, p)
    degree = np.zeros(n, dtype=np.int64)
    labels: dict[str, np.ndarray] = {}
    names: list[str] = []
    for g in range(n):
        mask, sign = _ext_decode(g, p)
        if mask is None:
            degree[g] = -1
        else:
            degree[g] = mask.bit_count()
    labels["degree"] = degree
    names.append("degree")
    if p > 2:
        sign_lbl = np.zeros(n, dtype=np.int64)
        for g in range(n):
            mask, sign = _ext_decode(g, p)
            if mask is not None and mask > 0:
                sign_lbl[g] = 1 if sign > 0 else 2
        labels["sign"] = sign_lbl
        names.append("sign")
    for r in range(k + 1):
        key = f"grade_{r}"
        arr = np.zeros(n, dtype=np.int64)
        for g in range(n):
            if degree[g] == r:
                arr[g] = g
        labels[key] = arr
        names.append(key)
    chance = {name: _chance_from_labels(labels[name]) for name in names}
    return WedderburnSpec(ring_id=ring_id, component_names=tuple(names), labels=labels, chance=chance)


def _lam_radical_joint_label_array(
    n: int, k: int, p: int, *, max_degree: int | None = None
) -> np.ndarray:
    """Joint radical coefficients (wedge grades >= 1) as [n, n_rad] int labels."""
    from .true_algebras import _coeff_decode, _lam_basis_masks

    masks = _lam_basis_masks(k, max_degree=max_degree)
    n_basis = len(masks)
    rad_indices = [bi for bi, mask in enumerate(masks) if mask.bit_count() >= 1]
    ncol = len(rad_indices)
    out = np.zeros((n, ncol), dtype=np.int64)
    for g in range(n):
        coeffs = _coeff_decode(g, n_basis, p)
        for j, bi in enumerate(rad_indices):
            out[g, j] = coeffs[bi]
    return out


def wedderburn_spec_lam_wedderburn_malcev(
    ring_id: str, k: int, p: int, *, max_degree: int | None = None
) -> WedderburnSpec:
    """Exterior algebra: semisimple scalar (grade 0) + single radical joint block.

    Grades 1 and 2 are not independent Wedderburn factors ($e_1 e_2 = e_{12}$);
    probe/IIA on the radical uses one joint coefficient vector.
    """
    from .true_algebras import _coeff_decode, _lam_basis_masks

    masks = _lam_basis_masks(k, max_degree=max_degree)
    n_basis = len(masks)
    n = p**n_basis
    field = np.zeros(n, dtype=np.int64)
    for g in range(n):
        field[g] = _coeff_decode(g, n_basis, p)[0]
    rad = _lam_radical_joint_label_array(n, k, p, max_degree=max_degree)
    labels: dict[str, np.ndarray] = {"lam_field": field, "lam_rad": rad}
    names = ["lam_field", "lam_rad"]
    chance = {
        "lam_field": _chance_from_labels(field),
        "lam_rad": _chance_from_label_array(rad),
    }
    return WedderburnSpec(ring_id=ring_id, component_names=tuple(names), labels=labels, chance=chance)


def _lam_basis_component_name(mask: int) -> str:
    """Human-readable name for a Λ(F_p^k) basis monomial (bitmask of generators)."""
    if mask == 0:
        return "lam_1"
    gens = "".join(str(i) for i in range(mask.bit_length()) if mask & (1 << i))
    return f"lam_e{gens}"


def wedderburn_spec_lam_basis(ring_id: str, k: int, p: int, *, max_degree: int | None = None) -> WedderburnSpec:
    """Exterior algebra: one component per basis-monomial coefficient in F_p.

    Independent labels (unlike grade buckets) — suitable for matched-pair IIA while
  retaining the algebra element as a coefficient vector in the fixed wedge basis.
    """
    from .true_algebras import _coeff_decode, _lam_basis_masks

    masks = _lam_basis_masks(k, max_degree=max_degree)
    n_basis = len(masks)
    n = p**n_basis
    labels: dict[str, np.ndarray] = {}
    names: list[str] = []
    for bi, mask in enumerate(masks):
        key = _lam_basis_component_name(mask)
        arr = np.zeros(n, dtype=np.int64)
        for g in range(n):
            arr[g] = _coeff_decode(g, n_basis, p)[bi]
        labels[key] = arr
        names.append(key)
    chance = {name: _chance_from_labels(labels[name]) for name in names}
    return WedderburnSpec(ring_id=ring_id, component_names=tuple(names), labels=labels, chance=chance)


def wedderburn_spec_lam_full(ring_id: str, k: int, p: int, *, max_degree: int | None = None) -> WedderburnSpec:
    """Full/truncated exterior algebra: max wedge grade + per-grade buckets."""
    from .true_algebras import _coeff_decode, _lam_basis_masks

    masks = _lam_basis_masks(k, max_degree=max_degree)
    n_basis = len(masks)
    n = p**n_basis
    deg_of_basis = [masks[i].bit_count() for i in range(n_basis)]
    max_deg = max_degree if max_degree is not None else k

    max_grade = np.zeros(n, dtype=np.int64)
    for g in range(n):
        coeffs = _coeff_decode(g, n_basis, p)
        active = [deg_of_basis[bi] for bi in range(n_basis) if coeffs[bi] != 0]
        max_grade[g] = max(active) if active else -1

    labels: dict[str, np.ndarray] = {"max_grade": max_grade}
    names: list[str] = ["max_grade"]
    for r in range(max_deg + 1):
        key = f"grade_{r}"
        arr = np.zeros(n, dtype=np.int64)
        for g in range(n):
            arr[g] = 1 if max_grade[g] == r else 0
        labels[key] = arr
        names.append(key)
    chance = {name: _chance_from_labels(labels[name]) for name in names}
    return WedderburnSpec(ring_id=ring_id, component_names=tuple(names), labels=labels, chance=chance)


def wedderburn_spec_fp_group_ring(ring_id: str, n_group: int, p: int) -> WedderburnSpec:
    """F_p[G]: scalar label per group-basis coefficient."""
    from .true_algebras import _coeff_decode

    n = p**n_group
    labels: dict[str, np.ndarray] = {}
    names: list[str] = []
    for gi in range(n_group):
        key = f"g{gi}"
        arr = np.zeros(n, dtype=np.int64)
        for elem in range(n):
            arr[elem] = _coeff_decode(elem, n_group, p)[gi]
        labels[key] = arr
        names.append(key)
    chance = {name: _chance_from_labels(labels[name]) for name in names}
    return WedderburnSpec(ring_id=ring_id, component_names=tuple(names), labels=labels, chance=chance)


def get_wedderburn_spec(ring_id: str, *, label_scheme: LabelScheme = SCALAR_SCHEME) -> WedderburnSpec:
    if label_scheme not in (
        SCALAR_SCHEME,
        COORDS_SCHEME,
        COORDS_JOINT_SCHEME,
        LAM_BASIS_SCHEME,
        LAM_WM_SCHEME,
    ):
        raise ValueError(f"unknown label_scheme {label_scheme!r}")

    if label_scheme == LAM_BASIS_SCHEME:
        if ring_id == "lam_f3_k2":
            return wedderburn_spec_lam_basis(ring_id, 2, 3)
        if ring_id == "lam_f2_k3t":
            return wedderburn_spec_lam_basis(ring_id, 3, 2, max_degree=2)
        raise ValueError(
            f"lam_basis labels not defined for {ring_id!r}; supported: lam_f3_k2, lam_f2_k3t"
        )

    if label_scheme == LAM_WM_SCHEME:
        if ring_id == "lam_f3_k2":
            return wedderburn_spec_lam_wedderburn_malcev(ring_id, 2, 3)
        if ring_id == "lam_f2_k3t":
            return wedderburn_spec_lam_wedderburn_malcev(ring_id, 3, 2, max_degree=2)
        raise ValueError(
            f"lam_wm labels not defined for {ring_id!r}; supported: lam_f3_k2, lam_f2_k3t"
        )

    if label_scheme in (COORDS_SCHEME, COORDS_JOINT_SCHEME):
        joint = label_scheme == COORDS_JOINT_SCHEME
        if "_x_" in ring_id:
            left, right = ring_id.split("_x_", 1)
            return _wedderburn_product_coords(left, right, joint=joint)
        if ring_id == "mat2_f2":
            return wedderburn_spec_mat2_f2_coords(joint=joint)
        if ring_id == "mat2_f3":
            return wedderburn_spec_mat2_f3_coords(joint=joint)
        if ring_id == "quat_f3":
            return wedderburn_spec_quat_f3_coords(joint=joint)
        raise ValueError(
            f"coordinate labels not defined for {ring_id!r}; supported: mat2_f2, mat2_f3, quat_f3, their products"
        )

    if "_x_" in ring_id:
        left, right = ring_id.split("_x_", 1)
        return _wedderburn_product(left, right)
    if ring_id == "tri3_f2":
        return wedderburn_spec_tri3_f2()
    if ring_id == "tri2_f3":
        return wedderburn_spec_tri2_f3()
    if ring_id == "tri2_f2":
        return wedderburn_spec_tri2_f2()
    if ring_id == "mat2_f3":
        return wedderburn_spec_mat2_f3()
    if ring_id == "quat_f3":
        return wedderburn_spec_quat_f3()
    if ring_id == "mat2_f2":
        return wedderburn_spec_mat2_f2()
    if ring_id == "f3":
        return wedderburn_spec_f3()
    if ring_id == "f2_x7":
        return wedderburn_spec_f2_x7()
    if ring_id.startswith("ext_f"):
        from .finite_rings import _ext_parse_ring_id

        p, k = _ext_parse_ring_id(ring_id)
        return wedderburn_spec_ext(k, p)
    if ring_id == "lam_f3_k2":
        return wedderburn_spec_lam_full(ring_id, 2, 3)
    if ring_id == "lam_f2_k3t":
        return wedderburn_spec_lam_full(ring_id, 3, 2, max_degree=2)
    if ring_id == "f2_s3":
        return wedderburn_spec_fp_group_ring(ring_id, 6, 2)
    if ring_id == "f2_d4":
        return wedderburn_spec_fp_group_ring(ring_id, 8, 2)
    if ring_id == "f3_s3":
        return wedderburn_spec_fp_group_ring(ring_id, 6, 3)
    if ring_id == "f2_z6":
        return wedderburn_spec_f2_z6()
    if ring_id == "f2_z7":
        return wedderburn_spec_fp_group_ring(ring_id, 7, 2)
    if ring_id == "f8":
        return wedderburn_spec_f8()
    if ring_id == "f9":
        return wedderburn_spec_f9()
    if ring_id == "f3_z4":
        return wedderburn_spec_f3_z4()
    raise ValueError(f"no Wedderburn spec for {ring_id!r}")


def labels_to_signatures(spec: WedderburnSpec) -> dict[str, np.ndarray]:
    sigs: dict[str, np.ndarray] = {}
    for name in spec.component_names:
        arr = spec.labels[name]
        if arr.ndim == 1:
            sigs[name] = arr.astype(np.float64).reshape(-1, 1)
        else:
            sigs[name] = arr.astype(np.float64)
    return sigs
