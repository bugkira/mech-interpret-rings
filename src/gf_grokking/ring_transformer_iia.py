"""IIA для GroupTransformer на конечных кольцах (Wedderburn-компоненты)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import torch

from .alternating_group import split_pair_indices

from .finite_rings import build_mul_table, get_ring_spec
from .iia import (
    split_ring_elements,
    IIAMetrics,
    IIAResult,
    _apply_subspace_patch,
    counterfactual_logit_diff,
    logit_diff_restoration,
    mlp_bottleneck_activation,
    nonlinear_bottleneck_patch,
    probe_subspace_basis_from_class_means,
    probe_subspace_basis_from_labels,
    probe_subspace_basis_from_vector_labels,
    random_orthonormal_basis,
    sample_matched_pairs,
)
from .iia_normalized import normalized_subspace_gain
from .models.group_transformer import GroupTransformer
from .ring_wedderburn import (
    WedderburnSpec,
    get_wedderburn_spec,
    labels_to_signatures,
    ring_identity_index,
)
from .transformer_iia import (
    TransformerPatchSite,
    evaluate_transformer_accuracy,
    forward_group_transformer_with_patch,
    hidden_at_site,
)

# Cap on scalar k_min from n_classes−1 (F₂ binary → 1; F₃ → 2).
SCALAR_K_MIN_CAP = 2
PairSplit = Literal[
    "full",
    "test",
    "test_strict",
    "test_elem",
    "test_elem_strict",
]
ProbeFitMode = Literal["all", "train_mult_elements", "element_train", "correct_identity"]
ProbeBasisMethod = Literal["logistic_svd", "class_mean_pca"]


def train_multiply_element_mask(
    n_elements: int,
    *,
    train_fraction: float,
    seed: int,
) -> np.ndarray:
    """Elements that appear in the multiplication train split (seed-matched)."""
    total = n_elements * n_elements
    train_size = int(total * train_fraction)
    test_size = total - train_size
    a_train, b_train, _, _ = split_pair_indices(n_elements, train_size, test_size, seed)
    train_el = {int(a) for a in a_train.tolist()} | {int(b) for b in b_train.tolist()}
    mask = np.zeros(n_elements, dtype=bool)
    for g in train_el:
        mask[g] = True
    return mask


@torch.no_grad()
def correct_identity_element_mask(
    model: GroupTransformer,
    mult_table: np.ndarray,
    ring_id: str,
    *,
    device: str,
) -> np.ndarray:
    """Elements g with correct prediction on canonical pair (g, e)."""
    n = mult_table.shape[0]
    e = ring_identity_index(ring_id, mult_table)
    dev = torch.device(device)
    mask = np.zeros(n, dtype=bool)
    for g in range(n):
        x = torch.tensor([[g, e]], dtype=torch.long, device=dev)
        pred = int(forward_group_transformer_with_patch(model, x).argmax(dim=1).item())
        mask[g] = pred == int(mult_table[g, e])
    return mask


def element_train_mask(
    n_elements: int,
    *,
    element_train_fraction: float,
    seed: int,
) -> np.ndarray:
    """Element-level train partition (disjoint holdout of ring indices)."""
    train_el, _test_el = split_ring_elements(
        n_elements, element_train_fraction, seed
    )
    mask = np.zeros(n_elements, dtype=bool)
    mask[train_el] = True
    return mask


def resolve_probe_fit_mask(
    mode: ProbeFitMode,
    *,
    n_elements: int,
    train_fraction: float,
    seed: int,
    model: GroupTransformer | None = None,
    mult_table: np.ndarray | None = None,
    ring_id: str | None = None,
    device: str = "cpu",
    element_train_fraction: float | None = None,
) -> np.ndarray | None:
    if mode == "all":
        return None
    if mode == "train_mult_elements":
        return train_multiply_element_mask(
            n_elements, train_fraction=train_fraction, seed=seed
        )
    if mode == "element_train":
        frac = (
            element_train_fraction
            if element_train_fraction is not None
            else train_fraction
        )
        return element_train_mask(
            n_elements, element_train_fraction=frac, seed=seed
        )
    if mode == "correct_identity":
        if model is None or mult_table is None or ring_id is None:
            raise ValueError("correct_identity probe fit requires model, mult_table, ring_id")
        return correct_identity_element_mask(
            model, mult_table, ring_id, device=device
        )
    raise ValueError(f"unknown probe_fit_mode: {mode!r}")


def held_out_base_pairs(
    n_elements: int,
    *,
    train_fraction: float,
    seed: int,
) -> set[tuple[int, int]]:
    """Base inputs (a, b) in the held-out multiplication split (seed-matched training)."""
    total = n_elements * n_elements
    train_size = int(total * train_fraction)
    test_size = total - train_size
    _, _, a_test, b_test = split_pair_indices(n_elements, train_size, test_size, seed)
    return {(int(a), int(b)) for a, b in zip(a_test.tolist(), b_test.tolist(), strict=True)}


def held_out_element_pairs(
    n_elements: int,
    *,
    train_fraction: float,
    seed: int,
) -> set[tuple[int, int]]:
    """Pairs (a,b) with both operands in the held-out element split (seed-matched)."""
    _train_el, test_el = split_ring_elements(n_elements, train_fraction, seed)
    test_set = {int(x) for x in test_el.tolist()}
    return {(a, b) for a in test_set for b in test_set}


def resolve_allowed_pair_filters(
    n_elements: int,
    pair_split: PairSplit,
    *,
    train_fraction: float | None,
    seed: int,
    split_seed: int | None = None,
) -> tuple[set[tuple[int, int]] | None, set[tuple[int, int]] | None]:
    """Return (allowed_base_pairs, allowed_counterfactual_pairs) for IIA sampling."""
    if pair_split == "full":
        return None, None
    if train_fraction is None:
        train_fraction = 0.5 if n_elements <= 64 else 0.3
    split_rng_seed = split_seed if split_seed is not None else seed
    held_out = held_out_base_pairs(
        n_elements, train_fraction=train_fraction, seed=split_rng_seed
    )
    if pair_split == "test":
        return held_out, None
    if pair_split == "test_strict":
        return held_out, held_out
    if pair_split == "test_elem":
        elem = held_out_element_pairs(
            n_elements, train_fraction=train_fraction, seed=split_rng_seed
        )
        return elem, None
    if pair_split == "test_elem_strict":
        elem = held_out_element_pairs(
            n_elements, train_fraction=train_fraction, seed=split_rng_seed
        )
        return elem, elem
    raise ValueError(f"unknown pair_split: {pair_split!r}")


def resolve_allowed_base_pairs(
    n_elements: int,
    pair_split: PairSplit,
    *,
    train_fraction: float | None,
    seed: int,
    split_seed: int | None = None,
) -> set[tuple[int, int]] | None:
    base, _ = resolve_allowed_pair_filters(
        n_elements,
        pair_split,
        train_fraction=train_fraction,
        seed=seed,
        split_seed=split_seed,
    )
    return base


def min_subspace_dim_for_probe(labels: np.ndarray, *, override: int | None = None) -> int:
    """Scalar probes: k_min = max(1, min(n_classes−1, SCALAR_K_MIN_CAP)); vector (a,b,c,d): full dim."""
    if override is not None:
        return int(override)
    if labels.ndim == 2:
        return int(labels.shape[1])
    n_classes = int(len(np.unique(labels)))
    return max(1, min(n_classes - 1, SCALAR_K_MIN_CAP))


def probe_subspace_basis_scalar(
    hidden_by_elem: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Legacy 1D lstsq probe (pre-Zheng); kept for regression tests."""
    y = labels.astype(np.float64).reshape(-1, 1)
    x = hidden_by_elem.astype(np.float64)
    x_aug = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
    coef, _, _, _ = np.linalg.lstsq(x_aug, y, rcond=None)
    w = coef[:-1].reshape(1, -1)
    _, _, vt = np.linalg.svd(w, full_matrices=False)
    return vt[:1], 1


def _proj_label(spec: WedderburnSpec, name: str, g: int) -> np.ndarray | int:
    return spec.labels[name][g]


def _labels_equal(a: np.ndarray | int, b: np.ndarray | int) -> bool:
    return bool(np.array_equal(np.asarray(a), np.asarray(b)))


@torch.no_grad()
def run_ring_iia_for_component(
    model: GroupTransformer,
    mult_table: np.ndarray,
    spec: WedderburnSpec,
    target: str,
    *,
    device: str,
    patch_site: TransformerPatchSite = "resid_pre_0",
    max_pairs: int = 200,
    rng: np.random.Generator,
    hidden_by_elem: np.ndarray,
    bottleneck_steps: int = 150,
    trace_mode: bool = False,
    allowed_base_pairs: set[tuple[int, int]] | None = None,
    allowed_counterfactual_pairs: set[tuple[int, int]] | None = None,
    probe_fit_mask: np.ndarray | None = None,
    probe_basis_method: ProbeBasisMethod = "logistic_svd",
    mlp_hidden_layer_sizes: tuple[int, ...] = (32,),
) -> IIAMetrics:
    n = mult_table.shape[0]
    dev = torch.device(device)
    labels = spec.labels[target]
    k_min = min_subspace_dim_for_probe(labels)
    if labels.ndim == 2:
        probe = probe_subspace_basis_from_vector_labels(
            hidden_by_elem,
            labels,
            min_subspace_dim=k_min,
            fit_mask=probe_fit_mask,
            mlp_hidden_layer_sizes=mlp_hidden_layer_sizes,
        )
    elif probe_basis_method == "class_mean_pca":
        probe = probe_subspace_basis_from_class_means(
            hidden_by_elem, labels, min_subspace_dim=k_min, fit_mask=probe_fit_mask
        )
    else:
        probe = probe_subspace_basis_from_labels(
            hidden_by_elem,
            labels,
            min_subspace_dim=k_min,
            fit_mask=probe_fit_mask,
            mlp_hidden_layer_sizes=mlp_hidden_layer_sizes,
        )
    basis, k = probe.basis, probe.k
    mlp_basis = probe.mlp_basis
    mlp_k = probe.mlp_patch_k if probe.mlp_patch_k > 0 else probe.mlp_k
    if mlp_basis is not None and mlp_k > 0:
        mlp_basis = mlp_basis[:mlp_k]
    rand_basis = random_orthonormal_basis(hidden_by_elem.shape[1], k, rng)
    mlp_rand_basis = (
        random_orthonormal_basis(hidden_by_elem.shape[1], mlp_k, rng)
        if mlp_basis is not None and mlp_k > 0
        else None
    )
    signatures = labels_to_signatures(spec)
    pairs = sample_matched_pairs(
        n,
        signatures,
        spec.component_names,
        target,
        max_pairs=max_pairs,
        rng=rng,
        allowed_base_pairs=allowed_base_pairs,
        allowed_counterfactual_pairs=allowed_counterfactual_pairs,
    )
    chance = spec.chance[target]
    if not pairs:
        return IIAMetrics(
            irrep=target,
            raw_iia=0.0,
            n_pairs=0,
            subspace_iia=0.0,
            random_subspace_iia=0.0,
            subspace_dim=k,
            error_preservation=float("nan"),
            error_preservation_chance=chance,
            conditional_ntp=float("nan"),
            probe_accuracy=probe.probe_accuracy,
            mlp_probe_accuracy=probe.mlp_probe_accuracy,
            n_classes=probe.n_classes,
            mlp_subspace_iia=float("nan"),
            mlp_random_subspace_iia=float("nan"),
            mlp_subspace_dim=mlp_k,
            mlp_bottleneck_iia=float("nan"),
            mlp_bottleneck_random_iia=float("nan"),
            raw_logit_restoration=float("nan"),
            subspace_logit_restoration=float("nan"),
            random_subspace_logit_restoration=float("nan"),
        )

    raw_ok = sub_ok = rand_ok = mlp_sub_ok = mlp_rand_ok = 0
    mlp_bn_ok = mlp_bn_rand_ok = 0
    raw_lr_vals: list[float] = []
    sub_lr_vals: list[float] = []
    rand_lr_vals: list[float] = []
    mlp_w0 = probe.mlp_w0
    mlp_b0 = probe.mlp_b0
    has_mlp_nl = mlp_w0 is not None and mlp_b0 is not None
    err_preserve = err_total = 0
    cond_ntp = cond_total = 0
    others = [c for c in spec.component_names if c != target]

    for a, a_prime, b in pairs:
        true_cf = int(mult_table[a_prime, b])
        true_base = int(mult_table[a, b])
        h_base = hidden_at_site(model, a, b, dev, site=patch_site)
        h_src = hidden_at_site(model, a_prime, b, dev, site=patch_site)
        x_base = torch.tensor([[a, b]], dtype=torch.long, device=dev)
        x_clean = torch.tensor([[a_prime, b]], dtype=torch.long, device=dev)
        base_pred = int(forward_group_transformer_with_patch(model, x_base).argmax(dim=1).item())

        ld_clean = counterfactual_logit_diff(
            forward_group_transformer_with_patch(model, x_clean), true_cf
        )
        ld_corrupt = counterfactual_logit_diff(
            forward_group_transformer_with_patch(model, x_base), true_cf
        )

        patch_modes: list[tuple[str, torch.Tensor]] = []
        if trace_mode:
            patch_modes = [("sub", _apply_subspace_patch(h_base, h_src, basis))]
        else:
            patch_modes = [
                ("full", h_src),
                ("sub", _apply_subspace_patch(h_base, h_src, basis)),
                ("rand", _apply_subspace_patch(h_base, h_src, rand_basis)),
            ]
            if mlp_basis is not None and mlp_k > 0 and mlp_rand_basis is not None:
                patch_modes.extend(
                    [
                        ("mlp_sub", _apply_subspace_patch(h_base, h_src, mlp_basis)),
                        ("mlp_rand", _apply_subspace_patch(h_base, h_src, mlp_rand_basis)),
                    ]
                )
        h_mlp_nl: torch.Tensor | None = None
        h_mlp_nl_rand: torch.Tensor | None = None
        if has_mlp_nl:
            h_mlp_nl = nonlinear_bottleneck_patch(
                h_base, h_src, mlp_w0, mlp_b0, steps=bottleneck_steps
            )
            g_rand = int(rng.integers(0, n))
            h_rand = hidden_at_site(model, g_rand, b, dev, site=patch_site)
            z_rand = mlp_bottleneck_activation(h_rand, mlp_w0, mlp_b0)
            h_mlp_nl_rand = nonlinear_bottleneck_patch(
                h_base, h_src, mlp_w0, mlp_b0, z_target=z_rand, steps=bottleneck_steps
            )
        for mode, h_patch in patch_modes:
            logits = forward_group_transformer_with_patch(
                model, x_base, patch_value=h_patch, patch_site=patch_site
            )
            pred = int(logits.argmax(dim=1).item())
            if pred == true_cf:
                if mode == "full":
                    raw_ok += 1
                elif mode == "sub":
                    sub_ok += 1
                elif mode == "rand":
                    rand_ok += 1
                elif mode == "mlp_sub":
                    mlp_sub_ok += 1
                elif mode == "mlp_rand":
                    mlp_rand_ok += 1
            if not trace_mode:
                ld_int = counterfactual_logit_diff(logits, true_cf)
                lr = logit_diff_restoration(ld_clean, ld_corrupt, ld_int)
                if math.isfinite(lr):
                    if mode == "full":
                        raw_lr_vals.append(lr)
                    elif mode == "sub":
                        sub_lr_vals.append(lr)
                    elif mode == "rand":
                        rand_lr_vals.append(lr)

        if h_mlp_nl is not None:
            logits = forward_group_transformer_with_patch(
                model, x_base, patch_value=h_mlp_nl, patch_site=patch_site
            )
            if int(logits.argmax(dim=1).item()) == true_cf:
                mlp_bn_ok += 1
        if h_mlp_nl_rand is not None:
            logits = forward_group_transformer_with_patch(
                model, x_base, patch_value=h_mlp_nl_rand, patch_site=patch_site
            )
            if int(logits.argmax(dim=1).item()) == true_cf:
                mlp_bn_rand_ok += 1

        if not trace_mode and base_pred != true_base:
            err_total += 1
            logits = forward_group_transformer_with_patch(
                model, x_base, patch_value=h_src, patch_site=patch_site
            )
            pred_p = int(logits.argmax(dim=1).item())
            if all(
                _labels_equal(_proj_label(spec, o, pred_p), _proj_label(spec, o, base_pred))
                for o in others
            ):
                err_preserve += 1
        elif not trace_mode:
            cond_total += 1
            logits = forward_group_transformer_with_patch(
                model, x_base, patch_value=h_src, patch_site=patch_site
            )
            pred_p = int(logits.argmax(dim=1).item())
            if all(
                _labels_equal(_proj_label(spec, o, pred_p), _proj_label(spec, o, base_pred))
                for o in others
            ):
                cond_ntp += 1

    m = len(pairs)
    return IIAMetrics(
        irrep=target,
        raw_iia=raw_ok / m if not trace_mode else float("nan"),
        n_pairs=m,
        subspace_iia=sub_ok / m,
        random_subspace_iia=rand_ok / m,
        subspace_dim=k,
        error_preservation=err_preserve / err_total if err_total else float("nan"),
        error_preservation_chance=chance,
        conditional_ntp=cond_ntp / cond_total if cond_total else float("nan"),
        probe_accuracy=probe.probe_accuracy,
        mlp_probe_accuracy=probe.mlp_probe_accuracy,
        n_classes=probe.n_classes,
        mlp_subspace_iia=mlp_sub_ok / m if mlp_k > 0 else float("nan"),
        mlp_random_subspace_iia=mlp_rand_ok / m if mlp_k > 0 else float("nan"),
        mlp_subspace_dim=mlp_k,
        mlp_bottleneck_iia=mlp_bn_ok / m if has_mlp_nl else float("nan"),
        mlp_bottleneck_random_iia=mlp_bn_rand_ok / m if has_mlp_nl else float("nan"),
        raw_logit_restoration=float(np.mean(raw_lr_vals)) if raw_lr_vals else float("nan"),
        subspace_logit_restoration=float(np.mean(sub_lr_vals)) if sub_lr_vals else float("nan"),
        random_subspace_logit_restoration=float(np.mean(rand_lr_vals)) if rand_lr_vals else float("nan"),
    )




@torch.no_grad()
def collect_ring_transformer_hidden_by_elem(
    model: GroupTransformer,
    mult_table: np.ndarray,
    ring_id: str,
    *,
    device: str,
    patch_site: TransformerPatchSite = "resid_pre_0",
) -> np.ndarray:
    n = mult_table.shape[0]
    e = ring_identity_index(ring_id, mult_table)
    dev = torch.device(device)
    return np.stack(
        [
            hidden_at_site(model, g, e, dev, site=patch_site).cpu().numpy().ravel()
            for g in range(n)
        ],
        axis=0,
    )


def build_ring_transformer_from_config(
    ring_id: str,
    cfg: dict[str, Any],
    device: str,
    *,
    checkpoint_path: str | Path | None = None,
) -> tuple[GroupTransformer, np.ndarray, WedderburnSpec]:
    """Создать GroupTransformer по config; опционально загрузить веса."""
    mult_table, _ = build_mul_table(ring_id)
    spec = get_wedderburn_spec(ring_id)
    n = int(cfg.get("num_elements", mult_table.shape[0]))
    model = GroupTransformer(
        num_elements=n,
        d_model=int(cfg.get("d_model", 128)),
        nhead=int(cfg.get("nhead", 4)),
        num_layers=int(cfg.get("num_layers", 1)),
        ffn_dim=int(cfg.get("ffn_dim", 512)),
        dropout=float(cfg.get("dropout", 0.0)),
        layernorm=bool(cfg.get("layernorm", False)),
    )
    if checkpoint_path is not None:
        ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, mult_table, spec


def default_ring_transformer_config(ring_id: str) -> dict[str, Any]:
    """Zheng protocol defaults for IIA hierarchy (random-init baseline)."""
    spec = get_ring_spec(ring_id)
    cfg: dict[str, Any] = {
        "ring_id": ring_id,
        "num_elements": spec.n_elements,
        "d_model": 128,
        "nhead": 4,
        "num_layers": 1,
        "ffn_dim": 512,
        "dropout": 0.0,
        "layernorm": False,
    }
    if getattr(spec, "encoding_version", None) is not None:
        cfg["encoding_version"] = spec.encoding_version
    return cfg


def load_ring_transformer(
    checkpoint: str,
    device: str,
) -> tuple[GroupTransformer, dict, np.ndarray, WedderburnSpec]:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    ring_id = cfg.get("ring_id")
    if ring_id is None:
        raise KeyError("checkpoint config missing ring_id")
    model, mult_table, spec = build_ring_transformer_from_config(
        ring_id, cfg, device, checkpoint_path=checkpoint
    )
    return model, ckpt, mult_table, spec


@torch.no_grad()
def run_ring_transformer_iia_analysis(
    model: GroupTransformer,
    mult_table: np.ndarray,
    spec: WedderburnSpec,
    *,
    checkpoint: str = "",
    device: str = "cpu",
    patch_site: TransformerPatchSite = "resid_pre_0",
    max_pairs_per_component: int = 200,
    seed: int = 42,
    pair_split: PairSplit = "full",
    train_fraction: float | None = None,
    split_seed: int | None = None,
    hidden_by_elem: np.ndarray | None = None,
    on_component: Callable[[str, int, int], None] | None = None,
    probe_fit_mode: ProbeFitMode = "element_train",
    probe_basis_method: ProbeBasisMethod = "logistic_svd",
    mlp_hidden_layer_sizes: tuple[int, ...] = (32,),
) -> IIAResult:
    rng = np.random.default_rng(seed)
    n = mult_table.shape[0]
    allowed_base_pairs, allowed_counterfactual_pairs = resolve_allowed_pair_filters(
        n,
        pair_split,
        train_fraction=train_fraction,
        seed=seed,
        split_seed=split_seed,
    )
    if hidden_by_elem is None:
        hidden_by_elem = collect_ring_transformer_hidden_by_elem(
            model, mult_table, spec.ring_id, device=device, patch_site=patch_site
        )
    if train_fraction is None:
        train_fraction = 0.5 if n <= 64 else 0.3
    split_rng_seed = split_seed if split_seed is not None else seed
    probe_fit_mask = resolve_probe_fit_mask(
        probe_fit_mode,
        n_elements=n,
        train_fraction=train_fraction,
        seed=split_rng_seed,
        model=model,
        mult_table=mult_table,
        ring_id=spec.ring_id,
        device=device,
    )
    test_acc = evaluate_transformer_accuracy(model, mult_table, device)
    component_names = spec.component_names
    per: list[IIAMetrics] = []
    for idx, name in enumerate(component_names):
        if on_component is not None:
            on_component(name, idx, len(component_names))
        per.append(
            run_ring_iia_for_component(
                model,
                mult_table,
                spec,
                name,
                device=device,
                patch_site=patch_site,
                max_pairs=max_pairs_per_component,
                rng=rng,
                hidden_by_elem=hidden_by_elem,
                allowed_base_pairs=allowed_base_pairs,
                allowed_counterfactual_pairs=allowed_counterfactual_pairs,
                probe_fit_mask=probe_fit_mask,
                probe_basis_method=probe_basis_method,
                mlp_hidden_layer_sizes=mlp_hidden_layer_sizes,
            )
        )
    raw_vals = [m.raw_iia for m in per if m.n_pairs > 0]
    ratios = [
        m.error_preservation / m.error_preservation_chance
        for m in per
        if m.n_pairs > 0
        and m.error_preservation_chance > 0
        and not np.isnan(m.error_preservation)
    ]
    return IIAResult(
        checkpoint=checkpoint,
        n_elements=n,
        test_accuracy=test_acc,
        patch_site=patch_site,
        per_irrep=per,
        mean_raw_iia=float(np.mean(raw_vals)) if raw_vals else 0.0,
        mean_error_preservation_ratio=float(np.mean(ratios)) if ratios else 0.0,
    )


@torch.no_grad()
def measure_ring_transformer_iia_row(
    model: GroupTransformer,
    mult_table: np.ndarray,
    spec: WedderburnSpec,
    hidden_by_elem: np.ndarray | None,
    *,
    device: str,
    patch_site: TransformerPatchSite = "resid_pre_0",
    max_pairs_per_component: int = 50,
    seed: int = 42,
    bottleneck_steps: int = 150,
    trace_mode: bool = False,
    pair_split: PairSplit = "full",
    train_fraction: float | None = None,
    split_seed: int | None = None,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = mult_table.shape[0]
    dev = torch.device(device)
    allowed_base_pairs, allowed_counterfactual_pairs = resolve_allowed_pair_filters(
        n,
        pair_split,
        train_fraction=train_fraction,
        seed=seed,
        split_seed=split_seed,
    )
    if hidden_by_elem is None:
        hidden_by_elem = collect_ring_transformer_hidden_by_elem(
            model, mult_table, spec.ring_id, device=device, patch_site=patch_site
        )
    per = [
        run_ring_iia_for_component(
            model,
            mult_table,
            spec,
            name,
            device=device,
            patch_site=patch_site,
            max_pairs=max_pairs_per_component,
            rng=rng,
            hidden_by_elem=hidden_by_elem,
            bottleneck_steps=bottleneck_steps,
            trace_mode=trace_mode,
            allowed_base_pairs=allowed_base_pairs,
            allowed_counterfactual_pairs=allowed_counterfactual_pairs,
        )
        for name in spec.component_names
    ]
    raw_vals = [m.raw_iia for m in per if m.n_pairs > 0]
    sub_vals = [m.subspace_iia for m in per if m.n_pairs > 0]
    rand_vals = [m.random_subspace_iia for m in per if m.n_pairs > 0]
    delta_vals = [s - r for s, r in zip(sub_vals, rand_vals)]
    gain_vals = [
        normalized_subspace_gain(m.subspace_iia, m.raw_iia, m.random_subspace_iia)
        for m in per
        if m.n_pairs > 0
    ]
    gain_vals = [g for g in gain_vals if g == g]
    mlp_sub_vals = [m.mlp_subspace_iia for m in per if m.n_pairs > 0 and not np.isnan(m.mlp_subspace_iia)]
    mlp_bn_vals = [m.mlp_bottleneck_iia for m in per if m.n_pairs > 0 and not np.isnan(m.mlp_bottleneck_iia)]
    row = {
        "mean_raw_iia": float(np.mean(raw_vals)) if raw_vals else 0.0,
        "mean_subspace_iia": float(np.mean(sub_vals)) if sub_vals else 0.0,
        "mean_subspace_delta": float(np.mean(delta_vals)) if delta_vals else 0.0,
        "mean_normalized_subspace_gain": float(np.mean(gain_vals)) if gain_vals else float("nan"),
        "mean_mlp_subspace_iia": float(np.mean(mlp_sub_vals)) if mlp_sub_vals else float("nan"),
        "mean_mlp_bottleneck_iia": float(np.mean(mlp_bn_vals)) if mlp_bn_vals else float("nan"),
        "mean_random_subspace_iia": float(np.mean(rand_vals)) if rand_vals else 0.0,
    }
    err_ratios = [
        m.error_preservation / m.error_preservation_chance
        for m in per
        if m.n_pairs > 0
        and m.error_preservation_chance > 0
        and not np.isnan(m.error_preservation)
    ]
    row["mean_error_preservation_ratio"] = (
        float(np.mean(err_ratios)) if err_ratios else float("nan")
    )
    for m in per:
        if m.n_pairs > 0:
            row[f"iia_raw_{m.irrep}"] = m.raw_iia
            row[f"iia_sub_{m.irrep}"] = m.subspace_iia
            row[f"iia_rand_{m.irrep}"] = m.random_subspace_iia
            row[f"iia_deff_{m.irrep}"] = float(m.subspace_dim)
            if not np.isnan(m.probe_accuracy):
                row[f"logistic_probe_{m.irrep}"] = m.probe_accuracy
                row[f"probe_acc_{m.irrep}"] = m.probe_accuracy
            if not np.isnan(m.mlp_probe_accuracy):
                row[f"mlp_probe_{m.irrep}"] = m.mlp_probe_accuracy
            if not np.isnan(m.mlp_subspace_iia):
                row[f"iia_mlp_sub_{m.irrep}"] = m.mlp_subspace_iia
                row[f"iia_mlp_rand_{m.irrep}"] = m.mlp_random_subspace_iia
            if not np.isnan(m.mlp_bottleneck_iia):
                row[f"iia_mlp_bn_{m.irrep}"] = m.mlp_bottleneck_iia
                row[f"iia_mlp_bn_rand_{m.irrep}"] = m.mlp_bottleneck_random_iia
            if not np.isnan(m.error_preservation):
                row[f"err_pres_{m.irrep}"] = m.error_preservation
    return row


@torch.no_grad()
def measure_ring_probe_row(
    model: GroupTransformer,
    mult_table: np.ndarray,
    spec: WedderburnSpec,
    hidden_by_elem: np.ndarray | None,
    *,
    device: str,
    patch_site: TransformerPatchSite = "resid_pre_0",
    seed: int = 42,
) -> dict[str, float]:
    """Fast predictive probes only (no interchange interventions)."""
    n = mult_table.shape[0]
    dev = torch.device(device)
    if hidden_by_elem is None:
        e = ring_identity_index(spec.ring_id, mult_table)
        hidden_by_elem = np.stack(
            [
                hidden_at_site(model, g, e, dev, site=patch_site).cpu().numpy().ravel()
                for g in range(n)
            ],
            axis=0,
        )
    logistic_vals: list[float] = []
    mlp_vals: list[float] = []
    row: dict[str, float] = {}
    for name in spec.component_names:
        labels = spec.labels[name]
        k_min = min_subspace_dim_for_probe(labels)
        if labels.ndim == 2:
            probe = probe_subspace_basis_from_vector_labels(
                hidden_by_elem, labels, min_subspace_dim=k_min, seed=seed
            )
        else:
            probe = probe_subspace_basis_from_labels(
                hidden_by_elem, labels, min_subspace_dim=k_min, seed=seed
            )
        if not np.isnan(probe.probe_accuracy):
            row[f"logistic_probe_{name}"] = float(probe.probe_accuracy)
            logistic_vals.append(float(probe.probe_accuracy))
        if not np.isnan(probe.mlp_probe_accuracy):
            row[f"mlp_probe_{name}"] = float(probe.mlp_probe_accuracy)
            mlp_vals.append(float(probe.mlp_probe_accuracy))
    if logistic_vals:
        row["mean_logistic_probe"] = float(np.mean(logistic_vals))
    if mlp_vals:
        row["mean_mlp_probe"] = float(np.mean(mlp_vals))
    return row
