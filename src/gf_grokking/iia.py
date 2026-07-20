"""Interchange Intervention Accuracy (IIA) for alt-group MLP.

Адаптация протокола Geiger et al. / «From Groups to Rings» для архитектуры
one-hot → Linear → ReLU → Linear. Аналог патчинга ``resid_pre_0`` у Transformer:
подмена скрытых активаций (``post_relu``) при фиксированном правом операнде ``b``.
"""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch
import torch.nn as nn

from .alt_group_interpret import compute_hidden_responses, load_alt_group_mlp
from .group_characters import conjugacy_data
from .models.mlp import MLP

PatchSite = Literal["pre_relu", "post_relu"]

_logger = logging.getLogger(__name__)


@dataclass
class MLPForwardCache:
    pre_relu: torch.Tensor
    post_relu: torch.Tensor
    logits: torch.Tensor


@dataclass
class ProbeSubspaceResult:
    basis: np.ndarray
    k: int
    probe_accuracy: float
    n_classes: int
    mlp_probe_accuracy: float = float("nan")
    mlp_basis: np.ndarray | None = None
    mlp_k: int = 0
    mlp_patch_k: int = 0
    mlp_w0: np.ndarray | None = None
    mlp_b0: np.ndarray | None = None


@dataclass
class IIAMetrics:
    irrep: str
    raw_iia: float
    n_pairs: int
    subspace_iia: float
    random_subspace_iia: float
    subspace_dim: int
    error_preservation: float
    error_preservation_chance: float
    conditional_ntp: float
    probe_accuracy: float = float("nan")
    mlp_probe_accuracy: float = float("nan")
    n_classes: int = 0
    mlp_subspace_iia: float = float("nan")
    mlp_random_subspace_iia: float = float("nan")
    mlp_subspace_dim: int = 0
    mlp_bottleneck_iia: float = float("nan")
    mlp_bottleneck_random_iia: float = float("nan")
    raw_logit_restoration: float = float("nan")
    subspace_logit_restoration: float = float("nan")
    random_subspace_logit_restoration: float = float("nan")


def counterfactual_logit_diff(logits: torch.Tensor, target: int) -> float:
    """Margin for counterfactual class: logit[target] - max_{j!=target} logit[j]."""
    flat = logits.detach().reshape(-1)
    t = int(target)
    target_logit = float(flat[t].item())
    if flat.numel() <= 1:
        return target_logit
    mask = torch.ones(flat.shape[0], dtype=torch.bool, device=flat.device)
    mask[t] = False
    runner_up = float(flat[mask].max().item())
    return target_logit - runner_up


def logit_diff_restoration(
    ld_clean: float,
    ld_corrupt: float,
    ld_intervened: float,
    *,
    eps: float = 0.05,
) -> float:
    """Fraction of clean--corrupt logit margin restored by intervention (clipped to [0, 1])."""
    denom = ld_clean - ld_corrupt
    if not math.isfinite(denom) or abs(denom) < eps:
        return float("nan")
    val = (ld_intervened - ld_corrupt) / denom
    if not math.isfinite(val):
        return float("nan")
    return float(max(0.0, min(1.0, val)))


@dataclass
class IIAResult:
    checkpoint: str
    n_elements: int
    test_accuracy: float
    patch_site: str
    per_irrep: list[IIAMetrics] = field(default_factory=list)
    mean_raw_iia: float = 0.0
    mean_error_preservation_ratio: float = 0.0


def irrep_character_vectors(
    mult_table: np.ndarray,
    irrep_specs: dict[str, tuple[int, dict[str, float | complex]]],
    names: tuple[str, ...],
) -> dict[str, np.ndarray]:
    """χ_i(g) для каждого g (complex)."""
    n = mult_table.shape[0]
    _, _, classes, class_labels = conjugacy_data(mult_table)
    elem_label: dict[int, str] = {}
    for i, cls in enumerate(classes):
        for g in cls:
            elem_label[g] = class_labels[i]
    out: dict[str, np.ndarray] = {}
    for name in names:
        _, char_by_class = irrep_specs[name]
        col = np.zeros(n, dtype=np.complex128)
        for g in range(n):
            raw = char_by_class[elem_label[g]]
            col[g] = complex(raw) if not isinstance(raw, complex) else raw
        out[name] = col
    return out


def _char_key(val: complex) -> tuple[float, float]:
    return (round(float(val.real), 8), round(float(val.imag), 8))


def _char_key_at(chars: dict[str, np.ndarray], name: str, g: int) -> tuple[float, float]:
    return _char_key(complex(chars[name][g]))


def _distinct_char_keys(chars: np.ndarray) -> int:
    return len({_char_key(complex(c)) for c in chars})


def _chance_preservation(chars: np.ndarray) -> float:
    d = _distinct_char_keys(chars)
    return 1.0 / d if d > 0 else 0.0


def _probe_train_test_split(
    x: np.ndarray,
    y: np.ndarray,
    *,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Same split policy as logistic probes (stratified when possible)."""
    from sklearn.model_selection import train_test_split

    y_arr = np.asarray(y)
    if y_arr.ndim == 1:
        labels = y_arr.astype(np.int64).reshape(-1)
        n_classes = len(np.unique(labels))
        _, counts = np.unique(labels, return_counts=True)
        stratify = labels if n_classes > 1 and int(counts.min()) >= 2 else None
    else:
        stratify = None
    if y_arr.shape[0] >= 5 and test_fraction > 0:
        return train_test_split(
            x,
            y_arr,
            test_size=test_fraction,
            random_state=seed,
            stratify=stratify,
        )
    return x, x, y_arr, y_arr


def _mlp_n_splits(y: np.ndarray) -> int:
    """Stratified CV folds for small ring element sets."""
    _, counts = np.unique(y.astype(np.int64).reshape(-1), return_counts=True)
    return max(2, min(5, int(counts.min())))


def _make_mlp_classifier(
    hidden_layer_sizes: tuple[int, ...],
    *,
    seed: int,
    alpha: float = 1.0,
) -> "MLPClassifier":
    from sklearn.neural_network import MLPClassifier

    return MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        max_iter=2000,
        early_stopping=False,
        alpha=alpha,
        learning_rate_init=1e-3,
        random_state=seed,
    )


def _make_mlp_regressor(
    hidden_layer_sizes: tuple[int, ...],
    *,
    seed: int,
    alpha: float = 1.0,
) -> "MLPRegressor":
    from sklearn.neural_network import MLPRegressor

    return MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        max_iter=2000,
        early_stopping=False,
        alpha=alpha,
        learning_rate_init=1e-3,
        random_state=seed,
    )


def _mlp_classification_accuracy(
    x: np.ndarray,
    y: np.ndarray,
    *,
    test_fraction: float = 0.2,
    seed: int = 42,
    hidden_layer_sizes: tuple[int, ...] = (32,),
    alpha: float = 1.0,
    use_cv_if_small: bool = True,
) -> float:
    """MLP probe accuracy: stratified CV on small n; else held-out split."""
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    y_flat = y.astype(np.int64).reshape(-1)
    if len(np.unique(y_flat)) <= 1:
        return 1.0
    clf = _make_mlp_classifier(hidden_layer_sizes, seed=seed, alpha=alpha)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
    if use_cv_if_small and x.shape[0] <= 512:
        n_splits = _mlp_n_splits(y_flat)
        _, counts = np.unique(y_flat, return_counts=True)
        if n_splits > 1 and int(counts.min()) >= n_splits:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            scores = cross_val_score(clf, x, y_flat, cv=cv, scoring="accuracy")
            return float(np.mean(scores))
        x_tr, x_te, y_tr, y_te = _probe_train_test_split(
            x, y_flat, test_fraction=test_fraction, seed=seed
        )
        clf.fit(x_tr, y_tr)
        return float(accuracy_score(y_te, clf.predict(x_te)))


def _fit_mlp_classifier_full(
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int = 42,
    hidden_layer_sizes: tuple[int, ...] = (32,),
    alpha: float = 1.0,
) -> "MLPClassifier":
    """Train MLP on all samples (for stable weight extraction)."""
    y_flat = y.astype(np.int64).reshape(-1)
    clf = _make_mlp_classifier(hidden_layer_sizes, seed=seed, alpha=alpha)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(x, y_flat)
    return clf


def mlp_probe_accuracy(
    hidden_by_elem: np.ndarray,
    labels: np.ndarray,
    *,
    test_fraction: float = 0.2,
    seed: int = 42,
    hidden_layer_sizes: tuple[int, ...] = (32,),
) -> float:
    """Predictive probe: 1-hidden-layer MLP (32 units); CV on small |R|."""
    x = hidden_by_elem.astype(np.float64)
    y = np.asarray(labels)
    if y.ndim == 1:
        return _mlp_classification_accuracy(
            x,
            y,
            test_fraction=test_fraction,
            seed=seed,
            hidden_layer_sizes=hidden_layer_sizes,
        )

    y_vec = y.astype(np.float64)
    if y_vec.ndim != 2:
        raise ValueError(f"labels must be 1D classes or 2D coordinates, got {y.shape}")
    from sklearn.metrics import r2_score
    from sklearn.model_selection import KFold, cross_val_score

    reg = _make_mlp_regressor(hidden_layer_sizes, seed=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if x.shape[0] <= 512:
            n_splits = max(2, min(5, x.shape[0] // 4))
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
            scores = cross_val_score(reg, x, y_vec, cv=cv, scoring="r2")
            return float(np.mean(scores))
        x_tr, x_te, y_tr, y_te = _probe_train_test_split(
            x, y_vec, test_fraction=test_fraction, seed=seed
        )
        reg.fit(x_tr, y_tr)
        return float(r2_score(y_te, reg.predict(x_te), multioutput="uniform_average"))


def _masked_probe_rows(
    x: np.ndarray,
    labels: np.ndarray,
    fit_mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    if fit_mask is None:
        return x, labels
    mask = np.asarray(fit_mask, dtype=bool).reshape(-1)
    if mask.shape[0] != x.shape[0]:
        raise ValueError(f"fit_mask length {mask.shape[0]} != n rows {x.shape[0]}")
    if int(mask.sum()) < 2:
        raise ValueError("fit_mask must select at least two elements")
    return x[mask], labels[mask]


def fit_mlp_probe_subspace(
    hidden_by_elem: np.ndarray,
    labels: np.ndarray,
    *,
    variance_frac: float = 0.95,
    test_fraction: float = 0.2,
    seed: int = 42,
    min_subspace_dim: int = 1,
    hidden_layer_sizes: tuple[int, ...] = (32,),
    logistic_k: int | None = None,
    fit_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, int, int, float, np.ndarray, np.ndarray]:
    """Causal MLP bottleneck: full-data MLP, SVD(W_in), patch k capped to logistic k.

    Returns basis, mlp_k, mlp_patch_k, cv_accuracy, w0 (d×h), b0 (h,).
    """
    x = hidden_by_elem.astype(np.float64)
    y_arr = np.asarray(labels)
    x, y_arr = _masked_probe_rows(x, y_arr, fit_mask)
    hidden_dim = x.shape[1]

    def _degenerate_basis() -> tuple[np.ndarray, int, int, float, np.ndarray, np.ndarray]:
        basis = np.zeros((min_subspace_dim, hidden_dim), dtype=np.float64)
        basis[0, 0] = 1.0
        if min_subspace_dim > 1:
            basis[1, 1 if hidden_dim > 1 else 0] = 1.0
        w0 = np.eye(hidden_dim, min(32, hidden_dim), dtype=np.float64)
        b0 = np.zeros(w0.shape[1], dtype=np.float64)
        return basis, min_subspace_dim, min_subspace_dim, 1.0, w0, b0

    if y_arr.ndim == 1:
        y_flat = y_arr.astype(np.int64).reshape(-1)
        if len(np.unique(y_flat)) <= 1:
            return _degenerate_basis()
        acc = _mlp_classification_accuracy(
            x, y_flat, test_fraction=test_fraction, seed=seed, hidden_layer_sizes=hidden_layer_sizes
        )
        clf = _fit_mlp_classifier_full(
            x, y_flat, seed=seed, hidden_layer_sizes=hidden_layer_sizes
        )
        w1 = clf.coefs_[0].T.astype(np.float64)
        w0 = clf.coefs_[0].astype(np.float64)
        b0 = clf.intercepts_[0].astype(np.float64)
    else:
        y_vec = y_arr.astype(np.float64)
        if y_vec.ndim != 2:
            raise ValueError(f"labels must be 1D classes or 2D coordinates, got {y_arr.shape}")
        acc = mlp_probe_accuracy(
            hidden_by_elem, y_arr, test_fraction=test_fraction, seed=seed, hidden_layer_sizes=hidden_layer_sizes
        )
        reg = _make_mlp_regressor(hidden_layer_sizes, seed=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reg.fit(x, y_vec)
        w1 = reg.coefs_[0].T.astype(np.float64)
        w0 = reg.coefs_[0].astype(np.float64)
        b0 = reg.intercepts_[0].astype(np.float64)

    basis, k = _svd_basis_from_weight_matrix(
        w1, variance_frac=variance_frac, min_subspace_dim=min_subspace_dim
    )
    k = max(min_subspace_dim, min(k, hidden_dim, w1.shape[0]))
    patch_k = k if logistic_k is None else max(min_subspace_dim, min(k, logistic_k))
    return basis[:k], k, patch_k, acc, w0, b0


def _mlp_hidden_pre_relu(h: torch.Tensor, w0: np.ndarray, b0: np.ndarray) -> torch.Tensor:
    w = torch.as_tensor(w0, dtype=h.dtype, device=h.device)
    b = torch.as_tensor(b0, dtype=h.dtype, device=h.device)
    flat = h.reshape(-1)
    return flat @ w + b


def mlp_bottleneck_activation(
    h: torch.Tensor,
    w0: np.ndarray,
    b0: np.ndarray,
) -> torch.Tensor:
    return torch.relu(_mlp_hidden_pre_relu(h, w0, b0))


def nonlinear_bottleneck_patch(
    h_base: torch.Tensor,
    h_src: torch.Tensor,
    w0: np.ndarray,
    b0: np.ndarray,
    *,
    z_target: torch.Tensor | None = None,
    steps: int = 150,
    lr: float = 0.1,
    lam: float = 0.02,
) -> torch.Tensor:
    """Nonlinear IIA: find h with ReLU(h W0+b0) ≈ z_target, penalize ||h-h_base||²."""
    with torch.enable_grad():
        if z_target is None:
            with torch.no_grad():
                z_target = mlp_bottleneck_activation(h_src, w0, b0).detach()

        h0 = h_base.detach().reshape(-1).float()
        h_var = (h0 + 0.25 * (h_src.detach().reshape(-1).float() - h0)).requires_grad_(True)
        w = torch.as_tensor(w0, dtype=torch.float32, device=h_var.device)
        b = torch.as_tensor(b0, dtype=torch.float32, device=h_var.device)
        z_tgt = z_target.reshape(-1).float().detach()
        opt = torch.optim.Adam([h_var], lr=lr)
        for _ in range(steps):
            z = torch.relu(h_var @ w + b)
            loss = ((z - z_tgt) ** 2).mean() + lam * ((h_var - h0) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        return h_var.detach().reshape(h_base.shape).to(dtype=h_base.dtype)


def _make_one_hot_pair(a: int, b: int, n: int, device: torch.device) -> torch.Tensor:
    x = torch.zeros(1, 2 * n, device=device)
    x[0, a] = 1.0
    x[0, n + b] = 1.0
    return x


@torch.no_grad()
def forward_mlp_with_patch(
    model: MLP,
    x: torch.Tensor,
    *,
    patch_value: torch.Tensor | None = None,
    patch_site: PatchSite | None = None,
    return_cache: bool = False,
) -> torch.Tensor | MLPForwardCache:
    """Forward с опциональной подменой pre/post ReLU (как ``hook_resid_pre_0`` для MLP)."""
    pre_relu: nn.Linear = model.net[0]  # type: ignore[assignment]
    act: nn.Module = model.net[1]  # type: ignore[assignment]
    w2: nn.Linear = model.net[2]  # type: ignore[assignment]

    h_pre = pre_relu(x)
    h_post = act(h_pre)
    if patch_site == "pre_relu" and patch_value is not None:
        h_pre = patch_value
        h_post = act(h_pre)
    elif patch_site == "post_relu" and patch_value is not None:
        h_post = patch_value
    logits = w2(h_post)
    if return_cache:
        return MLPForwardCache(pre_relu=h_pre, post_relu=h_post, logits=logits)
    return logits


def hidden_for_pair(
    model: MLP,
    a: int,
    b: int,
    n: int,
    device: torch.device,
    *,
    site: PatchSite = "post_relu",
) -> torch.Tensor:
    x = _make_one_hot_pair(a, b, n, device)
    cache = forward_mlp_with_patch(model, x, return_cache=True)
    assert isinstance(cache, MLPForwardCache)
    return cache.post_relu if site == "post_relu" else cache.pre_relu


def _apply_subspace_patch(
    h_base: torch.Tensor,
    h_source: torch.Tensor,
    basis: np.ndarray,
) -> torch.Tensor:
    """h_base + V V^T (h_source - h_base), V rows orthonormal [k, hidden]."""
    v = torch.from_numpy(basis.astype(np.float32)).to(h_base.device)
    delta = h_source - h_base
    proj = (delta @ v.T) @ v
    return h_base + proj


def probe_subspace_basis(
    hidden_by_elem: np.ndarray,
    char_vec: np.ndarray,
    *,
    variance_frac: float = 0.95,
) -> tuple[np.ndarray, int]:
    """Linear probe χ(g) from hidden[g]; SVD → orthonormal basis V [k, hidden]."""
    n, h = hidden_by_elem.shape
    y = np.column_stack([char_vec.real, char_vec.imag]).astype(np.float64)
    x = hidden_by_elem.astype(np.float64)
    x_aug = np.concatenate([x, np.ones((n, 1))], axis=1)
    coef, _, _, _ = np.linalg.lstsq(x_aug, y, rcond=None)
    w = coef[:-1].T  # [2, hidden]
    _, s, vt = np.linalg.svd(w, full_matrices=False)
    if s.sum() < 1e-15:
        k = 1
    else:
        cum = np.cumsum(s**2) / np.sum(s**2)
        k = int(np.searchsorted(cum, variance_frac) + 1)
        k = max(1, min(k, vt.shape[0]))
    return vt[:k], k


def _svd_basis_from_weight_matrix(
    w: np.ndarray,
    *,
    variance_frac: float = 0.95,
    min_subspace_dim: int = 1,
) -> tuple[np.ndarray, int]:
    """SVD(W) → top-k right singular vectors covering ``variance_frac`` of ||W||_F²."""
    _, s, vt = np.linalg.svd(w, full_matrices=False)
    if s.sum() < 1e-15:
        k = min_subspace_dim
    else:
        cum = np.cumsum(s**2) / np.sum(s**2)
        k = int(np.searchsorted(cum, variance_frac) + 1)
        k = max(min_subspace_dim, min(k, vt.shape[0]))
    return vt[:k], k


def _lstsq_onehot_probe_basis(
    hidden_by_elem: np.ndarray,
    labels: np.ndarray,
    *,
    variance_frac: float = 0.95,
    min_subspace_dim: int = 1,
) -> tuple[np.ndarray, int]:
    """Fallback: linear lstsq on one-hot π labels → SVD(W)."""
    uniq = np.unique(labels)
    n_classes = len(uniq)
    label_to_col = {int(u): i for i, u in enumerate(uniq)}
    y = np.zeros((labels.shape[0], n_classes), dtype=np.float64)
    for i, lab in enumerate(labels):
        y[i, label_to_col[int(lab)]] = 1.0
    x = hidden_by_elem.astype(np.float64)
    x_aug = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
    coef, _, _, _ = np.linalg.lstsq(x_aug, y, rcond=None)
    w = coef[:-1].T
    return _svd_basis_from_weight_matrix(
        w, variance_frac=variance_frac, min_subspace_dim=min_subspace_dim
    )


def class_mean_centroid_matrix(
    hidden_by_elem: np.ndarray,
    labels: np.ndarray,
    *,
    fit_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-class mean hiddens μ_c; returns (centroids [C, d], class ids [C])."""
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    x = hidden_by_elem.astype(np.float64)
    x_fit, labels_fit = _masked_probe_rows(x, labels, fit_mask)
    class_ids = np.unique(labels_fit)
    centroids = np.stack(
        [x_fit[labels_fit == c].mean(axis=0) for c in class_ids],
        axis=0,
    )
    return centroids, class_ids


def _nearest_centroid_accuracy(
    x: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    class_ids: np.ndarray,
) -> float:
    if centroids.shape[0] <= 1:
        return 1.0
    dists = ((x[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    pred_idx = np.argmin(dists, axis=1)
    pred = class_ids[pred_idx]
    return float(np.mean(pred == labels))


def probe_subspace_basis_from_class_means(
    hidden_by_elem: np.ndarray,
    labels: np.ndarray,
    *,
    variance_frac: float = 0.95,
    min_subspace_dim: int = 1,
    fit_mask: np.ndarray | None = None,
) -> ProbeSubspaceResult:
    """Class-mean PCA: SVD on centered centroid matrix M ∈ R^{C×d}."""
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    hidden_dim = hidden_by_elem.shape[1]
    x = hidden_by_elem.astype(np.float64)
    x_fit, labels_fit = _masked_probe_rows(x, labels, fit_mask)
    centroids, class_ids = class_mean_centroid_matrix(
        hidden_by_elem, labels, fit_mask=fit_mask
    )
    n_classes = int(len(class_ids))
    k_cap = min(n_classes, hidden_dim)
    if n_classes <= 1:
        basis = np.zeros((min_subspace_dim, hidden_dim), dtype=np.float64)
        basis[0, 0] = 1.0
        return ProbeSubspaceResult(
            basis=basis,
            k=min_subspace_dim,
            probe_accuracy=1.0,
            n_classes=n_classes,
        )
    m_centered = centroids - centroids.mean(axis=0, keepdims=True)
    basis, k = _svd_basis_from_weight_matrix(
        m_centered,
        variance_frac=variance_frac,
        min_subspace_dim=min_subspace_dim,
    )
    k = max(min_subspace_dim, min(k, k_cap))
    basis = basis[:k]
    probe_accuracy = _nearest_centroid_accuracy(x_fit, labels_fit, centroids, class_ids)
    return ProbeSubspaceResult(
        basis=basis,
        k=k,
        probe_accuracy=probe_accuracy,
        n_classes=n_classes,
    )


def probe_subspace_basis_from_labels(
    hidden_by_elem: np.ndarray,
    labels: np.ndarray,
    *,
    variance_frac: float = 0.95,
    test_fraction: float = 0.2,
    seed: int = 42,
    min_subspace_dim: int = 1,
    fit_mask: np.ndarray | None = None,
    mlp_hidden_layer_sizes: tuple[int, ...] = (32,),
) -> ProbeSubspaceResult:
    """Logistic probe πᵢ(a) → SVD(W); Zheng et al. subspace IIA protocol.

    By default, probe and subspace directions are fit on all ring elements (the
    IIA carrier set). Pass ``fit_mask`` to restrict fitting to a subset of
    elements; ``probe_accuracy`` is then in-sample accuracy on that subset.
    """
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    uniq = np.unique(labels)
    n_classes = int(len(uniq))
    hidden_dim = hidden_by_elem.shape[1]
    x = hidden_by_elem.astype(np.float64)
    x_fit, labels_fit = _masked_probe_rows(x, labels, fit_mask)
    k_cap = min(n_classes, hidden_dim)

    if n_classes <= 1:
        basis = np.zeros((min_subspace_dim, hidden_dim), dtype=np.float64)
        basis[0, 0] = 1.0
        if min_subspace_dim > 1:
            basis[1, 1 if hidden_dim > 1 else 0] = 1.0
        mlp_basis, mlp_k, mlp_patch_k, mlp_acc, mlp_w0, mlp_b0 = fit_mlp_probe_subspace(
            hidden_by_elem,
            labels,
            variance_frac=variance_frac,
            test_fraction=test_fraction,
            seed=seed,
            min_subspace_dim=min_subspace_dim,
            logistic_k=min_subspace_dim,
            fit_mask=fit_mask,
            hidden_layer_sizes=mlp_hidden_layer_sizes,
        )
        return ProbeSubspaceResult(
            basis=basis,
            k=min_subspace_dim,
            probe_accuracy=1.0,
            n_classes=n_classes,
            mlp_probe_accuracy=mlp_acc,
            mlp_basis=mlp_basis,
            mlp_k=mlp_k,
            mlp_patch_k=mlp_patch_k,
            mlp_w0=mlp_w0,
            mlp_b0=mlp_b0,
        )

    probe_accuracy = float("nan")
    mlp_basis: np.ndarray | None = None
    mlp_k = mlp_patch_k = 0
    mlp_acc = float("nan")

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score

        clf = LogisticRegression(max_iter=1000, solver="lbfgs")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(x_fit, labels_fit)
        probe_accuracy = float(accuracy_score(labels_fit, clf.predict(x_fit)))
        w = clf.coef_.astype(np.float64)
        basis, k = _svd_basis_from_weight_matrix(
            w, variance_frac=variance_frac, min_subspace_dim=min_subspace_dim
        )
        k = max(min_subspace_dim, min(k, k_cap))
        basis = basis[:k]
        mlp_basis, mlp_k, mlp_patch_k, mlp_acc, mlp_w0, mlp_b0 = fit_mlp_probe_subspace(
            hidden_by_elem,
            labels,
            variance_frac=variance_frac,
            test_fraction=test_fraction,
            seed=seed,
            min_subspace_dim=min_subspace_dim,
            logistic_k=k,
            fit_mask=fit_mask,
            hidden_layer_sizes=mlp_hidden_layer_sizes,
        )
        return ProbeSubspaceResult(
            basis=basis,
            k=k,
            probe_accuracy=probe_accuracy,
            n_classes=n_classes,
            mlp_probe_accuracy=mlp_acc,
            mlp_basis=mlp_basis,
            mlp_k=mlp_k,
            mlp_patch_k=mlp_patch_k,
            mlp_w0=mlp_w0,
            mlp_b0=mlp_b0,
        )
    except Exception as exc:
        _logger.warning("logistic probe failed (%s); falling back to lstsq one-hot", exc)
        basis, k = _lstsq_onehot_probe_basis(
            x_fit,
            labels_fit,
            variance_frac=variance_frac,
            min_subspace_dim=min_subspace_dim,
        )
        k = max(min_subspace_dim, min(k, k_cap))
        basis = basis[:k]
        uniq_fb = np.unique(labels_fit)
        label_to_col = {int(u): i for i, u in enumerate(uniq_fb)}
        y_oh = np.zeros((labels_fit.shape[0], len(uniq_fb)), dtype=np.float64)
        for i, lab in enumerate(labels_fit):
            y_oh[i, label_to_col[int(lab)]] = 1.0
        x_aug = np.concatenate([x_fit, np.ones((x_fit.shape[0], 1))], axis=1)
        coef, _, _, _ = np.linalg.lstsq(x_aug, y_oh, rcond=None)
        pred_idx = np.argmax(x_aug @ coef, axis=1)
        true_idx = np.array([label_to_col[int(lab)] for lab in labels_fit])
        probe_accuracy = float(np.mean(pred_idx == true_idx))
        mlp_basis, mlp_k, mlp_patch_k, mlp_acc, mlp_w0, mlp_b0 = fit_mlp_probe_subspace(
            hidden_by_elem,
            labels,
            variance_frac=variance_frac,
            test_fraction=test_fraction,
            seed=seed,
            min_subspace_dim=min_subspace_dim,
            logistic_k=k,
            fit_mask=fit_mask,
            hidden_layer_sizes=mlp_hidden_layer_sizes,
        )
        return ProbeSubspaceResult(
            basis=basis,
            k=k,
            probe_accuracy=probe_accuracy,
            n_classes=n_classes,
            mlp_probe_accuracy=mlp_acc,
            mlp_basis=mlp_basis,
            mlp_k=mlp_k,
            mlp_patch_k=mlp_patch_k,
            mlp_w0=mlp_w0,
            mlp_b0=mlp_b0,
        )


def probe_subspace_basis_from_vector_labels(
    hidden_by_elem: np.ndarray,
    labels_2d: np.ndarray,
    *,
    variance_frac: float = 0.95,
    min_subspace_dim: int = 1,
    test_fraction: float = 0.2,
    seed: int = 42,
    fit_mask: np.ndarray | None = None,
    mlp_hidden_layer_sizes: tuple[int, ...] = (32,),
) -> ProbeSubspaceResult:
    """Multi-output linear probe on coordinate vector (a,b,c,d) → SVD(W).

    ``probe_accuracy`` is mean per-coordinate in-sample exact-match accuracy
    (integer labels), comparable to scalar logistic probe accuracy.
    """
    y = np.asarray(labels_2d, dtype=np.float64)
    if y.ndim != 2:
        raise ValueError(f"labels_2d must be [n, d], got shape {y.shape}")
    n_coords = y.shape[1]
    hidden_dim = hidden_by_elem.shape[1]
    x = hidden_by_elem.astype(np.float64)
    x_fit, y_fit = _masked_probe_rows(x, y, fit_mask)
    x_aug = np.concatenate([x_fit, np.ones((x_fit.shape[0], 1))], axis=1)
    coef, _, _, _ = np.linalg.lstsq(x_aug, y_fit, rcond=None)
    w = coef[:-1].T  # [n_coords, hidden]
    basis, k = _svd_basis_from_weight_matrix(
        w, variance_frac=variance_frac, min_subspace_dim=min_subspace_dim
    )
    k = max(min_subspace_dim, min(k, hidden_dim))
    basis = basis[:k]
    pred = x_aug @ coef
    y_int = np.rint(y_fit).astype(np.int64)
    pred_int = np.rint(pred).astype(np.int64)
    coord_acc = [
        float(np.mean(pred_int[:, j] == y_int[:, j])) for j in range(n_coords)
    ]
    probe_accuracy = float(np.mean(coord_acc)) if coord_acc else float("nan")
    mlp_basis, mlp_k, mlp_patch_k, mlp_acc, mlp_w0, mlp_b0 = fit_mlp_probe_subspace(
        hidden_by_elem,
        y,
        variance_frac=variance_frac,
        test_fraction=test_fraction,
        seed=seed,
        min_subspace_dim=min_subspace_dim,
        logistic_k=k,
        fit_mask=fit_mask,
        hidden_layer_sizes=mlp_hidden_layer_sizes,
    )
    return ProbeSubspaceResult(
        basis=basis,
        k=k,
        probe_accuracy=probe_accuracy,
        n_classes=n_coords,
        mlp_probe_accuracy=mlp_acc,
        mlp_basis=mlp_basis,
        mlp_k=mlp_k,
        mlp_patch_k=mlp_patch_k,
        mlp_w0=mlp_w0,
        mlp_b0=mlp_b0,
    )


def random_orthonormal_basis(hidden_dim: int, k: int, rng: np.random.Generator) -> np.ndarray:
    m = rng.standard_normal((k, hidden_dim))
    q, _ = np.linalg.qr(m.T)
    return q.T[:k]


def identity_element(mult_table: np.ndarray) -> int:
    """Индекс двусторонней единицы в таблице умножения."""
    n = mult_table.shape[0]
    ar = np.arange(n, dtype=int)
    for e in range(n):
        if np.array_equal(mult_table[e], ar) and np.array_equal(mult_table[:, e], ar):
            return e
    raise ValueError("identity not found in multiplication table")


def conjugacy_class_labels(mult_table: np.ndarray) -> np.ndarray:
    """Метка класса сопряжённости для каждого g (все χ_j совпадают внутри класса)."""
    _, _, classes, _ = conjugacy_data(mult_table)
    labels = np.zeros(mult_table.shape[0], dtype=np.int32)
    for i, cls in enumerate(classes):
        for g in cls:
            labels[g] = i
    return labels


def _vec_key(vec: np.ndarray, *, decimals: int = 4) -> tuple[float, ...]:
    return tuple(round(float(x), decimals) for x in np.asarray(vec, dtype=np.float64).ravel())


def component_signatures_from_hidden(
    hidden_by_elem: np.ndarray,
    chars: dict[str, np.ndarray],
    nontriv: tuple[str, ...],
    *,
    variance_frac: float = 0.95,
) -> dict[str, np.ndarray]:
    """Координаты элемента g в probe-подпространствах V_i (аналог π_i для групп).

    Скалярные χ_i(g) — класс-функции: при совпадении всех χ_j, j≠i, совпадает и χ_i.
    Для умножения в группе используем многомерные проекции h(g) на SVD-базис пробы χ_i.
    """
    sigs: dict[str, np.ndarray] = {}
    for name in nontriv:
        basis, _ = probe_subspace_basis(
            hidden_by_elem, chars[name], variance_frac=variance_frac
        )
        sigs[name] = hidden_by_elem @ basis.T
    return sigs


def sample_matched_pairs(
    n_elements: int,
    signatures: dict[str, np.ndarray],
    nontriv: tuple[str, ...],
    target: str,
    *,
    max_pairs: int,
    rng: np.random.Generator,
    allowed_base_pairs: set[tuple[int, int]] | None = None,
    allowed_counterfactual_pairs: set[tuple[int, int]] | None = None,
) -> list[tuple[int, int, int]]:
    """(a, a_prime, b): π_j(a)=π_j(a′) для j≠target, π_target(a)≠π_target(a′).

    ``signatures[name]`` — массив [n, k_i] probe-координат (см. ``component_signatures_from_hidden``).
    If ``allowed_base_pairs`` is set, only pairs whose base input ``(a, b)`` is in that set are kept.
    If ``allowed_counterfactual_pairs`` is set, only pairs whose counterfactual input ``(a', b)``
    is in that set are kept (closes train-set leakage on the patched operand).
    """
    others = [n for n in nontriv if n != target]
    buckets: dict[tuple[tuple[float, ...], ...], list[int]] = {}
    for g in range(n_elements):
        key = tuple(_vec_key(signatures[o][g]) for o in others)
        buckets.setdefault(key, []).append(g)

    pairs: list[tuple[int, int, int]] = []
    for members in buckets.values():
        if len(members) < 2:
            continue
        by_target: dict[tuple[float, ...], list[int]] = {}
        for g in members:
            tk = _vec_key(signatures[target][g])
            by_target.setdefault(tk, []).append(g)
        target_keys = list(by_target.keys())
        if len(target_keys) < 2:
            continue
        attempts = 0
        max_attempts = max(max_pairs * 50, 500)
        while len(pairs) < max_pairs and attempts < max_attempts:
            attempts += 1
            tk0, tk1 = rng.choice(len(target_keys), size=2, replace=False)
            a = int(rng.choice(by_target[target_keys[tk0]]))
            a_prime = int(rng.choice(by_target[target_keys[tk1]]))
            if a == a_prime:
                continue
            b = int(rng.integers(0, n_elements))
            if allowed_base_pairs is not None and (a, b) not in allowed_base_pairs:
                continue
            if allowed_counterfactual_pairs is not None and (a_prime, b) not in allowed_counterfactual_pairs:
                continue
            pairs.append((a, a_prime, b))
            if len(pairs) >= max_pairs:
                return pairs
    return pairs


def split_ring_elements(
    n_elements: int,
    train_fraction: float = 0.7,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Disjoint partition of ring indices into train / val element sets."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_elements)
    n_train = max(2, int(n_elements * train_fraction))
    if n_train >= n_elements:
        n_train = n_elements - 1
    return perm[:n_train], perm[n_train:]


@dataclass
class DisjointElementSplit:
    """Matched-pair pools with element-level train/val separation."""

    train_elements: np.ndarray
    val_elements: np.ndarray
    train_pairs: list[tuple[int, int, int]]
    val_pairs: list[tuple[int, int, int]]
    bases_per_source: int
    n_sources_train: int


def sample_disjoint_dense_matched_pairs(
    n_elements: int,
    signatures: dict[str, np.ndarray],
    nontriv: tuple[str, ...],
    target: str,
    *,
    bases_per_source: int = 50,
    element_train_fraction: float = 0.7,
    element_seed: int = 42,
    max_val_pairs: int = 5000,
    rng: np.random.Generator,
) -> DisjointElementSplit:
    """Element-disjoint train/val pools with dense base sampling on train sources.

    Train: counterfactual source ``a_prime`` ∈ R_train; each source paired with
    ``bases_per_source`` random bases ``b``.
    Val: both ``a`` and ``a_prime`` ∈ R_val (unseen activations at patch time).
    """
    train_elems, val_elems = split_ring_elements(
        n_elements, element_train_fraction, element_seed
    )
    train_set = set(int(x) for x in train_elems)
    val_set = set(int(x) for x in val_elems)

    others = [n for n in nontriv if n != target]
    buckets: dict[tuple[tuple[float, ...], ...], list[int]] = {}
    for g in range(n_elements):
        key = tuple(_vec_key(signatures[o][g]) for o in others)
        buckets.setdefault(key, []).append(g)

    sources_train: dict[int, int] = {}
    # Val (a, a') pairs where both live in R_val and differ only on ``target``.
    # Dense base sampling (same bases_per_source as train) — one random b per
    # pair is too sparse for group-ring coeff labels (often ≤10 val pairs).
    val_sources: list[tuple[int, int]] = []

    for members in buckets.values():
        if len(members) < 2:
            continue
        by_target: dict[tuple[float, ...], list[int]] = {}
        for g in members:
            tk = _vec_key(signatures[target][g])
            by_target.setdefault(tk, []).append(g)
        target_keys = list(by_target.keys())
        if len(target_keys) < 2:
            continue
        for i in range(len(target_keys)):
            for j in range(i + 1, len(target_keys)):
                for a in by_target[target_keys[i]]:
                    for a_prime in by_target[target_keys[j]]:
                        if a == a_prime:
                            continue
                        if a_prime in train_set and a_prime not in sources_train:
                            sources_train[a_prime] = int(a)
                        if a in val_set and a_prime in val_set:
                            val_sources.append((int(a), int(a_prime)))
                for a in by_target[target_keys[j]]:
                    for a_prime in by_target[target_keys[i]]:
                        if a == a_prime:
                            continue
                        if a_prime in train_set and a_prime not in sources_train:
                            sources_train[a_prime] = int(a)
                        if a in val_set and a_prime in val_set:
                            val_sources.append((int(a), int(a_prime)))

    # Deduplicate val (a, a') while preserving order.
    seen_val: set[tuple[int, int]] = set()
    unique_val_sources: list[tuple[int, int]] = []
    for pair in val_sources:
        if pair in seen_val:
            continue
        seen_val.add(pair)
        unique_val_sources.append(pair)

    train_pairs: list[tuple[int, int, int]] = []
    for a_prime, a in sources_train.items():
        for _ in range(bases_per_source):
            b = int(rng.integers(0, n_elements))
            train_pairs.append((a, a_prime, b))

    val_pairs: list[tuple[int, int, int]] = []
    for a, a_prime in unique_val_sources:
        for _ in range(bases_per_source):
            b = int(rng.integers(0, n_elements))
            val_pairs.append((a, a_prime, b))

    if len(val_pairs) > max_val_pairs:
        pick = rng.choice(len(val_pairs), size=max_val_pairs, replace=False)
        val_pairs = [val_pairs[int(i)] for i in pick]

    rng.shuffle(train_pairs)
    rng.shuffle(val_pairs)

    return DisjointElementSplit(
        train_elements=train_elems,
        val_elements=val_elems,
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        bases_per_source=bases_per_source,
        n_sources_train=len(sources_train),
    )


def sample_matched_pairs_conjugacy(
    class_labels: np.ndarray,
    *,
    max_pairs: int,
    rng: np.random.Generator,
) -> list[tuple[int, int, int]]:
    """Пары (a, a′, b) в одном классе сопряжённости, a≠a′.

    Для групп с 1D характерами совпадение всех π_j, j≠i, эквивалентно одному классу;
    counterfactual a′·b отличается от a·b при a≠a′.
    """
    n = class_labels.shape[0]
    by_class: dict[int, list[int]] = {}
    for g in range(n):
        by_class.setdefault(int(class_labels[g]), []).append(g)

    pairs: list[tuple[int, int, int]] = []
    classes = [c for c, mem in by_class.items() if len(mem) >= 2]
    if not classes:
        return pairs
    for _ in range(max_pairs):
        c = int(rng.choice(classes))
        mem = by_class[c]
        a, a_prime = rng.choice(mem, size=2, replace=False)
        pairs.append((int(a), int(a_prime), int(rng.integers(0, n))))
    return pairs


@torch.no_grad()
def evaluate_accuracy(model: MLP, mult_table: np.ndarray, device: str) -> float:
    n = mult_table.shape[0]
    model.eval()
    correct = 0
    for a in range(n):
        for b in range(n):
            x = _make_one_hot_pair(a, b, n, torch.device(device))
            pred = int(forward_mlp_with_patch(model, x).argmax(dim=1).item())
            if pred == int(mult_table[a, b]):
                correct += 1
    return correct / (n * n)


@torch.no_grad()
def run_iia_for_irrep(
    model: MLP,
    mult_table: np.ndarray,
    chars: dict[str, np.ndarray],
    nontriv: tuple[str, ...],
    target: str,
    *,
    device: str,
    patch_site: PatchSite = "post_relu",
    max_pairs: int = 200,
    rng: np.random.Generator,
    hidden_by_elem: np.ndarray | None = None,
) -> IIAMetrics:
    n = mult_table.shape[0]
    if hidden_by_elem is None:
        resp = compute_hidden_responses(model, n, device=device)
        hidden_by_elem = resp.post_relu.mean(axis=2).T  # [n, hidden]

    basis, k = probe_subspace_basis(hidden_by_elem, chars[target])
    rand_basis = random_orthonormal_basis(hidden_by_elem.shape[1], k, rng)

    class_labels = conjugacy_class_labels(mult_table)
    pairs = sample_matched_pairs_conjugacy(class_labels, max_pairs=max_pairs, rng=rng)
    if not pairs:
        return IIAMetrics(
            irrep=target,
            raw_iia=0.0,
            n_pairs=0,
            subspace_iia=0.0,
            random_subspace_iia=0.0,
            subspace_dim=k,
            error_preservation=0.0,
            error_preservation_chance=_chance_preservation(chars[target]),
            conditional_ntp=0.0,
        )

    dev = torch.device(device)
    raw_ok = sub_ok = rand_ok = 0
    err_preserve = err_total = 0
    cond_ntp = cond_total = 0

    for a, a_prime, b in pairs:
        true_cf = int(mult_table[a_prime, b])
        true_base = int(mult_table[a, b])

        h_base = hidden_for_pair(model, a, b, n, dev, site=patch_site)
        h_src = hidden_for_pair(model, a_prime, b, n, dev, site=patch_site)

        x_base = _make_one_hot_pair(a, b, n, dev)
        base_pred = int(forward_mlp_with_patch(model, x_base).argmax(dim=1).item())

        for mode, h_patch in (
            ("full", h_src),
            ("sub", _apply_subspace_patch(h_base, h_src, basis)),
            ("rand", _apply_subspace_patch(h_base, h_src, rand_basis)),
        ):
            logits = forward_mlp_with_patch(
                model, x_base, patch_value=h_patch, patch_site=patch_site
            )
            pred = int(logits.argmax(dim=1).item())
            if pred == true_cf:
                if mode == "full":
                    raw_ok += 1
                elif mode == "sub":
                    sub_ok += 1
                else:
                    rand_ok += 1

        if base_pred != true_base:
            err_total += 1
            logits = forward_mlp_with_patch(
                model, x_base, patch_value=h_src, patch_site=patch_site
            )
            pred_p = int(logits.argmax(dim=1).item())
            preserved = all(
                _char_key_at(chars, o, pred_p) == _char_key_at(chars, o, base_pred)
                for o in nontriv
                if o != target
            )
            if preserved:
                err_preserve += 1
        else:
            cond_total += 1
            logits = forward_mlp_with_patch(
                model, x_base, patch_value=h_src, patch_site=patch_site
            )
            pred_p = int(logits.argmax(dim=1).item())
            if all(
                _char_key_at(chars, o, pred_p) == _char_key_at(chars, o, base_pred)
                for o in nontriv
                if o != target
            ):
                cond_ntp += 1

    m = len(pairs)
    chance = _chance_preservation(chars[target])
    return IIAMetrics(
        irrep=target,
        raw_iia=raw_ok / m,
        n_pairs=m,
        subspace_iia=sub_ok / m,
        random_subspace_iia=rand_ok / m,
        subspace_dim=k,
        error_preservation=err_preserve / err_total if err_total else float("nan"),
        error_preservation_chance=chance,
        conditional_ntp=cond_ntp / cond_total if cond_total else float("nan"),
    )


@torch.no_grad()
def run_iia_analysis(
    model: MLP,
    mult_table: np.ndarray,
    irrep_specs: dict[str, tuple[int, dict[str, float | complex]]],
    nontriv: tuple[str, ...],
    *,
    checkpoint: str = "",
    device: str = "cpu",
    patch_site: PatchSite = "post_relu",
    max_pairs_per_irrep: int = 200,
    seed: int = 42,
) -> IIAResult:
    rng = np.random.default_rng(seed)
    n = mult_table.shape[0]
    chars = irrep_character_vectors(mult_table, irrep_specs, nontriv)
    e = identity_element(mult_table)
    dev = torch.device(device)
    hidden_by_elem = np.stack(
        [
            hidden_for_pair(model, g, e, n, dev, site=patch_site)
            .cpu()
            .numpy()
            .ravel()
            for g in range(n)
        ],
        axis=0,
    )

    test_acc = evaluate_accuracy(model, mult_table, device)
    per: list[IIAMetrics] = []
    for name in nontriv:
        per.append(
            run_iia_for_irrep(
                model,
                mult_table,
                chars,
                nontriv,
                name,
                device=device,
                patch_site=patch_site,
                max_pairs=max_pairs_per_irrep,
                rng=rng,
                hidden_by_elem=hidden_by_elem,
            )
        )

    raw_vals = [m.raw_iia for m in per if m.n_pairs > 0]
    ratios = [
        m.error_preservation / m.error_preservation_chance
        for m in per
        if m.n_pairs > 0 and m.error_preservation_chance > 0 and not np.isnan(m.error_preservation)
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


def load_model_and_table_from_checkpoint(
    checkpoint: str,
    device: str,
) -> tuple[MLP, dict, np.ndarray, dict, tuple[str, ...]]:
    """Загрузка MLP + mult_table + irrep_specs + nontriv для A5/A6/PSL2."""
    from .alternating_group import build_alternating_mult_table
    from .group_characters import (
        PSL2_Q11_NONTRIV,
        PSL2_Q11_SPECS,
        PSL2_Q7_NONTRIV,
        PSL2_Q7_SPECS,
        PSL2_Q8_NONTRIV,
        PSL2_Q8_SPECS,
        _remap_a5_specs,
        _remap_a6_specs,
    )
    from .psl2_group import build_psl2_mult_table

    model, ckpt = load_alt_group_mlp(checkpoint, device)
    cfg = ckpt.get("config", {})
    if cfg.get("field_order") is not None:
        q = int(cfg["field_order"])
        mult_table, _ = build_psl2_mult_table(q)
        if q == 5:
            specs, nontriv = _remap_a5_specs(mult_table)
        elif q == 7:
            specs, nontriv = PSL2_Q7_SPECS, PSL2_Q7_NONTRIV
        elif q == 8:
            specs, nontriv = PSL2_Q8_SPECS, PSL2_Q8_NONTRIV
        elif q == 9:
            specs, nontriv = _remap_a6_specs(mult_table)
        elif q == 11:
            specs, nontriv = PSL2_Q11_SPECS, PSL2_Q11_NONTRIV
        else:
            raise ValueError(f"no irrep specs for PSL(2,{q})")
    elif cfg.get("group_n") is not None:
        gn = int(cfg["group_n"])
        mult_table, _ = build_alternating_mult_table(gn)
        if gn == 5:
            specs, nontriv = _remap_a5_specs(mult_table)
        elif gn == 6:
            specs, nontriv = _remap_a6_specs(mult_table)
        else:
            raise ValueError(f"no irrep specs for A_{gn}")
    else:
        raise KeyError("cannot infer group from checkpoint config")
    return model, ckpt, mult_table, specs, nontriv
