"""Distributed Alignment Search (DAS) for ring Transformer IIA.

Mirrors pyvene ``LowRankRotatedSpaceIntervention`` (Wu et al. 2023; Geiger et al.
2023) without wrapping HuggingFace models:

    h' = h_base + ((h_src - h_base) @ W) @ W.T

where W ∈ R^{d×k} has orthonormal columns (learned rotation + low-rank
interchange in rotated coordinates).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .geometry_audit import polar_feature_matrix
from .iia import _apply_subspace_patch, probe_subspace_basis_from_labels, random_orthonormal_basis
from .models.group_transformer import GroupTransformer
from .trained_patcher_iia import (
    MatchedPairHiddens,
    collect_disjoint_dense_pair_hiddens,
    forward_group_transformer_with_patch_grad,
    split_pair_indices,
)
from .transformer_iia import TransformerPatchSite


class LowRankRotateLayer(nn.Module):
    """Linear map x ↦ x @ W with W ∈ R^{d×k}; columns re-orthogonalized via QR."""

    def __init__(self, d_model: int, rank: int, *, init_orth: bool = True) -> None:
        super().__init__()
        self.d_model = d_model
        self.rank = rank
        weight = torch.empty(d_model, rank)
        if init_orth:
            nn.init.orthogonal_(weight)
        else:
            nn.init.normal_(weight, std=0.02)
        self.weight = nn.Parameter(weight)

    def project_orthogonal_(self) -> None:
        with torch.no_grad():
            q, _ = torch.linalg.qr(self.weight, mode="reduced")
            self.weight.copy_(q)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight


class LowRankDASIntervention(nn.Module):
    """Trainable low-rank rotated interchange patch (pyvene-compatible math)."""

    def __init__(self, d_model: int, rank: int, *, init_orth: bool = True) -> None:
        super().__init__()
        self.d_model = d_model
        self.rank = rank
        self.rotate_layer = LowRankRotateLayer(d_model, rank, init_orth=init_orth)

    def forward(self, h_base: torch.Tensor, h_src: torch.Tensor) -> torch.Tensor:
        if h_base.dim() == 1:
            h_base = h_base.unsqueeze(0)
            h_src = h_src.unsqueeze(0)
        rotated_base = self.rotate_layer(h_base)
        rotated_src = self.rotate_layer(h_src)
        delta_rot = rotated_src - rotated_base
        return h_base + delta_rot @ self.rotate_layer.weight.T

    def init_from_probe_basis(self, basis: np.ndarray) -> None:
        """Initialize W from logistic-probe SVD basis rows [k, d]."""
        b = np.asarray(basis, dtype=np.float32)
        k = min(self.rank, b.shape[0])
        w = torch.zeros(self.d_model, self.rank)
        w[:, :k] = torch.from_numpy(b[:k].T)
        with torch.no_grad():
            self.rotate_layer.weight.copy_(w)
        self.rotate_layer.project_orthogonal_()


class QuadraticDASIntervention(nn.Module):
    """Linear DAS + elementwise-square correction in rotated interchange coords."""

    def __init__(self, d_model: int, rank: int, *, init_orth: bool = True) -> None:
        super().__init__()
        self.d_model = d_model
        self.rank = rank
        self.rotate_layer = LowRankRotateLayer(d_model, rank, init_orth=init_orth)
        quad_out = torch.zeros(rank, d_model)
        self.quad_out = nn.Parameter(quad_out)

    def forward(self, h_base: torch.Tensor, h_src: torch.Tensor) -> torch.Tensor:
        if h_base.dim() == 1:
            h_base = h_base.unsqueeze(0)
            h_src = h_src.unsqueeze(0)
        rotated_base = self.rotate_layer(h_base)
        rotated_src = self.rotate_layer(h_src)
        delta_rot = rotated_src - rotated_base
        linear = delta_rot @ self.rotate_layer.weight.T
        quad = (delta_rot**2) @ self.quad_out
        return h_base + linear + quad

    def init_from_probe_basis(self, basis: np.ndarray) -> None:
        b = np.asarray(basis, dtype=np.float32)
        k = min(self.rank, b.shape[0])
        w = torch.zeros(self.d_model, self.rank)
        w[:, :k] = torch.from_numpy(b[:k].T)
        with torch.no_grad():
            self.rotate_layer.weight.copy_(w)
        self.rotate_layer.project_orthogonal_()


class BilinearDASIntervention(nn.Module):
    """DAS diff + gated bilinear term (A h_b) ⊙ (C h_s) in rank space."""

    def __init__(self, d_model: int, rank: int, *, init_orth: bool = True) -> None:
        super().__init__()
        self.d_model = d_model
        self.rank = rank
        self.rotate_layer = LowRankRotateLayer(d_model, rank, init_orth=init_orth)
        self.base_proj = LowRankRotateLayer(d_model, rank, init_orth=init_orth)
        self.src_proj = LowRankRotateLayer(d_model, rank, init_orth=init_orth)
        self.bilinear_gate = nn.Parameter(torch.zeros(rank))

    def forward(self, h_base: torch.Tensor, h_src: torch.Tensor) -> torch.Tensor:
        if h_base.dim() == 1:
            h_base = h_base.unsqueeze(0)
            h_src = h_src.unsqueeze(0)
        rotated_base = self.rotate_layer(h_base)
        rotated_src = self.rotate_layer(h_src)
        delta_rot = rotated_src - rotated_base
        bilinear_rot = self.base_proj(h_base) * self.src_proj(h_src)
        combined = delta_rot + self.bilinear_gate * bilinear_rot
        return h_base + combined @ self.rotate_layer.weight.T

    def init_from_probe_basis(self, basis: np.ndarray) -> None:
        b = np.asarray(basis, dtype=np.float32)
        k = min(self.rank, b.shape[0])
        w = torch.zeros(self.d_model, self.rank)
        w[:, :k] = torch.from_numpy(b[:k].T)
        with torch.no_grad():
            self.rotate_layer.weight.copy_(w)
            self.base_proj.weight.copy_(w)
            self.src_proj.weight.copy_(w)
        self.rotate_layer.project_orthogonal_()
        self.base_proj.project_orthogonal_()
        self.src_proj.project_orthogonal_()


def augment_quadratic_features(points: np.ndarray) -> np.ndarray:
    """Concatenate [h, h ⊙ h] along feature axis."""
    x = np.asarray(points, dtype=np.float64)
    return np.concatenate([x, x * x], axis=1)


def compute_quadratic_lift_matrix(hidden_by_elem: np.ndarray) -> np.ndarray:
    """Least-squares map [h, h⊙h] -> h, shape [2d, d]."""
    aug = augment_quadratic_features(hidden_by_elem)
    hidden = np.asarray(hidden_by_elem, dtype=np.float64)
    lift, _, _, _ = np.linalg.lstsq(aug, hidden, rcond=None)
    return lift.astype(np.float32)


def make_das_intervention(
    kind: str,
    d_model: int,
    rank: int,
    *,
    init_orth: bool = True,
) -> nn.Module:
    if kind == "linear":
        return LowRankDASIntervention(d_model, rank, init_orth=init_orth)
    if kind == "quadratic":
        return QuadraticDASIntervention(d_model, rank, init_orth=init_orth)
    if kind == "bilinear":
        return BilinearDASIntervention(d_model, rank, init_orth=init_orth)
    raise ValueError(f"unknown DAS intervention kind {kind!r}")


def compute_polar_lift_matrix(
    hidden_by_elem: np.ndarray,
    *,
    mode: str = "hyperspherical",
) -> np.ndarray:
    """Least-squares map polar features -> hidden coordinates, shape [F, d]."""
    polar = polar_feature_matrix(hidden_by_elem, mode=mode)
    hidden = np.asarray(hidden_by_elem, dtype=np.float64)
    lift, _, _, _ = np.linalg.lstsq(polar, hidden, rcond=None)
    return lift.astype(np.float32)


def polar_hidden_features(
    hidden: torch.Tensor | np.ndarray,
    *,
    mode: str = "hyperspherical",
) -> torch.Tensor:
    """Detach-and-map hidden vector(s) to fixed polar features (no grad through map)."""
    if isinstance(hidden, torch.Tensor):
        device = hidden.device
        dtype = hidden.dtype
        was_1d = hidden.dim() == 1
        h_np = hidden.detach().cpu().numpy()
    else:
        device = torch.device("cpu")
        dtype = torch.float32
        was_1d = np.asarray(hidden).ndim == 1
        h_np = np.asarray(hidden)
    if h_np.ndim == 1:
        h_np = h_np.reshape(1, -1)
    feats = polar_feature_matrix(h_np, mode=mode).astype(np.float32)
    out = torch.from_numpy(feats).to(device=device, dtype=dtype)
    return out.squeeze(0) if was_1d else out


class PolarPathDASIntervention(nn.Module):
    """Interchange in rotated polar-diff space, then linear lift back to hidden."""

    def __init__(
        self,
        d_model: int,
        polar_dim: int,
        rank: int,
        *,
        polar_mode: str = "hyperspherical",
        init_orth: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.polar_dim = polar_dim
        self.rank = rank
        self.polar_mode = polar_mode
        self.rotate_in = LowRankRotateLayer(polar_dim, rank, init_orth=init_orth)
        project_out = torch.empty(rank, d_model)
        nn.init.xavier_uniform_(project_out)
        self.project_out = nn.Parameter(project_out)

    def forward(self, h_base: torch.Tensor, h_src: torch.Tensor) -> torch.Tensor:
        single = h_base.dim() == 1
        if single:
            h_base = h_base.unsqueeze(0)
            h_src = h_src.unsqueeze(0)
        f_base = polar_hidden_features(h_base, mode=self.polar_mode)
        f_src = polar_hidden_features(h_src, mode=self.polar_mode)
        rotated_base = self.rotate_in(f_base)
        rotated_src = self.rotate_in(f_src)
        delta_rot = rotated_src - rotated_base
        return h_base + delta_rot @ self.project_out

    def init_from_probe_basis(self, basis_polar: np.ndarray) -> None:
        """Initialize polar rotation from logistic-probe rows [k, F]."""
        b = np.asarray(basis_polar, dtype=np.float32)
        k = min(self.rank, b.shape[0])
        w = torch.zeros(self.polar_dim, self.rank)
        w[:, :k] = torch.from_numpy(b[:k].T)
        with torch.no_grad():
            self.rotate_in.weight.copy_(w)
        self.rotate_in.project_orthogonal_()


def _project_das_orthogonal_(das: nn.Module) -> None:
    if hasattr(das, "rotate_layer"):
        das.rotate_layer.project_orthogonal_()
    if hasattr(das, "base_proj"):
        das.base_proj.project_orthogonal_()
    if hasattr(das, "src_proj"):
        das.src_proj.project_orthogonal_()
    elif hasattr(das, "rotate_in"):
        das.rotate_in.project_orthogonal_()


@dataclass
class DASMetrics:
    component: str
    subspace_dim: int
    n_train: int
    n_val: int
    train_iia: float
    val_das_iia: float
    val_logistic_subspace_iia: float
    val_random_rotation_iia: float
    val_raw_iia: float
    train_epochs: int
    best_val_iia: float
    val_polar_logistic_subspace_iia: float = float("nan")
    polar_mode: str = ""
    intervention: str = "linear"
    val_quadratic_probe_iia: float = float("nan")


@dataclass
class PolarDASDisjointRun:
    metrics: DASMetrics
    das: PolarPathDASIntervention
    train_data: MatchedPairHiddens
    val_data: MatchedPairHiddens
    split_meta: dict
    polar_lift: np.ndarray


@dataclass
class DASRun:
    metrics: DASMetrics
    das: LowRankDASIntervention
    data: MatchedPairHiddens
    train_idx: np.ndarray
    val_idx: np.ndarray


@dataclass
class DASDisjointRun:
    metrics: DASMetrics
    das: LowRankDASIntervention
    train_data: MatchedPairHiddens
    val_data: MatchedPairHiddens
    split_meta: dict


def _evaluate_das_iia(
    das: nn.Module,
    model: GroupTransformer,
    data: MatchedPairHiddens,
    idx: np.ndarray,
    *,
    device: str,
    patch_site: TransformerPatchSite = "resid_pre_0",
) -> float:
    ok = _evaluate_das_correctness(
        das, model, data, idx, device=device, patch_site=patch_site
    )
    if len(ok) == 0:
        return float("nan")
    return float(ok.mean())


def _evaluate_das_correctness(
    das: nn.Module,
    model: GroupTransformer,
    data: MatchedPairHiddens,
    idx: np.ndarray,
    *,
    device: str,
    patch_site: TransformerPatchSite = "resid_pre_0",
) -> np.ndarray:
    if len(idx) == 0:
        return np.array([], dtype=bool)
    dev = torch.device(device)
    das.eval()
    out = np.zeros(len(idx), dtype=bool)
    for j, i in enumerate(idx):
        h_patch = das(data.h_base[i], data.h_src[i])
        x = torch.tensor([[int(data.a[i]), int(data.b[i])]], dtype=torch.long, device=dev)
        pred = int(
            forward_group_transformer_with_patch_grad(
                model, x, h_patch, patch_site=patch_site
            )
            .argmax(dim=1)
            .item()
        )
        out[j] = pred == int(data.true_cf[i])
    return out


def _evaluate_basis_correctness(
    model: GroupTransformer,
    data: MatchedPairHiddens,
    idx: np.ndarray,
    basis: np.ndarray,
    *,
    device: str,
    patch_site: TransformerPatchSite = "resid_pre_0",
) -> np.ndarray:
    if len(idx) == 0:
        return np.array([], dtype=bool)
    dev = torch.device(device)
    out = np.zeros(len(idx), dtype=bool)
    for j, i in enumerate(idx):
        h_b = data.h_base[i].reshape(1, -1)
        h_s = data.h_src[i].reshape(1, -1)
        h_patch = _apply_subspace_patch(h_b, h_s, basis)
        x = torch.tensor([[int(data.a[i]), int(data.b[i])]], dtype=torch.long, device=dev)
        pred = int(
            forward_group_transformer_with_patch_grad(
                model, x, h_patch, patch_site=patch_site
            )
            .argmax(dim=1)
            .item()
        )
        out[j] = pred == int(data.true_cf[i])
    return out


def _evaluate_basis_iia(
    model: GroupTransformer,
    data: MatchedPairHiddens,
    idx: np.ndarray,
    basis: np.ndarray,
    *,
    device: str,
    patch_site: TransformerPatchSite = "resid_pre_0",
) -> float:
    ok = _evaluate_basis_correctness(
        model, data, idx, basis, device=device, patch_site=patch_site
    )
    if len(ok) == 0:
        return float("nan")
    return float(ok.mean())


def _evaluate_polar_basis_iia(
    model: GroupTransformer,
    data: MatchedPairHiddens,
    idx: np.ndarray,
    basis_polar: np.ndarray,
    polar_lift: np.ndarray,
    *,
    polar_mode: str,
    device: str,
    patch_site: TransformerPatchSite = "resid_pre_0",
) -> float:
    if len(idx) == 0:
        return float("nan")
    dev = torch.device(device)
    w_lift = torch.from_numpy(np.asarray(polar_lift, dtype=np.float32)).to(dev)
    ok = 0
    for i in idx:
        h_b = data.h_base[i].reshape(1, -1)
        h_s = data.h_src[i].reshape(1, -1)
        f_b = polar_hidden_features(h_b, mode=polar_mode)
        f_s = polar_hidden_features(h_s, mode=polar_mode)
        f_patch = _apply_subspace_patch(f_b, f_s, basis_polar)
        polar_delta = f_patch - f_b
        h_patch = h_b + polar_delta @ w_lift
        x = torch.tensor([[int(data.a[i]), int(data.b[i])]], dtype=torch.long, device=dev)
        pred = int(
            forward_group_transformer_with_patch_grad(
                model, x, h_patch, patch_site=patch_site
            )
            .argmax(dim=1)
            .item()
        )
        if pred == int(data.true_cf[i]):
            ok += 1
    return ok / len(idx)


def _evaluate_quadratic_probe_iia(
    model: GroupTransformer,
    data: MatchedPairHiddens,
    idx: np.ndarray,
    basis_quad: np.ndarray,
    quad_lift: np.ndarray,
    *,
    device: str,
    patch_site: TransformerPatchSite = "resid_pre_0",
) -> float:
    if len(idx) == 0:
        return float("nan")
    dev = torch.device(device)
    w_lift = torch.from_numpy(np.asarray(quad_lift, dtype=np.float32)).to(dev)
    ok = 0
    for i in idx:
        h_b = data.h_base[i].reshape(1, -1)
        h_s = data.h_src[i].reshape(1, -1)
        f_b = torch.from_numpy(augment_quadratic_features(h_b.cpu().numpy())).to(dev, dtype=h_b.dtype)
        f_s = torch.from_numpy(augment_quadratic_features(h_s.cpu().numpy())).to(dev, dtype=h_s.dtype)
        f_patch = _apply_subspace_patch(f_b, f_s, basis_quad)
        aug_delta = f_patch - f_b
        h_patch = h_b + aug_delta @ w_lift
        x = torch.tensor([[int(data.a[i]), int(data.b[i])]], dtype=torch.long, device=dev)
        pred = int(
            forward_group_transformer_with_patch_grad(
                model, x, h_patch, patch_site=patch_site
            )
            .argmax(dim=1)
            .item()
        )
        if pred == int(data.true_cf[i]):
            ok += 1
    return ok / len(idx)


def train_das_intervention(
    das: nn.Module,
    model: GroupTransformer,
    data: MatchedPairHiddens,
    train_idx: np.ndarray,
    *,
    device: str,
    epochs: int = 300,
    batch_size: int = 32,
    lr: float = 1e-2,
    patch_site: TransformerPatchSite = "resid_pre_0",
    val_idx: np.ndarray | None = None,
    val_data: MatchedPairHiddens | None = None,
    early_stop_patience: int = 40,
    shuffle_seed: int = 0,
) -> tuple[nn.Module, int, float]:
    dev = torch.device(device)
    eval_data = val_data if val_data is not None else data
    das = das.to(dev)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    opt = torch.optim.Adam(das.parameters(), lr=lr)
    das.eval()
    best_state = {k: v.detach().cpu().clone() for k, v in das.state_dict().items()}
    if val_idx is not None and len(val_idx) > 0:
        best_val = _evaluate_das_iia(
            das, model, eval_data, val_idx, device=device, patch_site=patch_site
        )
    else:
        best_val = -1.0
    stale = 0
    last_epoch = 0

    for epoch in range(epochs):
        last_epoch = epoch + 1
        das.train()
        perm = np.random.default_rng(shuffle_seed + epoch).permutation(train_idx)
        for start in range(0, len(perm), batch_size):
            idx = perm[start : start + batch_size]
            h_b = data.h_base[idx]
            h_s = data.h_src[idx]
            y = torch.as_tensor(data.true_cf[idx], dtype=torch.long, device=dev)
            x = torch.as_tensor(np.stack([data.a[idx], data.b[idx]], axis=1), dtype=torch.long, device=dev)

            h_patch = das(h_b, h_s)
            logits = forward_group_transformer_with_patch_grad(
                model, x, h_patch, patch_site=patch_site
            )
            loss = F.cross_entropy(logits, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            _project_das_orthogonal_(das)

        if val_idx is not None and len(val_idx) > 0:
            val_iia = _evaluate_das_iia(
                das, model, eval_data, val_idx, device=device, patch_site=patch_site
            )
            if val_iia > best_val:
                best_val = val_iia
                best_state = {k: v.detach().cpu().clone() for k, v in das.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= early_stop_patience:
                    break

    if best_state is not None:
        das.load_state_dict(best_state)
    das.eval()
    return das, last_epoch, best_val


def run_das_iia_component(
    model: GroupTransformer,
    data: MatchedPairHiddens,
    target: str,
    *,
    basis: np.ndarray,
    subspace_dim: int,
    device: str,
    patch_site: TransformerPatchSite = "resid_pre_0",
    train_fraction: float = 0.8,
    pair_seed: int = 42,
    train_epochs: int = 300,
    init_from_probe: bool = True,
    shuffle_seed: int = 0,
    rng: np.random.Generator | None = None,
    return_artifacts: bool = False,
) -> DASMetrics | DASRun:
    """Fit DAS rotation on train pairs; report val IIA vs axis-aligned logistic sub."""
    ps = pair_seed
    rng = rng or np.random.default_rng(ps + hash(target) % 1000)
    train_idx, val_idx = split_pair_indices(len(data.a), train_fraction, ps)
    if len(val_idx) == 0:
        val_idx = train_idx[-1:]
        train_idx = train_idx[:-1]

    das = LowRankDASIntervention(model.d_model, subspace_dim, init_orth=not init_from_probe)
    if init_from_probe:
        das.init_from_probe_basis(basis)

    das, epochs_run, best_val = train_das_intervention(
        das,
        model,
        data,
        train_idx,
        device=device,
        epochs=train_epochs,
        patch_site=patch_site,
        val_idx=val_idx,
        shuffle_seed=shuffle_seed,
    )

    rand_basis = random_orthonormal_basis(model.d_model, subspace_dim, rng=rng)
    from .trained_patcher_iia import _evaluate_raw_iia

    metrics = DASMetrics(
        component=target,
        subspace_dim=subspace_dim,
        n_train=len(train_idx),
        n_val=len(val_idx),
        train_iia=_evaluate_das_iia(
            das, model, data, train_idx, device=device, patch_site=patch_site
        ),
        val_das_iia=_evaluate_das_iia(
            das, model, data, val_idx, device=device, patch_site=patch_site
        ),
        val_logistic_subspace_iia=_evaluate_basis_iia(
            model, data, val_idx, basis, device=device, patch_site=patch_site
        ),
        val_random_rotation_iia=_evaluate_basis_iia(
            model, data, val_idx, rand_basis, device=device, patch_site=patch_site
        ),
        val_raw_iia=_evaluate_raw_iia(
            model, data, val_idx, device=device, patch_site=patch_site
        ),
        train_epochs=epochs_run,
        best_val_iia=best_val,
    )
    if return_artifacts:
        return DASRun(
            metrics=metrics,
            das=das,
            data=data,
            train_idx=train_idx,
            val_idx=val_idx,
        )
    return metrics


def run_polar_das_disjoint_dense(
    model: GroupTransformer,
    mult_table: np.ndarray,
    spec,
    target: str,
    hidden_by_elem: np.ndarray,
    *,
    basis_cartesian: np.ndarray,
    subspace_dim: int,
    device: str,
    polar_mode: str = "hyperspherical",
    patch_site: TransformerPatchSite = "resid_pre_0",
    bases_per_source: int = 50,
    element_train_fraction: float = 0.7,
    element_seed: int = 42,
    max_val_pairs: int = 5000,
    seed: int = 42,
    pair_seed: int | None = None,
    train_epochs: int = 300,
    init_from_probe: bool = True,
    rng: np.random.Generator | None = None,
    return_artifacts: bool = False,
) -> DASMetrics | PolarDASDisjointRun:
    """Element-disjoint DAS with interchange path through polar feature space."""
    ps = pair_seed if pair_seed is not None else seed
    rng = rng or np.random.default_rng(ps + hash(target) % 1000)
    labels = spec.labels[target]

    polar_by_elem = polar_feature_matrix(hidden_by_elem, mode=polar_mode)
    polar_dim = int(polar_by_elem.shape[1])
    polar_lift = compute_polar_lift_matrix(hidden_by_elem, mode=polar_mode)
    probe_polar = probe_subspace_basis_from_labels(
        polar_by_elem, labels, min_subspace_dim=subspace_dim
    )

    train_data, val_data, split_meta = collect_disjoint_dense_pair_hiddens(
        model,
        mult_table,
        spec,
        target,
        device=device,
        patch_site=patch_site,
        bases_per_source=bases_per_source,
        element_train_fraction=element_train_fraction,
        element_seed=element_seed,
        max_val_pairs=max_val_pairs,
        rng=rng,
    )
    train_idx = np.arange(len(train_data.a))
    val_idx = np.arange(len(val_data.a))

    das = PolarPathDASIntervention(
        model.d_model,
        polar_dim,
        subspace_dim,
        polar_mode=polar_mode,
        init_orth=not init_from_probe,
    )
    if init_from_probe:
        das.init_from_probe_basis(probe_polar.basis)

    das, epochs_run, best_val = train_das_intervention(
        das,
        model,
        train_data,
        train_idx,
        device=device,
        epochs=train_epochs,
        patch_site=patch_site,
        val_idx=val_idx,
        val_data=val_data,
    )

    rand_basis = random_orthonormal_basis(
        model.d_model, subspace_dim, rng=rng
    )
    from .trained_patcher_iia import _evaluate_raw_iia

    metrics = DASMetrics(
        component=target,
        subspace_dim=subspace_dim,
        n_train=len(train_idx),
        n_val=len(val_idx),
        train_iia=_evaluate_das_iia(
            das, model, train_data, train_idx, device=device, patch_site=patch_site
        ),
        val_das_iia=_evaluate_das_iia(
            das, model, val_data, val_idx, device=device, patch_site=patch_site
        ),
        val_logistic_subspace_iia=_evaluate_basis_iia(
            model, val_data, val_idx, basis_cartesian, device=device, patch_site=patch_site
        ),
        val_random_rotation_iia=_evaluate_basis_iia(
            model, val_data, val_idx, rand_basis, device=device, patch_site=patch_site
        ),
        val_raw_iia=_evaluate_raw_iia(
            model, val_data, val_idx, device=device, patch_site=patch_site
        ),
        train_epochs=epochs_run,
        best_val_iia=best_val,
        val_polar_logistic_subspace_iia=_evaluate_polar_basis_iia(
            model,
            val_data,
            val_idx,
            probe_polar.basis,
            polar_lift,
            polar_mode=polar_mode,
            device=device,
            patch_site=patch_site,
        ),
        polar_mode=polar_mode,
    )
    if return_artifacts:
        return PolarDASDisjointRun(
            metrics=metrics,
            das=das,
            train_data=train_data,
            val_data=val_data,
            split_meta=split_meta,
            polar_lift=polar_lift,
        )
    return metrics


def run_das_disjoint_dense(
    model: GroupTransformer,
    mult_table: np.ndarray,
    spec,
    target: str,
    hidden_by_elem: np.ndarray,
    *,
    basis: np.ndarray,
    subspace_dim: int,
    device: str,
    intervention: str = "linear",
    patch_site: TransformerPatchSite = "resid_pre_0",
    bases_per_source: int = 50,
    element_train_fraction: float = 0.7,
    element_seed: int = 42,
    max_val_pairs: int = 5000,
    seed: int = 42,
    pair_seed: int | None = None,
    train_epochs: int = 300,
    init_from_probe: bool = True,
    rng: np.random.Generator | None = None,
    return_artifacts: bool = False,
) -> DASMetrics | DASDisjointRun:
    """Element-disjoint train/val + dense base sampling per train source."""
    ps = pair_seed if pair_seed is not None else seed
    rng = rng or np.random.default_rng(ps + hash(target) % 1000)
    labels = spec.labels[target]
    aug_hidden = augment_quadratic_features(hidden_by_elem)
    quad_lift = compute_quadratic_lift_matrix(hidden_by_elem)
    probe_quad = probe_subspace_basis_from_labels(
        aug_hidden, labels, min_subspace_dim=subspace_dim
    )

    train_data, val_data, split_meta = collect_disjoint_dense_pair_hiddens(
        model,
        mult_table,
        spec,
        target,
        device=device,
        patch_site=patch_site,
        bases_per_source=bases_per_source,
        element_train_fraction=element_train_fraction,
        element_seed=element_seed,
        max_val_pairs=max_val_pairs,
        rng=rng,
    )
    train_idx = np.arange(len(train_data.a))
    val_idx = np.arange(len(val_data.a))

    das = make_das_intervention(
        intervention, model.d_model, subspace_dim, init_orth=not init_from_probe
    )
    if init_from_probe and hasattr(das, "init_from_probe_basis"):
        das.init_from_probe_basis(basis)

    das, epochs_run, best_val = train_das_intervention(
        das,
        model,
        train_data,
        train_idx,
        device=device,
        epochs=train_epochs,
        patch_site=patch_site,
        val_idx=val_idx,
        val_data=val_data,
    )

    rand_basis = random_orthonormal_basis(model.d_model, subspace_dim, rng=rng)
    from .trained_patcher_iia import _evaluate_raw_iia

    metrics = DASMetrics(
        component=target,
        subspace_dim=subspace_dim,
        n_train=len(train_idx),
        n_val=len(val_idx),
        train_iia=_evaluate_das_iia(
            das, model, train_data, train_idx, device=device, patch_site=patch_site
        ),
        val_das_iia=_evaluate_das_iia(
            das, model, val_data, val_idx, device=device, patch_site=patch_site
        ),
        val_logistic_subspace_iia=_evaluate_basis_iia(
            model, val_data, val_idx, basis, device=device, patch_site=patch_site
        ),
        val_random_rotation_iia=_evaluate_basis_iia(
            model, val_data, val_idx, rand_basis, device=device, patch_site=patch_site
        ),
        val_raw_iia=_evaluate_raw_iia(
            model, val_data, val_idx, device=device, patch_site=patch_site
        ),
        train_epochs=epochs_run,
        best_val_iia=best_val,
        intervention=intervention,
        val_quadratic_probe_iia=_evaluate_quadratic_probe_iia(
            model,
            val_data,
            val_idx,
            probe_quad.basis,
            quad_lift,
            device=device,
            patch_site=patch_site,
        ),
    )
    if return_artifacts:
        return DASDisjointRun(
            metrics=metrics,
            das=das,
            train_data=train_data,
            val_data=val_data,
            split_meta=split_meta,
        )
    return metrics
