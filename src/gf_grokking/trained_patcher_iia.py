"""Train a shared nonlinear hidden-state patcher for strict bottleneck IIA.

Unlike per-pair Adam optimization (``nonlinear_bottleneck_patch``), we fit one
small MLP on a fixed training set of matched pairs and evaluate causal transfer
on held-out pairs with the frozen patcher.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .iia import (
    _apply_subspace_patch,
    nonlinear_bottleneck_patch,
    probe_subspace_basis_from_labels,
    probe_subspace_basis_from_vector_labels,
    sample_disjoint_dense_matched_pairs,
    sample_matched_pairs,
)
from .ring_transformer_iia import min_subspace_dim_for_probe
from .ring_wedderburn import WedderburnSpec, labels_to_signatures
from .transformer_iia import TransformerPatchSite, hidden_at_site
from .models.group_transformer import GroupTransformer


@dataclass
class MatchedPairHiddens:
    """Precomputed hidden states for matched interchange pairs."""

    a: np.ndarray
    a_prime: np.ndarray
    b: np.ndarray
    true_cf: np.ndarray
    h_base: torch.Tensor
    h_src: torch.Tensor


@dataclass
class TrainedPatcherMetrics:
    component: str
    n_train: int
    n_val: int
    train_iia: float
    val_iia: float
    val_adam_bottleneck_iia: float
    val_logistic_subspace_iia: float
    val_random_input_iia: float
    val_raw_iia: float
    logistic_probe_acc: float
    mlp_probe_acc: float
    best_val_iia: float
    train_epochs: int
    subspace_dim: int


class HiddenPatcherMLP(nn.Module):
    """Unconditional residual patcher: h' = h_base + f(h_src)."""

    def __init__(self, d_model: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.d_model = d_model
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, d_model)

    def forward(self, h_base: torch.Tensor, h_src: torch.Tensor) -> torch.Tensor:
        if h_base.dim() == 1:
            h_base = h_base.unsqueeze(0)
            h_src = h_src.unsqueeze(0)
        delta = self.fc2(torch.relu(self.fc1(h_src)))
        return h_base + delta

    def delta_from_src(self, h_src: torch.Tensor) -> torch.Tensor:
        if h_src.dim() == 1:
            h_src = h_src.unsqueeze(0)
        return self.fc2(torch.relu(self.fc1(h_src)))

    def first_layer_weight(self) -> torch.Tensor:
        """Shape (hidden_dim, d_model); columns index h_src coordinates."""
        return self.fc1.weight


def first_layer_l1_norm(patcher: HiddenPatcherMLP, *, source_only: bool = False) -> torch.Tensor:
    del source_only  # unconditional patcher: all fc1 columns read h_src
    return patcher.first_layer_weight().abs().sum()


def count_active_input_dims(
    patcher: HiddenPatcherMLP,
    *,
    threshold: float = 1e-6,
    relative: float | None = None,
) -> int:
    """Input features with non-negligible first-layer weight mass."""
    w = patcher.first_layer_weight().detach().abs().cpu().numpy()
    col_l1 = w.sum(axis=0)
    if relative is not None:
        thr = float(relative) * float(col_l1.max()) if col_l1.max() > 0 else threshold
    else:
        thr = threshold
    return int((col_l1 > thr).sum())


def count_active_source_dims(
    patcher: HiddenPatcherMLP,
    *,
    threshold: float = 1e-6,
    relative: float | None = 0.01,
) -> int:
    """Active h_src columns of the first-layer weight matrix."""
    w = patcher.first_layer_weight().detach().abs().cpu().numpy()
    col_l1 = w.sum(axis=0)
    if relative is not None:
        thr = float(relative) * float(col_l1.max()) if col_l1.max() > 0 else threshold
    else:
        thr = threshold
    return int((col_l1 > thr).sum())


def forward_group_transformer_with_patch_grad(
    model: GroupTransformer,
    x: torch.Tensor,
    patch_value: torch.Tensor,
    *,
    patch_site: TransformerPatchSite = "resid_pre_0",
    patch_pos: int = 0,
) -> torch.Tensor:
    """Differentiable forward with patched activations at ``resid_pre_0``."""
    h = model.embedding(x)
    if patch_site == "resid_pre_0":
        h = h.clone()
        h[:, patch_pos, :] = patch_value
    for layer in model.layers:
        h = layer(h)
    return model.head(h[:, 0])


def collect_matched_pair_hiddens(
    model: GroupTransformer,
    mult_table: np.ndarray,
    spec: WedderburnSpec,
    target: str,
    *,
    device: str,
    patch_site: TransformerPatchSite = "resid_pre_0",
    max_pairs: int = 1000,
    rng: np.random.Generator,
) -> MatchedPairHiddens:
    signatures = labels_to_signatures(spec)
    pairs = sample_matched_pairs(
        mult_table.shape[0],
        signatures,
        spec.component_names,
        target,
        max_pairs=max_pairs,
        rng=rng,
    )
    if not pairs:
        raise ValueError(f"no matched pairs for component {target!r}")

    dev = torch.device(device)
    n = len(pairs)
    d = hidden_at_site(model, pairs[0][0], pairs[0][2], dev, site=patch_site).numel()
    h_base = torch.zeros(n, d, device=dev)
    h_src = torch.zeros(n, d, device=dev)
    a_arr = np.zeros(n, dtype=np.int64)
    a_prime_arr = np.zeros(n, dtype=np.int64)
    b_arr = np.zeros(n, dtype=np.int64)
    true_cf = np.zeros(n, dtype=np.int64)

    model.eval()
    with torch.no_grad():
        for i, (a, a_prime, b) in enumerate(pairs):
            a_arr[i] = a
            a_prime_arr[i] = a_prime
            b_arr[i] = b
            true_cf[i] = int(mult_table[a_prime, b])
            h_base[i] = hidden_at_site(model, a, b, dev, site=patch_site).reshape(-1)
            h_src[i] = hidden_at_site(model, a_prime, b, dev, site=patch_site).reshape(-1)

    return MatchedPairHiddens(
        a=a_arr,
        a_prime=a_prime_arr,
        b=b_arr,
        true_cf=true_cf,
        h_base=h_base,
        h_src=h_src,
    )


def _pairs_to_matched_hiddens(
    model: GroupTransformer,
    mult_table: np.ndarray,
    pairs: list[tuple[int, int, int]],
    *,
    device: str,
    patch_site: TransformerPatchSite = "resid_pre_0",
) -> MatchedPairHiddens:
    if not pairs:
        raise ValueError("empty pair list")
    dev = torch.device(device)
    n = len(pairs)
    d = hidden_at_site(model, pairs[0][0], pairs[0][2], dev, site=patch_site).numel()
    h_base = torch.zeros(n, d, device=dev)
    h_src = torch.zeros(n, d, device=dev)
    a_arr = np.zeros(n, dtype=np.int64)
    a_prime_arr = np.zeros(n, dtype=np.int64)
    b_arr = np.zeros(n, dtype=np.int64)
    true_cf = np.zeros(n, dtype=np.int64)
    model.eval()
    with torch.no_grad():
        for i, (a, a_prime, b) in enumerate(pairs):
            a_arr[i] = a
            a_prime_arr[i] = a_prime
            b_arr[i] = b
            true_cf[i] = int(mult_table[a_prime, b])
            h_base[i] = hidden_at_site(model, a, b, dev, site=patch_site).reshape(-1)
            h_src[i] = hidden_at_site(model, a_prime, b, dev, site=patch_site).reshape(-1)
    return MatchedPairHiddens(
        a=a_arr,
        a_prime=a_prime_arr,
        b=b_arr,
        true_cf=true_cf,
        h_base=h_base,
        h_src=h_src,
    )


def collect_disjoint_dense_pair_hiddens(
    model: GroupTransformer,
    mult_table: np.ndarray,
    spec: WedderburnSpec,
    target: str,
    *,
    device: str,
    patch_site: TransformerPatchSite = "resid_pre_0",
    bases_per_source: int = 50,
    element_train_fraction: float = 0.7,
    element_seed: int = 42,
    max_val_pairs: int = 5000,
    rng: np.random.Generator,
) -> tuple[MatchedPairHiddens, MatchedPairHiddens, dict]:
    signatures = labels_to_signatures(spec)
    split = sample_disjoint_dense_matched_pairs(
        mult_table.shape[0],
        signatures,
        spec.component_names,
        target,
        bases_per_source=bases_per_source,
        element_train_fraction=element_train_fraction,
        element_seed=element_seed,
        max_val_pairs=max_val_pairs,
        rng=rng,
    )
    if not split.train_pairs:
        raise ValueError(f"no train pairs for component {target!r} (disjoint split)")
    if not split.val_pairs:
        raise ValueError(f"no val pairs for component {target!r} (disjoint split)")

    train_data = _pairs_to_matched_hiddens(
        model, mult_table, split.train_pairs, device=device, patch_site=patch_site
    )
    val_data = _pairs_to_matched_hiddens(
        model, mult_table, split.val_pairs, device=device, patch_site=patch_site
    )
    meta = {
        "n_train_elements": int(len(split.train_elements)),
        "n_val_elements": int(len(split.val_elements)),
        "n_sources_train": split.n_sources_train,
        "bases_per_source": split.bases_per_source,
        "element_seed": element_seed,
        "element_train_fraction": element_train_fraction,
    }
    return train_data, val_data, meta


def split_pair_indices(n: int, train_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = max(1, int(n * train_fraction))
    if n_train >= n:
        n_train = n - 1
    return perm[:n_train], perm[n_train:]


def train_hidden_patcher(
    patcher: HiddenPatcherMLP,
    model: GroupTransformer,
    data: MatchedPairHiddens,
    train_idx: np.ndarray,
    *,
    device: str,
    epochs: int = 300,
    batch_size: int = 32,
    lr: float = 1e-3,
    stay_penalty: float = 0.05,
    l1_alpha: float = 0.0,
    l1_source_only: bool = False,
    patch_site: TransformerPatchSite = "resid_pre_0",
    val_idx: np.ndarray | None = None,
    val_data: MatchedPairHiddens | None = None,
    early_stop_patience: int = 40,
) -> tuple[HiddenPatcherMLP, int, float]:
    """Fit patcher to maximize counterfactual accuracy on train pairs."""
    dev = torch.device(device)
    eval_data = val_data if val_data is not None else data
    patcher = patcher.to(dev)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    opt = torch.optim.Adam(patcher.parameters(), lr=lr)
    best_val = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    last_epoch = 0

    for epoch in range(epochs):
        last_epoch = epoch + 1
        patcher.train()
        perm = np.random.default_rng(epoch).permutation(train_idx)
        for start in range(0, len(perm), batch_size):
            idx = perm[start : start + batch_size]
            h_b = data.h_base[idx]
            h_s = data.h_src[idx]
            y = torch.as_tensor(data.true_cf[idx], dtype=torch.long, device=dev)
            x = torch.as_tensor(np.stack([data.a[idx], data.b[idx]], axis=1), dtype=torch.long, device=dev)

            h_patch = patcher(h_b, h_s)
            logits = forward_group_transformer_with_patch_grad(
                model, x, h_patch, patch_site=patch_site
            )
            ce = F.cross_entropy(logits, y)
            stay = ((h_patch - h_b) ** 2).mean()
            loss = ce + stay_penalty * stay
            if l1_alpha > 0:
                loss = loss + l1_alpha * first_layer_l1_norm(patcher, source_only=l1_source_only)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        if val_idx is not None and len(val_idx) > 0:
            val_iia = _evaluate_patcher_iia(
                patcher, model, eval_data, val_idx, device=device, patch_site=patch_site
            )
            if val_iia > best_val:
                best_val = val_iia
                best_state = {k: v.detach().cpu().clone() for k, v in patcher.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= early_stop_patience:
                    break

    if best_state is not None:
        patcher.load_state_dict(best_state)
    patcher.eval()
    return patcher, last_epoch, best_val


@torch.no_grad()
def _evaluate_patcher_iia(
    patcher: HiddenPatcherMLP,
    model: GroupTransformer,
    data: MatchedPairHiddens,
    idx: np.ndarray,
    *,
    device: str,
    patch_site: TransformerPatchSite = "resid_pre_0",
    h_src_override: torch.Tensor | None = None,
) -> float:
    if len(idx) == 0:
        return float("nan")
    dev = torch.device(device)
    patcher.eval()
    ok = 0
    for i in idx:
        h_b = data.h_base[i]
        h_s = h_src_override[i] if h_src_override is not None else data.h_src[i]
        h_patch = patcher(h_b, h_s)
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


@torch.no_grad()
def _evaluate_adam_bottleneck_iia(
    model: GroupTransformer,
    data: MatchedPairHiddens,
    idx: np.ndarray,
    w0: np.ndarray,
    b0: np.ndarray,
    *,
    device: str,
    patch_site: TransformerPatchSite = "resid_pre_0",
    steps: int = 150,
    z_random: bool = False,
    rng: np.random.Generator,
    n_elements: int,
) -> float:
    if len(idx) == 0:
        return float("nan")
    dev = torch.device(device)
    ok = 0
    for i in idx:
        h_b = data.h_base[i].reshape(1, -1)
        h_s = data.h_src[i].reshape(1, -1)
        if z_random:
            g_rand = int(rng.integers(0, n_elements))
            h_rand = hidden_at_site(model, g_rand, int(data.b[i]), dev, site=patch_site)
            from .iia import mlp_bottleneck_activation

            z_tgt = mlp_bottleneck_activation(h_rand, w0, b0)
            h_patch = nonlinear_bottleneck_patch(
                h_b, h_s, w0, b0, z_target=z_tgt, steps=steps
            )
        else:
            h_patch = nonlinear_bottleneck_patch(h_b, h_s, w0, b0, steps=steps)
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


@torch.no_grad()
def _evaluate_subspace_iia(
    model: GroupTransformer,
    data: MatchedPairHiddens,
    idx: np.ndarray,
    basis: np.ndarray,
    *,
    device: str,
    patch_site: TransformerPatchSite = "resid_pre_0",
) -> float:
    if len(idx) == 0:
        return float("nan")
    dev = torch.device(device)
    ok = 0
    for i in idx:
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
        if pred == int(data.true_cf[i]):
            ok += 1
    return ok / len(idx)


@torch.no_grad()
def _evaluate_raw_iia(
    model: GroupTransformer,
    data: MatchedPairHiddens,
    idx: np.ndarray,
    *,
    device: str,
    patch_site: TransformerPatchSite = "resid_pre_0",
) -> float:
    if len(idx) == 0:
        return float("nan")
    dev = torch.device(device)
    ok = 0
    for i in idx:
        x = torch.tensor([[int(data.a[i]), int(data.b[i])]], dtype=torch.long, device=dev)
        pred = int(
            forward_group_transformer_with_patch_grad(
                model, x, data.h_src[i].reshape(1, -1), patch_site=patch_site
            )
            .argmax(dim=1)
            .item()
        )
        if pred == int(data.true_cf[i]):
            ok += 1
    return ok / len(idx)


def source_saliency_importance(
    patcher: HiddenPatcherMLP,
    data: MatchedPairHiddens,
    idx: np.ndarray,
    *,
    device: str,
) -> np.ndarray:
    """I_j = (1/N) sum_pairs sum_k |d h_patched,k / d h_src,j| on validation pairs."""
    if len(idx) == 0:
        return np.zeros(patcher.d_model, dtype=np.float64)
    dev = torch.device(device)
    patcher.eval()
    d = patcher.d_model
    accum = np.zeros(d, dtype=np.float64)
    for i in idx:
        h_b = data.h_base[i].detach()
        h_s = data.h_src[i].detach().clone().requires_grad_(True)

        def patch_from_src(hs: torch.Tensor) -> torch.Tensor:
            return patcher(h_b.unsqueeze(0), hs.unsqueeze(0)).squeeze(0)

        jac = torch.autograd.functional.jacobian(patch_from_src, h_s)
        accum += jac.abs().sum(dim=0).detach().cpu().numpy()
    return accum / len(idx)


def base_saliency_importance(
    patcher: HiddenPatcherMLP,
    data: MatchedPairHiddens,
    idx: np.ndarray,
    *,
    device: str,
) -> np.ndarray:
    """Jacobian saliency for h_base; zero for the unconditional patcher (delta independent of h_base)."""
    del data, device
    if len(idx) == 0:
        return np.zeros(patcher.d_model, dtype=np.float64)
    return np.zeros(patcher.d_model, dtype=np.float64)


def evaluate_patcher_with_source_mask(
    patcher: HiddenPatcherMLP,
    model: GroupTransformer,
    data: MatchedPairHiddens,
    idx: np.ndarray,
    keep_dims: np.ndarray,
    *,
    device: str,
    patch_site: TransformerPatchSite = "resid_pre_0",
) -> float:
    """IIA when only selected h_src coordinates are passed to the patcher."""
    if len(idx) == 0:
        return float("nan")
    dev = torch.device(device)
    patcher.eval()
    mask = torch.zeros(patcher.d_model, device=dev)
    mask[torch.as_tensor(keep_dims, device=dev)] = 1.0
    ok = 0
    for i in idx:
        h_b = data.h_base[i]
        h_s = data.h_src[i] * mask
        h_patch = patcher(h_b, h_s)
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


def topk_source_mask_from_saliency(importance: np.ndarray, k: int) -> np.ndarray:
    k = max(1, min(k, importance.shape[0]))
    return np.argsort(-importance)[:k].astype(np.int64)


@dataclass
class WeightSparsityStats:
    """First-layer weight sparsity diagnostics (not just binary 'active' count)."""

    col_l1_mean: float
    col_l1_std: float
    col_l1_max: float
    col_l1_min: float
    n_exact_zero_cols: int
    n_below_1e3: int
    n_below_1e2: int
    n_below_1e1: int
    n_relative_1pct: int
    n_relative_10pct: int
    effective_input_dims: float
    total_l1: float
    source_col_l1_mean: float
    source_col_l1_max: float


def weight_sparsity_stats(patcher: HiddenPatcherMLP) -> WeightSparsityStats:
    w = patcher.first_layer_weight().detach().abs().cpu().numpy()
    col_l1 = w.sum(axis=0)
    mx = float(col_l1.max()) if col_l1.size else 0.0
    smx = mx
    s1 = col_l1.sum()
    s2 = (col_l1**2).sum()
    eff = float(s1**2 / s2) if s2 > 0 else 0.0
    return WeightSparsityStats(
        col_l1_mean=float(col_l1.mean()),
        col_l1_std=float(col_l1.std()),
        col_l1_max=mx,
        col_l1_min=float(col_l1.min()),
        n_exact_zero_cols=int((col_l1 == 0).sum()),
        n_below_1e3=int((col_l1 < 1e-3).sum()),
        n_below_1e2=int((col_l1 < 1e-2).sum()),
        n_below_1e1=int((col_l1 < 1e-1).sum()),
        n_relative_1pct=int((col_l1 < 0.01 * mx).sum()) if mx > 0 else 0,
        n_relative_10pct=int((col_l1 < 0.10 * mx).sum()) if mx > 0 else 0,
        effective_input_dims=eff,
        total_l1=float(s1),
        source_col_l1_mean=float(col_l1.mean()) if col_l1.size else 0.0,
        source_col_l1_max=smx,
    )


@torch.no_grad()
def evaluate_patcher_with_dim_ablation(
    patcher: HiddenPatcherMLP,
    model: GroupTransformer,
    data: MatchedPairHiddens,
    idx: np.ndarray,
    *,
    ablate_dim: int | None = None,
    ablate_dims: np.ndarray | None = None,
    keep_only_dims: np.ndarray | None = None,
    ablate_hidden_unit: int | None = None,
    device: str,
    patch_site: TransformerPatchSite = "resid_pre_0",
) -> float:
    """Causal IIA under input/hidden ablations (zero-out interventions)."""
    if len(idx) == 0:
        return float("nan")
    dev = torch.device(device)
    patcher.eval()
    d = patcher.d_model
    ok = 0
    for i in idx:
        h_b = data.h_base[i].clone()
        h_s = data.h_src[i].clone()
        if ablate_dim is not None:
            h_s[ablate_dim] = 0.0
        if ablate_dims is not None:
            h_s[torch.as_tensor(ablate_dims, device=dev)] = 0.0
        if keep_only_dims is not None:
            mask = torch.zeros(d, device=dev)
            mask[torch.as_tensor(keep_only_dims, device=dev)] = 1.0
            h_s = h_s * mask

        if ablate_hidden_unit is not None:
            pre = patcher.fc1(h_s)
            pre[ablate_hidden_unit] = 0.0
            h_patch = h_b + patcher.fc2(torch.relu(pre))
        else:
            h_patch = patcher(h_b, h_s)

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


def leave_one_out_source_ablation(
    patcher: HiddenPatcherMLP,
    model: GroupTransformer,
    data: MatchedPairHiddens,
    idx: np.ndarray,
    *,
    device: str,
    patch_site: TransformerPatchSite = "resid_pre_0",
) -> tuple[float, list[dict]]:
    """Baseline val IIA and per-dim drop when h_src[j] is zeroed."""
    baseline = evaluate_patcher_with_dim_ablation(
        patcher, model, data, idx, device=device, patch_site=patch_site
    )
    rows: list[dict] = []
    d = patcher.d_model
    for j in range(d):
        ablated = evaluate_patcher_with_dim_ablation(
            patcher, model, data, idx, ablate_dim=j, device=device, patch_site=patch_site
        )
        rows.append(
            {
                "dim": j,
                "val_iia": ablated,
                "drop": baseline - ablated,
            }
        )
    rows.sort(key=lambda r: r["drop"], reverse=True)
    return baseline, rows


def topk_keep_ablation_curve(
    patcher: HiddenPatcherMLP,
    model: GroupTransformer,
    data: MatchedPairHiddens,
    idx: np.ndarray,
    importance: np.ndarray,
    *,
    ks: list[int],
    device: str,
    patch_site: TransformerPatchSite = "resid_pre_0",
) -> list[dict]:
    order = np.argsort(-importance)
    rows: list[dict] = []
    for k in ks:
        keep = order[:k]
        val = evaluate_patcher_with_dim_ablation(
            patcher,
            model,
            data,
            idx,
            keep_only_dims=keep,
            device=device,
            patch_site=patch_site,
        )
        rows.append({"k": int(k), "dims": keep.tolist(), "val_iia": float(val)})
    return rows


@dataclass
class TrainedPatcherRun:
    metrics: TrainedPatcherMetrics
    patcher: HiddenPatcherMLP
    data: MatchedPairHiddens
    train_idx: np.ndarray
    val_idx: np.ndarray


@dataclass
class TrainedPatcherDisjointRun:
    metrics: TrainedPatcherMetrics
    patcher: HiddenPatcherMLP
    train_data: MatchedPairHiddens
    val_data: MatchedPairHiddens
    split_meta: dict


def run_trained_patcher_pipeline(
    model: GroupTransformer,
    mult_table: np.ndarray,
    spec: WedderburnSpec,
    target: str,
    hidden_by_elem: np.ndarray,
    *,
    device: str,
    patch_site: TransformerPatchSite = "resid_pre_0",
    max_pairs: int = 800,
    train_fraction: float = 0.8,
    seed: int = 42,
    pair_seed: int | None = None,
    patcher_hidden: int = 64,
    train_epochs: int = 300,
    adam_steps: int = 150,
    l1_alpha: float = 0.0,
    l1_source_only: bool = False,
    rng: np.random.Generator | None = None,
) -> TrainedPatcherRun:
    result = run_trained_patcher_iia_component(
        model,
        mult_table,
        spec,
        target,
        hidden_by_elem,
        device=device,
        patch_site=patch_site,
        max_pairs=max_pairs,
        train_fraction=train_fraction,
        seed=seed,
        pair_seed=pair_seed,
        patcher_hidden=patcher_hidden,
        train_epochs=train_epochs,
        adam_steps=adam_steps,
        l1_alpha=l1_alpha,
        l1_source_only=l1_source_only,
        rng=rng,
        return_artifacts=True,
    )
    assert isinstance(result, TrainedPatcherRun)
    return result


def run_trained_patcher_iia_component(
    model: GroupTransformer,
    mult_table: np.ndarray,
    spec: WedderburnSpec,
    target: str,
    hidden_by_elem: np.ndarray,
    *,
    device: str,
    patch_site: TransformerPatchSite = "resid_pre_0",
    max_pairs: int = 800,
    train_fraction: float = 0.8,
    seed: int = 42,
    pair_seed: int | None = None,
    patcher_hidden: int = 64,
    train_epochs: int = 300,
    adam_steps: int = 150,
    l1_alpha: float = 0.0,
    l1_source_only: bool = False,
    rng: np.random.Generator | None = None,
    return_artifacts: bool = False,
) -> TrainedPatcherMetrics | TrainedPatcherRun:
    ps = pair_seed if pair_seed is not None else seed
    rng = rng or np.random.default_rng(ps + hash(target) % 1000)
    labels = spec.labels[target]
    k_min = min_subspace_dim_for_probe(labels)
    if labels.ndim == 2:
        probe = probe_subspace_basis_from_vector_labels(
            hidden_by_elem, labels, min_subspace_dim=k_min
        )
    else:
        probe = probe_subspace_basis_from_labels(
            hidden_by_elem, labels, min_subspace_dim=k_min
        )

    data = collect_matched_pair_hiddens(
        model,
        mult_table,
        spec,
        target,
        device=device,
        patch_site=patch_site,
        max_pairs=max_pairs,
        rng=rng,
    )
    train_idx, val_idx = split_pair_indices(len(data.a), train_fraction, ps)
    if len(val_idx) == 0:
        val_idx = train_idx[-max(1, len(train_idx) // 5) :]
        train_idx = train_idx[: len(train_idx) - len(val_idx)]

    d_model = data.h_base.shape[1]
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    patcher = HiddenPatcherMLP(d_model, hidden_dim=patcher_hidden)
    patcher, epochs_run, best_val = train_hidden_patcher(
        patcher,
        model,
        data,
        train_idx,
        device=device,
        epochs=train_epochs,
        val_idx=val_idx,
        patch_site=patch_site,
        l1_alpha=l1_alpha,
        l1_source_only=l1_source_only,
    )

    train_iia = _evaluate_patcher_iia(patcher, model, data, train_idx, device=device, patch_site=patch_site)
    val_iia = _evaluate_patcher_iia(patcher, model, data, val_idx, device=device, patch_site=patch_site)

    n_elem = mult_table.shape[0]
    h_rand = data.h_src.clone()
    dev = torch.device(device)
    for i in val_idx:
        g_rand = int(rng.integers(0, n_elem))
        h_rand[i] = hidden_at_site(model, g_rand, int(data.b[i]), dev, site=patch_site).reshape(-1)

    val_random = _evaluate_patcher_iia(
        patcher, model, data, val_idx, device=device, patch_site=patch_site, h_src_override=h_rand
    )
    val_sub = _evaluate_subspace_iia(
        model, data, val_idx, probe.basis, device=device, patch_site=patch_site
    )
    val_raw = _evaluate_raw_iia(model, data, val_idx, device=device, patch_site=patch_site)

    val_adam = float("nan")
    if probe.mlp_w0 is not None and probe.mlp_b0 is not None:
        val_adam = _evaluate_adam_bottleneck_iia(
            model,
            data,
            val_idx,
            probe.mlp_w0,
            probe.mlp_b0,
            device=device,
            patch_site=patch_site,
            steps=adam_steps,
            z_random=False,
            rng=rng,
            n_elements=mult_table.shape[0],
        )

    metrics = TrainedPatcherMetrics(
        component=target,
        n_train=int(len(train_idx)),
        n_val=int(len(val_idx)),
        train_iia=train_iia,
        val_iia=val_iia,
        val_adam_bottleneck_iia=val_adam,
        val_logistic_subspace_iia=val_sub,
        val_random_input_iia=val_random,
        val_raw_iia=val_raw,
        logistic_probe_acc=probe.probe_accuracy,
        mlp_probe_acc=probe.mlp_probe_accuracy,
        best_val_iia=best_val if best_val >= 0 else val_iia,
        train_epochs=epochs_run,
        subspace_dim=probe.k,
    )
    if return_artifacts:
        return TrainedPatcherRun(
            metrics=metrics,
            patcher=patcher,
            data=data,
            train_idx=train_idx,
            val_idx=val_idx,
        )
    return metrics


def run_trained_patcher_disjoint_dense(
    model: GroupTransformer,
    mult_table: np.ndarray,
    spec: WedderburnSpec,
    target: str,
    hidden_by_elem: np.ndarray,
    *,
    device: str,
    patch_site: TransformerPatchSite = "resid_pre_0",
    bases_per_source: int = 50,
    element_train_fraction: float = 0.7,
    element_seed: int = 42,
    max_val_pairs: int = 5000,
    seed: int = 42,
    pair_seed: int | None = None,
    patcher_hidden: int = 64,
    train_epochs: int = 300,
    adam_steps: int = 150,
    l1_alpha: float = 0.0,
    l1_source_only: bool = False,
    rng: np.random.Generator | None = None,
    return_artifacts: bool = False,
    min_subspace_dim: int | None = None,
) -> TrainedPatcherMetrics | TrainedPatcherDisjointRun:
    """Element-disjoint train/val + dense base sampling per train source."""
    ps = pair_seed if pair_seed is not None else seed
    rng = rng or np.random.default_rng(ps + hash(target) % 1000)
    labels = spec.labels[target]
    k_min = min_subspace_dim_for_probe(labels, override=min_subspace_dim)
    if labels.ndim == 2:
        probe = probe_subspace_basis_from_vector_labels(
            hidden_by_elem, labels, min_subspace_dim=k_min
        )
    else:
        probe = probe_subspace_basis_from_labels(
            hidden_by_elem, labels, min_subspace_dim=k_min
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

    d_model = train_data.h_base.shape[1]
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    patcher = HiddenPatcherMLP(d_model, hidden_dim=patcher_hidden)
    patcher, epochs_run, best_val = train_hidden_patcher(
        patcher,
        model,
        train_data,
        train_idx,
        device=device,
        epochs=train_epochs,
        val_idx=val_idx,
        val_data=val_data,
        patch_site=patch_site,
        l1_alpha=l1_alpha,
        l1_source_only=l1_source_only,
    )

    train_iia = _evaluate_patcher_iia(
        patcher, model, train_data, train_idx, device=device, patch_site=patch_site
    )
    val_iia = _evaluate_patcher_iia(
        patcher, model, val_data, val_idx, device=device, patch_site=patch_site
    )

    n_elem = mult_table.shape[0]
    h_rand = val_data.h_src.clone()
    dev = torch.device(device)
    for i in val_idx:
        g_rand = int(rng.integers(0, n_elem))
        h_rand[i] = hidden_at_site(
            model, g_rand, int(val_data.b[i]), dev, site=patch_site
        ).reshape(-1)

    val_random = _evaluate_patcher_iia(
        patcher,
        model,
        val_data,
        val_idx,
        device=device,
        patch_site=patch_site,
        h_src_override=h_rand,
    )
    val_sub = _evaluate_subspace_iia(
        model, val_data, val_idx, probe.basis, device=device, patch_site=patch_site
    )
    val_raw = _evaluate_raw_iia(model, val_data, val_idx, device=device, patch_site=patch_site)

    val_adam = float("nan")
    if probe.mlp_w0 is not None and probe.mlp_b0 is not None:
        val_adam = _evaluate_adam_bottleneck_iia(
            model,
            val_data,
            val_idx,
            probe.mlp_w0,
            probe.mlp_b0,
            device=device,
            patch_site=patch_site,
            steps=adam_steps,
            z_random=False,
            rng=rng,
            n_elements=mult_table.shape[0],
        )

    metrics = TrainedPatcherMetrics(
        component=target,
        n_train=int(len(train_idx)),
        n_val=int(len(val_idx)),
        train_iia=train_iia,
        val_iia=val_iia,
        val_adam_bottleneck_iia=val_adam,
        val_logistic_subspace_iia=val_sub,
        val_random_input_iia=val_random,
        val_raw_iia=val_raw,
        logistic_probe_acc=probe.probe_accuracy,
        mlp_probe_acc=probe.mlp_probe_accuracy,
        best_val_iia=best_val if best_val >= 0 else val_iia,
        train_epochs=epochs_run,
        subspace_dim=probe.k,
    )
    if return_artifacts:
        return TrainedPatcherDisjointRun(
            metrics=metrics,
            patcher=patcher,
            train_data=train_data,
            val_data=val_data,
            split_meta=split_meta,
        )
    return metrics
