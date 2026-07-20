"""Интерпретация MLP на A_n: SVD/PCA (Веддербёрн) и coset concentration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from .alt_group_subgroups import SubgroupSpec, all_subgroup_specs, left_coset_labels, right_coset_labels
from .alternating_group import build_alternating_mult_table
from .models.mlp import MLP

# Вещественные размерности неприводимых представлений (сумма квадратов = |G|).
REAL_IRREP_DIMS: dict[int, list[int]] = {
    5: [1, 3, 3, 4, 5],
    6: [1, 5, 5, 8, 8, 9, 10],
}


@dataclass
class HiddenResponses:
    """Отклики скрытого слоя на все пары (a, b)."""

    pre_relu: np.ndarray  # [hidden, n, n]
    post_relu: np.ndarray  # [hidden, n, n]


def mlp_weight_blocks(model: MLP, n_elements: int) -> dict[str, np.ndarray]:
    """Матрицы весов [n_elements, hidden] и [hidden, n_elements]."""
    w1: nn.Linear = model.net[0]  # type: ignore[assignment]
    w2: nn.Linear = model.net[2]  # type: ignore[assignment]
    w_in = w1.weight.detach().cpu().numpy()
    emb_a = w_in[:, :n_elements].T.astype(np.float64)  # [n, h]
    emb_b = w_in[:, n_elements:].T.astype(np.float64)
    emb_avg = 0.5 * (emb_a + emb_b)
    w_out = w2.weight.detach().cpu().numpy().astype(np.float64)  # [n, h]
    return {
        "W_a": emb_a,
        "W_b": emb_b,
        "W_avg": emb_avg,
        "W_out": w_out,
        "W_in_a": w_in[:, :n_elements].astype(np.float64),  # [h, n]
    }


def svd_spectrum(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """SVD матрицы [n, d]; возвращает U, S, Vt."""
    x = matrix - matrix.mean(axis=0, keepdims=True)
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    return u, s, vt


def irrep_cumulative_bounds(dims: list[int]) -> list[int]:
    """Кумулятивные границы одной копии разложения (1, 1+3, …)."""
    out: list[int] = []
    total = 0
    for d in dims:
        total += d
        out.append(total)
    return out


def coset_concentration(values: np.ndarray, labels: np.ndarray) -> float:
    """1 − (внутриклассовая SS / общая SS); 1 = идеальная константность на косетах."""
    v = np.asarray(values, dtype=np.float64)
    lab = np.asarray(labels, dtype=np.int64)
    if v.shape[0] != lab.shape[0]:
        raise ValueError("values and labels length mismatch")
    total_ss = float(np.sum((v - v.mean()) ** 2))
    if total_ss < 1e-15:
        return 0.0
    within = 0.0
    for c in np.unique(lab):
        mask = lab == c
        vc = v[mask]
        within += float(np.sum((vc - vc.mean()) ** 2))
    return 1.0 - within / total_ss


def neuron_responses_by_left(
    responses: HiddenResponses,
    *,
    use_post_relu: bool = True,
) -> np.ndarray:
    """[hidden, n] — средний отклик нейрона по правому аргументу b при фикс. a."""
    field = responses.post_relu if use_post_relu else responses.pre_relu
    return field.mean(axis=2)


def neuron_responses_by_right(
    responses: HiddenResponses,
    *,
    use_post_relu: bool = True,
) -> np.ndarray:
    """[hidden, n] — средний отклик при фикс. b (усреднение по a)."""
    field = responses.post_relu if use_post_relu else responses.pre_relu
    return field.mean(axis=1)


@torch.no_grad()
def compute_hidden_responses(
    model: MLP,
    n_elements: int,
    *,
    device: str = "cpu",
) -> HiddenResponses:
    """Полный forward на всех парах через разложение W_a + W_b."""
    model = model.to(device)
    model.eval()
    w1: nn.Linear = model.net[0]  # type: ignore[assignment]
    w_in = w1.weight
    bias = w1.bias
    h = w_in.shape[0]
    n = n_elements
    w_a = w_in[:, :n]  # [h, n]
    w_b = w_in[:, n:]  # [h, n]

    # pre[h, i, j] = w_a[:, i] + w_b[:, j] + bias
    pre = w_a[:, :, None] + w_b[:, None, :] + bias[:, None, None]
    post = torch.relu(pre)
    return HiddenResponses(
        pre_relu=pre.detach().cpu().numpy(),
        post_relu=post.detach().cpu().numpy(),
    )


def coset_concentration_matrix(
    neuron_values: np.ndarray,
    subgroups: tuple[SubgroupSpec, ...],
    mult_table: np.ndarray,
    *,
    side: str = "left",
) -> tuple[np.ndarray, list[str]]:
    """[hidden, n_subgroups] — concentration каждого нейрона по каждой подгруппе."""
    if side not in ("left", "right"):
        raise ValueError("side must be 'left' or 'right'")
    n_neurons, n_elems = neuron_values.shape
    if n_elems != mult_table.shape[0]:
        raise ValueError("neuron_values width must match |G|")

    cols: list[str] = []
    mat = np.zeros((n_neurons, len(subgroups)), dtype=np.float64)
    for j, sg in enumerate(subgroups):
        cols.append(sg.name)
        if side == "left":
            labels = left_coset_labels(sg, n_elems, mult_table=mult_table)
        else:
            labels = right_coset_labels(sg, n_elems, mult_table=mult_table)
        for i in range(n_neurons):
            mat[i, j] = coset_concentration(neuron_values[i], labels)
    return mat, cols


def subgroups_by_family(
    subgroups: tuple[SubgroupSpec, ...],
) -> dict[str, tuple[SubgroupSpec, ...]]:
    """Разбить каталог подгрупп по family."""
    buckets: dict[str, list[SubgroupSpec]] = {}
    for sg in subgroups:
        buckets.setdefault(sg.family, []).append(sg)
    return {fam: tuple(specs) for fam, specs in buckets.items()}


def coset_labels_for_subgroup(
    subgroup: SubgroupSpec,
    n_elements: int,
    mult_table: np.ndarray,
    *,
    side: str = "left",
) -> np.ndarray:
    if side == "left":
        return left_coset_labels(subgroup, n_elements, mult_table=mult_table)
    if side == "right":
        return right_coset_labels(subgroup, n_elements, mult_table=mult_table)
    raise ValueError("side must be 'left' or 'right'")


def best_coset_labels(
    values: np.ndarray,
    subgroup_specs: tuple[SubgroupSpec, ...],
    mult_table: np.ndarray,
    *,
    side: str = "left",
) -> tuple[np.ndarray, float, SubgroupSpec]:
    """Лучшее coset-разбиение для одного нейрона среди сопряжённых подгрупп."""
    best_labels: np.ndarray | None = None
    best_conc = -1.0
    best_spec = subgroup_specs[0]
    for sg in subgroup_specs:
        labels = coset_labels_for_subgroup(sg, values.shape[0], mult_table, side=side)
        conc = coset_concentration(values, labels)
        if conc > best_conc:
            best_conc = conc
            best_labels = labels
            best_spec = sg
    assert best_labels is not None
    return best_labels, best_conc, best_spec


def project_onto_coset_partition(values: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Проекция на кусочно-константные функции по coset-меткам."""
    out = np.zeros_like(values, dtype=np.float64)
    for c in np.unique(labels):
        mask = labels == c
        out[mask] = float(values[mask].mean())
    return out


def family_max_concentration(
    matrix: np.ndarray,
    subgroups: tuple[SubgroupSpec, ...],
) -> dict[str, np.ndarray]:
    """Максимум concentration по сопряжённым подгруппам внутри семейства."""
    families: dict[str, list[int]] = {}
    for j, sg in enumerate(subgroups):
        families.setdefault(sg.family, []).append(j)
    out: dict[str, np.ndarray] = {}
    for fam, idxs in families.items():
        out[fam] = matrix[:, idxs].max(axis=1)
    return out


def residual_family_max_concentration(
    neuron_values: np.ndarray,
    subgroups: tuple[SubgroupSpec, ...],
    mult_table: np.ndarray,
    *,
    coarse_family: str,
    target_family: str,
    side: str = "left",
) -> np.ndarray:
    """Concentration target_family после вычитания лучшей coarse-проекции."""
    by_fam = subgroups_by_family(subgroups)
    if coarse_family not in by_fam or target_family not in by_fam:
        raise KeyError(f"unknown family: {coarse_family} or {target_family}")
    n_neurons = neuron_values.shape[0]
    out = np.zeros(n_neurons, dtype=np.float64)
    for i in range(n_neurons):
        coarse_labels, _, _ = best_coset_labels(
            neuron_values[i], by_fam[coarse_family], mult_table, side=side
        )
        residual = neuron_values[i] - project_onto_coset_partition(neuron_values[i], coarse_labels)
        best_fine = -1.0
        for sg in by_fam[target_family]:
            labels = coset_labels_for_subgroup(sg, neuron_values.shape[1], mult_table, side=side)
            best_fine = max(best_fine, coset_concentration(residual, labels))
        out[i] = best_fine
    return out


def exclusive_family_scores(
    family_scores: dict[str, np.ndarray],
    *,
    target: str,
    competitors: tuple[str, ...],
    margin: float = 0.0,
) -> np.ndarray:
    """raw(target) − max(competitors) − margin; отрицательные → 0."""
    comp = np.zeros_like(family_scores[target])
    for fam in competitors:
        comp = np.maximum(comp, family_scores[fam])
    return np.maximum(0.0, family_scores[target] - comp - margin)


def neuron_mask_from_scores(
    scores: np.ndarray,
    *,
    threshold: float = 0.5,
    top_k: int | None = None,
) -> np.ndarray:
    """Булева маска нейронов по порогу и/или top-k."""
    mask = scores >= threshold
    if top_k is not None and top_k > 0:
        order = np.argsort(-scores)
        top = np.zeros(scores.shape[0], dtype=bool)
        top[order[:top_k]] = True
        mask = mask | top
    return mask


@torch.no_grad()
def forward_with_hidden_mask(
    model: MLP,
    inputs: torch.Tensor,
    keep_mask: torch.Tensor,
) -> torch.Tensor:
    """Forward с занулением скрытых нейронов вне keep_mask."""
    w1: nn.Linear = model.net[0]  # type: ignore[assignment]
    act: nn.Module = model.net[1]  # type: ignore[assignment]
    w2: nn.Linear = model.net[2]  # type: ignore[assignment]
    hidden = act(w1(inputs))
    mask = keep_mask.to(device=hidden.device, dtype=hidden.dtype)
    return w2(hidden * mask)


@torch.no_grad()
def evaluate_with_hidden_mask(
    model: MLP,
    loader: torch.utils.data.DataLoader,
    keep_mask: torch.Tensor,
    device: str,
) -> float:
    """Accuracy при маскировании скрытого слоя."""
    model.eval()
    correct = 0
    total = 0
    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        logits = forward_with_hidden_mask(model, inputs, keep_mask)
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        total += labels.size(0)
    return correct / total if total else 0.0


def project_elements_3d(
    emb: np.ndarray,
    singular_values: np.ndarray,
    vt: np.ndarray,
    start: int,
) -> np.ndarray:
    """Координаты элементов в 3D-блоке SVD (строки emb [n, h])."""
    x = emb - emb.mean(axis=0, keepdims=True)
    return x @ vt[start : start + 3].T


def _num_elements_from_config(cfg: dict) -> int:
    if "num_elements" in cfg:
        return int(cfg["num_elements"])
    if cfg.get("field_order") is not None:
        from .psl2_group import psl2_group_order

        return psl2_group_order(int(cfg["field_order"]))
    group_n = cfg.get("group_n")
    if group_n is not None:
        from math import factorial

        gn = int(group_n)
        return factorial(gn) // 2
    raise KeyError("cannot infer num_elements from checkpoint config")


def load_alt_group_mlp(
    checkpoint_path: str,
    device: str,
) -> tuple[MLP, dict]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    n = _num_elements_from_config(cfg)
    hidden = int(cfg["hidden"])
    model = MLP(n * 2, hidden, n).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt
