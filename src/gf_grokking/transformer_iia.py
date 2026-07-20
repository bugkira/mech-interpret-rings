"""IIA для GroupTransformer: патчинг ``resid_pre_0`` (post-embedding, pos 0)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

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
from .iia import (
    IIAResult,
    conjugacy_class_labels,
    identity_element,
    irrep_character_vectors,
    probe_subspace_basis,
    random_orthonormal_basis,
    sample_matched_pairs_conjugacy,
    _apply_subspace_patch,
    _chance_preservation,
    _char_key_at,
)
from .models.group_transformer import GroupTransformer
from .alternating_group import build_alternating_mult_table
from .psl2_group import build_psl2_mult_table

TransformerPatchSite = Literal["resid_pre_0", "post_block"]


@dataclass
class TransformerForwardCache:
    resid_pre: torch.Tensor
    post_block: torch.Tensor
    logits: torch.Tensor


def _pair_tensor(a: int, b: int, device: torch.device) -> torch.Tensor:
    return torch.tensor([[a, b]], dtype=torch.long, device=device)


@torch.no_grad()
def forward_group_transformer_with_patch(
    model: GroupTransformer,
    x: torch.Tensor,
    *,
    patch_value: torch.Tensor | None = None,
    patch_site: TransformerPatchSite | None = None,
    patch_pos: int = 0,
    return_cache: bool = False,
) -> torch.Tensor | TransformerForwardCache:
    """Forward с подменой активаций на ``resid_pre_0`` (Zheng et al.) или после блока."""
    h = model.embedding(x)
    resid_pre = h
    if patch_site == "resid_pre_0" and patch_value is not None:
        h = h.clone()
        h[:, patch_pos, :] = patch_value
    for layer in model.layers:
        h = layer(h)
    post_block = h
    if patch_site == "post_block" and patch_value is not None:
        h = h.clone()
        h[:, patch_pos, :] = patch_value
    logits = model.head(h[:, 0])
    if return_cache:
        return TransformerForwardCache(resid_pre=resid_pre, post_block=post_block, logits=logits)
    return logits


@torch.no_grad()
def hidden_at_site(
    model: GroupTransformer,
    a: int,
    b: int,
    device: torch.device,
    *,
    site: TransformerPatchSite = "resid_pre_0",
) -> torch.Tensor:
    x = _pair_tensor(a, b, device)
    cache = forward_group_transformer_with_patch(model, x, return_cache=True)
    assert isinstance(cache, TransformerForwardCache)
    if site == "resid_pre_0":
        return cache.resid_pre[:, 0, :]
    return cache.post_block[:, 0, :]


def load_group_transformer_and_table(
    checkpoint: str,
    device: str,
) -> tuple[GroupTransformer, dict, np.ndarray, dict, tuple[str, ...]]:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    n = int(cfg["num_elements"])
    model = GroupTransformer(
        num_elements=n,
        d_model=int(cfg.get("d_model", 128)),
        nhead=int(cfg.get("nhead", 4)),
        num_layers=int(cfg.get("num_layers", 1)),
        ffn_dim=int(cfg.get("ffn_dim", 512)),
        dropout=float(cfg.get("dropout", 0.0)),
        layernorm=bool(cfg.get("layernorm", False)),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

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


@torch.no_grad()
def evaluate_transformer_accuracy(
    model: GroupTransformer, mult_table: np.ndarray, device: str
) -> float:
    n = mult_table.shape[0]
    dev = torch.device(device)
    correct = 0
    for a in range(n):
        for b in range(n):
            x = _pair_tensor(a, b, dev)
            pred = int(forward_group_transformer_with_patch(model, x).argmax(dim=1).item())
            if pred == int(mult_table[a, b]):
                correct += 1
    return correct / (n * n)


@torch.no_grad()
def run_transformer_iia_for_irrep(
    model: GroupTransformer,
    mult_table: np.ndarray,
    chars: dict[str, np.ndarray],
    nontriv: tuple[str, ...],
    target: str,
    *,
    device: str,
    patch_site: TransformerPatchSite = "resid_pre_0",
    max_pairs: int = 200,
    rng: np.random.Generator,
    hidden_by_elem: np.ndarray,
) -> "IIAMetrics":
    from .iia import IIAMetrics

    n = mult_table.shape[0]
    dev = torch.device(device)
    basis, k = probe_subspace_basis(hidden_by_elem, chars[target])
    rand_basis = random_orthonormal_basis(hidden_by_elem.shape[1], k, rng)
    pairs = sample_matched_pairs_conjugacy(
        conjugacy_class_labels(mult_table), max_pairs=max_pairs, rng=rng
    )
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

    raw_ok = sub_ok = rand_ok = 0
    err_preserve = err_total = 0
    cond_ntp = cond_total = 0

    for a, a_prime, b in pairs:
        true_cf = int(mult_table[a_prime, b])
        true_base = int(mult_table[a, b])
        h_base = hidden_at_site(model, a, b, dev, site=patch_site)
        h_src = hidden_at_site(model, a_prime, b, dev, site=patch_site)
        x_base = _pair_tensor(a, b, dev)
        base_pred = int(forward_group_transformer_with_patch(model, x_base).argmax(dim=1).item())

        for mode, h_patch in (
            ("full", h_src),
            ("sub", _apply_subspace_patch(h_base, h_src, basis)),
            ("rand", _apply_subspace_patch(h_base, h_src, rand_basis)),
        ):
            logits = forward_group_transformer_with_patch(
                model,
                x_base,
                patch_value=h_patch,
                patch_site=patch_site,
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
            logits = forward_group_transformer_with_patch(
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
            logits = forward_group_transformer_with_patch(
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
def run_transformer_iia_analysis(
    model: GroupTransformer,
    mult_table: np.ndarray,
    irrep_specs: dict,
    nontriv: tuple[str, ...],
    *,
    checkpoint: str = "",
    device: str = "cpu",
    patch_site: TransformerPatchSite = "resid_pre_0",
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
            hidden_at_site(model, g, e, dev, site=patch_site).cpu().numpy().ravel()
            for g in range(n)
        ],
        axis=0,
    )

    test_acc = evaluate_transformer_accuracy(model, mult_table, device)
    per = [
        run_transformer_iia_for_irrep(
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
        for name in nontriv
    ]
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
def measure_transformer_iia_row(
    model: GroupTransformer,
    mult_table: np.ndarray,
    irrep_specs: dict,
    nontriv: tuple[str, ...],
    chars: dict[str, np.ndarray],
    hidden_by_elem: np.ndarray | None,
    *,
    device: str,
    patch_site: TransformerPatchSite = "resid_pre_0",
    max_pairs_per_irrep: int = 50,
    seed: int = 42,
) -> dict[str, float]:
    """Компактные IIA-метрики без полного eval accuracy (для dense trace)."""
    rng = np.random.default_rng(seed)
    n = mult_table.shape[0]
    dev = torch.device(device)
    if hidden_by_elem is None:
        e = identity_element(mult_table)
        hidden_by_elem = np.stack(
            [
                hidden_at_site(model, g, e, dev, site=patch_site).cpu().numpy().ravel()
                for g in range(n)
            ],
            axis=0,
        )
    per = [
        run_transformer_iia_for_irrep(
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
        for name in nontriv
    ]
    raw_vals = [m.raw_iia for m in per if m.n_pairs > 0]
    sub_vals = [m.subspace_iia for m in per if m.n_pairs > 0]
    row = {
        "mean_raw_iia": float(np.mean(raw_vals)) if raw_vals else 0.0,
        "mean_subspace_iia": float(np.mean(sub_vals)) if sub_vals else 0.0,
        "mean_random_subspace_iia": float(
            np.mean([m.random_subspace_iia for m in per if m.n_pairs > 0])
        )
        if any(m.n_pairs > 0 for m in per)
        else 0.0,
    }
    for m in per:
        if m.n_pairs > 0:
            row[f"iia_raw_{m.irrep}"] = m.raw_iia
            row[f"iia_sub_{m.irrep}"] = m.subspace_iia
    return row
