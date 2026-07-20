"""Tests for Zheng/Nanda init and AdamW param groups."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from gf_grokking.models.group_transformer import GroupTransformer
from gf_grokking.zheng_protocol import (
    build_adamw_optimizer,
    describe_optimizer_groups,
    init_group_transformer_zheng,
)


def test_zheng_init_scales_embedding():
    m = GroupTransformer(num_elements=81, d_model=128, layernorm=False)
    init_group_transformer_zheng(m)
    assert m.embedding.embedding.weight.std().item() == pytest.approx(0.02, abs=0.005)


def test_adamw_excludes_bias_from_decay():
    m = GroupTransformer(num_elements=81, d_model=128, layernorm=False)
    opt = build_adamw_optimizer(m, lr=1e-3, weight_decay=1.0, exclude_bias_from_decay=True)
    groups = describe_optimizer_groups(opt)
    assert groups["decay"] == 8  # embed + 7 linear weights
    assert groups["no_decay"] == 7  # biases
    for g in opt.param_groups:
        if g["weight_decay"] == 0:
            assert all(p.ndim < 2 for p in g["params"])


def test_legacy_adamw_decays_all():
    m = GroupTransformer(num_elements=81, d_model=128, layernorm=False)
    opt = build_adamw_optimizer(m, lr=1e-3, weight_decay=1.0, exclude_bias_from_decay=False)
    assert len(opt.param_groups) == 1
    assert opt.param_groups[0]["weight_decay"] == 1.0
    assert len(opt.param_groups[0]["params"]) == 15
