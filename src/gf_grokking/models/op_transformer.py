"""Transformer для GF(2^n): op-токен (+ / ×) или dual-head (+ и × одновременно)."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from .transformer import Embedding, TransformerBlock

OP_ADD = 0
OP_MUL = 1


class GfOpTransformer(nn.Module):
    """Режимы:
    - op_token: вход [batch, 3] = (op, a, b), один head
    - dual_head: вход [batch, 2] = (a, b), два head (add и mul)
    - single_head: вход [batch, 2] = (a, b), один head на обе метки (ablation)
  """

    def __init__(
        self,
        n: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 1,
        ffn_dim: int = 128,
        dropout: float = 0.1,
        mode: Literal["op_token", "dual_head", "single_head"] = "op_token",
        readout_index: int = 0,
    ) -> None:
        super().__init__()
        self.n = n
        self.num_elements = 1 << n
        self.mode = mode
        self.readout_index = readout_index
        self.d_model = d_model

        self.field_embedding = Embedding(self.num_elements, d_model)
        if mode == "op_token":
            self.op_embedding = nn.Embedding(2, d_model)
        else:
            self.op_embedding = None

        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, nhead, ffn_dim, dropout) for _ in range(num_layers)]
        )
        self.head = nn.Linear(d_model, self.num_elements)
        if mode == "dual_head":
            self.head_add = nn.Linear(d_model, self.num_elements)
            self.head_mul = nn.Linear(d_model, self.num_elements)
        else:
            self.head_add = None
            self.head_mul = None

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "op_token":
            if x.dim() != 2 or x.size(1) != 3:
                raise ValueError(f"op_token expects [batch, 3], got {tuple(x.shape)}")
            op, a, b = x[:, 0], x[:, 1], x[:, 2]
            assert self.op_embedding is not None
            h = torch.stack(
                [self.op_embedding(op), self.field_embedding.embedding(a), self.field_embedding.embedding(b)],
                dim=1,
            )
            return h
        if self.mode in ("dual_head", "single_head"):
            if x.dim() != 2 or x.size(1) != 2:
                raise ValueError(f"{self.mode} expects [batch, 2], got {tuple(x.shape)}")
            return self.field_embedding(x)
        raise ValueError(f"unknown mode: {self.mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        h = self._encode(x)
        for layer in self.layers:
            h = layer(h)
        readout = h[:, self.readout_index]
        if self.mode == "dual_head":
            assert self.head_add is not None and self.head_mul is not None
            return self.head_add(readout), self.head_mul(readout)
        return self.head(readout)
