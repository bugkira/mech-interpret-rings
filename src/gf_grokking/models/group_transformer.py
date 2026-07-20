"""Transformer для умножения в конечных группах (A_n, PSL(2,q)).

Архитектура по Nanda et al. / Zheng et al.: 1–2 слоя, без LayerNorm,
вход — два токена (a, b), readout с позиции 0.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .transformer import Embedding, FeedForward, MultiHeadAttention, TransformerBlock


class TransformerBlockNoNorm(nn.Module):
    """Residual attention + FFN без LayerNorm (Nanda-style)."""

    def __init__(self, d_model: int, nhead: int, ffn_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.attn = MultiHeadAttention(d_model, nhead, dropout)
        self.ffn = FeedForward(d_model, ffn_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.dropout(self.attn(x, mask))
        x = x + self.dropout(self.ffn(x))
        return x


class TransformerBlockPreNorm(nn.Module):
    """Pre-norm блок (опционально, для сравнения с GF-Transformer)."""

    def __init__(self, d_model: int, nhead: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self._block = TransformerBlock(d_model, nhead, ffn_dim, dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self._block(x, mask)


class GroupTransformer(nn.Module):
    """Умножение в группе: (a, b) → logits для a·b.

    Args:
        num_elements: |G| — размер словаря эмбеддингов и числа классов.
        layernorm: если False — Nanda / «From Groups to Rings» (без LN).
    """

    def __init__(
        self,
        num_elements: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 1,
        ffn_dim: int = 512,
        dropout: float = 0.0,
        *,
        layernorm: bool = False,
    ) -> None:
        super().__init__()
        self.num_elements = num_elements
        self.d_model = d_model
        self.layernorm = layernorm

        self.embedding = Embedding(num_elements, d_model)
        block_cls = TransformerBlockPreNorm if layernorm else TransformerBlockNoNorm
        self.layers = nn.ModuleList(
            [block_cls(d_model, nhead, ffn_dim, dropout) for _ in range(num_layers)]
        )
        self.head = nn.Linear(d_model, num_elements)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, 2] int64 — индексы (a, b).

        Returns:
            logits [batch, num_elements]
        """
        h = self.embedding(x)
        for layer in self.layers:
            h = layer(h)
        return self.head(h[:, 0])
