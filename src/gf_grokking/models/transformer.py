"""Мини-трансформер для задачи умножения в GF(2^n).

Гипотеза: attention-механизм (Q-K скалярное произведение ≈ билинейная
операция) может служить билинейным приоритетом и достичь 100% обобщения.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class Embedding(nn.Module):
    """Эмбеддинг-слой: one-hot → learnable continuous embedding."""

    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len] with integer symbols in [0, vocab_size).

        Returns:
            [batch, seq_len, d_model]
        """
        return self.embedding(x)


class MultiHeadAttention(nn.Module):
    """Multi-head attention (self-attention, no mask needed for this task)."""

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0) -> None:
        super().__init__()
        assert d_model % nhead == 0
        self.d_k = d_model // nhead
        self.nhead = nhead

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
            mask: optional [batch, nhead, q_len, k_len]

        Returns:
            [batch, seq_len, d_model]
        """
        B, S, _ = x.shape

        q = self.q_proj(x).view(B, S, self.nhead, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.nhead, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.nhead, self.d_k).transpose(1, 2)

        attn = torch.einsum("bnid, bnjd -> bnij", q, k) / math.sqrt(self.d_k)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.einsum("bnij, bnjd -> bnid", attn, v)
        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.out_proj(out)


class FeedForward(nn.Module):
    """Position-wise feed-forward network (GELU)."""

    def __init__(self, d_model: int, ffn_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Single transformer block: self-attention + FFN (pre-norm)."""

    def __init__(self, d_model: int, nhead: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = MultiHeadAttention(d_model, nhead, dropout)
        self.ffn = FeedForward(d_model, ffn_dim)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.norm1(x), mask))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class Transformer(nn.Module):
    """Мини-трансформер для задачи умножения в GF(2^n).

    Input: two symbols (a, b) → embed → TransformerBlock(s) → token a → MLP head → label.

    Args:
        n: степень поля GF(2^n).
        d_model: размерность эмбеддинга.
        nhead: количество attention head.
        num_layers: количество трансформер-блоков.
        ffn_dim: размерность FFN скрытого слоя.
        dropout: probability of dropout.
    """

    def __init__(
        self,
        n: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        ffn_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        num_elements = 1 << n  # 2^n

        self.embedding = Embedding(num_elements, d_model)
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, nhead, ffn_dim, dropout) for _ in range(num_layers)]
        )
        self.head = nn.Linear(d_model, num_elements)

        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, 2] with integer symbols in [0, 2^n).

        Returns:
            logits: [batch, 2^n]
        """
        # x shape: [batch, 2] where x[:,0]=a, x[:,1]=b
        h = self.embedding(x)  # [batch, 2, d_model]
        for layer in self.layers:
            h = layer(h)  # self-attention over the 2 tokens
        # Use the first token (a) to predict a * b
        out = self.head(h[:, 0])  # [batch, num_elements]
        return out
