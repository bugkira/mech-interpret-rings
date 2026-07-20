"""Базовый MLP для задачи умножения в GF(2^n).

Архитектура:
  Input (2^(n+1)) → Linear → ReLU → Linear → Label (2^n)
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Полносвязная сеть с одним скрытым линейным слоем + ReLU."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        activation: type[nn.Module] | None = None,
    ) -> None:
        super().__init__()
        act = activation() if activation is not None else nn.ReLU()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            act,
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
