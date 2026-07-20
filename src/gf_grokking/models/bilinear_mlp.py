"""Bilinear MLP — добавляет попарные произведения входов.

Гипотеза: билинейный prior позволит модели «увидеть» структуру
умножения в GF(2^n) и преодолеть барьер ~44%.
"""

import torch
import torch.nn as nn

from .mlp import MLP


class BilinearMLP(nn.Module):
    """MLP с входным билинейным слоем (попарные произведения признаков).

    Для двух one-hot a, b размерности m:
      - линейная часть [m+m, hidden]
      - билинейная часть [m*m, hidden_b]
      — concat → output [m]
    """

    def __init__(
        self,
        n: int,
        hidden_dim: int,
        bilinear_dim: int | None = None,
    ) -> None:
        """Инициализация.

        Args:
            n: Степень поля GF(2^n).
            hidden_dim: Размерность основного MLP.
            bilinear_dim: Размерность билинейного подслоя (%=64 для экономии памяти).
        """
        super().__init__()
        num_elements = 1 << n  # 2^n
        input_size = (num_elements + num_elements)

        self.hidden_dim = hidden_dim

        # Линейная ветка: стандартный MLP
        self.lin_net = MLP(input_size, hidden_dim, num_elements)

        # Билинейная ветка: попарные произведения + линейный слой
        self.bilinear_dim = bilinear_dim or min(hidden_dim, num_elements * num_elements)
        self.bilinear_in = num_elements * num_elements
        self.bilinear_proj = MLP(
            self.bilinear_in, bilinear_dim, num_elements
        ) if bilinear_dim else None

        # Слияние (если есть билинейная ветка)
        self.has_bilinear = self.bilinear_proj is not None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [batch, 2 * 2^n] — два one-hot вектора.

        Returns:
            logits: [batch, 2^n]
        """
        raise NotImplementedError("BilinearMLP forward ещё не реализован")
