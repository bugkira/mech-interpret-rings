"""DataLoader для умножения в конечных группах (A_n, PSL(2,q), OHE)."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from .alternating_group import build_alternating_mult_table, split_pair_indices
from .data import EAGER_MAX_BYTES, _GFIndexDataset, encode_pairs_one_hot
from .finite_rings import _mat2_encode, build_mul_table as build_ring_mul_table
from .finite_magmas import build_magma_table
from .psl2_group import build_psl2_mult_table


def mlp_input_dim(num_elements: int) -> int:
    return num_elements * 2


def estimate_one_hot_bytes(num_elements: int, train_size: int, test_size: int) -> int:
    input_dim = mlp_input_dim(num_elements)
    return (train_size + test_size) * input_dim * 4


def _labels_for_pairs(
    mult_table: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    return mult_table[a.long(), b.long()].long()


def _make_collate(num_elements: int):
    def collate(
        batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        a = torch.stack([item[0] for item in batch])
        b = torch.stack([item[1] for item in batch])
        labels = torch.stack([item[2] for item in batch])
        inputs = encode_pairs_one_hot(a, b, num_elements)
        return inputs, labels

    return collate


def _loader_kwargs(
    *,
    device: str | torch.device,
    batch_size: int,
    shuffle_train: bool,
    num_workers: int,
) -> dict:
    use_cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    pin_memory = use_cuda
    workers = num_workers if num_workers >= 0 else (2 if use_cuda else 0)
    kw: dict = {
        "batch_size": batch_size,
        "pin_memory": pin_memory,
        "num_workers": workers,
    }
    if workers > 0:
        kw["persistent_workers"] = True
    return kw


def create_group_loaders(
    train_size: int,
    test_size: int,
    batch_size: int = 256,
    device: str | torch.device = "cpu",
    shuffle_train: bool = True,
    seed: int = 42,
    *,
    mult_table: torch.Tensor | None = None,
    group_n: int | None = None,
    psl2_q: int | None = None,
    ring_id: str | None = None,
    num_workers: int = -1,
) -> tuple[DataLoader, DataLoader, torch.Tensor, int]:
    """Train/test DataLoader для умножения в группе/кольце.

    Источник таблицы (ровно один):
      - ``mult_table`` — готовая таблица;
      - ``group_n`` — A_{group_n};
      - ``psl2_q`` — PSL(2, q);
      - ``ring_id`` — конечное кольцо из ``finite_rings``.

    Returns:
        (train_loader, test_loader, mult_table, n_elements)
    """
    if mult_table is None:
        sources = sum(x is not None for x in (group_n, psl2_q, ring_id))
        if sources != 1:
            raise ValueError("specify exactly one of group_n, psl2_q, or ring_id")
        if group_n is not None:
            table_np, n_elements = build_alternating_mult_table(group_n)
        elif psl2_q is not None:
            table_np, n_elements = build_psl2_mult_table(psl2_q)
        else:
            table_np, _spec = build_ring_mul_table(ring_id)  # type: ignore[arg-type]
            n_elements = table_np.shape[0]
        mult_table = torch.from_numpy(table_np).long()
    else:
        n_elements = mult_table.shape[0]

    a_train, b_train, a_test, b_test = split_pair_indices(
        n_elements, train_size, test_size, seed
    )

    loader_kw = _loader_kwargs(
        device=device,
        batch_size=batch_size,
        shuffle_train=shuffle_train,
        num_workers=num_workers,
    )
    eager = estimate_one_hot_bytes(n_elements, train_size, test_size) <= EAGER_MAX_BYTES

    if eager:
        train_labels = _labels_for_pairs(mult_table, a_train, b_train)
        train_inputs = encode_pairs_one_hot(a_train, b_train, n_elements)
        train_loader = DataLoader(
            TensorDataset(train_inputs, train_labels),
            shuffle=shuffle_train,
            **loader_kw,
        )
        if len(a_test) == 0:
            test_loader = train_loader
        else:
            test_labels = _labels_for_pairs(mult_table, a_test, b_test)
            test_inputs = encode_pairs_one_hot(a_test, b_test, n_elements)
            test_loader = DataLoader(
                TensorDataset(test_inputs, test_labels),
                shuffle=False,
                **loader_kw,
            )
    else:
        collate = _make_collate(n_elements)
        train_labels = _labels_for_pairs(mult_table, a_train, b_train)
        train_loader = DataLoader(
            _GFIndexDataset(a_train, b_train, train_labels),
            shuffle=shuffle_train,
            collate_fn=collate,
            **loader_kw,
        )
        if len(a_test) == 0:
            test_loader = train_loader
        else:
            test_labels = _labels_for_pairs(mult_table, a_test, b_test)
            test_loader = DataLoader(
                _GFIndexDataset(a_test, b_test, test_labels),
                shuffle=False,
                collate_fn=collate,
                **loader_kw,
            )

    return train_loader, test_loader, mult_table, n_elements


def _collate_transformer_pairs(
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    a = torch.stack([item[0] for item in batch])
    b = torch.stack([item[1] for item in batch])
    labels = torch.stack([item[2] for item in batch])
    tokens = torch.stack([a, b], dim=1)
    return tokens, labels


def create_group_transformer_loaders(
    train_size: int,
    test_size: int,
    batch_size: int = 512,
    device: str | torch.device = "cpu",
    shuffle_train: bool = True,
    seed: int = 42,
    *,
    mult_table: torch.Tensor | None = None,
    group_n: int | None = None,
    psl2_q: int | None = None,
    ring_id: str | None = None,
    magma_id: str | None = None,
    num_workers: int = -1,
) -> tuple[DataLoader, DataLoader, torch.Tensor, int]:
    """DataLoader для GroupTransformer: вход [batch, 2] (a, b), label = a·b."""
    if mult_table is None:
        sources = sum(x is not None for x in (group_n, psl2_q, ring_id, magma_id))
        if sources != 1:
            raise ValueError("specify exactly one of group_n, psl2_q, ring_id, or magma_id")
        if group_n is not None:
            table_np, n_elements = build_alternating_mult_table(group_n)
        elif psl2_q is not None:
            table_np, n_elements = build_psl2_mult_table(psl2_q)
        elif ring_id is not None:
            table_np, _spec = build_ring_mul_table(ring_id)
            n_elements = table_np.shape[0]
        else:
            table_np, _spec = build_magma_table(magma_id)  # type: ignore[arg-type]
            n_elements = table_np.shape[0]
        mult_table = torch.from_numpy(table_np).long()
    else:
        n_elements = mult_table.shape[0]

    a_train, b_train, a_test, b_test = split_pair_indices(
        n_elements, train_size, test_size, seed
    )
    loader_kw = _loader_kwargs(
        device=device,
        batch_size=batch_size,
        shuffle_train=shuffle_train,
        num_workers=num_workers,
    )

    train_labels = _labels_for_pairs(mult_table, a_train, b_train)
    train_loader = DataLoader(
        _GFIndexDataset(a_train, b_train, train_labels),
        shuffle=shuffle_train,
        collate_fn=_collate_transformer_pairs,
        **loader_kw,
    )
    if len(a_test) == 0:
        test_loader = train_loader
    else:
        test_labels = _labels_for_pairs(mult_table, a_test, b_test)
        test_loader = DataLoader(
            _GFIndexDataset(a_test, b_test, test_labels),
            shuffle=False,
            collate_fn=_collate_transformer_pairs,
            **loader_kw,
        )
    return train_loader, test_loader, mult_table, n_elements


def create_alt_group_loaders(
    group_n: int,
    train_size: int,
    test_size: int,
    batch_size: int = 256,
    device: str | torch.device = "cpu",
    shuffle_train: bool = True,
    seed: int = 42,
    *,
    mult_table: torch.Tensor | None = None,
    num_workers: int = -1,
) -> tuple[DataLoader, DataLoader, torch.Tensor, int]:
    """Train/test DataLoader для умножения в A_{group_n}."""
    return create_group_loaders(
        train_size,
        test_size,
        batch_size=batch_size,
        device=device,
        shuffle_train=shuffle_train,
        seed=seed,
        mult_table=mult_table,
        group_n=group_n,
        num_workers=num_workers,
    )
