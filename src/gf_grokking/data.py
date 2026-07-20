"""Dataset and DataLoader helpers for GF(2^n) arithmetic."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Literal

InputEncoding = Literal["onehot", "bits"]

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

from .gf_field import (
    SpanningTreeSplit,
    gf_add,
    gf_mult_vectorized,
    load_spanning_tree_split,
    split_pair_indices,
)

# Порог: если one-hot train+test > этого значения — храним только индексы,
# one-hot строится векторизованно в collate по батчу (быстро, без OOM).
EAGER_MAX_BYTES = 1_500_000_000  # ~1.5 GiB


def estimate_one_hot_bytes(n: int, train_size: int, test_size: int) -> int:
    """Оценка RAM для материализованных one-hot входов (float32)."""
    input_dim = 2 * (1 << n)
    return (train_size + test_size) * input_dim * 4


def encode_pairs_one_hot(
    a_idx: torch.Tensor,
    b_idx: torch.Tensor,
    num_elements: int,
) -> torch.Tensor:
    """Векторизованный one-hot: [batch, 2 * num_elements]."""
    a_oh = F.one_hot(a_idx.long(), num_elements).float()
    b_oh = F.one_hot(b_idx.long(), num_elements).float()
    return torch.cat([a_oh, b_oh], dim=-1)


def encode_index_bits(idx: torch.Tensor, n: int) -> torch.Tensor:
    """Битовое представление элемента GF(2^n): [batch, n], младший бит = x^0."""
    idx = idx.long()
    shifts = torch.arange(n, device=idx.device, dtype=torch.long)
    return ((idx.unsqueeze(-1) >> shifts) & 1).float()


def encode_pairs_bits(
    a_idx: torch.Tensor,
    b_idx: torch.Tensor,
    n: int,
) -> torch.Tensor:
    """Конкатенация битовых векторов: [batch, 2 * n]."""
    return torch.cat([encode_index_bits(a_idx, n), encode_index_bits(b_idx, n)], dim=-1)


def mlp_input_dim(n: int, *, input_encoding: InputEncoding = "onehot") -> int:
    num_elements = 1 << n
    if input_encoding == "onehot":
        return num_elements * 2
    if input_encoding == "bits":
        return n * 2
    raise ValueError(f"unknown input_encoding: {input_encoding}")


class _GFIndexDataset(Dataset):
    """Хранит только индексы пар (a, b) и метки — без one-hot в памяти."""

    def __init__(
        self,
        a_indices: torch.Tensor,
        b_indices: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        self.a_indices = a_indices
        self.b_indices = b_indices
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.a_indices[idx], self.b_indices[idx], self.labels[idx]


def _make_collate(num_elements: int, *, n: int, input_encoding: InputEncoding) -> Callable:
    def collate(
        batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        a = torch.stack([item[0] for item in batch])
        b = torch.stack([item[1] for item in batch])
        labels = torch.stack([item[2] for item in batch])
        if input_encoding == "onehot":
            inputs = encode_pairs_one_hot(a, b, num_elements)
        else:
            inputs = encode_pairs_bits(a, b, n)
        return inputs, labels

    return collate


def _collate_transformer(
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collate для Transformer: пары (a, b) как [batch, 2] int64."""
    a = torch.stack([item[0] for item in batch])
    b = torch.stack([item[1] for item in batch])
    labels = torch.stack([item[2] for item in batch])
    pairs = torch.stack([a, b], dim=1)
    return pairs, labels


class _GFOpDataset(Dataset):
    """op, a, b, label — для op_token режима."""

    def __init__(
        self,
        op: torch.Tensor,
        a_indices: torch.Tensor,
        b_indices: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        self.op = op
        self.a_indices = a_indices
        self.b_indices = b_indices
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.op[idx], self.a_indices[idx], self.b_indices[idx], self.labels[idx]


class _GFDualDataset(Dataset):
    """a, b, label_add, label_mul — для dual_head."""

    def __init__(
        self,
        a_indices: torch.Tensor,
        b_indices: torch.Tensor,
        labels_add: torch.Tensor,
        labels_mul: torch.Tensor,
    ) -> None:
        self.a_indices = a_indices
        self.b_indices = b_indices
        self.labels_add = labels_add
        self.labels_mul = labels_mul

    def __len__(self) -> int:
        return len(self.a_indices)

    def __getitem__(
        self, idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.a_indices[idx],
            self.b_indices[idx],
            self.labels_add[idx],
            self.labels_mul[idx],
        )


def _collate_op_token(
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    op = torch.stack([item[0] for item in batch])
    a = torch.stack([item[1] for item in batch])
    b = torch.stack([item[2] for item in batch])
    labels = torch.stack([item[3] for item in batch])
    x = torch.stack([op, a, b], dim=1)
    return x, labels


def _collate_dual_head(
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    a = torch.stack([item[0] for item in batch])
    b = torch.stack([item[1] for item in batch])
    la = torch.stack([item[2] for item in batch])
    lm = torch.stack([item[3] for item in batch])
    pairs = torch.stack([a, b], dim=1)
    return pairs, la, lm


def expand_pairs_to_op_samples(
    n: int,
    poly: int,
    a: torch.Tensor,
    b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Из (a,b) пар строит 2× выборку: add и mul; и dual labels."""
    mul_table = gf_mult_vectorized(n, poly)
    labels_add = gf_add(a, b)
    labels_mul = mul_table[a.long(), b.long()].long()

    op_add = torch.zeros(len(a), dtype=torch.long)
    op_mul = torch.ones(len(a), dtype=torch.long)
    op = torch.cat([op_add, op_mul])
    a2 = torch.cat([a, a])
    b2 = torch.cat([b, b])
    labels_op = torch.cat([labels_add, labels_mul])
    return op, a2, b2, labels_op, labels_add, labels_mul


def create_gf_op_loaders(
    n: int,
    poly: int,
    train_size: int,
    test_size: int | None = None,
    *,
    mode: str = "op_token",
    batch_size: int = 256,
    device: str | torch.device = "cpu",
    shuffle_train: bool = True,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """DataLoader для + и × в GF(2^n).

    mode:
      - op_token: вход [op, a, b], одна метка
      - dual_head / single_head: вход [a, b], метки (add, mul) в батче
    """
    if test_size is None:
        test_size = train_size // 4

    a_train, b_train, a_test, b_test = split_pair_indices(n, train_size, test_size, seed)
    use_cuda = str(device).startswith("cuda")
    pin_memory = use_cuda and torch.cuda.is_available()

    tr = expand_pairs_to_op_samples(n, poly, a_train, b_train)
    te = expand_pairs_to_op_samples(n, poly, a_test, b_test)

    if mode == "op_token":
        train_loader = DataLoader(
            _GFOpDataset(tr[0], tr[1], tr[2], tr[3]),
            batch_size=batch_size,
            shuffle=shuffle_train,
            collate_fn=_collate_op_token,
            pin_memory=pin_memory,
        )
        test_loader = DataLoader(
            _GFOpDataset(te[0], te[1], te[2], te[3]),
            batch_size=len(te[3]),
            shuffle=False,
            collate_fn=_collate_op_token,
            pin_memory=pin_memory,
        )
        return train_loader, test_loader

    if mode in ("dual_head", "single_head"):
        train_loader = DataLoader(
            _GFDualDataset(a_train, b_train, tr[4], tr[5]),
            batch_size=batch_size,
            shuffle=shuffle_train,
            collate_fn=_collate_dual_head,
            pin_memory=pin_memory,
        )
        test_loader = DataLoader(
            _GFDualDataset(a_test, b_test, te[4], te[5]),
            batch_size=len(a_test),
            shuffle=False,
            collate_fn=_collate_dual_head,
            pin_memory=pin_memory,
        )
        return train_loader, test_loader

    raise ValueError(f"unknown mode: {mode}")


def _labels_for_pairs(
    n: int,
    poly: int,
    a_idx: torch.Tensor,
    b_idx: torch.Tensor,
) -> torch.Tensor:
    table = gf_mult_vectorized(n, poly)
    return table[a_idx, b_idx].long()


def create_transformer_loaders_from_split(
    split: SpanningTreeSplit,
    batch_size: int = 256,
    device: str | torch.device = "cpu",
    shuffle_train: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """DataLoader из готового SpanningTreeSplit."""
    return _create_transformer_loaders_from_indices(
        n=split.n,
        poly=split.poly,
        a_train=split.a_train,
        b_train=split.b_train,
        a_test=split.a_test,
        b_test=split.b_test,
        batch_size=batch_size,
        device=device,
        shuffle_train=shuffle_train,
    )


def create_transformer_loaders_from_path(
    dataset_path: str | Path,
    batch_size: int = 256,
    device: str | torch.device = "cpu",
    shuffle_train: bool = True,
) -> tuple[DataLoader, DataLoader, SpanningTreeSplit]:
    """Загружает JSON split и строит DataLoader'ы."""
    split = load_spanning_tree_split(dataset_path)
    loaders = create_transformer_loaders_from_split(
        split, batch_size=batch_size, device=device, shuffle_train=shuffle_train
    )
    return loaders[0], loaders[1], split


SPANNING_TREE_SUBSET_LABELS: dict[str, str] = {
    "tree": "Остовное дерево",
    "bulk_train": "Bulk train (докинутые)",
    "bulk_test": "Bulk test (не в train)",
    "s_related": "S-связанные",
}


def spanning_tree_pair_subsets(
    split: SpanningTreeSplit,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Разбивает spanning-tree split на 4 подмножества для eval по эпохам."""
    from .gf_field import is_s_related

    tree_set = {tuple(p) for p in split.tree_pairs}
    s_set = set(split.s_elements)
    table = gf_mult_vectorized(split.n, split.poly)

    buckets: dict[str, list[tuple[int, int]]] = {
        "tree": [],
        "bulk_train": [],
        "bulk_test": [],
        "s_related": [],
    }
    for a, b in zip(split.a_train.tolist(), split.b_train.tolist(), strict=True):
        key = "tree" if (a, b) in tree_set else "bulk_train"
        buckets[key].append((a, b))
    for a, b in zip(split.a_test.tolist(), split.b_test.tolist(), strict=True):
        prod = int(table[a, b].item())
        key = "s_related" if is_s_related(a, b, prod, s_set) else "bulk_test"
        buckets[key].append((a, b))

    out: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for name, pairs in buckets.items():
        if not pairs:
            out[name] = (torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long))
        else:
            out[name] = (
                torch.tensor([p[0] for p in pairs], dtype=torch.long),
                torch.tensor([p[1] for p in pairs], dtype=torch.long),
            )
    return out


def create_transformer_subset_loader(
    n: int,
    poly: int,
    a_idx: torch.Tensor,
    b_idx: torch.Tensor,
    batch_size: int = 512,
    device: str | torch.device = "cpu",
) -> DataLoader:
    """DataLoader для произвольного подмножества пар (a, b)."""
    labels = _labels_for_pairs(n, poly, a_idx, b_idx)
    use_cuda = str(device).startswith("cuda")
    pin_memory = use_cuda and torch.cuda.is_available()
    n_pairs = len(labels)
    eff_batch = min(batch_size, max(1, n_pairs))
    return DataLoader(
        _GFIndexDataset(a_idx, b_idx, labels),
        batch_size=eff_batch,
        shuffle=False,
        collate_fn=_collate_transformer,
        pin_memory=pin_memory,
    )


def pick_augment_element_a(split: SpanningTreeSplit, a: int | None = None) -> int:
    """Элемент a ∈ GF(2^n) \\ S, a ∉ {0, 1}."""
    s_set = set(split.s_elements)
    num = 1 << split.n
    if a is not None:
        if a in s_set:
            raise ValueError(f"a={a} must not be in S")
        if a in (0, 1):
            raise ValueError("a must not be 0 or 1")
        if not 0 <= a < num:
            raise ValueError(f"a={a} out of range for GF(2^{split.n})")
        return a
    for candidate in range(num):
        if candidate not in s_set and candidate not in (0, 1):
            return candidate
    raise RuntimeError("no valid augment element found")


def valid_augment_candidates(split: SpanningTreeSplit) -> list[int]:
    """Все a ∈ GF(2^n) \\ S, a ∉ {0, 1}."""
    s_set = set(split.s_elements)
    num = 1 << split.n
    return [x for x in range(num) if x not in s_set and x not in (0, 1)]


def sample_random_augment_a(
    split: SpanningTreeSplit,
    count: int,
    seed: int,
) -> list[int]:
    """Случайные distinct a для a×S-аугментации."""
    candidates = valid_augment_candidates(split)
    if count > len(candidates):
        raise ValueError(f"count={count} > available candidates {len(candidates)}")
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(candidates), generator=generator).tolist()
    return [candidates[i] for i in perm[:count]]


def sample_random_s_elements(
    split: SpanningTreeSplit,
    count: int,
    seed: int,
) -> list[int]:
    """Случайные distinct элементы из S."""
    if count > len(split.s_elements):
        raise ValueError(f"count={count} > |S|={len(split.s_elements)}")
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(split.s_elements), generator=generator).tolist()
    return [split.s_elements[i] for i in perm[:count]]


def a_cross_s_pairs(
    a: int,
    s_elements: list[int],
    *,
    both_directions: bool = False,
) -> list[tuple[int, int]]:
    """Пары (a, s) для всех s ∈ S; опционально и (s, a)."""
    pairs = [(a, s) for s in s_elements]
    if both_directions:
        pairs.extend((s, a) for s in s_elements)
    return pairs


def augment_spanning_tree_with_a_cross_s(
    split: SpanningTreeSplit,
    a: int,
    *,
    both_directions: bool = False,
) -> tuple[SpanningTreeSplit, list[tuple[int, int]]]:
    """Добавляет пары a×S в train и убирает их из test."""
    added = a_cross_s_pairs(a, split.s_elements, both_directions=both_directions)
    return augment_spanning_tree_with_pairs(
        split,
        added,
        augment_meta={
            "augment_mode": "aS",
            "aS_augment_a": a,
            "aS_both_directions": both_directions,
        },
    )


def s_cross_s_pairs(s_elements: list[int]) -> list[tuple[int, int]]:
    """Все пары (s, s') с обоими операндами в S."""
    return [(s, sp) for s in s_elements for sp in s_elements]


def s_cross_s_pairs_anchor(s_elements: list[int], s_anchor: int) -> list[tuple[int, int]]:
    """Пары из S×S с участием s_anchor (строка + столбец)."""
    if s_anchor not in s_elements:
        raise ValueError(f"s_anchor={s_anchor} not in S")
    pairs = [(s_anchor, sp) for sp in s_elements]
    pairs.extend((sp, s_anchor) for sp in s_elements)
    return list(dict.fromkeys(pairs))


def build_s_augment_pairs(
    mode: str,
    s_elements: list[int],
    *,
    a: int | None = None,
    b: int | None = None,
    s_anchor: int | None = None,
) -> list[tuple[int, int]]:
    """Собирает пары для режимов aS / aS_bS / sS / aS_sS.

    - aS:    {(a, s) : s ∈ S}
    - aS_bS: {(a, s), (b, s) : s ∈ S}
    - sS:    {(s, s') : s, s' ∈ S}
    - aS_sS: aS ∪ sS; если задан s_anchor — sS только с участием s_anchor
    """
    pairs: list[tuple[int, int]] = []
    if mode in ("aS", "aS_bS", "aS_sS"):
        if a is None:
            raise ValueError(f"mode {mode} requires a")
        pairs.extend((a, s) for s in s_elements)
    if mode == "aS_bS":
        if b is None:
            raise ValueError("mode aS_bS requires b")
        pairs.extend((b, s) for s in s_elements)
    if mode == "sS":
        pairs.extend(s_cross_s_pairs(s_elements))
    elif mode == "aS_sS":
        if s_anchor is not None:
            pairs.extend(s_cross_s_pairs_anchor(s_elements, s_anchor))
        else:
            pairs.extend(s_cross_s_pairs(s_elements))
    if mode not in ("aS", "aS_bS", "sS", "aS_sS"):
        raise ValueError(f"unknown augment mode: {mode}")
    return list(dict.fromkeys(pairs))


def augment_spanning_tree_with_pairs(
    split: SpanningTreeSplit,
    added: list[tuple[int, int]],
    *,
    augment_meta: dict | None = None,
) -> tuple[SpanningTreeSplit, list[tuple[int, int]]]:
    """Добавляет пары в train и убирает их из test."""
    added_set = set(added)

    train_pairs = list(zip(split.a_train.tolist(), split.b_train.tolist(), strict=True))
    for p in added:
        if p not in train_pairs:
            train_pairs.append(p)

    test_pairs = [
        p
        for p in zip(split.a_test.tolist(), split.b_test.tolist(), strict=True)
        if p not in added_set
    ]

    a_train = torch.tensor([p[0] for p in train_pairs], dtype=torch.long)
    b_train = torch.tensor([p[1] for p in train_pairs], dtype=torch.long)
    a_test = torch.tensor([p[0] for p in test_pairs], dtype=torch.long)
    b_test = torch.tensor([p[1] for p in test_pairs], dtype=torch.long)

    stats = dict(split.stats)
    stats["train_size"] = len(train_pairs)
    stats["test_size"] = len(test_pairs)
    stats["augment_pairs"] = len(added)
    if augment_meta:
        stats.update(augment_meta)

    augmented = SpanningTreeSplit(
        n=split.n,
        poly=split.poly,
        seed=split.seed,
        s_size=split.s_size,
        s_elements=list(split.s_elements),
        bulk_train_fraction=split.bulk_train_fraction,
        a_train=a_train,
        b_train=b_train,
        a_test=a_test,
        b_test=b_test,
        tree_pairs=list(split.tree_pairs),
        stats=stats,
    )
    return augmented, added


def augment_spanning_tree_mode(
    split: SpanningTreeSplit,
    mode: str,
    *,
    a: int | None = None,
    b: int | None = None,
    s_anchor: int | None = None,
) -> tuple[SpanningTreeSplit, list[tuple[int, int]]]:
    """Аугментация train по именованному режиму."""
    if mode == "aS":
        if a is None:
            a = pick_augment_element_a(split)
        pick_augment_element_a(split, a)
        added = build_s_augment_pairs(mode, split.s_elements, a=a)
    elif mode == "aS_bS":
        if a is None or b is None:
            sampled = sample_random_augment_a(split, 2, seed=split.seed + 99)
            a = a if a is not None else sampled[0]
            b = b if b is not None else (sampled[1] if len(sampled) > 1 else sampled[0])
        pick_augment_element_a(split, a)
        pick_augment_element_a(split, b)
        if a == b:
            raise ValueError("aS_bS requires distinct a and b")
        added = build_s_augment_pairs(mode, split.s_elements, a=a, b=b)
    elif mode in ("sS", "aS_sS"):
        if mode == "aS_sS" and a is None:
            a = pick_augment_element_a(split)
        if a is not None:
            pick_augment_element_a(split, a)
        if s_anchor is not None and s_anchor not in split.s_elements:
            raise ValueError(f"s_anchor={s_anchor} not in S")
        added = build_s_augment_pairs(
            mode, split.s_elements, a=a, b=b, s_anchor=s_anchor
        )
    else:
        raise ValueError(f"unknown mode: {mode}")

    meta = {
        "augment_mode": mode,
        "augment_a": a,
        "augment_b": b,
        "augment_s_anchor": s_anchor,
    }
    return augment_spanning_tree_with_pairs(split, added, augment_meta=meta)


def _create_transformer_loaders_from_indices(
    n: int,
    poly: int,
    a_train: torch.Tensor,
    b_train: torch.Tensor,
    a_test: torch.Tensor,
    b_test: torch.Tensor,
    batch_size: int,
    device: str | torch.device,
    shuffle_train: bool,
) -> tuple[DataLoader, DataLoader]:
    train_labels = _labels_for_pairs(n, poly, a_train, b_train)
    test_labels = _labels_for_pairs(n, poly, a_test, b_test)

    use_cuda = str(device).startswith("cuda")
    pin_memory = use_cuda and torch.cuda.is_available()

    train_loader = DataLoader(
        _GFIndexDataset(a_train, b_train, train_labels),
        batch_size=batch_size,
        shuffle=shuffle_train,
        collate_fn=_collate_transformer,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        _GFIndexDataset(a_test, b_test, test_labels),
        batch_size=len(test_labels),
        shuffle=False,
        collate_fn=_collate_transformer,
        pin_memory=pin_memory,
    )
    return train_loader, test_loader


def create_transformer_loaders(
    n: int,
    poly: int,
    train_size: int,
    test_size: int | None = None,
    batch_size: int = 256,
    device: str | torch.device = "cpu",
    shuffle_train: bool = True,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """DataLoader для Transformer: вход [batch, 2] (индексы a, b), без one-hot."""
    if test_size is None:
        test_size = train_size // 4

    a_train, b_train, a_test, b_test = split_pair_indices(
        n, train_size, test_size, seed
    )
    return _create_transformer_loaders_from_indices(
        n=n,
        poly=poly,
        a_train=a_train,
        b_train=b_train,
        a_test=a_test,
        b_test=b_test,
        batch_size=batch_size,
        device=device,
        shuffle_train=shuffle_train,
    )


def _create_eager_loaders(
    n: int,
    poly: int,
    a_train: torch.Tensor,
    b_train: torch.Tensor,
    a_test: torch.Tensor,
    b_test: torch.Tensor,
    batch_size: int,
    shuffle_train: bool,
    pin_memory: bool,
    *,
    input_encoding: InputEncoding = "onehot",
) -> tuple[DataLoader, DataLoader]:
    """Материализует входы целиком на CPU — быстрый путь для небольших полей."""
    num_elements = 1 << n

    train_labels = _labels_for_pairs(n, poly, a_train, b_train)
    test_labels = _labels_for_pairs(n, poly, a_test, b_test)

    if input_encoding == "onehot":
        train_inputs = encode_pairs_one_hot(a_train, b_train, num_elements)
        test_inputs = encode_pairs_one_hot(a_test, b_test, num_elements)
    else:
        train_inputs = encode_pairs_bits(a_train, b_train, n)
        test_inputs = encode_pairs_bits(a_test, b_test, n)

    train_loader = DataLoader(
        TensorDataset(train_inputs, train_labels),
        batch_size=batch_size,
        shuffle=shuffle_train,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        TensorDataset(test_inputs, test_labels),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory,
    )
    return train_loader, test_loader


def _create_indexed_loaders(
    n: int,
    poly: int,
    a_train: torch.Tensor,
    b_train: torch.Tensor,
    a_test: torch.Tensor,
    b_test: torch.Tensor,
    batch_size: int,
    shuffle_train: bool,
    pin_memory: bool,
    *,
    input_encoding: InputEncoding = "onehot",
) -> tuple[DataLoader, DataLoader]:
    """Индексы на CPU, вход строится в collate по батчу — для больших полей."""
    num_elements = 1 << n
    collate = _make_collate(num_elements, n=n, input_encoding=input_encoding)

    train_labels = _labels_for_pairs(n, poly, a_train, b_train)
    test_labels = _labels_for_pairs(n, poly, a_test, b_test)

    train_loader = DataLoader(
        _GFIndexDataset(a_train, b_train, train_labels),
        batch_size=batch_size,
        shuffle=shuffle_train,
        collate_fn=collate,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        _GFIndexDataset(a_test, b_test, test_labels),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate,
        pin_memory=pin_memory,
    )
    return train_loader, test_loader


def create_analysis_test_loader(
    n: int,
    poly: int,
    train_size: int,
    test_size: int,
    batch_size: int = 4096,
    seed: int = 42,
) -> DataLoader:
    """Test DataLoader с материализованным one-hot на CPU (для mech-interp).

    Не грузит train — только test, батчами, без pin_memory/GPU transfer всего датасета.
    """
    _, _, a_test, b_test = split_pair_indices(n, train_size, test_size, seed)
    num_elements = 1 << n
    test_labels = _labels_for_pairs(n, poly, a_test, b_test)
    test_inputs = encode_pairs_one_hot(a_test, b_test, num_elements)
    print(
        f"Analysis test set: {len(test_labels)} pairs, "
        f"one-hot RAM ≈ {test_inputs.numel() * 4 / 1e9:.2f} GiB (CPU)"
    )
    return DataLoader(
        TensorDataset(test_inputs, test_labels),
        batch_size=batch_size,
        shuffle=False,
    )


def create_loaders(
    n: int,
    poly: int,
    train_size: int,
    test_size: int | None = None,
    batch_size: int = 256,
    device: str | torch.device = "cpu",
    shuffle_train: bool = True,
    seed: int = 42,
    *,
    input_encoding: InputEncoding = "onehot",
) -> tuple[DataLoader, DataLoader]:
    """Создаёт train/test DataLoader для задачи умножения в GF(2^n).

    Train/test — **непересекающееся** разбиение всех пар (a, b) без возврата:
    ``randperm(2^n × 2^n)`` → первые train_size в train, следующие test_size в test.

    Данные всегда на CPU. Для небольших полей one-hot материализуется целиком
    (быстро); для больших хранятся только индексы, encoding — в collate.

    Args:
        n: Степень поля.
        poly: Неприводимый многочлен.
        train_size: Размер обучающей выборки.
        test_size: Размер тестовой выборки (по умолчанию = 25% от train).
        batch_size: Размер батча.
        device: Если CUDA — включает pin_memory для быстрого transfer.
        shuffle_train: Перемешивать ли обучающую выборку.
        seed: Seed для разбиения train/test.

    Returns:
        (train_loader, test_loader)
    """
    if test_size is None:
        test_size = train_size // 4

    a_train, b_train, a_test, b_test = split_pair_indices(
        n, train_size, test_size, seed
    )

    use_cuda = str(device).startswith("cuda")
    pin_memory = use_cuda and torch.cuda.is_available()

    eager = (
        input_encoding == "onehot"
        and estimate_one_hot_bytes(n, train_size, test_size) <= EAGER_MAX_BYTES
    )
    factory = _create_eager_loaders if eager else _create_indexed_loaders

    return factory(
        n=n,
        poly=poly,
        a_train=a_train,
        b_train=b_train,
        a_test=a_test,
        b_test=b_test,
        batch_size=batch_size,
        shuffle_train=shuffle_train,
        pin_memory=pin_memory,
        input_encoding=input_encoding,
    )
