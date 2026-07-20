"""Galois field GF(2^n) arithmetic — PyTorch, GPU-compatible.

Портирование NumPy-ядра из `gf_mult_vectorized` на PyTorch тензоры.
Все функции работают с device-агностичными тензорами (CPU/CUDA).
"""

from __future__ import annotations

import functools
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

TEST_SEED_OFFSET = 1


@functools.lru_cache(maxsize=32)
def gf_mult_vectorized(n: int, poly: int) -> torch.Tensor:
    """Генерирует полную таблицу умножения GF(2^n) по неприводимому многочлену poly.

    Пример для n=5 (GF(32)): poly=37 (x^5 + x^2 + 1)

    Args:
        n: Степень поля. Размер = 2^n.
        poly: Неприводимый многочлен как целое число (бита = коэффициенты).

    Returns:
        torch.Tensor размера (2^n, 2^n), где [i, j] = i * j в GF(2^n).
    """
    num_elements = 1 << n  # 2^n
    A = torch.arange(num_elements, dtype=torch.int32)[:, None]
    B = torch.arange(num_elements, dtype=torch.int32)[None, :]

    # 1. Многочленное умножение без переноса (XOR = сложение в GF(2))
    res = torch.zeros(num_elements, num_elements, dtype=torch.int32)
    for i in range(n):
        bit_i = (B >> i) & 1
        res ^= (A << i) * bit_i

    # 2. Редукция по модулю неприводимого многочлена poly
    max_deg = 2 * n - 2
    for i in range(max_deg, n - 1, -1):
        mask = (res >> i) & 1
        res ^= (poly << (i - n)) * mask

    return res


def gf_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Сложение в GF(2^n): XOR в полиномиальном базисе."""
    return (a.long() ^ b.long()).long()


def gf_add_vectorized(n: int) -> torch.Tensor:
    """Таблица сложения [2^n, 2^n] — popcount не нужен, это XOR индексов."""
    num = 1 << n
    a = torch.arange(num, dtype=torch.long).unsqueeze(1)
    b = torch.arange(num, dtype=torch.long).unsqueeze(0)
    return (a ^ b).long()


def split_pair_indices(
    n: int,
    train_size: int,
    test_size: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Непересекающееся разбиение пар (a, b) на train и test без возврата.

    Пары кодируются плоским индексом ``a * 2^n + b``, затем перемешиваются
    через ``randperm``.

    Args:
        n: Степень поля.
        train_size: Число пар в train.
        test_size: Число пар в test.
        seed: Seed для перемешивания.

    Returns:
        (a_train, b_train, a_test, b_test) — long-тензоры на CPU.
    """
    num_elements = 1 << n
    total_pairs = num_elements * num_elements
    if train_size + test_size > total_pairs:
        raise ValueError(
            f"train_size + test_size ({train_size + test_size}) "
            f"превышает число пар ({total_pairs})"
        )

    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(total_pairs, generator=generator)

    flat_train = perm[:train_size]
    flat_test = perm[train_size : train_size + test_size]

    a_train = (flat_train // num_elements).long()
    b_train = (flat_train % num_elements).long()
    a_test = (flat_test // num_elements).long()
    b_test = (flat_test % num_elements).long()
    return a_train, b_train, a_test, b_test


def is_s_related(
    a: int,
    b: int,
    product: int,
    s_set: set[int],
) -> bool:
    """Пара связана с S: операнд или произведение в S."""
    return a in s_set or b in s_set or product in s_set


def select_s_elements(n: int, s_size: int, seed: int) -> list[int]:
    """Выбирает подмножество S ⊂ GF(2^n), |S| = s_size."""
    num_elements = 1 << n
    if s_size > num_elements:
        raise ValueError(f"s_size={s_size} > field size {num_elements}")
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(num_elements, generator=generator)
    return perm[:s_size].tolist()


def _spanning_tree_directed_pairs(
    p_pairs: list[tuple[int, int]],
    vertices: list[int],
) -> list[tuple[int, int]]:
    """Остовное дерево на вершинах `vertices` по рёбрам из P (неориентированно)."""
    p_set = set(p_pairs)
    adj: dict[int, set[int]] = defaultdict(set)
    for a, b in p_pairs:
        adj[a].add(b)
        adj[b].add(a)

    root = vertices[0]
    seen = {root}
    tree: list[tuple[int, int]] = []
    stack = [root]
    while stack:
        u = stack.pop()
        for w in sorted(adj[u]):
            if w not in seen:
                seen.add(w)
                if (u, w) in p_set:
                    tree.append((u, w))
                elif (w, u) in p_set:
                    tree.append((w, u))
                else:
                    raise RuntimeError(f"edge ({u},{w}) missing from P")
                stack.append(w)

    if len(seen) != len(vertices):
        raise ValueError(
            f"P-граф несвязен на V\\S: {len(seen)}/{len(vertices)} вершин достижимо"
        )
    return tree


@dataclass
class SpanningTreeSplit:
    """Разбиение D на train (остов + bulk) и test (S-hole + остаток P)."""

    n: int
    poly: int
    seed: int
    s_size: int
    s_elements: list[int]
    bulk_train_fraction: float
    a_train: torch.Tensor
    b_train: torch.Tensor
    a_test: torch.Tensor
    b_test: torch.Tensor
    tree_pairs: list[tuple[int, int]]
    stats: dict[str, Any]

    @property
    def train_size(self) -> int:
        return len(self.a_train)

    @property
    def test_size(self) -> int:
        return len(self.a_test)


def split_spanning_tree(
    n: int,
    poly: int,
    *,
    s_size: int = 20,
    seed: int = 42,
    bulk_train_fraction: float = 0.30,
    s_elements: list[int] | None = None,
) -> SpanningTreeSplit:
    """Spanning-tree split: S-hole → test; остов P + bulk → train.

    P = пары без связи с S (ни операнд, ни ответ в S).
    На V\\S строится остовное дерево (|V\\S|-1 пар).
    Train = tree ∪ random(bulk_train_fraction × (P \\ tree)).
    Test = R_S ∪ остаток P.
    """
    if not 0.0 <= bulk_train_fraction <= 1.0:
        raise ValueError("bulk_train_fraction must be in [0, 1]")

    table = gf_mult_vectorized(n, poly)
    num_elements = 1 << n
    total_pairs = num_elements * num_elements

    if s_elements is None:
        s_elements = select_s_elements(n, s_size, seed)
    s_set = set(s_elements)

    r_pairs: list[tuple[int, int]] = []
    p_pairs: list[tuple[int, int]] = []
    for a in range(num_elements):
        for b in range(num_elements):
            prod = int(table[a, b].item())
            if is_s_related(a, b, prod, s_set):
                r_pairs.append((a, b))
            else:
                p_pairs.append((a, b))

    vertices = [x for x in range(num_elements) if x not in s_set]
    tree_pairs = _spanning_tree_directed_pairs(p_pairs, vertices)
    tree_set = set(tree_pairs)

    non_tree = [p for p in p_pairs if p not in tree_set]
    generator = torch.Generator().manual_seed(seed + 17)
    perm = torch.randperm(len(non_tree), generator=generator).tolist()
    n_bulk_train = int(round(bulk_train_fraction * len(non_tree)))
    bulk_train_idx = set(perm[:n_bulk_train])

    train_pairs = list(tree_pairs) + [non_tree[i] for i in bulk_train_idx]
    test_pairs = r_pairs + [non_tree[i] for i in range(len(non_tree)) if i not in bulk_train_idx]

    train_pairs.sort()
    test_pairs.sort()

    def _to_tensors(pairs: list[tuple[int, int]]) -> tuple[torch.Tensor, torch.Tensor]:
        if not pairs:
            return torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long)
        a = torch.tensor([p[0] for p in pairs], dtype=torch.long)
        b = torch.tensor([p[1] for p in pairs], dtype=torch.long)
        return a, b

    a_train, b_train = _to_tensors(train_pairs)
    a_test, b_test = _to_tensors(test_pairs)

    assert len(train_pairs) + len(test_pairs) == total_pairs
    assert set(train_pairs).isdisjoint(test_pairs)

    stats = {
        "total_pairs": total_pairs,
        "s_related_test": len(r_pairs),
        "core_pool_p": len(p_pairs),
        "tree_pairs": len(tree_pairs),
        "bulk_train": n_bulk_train,
        "bulk_test": len(non_tree) - n_bulk_train,
        "train_size": len(train_pairs),
        "test_size": len(test_pairs),
        "vertices_outside_s": len(vertices),
    }

    return SpanningTreeSplit(
        n=n,
        poly=poly,
        seed=seed,
        s_size=len(s_elements),
        s_elements=s_elements,
        bulk_train_fraction=bulk_train_fraction,
        a_train=a_train,
        b_train=b_train,
        a_test=a_test,
        b_test=b_test,
        tree_pairs=tree_pairs,
        stats=stats,
    )


def save_spanning_tree_split(split: SpanningTreeSplit, path: str | Path) -> Path:
    """Сохраняет split в JSON (пары + метаданные)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "split_type": "spanning_tree",
        "n": split.n,
        "poly": split.poly,
        "seed": split.seed,
        "s_size": split.s_size,
        "s_elements": split.s_elements,
        "bulk_train_fraction": split.bulk_train_fraction,
        "stats": split.stats,
        "tree_pairs": split.tree_pairs,
        "train_pairs": list(zip(split.a_train.tolist(), split.b_train.tolist())),
        "test_pairs": list(zip(split.a_test.tolist(), split.b_test.tolist())),
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def load_spanning_tree_split(path: str | Path) -> SpanningTreeSplit:
    """Загружает split из JSON."""
    path = Path(path)
    data = json.loads(path.read_text())
    train_pairs = data["train_pairs"]
    test_pairs = data["test_pairs"]
    a_train = torch.tensor([p[0] for p in train_pairs], dtype=torch.long)
    b_train = torch.tensor([p[1] for p in train_pairs], dtype=torch.long)
    a_test = torch.tensor([p[0] for p in test_pairs], dtype=torch.long)
    b_test = torch.tensor([p[1] for p in test_pairs], dtype=torch.long)
    return SpanningTreeSplit(
        n=data["n"],
        poly=data["poly"],
        seed=data["seed"],
        s_size=data["s_size"],
        s_elements=data["s_elements"],
        bulk_train_fraction=data["bulk_train_fraction"],
        a_train=a_train,
        b_train=b_train,
        a_test=a_test,
        b_test=b_test,
        tree_pairs=[tuple(p) for p in data["tree_pairs"]],
        stats=data["stats"],
    )


def generate_dataset(
    n: int,
    poly: int,
    sample_size: int | None = None,
    device: str | torch.device = "cpu",
    generator: torch.Generator | None = None,
    a_idx: torch.Tensor | None = None,
    b_idx: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Генерирует датасет пар (a, b) → label = a * b в GF(2^n).

    Args:
        n: Степень поля.
        poly: Неприводимый многочлен.
        sample_size: Размер выборки (по умолчанию — полная таблица: 2^n × 2^n).
        device: Целевое устройство для выходных тензоров.
        generator: Генератор для случайной выборки (игнорируется если заданы a_idx/b_idx).
        a_idx: Готовые индексы a (для бенчмарков и тестов).
        b_idx: Готовые индексы b.

    Returns:
        inputs: [sample_size, 2^(n+1)] — два one-hot-encoded аргумента.
        labels: [sample_size] — целевые элементы GF(2^n).
    """
    table = gf_mult_vectorized(n, poly)
    num_elements = 1 << n

    if a_idx is not None and b_idx is not None:
        a_idx = a_idx.to(device)
        b_idx = b_idx.to(device)
    else:
        if sample_size is None:
            sample_size = num_elements * num_elements
        gen_device = generator.device if generator is not None else "cpu"
        a_idx = torch.randint(
            0, num_elements, (sample_size,), device=gen_device, generator=generator
        ).to(device)
        b_idx = torch.randint(
            0, num_elements, (sample_size,), device=gen_device, generator=generator
        ).to(device)

    labels = table.to(device)[a_idx, b_idx].to(torch.long)

    one_hot = torch.eye(num_elements, dtype=torch.float32, device=device)
    inputs = torch.cat([one_hot[a_idx], one_hot[b_idx]], dim=1)

    return inputs, labels


def get_irreducible_poly(n: int) -> int:
    """Возвращает неприводимый многочлен для GF(2^n).

    Args:
        n: Степень поля.

    Returns:
        Целое число, представляющее бинарное представление многочлена.

    Examples:
        >>> get_irreducible_poly(5)
        37  # x^5 + x^2 + 1
        >>> get_irreducible_poly(8)
        283  # x^8 + x^4 + x^3 + x + 1
    """
    polynomials: dict[int, int] = {
        1: 3,    # x + 1
        2: 7,    # x^2 + x + 1
        3: 13,   # x^3 + x^2 + 1
        4: 19,   # x^4 + x + 1
        5: 37,   # x^5 + x^2 + 1
        6: 109,  # x^6 + x^5 + x^3 + x^2 + 1
        7: 137,  # x^7 + x^3 + 1
        8: 283,  # x^8 + x^4 + x^3 + x + 1 (AES)
        9: 515,  # x^9 + x + 1
        10: 1033, # x^10 + x^3 + 1
    }
    if n not in polynomials:
        raise ValueError(
            f"Нет предустановленного полинома для n={n}. Укажите poly вручную."
        )
    return polynomials[n]
