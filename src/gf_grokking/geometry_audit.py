"""Unsupervised geometry checks for ring Transformer representations.

Complements linear DAS/IIA: Mantel label–embedding distance correlation,
PCA spectrum / participation ratio, and optional persistent homology (Ripser).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import kneighbors_graph

try:
    from ripser import ripser as _ripser
except ImportError:  # pragma: no cover
    _ripser = None


@dataclass
class MantelResult:
    r: float
    n: int
    p_perm: float | None
    label_metric: str
    embed_metric: str


@dataclass
class PCASpectrum:
    n_points: int
    n_dims: int
    participation_ratio: float
    dims_90: int
    dims_95: int
    top5_var_frac: list[float]


@dataclass
class PersistenceSummary:
    backend: str
    n_points: int
    pca_dims: int
    beta0_mid: int
    beta1_mid: int
    beta0_long: int
    beta1_long: int
    max_h1_lifetime: float


@dataclass
class ResidualGeometry:
    subspace_dim: int
    frac_var_in_subspace: float
    residual_participation_ratio: float
    residual_dims_95: int
    mantel_residual_vs_label_r: float | None


def scalar_label_distance(labels: np.ndarray) -> np.ndarray:
    """0/1 mismatch matrix for a single component label vector."""
    lab = np.asarray(labels, dtype=np.int64).reshape(-1)
    return (lab[:, None] != lab[None, :]).astype(np.float64)


def mantel_correlation(
    label_dist: np.ndarray,
    embed_dist: np.ndarray,
    *,
    n_perm: int = 0,
    seed: int = 42,
) -> MantelResult:
    """Pearson r between upper triangles of two distance matrices."""
    n = label_dist.shape[0]
    if embed_dist.shape != (n, n):
        raise ValueError("distance matrices must match shape")
    tri = np.triu_indices(n, k=1)
    x = label_dist[tri]
    y = embed_dist[tri]
    if x.std() < 1e-15 or y.std() < 1e-15:
        r = float("nan")
    else:
        r = float(np.corrcoef(x, y)[0, 1])

    p_perm: float | None = None
    if n_perm > 0 and np.isfinite(r):
        rng = np.random.default_rng(seed)
        count = 0
        for _ in range(n_perm):
            perm = rng.permutation(n)
            yp = embed_dist[perm][:, perm][tri]
            if yp.std() < 1e-15:
                continue
            rp = float(np.corrcoef(x, yp)[0, 1])
            if rp >= r:
                count += 1
        p_perm = count / n_perm
    return MantelResult(r=r, n=n, p_perm=p_perm, label_metric="component", embed_metric="euclidean")


def pca_spectrum(points: np.ndarray, *, max_components: int = 20) -> PCASpectrum:
    x = np.asarray(points, dtype=np.float64)
    n, d = x.shape
    k = min(max_components, n - 1, d)
    if k < 1:
        return PCASpectrum(
            n_points=n,
            n_dims=d,
            participation_ratio=float("nan"),
            dims_90=0,
            dims_95=0,
            top5_var_frac=[],
        )
    x0 = x - x.mean(axis=0, keepdims=True)
    pca = PCA(n_components=k, random_state=0)
    pca.fit(x0)
    ev = pca.explained_variance_
    total = ev.sum()
    if total < 1e-15:
        pr = float("nan")
        dims_90 = dims_95 = 0
        top5 = [0.0] * min(5, k)
    else:
        pr = float((total**2) / (np.sum(ev**2) + 1e-15))
        cum = np.cumsum(ev) / total
        dims_90 = int(np.searchsorted(cum, 0.90) + 1)
        dims_95 = int(np.searchsorted(cum, 0.95) + 1)
        fr = ev / total
        top5 = [float(v) for v in fr[: min(5, len(fr))]]
    return PCASpectrum(
        n_points=n,
        n_dims=d,
        participation_ratio=pr,
        dims_90=dims_90,
        dims_95=dims_95,
        top5_var_frac=top5,
    )


def pca_reduce(points: np.ndarray, n_components: int) -> np.ndarray:
    x = np.asarray(points, dtype=np.float64)
    k = min(n_components, x.shape[0] - 1, x.shape[1])
    if k < 1:
        return x[:, :1]
    x0 = x - x.mean(axis=0, keepdims=True)
    return PCA(n_components=k, random_state=0).fit_transform(x0)


def persistence_summary(
    points: np.ndarray,
    *,
    pca_dims: int = 10,
    lifetime_frac: float = 0.3,
) -> PersistenceSummary:
    """Persistent homology on PCA-reduced cloud; long features = lifetime > frac * max."""
    x = pca_reduce(points, pca_dims)
    n = x.shape[0]
    if _ripser is None:
        return _persistence_knn_fallback(x, lifetime_frac=lifetime_frac)
    out = _ripser(x, maxdim=1)
    dgms = out["dgms"]
    beta0_mid, beta1_mid = _count_midlife(dgms, idx=0), _count_midlife(dgms, idx=1)
    beta0_long, beta1_long = _count_long(dgms, 0, lifetime_frac), _count_long(dgms, 1, lifetime_frac)
    max_h1 = _max_lifetime(dgms, 1)
    return PersistenceSummary(
        backend="ripser",
        n_points=n,
        pca_dims=x.shape[1],
        beta0_mid=beta0_mid,
        beta1_mid=beta1_mid,
        beta0_long=beta0_long,
        beta1_long=beta1_long,
        max_h1_lifetime=max_h1,
    )


def _max_lifetime(dgms: list, dim: int) -> float:
    d = dgms[dim]
    if len(d) == 0:
        return 0.0
    finite = d[np.isfinite(d[:, 1])]
    if len(finite) == 0:
        return 0.0
    return float(np.max(finite[:, 1] - finite[:, 0]))


def _count_midlife(dgms: list, idx: int) -> int:
    d = dgms[idx]
    if len(d) == 0:
        return 0
    finite = d[np.isfinite(d[:, 1])]
    if len(finite) == 0:
        return max(0, len(d) - 1)
    lifetimes = finite[:, 1] - finite[:, 0]
    med = float(np.median(lifetimes)) if len(lifetimes) else 0.0
    return int(np.sum(lifetimes > med))


def _count_long(dgms: list, dim: int, lifetime_frac: float) -> int:
    d = dgms[dim]
    if len(d) == 0:
        return 0
    finite = d[np.isfinite(d[:, 1])]
    if len(finite) == 0:
        return 0
    lifetimes = finite[:, 1] - finite[:, 0]
    if len(lifetimes) == 0:
        return 0
    span = float(np.max(finite[:, 1]) - np.min(finite[:, 0]))
    if span < 1e-15:
        return 0
    thr = max(lifetime_frac * span, float(np.percentile(lifetimes, 90)))
    return int(np.sum(lifetimes >= thr))


def _persistence_knn_fallback(x: np.ndarray, *, lifetime_frac: float) -> PersistenceSummary:
    """Rough β₀ from kNN connectivity at a few radii (no Ripser installed)."""
    dists = pairwise_distances(x)
    radii = np.quantile(dists[np.triu_indices(len(x), k=1)], [0.05, 0.15, 0.3])
    beta0 = []
    for r in radii:
        adj = (dists <= r).astype(int)
        beta0.append(_connected_components(adj))
    return PersistenceSummary(
        backend="knn_fallback",
        n_points=len(x),
        pca_dims=x.shape[1],
        beta0_mid=int(np.median(beta0)),
        beta1_mid=0,
        beta0_long=int(np.min(beta0)),
        beta1_long=0,
        max_h1_lifetime=0.0,
    )


def _connected_components(adj: np.ndarray) -> int:
    n = adj.shape[0]
    seen = np.zeros(n, dtype=bool)
    comps = 0
    for i in range(n):
        if seen[i]:
            continue
        comps += 1
        stack = [i]
        seen[i] = True
        while stack:
            u = stack.pop()
            for v in np.where(adj[u])[0]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
    return comps


def orthonormal_columns(w: np.ndarray) -> np.ndarray:
    """W [d, k] with orthonormal columns (QR)."""
    q, _ = np.linalg.qr(np.asarray(w, dtype=np.float64), mode="reduced")
    return q


def project_subspace(points: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (projected, residual) for columns of W."""
    x = np.asarray(points, dtype=np.float64)
    q = orthonormal_columns(w)
    proj = (x @ q) @ q.T
    return proj, x - proj


def residual_geometry(
    points: np.ndarray,
    labels: np.ndarray,
    w: np.ndarray,
    *,
    mantel_perm: int = 0,
    seed: int = 42,
) -> ResidualGeometry:
    x = np.asarray(points, dtype=np.float64)
    q = orthonormal_columns(w)
    k = q.shape[1]
    proj, resid = project_subspace(x, q)
    var_total = float(np.sum(x**2))
    var_proj = float(np.sum(proj**2))
    frac = var_proj / var_total if var_total > 1e-15 else float("nan")
    spec = pca_spectrum(resid)
    mantel_r: float | None = None
    if labels is not None:
        ld = scalar_label_distance(labels)
        rd = pairwise_distances(resid)
        mantel_r = mantel_correlation(ld, rd, n_perm=mantel_perm, seed=seed).r
    return ResidualGeometry(
        subspace_dim=k,
        frac_var_in_subspace=frac,
        residual_participation_ratio=spec.participation_ratio,
        residual_dims_95=spec.dims_95,
        mantel_residual_vs_label_r=mantel_r,
    )


def knn_label_purity(points: np.ndarray, labels: np.ndarray, k: int = 10) -> float:
    """Mean fraction of kNN sharing the same scalar label (geometry–label alignment)."""
    x = np.asarray(points, dtype=np.float64)
    lab = np.asarray(labels, dtype=np.int64).reshape(-1)
    n = len(lab)
    k = min(k, n - 1)
    if k < 1:
        return float("nan")
    graph = kneighbors_graph(x, k, mode="connectivity", include_self=False)
    purities = []
    for i in range(n):
        nbrs = graph.getrow(i).indices
        if len(nbrs) == 0:
            continue
        purities.append(float(np.mean(lab[nbrs] == lab[i])))
    return float(np.mean(purities)) if purities else float("nan")


def hyperspherical_sincos(x: np.ndarray) -> np.ndarray:
    """Map x ∈ R^d to [r, sin θ₁, cos θ₁, …] (standard hyperspherical angles)."""
    v = np.asarray(x, dtype=np.float64).ravel()
    d = v.shape[0]
    if d == 0:
        return np.zeros(1, dtype=np.float64)
    r = float(np.linalg.norm(v))
    if r < 1e-12:
        return np.zeros(1 + 2 * max(0, d - 1), dtype=np.float64)
    feats: list[float] = [r]
    for i in range(d - 1):
        tail = v[i:]
        ri = float(np.linalg.norm(tail))
        theta = 0.0 if ri < 1e-12 else float(np.arccos(np.clip(v[i] / ri, -1.0, 1.0)))
        feats.extend([np.sin(theta), np.cos(theta)])
    return np.asarray(feats, dtype=np.float64)


def block_polar_sincos(x: np.ndarray, *, block_size: int = 2) -> np.ndarray:
    """Block-wise polar coords: per block (r, sin φ, cos φ)."""
    v = np.asarray(x, dtype=np.float64).ravel()
    feats: list[float] = []
    for i in range(0, v.shape[0], block_size):
        block = v[i : i + block_size]
        if block.size == 1:
            feats.append(abs(float(block[0])))
            continue
        r = float(np.linalg.norm(block))
        phi = float(np.arctan2(block[1], block[0]))
        feats.extend([r, np.sin(phi), np.cos(phi)])
    return np.asarray(feats, dtype=np.float64)


def sequential_index_pairs(d: int) -> np.ndarray:
    """Adjacent coordinate pairs: (0,1), (2,3), …"""
    dim = int(d)
    pairs: list[tuple[int, int]] = []
    for i in range(0, dim - 1, 2):
        pairs.append((i, i + 1))
    if dim % 2 == 1:
        pairs.append((dim - 1, dim - 1))
    return np.asarray(pairs, dtype=np.int64)


def greedy_max_abs_correlation_pairs(points: np.ndarray) -> np.ndarray:
    """Unsupervised pairing via greedy max-|corr| matching on train cloud [n, d]."""
    x = np.asarray(points, dtype=np.float64)
    if x.shape[0] < 2:
        return sequential_index_pairs(x.shape[1])
    corr = np.corrcoef(x, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    d = corr.shape[0]
    score = np.abs(corr)
    np.fill_diagonal(score, 0.0)
    edges: list[tuple[float, int, int]] = []
    for i in range(d):
        for j in range(i + 1, d):
            edges.append((float(score[i, j]), i, j))
    edges.sort(reverse=True, key=lambda t: t[0])
    used: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for _, i, j in edges:
        if i in used or j in used:
            continue
        pairs.append((i, j))
        used.add(i)
        used.add(j)
        if len(used) >= d:
            break
    if d % 2 == 1:
        leftover = next(i for i in range(d) if i not in used)
        pairs.append((leftover, leftover))
    return np.asarray(pairs, dtype=np.int64)


def head_corr_index_pairs(points: np.ndarray, *, nhead: int) -> np.ndarray:
    """Greedy |corr| pairing separately within each attention-head slice."""
    x = np.asarray(points, dtype=np.float64)
    d = x.shape[1]
    if d % nhead != 0:
        raise ValueError(f"d={d} not divisible by nhead={nhead}")
    head_dim = d // nhead
    chunks: list[np.ndarray] = []
    for h in range(nhead):
        sl = slice(h * head_dim, (h + 1) * head_dim)
        local = greedy_max_abs_correlation_pairs(x[:, sl])
        global_pairs = np.column_stack(
            [local[:, 0] + sl.start, local[:, 1] + sl.start]
        )
        chunks.append(global_pairs)
    return np.vstack(chunks)


def block_polar_paired_sincos(x: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """Polar features for explicit coordinate pairs [[i,j], …]."""
    v = np.asarray(x, dtype=np.float64).ravel()
    feats: list[float] = []
    for i, j in np.asarray(pairs, dtype=np.int64):
        if int(i) == int(j):
            feats.append(abs(float(v[int(i)])))
            continue
        block = np.asarray([v[int(i)], v[int(j)]], dtype=np.float64)
        r = float(np.linalg.norm(block))
        phi = float(np.arctan2(block[1], block[0]))
        feats.extend([r, np.sin(phi), np.cos(phi)])
    return np.asarray(feats, dtype=np.float64)


def block_polar_paired_matrix(points: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    rows = [block_polar_paired_sincos(row, pairs) for row in np.asarray(points, dtype=np.float64)]
    max_len = max(r.shape[0] for r in rows)
    out = np.zeros((len(rows), max_len), dtype=np.float64)
    for i, row in enumerate(rows):
        out[i, : row.shape[0]] = row
    return out


def random_projection_polar_matrix(
    points: np.ndarray,
    *,
    n_projections: int = 32,
    seed: int = 0,
) -> np.ndarray:
    """Fixed random 2D projections → polar (r, sin φ, cos φ) each."""
    x = np.asarray(points, dtype=np.float64)
    bases = fit_random_projection_bases(x.shape[1], n_projections=n_projections, seed=seed)
    return random_projection_polar_matrix_fixed(x, bases)


def _random_projection_polar_row(
    row: np.ndarray,
    projections: list[np.ndarray],
) -> np.ndarray:
    feats: list[float] = []
    for q in projections:
        z = row @ q
        r = float(np.linalg.norm(z))
        if r < 1e-12:
            feats.extend([0.0, 0.0, 0.0])
        else:
            phi = float(np.arctan2(z[1], z[0]))
            feats.extend([r, np.sin(phi), np.cos(phi)])
    return np.asarray(feats, dtype=np.float64)


def fit_random_projection_bases(
    d: int,
    *,
    n_projections: int = 32,
    seed: int = 0,
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    bases: list[np.ndarray] = []
    for _ in range(int(n_projections)):
        q = rng.standard_normal((d, 2))
        q, _ = np.linalg.qr(q, mode="reduced")
        bases.append(q.astype(np.float64))
    return bases


def random_projection_polar_matrix_fixed(
    points: np.ndarray,
    bases: list[np.ndarray],
) -> np.ndarray:
    x = np.asarray(points, dtype=np.float64)
    rows = [_random_projection_polar_row(row, bases) for row in x]
    max_len = max(r.shape[0] for r in rows)
    out = np.zeros((len(rows), max_len), dtype=np.float64)
    for i, row in enumerate(rows):
        out[i, : row.shape[0]] = row
    return out


@dataclass
class PolarFeatureFitter:
    mode: str
    pairs: np.ndarray | None = None
    pca: PCA | None = None
    proj_bases: list[np.ndarray] | None = None

    def transform(self, points: np.ndarray) -> np.ndarray:
        x = np.asarray(points, dtype=np.float64)
        if self.mode == "cartesian":
            return x
        if self.mode == "block_naive":
            return block_polar_paired_matrix(x, sequential_index_pairs(x.shape[1]))
        if self.mode in {"corr_matching", "head_corr"}:
            assert self.pairs is not None
            return block_polar_paired_matrix(x, self.pairs)
        if self.mode == "pca_block":
            assert self.pca is not None
            z = self.pca.transform(x)
            return block_polar_paired_matrix(z, sequential_index_pairs(z.shape[1]))
        if self.mode == "random_proj":
            assert self.proj_bases is not None
            return random_projection_polar_matrix_fixed(x, self.proj_bases)
        raise ValueError(f"unknown polar fitter mode {self.mode!r}")

    @classmethod
    def fit(
        cls,
        train_points: np.ndarray,
        mode: str,
        *,
        nhead: int = 4,
        pca_dims: int = 32,
        n_projections: int = 32,
        proj_seed: int = 0,
    ) -> PolarFeatureFitter:
        x = np.asarray(train_points, dtype=np.float64)
        if mode == "cartesian":
            return cls(mode=mode)
        if mode == "block_naive":
            return cls(mode=mode, pairs=sequential_index_pairs(x.shape[1]))
        if mode == "corr_matching":
            return cls(mode=mode, pairs=greedy_max_abs_correlation_pairs(x))
        if mode == "head_corr":
            return cls(mode=mode, pairs=head_corr_index_pairs(x, nhead=nhead))
        if mode == "pca_block":
            k = min(int(pca_dims), x.shape[0] - 1, x.shape[1])
            pca = PCA(n_components=k, random_state=0)
            pca.fit(x)
            return cls(mode=mode, pca=pca)
        if mode == "random_proj":
            return cls(
                mode=mode,
                proj_bases=fit_random_projection_bases(
                    x.shape[1], n_projections=n_projections, seed=proj_seed
                ),
            )
        raise ValueError(f"unknown polar fitter mode {mode!r}")


def polar_feature_matrix(
    points: np.ndarray,
    *,
    mode: str = "hyperspherical",
    block_size: int = 2,
    pca_dims: int | None = None,
) -> np.ndarray:
    """Row-wise polar features for a point cloud [n, d]."""
    x = np.asarray(points, dtype=np.float64)
    if pca_dims is not None:
        x = pca_reduce(x, int(pca_dims))
    if mode == "hyperspherical":
        rows = [hyperspherical_sincos(row) for row in x]
    elif mode == "block":
        rows = [block_polar_sincos(row, block_size=block_size) for row in x]
    elif mode == "pca_hyperspherical":
        z = pca_reduce(x, min(16, x.shape[0] - 1, x.shape[1]))
        rows = [hyperspherical_sincos(row) for row in z]
    else:
        raise ValueError(f"unknown polar mode {mode!r}")
    max_len = max(r.shape[0] for r in rows)
    out = np.zeros((len(rows), max_len), dtype=np.float64)
    for i, row in enumerate(rows):
        out[i, : row.shape[0]] = row
    return out
