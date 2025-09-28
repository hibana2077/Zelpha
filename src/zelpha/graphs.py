"""Graph construction utilities for Zelpha experiments."""

from __future__ import annotations

import math
import random
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy.sparse import csr_matrix, issparse
from scipy.sparse.csgraph import shortest_path
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import NearestNeighbors


def set_seed(seed: int = 42) -> None:
    """Seed NumPy and Python's RNG for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)


def rbf_kernel(X: np.ndarray, gamma: float) -> np.ndarray:
    """Compute an RBF kernel matrix with bandwidth ``gamma``."""
    X_norm = (X ** 2).sum(axis=1)[:, None]
    sq_dists = X_norm + X_norm.T - 2 * (X @ X.T)
    sq_dists = np.maximum(sq_dists, 0.0)
    return np.exp(-gamma * sq_dists)


def _estimate_rbf_gamma(X: np.ndarray, n_neighbors: int = 50) -> float:
    """Heuristic to estimate RBF gamma from median neighbor distance.

    Uses median of Euclidean distances to the 1..k-th neighbors (excluding self)
    and sets gamma = 1 / (median^2 + eps).
    """
    n = X.shape[0]
    k = min(n_neighbors, n)
    nbrs = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(X)
    dists, _ = nbrs.kneighbors(X)
    med = np.median(dists[:, 1:])
    return float(1.0 / (med ** 2 + 1e-12))


def topk_sparsify(W: np.ndarray, k: int, symmetrize: bool = True) -> csr_matrix:
    """Keep the top-``k`` neighbors per row of ``W`` and return a sparse matrix."""
    n = W.shape[0]
    np.fill_diagonal(W, 0.0)
    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []
    for i in range(n):
        row = W[i]
        if k < len(row):
            idx = np.argpartition(row, -k)[-k:]
            idx = idx[np.argsort(-row[idx])]
        else:
            idx = np.argsort(-row)
        vals = row[idx]
        mask = vals > 0
        idx = idx[mask]
        vals = vals[mask]
        rows.extend([i] * len(idx))
        cols.extend(idx.tolist())
        data.extend(vals.tolist())
    A = csr_matrix((data, (rows, cols)), shape=(n, n))
    if symmetrize:
        A = A.maximum(A.T)
    return A


def build_cosine_knn_graph(X: np.ndarray, k: int) -> csr_matrix:
    """Construct a symmetric cosine k-NN similarity graph."""
    nbrs = NearestNeighbors(n_neighbors=min(k + 1, len(X)), metric="cosine")
    nbrs.fit(X)
    distances, indices = nbrs.kneighbors(X, return_distance=True)
    n = X.shape[0]
    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []
    for i in range(n):
        idx_row = indices[i]
        dist_row = distances[i]
        sims = 1.0 - dist_row
        mask = idx_row != i
        idx_row = idx_row[mask]
        sims = sims[mask]
        if len(idx_row) > k:
            top = np.argpartition(sims, -k)[-k:]
            idx_row = idx_row[top]
            sims = sims[top]
        sims = np.clip(sims, 0.0, 1.0)
        rows.extend([i] * len(idx_row))
        cols.extend(idx_row.tolist())
        data.extend(sims.tolist())
    A = csr_matrix((data, (rows, cols)), shape=(n, n))
    return A.maximum(A.T)


def build_rbf_knn_graph(X: np.ndarray, k: int, gamma: float | None = None) -> csr_matrix:
    """Construct a symmetric RBF/Gaussian k-NN affinity graph.

    Parameters
    ----------
    X : array-like of shape (n, d)
        Input features.
    k : int
        Number of neighbors to keep per node.
    gamma : float, optional
        RBF bandwidth; if None, estimated from median neighbor distance.

    Returns
    -------
    csr_matrix
        Symmetric kNN affinity matrix with weights in [0, 1].
    """
    if gamma is None:
        gamma = _estimate_rbf_gamma(X)
    K = rbf_kernel(X, gamma=gamma)
    # Normalize to cosine-style affinity (optional but common)
    diag = np.sqrt(np.clip(np.diag(K), 1e-12, None))
    H = (K / diag[:, None]) / diag[None, :]
    A = topk_sparsify(H, k=k, symmetrize=True)
    # Clip any numerical spill
    A.data = np.clip(A.data, 0.0, 1.0)
    return A


def build_snn_graph(
    X: np.ndarray,
    k: int,
    *,
    metric: str = "euclidean",
    sim: str = "jaccard",
    include_mutual_only: bool = False,
) -> csr_matrix:
    """Construct a Shared Nearest Neighbor (SNN) similarity graph.

    For each node i, compute its k-NN set N_k(i) under the chosen metric.
    The edge weight between i and j (where j in N_k(i) or i in N_k(j)) is:
      - sim="jaccard": |N_k(i) ∩ N_k(j)| / |N_k(i) ∪ N_k(j)| (in [0,1])
      - sim="count" (aka SNN count): |N_k(i) ∩ N_k(j)| (integer in [0,k])

    Parameters
    ----------
    X : array-like of shape (n, d)
        Input features.
    k : int
        Number of nearest neighbors to define the neighbor sets N_k(i).
    metric : str, default="euclidean"
        Distance metric for kNN search.
    sim : {"jaccard", "count"}, default="jaccard"
        Similarity function.
    include_mutual_only : bool, default=False
        If True, only keep edges where i in N_k(j) and j in N_k(i) (mutual kNN).

    Returns
    -------
    csr_matrix
        Symmetric SNN similarity matrix. For sim="count", values are ints; for
        sim="jaccard", values are floats in [0,1].
    """
    n = X.shape[0]
    # Compute kNN sets
    nbrs = NearestNeighbors(n_neighbors=min(k + 1, n), metric=metric)
    nbrs.fit(X)
    _, indices = nbrs.kneighbors(X, return_distance=True)

    # Build neighbor sets excluding self
    neigh_sets: List[set[int]] = []
    for i in range(n):
        row = indices[i]
        row = row[row != i]
        if len(row) > k:
            row = row[:k]
        neigh_sets.append(set(int(x) for x in row))

    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []

    for i in range(n):
        Ni = neigh_sets[i]
        if not Ni:
            continue
        # Candidate neighbors: either Ni or mutual ones based on flag
        candidates = list(Ni)
        for j in candidates:
            if include_mutual_only and i not in neigh_sets[j]:
                continue
            Nj = neigh_sets[j]
            inter = len(Ni & Nj)
            if inter == 0:
                continue
            if sim == "jaccard":
                uni = len(Ni | Nj)
                w = inter / max(uni, 1)
            elif sim == "count":
                w = float(inter)
            else:
                raise ValueError("sim must be one of {'jaccard','count'}")
            rows.append(i)
            cols.append(j)
            data.append(float(w))

    A = csr_matrix((data, (rows, cols)), shape=(n, n))
    # Symmetrize by max to keep strongest agreement
    A = A.maximum(A.T)
    if sim == "jaccard":
        # numerical stability
        A.data = np.clip(A.data, 0.0, 1.0)
    return A


def normalized_laplacian(A: csr_matrix) -> csr_matrix:
    if not issparse(A):
        A = csr_matrix(A)
    n = A.shape[0]
    d = np.array(A.sum(axis=1)).ravel()
    d = np.maximum(d, 1e-12)
    d_inv_sqrt = 1.0 / np.sqrt(d)
    D_inv_sqrt = csr_matrix((d_inv_sqrt, (np.arange(n), np.arange(n))), shape=(n, n))
    return csr_matrix(np.eye(n)) - (D_inv_sqrt @ A @ D_inv_sqrt)


def heat_kernel_from_graph(A: csr_matrix, t: float, rank: int = 128) -> np.ndarray:
    n = A.shape[0]
    L = normalized_laplacian(A)
    k = min(rank, n - 2) if n > 2 else 1
    vals, vecs = eigsh(L, k=k, which="SM")
    vals = np.clip(vals, 0.0, None)
    exp_vals = np.exp(-t * vals)
    K = (vecs * exp_vals) @ vecs.T
    return np.maximum(K, 0.0)


def zelpha_graph(
    X: np.ndarray,
    k: int,
    alpha: float = 0.5,
    t: float = 1.0,
    kernel: str = "rbf",
    gamma: float | None = None,
    heat_rank: int = 128,
    *,
    fusion: str = "power",
    return_dense: bool = False,
) -> csr_matrix | np.ndarray:
    """Construct a Zelpha graph (sparse by default) from features X.

    Parameters
    ----------
    X : np.ndarray
        Input features of shape (n, d).
    k : int
        Number of neighbors to keep per node when sparsifying.
    alpha : float, default=0.5
        Trade-off between feature affinity H and heat kernel K_heat.
        Only used when fusion="power" (geometric) or fusion="convex" (arithmetic).
    t : float, default=1.0
        Diffusion time for heat kernel exp(-t L).
    kernel : {"rbf", "linear"}, default="rbf"
        Base feature kernel to build K.
    gamma : float or None, default=None
        Bandwidth for RBF kernel. If None, estimated from median neighbor distance.
    heat_rank : int, default=128
        Rank for low-rank heat kernel approximation via eigensolver.
    fusion : {"power", "convex"}, default="power"
        Fusion strategy between H and K_heat.
        - "power": geometric-style fusion (H**alpha) * (K_heat**(1-alpha)) [original behavior].
        - "convex": arithmetic convex combination alpha*H + (1-alpha)*K_heat (PD-preserving before sparsify).
    return_dense : bool, default=False
        If True, return the dense fused affinity matrix (np.ndarray). Note that applying
        top-k sparsification breaks PD in general; use this to obtain a PD matrix for
        kernel/spectral methods. If False, return a symmetric kNN sparse graph (csr_matrix).

    Returns
    -------
    csr_matrix or np.ndarray
        By default, a symmetric kNN sparse adjacency (csr_matrix). If return_dense=True,
        returns the dense fused affinity matrix (np.ndarray).
    """
    n = X.shape[0]
    if kernel == "rbf":
        if gamma is None:
            gamma = _estimate_rbf_gamma(X)
        K = rbf_kernel(X, gamma=gamma)
    elif kernel == "linear":
        K_lin = X @ X.T
        diag = np.sqrt(np.clip(np.diag(K_lin), 1e-12, None))
        K = (K_lin / diag[:, None]) / diag[None, :]
    else:
        raise ValueError(f"Unsupported kernel: {kernel}")

    diag = np.sqrt(np.clip(np.diag(K), 1e-12, None))
    H = (K / diag[:, None]) / diag[None, :]
    A0 = topk_sparsify(H.copy(), k=k, symmetrize=True)
    K_heat = heat_kernel_from_graph(A0, t=t, rank=heat_rank)
    if fusion == "power":
        W_dense = (H ** alpha) * (K_heat ** (1.0 - alpha))
    elif fusion == "convex":
        W_dense = alpha * H + (1.0 - alpha) * K_heat
    else:
        raise ValueError("fusion must be one of {'power','convex'}")

    if return_dense:
        # Return dense affinity for kernel/spectral downstream; PD is preserved for fusion='convex'
        # (before sparsification). For fusion='power', PD is not guaranteed.
        return np.asarray(W_dense)

    return topk_sparsify(W_dense, k=k, symmetrize=True)


def labels_from_swiss_t(tvals: np.ndarray, n_bins: int = 20) -> np.ndarray:
    bins = np.quantile(tvals, np.linspace(0, 1, n_bins + 1))
    bins = np.unique(bins)
    labels = np.digitize(tvals, bins[1:-1], right=True)
    return labels.astype(int)


def neighbor_purity(A: csr_matrix, labels: np.ndarray, k: int) -> float:
    rows = A.tolil()
    purities: List[float] = []
    for i in range(A.shape[0]):
        neigh = rows.rows[i]
        neigh = [j for j in neigh if j != i]
        if not neigh:
            continue
        same = sum(1 for j in neigh if labels[j] == labels[i])
        purities.append(same / max(len(neigh), 1))
    return float(np.mean(purities)) if purities else 0.0


def cut_energy(A: csr_matrix, labels: np.ndarray) -> float:
    A_coo = A.tocoo()
    mask = A_coo.row < A_coo.col
    w = A_coo.data[mask]
    i = A_coo.row[mask]
    j = A_coo.col[mask]
    inter = w[labels[i] != labels[j]].sum()
    total = w.sum() + 1e-12
    return float(inter / total)


def geodesic_distortion(
    A: csr_matrix,
    tvals: np.ndarray,
    yvals: np.ndarray,
    pairs: int = 500,
    length_transform: str = "neglog",
) -> float:
    n = A.shape[0]
    t_std = (tvals - tvals.mean()) / (tvals.std() + 1e-12)
    y_std = (yvals - yvals.mean()) / (yvals.std() + 1e-12)
    U = np.stack([t_std, y_std], axis=1)

    B = A.tocsr(copy=True)
    if length_transform == "neglog":
        B.data = -np.log(np.clip(B.data, 1e-9, 1.0))
    elif length_transform == "recip":
        B.data = 1.0 / np.clip(B.data, 1e-9, None)
    elif length_transform == "one-minus":
        B.data = 1.0 - np.clip(B.data, 0.0, 1.0) + 1e-9
    else:
        raise ValueError("Unknown length_transform")

    src_nodes = np.random.choice(n, size=min(n, max(50, pairs // 2)), replace=False)
    D_graph = shortest_path(B, directed=False, indices=src_nodes, return_predecessors=False)

    dists: List[float] = []
    for _ in range(pairs):
        s = int(np.random.choice(src_nodes))
        t = int(np.random.randint(0, n))
        dg = D_graph[np.where(src_nodes == s)[0][0], t]
        if not np.isfinite(dg):
            continue
        du = np.linalg.norm(U[s] - U[t])
        if du <= 1e-9:
            continue
        dists.append(abs(dg - du) / du)
    return float(np.mean(dists)) if dists else float("nan")


__all__ = [
    "set_seed",
    "rbf_kernel",
    "build_rbf_knn_graph",
    "build_snn_graph",
    "topk_sparsify",
    "build_cosine_knn_graph",
    "normalized_laplacian",
    "heat_kernel_from_graph",
    "zelpha_graph",
    "labels_from_swiss_t",
    "neighbor_purity",
    "cut_energy",
    "geodesic_distortion",
]
