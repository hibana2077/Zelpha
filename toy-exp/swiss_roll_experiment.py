import json
import math
import os
import random
from dataclasses import dataclass, asdict
from typing import Dict, Tuple

import numpy as np
from scipy.sparse import csr_matrix, issparse
from scipy.sparse.csgraph import shortest_path
from scipy.sparse.linalg import eigsh
from sklearn.datasets import make_swiss_roll
from sklearn.neighbors import NearestNeighbors


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)


def rbf_kernel(X: np.ndarray, gamma: float) -> np.ndarray:
    # K_ij = exp(-gamma * ||x_i - x_j||^2)
    # Efficient pairwise squared distances using (x - y)^2 = ||x||^2 + ||y||^2 - 2 x.y
    X_norm = (X ** 2).sum(axis=1)[:, None]
    sq_dists = X_norm + X_norm.T - 2 * (X @ X.T)
    sq_dists = np.maximum(sq_dists, 0.0)
    K = np.exp(-gamma * sq_dists)
    return K


def topk_sparsify(W: np.ndarray, k: int, symmetrize: bool = True) -> csr_matrix:
    n = W.shape[0]
    # Zero diagonal
    np.fill_diagonal(W, 0.0)
    # For each row, keep top-k
    rows, cols, data = [], [], []
    for i in range(n):
        row = W[i]
        if k < len(row):
            idx = np.argpartition(row, -k)[-k:]
            idx = idx[np.argsort(-row[idx])]
        else:
            idx = np.argsort(-row)
        vals = row[idx]
        # Filter out non-positive weights (just in case)
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
    # sklearn cosine metric returns distance = 1 - cosine_similarity
    nbrs = NearestNeighbors(n_neighbors=min(k + 1, len(X)), metric="cosine")
    nbrs.fit(X)
    distances, indices = nbrs.kneighbors(X, return_distance=True)
    n = X.shape[0]
    rows, cols, data = [], [], []
    for i in range(n):
        # Skip self (distance 0 at first position)
        idx_row = indices[i]
        dist_row = distances[i]
        # Convert to similarity in [-1, 1]; ensure non-negative for weights
        sims = 1.0 - dist_row  # cosine sim
        # Drop self
        mask = idx_row != i
        idx_row = idx_row[mask]
        sims = sims[mask]
        # Keep top-k (k neighbors besides self)
        if len(idx_row) > k:
            top = np.argpartition(sims, -k)[-k:]
            idx_row = idx_row[top]
            sims = sims[top]
        # Clip to [0, 1] to avoid negative weights
        sims = np.clip(sims, 0.0, 1.0)
        rows.extend([i] * len(idx_row))
        cols.extend(idx_row.tolist())
        data.extend(sims.tolist())
    A = csr_matrix((data, (rows, cols)), shape=(n, n))
    A = A.maximum(A.T)
    return A


def normalized_laplacian(A: csr_matrix) -> csr_matrix:
    if not issparse(A):
        A = csr_matrix(A)
    n = A.shape[0]
    d = np.array(A.sum(axis=1)).ravel()
    d = np.maximum(d, 1e-12)
    d_inv_sqrt = 1.0 / np.sqrt(d)
    D_inv_sqrt = csr_matrix((d_inv_sqrt, (np.arange(n), np.arange(n))), shape=(n, n))
    L_sym = csr_matrix((n, n))
    L_sym = csr_matrix(np.eye(n)) - (D_inv_sqrt @ A @ D_inv_sqrt)
    return L_sym


def heat_kernel_from_graph(A: csr_matrix, t: float, rank: int = 128) -> np.ndarray:
    n = A.shape[0]
    L = normalized_laplacian(A)
    # Compute smallest eigenpairs of L (symmetric PSD, eigenvalues in [0, 2])
    k = min(rank, n - 2) if n > 2 else 1
    vals, vecs = eigsh(L, k=k, which="SM")
    # Ensure numerical stability
    vals = np.clip(vals, 0.0, None)
    exp_vals = np.exp(-t * vals)
    # Reconstruct approximate heat kernel: U exp(-t Λ) U^T
    K = (vecs * exp_vals) @ vecs.T
    # Clip small negatives due to numerical errors
    K = np.maximum(K, 0.0)
    return K


def zelpha_graph(X: np.ndarray, k: int, alpha: float = 0.5, t: float = 1.0,
                 kernel: str = "rbf", gamma: float | None = None,
                 heat_rank: int = 128) -> csr_matrix:
    n = X.shape[0]
    # Build H = D^{-1/2} K D^{-1/2}; for RBF, K_ii=1 so H=K
    if kernel == "rbf":
        if gamma is None:
            # median heuristic on pairwise distances
            nbrs = NearestNeighbors(n_neighbors=min(50, n), metric="euclidean").fit(X)
            dists, _ = nbrs.kneighbors(X)
            med = np.median(dists[:, 1:])  # skip self
            gamma = 1.0 / (med ** 2 + 1e-12)
        K = rbf_kernel(X, gamma=gamma)
    elif kernel == "linear":
        K = X @ X.T
        diag = np.sqrt(np.clip(np.diag(K), 1e-12, None))
        H = (K / diag[:, None]) / diag[None, :]
        K = H
    else:
        raise ValueError(f"Unsupported kernel: {kernel}")

    # For RBF, H = K (since K_ii = 1). For generality, compute H anyway.
    diag = np.sqrt(np.clip(np.diag(K), 1e-12, None))
    H = (K / diag[:, None]) / diag[None, :]

    # Initial sparse graph from H top-k (to define Laplacian)
    A0 = topk_sparsify(H.copy(), k=k, symmetrize=True)

    # Heat kernel on the initial graph
    K_heat = heat_kernel_from_graph(A0, t=t, rank=heat_rank)

    # Zelpha affinity: elementwise product H^alpha * K_heat^(1-alpha)
    W_dense = (H ** alpha) * (K_heat ** (1.0 - alpha))
    # Sparsify to kNN and symmetrize
    A = topk_sparsify(W_dense, k=k, symmetrize=True)
    return A


def labels_from_swiss_t(tvals: np.ndarray, n_bins: int = 20) -> np.ndarray:
    # Bin the unrolled coordinate t into bands
    bins = np.quantile(tvals, np.linspace(0, 1, n_bins + 1))
    # Handle potential duplicate bin edges
    bins = np.unique(bins)
    # np.digitize returns indices in 1..len(bins)-1
    labels = np.digitize(tvals, bins[1:-1], right=True)
    return labels.astype(int)


def neighbor_purity(A: csr_matrix, labels: np.ndarray, k: int) -> float:
    n = A.shape[0]
    rows = A.tolil()
    purities = []
    for i in range(n):
        neigh = rows.rows[i]
        if i in neigh:
            # shouldn't happen due to zero diagonal, but guard anyway
            neigh = [j for j in neigh if j != i]
        if len(neigh) == 0:
            continue
        same = sum(1 for j in neigh if labels[j] == labels[i])
        purities.append(same / max(len(neigh), 1))
    return float(np.mean(purities)) if purities else 0.0


def cut_energy(A: csr_matrix, labels: np.ndarray) -> float:
    # Fraction of total edge weight that crosses between different bands
    # Use upper triangle to avoid double counting
    A_coo = A.tocoo()
    mask = A_coo.row < A_coo.col
    i = A_coo.row[mask]
    j = A_coo.col[mask]
    w = A_coo.data[mask]
    inter = w[labels[i] != labels[j]].sum()
    total = w.sum() + 1e-12
    return float(inter / total)


def geodesic_distortion(A: csr_matrix, tvals: np.ndarray, yvals: np.ndarray,
                        pairs: int = 500, length_transform: str = "neglog") -> float:
    n = A.shape[0]
    # Ground-truth geodesic in unrolled plane: use (t, y), standardized
    t_std = (tvals - tvals.mean()) / (tvals.std() + 1e-12)
    y_std = (yvals - yvals.mean()) / (yvals.std() + 1e-12)
    U = np.stack([t_std, y_std], axis=1)

    # Build edge lengths from similarities
    B = A.tocsr(copy=True)
    if length_transform == "neglog":
        B.data = -np.log(np.clip(B.data, 1e-9, 1.0))
    elif length_transform == "recip":
        B.data = 1.0 / np.clip(B.data, 1e-9, None)
    elif length_transform == "one-minus":
        # Map similarity in [0,1] to non-negative length; encourages short paths through high similarity edges
        B.data = 1.0 - np.clip(B.data, 0.0, 1.0) + 1e-9
    else:
        raise ValueError("Unknown length_transform")

    # Sample source nodes and compute single-source shortest paths
    src_nodes = np.random.choice(n, size=min(n, max(50, pairs // 2)), replace=False)
    D_graph = shortest_path(B, directed=False, indices=src_nodes, return_predecessors=False)

    # Collect random pairs among computed sources/all nodes
    dists = []
    for _ in range(pairs):
        s = int(np.random.choice(src_nodes))
        t = int(np.random.randint(0, n))
        # Graph distance
        dg = D_graph[np.where(src_nodes == s)[0][0], t]
        if not np.isfinite(dg):
            continue  # skip unreachable
        # True geodesic distance in (t, y)
        du = np.linalg.norm(U[s] - U[t])
        if du <= 1e-9:
            continue
        dists.append(abs(dg - du) / du)
    return float(np.mean(dists)) if dists else float("nan")


@dataclass
class ExperimentConfig:
    n: int = 2000
    noise: float = 0.0
    k: int = 10
    alpha: float = 0.5
    t: float = 1.0
    heat_rank: int = 128
    n_bins: int = 20
    pairs: int = 500
    seed: int = 42


def run_experiment(cfg: ExperimentConfig, length_transform: str = "neglog") -> Dict[str, float]:
    set_seed(cfg.seed)
    X, tvals = make_swiss_roll(n_samples=cfg.n, noise=cfg.noise, random_state=cfg.seed)
    yvals = X[:, 1]  # vertical coordinate

    labels = labels_from_swiss_t(tvals, n_bins=cfg.n_bins)

    # Cosine kNN graph
    A_cos = build_cosine_knn_graph(X, k=cfg.k)
    purity_cos = neighbor_purity(A_cos, labels, k=cfg.k)
    cut_cos = cut_energy(A_cos, labels)
    dist_cos = geodesic_distortion(A_cos, tvals, yvals, pairs=cfg.pairs, length_transform=length_transform)

    # Zelpha graph
    A_zel = zelpha_graph(X, k=cfg.k, alpha=cfg.alpha, t=cfg.t, kernel="rbf", heat_rank=cfg.heat_rank)
    purity_zel = neighbor_purity(A_zel, labels, k=cfg.k)
    cut_zel = cut_energy(A_zel, labels)
    dist_zel = geodesic_distortion(A_zel, tvals, yvals, pairs=cfg.pairs, length_transform=length_transform)

    return {
        "purity_cosine": purity_cos,
        "purity_zelpha": purity_zel,
        "cut_energy_cosine": cut_cos,
        "cut_energy_zelpha": cut_zel,
        "geodesic_distortion_cosine": dist_cos,
        "geodesic_distortion_zelpha": dist_zel,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Swiss roll kNN graph quality: cosine vs Zelpha")
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--t", type=float, default=1.0)
    parser.add_argument("--heat-rank", type=int, default=128)
    parser.add_argument("--bins", type=int, default=20)
    parser.add_argument("--pairs", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", type=str, default="results")
    parser.add_argument("--length-transform", type=str, default="neglog", choices=["neglog", "recip", "one-minus"]) 
    args = parser.parse_args()

    cfg = ExperimentConfig(
        n=args.n,
        noise=args.noise,
        k=args.k,
        alpha=args.alpha,
        t=args.t,
        heat_rank=args.heat_rank,
        n_bins=args.bins,
        pairs=args.pairs,
        seed=args.seed,
    )

    print("Config:")
    print(cfg)
    results = run_experiment(cfg, length_transform=args.length_transform)

    print("\nResults:")
    for k, v in results.items():
        print(f"{k}: {v:.6f}")

    # Save
    os.makedirs(args.outdir, exist_ok=True)
    out_json = os.path.join(args.outdir, "swiss_roll_metrics.json")
    payload = {"config": asdict(cfg), "length_transform": args.length_transform, "results": results}
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved metrics to {out_json}")


if __name__ == "__main__":
    main()
