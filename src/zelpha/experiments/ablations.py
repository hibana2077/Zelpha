"""Ablation and visualization utilities for Zelpha graphs."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import matplotlib as mpl
mpl.use("Agg")  # set non-interactive backend before importing pyplot
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.datasets import make_circles, make_moons, make_swiss_roll
from sklearn.preprocessing import StandardScaler

from ..graphs import (
    heat_kernel_from_graph,
    neighbor_purity,
    rbf_kernel,
    topk_sparsify,
    zelpha_graph,
)


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)


def compute_H(X: np.ndarray, kernel: str = "rbf", gamma: float | None = None) -> np.ndarray:
    if kernel == "rbf":
        if gamma is None:
            Xs = StandardScaler().fit_transform(X)
            X_norm = (Xs ** 2).sum(axis=1)[:, None]
            sq_dists = X_norm + X_norm.T - 2 * (Xs @ Xs.T)
            sq_dists = np.maximum(sq_dists, 0.0)
            med = np.median(np.sqrt(np.clip(sq_dists + np.eye(len(X)) * 1e9, 0.0, None)))
            gamma = 1.0 / (med ** 2 + 1e-12)
        K = rbf_kernel(X, gamma=gamma)
    elif kernel == "linear":
        K = X @ X.T
    else:
        raise ValueError(f"Unsupported kernel: {kernel}")

    diag = np.sqrt(np.clip(np.diag(K), 1e-12, None))
    H = (K / diag[:, None]) / diag[None, :]
    return H


def graphs_from_ablation(
    X: np.ndarray,
    k: int,
    alpha: float,
    t: float,
    heat_rank: int = 128,
    kernel: str = "rbf",
) -> Dict[str, csr_matrix | np.ndarray]:
    H = compute_H(X, kernel=kernel)
    A0 = topk_sparsify(H.copy(), k=k, symmetrize=True)
    K_heat = heat_kernel_from_graph(A0, t=t, rank=heat_rank)
    A_H = topk_sparsify(H.copy(), k=k, symmetrize=True)
    A_heat = topk_sparsify(K_heat.copy(), k=k, symmetrize=True)
    A_zel = zelpha_graph(X, k=k, alpha=alpha, t=t, kernel=kernel, heat_rank=heat_rank)
    return {
        "H_dense": H,
        "K_heat_dense": K_heat,
        "A_H": A_H,
        "A_heat": A_heat,
        "A_zelpha": A_zel,
    }


def to_dense(A: csr_matrix | np.ndarray) -> np.ndarray:
    return A if isinstance(A, np.ndarray) else A.toarray()


def plot_adj_matrices(Ws: List[np.ndarray], titles: List[str], out_path: str, vmin: float = 0.0, vmax: float | None = None) -> None:
    m = len(Ws)
    ncols = min(m, 3)
    nrows = int(np.ceil(m / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows), constrained_layout=True)
    axes = np.array(axes).reshape(-1)
    vmax = vmax if vmax is not None else max(np.max(W) for W in Ws)
    for i, (W, title) in enumerate(zip(Ws, titles)):
        ax = axes[i]
        im = ax.imshow(W, cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    fig.suptitle("Adjacency / Affinity Matrices", fontsize=12)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_adj_differences(W1: np.ndarray, W2: np.ndarray, name1: str, name2: str, out_path: str) -> None:
    diff = W1 - W2
    lim = np.max(np.abs(diff)) + 1e-9
    fig, ax = plt.subplots(1, 1, figsize=(5, 4), constrained_layout=True)
    im = ax.imshow(diff, cmap="bwr", vmin=-lim, vmax=lim)
    ax.set_title(f"{name1} - {name2}")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def heat_diffuse_from_seeds(A: csr_matrix, seeds: List[int], t: float, rank: int = 128) -> np.ndarray:
    n = A.shape[0]
    K = heat_kernel_from_graph(A, t=t, rank=rank)
    S = np.zeros((n, len(seeds)), dtype=float)
    for j, s in enumerate(seeds):
        e = np.zeros(n, dtype=float)
        e[s] = 1.0
        S[:, j] = K @ e
    S = (S - S.min(axis=0, keepdims=True)) / (S.max(axis=0, keepdims=True) - S.min(axis=0, keepdims=True) + 1e-12)
    return S


def scatter_diffusion(X2d: np.ndarray, S: np.ndarray, seeds: List[int], title: str, out_path: str) -> None:
    intensity = S.mean(axis=1)
    fig, ax = plt.subplots(1, 1, figsize=(5, 4), constrained_layout=True)
    sc = ax.scatter(X2d[:, 0], X2d[:, 1], c=intensity, cmap="plasma", s=8, edgecolors="none")
    ax.scatter(X2d[seeds, 0], X2d[seeds, 1], c="cyan", s=50, edgecolors="k", marker="*", label="seeds")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="best", fontsize=8)
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="diffusion intensity")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


@dataclass
class AblationConfig:
    dataset: str = "moons"
    n: int = 800
    noise: float = 0.1
    k: int = 10
    alpha: float = 0.5
    t: float = 1.0
    heat_rank: int = 128
    kernel: str = "rbf"
    seed: int = 42
    max_heatmap_n: int = 400
    n_seeds: int = 3


def make_dataset(cfg: AblationConfig) -> Tuple[np.ndarray, np.ndarray | None]:
    set_seed(cfg.seed)
    if cfg.dataset == "moons":
        X, y = make_moons(n_samples=cfg.n, noise=cfg.noise, random_state=cfg.seed)
        return X.astype(np.float64), y.astype(int)
    if cfg.dataset == "circles":
        X, y = make_circles(n_samples=cfg.n, noise=cfg.noise, factor=0.5, random_state=cfg.seed)
        return X.astype(np.float64), y.astype(int)
    if cfg.dataset == "swiss":
        X, tvals = make_swiss_roll(n_samples=cfg.n, noise=cfg.noise, random_state=cfg.seed)
        return X.astype(np.float64), tvals.astype(np.float64)
    raise ValueError("dataset must be one of: moons, circles, swiss")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def run_ablation_and_visualization(cfg: AblationConfig, outdir: str = "results") -> Dict:
    ensure_dir(outdir)
    X, y_or_t = make_dataset(cfg)
    pkg = graphs_from_ablation(
        X=X,
        k=cfg.k,
        alpha=cfg.alpha,
        t=cfg.t,
        heat_rank=cfg.heat_rank,
        kernel=cfg.kernel,
    )

    H = pkg["H_dense"]
    K_heat = pkg["K_heat_dense"]
    A_H = pkg["A_H"]
    A_heat = pkg["A_heat"]
    A_zel = pkg["A_zelpha"]

    metrics: Dict[str, float] = {}
    if cfg.dataset in {"moons", "circles"}:
        labels = y_or_t.astype(int)
        metrics = {
            "neighbor_purity_H": neighbor_purity(A_H, labels, k=cfg.k),
            "neighbor_purity_heat": neighbor_purity(A_heat, labels, k=cfg.k),
            "neighbor_purity_zelpha": neighbor_purity(A_zel, labels, k=cfg.k),
        }

    idx_vis = np.arange(len(X))
    if len(X) > cfg.max_heatmap_n:
        idx_vis = np.random.RandomState(cfg.seed).choice(len(X), size=cfg.max_heatmap_n, replace=False)
        idx_vis.sort()

    H_vis = H[np.ix_(idx_vis, idx_vis)]
    Kh_vis = K_heat[np.ix_(idx_vis, idx_vis)]
    AH_vis = to_dense(A_H)[np.ix_(idx_vis, idx_vis)]
    Aheat_vis = to_dense(A_heat)[np.ix_(idx_vis, idx_vis)]
    Azel_vis = to_dense(A_zel)[np.ix_(idx_vis, idx_vis)]

    plot_adj_matrices(
        Ws=[H_vis, Kh_vis, AH_vis, Aheat_vis, Azel_vis],
        titles=["H (dense)", "Heat K (dense)", "A_H (top-k)", "A_heat (top-k)", "A_zelpha (top-k)"],
        out_path=os.path.join(outdir, f"{cfg.dataset}_adjacency_panels.png"),
    )
    plot_adj_differences(Azel_vis, AH_vis, "A_zelpha", "A_H", os.path.join(outdir, f"{cfg.dataset}_adj_diff_zelpha_minus_H.png"))
    plot_adj_differences(Azel_vis, Aheat_vis, "A_zelpha", "A_heat", os.path.join(outdir, f"{cfg.dataset}_adj_diff_zelpha_minus_heat.png"))
    plot_adj_differences(AH_vis, Aheat_vis, "A_H", "A_heat", os.path.join(outdir, f"{cfg.dataset}_adj_diff_H_minus_heat.png"))

    seeds = np.random.RandomState(cfg.seed).choice(len(X), size=min(cfg.n_seeds, len(X)), replace=False).tolist()
    if cfg.dataset in {"moons", "circles"}:
        X2d = X
    else:
        X2d = X[:, :2]

    S_H = heat_diffuse_from_seeds(A_H, seeds=seeds, t=cfg.t, rank=cfg.heat_rank)
    S_heat = heat_diffuse_from_seeds(A_heat, seeds=seeds, t=cfg.t, rank=cfg.heat_rank)
    S_zel = heat_diffuse_from_seeds(A_zel, seeds=seeds, t=cfg.t, rank=cfg.heat_rank)

    scatter_diffusion(X2d, S_H, seeds, f"Diffusion extent: A_H ({cfg.dataset})", os.path.join(outdir, f"{cfg.dataset}_diffusion_AH.png"))
    scatter_diffusion(X2d, S_heat, seeds, f"Diffusion extent: A_heat ({cfg.dataset})", os.path.join(outdir, f"{cfg.dataset}_diffusion_Aheat.png"))
    scatter_diffusion(X2d, S_zel, seeds, f"Diffusion extent: A_zelpha ({cfg.dataset})", os.path.join(outdir, f"{cfg.dataset}_diffusion_Azelpha.png"))

    payload = {
        "config": asdict(cfg),
        "seeds": seeds,
        "metrics": metrics,
        "outputs": {
            "adjacency_panels": f"{cfg.dataset}_adjacency_panels.png",
            "adj_diff_zelpha_minus_H": f"{cfg.dataset}_adj_diff_zelpha_minus_H.png",
            "adj_diff_zelpha_minus_heat": f"{cfg.dataset}_adj_diff_zelpha_minus_heat.png",
            "adj_diff_H_minus_heat": f"{cfg.dataset}_adj_diff_H_minus_heat.png",
            "diffusion_AH": f"{cfg.dataset}_diffusion_AH.png",
            "diffusion_Aheat": f"{cfg.dataset}_diffusion_Aheat.png",
            "diffusion_Azelpha": f"{cfg.dataset}_diffusion_Azelpha.png",
        },
    }
    with open(os.path.join(outdir, f"{cfg.dataset}_ablation_vis_summary.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return payload


def main() -> None:  # pragma: no cover - CLI wrapper
    import argparse

    parser = argparse.ArgumentParser(description="Ablation + Visualization for Zelpha components on toy datasets")
    parser.add_argument("--dataset", type=str, default="moons", choices=["moons", "circles", "swiss"], help="Dataset")
    parser.add_argument("--n", type=int, default=800, help="Number of samples")
    parser.add_argument("--noise", type=float, default=0.1, help="Noise level")
    parser.add_argument("--k", type=int, default=10, help="k for kNN sparsification")
    parser.add_argument("--alpha", type=float, default=0.5, help="Zelpha alpha (weight for H)")
    parser.add_argument("--t", type=float, default=1.0, help="Heat kernel time scale")
    parser.add_argument("--heat-rank", type=int, default=128, help="Eigen rank for heat kernel approx")
    parser.add_argument("--kernel", type=str, default="rbf", choices=["rbf", "linear"], help="Feature kernel for H")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-heatmap-n", type=int, default=400, help="Subsample size for adjacency heatmaps")
    parser.add_argument("--n-seeds", type=int, default=3, help="# seed nodes for diffusion visualization")
    parser.add_argument("--outdir", type=str, default="results", help="Output directory")

    args = parser.parse_args()

    cfg = AblationConfig(
        dataset=args.dataset,
        n=args.n,
        noise=args.noise,
        k=args.k,
        alpha=args.alpha,
        t=args.t,
        heat_rank=args.heat_rank,
        kernel=args.kernel,
        seed=args.seed,
        max_heatmap_n=args.max_heatmap_n,
        n_seeds=args.n_seeds,
    )

    run_ablation_and_visualization(cfg, outdir=args.outdir)


if __name__ == "__main__":  # pragma: no cover - CLI
    main()
