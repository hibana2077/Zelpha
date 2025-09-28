"""Swiss roll benchmark comparing cosine and Zelpha graphs."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Dict

from sklearn.datasets import make_swiss_roll

from ..graphs import (
    build_cosine_knn_graph,
    cut_energy,
    geodesic_distortion,
    labels_from_swiss_t,
    neighbor_purity,
    set_seed,
    zelpha_graph,
)


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
    yvals = X[:, 1]

    labels = labels_from_swiss_t(tvals, n_bins=cfg.n_bins)

    A_cos = build_cosine_knn_graph(X, k=cfg.k)
    purity_cos = neighbor_purity(A_cos, labels, k=cfg.k)
    cut_cos = cut_energy(A_cos, labels)
    dist_cos = geodesic_distortion(A_cos, tvals, yvals, pairs=cfg.pairs, length_transform=length_transform)

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


def main() -> None:  # pragma: no cover - CLI wrapper
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

    results = run_experiment(cfg, length_transform=args.length_transform)

    print("Config:")
    print(cfg)
    print("\nResults:")
    for k, v in results.items():
        print(f"{k}: {v:.6f}")

    os.makedirs(args.outdir, exist_ok=True)
    out_json = os.path.join(args.outdir, "swiss_roll_metrics.json")
    payload = {"config": asdict(cfg), "length_transform": args.length_transform, "results": results}
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved metrics to {out_json}")


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
