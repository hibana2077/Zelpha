"""Two moons / circles classification experiments with Zelpha graphs."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.datasets import make_circles, make_moons

from ..training import (
    GraphConfig,
    TrainConfig,
    build_graphs_for_point_cloud,
    train_and_eval,
    train_and_eval_logreg,
    train_and_eval_svm,
)


def make_dataset(name: str, n: int, noise: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if name == "moons":
        X, y = make_moons(n_samples=n, noise=noise, random_state=seed)
    elif name == "circles":
        X, y = make_circles(n_samples=n, noise=noise, factor=0.5, random_state=seed)
    else:
        raise ValueError("dataset must be 'moons' or 'circles'")
    return X.astype(np.float64), y.astype(int)


@dataclass
class ExperimentConfig:
    dataset: str = "moons"
    n_samples: int = 1000
    noise_levels: Tuple[float, ...] = (0.0, 0.1, 0.2, 0.3)
    graph: GraphConfig = field(default_factory=GraphConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def run_experiment(cfg: ExperimentConfig, outdir: str = "results", use_cuda: bool | None = None) -> Dict:
    device = torch.device("cuda" if (use_cuda if use_cuda is not None else torch.cuda.is_available()) else "cpu")
    results: Dict[str, Dict[float, Dict[str, float]]] = {"logreg": {}, "svm": {}}

    for noise in cfg.noise_levels:
        X, y = make_dataset(cfg.dataset, cfg.n_samples, noise=noise, seed=cfg.train.random_state)

        lr_metrics = train_and_eval_logreg(X, y, cfg.train, seed=cfg.train.random_state)
        results["logreg"][noise] = lr_metrics
        print(f"[{cfg.dataset} | noise={noise:.2f} | logreg] test_acc={lr_metrics['test_acc']:.4f} test_macro_f1={lr_metrics['test_macro_f1']:.4f}")

        svm_metrics = train_and_eval_svm(X, y, cfg.train, seed=cfg.train.random_state)
        results["svm"][noise] = svm_metrics
        print(f"[{cfg.dataset} | noise={noise:.2f} | svm]    test_acc={svm_metrics['test_acc']:.4f} test_macro_f1={svm_metrics['test_macro_f1']:.4f}")

        graphs = build_graphs_for_point_cloud(X, cfg.graph)
        for name, A in graphs.items():
            if name not in results:
                results[name] = {}
            metrics = train_and_eval(X, y, A, cfg.train, seed=cfg.train.random_state, device=device)
            results[name][noise] = metrics
            print(f"[{cfg.dataset} | noise={noise:.2f} | {name}] test_acc={metrics['test_acc']:.4f} test_macro_f1={metrics['test_macro_f1']:.4f}")

    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, f"{cfg.dataset}_two_moons_circles_metrics.json")
    payload = {
        "config": asdict(cfg),
        "results": results,
        "notes": {
            "model": "Two-layer GCN (PyTorch) with sparse adjacency, early stopping on val acc",
            "hidden_dim": cfg.train.hidden_dim,
            "lr": cfg.train.lr,
            "weight_decay": cfg.train.weight_decay,
            "dropout": cfg.train.dropout,
            "epochs": cfg.train.epochs,
            "patience": cfg.train.patience,
            "graph": "cosine, rbf, snn_jaccard kNN vs Zelpha",
            "baseline": "Logistic Regression (StandardScaler + LogisticRegression) and SVM (StandardScaler + SVC[RBF]) on raw features",
        },
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved results to {out_path}")
    return payload


def main() -> None:  # pragma: no cover - CLI wrapper
    import argparse

    parser = argparse.ArgumentParser(description="Two moons / circles classification with cosine vs Zelpha graphs (PyTorch GCN)")
    parser.add_argument("--dataset", type=str, default="moons", choices=["moons", "circles"])
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--noise-levels", type=float, nargs="*", default=[0.0, 0.1, 0.2, 0.3])
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--t", type=float, default=1.0)
    parser.add_argument("--heat-rank", type=int, default=128)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    parser.add_argument("--outdir", type=str, default="results")
    args = parser.parse_args()

    cfg = ExperimentConfig(
        dataset=args.dataset,
        n_samples=args.n,
        noise_levels=tuple(args.noise_levels),
        graph=GraphConfig(k=args.k, alpha=args.alpha, t=args.t, heat_rank=args.heat_rank, kernel="rbf"),
        train=TrainConfig(
            test_size=args.test_size,
            val_size=args.val_size,
            random_state=args.seed,
            hidden_dim=args.hidden_dim,
            lr=args.lr,
            weight_decay=args.weight_decay,
            dropout=args.dropout,
            epochs=args.epochs,
            patience=args.patience,
        ),
    )

    run_experiment(cfg, outdir=args.outdir, use_cuda=(not args.cpu))


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
