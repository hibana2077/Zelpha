"""UCI toy benchmark runner for Zelpha graphs."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import ttest_rel
from sklearn.datasets import load_breast_cancer, load_digits, load_wine
from sklearn.preprocessing import StandardScaler

from ..graphs import build_cosine_knn_graph, zelpha_graph
from ..training import TrainConfig, train_and_eval, train_and_eval_logreg, train_and_eval_svm


def load_ionosphere_csv(path: str) -> Tuple[np.ndarray, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", dtype=object)
    if data.ndim == 1:
        data = data[None, :]
    X = data[:, :-1].astype(np.float64)
    y_raw = data[:, -1].astype(str)
    y = np.array([1 if v.strip().lower() == "g" else 0 for v in y_raw], dtype=int)
    return X, y


def load_dataset(name: str, ionosphere_path: str) -> Tuple[np.ndarray, np.ndarray]:
    name = name.lower()
    if name == "digits":
        ds = load_digits()
        X, y = ds.data.astype(np.float64), ds.target.astype(int)
    elif name == "wine":
        ds = load_wine()
        X, y = ds.data.astype(np.float64), ds.target.astype(int)
    elif name == "breast_cancer":
        ds = load_breast_cancer()
        X, y = ds.data.astype(np.float64), ds.target.astype(int)
    elif name == "ionosphere":
        X, y = load_ionosphere_csv(ionosphere_path)
    else:
        raise ValueError(f"Unknown dataset: {name}")
    return X, y


def aggregate_stats(values: List[float]) -> Dict[str, float]:
    arr = np.array(values, dtype=np.float64)
    return {
        "mean": float(np.nanmean(arr)),
        "std": float(np.nanstd(arr, ddof=1) if len(arr) > 1 else 0.0),
    }


@dataclass
class UCIRunConfig:
    datasets: Tuple[str, ...] = ("digits", "wine", "breast_cancer", "ionosphere")
    k: int = 10
    alphas: Tuple[float, ...] = (0.25, 0.5, 0.75)
    times: Tuple[float, ...] = (0.1, 0.5, 1.0, 2.0)
    heat_rank: int = 128
    train: TrainConfig = field(
        default_factory=lambda: TrainConfig(
            test_size=0.2,
            val_size=0.2,
            random_state=42,
            hidden_dim=64,
            lr=0.01,
            weight_decay=5e-4,
            dropout=0.5,
            epochs=400,
            patience=50,
        )
    )
    seeds: Tuple[int, ...] = (42, 43, 44, 45, 46)
    standardize: bool = True
    outdir: str = "results"


def run_uci(cfg: UCIRunConfig, use_cuda: bool | None = None) -> Dict:
    import torch

    device = torch.device("cuda" if (use_cuda if use_cuda is not None else torch.cuda.is_available()) else "cpu")

    results: Dict[str, dict] = {}
    here = os.path.dirname(os.path.abspath(__file__))
    iono_csv = os.path.join(here, "ionosphere.csv")

    for ds_name in cfg.datasets:
        print(f"\n=== Dataset: {ds_name} ===")
        X_all, y_all = load_dataset(ds_name, ionosphere_path=iono_csv)

        scaler = StandardScaler() if cfg.standardize else None
        if scaler is not None:
            X_all = scaler.fit_transform(X_all)

        baseline_logreg: List[Dict[str, float]] = []
        baseline_svm: List[Dict[str, float]] = []
        cosine_acc_per_seed: List[float] = []
        cosine_f1_per_seed: List[float] = []
        sweep_records: Dict[Tuple[float, float], Dict[str, List[float]]] = {
            (a, t): {"acc": [], "f1": []} for a in cfg.alphas for t in cfg.times
        }

        for seed in cfg.seeds:
            tcfg = TrainConfig(
                test_size=cfg.train.test_size,
                val_size=cfg.train.val_size,
                random_state=seed,
                hidden_dim=cfg.train.hidden_dim,
                lr=cfg.train.lr,
                weight_decay=cfg.train.weight_decay,
                dropout=cfg.train.dropout,
                epochs=cfg.train.epochs,
                patience=cfg.train.patience,
            )

            lr_metrics = train_and_eval_logreg(X_all, y_all, tcfg, seed=seed)
            svm_metrics = train_and_eval_svm(X_all, y_all, tcfg, seed=seed)
            baseline_logreg.append(lr_metrics)
            baseline_svm.append(svm_metrics)

            A_cos = build_cosine_knn_graph(X_all, k=cfg.k)
            cos_metrics = train_and_eval(X_all, y_all, A_cos, tcfg, seed=seed, device=device)
            cosine_acc_per_seed.append(float(cos_metrics["test_acc"]))
            cosine_f1_per_seed.append(float(cos_metrics["test_macro_f1"]))

            for alpha in cfg.alphas:
                for t in cfg.times:
                    A_zel = zelpha_graph(
                        X_all,
                        k=cfg.k,
                        alpha=float(alpha),
                        t=float(t),
                        kernel="rbf",
                        heat_rank=cfg.heat_rank,
                    )
                    zel_metrics = train_and_eval(X_all, y_all, A_zel, tcfg, seed=seed, device=device)
                    sweep_records[(alpha, t)]["acc"].append(float(zel_metrics["test_acc"]))
                    sweep_records[(alpha, t)]["f1"].append(float(zel_metrics["test_macro_f1"]))

        ds_result = {
            "config": {
                "k": cfg.k,
                "alphas": cfg.alphas,
                "times": cfg.times,
                "heat_rank": cfg.heat_rank,
                "seeds": cfg.seeds,
                "standardize": cfg.standardize,
                "train": asdict(cfg.train),
            },
            "baselines": {
                "logreg": {
                    "test_acc": aggregate_stats([m["test_acc"] for m in baseline_logreg]),
                    "test_macro_f1": aggregate_stats([m["test_macro_f1"] for m in baseline_logreg]),
                },
                "svm": {
                    "test_acc": aggregate_stats([m["test_acc"] for m in baseline_svm]),
                    "test_macro_f1": aggregate_stats([m["test_macro_f1"] for m in baseline_svm]),
                },
            },
            "cosine": {
                "test_acc": aggregate_stats(cosine_acc_per_seed),
                "test_macro_f1": aggregate_stats(cosine_f1_per_seed),
                "per_seed": {"acc": cosine_acc_per_seed, "f1": cosine_f1_per_seed},
            },
            "zelpha": {},
        }

        for (alpha, t), rec in sweep_records.items():
            acc_vals = rec["acc"]
            f1_vals = rec["f1"]
            try:
                t_acc, p_acc = ttest_rel(acc_vals, cosine_acc_per_seed)
            except Exception:
                t_acc, p_acc = np.nan, np.nan
            try:
                t_f1, p_f1 = ttest_rel(f1_vals, cosine_f1_per_seed)
            except Exception:
                t_f1, p_f1 = np.nan, np.nan

            ds_result["zelpha"][f"alpha={alpha}_t={t}"] = {
                "test_acc": aggregate_stats(acc_vals),
                "test_macro_f1": aggregate_stats(f1_vals),
                "per_seed": {"acc": acc_vals, "f1": f1_vals},
                "ttest_vs_cosine": {
                    "acc": {"t": float(t_acc), "p": float(p_acc) if np.isfinite(p_acc) else np.nan},
                    "f1": {"t": float(t_f1), "p": float(p_f1) if np.isfinite(p_f1) else np.nan},
                },
            }

        results[ds_name] = ds_result

    os.makedirs(cfg.outdir, exist_ok=True)
    out_path = os.path.join(cfg.outdir, "uci_metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"uci_config": asdict(cfg), "results": results}, f, indent=2)
    print(f"\nSaved UCI results to {out_path}")

    return results


def main() -> None:  # pragma: no cover - CLI wrapper
    import argparse

    parser = argparse.ArgumentParser(description="UCI classification with cosine vs Zelpha graphs (PyTorch GCN)")
    parser.add_argument("--datasets", type=str, nargs="*", default=["digits", "wine", "breast_cancer", "ionosphere"], help="Subset of datasets")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--alphas", type=float, nargs="*", default=[0.25, 0.5, 0.75])
    parser.add_argument("--times", type=float, nargs="*", default=[0.1, 0.5, 1.0, 2.0])
    parser.add_argument("--heat-rank", type=int, default=128)
    parser.add_argument("--seeds", type=int, nargs="*", default=[42, 43, 44, 45, 46])
    parser.add_argument("--standardize", action="store_true")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    parser.add_argument("--outdir", type=str, default="results")
    args = parser.parse_args()

    run_cfg = UCIRunConfig(
        datasets=tuple(args.datasets),
        k=args.k,
        alphas=tuple(args.alphas),
        times=tuple(args.times),
        heat_rank=args.heat_rank,
        train=TrainConfig(
            test_size=args.test_size,
            val_size=args.val_size,
            random_state=42,
            hidden_dim=args.hidden_dim,
            lr=args.lr,
            weight_decay=args.weight_decay,
            dropout=args.dropout,
            epochs=args.epochs,
            patience=args.patience,
        ),
        seeds=tuple(args.seeds),
        standardize=bool(args.standardize),
        outdir=args.outdir,
    )

    run_uci(run_cfg, use_cuda=(not args.cpu))


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
