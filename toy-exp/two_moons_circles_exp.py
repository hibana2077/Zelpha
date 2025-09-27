import json
import os
from dataclasses import dataclass, asdict, field
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix, issparse
from sklearn.datasets import make_moons, make_circles
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

# Reuse graph builders from swiss_roll_experiment.py in the same folder
from swiss_roll_experiment import build_cosine_knn_graph, zelpha_graph, neighbor_purity
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


@dataclass
class GraphConfig:
    k: int = 10
    alpha: float = 0.5
    t: float = 1.0
    heat_rank: int = 128
    kernel: str = "rbf"


@dataclass
class TrainConfig:
    test_size: float = 0.2
    val_size: float = 0.2
    random_state: int = 42
    # GCN hyperparameters
    hidden_dim: int = 32
    lr: float = 0.01
    weight_decay: float = 5e-4
    dropout: float = 0.5
    epochs: int = 1000
    patience: int = 100


@dataclass
class ExperimentConfig:
    dataset: str = "moons"  # or "circles"
    n_samples: int = 1000
    noise_levels: Tuple[float, ...] = (0.0, 0.1, 0.2, 0.3)
    graph: GraphConfig = field(default_factory=GraphConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def csr_to_edge_index(A: csr_matrix, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    if not issparse(A):
        A = csr_matrix(A)
    A = A.tocoo()
    edge_index = torch.tensor(
        np.vstack([A.row, A.col]), dtype=torch.long, device=device
    )
    edge_weight = torch.tensor(A.data, dtype=torch.float32, device=device)
    return edge_index, edge_weight


def build_graphs(X: np.ndarray, cfg: GraphConfig) -> Dict[str, csr_matrix]:
    A_cos = build_cosine_knn_graph(X, k=cfg.k)
    A_zel = zelpha_graph(X, k=cfg.k, alpha=cfg.alpha, t=cfg.t, kernel=cfg.kernel, heat_rank=cfg.heat_rank)
    return {"cosine": A_cos, "zelpha": A_zel}


def make_dataset(name: str, n: int, noise: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if name == "moons":
        X, y = make_moons(n_samples=n, noise=noise, random_state=seed)
    elif name == "circles":
        X, y = make_circles(n_samples=n, noise=noise, factor=0.5, random_state=seed)
    else:
        raise ValueError("dataset must be 'moons' or 'circles'")
    return X.astype(np.float64), y.astype(int)

class PyGGCN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.5):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim, add_self_loops=True, normalize=True)
        self.conv2 = GCNConv(hidden_dim, out_dim, add_self_loops=True, normalize=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor | None = None) -> torch.Tensor:
        h = self.conv1(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.dropout(h)
        out = self.conv2(h, edge_index, edge_weight)
        return out


def train_and_eval(
    X: np.ndarray,
    y: np.ndarray,
    A: csr_matrix,
    tcfg: TrainConfig,
    seed: int,
    device: torch.device,
) -> Dict[str, float]:
    set_seed(seed)
    # Indices for stratified splits
    N = len(X)
    idx_all = np.arange(N)
    idx_trainval, idx_test = train_test_split(idx_all, test_size=tcfg.test_size, random_state=tcfg.random_state, stratify=y)
    rel_val = tcfg.val_size / (1.0 - tcfg.test_size)
    idx_train, idx_val = train_test_split(idx_trainval, test_size=rel_val, random_state=tcfg.random_state, stratify=y[idx_trainval])

    # Convert graph to PyG edge_index/edge_weight; normalization handled by GCNConv
    edge_index, edge_weight = csr_to_edge_index(A, device)

    # Features and labels
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    y_t = torch.tensor(y, dtype=torch.long, device=device)
    train_mask = torch.zeros(N, dtype=torch.bool, device=device)
    val_mask = torch.zeros(N, dtype=torch.bool, device=device)
    test_mask = torch.zeros(N, dtype=torch.bool, device=device)
    train_mask[idx_train] = True
    val_mask[idx_val] = True
    test_mask[idx_test] = True

    model = PyGGCN(in_dim=X.shape[1], hidden_dim=tcfg.hidden_dim, out_dim=int(y.max() + 1), dropout=tcfg.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)

    best_val_acc = -1.0
    best_state = None
    patience = tcfg.patience
    epochs_no_improve = 0

    for epoch in range(tcfg.epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(X_t, edge_index, edge_weight)
        loss = F.cross_entropy(logits[train_mask], y_t[train_mask])
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                logits = model(X_t, edge_index, edge_weight)
                pred_tr = logits[train_mask].argmax(dim=1)
                acc_tr = (pred_tr == y_t[train_mask]).float().mean().item()
                pred_val = logits[val_mask].argmax(dim=1)
                acc_val = (pred_val == y_t[val_mask]).float().mean().item()
            model.train()

        # Eval on val
        model.eval()
        with torch.no_grad():
            logits = model(X_t, edge_index, edge_weight)
            pred_val = logits[val_mask].argmax(dim=1)
            acc_val = (pred_val == y_t[val_mask]).float().mean().item()

        if acc_val > best_val_acc:
            best_val_acc = acc_val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    # Load best
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final metrics on val/test
    model.eval()
    with torch.no_grad():
        logits = model(X_t, edge_index, edge_weight)
        y_pred_val = logits[val_mask].argmax(dim=1).detach().cpu().numpy()
        # IMPORTANT: use the same boolean mask ordering for y_true to match y_pred
        y_true_val = y[val_mask.detach().cpu().numpy()]
        y_pred_test = logits[test_mask].argmax(dim=1).detach().cpu().numpy()
        y_true_test = y[test_mask.detach().cpu().numpy()]

    acc_val = accuracy_score(y_true_val, y_pred_val)
    f1_val = f1_score(y_true_val, y_pred_val, average="macro")
    acc_test = accuracy_score(y_true_test, y_pred_test)
    f1_test = f1_score(y_true_test, y_pred_test, average="macro")

    return {
        "val_acc": float(acc_val),
        "val_macro_f1": float(f1_val),
        "test_acc": float(acc_test),
        "test_macro_f1": float(f1_test),
        "best_val_acc": float(best_val_acc),
    }


def train_and_eval_logreg(
    X: np.ndarray,
    y: np.ndarray,
    tcfg: TrainConfig,
    seed: int,
) -> Dict[str, float]:
    """Train a Logistic Regression baseline on raw features with the same splits.

    Uses StandardScaler + LogisticRegression pipeline. Returns the same metric schema
    as the GCN training function for easy comparison.
    """
    set_seed(seed)

    # Indices for stratified splits (identical params to GCN path for fairness)
    N = len(X)
    idx_all = np.arange(N)
    idx_trainval, idx_test = train_test_split(
        idx_all,
        test_size=tcfg.test_size,
        random_state=tcfg.random_state,
        stratify=y,
    )
    rel_val = tcfg.val_size / (1.0 - tcfg.test_size)
    idx_train, idx_val = train_test_split(
        idx_trainval,
        test_size=rel_val,
        random_state=tcfg.random_state,
        stratify=y[idx_trainval],
    )

    # Model: Standardize then Logistic Regression
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=tcfg.random_state)
    )
    clf.fit(X[idx_train], y[idx_train])

    y_pred_val = clf.predict(X[idx_val])
    y_true_val = y[idx_val]
    y_pred_test = clf.predict(X[idx_test])
    y_true_test = y[idx_test]

    acc_val = accuracy_score(y_true_val, y_pred_val)
    f1_val = f1_score(y_true_val, y_pred_val, average="macro")
    acc_test = accuracy_score(y_true_test, y_pred_test)
    f1_test = f1_score(y_true_test, y_pred_test, average="macro")

    return {
        "val_acc": float(acc_val),
        "val_macro_f1": float(f1_val),
        "test_acc": float(acc_test),
        "test_macro_f1": float(f1_test),
        # No early stopping here; mirror key for compatibility
        "best_val_acc": float(acc_val),
    }


def train_and_eval_svm(
    X: np.ndarray,
    y: np.ndarray,
    tcfg: TrainConfig,
    seed: int,
) -> Dict[str, float]:
    """Train an SVM (RBF kernel) baseline on raw features with the same splits.

    Uses StandardScaler + SVC pipeline. Returns the same metric schema
    as the GCN training function for easy comparison.
    """
    set_seed(seed)

    # Indices for stratified splits (identical params to GCN path for fairness)
    N = len(X)
    idx_all = np.arange(N)
    idx_trainval, idx_test = train_test_split(
        idx_all,
        test_size=tcfg.test_size,
        random_state=tcfg.random_state,
        stratify=y,
    )
    rel_val = tcfg.val_size / (1.0 - tcfg.test_size)
    idx_train, idx_val = train_test_split(
        idx_trainval,
        test_size=rel_val,
        random_state=tcfg.random_state,
        stratify=y[idx_trainval],
    )

    # Model: Standardize then SVC (RBF)
    clf = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=1.0, gamma="scale", random_state=tcfg.random_state)
    )
    clf.fit(X[idx_train], y[idx_train])

    y_pred_val = clf.predict(X[idx_val])
    y_true_val = y[idx_val]
    y_pred_test = clf.predict(X[idx_test])
    y_true_test = y[idx_test]

    acc_val = accuracy_score(y_true_val, y_pred_val)
    f1_val = f1_score(y_true_val, y_pred_val, average="macro")
    acc_test = accuracy_score(y_true_test, y_pred_test)
    f1_test = f1_score(y_true_test, y_pred_test, average="macro")

    return {
        "val_acc": float(acc_val),
        "val_macro_f1": float(f1_val),
        "test_acc": float(acc_test),
        "test_macro_f1": float(f1_test),
        # No early stopping here; mirror key for compatibility
        "best_val_acc": float(acc_val),
    }


def run_experiment(cfg: ExperimentConfig, outdir: str = "results", use_cuda: bool | None = None) -> Dict:
    device = torch.device("cuda" if (use_cuda if use_cuda is not None else torch.cuda.is_available()) else "cpu")
    results = {"cosine": {}, "zelpha": {}, "logreg": {}, "svm": {}}
    for noise in cfg.noise_levels:
        # Dataset
        X, y = make_dataset(cfg.dataset, cfg.n_samples, noise=noise, seed=cfg.train.random_state)
        # Logistic Regression baseline on raw features (shared splits via same seeds/params)
        lr_metrics = train_and_eval_logreg(X, y, cfg.train, seed=cfg.train.random_state)
        results["logreg"][noise] = lr_metrics
        print(f"[{cfg.dataset} | noise={noise:.2f} | logreg] test_acc={lr_metrics['test_acc']:.4f} test_macro_f1={lr_metrics['test_macro_f1']:.4f}")
        # SVM baseline on raw features (shared splits)
        svm_metrics = train_and_eval_svm(X, y, cfg.train, seed=cfg.train.random_state)
        results["svm"][noise] = svm_metrics
        print(f"[{cfg.dataset} | noise={noise:.2f} | svm]    test_acc={svm_metrics['test_acc']:.4f} test_macro_f1={svm_metrics['test_macro_f1']:.4f}")
        # Graphs (built on the noisy features, as specified in the ablation)
        graphs = build_graphs(X, cfg.graph)
        for name, A in graphs.items():
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
            "graph": "cosine-kNN vs Zelpha",
            "baseline": "Logistic Regression (StandardScaler + LogisticRegression) and SVM (StandardScaler + SVC[RBF]) on raw features",
        },
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved results to {out_path}")
    return payload


def main():
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
    # GCN hyperparams
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


if __name__ == "__main__":
    main()
