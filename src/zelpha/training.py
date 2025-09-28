"""Training helpers shared across Zelpha experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix, issparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .graphs import build_cosine_knn_graph, zelpha_graph


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    hidden_dim: int = 32
    lr: float = 0.01
    weight_decay: float = 5e-4
    dropout: float = 0.5
    epochs: int = 1000
    patience: int = 100


class PyGGCN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.5) -> None:
        super().__init__()
        try:
            from torch_geometric.nn import GCNConv
        except ImportError as exc:  # pragma: no cover - handled at runtime
            raise ImportError("torch-geometric is required for PyGGCN") from exc

        self.conv1 = GCNConv(in_dim, hidden_dim, add_self_loops=True, normalize=True)
        self.conv2 = GCNConv(hidden_dim, out_dim, add_self_loops=True, normalize=True)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.conv1(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.dropout(h)
        return self.conv2(h, edge_index, edge_weight)


def csr_to_edge_index(A: csr_matrix, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    if not issparse(A):
        A = csr_matrix(A)
    A = A.tocoo()
    edge_index = torch.tensor(np.vstack([A.row, A.col]), dtype=torch.long, device=device)
    edge_weight = torch.tensor(A.data, dtype=torch.float32, device=device)
    return edge_index, edge_weight


def build_graphs_for_point_cloud(X: np.ndarray, cfg: GraphConfig) -> Dict[str, csr_matrix]:
    return {
        "cosine": build_cosine_knn_graph(X, k=cfg.k),
        "zelpha": zelpha_graph(
            X,
            k=cfg.k,
            alpha=cfg.alpha,
            t=cfg.t,
            kernel=cfg.kernel,
            heat_rank=cfg.heat_rank,
        ),
    }


def train_and_eval(
    X: np.ndarray,
    y: np.ndarray,
    A: csr_matrix,
    tcfg: TrainConfig,
    seed: int,
    device: torch.device,
) -> Dict[str, float]:
    seed_everything(seed)
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

    edge_index, edge_weight = csr_to_edge_index(A, device)
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
    epochs_no_improve = 0

    for _ in range(tcfg.epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(X_t, edge_index, edge_weight)
        loss = F.cross_entropy(logits[train_mask], y_t[train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(X_t, edge_index, edge_weight)
            pred_val = logits[val_mask].argmax(dim=1)
            acc_val = (pred_val == y_t[val_mask]).float().mean().item()
        model.train()

        if acc_val > best_val_acc:
            best_val_acc = acc_val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= tcfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        logits = model(X_t, edge_index, edge_weight)
        y_pred_val = logits[val_mask].argmax(dim=1).detach().cpu().numpy()
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


def _baseline_split(
    X: np.ndarray,
    y: np.ndarray,
    tcfg: TrainConfig,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    seed_everything(seed)
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
    return (
        X[idx_train],
        y[idx_train],
        X[idx_val],
        y[idx_val],
        X[idx_test],
        y[idx_test],
    )


def train_and_eval_logreg(
    X: np.ndarray,
    y: np.ndarray,
    tcfg: TrainConfig,
    seed: int,
) -> Dict[str, float]:
    X_tr, y_tr, X_val, y_val, X_te, y_te = _baseline_split(X, y, tcfg, seed)
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=tcfg.random_state),
    )
    clf.fit(X_tr, y_tr)
    y_pred_val = clf.predict(X_val)
    y_pred_test = clf.predict(X_te)
    return {
        "val_acc": float(accuracy_score(y_val, y_pred_val)),
        "val_macro_f1": float(f1_score(y_val, y_pred_val, average="macro")),
        "test_acc": float(accuracy_score(y_te, y_pred_test)),
        "test_macro_f1": float(f1_score(y_te, y_pred_test, average="macro")),
        "best_val_acc": float(accuracy_score(y_val, y_pred_val)),
    }


def train_and_eval_svm(
    X: np.ndarray,
    y: np.ndarray,
    tcfg: TrainConfig,
    seed: int,
) -> Dict[str, float]:
    X_tr, y_tr, X_val, y_val, X_te, y_te = _baseline_split(X, y, tcfg, seed)
    clf = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=1.0, gamma="scale", random_state=tcfg.random_state),
    )
    clf.fit(X_tr, y_tr)
    y_pred_val = clf.predict(X_val)
    y_pred_test = clf.predict(X_te)
    return {
        "val_acc": float(accuracy_score(y_val, y_pred_val)),
        "val_macro_f1": float(f1_score(y_val, y_pred_val, average="macro")),
        "test_acc": float(accuracy_score(y_te, y_pred_test)),
        "test_macro_f1": float(f1_score(y_te, y_pred_test, average="macro")),
        "best_val_acc": float(accuracy_score(y_val, y_pred_val)),
    }


__all__ = [
    "GraphConfig",
    "TrainConfig",
    "PyGGCN",
    "csr_to_edge_index",
    "build_graphs_for_point_cloud",
    "train_and_eval",
    "train_and_eval_logreg",
    "train_and_eval_svm",
]
