# Zelpha Toy Experiments

Reusable toy experiments and utilities that showcase the Zelpha graph construction recipe. The code is organised as a small Python package under `src/` so the whole repo can be cloned and used directly without publishing to PyPI.

## Repository layout

```text
├── docs/                      # research notes and reports (unchanged)
├── src/
│   └── zelpha/
│       ├── __init__.py
│       ├── graphs.py          # graph construction + quality metrics
│       ├── training.py        # GCN + classical baselines and shared configs
│       └── experiments/
│           ├── ablations.py   # adjacency heatmaps + diffusion visuals
│           ├── moons_circles.py
│           ├── swiss_roll.py
│           └── uci.py
├── requirements.txt
└── toy-exp/                   # legacy scripts kept for reference (now wrappers)
```

All entrypoints live inside `zelpha.experiments`. You can run them straight from the repository root once dependencies are installed.

## Graph builders

Available constructors in `zelpha.graphs`:

- `build_cosine_knn_graph`: cosine similarity k-NN (symmetric)
- `build_rbf_knn_graph`: RBF/Gaussian k-NN affinity; auto-estimates gamma if not provided
- `build_snn_graph`: Shared Nearest Neighbors with `sim="jaccard"` (default) or `sim="count"`
- `zelpha_graph`: fusion of RKHS-cosine and graph heat kernel (geometric or convex)

### Python API

```python
from zelpha.graphs import build_cosine_knn_graph, build_rbf_knn_graph, build_snn_graph, zelpha_graph

A_cos = build_cosine_knn_graph(X, k=10)
A_rbf = build_rbf_knn_graph(X, k=10)  # or pass gamma=...
A_snn = build_snn_graph(X, k=10, sim="jaccard")  # or sim="count"
A_zel = zelpha_graph(X, k=10, alpha=0.5, t=1.0, kernel="rbf", heat_rank=128)
```

## Quick start

1. **Create an environment and install dependencies**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

1. **Make the `src/` package visible** (set `PYTHONPATH` once per session):

```powershell
$env:PYTHONPATH = "src"
```

1. **Run an experiment** – invoke the module with `python -m` and override flags as needed:

```powershell
# Adjacency ablations + visualisations
python -m zelpha.experiments.ablations --dataset moons --outdir results/moons

# Swiss roll graph quality metrics
python -m zelpha.experiments.swiss_roll --n 3000 --k 12

# Two moons / circles classification (GCN + baselines)
python -m zelpha.experiments.moons_circles --dataset circles --noise-levels 0.1 0.2 0.3

# UCI sweeps (requires `toy-exp/ionosphere.csv` in place if you want that dataset)
python -m zelpha.experiments.uci --datasets digits wine
```

Outputs are written to the provided `--outdir` (default: `results/`).

## Notes

- The original `toy-exp/*.py` scripts are left in place but will eventually be removed. Each now simply imports the packaged entrypoints so older references continue to work.
- `torch-geometric` has its own platform-specific wheels. Follow the [official install guide](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) if `pip install` complains.
- To run the UCI benchmark with the Ionosphere dataset, download `ionosphere.csv` from the UCI repository and place it inside `src/zelpha/experiments/` (or update the path inside `uci.py`).

Happy graph building!
