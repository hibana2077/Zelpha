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