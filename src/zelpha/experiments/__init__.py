"""Experiment entrypoints for the Zelpha toy benchmarks."""

from .ablations import AblationConfig, run_ablation_and_visualization
from .moons_circles import ExperimentConfig as MoonsCirclesConfig, run_experiment as run_moons_circles
from .swiss_roll import ExperimentConfig as SwissRollConfig, run_experiment as run_swiss_roll
from .uci import UCIRunConfig, run_uci

__all__ = [
    "AblationConfig",
    "run_ablation_and_visualization",
    "MoonsCirclesConfig",
    "run_moons_circles",
    "SwissRollConfig",
    "run_swiss_roll",
    "UCIRunConfig",
    "run_uci",
]
