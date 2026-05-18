"""Core package for Game Theory Project experiments."""

from .graph_generator import GraphGenerator
from .hst_bound import compute_beta_hst_max
from .models import RecurrentGATAgent, SharedSequentialLoss
from .reporting import classify_regimes
from .signal_generator import PrivateSignalGenerator
from .training_pipeline import (
    ConditionRunResult,
    compute_epsilon_series,
    condition_result_to_dict,
    fit_beta_from_epsilon,
    run_condition_experiment,
    train_condition_model,
)

__all__ = [
    "GraphGenerator",
    "compute_beta_hst_max",
    "classify_regimes",
    "PrivateSignalGenerator",
    "RecurrentGATAgent",
    "SharedSequentialLoss",
    "ConditionRunResult",
    "train_condition_model",
    "compute_epsilon_series",
    "fit_beta_from_epsilon",
    "run_condition_experiment",
    "condition_result_to_dict",
]
