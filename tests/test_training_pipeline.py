import math

import pytest
import torch

from src.graph_generator import GraphGenerator
from src.training_pipeline import fit_beta_from_epsilon, run_condition_experiment


def test_fit_beta_from_synthetic_exponential_series() -> None:
    alpha = 0.4
    beta = 0.35
    epsilon_inf = 0.05
    epsilon_series = [
        alpha * math.exp(-beta * t) + epsilon_inf for t in range(1, 21)
    ]
    fit = fit_beta_from_epsilon(epsilon_series)
    assert fit["fit_success"] is True
    assert isinstance(fit["beta"], float)
    assert fit["beta"] > 0.0


def test_run_condition_experiment_outputs_expected_shapes() -> None:
    graph = GraphGenerator().generate_complete(10)
    result = run_condition_experiment(
        graph_data=graph,
        train_episodes=3,
        test_episodes=4,
        max_horizon=5,
        signal_quality=0.8,
        hidden_dim=16,
        num_heads=2,
        learning_rate=0.001,
        seed=11,
        disable_beta_fit=False,
    )

    assert len(result.train_loss_history) == 3
    assert len(result.epsilon_series) == 5
    assert all(0.0 <= value <= 1.0 for value in result.epsilon_series)
    assert "beta" in result.beta_fit
    assert result.beta_hst_max > 0.0
    if result.beta_fit["fit_success"] is True:
        assert isinstance(result.beta_gap, float)
        assert isinstance(result.exceeds_hst_bound, bool)


def test_run_condition_experiment_disable_beta_fit() -> None:
    graph = GraphGenerator().generate_watts_strogatz(10, k=2, p=0.1, seed=3)
    result = run_condition_experiment(
        graph_data=graph,
        train_episodes=2,
        test_episodes=2,
        max_horizon=4,
        signal_quality=0.8,
        hidden_dim=16,
        num_heads=2,
        learning_rate=0.001,
        seed=2,
        disable_beta_fit=True,
    )

    assert result.beta_fit["fit_success"] is False
    assert result.beta_fit["method"] == "disabled"
    assert torch.isfinite(torch.tensor(result.epsilon_series)).all()
    assert result.beta_hst_max > 0.0
    assert result.beta_gap is None
    assert result.exceeds_hst_bound is None


@pytest.mark.parametrize("series", [[0.1, 0.2], [float("nan"), 0.2, 0.3]])
def test_fit_beta_handles_invalid_input(series: list[float]) -> None:
    fit = fit_beta_from_epsilon(series)
    assert fit["fit_success"] is False or fit["method"] == "log_linear_fallback"
