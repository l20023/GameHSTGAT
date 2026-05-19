import math

import pytest
import torch

from src.graph_generator import GraphGenerator
from src.training_pipeline import (
    condition_result_to_dict,
    fit_beta_from_epsilon,
    run_condition_experiment,
)


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
    assert isinstance(fit["rmse"], float)
    assert isinstance(fit["r2"], float)
    assert fit["failure_reason"] == ""


def test_condition_result_to_dict_omits_heavy_series_by_default() -> None:
    graph = GraphGenerator().generate_complete(8)
    result = run_condition_experiment(
        graph_data=graph,
        train_episodes=2,
        test_episodes=2,
        max_horizon=3,
        signal_quality=0.8,
        hidden_dim=8,
        num_heads=2,
        learning_rate=0.001,
        seed=5,
        disable_beta_fit=True,
    )
    payload = condition_result_to_dict(result)
    assert "train_loss_history" not in payload
    assert "epsilon_series" not in payload
    assert payload["train_loss_final"] is not None


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
    assert "rmse" in result.beta_fit
    assert "r2" in result.beta_fit
    assert "failure_reason" in result.beta_fit
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
    assert result.beta_fit["failure_reason"] == "disabled"
    assert torch.isfinite(torch.tensor(result.epsilon_series)).all()
    assert result.beta_hst_max > 0.0
    assert result.beta_gap is None
    assert result.exceeds_hst_bound is None


def test_run_condition_experiment_vector_communication_mode() -> None:
    graph = GraphGenerator().generate_complete(10)
    result = run_condition_experiment(
        graph_data=graph,
        train_episodes=2,
        test_episodes=2,
        max_horizon=4,
        signal_quality=0.8,
        hidden_dim=16,
        num_heads=2,
        communication_mode="vector",
        communication_dim=4,
        learning_rate=0.001,
        seed=17,
        disable_beta_fit=True,
    )
    assert len(result.epsilon_series) == 4
    assert result.beta_fit["method"] == "disabled"


@pytest.mark.parametrize("series", [[0.1, 0.2], [float("nan"), 0.2, 0.3]])
def test_fit_beta_handles_invalid_input(series: list[float]) -> None:
    fit = fit_beta_from_epsilon(series)
    assert fit["fit_success"] is False or fit["method"] == "log_linear_fallback"


def test_fit_beta_flags_poor_fit_when_r2_is_very_bad() -> None:
    noisy_series = [0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4]
    fit = fit_beta_from_epsilon(noisy_series)
    if fit["fit_success"] is False:
        assert fit["failure_reason"] in {
            "poor_fit_r2",
            "non_finite_params",
            "negative_beta",
            "epsilon_inf_out_of_range",
        }
