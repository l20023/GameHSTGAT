import math
from types import SimpleNamespace

import pytest
import torch

from src.graph_generator import GraphGenerator
from src.training_pipeline import (
    _detect_fit_window,
    condition_result_to_dict,
    fit_beta_from_epsilon,
    resolve_runtime_device,
    run_condition_experiment,
)
import numpy as np


def test_fit_beta_from_synthetic_exponential_series() -> None:
    alpha = 0.45
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
    assert fit["method"] in {"scipy_anchored_t0", "anchored_t0_log_linear_fallback"}
    if fit["method"] == "scipy_anchored_t0":
        assert isinstance(fit["beta_std"], float)
        assert math.isfinite(float(fit["beta_std"]))
        assert isinstance(fit["beta_ci_lower"], float)
        assert isinstance(fit["beta_ci_upper"], float)
        assert float(fit["beta_ci_lower"]) <= float(fit["beta"]) <= float(fit["beta_ci_upper"])


def test_fit_beta_truncates_at_plateau() -> None:
    alpha = 0.4
    beta = 0.5
    epsilon_inf = 0.02
    plateau_length = 50
    decay_part = [alpha * math.exp(-beta * t) + epsilon_inf for t in range(1, 16)]
    plateau_part = [epsilon_inf] * plateau_length
    series = decay_part + plateau_part

    fit = fit_beta_from_epsilon(series)
    assert fit["fit_success"] is True
    assert fit["plateau_detected"] is True
    assert fit["n_full_series"] == len(series)
    fit_window = int(fit["fit_window_t_max"])
    assert fit_window < len(series)
    assert fit_window >= 5


def test_fit_beta_no_plateau_detection_when_decay_continues() -> None:
    alpha = 0.4
    beta = 0.05
    epsilon_inf = 0.0
    series = [alpha * math.exp(-beta * t) + epsilon_inf for t in range(1, 21)]
    fit = fit_beta_from_epsilon(series)
    assert fit["fit_success"] is True
    assert fit["plateau_detected"] is False
    assert int(fit["fit_window_t_max"]) == len(series)


def test_detect_fit_window_resists_single_outlier_below_threshold() -> None:
    """Single sub-threshold outlier with above-threshold neighbours must not trigger cutoff."""
    # Tail is flat at 0.0 (clear plateau), preceded by a clear above-threshold
    # decay region, with one isolated outlier point dipping below threshold.
    y = np.array(
        [0.40, 0.30, 0.25, 0.20, 0.15, 0.01, 0.12, 0.10, 0.08, 0.06,
         0.05, 0.04, 0.03, 0.02, 0.01, 0.00, 0.00, 0.00, 0.00, 0.00],
        dtype=float,
    )
    window = _detect_fit_window(y)
    # The plateau starts around index 14-15. The lone outlier at index 5 must
    # not cause a cutoff anywhere near that point.
    assert window > 10, f"Cutoff triggered prematurely: window={window}"


def test_run_condition_experiment_uses_validation_checkpoint() -> None:
    graph = GraphGenerator().generate_complete(8)
    result = run_condition_experiment(
        graph_data=graph,
        train_episodes=12,
        test_episodes=4,
        max_horizon=4,
        signal_quality=0.8,
        hidden_dim=8,
        num_heads=2,
        learning_rate=0.001,
        seed=3,
        device=torch.device("cpu"),
        disable_beta_fit=True,
        validation_episodes=4,
        validation_eval_every=4,
    )
    # Validation checkpoint must have been evaluated and recorded.
    assert result.best_validation_error is not None
    assert result.best_validation_episode_idx is not None
    assert len(result.validation_history) > 0
    payload = condition_result_to_dict(result)
    assert payload["best_validation_error"] is not None
    assert payload["best_validation_episode_idx"] is not None


def test_run_condition_experiment_skips_validation_when_disabled() -> None:
    graph = GraphGenerator().generate_complete(8)
    result = run_condition_experiment(
        graph_data=graph,
        train_episodes=8,
        test_episodes=4,
        max_horizon=4,
        signal_quality=0.8,
        hidden_dim=8,
        num_heads=2,
        learning_rate=0.001,
        seed=4,
        device=torch.device("cpu"),
        disable_beta_fit=True,
        validation_episodes=0,
    )
    assert result.best_validation_error is None
    assert result.best_validation_episode_idx is None
    assert result.validation_history == []


def test_run_condition_experiment_reports_stability_diagnostics() -> None:
    graph = GraphGenerator().generate_complete(8)
    result = run_condition_experiment(
        graph_data=graph,
        train_episodes=4,
        test_episodes=2,
        max_horizon=3,
        signal_quality=0.8,
        hidden_dim=8,
        num_heads=2,
        learning_rate=0.001,
        seed=7,
        device=torch.device("cpu"),
        disable_beta_fit=True,
    )
    # With train_episodes < running_mean_window the running mean is over
    # whatever points are available, but never None when train_loss_history is
    # populated.
    assert result.train_loss_running_mean_final is not None
    payload = condition_result_to_dict(result)
    assert "train_loss_running_mean_final" in payload
    assert "best_running_mean_loss" in payload
    assert "best_episode_idx" in payload


def test_exceeds_hst_bound_suppressed_under_convergence_warning() -> None:
    """A spurious super-bound beta from a non-converged run must not flag exceedance."""
    graph = GraphGenerator().generate_complete(6)
    # Train very briefly so the model never converges; eps_inf will be high
    # and the fit will likely produce a Bound-artifact beta, but the
    # convergence_warning gate must suppress exceeds_hst_bound.
    result = run_condition_experiment(
        graph_data=graph,
        train_episodes=2,
        test_episodes=2,
        max_horizon=8,
        signal_quality=0.6,
        hidden_dim=8,
        num_heads=2,
        learning_rate=0.001,
        seed=42,
        device=torch.device("cpu"),
        disable_beta_fit=False,
    )
    if result.convergence_warning and result.beta_gap is not None and result.beta_gap > 0:
        assert result.exceeds_hst_bound is False


def test_detect_fit_window_finds_plateau_with_consecutive_run() -> None:
    """Consecutive sub-threshold points should trigger cutoff at the run start."""
    y = np.array(
        [0.40, 0.30, 0.20, 0.12, 0.08, 0.05, 0.03, 0.02, 0.02, 0.02,
         0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02],
        dtype=float,
    )
    window = _detect_fit_window(y)
    assert window < len(y)
    assert window >= 5


def test_fit_beta_disabled_payload_includes_ci_fields() -> None:
    graph = GraphGenerator().generate_complete(6)
    result = run_condition_experiment(
        graph_data=graph,
        train_episodes=1,
        test_episodes=1,
        max_horizon=3,
        signal_quality=0.8,
        hidden_dim=8,
        num_heads=2,
        learning_rate=0.001,
        seed=0,
        device=torch.device("cpu"),
        disable_beta_fit=True,
    )
    assert "beta_std" in result.beta_fit
    assert "beta_ci_lower" in result.beta_fit
    assert "beta_ci_upper" in result.beta_fit


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
        device=torch.device("cpu"),
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
        device=torch.device("cpu"),
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
    assert isinstance(result.convergence_warning, bool)
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
        device=torch.device("cpu"),
        disable_beta_fit=True,
    )

    assert result.beta_fit["fit_success"] is False
    assert result.beta_fit["method"] == "disabled"
    assert result.beta_fit["failure_reason"] == "disabled"
    assert torch.isfinite(torch.tensor(result.epsilon_series)).all()
    assert result.beta_hst_max > 0.0
    assert result.beta_gap is None
    assert result.exceeds_hst_bound is None
    assert result.convergence_warning is False


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
        device=torch.device("cpu"),
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


def test_condition_result_to_dict_includes_convergence_warning() -> None:
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
        seed=6,
        device=torch.device("cpu"),
        disable_beta_fit=True,
    )
    payload = condition_result_to_dict(result)
    assert "convergence_warning" in payload
    assert payload["convergence_warning"] is False


def test_resolve_runtime_device_auto_prefers_cuda(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.backends,
        "mps",
        SimpleNamespace(is_available=lambda: True),
        raising=False,
    )
    device = resolve_runtime_device("auto")
    assert device.type == "cuda"


def test_resolve_runtime_device_auto_uses_mps_when_no_cuda(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        torch.backends,
        "mps",
        SimpleNamespace(is_available=lambda: True),
        raising=False,
    )
    device = resolve_runtime_device("auto")
    assert device.type == "mps"


def test_resolve_runtime_device_auto_falls_back_to_cpu(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        torch.backends,
        "mps",
        SimpleNamespace(is_available=lambda: False),
        raising=False,
    )
    device = resolve_runtime_device("auto")
    assert device.type == "cpu"


def test_resolve_runtime_device_raises_for_unavailable_explicit(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="CUDA is not available"):
        resolve_runtime_device("cuda")
