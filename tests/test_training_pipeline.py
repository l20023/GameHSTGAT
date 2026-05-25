import math
from types import SimpleNamespace

import pytest
import torch

from src.graph_generator import GraphGenerator
from src.training_pipeline import (
    _consensus_flags_at_timestep,
    _finalize_consensus_mode_metrics,
    _new_consensus_mode_accumulator,
    _update_consensus_mode_accumulator,
    _detect_fit_window,
    anchored_t1_decay_values,
    condition_result_to_dict,
    consensus_to_dict,
    fit_beta_from_epsilon,
    resolve_runtime_device,
    run_condition_experiment,
)
import numpy as np


def test_consensus_flags_unanimous_and_majority() -> None:
    preds = torch.tensor([1, 1, 1, 0])
    flags = _consensus_flags_at_timestep(preds, theta=1, num_nodes=4)
    assert flags["unanimous"] is False
    assert flags["majority"] is True
    assert flags["majority_correct"] is True
    assert flags["agreement_fraction"] == 0.75

    unanimous_preds = torch.tensor([0, 0, 0, 0])
    uni_flags = _consensus_flags_at_timestep(unanimous_preds, theta=1, num_nodes=4)
    assert uni_flags["unanimous"] is True
    assert uni_flags["unanimous_correct"] is False
    assert uni_flags["unanimous_wrong"] is True


def test_consensus_episode_aggregation() -> None:
    max_horizon = 3
    acc = _new_consensus_mode_accumulator(max_horizon)
    flags_episode = [
        _consensus_flags_at_timestep(torch.tensor([0, 0]), theta=0, num_nodes=2),
        _consensus_flags_at_timestep(torch.tensor([1, 1]), theta=0, num_nodes=2),
        _consensus_flags_at_timestep(torch.tensor([0, 0]), theta=0, num_nodes=2),
    ]
    _update_consensus_mode_accumulator(
        acc, max_horizon=max_horizon, flags_per_t=flags_episode, mode="unanimous"
    )
    metrics = _finalize_consensus_mode_metrics(
        acc, test_episodes=1, max_horizon=max_horizon, include_series=True
    )
    assert metrics["fraction_episodes_reach_consensus"] == 1.0
    assert metrics["fraction_episodes_consensus_correct"] == 1.0
    assert metrics["mean_first_consensus_t"] == 1.0
    assert metrics["fraction_correct_at_first_consensus"] == 1.0
    assert len(metrics["consensus_rate_series"]) == 3


def test_consensus_to_dict_omits_series_by_default() -> None:
    consensus = {
        "unanimous": {
            "fraction_episodes_reach_consensus": 0.5,
            "consensus_rate_series": [0.1, 0.2],
        },
        "majority": {"fraction_episodes_reach_consensus": 0.9},
    }
    payload = consensus_to_dict(consensus, save_consensus_series=False)
    assert "consensus_rate_series" not in payload["unanimous"]
    assert payload["majority"]["fraction_episodes_reach_consensus"] == 0.9


def test_fit_beta_from_synthetic_exponential_series() -> None:
    beta = 0.35
    epsilon_inf = 0.05
    epsilon_1 = 0.4
    epsilon_series = [
        (epsilon_1 - epsilon_inf) * math.exp(-beta * (t - 1)) + epsilon_inf
        for t in range(1, 21)
    ]
    fit = fit_beta_from_epsilon(epsilon_series, anchor="t1")
    assert fit["fit_success"] is True
    assert isinstance(fit["beta"], float)
    assert abs(float(fit["beta"]) - beta) < 0.05
    assert isinstance(fit["rmse"], float)
    assert isinstance(fit["r2"], float)
    assert fit["failure_reason"] == ""
    assert fit["method"] in {"scipy_anchored_t1", "anchored_t1_log_linear_fallback"}
    if fit["method"] == "scipy_anchored_t1":
        assert isinstance(fit["beta_std"], float)
        assert math.isfinite(float(fit["beta_std"]))
        assert isinstance(fit["beta_ci_lower"], float)
        assert isinstance(fit["beta_ci_upper"], float)
        assert float(fit["beta_ci_lower"]) <= float(fit["beta"]) <= float(fit["beta_ci_upper"])


def test_fit_beta_truncates_at_perfect_error_suffix() -> None:
    beta = 0.5
    epsilon_inf = 0.02
    epsilon_1 = 0.38
    perfect_length = 50
    decay_part = [
        (epsilon_1 - epsilon_inf) * math.exp(-beta * (t - 1)) + epsilon_inf
        for t in range(1, 16)
    ]
    perfect_part = [0.0] * perfect_length
    series = decay_part + perfect_part

    fit = fit_beta_from_epsilon(series, anchor="t1")
    assert fit["fit_success"] is True
    assert fit["plateau_detected"] is True
    assert fit["n_full_series"] == len(series)
    fit_window = int(fit["fit_window_t_max"])
    assert fit_window < len(series)
    assert fit_window >= 5
    assert fit_window == len(decay_part)


def test_fit_beta_default_anchor_is_t0() -> None:
    from src.training_pipeline import PRIOR_EPSILON_AT_T0

    beta = 0.35
    epsilon_inf = 0.05
    series = [
        (PRIOR_EPSILON_AT_T0 - epsilon_inf) * math.exp(-beta * t) + epsilon_inf
        for t in range(1, 21)
    ]
    fit = fit_beta_from_epsilon(series)
    assert fit["fit_anchor"] == "t0"
    assert fit["method"] in {"scipy_anchored_t0", "anchored_t0_log_linear_fallback"}


def test_fit_beta_anchor_t0_uses_prior_at_round_zero() -> None:
    from src.training_pipeline import PRIOR_EPSILON_AT_T0, anchored_t0_decay_values

    beta = 0.35
    epsilon_inf = 0.05
    series = [
        (PRIOR_EPSILON_AT_T0 - epsilon_inf) * math.exp(-beta * t) + epsilon_inf
        for t in range(1, 21)
    ]
    fit = fit_beta_from_epsilon(series, anchor="t0")
    assert fit["fit_success"] is True
    assert fit["fit_anchor"] == "t0"
    assert fit["method"] in {"scipy_anchored_t0", "anchored_t0_log_linear_fallback"}
    t1_pred = anchored_t0_decay_values(
        np.array([1.0]),
        beta=float(fit["beta"]),
        epsilon_inf=float(fit["epsilon_inf"]),
    )[0]
    expected_t1 = (PRIOR_EPSILON_AT_T0 - epsilon_inf) * math.exp(-beta) + epsilon_inf
    assert abs(float(t1_pred) - expected_t1) < 1e-9


def test_fit_beta_matches_empirical_at_t1() -> None:
    epsilon_1 = 0.33
    beta = 0.2
    epsilon_inf = 0.04
    series = [
        (epsilon_1 - epsilon_inf) * math.exp(-beta * (t - 1)) + epsilon_inf
        for t in range(1, 11)
    ]
    fit = fit_beta_from_epsilon(series, anchor="t1")
    assert fit["fit_success"] is True
    assert fit.get("fit_anchor", "t1") == "t1"
    t1_pred = anchored_t1_decay_values(
        np.array([1.0]),
        beta=float(fit["beta"]),
        epsilon_inf=float(fit["epsilon_inf"]),
        epsilon_1=epsilon_1,
    )[0]
    assert abs(float(t1_pred) - epsilon_1) < 1e-9


def test_fit_beta_no_plateau_detection_when_decay_continues() -> None:
    epsilon_1 = 0.4
    beta = 0.05
    epsilon_inf = 0.0
    series = [
        (epsilon_1 - epsilon_inf) * math.exp(-beta * (t - 1)) + epsilon_inf
        for t in range(1, 21)
    ]
    fit = fit_beta_from_epsilon(series, anchor="t1")
    assert fit["fit_success"] is True
    assert fit["plateau_detected"] is False
    assert int(fit["fit_window_t_max"]) == len(series)


def test_detect_fit_window_resists_single_outlier_before_perfect_tail() -> None:
    """Only a perfect-error suffix is excluded; mid-series outliers are ignored."""
    y = np.array(
        [0.40, 0.30, 0.25, 0.20, 0.15, 0.01, 0.12, 0.10, 0.08, 0.06,
         0.05, 0.04, 0.03, 0.02, 0.01, 0.00, 0.00, 0.00, 0.00, 0.00],
        dtype=float,
    )
    window = _detect_fit_window(y)
    assert window == 15, f"Expected cut before perfect tail, got window={window}"


def test_detect_fit_window_no_cut_on_nonzero_plateau() -> None:
    """Low but non-zero error plateaus must not be auto-truncated."""
    y = np.array(
        [0.40, 0.30, 0.20, 0.12, 0.08, 0.05, 0.03, 0.02, 0.02, 0.02,
         0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02],
        dtype=float,
    )
    assert _detect_fit_window(y) == len(y)


def test_detect_fit_window_truncates_perfect_suffix() -> None:
    y = np.array([0.4, 0.35, 0.3, 0.25, 0.2, 0.0, 0.0, 0.0], dtype=float)
    assert _detect_fit_window(y) == 5


def test_fit_beta_respects_manual_fit_window() -> None:
    series = [0.4, 0.25, 0.15, 0.1, 0.08, 0.07, 0.06]
    fit = fit_beta_from_epsilon(series, fit_window_t_max=3)
    assert int(fit["fit_window_t_max"]) == 3
    assert int(fit["n_full_series"]) == len(series)
    assert fit["plateau_detected"] is True


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


def test_detect_fit_window_no_cut_on_slow_tail() -> None:
    """Slow exponential decay with negative tail slope must keep the full series."""
    alpha = 0.4
    beta = 0.05
    series = np.array(
        [alpha * math.exp(-beta * t) for t in range(1, 21)],
        dtype=float,
    )
    window = _detect_fit_window(series)
    assert window == len(series)


def test_detect_fit_window_smoke_like_series() -> None:
    """Monotonic decay through t=20 must not cut early (smoke-run false positive)."""
    y = np.array(
        [
            0.40, 0.36, 0.33, 0.30, 0.27, 0.24, 0.21, 0.19, 0.17, 0.15,
            0.13, 0.12, 0.10, 0.09, 0.08, 0.07, 0.065, 0.060, 0.055, 0.050,
        ],
        dtype=float,
    )
    window = _detect_fit_window(y)
    assert window == len(y), f"Premature cutoff at window={window}"


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


def test_run_condition_experiment_includes_consensus_metrics() -> None:
    graph = GraphGenerator().generate_complete(10)
    result = run_condition_experiment(
        graph_data=graph,
        train_episodes=2,
        test_episodes=3,
        max_horizon=4,
        signal_quality=0.8,
        hidden_dim=16,
        num_heads=2,
        learning_rate=0.001,
        seed=19,
        device=torch.device("cpu"),
        disable_beta_fit=True,
    )
    for mode in ("unanimous", "majority"):
        mode_metrics = result.consensus[mode]
        assert 0.0 <= mode_metrics["fraction_episodes_reach_consensus"] <= 1.0
        assert 0.0 <= mode_metrics["fraction_episodes_consensus_correct"] <= 1.0
        assert "consensus_rate_series" not in mode_metrics

    payload = condition_result_to_dict(result)
    assert "consensus" in payload
    assert "unanimous" in payload["consensus"]


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


def test_condition_result_to_dict_includes_consensus_series_when_requested() -> None:
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
        seed=21,
        device=torch.device("cpu"),
        disable_beta_fit=True,
        include_consensus_series=True,
    )
    payload = condition_result_to_dict(result, save_consensus_series=True)
    assert "consensus_rate_series" in payload["consensus"]["unanimous"]


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
