"""Training and evaluation pipeline utilities for recurrent GAT experiments."""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch_geometric.data import Data

from .hst_bound import compute_beta_hst_max
from .majority_vote_baseline import (
    build_adjacency_matrix,
    build_neighbor_lists,
    rollout_majority_vote_episode,
)
from .models import RecurrentGATAgent, SharedSequentialLoss
from .signal_generator import PrivateSignalGenerator

EPISODE_SEED_STRIDE = 100_000_000
TRAIN_EPISODE_OFFSET = 0
VALIDATION_EPISODE_OFFSET = 5_000_000
TEST_EPISODE_OFFSET = 10_000_000
DEFAULT_CONVERGENCE_WARNING_THRESHOLD = 0.05
DEFAULT_GRAD_CLIP_MAX_NORM = 0.0  # disabled by default; opt-in for stability
DEFAULT_RUNNING_MEAN_WINDOW = 100
DEFAULT_VALIDATION_EPISODES = 50
DEFAULT_VALIDATION_EVAL_EVERY = 100
DEFAULT_DROPOUT = 0.1
DEFAULT_WEIGHT_DECAY = 1e-4


def resolve_runtime_device(device: str) -> torch.device:
    """Resolve runtime device from config value."""
    requested = device.strip().lower()
    if requested not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError("device must be one of: auto, cpu, cuda, mps.")

    mps_available = bool(
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()  # type: ignore[attr-defined]
    )

    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if mps_available:
            return torch.device("mps")
        return torch.device("cpu")

    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("device='cuda' requested but CUDA is not available.")
    if requested == "mps" and not mps_available:
        raise ValueError("device='mps' requested but MPS is not available.")
    return torch.device(requested)


@dataclass
class ConditionRunResult:
    """Container for one graph condition's training/evaluation output."""

    train_loss_history: list[float]
    epsilon_series: list[float]
    consensus: dict[str, Any]
    beta_fit: dict[str, float | bool | str]
    beta_hst_max: float
    beta_gap: float | None
    exceeds_hst_bound: bool | None
    convergence_warning: bool
    train_loss_running_mean_final: float | None
    best_running_mean_loss: float | None
    best_episode_idx: int | None
    best_validation_error: float | None
    best_validation_episode_idx: int | None
    validation_history: list[tuple[int, float]]


def _episode_to_model_inputs(
    episode: dict[str, int | torch.Tensor], *, num_nodes: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    private_signals = episode["private_signals"]
    theta = episode["theta"]
    if not isinstance(private_signals, torch.Tensor):
        raise ValueError("episode['private_signals'] must be a torch.Tensor.")
    if not isinstance(theta, int):
        raise ValueError("episode['theta'] must be an int.")

    x_sequences = (
        private_signals.to(device=device, dtype=torch.float32).t().unsqueeze(-1).contiguous()
    )
    targets = torch.full((num_nodes,), theta, dtype=torch.long, device=device)
    return x_sequences, targets


def _episode_seed(base_seed: int, episode_idx: int, offset: int) -> int:
    if episode_idx < 0:
        raise ValueError("episode_idx must be >= 0.")
    if offset < 0 or offset >= EPISODE_SEED_STRIDE:
        raise ValueError(
            f"offset must be in [0, {EPISODE_SEED_STRIDE - 1}] to avoid split collisions."
        )
    if episode_idx + offset >= EPISODE_SEED_STRIDE:
        raise ValueError(
            "episode_idx + offset exceeds split stride; choose smaller episode count or different split design."
        )
    return base_seed * EPISODE_SEED_STRIDE + offset + episode_idx


@dataclass
class TrainOutcome:
    """Detailed training outcome including stability diagnostics."""

    model: RecurrentGATAgent
    train_loss_history: list[float]
    train_loss_running_mean_final: float | None
    best_running_mean_loss: float | None
    best_episode_idx: int | None
    best_validation_error: float | None
    best_validation_episode_idx: int | None
    validation_history: list[tuple[int, float]]


def _evaluate_on_validation(
    *,
    model: RecurrentGATAgent,
    graph_data: Data,
    num_validation_episodes: int,
    max_horizon: int,
    signal_quality: float,
    seed: int,
    device: torch.device,
) -> float:
    """Compute mean error rate on a held-out validation pool of episodes."""
    num_nodes = int(graph_data.num_nodes)
    signal_generator = PrivateSignalGenerator(signal_quality=signal_quality, default_seed=seed)
    error_count = 0.0
    total = 0.0
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for ep_idx in range(num_validation_episodes):
            episode = signal_generator.generate_episode(
                num_nodes=num_nodes,
                max_horizon=max_horizon,
                seed=_episode_seed(seed, ep_idx, offset=VALIDATION_EPISODE_OFFSET),
            )
            x_seq, targets = _episode_to_model_inputs(
                episode, num_nodes=num_nodes, device=device
            )
            all_logits = model(x_seq, graph_data.edge_index, max_horizon=max_horizon)
            preds = all_logits.argmax(dim=-1)
            errors = (preds != targets.unsqueeze(0)).sum().item()
            error_count += float(errors)
            total += float(max_horizon * num_nodes)
    if was_training:
        model.train()
    if total <= 0.0:
        return float("nan")
    return error_count / total


def train_condition_model(
    *,
    graph_data: Data,
    train_episodes: int,
    max_horizon: int,
    signal_quality: float,
    hidden_dim: int,
    num_heads: int,
    communication_mode: str = "fair_1bit",
    communication_dim: int | None = None,
    learning_rate: float,
    seed: int,
    device: torch.device,
    grad_clip_max_norm: float = DEFAULT_GRAD_CLIP_MAX_NORM,
    running_mean_window: int = DEFAULT_RUNNING_MEAN_WINDOW,
    use_best_checkpoint: bool = False,
    dropout: float = DEFAULT_DROPOUT,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    validation_episodes: int = DEFAULT_VALIDATION_EPISODES,
    validation_eval_every: int = DEFAULT_VALIDATION_EVAL_EVERY,
) -> TrainOutcome:
    """Train one model for one graph condition.

    Stability features:
    - dropout in the GAT layer and before the MLP head (regularization)
    - weight_decay in Adam (regularization)
    - optional gradient clipping (off by default)
    - validation-best checkpoint: every `validation_eval_every` episodes the
      model is evaluated on a held-out validation pool of `validation_episodes`
      episodes. The model with the lowest validation error is reloaded at the
      end of training. Set `validation_episodes=0` to disable this.
    - running-mean of training loss tracked for diagnostics; auto-reload only
      if `use_best_checkpoint=True` (deprecated, prefer validation-based).
    """
    num_nodes = int(graph_data.num_nodes)
    signal_generator = PrivateSignalGenerator(signal_quality=signal_quality, default_seed=seed)
    model = RecurrentGATAgent(
        num_features_signal=1,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        dropout=dropout,
        communication_mode=communication_mode,
        communication_dim=communication_dim,
    ).to(device)
    criterion = SharedSequentialLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    graph_data_device = graph_data.to(device)

    train_loss_history: list[float] = []
    best_running_mean_loss: float | None = None
    best_episode_idx: int | None = None
    best_running_mean_state: dict[str, torch.Tensor] | None = None
    best_validation_error: float | None = None
    best_validation_episode_idx: int | None = None
    best_validation_state: dict[str, torch.Tensor] | None = None
    validation_history: list[tuple[int, float]] = []

    model.train()
    for episode_idx in range(train_episodes):
        episode = signal_generator.generate_episode(
            num_nodes=num_nodes,
            max_horizon=max_horizon,
            seed=_episode_seed(seed, episode_idx, offset=TRAIN_EPISODE_OFFSET),
        )
        x_sequences, targets = _episode_to_model_inputs(
            episode, num_nodes=num_nodes, device=device
        )
        optimizer.zero_grad()
        all_logits = model(x_sequences, graph_data_device.edge_index, max_horizon=max_horizon)
        loss = criterion(all_logits, targets)
        loss.backward()
        if grad_clip_max_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_max_norm)
        optimizer.step()
        train_loss_history.append(float(loss.detach().item()))

        # Best-checkpoint tracking by trailing-window mean loss (diagnostic).
        if len(train_loss_history) >= running_mean_window:
            running = sum(train_loss_history[-running_mean_window:]) / running_mean_window
            if best_running_mean_loss is None or running < best_running_mean_loss:
                best_running_mean_loss = running
                best_episode_idx = episode_idx
                if use_best_checkpoint:
                    best_running_mean_state = {
                        name: tensor.detach().clone()
                        for name, tensor in model.state_dict().items()
                    }

        # Periodic validation evaluation -> best-by-validation checkpoint.
        if (
            validation_episodes > 0
            and (episode_idx + 1) % validation_eval_every == 0
        ):
            val_error = _evaluate_on_validation(
                model=model,
                graph_data=graph_data_device,
                num_validation_episodes=validation_episodes,
                max_horizon=max_horizon,
                signal_quality=signal_quality,
                seed=seed,
                device=device,
            )
            validation_history.append((episode_idx, val_error))
            if (
                np.isfinite(val_error)
                and (best_validation_error is None or val_error < best_validation_error)
            ):
                best_validation_error = val_error
                best_validation_episode_idx = episode_idx
                best_validation_state = {
                    name: tensor.detach().clone() for name, tensor in model.state_dict().items()
                }

    # Validation-best checkpoint is the preferred reload target when available.
    if best_validation_state is not None:
        model.load_state_dict(best_validation_state)
    elif use_best_checkpoint and best_running_mean_state is not None:
        model.load_state_dict(best_running_mean_state)

    if len(train_loss_history) >= running_mean_window:
        train_loss_running_mean_final: float | None = (
            sum(train_loss_history[-running_mean_window:]) / running_mean_window
        )
    elif train_loss_history:
        train_loss_running_mean_final = sum(train_loss_history) / len(train_loss_history)
    else:
        train_loss_running_mean_final = None

    return TrainOutcome(
        model=model,
        train_loss_history=train_loss_history,
        train_loss_running_mean_final=train_loss_running_mean_final,
        best_running_mean_loss=best_running_mean_loss,
        best_episode_idx=best_episode_idx,
        best_validation_error=best_validation_error,
        best_validation_episode_idx=best_validation_episode_idx,
        validation_history=validation_history,
    )


def _consensus_flags_at_timestep(
    predictions_t: torch.Tensor, *, theta: int, num_nodes: int
) -> dict[str, bool | float]:
    """Per-timestep unanimous/majority consensus and agreement fraction."""
    counts = torch.bincount(predictions_t.to(torch.int64), minlength=2).to(torch.float64)
    max_count = float(counts.max().item())
    agreement_fraction = max_count / float(num_nodes)
    unanimous = bool(predictions_t.min().item() == predictions_t.max().item())
    majority = max_count > (num_nodes / 2.0)
    unanimous_label = int(predictions_t[0].item()) if unanimous else -1
    majority_label = int(counts.argmax().item()) if majority else -1
    return {
        "unanimous": unanimous,
        "unanimous_correct": unanimous and unanimous_label == theta,
        "unanimous_wrong": unanimous and unanimous_label != theta,
        "majority": majority,
        "majority_correct": majority and majority_label == theta,
        "majority_wrong": majority and majority_label != theta,
        "agreement_fraction": agreement_fraction,
    }


def _new_consensus_mode_accumulator(max_horizon: int) -> dict[str, Any]:
    return {
        "reach_episodes": 0,
        "correct_episodes": 0,
        "wrong_only_episodes": 0,
        "first_correct_count": 0,
        "first_consensus_times": [],
        "consensus_at_t": [0] * max_horizon,
        "correct_at_t": [0] * max_horizon,
        "wrong_at_t": [0] * max_horizon,
        "agreement_sum_at_t": [0.0] * max_horizon,
    }


def _update_consensus_mode_accumulator(
    accumulator: dict[str, Any],
    *,
    max_horizon: int,
    flags_per_t: list[dict[str, bool | float]],
    mode: str,
) -> None:
    reached = False
    ever_correct = False
    first_consensus_t: int | None = None
    first_consensus_correct = False

    for t_idx, flags in enumerate(flags_per_t):
        has_consensus = bool(flags[mode])
        if has_consensus:
            reached = True
            if first_consensus_t is None:
                first_consensus_t = t_idx + 1
                first_consensus_correct = bool(flags[f"{mode}_correct"])
            if flags[f"{mode}_correct"]:
                ever_correct = True
            accumulator["consensus_at_t"][t_idx] += 1
            if flags[f"{mode}_correct"]:
                accumulator["correct_at_t"][t_idx] += 1
            if flags[f"{mode}_wrong"]:
                accumulator["wrong_at_t"][t_idx] += 1
        accumulator["agreement_sum_at_t"][t_idx] += float(flags["agreement_fraction"])

    if reached:
        accumulator["reach_episodes"] += 1
        if ever_correct:
            accumulator["correct_episodes"] += 1
        else:
            accumulator["wrong_only_episodes"] += 1
        if first_consensus_t is not None:
            accumulator["first_consensus_times"].append(first_consensus_t)
            if first_consensus_correct:
                accumulator["first_correct_count"] += 1


def _finalize_consensus_mode_metrics(
    accumulator: dict[str, Any],
    *,
    test_episodes: int,
    max_horizon: int,
    include_series: bool,
) -> dict[str, Any]:
    test_episodes_f = float(test_episodes)
    first_times = accumulator["first_consensus_times"]
    reach = int(accumulator["reach_episodes"])
    metrics: dict[str, Any] = {
        "fraction_episodes_reach_consensus": reach / test_episodes_f,
        "fraction_episodes_consensus_correct": int(accumulator["correct_episodes"]) / test_episodes_f,
        "fraction_episodes_consensus_wrong_only": int(accumulator["wrong_only_episodes"])
        / test_episodes_f,
        "fraction_correct_at_first_consensus": (
            float(accumulator["first_correct_count"]) / float(reach) if reach > 0 else None
        ),
        "mean_first_consensus_t": (
            float(sum(first_times)) / float(len(first_times)) if first_times else None
        ),
        "median_first_consensus_t": (
            float(statistics.median(first_times)) if first_times else None
        ),
    }
    if include_series:
        metrics["consensus_rate_series"] = [
            count / test_episodes_f for count in accumulator["consensus_at_t"]
        ]
        metrics["correct_consensus_rate_series"] = [
            count / test_episodes_f for count in accumulator["correct_at_t"]
        ]
        metrics["wrong_consensus_rate_series"] = [
            count / test_episodes_f for count in accumulator["wrong_at_t"]
        ]
        metrics["agreement_fraction_series"] = [
            total / test_episodes_f for total in accumulator["agreement_sum_at_t"]
        ]
    return metrics


def evaluate_test_episodes(
    *,
    model: RecurrentGATAgent,
    graph_data: Data,
    test_episodes: int,
    max_horizon: int,
    signal_quality: float,
    seed: int,
    device: torch.device,
    include_consensus_series: bool = False,
) -> tuple[list[float], dict[str, Any]]:
    """Evaluate epsilon(t) and consensus metrics in one test pass per episode."""
    num_nodes = int(graph_data.num_nodes)
    signal_generator = PrivateSignalGenerator(signal_quality=signal_quality, default_seed=seed)
    error_counts = torch.zeros(max_horizon, dtype=torch.float64, device=device)
    total_predictions_per_t = float(test_episodes * num_nodes)
    graph_data_device = graph_data.to(device)
    unanimous_acc = _new_consensus_mode_accumulator(max_horizon)
    majority_acc = _new_consensus_mode_accumulator(max_horizon)

    model.eval()
    with torch.no_grad():
        for episode_idx in range(test_episodes):
            episode = signal_generator.generate_episode(
                num_nodes=num_nodes,
                max_horizon=max_horizon,
                seed=_episode_seed(seed, episode_idx, offset=TEST_EPISODE_OFFSET),
            )
            theta = int(episode["theta"])
            x_sequences, targets = _episode_to_model_inputs(
                episode, num_nodes=num_nodes, device=device
            )
            all_logits = model(x_sequences, graph_data_device.edge_index, max_horizon=max_horizon)
            predictions = all_logits.argmax(dim=-1)  # [T, N]
            errors_per_t = (predictions != targets.unsqueeze(0)).sum(dim=1).to(torch.float64)
            error_counts += errors_per_t

            flags_per_t = [
                _consensus_flags_at_timestep(predictions[t], theta=theta, num_nodes=num_nodes)
                for t in range(max_horizon)
            ]
            _update_consensus_mode_accumulator(
                unanimous_acc, max_horizon=max_horizon, flags_per_t=flags_per_t, mode="unanimous"
            )
            _update_consensus_mode_accumulator(
                majority_acc, max_horizon=max_horizon, flags_per_t=flags_per_t, mode="majority"
            )

    epsilon = (error_counts / total_predictions_per_t).cpu().tolist()
    consensus = {
        "unanimous": _finalize_consensus_mode_metrics(
            unanimous_acc,
            test_episodes=test_episodes,
            max_horizon=max_horizon,
            include_series=include_consensus_series,
        ),
        "majority": _finalize_consensus_mode_metrics(
            majority_acc,
            test_episodes=test_episodes,
            max_horizon=max_horizon,
            include_series=include_consensus_series,
        ),
    }
    return epsilon, consensus


def evaluate_majority_baseline_episodes(
    *,
    graph_data: Data,
    test_episodes: int,
    max_horizon: int,
    signal_quality: float,
    seed: int,
    include_consensus_series: bool = False,
) -> tuple[list[float], dict[str, Any]]:
    """Evaluate cumulative majority-vote baseline on test episodes."""
    num_nodes = int(graph_data.num_nodes)
    neighbors = build_neighbor_lists(num_nodes, graph_data.edge_index.cpu())
    adjacency = build_adjacency_matrix(neighbors)
    signal_generator = PrivateSignalGenerator(signal_quality=signal_quality, default_seed=seed)
    error_counts = np.zeros(max_horizon, dtype=np.float64)
    total_predictions_per_t = float(test_episodes * num_nodes)
    unanimous_acc = _new_consensus_mode_accumulator(max_horizon)
    majority_acc = _new_consensus_mode_accumulator(max_horizon)
    episode_rng = np.random.default_rng(seed)

    for episode_idx in range(test_episodes):
        episode = signal_generator.generate_episode(
            num_nodes=num_nodes,
            max_horizon=max_horizon,
            seed=_episode_seed(seed, episode_idx, offset=TEST_EPISODE_OFFSET),
        )
        theta = int(episode["theta"])
        private_signals = episode["private_signals"]
        if not isinstance(private_signals, torch.Tensor):
            raise ValueError("episode['private_signals'] must be a torch.Tensor.")

        predictions = rollout_majority_vote_episode(
            private_signals,
            neighbors,
            max_horizon=max_horizon,
            rng=episode_rng,
            adjacency=adjacency,
        )
        errors_per_t = (predictions != theta).sum(dim=1).numpy().astype(np.float64)
        error_counts += errors_per_t

        flags_per_t = [
            _consensus_flags_at_timestep(predictions[t], theta=theta, num_nodes=num_nodes)
            for t in range(max_horizon)
        ]
        _update_consensus_mode_accumulator(
            unanimous_acc, max_horizon=max_horizon, flags_per_t=flags_per_t, mode="unanimous"
        )
        _update_consensus_mode_accumulator(
            majority_acc, max_horizon=max_horizon, flags_per_t=flags_per_t, mode="majority"
        )

    epsilon = (error_counts / total_predictions_per_t).tolist()
    consensus = {
        "unanimous": _finalize_consensus_mode_metrics(
            unanimous_acc,
            test_episodes=test_episodes,
            max_horizon=max_horizon,
            include_series=include_consensus_series,
        ),
        "majority": _finalize_consensus_mode_metrics(
            majority_acc,
            test_episodes=test_episodes,
            max_horizon=max_horizon,
            include_series=include_consensus_series,
        ),
    }
    return epsilon, consensus


def compute_epsilon_series(
    *,
    model: RecurrentGATAgent,
    graph_data: Data,
    test_episodes: int,
    max_horizon: int,
    signal_quality: float,
    seed: int,
    device: torch.device,
) -> list[float]:
    """Evaluate error rate epsilon(t) across test episodes and nodes."""
    epsilon_series, _ = evaluate_test_episodes(
        model=model,
        graph_data=graph_data,
        test_episodes=test_episodes,
        max_horizon=max_horizon,
        signal_quality=signal_quality,
        seed=seed,
        device=device,
        include_consensus_series=False,
    )
    return epsilon_series


WALD_CI_Z = 1.959963984540054  # 95% normal-approx z-score
PERFECT_ERROR_TOLERANCE = 0.0  # epsilon(t) at or below this counts as perfect
PLATEAU_MIN_FIT_POINTS = 5  # minimum number of points required to fit
FIT_START_T = 2


def anchored_t2_decay_values(
    t_values: np.ndarray,
    *,
    alpha: float,
    beta: float,
    epsilon_inf: float,
    fit_start_t: float = FIT_START_T,
) -> np.ndarray:
    """Evaluate epsilon(t) = alpha * exp(-beta * (t - fit_start_t)) + epsilon_inf."""
    return alpha * np.exp(-beta * (t_values - float(fit_start_t))) + epsilon_inf


def hst_alpha_at_t2_intersection(*, gat_alpha: float) -> float:
    """HST reference shares GAT epsilon_inf and meets the GAT curve at t=2."""
    return float(gat_alpha)


def _free_t2_decay(
    t_values: np.ndarray, alpha: float, beta: float, epsilon_inf: float
) -> np.ndarray:
    return anchored_t2_decay_values(
        t_values, alpha=alpha, beta=beta, epsilon_inf=epsilon_inf
    )


def _fit_quality_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    residuals = y_true - y_pred
    rmse = float(np.sqrt(np.mean(np.square(residuals))))
    ss_res = float(np.sum(np.square(residuals)))
    ss_tot = float(np.sum(np.square(y_true - float(np.mean(y_true)))))
    if ss_tot <= 1e-12:
        r2 = float("nan")
    else:
        r2 = 1.0 - (ss_res / ss_tot)
    return rmse, r2


def _detect_fit_window(
    y: np.ndarray,
    *,
    perfect_error_tolerance: float = PERFECT_ERROR_TOLERANCE,
    min_fit_points: int = PLATEAU_MIN_FIT_POINTS,
) -> int:
    """
    Find the number of leading points to use for an exponential decay fit.

    Excludes only a trailing suffix where every point has perfect error rate
    (epsilon(t) <= perfect_error_tolerance). Low but non-zero error plateaus
    are never auto-truncated.
    """
    n = len(y)
    if n <= min_fit_points:
        return n
    perfect_suffix_start = n
    for i in range(n - 1, -1, -1):
        if float(y[i]) <= perfect_error_tolerance:
            perfect_suffix_start = i
        else:
            break
    if perfect_suffix_start >= n:
        return n
    window = perfect_suffix_start
    if window < min_fit_points:
        return n
    return window


def _finalize_fit_result(
    *,
    method: str,
    alpha: float,
    beta: float,
    epsilon_inf: float,
    y_true: np.ndarray,
    t_values: np.ndarray,
    beta_std: float | None = None,
    fit_window_t_max: int | None = None,
    plateau_detected: bool = False,
    n_full_series: int | None = None,
    fit_start_t: int | None = None,
) -> dict[str, float | bool | str]:
    start_t = float(fit_start_t if fit_start_t is not None else FIT_START_T)
    y_pred = anchored_t2_decay_values(
        t_values, alpha=alpha, beta=beta, epsilon_inf=epsilon_inf, fit_start_t=start_t
    )
    rmse, r2 = _fit_quality_metrics(y_true, y_pred)

    fit_success = True
    failure_reason = ""
    if not np.isfinite(alpha) or not np.isfinite(beta) or not np.isfinite(epsilon_inf):
        fit_success = False
        failure_reason = "non_finite_params"
    elif beta < 0.0:
        fit_success = False
        failure_reason = "negative_beta"
    elif epsilon_inf < 0.0 or epsilon_inf > 1.0:
        fit_success = False
        failure_reason = "epsilon_inf_out_of_range"
    elif len(y_true) >= 5 and np.isfinite(r2) and r2 < -0.1:
        fit_success = False
        failure_reason = "poor_fit_r2"

    if beta_std is None or not np.isfinite(beta_std):
        beta_std_value = float("nan")
        beta_ci_lower = float("nan")
        beta_ci_upper = float("nan")
    else:
        beta_std_value = float(beta_std)
        beta_ci_lower = float(beta - WALD_CI_Z * beta_std_value)
        beta_ci_upper = float(beta + WALD_CI_Z * beta_std_value)

    if fit_window_t_max is None:
        fit_window_t_max_value = len(t_values)
    else:
        fit_window_t_max_value = int(fit_window_t_max)
    n_full = int(n_full_series) if n_full_series is not None else fit_window_t_max_value

    return {
        "alpha": float(alpha),
        "beta": float(beta),
        "beta_std": beta_std_value,
        "beta_ci_lower": beta_ci_lower,
        "beta_ci_upper": beta_ci_upper,
        "epsilon_inf": float(epsilon_inf),
        "fit_success": fit_success,
        "method": method,
        "rmse": rmse,
        "r2": r2,
        "fit_window_t_max": fit_window_t_max_value,
        "n_full_series": n_full,
        "plateau_detected": bool(plateau_detected),
        "fit_anchor": "t2",
        "fit_start_t": int(fit_start_t if fit_start_t is not None else FIT_START_T),
        "failure_reason": failure_reason if not fit_success else "",
    }


def _fallback_beta_fit(
    epsilon_series: list[float],
    *,
    fit_window_t_max: int | None = None,
) -> dict[str, float | bool | str]:
    """Log-linear fallback for t>=2 anchored epsilon decay."""
    y_full = np.asarray(epsilon_series, dtype=float)
    n_full = len(y_full)
    window = n_full if fit_window_t_max is None else int(fit_window_t_max)
    plateau_detected = window < n_full
    y = y_full[:window]
    min_positive = max(float(np.max(y)) * 1e-6, 1e-9)

    if len(y) < FIT_START_T + 2:
        return _empty_fit_result("anchored_t2_log_linear_fallback", "insufficient_points")

    y_fit = y[FIT_START_T - 1 :]
    t_fit = np.arange(FIT_START_T, FIT_START_T + len(y_fit), dtype=float)
    epsilon_2 = float(y_fit[0])
    y_min = float(np.min(y_fit))
    epsilon_upper = min(epsilon_2 - min_positive, y_min - min_positive)
    if epsilon_upper <= 0.0:
        epsilon_upper = min_positive
    candidates = np.linspace(0.0, epsilon_upper, num=80, dtype=float)
    if not np.any(np.isclose(candidates, epsilon_upper)):
        candidates = np.append(candidates, epsilon_upper)

    best_result: tuple[float, float, float, float] | None = None
    for epsilon_inf in candidates:
        if epsilon_inf >= epsilon_2:
            continue
        shifted = y_fit - float(epsilon_inf)
        if np.any(shifted <= 0.0):
            continue
        slope, intercept = np.polyfit(t_fit - float(FIT_START_T), np.log(shifted), 1)
        beta = max(0.0, float(-slope))
        alpha = float(np.exp(intercept))
        if alpha <= 0.0:
            continue
        y_pred = anchored_t2_decay_values(
            t_fit, alpha=alpha, beta=beta, epsilon_inf=float(epsilon_inf)
        )
        rmse, _ = _fit_quality_metrics(y_fit, y_pred)
        if best_result is None or rmse < best_result[0]:
            best_result = (rmse, alpha, beta, float(epsilon_inf))

    if best_result is None:
        empty = _empty_fit_result("anchored_t2_log_linear_fallback", "fallback_failed")
        empty["fit_window_t_max"] = window
        empty["n_full_series"] = n_full
        empty["plateau_detected"] = plateau_detected
        empty["fit_start_t"] = FIT_START_T
        return empty

    _, alpha, beta, epsilon_inf = best_result
    return _finalize_fit_result(
        method="anchored_t2_log_linear_fallback",
        alpha=alpha,
        beta=beta,
        epsilon_inf=epsilon_inf,
        y_true=y_fit,
        t_values=t_fit,
        fit_window_t_max=window,
        plateau_detected=plateau_detected,
        n_full_series=n_full,
        fit_start_t=FIT_START_T,
    )


def _empty_fit_result(
    method: str, failure_reason: str
) -> dict[str, float | bool | str]:
    return {
        "alpha": float("nan"),
        "beta": float("nan"),
        "beta_std": float("nan"),
        "beta_ci_lower": float("nan"),
        "beta_ci_upper": float("nan"),
        "epsilon_inf": float("nan"),
        "fit_success": False,
        "method": method,
        "rmse": float("nan"),
        "r2": float("nan"),
        "fit_window_t_max": 0,
        "n_full_series": 0,
        "plateau_detected": False,
        "fit_anchor": "t2",
        "fit_start_t": FIT_START_T,
        "failure_reason": failure_reason,
    }


def fit_beta_from_epsilon(
    epsilon_series: list[float],
    *,
    fit_window_t_max: int | None = None,
) -> dict[str, float | bool | str]:
    """Fit t>=2 anchored decay: epsilon(t) = alpha * exp(-beta * (t-2)) + epsilon_inf."""
    min_points = FIT_START_T + 2
    if len(epsilon_series) < min_points:
        return _empty_fit_result("insufficient_points", "insufficient_points")

    y_full = np.asarray(epsilon_series, dtype=float)
    if not np.all(np.isfinite(y_full)):
        return _empty_fit_result("non_finite_input", "non_finite_input")

    n_full = len(y_full)
    if fit_window_t_max is None:
        window = _detect_fit_window(y_full)
    else:
        window = min(max(1, int(fit_window_t_max)), n_full)
    plateau_detected = window < n_full
    y = y_full[:window]
    try:
        from scipy.optimize import curve_fit  # type: ignore

        if len(y) < FIT_START_T + 2:
            return _empty_fit_result("insufficient_points", "insufficient_points")

        y_fit = y[FIT_START_T - 1 :]
        t_fit = np.arange(FIT_START_T, FIT_START_T + len(y_fit), dtype=float)
        epsilon_2 = float(y_fit[0])
        epsilon_inf_guess = max(
            0.0,
            min(epsilon_2 - 1e-6, float(np.mean(y_fit[-min(5, len(y_fit)) :]))),
        )
        alpha_guess = max(epsilon_2 - epsilon_inf_guess, 1e-6)
        initial_guess = (alpha_guess, 0.1, epsilon_inf_guess)
        bounds = (
            (0.0, 0.0, 0.0),
            (1.0, 10.0, max(epsilon_2 - 1e-9, 1e-9)),
        )
        params, pcov = curve_fit(
            _free_t2_decay,
            t_fit,
            y_fit,
            p0=initial_guess,
            bounds=bounds,
            maxfev=10_000,
        )
        alpha = float(params[0])
        beta = float(params[1])
        epsilon_inf = float(params[2])
        beta_variance = float(pcov[1, 1]) if np.all(np.isfinite(pcov)) else float("nan")
        beta_std = float(np.sqrt(beta_variance)) if beta_variance >= 0.0 else float("nan")
        return _finalize_fit_result(
            method="scipy_anchored_t2",
            alpha=alpha,
            beta=beta,
            epsilon_inf=epsilon_inf,
            y_true=y_fit,
            t_values=t_fit,
            beta_std=beta_std,
            fit_window_t_max=window,
            plateau_detected=plateau_detected,
            n_full_series=n_full,
            fit_start_t=FIT_START_T,
        )
    except Exception:
        return _fallback_beta_fit(epsilon_series, fit_window_t_max=window)


def run_condition_experiment(
    *,
    graph_data: Data,
    train_episodes: int,
    test_episodes: int,
    max_horizon: int,
    signal_quality: float,
    hidden_dim: int,
    num_heads: int,
    communication_mode: str = "fair_1bit",
    communication_dim: int | None = None,
    learning_rate: float,
    seed: int,
    device: torch.device,
    disable_beta_fit: bool = False,
    convergence_warning_threshold: float = DEFAULT_CONVERGENCE_WARNING_THRESHOLD,
    dropout: float = DEFAULT_DROPOUT,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    validation_episodes: int = DEFAULT_VALIDATION_EPISODES,
    validation_eval_every: int = DEFAULT_VALIDATION_EVAL_EVERY,
    include_consensus_series: bool = False,
) -> ConditionRunResult:
    """Run train + eval + beta fit for one graph condition."""
    train_outcome = train_condition_model(
        graph_data=graph_data,
        train_episodes=train_episodes,
        max_horizon=max_horizon,
        signal_quality=signal_quality,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        communication_mode=communication_mode,
        communication_dim=communication_dim,
        learning_rate=learning_rate,
        seed=seed,
        device=device,
        dropout=dropout,
        weight_decay=weight_decay,
        validation_episodes=validation_episodes,
        validation_eval_every=validation_eval_every,
    )
    epsilon_series, consensus = evaluate_test_episodes(
        model=train_outcome.model,
        graph_data=graph_data,
        test_episodes=test_episodes,
        max_horizon=max_horizon,
        signal_quality=signal_quality,
        seed=seed,
        device=device,
        include_consensus_series=include_consensus_series,
    )
    if disable_beta_fit:
        beta_fit: dict[str, float | bool | str] = _empty_fit_result("disabled", "disabled")
    else:
        beta_fit = fit_beta_from_epsilon(epsilon_series)

    beta_hst_max = compute_beta_hst_max(signal_quality)
    epsilon_inf_value = beta_fit.get("epsilon_inf")
    convergence_warning = bool(
        isinstance(epsilon_inf_value, (int, float))
        and np.isfinite(float(epsilon_inf_value))
        and float(epsilon_inf_value) > convergence_warning_threshold
    )
    beta_gap: float | None = None
    exceeds_hst_bound: bool | None = None
    if bool(beta_fit.get("fit_success", False)):
        beta_gat_raw = beta_fit.get("beta")
        if isinstance(beta_gat_raw, (int, float)) and np.isfinite(float(beta_gat_raw)):
            beta_gap = float(beta_gat_raw) - beta_hst_max
            # exceeds_hst_bound is only meaningful when the model has actually
            # converged. Otherwise a non-converged run with a steep but
            # spurious early decay can produce false counter-evidence.
            exceeds_hst_bound = (beta_gap > 0.0) and not convergence_warning

    return ConditionRunResult(
        train_loss_history=train_outcome.train_loss_history,
        epsilon_series=epsilon_series,
        consensus=consensus,
        beta_fit=beta_fit,
        beta_hst_max=beta_hst_max,
        beta_gap=beta_gap,
        exceeds_hst_bound=exceeds_hst_bound,
        convergence_warning=convergence_warning,
        train_loss_running_mean_final=train_outcome.train_loss_running_mean_final,
        best_running_mean_loss=train_outcome.best_running_mean_loss,
        best_episode_idx=train_outcome.best_episode_idx,
        best_validation_error=train_outcome.best_validation_error,
        best_validation_episode_idx=train_outcome.best_validation_episode_idx,
        validation_history=train_outcome.validation_history,
    )


def run_majority_baseline_condition(
    *,
    graph_data: Data,
    test_episodes: int,
    max_horizon: int,
    signal_quality: float,
    seed: int,
    disable_beta_fit: bool = False,
    convergence_warning_threshold: float = DEFAULT_CONVERGENCE_WARNING_THRESHOLD,
    include_consensus_series: bool = False,
) -> ConditionRunResult:
    """Evaluate cumulative majority-vote baseline (no training) for one graph condition."""
    epsilon_series, consensus = evaluate_majority_baseline_episodes(
        graph_data=graph_data,
        test_episodes=test_episodes,
        max_horizon=max_horizon,
        signal_quality=signal_quality,
        seed=seed,
        include_consensus_series=include_consensus_series,
    )
    if disable_beta_fit:
        beta_fit: dict[str, float | bool | str] = _empty_fit_result("disabled", "disabled")
    else:
        beta_fit = fit_beta_from_epsilon(epsilon_series)

    beta_hst_max = compute_beta_hst_max(signal_quality)
    epsilon_inf_value = beta_fit.get("epsilon_inf")
    convergence_warning = bool(
        isinstance(epsilon_inf_value, (int, float))
        and np.isfinite(float(epsilon_inf_value))
        and float(epsilon_inf_value) > convergence_warning_threshold
    )
    beta_gap: float | None = None
    exceeds_hst_bound: bool | None = None
    if bool(beta_fit.get("fit_success", False)):
        beta_gat_raw = beta_fit.get("beta")
        if isinstance(beta_gat_raw, (int, float)) and np.isfinite(float(beta_gat_raw)):
            beta_gap = float(beta_gat_raw) - beta_hst_max
            exceeds_hst_bound = (beta_gap > 0.0) and not convergence_warning

    return ConditionRunResult(
        train_loss_history=[],
        epsilon_series=epsilon_series,
        consensus=consensus,
        beta_fit=beta_fit,
        beta_hst_max=beta_hst_max,
        beta_gap=beta_gap,
        exceeds_hst_bound=exceeds_hst_bound,
        convergence_warning=convergence_warning,
        train_loss_running_mean_final=None,
        best_running_mean_loss=None,
        best_episode_idx=None,
        best_validation_error=None,
        best_validation_episode_idx=None,
        validation_history=[],
    )


_CONSENSUS_SERIES_KEYS = (
    "consensus_rate_series",
    "correct_consensus_rate_series",
    "wrong_consensus_rate_series",
    "agreement_fraction_series",
)


def consensus_to_dict(
    consensus: dict[str, Any], *, save_consensus_series: bool = False
) -> dict[str, Any]:
    """Serialize consensus metrics; time series are optional."""
    serialized: dict[str, Any] = {}
    for mode in ("unanimous", "majority"):
        mode_metrics = consensus.get(mode, {})
        if not isinstance(mode_metrics, dict):
            continue
        mode_payload = dict(mode_metrics)
        if not save_consensus_series:
            for key in _CONSENSUS_SERIES_KEYS:
                mode_payload.pop(key, None)
        serialized[mode] = mode_payload
    return serialized


def condition_result_to_dict(
    result: ConditionRunResult,
    *,
    save_train_loss_history: bool = False,
    save_epsilon_series: bool = False,
    save_consensus_series: bool = False,
    algorithm: str | None = None,
) -> dict[str, Any]:
    """Convert dataclass result to JSON-serializable dictionary."""
    payload: dict[str, Any] = {
        "train_loss_final": result.train_loss_history[-1] if result.train_loss_history else None,
        "train_loss_running_mean_final": result.train_loss_running_mean_final,
        "best_running_mean_loss": result.best_running_mean_loss,
        "best_episode_idx": result.best_episode_idx,
        "best_validation_error": result.best_validation_error,
        "best_validation_episode_idx": result.best_validation_episode_idx,
        "validation_history": result.validation_history,
        "consensus": consensus_to_dict(
            result.consensus, save_consensus_series=save_consensus_series
        ),
        "beta_fit": result.beta_fit,
        "beta_hst_max": result.beta_hst_max,
        "beta_gap": result.beta_gap,
        "exceeds_hst_bound": result.exceeds_hst_bound,
        "convergence_warning": result.convergence_warning,
    }
    if save_train_loss_history:
        payload["train_loss_history"] = result.train_loss_history
    if save_epsilon_series:
        payload["epsilon_series"] = result.epsilon_series
    if algorithm is not None:
        payload["algorithm"] = algorithm
    return payload


__all__ = [
    "ConditionRunResult",
    "TrainOutcome",
    "condition_result_to_dict",
    "consensus_to_dict",
    "compute_epsilon_series",
    "evaluate_majority_baseline_episodes",
    "evaluate_test_episodes",
    "FIT_START_T",
    "anchored_t2_decay_values",
    "fit_beta_from_epsilon",
    "hst_alpha_at_t2_intersection",
    "resolve_runtime_device",
    "run_condition_experiment",
    "run_majority_baseline_condition",
    "train_condition_model",
]
