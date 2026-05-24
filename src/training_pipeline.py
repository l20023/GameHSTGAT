"""Training and evaluation pipeline utilities for recurrent GAT experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch_geometric.data import Data

from .hst_bound import compute_beta_hst_max
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
    num_nodes = int(graph_data.num_nodes)
    signal_generator = PrivateSignalGenerator(signal_quality=signal_quality, default_seed=seed)
    error_counts = torch.zeros(max_horizon, dtype=torch.float64, device=device)
    total_predictions_per_t = float(test_episodes * num_nodes)
    graph_data_device = graph_data.to(device)

    model.eval()
    with torch.no_grad():
        for episode_idx in range(test_episodes):
            episode = signal_generator.generate_episode(
                num_nodes=num_nodes,
                max_horizon=max_horizon,
                seed=_episode_seed(seed, episode_idx, offset=TEST_EPISODE_OFFSET),
            )
            x_sequences, targets = _episode_to_model_inputs(
                episode, num_nodes=num_nodes, device=device
            )
            all_logits = model(x_sequences, graph_data_device.edge_index, max_horizon=max_horizon)
            predictions = all_logits.argmax(dim=-1)  # [T, N]
            errors_per_t = (predictions != targets.unsqueeze(0)).sum(dim=1).to(torch.float64)
            error_counts += errors_per_t

    epsilon = (error_counts / total_predictions_per_t).cpu()
    return epsilon.tolist()


def exponential_decay_values(
    t_values: np.ndarray, *, alpha: float, beta: float, epsilon_inf: float
) -> np.ndarray:
    """Evaluate epsilon(t) = alpha * exp(-beta * t) + epsilon_inf."""
    return alpha * np.exp(-beta * t_values) + epsilon_inf


def anchored_t0_decay_values(
    t_values: np.ndarray, *, beta: float, epsilon_inf: float
) -> np.ndarray:
    """Evaluate epsilon(t) with epsilon(0)=0.5 anchor and free beta/epsilon_inf."""
    alpha = 0.5 - epsilon_inf
    return exponential_decay_values(t_values, alpha=alpha, beta=beta, epsilon_inf=epsilon_inf)


def _exponential_decay(
    t_values: np.ndarray, alpha: float, beta: float, epsilon_inf: float
) -> np.ndarray:
    return exponential_decay_values(
        t_values, alpha=alpha, beta=beta, epsilon_inf=epsilon_inf
    )


def _anchored_t0_decay(
    t_values: np.ndarray, beta: float, epsilon_inf: float
) -> np.ndarray:
    return anchored_t0_decay_values(t_values, beta=beta, epsilon_inf=epsilon_inf)


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


WALD_CI_Z = 1.959963984540054  # 95% normal-approx z-score
PLATEAU_TOLERANCE = 0.05  # plateau threshold as fraction of initial decay range
PLATEAU_MIN_FIT_POINTS = 5  # minimum number of points required to fit
PLATEAU_CONSECUTIVE_BELOW = 3  # consecutive sub-threshold points needed


def _detect_fit_window(
    y: np.ndarray,
    *,
    plateau_tolerance: float = PLATEAU_TOLERANCE,
    min_fit_points: int = PLATEAU_MIN_FIT_POINTS,
    consecutive_below: int = PLATEAU_CONSECUTIVE_BELOW,
) -> int:
    """
    Find the maximum t index to use for an exponential decay fit before
    epsilon(t) plateaus near epsilon_inf.

    A plateau is only declared when ALL of the following hold:
    1. The tail (last 10% of points) is itself approximately flat
       (variation in the tail < plateau_tolerance * total decay range).
    2. There are `consecutive_below` consecutive points below the threshold
       `eps_inf_estimate + plateau_tolerance * decay_range`. This protects
       against single noisy outlier points triggering a premature cutoff.
    3. The window keeps at least `min_fit_points` leading points for the fit.

    Returns the number of leading points to include. A return value < n
    indicates a plateau was detected.
    """
    n = len(y)
    if n <= min_fit_points:
        return n
    tail_size = max(3, n // 10)
    tail = y[-tail_size:]
    eps_inf_estimate = float(np.median(tail))
    decay_range = float(y[0]) - eps_inf_estimate
    if decay_range <= 0.0:
        return n
    # Require the tail itself to be approximately flat (still-decaying tails
    # would make eps_inf_estimate biased and produce false plateau signals).
    tail_decay = float(tail[0] - tail[-1])
    if tail_decay > plateau_tolerance * decay_range:
        return n
    threshold = eps_inf_estimate + plateau_tolerance * decay_range
    streak = 0
    plateau_start: int | None = None
    for i in range(n):
        if float(y[i]) < threshold:
            streak += 1
            if streak >= consecutive_below:
                plateau_start = i - consecutive_below + 1
                break
        else:
            streak = 0
    if plateau_start is None:
        return n
    window_size = plateau_start + 1
    if window_size < min_fit_points:
        return n
    return window_size


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
) -> dict[str, float | bool | str]:
    y_pred = _exponential_decay(t_values, alpha, beta, epsilon_inf)
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
        "failure_reason": failure_reason if not fit_success else "",
    }


def _fallback_beta_fit(
    epsilon_series: list[float],
    *,
    fit_window_t_max: int | None = None,
) -> dict[str, float | bool | str]:
    """Fit anchored epsilon decay using a log-linear fallback without scipy."""
    y_full = np.asarray(epsilon_series, dtype=float)
    n_full = len(y_full)
    window = n_full if fit_window_t_max is None else int(fit_window_t_max)
    plateau_detected = window < n_full
    y = y_full[:window]
    t = np.arange(1, len(y) + 1, dtype=float)
    min_positive = max(float(np.max(y)) * 1e-6, 1e-9)

    # Search epsilon_inf candidates to stabilize the log-linear fallback.
    y_min = float(np.min(y))
    epsilon_upper = min(0.5 - min_positive, y_min - min_positive)
    if epsilon_upper <= 0.0:
        epsilon_upper = min_positive
    candidates = np.linspace(0.0, epsilon_upper, num=80, dtype=float)
    if not np.any(np.isclose(candidates, epsilon_upper)):
        candidates = np.append(candidates, epsilon_upper)

    best_result: tuple[float, float, float] | None = None
    for epsilon_inf in candidates:
        if epsilon_inf >= 0.5:
            continue
        anchor_scale = 0.5 - float(epsilon_inf)
        if anchor_scale <= 0.0:
            continue
        shifted = y - float(epsilon_inf)
        if np.any(shifted <= 0.0):
            continue
        normalized = shifted / anchor_scale
        if np.any(normalized <= 0.0):
            continue
        slope, _ = np.polyfit(t, np.log(normalized), 1)
        beta = max(0.0, float(-slope))
        alpha = anchor_scale
        y_pred = anchored_t0_decay_values(t, beta=beta, epsilon_inf=float(epsilon_inf))
        rmse, _ = _fit_quality_metrics(y, y_pred)
        if best_result is None or rmse < best_result[0]:
            best_result = (rmse, beta, float(epsilon_inf))

    if best_result is None:
        empty = _empty_fit_result(
            "anchored_t0_log_linear_fallback", "fallback_failed"
        )
        empty["fit_window_t_max"] = window
        empty["n_full_series"] = n_full
        empty["plateau_detected"] = plateau_detected
        return empty

    _, beta, epsilon_inf = best_result
    alpha = 0.5 - epsilon_inf
    return _finalize_fit_result(
        method="anchored_t0_log_linear_fallback",
        alpha=alpha,
        beta=beta,
        epsilon_inf=epsilon_inf,
        y_true=y,
        t_values=t,
        fit_window_t_max=window,
        plateau_detected=plateau_detected,
        n_full_series=n_full,
    )


def _empty_fit_result(method: str, failure_reason: str) -> dict[str, float | bool | str]:
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
        "failure_reason": failure_reason,
    }


def fit_beta_from_epsilon(epsilon_series: list[float]) -> dict[str, float | bool | str]:
    """Fit anchored epsilon(t) with epsilon(0)=0.5 and robust fallback.

    Uses a slope-based plateau cutoff to truncate trailing flat data points
    before fitting. The full series is still stored as `n_full_series` and the
    truncation length as `fit_window_t_max` in the returned dict.
    """
    if len(epsilon_series) < 3:
        return _empty_fit_result("insufficient_points", "insufficient_points")

    y_full = np.asarray(epsilon_series, dtype=float)
    if not np.all(np.isfinite(y_full)):
        return _empty_fit_result("non_finite_input", "non_finite_input")

    n_full = len(y_full)
    window = _detect_fit_window(y_full)
    plateau_detected = window < n_full
    y = y_full[:window]
    t = np.arange(1, len(y) + 1, dtype=float)
    try:
        from scipy.optimize import curve_fit  # type: ignore

        epsilon_inf_guess = max(0.0, min(0.49, float(np.mean(y[-min(5, len(y)) :]))))
        initial_guess = (0.1, epsilon_inf_guess)
        bounds = ((0.0, 0.0), (10.0, 0.499999))
        params, pcov = curve_fit(
            _anchored_t0_decay,
            t,
            y,
            p0=initial_guess,
            bounds=bounds,
            maxfev=10_000,
        )
        beta = float(params[0])
        epsilon_inf = float(params[1])
        beta_variance = float(pcov[0, 0]) if np.all(np.isfinite(pcov)) else float("nan")
        beta_std = float(np.sqrt(beta_variance)) if beta_variance >= 0.0 else float("nan")
        return _finalize_fit_result(
            method="scipy_anchored_t0",
            alpha=0.5 - epsilon_inf,
            beta=beta,
            epsilon_inf=epsilon_inf,
            y_true=y,
            t_values=t,
            beta_std=beta_std,
            fit_window_t_max=window,
            plateau_detected=plateau_detected,
            n_full_series=n_full,
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
    epsilon_series = compute_epsilon_series(
        model=train_outcome.model,
        graph_data=graph_data,
        test_episodes=test_episodes,
        max_horizon=max_horizon,
        signal_quality=signal_quality,
        seed=seed,
        device=device,
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


def condition_result_to_dict(
    result: ConditionRunResult,
    *,
    save_train_loss_history: bool = False,
    save_epsilon_series: bool = False,
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
    return payload


__all__ = [
    "ConditionRunResult",
    "TrainOutcome",
    "condition_result_to_dict",
    "compute_epsilon_series",
    "anchored_t0_decay_values",
    "exponential_decay_values",
    "fit_beta_from_epsilon",
    "resolve_runtime_device",
    "run_condition_experiment",
    "train_condition_model",
]
