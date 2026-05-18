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


@dataclass
class ConditionRunResult:
    """Container for one graph condition's training/evaluation output."""

    train_loss_history: list[float]
    epsilon_series: list[float]
    beta_fit: dict[str, float | bool | str]
    beta_hst_max: float
    beta_gap: float | None
    exceeds_hst_bound: bool | None


def _episode_to_model_inputs(
    episode: dict[str, int | torch.Tensor], *, num_nodes: int
) -> tuple[torch.Tensor, torch.Tensor]:
    private_signals = episode["private_signals"]
    theta = episode["theta"]
    if not isinstance(private_signals, torch.Tensor):
        raise ValueError("episode['private_signals'] must be a torch.Tensor.")
    if not isinstance(theta, int):
        raise ValueError("episode['theta'] must be an int.")

    x_sequences = private_signals.to(dtype=torch.float32).t().unsqueeze(-1).contiguous()
    targets = torch.full((num_nodes,), theta, dtype=torch.long)
    return x_sequences, targets


def _episode_seed(base_seed: int, episode_idx: int, offset: int) -> int:
    return base_seed * 1_000_000 + offset + episode_idx


def train_condition_model(
    *,
    graph_data: Data,
    train_episodes: int,
    max_horizon: int,
    signal_quality: float,
    hidden_dim: int,
    num_heads: int,
    learning_rate: float,
    seed: int,
) -> tuple[RecurrentGATAgent, list[float]]:
    """Train one model for one graph condition and return per-episode loss."""
    num_nodes = int(graph_data.num_nodes)
    signal_generator = PrivateSignalGenerator(signal_quality=signal_quality, default_seed=seed)
    model = RecurrentGATAgent(
        num_features_signal=1,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
    )
    criterion = SharedSequentialLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_loss_history: list[float] = []
    model.train()
    for episode_idx in range(train_episodes):
        episode = signal_generator.generate_episode(
            num_nodes=num_nodes,
            max_horizon=max_horizon,
            seed=_episode_seed(seed, episode_idx, offset=0),
        )
        x_sequences, targets = _episode_to_model_inputs(episode, num_nodes=num_nodes)
        optimizer.zero_grad()
        all_logits = model(x_sequences, graph_data.edge_index, max_horizon=max_horizon)
        loss = criterion(all_logits, targets)
        loss.backward()
        optimizer.step()
        train_loss_history.append(float(loss.detach().item()))

    return model, train_loss_history


def compute_epsilon_series(
    *,
    model: RecurrentGATAgent,
    graph_data: Data,
    test_episodes: int,
    max_horizon: int,
    signal_quality: float,
    seed: int,
) -> list[float]:
    """Evaluate error rate epsilon(t) across test episodes and nodes."""
    num_nodes = int(graph_data.num_nodes)
    signal_generator = PrivateSignalGenerator(signal_quality=signal_quality, default_seed=seed)
    error_counts = torch.zeros(max_horizon, dtype=torch.float64)
    total_predictions_per_t = float(test_episodes * num_nodes)

    model.eval()
    with torch.no_grad():
        for episode_idx in range(test_episodes):
            episode = signal_generator.generate_episode(
                num_nodes=num_nodes,
                max_horizon=max_horizon,
                seed=_episode_seed(seed, episode_idx, offset=100_000),
            )
            x_sequences, targets = _episode_to_model_inputs(episode, num_nodes=num_nodes)
            all_logits = model(x_sequences, graph_data.edge_index, max_horizon=max_horizon)
            predictions = all_logits.argmax(dim=-1)  # [T, N]
            errors_per_t = (predictions != targets.unsqueeze(0)).sum(dim=1).to(torch.float64)
            error_counts += errors_per_t

    epsilon = error_counts / total_predictions_per_t
    return epsilon.tolist()


def _fallback_beta_fit(epsilon_series: list[float]) -> dict[str, float | bool | str]:
    """Fit exponential decay using a log-linear fallback without scipy."""
    y = np.asarray(epsilon_series, dtype=float)
    t = np.arange(1, len(y) + 1, dtype=float)
    tail_len = min(5, len(y))
    epsilon_inf = float(np.mean(y[-tail_len:]))
    shifted = y - epsilon_inf
    min_positive = max(float(np.max(y)) * 1e-6, 1e-9)
    shifted = np.maximum(shifted, min_positive)
    slope, intercept = np.polyfit(t, np.log(shifted), 1)
    beta = max(0.0, float(-slope))
    alpha = float(np.exp(intercept))
    return {
        "alpha": alpha,
        "beta": beta,
        "epsilon_inf": epsilon_inf,
        "fit_success": True,
        "method": "log_linear_fallback",
    }


def fit_beta_from_epsilon(epsilon_series: list[float]) -> dict[str, float | bool | str]:
    """Fit epsilon(t)=alpha*exp(-beta*t)+epsilon_inf with robust fallback."""
    if len(epsilon_series) < 3:
        return {
            "alpha": float("nan"),
            "beta": float("nan"),
            "epsilon_inf": float("nan"),
            "fit_success": False,
            "method": "insufficient_points",
        }

    y = np.asarray(epsilon_series, dtype=float)
    if not np.all(np.isfinite(y)):
        return {
            "alpha": float("nan"),
            "beta": float("nan"),
            "epsilon_inf": float("nan"),
            "fit_success": False,
            "method": "non_finite_input",
        }

    t = np.arange(1, len(y) + 1, dtype=float)
    try:
        from scipy.optimize import curve_fit  # type: ignore

        def exponential_decay(
            t_values: np.ndarray, alpha: float, beta: float, epsilon_inf: float
        ) -> np.ndarray:
            return alpha * np.exp(-beta * t_values) + epsilon_inf

        epsilon_inf_guess = float(np.mean(y[-min(5, len(y)) :]))
        alpha_guess = max(float(y[0] - epsilon_inf_guess), 1e-6)
        initial_guess = (alpha_guess, 0.1, max(0.0, epsilon_inf_guess))
        bounds = ((0.0, 0.0, 0.0), (1.0, 10.0, 1.0))
        params, _ = curve_fit(
            exponential_decay,
            t,
            y,
            p0=initial_guess,
            bounds=bounds,
            maxfev=10_000,
        )
        return {
            "alpha": float(params[0]),
            "beta": float(params[1]),
            "epsilon_inf": float(params[2]),
            "fit_success": True,
            "method": "scipy_curve_fit",
        }
    except Exception:
        return _fallback_beta_fit(epsilon_series)


def run_condition_experiment(
    *,
    graph_data: Data,
    train_episodes: int,
    test_episodes: int,
    max_horizon: int,
    signal_quality: float,
    hidden_dim: int,
    num_heads: int,
    learning_rate: float,
    seed: int,
    disable_beta_fit: bool = False,
) -> ConditionRunResult:
    """Run train + eval + beta fit for one graph condition."""
    model, train_loss_history = train_condition_model(
        graph_data=graph_data,
        train_episodes=train_episodes,
        max_horizon=max_horizon,
        signal_quality=signal_quality,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        learning_rate=learning_rate,
        seed=seed,
    )
    epsilon_series = compute_epsilon_series(
        model=model,
        graph_data=graph_data,
        test_episodes=test_episodes,
        max_horizon=max_horizon,
        signal_quality=signal_quality,
        seed=seed,
    )
    if disable_beta_fit:
        beta_fit: dict[str, float | bool | str] = {
            "alpha": float("nan"),
            "beta": float("nan"),
            "epsilon_inf": float("nan"),
            "fit_success": False,
            "method": "disabled",
        }
    else:
        beta_fit = fit_beta_from_epsilon(epsilon_series)

    beta_hst_max = compute_beta_hst_max(signal_quality)
    beta_gap: float | None = None
    exceeds_hst_bound: bool | None = None
    if bool(beta_fit.get("fit_success", False)):
        beta_gat_raw = beta_fit.get("beta")
        if isinstance(beta_gat_raw, (int, float)) and np.isfinite(float(beta_gat_raw)):
            beta_gap = float(beta_gat_raw) - beta_hst_max
            exceeds_hst_bound = beta_gap > 0.0

    return ConditionRunResult(
        train_loss_history=train_loss_history,
        epsilon_series=epsilon_series,
        beta_fit=beta_fit,
        beta_hst_max=beta_hst_max,
        beta_gap=beta_gap,
        exceeds_hst_bound=exceeds_hst_bound,
    )


def condition_result_to_dict(result: ConditionRunResult) -> dict[str, Any]:
    """Convert dataclass result to JSON-serializable dictionary."""
    return {
        "train_loss_history": result.train_loss_history,
        "train_loss_final": result.train_loss_history[-1] if result.train_loss_history else None,
        "epsilon_series": result.epsilon_series,
        "beta_fit": result.beta_fit,
        "beta_hst_max": result.beta_hst_max,
        "beta_gap": result.beta_gap,
        "exceeds_hst_bound": result.exceeds_hst_bound,
    }


__all__ = [
    "ConditionRunResult",
    "condition_result_to_dict",
    "compute_epsilon_series",
    "fit_beta_from_epsilon",
    "run_condition_experiment",
    "train_condition_model",
]
