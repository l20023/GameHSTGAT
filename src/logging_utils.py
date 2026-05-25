"""Logging helpers for experiment tracking integrations."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def init_wandb_run(
    *,
    project: str,
    entity: str | None,
    seed: int,
    config: dict[str, Any],
) -> Any:
    """Initialize one Weights & Biases run."""
    try:
        import wandb  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised through runtime checks
        raise RuntimeError(
            "wandb is required for training runs. Install it with `pip install wandb`."
        ) from exc

    run_name = f"seed_{seed}"
    return wandb.init(
        project=project,
        entity=entity,
        config=config,
        name=run_name,
        reinit=True,
    )


def log_train_loss_history(
    *,
    run: Any,
    condition_key: str,
    train_loss_history: list[float],
) -> None:
    """Log per-episode training loss for one condition to W&B."""
    if not train_loss_history:
        return

    metric_key = f"{condition_key}/train_loss"
    for episode_idx, loss_value in enumerate(train_loss_history):
        run.log({metric_key: float(loss_value)}, step=episode_idx)


_CONSENSUS_SCALAR_KEYS = (
    "fraction_episodes_reach_consensus",
    "fraction_episodes_consensus_correct",
    "fraction_episodes_consensus_wrong_only",
    "fraction_correct_at_first_consensus",
    "mean_first_consensus_t",
    "median_first_consensus_t",
)

_CONSENSUS_SERIES_KEYS = (
    "consensus_rate_series",
    "correct_consensus_rate_series",
    "wrong_consensus_rate_series",
    "agreement_fraction_series",
)


def _append_consensus_payload(
    payload: dict[str, Any], *, condition_key: str, metrics: dict[str, Any]
) -> None:
    consensus = metrics.get("consensus", {})
    if not isinstance(consensus, dict):
        return
    for mode in ("unanimous", "majority"):
        mode_metrics = consensus.get(mode, {})
        if not isinstance(mode_metrics, dict):
            continue
        prefix = f"{condition_key}/consensus/{mode}"
        for key in _CONSENSUS_SCALAR_KEYS:
            value = mode_metrics.get(key)
            if isinstance(value, (int, float)):
                payload[f"{prefix}/{key}"] = float(value)
        for series_key in _CONSENSUS_SERIES_KEYS:
            series = mode_metrics.get(series_key, [])
            if not isinstance(series, list):
                continue
            series_name = series_key.removesuffix("_series")
            for t_idx, value in enumerate(series, start=1):
                if isinstance(value, (int, float)):
                    payload[f"{prefix}/{series_name}_t/{t_idx}"] = float(value)


def log_condition_metrics(
    *,
    run: Any,
    condition_key: str,
    condition_index: int,
    metrics: dict[str, Any],
    train_loss_history: list[float] | None = None,
) -> None:
    """Log one condition's summary, training loss curve, and optional series to W&B."""
    beta_fit = metrics.get("beta_fit", {})
    payload: dict[str, Any] = {
        "condition/index": condition_index,
        "condition/key": condition_key,
        f"{condition_key}/train_loss_final": metrics.get("train_loss_final"),
        f"{condition_key}/beta_fit/alpha": beta_fit.get("alpha"),
        f"{condition_key}/beta_fit/beta": beta_fit.get("beta"),
        f"{condition_key}/beta_fit/epsilon_inf": beta_fit.get("epsilon_inf"),
        f"{condition_key}/beta_fit/fit_success": beta_fit.get("fit_success"),
        f"{condition_key}/beta_fit/method": beta_fit.get("method"),
        f"{condition_key}/beta_hst_max": metrics.get("beta_hst_max"),
        f"{condition_key}/beta_gap": metrics.get("beta_gap"),
        f"{condition_key}/exceeds_hst_bound": metrics.get("exceeds_hst_bound"),
        f"{condition_key}/convergence_warning": metrics.get("convergence_warning"),
    }
    epsilon_series = metrics.get("epsilon_series", [])
    for t, epsilon_t in enumerate(epsilon_series, start=1):
        payload[f"{condition_key}/epsilon_t/{t}"] = float(epsilon_t)

    _append_consensus_payload(payload, condition_key=condition_key, metrics=metrics)

    plot_path = metrics.get("learning_rate_plot")
    if isinstance(plot_path, str) and Path(plot_path).is_file():
        try:
            import wandb  # type: ignore

            payload[f"{condition_key}/learning_rate_plot"] = wandb.Image(plot_path)
        except Exception:
            pass

    anchored_plot = metrics.get("learning_rate_plot_anchored_t0")
    if isinstance(anchored_plot, str) and Path(anchored_plot).is_file():
        try:
            import wandb  # type: ignore

            payload[f"{condition_key}/learning_rate_plot_anchored_t0"] = wandb.Image(anchored_plot)
        except Exception:
            pass

    train_loss_plot = metrics.get("train_loss_plot")
    if isinstance(train_loss_plot, str) and Path(train_loss_plot).is_file():
        try:
            import wandb  # type: ignore

            payload[f"{condition_key}/train_loss_plot"] = wandb.Image(train_loss_plot)
        except Exception:
            pass

    run.log(payload)

    history = train_loss_history
    if history is None:
        raw_history = metrics.get("train_loss_history")
        if isinstance(raw_history, list):
            history = raw_history
    if history is not None:
        log_train_loss_history(
            run=run,
            condition_key=condition_key,
            train_loss_history=history,
        )


def finish_wandb_run(run: Any) -> None:
    """Safely finish an initialized W&B run."""
    run.finish()


__all__ = [
    "finish_wandb_run",
    "init_wandb_run",
    "log_condition_metrics",
    "log_train_loss_history",
]
