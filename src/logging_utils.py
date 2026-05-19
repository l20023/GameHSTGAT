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


def log_condition_metrics(
    *,
    run: Any,
    condition_key: str,
    condition_index: int,
    metrics: dict[str, Any],
) -> None:
    """Log one condition's summary and epsilon(t) series to W&B."""
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
    }
    epsilon_series = metrics.get("epsilon_series", [])
    for t, epsilon_t in enumerate(epsilon_series, start=1):
        payload[f"{condition_key}/epsilon_t/{t}"] = float(epsilon_t)

    plot_path = metrics.get("learning_rate_plot")
    if isinstance(plot_path, str) and Path(plot_path).is_file():
        try:
            import wandb  # type: ignore

            payload[f"{condition_key}/learning_rate_plot"] = wandb.Image(plot_path)
        except Exception:
            pass

    run.log(payload)


def finish_wandb_run(run: Any) -> None:
    """Safely finish an initialized W&B run."""
    run.finish()


__all__ = ["finish_wandb_run", "init_wandb_run", "log_condition_metrics"]
