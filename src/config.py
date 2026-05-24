"""Configuration helpers for training entrypoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml_config(path: Path) -> dict[str, Any]:
    """Load YAML config file and return dict."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError("YAML config root must be a mapping/object.")
    return payload


def merge_flat_config(
    *,
    defaults: dict[str, Any],
    yaml_config: dict[str, Any],
    cli_overrides: dict[str, Any],
) -> dict[str, Any]:
    """Merge defaults < yaml_config < cli_overrides for flat configs."""
    merged = dict(defaults)
    merged.update(yaml_config)
    merged.update({key: value for key, value in cli_overrides.items() if value is not None})
    return merged


def resolve_replication_seeds(
    *,
    cli_seeds: str | None = None,
    run_config: dict[str, Any] | None = None,
    default_count: int = 5,
) -> list[int]:
    """
    Resolve replication seeds for grid runs.

    Precedence: CLI comma-list > explicit YAML ``seeds`` list > ``num_seeds`` in YAML
    > ``default_count`` seeds ``0 .. default_count-1`` (default 5).
    """
    if cli_seeds is not None and cli_seeds.strip():
        parsed = [int(item.strip()) for item in cli_seeds.split(",") if item.strip()]
        if not parsed:
            raise ValueError("Expected at least one seed in --seeds.")
        return parsed

    config = run_config or {}
    yaml_seeds = config.get("seeds")
    if yaml_seeds is not None:
        if not isinstance(yaml_seeds, list) or not yaml_seeds:
            raise ValueError("Config 'seeds' must be a non-empty list of integers.")
        return [int(seed) for seed in yaml_seeds]

    num_seeds = config.get("num_seeds", default_count)
    if not isinstance(num_seeds, int) or num_seeds < 1:
        raise ValueError("Config 'num_seeds' must be an integer >= 1.")
    return list(range(num_seeds))


def build_grid_config_snapshot(
    *,
    run_config: dict[str, Any],
    seeds: list[int],
    num_nodes_list: list[int],
    signal_quality_list: list[float],
    graph_cache_dir: Path,
    artifacts_root: Path,
    communication_mode: str,
    communication_dim: int | None,
    train_episodes_per_n: dict[int, int] | None,
) -> dict[str, Any]:
    """Build a reproducibility snapshot for grid summary JSON."""
    return {
        "seeds": seeds,
        "num_nodes_list": num_nodes_list,
        "signal_quality_list": signal_quality_list,
        "train_episodes": int(run_config["train_episodes"]),
        "train_episodes_per_n": train_episodes_per_n,
        "test_episodes": int(run_config["test_episodes"]),
        "max_horizon": int(run_config["max_horizon"]),
        "hidden_dim": int(run_config["hidden_dim"]),
        "num_heads": int(run_config["num_heads"]),
        "communication_mode": communication_mode,
        "communication_dim": communication_dim,
        "learning_rate": float(run_config["learning_rate"]),
        "weight_decay": float(run_config["weight_decay"]),
        "dropout": float(run_config["dropout"]),
        "validation_episodes": int(run_config["validation_episodes"]),
        "validation_eval_every": int(run_config["validation_eval_every"]),
        "device": str(run_config["device"]),
        "disable_beta_fit": bool(run_config["disable_beta_fit"]),
        "save_train_loss_history": bool(run_config["save_train_loss_history"]),
        "save_epsilon_series": bool(run_config["save_epsilon_series"]),
        "save_learning_rate_plots": bool(run_config["save_learning_rate_plots"]),
        "wandb_project": str(run_config["wandb_project"]),
        "wandb_entity": run_config["wandb_entity"],
        "graph_cache_dir": str(graph_cache_dir),
        "artifacts_root": str(artifacts_root),
    }


__all__ = [
    "build_grid_config_snapshot",
    "load_yaml_config",
    "merge_flat_config",
    "resolve_replication_seeds",
]
