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
    default_count: int = 10,
) -> list[int]:
    """
    Resolve replication seeds for grid runs.

    Precedence: CLI comma-list > explicit YAML ``seeds`` list > ``num_seeds`` in YAML
    > ``default_count`` seeds ``0 .. default_count-1``.
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


__all__ = ["load_yaml_config", "merge_flat_config", "resolve_replication_seeds"]
