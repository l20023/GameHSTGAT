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


__all__ = ["load_yaml_config", "merge_flat_config"]
