"""Grid task indexing for parallel SLURM array runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROPOSAL_GRID_SIGNAL_QUALITIES: tuple[float, ...] = (0.55, 0.65, 0.8)
DEFAULT_MAX_HORIZON = 50
EXTENDED_MAX_HORIZON = 100
EXTENDED_MAX_HORIZON_Q = 0.55


@dataclass(frozen=True)
class GridTask:
    """One parallelizable grid cell: (seed, num_nodes, signal_quality)."""

    task_index: int
    seed: int
    num_nodes: int
    signal_quality: float
    setting_key: str
    quality_key: str


def parse_csv_ints(value: str) -> list[int]:
    parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed:
        raise ValueError("Expected at least one integer value.")
    return parsed


def parse_csv_floats(value: str) -> list[float]:
    parsed = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed:
        raise ValueError("Expected at least one float value.")
    return parsed


def format_signal_quality_label(signal_quality: float) -> str:
    """Human-readable q label for setting keys, e.g. 0.55 -> '0.55'."""
    return f"{signal_quality:.2f}".rstrip("0").rstrip(".")


def format_signal_quality_key(signal_quality: float) -> str:
    """Filesystem-safe q key without rounding collisions, e.g. 0.55 -> '0p55'."""
    return format_signal_quality_label(signal_quality).replace(".", "p")


def format_signal_quality(signal_quality: float) -> str:
    """Alias for filesystem key (backward compatibility)."""
    return format_signal_quality_key(signal_quality)


def parse_train_episodes_per_n(raw: Any) -> dict[int, int] | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("train_episodes_per_n must map int -> int.")
    try:
        return {int(key): int(value) for key, value in raw.items()}
    except (TypeError, ValueError) as exc:
        raise ValueError("train_episodes_per_n must map int -> int.") from exc


def proposal_grid_signal_quality_csv() -> str:
    """Default comma-separated q list for grid CLI defaults."""
    return ",".join(format_signal_quality_label(q) for q in PROPOSAL_GRID_SIGNAL_QUALITIES)


def resolve_grid_signal_quality_list(
    *,
    cli_value: str | None = None,
    run_config: dict[str, Any] | None = None,
) -> list[float]:
    """CLI comma-list overrides YAML ``grid_signal_quality_list``, then proposal default."""
    if cli_value is not None and str(cli_value).strip():
        return parse_csv_floats(str(cli_value))
    config = run_config or {}
    yaml_list = config.get("grid_signal_quality_list")
    if yaml_list is not None:
        if not isinstance(yaml_list, list) or not yaml_list:
            raise ValueError("grid_signal_quality_list must be a non-empty list.")
        return [float(q) for q in yaml_list]
    return list(PROPOSAL_GRID_SIGNAL_QUALITIES)


def normalize_topology_name(condition_name: str) -> str:
    """Return topology segment from a condition key or bare name."""
    if "/" in condition_name:
        return condition_name.split("/", 1)[1]
    return condition_name


def is_complete_topology(condition_name: str) -> bool:
    return normalize_topology_name(condition_name) == "complete"


def resolve_max_horizon(
    *,
    signal_quality: float,
    topology_name: str,
    base_horizon: int = DEFAULT_MAX_HORIZON,
    extended_horizon: int = EXTENDED_MAX_HORIZON,
    extended_q: float = EXTENDED_MAX_HORIZON_Q,
) -> int:
    """
    Horizon policy: T=100 only for q=0.55 on non-complete topologies; otherwise T=50.
    """
    if is_complete_topology(topology_name):
        return base_horizon
    if abs(float(signal_quality) - extended_q) < 1e-9:
        return extended_horizon
    return base_horizon


def resolve_train_episodes(
    *,
    num_nodes: int,
    train_episodes_per_n: dict[int, int] | None,
    train_episodes_default: int,
) -> int:
    if train_episodes_per_n and num_nodes in train_episodes_per_n:
        return int(train_episodes_per_n[num_nodes])
    return int(train_episodes_default)


def build_grid_tasks(
    *,
    seeds: list[int],
    num_nodes_list: list[int],
    signal_quality_list: list[float],
) -> list[GridTask]:
    """Build tasks in the same order as run_grid_experiments (q, n, seed)."""
    tasks: list[GridTask] = []
    task_index = 0
    for signal_quality in signal_quality_list:
        for num_nodes in num_nodes_list:
            quality_key = format_signal_quality_key(signal_quality)
            setting_key = f"n_{num_nodes}/q_{format_signal_quality_label(signal_quality)}"
            for seed in seeds:
                tasks.append(
                    GridTask(
                        task_index=task_index,
                        seed=seed,
                        num_nodes=num_nodes,
                        signal_quality=signal_quality,
                        setting_key=setting_key,
                        quality_key=quality_key,
                    )
                )
                task_index += 1
    return tasks


def task_artifacts_dir(artifacts_root: Path, task: GridTask) -> Path:
    return artifacts_root / f"n_{task.num_nodes}" / f"q_{task.quality_key}"


__all__ = [
    "DEFAULT_MAX_HORIZON",
    "EXTENDED_MAX_HORIZON",
    "EXTENDED_MAX_HORIZON_Q",
    "GridTask",
    "PROPOSAL_GRID_SIGNAL_QUALITIES",
    "build_grid_tasks",
    "format_signal_quality",
    "format_signal_quality_key",
    "format_signal_quality_label",
    "is_complete_topology",
    "normalize_topology_name",
    "parse_csv_floats",
    "parse_csv_ints",
    "parse_train_episodes_per_n",
    "proposal_grid_signal_quality_csv",
    "resolve_grid_signal_quality_list",
    "resolve_max_horizon",
    "resolve_train_episodes",
    "task_artifacts_dir",
]
