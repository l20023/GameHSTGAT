"""Grid task indexing for parallel SLURM array runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


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
    "GridTask",
    "build_grid_tasks",
    "format_signal_quality",
    "format_signal_quality_key",
    "format_signal_quality_label",
    "parse_csv_floats",
    "parse_csv_ints",
    "parse_train_episodes_per_n",
    "resolve_train_episodes",
    "task_artifacts_dir",
]
