"""Tests for grid task indexing."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.grid_tasks import (
    build_grid_tasks,
    parse_train_episodes_per_n,
    resolve_train_episodes,
    task_artifacts_dir,
)


def test_build_grid_tasks_default_grid_has_sixty_tasks() -> None:
    tasks = build_grid_tasks(
        seeds=[0, 1, 2, 3, 4],
        num_nodes_list=[10, 100, 1000],
        signal_quality_list=[0.55, 0.6, 0.7, 0.8],
    )
    assert len(tasks) == 60
    assert tasks[0].task_index == 0
    assert tasks[-1].task_index == 59


def test_build_grid_tasks_order_is_q_then_n_then_seed() -> None:
    tasks = build_grid_tasks(
        seeds=[0, 1],
        num_nodes_list=[10, 100],
        signal_quality_list=[0.6, 0.8],
    )
    assert (tasks[0].seed, tasks[0].num_nodes, tasks[0].signal_quality) == (0, 10, 0.6)
    assert (tasks[1].seed, tasks[1].num_nodes, tasks[1].signal_quality) == (1, 10, 0.6)
    assert (tasks[2].seed, tasks[2].num_nodes, tasks[2].signal_quality) == (0, 100, 0.6)
    assert (tasks[3].seed, tasks[3].num_nodes, tasks[3].signal_quality) == (1, 100, 0.6)
    assert (tasks[4].seed, tasks[4].num_nodes, tasks[4].signal_quality) == (0, 10, 0.8)


def test_task_artifacts_dir_matches_grid_layout() -> None:
    tasks = build_grid_tasks(
        seeds=[3],
        num_nodes_list=[10],
        signal_quality_list=[0.6],
    )
    path = task_artifacts_dir(Path("artifacts/training_metrics_fair/grid_runs"), tasks[0])
    assert path == Path("artifacts/training_metrics_fair/grid_runs/n_10/q_0p6")


def test_resolve_train_episodes_uses_per_n_mapping() -> None:
    mapping = {10: 5000, 100: 7000, 1000: 10000}
    assert resolve_train_episodes(
        num_nodes=100,
        train_episodes_per_n=mapping,
        train_episodes_default=999,
    ) == 7000
    assert resolve_train_episodes(
        num_nodes=42,
        train_episodes_per_n=mapping,
        train_episodes_default=999,
    ) == 999


def test_parse_train_episodes_per_n_rejects_invalid_payload() -> None:
    with pytest.raises(ValueError, match="train_episodes_per_n"):
        parse_train_episodes_per_n(["not", "a", "dict"])
