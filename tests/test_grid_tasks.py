"""Tests for grid task indexing."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.grid_tasks import (
    PROPOSAL_GRID_SIGNAL_QUALITIES,
    build_grid_tasks,
    format_signal_quality_key,
    format_signal_quality_label,
    parse_train_episodes_per_n,
    resolve_grid_signal_quality_list,
    resolve_max_horizon,
    resolve_train_episodes,
    task_artifacts_dir,
)


def test_format_signal_quality_key_distinguishes_055_and_06() -> None:
    assert format_signal_quality_key(0.55) == "0p55"
    assert format_signal_quality_key(0.65) == "0p65"
    assert format_signal_quality_key(0.6) == "0p6"
    assert format_signal_quality_key(0.8) == "0p8"


def test_format_signal_quality_label() -> None:
    assert format_signal_quality_label(0.55) == "0.55"
    assert format_signal_quality_label(0.6) == "0.6"


def test_build_grid_tasks_default_grid_has_forty_five_tasks() -> None:
    tasks = build_grid_tasks(
        seeds=[0, 1, 2, 3, 4],
        num_nodes_list=[10, 100, 1000],
        signal_quality_list=list(PROPOSAL_GRID_SIGNAL_QUALITIES),
    )
    assert len(tasks) == 45
    assert tasks[0].task_index == 0
    assert tasks[-1].task_index == 44


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


def test_q055_and_q06_have_distinct_artifact_dirs() -> None:
    tasks = build_grid_tasks(
        seeds=[0],
        num_nodes_list=[10],
        signal_quality_list=[0.55, 0.6],
    )
    root = Path("artifacts/training_metrics_fair/grid_runs")
    path_055 = task_artifacts_dir(root, tasks[0])
    path_06 = task_artifacts_dir(root, tasks[1])
    assert path_055 == Path("artifacts/training_metrics_fair/grid_runs/n_10/q_0p55")
    assert path_06 == Path("artifacts/training_metrics_fair/grid_runs/n_10/q_0p6")
    assert path_055 != path_06
    assert tasks[0].setting_key == "n_10/q_0.55"
    assert tasks[1].setting_key == "n_10/q_0.6"


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


def test_resolve_max_horizon_policy() -> None:
    assert (
        resolve_max_horizon(signal_quality=0.55, topology_name="ws_p_0.1_seed_0")
        == 100
    )
    assert resolve_max_horizon(signal_quality=0.55, topology_name="complete") == 50
    assert resolve_max_horizon(signal_quality=0.65, topology_name="ws_p_0.0_seed_1") == 50
    assert resolve_max_horizon(signal_quality=0.8, topology_name="complete") == 50


def test_resolve_grid_signal_quality_list_from_yaml() -> None:
    qualities = resolve_grid_signal_quality_list(
        run_config={"grid_signal_quality_list": [0.55, 0.65, 0.8]}
    )
    assert qualities == [0.55, 0.65, 0.8]


def test_parse_train_episodes_per_n_rejects_invalid_payload() -> None:
    with pytest.raises(ValueError, match="train_episodes_per_n"):
        parse_train_episodes_per_n(["not", "a", "dict"])
