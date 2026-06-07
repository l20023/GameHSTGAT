"""Tests for staging anchored_t2 evaluation plots in episode viewers."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import networkx as nx
import pytest

from src.episode_animation import save_interactive_episode_view
from src.learning_rate_plots import (
    condition_key_for_topology,
    resolve_anchored_t2_plot_path,
    stage_anchored_t2_plot_for_viewer,
)


def test_condition_key_for_ws_topology_includes_seed() -> None:
    assert condition_key_for_topology(num_nodes=10, topology="ws_p_0.1", seed=0) == (
        "n_10/ws_p_0.1_seed_0"
    )


def test_resolve_anchored_t2_plot_path_for_fair_n10() -> None:
    project_root = Path(__file__).resolve().parents[1]
    plot = resolve_anchored_t2_plot_path(
        communication_mode="fair_1bit",
        num_nodes=10,
        signal_quality=0.55,
        topology="complete",
        seed=0,
        project_root=project_root,
    )
    assert plot is not None
    assert plot.name.endswith("__anchored_t2.png")
    assert plot.exists()


def test_stage_anchored_t2_plot_for_viewer_copies_png(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    source = resolve_anchored_t2_plot_path(
        communication_mode="fair_1bit",
        num_nodes=10,
        signal_quality=0.55,
        topology="complete",
        seed=0,
        project_root=project_root,
    )
    assert source is not None
    html_path = tmp_path / "seed_0.html"
    html_path.write_text("<html></html>", encoding="utf-8")
    filename = stage_anchored_t2_plot_for_viewer(
        communication_mode="fair_1bit",
        num_nodes=10,
        signal_quality=0.55,
        topology="complete",
        seed=0,
        html_path=html_path,
        project_root=project_root,
    )
    assert filename == "seed_0__anchored_t2.png"
    copied = html_path.with_name(filename)
    assert copied.exists()
    assert copied.stat().st_size == source.stat().st_size


def test_save_interactive_episode_view_embeds_eval_plot(tmp_path: Path) -> None:
    from tests.test_episode_animation import _synthetic_trace

    trace = _synthetic_trace(num_nodes=4, max_horizon=5)
    graph = nx.complete_graph(4)
    out = save_interactive_episode_view(
        trace,
        graph,
        tmp_path / "view.html",
        eval_plot_filename="view__anchored_t2.png",
    )
    html = out.read_text(encoding="utf-8")
    assert "Test-set error decay (anchored at t=2)" in html
    assert 'src="view__anchored_t2.png"' in html
    assert 'class="eval-plot"' in html
