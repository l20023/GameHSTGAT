"""Tests for per-signal anchored_t2 evaluation plots in episode viewers."""

from __future__ import annotations

from pathlib import Path

import networkx as nx

from src.episode_animation import save_interactive_episode_view
from src.learning_rate_plots import (
    condition_key_for_topology,
    resolve_anchored_t2_plot_path,
    stage_anchored_t2_plot_for_viewer,
    stage_per_episode_eval_plots,
)
from tests.test_episode_animation import _synthetic_trace


def test_stage_per_episode_eval_plots_writes_one_png_per_signal(tmp_path: Path) -> None:
    traces = [_synthetic_trace(num_nodes=4, max_horizon=8) for _ in range(2)]
    html_path = tmp_path / "seed_0.html"
    html_path.write_text("<html></html>", encoding="utf-8")
    mapping = stage_per_episode_eval_plots(
        traces=traces,
        episode_seeds=[4242, 4243],
        html_path=html_path,
        signal_quality=0.55,
        condition_key="n_4/complete",
    )
    assert mapping == {
        4242: "seed_0_ep4242__anchored_t2.png",
        4243: "seed_0_ep4243__anchored_t2.png",
    }
    for filename in mapping.values():
        plot = html_path.with_name(filename)
        assert plot.exists()
        assert plot.stat().st_size > 0


def test_stage_anchored_t2_plot_for_viewer_copies_summary(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
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
    source = resolve_anchored_t2_plot_path(
        communication_mode="fair_1bit",
        num_nodes=10,
        signal_quality=0.55,
        topology="complete",
        seed=0,
        project_root=project_root,
    )
    assert source is not None
    assert copied.exists()
    assert copied.stat().st_size == source.stat().st_size


def test_save_interactive_episode_view_embeds_per_signal_and_summary_plots(
    tmp_path: Path,
) -> None:
    traces = [_synthetic_trace(num_nodes=4, max_horizon=6) for _ in range(2)]
    graph = nx.complete_graph(4)
    out = save_interactive_episode_view(
        traces,
        graph,
        tmp_path / "view.html",
        episode_seeds=[4242, 4243],
        condition_key=condition_key_for_topology(
            num_nodes=4,
            topology="complete",
            seed=0,
        ),
        summary_plot_filename="view__anchored_t2.png",
    )
    html = out.read_text(encoding="utf-8")
    assert '"eval_plot":"view_ep4242__anchored_t2.png"' in html
    assert '"eval_plot":"view_ep4243__anchored_t2.png"' in html
    assert 'id="eval-plot"' in html
    assert 'id="summary-plot"' in html
    assert 'src="view__anchored_t2.png"' in html
    assert "Training evaluation summary" in html
    assert "function updateEvalPlot" in html
    assert (tmp_path / "view_ep4242__anchored_t2.png").exists()
    assert (tmp_path / "view_ep4243__anchored_t2.png").exists()
