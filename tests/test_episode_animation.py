"""Tests for interactive episode viewer helpers."""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from src.episode_animation import (
    EpisodeTrace,
    circular_layout,
    compute_unanimous_consensus,
    load_rollouts_cache,
    node_radius_for_count,
    pack_correct_bitmap,
    rollouts_cache_is_valid,
    rollouts_cache_path,
    save_interactive_episode_view,
    save_rollouts_cache,
    viewer_edge_payload,
)


def _synthetic_trace(
    *,
    theta: int = 1,
    num_nodes: int = 5,
    max_horizon: int = 8,
) -> EpisodeTrace:
    predictions = np.zeros((max_horizon, num_nodes), dtype=np.int64)
    for t in range(max_horizon):
        if t < 4:
            predictions[t, :] = np.arange(num_nodes) % 2
        else:
            predictions[t, :] = 1
    correct = predictions == theta
    return EpisodeTrace(
        theta=theta,
        num_nodes=num_nodes,
        max_horizon=max_horizon,
        topology="complete",
        signal_quality=0.55,
        private_signals=np.zeros_like(predictions),
        predictions=predictions,
        probs_one=predictions.astype(float),
        comm_messages=predictions.astype(float),
        correct=correct,
    )


def test_circular_layout_places_nodes_on_unit_circle() -> None:
    positions = circular_layout(6)
    assert len(positions) == 6
    radii = [np.hypot(x, y) for x, y in positions.values()]
    assert all(abs(r - 1.0) < 1e-9 for r in radii)


def test_node_radius_scales_down_for_large_graphs() -> None:
    assert node_radius_for_count(10) > node_radius_for_count(1000)


def test_compute_unanimous_consensus_finds_first_round() -> None:
    trace = _synthetic_trace(theta=1, num_nodes=4, max_horizon=6)
    consensus = compute_unanimous_consensus(trace)
    assert consensus["first_unanimous_t"] == 5
    assert consensus["first_unanimous_correct"] is True
    assert consensus["per_round"][3]["unanimous"] is False
    assert consensus["per_round"][4]["unanimous"] is True


def test_compute_unanimous_consensus_wrong_unanimous() -> None:
    trace = _synthetic_trace(theta=0, num_nodes=4, max_horizon=6)
    consensus = compute_unanimous_consensus(trace)
    assert consensus["first_unanimous_t"] == 5
    assert consensus["first_unanimous_correct"] is False


def test_save_interactive_episode_view_writes_html(tmp_path: Path) -> None:
    trace = _synthetic_trace(num_nodes=4, max_horizon=5)
    graph = nx.complete_graph(4)
    out = save_interactive_episode_view(trace, graph, tmp_path / "view.html")
    html = out.read_text(encoding="utf-8")
    assert out.exists()
    assert "round-slider" in html
    assert '"edges"' in html
    assert "first_unanimous_t" in html
    assert "Unanimous consensus" in html
    assert '"num_nodes":4' in html
    assert '"edge_hint":"complete"' in html
    assert '"episodes"' in html
    assert '"correct_packed"' in html
    assert '"correct_shape"' in html
    assert "function unpackCorrect" in html
    assert "edge-canvas" in html
    assert "function drawEdges" in html
    assert 'id="prev-btn"' in html
    assert 'id="next-btn"' in html
    assert 'id="new-signal-btn"' in html


def test_viewer_edge_payload_omits_dense_complete_edges() -> None:
    graph = nx.complete_graph(100)
    edges, hint, count = viewer_edge_payload(graph, topology="complete")
    assert edges == []
    assert hint == "complete"
    assert count == 4950


def test_viewer_edge_payload_keeps_ws_edges() -> None:
    graph = nx.cycle_graph(1000)
    edges, hint, count = viewer_edge_payload(graph, topology="watts_strogatz")
    assert hint == "graph"
    assert count == 1000
    assert len(edges) == 1000


def test_pack_correct_bitmap_roundtrip() -> None:
    correct = np.array(
        [
            [True, False, True],
            [False, False, True],
        ],
        dtype=bool,
    )
    packed = pack_correct_bitmap(correct)
    flat = np.unpackbits(np.frombuffer(__import__("base64").b64decode(packed), dtype=np.uint8))[
        : correct.size
    ]
    assert flat.reshape(correct.shape).astype(bool).tolist() == correct.tolist()


def test_rollouts_cache_roundtrip(tmp_path: Path) -> None:
    trace = _synthetic_trace(num_nodes=4, max_horizon=5)
    html_path = tmp_path / "view.html"
    cache_path = rollouts_cache_path(html_path)
    episode_seeds = [4242, 4243]
    save_rollouts_cache(cache_path, [trace, trace], episode_seeds=episode_seeds)
    assert rollouts_cache_is_valid(
        cache_path,
        episode_seeds=episode_seeds,
        num_nodes=4,
        max_horizon=5,
        signal_quality=0.55,
        topology="complete",
    )
    loaded, seeds = load_rollouts_cache(cache_path)
    assert seeds == episode_seeds
    assert len(loaded) == 2
    assert np.array_equal(loaded[0].correct, trace.correct)
    assert np.array_equal(loaded[0].predictions, trace.predictions)
    assert compute_unanimous_consensus(loaded[0]) == compute_unanimous_consensus(trace)


def test_large_complete_html_stays_small(tmp_path: Path) -> None:
    trace = _synthetic_trace(num_nodes=100, max_horizon=5)
    graph = nx.complete_graph(100)
    out = save_interactive_episode_view(trace, graph, tmp_path / "big.html")
    html = out.read_text(encoding="utf-8")
    assert '"edge_count":4950' in html
    assert len(html) < 200_000
