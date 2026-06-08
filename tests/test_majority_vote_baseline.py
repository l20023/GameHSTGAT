"""Tests for cumulative majority-vote baseline rollout and evaluation."""

from __future__ import annotations

import numpy as np
import torch

from src.graph_generator import GraphGenerator
from src.majority_vote_baseline import build_neighbor_lists, rollout_majority_vote_episode
from src.training_pipeline import fit_beta_from_epsilon, run_majority_baseline_condition


def _complete_neighbors(num_nodes: int) -> list[list[int]]:
    graph = GraphGenerator().generate_complete(num_nodes)
    return build_neighbor_lists(num_nodes, graph.edge_index)


def test_t0_predicts_private_signal() -> None:
    private = torch.tensor([[1, 0, 1], [0, 1, 0], [1, 1, 0]], dtype=torch.int64)
    rng = np.random.default_rng(0)
    predictions = rollout_majority_vote_episode(
        private,
        _complete_neighbors(3),
        max_horizon=3,
        rng=rng,
    )
    assert predictions[0].tolist() == [1, 0, 1]


def test_complete_graph_identical_private_signals_converge() -> None:
    private = torch.ones((4, 3), dtype=torch.int64)
    rng = np.random.default_rng(0)
    predictions = rollout_majority_vote_episode(
        private,
        _complete_neighbors(3),
        max_horizon=4,
        rng=rng,
    )
    assert torch.all(predictions[1:] == 1).item()


def test_tie_break_is_seeded() -> None:
    # Single neighbor: one 0 vote + private 1 => tie at t=1.
    neighbors = [[1], [0], []]
    private = torch.tensor([[0, 0, 0], [1, 1, 1]], dtype=torch.int64)
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)
    pred_a = rollout_majority_vote_episode(private, neighbors, max_horizon=2, rng=rng_a)
    pred_b = rollout_majority_vote_episode(private, neighbors, max_horizon=2, rng=rng_b)
    assert pred_a[1, 0].item() == pred_b[1, 0].item()
    assert pred_a[1, 0].item() in (0, 1)


def test_cumulative_neighbor_history_not_only_last_message() -> None:
    private = torch.ones((4, 3), dtype=torch.int64)
    private[3, :] = 0
    rng = np.random.default_rng(0)
    predictions = rollout_majority_vote_episode(
        private,
        _complete_neighbors(3),
        max_horizon=4,
        rng=rng,
    )
    # Three rounds of unanimous 1-broadcasts outweigh a single final private 0.
    assert torch.all(predictions[3] == 1).item()


def test_integration_smoke_fit_beta() -> None:
    graph = GraphGenerator().generate_complete(8)
    result = run_majority_baseline_condition(
        graph_data=graph,
        test_episodes=20,
        max_horizon=15,
        signal_quality=0.8,
        seed=0,
    )
    assert len(result.epsilon_series) == 15
    assert result.train_loss_history == []
    beta = result.beta_fit.get("beta")
    assert isinstance(beta, (int, float))
    assert np.isfinite(float(beta))
    refit = fit_beta_from_epsilon(result.epsilon_series)
    assert refit.get("fit_success") == result.beta_fit.get("fit_success")
