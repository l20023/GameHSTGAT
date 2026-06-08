"""Cumulative majority-vote baseline aligned with the RGAT communication protocol."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from numpy.random import Generator


def build_neighbor_lists(num_nodes: int, edge_index: torch.Tensor) -> list[list[int]]:
    """Build sorted neighbor lists from an undirected edge_index."""
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E].")
    if int(edge_index.max().item()) >= num_nodes or int(edge_index.min().item()) < 0:
        raise ValueError("edge_index contains out-of-range node indices.")

    neighbors: list[set[int]] = [set() for _ in range(num_nodes)]
    sources = edge_index[0].tolist()
    targets = edge_index[1].tolist()
    for source, target in zip(sources, targets, strict=True):
        if source == target:
            continue
        neighbors[int(source)].add(int(target))
        neighbors[int(target)].add(int(source))
    return [sorted(node_neighbors) for node_neighbors in neighbors]


def build_adjacency_matrix(neighbors: list[list[int]]) -> np.ndarray:
    """Build a binary adjacency matrix from neighbor lists."""
    num_nodes = len(neighbors)
    adjacency = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    for node_idx, node_neighbors in enumerate(neighbors):
        for neighbor_idx in node_neighbors:
            adjacency[node_idx, neighbor_idx] = 1.0
    return adjacency


def _adjacency_from_neighbors(neighbors: list[list[int]]) -> np.ndarray:
    return build_adjacency_matrix(neighbors)


def rollout_majority_vote_episode(
    private_signals: torch.Tensor | np.ndarray,
    neighbors: list[list[int]],
    *,
    max_horizon: int,
    rng: Generator,
    adjacency: np.ndarray | None = None,
) -> torch.Tensor:
    """
    Roll out cumulative majority voting for one episode.

    Round t=0: predict and broadcast the private signal.
    Round t>=1: add neighbor broadcasts from t-1 to cumulative counts, add the
    current private signal, predict majority (random tie-break), broadcast.
    """
    private = np.asarray(private_signals[:max_horizon], dtype=np.int64)
    if private.ndim != 2:
        raise ValueError("private_signals must be 2D with shape [T, N].")
    horizon, num_nodes = private.shape
    if len(neighbors) != num_nodes:
        raise ValueError("neighbors length must match num_nodes.")
    if horizon <= 0:
        raise ValueError("max_horizon must be positive.")

    adjacency = (
        adjacency
        if adjacency is not None
        else _adjacency_from_neighbors(neighbors)
    )
    counts_zero = np.zeros(num_nodes, dtype=np.int64)
    counts_one = np.zeros(num_nodes, dtype=np.int64)
    predictions = np.zeros((horizon, num_nodes), dtype=np.int64)

    predictions[0] = private[0]
    for timestep in range(1, horizon):
        previous_broadcasts = predictions[timestep - 1].astype(np.float64)
        neighbor_ones = adjacency @ previous_broadcasts
        neighbor_zeros = adjacency.sum(axis=1).astype(np.int64) - neighbor_ones.astype(np.int64)
        counts_zero += neighbor_zeros
        counts_one += neighbor_ones.astype(np.int64)
        counts_zero += 1 - private[timestep]
        counts_one += private[timestep]

        round_predictions = np.empty(num_nodes, dtype=np.int64)
        greater_zero = counts_zero > counts_one
        greater_one = counts_one > counts_zero
        ties = ~(greater_zero | greater_one)
        round_predictions[greater_zero] = 0
        round_predictions[greater_one] = 1
        if np.any(ties):
            round_predictions[ties] = rng.integers(0, 2, size=int(ties.sum()))
        predictions[timestep] = round_predictions

    return torch.from_numpy(predictions)


__all__ = [
    "build_neighbor_lists",
    "rollout_majority_vote_episode",
]
