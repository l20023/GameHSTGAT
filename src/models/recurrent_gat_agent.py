"""Recurrent GAT agent and shared sequential training loss."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class RecurrentGATAgent(nn.Module):
    """Run GAT message passing + GRU belief updates over a signal sequence."""

    def __init__(
        self,
        *,
        num_features_signal: int = 1,
        hidden_dim: int = 32,
        num_heads: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be > 0.")
        if num_heads <= 0:
            raise ValueError("num_heads must be > 0.")
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        if num_features_signal <= 0:
            raise ValueError("num_features_signal must be > 0.")

        self.hidden_dim = hidden_dim
        self.num_features_signal = num_features_signal
        self.num_heads = num_heads

        self.gat_conv = GATv2Conv(
            in_channels=hidden_dim,
            out_channels=hidden_dim // num_heads,
            heads=num_heads,
            concat=True,
            dropout=dropout,
        )
        self.gru_cell = nn.GRUCell(
            input_size=num_features_signal + hidden_dim,
            hidden_size=hidden_dim,
        )
        self.mlp_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(
        self,
        x_sequences: torch.Tensor,
        edge_index: torch.Tensor,
        max_horizon: int | None = None,
        initial_hidden_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x_sequences: Private signals shaped [num_nodes, horizon, num_features_signal].
            edge_index: PyG edge index [2, num_edges].
            max_horizon: Number of rolled-out rounds; defaults to x_sequences.shape[1].
            initial_hidden_state: Optional initial belief state [num_nodes, hidden_dim].

        Returns:
            Logits over time with shape [horizon, num_nodes, 2].
        """
        if x_sequences.ndim != 3:
            raise ValueError(
                "x_sequences must have shape [num_nodes, horizon, num_features_signal]."
            )
        if x_sequences.size(-1) != self.num_features_signal:
            raise ValueError(
                "x_sequences feature dimension must match num_features_signal."
            )
        if edge_index.ndim != 2 or edge_index.size(0) != 2:
            raise ValueError("edge_index must have shape [2, num_edges].")

        num_nodes, horizon, _ = x_sequences.shape
        if max_horizon is None:
            max_horizon = horizon
        if max_horizon <= 0:
            raise ValueError("max_horizon must be > 0.")
        if max_horizon > horizon:
            raise ValueError("max_horizon cannot exceed x_sequences horizon.")

        device = x_sequences.device
        if initial_hidden_state is None:
            hidden_state = torch.zeros((num_nodes, self.hidden_dim), device=device)
        else:
            if initial_hidden_state.shape != (num_nodes, self.hidden_dim):
                raise ValueError(
                    "initial_hidden_state must have shape [num_nodes, hidden_dim]."
                )
            hidden_state = initial_hidden_state

        logits_per_round: list[torch.Tensor] = []
        for t in range(max_horizon):
            # 1) Message passing on prior beliefs.
            neighbor_aggregation = F.relu(self.gat_conv(hidden_state, edge_index))
            # 2) Inject fresh private signal after message passing.
            current_private_signal = x_sequences[:, t, :]
            gru_input = torch.cat([current_private_signal, neighbor_aggregation], dim=-1)
            # 3) Belief-state update.
            hidden_state = self.gru_cell(gru_input, hidden_state)
            # 4) Time-local prediction for shared sequential supervision.
            logits_per_round.append(self.mlp_head(hidden_state))

        return torch.stack(logits_per_round, dim=0)


class SharedSequentialLoss(nn.Module):
    """Average cross-entropy across all rounds (and all nodes)."""

    def __init__(self) -> None:
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, all_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if all_logits.ndim != 3 or all_logits.size(-1) != 2:
            raise ValueError("all_logits must have shape [horizon, num_nodes, 2].")
        horizon, num_nodes, _ = all_logits.shape
        if horizon <= 0:
            raise ValueError("all_logits horizon must be > 0.")

        if targets.ndim == 0:
            targets = targets.expand(num_nodes)
        elif targets.ndim == 1 and targets.numel() == 1:
            targets = targets.expand(num_nodes)
        elif targets.ndim != 1 or targets.numel() != num_nodes:
            raise ValueError("targets must be scalar-like or shape [num_nodes].")

        targets = targets.to(device=all_logits.device, dtype=torch.long)
        total_loss = torch.zeros((), device=all_logits.device)
        for t in range(horizon):
            total_loss = total_loss + self.criterion(all_logits[t], targets)
        return total_loss / horizon


__all__ = ["RecurrentGATAgent", "SharedSequentialLoss"]
