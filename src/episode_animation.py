"""Animate per-node RGAT predictions on one social-learning episode."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Patch
from torch_geometric.data import Data

from src.models.recurrent_gat_agent import RecurrentGATAgent
from src.signal_generator import PrivateSignalGenerator
from src.training_pipeline import _episode_to_model_inputs


@dataclass(frozen=True)
class EpisodeTrace:
    """Per-round node-level rollout for visualization."""

    theta: int
    num_nodes: int
    max_horizon: int
    topology: str
    signal_quality: float
    private_signals: np.ndarray
    predictions: np.ndarray
    probs_one: np.ndarray
    comm_messages: np.ndarray
    correct: np.ndarray

    @property
    def error_rates(self) -> np.ndarray:
        return 1.0 - self.correct.mean(axis=1)

    @property
    def agreement_fractions(self) -> np.ndarray:
        counts = []
        for t in range(self.max_horizon):
            preds = self.predictions[t]
            _, freq = np.unique(preds, return_counts=True)
            counts.append(float(freq.max()) / self.num_nodes)
        return np.asarray(counts)


def pyg_to_networkx(graph_data: Data) -> nx.Graph:
    """Convert undirected PyG graph to NetworkX (deduplicate directed edges)."""
    edge_index = graph_data.edge_index.cpu().numpy()
    graph = nx.Graph()
    graph.add_nodes_from(range(int(graph_data.num_nodes)))
    seen: set[tuple[int, int]] = set()
    for src, dst in edge_index.T:
        u, v = int(src), int(dst)
        key = (min(u, v), max(u, v))
        if key not in seen:
            seen.add(key)
            graph.add_edge(u, v)
    return graph


def layout_positions(graph: nx.Graph, *, topology: str, seed: int) -> dict[int, np.ndarray]:
    """Stable node positions for animation frames."""
    if topology == "complete":
        positions = nx.circular_layout(graph)
    else:
        positions = nx.spring_layout(graph, seed=seed, k=1.2 / np.sqrt(graph.number_of_nodes()))
    return {node: np.asarray(coords, dtype=float) for node, coords in positions.items()}


def rollout_episode(
    *,
    model: RecurrentGATAgent,
    graph_data: Data,
    episode: dict[str, int | torch.Tensor],
    device: torch.device,
    signal_quality: float,
    topology: str,
) -> EpisodeTrace:
    """Run one episode and collect per-node predictions round by round."""
    num_nodes = int(graph_data.num_nodes)
    x_sequences, _ = _episode_to_model_inputs(
        episode, num_nodes=num_nodes, device=device
    )
    horizon = int(x_sequences.shape[1])
    theta = int(episode["theta"])
    private_signals = episode["private_signals"].cpu().numpy()

    was_training = model.training
    model.eval()
    with torch.no_grad():
        logits = model(
            x_sequences,
            graph_data.edge_index,
            max_horizon=horizon,
        )
        probs = F.softmax(logits, dim=-1)
        predictions = logits.argmax(dim=-1).cpu().numpy()
        probs_one = probs[..., 1].cpu().numpy()
        comm_messages = predictions.astype(np.float64)

    if was_training:
        model.train()

    correct = predictions == theta
    return EpisodeTrace(
        theta=theta,
        num_nodes=num_nodes,
        max_horizon=horizon,
        topology=topology,
        signal_quality=signal_quality,
        private_signals=private_signals,
        predictions=predictions,
        probs_one=probs_one,
        comm_messages=comm_messages,
        correct=correct,
    )


def sample_episode(
    *,
    num_nodes: int,
    max_horizon: int,
    signal_quality: float,
    seed: int,
) -> dict[str, int | torch.Tensor]:
    generator = PrivateSignalGenerator(signal_quality=signal_quality, default_seed=seed)
    return generator.generate_episode(
        num_nodes=num_nodes,
        max_horizon=max_horizon,
        seed=seed,
    )


def save_episode_animation(
    trace: EpisodeTrace,
    graph: nx.Graph,
    output_path: str | Path,
    *,
    layout_seed: int = 0,
    fps: int = 2,
    dpi: int = 120,
    frame_step: int = 1,
) -> Path:
    """Write a GIF showing node predictions evolving over rounds."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    positions = layout_positions(graph, topology=trace.topology, seed=layout_seed)

    pred_colors = {0: "#4C78A8", 1: "#F58518"}
    signal_colors = {0: "#9ECAE9", 1: "#FDAE6B"}

    fig, ax = plt.subplots(figsize=(8.5, 7.0))
    fig.subplots_adjust(top=0.86)

    timesteps = list(range(0, trace.max_horizon, max(1, frame_step)))
    if timesteps[-1] != trace.max_horizon - 1:
        timesteps.append(trace.max_horizon - 1)

    def _draw_frame(t: int) -> None:
        ax.clear()
        ax.set_aspect("equal")
        ax.axis("off")

        nx.draw_networkx_edges(
            graph,
            positions,
            ax=ax,
            alpha=0.25,
            width=1.0,
            edge_color="#BDBDBD",
        )
        for node in range(trace.num_nodes):
            x, y = positions[node]
            pred = int(trace.predictions[t, node])
            signal = int(trace.private_signals[t, node])
            is_correct = bool(trace.correct[t, node])
            prob = float(trace.probs_one[t, node])

            nx.draw_networkx_nodes(
                graph,
                positions,
                nodelist=[node],
                node_color=[pred_colors[pred]],
                edgecolors="#2CA02C" if is_correct else "#D62728",
                linewidths=2.8,
                node_size=1100,
                ax=ax,
            )
            ax.text(
                x,
                y,
                f"{node}\nŷ={pred} ({prob:.2f})\ns={signal}",
                ha="center",
                va="center",
                fontsize=8,
                color="white",
                fontweight="bold",
            )
            comm = int(trace.comm_messages[t, node])
            ax.scatter(
                [x + 0.11],
                [y + 0.11],
                s=180,
                c=signal_colors[comm],
                edgecolors="black",
                linewidths=0.6,
                zorder=5,
                marker="s",
            )

        err = 1.0 - trace.correct[t].mean()
        agree = trace.agreement_fractions[t]
        unanimous = trace.predictions[t].min() == trace.predictions[t].max()
        title = (
            f"Round {t + 1}/{trace.max_horizon}  |  "
            f"θ={trace.theta}  |  q={trace.signal_quality:.2f}  |  "
            f"error={err:.0%}  |  agreement={agree:.0%}"
        )
        if unanimous:
            title += "  |  unanimous"
        ax.set_title(title, fontsize=11, pad=8)
        legend_handles = [
            Patch(facecolor=pred_colors[0], edgecolor="black", label="prediction ŷ=0"),
            Patch(facecolor=pred_colors[1], edgecolor="black", label="prediction ŷ=1"),
            Patch(facecolor="white", edgecolor="#2CA02C", linewidth=2, label="correct vs θ"),
            Patch(facecolor="white", edgecolor="#D62728", linewidth=2, label="wrong vs θ"),
            Patch(facecolor=signal_colors[0], edgecolor="black", label="comm. bit (square)"),
        ]
        ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(0.0, -0.02), fontsize=8)

    def _update(frame_idx: int) -> None:
        _draw_frame(timesteps[frame_idx])

    anim = FuncAnimation(fig, _update, frames=len(timesteps), interval=1000 // fps)
    writer = PillowWriter(fps=fps)
    anim.save(output, writer=writer, dpi=dpi)
    plt.close(fig)
    return output


def save_checkpoint(model: RecurrentGATAgent, path: str | Path, metadata: dict[str, Any]) -> Path:
    """Persist model weights and run metadata for animation reuse."""
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "metadata": metadata,
    }
    torch.save(payload, checkpoint_path)
    return checkpoint_path


def load_checkpoint(
    path: str | Path,
    *,
    device: torch.device,
) -> tuple[RecurrentGATAgent, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    metadata = dict(payload["metadata"])
    model = RecurrentGATAgent(
        num_features_signal=1,
        hidden_dim=int(metadata["hidden_dim"]),
        num_heads=int(metadata["num_heads"]),
        dropout=float(metadata.get("dropout", 0.0)),
        communication_mode=str(metadata["communication_mode"]),
        communication_dim=metadata.get("communication_dim"),
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, metadata


__all__ = [
    "EpisodeTrace",
    "layout_positions",
    "load_checkpoint",
    "pyg_to_networkx",
    "rollout_episode",
    "sample_episode",
    "save_checkpoint",
    "save_episode_animation",
]
