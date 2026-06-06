"""Tests for the training entrypoint (local artifacts only)."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch_geometric.data import Data

import scripts.train as train_script


class _FakeGenerator:
    def __init__(self, graph_data: dict[int, dict[str, Data]]) -> None:
        self.graph_data = graph_data
        self.ws_probs_seen: list[float] | None = None

    def generate_and_store_experiment_graphs(
        self,
        *,
        storage_dir: Path,
        num_nodes_list: list[int],
        ws_probs: list[float] | None = None,
        k: int | None = None,
        seed: int | None = None,
    ) -> dict[int, dict[str, Data]]:
        del storage_dir, num_nodes_list, k, seed
        self.ws_probs_seen = ws_probs
        return self.graph_data


def _make_graph(num_nodes: int) -> Data:
    adjacency = torch.ones((num_nodes, num_nodes), dtype=torch.long) - torch.eye(
        num_nodes, dtype=torch.long
    )
    return Data(edge_index=adjacency.nonzero(as_tuple=False).t().contiguous(), num_nodes=num_nodes)


def test_run_single_seed_writes_local_artifacts(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        train_script,
        "run_condition_experiment",
        lambda **kwargs: type(
            "Result",
            (),
            {
                "train_loss_history": [0.9, 0.123],
                "epsilon_series": [0.4, 0.3, 0.2],
                "beta_fit": {
                    "alpha": 0.5,
                    "beta": 0.3,
                    "epsilon_inf": 0.1,
                    "fit_success": True,
                    "method": "mock",
                },
                "beta_hst_max": 0.81,
                "beta_gap": -0.51,
                "exceeds_hst_bound": False,
                "convergence_warning": False,
                "consensus": {
                    "unanimous": {"fraction_episodes_reach_consensus": 0.5},
                    "majority": {"fraction_episodes_reach_consensus": 0.8},
                },
                "train_loss_running_mean_final": 0.123,
                "best_running_mean_loss": 0.1,
                "best_episode_idx": 0,
            },
        )(),
    )
    monkeypatch.setattr(
        train_script,
        "condition_result_to_dict",
        lambda result, **kwargs: {
            "train_loss_final": result.train_loss_history[-1],
            "beta_fit": result.beta_fit,
            "beta_hst_max": result.beta_hst_max,
            "beta_gap": result.beta_gap,
            "exceeds_hst_bound": result.exceeds_hst_bound,
        },
    )
    monkeypatch.setattr(train_script, "save_learning_rate_plot", lambda **kwargs: kwargs["output_path"])
    monkeypatch.setattr(train_script, "save_train_loss_plot", lambda **kwargs: kwargs["output_path"])

    graph_data = {
        10: {
            "complete": _make_graph(10),
            "ws_p_0.0_seed_0": _make_graph(10),
            "ws_p_0.1_seed_0": _make_graph(10),
        }
    }
    fake_generator = _FakeGenerator(graph_data=graph_data)

    summary = train_script.run_single_seed(
        seed=0,
        generator=fake_generator,  # type: ignore[arg-type]
        num_nodes=10,
        graph_cache_dir=tmp_path / "graphs",
        artifacts_dir=tmp_path / "metrics",
        train_episodes=1,
        test_episodes=1,
        max_horizon=3,
        signal_quality=0.8,
        hidden_dim=8,
        num_heads=2,
        communication_mode="fair_1bit",
        communication_dim=None,
        learning_rate=0.001,
        weight_decay=0.0,
        dropout=0.0,
        validation_episodes=0,
        validation_eval_every=100,
        device="cpu",
        disable_beta_fit=False,
        save_train_loss_history=False,
        save_epsilon_series=False,
        save_consensus_series=False,
        save_learning_rate_plots=True,
    )

    assert fake_generator.ws_probs_seen == train_script.PROPOSAL_WS_PROBS
    assert summary["num_conditions"] == 3
    assert summary["mean_final_loss"] == 0.123

    metrics_path = tmp_path / "metrics" / "seed_0" / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["seed"] == 0
    assert metrics["signal_quality"] == 0.8
    assert metrics["num_nodes"] == 10
    assert len(metrics["conditions"]) == 3
    complete_metrics = metrics["conditions"]["n_10/complete"]
    assert complete_metrics["learning_rate_plot"].endswith("__anchored_t2.png")
    assert complete_metrics["learning_rate_plot_anchored_t2"].endswith("__anchored_t2.png")
