import json
from pathlib import Path

import torch
from torch_geometric.data import Data

import scripts.train as train_script


class _DummyRun:
    def __init__(self) -> None:
        self.logs: list[dict] = []
        self.finished = False

    def log(self, payload: dict) -> None:
        self.logs.append(payload)

    def finish(self) -> None:
        self.finished = True


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


def test_run_single_seed_logs_wandb_and_keeps_local_artifacts(tmp_path, monkeypatch) -> None:
    dummy_run = _DummyRun()
    captured: dict[str, object] = {}

    def _fake_init_wandb_run(**kwargs):
        captured["init_kwargs"] = kwargs
        return dummy_run

    monkeypatch.setattr(
        train_script,
        "init_wandb_run",
        _fake_init_wandb_run,
    )
    monkeypatch.setattr(train_script, "finish_wandb_run", lambda run: run.finish())
    monkeypatch.setattr(
        train_script,
        "run_condition_experiment",
        lambda **kwargs: {"dummy": kwargs["graph_data"].num_nodes},
    )
    monkeypatch.setattr(
        train_script,
        "condition_result_to_dict",
        lambda _: {
            "train_loss_final": 0.123,
            "epsilon_series": [0.4, 0.3, 0.2],
            "beta_fit": {
                "alpha": 0.5,
                "beta": 0.3,
                "epsilon_inf": 0.1,
                "fit_success": True,
                "method": "mock",
            },
        },
    )

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
        wandb_project="unit-test-project",
        wandb_entity=None,
        train_episodes=1,
        test_episodes=1,
        max_horizon=3,
        signal_quality=0.8,
        hidden_dim=8,
        num_heads=2,
        learning_rate=0.001,
        disable_beta_fit=False,
    )

    assert fake_generator.ws_probs_seen == train_script.PROPOSAL_WS_PROBS
    assert summary["num_conditions"] == 3
    assert summary["mean_final_loss"] == 0.123
    assert dummy_run.finished is True
    assert len(dummy_run.logs) == 3

    init_kwargs = captured["init_kwargs"]
    assert isinstance(init_kwargs, dict)
    assert init_kwargs["project"] == "unit-test-project"
    assert init_kwargs["config"]["ws_probs"] == train_script.PROPOSAL_WS_PROBS

    metrics_path = tmp_path / "metrics" / "seed_0" / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["seed"] == 0
    assert len(metrics["conditions"]) == 3
