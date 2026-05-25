import json
from pathlib import Path

import torch
from torch_geometric.data import Data

import scripts.train as train_script


class _DummyRun:
    def __init__(self) -> None:
        self.logs: list[tuple[dict, int | None]] = []
        self.finished = False

    def log(self, payload: dict, step: int | None = None) -> None:
        self.logs.append((payload, step))

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
        wandb_project="unit-test-project",
        wandb_entity=None,
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
    assert dummy_run.finished is True
    summary_logs = [entry for entry in dummy_run.logs if entry[1] is None]
    train_loss_logs = [entry for entry in dummy_run.logs if entry[1] is not None]
    assert len(summary_logs) == 3
    assert len(train_loss_logs) == 6
    assert train_loss_logs[0][0]["n_10/complete/train_loss"] == 0.9
    assert train_loss_logs[1][0]["n_10/complete/train_loss"] == 0.123

    init_kwargs = captured["init_kwargs"]
    assert isinstance(init_kwargs, dict)
    assert init_kwargs["project"] == "unit-test-project"
    assert init_kwargs["config"]["ws_probs"] == train_script.PROPOSAL_WS_PROBS
    assert init_kwargs["config"]["communication_mode"] == "fair_1bit"
    assert init_kwargs["config"]["communication_dim"] is None

    metrics_path = tmp_path / "metrics" / "seed_0" / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["seed"] == 0
    assert len(metrics["conditions"]) == 3
    complete_metrics = metrics["conditions"]["n_10/complete"]
    assert complete_metrics["learning_rate_plot"].endswith("__anchored_t0.png")
    assert complete_metrics["learning_rate_plot_anchored_t0"].endswith("__anchored_t0.png")
