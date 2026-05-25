import json
from pathlib import Path

import scripts.run_grid as run_grid


def test_run_grid_experiments_aggregates_and_normalizes_conditions(
    tmp_path, monkeypatch
) -> None:
    def _fake_run_single_seed(**kwargs):
        seed = int(kwargs["seed"])
        num_nodes = int(kwargs["num_nodes"])
        artifacts_dir = Path(kwargs["artifacts_dir"])
        metrics_path = artifacts_dir / f"seed_{seed}" / "metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_payload = {
            "seed": seed,
            "conditions": {
                f"n_{num_nodes}/complete": {
                    "train_loss_final": 0.1,
                    "epsilon_series": [0.4, 0.3],
                    "beta_fit": {
                        "alpha": 0.5,
                        "beta": 0.2 + 0.01 * seed,
                        "epsilon_inf": 0.1,
                        "fit_success": True,
                        "method": "mock",
                    },
                    "beta_hst_max": 0.4,
                    "beta_gap": -0.2 + 0.01 * seed,
                    "exceeds_hst_bound": False,
                },
                f"n_{num_nodes}/ws_p_0.1_seed_{seed}": {
                    "train_loss_final": 0.1,
                    "epsilon_series": [0.4, 0.3],
                    "beta_fit": {
                        "alpha": 0.5,
                        "beta": 0.3 + 0.01 * seed,
                        "epsilon_inf": 0.1,
                        "fit_success": True,
                        "method": "mock",
                    },
                    "beta_hst_max": 0.4,
                    "beta_gap": -0.1 + 0.01 * seed,
                    "exceeds_hst_bound": False,
                },
            },
        }
        metrics_path.write_text(json.dumps(metrics_payload), encoding="utf-8")
        return {"seed": seed, "num_conditions": 2, "mean_final_loss": 0.1}

    monkeypatch.setattr(run_grid, "run_single_seed", _fake_run_single_seed)

    summary = run_grid.run_grid_experiments(
        seeds=[0, 1],
        num_nodes_list=[10],
        signal_quality_list=[0.6],
        graph_cache_dir=tmp_path / "graphs",
        artifacts_root=tmp_path / "grid_runs",
        wandb_project="game-theory-project",
        wandb_entity="GameHSTGAT",
        train_episodes=2,
        test_episodes=2,
        max_horizon=5,
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
        save_learning_rate_plots=False,
    )

    assert summary["num_runs"] == 2
    assert summary["num_condition_records"] == 4

    by_setting = summary["aggregates"]["by_setting"]
    assert "n_10/q_0.6" in by_setting
    assert by_setting["n_10/q_0.6"]["num_records"] == 4
    assert by_setting["n_10/q_0.6"]["mean_beta_gat"] is not None

    by_condition = summary["aggregates"]["by_condition"]
    assert "ws_p_0.1" in by_condition  # seed suffix should be normalized away
    assert by_condition["ws_p_0.1"]["num_records"] == 2
    assert "regime_classification" in summary
    assert "headline_label" in summary["regime_classification"]
    assert "consistent_with_equilibrium_bound" in summary["regime_classification"]


def test_run_grid_uses_adaptive_train_episodes_per_n(tmp_path, monkeypatch) -> None:
    """train_episodes_per_n should override train_episodes default for matching n."""
    captured: list[tuple[int, int]] = []

    def _fake_run_single_seed(**kwargs):
        seed = int(kwargs["seed"])
        num_nodes = int(kwargs["num_nodes"])
        captured.append((num_nodes, int(kwargs["train_episodes"])))
        artifacts_dir = Path(kwargs["artifacts_dir"])
        metrics_path = artifacts_dir / f"seed_{seed}" / "metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(
            json.dumps({"seed": seed, "conditions": {}}), encoding="utf-8"
        )
        return {"seed": seed, "num_conditions": 0, "mean_final_loss": 0.0}

    monkeypatch.setattr(run_grid, "run_single_seed", _fake_run_single_seed)

    run_grid.run_grid_experiments(
        seeds=[0],
        num_nodes_list=[10, 100, 1000],
        signal_quality_list=[0.6],
        graph_cache_dir=tmp_path / "graphs",
        artifacts_root=tmp_path / "grid_runs",
        wandb_project="game-theory-project",
        wandb_entity=None,
        train_episodes=999,  # default fallback (should NOT be used for any n in the map)
        test_episodes=2,
        max_horizon=3,
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
        disable_beta_fit=True,
        save_train_loss_history=False,
        save_epsilon_series=False,
        save_consensus_series=False,
        save_learning_rate_plots=False,
        train_episodes_per_n={10: 5000, 100: 7000, 1000: 10000},
    )

    captured_dict = dict(captured)
    assert captured_dict[10] == 5000
    assert captured_dict[100] == 7000
    assert captured_dict[1000] == 10000


def test_run_grid_falls_back_to_train_episodes_when_n_not_in_map(
    tmp_path, monkeypatch
) -> None:
    captured: list[tuple[int, int]] = []

    def _fake_run_single_seed(**kwargs):
        seed = int(kwargs["seed"])
        num_nodes = int(kwargs["num_nodes"])
        captured.append((num_nodes, int(kwargs["train_episodes"])))
        artifacts_dir = Path(kwargs["artifacts_dir"])
        metrics_path = artifacts_dir / f"seed_{seed}" / "metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(
            json.dumps({"seed": seed, "conditions": {}}), encoding="utf-8"
        )
        return {"seed": seed, "num_conditions": 0, "mean_final_loss": 0.0}

    monkeypatch.setattr(run_grid, "run_single_seed", _fake_run_single_seed)

    run_grid.run_grid_experiments(
        seeds=[0],
        num_nodes_list=[42],  # not in the map
        signal_quality_list=[0.6],
        graph_cache_dir=tmp_path / "graphs",
        artifacts_root=tmp_path / "grid_runs",
        wandb_project="game-theory-project",
        wandb_entity=None,
        train_episodes=777,
        test_episodes=2,
        max_horizon=3,
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
        disable_beta_fit=True,
        save_train_loss_history=False,
        save_epsilon_series=False,
        save_consensus_series=False,
        save_learning_rate_plots=False,
        train_episodes_per_n={10: 5000, 100: 7000, 1000: 10000},
    )
    assert captured[0] == (42, 777)
