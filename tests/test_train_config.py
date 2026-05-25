import argparse
import sys

import scripts.train as train_script


def test_resolve_run_config_yaml_and_cli_override(tmp_path) -> None:
    config_path = tmp_path / "run.yaml"
    config_path.write_text(
        "\n".join(
            [
                "seed: 11",
                "num_nodes: 50",
                "train_episodes: 123",
                "communication_mode: vector",
                "communication_dim: 4",
            ]
        ),
        encoding="utf-8",
    )

    args = argparse.Namespace(
        config=config_path,
        seed=3,
        num_nodes=None,
        graph_cache_dir=None,
        artifacts_dir=None,
        train_episodes=999,
        test_episodes=None,
        max_horizon=None,
        signal_quality=None,
        learning_rate=None,
        weight_decay=None,
        dropout=None,
        validation_episodes=None,
        validation_eval_every=None,
        device=None,
        hidden_dim=None,
        num_heads=None,
        communication_mode=None,
        communication_dim=None,
        disable_beta_fit=False,
    )
    resolved = train_script.resolve_run_config(args)
    assert resolved["seed"] == 3  # CLI overrides YAML
    assert resolved["num_nodes"] == 50  # YAML overrides defaults
    assert resolved["train_episodes"] == 999  # CLI overrides YAML
    assert resolved["communication_mode"] == "vector"
    assert resolved["communication_dim"] == 4
    assert resolved["device"] == "auto"


def test_parse_args_accepts_single_seed_and_num_nodes(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--seed",
            "7",
            "--num-nodes",
            "100",
            "--device",
            "cpu",
        ],
    )
    args = train_script.parse_args()
    assert args.seed == 7
    assert args.num_nodes == 100
    assert args.device == "cpu"
