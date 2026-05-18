"""Training entrypoint for one configured RGAT run."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.graph_generator import GraphGenerator
from src.config import load_yaml_config, merge_flat_config
from src.logging_utils import finish_wandb_run, init_wandb_run, log_condition_metrics
from src.training_pipeline import condition_result_to_dict, run_condition_experiment


PROPOSAL_WS_PROBS = [0.0, 0.1]
DEFAULT_CONFIG_PATH = Path("configs/default.yaml")
DEFAULT_RUN_CONFIG: dict[str, Any] = {
    "seed": 0,
    "num_nodes": 10,
    "wandb_project": "game-theory-project",
    "wandb_entity": "GameHSTGAT",
    "graph_cache_dir": "artifacts/graphs",
    "artifacts_dir": "artifacts/training_metrics",
    "train_episodes": 5000,
    "test_episodes": 1000,
    "max_horizon": 50,
    "signal_quality": 0.8,
    "learning_rate": 0.001,
    "hidden_dim": 32,
    "num_heads": 2,
    "disable_beta_fit": False,
}


def set_global_seed(seed: int) -> None:
    """Set all relevant RNG seeds for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_single_seed(
    seed: int,
    *,
    generator: GraphGenerator,
    num_nodes: int,
    graph_cache_dir: Path,
    artifacts_dir: Path,
    wandb_project: str,
    wandb_entity: str | None,
    train_episodes: int,
    test_episodes: int,
    max_horizon: int,
    signal_quality: float,
    hidden_dim: int,
    num_heads: int,
    learning_rate: float,
    disable_beta_fit: bool,
) -> dict:
    """Run full train/eval pipeline for one seed across all graph conditions."""
    set_global_seed(seed)
    graphs = generator.generate_and_store_experiment_graphs(
        storage_dir=graph_cache_dir,
        num_nodes_list=[num_nodes],
        ws_probs=PROPOSAL_WS_PROBS,
        seed=seed,
    )
    run_config = {
        "seed": seed,
        "num_nodes": num_nodes,
        "ws_probs": PROPOSAL_WS_PROBS,
        "train_episodes": train_episodes,
        "test_episodes": test_episodes,
        "max_horizon": max_horizon,
        "signal_quality": signal_quality,
        "hidden_dim": hidden_dim,
        "num_heads": num_heads,
        "learning_rate": learning_rate,
        "disable_beta_fit": disable_beta_fit,
    }
    wandb_run = init_wandb_run(
        project=wandb_project,
        entity=wandb_entity,
        seed=seed,
        config=run_config,
    )

    per_condition_metrics: dict[str, dict] = {}
    try:
        condition_index = 0
        for num_nodes, conditions in graphs.items():
            for condition_name, graph_data in conditions.items():
                key = f"n_{num_nodes}/{condition_name}"
                result = run_condition_experiment(
                    graph_data=graph_data,
                    train_episodes=train_episodes,
                    test_episodes=test_episodes,
                    max_horizon=max_horizon,
                    signal_quality=signal_quality,
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    learning_rate=learning_rate,
                    seed=seed,
                    disable_beta_fit=disable_beta_fit,
                )
                condition_metrics = condition_result_to_dict(result)
                per_condition_metrics[key] = condition_metrics
                log_condition_metrics(
                    run=wandb_run,
                    condition_key=key,
                    condition_index=condition_index,
                    metrics=condition_metrics,
                )
                condition_index += 1

        save_seed_metrics(
            artifacts_dir=artifacts_dir,
            seed=seed,
            metrics=per_condition_metrics,
        )
    finally:
        finish_wandb_run(wandb_run)

    return {
        "seed": seed,
        "num_conditions": len(per_condition_metrics),
        "mean_final_loss": float(
            np.mean(
                [
                    entry["train_loss_final"]
                    for entry in per_condition_metrics.values()
                    if entry["train_loss_final"] is not None
                ]
            )
        ),
    }


def save_seed_metrics(*, artifacts_dir: Path, seed: int, metrics: dict[str, dict]) -> None:
    """Persist one seed's condition metrics to JSON artifacts."""
    seed_dir = artifacts_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    output_path = seed_dir / "metrics.json"
    output_path.write_text(
        json.dumps(
            {
                "seed": seed,
                "conditions": metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one configured RGAT training job.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="One seed value for this run.",
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        default=None,
        help="One node count value for this run.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="Weights & Biases project name.",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Weights & Biases entity/team (optional).",
    )
    parser.add_argument(
        "--graph-cache-dir",
        type=Path,
        default=None,
        help="Directory where generated graph files are cached.",
    )
    parser.add_argument(
        "--train-episodes",
        type=int,
        default=None,
        help="Number of training episodes per graph condition.",
    )
    parser.add_argument(
        "--test-episodes",
        type=int,
        default=None,
        help="Number of evaluation episodes per graph condition.",
    )
    parser.add_argument(
        "--max-horizon",
        type=int,
        default=None,
        help="Maximum number of signal rounds for sequence modeling.",
    )
    parser.add_argument(
        "--signal-quality",
        type=float,
        default=None,
        help="Probability that a private signal matches the true state.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Optimizer learning rate.",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=None,
        help="Hidden state dimension for the recurrent GAT model.",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=None,
        help="Number of GAT attention heads.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=None,
        help="Directory for per-seed JSON training/evaluation metrics.",
    )
    parser.add_argument(
        "--disable-beta-fit",
        action="store_true",
        help="Skip exponential beta fitting and only log epsilon(t).",
    )
    return parser.parse_args()


def resolve_run_config(args: argparse.Namespace) -> dict[str, Any]:
    """Resolve final run config from defaults, YAML, and CLI overrides."""
    yaml_payload = load_yaml_config(args.config)
    cli_overrides: dict[str, Any] = {
        "seed": args.seed,
        "num_nodes": args.num_nodes,
        "wandb_project": args.wandb_project,
        "wandb_entity": args.wandb_entity,
        "graph_cache_dir": str(args.graph_cache_dir) if args.graph_cache_dir is not None else None,
        "artifacts_dir": str(args.artifacts_dir) if args.artifacts_dir is not None else None,
        "train_episodes": args.train_episodes,
        "test_episodes": args.test_episodes,
        "max_horizon": args.max_horizon,
        "signal_quality": args.signal_quality,
        "learning_rate": args.learning_rate,
        "hidden_dim": args.hidden_dim,
        "num_heads": args.num_heads,
        "disable_beta_fit": args.disable_beta_fit if args.disable_beta_fit else None,
    }
    config = merge_flat_config(
        defaults=DEFAULT_RUN_CONFIG,
        yaml_config=yaml_payload,
        cli_overrides=cli_overrides,
    )
    return config


def main() -> None:
    args = parse_args()
    run_config = resolve_run_config(args)
    generator = GraphGenerator()
    graph_cache_dir = Path(str(run_config["graph_cache_dir"]))
    artifacts_dir = Path(str(run_config["artifacts_dir"]))
    graph_cache_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    run_summary = run_single_seed(
        seed=int(run_config["seed"]),
        generator=generator,
        num_nodes=int(run_config["num_nodes"]),
        graph_cache_dir=graph_cache_dir,
        artifacts_dir=artifacts_dir,
        wandb_project=str(run_config["wandb_project"]),
        wandb_entity=(None if run_config["wandb_entity"] is None else str(run_config["wandb_entity"])),
        train_episodes=int(run_config["train_episodes"]),
        test_episodes=int(run_config["test_episodes"]),
        max_horizon=int(run_config["max_horizon"]),
        signal_quality=float(run_config["signal_quality"]),
        hidden_dim=int(run_config["hidden_dim"]),
        num_heads=int(run_config["num_heads"]),
        learning_rate=float(run_config["learning_rate"]),
        disable_beta_fit=bool(run_config["disable_beta_fit"]),
    )
    print(
        f"Finished seed={run_summary['seed']} "
        f"for {run_summary['num_conditions']} conditions "
        f"(mean final loss={run_summary['mean_final_loss']:.4f})."
    )
    print(f"Artifacts written to: {artifacts_dir}")


if __name__ == "__main__":
    main()
