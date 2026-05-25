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
from src.grid_tasks import parse_train_episodes_per_n, resolve_train_episodes
from src.learning_rate_plots import learning_rate_plot_path, save_learning_rate_plot
from src.train_loss_plots import save_train_loss_plot, train_loss_plot_path
from src.training_pipeline import (
    condition_result_to_dict,
    resolve_runtime_device,
    run_condition_experiment,
)


PROPOSAL_WS_PROBS = [0.0, 0.1]
DEFAULT_CONFIG_PATH = Path("configs/default.yaml")
DEFAULT_RUN_CONFIG: dict[str, Any] = {
    "seed": 0,
    "num_nodes": 10,
    "graph_cache_dir": "artifacts/graphs",
    "artifacts_dir": "artifacts/training_metrics",
    "train_episodes": 5000,
    "train_episodes_per_n": None,
    "test_episodes": 1000,
    "max_horizon": 100,
    "signal_quality": 0.8,
    "learning_rate": 0.001,
    "weight_decay": 1e-4,
    "dropout": 0.1,
    "device": "auto",
    "hidden_dim": 64,
    "num_heads": 2,
    "communication_mode": "fair_1bit",
    "communication_dim": None,
    "validation_episodes": 50,
    "validation_eval_every": 100,
    "disable_beta_fit": False,
    "save_train_loss_history": False,
    "save_epsilon_series": True,
    "save_consensus_series": False,
    "save_learning_rate_plots": True,
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
    train_episodes: int,
    test_episodes: int,
    max_horizon: int,
    signal_quality: float,
    hidden_dim: int,
    num_heads: int,
    communication_mode: str,
    communication_dim: int | None,
    learning_rate: float,
    weight_decay: float,
    dropout: float,
    validation_episodes: int,
    validation_eval_every: int,
    device: str,
    disable_beta_fit: bool,
    save_train_loss_history: bool,
    save_epsilon_series: bool,
    save_consensus_series: bool,
    save_learning_rate_plots: bool,
) -> dict:
    """Run full train/eval pipeline for one seed across all graph conditions."""
    set_global_seed(seed)
    resolved_device = resolve_runtime_device(device)
    graphs = generator.generate_and_store_experiment_graphs(
        storage_dir=graph_cache_dir,
        num_nodes_list=[num_nodes],
        ws_probs=PROPOSAL_WS_PROBS,
        seed=seed,
    )
    per_condition_metrics: dict[str, dict] = {}
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
                communication_mode=communication_mode,
                communication_dim=communication_dim,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                dropout=dropout,
                validation_episodes=validation_episodes,
                validation_eval_every=validation_eval_every,
                seed=seed,
                device=resolved_device,
                disable_beta_fit=disable_beta_fit,
                include_consensus_series=save_consensus_series,
            )
            condition_metrics = condition_result_to_dict(
                result,
                save_train_loss_history=save_train_loss_history,
                save_epsilon_series=save_epsilon_series,
                save_consensus_series=save_consensus_series,
            )
            if save_learning_rate_plots:
                anchored_t1_plot_path = learning_rate_plot_path(
                    artifacts_dir=artifacts_dir,
                    seed=seed,
                    condition_key=key,
                    plot_variant="anchored_t1",
                )
                save_learning_rate_plot(
                    output_path=anchored_t1_plot_path,
                    epsilon_series=result.epsilon_series,
                    beta_fit=result.beta_fit,
                    beta_hst_max=result.beta_hst_max,
                    condition_key=key,
                    signal_quality=signal_quality,
                    beta_gap=result.beta_gap,
                    exceeds_hst_bound=result.exceeds_hst_bound,
                    convergence_warning=result.convergence_warning,
                    plot_variant="anchored_t1",
                )
                condition_metrics["learning_rate_plot"] = str(anchored_t1_plot_path)
                condition_metrics["learning_rate_plot_anchored_t1"] = str(anchored_t1_plot_path)
                train_loss_path = train_loss_plot_path(
                    artifacts_dir=artifacts_dir,
                    seed=seed,
                    condition_key=key,
                )
                save_train_loss_plot(
                    output_path=train_loss_path,
                    train_loss_history=result.train_loss_history,
                    condition_key=key,
                )
                condition_metrics["train_loss_plot"] = str(train_loss_path)
            per_condition_metrics[key] = condition_metrics

    save_seed_metrics(
        artifacts_dir=artifacts_dir,
        seed=seed,
        metrics=per_condition_metrics,
        signal_quality=signal_quality,
        num_nodes=num_nodes,
    )

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


def save_seed_metrics(
    *,
    artifacts_dir: Path,
    seed: int,
    metrics: dict[str, dict],
    signal_quality: float | None = None,
    num_nodes: int | None = None,
) -> None:
    """Persist one seed's condition metrics to JSON artifacts."""
    seed_dir = artifacts_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    output_path = seed_dir / "metrics.json"
    payload: dict[str, Any] = {
        "seed": seed,
        "conditions": metrics,
    }
    if signal_quality is not None:
        payload["signal_quality"] = signal_quality
    if num_nodes is not None:
        payload["num_nodes"] = num_nodes
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


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
        "--weight-decay",
        type=float,
        default=None,
        help="Adam weight decay (L2 regularization).",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="Dropout probability for GAT and head.",
    )
    parser.add_argument(
        "--validation-episodes",
        type=int,
        default=None,
        help="Held-out validation episodes for best-checkpoint selection (0 disables).",
    )
    parser.add_argument(
        "--validation-eval-every",
        type=int,
        default=None,
        help="Evaluate validation pool every N training episodes.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda", "mps"],
        default=None,
        help="Runtime device selection (auto, cpu, cuda, mps).",
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
        "--communication-mode",
        type=str,
        choices=["fair_1bit", "vector"],
        default=None,
        help="Communication channel mode: fair 1-bit bottleneck or vector ablation.",
    )
    parser.add_argument(
        "--communication-dim",
        type=int,
        default=None,
        help="Visible communication dimension (used in vector mode).",
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
    parser.add_argument(
        "--save-train-loss-history",
        action="store_true",
        help="Persist full per-episode train loss arrays in metrics.json.",
    )
    parser.add_argument(
        "--save-epsilon-series",
        action="store_true",
        help="Persist full epsilon(t) arrays in metrics.json.",
    )
    parser.add_argument(
        "--save-consensus-series",
        action="store_true",
        help="Persist consensus time-series arrays in metrics.json.",
    )
    parser.add_argument(
        "--no-learning-rate-plots",
        action="store_true",
        help="Disable PNG learning-rate plots under seed_<id>/plots/.",
    )
    return parser.parse_args()


def resolve_run_config(args: argparse.Namespace) -> dict[str, Any]:
    """Resolve final run config from defaults, YAML, and CLI overrides."""
    yaml_payload = load_yaml_config(args.config)
    cli_overrides: dict[str, Any] = {
        "seed": args.seed,
        "num_nodes": args.num_nodes,
        "graph_cache_dir": str(args.graph_cache_dir) if args.graph_cache_dir is not None else None,
        "artifacts_dir": str(args.artifacts_dir) if args.artifacts_dir is not None else None,
        "train_episodes": args.train_episodes,
        "test_episodes": args.test_episodes,
        "max_horizon": args.max_horizon,
        "signal_quality": args.signal_quality,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "validation_episodes": args.validation_episodes,
        "validation_eval_every": args.validation_eval_every,
        "device": args.device,
        "hidden_dim": args.hidden_dim,
        "num_heads": args.num_heads,
        "communication_mode": args.communication_mode,
        "communication_dim": args.communication_dim,
        "disable_beta_fit": args.disable_beta_fit if args.disable_beta_fit else None,
        "save_train_loss_history": (
            True if getattr(args, "save_train_loss_history", False) else None
        ),
        "save_epsilon_series": True if getattr(args, "save_epsilon_series", False) else None,
        "save_consensus_series": (
            True if getattr(args, "save_consensus_series", False) else None
        ),
        "save_learning_rate_plots": (
            False if getattr(args, "no_learning_rate_plots", False) else None
        ),
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

    train_episodes_per_n = parse_train_episodes_per_n(run_config.get("train_episodes_per_n"))
    num_nodes = int(run_config["num_nodes"])
    effective_train_episodes = resolve_train_episodes(
        num_nodes=num_nodes,
        train_episodes_per_n=train_episodes_per_n,
        train_episodes_default=int(run_config["train_episodes"]),
    )

    run_summary = run_single_seed(
        seed=int(run_config["seed"]),
        generator=generator,
        num_nodes=num_nodes,
        graph_cache_dir=graph_cache_dir,
        artifacts_dir=artifacts_dir,
        train_episodes=effective_train_episodes,
        test_episodes=int(run_config["test_episodes"]),
        max_horizon=int(run_config["max_horizon"]),
        signal_quality=float(run_config["signal_quality"]),
        hidden_dim=int(run_config["hidden_dim"]),
        num_heads=int(run_config["num_heads"]),
        communication_mode=str(run_config["communication_mode"]),
        communication_dim=(
            None if run_config["communication_dim"] is None else int(run_config["communication_dim"])
        ),
        learning_rate=float(run_config["learning_rate"]),
        weight_decay=float(run_config["weight_decay"]),
        dropout=float(run_config["dropout"]),
        validation_episodes=int(run_config["validation_episodes"]),
        validation_eval_every=int(run_config["validation_eval_every"]),
        device=str(run_config["device"]),
        disable_beta_fit=bool(run_config["disable_beta_fit"]),
        save_train_loss_history=bool(run_config["save_train_loss_history"]),
        save_epsilon_series=bool(run_config["save_epsilon_series"]),
        save_consensus_series=bool(run_config["save_consensus_series"]),
        save_learning_rate_plots=bool(run_config["save_learning_rate_plots"]),
    )
    print(
        f"Finished seed={run_summary['seed']} "
        f"for {run_summary['num_conditions']} conditions "
        f"(mean final loss={run_summary['mean_final_loss']:.4f})."
    )
    print(f"Artifacts written to: {artifacts_dir}")


if __name__ == "__main__":
    main()
