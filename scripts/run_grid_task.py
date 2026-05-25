"""Run a single grid task by index (for SLURM array workers)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train import DEFAULT_CONFIG_PATH, DEFAULT_RUN_CONFIG, run_single_seed
from src.config import load_yaml_config, merge_flat_config, resolve_replication_seeds
from src.graph_generator import GraphGenerator
from src.grid_tasks import (
    build_grid_tasks,
    parse_csv_floats,
    parse_csv_ints,
    parse_train_episodes_per_n,
    resolve_train_episodes,
    task_artifacts_dir,
)


def resolve_grid_context(args: argparse.Namespace) -> dict[str, Any]:
    yaml_payload = load_yaml_config(args.config)
    cli_overrides: dict[str, Any] = {
        "graph_cache_dir": str(args.graph_cache_dir) if args.graph_cache_dir is not None else None,
        "wandb_project": args.wandb_project,
        "wandb_entity": args.wandb_entity,
        "train_episodes": args.train_episodes,
        "test_episodes": args.test_episodes,
        "max_horizon": args.max_horizon,
        "hidden_dim": args.hidden_dim,
        "num_heads": args.num_heads,
        "communication_mode": args.communication_mode,
        "communication_dim": args.communication_dim,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "validation_episodes": args.validation_episodes,
        "validation_eval_every": args.validation_eval_every,
        "device": args.device,
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
    run_config = merge_flat_config(
        defaults=DEFAULT_RUN_CONFIG,
        yaml_config=yaml_payload,
        cli_overrides=cli_overrides,
    )
    seeds = resolve_replication_seeds(cli_seeds=args.seeds, run_config=run_config)
    num_nodes_list = parse_csv_ints(args.num_nodes_list)
    signal_quality_list = parse_csv_floats(args.signal_quality_list)
    tasks = build_grid_tasks(
        seeds=seeds,
        num_nodes_list=num_nodes_list,
        signal_quality_list=signal_quality_list,
    )
    train_episodes_per_n = parse_train_episodes_per_n(run_config.get("train_episodes_per_n"))
    return {
        "run_config": run_config,
        "seeds": seeds,
        "num_nodes_list": num_nodes_list,
        "signal_quality_list": signal_quality_list,
        "tasks": tasks,
        "train_episodes_per_n": train_episodes_per_n,
    }


def run_grid_task(
    *,
    task_index: int,
    context: dict[str, Any],
    artifacts_root: Path,
    communication_mode: str,
    communication_dim: int | None,
) -> dict[str, Any]:
    tasks = context["tasks"]
    if task_index < 0 or task_index >= len(tasks):
        raise IndexError(
            f"task_index {task_index} out of range for {len(tasks)} grid tasks."
        )
    task = tasks[task_index]
    run_config = context["run_config"]
    train_episodes_per_n = context["train_episodes_per_n"]
    effective_train_episodes = resolve_train_episodes(
        num_nodes=task.num_nodes,
        train_episodes_per_n=train_episodes_per_n,
        train_episodes_default=int(run_config["train_episodes"]),
    )
    setting_artifacts_dir = task_artifacts_dir(artifacts_root, task)
    setting_artifacts_dir.mkdir(parents=True, exist_ok=True)
    graph_cache_dir = Path(str(run_config["graph_cache_dir"]))
    graph_cache_dir.mkdir(parents=True, exist_ok=True)

    generator = GraphGenerator()
    run_summary = run_single_seed(
        seed=task.seed,
        generator=generator,
        num_nodes=task.num_nodes,
        graph_cache_dir=graph_cache_dir,
        artifacts_dir=setting_artifacts_dir,
        wandb_project=str(run_config["wandb_project"]),
        wandb_entity=(
            None if run_config["wandb_entity"] is None else str(run_config["wandb_entity"])
        ),
        train_episodes=effective_train_episodes,
        test_episodes=int(run_config["test_episodes"]),
        max_horizon=int(run_config["max_horizon"]),
        signal_quality=task.signal_quality,
        hidden_dim=int(run_config["hidden_dim"]),
        num_heads=int(run_config["num_heads"]),
        communication_mode=communication_mode,
        communication_dim=communication_dim,
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
    return {
        "task_index": task_index,
        "seed": task.seed,
        "num_nodes": task.num_nodes,
        "signal_quality": task.signal_quality,
        "setting_key": task.setting_key,
        "train_episodes": effective_train_episodes,
        "summary": run_summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one grid task by index or print task count."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to YAML config file used as baseline defaults.",
    )
    parser.add_argument(
        "--count",
        action="store_true",
        help="Print number of grid tasks and exit.",
    )
    parser.add_argument(
        "--task-index",
        type=int,
        default=None,
        help="Zero-based grid task index (maps to SLURM_ARRAY_TASK_ID - 1).",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated seeds. If omitted, uses config num_seeds.",
    )
    parser.add_argument(
        "--num-nodes-list",
        type=str,
        default="10,100,1000",
        help="Comma-separated node counts.",
    )
    parser.add_argument(
        "--signal-quality-list",
        type=str,
        default="0.55,0.6,0.7,0.8",
        help="Comma-separated signal qualities.",
    )
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=None,
        help="Root directory for grid run artifacts.",
    )
    parser.add_argument("--graph-cache-dir", type=Path, default=None)
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--train-episodes", type=int, default=None)
    parser.add_argument("--test-episodes", type=int, default=None)
    parser.add_argument("--max-horizon", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument(
        "--communication-mode",
        type=str,
        choices=["fair_1bit", "vector"],
        default="fair_1bit",
    )
    parser.add_argument("--communication-dim", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--validation-episodes", type=int, default=None)
    parser.add_argument("--validation-eval-every", type=int, default=None)
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda", "mps"],
        default=None,
    )
    parser.add_argument("--disable-beta-fit", action="store_true")
    parser.add_argument("--save-train-loss-history", action="store_true")
    parser.add_argument("--save-epsilon-series", action="store_true")
    parser.add_argument("--no-learning-rate-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    context = resolve_grid_context(args)

    if args.count:
        print(len(context["tasks"]))
        return

    if args.task_index is None:
        raise SystemExit("Either --count or --task-index is required.")

    if args.artifacts_root is not None:
        artifacts_root = args.artifacts_root
    else:
        base_artifacts_dir = Path(str(context["run_config"]["artifacts_dir"]))
        artifacts_root = base_artifacts_dir / "grid_runs"

    communication_dim = (
        None if args.communication_dim is None else int(args.communication_dim)
    )
    result = run_grid_task(
        task_index=int(args.task_index),
        context=context,
        artifacts_root=artifacts_root,
        communication_mode=str(args.communication_mode),
        communication_dim=communication_dim,
    )
    print(
        f"Finished task_index={result['task_index']} "
        f"seed={result['seed']} n={result['num_nodes']} q={result['signal_quality']:.2f} "
        f"(mean final loss={result['summary']['mean_final_loss']:.4f})."
    )


if __name__ == "__main__":
    main()
