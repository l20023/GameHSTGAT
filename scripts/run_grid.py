"""Orchestrate proposal grid runs across node sizes, signal qualities, and seeds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train import DEFAULT_CONFIG_PATH, DEFAULT_RUN_CONFIG, run_single_seed
from src.config import load_yaml_config, merge_flat_config, resolve_replication_seeds
from src.graph_generator import GraphGenerator
from src.grid_summary import build_grid_summary, normalize_condition_name
from src.grid_tasks import (
    build_grid_tasks,
    parse_csv_floats,
    parse_csv_ints,
    parse_train_episodes_per_n,
    resolve_train_episodes,
    task_artifacts_dir,
)


def _load_condition_metrics(metrics_path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    conditions = payload.get("conditions", {})
    if not isinstance(conditions, dict):
        raise ValueError(f"Invalid metrics payload at {metrics_path}: 'conditions' must be a dict.")
    return conditions


def run_grid_experiments(
    *,
    seeds: list[int],
    num_nodes_list: list[int],
    signal_quality_list: list[float],
    graph_cache_dir: Path,
    artifacts_root: Path,
    wandb_project: str,
    wandb_entity: str | None,
    train_episodes: int,
    test_episodes: int,
    max_horizon: int,
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
    save_learning_rate_plots: bool,
    train_episodes_per_n: dict[int, int] | None = None,
) -> dict[str, Any]:
    generator = GraphGenerator()
    graph_cache_dir.mkdir(parents=True, exist_ok=True)
    artifacts_root.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    run_summaries: list[dict[str, Any]] = []
    tasks = build_grid_tasks(
        seeds=seeds,
        num_nodes_list=num_nodes_list,
        signal_quality_list=signal_quality_list,
    )

    for task in tasks:
        setting_artifacts_dir = task_artifacts_dir(artifacts_root, task)
        effective_train_episodes = resolve_train_episodes(
            num_nodes=task.num_nodes,
            train_episodes_per_n=train_episodes_per_n,
            train_episodes_default=train_episodes,
        )
        run_summary = run_single_seed(
            seed=task.seed,
            generator=generator,
            num_nodes=task.num_nodes,
            graph_cache_dir=graph_cache_dir,
            artifacts_dir=setting_artifacts_dir,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
            train_episodes=effective_train_episodes,
            test_episodes=test_episodes,
            max_horizon=max_horizon,
            signal_quality=task.signal_quality,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            communication_mode=communication_mode,
            communication_dim=communication_dim,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            dropout=dropout,
            validation_episodes=validation_episodes,
            validation_eval_every=validation_eval_every,
            device=device,
            disable_beta_fit=disable_beta_fit,
            save_train_loss_history=save_train_loss_history,
            save_epsilon_series=save_epsilon_series,
            save_learning_rate_plots=save_learning_rate_plots,
        )
        run_summaries.append(
            {
                "seed": task.seed,
                "num_nodes": task.num_nodes,
                "signal_quality": task.signal_quality,
                "setting_key": task.setting_key,
                "summary": run_summary,
            }
        )
        metrics_path = setting_artifacts_dir / f"seed_{task.seed}" / "metrics.json"
        condition_metrics = _load_condition_metrics(metrics_path)
        for condition_key, metrics in condition_metrics.items():
            beta_fit = metrics.get("beta_fit", {})
            beta_gat = beta_fit.get("beta") if isinstance(beta_fit, dict) else None
            records.append(
                {
                    "seed": task.seed,
                    "num_nodes": task.num_nodes,
                    "signal_quality": task.signal_quality,
                    "setting_key": task.setting_key,
                    "condition_key": condition_key,
                    "condition_name": normalize_condition_name(condition_key),
                    "beta_gat": beta_gat,
                    "beta_hst_max": metrics.get("beta_hst_max"),
                    "beta_gap": metrics.get("beta_gap"),
                    "exceeds_hst_bound": metrics.get("exceeds_hst_bound"),
                    "artifact_path": str(metrics_path),
                }
            )

    grid_config = {
        "seeds": seeds,
        "num_nodes_list": num_nodes_list,
        "signal_quality_list": signal_quality_list,
        "train_episodes": train_episodes,
        "train_episodes_per_n": train_episodes_per_n,
        "test_episodes": test_episodes,
        "max_horizon": max_horizon,
        "hidden_dim": hidden_dim,
        "num_heads": num_heads,
        "communication_mode": communication_mode,
        "communication_dim": communication_dim,
        "learning_rate": learning_rate,
        "device": device,
        "disable_beta_fit": disable_beta_fit,
        "wandb_project": wandb_project,
        "wandb_entity": wandb_entity,
        "graph_cache_dir": str(graph_cache_dir),
        "artifacts_root": str(artifacts_root),
    }
    return build_grid_summary(
        records=records,
        grid_config=grid_config,
        run_summaries=run_summaries,
        artifacts_root=artifacts_root,
        num_nodes_list=num_nodes_list,
        signal_quality_list=signal_quality_list,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run proposal grid experiments across node sizes and signal qualities."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to YAML config file used as baseline defaults.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help=(
            "Comma-separated seeds, e.g. '0,1,2,3,4'. "
            "If omitted, uses config num_seeds (default 5 -> seeds 0..4)."
        ),
    )
    parser.add_argument(
        "--num-nodes-list",
        type=str,
        default="10,100,1000",
        help="Comma-separated node counts, e.g. '10,100,1000'.",
    )
    parser.add_argument(
        "--signal-quality-list",
        type=str,
        default="0.55,0.6,0.7,0.8",
        help="Comma-separated signal qualities, e.g. '0.55,0.6,0.7,0.8'.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/grid_summary.json"),
        help="Path to write aggregated grid summary JSON.",
    )
    parser.add_argument(
        "--graph-cache-dir",
        type=Path,
        default=None,
        help="Override graph cache directory.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=None,
        help="Root output directory for per-run metrics in grid mode.",
    )
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
        default=None,
    )
    parser.add_argument("--communication-dim", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda", "mps"],
        default=None,
    )
    parser.add_argument("--disable-beta-fit", action="store_true")
    return parser.parse_args()


def resolve_run_config(args: argparse.Namespace) -> dict[str, Any]:
    yaml_payload = load_yaml_config(args.config)
    cli_overrides: dict[str, Any] = {
        "graph_cache_dir": str(args.graph_cache_dir) if args.graph_cache_dir is not None else None,
        "artifacts_dir": str(args.artifacts_dir) if args.artifacts_dir is not None else None,
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
        "device": args.device,
        "disable_beta_fit": args.disable_beta_fit if args.disable_beta_fit else None,
    }
    return merge_flat_config(
        defaults=DEFAULT_RUN_CONFIG,
        yaml_config=yaml_payload,
        cli_overrides=cli_overrides,
    )


def main() -> None:
    args = parse_args()
    run_config = resolve_run_config(args)

    seeds = resolve_replication_seeds(cli_seeds=args.seeds, run_config=run_config)
    print(f"Running {len(seeds)} replication seeds: {seeds}")
    num_nodes_list = parse_csv_ints(args.num_nodes_list)
    signal_quality_list = parse_csv_floats(args.signal_quality_list)

    graph_cache_dir = Path(str(run_config["graph_cache_dir"]))
    base_artifacts_dir = Path(str(run_config["artifacts_dir"]))
    artifacts_root = base_artifacts_dir / "grid_runs"
    train_episodes_per_n = parse_train_episodes_per_n(run_config.get("train_episodes_per_n"))

    summary = run_grid_experiments(
        seeds=seeds,
        num_nodes_list=num_nodes_list,
        signal_quality_list=signal_quality_list,
        graph_cache_dir=graph_cache_dir,
        artifacts_root=artifacts_root,
        wandb_project=str(run_config["wandb_project"]),
        wandb_entity=(
            None if run_config["wandb_entity"] is None else str(run_config["wandb_entity"])
        ),
        train_episodes=int(run_config["train_episodes"]),
        test_episodes=int(run_config["test_episodes"]),
        max_horizon=int(run_config["max_horizon"]),
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
        save_learning_rate_plots=bool(run_config["save_learning_rate_plots"]),
        train_episodes_per_n=train_episodes_per_n,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Grid run summary written to: {args.output}")


if __name__ == "__main__":
    main()
