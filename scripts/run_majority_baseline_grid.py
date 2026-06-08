"""Run one majority-vote baseline grid task by index (for SLURM array workers)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_majority_baseline import run_single_seed
from scripts.train import DEFAULT_CONFIG_PATH, DEFAULT_RUN_CONFIG
from src.config import load_yaml_config, merge_flat_config, resolve_replication_seeds
from src.graph_generator import GraphGenerator
from src.grid_tasks import (
    build_grid_tasks,
    parse_csv_ints,
    proposal_grid_signal_quality_csv,
    resolve_grid_signal_quality_list,
    task_artifacts_dir,
)

DEFAULT_ARTIFACTS_ROOT = Path("artifacts/training_metrics_majority/grid_runs")


def resolve_grid_context(args: argparse.Namespace) -> dict[str, Any]:
    yaml_payload = load_yaml_config(args.config)
    cli_overrides: dict[str, Any] = {
        "graph_cache_dir": str(args.graph_cache_dir) if args.graph_cache_dir is not None else None,
        "test_episodes": args.test_episodes,
        "max_horizon": args.max_horizon,
        "disable_beta_fit": args.disable_beta_fit if args.disable_beta_fit else None,
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
    signal_quality_list = resolve_grid_signal_quality_list(
        cli_value=args.signal_quality_list,
        run_config=run_config,
    )
    tasks = build_grid_tasks(
        seeds=seeds,
        num_nodes_list=num_nodes_list,
        signal_quality_list=signal_quality_list,
    )
    return {
        "run_config": run_config,
        "seeds": seeds,
        "num_nodes_list": num_nodes_list,
        "signal_quality_list": signal_quality_list,
        "tasks": tasks,
    }


def run_majority_grid_task(
    *,
    task_index: int,
    context: dict[str, Any],
    artifacts_root: Path,
    skip_existing: bool = False,
) -> dict[str, Any]:
    tasks = context["tasks"]
    if task_index < 0 or task_index >= len(tasks):
        raise IndexError(
            f"task_index {task_index} out of range for {len(tasks)} grid tasks."
        )
    task = tasks[task_index]
    run_config = context["run_config"]
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
        test_episodes=int(run_config["test_episodes"]),
        max_horizon=int(run_config["max_horizon"]),
        signal_quality=task.signal_quality,
        disable_beta_fit=bool(run_config["disable_beta_fit"]),
        save_epsilon_series=bool(run_config["save_epsilon_series"]),
        save_consensus_series=bool(run_config["save_consensus_series"]),
        save_learning_rate_plots=bool(run_config["save_learning_rate_plots"]),
        skip_existing=skip_existing,
    )
    return {
        "task_index": task_index,
        "seed": task.seed,
        "num_nodes": task.num_nodes,
        "signal_quality": task.signal_quality,
        "setting_key": task.setting_key,
        "skipped": bool(run_summary.get("skipped", False)),
        "summary": run_summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one majority-vote baseline grid task by index or print task count."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--count", action="store_true")
    parser.add_argument("--all", action="store_true", help="Run all grid tasks sequentially.")
    parser.add_argument("--task-index", type=int, default=None)
    parser.add_argument("--seeds", type=str, default=None)
    parser.add_argument("--num-nodes-list", type=str, default="10,100,1000")
    parser.add_argument(
        "--signal-quality-list",
        type=str,
        default=proposal_grid_signal_quality_csv(),
    )
    parser.add_argument("--artifacts-root", type=Path, default=DEFAULT_ARTIFACTS_ROOT)
    parser.add_argument("--graph-cache-dir", type=Path, default=None)
    parser.add_argument("--test-episodes", type=int, default=None)
    parser.add_argument("--max-horizon", type=int, default=None)
    parser.add_argument("--disable-beta-fit", action="store_true")
    parser.add_argument("--save-epsilon-series", action="store_true")
    parser.add_argument("--save-consensus-series", action="store_true")
    parser.add_argument("--no-learning-rate-plots", action="store_true")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip tasks whose seed metrics.json already exists (default: on).",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_false",
        dest="skip_existing",
        help="Re-run all tasks even when metrics.json exists.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Alias for --no-skip-existing (overwrite existing outputs).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    context = resolve_grid_context(args)

    if args.count:
        print(len(context["tasks"]))
        return

    skip_existing = bool(args.skip_existing) and not bool(args.force)

    if getattr(args, "all", False):
        for task_index in range(len(context["tasks"])):
            result = run_majority_grid_task(
                task_index=task_index,
                context=context,
                artifacts_root=args.artifacts_root,
                skip_existing=skip_existing,
            )
            status = "Skipped" if result["skipped"] else "Completed"
            print(
                f"{status} majority baseline task {result['task_index']} "
                f"(seed={result['seed']}, n={result['num_nodes']}, q={result['signal_quality']})",
                flush=True,
            )
        return

    if args.task_index is None:
        raise SystemExit("Either --count or --task-index is required.")

    result = run_majority_grid_task(
        task_index=args.task_index,
        context=context,
        artifacts_root=args.artifacts_root,
        skip_existing=skip_existing,
    )
    status = "Skipped" if result["skipped"] else "Completed"
    print(
        f"{status} majority baseline task {result['task_index']} "
        f"(seed={result['seed']}, n={result['num_nodes']}, q={result['signal_quality']})"
    )


if __name__ == "__main__":
    main()
