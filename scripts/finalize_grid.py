"""Finalize grid runs: aggregate metrics, plots, and CSV summaries."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train import DEFAULT_CONFIG_PATH, DEFAULT_RUN_CONFIG
from src.config import load_yaml_config, merge_flat_config, resolve_replication_seeds
from src.grid_summary import build_grid_summary, collect_records_from_artifacts
from src.grid_tasks import parse_csv_floats, parse_csv_ints, parse_train_episodes_per_n


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate grid artifacts into summary JSON and CSV tables."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to YAML config file used as baseline defaults.",
    )
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        required=True,
        help="Root directory containing grid run metrics (grid_runs/).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to write grid summary JSON.",
    )
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=None,
        help="Path to write per-run metrics CSV.",
    )
    parser.add_argument(
        "--aggregate-csv",
        type=Path,
        default=None,
        help="Path to write aggregated metrics CSV.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated seeds used in the grid (for metadata).",
    )
    parser.add_argument(
        "--num-nodes-list",
        type=str,
        default="10,100,1000",
        help="Comma-separated node counts used in the grid.",
    )
    parser.add_argument(
        "--signal-quality-list",
        type=str,
        default="0.55,0.6,0.7,0.8",
        help="Comma-separated signal qualities used in the grid.",
    )
    parser.add_argument(
        "--communication-mode",
        type=str,
        choices=["fair_1bit", "vector"],
        default="fair_1bit",
    )
    parser.add_argument("--communication-dim", type=int, default=None)
    parser.add_argument("--skip-csv", action="store_true")
    return parser.parse_args()


def resolve_run_config(args: argparse.Namespace) -> dict[str, Any]:
    yaml_payload = load_yaml_config(args.config)
    return merge_flat_config(
        defaults=DEFAULT_RUN_CONFIG,
        yaml_config=yaml_payload,
        cli_overrides={},
    )


def main() -> None:
    args = parse_args()
    run_config = resolve_run_config(args)
    seeds = resolve_replication_seeds(cli_seeds=args.seeds, run_config=run_config)
    num_nodes_list = parse_csv_ints(args.num_nodes_list)
    signal_quality_list = parse_csv_floats(args.signal_quality_list)
    train_episodes_per_n = parse_train_episodes_per_n(run_config.get("train_episodes_per_n"))

    artifacts_root = args.artifacts_root
    records = collect_records_from_artifacts(artifacts_root)
    grid_config = {
        "seeds": seeds,
        "num_nodes_list": num_nodes_list,
        "signal_quality_list": signal_quality_list,
        "train_episodes": int(run_config["train_episodes"]),
        "train_episodes_per_n": train_episodes_per_n,
        "test_episodes": int(run_config["test_episodes"]),
        "max_horizon": int(run_config["max_horizon"]),
        "hidden_dim": int(run_config["hidden_dim"]),
        "num_heads": int(run_config["num_heads"]),
        "communication_mode": args.communication_mode,
        "communication_dim": args.communication_dim,
        "learning_rate": float(run_config["learning_rate"]),
        "device": str(run_config["device"]),
        "disable_beta_fit": bool(run_config["disable_beta_fit"]),
        "wandb_project": str(run_config["wandb_project"]),
        "wandb_entity": run_config["wandb_entity"],
        "graph_cache_dir": str(run_config["graph_cache_dir"]),
        "artifacts_root": str(artifacts_root),
    }
    summary = build_grid_summary(
        records=records,
        grid_config=grid_config,
        run_summaries=None,
        artifacts_root=artifacts_root,
        num_nodes_list=num_nodes_list,
        signal_quality_list=signal_quality_list,
    )

    if args.output is None:
        parent = artifacts_root.parent
        suffix = "fair" if args.communication_mode == "fair_1bit" else "vector"
        output_path = parent.parent / f"grid_summary_{suffix}.json"
    else:
        output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Grid summary written to: {output_path}")

    if args.skip_csv:
        return

    metrics_csv = args.metrics_csv
    aggregate_csv = args.aggregate_csv
    if metrics_csv is None or aggregate_csv is None:
        parent = artifacts_root.parent
        suffix = "fair" if args.communication_mode == "fair_1bit" else "vector"
        if metrics_csv is None:
            metrics_csv = parent.parent / f"metrics_summary_{suffix}.csv"
        if aggregate_csv is None:
            aggregate_csv = parent.parent / f"metrics_summary_{suffix}_aggregated.csv"

    summarize_script = PROJECT_ROOT / "scripts" / "summarize_metrics.py"
    subprocess.run(
        [
            sys.executable,
            str(summarize_script),
            "--root",
            str(artifacts_root),
            "--csv",
            str(metrics_csv),
            "--aggregate-csv",
            str(aggregate_csv),
        ],
        check=True,
    )
    print(f"Metrics CSV written to: {metrics_csv}")
    print(f"Aggregated CSV written to: {aggregate_csv}")


if __name__ == "__main__":
    main()
