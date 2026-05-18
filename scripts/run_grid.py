"""Orchestrate proposal grid runs across node sizes, signal qualities, and seeds."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train import DEFAULT_CONFIG_PATH, DEFAULT_RUN_CONFIG, run_single_seed
from src.config import load_yaml_config, merge_flat_config
from src.graph_generator import GraphGenerator
from src.reporting import classify_regimes


def _parse_csv_ints(value: str) -> list[int]:
    parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed:
        raise ValueError("Expected at least one integer value.")
    return parsed


def _parse_csv_floats(value: str) -> list[float]:
    parsed = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed:
        raise ValueError("Expected at least one float value.")
    return parsed


def _format_signal_quality(signal_quality: float) -> str:
    return f"{signal_quality:.1f}".replace(".", "p")


def _normalize_condition_name(condition_key: str) -> str:
    if "/" in condition_key:
        _, condition_name = condition_key.split("/", 1)
    else:
        condition_name = condition_key
    return re.sub(r"_seed_\d+$", "", condition_name)


def _safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _safe_bool_ratio(values: list[bool]) -> float | None:
    if not values:
        return None
    return float(sum(1 for item in values if item) / len(values))


def _aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_setting: dict[str, dict[str, Any]] = {}
    by_condition: dict[str, dict[str, Any]] = {}
    by_setting_and_condition: dict[str, dict[str, Any]] = {}

    for record in records:
        setting_key = str(record["setting_key"])
        condition_name = str(record["condition_name"])
        composite_key = f"{setting_key}/{condition_name}"

        for key, bucket_dict in (
            (setting_key, by_setting),
            (condition_name, by_condition),
            (composite_key, by_setting_and_condition),
        ):
            if key not in bucket_dict:
                bucket_dict[key] = {
                    "beta_gat_values": [],
                    "beta_gap_values": [],
                    "exceeds_values": [],
                    "artifact_paths": set(),
                    "num_records": 0,
                }
            bucket = bucket_dict[key]
            bucket["num_records"] += 1
            beta_gat = record.get("beta_gat")
            if isinstance(beta_gat, (int, float)):
                bucket["beta_gat_values"].append(float(beta_gat))
            beta_gap = record.get("beta_gap")
            if isinstance(beta_gap, (int, float)):
                bucket["beta_gap_values"].append(float(beta_gap))
            exceeds = record.get("exceeds_hst_bound")
            if isinstance(exceeds, bool):
                bucket["exceeds_values"].append(exceeds)
            artifact_path = record.get("artifact_path")
            if isinstance(artifact_path, str):
                bucket["artifact_paths"].add(artifact_path)

    def _finalize(source: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        finalized: dict[str, dict[str, Any]] = {}
        for key, bucket in source.items():
            finalized[key] = {
                "num_records": int(bucket["num_records"]),
                "num_beta_gat_values": len(bucket["beta_gat_values"]),
                "num_exceeds_values": len(bucket["exceeds_values"]),
                "mean_beta_gat": _safe_mean(bucket["beta_gat_values"]),
                "mean_beta_gap": _safe_mean(bucket["beta_gap_values"]),
                "proportion_exceeds_hst_bound": _safe_bool_ratio(bucket["exceeds_values"]),
                "artifact_paths": sorted(bucket["artifact_paths"]),
            }
        return finalized

    return {
        "by_setting": _finalize(by_setting),
        "by_condition": _finalize(by_condition),
        "by_setting_and_condition": _finalize(by_setting_and_condition),
    }


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
    learning_rate: float,
    disable_beta_fit: bool,
) -> dict[str, Any]:
    generator = GraphGenerator()
    graph_cache_dir.mkdir(parents=True, exist_ok=True)
    artifacts_root.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    run_summaries: list[dict[str, Any]] = []

    for signal_quality in signal_quality_list:
        for num_nodes in num_nodes_list:
            quality_key = _format_signal_quality(signal_quality)
            setting_key = f"n_{num_nodes}/q_{signal_quality:.1f}"
            setting_artifacts_dir = artifacts_root / f"n_{num_nodes}" / f"q_{quality_key}"
            for seed in seeds:
                run_summary = run_single_seed(
                    seed=seed,
                    generator=generator,
                    num_nodes=num_nodes,
                    graph_cache_dir=graph_cache_dir,
                    artifacts_dir=setting_artifacts_dir,
                    wandb_project=wandb_project,
                    wandb_entity=wandb_entity,
                    train_episodes=train_episodes,
                    test_episodes=test_episodes,
                    max_horizon=max_horizon,
                    signal_quality=signal_quality,
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    learning_rate=learning_rate,
                    disable_beta_fit=disable_beta_fit,
                )
                run_summaries.append(
                    {
                        "seed": seed,
                        "num_nodes": num_nodes,
                        "signal_quality": signal_quality,
                        "setting_key": setting_key,
                        "summary": run_summary,
                    }
                )
                metrics_path = setting_artifacts_dir / f"seed_{seed}" / "metrics.json"
                condition_metrics = _load_condition_metrics(metrics_path)
                for condition_key, metrics in condition_metrics.items():
                    beta_fit = metrics.get("beta_fit", {})
                    beta_gat = beta_fit.get("beta") if isinstance(beta_fit, dict) else None
                    records.append(
                        {
                            "seed": seed,
                            "num_nodes": num_nodes,
                            "signal_quality": signal_quality,
                            "setting_key": setting_key,
                            "condition_key": condition_key,
                            "condition_name": _normalize_condition_name(condition_key),
                            "beta_gat": beta_gat,
                            "beta_hst_max": metrics.get("beta_hst_max"),
                            "beta_gap": metrics.get("beta_gap"),
                            "exceeds_hst_bound": metrics.get("exceeds_hst_bound"),
                            "artifact_path": str(metrics_path),
                        }
                    )

    aggregates = _aggregate_records(records)
    regime_classification = classify_regimes(aggregates)
    return {
        "grid_config": {
            "seeds": seeds,
            "num_nodes_list": num_nodes_list,
            "signal_quality_list": signal_quality_list,
            "train_episodes": train_episodes,
            "test_episodes": test_episodes,
            "max_horizon": max_horizon,
            "hidden_dim": hidden_dim,
            "num_heads": num_heads,
            "learning_rate": learning_rate,
            "disable_beta_fit": disable_beta_fit,
            "wandb_project": wandb_project,
            "wandb_entity": wandb_entity,
            "graph_cache_dir": str(graph_cache_dir),
            "artifacts_root": str(artifacts_root),
        },
        "num_runs": len(run_summaries),
        "num_condition_records": len(records),
        "run_summaries": run_summaries,
        "aggregates": aggregates,
        "regime_classification": regime_classification,
    }


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
        default="0",
        help="Comma-separated seeds, e.g. '0,1,2'.",
    )
    parser.add_argument(
        "--num-nodes-list",
        type=str,
        default="10,50,100",
        help="Comma-separated node counts, e.g. '10,50,100'.",
    )
    parser.add_argument(
        "--signal-quality-list",
        type=str,
        default="0.6,0.8",
        help="Comma-separated signal qualities, e.g. '0.6,0.8'.",
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
    parser.add_argument("--learning-rate", type=float, default=None)
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
        "learning_rate": args.learning_rate,
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

    seeds = _parse_csv_ints(args.seeds)
    num_nodes_list = _parse_csv_ints(args.num_nodes_list)
    signal_quality_list = _parse_csv_floats(args.signal_quality_list)

    graph_cache_dir = Path(str(run_config["graph_cache_dir"]))
    base_artifacts_dir = Path(str(run_config["artifacts_dir"]))
    artifacts_root = base_artifacts_dir / "grid_runs"

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
        learning_rate=float(run_config["learning_rate"]),
        disable_beta_fit=bool(run_config["disable_beta_fit"]),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Grid run summary written to: {args.output}")


if __name__ == "__main__":
    main()
