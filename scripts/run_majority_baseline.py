"""Evaluate cumulative majority-vote baseline for one grid seed setting."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train import DEFAULT_CONFIG_PATH, DEFAULT_RUN_CONFIG, save_seed_metrics
from src.config import load_yaml_config, merge_flat_config
from src.graph_generator import GraphGenerator
from src.grid_tasks import resolve_max_horizon
from src.learning_rate_plots import learning_rate_plot_path, save_learning_rate_plot
from src.training_pipeline import condition_result_to_dict, run_majority_baseline_condition

PROPOSAL_WS_PROBS = [0.0, 0.1]
DEFAULT_ARTIFACTS_DIR = "artifacts/training_metrics_majority"
MAJORITY_ALGORITHM_LABEL = "majority vote baseline"


def seed_metrics_path(artifacts_dir: Path, seed: int) -> Path:
    """Return the metrics JSON path for one seed run."""
    return artifacts_dir / f"seed_{seed}" / "metrics.json"


def set_global_seed(seed: int) -> None:
    """Set CPU RNG seeds for reproducible baseline evaluation."""
    random.seed(seed)
    np.random.seed(seed)


def run_single_seed(
    seed: int,
    *,
    generator: GraphGenerator,
    num_nodes: int,
    graph_cache_dir: Path,
    artifacts_dir: Path,
    test_episodes: int,
    max_horizon: int,
    signal_quality: float,
    disable_beta_fit: bool,
    save_epsilon_series: bool,
    save_consensus_series: bool,
    save_learning_rate_plots: bool,
    skip_existing: bool = False,
) -> dict[str, Any]:
    """Run majority baseline evaluation for one seed across all graph conditions."""
    metrics_path = seed_metrics_path(artifacts_dir, seed)
    if skip_existing and metrics_path.is_file():
        return {
            "seed": seed,
            "num_nodes": num_nodes,
            "signal_quality": signal_quality,
            "skipped": True,
            "metrics_path": str(metrics_path),
        }

    set_global_seed(seed)
    graphs = generator.generate_and_store_experiment_graphs(
        storage_dir=graph_cache_dir,
        num_nodes_list=[num_nodes],
        ws_probs=PROPOSAL_WS_PROBS,
        seed=seed,
    )
    per_condition_metrics: dict[str, dict[str, Any]] = {}
    for num_nodes_key, conditions in graphs.items():
        for condition_name, graph_data in conditions.items():
            key = f"n_{num_nodes_key}/{condition_name}"
            condition_max_horizon = resolve_max_horizon(
                signal_quality=signal_quality,
                topology_name=condition_name,
                base_horizon=max_horizon,
            )
            result = run_majority_baseline_condition(
                graph_data=graph_data,
                test_episodes=test_episodes,
                max_horizon=condition_max_horizon,
                signal_quality=signal_quality,
                seed=seed,
                disable_beta_fit=disable_beta_fit,
                include_consensus_series=save_consensus_series,
            )
            condition_metrics = condition_result_to_dict(
                result,
                save_epsilon_series=save_epsilon_series,
                save_consensus_series=save_consensus_series,
                algorithm="majority_vote",
            )
            if save_learning_rate_plots:
                anchored_t2_plot_path = learning_rate_plot_path(
                    artifacts_dir=artifacts_dir,
                    seed=seed,
                    condition_key=key,
                )
                save_learning_rate_plot(
                    output_path=anchored_t2_plot_path,
                    epsilon_series=result.epsilon_series,
                    beta_fit=result.beta_fit,
                    beta_hst_max=result.beta_hst_max,
                    condition_key=key,
                    signal_quality=signal_quality,
                    beta_gap=result.beta_gap,
                    exceeds_hst_bound=result.exceeds_hst_bound,
                    convergence_warning=result.convergence_warning,
                    algorithm_label=MAJORITY_ALGORITHM_LABEL,
                )
                condition_metrics["learning_rate_plot"] = str(anchored_t2_plot_path)
                condition_metrics["learning_rate_plot_anchored_t2"] = str(anchored_t2_plot_path)
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
        "num_nodes": num_nodes,
        "signal_quality": signal_quality,
        "skipped": False,
        "metrics_path": str(metrics_path),
        "conditions": list(per_condition_metrics.keys()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate cumulative majority-vote baseline for one seed setting."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-nodes", type=int, default=None)
    parser.add_argument("--graph-cache-dir", type=Path, default=None)
    parser.add_argument("--artifacts-dir", type=Path, default=None)
    parser.add_argument("--test-episodes", type=int, default=None)
    parser.add_argument("--max-horizon", type=int, default=None)
    parser.add_argument("--signal-quality", type=float, default=None)
    parser.add_argument("--disable-beta-fit", action="store_true")
    parser.add_argument("--save-epsilon-series", action="store_true")
    parser.add_argument("--save-consensus-series", action="store_true")
    parser.add_argument("--no-learning-rate-plots", action="store_true")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip when seed metrics.json already exists.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing seed metrics (disables --skip-existing).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    yaml_payload = load_yaml_config(args.config)
    cli_overrides: dict[str, Any] = {
        "seed": args.seed,
        "num_nodes": args.num_nodes,
        "graph_cache_dir": str(args.graph_cache_dir) if args.graph_cache_dir is not None else None,
        "artifacts_dir": str(args.artifacts_dir) if args.artifacts_dir is not None else None,
        "test_episodes": args.test_episodes,
        "max_horizon": args.max_horizon,
        "signal_quality": args.signal_quality,
        "disable_beta_fit": True if args.disable_beta_fit else None,
        "save_epsilon_series": True if args.save_epsilon_series else None,
        "save_consensus_series": True if args.save_consensus_series else None,
        "save_learning_rate_plots": False if args.no_learning_rate_plots else None,
    }
    run_config = merge_flat_config(
        defaults={**DEFAULT_RUN_CONFIG, "artifacts_dir": DEFAULT_ARTIFACTS_DIR},
        yaml_config=yaml_payload,
        cli_overrides=cli_overrides,
    )
    seed = int(run_config["seed"])
    num_nodes = int(run_config["num_nodes"])
    graph_cache_dir = Path(str(run_config["graph_cache_dir"]))
    artifacts_dir = Path(str(run_config["artifacts_dir"]))
    graph_cache_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    generator = GraphGenerator()
    summary = run_single_seed(
        seed=seed,
        generator=generator,
        num_nodes=num_nodes,
        graph_cache_dir=graph_cache_dir,
        artifacts_dir=artifacts_dir,
        test_episodes=int(run_config["test_episodes"]),
        max_horizon=int(run_config["max_horizon"]),
        signal_quality=float(run_config["signal_quality"]),
        disable_beta_fit=bool(run_config["disable_beta_fit"]),
        save_epsilon_series=bool(run_config["save_epsilon_series"]),
        save_consensus_series=bool(run_config["save_consensus_series"]),
        save_learning_rate_plots=bool(run_config["save_learning_rate_plots"]),
        skip_existing=bool(args.skip_existing) and not bool(args.force),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
