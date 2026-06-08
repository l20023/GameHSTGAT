"""Render interactive HTML viewers for the cumulative majority-vote baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.episode_animation import (
    compute_unanimous_consensus,
    load_rollouts_cache,
    pyg_to_networkx,
    rollout_majority_trace,
    rollouts_cache_is_valid,
    rollouts_cache_path,
    sample_episode,
    save_interactive_episode_view,
    save_rollouts_cache,
)
from src.graph_generator import GraphGenerator
from src.grid_tasks import resolve_max_horizon
from src.learning_rate_plots import (
    condition_key_for_topology,
    stage_anchored_t2_plot_for_viewer,
)

MAJORITY_ALGORITHM_LABEL = "majority vote baseline"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-nodes", type=int, default=10)
    parser.add_argument("--signal-quality", type=float, default=0.55)
    parser.add_argument("--topology", type=str, default="complete", help="complete | ws_p_0.0 | ws_p_0.1")
    parser.add_argument("--max-horizon", type=int, default=50)
    parser.add_argument("--graph-cache-dir", type=Path, default=Path("artifacts/graphs"))
    parser.add_argument("--episode-seed", type=int, default=4242)
    parser.add_argument("--episode-variants", type=int, default=12)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--frame-step", type=int, default=1, help="Reserved for parity with RGAT script.")
    parser.add_argument("--force-reroll", action="store_true")
    return parser.parse_args()


def resolve_topology_name(topology: str, *, seed: int) -> str:
    if topology == "complete":
        return "complete"
    if topology in {"ws_p_0.0", "ws_p_0.1"}:
        return f"{topology}_seed_{seed}"
    raise ValueError("topology must be one of: complete, ws_p_0.0, ws_p_0.1")


def default_output_path(
    *,
    num_nodes: int,
    signal_quality: float,
    topology: str,
    seed: int,
) -> Path:
    q_key = f"{signal_quality:.2f}".replace(".", "p")
    return Path(
        f"artifacts/animations/majority_vote/n_{num_nodes}/q_{q_key}/{topology}/seed_{seed}.html"
    )


def main() -> None:
    args = parse_args()
    topology_key = resolve_topology_name(args.topology, seed=args.seed)
    max_horizon = resolve_max_horizon(
        signal_quality=args.signal_quality,
        topology_name=topology_key,
        base_horizon=args.max_horizon,
    )

    generator = GraphGenerator()
    graphs = generator.generate_and_store_experiment_graphs(
        storage_dir=args.graph_cache_dir,
        num_nodes_list=[args.num_nodes],
        seed=args.seed,
    )
    graph_data = graphs[args.num_nodes][topology_key]
    topology_label = str(getattr(graph_data, "topology", args.topology))

    primary = args.output or default_output_path(
        num_nodes=args.num_nodes,
        signal_quality=args.signal_quality,
        topology=args.topology,
        seed=args.seed,
    )
    episode_seeds = [
        args.episode_seed + offset for offset in range(max(args.episode_variants, 1))
    ]
    cache_path = rollouts_cache_path(primary)
    use_cache = (
        not args.force_reroll
        and rollouts_cache_is_valid(
            cache_path,
            episode_seeds=episode_seeds,
            num_nodes=args.num_nodes,
            max_horizon=max_horizon,
            signal_quality=args.signal_quality,
            topology=topology_label,
            checkpoint_path=None,
        )
    )

    if use_cache:
        traces, episode_seeds = load_rollouts_cache(cache_path)
        print(f"Loaded rollouts cache: {cache_path}")
    else:
        traces = []
        for episode_seed in episode_seeds:
            episode = sample_episode(
                num_nodes=args.num_nodes,
                max_horizon=max_horizon,
                signal_quality=args.signal_quality,
                seed=episode_seed,
            )
            traces.append(
                rollout_majority_trace(
                    graph_data=graph_data,
                    episode=episode,
                    signal_quality=args.signal_quality,
                    topology=topology_label,
                    rollout_seed=episode_seed,
                )
            )
        save_rollouts_cache(
            cache_path,
            traces,
            episode_seeds=episode_seeds,
            checkpoint_path=None,
        )
        print(f"Saved rollouts cache: {cache_path}")

    trace = traces[0]
    nx_graph = pyg_to_networkx(graph_data)
    consensus = compute_unanimous_consensus(trace)
    html_path = primary if primary.suffix == ".html" else primary.with_suffix(".html")
    condition_key = condition_key_for_topology(
        num_nodes=args.num_nodes,
        topology=args.topology,
        seed=args.seed,
    )
    summary_plot_filename = stage_anchored_t2_plot_for_viewer(
        communication_mode="majority_vote",
        num_nodes=args.num_nodes,
        signal_quality=args.signal_quality,
        topology=args.topology,
        seed=args.seed,
        html_path=html_path,
        project_root=PROJECT_ROOT,
    )
    save_interactive_episode_view(
        traces,
        nx_graph,
        html_path,
        episode_seeds=episode_seeds,
        condition_key=condition_key,
        summary_plot_filename=summary_plot_filename,
        algorithm_label=MAJORITY_ALGORITHM_LABEL,
    )
    print(f"Interactive viewer saved: {html_path}")
    if summary_plot_filename:
        print(f"Test-set summary plot: {html_path.with_name(summary_plot_filename)}")

    summary = {
        "algorithm": "majority_vote",
        "theta": trace.theta,
        "max_horizon": trace.max_horizon,
        "final_error_rate": float(trace.error_rates[-1]),
        "consensus": consensus,
        "outputs": [str(html_path)],
    }
    summary_path = html_path.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
