"""Train (or load) an RGAT and animate per-node predictions on one episode."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.episode_animation import (
    compute_unanimous_consensus,
    load_checkpoint,
    pyg_to_networkx,
    rollout_episode,
    sample_episode,
    save_checkpoint,
    save_episode_animation,
    save_interactive_episode_view,
)
from src.graph_generator import GraphGenerator
from src.grid_tasks import resolve_max_horizon
from src.training_pipeline import resolve_runtime_device, train_condition_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-nodes", type=int, default=10)
    parser.add_argument("--signal-quality", type=float, default=0.55)
    parser.add_argument("--topology", type=str, default="complete", help="complete | ws_p_0.0 | ws_p_0.1")
    parser.add_argument("--communication-mode", type=str, default="fair_1bit", choices=["fair_1bit", "vector"])
    parser.add_argument("--communication-dim", type=int, default=None)
    parser.add_argument("--train-episodes", type=int, default=5000)
    parser.add_argument("--max-horizon", type=int, default=50)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--graph-cache-dir", type=Path, default=Path("artifacts/graphs"))
    parser.add_argument("--checkpoint", type=Path, default=None, help="Reuse saved weights (.pt).")
    parser.add_argument("--save-checkpoint", type=Path, default=None, help="Write weights after training.")
    parser.add_argument("--episode-seed", type=int, default=4242, help="Base seed for the first animated episode.")
    parser.add_argument(
        "--episode-variants",
        type=int,
        default=12,
        help="Number of pre-rendered signal episodes bundled in the HTML viewer.",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["html", "gif", "both"],
        default="html",
        help="html: interactive scrubbable viewer (default); gif: legacy animation.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Primary output path (.html or .gif).")
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--frame-step", type=int, default=1, help="Use every k-th round (e.g. 2 for T=100).")
    parser.add_argument("--skip-train", action="store_true", help="Require --checkpoint.")
    return parser.parse_args()


def resolve_topology_name(topology: str, *, seed: int) -> str:
    if topology == "complete":
        return "complete"
    if topology in {"ws_p_0.0", "ws_p_0.1"}:
        return f"{topology}_seed_{seed}"
    raise ValueError("topology must be one of: complete, ws_p_0.0, ws_p_0.1")


def default_output_path(
    *,
    communication_mode: str,
    num_nodes: int,
    signal_quality: float,
    topology: str,
    seed: int,
    fmt: str,
) -> Path:
    q_key = f"{signal_quality:.2f}".replace(".", "p")
    ext = ".html" if fmt == "html" else ".gif"
    return Path(
        f"artifacts/animations/{communication_mode}/n_{num_nodes}"
        f"/q_{q_key}/{topology}/seed_{seed}{ext}"
    )


def main() -> None:
    args = parse_args()
    device = resolve_runtime_device(args.device)
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
    graph_data = graphs[args.num_nodes][topology_key].to(device)
    topology_label = str(getattr(graph_data, "topology", args.topology))

    metadata = {
        "seed": args.seed,
        "num_nodes": args.num_nodes,
        "signal_quality": args.signal_quality,
        "topology": args.topology,
        "communication_mode": args.communication_mode,
        "communication_dim": args.communication_dim,
        "hidden_dim": args.hidden_dim,
        "num_heads": args.num_heads,
        "dropout": args.dropout,
        "max_horizon": max_horizon,
        "train_episodes": args.train_episodes,
    }

    if args.checkpoint is not None:
        model, metadata = load_checkpoint(args.checkpoint, device=device)
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        if args.skip_train:
            raise ValueError("--skip-train requires --checkpoint.")
        print(
            f"Training RGAT ({args.communication_mode}) on "
            f"n={args.num_nodes}, q={args.signal_quality}, {args.topology}, "
            f"T={max_horizon}, episodes={args.train_episodes} ..."
        )
        train_outcome = train_condition_model(
            graph_data=graph_data,
            train_episodes=args.train_episodes,
            max_horizon=max_horizon,
            signal_quality=args.signal_quality,
            hidden_dim=args.hidden_dim,
            num_heads=args.num_heads,
            communication_mode=args.communication_mode,
            communication_dim=args.communication_dim,
            learning_rate=args.learning_rate,
            seed=args.seed,
            device=device,
            dropout=args.dropout,
            weight_decay=args.weight_decay,
        )
        model = train_outcome.model
        model.eval()
        print(
            f"Training done. best_validation_error="
            f"{train_outcome.best_validation_error}"
        )
        q_key = f"{args.signal_quality:.2f}".replace(".", "p")
        checkpoint_path = args.save_checkpoint or Path(
            f"artifacts/checkpoints/{args.communication_mode}/n_{args.num_nodes}"
            f"/q_{q_key}/{args.topology}/seed_{args.seed}.pt"
        )
        save_checkpoint(model, checkpoint_path, metadata)
        print(f"Saved checkpoint: {checkpoint_path}")

    episode_seeds = [args.episode_seed + offset for offset in range(max(args.episode_variants, 1))]
    traces = []
    for episode_seed in episode_seeds:
        episode = sample_episode(
            num_nodes=args.num_nodes,
            max_horizon=max_horizon,
            signal_quality=args.signal_quality,
            seed=episode_seed,
        )
        traces.append(
            rollout_episode(
                model=model,
                graph_data=graph_data,
                episode=episode,
                device=device,
                signal_quality=args.signal_quality,
                topology=topology_label,
            )
        )
    trace = traces[0]
    nx_graph = pyg_to_networkx(graph_data.cpu())
    consensus = compute_unanimous_consensus(trace)

    outputs: list[Path] = []
    primary = args.output
    if primary is None:
        primary = default_output_path(
            communication_mode=args.communication_mode,
            num_nodes=args.num_nodes,
            signal_quality=args.signal_quality,
            topology=args.topology,
            seed=args.seed,
            fmt="html" if args.format in {"html", "both"} else "gif",
        )

    if args.format in {"html", "both"}:
        html_path = primary if primary.suffix == ".html" else primary.with_suffix(".html")
        outputs.append(
            save_interactive_episode_view(
                traces,
                nx_graph,
                html_path,
                episode_seeds=episode_seeds,
            )
        )
        print(f"Interactive viewer saved: {html_path}")

    if args.format in {"gif", "both"}:
        gif_path = primary if primary.suffix == ".gif" else primary.with_suffix(".gif")
        outputs.append(
            save_episode_animation(
                trace,
                nx_graph,
                gif_path,
                layout_seed=args.seed,
                fps=args.fps,
                frame_step=args.frame_step,
            )
        )
        print(f"GIF saved: {gif_path}")

    summary = {
        "theta": trace.theta,
        "max_horizon": trace.max_horizon,
        "final_error_rate": float(trace.error_rates[-1]),
        "consensus": consensus,
        "outputs": [str(p) for p in outputs],
        "metadata": metadata,
    }
    summary_path = outputs[0].with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Summary saved: {summary_path}")
    print(f"Episode θ={trace.theta}, final error={trace.error_rates[-1]:.1%}")
    if consensus["first_unanimous_t"] is not None:
        tag = "correct" if consensus["first_unanimous_correct"] else "wrong"
        print(f"First unanimous consensus at t={consensus['first_unanimous_t']} ({tag})")
    else:
        print("Unanimous consensus never reached in this episode.")


if __name__ == "__main__":
    main()
