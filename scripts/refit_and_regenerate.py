"""Refit t>=2 free alpha/beta metrics and regenerate all learning-rate plots."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.grid_refit import (
    refit_grid_artifacts,
    regenerate_cell_plots,
    remove_stale_plots,
    write_q_cell_summaries,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Refit beta with free alpha/beta/epsilon_inf from t=2, update metrics.json, "
            "regenerate seed plots, cell-aggregated plots, and latex figure copies."
        )
    )
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        required=True,
        help="Grid artifacts root containing n_*/q_*/seed_*/metrics.json",
    )
    parser.add_argument(
        "--communication-mode",
        type=str,
        choices=["fair_1bit", "vector"],
        default="fair_1bit",
        help="Used for latex figure channel folder name.",
    )
    parser.add_argument(
        "--latex-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "latex_figures",
        help="Root directory for LaTeX-ready figure copies.",
    )
    parser.add_argument(
        "--skip-seed-plots",
        action="store_true",
        help="Only update metrics.json, skip per-seed PNG regeneration.",
    )
    parser.add_argument(
        "--skip-summarize",
        action="store_true",
        help="Skip metrics CSV/aggregate regeneration.",
    )
    parser.add_argument(
        "--cleanup-only",
        action="store_true",
        help=(
            "Skip refit/plot regeneration; only remove legacy plots, refresh cell "
            "summaries, and rewrite top-level metrics CSV/JSON/plots."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    grid_root = args.artifacts_root.expanduser().resolve()
    channel_label = "fair" if args.communication_mode == "fair_1bit" else "vector"
    latex_root = args.latex_root.expanduser().resolve() if args.latex_root else None

    if args.cleanup_only:
        cleanup = remove_stale_plots(grid_root)
        q_summaries = write_q_cell_summaries(grid_root)
        cell_paths = regenerate_cell_plots(
            grid_root,
            latex_root=latex_root,
            channel_label=channel_label,
        )
        print(
            f"Cleanup: removed {cleanup['legacy_plots_removed']} legacy plots, "
            f"{cleanup['stale_plots_removed']} stale plots, "
            f"{cleanup['sidecar_summaries_removed']} sidecar summaries."
        )
        print(f"Wrote {q_summaries} q-cell summaries; refreshed {len(cell_paths)} cell/latex plots.")
    else:
        counts = refit_grid_artifacts(
            grid_root,
            latex_root=latex_root,
            channel_label=channel_label,
            regenerate_seed_plots=not args.skip_seed_plots,
        )
        print(
            f"Refitted {counts['conditions']} conditions across "
            f"{counts['metrics_files']} metrics files; "
            f"wrote {counts['cell_plots']} cell/latex plot paths; "
            f"removed {counts.get('legacy_plots_removed', 0)} legacy plots."
        )

    if args.skip_summarize:
        return

    suffix = channel_label
    parent = grid_root.parent.parent
    metrics_csv = parent / f"metrics_summary_{suffix}.csv"
    aggregate_csv = parent / f"metrics_summary_{suffix}_aggregated.csv"
    aggregate_json = aggregate_csv.with_suffix(".json")
    aggregate_plots_dir = grid_root / "aggregate_plots"

    summarize_script = PROJECT_ROOT / "scripts" / "summarize_metrics.py"
    cmd = [
        sys.executable,
        str(summarize_script),
        "--root",
        str(grid_root),
        "--csv",
        str(metrics_csv),
        "--aggregate-csv",
        str(aggregate_csv),
        "--aggregate-json",
        str(aggregate_json),
        "--aggregate-plots-dir",
        str(aggregate_plots_dir),
    ]
    subprocess.run(cmd, check=True)
    print(f"Metrics CSV: {metrics_csv}")
    print(f"Aggregated CSV: {aggregate_csv}")
    print(f"Aggregate plots: {aggregate_plots_dir}")


if __name__ == "__main__":
    main()
