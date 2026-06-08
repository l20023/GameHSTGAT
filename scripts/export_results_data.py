"""Export aggregated metrics JSON for the GitHub Pages results browser."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_OUTPUT = PROJECT_ROOT / "docs" / "results-data.json"
DEFAULT_SOURCES: dict[str, Path] = {
    "fair_1bit": PROJECT_ROOT / "artifacts" / "metrics_summary_fair_aggregated.csv",
    "vector": PROJECT_ROOT / "artifacts" / "metrics_summary_vector_aggregated.csv",
    "majority_vote": PROJECT_ROOT / "artifacts" / "metrics_summary_majority_aggregated.csv",
}


def _float_or_none(value: str) -> float | None:
    text = value.strip()
    if not text or text.lower() in {"nan", "none", ""}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _bool_or_none(value: str) -> bool | None:
    text = value.strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def load_aggregated_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "num_nodes": int(row["num_nodes"]),
                    "signal_quality": float(row["signal_quality"]),
                    "topology": row["topology"],
                    "n_seeds": int(row["n_seeds"]),
                    "beta_gat_mean": _float_or_none(row.get("beta_gat_mean", "")),
                    "beta_gat_std": _float_or_none(row.get("beta_gat_std", "")),
                    "beta_hst_max_mean": _float_or_none(row.get("beta_hst_max_mean", "")),
                    "beta_gap_mean": _float_or_none(row.get("beta_gap_mean", "")),
                    "exceeds_hst_bound_rate": _float_or_none(row.get("exceeds_hst_bound_rate", "")),
                    "convergence_warning_rate": _float_or_none(row.get("convergence_warning_rate", "")),
                    "fit_success_rate": _float_or_none(row.get("fit_success_rate", "")),
                }
            )
    return rows


def build_payload(sources: dict[str, Path]) -> dict[str, Any]:
    algorithms: dict[str, list[dict[str, Any]]] = {}
    for algorithm, path in sources.items():
        algorithms[algorithm] = load_aggregated_csv(path)
    return {
        "algorithms": algorithms,
        "aggregate_plot_paths": {
            "fair_1bit": {
                "beta_vs_q": "artifacts/training_metrics_fair/grid_runs/aggregate_plots/beta_vs_q.png",
                "beta_vs_n": "artifacts/training_metrics_fair/grid_runs/aggregate_plots/beta_vs_n.png",
            },
            "vector": {
                "beta_vs_q": "artifacts/training_metrics_vector/grid_runs/aggregate_plots/beta_vs_q.png",
                "beta_vs_n": "artifacts/training_metrics_vector/grid_runs/aggregate_plots/beta_vs_n.png",
            },
            "majority_vote": {
                "beta_vs_q": "artifacts/training_metrics_majority/grid_runs/aggregate_plots/beta_vs_q.png",
                "beta_vs_n": "artifacts/training_metrics_majority/grid_runs/aggregate_plots/beta_vs_n.png",
            },
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export docs/results-data.json for Pages.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_payload(DEFAULT_SOURCES)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    counts = {key: len(value) for key, value in payload["algorithms"].items()}
    print(f"Wrote {args.output} ({counts})")


if __name__ == "__main__":
    main()
