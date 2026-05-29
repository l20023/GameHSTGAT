"""Aggregate condition-level metrics from all metrics.json artifacts into tables."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.aggregate_plots import emit_seed_aggregate_plots
from src.metrics_aggregate import (
    AGGREGATE_COLUMNS,
    aggregate_records,
    build_seed_aggregate_summary,
)

CONDITION_KEY_RE = re.compile(r"^n_(\d+)/(.+)$")
GRID_PATH_RE = re.compile(
    r"(?:^|/)n_(\d+)/q_([0-9]+p[0-9]+)/seed_(\d+)/metrics\.json$"
)
SEED_PATH_RE = re.compile(r"seed_(\d+)/metrics\.json$")


def _normalize_condition_name(condition_key: str) -> str:
    if "/" in condition_key:
        _, condition_name = condition_key.split("/", 1)
    else:
        condition_name = condition_key
    return re.sub(r"_seed_\d+$", "", condition_name)


def _parse_signal_quality_from_key(quality_key: str) -> float | None:
    match = re.fullmatch(r"([0-9]+)p([0-9]+)", quality_key)
    if not match:
        return None
    return float(f"{match.group(1)}.{match.group(2)}")


def _parse_path_context(metrics_path: Path) -> dict[str, Any]:
    posix = metrics_path.as_posix()
    seed: int | None = None
    num_nodes: int | None = None
    signal_quality: float | None = None
    layout = "unknown"

    grid_match = GRID_PATH_RE.search(posix)
    if grid_match:
        layout = "grid"
        num_nodes = int(grid_match.group(1))
        signal_quality = _parse_signal_quality_from_key(grid_match.group(2))
        seed = int(grid_match.group(3))
    else:
        seed_match = SEED_PATH_RE.search(posix)
        if seed_match:
            layout = "flat"
            seed = int(seed_match.group(1))

    return {
        "layout": layout,
        "seed": seed,
        "num_nodes": num_nodes,
        "signal_quality": signal_quality,
        "metrics_path": str(metrics_path),
    }


def _finite_float(value: Any) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    if numeric != numeric:  # NaN
        return None
    return numeric


def _consensus_record_fields(metrics: dict[str, Any]) -> dict[str, float | None]:
    """Flatten unanimous/majority consensus scalars for tabular export."""
    fields: dict[str, float | None] = {}
    consensus = metrics.get("consensus", {})
    if not isinstance(consensus, dict):
        return fields
    for mode in ("unanimous", "majority"):
        mode_metrics = consensus.get(mode, {})
        if not isinstance(mode_metrics, dict):
            continue
        fields[f"{mode}_reach_consensus_rate"] = _finite_float(
            mode_metrics.get("fraction_episodes_reach_consensus")
        )
        fields[f"{mode}_correct_consensus_rate"] = _finite_float(
            mode_metrics.get("fraction_episodes_consensus_correct")
        )
        fields[f"{mode}_wrong_only_consensus_rate"] = _finite_float(
            mode_metrics.get("fraction_episodes_consensus_wrong_only")
        )
        fields[f"{mode}_correct_at_first_consensus_rate"] = _finite_float(
            mode_metrics.get("fraction_correct_at_first_consensus")
        )
        fields[f"{mode}_mean_first_consensus_t"] = _finite_float(
            mode_metrics.get("mean_first_consensus_t")
        )
    return fields


def discover_metrics_files(root: Path) -> list[Path]:
    return sorted(root.rglob("metrics.json"))


def load_metrics_records(metrics_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    conditions = payload.get("conditions", {})
    if not isinstance(conditions, dict):
        raise ValueError(f"Invalid metrics payload at {metrics_path}: 'conditions' must be a dict.")

    path_ctx = _parse_path_context(metrics_path)
    file_seed = payload.get("seed", path_ctx["seed"])
    file_signal_quality = payload.get("signal_quality", path_ctx["signal_quality"])
    file_num_nodes = payload.get("num_nodes", path_ctx["num_nodes"])
    records: list[dict[str, Any]] = []

    for condition_key, metrics in conditions.items():
        if not isinstance(metrics, dict):
            continue

        condition_match = CONDITION_KEY_RE.match(str(condition_key))
        num_nodes = file_num_nodes
        topology = _normalize_condition_name(str(condition_key))
        if condition_match:
            num_nodes = int(condition_match.group(1))
            topology = _normalize_condition_name(str(condition_key))

        beta_fit = metrics.get("beta_fit", {})
        if not isinstance(beta_fit, dict):
            beta_fit = {}

        beta_gat = _finite_float(beta_fit.get("beta"))
        beta_hst = _finite_float(metrics.get("beta_hst_max"))
        beta_gap = metrics.get("beta_gap")
        beta_gap_value = _finite_float(beta_gap) if beta_gap is not None else None

        exceeds = metrics.get("exceeds_hst_bound")
        exceeds_bool = exceeds if isinstance(exceeds, bool) else None

        convergence_warning = metrics.get("convergence_warning")
        convergence_warning_bool = (
            convergence_warning if isinstance(convergence_warning, bool) else None
        )

        records.append(
            {
                "seed": file_seed,
                "num_nodes": num_nodes,
                "signal_quality": file_signal_quality,
                "topology": topology,
                "condition_key": condition_key,
                "beta_gat": beta_gat,
                "beta_gat_se": _finite_float(beta_fit.get("beta_std")),
                "beta_gat_ci_lower": _finite_float(beta_fit.get("beta_ci_lower")),
                "beta_gat_ci_upper": _finite_float(beta_fit.get("beta_ci_upper")),
                "beta_hst_max": beta_hst,
                "beta_gap": beta_gap_value,
                "exceeds_hst_bound": exceeds_bool,
                "convergence_warning": convergence_warning_bool,
                "fit_method": beta_fit.get("method"),
                "fit_success": beta_fit.get("fit_success"),
                "fit_rmse": _finite_float(beta_fit.get("rmse")),
                "fit_r2": _finite_float(beta_fit.get("r2")),
                "fit_failure_reason": beta_fit.get("failure_reason") or "",
                "train_loss_final": _finite_float(metrics.get("train_loss_final")),
                "layout": path_ctx["layout"],
                "metrics_path": str(metrics_path),
                **_consensus_record_fields(metrics),
            }
        )

    return records


def collect_records(roots: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for root in roots:
        if not root.exists():
            raise FileNotFoundError(f"Artifacts root does not exist: {root}")
        for metrics_path in discover_metrics_files(root):
            records.extend(load_metrics_records(metrics_path))
    return records


TABLE_COLUMNS = [
    "seed",
    "num_nodes",
    "signal_quality",
    "topology",
    "beta_gat",
    "beta_gat_se",
    "beta_gat_ci_lower",
    "beta_gat_ci_upper",
    "beta_hst_max",
    "beta_gap",
    "exceeds_hst_bound",
    "convergence_warning",
    "fit_method",
    "fit_success",
    "fit_rmse",
    "fit_r2",
    "train_loss_final",
    "unanimous_reach_consensus_rate",
    "unanimous_correct_consensus_rate",
    "unanimous_wrong_only_consensus_rate",
    "unanimous_mean_first_consensus_t",
    "majority_reach_consensus_rate",
    "majority_correct_consensus_rate",
    "majority_wrong_only_consensus_rate",
    "majority_mean_first_consensus_t",
    "condition_key",
    "metrics_path",
]


def _format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_csv(
    records: list[dict[str, Any]], output_path: Path, *, columns: list[str] | None = None
) -> None:
    fieldnames = columns or TABLE_COLUMNS
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow({column: _format_cell(record.get(column)) for column in fieldnames})


def write_markdown(
    records: list[dict[str, Any]], output_path: Path, *, columns: list[str] | None = None
) -> None:
    fieldnames = columns or TABLE_COLUMNS
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = "| " + " | ".join(fieldnames) + " |"
    separator = "| " + " | ".join("---" for _ in fieldnames) + " |"
    lines = [header, separator]
    for record in records:
        row = "| " + " | ".join(_format_cell(record.get(column)) for column in fieldnames) + " |"
        lines.append(row)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_json_summary(records: list[dict[str, Any]], output_path: Path) -> dict[str, Any]:
    summary = {
        "num_records": len(records),
        "num_metrics_files": len({record["metrics_path"] for record in records}),
        "records": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize all metrics.json files under artifact directories."
    )
    parser.add_argument(
        "--root",
        action="append",
        default=None,
        help="Artifact root to scan (repeatable). Default: artifacts/training_metrics",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "metrics_summary.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=None,
        help="Optional Markdown table output path",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Optional JSON output with full record list",
    )
    parser.add_argument(
        "--aggregate-csv",
        type=Path,
        default=None,
        help="Write seed-aggregated table (mean/std per topology and setting)",
    )
    parser.add_argument(
        "--aggregate-markdown",
        type=Path,
        default=None,
        help="Optional markdown table for seed-aggregated summary",
    )
    parser.add_argument(
        "--aggregate-json",
        type=Path,
        default=None,
        help="Write seed-aggregated summary JSON (cells keyed by n/q/topology)",
    )
    parser.add_argument(
        "--aggregate-plots-dir",
        type=Path,
        default=None,
        help="Write beta_vs_q.png and beta_vs_n.png from aggregated records",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    roots = [Path(item) for item in args.root] if args.root else [PROJECT_ROOT / "artifacts" / "training_metrics"]
    records = collect_records(roots)
    if not records:
        print(f"No metrics.json files found under: {', '.join(str(root) for root in roots)}")
        return

    write_csv(records, args.csv)
    print(f"Wrote {len(records)} rows to {args.csv}")

    if args.markdown is not None:
        write_markdown(records, args.markdown)
        print(f"Wrote markdown table to {args.markdown}")

    if args.json is not None:
        write_json_summary(records, args.json)
        print(f"Wrote JSON summary to {args.json}")

    need_aggregate = any(
        path is not None
        for path in (
            args.aggregate_csv,
            args.aggregate_markdown,
            args.aggregate_json,
            args.aggregate_plots_dir,
        )
    )
    if need_aggregate:
        aggregated = aggregate_records(records)
        if args.aggregate_csv is not None:
            write_csv(aggregated, args.aggregate_csv, columns=AGGREGATE_COLUMNS)
            print(f"Wrote {len(aggregated)} aggregated rows to {args.aggregate_csv}")
        if args.aggregate_markdown is not None:
            write_markdown(aggregated, args.aggregate_markdown, columns=AGGREGATE_COLUMNS)
            print(f"Wrote aggregated markdown table to {args.aggregate_markdown}")
        if args.aggregate_json is not None:
            summary = build_seed_aggregate_summary(records)
            args.aggregate_json.parent.mkdir(parents=True, exist_ok=True)
            args.aggregate_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(f"Wrote seed aggregate JSON to {args.aggregate_json}")
        if args.aggregate_plots_dir is not None:
            plot_paths = emit_seed_aggregate_plots(
                records=records,
                output_dir=args.aggregate_plots_dir,
            )
            for key, path in plot_paths.items():
                print(f"  {key}: {path}")


if __name__ == "__main__":
    main()
