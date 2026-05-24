"""Aggregate condition-level metrics from all metrics.json artifacts into tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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


def discover_metrics_files(root: Path) -> list[Path]:
    return sorted(root.rglob("metrics.json"))


def load_metrics_records(metrics_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    conditions = payload.get("conditions", {})
    if not isinstance(conditions, dict):
        raise ValueError(f"Invalid metrics payload at {metrics_path}: 'conditions' must be a dict.")

    path_ctx = _parse_path_context(metrics_path)
    file_seed = payload.get("seed", path_ctx["seed"])
    records: list[dict[str, Any]] = []

    for condition_key, metrics in conditions.items():
        if not isinstance(metrics, dict):
            continue

        condition_match = CONDITION_KEY_RE.match(str(condition_key))
        num_nodes = path_ctx["num_nodes"]
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
                "signal_quality": path_ctx["signal_quality"],
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
            }
        )

    return records


AGGREGATE_GROUP_KEYS = ("num_nodes", "signal_quality", "topology")
AGGREGATE_COLUMNS = [
    "num_nodes",
    "signal_quality",
    "topology",
    "n_seeds",
    "beta_gat_mean",
    "beta_gat_std",
    "beta_gat_se_mean",
    "beta_gat_ci_lower_mean",
    "beta_gat_ci_upper_mean",
    "beta_gat_best",
    "beta_gat_best_seed",
    "beta_gap_at_best",
    "exceeds_hst_bound_at_best",
    "beta_hst_max_mean",
    "beta_gap_mean",
    "beta_gap_std",
    "exceeds_hst_bound_rate",
    "convergence_warning_rate",
    "fit_success_rate",
    "fit_r2_mean",
]


def _std(values: list[float]) -> float | None:
    if len(values) < 2:
        return 0.0 if len(values) == 1 else None
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def _best_run_by_beta_gat(group: list[dict[str, Any]]) -> dict[str, Any]:
    """Pick the seed run with the highest empirical learning rate beta_gat."""
    best_record: dict[str, Any] | None = None
    best_beta: float | None = None
    for item in group:
        beta = item.get("beta_gat")
        if not isinstance(beta, (int, float)) or not math.isfinite(float(beta)):
            continue
        beta_value = float(beta)
        if best_beta is None or beta_value > best_beta:
            best_beta = beta_value
            best_record = item

    if best_record is None or best_beta is None:
        return {
            "beta_gat_best": None,
            "beta_gat_best_seed": None,
            "beta_gap_at_best": None,
            "exceeds_hst_bound_at_best": None,
        }

    beta_gap = best_record.get("beta_gap")
    beta_gap_value = (
        float(beta_gap)
        if isinstance(beta_gap, (int, float)) and math.isfinite(float(beta_gap))
        else None
    )
    exceeds = best_record.get("exceeds_hst_bound")
    return {
        "beta_gat_best": best_beta,
        "beta_gat_best_seed": best_record.get("seed"),
        "beta_gap_at_best": beta_gap_value,
        "exceeds_hst_bound_at_best": exceeds if isinstance(exceeds, bool) else None,
    }


def aggregate_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mean/std and best-seed stats for each (num_nodes, signal_quality, topology) cell."""
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("seed") is None:
            continue
        key = tuple(record.get(field) for field in AGGREGATE_GROUP_KEYS)
        buckets[key].append(record)

    aggregated: list[dict[str, Any]] = []
    for key, group in sorted(buckets.items(), key=lambda item: item[0]):
        beta_gat_values = [
            float(item["beta_gat"])
            for item in group
            if isinstance(item.get("beta_gat"), (int, float))
            and math.isfinite(float(item["beta_gat"]))
        ]
        beta_gat_se_values = [
            float(item["beta_gat_se"])
            for item in group
            if isinstance(item.get("beta_gat_se"), (int, float))
            and math.isfinite(float(item["beta_gat_se"]))
        ]
        beta_gat_ci_lower_values = [
            float(item["beta_gat_ci_lower"])
            for item in group
            if isinstance(item.get("beta_gat_ci_lower"), (int, float))
            and math.isfinite(float(item["beta_gat_ci_lower"]))
        ]
        beta_gat_ci_upper_values = [
            float(item["beta_gat_ci_upper"])
            for item in group
            if isinstance(item.get("beta_gat_ci_upper"), (int, float))
            and math.isfinite(float(item["beta_gat_ci_upper"]))
        ]
        beta_gap_values = [
            float(item["beta_gap"])
            for item in group
            if isinstance(item.get("beta_gap"), (int, float))
            and math.isfinite(float(item["beta_gap"]))
        ]
        beta_hst_values = [
            float(item["beta_hst_max"])
            for item in group
            if isinstance(item.get("beta_hst_max"), (int, float))
            and math.isfinite(float(item["beta_hst_max"]))
        ]
        r2_values = [
            float(item["fit_r2"])
            for item in group
            if isinstance(item.get("fit_r2"), (int, float))
            and math.isfinite(float(item["fit_r2"]))
        ]
        exceeds_values = [
            bool(item["exceeds_hst_bound"])
            for item in group
            if isinstance(item.get("exceeds_hst_bound"), bool)
        ]
        convergence_warning_values = [
            bool(item["convergence_warning"])
            for item in group
            if isinstance(item.get("convergence_warning"), bool)
        ]
        fit_success_values = [
            bool(item["fit_success"])
            for item in group
            if isinstance(item.get("fit_success"), bool)
        ]

        num_nodes, signal_quality, topology = key
        best_run = _best_run_by_beta_gat(group)
        aggregated.append(
            {
                "num_nodes": num_nodes,
                "signal_quality": signal_quality,
                "topology": topology,
                "n_seeds": len({item["seed"] for item in group}),
                "beta_gat_mean": (
                    sum(beta_gat_values) / len(beta_gat_values) if beta_gat_values else None
                ),
                "beta_gat_std": _std(beta_gat_values),
                "beta_gat_se_mean": (
                    sum(beta_gat_se_values) / len(beta_gat_se_values)
                    if beta_gat_se_values
                    else None
                ),
                "beta_gat_ci_lower_mean": (
                    sum(beta_gat_ci_lower_values) / len(beta_gat_ci_lower_values)
                    if beta_gat_ci_lower_values
                    else None
                ),
                "beta_gat_ci_upper_mean": (
                    sum(beta_gat_ci_upper_values) / len(beta_gat_ci_upper_values)
                    if beta_gat_ci_upper_values
                    else None
                ),
                **best_run,
                "beta_hst_max_mean": (
                    sum(beta_hst_values) / len(beta_hst_values) if beta_hst_values else None
                ),
                "beta_gap_mean": (
                    sum(beta_gap_values) / len(beta_gap_values) if beta_gap_values else None
                ),
                "beta_gap_std": _std(beta_gap_values),
                "exceeds_hst_bound_rate": (
                    sum(1 for value in exceeds_values if value) / len(exceeds_values)
                    if exceeds_values
                    else None
                ),
                "convergence_warning_rate": (
                    sum(1 for value in convergence_warning_values if value)
                    / len(convergence_warning_values)
                    if convergence_warning_values
                    else None
                ),
                "fit_success_rate": (
                    sum(1 for value in fit_success_values if value) / len(fit_success_values)
                    if fit_success_values
                    else None
                ),
                "fit_r2_mean": sum(r2_values) / len(r2_values) if r2_values else None,
            }
        )
    return aggregated


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

    if args.aggregate_csv is not None or args.aggregate_markdown is not None:
        aggregated = aggregate_records(records)
        if args.aggregate_csv is not None:
            write_csv(aggregated, args.aggregate_csv, columns=AGGREGATE_COLUMNS)
            print(f"Wrote {len(aggregated)} aggregated rows to {args.aggregate_csv}")
        if args.aggregate_markdown is not None:
            write_markdown(aggregated, args.aggregate_markdown, columns=AGGREGATE_COLUMNS)
            print(f"Wrote aggregated markdown table to {args.aggregate_markdown}")


if __name__ == "__main__":
    main()
