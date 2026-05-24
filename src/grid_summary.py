"""Aggregate grid run metrics into summary JSON and plots."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from src.aggregate_plots import save_beta_vs_n_plot, save_beta_vs_q_plot
from src.grid_tasks import format_signal_quality_label
from src.reporting import classify_regimes

GRID_METRICS_RE = re.compile(
    r"n_(\d+)/q_([0-9]+p[0-9]+)/seed_(\d+)/metrics\.json$"
)


def normalize_condition_name(condition_key: str) -> str:
    if "/" in condition_key:
        _, condition_name = condition_key.split("/", 1)
    else:
        condition_name = condition_key
    return re.sub(r"_seed_\d+$", "", condition_name)


def _finite_float(value: Any) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    if not math.isfinite(numeric):
        return None
    return numeric


def _safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _safe_bool_ratio(values: list[bool]) -> float | None:
    if not values:
        return None
    return float(sum(1 for item in values if item) / len(values))


def _new_aggregate_bucket() -> dict[str, Any]:
    return {
        "beta_gat_values": [],
        "beta_gap_values": [],
        "exceeds_values": [],
        "convergence_warning_values": [],
        "fit_success_values": [],
        "artifact_paths": set(),
        "num_records": 0,
    }


def aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
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
                bucket_dict[key] = _new_aggregate_bucket()
            bucket = bucket_dict[key]
            bucket["num_records"] += 1

            beta_gat = _finite_float(record.get("beta_gat"))
            if beta_gat is not None:
                bucket["beta_gat_values"].append(beta_gat)

            beta_gap = _finite_float(record.get("beta_gap"))
            if beta_gap is not None:
                bucket["beta_gap_values"].append(beta_gap)

            exceeds = record.get("exceeds_hst_bound")
            if isinstance(exceeds, bool):
                bucket["exceeds_values"].append(exceeds)

            convergence_warning = record.get("convergence_warning")
            if isinstance(convergence_warning, bool):
                bucket["convergence_warning_values"].append(convergence_warning)

            fit_success = record.get("fit_success")
            if isinstance(fit_success, bool):
                bucket["fit_success_values"].append(fit_success)

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
                "num_fit_success_values": len(bucket["fit_success_values"]),
                "mean_beta_gat": _safe_mean(bucket["beta_gat_values"]),
                "mean_beta_gap": _safe_mean(bucket["beta_gap_values"]),
                "proportion_exceeds_hst_bound": _safe_bool_ratio(bucket["exceeds_values"]),
                "convergence_warning_rate": _safe_bool_ratio(
                    bucket["convergence_warning_values"]
                ),
                "fit_success_rate": _safe_bool_ratio(bucket["fit_success_values"]),
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


def _parse_signal_quality_from_key(quality_key: str) -> float:
    major, minor = quality_key.split("p", 1)
    return float(f"{major}.{minor}")


def metrics_to_record(
    *,
    seed: int,
    num_nodes: int,
    signal_quality: float,
    setting_key: str,
    condition_key: str,
    metrics: dict[str, Any],
    artifact_path: str,
) -> dict[str, Any]:
    beta_fit = metrics.get("beta_fit", {})
    beta_gat = beta_fit.get("beta") if isinstance(beta_fit, dict) else None
    fit_success = beta_fit.get("fit_success") if isinstance(beta_fit, dict) else None
    convergence_warning = metrics.get("convergence_warning")
    return {
        "seed": seed,
        "num_nodes": num_nodes,
        "signal_quality": signal_quality,
        "setting_key": setting_key,
        "condition_key": condition_key,
        "condition_name": normalize_condition_name(condition_key),
        "beta_gat": beta_gat,
        "beta_hst_max": metrics.get("beta_hst_max"),
        "beta_gap": metrics.get("beta_gap"),
        "exceeds_hst_bound": metrics.get("exceeds_hst_bound"),
        "convergence_warning": (
            convergence_warning if isinstance(convergence_warning, bool) else None
        ),
        "fit_success": fit_success if isinstance(fit_success, bool) else None,
        "artifact_path": artifact_path,
    }


def collect_records_from_artifacts(artifacts_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for metrics_path in sorted(artifacts_root.rglob("metrics.json")):
        match = GRID_METRICS_RE.search(metrics_path.as_posix())
        if not match:
            continue
        num_nodes = int(match.group(1))
        quality_key = match.group(2)
        seed = int(match.group(3))
        signal_quality = _parse_signal_quality_from_key(quality_key)
        setting_key = f"n_{num_nodes}/q_{format_signal_quality_label(signal_quality)}"
        condition_metrics = _load_condition_metrics(metrics_path)
        for condition_key, metrics in condition_metrics.items():
            records.append(
                metrics_to_record(
                    seed=seed,
                    num_nodes=num_nodes,
                    signal_quality=signal_quality,
                    setting_key=setting_key,
                    condition_key=condition_key,
                    metrics=metrics,
                    artifact_path=str(metrics_path),
                )
            )
    return records


def emit_aggregate_plots(
    *,
    records: list[dict[str, Any]],
    artifacts_root: Path,
    num_nodes_list: list[int],
    signal_quality_list: list[float],
) -> dict[str, str]:
    plots_dir = artifacts_root / "aggregate_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, str] = {}
    try:
        beta_vs_q_path = save_beta_vs_q_plot(
            output_path=plots_dir / "beta_vs_q.png",
            records=records,
            num_nodes_values=num_nodes_list,
        )
        paths["beta_vs_q"] = str(beta_vs_q_path)
    except Exception as exc:  # pragma: no cover - plotting is best-effort
        paths["beta_vs_q_error"] = str(exc)

    try:
        beta_vs_n_path = save_beta_vs_n_plot(
            output_path=plots_dir / "beta_vs_n.png",
            records=records,
            signal_quality_values=signal_quality_list,
        )
        paths["beta_vs_n"] = str(beta_vs_n_path)
    except Exception as exc:  # pragma: no cover - plotting is best-effort
        paths["beta_vs_n_error"] = str(exc)

    return paths


def build_grid_summary(
    *,
    records: list[dict[str, Any]],
    grid_config: dict[str, Any],
    run_summaries: list[dict[str, Any]] | None = None,
    artifacts_root: Path,
    num_nodes_list: list[int],
    signal_quality_list: list[float],
) -> dict[str, Any]:
    aggregates = aggregate_records(records)
    regime_classification = classify_regimes(aggregates)
    aggregate_plot_paths = emit_aggregate_plots(
        records=records,
        artifacts_root=artifacts_root,
        num_nodes_list=num_nodes_list,
        signal_quality_list=signal_quality_list,
    )
    return {
        "grid_config": grid_config,
        "num_runs": len(run_summaries) if run_summaries is not None else None,
        "num_condition_records": len(records),
        "run_summaries": run_summaries or [],
        "aggregates": aggregates,
        "regime_classification": regime_classification,
        "aggregate_plots": aggregate_plot_paths,
    }


__all__ = [
    "aggregate_records",
    "build_grid_summary",
    "collect_records_from_artifacts",
    "emit_aggregate_plots",
    "metrics_to_record",
    "normalize_condition_name",
]
