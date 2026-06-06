"""Refit beta metrics and regenerate learning-rate plots from saved grid artifacts."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from src.hst_bound import compute_beta_hst_max
from src.learning_rate_plots import PLOT_VARIANT, save_learning_rate_plot
from src.training_pipeline import DEFAULT_CONVERGENCE_WARNING_THRESHOLD, fit_beta_from_epsilon

GRID_METRICS_RE = re.compile(
    r"(?:^|/)n_(\d+)/q_([0-9]+p[0-9]+)/seed_(\d+)/metrics\.json$"
)


def _finite_float(value: Any) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    if not np.isfinite(numeric):
        return None
    return numeric


def _parse_signal_quality(quality_key: str) -> float:
    match = re.fullmatch(r"([0-9]+)p([0-9]+)", quality_key)
    if not match:
        raise ValueError(f"Invalid q key: {quality_key!r}")
    return float(f"{match.group(1)}.{match.group(2)}")


def _convergence_warning(
    beta_fit: dict[str, Any],
    *,
    threshold: float = DEFAULT_CONVERGENCE_WARNING_THRESHOLD,
) -> bool:
    epsilon_inf = beta_fit.get("epsilon_inf")
    return bool(
        isinstance(epsilon_inf, (int, float))
        and np.isfinite(float(epsilon_inf))
        and float(epsilon_inf) > threshold
    )


def _beta_gap_and_exceed(
    beta_fit: dict[str, Any],
    *,
    beta_hst_max: float,
    convergence_warning: bool,
) -> tuple[float | None, bool | None]:
    if not bool(beta_fit.get("fit_success", False)):
        return None, None
    beta_raw = beta_fit.get("beta")
    if not isinstance(beta_raw, (int, float)) or not np.isfinite(float(beta_raw)):
        return None, None
    gap = float(beta_raw) - beta_hst_max
    exceeds = (gap > 0.0) and not convergence_warning
    return gap, exceeds


def refit_condition_metrics(
    condition: dict[str, Any],
    *,
    signal_quality: float,
) -> dict[str, Any]:
    """Return updated beta_fit / gap fields for one condition dict."""
    epsilon_series = condition.get("epsilon_series")
    if not isinstance(epsilon_series, list) or not epsilon_series:
        return {}

    beta_fit = fit_beta_from_epsilon([float(v) for v in epsilon_series])
    beta_hst_max = compute_beta_hst_max(float(signal_quality))
    convergence_warning = _convergence_warning(beta_fit)
    beta_gap, exceeds_hst_bound = _beta_gap_and_exceed(
        beta_fit,
        beta_hst_max=beta_hst_max,
        convergence_warning=convergence_warning,
    )
    return {
        "beta_fit": beta_fit,
        "beta_hst_max": beta_hst_max,
        "beta_gap": beta_gap,
        "exceeds_hst_bound": exceeds_hst_bound,
        "convergence_warning": convergence_warning,
    }


def mean_epsilon_series(series_list: list[list[float]]) -> list[float]:
    """Element-wise mean over equal-length epsilon series."""
    if not series_list:
        return []
    length = len(series_list[0])
    if any(len(series) != length for series in series_list):
        raise ValueError("All epsilon series must share the same length.")
    stacked = np.asarray(series_list, dtype=float)
    return [float(v) for v in np.mean(stacked, axis=0)]


def _topology_filename(condition_key: str) -> str:
    if "/" in condition_key:
        _, topology = condition_key.split("/", 1)
    else:
        topology = condition_key
    return re.sub(r"_seed_\d+$", "", topology)


def refit_metrics_file(
    metrics_path: Path,
    *,
    signal_quality: float,
    regenerate_seed_plots: bool = True,
) -> int:
    """Refit all conditions in one metrics.json; optionally rewrite seed plots."""
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    conditions = payload.get("conditions", {})
    if not isinstance(conditions, dict):
        raise ValueError(f"Invalid conditions in {metrics_path}")

    grid_match = GRID_METRICS_RE.search(metrics_path.as_posix())
    updated = 0

    for condition_key, condition in conditions.items():
        if not isinstance(condition, dict):
            continue
        refit = refit_condition_metrics(condition, signal_quality=signal_quality)
        if not refit:
            continue
        condition.update(refit)
        updated += 1

        if regenerate_seed_plots and isinstance(condition.get("epsilon_series"), list):
            plots_dir = metrics_path.parent / "plots"
            safe_name = condition_key.replace("/", "__")
            plot_path = plots_dir / f"{safe_name}__{PLOT_VARIANT}.png"
            save_learning_rate_plot(
                output_path=plot_path,
                epsilon_series=[float(v) for v in condition["epsilon_series"]],
                beta_fit=refit["beta_fit"],
                beta_hst_max=float(refit["beta_hst_max"]),
                condition_key=condition_key,
                signal_quality=signal_quality,
                beta_gap=refit["beta_gap"],
                exceeds_hst_bound=refit["exceeds_hst_bound"],
                convergence_warning=bool(refit["convergence_warning"]),
            )
            condition["learning_rate_plot"] = str(plot_path)
            condition[f"learning_rate_plot_{PLOT_VARIANT}"] = str(plot_path)

    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return updated


CellKey = tuple[int, str, str]  # num_nodes, q_key, condition_key


def _collect_cell_series(
    grid_root: Path,
) -> dict[CellKey, list[tuple[Path, list[float], float]]]:
    """Map (n, q, condition) -> list of (metrics_path, epsilon_series, signal_quality)."""
    buckets: dict[CellKey, list[tuple[Path, list[float], float]]] = defaultdict(list)
    for metrics_path in sorted(grid_root.rglob("metrics.json")):
        if not metrics_path.parent.name.startswith("seed_"):
            continue
        grid_match = GRID_METRICS_RE.search(metrics_path.as_posix())
        if not grid_match:
            continue
        num_nodes = int(grid_match.group(1))
        q_key = grid_match.group(2)
        signal_quality = _parse_signal_quality(q_key)
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        conditions = payload.get("conditions", {})
        if not isinstance(conditions, dict):
            continue
        for condition_key, condition in conditions.items():
            epsilon_series = condition.get("epsilon_series")
            if not isinstance(epsilon_series, list) or not epsilon_series:
                continue
            buckets[(num_nodes, q_key, str(condition_key))].append(
                (
                    metrics_path,
                    [float(v) for v in epsilon_series],
                    signal_quality,
                )
            )
    return buckets


def regenerate_cell_plots(
    grid_root: Path,
    *,
    latex_root: Path | None = None,
    channel_label: str = "fair",
) -> dict[str, str]:
    """Write cross-seed aggregated t2 plots per grid cell and optional latex copies."""
    buckets = _collect_cell_series(grid_root)
    written: dict[str, str] = {}

    for (num_nodes, q_key, condition_key), entries in sorted(buckets.items()):
        signal_quality = entries[0][2]
        mean_series = mean_epsilon_series([entry[1] for entry in entries])
        beta_fit = fit_beta_from_epsilon(mean_series)
        beta_hst_max = compute_beta_hst_max(signal_quality)
        convergence_warning = _convergence_warning(beta_fit)
        beta_gap, exceeds_hst_bound = _beta_gap_and_exceed(
            beta_fit,
            beta_hst_max=beta_hst_max,
            convergence_warning=convergence_warning,
        )

        cell_dir = grid_root / f"n_{num_nodes}" / f"q_{q_key}"
        plots_dir = cell_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        safe_name = condition_key.replace("/", "__")
        plot_path = plots_dir / f"{safe_name}__{PLOT_VARIANT}_aggregated.png"
        save_learning_rate_plot(
            output_path=plot_path,
            epsilon_series=mean_series,
            beta_fit=beta_fit,
            beta_hst_max=beta_hst_max,
            condition_key=f"{condition_key} (mean over {len(entries)} seeds)",
            signal_quality=signal_quality,
            beta_gap=beta_gap,
            exceeds_hst_bound=exceeds_hst_bound,
            convergence_warning=convergence_warning,
        )
        written[str(plot_path)] = str(plot_path)

        if latex_root is not None:
            topology = _topology_filename(condition_key)
            latex_path = latex_root / channel_label / f"n_{num_nodes}" / f"q_{q_key}" / f"{topology}.png"
            latex_path.parent.mkdir(parents=True, exist_ok=True)
            latex_path.write_bytes(plot_path.read_bytes())
            written[str(latex_path)] = str(latex_path)

    return written


def write_q_cell_summaries(grid_root: Path) -> int:
    """Write metrics_summary.csv/json per (n, q) cell from refitted seed metrics."""
    from scripts.summarize_metrics import TABLE_COLUMNS, load_metrics_records, write_csv
    from src.metrics_aggregate import build_seed_aggregate_summary

    written = 0
    for q_dir in sorted(grid_root.glob("n_*/q_*")):
        records: list[dict[str, Any]] = []
        for metrics_path in sorted(q_dir.glob("seed_*/metrics.json")):
            records.extend(load_metrics_records(metrics_path))
        if not records:
            continue
        write_csv(records, q_dir / "metrics_summary.csv", columns=TABLE_COLUMNS)
        summary = build_seed_aggregate_summary(records)
        (q_dir / "metrics_summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
        written += 1
    return written


def remove_stale_plots(grid_root: Path) -> dict[str, int]:
    """Delete obsolete plot variants and legacy metric sidecars."""
    counts = {
        "legacy_plots_removed": 0,
        "stale_plots_removed": 0,
        "sidecar_summaries_removed": 0,
    }

    for pattern in ("*anchored_t0*.png", "*anchored_t1*.png"):
        for plot_path in grid_root.rglob(pattern):
            plot_path.unlink(missing_ok=True)
            counts["legacy_plots_removed"] += 1

    for plot_path in grid_root.rglob("*_seed_*__anchored_t2_aggregated.png"):
        plot_path.unlink(missing_ok=True)
        counts["stale_plots_removed"] += 1

    for sidecar in grid_root.rglob("metrics_summary_t2.*"):
        sidecar.unlink(missing_ok=True)
        counts["sidecar_summaries_removed"] += 1

    return counts


def refit_grid_artifacts(
    grid_root: Path,
    *,
    latex_root: Path | None = None,
    channel_label: str = "fair",
    regenerate_seed_plots: bool = True,
) -> dict[str, int]:
    """Refit all metrics under grid_runs, regenerate seed + cell plots."""
    counts = {"metrics_files": 0, "conditions": 0, "cell_plots": 0}

    for metrics_path in sorted(grid_root.rglob("metrics.json")):
        if not metrics_path.parent.name.startswith("seed_"):
            continue
        q_match = re.search(r"/q_([0-9]+p[0-9]+)/", metrics_path.as_posix())
        if not q_match:
            continue
        signal_quality = _parse_signal_quality(q_match.group(1))
        updated = refit_metrics_file(
            metrics_path,
            signal_quality=signal_quality,
            regenerate_seed_plots=regenerate_seed_plots,
        )
        counts["metrics_files"] += 1
        counts["conditions"] += updated

    cell_paths = regenerate_cell_plots(
        grid_root,
        latex_root=latex_root,
        channel_label=channel_label,
    )
    counts["cell_plots"] = len(cell_paths)
    counts["q_summaries"] = write_q_cell_summaries(grid_root)
    counts.update(remove_stale_plots(grid_root))
    return counts


__all__ = [
    "PLOT_VARIANT",
    "mean_epsilon_series",
    "refit_condition_metrics",
    "refit_grid_artifacts",
    "refit_metrics_file",
    "regenerate_cell_plots",
    "remove_stale_plots",
    "write_q_cell_summaries",
]
