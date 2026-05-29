"""Aggregated cross-seed plots comparing beta_GAT to the HST bound."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import math

from .hst_bound import compute_beta_hst_max


GroupKey = tuple[int, float, str]


def _finite(value: Any) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    if not math.isfinite(numeric):
        return None
    return numeric


def _bucket_records(records: list[dict[str, Any]]) -> dict[GroupKey, list[float]]:
    buckets: dict[GroupKey, list[float]] = defaultdict(list)
    for record in records:
        num_nodes = record.get("num_nodes")
        signal_quality = record.get("signal_quality")
        condition_name = record.get("condition_name") or record.get("topology")
        beta_gat = _finite(record.get("beta_gat"))
        if (
            beta_gat is None
            or not isinstance(num_nodes, int)
            or not isinstance(signal_quality, (int, float))
            or not isinstance(condition_name, str)
        ):
            continue
        buckets[(int(num_nodes), float(signal_quality), condition_name)].append(beta_gat)
    return buckets


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return mean, math.sqrt(variance)


def _topologies(buckets: dict[GroupKey, list[float]]) -> list[str]:
    return sorted({key[2] for key in buckets.keys()})


def save_beta_vs_q_plot(
    *,
    output_path: Path,
    records: list[dict[str, Any]],
    num_nodes_values: list[int],
) -> Path:
    """Plot beta_GAT (mean ± std over seeds) vs q for each topology and n."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    buckets = _bucket_records(records)
    topologies = _topologies(buckets)
    if not topologies:
        raise ValueError("No usable records to plot.")

    q_values = sorted({key[1] for key in buckets.keys()})

    n_cols = len(topologies)
    fig, axes = plt.subplots(1, n_cols, figsize=(5.5 * n_cols, 5), sharey=True)
    if n_cols == 1:
        axes = [axes]

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for ax, topology in zip(axes, topologies):
        for idx, num_nodes in enumerate(num_nodes_values):
            xs: list[float] = []
            ys: list[float] = []
            yerrs: list[float] = []
            for q in q_values:
                values = buckets.get((num_nodes, q, topology), [])
                if not values:
                    continue
                mean, std = _mean_std(values)
                xs.append(q)
                ys.append(mean)
                yerrs.append(std)
            if xs:
                ax.errorbar(
                    xs,
                    ys,
                    yerr=yerrs,
                    marker="o",
                    capsize=3,
                    linewidth=1.5,
                    color=color_cycle[idx % len(color_cycle)],
                    label=f"n={num_nodes}",
                )
        hst_xs = q_values
        hst_ys = [compute_beta_hst_max(q) for q in q_values]
        ax.plot(
            hst_xs,
            hst_ys,
            "--",
            color="#d62728",
            linewidth=2.0,
            label=r"$\beta_{\mathrm{HST}}^{\max}(q)$",
        )
        ax.set_title(topology)
        ax.set_xlabel(r"signal quality $q$")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)
    axes[0].set_ylabel(r"$\beta_{\mathrm{GAT}}$")
    fig.suptitle(r"Empirical learning rate vs signal quality (mean$\pm$std over seeds)")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_beta_vs_n_plot(
    *,
    output_path: Path,
    records: list[dict[str, Any]],
    signal_quality_values: list[float],
) -> Path:
    """Plot beta_GAT (mean ± std over seeds) vs n on log-x for each topology and q."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    buckets = _bucket_records(records)
    topologies = _topologies(buckets)
    if not topologies:
        raise ValueError("No usable records to plot.")

    n_values = sorted({key[0] for key in buckets.keys()})

    n_cols = len(topologies)
    fig, axes = plt.subplots(1, n_cols, figsize=(5.5 * n_cols, 5), sharey=True)
    if n_cols == 1:
        axes = [axes]

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for ax, topology in zip(axes, topologies):
        for idx, q in enumerate(signal_quality_values):
            xs: list[int] = []
            ys: list[float] = []
            yerrs: list[float] = []
            for n in n_values:
                values = buckets.get((n, q, topology), [])
                if not values:
                    continue
                mean, std = _mean_std(values)
                xs.append(n)
                ys.append(mean)
                yerrs.append(std)
            if xs:
                ax.errorbar(
                    xs,
                    ys,
                    yerr=yerrs,
                    marker="o",
                    capsize=3,
                    linewidth=1.5,
                    color=color_cycle[idx % len(color_cycle)],
                    label=f"q={q:.2f}",
                )
                hst_value = compute_beta_hst_max(q)
                ax.axhline(
                    hst_value,
                    linestyle="--",
                    linewidth=1.0,
                    color=color_cycle[idx % len(color_cycle)],
                    alpha=0.6,
                )
        ax.set_xscale("log")
        ax.set_title(topology)
        ax.set_xlabel("network size $n$ (log)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="best", fontsize=9)
    axes[0].set_ylabel(r"$\beta_{\mathrm{GAT}}$")
    fig.suptitle(
        r"Empirical learning rate vs network size (mean$\pm$std over seeds; "
        r"dashed lines = $\beta_{\mathrm{HST}}^{\max}$ per $q$)"
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def emit_seed_aggregate_plots(
    *,
    records: list[dict[str, Any]],
    output_dir: Path,
    num_nodes_list: list[int] | None = None,
    signal_quality_list: list[float] | None = None,
) -> dict[str, str]:
    """Write cross-seed beta vs q / beta vs n plots; returns paths or error keys."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if num_nodes_list is None:
        num_nodes_list = sorted(
            {
                int(record["num_nodes"])
                for record in records
                if isinstance(record.get("num_nodes"), int)
            }
        )
    if signal_quality_list is None:
        signal_quality_list = sorted(
            {
                float(record["signal_quality"])
                for record in records
                if isinstance(record.get("signal_quality"), (int, float))
            }
        )

    paths: dict[str, str] = {}
    try:
        beta_vs_q_path = save_beta_vs_q_plot(
            output_path=output_dir / "beta_vs_q.png",
            records=records,
            num_nodes_values=num_nodes_list,
        )
        paths["beta_vs_q"] = str(beta_vs_q_path)
    except Exception as exc:  # pragma: no cover
        paths["beta_vs_q_error"] = str(exc)

    try:
        beta_vs_n_path = save_beta_vs_n_plot(
            output_path=output_dir / "beta_vs_n.png",
            records=records,
            signal_quality_values=signal_quality_list,
        )
        paths["beta_vs_n"] = str(beta_vs_n_path)
    except Exception as exc:  # pragma: no cover
        paths["beta_vs_n_error"] = str(exc)

    return paths


__all__ = ["emit_seed_aggregate_plots", "save_beta_vs_n_plot", "save_beta_vs_q_plot"]
