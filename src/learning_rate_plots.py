"""Plots comparing empirical error decay with GAT fit and HST learning-rate bound."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from .training_pipeline import (
    DEFAULT_CONVERGENCE_WARNING_THRESHOLD,
    FIT_START_T,
    anchored_t2_decay_values,
    hst_alpha_at_t2_intersection,
)

PLOT_VARIANT = "anchored_t2"


def _build_learning_rate_suptitle(
    *,
    condition_key: str,
    signal_quality: float,
    beta_gap: float | None,
    exceeds_hst_bound: bool | None,
    convergence_warning: bool,
    epsilon_inf: float | None,
    fit_success: bool,
    convergence_warning_threshold: float,
    algorithm_label: str | None = None,
) -> str:
    """Assemble the figure suptitle including quality and bound flags."""
    anchor_label = rf"shared origin $t={FIT_START_T}$ (fit from $t\geq {FIT_START_T}$)"
    parts = [f"{condition_key}  |  q={signal_quality:.2f}  |  anchor={anchor_label}"]
    if algorithm_label:
        parts.insert(0, algorithm_label)

    if isinstance(beta_gap, float) and np.isfinite(beta_gap):
        parts.append(f"gap={beta_gap:.3f}")

    if convergence_warning:
        eps_label = (
            f"{epsilon_inf:.3f}"
            if isinstance(epsilon_inf, (int, float)) and np.isfinite(float(epsilon_inf))
            else "?"
        )
        parts.append(
            f"convergence warning (fitted "
            rf"$\varepsilon_\infty$={eps_label} > {convergence_warning_threshold:.2f})"
        )
        parts.append("bound comparison suppressed")
    elif isinstance(exceeds_hst_bound, bool):
        parts.append("exceeds bound" if exceeds_hst_bound else "within bound")
    elif not fit_success:
        parts.append("bound comparison n/a (fit failed)")

    if not fit_success:
        parts.append("fit failed")

    return "  |  ".join(parts)


def learning_rate_plot_path(
    *,
    artifacts_dir: Path,
    seed: int,
    condition_key: str,
) -> Path:
    """Return the PNG path for one condition's learning-rate plot."""
    safe_name = condition_key.replace("/", "__")
    return artifacts_dir / f"seed_{seed}" / "plots" / f"{safe_name}__{PLOT_VARIANT}.png"


def save_learning_rate_plot(
    *,
    output_path: Path,
    epsilon_series: list[float],
    beta_fit: dict[str, Any],
    beta_hst_max: float,
    condition_key: str,
    signal_quality: float,
    beta_gap: float | None = None,
    exceeds_hst_bound: bool | None = None,
    convergence_warning: bool = False,
    convergence_warning_threshold: float = DEFAULT_CONVERGENCE_WARNING_THRESHOLD,
    algorithm_label: str | None = None,
) -> Path:
    """
    Render a dual-panel decay plot for one condition.

    GAT free fit on t>=2; HST uses the same alpha and epsilon_inf so both curves meet at t=2.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not epsilon_series:
        raise ValueError("epsilon_series must be non-empty to plot learning rate.")

    fit_start_t = float(beta_fit.get("fit_start_t", FIT_START_T))
    empirical_full = np.asarray(epsilon_series, dtype=float)
    start_idx = int(fit_start_t) - 1
    if start_idx < 0 or start_idx >= len(empirical_full):
        raise ValueError(
            f"fit_start_t={fit_start_t} is out of range for epsilon_series length "
            f"{len(empirical_full)}."
        )
    empirical = empirical_full[start_idx:]
    t_values = np.arange(fit_start_t, len(epsilon_series) + 1, dtype=float)
    curve_t_plot = t_values.copy()

    fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(13, 5))

    fit_success = bool(beta_fit.get("fit_success", False))
    alpha = beta_fit.get("alpha")
    beta_gat = beta_fit.get("beta")
    epsilon_inf = beta_fit.get("epsilon_inf")
    fit_window_raw = beta_fit.get("fit_window_t_max")
    plateau_detected = bool(beta_fit.get("plateau_detected", False))
    fit_window_t_max = (
        int(fit_window_raw)
        if isinstance(fit_window_raw, (int, float)) and fit_window_raw > 0
        else len(epsilon_series)
    )
    can_plot_curves = (
        fit_success
        and isinstance(alpha, (int, float))
        and isinstance(beta_gat, (int, float))
        and isinstance(epsilon_inf, (int, float))
        and np.isfinite([alpha, beta_gat, epsilon_inf]).all()
    )

    ax_lin.plot(
        t_values,
        empirical,
        "o-",
        color="#1f77b4",
        linewidth=1.5,
        markersize=4,
        label=r"$\varepsilon(t)$ empirical",
    )
    floor = 1e-4
    if can_plot_curves:
        beta_gat_f = float(beta_gat)
        epsilon_inf_f = float(epsilon_inf)
        alpha_f = float(alpha)
        beta_hst_f = float(beta_hst_max)
        hst_alpha = hst_alpha_at_t2_intersection(gat_alpha=alpha_f)
        gat_curve = anchored_t2_decay_values(
            curve_t_plot,
            alpha=alpha_f,
            beta=beta_gat_f,
            epsilon_inf=epsilon_inf_f,
            fit_start_t=fit_start_t,
        )
        hst_curve = anchored_t2_decay_values(
            curve_t_plot,
            alpha=hst_alpha,
            beta=beta_hst_f,
            epsilon_inf=epsilon_inf_f,
            fit_start_t=fit_start_t,
        )
        anchor_tag = str(int(fit_start_t))
        shared_anchor_epsilon = float(gat_curve[0])
        ax_lin.plot(
            curve_t_plot,
            gat_curve,
            "--",
            color="#2ca02c",
            linewidth=2.0,
            label=(
                rf"GAT fit ($\beta_{{\mathrm{{GAT}}}}$={beta_gat_f:.3f}, "
                rf"free $\alpha,\beta$ from $t={anchor_tag}$)"
            ),
        )
        ax_lin.plot(
            curve_t_plot,
            hst_curve,
            "--",
            color="#d62728",
            linewidth=2.0,
            label=(
                rf"HST max slope ($\beta_{{\mathrm{{HST}}}}$={beta_hst_f:.3f}, "
                rf"$\alpha$ matched at $t={anchor_tag}$)"
            ),
        )
        ax_lin.plot(
            [fit_start_t],
            [shared_anchor_epsilon],
            "s",
            color="#9467bd",
            markersize=7,
            markerfacecolor="white",
            markeredgewidth=1.5,
            label=rf"shared origin $\varepsilon({int(fit_start_t)})={shared_anchor_epsilon:.3f}$",
            zorder=5,
        )
        anchor_residual = max(alpha_f, floor)

    if plateau_detected and 0 < fit_window_t_max < len(epsilon_series):
        ax_lin.axvline(
            x=float(fit_window_t_max),
            color="grey",
            linestyle=":",
            linewidth=1.0,
            label=f"fit window ($t \\leq {fit_window_t_max}$)",
        )
        ax_lin.axvspan(
            float(fit_window_t_max),
            float(t_values[-1]),
            color="grey",
            alpha=0.08,
        )
    threshold_style = {
        "color": "#ff7f0e",
        "linestyle": "-.",
        "linewidth": 1.8 if convergence_warning else 1.0,
        "alpha": 1.0 if convergence_warning else 0.55,
    }
    ax_lin.axhline(
        y=float(convergence_warning_threshold),
        label=rf"convergence threshold ($\varepsilon_\infty \leq "
        rf"{convergence_warning_threshold:.2f}$)",
        **threshold_style,
    )
    if (
        convergence_warning
        and can_plot_curves
        and isinstance(epsilon_inf, (int, float))
        and np.isfinite(float(epsilon_inf))
    ):
        epsilon_inf_f = float(epsilon_inf)
        ax_lin.axhline(
            y=epsilon_inf_f,
            color="#ff7f0e",
            linestyle=":",
            linewidth=1.2,
            alpha=0.9,
            label=rf"fitted $\varepsilon_\infty$={epsilon_inf_f:.3f}",
        )
    ax_lin.set_xlabel("Round $t$")
    ax_lin.set_ylabel(r"Error rate $\varepsilon(t)$")
    ax_lin.set_ylim(bottom=0.0)
    ax_lin.set_xlim(left=fit_start_t - 0.2)
    ax_lin.grid(True, alpha=0.3)
    ax_lin.legend(loc="best", fontsize=9)
    ax_lin.set_title("Empirical decay (linear)", fontsize=11)

    if can_plot_curves:
        residual = np.maximum(empirical - float(epsilon_inf), floor)
    else:
        residual = np.maximum(empirical, floor)

    ax_log.semilogy(
        t_values,
        residual,
        "o-",
        color="#1f77b4",
        linewidth=1.5,
        markersize=4,
        label=r"$\varepsilon(t)-\varepsilon_\infty$ empirical",
    )

    if can_plot_curves:
        beta_gat_f = float(beta_gat)
        beta_hst_f = float(beta_hst_max)
        epsilon_inf_f = float(epsilon_inf)
        alpha_f = float(alpha)
        slope_t_axis = curve_t_plot
        slope_base = max(alpha_f, floor)
        gat_slope_line = slope_base * np.exp(-beta_gat_f * (slope_t_axis - fit_start_t))
        hst_slope_line = slope_base * np.exp(-beta_hst_f * (slope_t_axis - fit_start_t))
        ax_log.semilogy(
            slope_t_axis,
            gat_slope_line,
            "--",
            color="#2ca02c",
            linewidth=2.0,
            label=rf"GAT slope ($\beta_{{\mathrm{{GAT}}}}$={beta_gat_f:.3f})",
        )
        ax_log.semilogy(
            slope_t_axis,
            hst_slope_line,
            "--",
            color="#d62728",
            linewidth=2.0,
            label=rf"HST max slope ($\beta_{{\mathrm{{HST}}}}$={beta_hst_f:.3f})",
        )
        ax_log.set_xlim(left=fit_start_t - 0.2)
    if plateau_detected and 0 < fit_window_t_max < len(epsilon_series):
        ax_log.axvline(
            x=float(fit_window_t_max),
            color="grey",
            linestyle=":",
            linewidth=1.0,
            label=f"fit window ($t \\leq {fit_window_t_max}$)",
        )
        ax_log.axvspan(
            float(fit_window_t_max),
            float(t_values[-1]),
            color="grey",
            alpha=0.08,
        )
    ax_log.set_xlabel("Round $t$")
    ax_log.set_ylabel(r"$\varepsilon(t)-\varepsilon_\infty$ (log scale)")
    ax_log.grid(True, which="both", alpha=0.3)
    ax_log.legend(loc="best", fontsize=9)
    ax_log.set_title("Slope view (log y) — slope = $-\\beta$", fontsize=11)
    empirical_min = float(np.min(residual))
    y_floor = max(empirical_min * 1e-3, 1e-6)
    y_ceiling = max(float(np.max(residual)) * 1.5, 1.0)
    ax_log.set_ylim(bottom=y_floor, top=y_ceiling)

    epsilon_inf_value = (
        float(epsilon_inf)
        if isinstance(epsilon_inf, (int, float)) and np.isfinite(float(epsilon_inf))
        else None
    )
    suptitle = _build_learning_rate_suptitle(
        condition_key=condition_key,
        signal_quality=signal_quality,
        beta_gap=beta_gap,
        exceeds_hst_bound=exceeds_hst_bound,
        convergence_warning=convergence_warning,
        epsilon_inf=epsilon_inf_value,
        fit_success=fit_success,
        convergence_warning_threshold=convergence_warning_threshold,
        algorithm_label=algorithm_label,
    )
    fig.suptitle(suptitle, fontsize=11)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def training_metrics_root(communication_mode: str) -> str:
    if communication_mode == "fair_1bit":
        return "training_metrics_fair"
    if communication_mode == "vector":
        return "training_metrics_vector"
    if communication_mode == "majority_vote":
        return "training_metrics_majority"
    raise ValueError(
        "communication_mode must be fair_1bit, vector, or majority_vote."
    )


def metrics_json_path(
    *,
    communication_mode: str,
    num_nodes: int,
    signal_quality: float,
    seed: int,
    project_root: Path | None = None,
) -> Path:
    q_key = f"{signal_quality:.2f}".replace(".", "p")
    root = training_metrics_root(communication_mode)
    rel = Path(
        f"artifacts/{root}/grid_runs/n_{num_nodes}/q_{q_key}/seed_{seed}/metrics.json"
    )
    return rel if project_root is None else project_root / rel


def condition_key_for_topology(*, num_nodes: int, topology: str, seed: int) -> str:
    topo_key = topology if topology == "complete" else f"{topology}_seed_{seed}"
    return f"n_{num_nodes}/{topo_key}"


def resolve_anchored_t2_plot_path(
    *,
    communication_mode: str,
    num_nodes: int,
    signal_quality: float,
    topology: str,
    seed: int,
    project_root: Path | None = None,
) -> Path | None:
    """Return the saved anchored_t2 PNG for a training run, if metrics exist."""
    metrics_path = metrics_json_path(
        communication_mode=communication_mode,
        num_nodes=num_nodes,
        signal_quality=signal_quality,
        seed=seed,
        project_root=project_root,
    )
    if not metrics_path.exists():
        return None
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    conditions = payload.get("conditions", {})
    if not isinstance(conditions, dict):
        return None
    condition_key = condition_key_for_topology(
        num_nodes=num_nodes,
        topology=topology,
        seed=seed,
    )
    condition = conditions.get(condition_key)
    if not isinstance(condition, dict):
        return None
    plot_raw = condition.get(f"learning_rate_plot_{PLOT_VARIANT}") or condition.get(
        "learning_rate_plot"
    )
    if not isinstance(plot_raw, str) or not plot_raw:
        return None
    plot_path = Path(plot_raw)
    if project_root is not None and not plot_path.is_absolute():
        plot_path = project_root / plot_path
    return plot_path if plot_path.exists() else None


def _convergence_warning_from_fit(
    beta_fit: dict[str, Any],
    *,
    threshold: float,
) -> bool:
    epsilon_inf = beta_fit.get("epsilon_inf")
    return bool(
        isinstance(epsilon_inf, (int, float))
        and np.isfinite(float(epsilon_inf))
        and float(epsilon_inf) > threshold
    )


def _beta_gap_and_exceed_from_fit(
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


def stage_per_episode_eval_plots(
    *,
    traces: list[Any],
    episode_seeds: list[int],
    html_path: str | Path,
    signal_quality: float,
    condition_key: str,
    algorithm_label: str | None = None,
) -> dict[int, str]:
    """Render one anchored_t2 PNG per signal episode from its rollout error curve."""
    from .hst_bound import compute_beta_hst_max
    from .training_pipeline import (
        DEFAULT_CONVERGENCE_WARNING_THRESHOLD,
        fit_beta_from_epsilon,
    )

    if len(traces) != len(episode_seeds):
        raise ValueError("traces and episode_seeds must have the same length.")

    html = Path(html_path)
    beta_hst_max = compute_beta_hst_max(float(signal_quality))
    mapping: dict[int, str] = {}
    for trace, seed in zip(traces, episode_seeds, strict=True):
        epsilon_series = [float(v) for v in trace.error_rates]
        beta_fit = fit_beta_from_epsilon(epsilon_series)
        convergence_warning = _convergence_warning_from_fit(
            beta_fit,
            threshold=DEFAULT_CONVERGENCE_WARNING_THRESHOLD,
        )
        beta_gap, exceeds_hst_bound = _beta_gap_and_exceed_from_fit(
            beta_fit,
            beta_hst_max=beta_hst_max,
            convergence_warning=convergence_warning,
        )
        dest = html.with_name(f"{html.stem}_ep{seed}__{PLOT_VARIANT}.png")
        save_learning_rate_plot(
            output_path=dest,
            epsilon_series=epsilon_series,
            beta_fit=beta_fit,
            beta_hst_max=beta_hst_max,
            condition_key=f"{condition_key} · signal ep={seed}",
            signal_quality=float(signal_quality),
            beta_gap=beta_gap,
            exceeds_hst_bound=exceeds_hst_bound,
            convergence_warning=convergence_warning,
            algorithm_label=algorithm_label,
        )
        mapping[int(seed)] = dest.name
    return mapping


def stage_anchored_t2_plot_for_viewer(
    *,
    communication_mode: str,
    num_nodes: int,
    signal_quality: float,
    topology: str,
    seed: int,
    html_path: str | Path,
    project_root: Path | None = None,
) -> str | None:
    """Copy the anchored_t2 evaluation PNG next to an HTML viewer."""
    source = resolve_anchored_t2_plot_path(
        communication_mode=communication_mode,
        num_nodes=num_nodes,
        signal_quality=signal_quality,
        topology=topology,
        seed=seed,
        project_root=project_root,
    )
    if source is None:
        return None
    html = Path(html_path)
    dest = html.with_name(f"{html.stem}__{PLOT_VARIANT}.png")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)
    return dest.name


__all__ = [
    "PLOT_VARIANT",
    "_build_learning_rate_suptitle",
    "condition_key_for_topology",
    "learning_rate_plot_path",
    "metrics_json_path",
    "resolve_anchored_t2_plot_path",
    "save_learning_rate_plot",
    "stage_anchored_t2_plot_for_viewer",
    "stage_per_episode_eval_plots",
    "training_metrics_root",
]
