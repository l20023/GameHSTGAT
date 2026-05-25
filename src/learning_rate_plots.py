"""Plots comparing empirical error decay with GAT fit and HST learning-rate bound."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np

from .training_pipeline import (
    DEFAULT_CONVERGENCE_WARNING_THRESHOLD,
    FitAnchor,
    PRIOR_EPSILON_AT_T0,
    anchored_t0_decay_values,
    anchored_t1_decay_values,
    normalize_fit_anchor,
)

PlotVariant = Literal["anchored_t1", "anchored_t0"]


def _decay_curve(
    t_values: np.ndarray,
    *,
    beta: float,
    epsilon_inf: float,
    anchor: FitAnchor,
    epsilon_1: float,
) -> np.ndarray:
    if anchor == "t1":
        return anchored_t1_decay_values(
            t_values, beta=beta, epsilon_inf=epsilon_inf, epsilon_1=epsilon_1
        )
    return anchored_t0_decay_values(t_values, beta=beta, epsilon_inf=epsilon_inf)


def _resolve_plot_anchor(
    beta_fit: dict[str, Any],
    plot_variant: PlotVariant,
    *,
    fit_anchor: FitAnchor | None = None,
) -> FitAnchor:
    """Resolve anchor: explicit arg > stored fit > plot variant; default t0."""
    if fit_anchor is not None:
        return normalize_fit_anchor(fit_anchor)
    stored = beta_fit.get("fit_anchor")
    if isinstance(stored, str) and stored in ("t1", "t0"):
        return normalize_fit_anchor(stored)
    if plot_variant == "anchored_t1":
        return "t1"
    return "t0"


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
    fit_anchor: FitAnchor,
) -> str:
    """Assemble the figure suptitle including quality and bound flags."""
    anchor_label = (
        r"$\varepsilon(1)$ empirical"
        if fit_anchor == "t1"
        else rf"$\varepsilon(0)={0.5:.1f}$ prior"
    )
    parts = [f"{condition_key}  |  q={signal_quality:.2f}  |  anchor={anchor_label}"]

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
    plot_variant: PlotVariant = "anchored_t0",
) -> Path:
    """Return the PNG path for one condition's learning-rate plot."""
    safe_name = condition_key.replace("/", "__")
    suffix = plot_variant
    return artifacts_dir / f"seed_{seed}" / "plots" / f"{safe_name}__{suffix}.png"


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
    plot_variant: PlotVariant = "anchored_t0",
    fit_anchor: FitAnchor | None = "t0",
) -> Path:
    """
    Render a dual-panel decay plot for one condition.

    Default fit_anchor is t0: GAT and HST share epsilon(0)=0.5 at t=0 (shown on the x-axis).
    With t1, both curves pass through empirical epsilon(1) at t=1. The fit-window marker
    applies to measured rounds t>=1 only.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not epsilon_series:
        raise ValueError("epsilon_series must be non-empty to plot learning rate.")

    resolved_anchor = _resolve_plot_anchor(
        beta_fit, plot_variant, fit_anchor=fit_anchor
    )

    t_values = np.arange(1, len(epsilon_series) + 1, dtype=float)
    empirical = np.asarray(epsilon_series, dtype=float)
    epsilon_1_f = float(empirical[0])
    if resolved_anchor == "t0":
        curve_t_plot = np.concatenate([[0.0], t_values])
    else:
        curve_t_plot = t_values

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
    fit_t_values = t_values[:fit_window_t_max]
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
    anchor_residual = 0.0
    if can_plot_curves:
        beta_gat_f = float(beta_gat)
        epsilon_inf_f = float(epsilon_inf)
        beta_hst_f = float(beta_hst_max)
        gat_curve = _decay_curve(
            curve_t_plot,
            beta=beta_gat_f,
            epsilon_inf=epsilon_inf_f,
            anchor=resolved_anchor,
            epsilon_1=epsilon_1_f,
        )
        hst_curve = _decay_curve(
            curve_t_plot,
            beta=beta_hst_f,
            epsilon_inf=epsilon_inf_f,
            anchor=resolved_anchor,
            epsilon_1=epsilon_1_f,
        )
        anchor_tag = "1" if resolved_anchor == "t1" else "0"
        ax_lin.plot(
            curve_t_plot,
            gat_curve,
            "--",
            color="#2ca02c",
            linewidth=2.0,
            label=(
                rf"GAT fit ($\beta_{{\mathrm{{GAT}}}}$={beta_gat_f:.3f}, "
                rf"anchor $t={anchor_tag}$)"
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
                rf"anchor $t={anchor_tag}$)"
            ),
        )
        if resolved_anchor == "t0":
            ax_lin.plot(
                [0.0],
                [PRIOR_EPSILON_AT_T0],
                "s",
                color="#9467bd",
                markersize=7,
                markerfacecolor="white",
                markeredgewidth=1.5,
                label=rf"shared anchor $\varepsilon(0)={PRIOR_EPSILON_AT_T0}$",
                zorder=5,
            )
            anchor_residual = max(PRIOR_EPSILON_AT_T0 - epsilon_inf_f, floor)
        else:
            anchor_residual = max(epsilon_1_f - epsilon_inf_f, floor)
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
    if resolved_anchor == "t0":
        ax_lin.set_xlim(left=-0.2)
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
        if resolved_anchor == "t0":
            slope_t_axis = curve_t_plot
            slope_base = max(PRIOR_EPSILON_AT_T0 - epsilon_inf_f, floor)
            gat_slope_line = slope_base * np.exp(-beta_gat_f * slope_t_axis)
            hst_slope_line = slope_base * np.exp(-beta_hst_f * slope_t_axis)
        else:
            slope_t_axis = t_values
            slope_residual = max(anchor_residual, floor)
            gat_slope_line = slope_residual * np.exp(-beta_gat_f * (slope_t_axis - 1.0))
            hst_slope_line = slope_residual * np.exp(-beta_hst_f * (slope_t_axis - 1.0))
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
        if resolved_anchor == "t0":
            ax_log.set_xlim(left=-0.2)
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
        fit_anchor=resolved_anchor,
    )
    fig.suptitle(suptitle, fontsize=11)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


__all__ = [
    "PlotVariant",
    "_build_learning_rate_suptitle",
    "learning_rate_plot_path",
    "save_learning_rate_plot",
]
