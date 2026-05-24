"""Plots comparing empirical error decay with GAT fit and HST learning-rate bound."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np

from .training_pipeline import exponential_decay_values

PlotVariant = Literal["anchored_t0"]


def learning_rate_plot_path(
    *,
    artifacts_dir: Path,
    seed: int,
    condition_key: str,
    plot_variant: PlotVariant = "anchored_t0",
) -> Path:
    """Return the PNG path for one condition's learning-rate plot."""
    safe_name = condition_key.replace("/", "__")
    suffix = "anchored_t0"
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
    plot_variant: PlotVariant = "anchored_t0",
) -> Path:
    """
    Render a dual-panel decay plot for one condition.

    Left panel (linear y):
        Empirical error rate epsilon(t) over rounds with the fitted GAT decay
        curve overlaid. This panel shows absolute error magnitude over time.

    Right panel (log y):
        Log-scale view of (epsilon(t) - epsilon_inf), which makes the slope
        directly equal to -beta. The GAT fit and the HST upper bound are drawn
        as straight reference lines anchored at the same starting point so that
        only the slopes are compared. The HST line represents the maximal
        slope permitted by Theorem 1, not a predicted error trajectory.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not epsilon_series:
        raise ValueError("epsilon_series must be non-empty to plot learning rate.")

    t_values = np.arange(1, len(epsilon_series) + 1, dtype=float)
    empirical = np.asarray(epsilon_series, dtype=float)

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

    # ---- Left panel: linear y, empirical error vs GAT fit ----
    ax_lin.plot(
        t_values,
        empirical,
        "o-",
        color="#1f77b4",
        linewidth=1.5,
        markersize=4,
        label=r"$\varepsilon(t)$ empirical",
    )
    if can_plot_curves:
        alpha_f = float(alpha)
        beta_gat_f = float(beta_gat)
        epsilon_inf_f = float(epsilon_inf)
        gat_curve = exponential_decay_values(
            t_values, alpha=alpha_f, beta=beta_gat_f, epsilon_inf=epsilon_inf_f
        )
        ax_lin.plot(
            t_values,
            gat_curve,
            "--",
            color="#2ca02c",
            linewidth=2.0,
            label=rf"GAT fit ($\beta_{{\mathrm{{GAT}}}}$={beta_gat_f:.3f})",
        )
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
    ax_lin.set_xlabel("Round $t$")
    ax_lin.set_ylabel(r"Error rate $\varepsilon(t)$")
    ax_lin.set_ylim(bottom=0.0)
    ax_lin.grid(True, alpha=0.3)
    ax_lin.legend(loc="best", fontsize=9)
    ax_lin.set_title("Empirical decay (linear)", fontsize=11)

    # ---- Right panel: log y, slope comparison ----
    floor = 1e-4
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

    # Anchor both slope reference lines at the first finite empirical residual.
    anchor_t = float(t_values[0])
    anchor_y = float(residual[0])
    if can_plot_curves and anchor_y > 0.0:
        beta_gat_f = float(beta_gat)
        beta_hst_f = float(beta_hst_max)
        gat_slope_line = anchor_y * np.exp(-beta_gat_f * (t_values - anchor_t))
        hst_slope_line = anchor_y * np.exp(-beta_hst_f * (t_values - anchor_t))
        ax_log.semilogy(
            t_values,
            gat_slope_line,
            "--",
            color="#2ca02c",
            linewidth=2.0,
            label=rf"GAT slope ($\beta_{{\mathrm{{GAT}}}}$={beta_gat_f:.3f})",
        )
        ax_log.semilogy(
            t_values,
            hst_slope_line,
            "--",
            color="#d62728",
            linewidth=2.0,
            label=rf"HST max slope ($\beta_{{\mathrm{{HST}}}}$={beta_hst_f:.3f})",
        )
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
    # Cap the visible y-range so that extreme exponential decays from
    # degenerate fits do not collapse the panel onto a single line.
    empirical_min = float(np.min(residual))
    y_floor = max(empirical_min * 1e-3, 1e-6)
    y_ceiling = max(float(np.max(residual)) * 1.5, 1.0)
    ax_log.set_ylim(bottom=y_floor, top=y_ceiling)

    suptitle = f"{condition_key}  |  q={signal_quality:.2f}"
    if isinstance(beta_gap, float) and np.isfinite(beta_gap):
        suptitle += f"  |  gap={beta_gap:.3f}"
    if isinstance(exceeds_hst_bound, bool):
        suptitle += "  |  exceeds bound" if exceeds_hst_bound else "  |  within bound"
    fig.suptitle(suptitle, fontsize=12)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


__all__ = ["PlotVariant", "learning_rate_plot_path", "save_learning_rate_plot"]
