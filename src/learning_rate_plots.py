"""Plots comparing empirical error decay with GAT fit and HST learning-rate bound."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np

from .training_pipeline import exponential_decay_values

PlotVariant = Literal["free_alpha", "anchored_t1"]


def learning_rate_plot_path(
    *,
    artifacts_dir: Path,
    seed: int,
    condition_key: str,
    plot_variant: PlotVariant = "free_alpha",
) -> Path:
    """Return the PNG path for one condition's learning-rate plot."""
    safe_name = condition_key.replace("/", "__")
    suffix = "free_alpha" if plot_variant == "free_alpha" else "anchored_t1"
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
    plot_variant: PlotVariant = "free_alpha",
) -> Path:
    """
    Plot empirical epsilon(t), GAT exponential fit, and HST max-rate reference curve.

    Variants:
    - free_alpha: Both GAT and HST curves share fitted (alpha, epsilon_inf), differing only in beta.
    - anchored_t1: HST reference is anchored at empirical epsilon(1), then decays with beta_HST_max.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not epsilon_series:
        raise ValueError("epsilon_series must be non-empty to plot learning rate.")

    t_values = np.arange(1, len(epsilon_series) + 1, dtype=float)
    empirical = np.asarray(epsilon_series, dtype=float)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        t_values,
        empirical,
        "o-",
        color="#1f77b4",
        linewidth=1.5,
        markersize=4,
        label=r"$\varepsilon(t)$ empirical",
    )

    fit_success = bool(beta_fit.get("fit_success", False))
    alpha = beta_fit.get("alpha")
    beta_gat = beta_fit.get("beta")
    epsilon_inf = beta_fit.get("epsilon_inf")
    can_plot_curves = (
        fit_success
        and isinstance(alpha, (int, float))
        and isinstance(beta_gat, (int, float))
        and isinstance(epsilon_inf, (int, float))
        and np.isfinite([alpha, beta_gat, epsilon_inf]).all()
    )

    if can_plot_curves:
        alpha_f = float(alpha)
        beta_gat_f = float(beta_gat)
        epsilon_inf_f = float(epsilon_inf)
        epsilon_1 = float(empirical[0])
        gat_curve = exponential_decay_values(
            t_values, alpha=alpha_f, beta=beta_gat_f, epsilon_inf=epsilon_inf_f
        )
        if plot_variant == "anchored_t1":
            anchored_alpha = epsilon_1 - epsilon_inf_f
            hst_curve = epsilon_inf_f + anchored_alpha * np.exp(
                -float(beta_hst_max) * (t_values - 1.0)
            )
            hst_label = rf"HST anchored at $\varepsilon(1)$ ($\beta_{{\mathrm{{HST}}}}$={beta_hst_max:.3f})"
        else:
            hst_curve = exponential_decay_values(
                t_values, alpha=alpha_f, beta=float(beta_hst_max), epsilon_inf=epsilon_inf_f
            )
            hst_label = rf"HST upper bound ($\beta_{{\mathrm{{HST}}}}$={beta_hst_max:.3f})"
        ax.plot(
            t_values,
            gat_curve,
            "--",
            color="#2ca02c",
            linewidth=2.0,
            label=rf"GAT fit ($\beta_{{\mathrm{{GAT}}}}$={beta_gat_f:.3f})",
        )
        ax.plot(
            t_values,
            hst_curve,
            "--",
            color="#d62728",
            linewidth=2.0,
            label=hst_label,
        )
    else:
        ax.axhline(
            y=float(np.mean(empirical)),
            color="#d62728",
            linestyle=":",
            linewidth=1.5,
            label=rf"HST $\beta_{{\max}}$={beta_hst_max:.3f} (fit unavailable)",
        )

    title = f"{condition_key}  |  q={signal_quality:.2f}"
    if isinstance(beta_gap, float) and np.isfinite(beta_gap):
        title += f"  |  gap={beta_gap:.3f}"
    if isinstance(exceeds_hst_bound, bool):
        title += "  |  exceeds bound" if exceeds_hst_bound else "  |  within bound"
    title += "  |  anchored HST@t1" if plot_variant == "anchored_t1" else "  |  free alpha"
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Round t")
    ax.set_ylabel(r"Error rate $\varepsilon(t)$")
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


__all__ = ["PlotVariant", "learning_rate_plot_path", "save_learning_rate_plot"]
