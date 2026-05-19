"""Tests for learning-rate comparison plots."""

from __future__ import annotations

from pathlib import Path

from src.learning_rate_plots import learning_rate_plot_path, save_learning_rate_plot


def test_save_learning_rate_plot_writes_png(tmp_path: Path) -> None:
    epsilon_series = [0.4, 0.25, 0.15, 0.1, 0.08]
    beta_fit = {
        "alpha": 0.45,
        "beta": 0.3,
        "epsilon_inf": 0.05,
        "fit_success": True,
        "method": "scipy_curve_fit",
    }
    output_path = tmp_path / "plot.png"
    save_learning_rate_plot(
        output_path=output_path,
        epsilon_series=epsilon_series,
        beta_fit=beta_fit,
        beta_hst_max=0.81,
        condition_key="n_10/complete",
        signal_quality=0.6,
        beta_gap=-0.51,
        exceeds_hst_bound=False,
    )
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_learning_rate_plot_path_sanitizes_condition_key() -> None:
    path = learning_rate_plot_path(
        artifacts_dir=Path("artifacts/fair"),
        seed=2,
        condition_key="n_10/ws_p_0.1_seed_2",
    )
    assert path == Path("artifacts/fair/seed_2/plots/n_10__ws_p_0.1_seed_2.png")
