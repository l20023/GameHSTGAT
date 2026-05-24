"""Tests for learning-rate comparison plots."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.learning_rate_plots import (
    _build_learning_rate_suptitle,
    learning_rate_plot_path,
    save_learning_rate_plot,
)


def test_build_suptitle_shows_convergence_warning_instead_of_within_bound() -> None:
    title = _build_learning_rate_suptitle(
        condition_key="n_10/complete",
        signal_quality=0.6,
        beta_gap=0.12,
        exceeds_hst_bound=False,
        convergence_warning=True,
        epsilon_inf=0.08,
        fit_success=True,
        convergence_warning_threshold=0.05,
    )
    assert "convergence warning" in title
    assert "bound comparison suppressed" in title
    assert "within bound" not in title


def test_build_suptitle_shows_fit_failed() -> None:
    title = _build_learning_rate_suptitle(
        condition_key="n_10/ws",
        signal_quality=0.8,
        beta_gap=None,
        exceeds_hst_bound=None,
        convergence_warning=False,
        epsilon_inf=None,
        fit_success=False,
        convergence_warning_threshold=0.05,
    )
    assert "fit failed" in title
    assert "bound comparison n/a" in title


@pytest.mark.parametrize("plot_variant", ["anchored_t0"])
def test_save_learning_rate_plot_writes_png(tmp_path: Path, plot_variant: str) -> None:
    epsilon_series = [0.4, 0.25, 0.15, 0.1, 0.08]
    beta_fit = {
        "alpha": 0.45,
        "beta": 0.3,
        "epsilon_inf": 0.05,
        "fit_success": True,
        "method": "scipy_anchored_t0",
    }
    output_path = tmp_path / f"plot_{plot_variant}.png"
    save_learning_rate_plot(
        output_path=output_path,
        epsilon_series=epsilon_series,
        beta_fit=beta_fit,
        beta_hst_max=0.81,
        condition_key="n_10/complete",
        signal_quality=0.6,
        beta_gap=-0.51,
        exceeds_hst_bound=False,
        convergence_warning=False,
        plot_variant=plot_variant,  # type: ignore[arg-type]
    )


def test_save_learning_rate_plot_with_convergence_warning(tmp_path: Path) -> None:
    output_path = tmp_path / "plot_warn.png"
    save_learning_rate_plot(
        output_path=output_path,
        epsilon_series=[0.4, 0.25, 0.15, 0.1, 0.08],
        beta_fit={
            "alpha": 0.45,
            "beta": 0.3,
            "epsilon_inf": 0.08,
            "fit_success": True,
            "method": "scipy_anchored_t0",
        },
        beta_hst_max=0.81,
        condition_key="n_10/complete",
        signal_quality=0.6,
        beta_gap=0.1,
        exceeds_hst_bound=False,
        convergence_warning=True,
    )
    assert output_path.exists()
    assert output_path.stat().st_size > 0
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_learning_rate_plot_path_sanitizes_condition_key() -> None:
    anchored_t0_path = learning_rate_plot_path(
        artifacts_dir=Path("artifacts/fair"),
        seed=2,
        condition_key="n_10/ws_p_0.1_seed_2",
        plot_variant="anchored_t0",
    )
    assert anchored_t0_path == Path(
        "artifacts/fair/seed_2/plots/n_10__ws_p_0.1_seed_2__anchored_t0.png"
    )
