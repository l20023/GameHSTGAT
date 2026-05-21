"""Tests for training-loss curve plots."""

from __future__ import annotations

from pathlib import Path

from src.train_loss_plots import save_train_loss_plot, train_loss_plot_path


def test_save_train_loss_plot_writes_png(tmp_path: Path) -> None:
    output_path = tmp_path / "train_loss.png"
    save_train_loss_plot(
        output_path=output_path,
        train_loss_history=[0.8, 0.5, 0.3, 0.2],
        condition_key="n_10/complete",
    )
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_train_loss_plot_path_sanitizes_condition_key() -> None:
    path = train_loss_plot_path(
        artifacts_dir=Path("artifacts/fair"),
        seed=2,
        condition_key="n_10/ws_p_0.1_seed_2",
    )
    assert path == Path(
        "artifacts/fair/seed_2/plots/n_10__ws_p_0.1_seed_2__train_loss.png"
    )
