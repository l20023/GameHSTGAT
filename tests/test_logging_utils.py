"""Tests for W&B logging helpers."""

from __future__ import annotations

from src.logging_utils import log_condition_metrics, log_train_loss_history


class _DummyRun:
    def __init__(self) -> None:
        self.logs: list[tuple[dict, int | None]] = []

    def log(self, payload: dict, step: int | None = None) -> None:
        self.logs.append((payload, step))


def test_log_train_loss_history_logs_each_episode_with_step() -> None:
    run = _DummyRun()
    log_train_loss_history(
        run=run,
        condition_key="n_10/complete",
        train_loss_history=[0.9, 0.5, 0.2],
    )
    assert len(run.logs) == 3
    assert run.logs[0] == ({"n_10/complete/train_loss": 0.9}, 0)
    assert run.logs[1] == ({"n_10/complete/train_loss": 0.5}, 1)
    assert run.logs[2] == ({"n_10/complete/train_loss": 0.2}, 2)


def test_log_train_loss_history_skips_empty_series() -> None:
    run = _DummyRun()
    log_train_loss_history(
        run=run,
        condition_key="n_10/complete",
        train_loss_history=[],
    )
    assert run.logs == []


def test_log_condition_metrics_logs_train_loss_from_argument() -> None:
    run = _DummyRun()
    log_condition_metrics(
        run=run,
        condition_key="n_10/complete",
        condition_index=0,
        metrics={"train_loss_final": 0.1, "beta_fit": {}},
        train_loss_history=[1.0, 0.4],
    )
    summary_logs = [entry for entry in run.logs if entry[1] is None]
    step_logs = [entry for entry in run.logs if entry[1] is not None]
    assert len(summary_logs) == 1
    assert summary_logs[0][0]["n_10/complete/train_loss_final"] == 0.1
    assert len(step_logs) == 2
    assert step_logs[0][0]["n_10/complete/train_loss"] == 1.0
