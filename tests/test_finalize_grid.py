"""Tests for grid finalization from artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from src.grid_summary import (
    aggregate_records,
    build_grid_summary,
    collect_records_from_artifacts,
)


def _write_metrics(
    path: Path,
    *,
    beta: float,
    beta_hst_max: float,
    exceeds: bool,
    convergence_warning: bool = False,
    fit_success: bool = True,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": 0,
        "conditions": {
            "n_10/complete": {
                "beta_fit": {"beta": beta, "fit_success": fit_success},
                "beta_hst_max": beta_hst_max,
                "beta_gap": beta - beta_hst_max,
                "exceeds_hst_bound": exceeds,
                "convergence_warning": convergence_warning,
            }
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_collect_records_from_artifacts_q055_path(tmp_path: Path) -> None:
    metrics_path = tmp_path / "n_10" / "q_0p55" / "seed_0" / "metrics.json"
    _write_metrics(
        metrics_path,
        beta=0.5,
        beta_hst_max=0.3,
        exceeds=False,
    )
    records = collect_records_from_artifacts(tmp_path)
    assert len(records) == 1
    assert records[0]["signal_quality"] == 0.55
    assert records[0]["setting_key"] == "n_10/q_0.55"


def test_collect_records_from_artifacts(tmp_path: Path) -> None:
    metrics_path = tmp_path / "n_10" / "q_0p6" / "seed_0" / "metrics.json"
    _write_metrics(
        metrics_path,
        beta=0.5,
        beta_hst_max=0.3,
        exceeds=False,
    )
    records = collect_records_from_artifacts(tmp_path)
    assert len(records) == 1
    assert records[0]["num_nodes"] == 10
    assert records[0]["signal_quality"] == 0.6
    assert records[0]["seed"] == 0
    assert records[0]["beta_gat"] == 0.5
    assert records[0]["fit_success"] is True


def test_aggregate_records_tracks_fit_and_convergence_rates() -> None:
    records = [
        {
            "setting_key": "n_10/q_0.6",
            "condition_name": "complete",
            "beta_gat": 0.4,
            "beta_gap": 0.1,
            "exceeds_hst_bound": False,
            "convergence_warning": False,
            "fit_success": True,
            "artifact_path": "a",
        },
        {
            "setting_key": "n_10/q_0.6",
            "condition_name": "complete",
            "beta_gat": float("nan"),
            "beta_gap": None,
            "exceeds_hst_bound": None,
            "convergence_warning": True,
            "fit_success": False,
            "artifact_path": "b",
        },
    ]
    aggregates = aggregate_records(records)
    bucket = aggregates["by_setting"]["n_10/q_0.6"]
    assert bucket["num_records"] == 2
    assert bucket["num_beta_gat_values"] == 1
    assert bucket["fit_success_rate"] == 0.5
    assert bucket["convergence_warning_rate"] == 0.5


def test_build_grid_summary_from_mock_artifacts(tmp_path: Path) -> None:
    metrics_path = tmp_path / "n_10" / "q_0p6" / "seed_0" / "metrics.json"
    _write_metrics(
        metrics_path,
        beta=0.4,
        beta_hst_max=0.3,
        exceeds=False,
    )
    records = collect_records_from_artifacts(tmp_path)
    summary = build_grid_summary(
        records=records,
        grid_config={"seeds": [0], "num_nodes_list": [10], "signal_quality_list": [0.6]},
        run_summaries=[],
        artifacts_root=tmp_path,
        num_nodes_list=[10],
        signal_quality_list=[0.6],
    )
    assert summary["num_condition_records"] == 1
    assert "aggregates" in summary
    assert "regime_classification" in summary
    assert summary["aggregates"]["by_setting"]["n_10/q_0.6"]["num_records"] == 1
