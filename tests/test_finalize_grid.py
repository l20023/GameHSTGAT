"""Tests for grid finalization from artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from src.grid_summary import build_grid_summary, collect_records_from_artifacts


def _write_metrics(
    path: Path,
    *,
    beta: float,
    beta_hst_max: float,
    exceeds: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": 0,
        "conditions": {
            "n_10/complete": {
                "beta_fit": {"beta": beta, "fit_success": True},
                "beta_hst_max": beta_hst_max,
                "beta_gap": beta - beta_hst_max,
                "exceeds_hst_bound": exceeds,
            }
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


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
