"""Tests for metrics aggregation script."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.summarize_metrics import (
    aggregate_records,
    collect_records,
    load_metrics_records,
    write_csv,
)


def test_load_metrics_records_parses_condition_and_fit_fields(tmp_path: Path) -> None:
    metrics_path = tmp_path / "seed_0" / "metrics.json"
    metrics_path.parent.mkdir(parents=True)
    metrics_path.write_text(
        json.dumps(
            {
                "seed": 0,
                "conditions": {
                    "n_10/complete": {
                        "train_loss_final": 0.5,
                        "beta_fit": {
                            "beta": 0.12,
                            "fit_success": True,
                            "method": "log_linear_fallback",
                            "rmse": 0.08,
                            "r2": 0.4,
                            "failure_reason": "",
                        },
                        "beta_hst_max": 2.77,
                        "beta_gap": -2.65,
                        "exceeds_hst_bound": False,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    records = load_metrics_records(metrics_path)
    assert len(records) == 1
    row = records[0]
    assert row["seed"] == 0
    assert row["num_nodes"] == 10
    assert row["topology"] == "complete"
    assert row["beta_gat"] == 0.12
    assert row["beta_hst_max"] == 2.77
    assert row["beta_gap"] == -2.65
    assert row["exceeds_hst_bound"] is False
    assert row["fit_method"] == "log_linear_fallback"


def test_collect_records_from_grid_layout(tmp_path: Path) -> None:
    metrics_path = tmp_path / "grid_runs" / "n_10" / "q_0p8" / "seed_1" / "metrics.json"
    metrics_path.parent.mkdir(parents=True)
    metrics_path.write_text(
        json.dumps(
            {
                "seed": 1,
                "conditions": {
                    "n_10/ws_p_0.1_seed_0": {
                        "train_loss_final": 0.7,
                        "beta_fit": {"beta": 0.2, "fit_success": True, "method": "scipy_curve_fit"},
                        "beta_hst_max": 2.77,
                        "beta_gap": -2.57,
                        "exceeds_hst_bound": False,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    records = collect_records([tmp_path])
    assert len(records) == 1
    assert records[0]["seed"] == 1
    assert records[0]["num_nodes"] == 10
    assert records[0]["signal_quality"] == 0.8
    assert records[0]["topology"] == "ws_p_0.1"


def test_aggregate_records_computes_mean_and_std_across_seeds() -> None:
    records = [
        {
            "seed": 0,
            "num_nodes": 10,
            "signal_quality": 0.8,
            "topology": "complete",
            "beta_gat": 0.1,
            "beta_hst_max": 2.77,
            "beta_gap": -2.67,
            "exceeds_hst_bound": False,
            "fit_success": True,
            "fit_r2": 0.4,
        },
        {
            "seed": 1,
            "num_nodes": 10,
            "signal_quality": 0.8,
            "topology": "complete",
            "beta_gat": 0.2,
            "beta_hst_max": 2.77,
            "beta_gap": -2.57,
            "exceeds_hst_bound": False,
            "fit_success": True,
            "fit_r2": 0.6,
        },
    ]
    aggregated = aggregate_records(records)
    assert len(aggregated) == 1
    row = aggregated[0]
    assert row["n_seeds"] == 2
    assert row["beta_gat_mean"] == pytest.approx(0.15)
    assert row["beta_gat_std"] == pytest.approx(0.07071067811865476)
    assert row["beta_gat_best"] == pytest.approx(0.2)
    assert row["beta_gat_best_seed"] == 1
    assert row["beta_gap_at_best"] == pytest.approx(-2.57)
    assert row["exceeds_hst_bound_at_best"] is False
    assert row["fit_r2_mean"] == pytest.approx(0.5)


def test_aggregate_records_computes_se_ci_and_convergence_rate() -> None:
    records = [
        {
            "seed": 0,
            "num_nodes": 10,
            "signal_quality": 0.6,
            "topology": "complete",
            "beta_gat": 0.10,
            "beta_gat_se": 0.02,
            "beta_gat_ci_lower": 0.06,
            "beta_gat_ci_upper": 0.14,
            "beta_hst_max": 0.81,
            "beta_gap": -0.71,
            "exceeds_hst_bound": False,
            "convergence_warning": True,
            "fit_success": True,
            "fit_r2": 0.9,
        },
        {
            "seed": 1,
            "num_nodes": 10,
            "signal_quality": 0.6,
            "topology": "complete",
            "beta_gat": 0.12,
            "beta_gat_se": 0.04,
            "beta_gat_ci_lower": 0.04,
            "beta_gat_ci_upper": 0.20,
            "beta_hst_max": 0.81,
            "beta_gap": -0.69,
            "exceeds_hst_bound": False,
            "convergence_warning": False,
            "fit_success": True,
            "fit_r2": 0.95,
        },
    ]
    aggregated = aggregate_records(records)
    assert len(aggregated) == 1
    row = aggregated[0]
    assert row["beta_gat_se_mean"] == pytest.approx(0.03)
    assert row["beta_gat_ci_lower_mean"] == pytest.approx(0.05)
    assert row["beta_gat_ci_upper_mean"] == pytest.approx(0.17)
    assert row["convergence_warning_rate"] == pytest.approx(0.5)


def test_load_metrics_records_reads_run_metadata_from_payload(tmp_path: Path) -> None:
    metrics_path = tmp_path / "seed_3" / "metrics.json"
    metrics_path.parent.mkdir(parents=True)
    metrics_path.write_text(
        json.dumps(
            {
                "seed": 3,
                "signal_quality": 0.65,
                "num_nodes": 10,
                "conditions": {
                    "n_10/complete": {
                        "beta_fit": {"beta": 0.2, "fit_success": True},
                        "beta_hst_max": 1.0,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    records = load_metrics_records(metrics_path)
    assert len(records) == 1
    assert records[0]["signal_quality"] == 0.65
    assert records[0]["num_nodes"] == 10


def test_write_csv_creates_table(tmp_path: Path) -> None:
    metrics_path = tmp_path / "seed_2" / "metrics.json"
    metrics_path.parent.mkdir(parents=True)
    metrics_path.write_text(
        json.dumps(
            {
                "seed": 2,
                "conditions": {
                    "n_50/complete": {
                        "train_loss_final": 0.4,
                        "beta_fit": {"beta": 1.0, "fit_success": True, "method": "scipy_curve_fit"},
                        "beta_hst_max": 2.0,
                        "beta_gap": -1.0,
                        "exceeds_hst_bound": True,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    output_csv = tmp_path / "summary.csv"
    records = collect_records([tmp_path])
    write_csv(records, output_csv)
    text = output_csv.read_text(encoding="utf-8")
    assert "seed,num_nodes" in text
    assert "complete" in text
    assert "1" in text
