from src.reporting import classify_regimes


def test_classify_regimes_consistent_with_equilibrium_bound() -> None:
    aggregates = {
        "by_setting": {
            "n_10/q_0.6": {
                "mean_beta_gap": -0.10,
                "proportion_exceeds_hst_bound": 0.0,
                "mean_beta_gat": 0.20,
                "fit_success_rate": 1.0,
                "convergence_warning_rate": 0.0,
            },
            "n_50/q_0.6": {
                "mean_beta_gap": -0.05,
                "proportion_exceeds_hst_bound": 0.0,
                "mean_beta_gat": 0.20,
                "fit_success_rate": 1.0,
                "convergence_warning_rate": 0.0,
            },
        }
    }
    result = classify_regimes(aggregates)
    assert result["headline_label"] == "consistent_with_equilibrium_bound"
    assert result["consistent_with_equilibrium_bound"] is True
    assert result["empirical_counter_evidence"] is False
    assert result["boundary_condition_evidence"] is False


def test_classify_regimes_detects_empirical_counter_evidence() -> None:
    aggregates = {
        "by_setting": {
            "n_10/q_0.8": {
                "mean_beta_gap": 0.02,
                "proportion_exceeds_hst_bound": 0.4,
                "mean_beta_gat": 0.25,
                "fit_success_rate": 1.0,
                "convergence_warning_rate": 0.0,
            }
        }
    }
    result = classify_regimes(aggregates)
    assert result["headline_label"] == "empirical_counter_evidence"
    assert result["consistent_with_equilibrium_bound"] is False
    assert result["empirical_counter_evidence"] is True


def test_classify_regimes_detects_boundary_condition_evidence() -> None:
    aggregates = {
        "by_setting": {
            "n_10/q_0.6": {
                "mean_beta_gap": -0.01,
                "proportion_exceeds_hst_bound": 0.0,
                "mean_beta_gat": 0.12,
                "fit_success_rate": 1.0,
                "convergence_warning_rate": 0.0,
            },
            "n_100/q_0.6": {
                "mean_beta_gap": -0.01,
                "proportion_exceeds_hst_bound": 0.0,
                "mean_beta_gat": 0.30,
                "fit_success_rate": 1.0,
                "convergence_warning_rate": 0.0,
            },
        }
    }
    result = classify_regimes(aggregates)
    assert result["headline_label"] == "boundary_condition_evidence"
    assert result["boundary_condition_evidence"] is True


def test_classify_regimes_inconclusive_when_fit_quality_low() -> None:
    aggregates = {
        "by_setting": {
            "n_10/q_0.6": {
                "mean_beta_gap": -0.01,
                "proportion_exceeds_hst_bound": 0.0,
                "mean_beta_gat": 0.12,
                "fit_success_rate": 0.2,
                "convergence_warning_rate": 0.0,
            }
        }
    }
    result = classify_regimes(aggregates)
    assert result["headline_label"] == "inconclusive"
    assert result["insufficient_fit_quality"] is True


def test_classify_regimes_inconclusive_when_convergence_warnings_high() -> None:
    aggregates = {
        "by_setting": {
            "n_10/q_0.6": {
                "mean_beta_gap": -0.01,
                "proportion_exceeds_hst_bound": 0.0,
                "mean_beta_gat": 0.12,
                "fit_success_rate": 1.0,
                "convergence_warning_rate": 0.8,
            }
        }
    }
    result = classify_regimes(aggregates)
    assert result["headline_label"] == "inconclusive"
    assert result["high_convergence_warning_rate"] is True
