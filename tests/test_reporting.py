from src.reporting import classify_regimes


def test_classify_regimes_supports_information_theoretic_limit() -> None:
    aggregates = {
        "by_setting": {
            "n_10/q_0.6": {
                "mean_beta_gap": -0.10,
                "proportion_exceeds_hst_bound": 0.0,
                "mean_beta_gat": 0.20,
            },
            "n_50/q_0.6": {
                "mean_beta_gap": -0.05,
                "proportion_exceeds_hst_bound": 0.0,
                "mean_beta_gat": 0.20,
            },
        }
    }
    result = classify_regimes(aggregates)
    assert result["headline_label"] == "supports_information_theoretic_limit"
    assert result["supports_information_theoretic_limit"] is True
    assert result["empirical_counter_evidence"] is False
    assert result["boundary_condition_evidence"] is False


def test_classify_regimes_detects_empirical_counter_evidence() -> None:
    aggregates = {
        "by_setting": {
            "n_10/q_0.8": {
                "mean_beta_gap": 0.02,
                "proportion_exceeds_hst_bound": 0.4,
                "mean_beta_gat": 0.25,
            }
        }
    }
    result = classify_regimes(aggregates)
    assert result["headline_label"] == "empirical_counter_evidence"
    assert result["supports_information_theoretic_limit"] is False
    assert result["empirical_counter_evidence"] is True


def test_classify_regimes_detects_boundary_condition_evidence() -> None:
    aggregates = {
        "by_setting": {
            "n_10/q_0.6": {
                "mean_beta_gap": -0.01,
                "proportion_exceeds_hst_bound": 0.0,
                "mean_beta_gat": 0.12,
            },
            "n_100/q_0.6": {
                "mean_beta_gap": -0.01,
                "proportion_exceeds_hst_bound": 0.0,
                "mean_beta_gat": 0.30,
            },
        }
    }
    result = classify_regimes(aggregates)
    assert result["headline_label"] == "boundary_condition_evidence"
    assert result["boundary_condition_evidence"] is True
