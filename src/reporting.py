"""Reporting helpers for interpreting aggregated grid results."""

from __future__ import annotations

import math
from typing import Any


def _collect_metric(aggregates: dict[str, Any], metric_name: str) -> list[float]:
    by_setting = aggregates.get("by_setting", {})
    if not isinstance(by_setting, dict):
        return []
    values: list[float] = []
    for payload in by_setting.values():
        if not isinstance(payload, dict):
            continue
        value = payload.get(metric_name)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return values


def classify_regimes(aggregates: dict[str, Any], *, epsilon: float = 1e-9) -> dict[str, Any]:
    """
    Classify proposal regimes from aggregated beta and exceedance statistics.

    Note:
        This classification tracks consistency with the HST equilibrium bound,
        not an information-theoretic impossibility bound for arbitrary learners.

    Returns a deterministic dictionary with booleans and diagnostics.
    """
    by_setting = aggregates.get("by_setting", {})
    fit_success_rates = _collect_metric(aggregates, "fit_success_rate")
    convergence_warning_rates = _collect_metric(aggregates, "convergence_warning_rate")

    insufficient_fit_quality = not fit_success_rates or not any(
        rate >= 0.5 for rate in fit_success_rates
    )
    high_convergence_warning_rate = any(
        rate > 0.5 for rate in convergence_warning_rates
    )

    mean_beta_gaps = _collect_metric(aggregates, "mean_beta_gap")
    exceed_proportions = _collect_metric(aggregates, "proportion_exceeds_hst_bound")
    mean_beta_gat_values = _collect_metric(aggregates, "mean_beta_gat")

    any_positive_gap = any(value > epsilon for value in mean_beta_gaps)
    any_exceedance = any(value > epsilon for value in exceed_proportions)
    all_non_positive_gap = bool(mean_beta_gaps) and all(value <= epsilon for value in mean_beta_gaps)
    all_zero_exceedance = bool(exceed_proportions) and all(
        value <= epsilon for value in exceed_proportions
    )

    consistent_with_equilibrium_bound = all_non_positive_gap and all_zero_exceedance
    empirical_counter_evidence = any_positive_gap or any_exceedance

    beta_gat_variation = 0.0
    if len(mean_beta_gat_values) >= 2:
        beta_gat_variation = max(mean_beta_gat_values) - min(mean_beta_gat_values)
    exceedance_variation = 0.0
    if len(exceed_proportions) >= 2:
        exceedance_variation = max(exceed_proportions) - min(exceed_proportions)

    boundary_condition_evidence = (beta_gat_variation > epsilon) or (
        exceedance_variation > epsilon
    )

    if insufficient_fit_quality or high_convergence_warning_rate:
        headline_label = "inconclusive"
    elif empirical_counter_evidence:
        headline_label = "empirical_counter_evidence"
    elif boundary_condition_evidence:
        headline_label = "boundary_condition_evidence"
    elif consistent_with_equilibrium_bound:
        headline_label = "consistent_with_equilibrium_bound"
    else:
        headline_label = "inconclusive"

    return {
        "headline_label": headline_label,
        "consistent_with_equilibrium_bound": consistent_with_equilibrium_bound,
        "empirical_counter_evidence": empirical_counter_evidence,
        "boundary_condition_evidence": boundary_condition_evidence,
        "insufficient_fit_quality": insufficient_fit_quality,
        "high_convergence_warning_rate": high_convergence_warning_rate,
        "diagnostics": {
            "num_settings": len(by_setting) if isinstance(by_setting, dict) else 0,
            "num_settings_with_beta_gap": len(mean_beta_gaps),
            "num_settings_with_exceedance": len(exceed_proportions),
            "num_settings_with_fit_success_rate": len(fit_success_rates),
            "num_settings_with_convergence_warning_rate": len(convergence_warning_rates),
            "mean_beta_gap_min": min(mean_beta_gaps) if mean_beta_gaps else None,
            "mean_beta_gap_max": max(mean_beta_gaps) if mean_beta_gaps else None,
            "proportion_exceeds_min": min(exceed_proportions) if exceed_proportions else None,
            "proportion_exceeds_max": max(exceed_proportions) if exceed_proportions else None,
            "fit_success_rate_min": min(fit_success_rates) if fit_success_rates else None,
            "fit_success_rate_max": max(fit_success_rates) if fit_success_rates else None,
            "convergence_warning_rate_max": (
                max(convergence_warning_rates) if convergence_warning_rates else None
            ),
            "beta_gat_variation": beta_gat_variation,
            "exceedance_variation": exceedance_variation,
        },
    }


__all__ = ["classify_regimes"]
