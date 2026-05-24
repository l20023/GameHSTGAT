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

    if empirical_counter_evidence:
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
        "diagnostics": {
            "num_settings_with_beta_gap": len(mean_beta_gaps),
            "num_settings_with_exceedance": len(exceed_proportions),
            "mean_beta_gap_min": min(mean_beta_gaps) if mean_beta_gaps else None,
            "mean_beta_gap_max": max(mean_beta_gaps) if mean_beta_gaps else None,
            "proportion_exceeds_min": min(exceed_proportions) if exceed_proportions else None,
            "proportion_exceeds_max": max(exceed_proportions) if exceed_proportions else None,
            "beta_gat_variation": beta_gat_variation,
            "exceedance_variation": exceedance_variation,
        },
    }


__all__ = ["classify_regimes"]
