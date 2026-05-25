"""Seed-aggregated metrics across replication runs."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any


AGGREGATE_GROUP_KEYS = ("num_nodes", "signal_quality", "topology")
AGGREGATE_COLUMNS = [
    "num_nodes",
    "signal_quality",
    "topology",
    "n_seeds",
    "beta_gat_mean",
    "beta_gat_std",
    "beta_gat_se_mean",
    "beta_gat_ci_lower_mean",
    "beta_gat_ci_upper_mean",
    "beta_gat_best",
    "beta_gat_best_seed",
    "beta_gap_at_best",
    "exceeds_hst_bound_at_best",
    "beta_hst_max_mean",
    "beta_gap_mean",
    "beta_gap_std",
    "exceeds_hst_bound_rate",
    "convergence_warning_rate",
    "fit_success_rate",
    "fit_r2_mean",
    "unanimous_reach_consensus_rate_mean",
    "unanimous_correct_consensus_rate_mean",
    "unanimous_wrong_only_consensus_rate_mean",
    "unanimous_mean_first_consensus_t_mean",
    "majority_reach_consensus_rate_mean",
    "majority_correct_consensus_rate_mean",
    "majority_wrong_only_consensus_rate_mean",
    "majority_mean_first_consensus_t_mean",
]


def _std(values: list[float]) -> float | None:
    if len(values) < 2:
        return 0.0 if len(values) == 1 else None
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def _mean_field(group: list[dict[str, Any]], field: str) -> float | None:
    values = [
        float(item[field])
        for item in group
        if isinstance(item.get(field), (int, float)) and math.isfinite(float(item[field]))
    ]
    if not values:
        return None
    return sum(values) / len(values)


def _best_run_by_beta_gat(group: list[dict[str, Any]]) -> dict[str, Any]:
    best_record: dict[str, Any] | None = None
    best_beta: float | None = None
    for item in group:
        beta = item.get("beta_gat")
        if not isinstance(beta, (int, float)) or not math.isfinite(float(beta)):
            continue
        beta_value = float(beta)
        if best_beta is None or beta_value > best_beta:
            best_beta = beta_value
            best_record = item

    if best_record is None or best_beta is None:
        return {
            "beta_gat_best": None,
            "beta_gat_best_seed": None,
            "beta_gap_at_best": None,
            "exceeds_hst_bound_at_best": None,
        }

    beta_gap = best_record.get("beta_gap")
    beta_gap_value = (
        float(beta_gap)
        if isinstance(beta_gap, (int, float)) and math.isfinite(float(beta_gap))
        else None
    )
    exceeds = best_record.get("exceeds_hst_bound")
    return {
        "beta_gat_best": best_beta,
        "beta_gat_best_seed": best_record.get("seed"),
        "beta_gap_at_best": beta_gap_value,
        "exceeds_hst_bound_at_best": exceeds if isinstance(exceeds, bool) else None,
    }


def aggregate_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mean/std and best-seed stats for each (num_nodes, signal_quality, topology) cell."""
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("seed") is None:
            continue
        key = tuple(record.get(field) for field in AGGREGATE_GROUP_KEYS)
        buckets[key].append(record)

    aggregated: list[dict[str, Any]] = []
    for key, group in sorted(buckets.items(), key=lambda item: item[0]):
        beta_gat_values = [
            float(item["beta_gat"])
            for item in group
            if isinstance(item.get("beta_gat"), (int, float))
            and math.isfinite(float(item["beta_gat"]))
        ]
        beta_gat_se_values = [
            float(item["beta_gat_se"])
            for item in group
            if isinstance(item.get("beta_gat_se"), (int, float))
            and math.isfinite(float(item["beta_gat_se"]))
        ]
        beta_gat_ci_lower_values = [
            float(item["beta_gat_ci_lower"])
            for item in group
            if isinstance(item.get("beta_gat_ci_lower"), (int, float))
            and math.isfinite(float(item["beta_gat_ci_lower"]))
        ]
        beta_gat_ci_upper_values = [
            float(item["beta_gat_ci_upper"])
            for item in group
            if isinstance(item.get("beta_gat_ci_upper"), (int, float))
            and math.isfinite(float(item["beta_gat_ci_upper"]))
        ]
        beta_gap_values = [
            float(item["beta_gap"])
            for item in group
            if isinstance(item.get("beta_gap"), (int, float))
            and math.isfinite(float(item["beta_gap"]))
        ]
        beta_hst_values = [
            float(item["beta_hst_max"])
            for item in group
            if isinstance(item.get("beta_hst_max"), (int, float))
            and math.isfinite(float(item["beta_hst_max"]))
        ]
        r2_values = [
            float(item["fit_r2"])
            for item in group
            if isinstance(item.get("fit_r2"), (int, float))
            and math.isfinite(float(item["fit_r2"]))
        ]
        exceeds_values = [
            bool(item["exceeds_hst_bound"])
            for item in group
            if isinstance(item.get("exceeds_hst_bound"), bool)
        ]
        convergence_warning_values = [
            bool(item["convergence_warning"])
            for item in group
            if isinstance(item.get("convergence_warning"), bool)
        ]
        fit_success_values = [
            bool(item["fit_success"])
            for item in group
            if isinstance(item.get("fit_success"), bool)
        ]

        num_nodes, signal_quality, topology = key
        best_run = _best_run_by_beta_gat(group)
        aggregated.append(
            {
                "num_nodes": num_nodes,
                "signal_quality": signal_quality,
                "topology": topology,
                "n_seeds": len({item["seed"] for item in group}),
                "beta_gat_mean": (
                    sum(beta_gat_values) / len(beta_gat_values) if beta_gat_values else None
                ),
                "beta_gat_std": _std(beta_gat_values),
                "beta_gat_se_mean": (
                    sum(beta_gat_se_values) / len(beta_gat_se_values)
                    if beta_gat_se_values
                    else None
                ),
                "beta_gat_ci_lower_mean": (
                    sum(beta_gat_ci_lower_values) / len(beta_gat_ci_lower_values)
                    if beta_gat_ci_lower_values
                    else None
                ),
                "beta_gat_ci_upper_mean": (
                    sum(beta_gat_ci_upper_values) / len(beta_gat_ci_upper_values)
                    if beta_gat_ci_upper_values
                    else None
                ),
                **best_run,
                "beta_hst_max_mean": (
                    sum(beta_hst_values) / len(beta_hst_values) if beta_hst_values else None
                ),
                "beta_gap_mean": (
                    sum(beta_gap_values) / len(beta_gap_values) if beta_gap_values else None
                ),
                "beta_gap_std": _std(beta_gap_values),
                "exceeds_hst_bound_rate": (
                    sum(1 for value in exceeds_values if value) / len(exceeds_values)
                    if exceeds_values
                    else None
                ),
                "convergence_warning_rate": (
                    sum(1 for value in convergence_warning_values if value)
                    / len(convergence_warning_values)
                    if convergence_warning_values
                    else None
                ),
                "fit_success_rate": (
                    sum(1 for value in fit_success_values if value) / len(fit_success_values)
                    if fit_success_values
                    else None
                ),
                "fit_r2_mean": sum(r2_values) / len(r2_values) if r2_values else None,
                "unanimous_reach_consensus_rate_mean": _mean_field(
                    group, "unanimous_reach_consensus_rate"
                ),
                "unanimous_correct_consensus_rate_mean": _mean_field(
                    group, "unanimous_correct_consensus_rate"
                ),
                "unanimous_wrong_only_consensus_rate_mean": _mean_field(
                    group, "unanimous_wrong_only_consensus_rate"
                ),
                "unanimous_mean_first_consensus_t_mean": _mean_field(
                    group, "unanimous_mean_first_consensus_t"
                ),
                "majority_reach_consensus_rate_mean": _mean_field(
                    group, "majority_reach_consensus_rate"
                ),
                "majority_correct_consensus_rate_mean": _mean_field(
                    group, "majority_correct_consensus_rate"
                ),
                "majority_wrong_only_consensus_rate_mean": _mean_field(
                    group, "majority_wrong_only_consensus_rate"
                ),
                "majority_mean_first_consensus_t_mean": _mean_field(
                    group, "majority_mean_first_consensus_t"
                ),
            }
        )
    return aggregated


def build_seed_aggregate_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """JSON-serializable seed aggregate over (n, q, topology) cells."""
    cells = aggregate_records(records)
    seeds = sorted({int(r["seed"]) for r in records if r.get("seed") is not None})
    by_cell_key: dict[str, dict[str, Any]] = {}
    for cell in cells:
        n = cell.get("num_nodes")
        q = cell.get("signal_quality")
        topo = cell.get("topology")
        key = f"n_{n}/q_{q}/{topo}"
        by_cell_key[key] = cell
    return {
        "num_records": len(records),
        "num_seeds": len(seeds),
        "seeds": seeds,
        "cells": cells,
        "by_cell_key": by_cell_key,
    }


__all__ = [
    "AGGREGATE_COLUMNS",
    "AGGREGATE_GROUP_KEYS",
    "aggregate_records",
    "build_seed_aggregate_summary",
]
