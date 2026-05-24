"""Utilities for computing the theoretical HST learning-rate bound."""

from __future__ import annotations

import math


def compute_beta_hst_max(signal_quality: float) -> float:
    """
    Compute the HST upper bound on equilibrium learning speed for binary signals.

    Notes:
        Huang, Strack, and Tamuz (2024), Theorem 1, show that equilibrium speed
        is bounded by:

            M = 2 * sup |ell(s)|

        where ell(s) is the log-likelihood ratio induced by one private signal.
        In our binary symmetric signal model:

            P(s=theta) = q, P(s!=theta) = 1-q
            ell in {+log(q/(1-q)), -log(q/(1-q))}

        hence:

            beta_HST_max(q) = M = 2 * log(q/(1-q))

        for q in (0.5, 1). This matches the paper's calibration intuition:
        q=0.9 gives M ≈ 4.394.

        Source: https://arxiv.org/pdf/2112.14265 (Eq. (1), Theorem 1).

        Interpretation:
            This is an upper bound for equilibrium learning speed under the HST
            model assumptions (strategic agents in Nash equilibrium), not a
            universal information-theoretic impossibility bound for arbitrary
            learning algorithms.
    """
    if not (0.5 < signal_quality < 1.0):
        raise ValueError("signal_quality must satisfy 0.5 < signal_quality < 1.0.")

    q = float(signal_quality)
    beta_hst = 2.0 * math.log(q / (1.0 - q))
    if not math.isfinite(beta_hst) or beta_hst <= 0.0:
        raise ValueError("Computed beta_hst_max is invalid.")
    return beta_hst


__all__ = ["compute_beta_hst_max"]
