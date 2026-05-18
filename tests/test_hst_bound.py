import math

import pytest

from src.hst_bound import compute_beta_hst_max


@pytest.mark.parametrize("q", [0.6, 0.8, 0.9])
def test_compute_beta_hst_max_valid_inputs(q: float) -> None:
    beta = compute_beta_hst_max(q)
    assert math.isfinite(beta)
    assert beta > 0.0


@pytest.mark.parametrize("q", [0.5, 1.0, -0.1, 1.2])
def test_compute_beta_hst_max_invalid_inputs_raise(q: float) -> None:
    with pytest.raises(ValueError):
        compute_beta_hst_max(q)


def test_compute_beta_hst_max_matches_hst_expression_for_q_0p9() -> None:
    beta = compute_beta_hst_max(0.9)
    assert beta == pytest.approx(2.0 * math.log(9.0), rel=1e-9)
