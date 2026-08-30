"""Comprehensive unit tests and oracle parity benchmarks for proper_scoring_v11."""

import math

import pytest
import scipy.stats as stats

from research.volatility_forecasting.proper_scoring_v11 import (
    compute_student_t_crps_mc,
    compute_student_t_crps_quadrature,
    student_t_crps,
)


def test_crps_reference_anchor():
    # Known benchmark: Student-t(df=5, loc=0, scale=1, y=0)
    crps_val = student_t_crps(y_true=0.0, loc=0.0, scale=1.0, df=5.0)
    quad_val = compute_student_t_crps_quadrature(y=0.0, loc=0.0, scale=1.0, df=5.0)
    assert isinstance(crps_val, float)
    assert abs(crps_val - quad_val) < 1e-10
    assert abs(crps_val - 0.25702536) < 1e-6


@pytest.mark.parametrize("df", [3.0, 5.0, 10.0, 30.0])
@pytest.mark.parametrize("z_val", [-3.0, -1.0, 0.0, 1.0, 3.0])
@pytest.mark.parametrize("scale", [0.005, 0.02, 0.10])
def test_crps_grid_against_quadrature_oracle(df: float, z_val: float, scale: float):
    loc = 0.0
    y = loc + z_val * scale
    prod_val = student_t_crps(y_true=y, loc=loc, scale=scale, df=df)
    quad_val = compute_student_t_crps_quadrature(y=y, loc=loc, scale=scale, df=df)
    assert abs(prod_val - quad_val) < 1e-8


def test_crps_mathematical_invariants():
    df = 5.0
    loc = 10.0
    scale = 2.0

    # 1. Symmetry around location: CRPS(loc + delta) == CRPS(loc - delta)
    crps_pos = student_t_crps(loc + 3.0, loc, scale, df)
    crps_neg = student_t_crps(loc - 3.0, loc, scale, df)
    assert abs(crps_pos - crps_neg) < 1e-12

    # 2. Translation invariance: CRPS(y + c, loc + c, scale) == CRPS(y, loc, scale)
    c = 50.0
    crps_shifted = student_t_crps(13.0 + c, loc + c, scale, df)
    assert abs(crps_pos - crps_shifted) < 1e-12

    # 3. Positive scale homogeneity: CRPS(alpha*y, alpha*loc, alpha*scale) == alpha * CRPS(y, loc, scale)
    alpha = 3.5
    crps_scaled = student_t_crps(alpha * 13.0, alpha * loc, alpha * scale, df)
    assert abs(crps_scaled - alpha * crps_pos) < 1e-12


def test_crps_monte_carlo_oracle_tolerance():
    prod_val = student_t_crps(y_true=0.05, loc=0.0, scale=0.02, df=5.0)
    mc_val = compute_student_t_crps_mc(y=0.05, loc=0.0, scale=0.02, df=5.0, n_samples=500_000)
    assert abs(prod_val - mc_val) < 0.001


def test_crps_gaussian_limit():
    # As df -> inf (e.g. df = 200), Student-t CRPS matches Gaussian CRPS:
    # Gaussian CRPS: s * [ z*(2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi) ]
    scale = 0.02
    y = 0.03
    loc = 0.0
    z = (y - loc) / scale

    t_crps = student_t_crps(y, loc, scale, df=200.0)
    gauss_crps = scale * (
        z * (2.0 * stats.norm.cdf(z) - 1.0) + 2.0 * stats.norm.pdf(z) - 1.0 / math.sqrt(math.pi)
    )
    assert abs(t_crps - gauss_crps) < 1e-4
