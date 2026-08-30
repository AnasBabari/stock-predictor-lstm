"""Mathematically validated proper scoring rules including exact Student-t Continuous Ranked Probability Score (CRPS)."""

from __future__ import annotations

import math

import numpy as np
import scipy.integrate as integrate
import scipy.special as special
import scipy.stats as stats


def compute_student_t_crps_quadrature(
    y: float,
    loc: float,
    scale: float,
    df: float,
) -> float:
    """Numerical quadrature oracle for CRPS: integral of (F(z) - 1{z >= y})^2 dz."""
    dist = stats.t(df=df, loc=loc, scale=scale)

    def integrand(z: float) -> float:
        f_val = dist.cdf(z)
        indicator = 1.0 if z >= y else 0.0
        return float((f_val - indicator) ** 2)

    val1, _ = integrate.quad(integrand, -np.inf, y, limit=100)
    val2, _ = integrate.quad(integrand, y, np.inf, limit=100)
    return float(val1 + val2)


def compute_student_t_crps_mc(
    y: float,
    loc: float,
    scale: float,
    df: float,
    n_samples: int = 1_000_000,
    seed: int = 42,
) -> float:
    """Monte Carlo verification oracle based on energy representation: E|X - y| - 0.5 E|X - X'|."""
    rng = np.random.default_rng(seed)
    dist = stats.t(df=df, loc=loc, scale=scale)
    x1 = dist.rvs(size=n_samples, random_state=rng)
    x2 = dist.rvs(size=n_samples, random_state=rng)
    term1 = float(np.mean(np.abs(x1 - y)))
    term2 = float(0.5 * np.mean(np.abs(x1 - x2)))
    return term1 - term2


def student_t_crps(
    y_true: np.ndarray | float,
    loc: np.ndarray | float,
    scale: np.ndarray | float,
    df: float = 5.0,
) -> np.ndarray | float:
    """Production analytical vectorized Student-t CRPS.

    CRPS(F, y) = scale * [ z * (2*F(z) - 1) + 2*f(z)*(df + z^2)/(df - 1) - 0.5*E|Z - Z'| ]
    where 0.5*E|Z - Z'| = 2*sqrt(df)*B(df - 0.5, 0.5) / ( (df - 1) * B(df/2, 0.5)^2 ).
    Requires df > 1.0 (variance requires df > 2.0). Scale must be strictly positive.
    """
    if df <= 1.0:
        raise ValueError(f"CRPS for Student-t requires df > 1.0, got {df}")

    y_arr = np.asarray(y_true, dtype=np.float64)
    loc_arr = np.asarray(loc, dtype=np.float64)
    scale_arr = np.asarray(scale, dtype=np.float64)

    if np.any(scale_arr <= 0.0):
        raise ValueError("Student-t scale parameter must be strictly positive (> 0.0).")

    # Standardized residual
    z = (y_arr - loc_arr) / scale_arr

    # Evaluate standard Student-t CDF and PDF
    dist = stats.t(df=df)
    cdf_z = dist.cdf(z)
    pdf_z = dist.pdf(z)

    # First term: E|Z - z|
    e_abs_z = z * (2.0 * cdf_z - 1.0) + 2.0 * pdf_z * ((df + z**2) / (df - 1.0))

    # Second term: 0.5 * E|Z - Z'|
    half_diff = (
        2.0
        * math.sqrt(df)
        * special.beta(df - 0.5, 0.5)
        / (special.beta(0.5 * df, 0.5) ** 2 * (df - 1.0))
    )

    crps = scale_arr * (e_abs_z - half_diff)

    if np.ndim(y_true) == 0 and np.ndim(loc) == 0 and np.ndim(scale) == 0:
        return float(crps)
    return crps
