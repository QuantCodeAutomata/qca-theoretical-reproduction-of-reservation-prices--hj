import pytest
import numpy as np
import sympy as sp
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from microstructure.intensity_derivation import (
    derive_exponential_intensity,
    derive_powerlaw_intensity,
    compute_exponential_params,
    compute_powerlaw_params,
    exponential_intensity_func,
    powerlaw_intensity_func,
)


def test_exponential_intensity_decreasing():
    """lambda(1) > lambda(2) > lambda(3) for paper params A=140, k=1.5."""
    lam1 = exponential_intensity_func(np.array([1.0]), A=140.0, k=1.5)[0]
    lam2 = exponential_intensity_func(np.array([2.0]), A=140.0, k=1.5)[0]
    lam3 = exponential_intensity_func(np.array([3.0]), A=140.0, k=1.5)[0]
    assert lam1 > lam2 > lam3


def test_exponential_intensity_at_zero():
    """lambda(0) == A = 140."""
    lam0 = exponential_intensity_func(np.array([0.0]), A=140.0, k=1.5)[0]
    assert np.isclose(lam0, 140.0)


def test_exponential_intensity_positive():
    """All values on delta in [0.01, 5] are positive."""
    delta_grid = np.linspace(0.01, 5, 500)
    lam = exponential_intensity_func(delta_grid, A=140.0, k=1.5)
    assert np.all(lam > 0)


def test_powerlaw_intensity_decreasing():
    """lambda(1) > lambda(2) > lambda(3) for any B > 0, rho > 0."""
    lam1 = powerlaw_intensity_func(np.array([1.0]), B=5.0, rho=2.0)[0]
    lam2 = powerlaw_intensity_func(np.array([2.0]), B=5.0, rho=2.0)[0]
    lam3 = powerlaw_intensity_func(np.array([3.0]), B=5.0, rho=2.0)[0]
    assert lam1 > lam2 > lam3


def test_powerlaw_raises_on_nonpositive_delta():
    """powerlaw_intensity_func raises ValueError when delta contains non-positive values."""
    with pytest.raises(ValueError):
        powerlaw_intensity_func(np.array([0.0, 1.0]), B=1.0, rho=1.0)


def test_exponential_params_k_formula():
    """k = alpha * K; compute_exponential_params(Lambda=1, alpha=1.5, K=1.0) -> k == 1.5."""
    A, k = compute_exponential_params(Lambda_val=1.0, alpha_val=1.5, K_val=1.0)
    assert np.isclose(k, 1.5)


def test_powerlaw_params_exponent():
    """rho = alpha / beta; compute_powerlaw_params(Lambda=1, alpha=1.5, beta=0.5) -> rho == 3.0."""
    B, rho = compute_powerlaw_params(Lambda_val=1.0, alpha_val=1.5, beta_val=0.5, c_val=1.0)
    assert np.isclose(rho, 3.0)


def test_derive_exponential_returns_correct_keys():
    """derive_exponential_intensity() returns dict with required keys."""
    result = derive_exponential_intensity()
    assert 'intensity_expr' in result
    assert 'k_formula' in result
    assert 'A_formula' in result
    assert 'tail_probability' in result
    assert 'derivation_steps' in result


def test_log_linear_property():
    """
    Exponential intensity is linear on log scale.
    For any delta1, delta2: ln(lambda(delta2) / lambda(delta1)) == -k * (delta2 - delta1).
    """
    A, k = 140.0, 1.5
    delta1, delta2 = 1.0, 2.5
    lam1 = exponential_intensity_func(np.array([delta1]), A=A, k=k)[0]
    lam2 = exponential_intensity_func(np.array([delta2]), A=A, k=k)[0]
    log_ratio = np.log(lam2 / lam1)
    expected = -k * (delta2 - delta1)
    assert np.isclose(log_ratio, expected, rtol=1e-10)


def test_log_log_property():
    """
    Power-law intensity is linear on log-log scale.
    ln(lambda(2)) - ln(lambda(1)) == -rho * (ln(2) - ln(1)).
    """
    B, rho = 1.0, 2.0
    delta1, delta2 = 1.0, 2.0
    lam1 = powerlaw_intensity_func(np.array([delta1]), B=B, rho=rho)[0]
    lam2 = powerlaw_intensity_func(np.array([delta2]), B=B, rho=rho)[0]
    log_diff = np.log(lam2) - np.log(lam1)
    expected = -rho * (np.log(delta2) - np.log(delta1))
    assert np.isclose(log_diff, expected, rtol=1e-10)
