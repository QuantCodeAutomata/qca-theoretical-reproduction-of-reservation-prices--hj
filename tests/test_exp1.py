"""
Pytest test suite for Experiment 1: Avellaneda-Stoikov theoretical derivations.

Tests cover:
  - Symbolic frozen value function properties
  - Reservation price formulas (numerical)
  - Quote distance formulas (numerical)
  - Indifference conditions (symbolic)
  - Spread decomposition identity
  - Paper-table spread values
"""
import pytest
import numpy as np
import sympy as sp
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from theory.derivations import (
    frozen_value_function,
    reservation_ask_price,
    reservation_bid_price,
    reservation_price,
    quote_distance_ask_approx,
    quote_distance_bid_approx,
    full_spread_approx,
    table_faithful_spread,
    verify_indifference_ask,
    verify_indifference_bid,
    verify_spread_decomposition,
)
from theory.checks import (
    numerical_reservation_price,
    numerical_ask_distance,
    numerical_bid_distance,
    check_zero_inventory_property,
    check_monotonicity_in_q,
    check_convergence_at_maturity,
    check_gamma_zero_limit,
    check_spread_values,
)

# Shared sympy symbols for tests
_x = sp.Symbol('x', real=True, positive=True)
_s = sp.Symbol('s', real=True, positive=True)
_q = sp.Symbol('q', real=True)
_gamma = sp.Symbol('gamma', real=True, positive=True)
_sigma = sp.Symbol('sigma', real=True, positive=True)
_T = sp.Symbol('T', real=True, positive=True)
_t = sp.Symbol('t', real=True, nonneg=True)


# ---------------------------------------------------------------------------
# 1. Frozen value function — zero inventory
# ---------------------------------------------------------------------------
def test_frozen_value_function_zero_inventory():
    """For q=0, v(x,s,0,t) = -exp(-gamma*x)."""
    v = frozen_value_function(_x, _s, sp.Integer(0), _t, _T, _sigma, _gamma)
    # At q=0 the second and third exponential factors both equal 1
    simplified = sp.simplify(v - (-sp.exp(-_gamma * _x)))
    assert simplified == 0, f"Expected 0, got {simplified}"


# ---------------------------------------------------------------------------
# 2. Reservation price — zero inventory equals s
# ---------------------------------------------------------------------------
def test_reservation_price_zero_inventory():
    """r(s, 0, t) should equal s exactly."""
    assert np.isclose(
        numerical_reservation_price(100, 0, 0.5, 1.0, 2.0, 0.1), 100.0
    )


# ---------------------------------------------------------------------------
# 3. Reservation price — positive inventory below mid
# ---------------------------------------------------------------------------
def test_reservation_price_positive_inventory_below_mid():
    """For q > 0, r(s, q, t) < s (market maker quotes lower to attract buyers)."""
    r = numerical_reservation_price(100, 3, 0.0, 1.0, 2.0, 0.1)
    assert r < 100.0, f"Expected r < 100, got {r}"


# ---------------------------------------------------------------------------
# 4. Reservation price — negative inventory above mid
# ---------------------------------------------------------------------------
def test_reservation_price_negative_inventory_above_mid():
    """For q < 0, r(s, q, t) > s."""
    r = numerical_reservation_price(100, -3, 0.0, 1.0, 2.0, 0.1)
    assert r > 100.0, f"Expected r > 100, got {r}"


# ---------------------------------------------------------------------------
# 5. Reservation price vanishes at maturity
# ---------------------------------------------------------------------------
def test_reservation_price_vanishes_at_maturity():
    """At t=T, T-t=0, so r(s, q, T) = s regardless of q."""
    r = numerical_reservation_price(100, 5, 1.0, 1.0, 2.0, 0.1)
    assert np.isclose(r, 100.0), f"Expected 100.0 at maturity, got {r}"


# ---------------------------------------------------------------------------
# 6. Spread values match paper tables
# ---------------------------------------------------------------------------
def test_spread_values_match_paper_tables():
    """(2/gamma)*ln(1+gamma/k) matches paper table values within tolerance."""
    k = 1.5
    cases = [(0.1, 1.29), (0.01, 1.33), (0.5, 1.15)]
    for gamma, expected in cases:
        spread = float(table_faithful_spread(gamma, k).evalf())
        assert abs(spread - expected) < 0.02, (
            f"gamma={gamma}: spread={spread:.4f}, expected≈{expected}"
        )


# ---------------------------------------------------------------------------
# 7. Spread decomposition: delta_a + delta_b == full_spread
# ---------------------------------------------------------------------------
def test_spread_decomposition():
    """delta_a + delta_b == full_spread for q=2, t=0.5, T=1, sigma=2, gamma=0.1, k=1.5."""
    q_val, t_val, T_val, sigma_val, gamma_val, k_val = 2, 0.5, 1.0, 2.0, 0.1, 1.5

    da = numerical_ask_distance(q_val, t_val, T_val, sigma_val, gamma_val, k_val)
    db = numerical_bid_distance(q_val, t_val, T_val, sigma_val, gamma_val, k_val)
    full = float(full_spread_approx(t_val, T_val, sigma_val, gamma_val, k_val).evalf())

    assert np.isclose(da + db, full, rtol=1e-10), (
        f"delta_a + delta_b = {da + db:.6f}, full_spread = {full:.6f}"
    )


# ---------------------------------------------------------------------------
# 8. Ask and bid distances positive at q=0
# ---------------------------------------------------------------------------
def test_ask_distance_positive_small_q():
    """For q=0, both delta_a and delta_b are positive."""
    da = numerical_ask_distance(0, 0.5, 1.0, 2.0, 0.1, 1.5)
    db = numerical_bid_distance(0, 0.5, 1.0, 2.0, 0.1, 1.5)
    assert da > 0, f"delta_a should be positive, got {da}"
    assert db > 0, f"delta_b should be positive, got {db}"


# ---------------------------------------------------------------------------
# 9. Symbolic indifference — ask
# ---------------------------------------------------------------------------
def test_indifference_ask_verification():
    """verify_indifference_ask() returns True."""
    assert verify_indifference_ask()


# ---------------------------------------------------------------------------
# 10. Symbolic indifference — bid
# ---------------------------------------------------------------------------
def test_indifference_bid_verification():
    """verify_indifference_bid() returns True."""
    assert verify_indifference_bid()


# ---------------------------------------------------------------------------
# 11. Monotonicity in inventory
# ---------------------------------------------------------------------------
def test_monotonicity_in_inventory():
    """check_monotonicity_in_q() passes for all parameter combinations."""
    assert check_monotonicity_in_q()


# ---------------------------------------------------------------------------
# 12. Gamma→0 limit
# ---------------------------------------------------------------------------
def test_gamma_zero_limit():
    """check_gamma_zero_limit() passes (impact term approaches 1/k)."""
    assert check_gamma_zero_limit()


# ---------------------------------------------------------------------------
# 13. Ask decreases, bid increases with positive inventory
# ---------------------------------------------------------------------------
def test_ask_increases_bid_decreases_with_positive_q():
    """Higher positive q -> smaller delta_a and larger delta_b."""
    t_val, T_val, sigma_val, gamma_val, k_val = 0.5, 1.0, 2.0, 0.1, 1.5

    da_q0 = numerical_ask_distance(0, t_val, T_val, sigma_val, gamma_val, k_val)
    da_q2 = numerical_ask_distance(2, t_val, T_val, sigma_val, gamma_val, k_val)
    db_q0 = numerical_bid_distance(0, t_val, T_val, sigma_val, gamma_val, k_val)
    db_q2 = numerical_bid_distance(2, t_val, T_val, sigma_val, gamma_val, k_val)

    assert da_q0 > da_q2, (
        f"delta_a(q=0)={da_q0:.4f} should exceed delta_a(q=2)={da_q2:.4f}"
    )
    assert db_q0 < db_q2, (
        f"delta_b(q=0)={db_q0:.4f} should be less than delta_b(q=2)={db_q2:.4f}"
    )


# ---------------------------------------------------------------------------
# 14. Table-faithful spread formula sanity check
# ---------------------------------------------------------------------------
def test_table_faithful_spread_formula():
    """(2/0.1)*ln(1+0.1/1.5) should be approximately 1.29 (within 0.02)."""
    spread = (2 / 0.1) * np.log(1 + 0.1 / 1.5)
    assert abs(spread - 1.29) < 0.02, f"spread={spread:.5f}, expected≈1.29"
