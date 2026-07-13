import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim.common import (
    table_faithful_spread,
    reservation_price_num,
    compute_intensity,
    full_spread,
)
from sim.table_faithful import (
    simulate_single_path_inventory_aware,
    simulate_single_path_symmetric,
)


def test_table_faithful_spread_values():
    """Table-faithful spreads should match paper-reported values within 0.02."""
    assert abs(table_faithful_spread(0.1) - 1.29) < 0.02
    assert abs(table_faithful_spread(0.01) - 1.33) < 0.02
    assert abs(table_faithful_spread(0.5) - 1.15) < 0.02


def test_reservation_price_zero_inventory():
    """Reservation price equals mid-price when inventory is zero."""
    r = reservation_price_num(S=100.0, q=0.0, t=0.5, T=1.0, sigma=2.0, gamma=0.1)
    assert r == 100.0


def test_single_path_runs_without_error_inventory_aware():
    """Inventory-aware single path should return a finite (profit, q) tuple."""
    result = simulate_single_path_inventory_aware(0.1, seed=42)
    assert len(result) == 2
    profit, q = result
    assert np.isfinite(profit)


def test_single_path_runs_without_error_symmetric():
    """Symmetric single path should return a finite (profit, q) tuple."""
    result = simulate_single_path_symmetric(0.1, seed=42)
    assert len(result) == 2
    profit, q = result
    assert np.isfinite(profit)


def test_intensity_decreases_with_distance():
    """Fill intensity must be strictly decreasing in the quote distance."""
    assert compute_intensity(1.0) > compute_intensity(2.0) > compute_intensity(3.0)


def test_inventory_aware_lower_q_variance():
    """Core paper claim: inventory-aware strategy has lower final-inventory variance."""
    n_paths = 200
    gamma = 0.1

    inv_qs = [
        simulate_single_path_inventory_aware(gamma, seed=i)[1]
        for i in range(n_paths)
    ]
    sym_qs = [
        simulate_single_path_symmetric(gamma, seed=i)[1]
        for i in range(n_paths)
    ]

    std_inv = np.std(inv_qs, ddof=1)
    std_sym = np.std(sym_qs, ddof=1)
    assert std_inv < std_sym, (
        f"Expected inventory-aware std ({std_inv:.4f}) < symmetric std ({std_sym:.4f})"
    )


def test_profit_is_finite_all_gammas():
    """All profits should be finite across gamma values."""
    for gamma in [0.1, 0.01, 0.5]:
        for i in range(10):
            profit, _ = simulate_single_path_inventory_aware(gamma, seed=i)
            assert np.isfinite(profit), f"Non-finite profit for gamma={gamma}, seed={i}"
            profit_s, _ = simulate_single_path_symmetric(gamma, seed=i)
            assert np.isfinite(profit_s), f"Non-finite profit for gamma={gamma}, seed={i}"
