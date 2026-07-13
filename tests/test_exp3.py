import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim.common import (
    ask_distance_theory,
    bid_distance_theory,
    reservation_price_num,
    compute_intensity,
)
from sim.theory_faithful import (
    simulate_single_path_inventory_aware_theory,
    simulate_single_path_symmetric_theory,
)


def test_single_path_runs_without_error_inventory_aware_theory():
    """Theory-faithful inventory-aware path should return finite (profit, q, diag)."""
    profit, q, diag = simulate_single_path_inventory_aware_theory(0.1, seed=42)
    assert np.isfinite(profit)
    assert isinstance(diag, dict)


def test_single_path_runs_without_error_symmetric_theory():
    """Theory-faithful symmetric path should return finite (profit, q, diag)."""
    profit, q, diag = simulate_single_path_symmetric_theory(0.1, seed=42)
    assert np.isfinite(profit)


def test_poisson_fills_nonnegative():
    """Fill counts drawn from Poisson must always be non-negative (here via inventory check)."""
    # Run a path and verify that inventory only changes by integer amounts consistent
    # with non-negative Poisson draws (inventory can grow unboundedly but final_q is int-valued).
    profit, q, diag = simulate_single_path_inventory_aware_theory(0.1, seed=7)
    # q must be an integer (sum of non-negative Poisson draws)
    assert q == int(q), f"final_q={q} is not integer-valued"
    assert diag["avg_fills_per_step"] >= 0.0


def test_theory_faithful_inventory_lower_variance():
    """Core paper claim: inventory-aware strategy has lower final-inventory variance."""
    n_paths = 200
    gamma = 0.1

    inv_qs = [
        simulate_single_path_inventory_aware_theory(gamma, seed=i)[1]
        for i in range(n_paths)
    ]
    sym_qs = [
        simulate_single_path_symmetric_theory(gamma, seed=i)[1]
        for i in range(n_paths)
    ]

    std_inv = np.std(inv_qs, ddof=1)
    std_sym = np.std(sym_qs, ddof=1)
    assert std_inv < std_sym, (
        f"Expected inventory-aware std ({std_inv:.4f}) < symmetric std ({std_sym:.4f})"
    )


def test_at_zero_inventory_ask_bid_symmetric():
    """At q=0 the ask and bid distances should be equal (symmetric quotes)."""
    for gamma in [0.1, 0.01, 0.5]:
        da = ask_distance_theory(q=0.0, t=0.5, T=1.0, sigma=2.0, gamma=gamma)
        db = bid_distance_theory(q=0.0, t=0.5, T=1.0, sigma=2.0, gamma=gamma)
        assert abs(da - db) < 1e-12, (
            f"gamma={gamma}: delta_a={da:.6f} != delta_b={db:.6f} at q=0"
        )


def test_profit_is_finite_all_gammas_theory():
    """All theory-faithful profits should be finite across gamma values."""
    for gamma in [0.1, 0.01, 0.5]:
        for i in range(10):
            profit, _, _ = simulate_single_path_inventory_aware_theory(gamma, seed=i)
            assert np.isfinite(profit), f"Non-finite profit: gamma={gamma}, seed={i}"
            profit_s, _, _ = simulate_single_path_symmetric_theory(gamma, seed=i)
            assert np.isfinite(profit_s), f"Non-finite sym profit: gamma={gamma}, seed={i}"


def test_diagnostics_structure():
    """Diagnostics dict must contain all required keys."""
    required_keys = {
        "n_negative_delta_a",
        "n_negative_delta_b",
        "max_intensity",
        "avg_fills_per_step",
    }
    _, _, diag = simulate_single_path_inventory_aware_theory(0.1, seed=0)
    assert required_keys.issubset(diag.keys()), f"Missing keys: {required_keys - diag.keys()}"
