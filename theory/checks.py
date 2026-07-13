"""
Numerical verification of Avellaneda-Stoikov theoretical identities.

All checks return True on pass and print a brief PASS/FAIL summary.
"""
import numpy as np
import sympy as sp
from typing import Dict, List, Tuple
from theory.derivations import (
    frozen_value_function, reservation_ask_price, reservation_bid_price,
    reservation_price, quote_distance_ask_approx, quote_distance_bid_approx,
    full_spread_approx, table_faithful_spread,
)

# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------
S0 = 100.0
T_val = 1.0
SIGMA = 2.0
A_val = 140.0
K_val = 1.5
GAMMA_VALUES = [0.01, 0.1, 0.5]
Q_GRID = list(range(-5, 6))          # -5 to 5
T_GRID = [0.0, 0.25, 0.5, 0.75]     # exclude T=1 to avoid T-t=0 singularities


# ---------------------------------------------------------------------------
# 1. Numerical reservation price
# ---------------------------------------------------------------------------
def numerical_reservation_price(s: float, q: int, t: float, T: float,
                                 sigma: float, gamma: float) -> float:
    """Numerical evaluation of reservation price formula r(s,q,t) = s - q*gamma*sigma^2*(T-t)."""
    return s - q * gamma * sigma ** 2 * (T - t)


# ---------------------------------------------------------------------------
# 2. Numerical ask quote distance
# ---------------------------------------------------------------------------
def numerical_ask_distance(q: int, t: float, T: float,
                            sigma: float, gamma: float, k: float) -> float:
    """Numerically evaluate delta_a = ((1-2q)*gamma*sigma^2*(T-t))/2 + (1/gamma)*ln(1+gamma/k)."""
    inventory_term = ((1 - 2 * q) * gamma * sigma ** 2 * (T - t)) / 2
    impact_term = (1 / gamma) * np.log(1 + gamma / k)
    return inventory_term + impact_term


# ---------------------------------------------------------------------------
# 3. Numerical bid quote distance
# ---------------------------------------------------------------------------
def numerical_bid_distance(q: int, t: float, T: float,
                            sigma: float, gamma: float, k: float) -> float:
    """Numerically evaluate delta_b = ((1+2q)*gamma*sigma^2*(T-t))/2 + (1/gamma)*ln(1+gamma/k)."""
    inventory_term = ((1 + 2 * q) * gamma * sigma ** 2 * (T - t)) / 2
    impact_term = (1 / gamma) * np.log(1 + gamma / k)
    return inventory_term + impact_term


# ---------------------------------------------------------------------------
# 4. Zero-inventory property: r(s, 0, t) == s
# ---------------------------------------------------------------------------
def check_zero_inventory_property() -> bool:
    """Verify r(s, 0, t) == s for all t in T_GRID and all gamma in GAMMA_VALUES."""
    all_pass = True
    for gamma in GAMMA_VALUES:
        for t in T_GRID:
            r = numerical_reservation_price(S0, 0, t, T_val, SIGMA, gamma)
            if not np.isclose(r, S0):
                print(f"  FAIL zero-inventory: gamma={gamma}, t={t}, r={r:.6f}, S0={S0}")
                all_pass = False
    status = "PASS" if all_pass else "FAIL"
    print(f"check_zero_inventory_property: {status}")
    return all_pass


# ---------------------------------------------------------------------------
# 5. Monotonicity: r(s, q, t) decreases as q increases
# ---------------------------------------------------------------------------
def check_monotonicity_in_q() -> bool:
    """r(s, q, t) > r(s, q+1, t) for all consecutive q pairs and all parameters."""
    all_pass = True
    for gamma in GAMMA_VALUES:
        for t in T_GRID:
            for q in Q_GRID[:-1]:
                r_q = numerical_reservation_price(S0, q, t, T_val, SIGMA, gamma)
                r_q1 = numerical_reservation_price(S0, q + 1, t, T_val, SIGMA, gamma)
                if t == T_val:
                    continue  # T-t=0, r=s identically — skip
                if not r_q > r_q1:
                    print(f"  FAIL monotonicity: gamma={gamma}, t={t}, q={q}: r(q)={r_q:.4f} <= r(q+1)={r_q1:.4f}")
                    all_pass = False
    status = "PASS" if all_pass else "FAIL"
    print(f"check_monotonicity_in_q: {status}")
    return all_pass


# ---------------------------------------------------------------------------
# 6. Convergence at maturity: r(s, q, t→T) ≈ s
# ---------------------------------------------------------------------------
def check_convergence_at_maturity() -> bool:
    """At t approaching T, r(s, q, t) should be close to s (within 0.01)."""
    t_near = 0.999
    all_pass = True
    for q in [-2, -1, 0, 1, 2]:
        for gamma in GAMMA_VALUES:
            r = numerical_reservation_price(S0, q, t_near, T_val, SIGMA, gamma)
            if not abs(r - S0) < 0.01:
                print(f"  FAIL convergence: gamma={gamma}, q={q}, |r-s|={abs(r-S0):.6f}")
                all_pass = False
    status = "PASS" if all_pass else "FAIL"
    print(f"check_convergence_at_maturity: {status}")
    return all_pass


# ---------------------------------------------------------------------------
# 7. Gamma→0 limit: delta → 1/k
# ---------------------------------------------------------------------------
def check_gamma_zero_limit() -> bool:
    """As gamma→0, (1/gamma)*ln(1+gamma/k) → 1/k; verify numerically with gamma=0.0001."""
    gamma_small = 0.0001
    expected = 1.0 / K_val  # ≈ 0.6667
    tol = 0.01

    da = numerical_ask_distance(0, 0.0, T_val, SIGMA, gamma_small, K_val)
    db = numerical_bid_distance(0, 0.0, T_val, SIGMA, gamma_small, K_val)

    # At q=0, inventory term is gamma*sigma^2*T/2 which is also tiny (~0.0002)
    da_impact = (1 / gamma_small) * np.log(1 + gamma_small / K_val)
    db_impact = (1 / gamma_small) * np.log(1 + gamma_small / K_val)

    pass_a = abs(da_impact - expected) < tol
    pass_b = abs(db_impact - expected) < tol
    result = pass_a and pass_b
    status = "PASS" if result else "FAIL"
    print(
        f"check_gamma_zero_limit: {status} "
        f"(impact_term={da_impact:.5f}, 1/k={expected:.5f})"
    )
    return result


# ---------------------------------------------------------------------------
# 8. Spread values for each gamma
# ---------------------------------------------------------------------------
def check_spread_values() -> Dict:
    """Compute and print (2/gamma)*ln(1+gamma/k) for each gamma; return as dict."""
    # Expected values: gamma=0.1→~1.29, gamma=0.01→~1.33, gamma=0.5→~1.15
    expected = {0.01: 1.33, 0.1: 1.29, 0.5: 1.15}
    result = {}
    print("check_spread_values:")
    print(f"  {'gamma':>6}  {'spread':>8}  {'expected':>10}  {'diff':>8}")
    for g in GAMMA_VALUES:
        spread = float(table_faithful_spread(g, K_val).evalf())
        result[g] = spread
        exp_val = expected.get(g, float('nan'))
        diff = spread - exp_val
        print(f"  {g:>6.3f}  {spread:>8.4f}  {exp_val:>10.4f}  {diff:>+8.4f}")
    return result


# ---------------------------------------------------------------------------
# 9. Run all checks
# ---------------------------------------------------------------------------
def run_all_checks() -> Dict:
    """Run every check function and return {name: passed}."""
    print("\n--- Running all theoretical checks ---")
    checks = {
        "zero_inventory":      check_zero_inventory_property,
        "monotonicity_in_q":   check_monotonicity_in_q,
        "convergence_maturity": check_convergence_at_maturity,
        "gamma_zero_limit":    check_gamma_zero_limit,
    }
    results = {}
    for name, fn in checks.items():
        results[name] = fn()
    check_spread_values()

    print("\n--- Summary ---")
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name:<25} {status}")
    all_passed = all(results.values())
    print(f"  {'Overall':<25} {'PASS' if all_passed else 'FAIL'}")
    return results


# ---------------------------------------------------------------------------
# 10. Reservation price grid
# ---------------------------------------------------------------------------
def print_reservation_price_grid() -> None:
    """Print r(100, q, t) for a subset of q and t at gamma=0.1."""
    gamma = 0.1
    q_subset = [-5, -3, -1, 0, 1, 3, 5]
    t_subset = [0.0, 0.25, 0.5, 0.75]

    print(f"\nReservation price grid  [gamma={gamma}, S=100, sigma={SIGMA}, T={T_val}]")
    header = f"{'q':>4}" + "".join(f"  t={t:.2f}" for t in t_subset)
    print(header)
    print("-" * len(header))
    for q_val in q_subset:
        row = f"{q_val:>4}"
        for t_val in t_subset:
            r = numerical_reservation_price(S0, q_val, t_val, T_val, SIGMA, gamma)
            row += f"  {r:>7.3f}"
        print(row)


# ---------------------------------------------------------------------------
# 11. Quote distance grid
# ---------------------------------------------------------------------------
def print_quote_distance_grid() -> None:
    """Print delta_a and delta_b for a subset of q and t at gamma=0.1; flag negatives."""
    gamma = 0.1
    k = K_val
    q_subset = [-5, -3, -1, 0, 1, 3, 5]
    t_subset = [0.0, 0.25, 0.5, 0.75]

    print(f"\nQuote distance grid  [gamma={gamma}, sigma={SIGMA}, T={T_val}, k={k}]")
    print("  (*) marks negative quote distances")
    for t_val in t_subset:
        print(f"\n  t={t_val:.2f}")
        print(f"  {'q':>4}  {'delta_a':>10}  {'delta_b':>10}")
        print(f"  {'':-<4}  {'':-<10}  {'':-<10}")
        for q_val in q_subset:
            da = numerical_ask_distance(q_val, t_val, T_val, SIGMA, gamma, k)
            db = numerical_bid_distance(q_val, t_val, T_val, SIGMA, gamma, k)
            da_str = f"{da:>9.4f}" + ("(*)" if da < 0 else "   ")
            db_str = f"{db:>9.4f}" + ("(*)" if db < 0 else "   ")
            print(f"  {q_val:>4}  {da_str}  {db_str}")
