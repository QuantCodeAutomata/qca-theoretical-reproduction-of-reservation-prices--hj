"""
Symbolic and numerical formulas from the Avellaneda-Stoikov (2008) market-making model.

All derivations follow the paper's notation.  Module-level sympy symbols are
defined once and reused by every function so callers can compose expressions
freely.
"""
import numpy as np
import sympy as sp
from typing import Dict, Tuple, List

# ---------------------------------------------------------------------------
# Module-level symbolic variables — paper notation
# ---------------------------------------------------------------------------
x, s, q, t, T, sigma, gamma, A, k, omega = sp.symbols(
    'x s q t T sigma gamma A k omega',
    real=True, positive=True
)
# q can be negative (inventory); t is non-negative
q = sp.Symbol('q', real=True)
t = sp.Symbol('t', real=True, nonneg=True)


# ---------------------------------------------------------------------------
# 1. Frozen-inventory value function
# ---------------------------------------------------------------------------
def frozen_value_function(x_val, s_val, q_val, t_val, T_val, sigma_val, gamma_val) -> sp.Expr:
    """Frozen-inventory value function under exponential utility.

    S_T ~ N(s, q^2*sigma^2*(T-t)), so
    E[-exp(-gamma*(x+q*S_T))] =
        -exp(-gamma*x) * exp(-gamma*q*s) * exp(gamma^2*q^2*sigma^2*(T-t)/2)
    """
    # Custom — Context7 found no library equivalent (derived from Gaussian MGF)
    exponent = (
        -gamma_val * x_val
        - gamma_val * q_val * s_val
        + (gamma_val ** 2 * q_val ** 2 * sigma_val ** 2 * (T_val - t_val)) / 2
    )
    return -sp.exp(exponent)


# ---------------------------------------------------------------------------
# 2. Reservation ask price
# ---------------------------------------------------------------------------
def reservation_ask_price(s, q, t, T, sigma, gamma) -> sp.Expr:
    """Reservation ask price from indifference condition v(x+r^a, s, q-1, t) = v(x, s, q, t)."""
    # Custom — derived algebraically from frozen value function equality
    return s + ((1 - 2 * q) * gamma * sigma ** 2 * (T - t)) / 2


# ---------------------------------------------------------------------------
# 3. Reservation bid price
# ---------------------------------------------------------------------------
def reservation_bid_price(s, q, t, T, sigma, gamma) -> sp.Expr:
    """Reservation bid price from indifference condition v(x-r^b, s, q+1, t) = v(x, s, q, t)."""
    return s + ((-1 - 2 * q) * gamma * sigma ** 2 * (T - t)) / 2


# ---------------------------------------------------------------------------
# 4. Reservation mid-price
# ---------------------------------------------------------------------------
def reservation_price(s, q, t, T, sigma, gamma) -> sp.Expr:
    """Mid-point of reservation bid/ask = s - q*gamma*sigma^2*(T-t)."""
    r_a = reservation_ask_price(s, q, t, T, sigma, gamma)
    r_b = reservation_bid_price(s, q, t, T, sigma, gamma)
    return sp.simplify((r_a + r_b) / 2)


# ---------------------------------------------------------------------------
# 5. Symbolic indifference verification — ask side
# ---------------------------------------------------------------------------
def verify_indifference_ask(symbolic: bool = True) -> bool:
    """Verify symbolically that v(x+r^a, s, q-1, t) = v(x, s, q, t).

    Strategy: compare the exponents of the two frozen value functions;
    their difference must simplify to zero under sp.expand.
    """
    r_a = reservation_ask_price(s, q, t, T, sigma, gamma)

    # Exponents of lhs = frozen_value_function(x+r_a, s, q-1, t, ...)
    exp_lhs = (
        -gamma * (x + r_a)
        - gamma * (q - 1) * s
        + gamma ** 2 * (q - 1) ** 2 * sigma ** 2 * (T - t) / 2
    )
    # Exponents of rhs = frozen_value_function(x, s, q, t, ...)
    exp_rhs = (
        -gamma * x
        - gamma * q * s
        + gamma ** 2 * q ** 2 * sigma ** 2 * (T - t) / 2
    )

    diff = sp.expand(exp_lhs - exp_rhs)
    result = diff == 0
    print(f"verify_indifference_ask: {'PASS' if result else 'FAIL'} (exponent diff = {diff})")
    return result


# ---------------------------------------------------------------------------
# 6. Symbolic indifference verification — bid side
# ---------------------------------------------------------------------------
def verify_indifference_bid(symbolic: bool = True) -> bool:
    """Verify symbolically that v(x-r^b, s, q+1, t) = v(x, s, q, t)."""
    r_b = reservation_bid_price(s, q, t, T, sigma, gamma)

    exp_lhs = (
        -gamma * (x - r_b)
        - gamma * (q + 1) * s
        + gamma ** 2 * (q + 1) ** 2 * sigma ** 2 * (T - t) / 2
    )
    exp_rhs = (
        -gamma * x
        - gamma * q * s
        + gamma ** 2 * q ** 2 * sigma ** 2 * (T - t) / 2
    )

    diff = sp.expand(exp_lhs - exp_rhs)
    result = diff == 0
    print(f"verify_indifference_bid: {'PASS' if result else 'FAIL'} (exponent diff = {diff})")
    return result


# ---------------------------------------------------------------------------
# 7. Approximate optimal ask quote distance
# ---------------------------------------------------------------------------
def quote_distance_ask_approx(q_val, t_val, T_val, sigma_val, gamma_val, k_val) -> sp.Expr:
    """Approximate optimal ask quote distance under exponential intensity lambda=A*exp(-k*delta).

    delta_a = ((1-2*q)*gamma*sigma^2*(T-t))/2 + (1/gamma)*ln(1+gamma/k)
    """
    # Custom — Context7 found no library equivalent (paper Eq. approximate formula)
    inventory_term = ((1 - 2 * q_val) * gamma_val * sigma_val ** 2 * (T_val - t_val)) / 2
    impact_term = (1 / gamma_val) * sp.log(1 + gamma_val / k_val)
    return inventory_term + impact_term


# ---------------------------------------------------------------------------
# 8. Approximate optimal bid quote distance
# ---------------------------------------------------------------------------
def quote_distance_bid_approx(q_val, t_val, T_val, sigma_val, gamma_val, k_val) -> sp.Expr:
    """Approximate optimal bid quote distance under exponential intensity lambda=A*exp(-k*delta).

    delta_b = ((1+2*q)*gamma*sigma^2*(T-t))/2 + (1/gamma)*ln(1+gamma/k)
    """
    inventory_term = ((1 + 2 * q_val) * gamma_val * sigma_val ** 2 * (T_val - t_val)) / 2
    impact_term = (1 / gamma_val) * sp.log(1 + gamma_val / k_val)
    return inventory_term + impact_term


# ---------------------------------------------------------------------------
# 9. Full spread (ask distance + bid distance)
# ---------------------------------------------------------------------------
def full_spread_approx(t_val, T_val, sigma_val, gamma_val, k_val) -> sp.Expr:
    """Full bid-ask spread = delta_a + delta_b = gamma*sigma^2*(T-t) + (2/gamma)*ln(1+gamma/k)."""
    vol_term = gamma_val * sigma_val ** 2 * (T_val - t_val)
    impact_term = (2 / gamma_val) * sp.log(1 + gamma_val / k_val)
    return vol_term + impact_term


# ---------------------------------------------------------------------------
# 10. Inventory-independent (constant) spread component
# ---------------------------------------------------------------------------
def table_faithful_spread(gamma_val, k_val) -> sp.Expr:
    """Constant spread component used in table-faithful simulations.

    Returns (2/gamma)*ln(1+gamma/k).
    """
    return (2 / gamma_val) * sp.log(1 + gamma_val / k_val)


# ---------------------------------------------------------------------------
# 11. Symbolic spread decomposition verification
# ---------------------------------------------------------------------------
def verify_spread_decomposition() -> bool:
    """Verify symbolically that delta_a + delta_b == full_spread_approx."""
    delta_a = quote_distance_ask_approx(q, t, T, sigma, gamma, k)
    delta_b = quote_distance_bid_approx(q, t, T, sigma, gamma, k)
    full = full_spread_approx(t, T, sigma, gamma, k)

    diff = sp.simplify(delta_a + delta_b - full)
    result = diff == 0
    print(f"verify_spread_decomposition: {'PASS' if result else 'FAIL'} (diff = {diff})")
    return result


# ---------------------------------------------------------------------------
# 12. Stationary (infinite-horizon) reservation price
# ---------------------------------------------------------------------------
def stationary_reservation_price(s, q, gamma_val, sigma_val, omega_val, k_val) -> sp.Expr:
    """Stationary analog of the reservation price under an exponential discount rate omega.

    Returns s - q * gamma * sigma^2 / (2*omega).

    Admissibility condition: omega > 0.5 * gamma^2 * sigma^2 * q^2
    (ensures the value function integral converges).
    """
    return s - q * gamma_val * sigma_val ** 2 / (2 * omega_val)
