import numpy as np
from typing import Tuple

# Model parameters (paper values)
S0 = 100.0
T = 1.0
SIGMA = 2.0
A = 140.0
K = 1.5
DT = 0.005
N_STEPS = 200
N_PATHS = 1000
GAMMA_VALUES = [0.1, 0.01, 0.5]


def table_faithful_spread(gamma: float, k: float = K) -> float:
    """Constant spread component used in table-faithful track."""
    # Custom — Context7 found no library equivalent (paper table interpretation)
    return (2 / gamma) * np.log(1 + gamma / k)


def full_spread(t: float, T: float, sigma: float, gamma: float, k: float = K) -> float:
    """Full approximate spread from printed paper equations."""
    return gamma * sigma**2 * (T - t) + (2 / gamma) * np.log(1 + gamma / k)


def reservation_price_num(S: float, q: float, t: float, T: float, sigma: float, gamma: float) -> float:
    """Reservation (indifference) price for a market maker with inventory q."""
    return S - q * gamma * sigma**2 * (T - t)


def ask_distance_theory(q: float, t: float, T: float, sigma: float, gamma: float, k: float = K) -> float:
    """Theory-faithful ask quote distance from the paper equations."""
    return ((1 - 2 * q) * gamma * sigma**2 * (T - t)) / 2 + (1 / gamma) * np.log(1 + gamma / k)


def bid_distance_theory(q: float, t: float, T: float, sigma: float, gamma: float, k: float = K) -> float:
    """Theory-faithful bid quote distance from the paper equations."""
    return ((1 + 2 * q) * gamma * sigma**2 * (T - t)) / 2 + (1 / gamma) * np.log(1 + gamma / k)


def compute_intensity(delta: float, A: float = A, k: float = K) -> float:
    """Arrival intensity for a limit order at distance delta from mid."""
    return A * np.exp(-k * delta)


def print_spread_comparison() -> None:
    """Print spread values across tracks and compare with paper-reported figures."""
    paper_values = {0.1: 1.29, 0.01: 1.33, 0.5: 1.15}
    print("\n--- Spread Comparison ---")
    print(f"{'gamma':>8}  {'table_spread':>14}  {'full_spread(t=0)':>18}  {'paper_value':>12}")
    for gamma in GAMMA_VALUES:
        ts = table_faithful_spread(gamma)
        fs = full_spread(0.0, T, SIGMA, gamma)
        pv = paper_values.get(gamma, float('nan'))
        print(f"{gamma:>8.3f}  {ts:>14.4f}  {fs:>18.4f}  {pv:>12.2f}")
    print()
