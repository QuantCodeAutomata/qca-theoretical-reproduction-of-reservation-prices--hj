"""
Experiment 1: Theoretical Reproduction of Reservation Prices,
HJB Quote Conditions, and Approximate Closed-Form Quotes.

Runs symbolic verifications, numerical checks, prints grids, and saves plots
and a summary text file to results/.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from theory.derivations import (
    frozen_value_function, reservation_ask_price, reservation_bid_price,
    reservation_price, quote_distance_ask_approx, quote_distance_bid_approx,
    full_spread_approx, table_faithful_spread,
    verify_indifference_ask, verify_indifference_bid, verify_spread_decomposition,
    stationary_reservation_price,
)
from theory.checks import (
    numerical_reservation_price, numerical_ask_distance, numerical_bid_distance,
    check_zero_inventory_property, check_monotonicity_in_q,
    check_convergence_at_maturity, check_gamma_zero_limit, check_spread_values,
    run_all_checks, print_reservation_price_grid, print_quote_distance_grid,
    S0, T_val, SIGMA, K_val, GAMMA_VALUES,
)

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
def _save(fig, filename: str) -> None:
    path = os.path.join(RESULTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 1: Reservation price shift vs inventory q
# ---------------------------------------------------------------------------
def plot_reservation_price_vs_q() -> None:
    """r(100, q, 0, 1, 2, gamma) - 100 for q in [-5, 5], multiple gamma."""
    q_vals = np.arange(-5, 6)
    fig, ax = plt.subplots(figsize=(7, 4))

    for g in GAMMA_VALUES:
        shifts = [
            numerical_reservation_price(S0, int(q), 0.0, T_val, SIGMA, g) - S0
            for q in q_vals
        ]
        ax.plot(q_vals, shifts, marker='o', label=f"γ={g}")

    ax.axhline(0, color='k', linewidth=0.7, linestyle='--')
    ax.axvline(0, color='k', linewidth=0.7, linestyle='--')
    ax.set_xlabel("Inventory q")
    ax.set_ylabel("Reservation price shift  r(s,q,t) − s")
    ax.set_title("Reservation Price Shift vs Inventory\n"
                 r"(t=0, T=1, σ=2, S=100)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, "exp1_reservation_price_vs_q.png")


# ---------------------------------------------------------------------------
# Plot 2: Quote distances vs time-to-maturity (q=2)
# ---------------------------------------------------------------------------
def plot_quote_distances_vs_time() -> None:
    """delta_a and delta_b vs T-t for q=2, panels per gamma."""
    tau_vals = np.linspace(0, 1, 100)   # T - t
    q_fixed = 2
    k = K_val

    fig, axes = plt.subplots(1, len(GAMMA_VALUES), figsize=(14, 4), sharey=False)

    for ax, g in zip(axes, GAMMA_VALUES):
        da_vals = [
            numerical_ask_distance(q_fixed, T_val - tau, T_val, SIGMA, g, k)
            for tau in tau_vals
        ]
        db_vals = [
            numerical_bid_distance(q_fixed, T_val - tau, T_val, SIGMA, g, k)
            for tau in tau_vals
        ]
        ax.plot(tau_vals, da_vals, label=r"$\delta^a$")
        ax.plot(tau_vals, db_vals, label=r"$\delta^b$", linestyle='--')
        ax.axhline(0, color='k', linewidth=0.6, linestyle=':')
        ax.set_title(f"γ = {g}")
        ax.set_xlabel("Time-to-maturity  T − t")
        ax.set_ylabel("Quote distance")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Quote Distances vs Time-to-Maturity  (q={q_fixed}, σ={SIGMA}, k={k})",
                 y=1.02)
    fig.tight_layout()
    _save(fig, "exp1_quote_distances_vs_time.png")


# ---------------------------------------------------------------------------
# Plot 3: Full spread decomposition vs time
# ---------------------------------------------------------------------------
def plot_spread_decomposition() -> None:
    """Inventory component (gamma*sigma^2*tau) and constant component (2/gamma)*ln(1+gamma/k)."""
    tau_vals = np.linspace(0, 1, 200)
    k = K_val

    fig, ax = plt.subplots(figsize=(8, 5))

    for g in GAMMA_VALUES:
        inv_comp = g * SIGMA ** 2 * tau_vals                     # inventory component
        const_comp = (2 / g) * np.log(1 + g / k)                # constant (scalar)
        total = inv_comp + const_comp

        color = ax.plot(tau_vals, total, label=f"total spread (γ={g})")[0].get_color()
        ax.plot(tau_vals, inv_comp, linestyle=':', color=color,
                label=f"inv. component (γ={g})")
        ax.axhline(const_comp, linestyle='--', color=color, alpha=0.6,
                   label=f"const. component (γ={g}): {const_comp:.3f}")

    ax.set_xlabel("Time-to-maturity  T − t")
    ax.set_ylabel("Spread")
    ax.set_title(r"Spread Decomposition: $\gamma\sigma^2(T-t)$ + $\frac{2}{\gamma}\ln(1+\frac{\gamma}{k})$")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    _save(fig, "exp1_spread_decomposition.png")


# ---------------------------------------------------------------------------
# Save results text file
# ---------------------------------------------------------------------------
def save_results_txt(check_results: dict, spread_vals: dict) -> None:
    path = os.path.join(RESULTS_DIR, "exp1_results.txt")
    lines = []
    lines.append("=" * 60)
    lines.append("Experiment 1: Theoretical Derivations — Results")
    lines.append("=" * 60)

    lines.append("\n--- Check outcomes ---")
    for name, passed in check_results.items():
        lines.append(f"  {name:<25} {'PASS' if passed else 'FAIL'}")

    lines.append("\n--- Spread values (2/gamma)*ln(1+gamma/k) ---")
    lines.append(f"  k = {K_val}")
    for g, val in spread_vals.items():
        lines.append(f"  gamma={g:.3f}  spread={val:.4f}")

    lines.append("\n--- Sample reservation prices (gamma=0.1, S=100, sigma=2, T=1) ---")
    g = 0.1
    lines.append(f"  {'q':>4}  {'t=0.00':>8}  {'t=0.25':>8}  {'t=0.50':>8}  {'t=0.75':>8}")
    for q_val in [-3, -1, 0, 1, 3]:
        row = f"  {q_val:>4}"
        for t_val in [0.0, 0.25, 0.5, 0.75]:
            r = numerical_reservation_price(S0, q_val, t_val, T_val, SIGMA, g)
            row += f"  {r:>8.4f}"
        lines.append(row)

    lines.append("\n--- Sample quote distances (gamma=0.1, sigma=2, T=1, k=1.5, q=2) ---")
    g = 0.1
    q_val = 2
    lines.append(f"  {'t':>6}  {'delta_a':>10}  {'delta_b':>10}")
    for t_val in [0.0, 0.25, 0.5, 0.75]:
        da = numerical_ask_distance(q_val, t_val, T_val, SIGMA, g, K_val)
        db = numerical_bid_distance(q_val, t_val, T_val, SIGMA, g, K_val)
        lines.append(f"  {t_val:>6.2f}  {da:>10.5f}  {db:>10.5f}")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=== Experiment 1: Theoretical Derivations ===\n")

    # 1. Symbolic indifference verifications
    print("--- Symbolic indifference verifications ---")
    ask_ok = verify_indifference_ask()
    bid_ok = verify_indifference_bid()

    # 2. Spread decomposition
    print()
    spread_ok = verify_spread_decomposition()

    # 3. Numerical checks
    check_results = run_all_checks()

    # 4. Grids
    print_reservation_price_grid()
    print_quote_distance_grid()

    # 5. Spread values per gamma
    print("\n--- Spread values per gamma ---")
    spread_vals = check_spread_values()

    # 6. Plots
    print("\n--- Generating plots ---")
    plot_reservation_price_vs_q()
    plot_quote_distances_vs_time()
    plot_spread_decomposition()

    # 7. Save text results
    print("\n--- Saving results ---")
    save_results_txt(check_results, spread_vals)

    # 8. Final summary
    print("\n=== Experiment 1 Complete ===")
    symbolic_ok = ask_ok and bid_ok and spread_ok
    numerical_ok = all(check_results.values())
    print(f"  Symbolic verifications: {'PASS' if symbolic_ok else 'FAIL'}")
    print(f"  Numerical checks:       {'PASS' if numerical_ok else 'FAIL'}")


if __name__ == "__main__":
    main()
