import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim.theory_faithful import (
    run_all_gammas_theory_faithful,
    generate_sample_path_data_theory,
)
from sim.common import full_spread, table_faithful_spread, print_spread_comparison
from analysis.tables import (
    compute_summary_stats,
    format_summary_table,
    add_spread_column,
    compare_with_paper_targets,
)
from analysis.plots import plot_profit_histograms, plot_sample_path, plot_all_profit_histograms


GAMMAS = [0.1, 0.01, 0.5]
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
TRACK = "theory_faithful"


def _compare_spreads() -> None:
    """Print a note on where theory-faithful spread differs from table-faithful."""
    print("\n--- Spread Comparison: theory_faithful vs table_faithful ---")
    print(
        f"{'gamma':>8}  {'table_spread':>14}  {'theory_spread(t=0)':>20}  {'difference':>12}"
    )
    for gamma in GAMMAS:
        ts = table_faithful_spread(gamma)
        fs = full_spread(0.0, 1.0, 2.0, gamma)
        diff = fs - ts
        print(f"{gamma:>8.3f}  {ts:>14.4f}  {fs:>20.4f}  {diff:>+12.4f}")
    print(
        "\nNote: theory_faithful uses the time-varying spread "
        "gamma*sigma^2*(T-t) + (2/gamma)*ln(1+gamma/k), "
        "whereas table_faithful uses only (2/gamma)*ln(1+gamma/k).\n"
    )


def main() -> None:
    print("=== Experiment 3: Theory-Faithful Monte Carlo Simulation ===\n")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Step 1 – spread note
    print_spread_comparison()
    _compare_spreads()

    # Step 2 – run simulation
    t0 = time.time()
    df = run_all_gammas_theory_faithful(GAMMAS, n_paths=1000, seed=42)
    elapsed = time.time() - t0
    print(f"\nSimulation complete in {elapsed:.1f}s  ({len(df)} total path records)\n")

    # Step 3 – summary statistics
    summary = compute_summary_stats(df)
    summary = add_spread_column(summary, track=TRACK)
    print("--- Summary Statistics ---")
    print(format_summary_table(summary))

    # Step 4 – compare with paper targets
    comparison = compare_with_paper_targets(summary)
    print("\n--- Comparison with Paper Targets (gamma=0.1) ---")
    print(comparison.to_string(index=False))

    # Step 5 – save results
    csv_path = os.path.join(RESULTS_DIR, "exp3_summary.csv")
    summary.to_csv(csv_path, index=False)
    print(f"\nSummary saved to {csv_path}")

    # Step 6 – profit histograms
    print("\nGenerating profit histograms...")
    plot_all_profit_histograms(df, GAMMAS, track_name=TRACK, results_dir=RESULTS_DIR)

    # Step 7 – sample path plot
    print("\nGenerating sample path plot (gamma=0.1)...")
    path_data = generate_sample_path_data_theory(gamma=0.1, seed=0)
    sp_path = os.path.join(RESULTS_DIR, "exp3_sample_path_gamma_0.1.png")
    plot_sample_path(path_data, gamma=0.1, track_name=TRACK, save_path=sp_path)
    print(f"  Saved: {sp_path}")

    # Step 8 – compare theory vs table-faithful
    print("\n--- Theory-Faithful vs Table-Faithful: mean profit comparison ---")
    tf_csv = os.path.join(RESULTS_DIR, "exp2_summary.csv")
    if os.path.exists(tf_csv):
        tbl = pd.read_csv(tf_csv)
        tbl["track"] = "table_faithful"
        summary_tag = summary.copy()
        summary_tag["track"] = "theory_faithful"
        combined = pd.concat([tbl, summary_tag], ignore_index=True)
        pivot = combined.pivot_table(
            index=["gamma", "strategy"],
            columns="track",
            values="mean_profit",
        ).reset_index()
        print(pivot.to_string(index=False))
    else:
        print("  (Run Experiment 2 first to enable cross-track comparison)")

    # Step 9 – text results file
    txt_path = os.path.join(RESULTS_DIR, "exp3_results.txt")
    with open(txt_path, "w") as f:
        f.write("=== Experiment 3: Theory-Faithful Monte Carlo Results ===\n\n")
        f.write(f"Simulation time: {elapsed:.1f}s\n")
        f.write(f"Total path records: {len(df)}\n\n")
        f.write("--- Summary Table ---\n")
        f.write(format_summary_table(summary))
        f.write("\n\n--- Paper Target Comparison (gamma=0.1) ---\n")
        f.write(comparison.to_string(index=False))
        f.write("\n")
    print(f"Results text saved to {txt_path}")

    print("\n=== Experiment 3 Complete ===")


if __name__ == "__main__":
    main()
