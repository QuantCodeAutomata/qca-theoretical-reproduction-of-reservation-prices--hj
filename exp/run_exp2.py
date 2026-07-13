import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim.table_faithful import run_all_gammas_table_faithful, generate_sample_path_data
from sim.common import table_faithful_spread, print_spread_comparison
from analysis.tables import (
    compute_summary_stats,
    format_summary_table,
    add_spread_column,
    compare_with_paper_targets,
)
from analysis.plots import plot_profit_histograms, plot_sample_path, plot_all_profit_histograms


GAMMAS = [0.1, 0.01, 0.5]
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
TRACK = "table_faithful"


def main() -> None:
    print("=== Experiment 2: Table-Faithful Monte Carlo Simulation ===\n")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Step 1 – spread comparison
    print_spread_comparison()

    # Step 2 – run simulation
    t0 = time.time()
    df = run_all_gammas_table_faithful(GAMMAS, n_paths=1000, seed=42)
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
    csv_path = os.path.join(RESULTS_DIR, "exp2_summary.csv")
    summary.to_csv(csv_path, index=False)
    print(f"\nSummary saved to {csv_path}")

    # Step 6 – profit histograms
    print("\nGenerating profit histograms...")
    plot_all_profit_histograms(df, GAMMAS, track_name=TRACK, results_dir=RESULTS_DIR)

    # Step 7 – sample path plot
    print("\nGenerating sample path plot (gamma=0.1)...")
    path_data = generate_sample_path_data(gamma=0.1, seed=0)
    sp_path = os.path.join(RESULTS_DIR, "exp2_sample_path_gamma_0.1.png")
    plot_sample_path(path_data, gamma=0.1, track_name=TRACK, save_path=sp_path)
    print(f"  Saved: {sp_path}")

    # Step 8 – text results file
    txt_path = os.path.join(RESULTS_DIR, "exp2_results.txt")
    with open(txt_path, "w") as f:
        f.write("=== Experiment 2: Table-Faithful Monte Carlo Results ===\n\n")
        f.write(f"Simulation time: {elapsed:.1f}s\n")
        f.write(f"Total path records: {len(df)}\n\n")
        f.write("--- Summary Table ---\n")
        f.write(format_summary_table(summary))
        f.write("\n\n--- Paper Target Comparison (gamma=0.1) ---\n")
        f.write(comparison.to_string(index=False))
        f.write("\n")
    print(f"Results text saved to {txt_path}")

    print("\n=== Experiment 2 Complete ===")


if __name__ == "__main__":
    main()
