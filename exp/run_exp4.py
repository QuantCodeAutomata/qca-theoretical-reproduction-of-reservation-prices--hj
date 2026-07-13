import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from microstructure.intensity_derivation import (
    derive_exponential_intensity,
    derive_powerlaw_intensity,
    compute_exponential_params,
    compute_powerlaw_params,
    exponential_intensity_func,
    powerlaw_intensity_func,
    print_derivation_summary,
)
from microstructure.intensity_plots import (
    plot_exponential_intensity_curves,
    plot_powerlaw_intensity_curves,
    plot_shape_comparison,
    plot_paper_intensity,
)


def main():
    print("=== Experiment 4: Execution Intensity Derivation ===\n")

    results_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results'
    )
    os.makedirs(results_dir, exist_ok=True)

    # 1. Print derivation summary
    print_derivation_summary()

    # 2. Symbolic derivation – exponential intensity
    print("\n--- Symbolic Derivation: Exponential Intensity ---")
    exp_result = derive_exponential_intensity()
    print(f"  Intensity expression : {exp_result['intensity_expr']}")
    print(f"  k formula            : {exp_result['k_formula']}")
    print(f"  A formula            : {exp_result['A_formula']}")
    print(f"  Tail probability     : {exp_result['tail_probability']}")
    print("  Derivation steps:")
    for step in exp_result['derivation_steps']:
        print(f"    {step}")

    # 3. Symbolic derivation – power-law intensity
    print("\n--- Symbolic Derivation: Power-Law Intensity ---")
    pow_result = derive_powerlaw_intensity()
    print(f"  Intensity expression : {pow_result['intensity_expr']}")
    print(f"  Exponent             : {pow_result['exponent']}")
    print("  Derivation steps:")
    for step in pow_result['derivation_steps']:
        print(f"    {step}")

    # 4. Numerical illustrations: (A, k) for various alpha and K
    print("\n--- Numerical Parameter Illustrations ---")
    print(f"  {'alpha':>6}  {'K':>5}  {'A':>10}  {'k':>6}")
    print("  " + "-" * 34)
    for alpha in [1.4, 1.5, 1.53]:
        for K in [0.5, 1.0, 2.0]:
            A, k = compute_exponential_params(Lambda_val=1.0, alpha_val=alpha, K_val=K)
            print(f"  {alpha:>6.2f}  {K:>5.1f}  {A:>10.4f}  {k:>6.3f}")

    # 5. Generate plots
    alpha_values = [1.4, 1.5, 1.53]
    K_values = [0.5, 1.0, 2.0]
    beta_values = [0.3, 0.5, 1.0]

    print("\n--- Generating Plots ---")

    path_exp = os.path.join(results_dir, 'exp4_exponential_intensity.png')
    plot_exponential_intensity_curves(
        alpha_values=alpha_values,
        K_values=K_values,
        save_path=path_exp,
    )
    print(f"  Saved: {path_exp}")

    path_pow = os.path.join(results_dir, 'exp4_powerlaw_intensity.png')
    plot_powerlaw_intensity_curves(
        alpha_values=alpha_values,
        beta_values=beta_values,
        save_path=path_pow,
    )
    print(f"  Saved: {path_pow}")

    path_cmp = os.path.join(results_dir, 'exp4_shape_comparison.png')
    plot_shape_comparison(
        alpha_val=1.5, K_val=1.0, beta_val=0.5,
        save_path=path_cmp,
    )
    print(f"  Saved: {path_cmp}")

    path_paper = os.path.join(results_dir, 'exp4_paper_intensity.png')
    plot_paper_intensity(
        A=140.0, k=1.5,
        save_path=path_paper,
    )
    print(f"  Saved: {path_paper}")

    # 6. Save text results
    results_txt = os.path.join(results_dir, 'exp4_results.txt')
    with open(results_txt, 'w') as f:
        f.write("=== Experiment 4: Execution Intensity Derivation Results ===\n\n")

        f.write("--- Exponential Intensity ---\n")
        f.write(f"  intensity_expr : {exp_result['intensity_expr']}\n")
        f.write(f"  k_formula      : {exp_result['k_formula']}\n")
        f.write(f"  A_formula      : {exp_result['A_formula']}\n")
        f.write(f"  tail_prob      : {exp_result['tail_probability']}\n")
        for step in exp_result['derivation_steps']:
            f.write(f"  {step}\n")

        f.write("\n--- Power-Law Intensity ---\n")
        f.write(f"  intensity_expr : {pow_result['intensity_expr']}\n")
        f.write(f"  exponent       : {pow_result['exponent']}\n")
        for step in pow_result['derivation_steps']:
            f.write(f"  {step}\n")

        f.write("\n--- Numerical Parameter Illustrations ---\n")
        f.write(f"  {'alpha':>6}  {'K':>5}  {'A':>10}  {'k':>6}\n")
        f.write("  " + "-" * 34 + "\n")
        for alpha in [1.4, 1.5, 1.53]:
            for K in [0.5, 1.0, 2.0]:
                A, k = compute_exponential_params(Lambda_val=1.0, alpha_val=alpha, K_val=K)
                f.write(f"  {alpha:>6.2f}  {K:>5.1f}  {A:>10.4f}  {k:>6.3f}\n")

        f.write("\n--- Plots Saved ---\n")
        for p in [path_exp, path_pow, path_cmp, path_paper]:
            f.write(f"  {p}\n")

    print(f"\n  Results written to: {results_txt}")
    print("\n=== Experiment 4 Complete ===")


if __name__ == '__main__':
    main()
