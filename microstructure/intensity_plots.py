import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from typing import List, Tuple, Optional
from microstructure.intensity_derivation import (
    exponential_intensity_func, powerlaw_intensity_func,
    compute_exponential_params, compute_powerlaw_params,
    normalize_at_reference,
)


def plot_exponential_intensity_curves(
    alpha_values: List[float],
    K_values: List[float],
    delta_grid: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
) -> None:
    """Plot exponential intensity curves for various (alpha, K) parameter pairs."""
    if delta_grid is None:
        delta_grid = np.linspace(0.01, 5, 500)

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = plt.cm.tab10(np.linspace(0, 0.9, len(alpha_values) * len(K_values)))
    color_idx = 0

    for alpha in alpha_values:
        for K in K_values:
            A, k = compute_exponential_params(Lambda_val=1.0, alpha_val=alpha, K_val=K)
            lam = exponential_intensity_func(delta_grid, A=A, k=k)
            ax.semilogy(
                delta_grid, lam,
                color=colors[color_idx],
                label=f'alpha={alpha}, K={K}  (A={A:.2f}, k={k:.2f})',
                linewidth=1.4,
            )
            color_idx += 1

    # Paper params in bold
    lam_paper = exponential_intensity_func(delta_grid, A=140.0, k=1.5)
    ax.semilogy(
        delta_grid, lam_paper,
        color='black', linewidth=2.5, linestyle='--',
        label='Paper: A=140, k=1.5',
    )

    ax.set_xlabel('delta (price distance)')
    ax.set_ylabel('lambda(delta)  [log scale]')
    ax.set_title('Exponential Execution Intensity: Effect of alpha and K')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_powerlaw_intensity_curves(
    alpha_values: List[float],
    beta_values: List[float],
    delta_grid: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
) -> None:
    """Plot power-law intensity curves for various (alpha, beta) parameter pairs."""
    if delta_grid is None:
        delta_grid = np.linspace(0.1, 5, 500)   # exclude 0 to avoid singularity

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = plt.cm.tab10(np.linspace(0, 0.9, len(alpha_values) * len(beta_values)))
    color_idx = 0

    for alpha in alpha_values:
        for beta in beta_values:
            B, rho = compute_powerlaw_params(
                Lambda_val=1.0, alpha_val=alpha, beta_val=beta, c_val=1.0
            )
            # Normalise B=1 to compare shapes
            lam = powerlaw_intensity_func(delta_grid, B=1.0, rho=rho)
            ax.loglog(
                delta_grid, lam,
                color=colors[color_idx],
                label=f'alpha={alpha}, beta={beta}  (rho={rho:.2f})',
                linewidth=1.4,
            )
            color_idx += 1

    ax.set_xlabel('delta (price distance)')
    ax.set_ylabel('lambda(delta)  [log scale]')
    ax.set_title('Power-Law Execution Intensity: Effect of alpha and beta')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_shape_comparison(
    alpha_val: float = 1.5,
    K_val: float = 1.0,
    beta_val: float = 0.5,
    delta_grid: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Compare normalized exponential vs power-law intensities on the same plot.
    Uses a 2x1 subplot: linear y-axis and log-linear y-axis.
    """
    if delta_grid is None:
        delta_grid = np.linspace(0.1, 5, 500)

    A, k = compute_exponential_params(Lambda_val=1.0, alpha_val=alpha_val, K_val=K_val)
    B, rho = compute_powerlaw_params(
        Lambda_val=1.0, alpha_val=alpha_val, beta_val=beta_val, c_val=1.0
    )

    lam_exp = exponential_intensity_func(delta_grid, A=A, k=k)
    lam_pow = powerlaw_intensity_func(delta_grid, B=B, rho=rho)

    norm_exp, norm_pow = normalize_at_reference(delta_grid, lam_exp, lam_pow, delta_ref=1.0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

    # Linear panel
    ax1.plot(delta_grid, norm_exp, label=f'Exponential (k={k:.2f})', color='steelblue', linewidth=1.8)
    ax1.plot(delta_grid, norm_pow, label=f'Power-law (rho={rho:.2f})', color='tomato', linewidth=1.8)
    ax1.set_xlabel('delta')
    ax1.set_ylabel('Normalized lambda(delta)')
    ax1.set_title('Shape Comparison: Exponential vs Power-Law Intensity')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Log-linear panel
    ax2.semilogy(delta_grid, norm_exp, label=f'Exponential (k={k:.2f})', color='steelblue', linewidth=1.8)
    ax2.semilogy(delta_grid, norm_pow, label=f'Power-law (rho={rho:.2f})', color='tomato', linewidth=1.8)
    ax2.set_xlabel('delta')
    ax2.set_ylabel('Normalized lambda(delta)  [log scale]')
    ax2.set_title('Shape Comparison: Log-Linear View')
    ax2.legend()
    ax2.grid(True, which='both', alpha=0.3)

    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_paper_intensity(
    A: float = 140.0,
    k: float = 1.5,
    delta_grid: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot the specific intensity function used in the paper's simulation:
    lambda(delta) = 140 * exp(-1.5 * delta).
    Annotates key values lambda(0) = A and lambda(1/k) = A/e.
    """
    if delta_grid is None:
        delta_grid = np.linspace(0.0, 5, 500)

    lam = exponential_intensity_func(delta_grid, A=A, k=k)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(delta_grid, lam, color='steelblue', linewidth=2.2,
            label=f'lambda(delta) = {A} * exp(-{k} * delta)')

    # Annotate lambda(0) = A
    ax.axhline(A, color='gray', linestyle=':', linewidth=1)
    ax.annotate(
        f'lambda(0) = A = {A}',
        xy=(0, A), xytext=(0.5, A * 1.03),
        fontsize=9, color='gray',
    )

    # Annotate lambda(1/k) = A/e
    delta_half = 1.0 / k
    lam_half = A / np.e
    ax.axvline(delta_half, color='tomato', linestyle='--', linewidth=1)
    ax.axhline(lam_half, color='tomato', linestyle='--', linewidth=1)
    ax.scatter([delta_half], [lam_half], color='tomato', zorder=5)
    ax.annotate(
        f'lambda(1/k) = A/e ≈ {lam_half:.1f}',
        xy=(delta_half, lam_half),
        xytext=(delta_half + 0.2, lam_half + 5),
        fontsize=9, color='tomato',
        arrowprops=dict(arrowstyle='->', color='tomato'),
    )

    ax.set_xlabel('delta (price distance from mid)')
    ax.set_ylabel('lambda(delta)  (executions per unit time)')
    ax.set_title(f'Paper Simulation Intensity: lambda(delta) = {A} * exp(-{k} * delta)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.close(fig)
