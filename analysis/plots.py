import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import os


def plot_profit_histograms(
    df: pd.DataFrame,
    gamma: float,
    track_name: str = "table_faithful",
    save_path: Optional[str] = None,
) -> None:
    """Plot overlaid terminal profit histograms for both strategies at a given gamma."""
    fig, ax = plt.subplots(figsize=(8, 5))

    inv_profits = df[(df["gamma"] == gamma) & (df["strategy"] == "inventory_aware")]["profit"]
    sym_profits = df[(df["gamma"] == gamma) & (df["strategy"] == "symmetric")]["profit"]

    ax.hist(inv_profits, bins=40, alpha=0.6, color="steelblue", label="Inventory-Aware")
    ax.hist(sym_profits, bins=40, alpha=0.6, color="tomato", label="Symmetric")

    ax.set_title(f"Terminal Profit Distribution - gamma={gamma} ({track_name})")
    ax.set_xlabel("Terminal Profit")
    ax.set_ylabel("Frequency")
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_sample_path(
    path_data: Dict,
    gamma: float,
    track_name: str = "table_faithful",
    save_path: Optional[str] = None,
) -> None:
    """Plot mid-price, reservation price, bid and ask quotes with inventory on a secondary axis."""
    t = path_data["t_n"]

    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.plot(t, path_data["S_t"], color="black", lw=1.5, label="S (mid)")
    ax1.plot(t, path_data["r_t"], color="blue", lw=1.2, ls="--", label="r (reservation)")
    ax1.plot(t, path_data["p_b_t"], color="green", lw=1.0, label="bid")
    ax1.plot(t, path_data["p_a_t"], color="red", lw=1.0, label="ask")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Price")
    ax1.set_title(f"Sample Path - gamma={gamma} ({track_name})")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.step(t, path_data["q_t"], color="purple", lw=0.8, alpha=0.6, label="inventory q")
    ax2.set_ylabel("Inventory q", color="purple")
    ax2.tick_params(axis="y", labelcolor="purple")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_all_profit_histograms(
    df: pd.DataFrame,
    gammas: List[float],
    track_name: str,
    results_dir: str = "results",
) -> None:
    """Create one profit histogram per gamma and save to results directory."""
    os.makedirs(results_dir, exist_ok=True)
    for gamma in gammas:
        filename = f"{track_name}_profit_hist_gamma_{gamma}.png"
        save_path = os.path.join(results_dir, filename)
        plot_profit_histograms(df, gamma, track_name=track_name, save_path=save_path)
        print(f"  Saved: {save_path}")


def plot_inventory_distribution(
    df: pd.DataFrame,
    gamma: float,
    track_name: str = "table_faithful",
    save_path: Optional[str] = None,
) -> None:
    """Plot final inventory distribution for both strategies at a given gamma."""
    fig, ax = plt.subplots(figsize=(8, 5))

    inv_q = df[(df["gamma"] == gamma) & (df["strategy"] == "inventory_aware")]["final_q"]
    sym_q = df[(df["gamma"] == gamma) & (df["strategy"] == "symmetric")]["final_q"]

    bins = np.arange(
        min(inv_q.min(), sym_q.min()) - 0.5,
        max(inv_q.max(), sym_q.max()) + 1.5,
        1.0,
    )
    ax.hist(inv_q, bins=bins, alpha=0.6, color="steelblue", label="Inventory-Aware")
    ax.hist(sym_q, bins=bins, alpha=0.6, color="tomato", label="Symmetric")

    ax.set_title(f"Final Inventory Distribution - gamma={gamma}")
    ax.set_xlabel("Final Inventory q")
    ax.set_ylabel("Frequency")
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
