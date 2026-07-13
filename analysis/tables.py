import numpy as np
import pandas as pd
from typing import Dict, List


def compute_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute summary statistics matching paper table format.

    Groups by (gamma, strategy) and computes mean/std of profit and final_q.
    """
    summary = (
        df.groupby(["gamma", "strategy"])
        .agg(
            mean_profit=("profit", "mean"),
            std_profit=("profit", lambda x: x.std(ddof=1)),
            mean_final_q=("final_q", "mean"),
            std_final_q=("final_q", lambda x: x.std(ddof=1)),
        )
        .reset_index()
    )
    return summary


def format_summary_table(summary_df: pd.DataFrame) -> str:
    """Format the summary DataFrame as a printable aligned string table."""
    header = (
        f"{'gamma':>8}  {'strategy':>16}  {'spread':>8}  "
        f"{'mean_profit':>12}  {'std_profit':>11}  "
        f"{'mean_final_q':>13}  {'std_final_q':>11}"
    )
    sep = "-" * len(header)
    lines = [sep, header, sep]

    spread_col = "spread" if "spread" in summary_df.columns else None

    for _, row in summary_df.iterrows():
        spread_val = f"{row['spread']:>8.4f}" if spread_col else f"{'N/A':>8}"
        lines.append(
            f"{row['gamma']:>8.3f}  {row['strategy']:>16}  {spread_val}  "
            f"{row['mean_profit']:>12.4f}  {row['std_profit']:>11.4f}  "
            f"{row['mean_final_q']:>13.4f}  {row['std_final_q']:>11.4f}"
        )

    lines.append(sep)
    return "\n".join(lines)


def add_spread_column(
    summary_df: pd.DataFrame,
    track: str = "table_faithful",
    k: float = 1.5,
    sigma: float = 2.0,
    T: float = 1.0,
) -> pd.DataFrame:
    """Add a spread column appropriate to the simulation track.

    For 'table_faithful': spread = (2/gamma)*ln(1+gamma/k)
    For 'theory_faithful': spread = gamma*sigma^2*T + (2/gamma)*ln(1+gamma/k) at t=0
    """
    df = summary_df.copy()

    def _spread(gamma: float) -> float:
        base = (2 / gamma) * np.log(1 + gamma / k)
        if track == "theory_faithful":
            return gamma * sigma**2 * T + base
        return base

    df["spread"] = df["gamma"].apply(_spread)
    return df


def compare_with_paper_targets(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Compare simulation results against paper-reported targets for gamma=0.1.

    Paper targets (gamma=0.1):
      inventory_aware: mean_profit~62.94, std_profit~5.89, mean_q~0.10, std_q~2.80
      symmetric:       mean_profit~67.21, std_profit~13.43, mean_q~-0.018, std_q~8.66

    Returns a DataFrame with absolute and percentage deviations.
    """
    targets = {
        ("inventory_aware", 0.1): {
            "mean_profit": 62.94,
            "std_profit": 5.89,
            "mean_final_q": 0.10,
            "std_final_q": 2.80,
        },
        ("symmetric", 0.1): {
            "mean_profit": 67.21,
            "std_profit": 13.43,
            "mean_final_q": -0.018,
            "std_final_q": 8.66,
        },
    }

    rows: List[Dict] = []
    for (strategy, gamma), target in targets.items():
        mask = (summary_df["gamma"] == gamma) & (summary_df["strategy"] == strategy)
        subset = summary_df[mask]
        if subset.empty:
            continue
        row = subset.iloc[0]
        for metric, paper_val in target.items():
            sim_val = row[metric]
            abs_dev = sim_val - paper_val
            pct_dev = (abs_dev / abs(paper_val) * 100) if paper_val != 0 else float("nan")
            rows.append({
                "gamma": gamma,
                "strategy": strategy,
                "metric": metric,
                "paper_value": paper_val,
                "sim_value": sim_val,
                "abs_deviation": abs_dev,
                "pct_deviation": pct_dev,
            })

    return pd.DataFrame(rows)
