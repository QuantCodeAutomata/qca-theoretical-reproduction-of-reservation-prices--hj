import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sim.common import (
    ask_distance_theory,
    bid_distance_theory,
    full_spread,
    reservation_price_num,
    compute_intensity,
)


def simulate_single_path_inventory_aware_theory(
    gamma: float,
    seed: Optional[int] = None,
    S0: float = 100.0,
    T: float = 1.0,
    sigma: float = 2.0,
    A: float = 140.0,
    k: float = 1.5,
    dt: float = 0.005,
    N: int = 200,
) -> Tuple[float, float, Dict]:
    """Simulate one path using the exact paper equations (theory-faithful track).

    Ask and bid distances are computed from the full Avellaneda-Stoikov formula,
    and fill counts are drawn from Poisson distributions.
    """
    # Custom — Context7 found no library equivalent (custom Poisson event simulation)
    rng = np.random.default_rng(seed)

    S = S0
    q = 0.0
    X = 0.0

    n_negative_delta_a = 0
    n_negative_delta_b = 0
    max_intensity = 0.0
    total_fills = 0.0

    for n in range(N):
        t_n = n * dt

        delta_a = ask_distance_theory(q, t_n, T, sigma, gamma, k)
        delta_b = bid_distance_theory(q, t_n, T, sigma, gamma, k)

        if delta_a < 0:
            n_negative_delta_a += 1
        if delta_b < 0:
            n_negative_delta_b += 1

        p_a = S + delta_a
        p_b = S - delta_b

        # Clip at 0 so intensity stays bounded when quotes cross mid-price
        lambda_a = compute_intensity(max(delta_a, 0.0), A, k)
        lambda_b = compute_intensity(max(delta_b, 0.0), A, k)

        max_intensity = max(max_intensity, lambda_a, lambda_b)

        N_a = int(rng.poisson(lambda_a * dt))
        N_b = int(rng.poisson(lambda_b * dt))

        total_fills += N_a + N_b

        X += N_a * p_a - N_b * p_b
        q += -N_a + N_b

        epsilon = 1.0 if rng.random() < 0.5 else -1.0
        S += epsilon * sigma * np.sqrt(dt)

    avg_fills_per_step = total_fills / N
    diagnostics = {
        "n_negative_delta_a": n_negative_delta_a,
        "n_negative_delta_b": n_negative_delta_b,
        "max_intensity": max_intensity,
        "avg_fills_per_step": avg_fills_per_step,
    }
    return X + q * S, q, diagnostics


def simulate_single_path_symmetric_theory(
    gamma: float,
    seed: Optional[int] = None,
    S0: float = 100.0,
    T: float = 1.0,
    sigma: float = 2.0,
    A: float = 140.0,
    k: float = 1.5,
    dt: float = 0.005,
    N: int = 200,
) -> Tuple[float, float, Dict]:
    """Simulate one path of the symmetric benchmark (theory-faithful track).

    Uses the full time-varying spread from the paper, with Poisson fill counts.
    """
    rng = np.random.default_rng(seed)

    S = S0
    q = 0.0
    X = 0.0

    max_intensity = 0.0
    total_fills = 0.0

    for n in range(N):
        t_n = n * dt
        psi = full_spread(t_n, T, sigma, gamma, k)
        half = psi / 2

        p_a = S + half
        p_b = S - half

        lambda_a = compute_intensity(half, A, k)
        lambda_b = compute_intensity(half, A, k)

        max_intensity = max(max_intensity, lambda_a, lambda_b)

        N_a = int(rng.poisson(lambda_a * dt))
        N_b = int(rng.poisson(lambda_b * dt))

        total_fills += N_a + N_b

        X += N_a * p_a - N_b * p_b
        q += -N_a + N_b

        epsilon = 1.0 if rng.random() < 0.5 else -1.0
        S += epsilon * sigma * np.sqrt(dt)

    avg_fills_per_step = total_fills / N
    diagnostics = {
        "n_negative_delta_a": 0,
        "n_negative_delta_b": 0,
        "max_intensity": max_intensity,
        "avg_fills_per_step": avg_fills_per_step,
    }
    return X + q * S, q, diagnostics


def run_monte_carlo_theory_faithful(
    gamma: float,
    n_paths: int = 1000,
    seed: int = 42,
    S0: float = 100.0,
    T: float = 1.0,
    sigma: float = 2.0,
    A: float = 140.0,
    k: float = 1.5,
    dt: float = 0.005,
    N: int = 200,
) -> pd.DataFrame:
    """Run n_paths Monte Carlo paths for both theory-faithful strategies at a given gamma.

    Returns a DataFrame with columns: path_id, gamma, strategy, profit, final_q.
    """
    records: List[Dict] = []

    for path_id in range(n_paths):
        path_seed = seed * 100_000 + path_id

        profit_inv, q_inv, _ = simulate_single_path_inventory_aware_theory(
            gamma, seed=path_seed, S0=S0, T=T, sigma=sigma,
            A=A, k=k, dt=dt, N=N,
        )
        profit_sym, q_sym, _ = simulate_single_path_symmetric_theory(
            gamma, seed=path_seed, S0=S0, T=T, sigma=sigma,
            A=A, k=k, dt=dt, N=N,
        )

        records.append({"path_id": path_id, "gamma": gamma, "strategy": "inventory_aware",
                         "profit": profit_inv, "final_q": q_inv})
        records.append({"path_id": path_id, "gamma": gamma, "strategy": "symmetric",
                         "profit": profit_sym, "final_q": q_sym})

        if (path_id + 1) % 100 == 0:
            print(f"  gamma={gamma}: completed {path_id + 1}/{n_paths} paths")

    return pd.DataFrame(records)


def run_all_gammas_theory_faithful(
    gammas: List[float] = [0.1, 0.01, 0.5],
    n_paths: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Run theory-faithful Monte Carlo for all specified gamma values.

    Returns concatenated DataFrame of all results.
    """
    frames: List[pd.DataFrame] = []
    for gamma in gammas:
        print(f"\nRunning theory-faithful simulation: gamma={gamma}")
        df = run_monte_carlo_theory_faithful(gamma, n_paths=n_paths, seed=seed)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def generate_sample_path_data_theory(
    gamma: float = 0.1,
    seed: int = 0,
    S0: float = 100.0,
    T: float = 1.0,
    sigma: float = 2.0,
    A: float = 140.0,
    k: float = 1.5,
    dt: float = 0.005,
    N: int = 200,
) -> Dict:
    """Run one detailed theory-faithful inventory-aware path, recording state at every step.

    Returns a dict with arrays: S_t, r_t, p_a_t, p_b_t, delta_a_t, delta_b_t, q_t, X_t, t_n.
    """
    rng = np.random.default_rng(seed)

    S = S0
    q = 0.0
    X = 0.0

    S_arr, r_arr, p_a_arr, p_b_arr = [], [], [], []
    da_arr, db_arr, q_arr, X_arr, t_arr = [], [], [], [], []

    for n in range(N):
        t_n = n * dt
        r = reservation_price_num(S, q, t_n, T, sigma, gamma)
        delta_a = ask_distance_theory(q, t_n, T, sigma, gamma, k)
        delta_b = bid_distance_theory(q, t_n, T, sigma, gamma, k)
        p_a = S + delta_a
        p_b = S - delta_b

        S_arr.append(S)
        r_arr.append(r)
        p_a_arr.append(p_a)
        p_b_arr.append(p_b)
        da_arr.append(delta_a)
        db_arr.append(delta_b)
        q_arr.append(q)
        X_arr.append(X)
        t_arr.append(t_n)

        # Clip at 0 so intensity stays bounded when quotes cross mid-price
        lambda_a = compute_intensity(max(delta_a, 0.0), A, k)
        lambda_b = compute_intensity(max(delta_b, 0.0), A, k)

        N_a = int(rng.poisson(lambda_a * dt))
        N_b = int(rng.poisson(lambda_b * dt))

        X += N_a * p_a - N_b * p_b
        q += -N_a + N_b

        epsilon = 1.0 if rng.random() < 0.5 else -1.0
        S += epsilon * sigma * np.sqrt(dt)

    return {
        "S_t": np.array(S_arr),
        "r_t": np.array(r_arr),
        "p_a_t": np.array(p_a_arr),
        "p_b_t": np.array(p_b_arr),
        "delta_a_t": np.array(da_arr),
        "delta_b_t": np.array(db_arr),
        "q_t": np.array(q_arr),
        "X_t": np.array(X_arr),
        "t_n": np.array(t_arr),
    }
