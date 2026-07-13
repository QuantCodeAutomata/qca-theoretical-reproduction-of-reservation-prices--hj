import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sim.common import table_faithful_spread, reservation_price_num, compute_intensity

S0 = 100.0
T = 1.0
SIGMA = 2.0
A = 140.0
K = 1.5
DT = 0.005
N_STEPS = 200
N_PATHS = 1000


def simulate_single_path_inventory_aware(
    gamma: float,
    seed: Optional[int] = None,
    S0: float = 100.0,
    T: float = 1.0,
    sigma: float = 2.0,
    A: float = 140.0,
    k: float = 1.5,
    dt: float = 0.005,
    N: int = 200,
) -> Tuple[float, float]:
    """Simulate one path of the inventory-aware strategy (table-faithful track).

    Uses a constant spread derived from the paper table, centred around the
    reservation price, with capped Bernoulli fill probabilities.
    """
    # Custom — Context7 found no library equivalent (custom microstructure simulation)
    rng = np.random.default_rng(seed)

    S = S0
    q = 0.0
    X = 0.0
    n_capped = 0

    psi = table_faithful_spread(gamma, k)

    for n in range(N):
        t_n = n * dt

        # Reservation price
        r = reservation_price_num(S, q, t_n, T, sigma, gamma)

        # Quote prices centred on reservation price
        p_a = r + psi / 2
        p_b = r - psi / 2

        # Distances from mid
        delta_a = p_a - S
        delta_b = S - p_b

        # Arrival intensities
        lambda_a = compute_intensity(delta_a, A, k)
        lambda_b = compute_intensity(delta_b, A, k)

        # Capped Bernoulli fill probabilities
        prob_a = min(lambda_a * dt, 1.0)
        prob_b = min(lambda_b * dt, 1.0)
        if lambda_a * dt >= 1.0 or lambda_b * dt >= 1.0:
            n_capped += 1

        fill_a = int(rng.random() < prob_a)
        fill_b = int(rng.random() < prob_b)

        if fill_a:
            X += p_a
            q -= 1
        if fill_b:
            X -= p_b
            q += 1

        # Mid-price random walk
        epsilon = 1.0 if rng.random() < 0.5 else -1.0
        S += epsilon * sigma * np.sqrt(dt)

    terminal_profit = X + q * S
    return terminal_profit, q


def simulate_single_path_symmetric(
    gamma: float,
    seed: Optional[int] = None,
    S0: float = 100.0,
    T: float = 1.0,
    sigma: float = 2.0,
    A: float = 140.0,
    k: float = 1.5,
    dt: float = 0.005,
    N: int = 200,
) -> Tuple[float, float]:
    """Simulate one path of the symmetric strategy (table-faithful track).

    Quotes are placed symmetrically around the current mid-price using the
    constant paper-table spread, with capped Bernoulli fill probabilities.
    """
    rng = np.random.default_rng(seed)

    S = S0
    q = 0.0
    X = 0.0

    psi = table_faithful_spread(gamma, k)

    for n in range(N):
        # Symmetric quotes around mid
        p_a = S + psi / 2
        p_b = S - psi / 2

        delta_a = psi / 2
        delta_b = psi / 2

        lambda_a = compute_intensity(delta_a, A, k)
        lambda_b = compute_intensity(delta_b, A, k)

        prob_a = min(lambda_a * dt, 1.0)
        prob_b = min(lambda_b * dt, 1.0)

        fill_a = int(rng.random() < prob_a)
        fill_b = int(rng.random() < prob_b)

        if fill_a:
            X += p_a
            q -= 1
        if fill_b:
            X -= p_b
            q += 1

        epsilon = 1.0 if rng.random() < 0.5 else -1.0
        S += epsilon * sigma * np.sqrt(dt)

    return X + q * S, q


def run_monte_carlo_table_faithful(
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
    """Run n_paths Monte Carlo paths for both strategies at a given gamma.

    Returns a DataFrame with columns: path_id, gamma, strategy, profit, final_q.
    """
    records: List[Dict] = []

    for path_id in range(n_paths):
        path_seed_inv = seed * 100_000 + path_id
        path_seed_sym = seed * 100_000 + path_id

        profit_inv, q_inv = simulate_single_path_inventory_aware(
            gamma, seed=path_seed_inv, S0=S0, T=T, sigma=sigma,
            A=A, k=k, dt=dt, N=N,
        )
        profit_sym, q_sym = simulate_single_path_symmetric(
            gamma, seed=path_seed_sym, S0=S0, T=T, sigma=sigma,
            A=A, k=k, dt=dt, N=N,
        )

        records.append({"path_id": path_id, "gamma": gamma, "strategy": "inventory_aware",
                         "profit": profit_inv, "final_q": q_inv})
        records.append({"path_id": path_id, "gamma": gamma, "strategy": "symmetric",
                         "profit": profit_sym, "final_q": q_sym})

        if (path_id + 1) % 100 == 0:
            print(f"  gamma={gamma}: completed {path_id + 1}/{n_paths} paths")

    return pd.DataFrame(records)


def run_all_gammas_table_faithful(
    gammas: List[float] = [0.1, 0.01, 0.5],
    n_paths: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Run table-faithful Monte Carlo for all specified gamma values.

    Returns concatenated DataFrame of all results.
    """
    frames: List[pd.DataFrame] = []
    for gamma in gammas:
        print(f"\nRunning table-faithful simulation: gamma={gamma}")
        df = run_monte_carlo_table_faithful(gamma, n_paths=n_paths, seed=seed)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def generate_sample_path_data(
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
    """Run one detailed inventory-aware path and record state at every step.

    Returns a dict with arrays: S_t, r_t, p_a_t, p_b_t, q_t, X_t, t_n.
    Used for sample path plotting.
    """
    rng = np.random.default_rng(seed)

    S = S0
    q = 0.0
    X = 0.0
    psi = table_faithful_spread(gamma, k)

    S_arr, r_arr, p_a_arr, p_b_arr, q_arr, X_arr, t_arr = [], [], [], [], [], [], []

    for n in range(N):
        t_n = n * dt
        r = reservation_price_num(S, q, t_n, T, sigma, gamma)
        p_a = r + psi / 2
        p_b = r - psi / 2

        S_arr.append(S)
        r_arr.append(r)
        p_a_arr.append(p_a)
        p_b_arr.append(p_b)
        q_arr.append(q)
        X_arr.append(X)
        t_arr.append(t_n)

        delta_a = p_a - S
        delta_b = S - p_b
        lambda_a = compute_intensity(delta_a, A, k)
        lambda_b = compute_intensity(delta_b, A, k)
        prob_a = min(lambda_a * dt, 1.0)
        prob_b = min(lambda_b * dt, 1.0)

        if rng.random() < prob_a:
            X += p_a
            q -= 1
        if rng.random() < prob_b:
            X -= p_b
            q += 1

        epsilon = 1.0 if rng.random() < 0.5 else -1.0
        S += epsilon * sigma * np.sqrt(dt)

    return {
        "S_t": np.array(S_arr),
        "r_t": np.array(r_arr),
        "p_a_t": np.array(p_a_arr),
        "p_b_t": np.array(p_b_arr),
        "q_t": np.array(q_arr),
        "X_t": np.array(X_arr),
        "t_n": np.array(t_arr),
    }
