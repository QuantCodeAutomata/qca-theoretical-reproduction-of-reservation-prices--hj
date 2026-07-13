# AGENTS.md — Repository Memory

Persistent knowledge base for AI coding agents working in this repository.
Update this file whenever you discover new patterns, make structural decisions,
or identify important constraints.

---

## Repository Purpose

Reproduction and extension of the **Avellaneda–Stoikov (2008)
inventory-based market-making model** using Python.

The repository covers:
- Symbolic derivation and verification of HJB equations (sympy)
- Monte Carlo simulation of optimal market-making strategies
- Calibration of execution intensity from microstructure data

---

## Key Model Parameters

These values are the canonical defaults. Always load them from
`config/params.yaml` rather than hardcoding in source files.

| Parameter | Value | Notes |
|-----------|-------|-------|
| `S0` | 100 | Initial mid-price |
| `T` | 1.0 | Time horizon (normalised) |
| `sigma` | 2.0 | Mid-price volatility |
| `A` | 140.0 | Poisson arrival base rate |
| `k` | 1.5 | Fill-probability decay constant |
| `q0` | 0 | Initial inventory |
| `x0` | 0.0 | Initial cash |
| `dt` | 0.005 | Simulation time step |
| `N_steps` | 200 | Steps per path (= T / dt) |
| `n_paths` | 1000 | Monte Carlo paths |
| `gammas` | [0.1, 0.01, 0.5] | Risk-aversion values for simulation sweeps |
| `seed` | 42 | Global random seed |

---

## Key Formulas

**Reservation price** — inventory-adjusted indifference mid-price:
```
r(s, q, t) = s - q * gamma * sigma^2 * (T - t)
```

**Optimal bid–ask spread**:
```
delta* = gamma * sigma^2 * (T - t) + (2 / gamma) * ln(1 + gamma / k)
```

**Execution intensity** — Poisson arrival rate as a function of quote distance δ:
```
lambda(delta) = A * exp(-k * delta)
```

**Optimal bid and ask quotes**:
```
s_bid = r - delta* / 2
s_ask = r + delta* / 2
```

---

## Code Structure

```
src/            Core shared logic (utilities, config loading, common types)
theory/         Symbolic derivations using sympy (Exp1)
sim/            Monte Carlo simulation engines (Exp2, Exp3)
microstructure/ Execution intensity models and LOB calibration (Exp4)
analysis/       Post-processing, plotting, and reporting helpers
exp/            Top-level experiment runner scripts (run_exp1.py … run_exp4.py)
tests/          Pytest test suite — mirrors src/theory/sim/microstructure structure
config/         YAML configuration (params.yaml)
data/           Raw or cached input data (not committed unless small)
results/        All generated outputs: CSV tables, PNG figures, RESULTS.md
```

### Module Conventions
- All experiment runners in `exp/` load parameters exclusively from
  `config/params.yaml` via a shared loader in `src/`.
- Figures are saved as `results/exp{N}_<description>.png`.
- Tables are saved as `results/exp{N}_<description>.csv`.
- A summary of all results is maintained in `results/RESULTS.md`.

---

## Testing

Run the full test suite from the repository root:

```bash
pytest tests/
```

- Unit tests live in `tests/` and mirror the module structure under `src/`,
  `theory/`, `sim/`, and `microstructure/`.
- Use `pytest -v` for verbose output.
- Do not import from `exp/` runners in tests; test the underlying modules directly.

---

## Results

All generated outputs are written to `results/`:
- `results/exp1_reservation_prices.csv` — Exp1 reservation price table
- `results/exp2_table.csv` — Exp2 Monte Carlo summary statistics
- `results/exp3_comparison.csv` — Exp3 strategy comparison
- `results/exp4_intensity.csv` — Exp4 intensity calibration
- `results/RESULTS.md` — Human-readable summary of all experiment outputs

The `results/` directory is tracked by git (via `.gitkeep`) but generated
files are not — add them to `.gitignore` if they become large.

---

## Dependencies

```
sympy        # Symbolic mathematics (Exp1)
numpy        # Numerical arrays and random number generation
pandas       # DataFrames for result tables
matplotlib   # Base plotting
seaborn      # Statistical visualisation
scipy        # Optimisation and statistical functions
pytest       # Testing framework
pyyaml       # YAML config loading
```

Install all with:
```bash
pip install -r requirements.txt
```

---

## Important Constraints and Decisions

- **No hardcoded parameters**: all model constants come from `config/params.yaml`.
- **Reproducibility**: all stochastic experiments use the global `seed` from config.
- **Sympy purity**: Exp1 must perform all derivations symbolically — no numerical
  shortcuts inside `theory/` modules.
- **No circular imports**: `src/` must not import from `theory/`, `sim/`, or
  `microstructure/`; those packages import from `src/` only.
- **Python ≥ 3.9** required.

---

## Experiment 1 Implementation Notes

**Files implemented:**
- `theory/derivations.py` — Sympy symbolic formulas + 12 functions (frozen value fn, reservation prices, quote distances, spread, verifications)
- `theory/checks.py` — Numerical verification functions + grids; module-level constants `S0=100, T_val=1, SIGMA=2, A_val=140, K_val=1.5`
- `exp/run_exp1.py` — Full experiment runner; produces 3 PNGs + `exp1_results.txt`
- `tests/test_exp1.py` — 14 pytest tests (all pass)

**Key symbolic verification technique:**
Indifference conditions are verified by comparing **exponents** of the frozen value function (both sides are `-exp(...)`). Use `sp.expand(exp_lhs - exp_rhs) == 0` rather than `sp.simplify(lhs - rhs)` on full exponentials — the latter can fail to reach zero in sympy.

**Spread values at k=1.5:**
- γ=0.01 → 1.3289, γ=0.1 → 1.2908, γ=0.5 → 1.1507

**Quote distance signs:**
- Large positive q makes δᵃ negative (widened ask beyond market); flagged with `(*)` in grid output
- Large negative q makes δᵇ negative; same flag

**Sympy symbol conventions in derivations.py:**
- Module-level: `x, s, q, t, T, sigma, gamma, A, k, omega` (q redefined as `real=True` without positive; t as `nonneg=True`)
- `verify_spread_decomposition` uses `sp.simplify`; indifference verifications use `sp.expand` on exponents

---

## Experiment 4 Implementation Notes

**Files implemented:**
- `microstructure/intensity_derivation.py` — Sympy symbolic derivations + numerical helpers
- `microstructure/intensity_plots.py` — 4 plotting functions (semilogy, loglog, comparison, paper)
- `exp/run_exp4.py` — Full experiment runner; produces 4 PNGs + `exp4_results.txt`
- `tests/test_exp4.py` — 10 pytest tests (all pass)

**Derivation summary:**
- Exponential: `lambda(delta) = A*exp(-k*delta)` from power-law sizes + log impact; `k = alpha*K`
- Power-law: `lambda(delta) = B*delta^{-rho}` from power-law sizes + power-law impact; `rho = alpha/beta`
- Paper uses A=140, k=1.5 directly (functional form from derivation, values not estimated)

**Implementation notes:**
- `matplotlib.use('Agg')` must appear before any `pyplot` import (headless env)
- `powerlaw_intensity_func` raises `ValueError` for any `delta <= 0`
- Sympy symbols declared at module level in `intensity_derivation.py`
- `exp/run_exp4.py` uses `sys.path.insert(0, project_root)` for local imports
- Results are saved to `results/exp4_*.{png,txt}`
