# Inventory-Based Market-Making: Avellaneda–Stoikov Model Reproduction

A quantitative finance research repository that reproduces and extends the
Avellaneda–Stoikov (2008) inventory-based market-making model through four
structured experiments covering symbolic derivation, Monte Carlo simulation,
and microstructure intensity modelling.

---

## Overview

The Avellaneda–Stoikov framework models a market maker who continuously posts
limit orders on both sides of the order book. The market maker faces two
competing objectives: capturing the bid–ask spread and controlling inventory
risk. The model produces closed-form expressions for the **reservation price**
(the indifference mid-price adjusted for inventory) and the **optimal spread**,
derived from a Hamilton–Jacobi–Bellman (HJB) partial differential equation.

This repository reproduces the theoretical results, validates them numerically
via Monte Carlo methods, and extends the analysis to microstructure-level
execution intensity modelling.

---

## Experiments

### Experiment 1 — Theoretical Reproduction of Reservation Prices and HJB Conditions

**Runner:** `python exp/run_exp1.py`

Uses `sympy` to derive the reservation price and optimal spread symbolically
from first principles. Verifies that the closed-form ansatz satisfies the HJB
PDE exactly, computes partial derivatives, and tabulates reservation prices
across a grid of inventory levels `q ∈ {−5, …, 5}` and time points
`t ∈ {0.0, 0.25, 0.5, 0.75, 1.0}` for risk-aversion coefficients
`γ ∈ {0.01, 0.1, 0.5}`.

Key outputs:
- Symbolic verification of HJB residual (should be identically zero)
- Reservation price table (q × t grid) for each γ
- Optimal spread as a function of time-to-maturity

**Module path:** `theory/`

---

### Experiment 2 — Monte Carlo Table-Faithful Market-Making Simulation

**Runner:** `python exp/run_exp2.py`

Reproduces the numerical tables from the original Avellaneda–Stoikov paper
using Monte Carlo simulation. The simulation faithfully follows the paper's
parameter choices (`S0=100`, `T=1`, `σ=2`, `A=140`, `k=1.5`) and execution
mechanics (symmetric Poisson arrivals, fill probability as a function of
distance to mid-price).

The experiment sweeps over `γ ∈ {0.1, 0.01, 0.5}` and reports:
- Mean terminal P&L and standard deviation
- Mean terminal inventory
- Fill rate (fraction of posted quotes that execute)
- Sharpe ratio of terminal wealth

Results are saved to `results/exp2_table.csv` and figures to `results/`.

**Module path:** `sim/`

---

### Experiment 3 — Monte Carlo Theory-Faithful Market-Making Simulation

**Runner:** `python exp/run_exp3.py`

A theory-grounded Monte Carlo study that directly implements the optimal
control policy derived from the HJB solution. Rather than replicating paper
tables, this experiment validates that the theoretical optimal quotes
(computed from the closed-form reservation price and spread) outperform
heuristic alternatives across market conditions.

Compares three strategies:
1. **Optimal HJB** — quotes derived from the exact analytical solution
2. **Fixed spread** — symmetric quotes at a constant spread
3. **Naïve mid** — always quotes at the current mid-price

Reports distribution of terminal wealth, inventory paths, and quote distances
for each strategy. Results saved to `results/exp3_comparison.csv`.

**Module path:** `sim/`

---

### Experiment 4 — Execution Intensity Derivation from Microstructure

**Runner:** `python exp/run_exp4.py`

Derives and calibrates the execution intensity function `λ(δ) = A · exp(−k·δ)`
from microstructure principles. The experiment:
1. Derives the intensity functional form from order-book depth assumptions
2. Fits `A` and `k` to synthetic order-flow data generated from a realistic
   LOB (limit order book) model
3. Validates the fitted intensity against held-out simulation paths
4. Analyses the sensitivity of optimal spread to `k` and `A` perturbations

Outputs include calibration curves, residual plots, and a sensitivity table
saved to `results/exp4_intensity.csv`.

**Module path:** `microstructure/`

---

## Project Structure

```
.
├── README.md
├── AGENTS.md                    # Persistent memory for AI agents
├── requirements.txt
├── .gitignore
│
├── config/
│   └── params.yaml              # Centralised model and simulation parameters
│
├── src/                         # Shared core logic and utilities
│   └── __init__.py
│
├── theory/                      # Symbolic derivations (sympy)
│   └── __init__.py
│
├── sim/                         # Monte Carlo simulation engines
│   └── __init__.py
│
├── microstructure/              # Execution intensity and LOB models
│   └── __init__.py
│
├── analysis/                    # Post-processing and plotting utilities
│   └── __init__.py
│
├── exp/                         # Experiment runner scripts
│   ├── run_exp1.py
│   ├── run_exp2.py
│   ├── run_exp3.py
│   └── run_exp4.py
│
├── tests/                       # Pytest test suite
│   └── __init__.py
│
├── data/                        # Raw or cached input data
│   └── .gitkeep
│
└── results/                     # Generated outputs (CSV, PNG, RESULTS.md)
    └── .gitkeep
```

---

## Installation

Requires Python 3.9 or later. Install all dependencies with:

```bash
pip install sympy numpy pandas matplotlib seaborn scipy
```

To also install development and testing dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

Run each experiment independently from the repository root:

```bash
python exp/run_exp1.py   # Symbolic HJB derivation and reservation price tables
python exp/run_exp2.py   # Table-faithful Monte Carlo reproduction
python exp/run_exp3.py   # Theory-faithful optimal policy validation
python exp/run_exp4.py   # Execution intensity calibration
```

Run the test suite:

```bash
pytest tests/
```

All results (tables, figures, summary) are written to `results/`.

---

## Key Model Parameters

| Parameter | Symbol | Value | Description |
|-----------|--------|-------|-------------|
| Initial mid-price | S₀ | 100 | Starting asset price |
| Time horizon | T | 1.0 | Trading session length (normalised) |
| Volatility | σ | 2.0 | Mid-price diffusion coefficient |
| Arrival rate | A | 140.0 | Base Poisson intensity of market orders |
| Decay parameter | k | 1.5 | Exponential decay of fill probability with distance |
| Initial inventory | q₀ | 0 | Starting inventory |
| Initial cash | x₀ | 0.0 | Starting cash (normalised) |

---

## Key Formulas

**Reservation price** (inventory-adjusted indifference price):

```
r(s, q, t) = s − q · γ · σ² · (T − t)
```

**Optimal bid–ask spread**:

```
δ* = γ · σ² · (T − t) + (2/γ) · ln(1 + γ/k)
```

**Execution intensity** (Poisson arrival rate as function of quote distance δ):

```
λ(δ) = A · exp(−k · δ)
```

where γ is the market maker's risk-aversion coefficient.

---

## References

- Avellaneda, M. & Stoikov, S. (2008). *High-frequency trading in a limit order
  book.* Quantitative Finance, 8(3), 217–224.
- Cartea, Á., Jaimungal, S. & Penalva, J. (2015). *Algorithmic and
  High-Frequency Trading.* Cambridge University Press.
