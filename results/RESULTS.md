# Experiment Results: Inventory-Based Market Making (Avellaneda-Stoikov Model)

> All experiments ran with 38/38 tests passing. Results generated on 2026-07-13.

---

## Experiment 1: Theoretical Derivations

**Objective:** Reproduce the paper's theoretical formulas for reservation prices, HJB optimality conditions, and approximate closed-form quotes symbolically and numerically.

### Symbolic Verifications

| Check | Result |
|-------|--------|
| Indifference ask: v(x+r^a, s, q-1, t) = v(x, s, q, t) | **PASS** (exponent diff = 0) |
| Indifference bid: v(x-r^b, s, q+1, t) = v(x, s, q, t) | **PASS** (exponent diff = 0) |
| Spread decomposition: δ^a + δ^b = full spread | **PASS** (diff = 0) |

### Numerical Checks

| Check | Result |
|-------|--------|
| r(s, 0, t) = s (zero inventory) | **PASS** |
| r decreases in q (monotonicity) | **PASS** |
| r → s as t → T (maturity convergence) | **PASS** |
| γ → 0 limit: δ^a ≈ δ^b ≈ 1/k | **PASS** (impact=0.66664, 1/k=0.66667) |

### Spread Values vs Paper Tables

| γ | Computed Spread (2/γ)·ln(1+γ/k) | Paper Reported | Difference |
|---|----------------------------------|----------------|------------|
| 0.01 | 1.3289 | 1.3300 | -0.0011 |
| 0.10 | 1.2908 | 1.2900 | +0.0008 |
| 0.50 | 1.1507 | 1.1500 | +0.0007 |

> **Finding:** The table-faithful spread formula (2/γ)·ln(1+γ/k) reproduces the paper's reported spread values with < 0.1% error.

### Key Formulas Verified

- **Frozen-inventory value function:**  
  `v(x,s,q,t) = -exp(-γx) · exp(-γqs) · exp(γ²q²σ²(T-t)/2)`

- **Reservation ask price:**  
  `r^a(s,q,t) = s + ((1-2q)·γσ²(T-t))/2`

- **Reservation bid price:**  
  `r^b(s,q,t) = s + ((-1-2q)·γσ²(T-t))/2`

- **Reservation price:**  
  `r(s,q,t) = s - q·γσ²(T-t)`

- **Approximate optimal ask distance (exp. intensity):**  
  `δ^{a,*} = ((1-2q)·γσ²(T-t))/2 + (1/γ)·ln(1+γ/k)`

- **Approximate optimal bid distance:**  
  `δ^{b,*} = ((1+2q)·γσ²(T-t))/2 + (1/γ)·ln(1+γ/k)`

### Sample Reservation Price Grid (γ=0.1, S=100, σ=2, T=1)

| q | t=0.00 | t=0.25 | t=0.50 | t=0.75 |
|---|--------|--------|--------|--------|
| -5 | 102.000 | 101.500 | 101.000 | 100.500 |
| -3 | 101.200 | 100.900 | 100.600 | 100.300 |
| -1 | 100.400 | 100.300 | 100.200 | 100.100 |
|  0 | 100.000 | 100.000 | 100.000 | 100.000 |
|  1 |  99.600 |  99.700 |  99.800 |  99.900 |
|  3 |  98.800 |  99.100 |  99.400 |  99.700 |
|  5 |  98.000 |  98.500 |  99.000 |  99.500 |

### Plots Generated
- `results/exp1_reservation_price_vs_q.png` — Reservation price shift vs inventory q for multiple γ
- `results/exp1_quote_distances_vs_time.png` — Quote distances δ^a, δ^b vs time-to-maturity
- `results/exp1_spread_decomposition.png` — Spread components (inventory + constant)

---

## Experiment 2: Monte Carlo Simulation – Table-Faithful Track

**Setup:** 1000 Monte Carlo paths per γ, N=200 steps, dt=0.005, S₀=100, σ=2, A=140, k=1.5.  
**Spread:** Constant `ψ(γ) = (2/γ)·ln(1+γ/k)` centered on reservation price (inventory-aware) or mid-price (symmetric).  
**Fill model:** Independent capped Bernoulli draws, prob = min(λ·dt, 1).

### Summary Statistics

| γ | Strategy | Spread | Mean Profit | Std Profit | Mean Final q | Std Final q |
|---|----------|--------|-------------|------------|--------------|-------------|
| 0.10 | Inventory-Aware | 1.2908 | **64.04** | **6.04** | 0.031 | **2.89** |
| 0.10 | Symmetric | 1.2908 | **68.47** | **13.19** | -0.135 | **8.55** |
| 0.01 | Inventory-Aware | 1.3289 | 68.29 | 8.82 | 0.022 | 5.14 |
| 0.01 | Symmetric | 1.3289 | 68.50 | 13.09 | -0.103 | 8.49 |
| 0.50 | Inventory-Aware | 1.1507 | 23.85 | 5.18 | 0.075 | 1.97 |
| 0.50 | Symmetric | 1.1507 | 67.77 | 13.37 | -0.163 | 8.75 |

### Comparison with Paper Targets (γ=0.1)

| Strategy | Metric | Paper Value | Simulated | Deviation |
|----------|--------|-------------|-----------|-----------|
| Inventory-Aware | Mean Profit | 62.94 | 64.04 | +1.75% |
| Inventory-Aware | Std Profit | 5.89 | 6.04 | +2.58% |
| Inventory-Aware | Mean Final q | 0.100 | 0.031 | — |
| Inventory-Aware | Std Final q | 2.80 | 2.89 | +3.20% |
| Symmetric | Mean Profit | 67.21 | 68.47 | +1.87% |
| Symmetric | Std Profit | 13.43 | 13.19 | -1.82% |
| Symmetric | Std Final q | 8.66 | 8.55 | -1.23% |

### Key Findings

1. ✅ **Inventory-aware strategy achieves lower P&L variance** (std ≈6.04 vs 13.19 for γ=0.1)
2. ✅ **Inventory-aware strategy achieves lower final inventory variance** (std ≈2.89 vs 8.55 for γ=0.1)
3. ✅ **Symmetric benchmark delivers higher mean profit** (68.47 vs 64.04 for γ=0.1)
4. ✅ **As γ → 0.01, strategies converge** (std profits: 8.82 vs 13.09, closer than at γ=0.5)
5. ✅ **Spread formula reproduces paper table values** within 0.1%

### Plots Generated
- `results/exp2_sample_path_gamma_0.1.png` — Sample path with S, r, bid, ask, and inventory
- `results/table_faithful_profit_hist_gamma_0.1.png` — Terminal profit distributions, γ=0.1
- `results/table_faithful_profit_hist_gamma_0.01.png` — Terminal profit distributions, γ=0.01
- `results/table_faithful_profit_hist_gamma_0.5.png` — Terminal profit distributions, γ=0.5

---

## Experiment 3: Monte Carlo Simulation – Theory-Faithful Track

**Setup:** Same parameters as Exp 2.  
**Spread:** Full time-varying formula `ψ(t) = γσ²(T-t) + (2/γ)·ln(1+γ/k)`.  
**Fill model:** Exact Poisson counts N_fill ~ Poisson(λ·dt).

### Spread Comparison: Theory-Faithful vs Table-Faithful

| γ | Table-Faithful Spread | Theory-Faithful Spread (t=0) | Difference |
|---|----------------------|------------------------------|------------|
| 0.10 | 1.2908 | 1.6908 | +0.40 |
| 0.01 | 1.3289 | 1.3689 | +0.04 |
| 0.50 | 1.1507 | 3.1507 | +2.00 |

> **Note:** For large γ (=0.5), the full spread at t=0 is ~2.7× larger than the table-faithful spread, explaining significant differences in results.

### Summary Statistics

| γ | Strategy | Spread | Mean Profit | Std Profit | Mean Final q | Std Final q |
|---|----------|--------|-------------|------------|--------------|-------------|
| 0.10 | Inventory-Aware | 1.6908 | **63.48** | **7.43** | 0.127 | **3.43** |
| 0.10 | Symmetric | 1.6908 | **68.04** | **15.27** | -0.084 | **9.75** |
| 0.01 | Inventory-Aware | 1.3689 | 68.70 | 10.34 | -0.040 | 6.11 |
| 0.01 | Symmetric | 1.3689 | 68.90 | 16.10 | -0.155 | 10.11 |
| 0.50 | Inventory-Aware | 3.1507 | 36.09 | 7.17 | 0.157 | 2.34 |
| 0.50 | Symmetric | 3.1507 | 57.98 | 12.40 | -0.003 | 7.99 |

### Key Findings

1. ✅ **Qualitative paper claims hold under theory-faithful equations:** inventory-aware strategy still shows lower P&L variance and lower inventory variance
2. ✅ **Symmetric benchmark still earns higher mean profit** in all γ cases
3. ⚠️ **Quantitative discrepancy vs paper tables:** std profit is higher (7.43 vs 5.89 for γ=0.1) because the full spread formula is wider
4. ✅ **The paper's equations and tables are NOT numerically consistent** for γ=0.5 (full spread = 3.15 vs table spread = 1.15 — a factor of ~2.7)
5. ✅ **γ=0.01: both strategies are most similar** — confirms convergence claim

### Cross-Track Comparison (Mean Profit)

| γ | Strategy | Table-Faithful | Theory-Faithful |
|---|----------|----------------|-----------------|
| 0.10 | Inventory-Aware | 64.04 | 63.48 |
| 0.10 | Symmetric | 68.47 | 68.04 |
| 0.50 | Inventory-Aware | 23.85 | 36.09 |
| 0.50 | Symmetric | 67.77 | 57.98 |

### Plots Generated
- `results/exp3_sample_path_gamma_0.1.png` — Theory-faithful sample path with δ^a, δ^b
- `results/theory_faithful_profit_hist_gamma_0.1.png` — Terminal profit distributions, γ=0.1
- `results/theory_faithful_profit_hist_gamma_0.01.png` — Terminal profit distributions, γ=0.01
- `results/theory_faithful_profit_hist_gamma_0.5.png` — Terminal profit distributions, γ=0.5

---

## Experiment 4: Execution Intensity Derivation

**Objective:** Derive and illustrate the execution intensity function λ(δ) from microstructure assumptions about market-order size distributions and temporary price impact.

### Case A: Exponential Intensity (Logarithmic Impact)

**Assumptions:**
- Market-order sizes: P(Q > q) ∝ q^{−α} (power-law tail)
- Temporary impact: ΔP = (1/K)·ln(Q) ⟹ Q = exp(K·ΔP)

**Derivation:**
```
λ(δ) = Λ · P(ΔP > δ)
      = Λ · P(Q > c·exp(K·δ))
      ∝ Λ · [c·exp(K·δ)]^{−α}
      = A · exp(−k·δ)
```
**Parameter mapping:** k = α·K,  A ∝ Λ·c^{−α}

### Case B: Power-Law Intensity (Power-Law Impact)

**Assumptions:**
- Market-order sizes: P(Q > q) ∝ q^{−α}
- Temporary impact: ΔP ∝ Q^β ⟹ Q ∝ (ΔP)^{1/β}

**Derivation:**
```
λ(δ) = Λ · P(ΔP > δ)
      ∝ Λ · [c·δ^{1/β}]^{−α}
      = B · δ^{−α/β}
```
**Parameter mapping:** ρ = α/β,  B ∝ Λ·c^{−α}

### Numerical Illustrations (Exponential Parameters from Microstructure)

| α | K | A (prop.) | k |
|---|---|-----------|---|
| 1.40 | 0.5 | 0.7143 | 0.700 |
| 1.40 | 1.0 | 0.7143 | 1.400 |
| 1.40 | 2.0 | 0.7143 | 2.800 |
| 1.50 | 0.5 | 0.6667 | 0.750 |
| 1.50 | 1.0 | 0.6667 | 1.500 |
| 1.50 | 2.0 | 0.6667 | 3.000 |
| 1.53 | 0.5 | 0.6536 | 0.765 |
| 1.53 | 1.0 | 0.6536 | 1.530 |
| 1.53 | 2.0 | 0.6536 | 3.060 |

> **Paper parameters A=140, k=1.5** are imposed directly in simulations and not estimated from the above microstructure parameters. The derivation explains the *functional form* (exponential in δ), not the specific numerical values.

### Key Findings

1. ✅ **Exponential intensity λ(δ) = A·exp(−k·δ) arises naturally** from power-law order sizes with logarithmic market impact
2. ✅ **Power-law intensity λ(δ) = B·δ^{−α/β} arises** from power-law order sizes with power-law impact
3. ✅ **Exponential form is more analytically tractable** (HJB simplification yields closed-form quotes)
4. ✅ **Paper's k=1.5 is consistent with** α≈1.5, K≈1.0 (literature-based values)

### Plots Generated
- `results/exp4_exponential_intensity.png` — Semilog intensity curves for multiple (α, K)
- `results/exp4_powerlaw_intensity.png` — Log-log intensity curves for multiple (α, β)
- `results/exp4_shape_comparison.png` — Normalized exponential vs power-law shape comparison
- `results/exp4_paper_intensity.png` — Paper's specific intensity λ(δ) = 140·exp(−1.5·δ)

---

## Test Suite Summary

```
38 tests collected, 38 passed (0 failed)

test_exp1.py  — 14 tests  ✅
test_exp2.py  —  7 tests  ✅
test_exp3.py  —  7 tests  ✅
test_exp4.py  — 10 tests  ✅
```

---

## Overall Conclusions

1. **The paper's theoretical formulas are self-consistent.** Symbolic verification confirms that all reservation price identities and spread decompositions hold exactly.

2. **The table-faithful constant-spread interpretation reproduces the paper's tables.** Using ψ(γ) = (2/γ)·ln(1+γ/k) as a constant spread yields spread values within 0.1% of the paper's reported values and qualitative statistics consistent with the reported tables.

3. **The theory-faithful implementation confirms qualitative findings but does not numerically match the tables.** The full time-varying spread γσ²(T-t) + (2/γ)·ln(1+γ/k) generates wider spreads (especially for large γ), producing different numerical outcomes while preserving the paper's directional claims.

4. **Core paper claims are validated in both tracks:**
   - Inventory-aware quoting reduces P&L variance relative to symmetric benchmark ✅
   - Inventory-aware quoting reduces final inventory dispersion ✅
   - Symmetric benchmark typically achieves higher mean profit ✅
   - Strategies converge as γ → 0 ✅

5. **The execution intensity derivation clarifies the microstructure foundation** of the exponential form λ(δ) = A·exp(−k·δ), showing it follows from power-law order-size tails under logarithmic market impact.
