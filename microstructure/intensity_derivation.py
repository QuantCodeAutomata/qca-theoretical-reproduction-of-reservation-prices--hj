import numpy as np
import sympy as sp
from typing import Dict, Tuple, List

# Symbolic variables for microstructure model
delta, Q, Lambda_sym, alpha_sym, K_sym, beta_sym, A_sym, k_sym, B_sym = sp.symbols(
    'delta Q Lambda alpha K beta A k B',
    positive=True
)
c, q_var = sp.symbols('c q_var', positive=True)


def derive_exponential_intensity() -> Dict:
    """
    Derive lambda(delta) = A*exp(-k*delta) from power-law order-size tails with log impact.

    Assumptions:
    - Market orders arrive at rate Lambda (aggregate frequency)
    - Size Q has power-law tail: P(Q > q) ~ q^{-alpha} for large q
    - Logarithmic impact: Delta_p = (1/K)*ln(Q) => Q ~ exp(K*Delta_p)

    Derivation:
    lambda(delta) = Lambda * P(Delta_p > delta)
                  = Lambda * P(Q > c*exp(K*delta))  [for some constant c>0]
                  ~ Lambda * (c*exp(K*delta))^{-alpha}
                  = A * exp(-alpha*K*delta)

    So: k = alpha*K, A = Lambda * c^{-alpha}
    """
    # Custom — Context7 found no library equivalent (paper microstructure derivation)

    # Tail probability: P(Q > q_var) ~ q_var^{-alpha}
    tail_prob = q_var ** (-alpha_sym)

    # Substitute Q = c * exp(K * delta) via logarithmic impact inversion
    q_substituted = c * sp.exp(K_sym * delta)
    tail_at_delta = tail_prob.subs(q_var, q_substituted)
    tail_simplified = sp.simplify(tail_at_delta)

    # lambda(delta) = Lambda * P(Delta_p > delta)
    intensity_full = Lambda_sym * tail_simplified

    # Factor into A * exp(-k*delta) form
    # intensity_full = Lambda * c^{-alpha} * exp(-alpha*K*delta)
    intensity_factored = sp.collect(sp.expand(intensity_full), sp.exp(-alpha_sym * K_sym * delta))

    derivation_steps = [
        "Step 1: Assume P(Q > q) ~ q^{-alpha}  (power-law tail, normalized at q=1)",
        "Step 2: Logarithmic price impact: Delta_p = (1/K) * ln(Q)  =>  Q = exp(K * Delta_p)",
        "Step 3: P(Delta_p > delta) = P(Q > c * exp(K * delta))  for threshold constant c > 0",
        "Step 4: Substitute: P(Q > c*exp(K*delta)) ~ [c*exp(K*delta)]^{-alpha}",
        "        = c^{-alpha} * exp(-alpha * K * delta)",
        "Step 5: lambda(delta) = Lambda * c^{-alpha} * exp(-alpha * K * delta)",
        "        = A * exp(-k * delta)  where A = Lambda * c^{-alpha},  k = alpha * K",
    ]

    return {
        'intensity_expr': A_sym * sp.exp(-k_sym * delta),
        'k_formula': alpha_sym * K_sym,
        'A_formula': Lambda_sym / alpha_sym,   # proportional to (c absorbed)
        'tail_probability': tail_simplified,
        'derivation_steps': derivation_steps,
    }


def derive_powerlaw_intensity() -> Dict:
    """
    Derive lambda(delta) = B*delta^{-alpha/beta} from power-law order-size tails
    with power-law impact.

    Assumptions:
    - Size Q has power-law tail: P(Q > q) ~ q^{-alpha}
    - Power-law impact: Delta_p ~ Q^{beta} => Q ~ (Delta_p)^{1/beta}

    Derivation:
    lambda(delta) = Lambda * P(Delta_p > delta)
                  = Lambda * P(Q > c*delta^{1/beta})
                  ~ Lambda * (c*delta^{1/beta})^{-alpha}
                  = B * delta^{-alpha/beta}
    """
    # Tail probability: P(Q > q_var) ~ q_var^{-alpha}
    tail_prob = q_var ** (-alpha_sym)

    # Substitute Q = c * delta^{1/beta} via power-law impact inversion
    q_substituted = c * delta ** (sp.Integer(1) / beta_sym)
    tail_at_delta = tail_prob.subs(q_var, q_substituted)
    tail_simplified = sp.simplify(tail_at_delta)

    # lambda(delta) = Lambda * P(Delta_p > delta) = Lambda * c^{-alpha} * delta^{-alpha/beta}
    intensity_full = sp.simplify(Lambda_sym * tail_simplified)

    derivation_steps = [
        "Step 1: Assume P(Q > q) ~ q^{-alpha}  (power-law tail, normalized at q=1)",
        "Step 2: Power-law price impact: Delta_p ~ Q^{beta}  =>  Q ~ (Delta_p)^{1/beta}",
        "Step 3: P(Delta_p > delta) = P(Q > c * delta^{1/beta})  for threshold constant c > 0",
        "Step 4: Substitute: P(Q > c*delta^{1/beta}) ~ [c * delta^{1/beta}]^{-alpha}",
        "        = c^{-alpha} * delta^{-alpha/beta}",
        "Step 5: lambda(delta) = Lambda * c^{-alpha} * delta^{-alpha/beta}",
        "        = B * delta^{-alpha/beta}  where B = Lambda * c^{-alpha},  rho = alpha/beta",
    ]

    return {
        'intensity_expr': B_sym * delta ** (-alpha_sym / beta_sym),
        'exponent': alpha_sym / beta_sym,
        'derivation_steps': derivation_steps,
    }


def compute_exponential_params(
    Lambda_val: float, alpha_val: float, K_val: float
) -> Tuple[float, float]:
    """Map microstructure parameters to exponential intensity parameters."""
    A = Lambda_val / alpha_val   # proportional; exact constant absorbed
    k = alpha_val * K_val
    return (A, k)


def compute_powerlaw_params(
    Lambda_val: float, alpha_val: float, beta_val: float, c_val: float = 1.0
) -> Tuple[float, float]:
    """Map microstructure parameters to power-law intensity parameters."""
    B = Lambda_val * c_val ** (-alpha_val)
    rho = alpha_val / beta_val
    return (B, rho)


def exponential_intensity_func(
    delta: np.ndarray, A: float = 140.0, k: float = 1.5
) -> np.ndarray:
    """Exponential execution intensity function used in the simulation study."""
    return A * np.exp(-k * delta)


def powerlaw_intensity_func(
    delta: np.ndarray, B: float, rho: float
) -> np.ndarray:
    """Power-law execution intensity function."""
    delta = np.asarray(delta, dtype=float)
    if np.any(delta <= 0):
        raise ValueError("delta must be strictly positive for power-law intensity")
    return B * delta ** (-rho)


def normalize_at_reference(
    delta_grid: np.ndarray,
    func1: np.ndarray,
    func2: np.ndarray,
    delta_ref: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize both intensity functions to have the same value at delta=delta_ref.
    Used to compare shapes.
    """
    # Find index closest to delta_ref
    idx = np.argmin(np.abs(delta_grid - delta_ref))
    val1_ref = func1[idx]
    val2_ref = func2[idx]
    return func1 / val1_ref, func2 / val2_ref


def print_derivation_summary() -> None:
    """Print a clean summary of both intensity derivations."""
    sep = "=" * 60
    print(sep)
    print("EXECUTION INTENSITY DERIVATION SUMMARY")
    print(sep)

    print("\n--- Exponential Intensity ---")
    print("  Model:  lambda(delta) = A * exp(-k * delta)")
    print("  Origin: power-law order sizes + logarithmic market impact")
    print("  Parameter mapping:")
    print("    k = alpha * K     (tail index * log-impact coefficient)")
    print("    A = Lambda * c^{-alpha}  (proportional to arrival rate)")
    print("  Paper params: A = 140,  k = 1.5  (used directly, not estimated)")

    print("\n--- Power-Law Intensity ---")
    print("  Model:  lambda(delta) = B * delta^{-alpha/beta}")
    print("  Origin: power-law order sizes + power-law market impact")
    print("  Parameter mapping:")
    print("    rho = alpha / beta   (exponent ratio)")
    print("    B   = Lambda * c^{-alpha}")

    print("\n--- Note ---")
    print("  The paper uses A=140, k=1.5 directly in simulations.")
    print("  The microstructure derivation explains the *functional form*")
    print("  (exponential in delta), not the exact parameter values.")
    print(sep)
