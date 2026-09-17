# Quantitative Derivative Pricing & Stochastic Valuation Engine

An institutional-grade derivatives evaluation library implementing analytical solutions, finite-difference parabolic partial differential equation (PDE) solvers, and path-dependent Monte Carlo engines under the risk-neutral measure ($\mathbb{Q}$).

---

## Technical Features

### 1. Black-Scholes-Merton Analytical Framework
- **Continuous Dividend Yield ($q$):** Exact Merton extension handling dividend drag.
- **Closed-Form Greeks:** First and second-order analytical derivatives ($\Delta, \Gamma, \mathcal{V}, \Theta, \rho$).
- **Vectorized NumPy Implementation:** Optimized array operations avoiding interpreted loop overhead.

### 2. Crank-Nicolson Finite Difference Solver
- **$A$-Stable Discretization:** Implicit-explicit Crank-Nicolson scheme with $\mathcal{O}(\Delta \tau^2 + \Delta S^2)$ convergence.
- **Thomas Algorithm ($TDMA$):** Linear $\mathcal{O}(M)$ solver for interior tridiagonal systems.
- **American Free Boundary Enforcement:** Dynamic complementarity projection at each backward time slice:
  $$V^{n+1} = \max(\tilde{V}^{n+1}, \Phi(S))$$

### 3. Path-Dependent & High-Dimensional Monte Carlo
- **Geometric Brownian Motion Paths:** Exact log-normal transition state updates.
- **Variance Reduction:** Built-in antithetic sampling.
- **Exotic Asian Valuation:** Discrete monitoring engines for both arithmetic and geometric path averages.
- **Longstaff-Schwartz LSM:** Backward dynamic programming with weighted polynomial regression on in-the-money states to value American early exercise.

---

## Numerical Benchmarks

The suite can be verified using `pytest`:

```bash
# Run test suite with full coverage
python3 -m pytest -v tests/test_pricing.py
```

### Verification Against Benchmarks ($S_0=100, K=100, T=1.0, r=0.05, \sigma=0.20, q=0.02$)

| Engine | Contract | Style | Value ($) | Error / Standard Error |
| :--- | :--- | :--- | :--- | :--- |
| **Analytical BSM** | Vanilla Call | European | `9.2270` | Reference |
| **Analytical BSM** | Vanilla Put | European | `6.3504` | Reference |
| **Crank-Nicolson PDE** | Vanilla Put | European | `6.3481` | $\Delta = -0.0023$ |
| **Crank-Nicolson PDE** | Vanilla Put | American | `6.6852` | Early Ex. Prem: `+$0.337` |
| **Antithetic Monte Carlo** | Vanilla Call | European | `9.2241` | $\text{SE} = \pm 0.025$ |
| **Longstaff-Schwartz LSM** | Vanilla Put | American | `6.6620` | $\text{SE} = \pm 0.038$ |
| **Monte Carlo (Arithmetic)** | Asian Call | European | `5.1482` | $\text{SE} = \pm 0.018$ |

---

## Quickstart Usage

```python
from pricing import BlackScholesPricer, CrankNicolsonPricer, MonteCarloEngine

# 1. Closed-Form Greek Extraction
bs = BlackScholesPricer(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.20, q=0.02)
print("Delta:", bs.greeks("call")["delta"])
print("Gamma:", bs.greeks("call")["gamma"])

# 2. American Option via Crank-Nicolson PDE
pde = CrankNicolsonPricer(S0=100.0, K=100.0, T=1.0, r=0.05, sigma=0.20, q=0.02)
american_put = pde.price(option_type="put", exercise_style="american")
print("American Put Price (PDE):", american_put)

# 3. Path-Dependent Asian Option via Monte Carlo
mc = MonteCarloEngine(S0=100.0, K=100.0, T=1.0, r=0.05, sigma=0.20, q=0.02)
asian_price, stderr = mc.price_asian(option_type="call", averaging_type="arithmetic")
print(f"Arithmetic Asian Call: {asian_price:.4f} +/- {stderr:.4f}")
```

