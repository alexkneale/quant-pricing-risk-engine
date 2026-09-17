# Quantitative Derivative Pricing: Intuition, Mathematics, and Numerical Schemes

This document bridges the gap between theoretical financial mathematics and production numerical code. It covers the core mathematical foundations of option pricing across analytical formulas, finite difference PDE solvers, and Monte Carlo simulation.

---

## Table of Contents
1. [The Core Philosophy: Risk-Neutral Pricing](#1-the-core-philosophy-risk-neutral-pricing)
2. [Analytical Pricing & Sensitivities (The Greeks)](#2-analytical-pricing--sensitivities-the-greeks)
3. [The PDE Approach & The Crank-Nicolson Scheme](#3-the-pde-approach--the-crank-nicolson-scheme)
4. [Path-Dependent Options & Monte Carlo Methods](#4-path-dependent-options--monte-carlo-methods)
5. [American Options & Early Exercise: PDE vs. Longstaff-Schwartz](#5-american-options--early-exercise-pde-vs-longstaff-schwartz)

---

## 1. The Core Philosophy: Risk-Neutral Pricing

### 1.1 The Fundamental Problem: Why Drift Doesn't Matter
In the real world ($\mathbb{P}$-measure), stock prices follow Geometric Brownian Motion (GBM):
$$dS_t = \mu S_t dt + \sigma S_t dW_t^\mathbb{P}$$

Here, $\mu$ is the expected real-world return (the drift). In reality, estimating $\mu$ from historical data is notoriously noisy. 

However, **the price of an option does not depend on $\mu$ at all.** 

#### The Intuition
If you buy an option and continuously hedge your risk by selling $\Delta = \frac{\partial V}{\partial S}$ shares of stock, you create a portfolio with **zero directional market risk**:
$$\Pi_t = V_t - \Delta_t S_t$$

Because the portfolio is instantaneously riskless, it *must* grow at the risk-free rate $r$. If it grew any faster or slower, a trader could borrow at rate $r$, buy or short the portfolio, and lock in guaranteed risk-free profit (arbitrage). 

Because the risk has been hedged out, the expected growth rate of the stock ($\mu$) cancels out entirely. The only parameters that survive are:
1. Current spot price $S$
2. Strike price $K$
3. Time to expiry $T$
4. Risk-free rate $r$
5. Volatility $\sigma$

### 1.2 The Risk-Neutral Measure ($\mathbb{Q}$) and Martingales
Because derivative pricing is independent of investor risk preferences, we can pretend that all investors are risk-neutral. In a risk-neutral world:
- Investors demand no risk premium to hold risky assets.
- Every asset's expected rate of return is the risk-free rate $r$ (adjusted for continuous dividend yield $q$).

Mathematically, this transformation is formalized by **Girsanov's Theorem**:
We shift from physical measure $\mathbb{P}$ to risk-neutral measure $\mathbb{Q}$ by subtracting the *market price of risk* $\theta = \frac{\mu - r}{\sigma}$:
$$dW_t^\mathbb{Q} = dW_t^\mathbb{P} + \theta dt$$

Under the risk-neutral measure $\mathbb{Q}$, the asset dynamics become:
$$dS_t = (r - q) S_t dt + \sigma S_t dW_t^\mathbb{Q}$$

**The Discounted Expectation Formula:**
By the First Fundamental Theorem of Asset Pricing, the fair value of any European contingent claim with terminal payoff $\Phi(S_T)$ is simply its **discounted expected payoff under $\mathbb{Q}$**:
$$V(S, t) = e^{-r(T - t)} \mathbb{E}^\mathbb{Q}\left[ \Phi(S_T) \;\Big|\; S_t = S \right]$$

---

## 2. Analytical Pricing & Sensitivities (The Greeks)

### 2.1 Black-Scholes-Merton Formula Deconstructed
For a European Call option with continuous dividend yield $q$, the terminal payoff is $\max(S_T - K, 0)$. Solving the risk-neutral expectation integral yields:
$$C(S, t) = S e^{-q\tau} \mathcal{N}(d_1) - K e^{-r\tau} \mathcal{N}(d_2)$$
where $\mathcal{N}$ is the CDF of the normal distribution, $\tau = T - t$ is time to maturity and
$$d_1 = \frac{\ln(S / K) + \left(r - q + \frac{1}{2}\sigma^2\right)\tau}{\sigma \sqrt{\tau}}, \quad d_2 = d_1 - \sigma \sqrt{\tau}$$

#### Intuitive Meaning of $d_1$ and $d_2$
Do not view $d_1$ and $d_2$ as arbitrary formulas; both have clean probabilistic meanings:
1. **$\mathcal{N}(d_2)$ = Probability of Exercise:**
   $$\mathcal{N}(d_2) = \mathbb{Q}(S_T > K)$$
   It is the exact risk-neutral probability that the option expires in-the-money. The term $K e^{-r\tau}\mathcal{N}(d_2)$ is therefore the discounted expected strike cash payment you must make to exercise.
2. **$\mathcal{N}(d_1)$ = Asset-Weighted Probability:**
   $$\mathcal{N}(d_1) = \frac{\mathbb{E}^\mathbb{Q}[S_T \cdot \mathbf{1}_{S_T > K}]}{\mathbb{E}^\mathbb{Q}[S_T]}$$
   It reflects the expected stock value received conditional on exercise, scaled by current spot.

For a European Put option ($\Phi(S_T) = \max(K - S_T, 0)$):
$$P(S, t) = K e^{-r\tau} \mathcal{N}(-d_2) - S e^{-q\tau} \mathcal{N}(-d_1)$$

Put and call prices are linked by **Put-Call Parity**:
$$C - P = S e^{-q\tau} - K e^{-r\tau}$$

---

### 2.2 The Greeks: Sensitivities as Physical Derivatives

The Greeks are partial derivatives that tell a risk manager how to hedge their portfolio:

| Greek | Math Definition | Physical Analogy | Financial Meaning |
| :--- | :--- | :--- | :--- |
| **Delta ($\Delta$)** | $\frac{\partial V}{\partial S}$ | Velocity | Number of shares needed to hedge 1 option. For calls, $\Delta \in (0, 1)$; for puts, $\Delta \in (-1, 0)$. |
| **Gamma ($\Gamma$)** | $\frac{\partial^2 V}{\partial S^2}$ | Acceleration / Curvature | How fast Delta changes as spot moves. High Gamma near strike means the hedge ratio changes rapidly. |
| **Vega ($\mathcal{V}$)** | $\frac{\partial V}{\partial \sigma}$ | Diffusion sensitivity | Option value gained per $1\%$ rise in volatility. Always positive for vanilla options. |
| **Theta ($\Theta$)** | $\frac{\partial V}{\partial t} = -\frac{\partial V}{\partial \tau}$ | Radioactive decay | Time decay. Options lose value as expiry approaches (all else equal). Usually negative for long options. |
| **Rho ($\rho$)** | $\frac{\partial V}{\partial r}$ | Thermal expansion | Sensitivity to interest rates. Higher interest rates increase forward price, helping calls and hurting puts. |

#### Analytical Expressions
$$\Delta_{\text{call}} = e^{-q\tau}\mathcal{N}(d_1), \quad \Delta_{\text{put}} = -e^{-q\tau}\mathcal{N}(-d_1)$$
$$\Gamma = \frac{e^{-q\tau} \mathcal{N}'(d_1)}{S \sigma \sqrt{\tau}}$$
$$\mathcal{V} = S e^{-q\tau} \sqrt{\tau} \mathcal{N}'(d_1)$$

---

## 3. The PDE Approach & The Crank-Nicolson Scheme

### 3.1 From Portfolio Hedging to the Heat Equation
Applying Itô's Lemma to an option's price $V(S, t)$ and constructing a delta-hedged portfolio $\Pi = V - \Delta S$ yields the **Black-Scholes Partial Differential Equation**:
$$\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + (r - q)S \frac{\partial V}{\partial S} - rV = 0$$

#### The Physics Connection
If you make the coordinate change $x = \ln(S/K)$ and reverse time $\tau = T - t$, this PDE transforms directly into a **convection-diffusion-reaction equation** (equivalent to heat conduction with drift and radiative cooling):
$$\partial_\tau V = \underbrace{\frac{1}{2}\sigma^2 \partial_{xx}V}_{\text{Diffusion (Volatility)}} + \underbrace{\left(r - q - \frac{1}{2}\sigma^2\right)\partial_x V}_{\text{Advection (Drift)}} - \underbrace{rV}_{\text{Decay (Discounting)}}$$

### 3.2 Finite Difference Schemes
We discretize price $S \in [0, S_{\max}]$ into $M$ intervals ($\Delta S = S_{\max}/M$) and time-to-expiry $\tau \in [0, T]$ into $N$ intervals ($\Delta \tau = T/N$). Let $V_i^n \approx V(i \Delta S, n \Delta \tau)$.

Spatial derivatives use central differences:
$$\frac{\partial V}{\partial S} \approx \frac{V_{i+1} - V_{i-1}}{2\Delta S}, \quad \frac{\partial^2 V}{\partial S^2} \approx \frac{V_{i+1} - 2V_i + V_{i-1}}{\Delta S^2}$$

This reduces the differential operator $\mathcal{L}_{BS} V$ to:
$$\mathcal{L}_{BS} V_i = \alpha_i V_{i-1} + \beta_i V_i + \gamma_i V_{i+1}$$
where:
$$\alpha_i = \frac{1}{2}\left[\sigma^2 i^2 - (r - q)i\right], \quad \beta_i = -(\sigma^2 i^2 + r), \quad \gamma_i = \frac{1}{2}\left[\sigma^2 i^2 + (r - q)i\right]$$

### 3.3 Why Crank-Nicolson?
- **Explicit Scheme ($\theta = 0$):** Evaluates spatial derivatives at the known time step $n$. It is computationally fast, but **conditionally stable** (CFL condition: $\Delta \tau \le \frac{\Delta S^2}{\sigma^2 S^2}$). If your time step is even slightly too large, the solution blows up to infinity.
- **Fully Implicit Scheme ($\theta = 1$):** Evaluates spatial derivatives at the future step $n+1$. It is unconditionally stable, but only first-order accurate in time: $\mathcal{O}(\Delta \tau + \Delta S^2)$.
- **Crank-Nicolson Scheme ($\theta = 1/2$):** Averages the explicit and implicit steps (equivalent to the trapezoidal rule in time):
  $$\frac{V_i^{n+1} - V_i^n}{\Delta \tau} = \frac{1}{2}\left[\mathcal{L}_{BS} V_i^{n+1} + \mathcal{L}_{BS} V_i^n\right]$$

**Key Properties:**
- **Unconditionally stable ($A$-stable):** No numerical explosions, regardless of grid size.
- **Second-order accurate:** Convergence rate $\mathcal{O}(\Delta \tau^2 + \Delta S^2)$.

Rearranging gives a tridiagonal linear equation for interior nodes at each backward time step:
$$A V^{n+1} = B V^n + d^n$$
where $A$ and $B$ are tridiagonal matrices:
- $A$: Implicit operator applied to $V^{n+1}$ (contains sub-diagonal, main diagonal, super-diagonal).
- $B$: Explicit operator applied to $V^n$.
- $d^n$: Vector holding boundary values at $S=0$ and $S=S_{\max}$.

### 3.4 Solving in $\mathcal{O}(M)$ Time: The Thomas Algorithm
A general $M \times M$ linear system takes $\mathcal{O}(M^3)$ via Gaussian elimination. Because $A$ is tridiagonal, the **Thomas algorithm** (Tridiagonal Matrix Algorithm, TDMA) solves it in **$\mathcal{O}(M)$ operations** via forward elimination and backward substitution:
```text
Forward sweep: eliminate sub-diagonal -> Backward sweep: back-substitute
```

---

## 4. Path-Dependent Options & Monte Carlo Methods

### 4.1 Monte Carlo and Feynman-Kac
By the **Feynman-Kac Theorem**, solving the Black-Scholes PDE is mathematically identical to simulating sample paths of the SDE and taking the average discounted payoff:
$$V_0 = e^{-rT} \mathbb{E}^\mathbb{Q}[\Phi(S_T)] \approx \frac{1}{M} \sum_{m=1}^M e^{-rT} \Phi(S_T^{(m)})$$

#### Path Simulation
Under $\mathbb{Q}$, by applying Itô's lemma to $\ln S_t$, the exact transition over a discrete time step $\Delta t$ is:
$$S_{t+\Delta t} = S_t \exp\left( \left(r - q - \frac{1}{2}\sigma^2\right)\Delta t + \sigma \sqrt{\Delta t} Z \right), \quad Z \sim \mathcal{N}(0, 1)$$
Because this uses the exact solution to the log-SDE, there is **zero discretization bias** between time steps (unlike Euler-Maruyama approximations for non-linear SDEs).

### 4.2 Variance Reduction: Antithetic Variates
The standard error of a Monte Carlo estimate with $M$ paths scales as:
$$\text{SE} = \frac{\sigma_{\text{payoff}}}{\sqrt{M}}$$
To halve the standard error, you need $4\times$ more paths.

**Antithetic Sampling Intuition:**
For every Gaussian draw $Z \sim \mathcal{N}(0, 1)$ generating a path $S^{(1)}$, we also generate a twin path $S^{(2)}$ using $-Z$.
- If $Z$ drove the stock higher than average, $-Z$ drives it lower than average.
- Since option payoffs are monotonic with respect to the noise, $\text{Cov}(\Phi(S^{(1)}), \Phi(S^{(2)})) < 0$.
- When we average the two payoffs, the negative covariance cancels out a significant portion of the variance at zero additional sampling cost:
$$\text{Var}\left(\frac{\Phi_1 + \Phi_2}{2}\right) = \frac{1}{4}\text{Var}(\Phi_1) + \frac{1}{4}\text{Var}(\Phi_2) + \frac{1}{2}\underbrace{\text{Cov}(\Phi_1, \Phi_2)}_{< 0} < \frac{1}{2}\text{Var}(\Phi)$$

### 4.3 Path-Dependent Example: Asian Options
An Asian option's payoff depends on the **average price** over the life of the option:
$$\Phi(S) = \max\left(\frac{1}{K_{\text{steps}}}\sum_{k=1}^{K_{\text{steps}}} S_{t_k} - K, \; 0\right)$$
- **Why PDEs struggle:** To price this via PDE, you must add the running average as a second spatial state variable, turning a 1D PDE into a 2D PDE (exponentially increasing grid points).
- **Why Monte Carlo excels:** Monte Carlo tracks the running average along each trajectory automatically, adding virtually zero computational cost.

---

## 5. American Options & Early Exercise: PDE vs. Longstaff-Schwartz

An American option grants the holder the right to exercise at any stopping time $\tau^* \in [0, T]$. This flexibility makes the contract strictly more valuable than its European counterpart:
$$V_{\text{American}} \ge V_{\text{European}}$$
The difference is the **Early Exercise Premium**:
$$\text{Premium} = V_{\text{American}} - V_{\text{European}} \ge 0$$

### 5.1 The PDE Approach: A Free-Boundary Problem
In PDE terms, American exercise creates an unknown boundary $S^*(t)$ (the optimal exercise frontier). At any point in space and time, the option cannot trade below its intrinsic value $\Phi(S) = \max(K - S, 0)$.

This yields a **Linear Complementarity Problem (Variational Inequality)**:
$$\max\left( \partial_\tau V - \mathcal{L}_{BS} V, \; \Phi(S) - V \right) = 0$$

#### Implementation in Crank-Nicolson
At every backward time step:
1. Solve the tridiagonal system to obtain the continuation value: $\tilde{V}^{n+1} = A^{-1} (B V^n + d^n)$.
2. Project onto the early-exercise constraint:
   $$V_i^{n+1} = \max\left(\tilde{V}_i^{n+1}, \; \Phi(S_i)\right)$$

This guarantees that whenever the continuation value drops below the immediate exercise payoff, early exercise is chosen.

---

### 5.2 The Monte Carlo Dilemma and Longstaff-Schwartz (LSM)
Standard Monte Carlo simulates paths **forward in time**, but early exercise decisions require knowing the **future continuation value**:
$$C(S_t) = \mathbb{E}^\mathbb{Q}\left[ e^{-r\Delta t} V(S_{t+\Delta t}, t+\Delta t) \;\Big|\; S_t \right]$$

How do you compute a conditional expectation conditional on being at state $S_t$ when each simulated path is independent?

#### The Longstaff-Schwartz Insight (2001)
Longstaff and Schwartz solved this using **backward induction across paths combined with cross-sectional Ordinary Least Squares (OLS) regression**:

```text
At Maturity (t = T):
  Payoff is known on all paths: V(S_T) = max(K - S_T, 0)

Step Backward (t = T - dt down to t = 1):
  1. Filter only paths that are In-The-Money (ITM: K - S_t > 0).
  2. Set Independent Variable: X = S_t (current spot on ITM paths).
  3. Set Dependent Variable:   Y = Discounted realized future cashflow along that path.
  4. Fit polynomial regression:
       E[Y | X] ≈ β_0 + β_1*X + β_2*X^2 + ...
  5. The fitted curve gives the predicted Continuation Value C(S_t).
  6. Exercise Policy:
       If Payoff(S_t) > C(S_t) -> Exercise now! Update cashflow to Payoff(S_t).
       Else                    -> Keep holding. Discount future cashflow.
```

By regressing realized future cash flows onto low-degree polynomials (or Laguerre polynomials) across all in-the-money paths, LSM extracts a smooth, global estimate of the conditional continuation function.

---

## 6. Summary Comparison Matrix

| Feature | Closed-Form (BSM) | Finite Difference (Crank-Nicolson) | Monte Carlo (Standard / LSM) |
| :--- | :--- | :--- | :--- |
| **Speed** | Instant ($\sim \mu\text{s}$) | Fast ($\sim \text{ms}$) | Medium / Slow ($\sim 10\text{ms} - 1\text{s}$) |
| **Contract Styles** | European vanilla only | European, American, Bermudan | European, American (LSM), Exotic |
| **Path-Dependent Payoffs** | Fails (except simple barriers) | Cumbersome (adds spatial dimensions) | **Native & trivial** (Asian, Lookback) |
| **Dimension Scaling** | Single asset | Suffers curse of dimensionality ($d \le 3$) | **Immune to curse of dimensionality** |
| **Greeks Quality** | Exact closed-form | High (direct grid differentiation) | Noisy (requires pathwise or bump-and-reprice) |
