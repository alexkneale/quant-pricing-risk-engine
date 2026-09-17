# Module 3: Portfolio Risk, Coherent Measures & Attribution Theory

This document details the mathematical theory, statistical derivations, and numerical implementations underpinning **Portfolio Risk & Attribution** under the physical probability measure $\mathbb{P}$.

---

## 1. Mathematical Foundations of Risk Measures

Let $(\Omega, \mathcal{F}, \mathbb{P})$ be a probability space. Consider an investment horizon $\Delta t$ and an asset portfolio with initial value $V_0$. The portfolio value at $t_1 = t_0 + \Delta t$ is a random variable $V_1$. 

We define the **Loss Random Variable** $L$ as the negative portfolio profit-and-loss ($\text{PnL}$):
$$L = -\Delta V = -(V_1 - V_0) = -V_0 R_p$$
where $R_p = \frac{V_1 - V_0}{V_0} = \sum_{i=1}^N w_i R_i$ is the linear return of the portfolio with weights $w \in \mathbb{R}^N$. By convention, positive values of $L$ denote financial loss.

Let $F_L(l) = \mathbb{P}(L \le l)$ be the cumulative distribution function (CDF) of the loss variable $L$.

### 1.1 Axiomatic Definition of Coherent Risk Measures
Following **Artzner, Delbaen, Eber, and Heath (1999)**, a functional $\rho: \mathcal{L}^\infty \to \mathbb{R}$ mapping a loss distribution to a capital requirement is a **coherent risk measure** if and only if it satisfies four axioms:

1. **Translation Invariance:**
   $$\forall c \in \mathbb{R}, \quad \rho(L + c) = \rho(L) + c$$
   Adding a deterministic cash loss $c$ increases the required capital reserve by exactly $c$.
2. **Subadditivity:**
   $$\rho(L_1 + L_2) \le \rho(L_1) + \rho(L_2)$$
   A portfolio merger or diversification cannot create more risk than the sum of the standalone risks.
3. **Positive Homogeneity:**
   $$\forall \lambda \ge 0, \quad \rho(\lambda L) = \lambda \rho(L)$$
   Scaling exposure scales the risk proportionally (absence of liquidity feedback).
4. **Monotonicity:**
   $$\text{If } L_1 \le L_2 \text{ a.s.}, \quad \text{then } \rho(L_1) \le \rho(L_2)$$
   If position 2 incurs larger losses than position 1 in all states of the world, its risk must be greater.

---

## 2. Value at Risk ($\text{VaR}$)

### 2.1 Definition
For a given confidence level $\alpha \in (0, 1)$ (typically $\alpha = 0.95$ or $\alpha = 0.99$) over horizon $\Delta t$, the Value at Risk ($\text{VaR}_\alpha$) is the generalized $(1 - \alpha)$-quantile of the loss distribution:

$$\text{VaR}_\alpha(L) = \inf \{ l \in \mathbb{R} : F_L(l) \ge \alpha \} = F_L^{-1}(\alpha)$$

In terms of portfolio return $R_p$:
$$\mathbb{P}(L > \text{VaR}_\alpha(L)) = 1 - \alpha \iff \mathbb{P}\left(R_p < -\frac{\text{VaR}_\alpha(L)}{V_0}\right) = 1 - \alpha$$

### 2.2 Flaw of $\text{VaR}$: Failure of Subadditivity
$\text{VaR}$ is **not** a coherent risk measure because it fails the **subadditivity** axiom. 

#### Counterexample:
Consider two independent digital bond positions $A$ and $B$. Each bond defaults with probability $p = 0.04$ independently.
- If default occurs, the loss is $\$1,000$.
- If no default occurs, the loss is $\$0$.

At $\alpha = 0.95$:
- For position $A$: $\mathbb{P}(L_A = 1000) = 0.04 \implies F_{L_A}(0) = 0.96 \ge 0.95 \implies \text{VaR}_{0.95}(L_A) = \$0$.
- For position $B$: similarly, $\text{VaR}_{0.95}(L_B) = \$0$.
- For the combined portfolio $L_{A+B} = L_A + L_B$:
  $$\mathbb{P}(L_{A+B} = 0) = (1 - 0.04)^2 = 0.9216 < 0.95$$
  $$\mathbb{P}(L_{A+B} \ge 1000) = 1 - 0.9216 = 0.0784 > 0.05$$
  Therefore:
  $$\text{VaR}_{0.95}(L_{A+B}) = \$1000 > \text{VaR}_{0.95}(L_A) + \text{VaR}_{0.95}(L_B) = \$0$$

$\text{VaR}$ penalizes diversification in fat-tailed or non-sub-Gaussian scenarios and is blind to the severity of losses beyond the $(1-\alpha)$ quantile.

---

## 3. Expected Shortfall ($\text{ES}$) / Conditional Value at Risk ($\text{CVaR}$)

### 3.1 Definition and Derivation
Expected Shortfall at confidence level $\alpha$ measures the conditional expectation of loss given that the loss exceeds $\text{VaR}_\alpha$:

$$\text{ES}_\alpha(L) = \mathbb{E}[L \mid L \ge \text{VaR}_\alpha(L)]$$

For continuous loss distributions with density $f_L(l)$:

$$\text{ES}_\alpha(L) = \frac{1}{1 - \alpha} \int_{\text{VaR}_\alpha(L)}^\infty l f_L(l) \, dl$$

Equivalently, substituting $u = F_L(l)$ (the quantile substitution):

$$\text{ES}_\alpha(L) = \frac{1}{1 - \alpha} \int_\alpha^1 \text{VaR}_u(L) \, du$$

### 3.2 Proof of Coherence
Rockafellar and Uryasev (2000, 2002) proved that $\text{ES}_\alpha$ can be represented as the solution to a convex optimization problem:

$$\text{ES}_\alpha(L) = \min_{z \in \mathbb{R}} \left\{ z + \frac{1}{1 - \alpha} \mathbb{E}\left[ (L - z)^+ \right] \right\}$$
where $(x)^+ = \max(x, 0)$. 

Because $(L - z)^+$ is convex with respect to $L$, subadditivity and convexity follow directly:
$$(L_1 + L_2 - (z_1 + z_2))^+ \le (L_1 - z_1)^+ + (L_2 - z_2)^+$$
Taking expectations and minimizing verifies that $\text{ES}_\alpha(L_1 + L_2) \le \text{ES}_\alpha(L_1) + \text{ES}_\alpha(L_2)$. Thus, **Expected Shortfall is a coherent risk measure**.

---

## 4. Analytical & Numerical Estimators for $\text{VaR}$ and $\text{ES}$

### 4.1 Parametric Normal Distribution
Assume portfolio returns follow a normal distribution $R_p \sim \mathcal{N}(\mu_p, \sigma_p^2)$, where:
$$\mu_p = w^T \mu, \quad \sigma_p = \sqrt{w^T \Sigma w}$$

Let $Z \sim \mathcal{N}(0, 1)$, with standard normal CDF $\Phi(z)$ and PDF $\phi(z) = \frac{1}{\sqrt{2\pi}} e^{-z^2/2}$.

#### Normal $\text{VaR}$:
$$\text{VaR}_\alpha(R_p) = -(\mu_p + \sigma_p \Phi^{-1}(1 - \alpha)) = -\mu_p + \sigma_p \Phi^{-1}(\alpha)$$

#### Normal $\text{ES}$:
$$\text{ES}_\alpha(R_p) = \mathbb{E}[-R_p \mid -R_p \ge \text{VaR}_\alpha] = -\mu_p + \sigma_p \frac{\phi(\Phi^{-1}(\alpha))}{1 - \alpha}$$

*Derivation:*
$$\int_{\Phi^{-1}(\alpha)}^\infty z \phi(z) \, dz = \int_{\Phi^{-1}(\alpha)}^\infty \frac{z}{\sqrt{2\pi}} e^{-z^2/2} \, dz = \left[ -\frac{1}{\sqrt{2\pi}} e^{-z^2/2} \right]_{\Phi^{-1}(\alpha)}^\infty = \phi(\Phi^{-1}(\alpha))$$

---

### 4.2 Parametric Student's $t$-Distribution
Financial asset log-returns exhibit leptokurtosis (fat tails). Standardize returns using a Student-$t$ distribution with $\nu > 2$ degrees of freedom:
$$\frac{R_p - \mu_p}{\sigma_{\text{scale}}} \sim t_\nu, \quad \text{where } \sigma_{\text{scale}} = \sigma_p \sqrt{\frac{\nu - 2}{\nu}}$$

Let $t_\nu(x)$ be the density and $F_\nu(x)$ be the CDF of a standard Student-$t$ distribution with $\nu$ degrees of freedom:

#### Student-$t$ $\text{VaR}$:
$$\text{VaR}_\alpha(R_p) = -\mu_p - \sigma_p \sqrt{\frac{\nu - 2}{\nu}} F_\nu^{-1}(1 - \alpha)$$

#### Student-$t$ $\text{ES}$:
$$\text{ES}_\alpha(R_p) = -\mu_p + \sigma_p \sqrt{\frac{\nu - 2}{\nu}} \left( \frac{t_\nu(F_\nu^{-1}(\alpha))}{1 - \alpha} \left( \frac{\nu + (F_\nu^{-1}(\alpha))^2}{\nu - 1} \right) \right)$$

---

### 4.3 Cornish-Fisher Expansion (Semi-Parametric)
To incorporate empirical skewness $S = \mathbb{E}[(R_p - \mu)^3]/\sigma^3$ and excess kurtosis $K = \mathbb{E}[(R_p - \mu)^4]/\sigma^4 - 3$ without assuming a parametric likelihood:

Let $z_\alpha = \Phi^{-1}(\alpha)$. The Cornish-Fisher quantile expansion adjusts the normal critical value:
$$\tilde{z}_\alpha = z_\alpha + \frac{1}{6}(z_\alpha^2 - 1)S + \frac{1}{24}(z_\alpha^3 - 3z_\alpha)K - \frac{1}{36}(2z_\alpha^3 - 5z_\alpha)S^2$$

Then:
$$\text{VaR}_\alpha^{\text{CF}} = -\mu_p + \sigma_p \tilde{z}_\alpha$$

---

### 4.4 Non-Parametric Historical Simulation
Given historical return vectors $r_t \in \mathbb{R}^N$ for $t = 1, \dots, T$:
1. Compute the pseudo-realized portfolio returns:
   $$R_{p, t} = \sum_{i=1}^N w_i r_{i, t}, \quad \forall t \in \{1, \dots, T\}$$
2. Sort $\{R_{p, t}\}_{t=1}^T$ in ascending order: $R_{p, (1)} \le R_{p, (2)} \le \dots \le R_{p, (T)}$.
3. Let $k = \lfloor (1 - \alpha) T \rfloor$. The historical empirical estimators are:
   $$\widehat{\text{VaR}}_\alpha^{\text{Hist}} = -R_{p, (k)}$$
   $$\widehat{\text{ES}}_\alpha^{\text{Hist}} = -\frac{1}{k} \sum_{j=1}^k R_{p, (j)}$$

---

### 4.5 Filtered Historical Simulation (FHS)
Standard historical simulation assumes independent and identically distributed (i.i.d.) returns, failing to account for **volatility clustering**. Filtered Historical Simulation (Hull & White, 1998) scales historical shocks by current volatility dynamics.

1. Fit univariate $\text{GARCH}(1,1)$ models to each asset $i$:
   $$r_{i, t} = \sigma_{i, t} \epsilon_{i, t}, \quad \sigma_{i, t}^2 = \omega_i + \alpha_i r_{i, t-1}^2 + \beta_i \sigma_{i, t-1}^2$$
2. Extract the standardized empirical residuals:
   $$\hat{\epsilon}_{i, t} = \frac{r_{i, t}}{\hat{\sigma}_{i, t}}$$
3. Obtain the forecast conditional volatility for tomorrow $T+1$: $\hat{\sigma}_{i, T+1}$.
4. Generate forward return scenarios by bootstrapping standardized residuals:
   $$r_{i, \tau}^* = \hat{\sigma}_{i, T+1} \cdot \hat{\epsilon}_{i, t^*}, \quad t^* \sim \text{Uniform}(\{1, \dots, T\})$$
   *(Note: Draw identical time indices $t^*$ across all assets simultaneously to preserve empirical copula/cross-asset dependence).*
5. Calculate the bootstrapped portfolio return $R_{p, \tau}^* = w^T r_\tau^*$, and apply quantiles to obtain $\text{VaR}$ and $\text{ES}$.

---

## 5. Integrating High-Frequency Estimators into the Covariance Engine

High-frequency realized volatility models (e.g., Yang-Zhang) estimate univariate instantaneous variance $\sigma_i^2$, capturing intraday drift, jumps, and open gaps. However, they do not provide cross-asset covariances.

### 5.1 Covariance Decomposition: $\Sigma = D R D$
We assemble the portfolio covariance matrix $\Sigma \in \mathbb{R}^{N \times N}$ via:
$$\Sigma = D \, R \, D$$
where:
- $D = \text{diag}(\sigma_1^{\text{YZ}}, \sigma_2^{\text{YZ}}, \dots, \sigma_N^{\text{YZ}})$ is the diagonal matrix of localized annualized standard deviations from the Yang-Zhang estimator.
- $R \in \mathbb{R}^{N \times N}$ is the asset correlation matrix computed from multi-day synchronous log-returns.

### 5.2 Ledoit-Wolf Shrinkage for Correlation
Sample correlation matrices $S$ suffer from sample noise and rank deficiency when $N \approx T$. We compute a **Ledoit-Wolf shrinkage** estimator toward a structured constant-correlation target $F$:
$$R^* = \delta F + (1 - \delta) S, \quad \delta \in [0, 1]$$
where the average correlation is:
$$\bar{\rho} = \frac{2}{N(N-1)} \sum_{i < j} S_{i, j}$$
and $F_{i, i} = 1$, $F_{i, j} = \bar{\rho}$ for $i \ne j$. The shrinkage intensity $\delta$ minimizes the expected Frobenius loss $\mathbb{E}[\|R^* - \Sigma_{\text{true}}\|_F^2]$.

### 5.3 Numerical Positive Semi-Definiteness (PSD) Guarantee
To guarantee that $\Sigma$ is symmetric positive semi-definite (SPSD) and avoid negative eigenvalues caused by asynchronous sampling or numerical errors:
1. Perform an eigendecomposition: $\Sigma = V \Lambda V^T$, where $\Lambda = \text{diag}(\lambda_1, \dots, \lambda_N)$.
2. Threshold eigenvalues: $\tilde{\lambda}_i = \max(\lambda_i, \epsilon_{\text{floor}})$, with $\epsilon_{\text{floor}} = 10^{-8}$.
3. Reconstruct: $\tilde{\Sigma} = V \tilde{\Lambda} V^T$.

---

## 6. Risk Attribution & Euler's Allocation Principle

To attribute portfolio risk to individual assets, we use Euler's homogeneous function theorem.

### 6.1 Euler's Decomposition Theorem
A risk measure $\rho: \mathbb{R}^N \to \mathbb{R}$ that is **positively homogeneous of degree 1** satisfies:
$$\forall \lambda > 0, \quad \rho(\lambda w) = \lambda \rho(w)$$

Differentiating both sides with respect to $\lambda$ and setting $\lambda = 1$:
$$\rho(w) = \sum_{i=1}^N w_i \frac{\partial \rho(w)}{\partial w_i}$$

We define:
- **Marginal Risk Contribution ($\text{MRC}_i$):**
  $$\text{MRC}_i = \frac{\partial \rho(w)}{\partial w_i}$$
- **Component Risk Contribution ($\text{CRC}_i$):**
  $$\text{CRC}_i = w_i \frac{\partial \rho(w)}{\partial w_i}$$
- **Percentage Risk Contribution ($\% \text{RC}_i$):**
  $$\% \text{RC}_i = \frac{\text{CRC}_i}{\rho(w)}, \quad \sum_{i=1}^N \% \text{RC}_i = 1$$

---

### 6.2 Volatility Decomposition
Let $\sigma_p(w) = \sqrt{w^T \Sigma w}$. The gradient with respect to $w$ is:
$$\nabla_w \sigma_p = \frac{\Sigma w}{\sqrt{w^T \Sigma w}} = \frac{\Sigma w}{\sigma_p}$$
Thus, the Marginal Volatility is:
$$\text{MVol}_i = \frac{(\Sigma w)_i}{\sigma_p} = \text{Cov}(R_i, R_p) \cdot \frac{1}{\sigma_p} = \beta_i^p \sigma_p$$
where $\beta_i^p = \frac{\text{Cov}(R_i, R_p)}{\sigma_p^2}$ is the asset's beta relative to the portfolio.
The component volatility is:
$$\text{CVol}_i = w_i \text{MVol}_i \implies \sum_{i=1}^N \text{CVol}_i = \sum_{i=1}^N w_i \frac{(\Sigma w)_i}{\sigma_p} = \frac{w^T \Sigma w}{\sigma_p} = \sigma_p$$

---

### 6.3 Marginal and Component $\text{VaR}$
For normally distributed returns $R \sim \mathcal{N}(\mu, \Sigma)$, using $\text{VaR}_\alpha(w) = -w^T \mu + z_\alpha \sqrt{w^T \Sigma w}$:
$$\text{MVaR}_i = \frac{\partial \text{VaR}_\alpha}{\partial w_i} = -\mu_i + z_\alpha \frac{(\Sigma w)_i}{\sigma_p}$$
$$\text{CVaR}_i = w_i \text{MVaR}_i$$

For historical simulation, Scaillet (2004) showed that using Gourieroux's kernel smoothing:
$$\frac{\partial \text{VaR}_\alpha}{\partial w_i} = -\mathbb{E}[R_i \mid R_p = -\text{VaR}_\alpha(R_p)]$$

---

### 6.4 Marginal and Component $\text{ES}$
Using the definition $\text{ES}_\alpha(w) = -\mathbb{E}[R_p \mid R_p \le -\text{VaR}_\alpha]$:
$$\text{MES}_i = \frac{\partial \text{ES}_\alpha}{\partial w_i} = -\mathbb{E}[R_i \mid R_p \le -\text{VaR}_\alpha]$$
Component $\text{ES}$ represents the expected loss of position $i$ during tail events of the overall portfolio:
$$\text{CES}_i = -w_i \mathbb{E}[R_i \mid R_p \le -\text{VaR}_\alpha]$$
Notice the additivity:
$$\sum_{i=1}^N \text{CES}_i = -\mathbb{E}\left[ \sum_{i=1}^N w_i R_i \,\middle|\, R_p \le -\text{VaR}_\alpha \right] = -\mathbb{E}[R_p \mid R_p \le -\text{VaR}_\alpha] = \text{ES}_\alpha(w)$$

---

## 7. Factor Models & Systematic vs. Idiosyncratic Risk

### 7.1 Single-Index Model (Capital Asset Pricing Model representation)
Let $R_m$ be the benchmark market return with variance $\sigma_m^2$. Decompose asset returns as:
$$R_{i, t} = \alpha_i + \beta_i R_{m, t} + \epsilon_{i, t}$$
under the standard Gauss-Markov assumptions:
$$\mathbb{E}[\epsilon_i] = 0, \quad \text{Cov}(\epsilon_i, R_m) = 0, \quad \text{Cov}(\epsilon_i, \epsilon_j) = 0 \; (\forall i \ne j)$$

The asset regression beta is:
$$\beta_i = \frac{\text{Cov}(R_i, R_m)}{\sigma_m^2}$$

### 7.2 Portfolio Variance Decomposition
The portfolio return is:
$$R_p = \sum_{i=1}^N w_i (\alpha_i + \beta_i R_m + \epsilon_i) = \alpha_p + \beta_p R_m + \epsilon_p$$
where:
$$\beta_p = \sum_{i=1}^N w_i \beta_i = w^T \beta, \quad \alpha_p = w^T \alpha, \quad \epsilon_p = w^T \epsilon$$

Because $\text{Cov}(R_m, \epsilon_p) = 0$, the total portfolio variance decomposes orthogonally into:
$$\sigma_p^2 = \text{Var}(R_p) = \underbrace{\beta_p^2 \sigma_m^2}_{\text{Systematic Variance}} + \underbrace{\sum_{i=1}^N w_i^2 \sigma_{\epsilon_i}^2}_{\text{Idiosyncratic Variance}}$$
where $\sigma_{\epsilon_i}^2 = \text{Var}(\epsilon_i)$ is the residual variance of asset $i$.

---

## 8. Beta-Neutral Hedging via Convex Optimization

A quantitative equity market-neutral (QEMN) strategy eliminates systematic directional market exposure ($\beta_p = 0$) while minimizing total portfolio variance or idiosyncratic tracking error.

### 8.1 Mathematical Optimization Formulation
Given an existing asset allocation or alpha vector $\alpha \in \mathbb{R}^N$, a covariance matrix $\Sigma \in \mathbb{S}_{++}^N$, and asset betas $\beta \in \mathbb{R}^N$:

$$\min_{w \in \mathbb{R}^N} \quad \frac{1}{2} w^T \Sigma w - \lambda \alpha^T w$$

Subject to:
1. **Beta Neutrality:**
   $$\beta^T w = 0$$
2. **Gross or Net Exposure:**
   $$\mathbf{1}^T w = 0 \quad (\text{Dollar Neutral}) \quad \text{or} \quad \mathbf{1}^T w = 1 \quad (\text{Fully Invested})$$
   $$\|w\|_1 \le L_{\max} \quad (\text{Leverage / Gross Exposure Bound})$$
3. **Position Bounds:**
   $$w_{\min} \le w_i \le w_{\max}, \quad \forall i \in \{1, \dots, N\}$$

### 8.2 Analytical Solution (Lagrangian Formulation)
Consider the case without inequality bounds:
$$\min_w \frac{1}{2} w^T \Sigma w \quad \text{s.t.} \quad C^T w = b$$
where $C = [\beta, \mathbf{1}] \in \mathbb{R}^{N \times 2}$ and $b = [0, 1]^T \in \mathbb{R}^2$.

The Lagrangian is:
$$\mathcal{L}(w, \lambda) = \frac{1}{2} w^T \Sigma w - \lambda^T (C^T w - b)$$

The Karush-Kuhn-Tucker (KKT) first-order optimality condition is:
$$\nabla_w \mathcal{L} = \Sigma w - C \lambda = 0 \implies w = \Sigma^{-1} C \lambda$$

Substitute $w$ into the constraint $C^T w = b$:
$$C^T \Sigma^{-1} C \lambda = b \implies \lambda = (C^T \Sigma^{-1} C)^{-1} b$$

Thus, the exact analytical optimal beta-neutral minimum-variance weights are:
$$w^* = \Sigma^{-1} C (C^T \Sigma^{-1} C)^{-1} b$$

Because $\Sigma \succ 0$, $\Sigma^{-1}$ is symmetric positive definite, ensuring $C^T \Sigma^{-1} C$ is invertible and the objective function is strictly convex, which guarantees a unique global minimum.
