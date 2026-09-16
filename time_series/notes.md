## Volatility Metrics

Close-to-close volatility is defined as:
$\sigma_{close}^{2} = \frac{1}{N-1}\sum_{i=1}^{N}(r_i - \bar{r})^2$

Where 

$r_i = \ln(\frac{C_i}{C_{i-1}})$

$C_i$ denotes close price on day ${i}$


The Problem: Close-to-close volatility throws away everything that happened during the trading day between 09:30 and 16:00. If a stock swings up +5% and crashes down −5% before closing at 0%, close-to-close volatility records 0 variance.

To capture intraday variance without needing tick-by-tick data, we can use OHLC estimators:

Let:
- $o_t = \ln(\frac{O_t}{C_{t-1}})$ (normalised open/overnight jump)
- $u_t = \ln(\frac{H_t}{O_{t}})$ (normalised high relative to open)
- $d_t = \ln(\frac{L_t}{O_{t}})$ (normalised low relative to open)
- $c_t = \ln(\frac{C_t}{O_{t}})$ (normalised close relative to open)

Other metrics for volatility that we can use instead of close-to-close volatility include Parkinson Volatility (1980), Garman-Klass (1980), Yang-Zhang (2000). 

---

#### Parkinson

Uses only High and Low prices:

$\sigma_{P}^2 = \frac{1}{4\ln(2)}\frac{1}{N}\sum_{t=1}^{N}(\ln(\frac{H_t}{L_t}))^2$

Note how we require twice the number of data points here vs close-to-close volatility. It's roughly 5 times more efficient than close-to-close volatility (see note on efficiency of volatility estimate below for explanation of what this means).
Also, another major flaw is that it assumes continuous trading - ignores overnight jumps and opening jumps. 

---

#### Garman-Klass

Incorporates Open, High, Low and Close:

$\sigma_{GK}^2 = \frac{1}{N}\sum_{t=1}^{N}[0.5(\ln(\frac{H_t}{L_t}))^2 - (2\ln2 - 1)(\ln\frac{C_t}{O_t})^2]$

Roughly 7.4 times more efficient than close-to-close estimate. It has a major flaw that it assumes zero opening jump ($O_t = C_{t-1}$) in the derivation of the estimate. In equities, earnings releases and macro events happen overnight, creating large open gaps.

---

#### Yang-Zhang

$\sigma_{YZ}^2 = \sigma_{o}^2 + k\sigma_{c}^2 + (1 - k)\sigma_{RS}^2$

Where

- $\sigma_{o}^2 = \frac{1}{N-1}\sum_{t=1}^{N}(o_t - \bar{o})^2$ (Overnight variance)
- $\sigma_{c}^2 = \frac{1}{N-1}\sum_{t=1}^{N}(c_t - \bar{c})^2$ (Open-to-close variance)
- $\sigma_{RS}^2 = \frac{1}{N}[u_t(u_t-c_t) + d_t(d_t - c_t)]$ (Rogers-Satchell continuous variance - unbiased by upward or downward price trends)
- $k = \frac{0.34}{1.34 + \frac{N+1}{N-1}}$ (weight chosen to minimize the estimator's variance)

Yang-Zhang is the benchmark OHLC estimator in quant finance. It is unbiased, independent of continuous drift $\mu$, and accounts for both overnight jumps and intraday continuous diffusion.

---

#### Note on Efficiency of volatility estimate

What does it mean when we say one volatility estimate is $x$ times more efficient than another volatility estimate?

In statistical estimation theory, the **Relative Efficiency** of an unbiased estimator $\hat{\theta}_A$ relative to a benchmark unbiased estimator $\hat{\theta}_B$ is defined as the ratio of their sampling variances:

$$\text{Eff}(\hat{\theta}_A, \hat{\theta}_B) = \frac{\text{Var}(\hat{\theta}_B)}{\text{Var}(\hat{\theta}_A)}$$

When we state that an OHLC volatility estimator is $x$ times more efficient than the standard Close-to-Close estimator ($\sigma_{close}^2$):

1. **Variance Reduction:** The variance of the estimation error for the OHLC estimator is only $\frac{1}{x}$ of the Close-to-Close variance.
2. **Sample Size Equivalence:** To achieve the same estimation precision (standard error / confidence interval width) as $N$ days of OHLC data, a Close-to-Close estimator requires $x \times N$ days of data.
   - For example, if $x = 7.4$ (Garman-Klass), $1\text{ month}$ (21 trading days) of OHLC data provides the same statistical precision as $7.4 \times 21 \approx 155\text{ trading days}$ (over 7 months) of Close-to-Close prices.

---

#### Mathematical Derivation of $x$

Assume an underlying asset price $S_t$ follows a continuous driftless Geometric Brownian Motion:

$$d\ln S_t = \sigma dW_t$$

Let:
- $u = \ln(H / O)$
- $d = \ln(L / O)$
- $c = \ln(C / O)$
- $w = u - d = \ln(H / L)$ (Normalized high-low range)

##### Close-to-Close Estimator Benchmark ($\hat{\sigma}^2_{CC}$)
For daily log-return $r = \ln(C_t / C_{t-1}) \sim \mathcal{N}(0, \sigma^2)$:

$$\hat{\sigma}^2_{CC} = r^2$$

Since $\frac{r}{\sigma} \sim \mathcal{N}(0, 1)$, its square follows a chi-squared distribution with 1 degree of freedom: $\left(\frac{r}{\sigma}\right)^2 \sim \chi^2(1)$.

$$\text{Var}(\hat{\sigma}^2_{CC}) = \text{Var}(r^2) = \sigma^4 \text{Var}\left(\left(\frac{r}{\sigma}\right)^2\right) = 2\sigma^4$$

---

##### Derivation for Parkinson (1980): $x \approx 5.0$
Parkinson applied the reflection principle of standard Brownian motion to determine the probability density function $f(w)$ of the price range $w = \ln(H/L)$ over a unit time interval:

$$f(w) = 4 \sum_{k=1}^\infty (-1)^{k-1} k \left(\frac{2}{\sqrt{2\pi}\sigma}\right) \exp\left(-\frac{k^2 w^2}{2\sigma^2}\right)$$

Evaluating the first two non-central moments of $w^2$:
- $\mathbb{E}[w^2] = 4\ln(2)\sigma^2 \implies \hat{\sigma}_P^2 = \frac{w^2}{4\ln(2)}$ is an unbiased estimator of $\sigma^2$.
- $\text{Var}(\hat{\sigma}_P^2) = \frac{\mathbb{E}[w^4] - (\mathbb{E}[w^2])^2}{(4\ln 2)^2} \approx 0.4073 \sigma^4$

Calculating relative efficiency:

$$\text{Eff}(\hat{\sigma}_P^2, \hat{\sigma}_{CC}^2) = \frac{\text{Var}(\hat{\sigma}_{CC}^2)}{\text{Var}(\hat{\sigma}_P^2)} = \frac{2.0 \sigma^4}{0.4073 \sigma^4} \approx 4.91 \approx \mathbf{5.0}$$

---

##### Derivation for Garman-Klass (1980): $x \approx 7.4$
Garman and Klass sought the optimal linear combination of range variance $w^2 = (u - d)^2$ and close variance $c^2$:

$$\hat{\sigma}_{GK}^2 = a_1 (u - d)^2 - a_2 c^2$$

Minimizing the estimation variance $\text{Var}(\hat{\sigma}_{GK}^2)$ subject to the unbiasedness constraint $\mathbb{E}[\hat{\sigma}_{GK}^2] = \sigma^2$ yields the optimal coefficients:
- $a_1 = 0.5$
- $a_2 = 2\ln(2) - 1 \approx 0.3863$

The minimum variance is:

$$\text{Var}(\hat{\sigma}_{GK}^2) \approx 0.270 \sigma^4$$

Calculating relative efficiency:

$$\text{Eff}(\hat{\sigma}_{GK}^2, \hat{\sigma}_{CC}^2) = \frac{\text{Var}(\hat{\sigma}_{CC}^2)}{\text{Var}(\hat{\sigma}_{GK}^2)} = \frac{2.0 \sigma^4}{0.270 \sigma^4} \approx \mathbf{7.4}$$

---


## Models for Volatility

We will look at three models: Geometric Brownian Motion (GBM), Merton Jump-Diffusion (1976) and GARCH(1,1) (Generalized Autoregressive Conditional Heteroskedasticity)

#### Geometric Brownian Motion (GBM)

Underlying dynamic of timeseries $\{ dS_t \}$ given by:

$dS_t = \mu S_t dt + \sigma S_t dW_t$

Where $\mu$ gives the mean return (per $dt$), $\sigma$ is the volatility of the security's price, and $dW_t$ is the infinitesimal increment of a standard Brownian Motion process/ 

By Itô's lemma, difining $x_t = \ln S_t$:

$dx_t = (\mu - \frac{1}{2}\sigma^2)dt + \sigma dW_t$

Above has exact discrete solution over time step $\delta t$ given by:

$S_{t+\delta t} = S_t exp((\mu - \frac{1}{2}\sigma^2)\delta t + \sigma \sqrt{\delta t}Z)$

Where 

$Z \sim \mathcal{N}(0, 1)$

#### Merton Jump-Diffusion


GBM cannot account for sudden market shocks (e.g., flash crashes, earnings gaps). Merton added a compound Poisson jump process:

$dS_t = (\mu - \lambda k)S_t dt + \sigma S_t dW_t + (Y - 1) S_t dN_t$

where:

- $N_t \sim Poisson(\lambda t)$: Poisson process with jump intensity $\lambda$ (expected number of jumps per unit time).

- $Y$: Jump multiplier. When a jump occurs, $S_t \to S_t - Y$

$\ln(Y) \sim \mathcal{N}(\mu_J, \sigma_J^2)$

- $k = \mathbb{E}[Y-1] = exp(\mu_J + \frac{1}{2}\sigma_J^2) - 1$: The compensator drift term, ensuring that the expected rate of return remains $\mu$

Exact discrete simulation step:

For an interval $\delta t$, the number of jumps $N_{\delta t}$ is given by $N_{\delta t} \sim Poisson(\lambda \delta t)$. If $N_{\delta t} = m$, then we have that:

$\ln (\frac{S_{t+\delta t}}{S_t}) = (\mu - \lambda k - \frac{1}{2} \sigma^2) \delta t + \sigma \sqrt{\delta t} Z + \sum_{j=1}^{m}J_j $

Where $J_j \sim \mathcal{N}(\mu_J, \sigma_J^2)$

#### GARCH(1,1) 

Financial volatility is not constant; it exhibits heteroskedasticity (variance varies over time) and clustering (large changes follow large changes, regardless of sign).

Bollerslev (1986) formulated the discrete-time return dynamic:

$r_t = \mu + \epsilon_t$

$\epsilon_t = \sigma_t z_t$

$z_t \sim i.i.d. \mathcal{N}(0,1)$

$\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2$

The above dynamics are stationary if $\alpha + \beta < 1$. 

The long-run unconditional variance is:

$\sigma_L^2 = \frac{\omega}{1- \alpha - \beta}$

$\alpha$ (ARCH parameter): reaction to immediate market shocks ($\epsilon_{t-1}^2$)

$\beta$ (GARCH parameter): persistence of volatility memory ($\sigma_{t-1}^2$)

Calibrate values of $\alpha$ and $\beta$ via Max Likelihood Estimation (MLE):

Given historical return series $\{r_1, r_2, ..., r_T\}$, the conditional distribution $r_t | \mathcal{F}_{t-1} \sim \mathcal{N}(\mu, \sigma_t^2)$

Then have to maximize the log likelihood given by:

$\ln \mathcal{L}(\omega, \alpha, \beta, \mu) = - \frac{1}{2} \sum_{t=1}^T(\ln(2\pi) + \ln(\sigma_t^2) + \frac{(r_t-\mu)^2}{\sigma_t^2})$

Subject to bounds $\omega > 0$, $\alpha \geq 0$, $\beta \geq 0$ and constraints $\alpha + \beta < 1$

## Stylized Facts of Asset Returns

The Black-Scholes-Merton and standard Geometric Brownian Motion (GBM) models rely on three core assumptions:
1. Returns are normally distributed ($r_t \sim \mathcal{N}$).
2. Returns across disjoint intervals are independent and identically distributed ($i.i.d.$).
3. Instantaneous volatility $\sigma$ is constant.

In reality, empirical financial time series systematically violate every single one of these assumptions. The statistical properties that consistently emerge across asset classes, markets, and time frames are known as **Stylized Facts** (Cont, 2001).

Our statistical battery (`stylized_facts.py`) tests for four key empirical stylized facts:

1. **Absence of Autocorrelation in Raw Returns** (Weak-Form Market Efficiency)
2. **Heavy Tails / Leptokurtosis** (Fat Tails)
3. **Volatility Clustering** (Non-linear Autoregressive Variance)
4. **The Leverage Effect** (Asymmetric Volatility Response)

---

#### 1. Absence of Autocorrelation in Raw Returns

Price log-returns exhibit negligible linear autocorrelation at daily or lower frequencies:

$\text{Corr}(r_t, r_{t+\tau}) \approx 0 \quad \forall \ \tau \ge 1$

This is consistent with the Martingale hypothesis of asset prices under weak-form market efficiency:

$\mathbb{E}[r_t \mid \mathcal{F}_{t-1}] = 0$

Where $\mathcal{F}_{t-1}$ is the filtration (information set) up to day $t-1$. If returns had predictable linear persistence, simple momentum or mean-reversion filters would yield riskless excess returns.

**Statistical Test: Ljung-Box on Raw Returns**

We evaluate the joint hypothesis that the first $m$ autocorrelations are zero:

$H_0: \rho_1 = \rho_2 = \dots = \rho_m = 0$

$Q_{\text{LB}} = T(T + 2) \sum_{k=1}^m \frac{\hat{\rho}_k^2}{T - k} \sim \chi^2(m)$

For raw daily equity returns, $p > 0.05$. We fail to reject $H_0$, confirming that raw returns resemble white noise.

---

#### 2. Heavy Tails and Leptokurtosis

The unconditional distribution of daily log-returns exhibits fat tails and sharp peaked modes relative to a standard Gaussian distribution. Extreme deviations (e.g., $3\sigma$ or $4\sigma$ market crashes) occur orders of magnitude more frequently than predicted by standard normal statistics.

Let $\mu = \mathbb{E}[r_t]$ and $\sigma = \sqrt{\mathbb{E}[(r_t - \mu)^2]}$:

- **Skewness ($S$):** Measures distribution asymmetry.
  
  $S = \frac{\mathbb{E}[(r_t - \mu)^3]}{\sigma^3}$
  
  Equities typically exhibit negative skewness ($S < 0$) due to large market crashes.

- **Excess Kurtosis ($\kappa_{\text{excess}}$):** Measures tail weight and outlier propensity.
  
  $\kappa_{\text{excess}} = \frac{\mathbb{E}[(r_t - \mu)^4]}{\sigma^4} - 3$
  
  - Under Gaussian GBM: $\kappa_{\text{excess}} = 0$
  - Empirically for financial assets: $\kappa_{\text{excess}} > 1.0$ (often $3$ to $10+$)

Instead of the exponential tail decay of a normal distribution ($e^{-x^2}$), empirical returns decay via a power-law tail:

$P(|r_t| > x) \sim x^{-\alpha} \quad \text{as } x \to \infty, \quad \alpha \in [3, 5]$

**Statistical Test: Jarque-Bera Test**

Measures departure from normality using sample skewness $\hat{S}$ and sample kurtosis $\hat{K}$:

$JB = \frac{T}{6} \left( \hat{S}^2 + \frac{(\hat{K} - 3)^2}{4} \right) \sim \chi^2(2)$

For historical returns, $JB$ yields very large test statistics ($p < 0.001$), decisively rejecting the Gaussian distribution.

---

#### 3. Volatility Clustering

While raw returns $r_t$ are uncorrelated, non-linear transformations such as squared returns $(r_t - \mu)^2$ or absolute returns $|r_t|$ show strong, persistent, positive autocorrelation.

$\text{Corr}(r_t, r_{t+\tau}) \approx 0 \quad \not \implies \quad \text{Corr}(r_t^2, r_{t+\tau}^2) = 0$

Large price movements (of either sign) are followed by large price movements, and small movements are followed by small movements. 

**Statistical Test: Ljung-Box on Squared Returns (ARCH Effect Test)**

We run the Ljung-Box test on squared de-meaned returns $(r_t - \bar{r})^2$:

$Q_{\text{LB, sq}} = T(T + 2) \sum_{k=1}^m \frac{\hat{\rho}_{k, r^2}^2}{T - k} \sim \chi^2(m)$

Empirically, $p \ll 0.01$. The null hypothesis of independent increments is rejected, confirming the presence of conditional heteroskedasticity (ARCH/GARCH effects).

---

#### 4. The Leverage Effect

Volatility responds asymmetrically to price movements. A negative return (price drop) generates a larger increase in future volatility than a positive return (rally) of the exact same magnitude.

**Economic Intuition:**
- **Financial Leverage (Black, 1976):** As a firm's equity drops, its debt-to-equity ratio increases, making the equity riskier and increasing its volatility.
- **Order Imbalance:** Panic selloffs cause rapid liquidity evaporation, increasing bid-ask spreads and volatility.

**Empirical Metric:**
We measure the cross-correlation between current return $r_t$ and future absolute return $|r_{t+1}|$:

$L = \text{Corr}(r_t, |r_{t+1}|)$

- Under GBM: $L = 0$ (independent increments).
- Empirically: $L < 0$ (statistically significant negative correlation).

---

## Model Comparison Against Stylized Facts

| Stylized Fact | Geometric Brownian Motion (GBM) | Merton Jump-Diffusion (MJD) | GARCH(1,1) |
| :--- | :--- | :--- | :--- |
| **Linear Independence ($\rho_r \approx 0$)** | Yes | Yes | Yes |
| **Fat Tails ($\kappa_{\text{excess}} > 0$)** | **No** ($\kappa = 0$) | **Yes** (via Poisson jumps) | **Yes** (via dynamic variance) |
| **Volatility Clustering ($\rho_{r^2} > 0$)** | **No** (constant $\sigma$) | **No** (Poisson arrivals are memoryless) | **Yes** (via ARMA variance structure) |
| **Leverage Effect ($L < 0$)** | **No** ($L = 0$) | Partial (if mean jump $\mu_J < 0$) | **No** (needs EGARCH / GJR-GARCH) |

---

#### Why GBM Fails
1. **Fails Fat Tails:** Because $r_t \sim \mathcal{N}$, the probability of an extreme move (e.g., Black Monday $-20\%$ drop) is on the order of $10^{-50}$, making it mathematically impossible in a GBM model, even though such crashes occur periodically in real markets.
2. **Fails Volatility Clustering:** Since $\sigma$ is constant, $\text{Corr}(r_t^2, r_{t+\tau}^2) = 0$ for all $\tau \ge 1$. The Autocorrelation Function (ACF) of squared returns is zero everywhere.

---

#### How Merton Jump-Diffusion Fixes Fat Tails
MJD superimposes discontinuous Poisson jumps onto continuous Brownian paths:

$\ln\left(\frac{S_{t+\delta t}}{S_t}\right) = \left(\mu - \lambda k - \frac{1}{2}\sigma^2\right)\delta t + \sigma\sqrt{\delta t}Z + \sum_{j=1}^{N_{\delta t}} J_j$

- Over discrete intervals, the return distribution is a **Poisson-weighted mixture of normals**:
  
  $f(r) = \sum_{m=0}^\infty P(N_{\delta t} = m) \cdot \mathcal{N}\left(r; \ \mu_m, \sigma_m^2\right)$

- A mixture of Gaussians with differing variances produces positive excess kurtosis ($\kappa_{\text{excess}} > 0$).
- On a **Normal QQ-Plot**, GBM produces a straight diagonal line ($45^\circ$), failing to match market outliers. MJD produces an **S-shaped curve**, matching the fat tails of empirical market returns.
- **Remaining Limitation:** Jumps follow a Poisson process, meaning jump arrivals are independent and memoryless ($T \sim \text{Exp}(\lambda)$). Hence, MJD cannot produce volatility clustering.

---

#### How GARCH(1,1) Fixes Volatility Clustering and Induces Fat Tails

**1. Mechanism for Volatility Clustering:**
Rewriting the GARCH(1,1) variance equation using the zero-mean shock $\nu_t = \epsilon_t^2 - \sigma_t^2$:

$\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2$

$\epsilon_t^2 = \omega + (\alpha + \beta)\epsilon_{t-1}^2 + \nu_t - \beta \nu_{t-1}$

This shows that $\epsilon_t^2$ follows an **ARMA(1,1) process**. The ACF of squared returns decays at the persistence rate $(\alpha + \beta)$:

$\rho_{\epsilon^2}(\tau) \propto (\alpha + \beta)^\tau$

Because $\alpha + \beta \approx 0.95 - 0.99$ for daily financial data, the ACF of squared returns decays slowly over long lags, matching the volatility clustering observed in real data.

**2. Mechanism for Fat Tails:**
Even with standard normal innovations $z_t \sim \mathcal{N}(0, 1)$ ($\mathbb{E}[z_t^4] = 3$), the unconditional distribution of $\epsilon_t$ has excess kurtosis:

$\mathbb{E}[\epsilon_t^4] = \mathbb{E}[\sigma_t^4 z_t^4] = 3 \mathbb{E}[\sigma_t^4]$

By Jensen’s inequality, since $g(x) = x^2$ is convex:

$\mathbb{E}[\sigma_t^4] = \mathbb{E}[(\sigma_t^2)^2] > (\mathbb{E}[\sigma_t^2])^2$

$\text{Kurtosis}(\epsilon_t) = \frac{\mathbb{E}[\epsilon_t^4]}{(\mathbb{E}[\epsilon_t^2])^2} = 3 \frac{\mathbb{E}[\sigma_t^4]}{(\mathbb{E}[\sigma_t^2])^2} = 3 \left(1 + \frac{\text{Var}(\sigma_t^2)}{(\mathbb{E}[\sigma_t^2])^2}\right) > 3$

Solving explicitly for the GARCH(1,1) process:

$\kappa_{\text{excess}} = \frac{6\alpha^2}{1 - \beta^2 - 2\alpha\beta - 3\alpha^2} > 0$

Thus, time variation in conditional variance automatically generates fat tails in the unconditional return distribution.
