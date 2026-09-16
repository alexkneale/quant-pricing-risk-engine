import numpy as np
from scipy.optimize import minimize
from typing import Tuple, Dict, Any
from scipy.special import factorial
from scipy import stats


def simulate_gbm(
    S0: float, mu: float, sigma: float, T: float, dt: float, n_paths: int, seed: int = 42
) -> np.ndarray:
    """Vectorized simulation of Geometric Brownian Motion paths.
    Returns array of shape (n_steps + 1, n_paths).
    """
    np.random.seed(seed)
    n_steps = int(T / dt)
    
    # Pre-generate standard normal innovations
    Z = np.random.normal(0, 1, size=(n_steps, n_paths))
    
    # Compute log increments
    drift_term = (mu - 0.5 * sigma**2) * dt
    diffusion_term = sigma * np.sqrt(dt) * Z
    log_increments = drift_term + diffusion_term
    
    # Cumulative sum to get price paths
    log_paths = np.vstack([np.zeros((1, n_paths)), np.cumsum(log_increments, axis=0)])
    price_paths = S0 * np.exp(log_paths)
    
    return price_paths


def simulate_merton_jump(
    S0: float,
    mu: float,
    sigma: float,
    lam: float,
    mu_J: float,
    sigma_J: float,
    T: float,
    dt: float,
    n_paths: int,
    seed: int = 42
) -> np.ndarray:
    """Vectorized simulation of Merton Jump-Diffusion paths.
    Returns array of shape (n_steps + 1, n_paths).
    """
    np.random.seed(seed)
    n_steps = int(T / dt)
    
    # Jump compensator k = E[Y - 1]
    k = np.exp(mu_J + 0.5 * sigma_J**2) - 1.0
    
    # 1. Continuous diffusion part
    Z = np.random.normal(0, 1, size=(n_steps, n_paths))
    drift = (mu - lam * k - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt) * Z
    
    # 2. Compound Poisson Jump part
    # Number of jumps in each step for each path
    jump_counts = np.random.poisson(lam * dt, size=(n_steps, n_paths))
    
    # Generate aggregate jump size where jumps occur
    # For small dt, jump_counts > 1 is rare, but we handle arbitrary counts
    jump_magnitudes = np.zeros((n_steps, n_paths))
    total_jumps = np.sum(jump_counts)
    if total_jumps > 0:
        # Standard deviation scales with sqrt(N_jumps), mean scales with N_jumps
        non_zero_mask = jump_counts > 0
        counts = jump_counts[non_zero_mask]
        jump_magnitudes[non_zero_mask] = np.random.normal(
            loc=counts * mu_J,
            scale=np.sqrt(counts) * sigma_J
        )

    log_increments = drift + diffusion + jump_magnitudes
    log_paths = np.vstack([np.zeros((1, n_paths)), np.cumsum(log_increments, axis=0)])
    price_paths = S0 * np.exp(log_paths)

    return price_paths


def simulate_garch(
    omega: float,
    alpha: float,
    beta: float,
    mu: float,
    n_steps: int,
    n_paths: int,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulates GARCH(1,1) log-returns and conditional variance paths.
    Returns: (returns, variances) of shape (n_steps, n_paths).
    """
    np.random.seed(seed)
    
    # Initialize long-run unconditional variance
    long_run_var = omega / (1.0 - alpha - beta)
    
    returns = np.zeros((n_steps, n_paths))
    variances = np.zeros((n_steps, n_paths))
    
    # Initial step
    variances[0] = long_run_var
    Z = np.random.normal(0, 1, size=(n_steps, n_paths))
    returns[0] = mu + np.sqrt(variances[0]) * Z[0]
    
    for t in range(1, n_steps):
        eps_prev = returns[t - 1] - mu
        # GARCH conditional variance update: omega + alpha * eps^2 + beta * sigma^2
        variances[t] = omega + alpha * (eps_prev**2) + beta * variances[t - 1]
        returns[t] = mu + np.sqrt(variances[t]) * Z[t]
        
    return returns, variances


def fit_garch_mle(returns: np.ndarray) -> Dict[str, float]:
    """Fits GARCH(1,1) parameters (omega, alpha, beta, mu) via Maximum Likelihood Estimation (MLE).
    returns: 1D array of historical log-returns.
    """
    T = len(returns)
    sample_var = np.var(returns)

    def neg_log_likelihood(params: np.ndarray) -> float:
        mu, omega, alpha, beta = params
        
        # Enforce stationarity and positivity during optimization
        if omega <= 1e-8 or alpha < 0.0 or beta < 0.0 or (alpha + beta) >= 0.9999:
            return 1e10

        eps = returns - mu
        sigma2 = np.zeros(T)
        sigma2[0] = sample_var
        
        for t in range(1, T):
            sigma2[t] = omega + alpha * (eps[t - 1]**2) + beta * sigma2[t - 1]
            if sigma2[t] <= 1e-8:
                return 1e10

        # Negative log-likelihood of Gaussian innovations
        ll = -0.5 * np.sum(np.log(2 * np.pi) + np.log(sigma2) + (eps**2) / sigma2)
        return -ll

    # Initial guesses: mu=mean, alpha=0.08, beta=0.88, omega=var*(1 - alpha - beta)
    init_mu = np.mean(returns)
    init_alpha = 0.08
    init_beta = 0.88
    init_omega = sample_var * (1.0 - init_alpha - init_beta)
    
    initial_guess = np.array([init_mu, init_omega, init_alpha, init_beta])
    
    bounds = [
        (-1.0, 1.0),      # mu
        (1e-8, None),     # omega
        (0.0, 1.0),       # alpha
        (0.0, 1.0)        # beta
    ]
    
    result = minimize(
        neg_log_likelihood,
        initial_guess,
        method='L-BFGS-B',
        bounds=bounds
    )

    if not result.success:
        raise RuntimeError(f"GARCH MLE failed to converge: {result.message}")

    mu_opt, omega_opt, alpha_opt, beta_opt = result.x
    return {
        "mu": mu_opt,
        "omega": omega_opt,
        "alpha": alpha_opt,
        "beta": beta_opt,
        "half_life": np.log(0.5) / np.log(alpha_opt + beta_opt)  # Half-life of a volatility shock in days
    }

def fit_merton_mle(returns: np.ndarray, dt: float = 1/252, max_jumps: int = 10) -> dict:
    """
    Fits Merton Jump-Diffusion parameters (mu, sigma, lam, mu_J, sigma_J) 
    via Maximum Likelihood Estimation on historical log-returns.
    """
    clean_r = returns[~np.isnan(returns)]
    T = len(clean_r)
    sample_mean = np.mean(clean_r)
    sample_std = np.std(clean_r, ddof=1)
    
    # Precompute factorials for jump probabilities
    j_arr = np.arange(max_jumps + 1)
    fact_j = factorial(j_arr)

    def neg_log_likelihood(params):
        mu, sigma, lam, mu_J, sigma_J = params
        
        # Enforce strict positive variance and jump intensity
        if sigma <= 1e-6 or lam < 1e-6 or sigma_J <= 1e-6:
            return 1e10
            
        k = np.exp(mu_J + 0.5 * sigma_J**2) - 1.0
        
        # Conditional parameters for j = 0, ..., max_jumps
        # mu_j shape: (max_jumps + 1,)
        mu_j = (mu - lam * k - 0.5 * sigma**2) * dt + j_arr * mu_J
        var_j = sigma**2 * dt + j_arr * (sigma_J**2)
        std_j = np.sqrt(var_j)
        
        # Poisson probabilities P(N = j)
        pois_prob = np.exp(-lam * dt) * ((lam * dt) ** j_arr) / fact_j
        
        # Compute mixture likelihood matrix: shape (T, max_jumps + 1)
        r_expanded = clean_r[:, np.newaxis]  # (T, 1)
        z = (r_expanded - mu_j) / std_j
        densities = np.exp(-0.5 * z**2) / (np.sqrt(2 * np.pi) * std_j)
        
        # Marginal density per observation
        marginal_pdf = np.dot(densities, pois_prob)
        
        # Safeguard against zero/negative densities
        marginal_pdf = np.where(marginal_pdf <= 1e-15, 1e-15, marginal_pdf)
        
        return -np.sum(np.log(marginal_pdf))

    # Initial parameter guesses
    init_params = np.array([
        sample_mean / dt,                # Annualized drift mu
        sample_std * 0.7 / np.sqrt(dt),  # Continuous diffusion vol sigma
        5.0,                             # Jump intensity lambda (5 jumps/year)
        -0.02,                           # Mean jump size mu_J (-2%)
        sample_std * 1.5                 # Jump volatility sigma_J
    ])
    
    bounds = [
        (-2.0, 2.0),       # mu
        (1e-4, 2.0),       # sigma
        (1e-3, 50.0),      # lam
        (-0.5, 0.5),       # mu_J
        (1e-4, 1.0)        # sigma_J
    ]
    
    result = minimize(neg_log_likelihood, init_params, method='L-BFGS-B', bounds=bounds)
    
    if not result.success:
        raise RuntimeError(f"Merton MLE failed to converge: {result.message}")
        
    mu_opt, sig_opt, lam_opt, mu_J_opt, sig_J_opt = result.x
    return {
        "mu": mu_opt,
        "sigma": sig_opt,
        "lambda": lam_opt,
        "mu_J": mu_J_opt,
        "sigma_J": sig_J_opt,
        "implied_total_annual_vol": np.sqrt(sig_opt**2 + lam_opt * (mu_J_opt**2 + sig_J_opt**2))
    }