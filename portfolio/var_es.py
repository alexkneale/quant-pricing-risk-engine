"""
Description: Parametric, Historical, and Filtered Historical Value-at-Risk (VaR)
             and Expected Shortfall (ES) engines.
"""

from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
from scipy import stats
from arch import arch_model


class PortfolioRiskEngine:
    """
    Computes portfolio-level Value at Risk (VaR) and Expected Shortfall (ES).
    All outputs represent loss values (positive float values denote losses).
    """

    def __init__(self, confidence_level: float = 0.95):
        if not 0.0 < confidence_level < 1.0:
            raise ValueError("confidence_level must be between 0 and 1.")
        self.alpha = confidence_level

    # ----------------------------------------------------------------------
    # 1. Parametric Engine
    # ----------------------------------------------------------------------
    def parametric_risk(
        self,
        weights: np.ndarray,
        cov_matrix: np.ndarray,
        expected_returns: Optional[np.ndarray] = None,
        distribution: str = "gaussian",
        dof: Optional[float] = 5.0,
        skew: float = 0.0,
        kurt: float = 0.0,
    ) -> Dict[str, float]:
        """
        Compute parametric VaR and Expected Shortfall.

        Parameters:
        -----------
        weights : np.ndarray
            Portfolio weights, shape (N,).
        cov_matrix : np.ndarray
            Annualized or daily covariance matrix, shape (N, N).
        expected_returns : np.ndarray, optional
            Expected mean returns, shape (N,). Defaults to 0 if None.
        distribution : str
            One of: 'gaussian', 'student_t', 'cornish_fisher'.
        dof : float, optional
            Degrees of freedom for Student-t distribution (> 2).
        skew : float
            Empirical skewness of portfolio returns (for Cornish-Fisher).
        kurt : float
            Empirical excess kurtosis of portfolio returns (for Cornish-Fisher).
        """
        weights = np.asarray(weights)
        cov_matrix = np.asarray(cov_matrix)

        mu_p = 0.0 if expected_returns is None else float(np.dot(weights, expected_returns))
        sigma_p = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

        if distribution == "gaussian":
            z_alpha = stats.norm.ppf(self.alpha)
            phi_z = stats.norm.pdf(z_alpha)
            var = -mu_p + sigma_p * z_alpha
            es = -mu_p + sigma_p * (phi_z / (1.0 - self.alpha))

        elif distribution == "student_t":
            if dof is None or dof <= 2:
                raise ValueError("Student-t distribution requires dof > 2.")
            
            # Scale factor so that variance equals sigma_p^2
            scale = sigma_p * np.sqrt((dof - 2.0) / dof)
            t_inv = stats.t.ppf(self.alpha, df=dof)
            pdf_t = stats.t.pdf(t_inv, df=dof)

            var = -mu_p + scale * t_inv
            es = -mu_p + scale * (pdf_t / (1.0 - self.alpha)) * ((dof + t_inv**2) / (dof - 1.0))

        elif distribution == "cornish_fisher":
            z = stats.norm.ppf(self.alpha)
            # Cornish-Fisher quantile expansion for fat tails / skew
            z_cf = (
                z
                + (z**2 - 1.0) * skew / 6.0
                + (z**3 - 3.0 * z) * kurt / 24.0
                - (2.0 * z**3 - 5.0 * z) * (skew**2) / 36.0
            )
            var = -mu_p + sigma_p * z_cf
            # Analytical ES under CF approximated via Gaussian tail integration with z_cf
            phi_z_cf = stats.norm.pdf(z_cf)
            es = -mu_p + sigma_p * (phi_z_cf / (1.0 - self.alpha))

        else:
            raise ValueError(f"Unknown distribution: {distribution}")

        return {"VaR": float(var), "ES": float(es), "Portfolio_Vol": float(sigma_p)}

    # ----------------------------------------------------------------------
    # 2. Historical Simulation Engine
    # ----------------------------------------------------------------------
    def historical_risk(
        self,
        weights: np.ndarray,
        returns: pd.DataFrame,
    ) -> Dict[str, float]:
        """
        Non-parametric historical simulation.

        Parameters:
        -----------
        weights : np.ndarray
            Portfolio weights, shape (N,).
        returns : pd.DataFrame
            T x N matrix of synchronous asset returns.
        """
        w = np.asarray(weights)
        pnl_returns = returns.values @ w  # Array of length T

        # VaR is the empirical (1 - alpha) quantile of returns
        cutoff_quantile = 1.0 - self.alpha
        var = -np.percentile(pnl_returns, cutoff_quantile * 100.0)

        # ES is the mean of all returns falling beyond the VaR threshold
        tail_losses = pnl_returns[pnl_returns <= -var]
        es = -tail_losses.mean() if len(tail_losses) > 0 else var

        return {"VaR": float(var), "ES": float(es)}

    # ----------------------------------------------------------------------
    # 3. Filtered Historical Simulation (FHS) Engine
    # ----------------------------------------------------------------------
    def filtered_historical_simulation(
        self,
        weights: np.ndarray,
        returns: pd.DataFrame,
        n_scenarios: int = 5000,
        random_state: int = 42,
    ) -> Dict[str, float]:
        """
        Filtered Historical Simulation (Hull & White, 1998).
        Fits univariate GARCH(1,1) per asset, standardizes residuals,
        and bootstraps synchronous innovations scaled by current conditional volatility.
        """
        rng = np.random.default_rng(random_state)
        assets = returns.columns
        T, N = returns.shape

        standardized_residuals = np.zeros((T, N))
        next_day_vols = np.zeros(N)

        for i, col in enumerate(assets):
            # Scale by 100 for numerical stability during GARCH optimization
            series = returns[col] * 100.0
            am = arch_model(series, vol="Garch", p=1, q=1, mean="Constant", rescale=False)
            res = am.fit(disp="off", show_warning=False)

            cond_vol = res.conditional_volatility / 100.0
            # Standardized residuals: epsilon_t = r_t / sigma_t
            standardized_residuals[:, i] = (returns[col].values) / cond_vol.values

            # One-step-ahead conditional volatility forecast: sigma_{T+1}
            forecast = res.forecast(horizon=1)
            next_day_vols[i] = np.sqrt(forecast.variance.iloc[-1, 0]) / 100.0

        # Preserve empirical cross-asset copula by sampling the same time row index
        boot_idx = rng.choice(T, size=n_scenarios, replace=True)
        bootstrapped_innovations = standardized_residuals[boot_idx, :]  # (n_scenarios, N)

        # Rescale shocks by tomorrow's conditional volatility
        simulated_returns = bootstrapped_innovations * next_day_vols  # (n_scenarios, N)

        # Portfolio returns across scenarios
        sim_portfolio_returns = simulated_returns @ np.asarray(weights)

        cutoff = 1.0 - self.alpha
        var = -np.percentile(sim_portfolio_returns, cutoff * 100.0)
        tail_losses = sim_portfolio_returns[sim_portfolio_returns <= -var]
        es = -tail_losses.mean() if len(tail_losses) > 0 else var

        return {"VaR": float(var), "ES": float(es)}
