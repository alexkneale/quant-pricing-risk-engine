"""
Description: Euler's risk allocation, component VaR/ES, and systematic vs. idiosyncratic variance.
"""

from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.covariance import LedoitWolf


class RiskAttributionEngine:
    """
    Computes Euler risk allocation, beta decomposition, and covariance assembly.
    """

    @staticmethod
    def build_covariance_from_realized(
        realized_vols: np.ndarray,
        returns: pd.DataFrame,
        shrinkage: bool = True,
    ) -> np.ndarray:
        """
        Assemble Sigma = D @ R @ D where D contains instantaneous realized vols
        (e.g., Yang-Zhang) and R is the Ledoit-Wolf shrinkage correlation matrix.
        """
        realized_vols = np.asarray(realized_vols)
        D = np.diag(realized_vols)

        if shrinkage:
            lw = LedoitWolf()
            lw.fit(returns.values)
            sample_cov = lw.covariance_
            # Convert shrinkage covariance to correlation matrix R
            inv_std = 1.0 / np.sqrt(np.diag(sample_cov))
            R = sample_cov * np.outer(inv_std, inv_std)
        else:
            R = returns.corr().values

        cov = D @ R @ D

        # Guarantee positive semi-definiteness via spectral clipping
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.clip(eigvals, a_min=1e-8, a_max=None)
        cov_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
        return 0.5 * (cov_psd + cov_psd.T)

    @staticmethod
    def euler_volatility_attribution(
        weights: np.ndarray,
        cov_matrix: np.ndarray,
    ) -> pd.DataFrame:
        """
        Decomposes total portfolio volatility into marginal, component,
        and percentage contributions via Euler's theorem:
            sigma_p = sum( w_i * (Sigma @ w)_i / sigma_p )
        """
        w = np.asarray(weights)
        sigma_p = np.sqrt(w.T @ cov_matrix @ w)

        marginal_vol = (cov_matrix @ w) / sigma_p
        component_vol = w * marginal_vol
        pct_contrib = component_vol / sigma_p

        return pd.DataFrame({
            "Weight": w,
            "Marginal_Vol": marginal_vol,
            "Component_Vol": component_vol,
            "Pct_Contribution": pct_contrib,
        })

    @staticmethod
    def euler_parametric_var_es_attribution(
        weights: np.ndarray,
        cov_matrix: np.ndarray,
        alpha: float = 0.95,
    ) -> pd.DataFrame:
        """
        Parametric Gaussian Euler decomposition for VaR and ES.
        """
        w = np.asarray(weights)
        sigma_p = np.sqrt(w.T @ cov_matrix @ w)
        z_alpha = stats.norm.ppf(alpha)
        phi_z = stats.norm.pdf(z_alpha)

        # Marginal Volatility
        marginal_vol = (cov_matrix @ w) / sigma_p

        # VaR Contributions
        marginal_var = z_alpha * marginal_vol
        component_var = w * marginal_var
        var_total = z_alpha * sigma_p

        # ES Contributions
        es_factor = phi_z / (1.0 - alpha)
        marginal_es = es_factor * marginal_vol
        component_es = w * marginal_es
        es_total = es_factor * sigma_p

        return pd.DataFrame({
            "Weight": w,
            "Marginal_VaR": marginal_var,
            "Component_VaR": component_var,
            "Pct_VaR": component_var / var_total,
            "Marginal_ES": marginal_es,
            "Component_ES": component_es,
            "Pct_ES": component_es / es_total,
        })

    @staticmethod
    def historical_es_attribution(
        weights: np.ndarray,
        returns: pd.DataFrame,
        alpha: float = 0.95,
    ) -> pd.DataFrame:
        """
        Non-parametric historical Component ES decomposition:
            CES_i = -w_i * E[R_i | R_p <= -VaR_alpha]
        """
        w = np.asarray(weights)
        pnl = returns.values @ w
        cutoff = 1.0 - alpha
        var = -np.percentile(pnl, cutoff * 100.0)

        # Indicator of tail events
        tail_mask = pnl <= -var
        total_es = -pnl[tail_mask].mean()

        # Conditional expectation of each individual asset during the portfolio tail
        tail_returns = returns.values[tail_mask, :]
        marginal_tail_loss = -tail_returns.mean(axis=0)
        component_es = w * marginal_tail_loss

        return pd.DataFrame({
            "Weight": w,
            "Marginal_ES": marginal_tail_loss,
            "Component_ES": component_es,
            "Pct_ES": component_es / total_es,
        }, index=returns.columns)

    @staticmethod
    def factor_beta_decomposition(
        asset_returns: pd.DataFrame,
        market_returns: pd.Series,
        weights: np.ndarray,
    ) -> Dict[str, any]:
        """
        Decomposes total portfolio variance into Systematic and Idiosyncratic risk:
            sigma_p^2 = beta_p^2 * sigma_m^2 + sum(w_i^2 * sigma_epsilon_i^2)
        """
        w = np.asarray(weights)
        m_var = market_returns.var()
        m_mean = market_returns.mean()

        betas = []
        res_vars = []

        for col in asset_returns.columns:
            r = asset_returns[col]
            cov_im = np.cov(r, market_returns)[0, 1]
            b = cov_im / m_var
            betas.append(b)

            residuals = (r - r.mean()) - b * (market_returns - m_mean)
            res_vars.append(residuals.var())

        betas = np.array(betas)
        res_vars = np.array(res_vars)

        # Portfolio beta
        beta_p = float(np.dot(w, betas))
        systematic_var = (beta_p ** 2) * m_var
        idiosyncratic_var = float(np.dot(w**2, res_vars))
        total_modeled_var = systematic_var + idiosyncratic_var

        return {
            "Portfolio_Beta": beta_p,
            "Systematic_Variance": systematic_var,
            "Idiosyncratic_Variance": idiosyncratic_var,
            "Total_Modeled_Variance": total_modeled_var,
            "Pct_Systematic": systematic_var / total_modeled_var,
            "Pct_Idiosyncratic": idiosyncratic_var / total_modeled_var,
            "Asset_Betas": pd.Series(betas, index=asset_returns.columns),
            "Residual_Variances": pd.Series(res_vars, index=asset_returns.columns),
        }
