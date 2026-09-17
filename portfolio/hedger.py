"""
Description: Convex optimizer for beta-neutral minimum variance allocation.
"""

from typing import Optional, Tuple, Dict
import numpy as np
import pandas as pd
from scipy.optimize import minimize


class BetaNeutralHedger:
    """
    Constrained optimization solver for market-neutral and minimum-variance hedging.
    """

    def __init__(self, cov_matrix: np.ndarray, betas: np.ndarray):
        """
        Parameters:
        -----------
        cov_matrix : np.ndarray
            N x N asset covariance matrix.
        betas : np.ndarray
            N-dimensional array of asset betas to the market benchmark.
        """
        self.cov = np.asarray(cov_matrix)
        self.betas = np.asarray(betas)
        self.n_assets = len(betas)

        if self.cov.shape != (self.n_assets, self.n_assets):
            raise ValueError("Covariance matrix dimensions do not match betas length.")

    def optimize_beta_neutral(
        self,
        target_beta: float = 0.0,
        dollar_neutral: bool = True,
        weight_bounds: Tuple[float, float] = (-0.5, 0.5),
        expected_returns: Optional[np.ndarray] = None,
        risk_aversion: float = 0.0,
    ) -> Dict[str, any]:
        """
        Solve:
            min_w  (1/2) w^T Sigma w - lambda * mu^T w
        Subject to:
            beta^T w = target_beta
            sum(w) = 0 (if dollar_neutral) or 1 (if fully invested)
            w_min <= w_i <= w_max
        """
        def objective(w: np.ndarray) -> float:
            port_var = 0.5 * (w.T @ self.cov @ w)
            if expected_returns is not None and risk_aversion > 0.0:
                port_ret = np.dot(w, expected_returns)
                return port_var - risk_aversion * port_ret
            return port_var

        def objective_jacobian(w: np.ndarray) -> np.ndarray:
            grad = self.cov @ w
            if expected_returns is not None and risk_aversion > 0.0:
                grad -= risk_aversion * expected_returns
            return grad

        constraints = [
            {
                "type": "eq",
                "fun": lambda w: np.dot(w, self.betas) - target_beta,
                "jac": lambda w: self.betas,
            }
        ]

        if dollar_neutral:
            # Net zero cash exposure (sum of weights = 0)
            constraints.append({
                "type": "eq",
                "fun": lambda w: np.sum(w),
                "jac": lambda w: np.ones(self.n_assets),
            })
        else:
            # Fully invested (sum of weights = 1)
            constraints.append({
                "type": "eq",
                "fun": lambda w: np.sum(w) - 1.0,
                "jac": lambda w: np.ones(self.n_assets),
            })

        bounds = [weight_bounds for _ in range(self.n_assets)]

        # Initial feasible guess
        w0 = np.zeros(self.n_assets)
        if not dollar_neutral:
            w0 = np.ones(self.n_assets) / self.n_assets

        res = minimize(
            objective,
            w0,
            jac=objective_jacobian,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-11, "maxiter": 1000},
        )

        if not res.success:
            raise RuntimeError(f"Optimization failed: {res.message}")

        opt_weights = res.x
        opt_vol = np.sqrt(opt_weights.T @ self.cov @ opt_weights)
        realized_beta = float(np.dot(opt_weights, self.betas))

        return {
            "Optimal_Weights": opt_weights,
            "Optimized_Vol": float(opt_vol),
            "Realized_Beta": realized_beta,
            "Net_Exposure": float(np.sum(opt_weights)),
            "Gross_Exposure": float(np.sum(np.abs(opt_weights))),
            "Convergence": res.success,
        }
