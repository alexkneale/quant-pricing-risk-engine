"""
Monte Carlo Pricing Engine.
Includes vectorised geometric Brownian motion path generation, variance reduction
via antithetic variates, path-dependent Asian pricer, and Longstaff-Schwartz
Least Squares Monte Carlo (LSM) for American options.
"""

from typing import Literal, Tuple
import numpy as np


class MonteCarloEngine:
    """
    Monte Carlo framework for path-dependent and early-exercise contingent claims.
    """

    def __init__(
        self,
        S0: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
        seed: int = 42,
    ):
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.q = q
        self.rng = np.random.default_rng(seed)

    def _generate_gbm_paths(
        self,
        n_paths: int,
        n_steps: int,
        antithetic: bool = True,
    ) -> np.ndarray:
        """
        Simulates asset price paths under risk-neutral measure Q.
        Uses exact log-Euler discrete transition.
        Shape returned: (effective_paths, n_steps + 1)
        """
        dt = self.T / n_steps
        drift = (self.r - self.q - 0.5 * self.sigma**2) * dt
        vol_diffusion = self.sigma * np.sqrt(dt)

        if antithetic:
            half_paths = n_paths // 2
            z = self.rng.standard_normal(size=(half_paths, n_steps))
            z = np.vstack([z, -z])
        else:
            z = self.rng.standard_normal(size=(n_paths, n_steps))

        log_returns = drift + vol_diffusion * z
        log_paths = np.zeros((z.shape[0], n_steps + 1))
        log_paths[:, 0] = np.log(self.S0)
        log_paths[:, 1:] = np.log(self.S0) + np.cumsum(log_returns, axis=1)

        return np.exp(log_paths)

    def price_european(
        self,
        option_type: Literal["call", "put"] = "call",
        n_paths: int = 100_000,
        antithetic: bool = True,
    ) -> Tuple[float, float]:
        """
        Prices a European vanilla option via vectorized Monte Carlo.
        Returns: (price, standard_error)
        """
        paths = self._generate_gbm_paths(n_paths=n_paths, n_steps=1, antithetic=antithetic)
        ST = paths[:, -1]

        if option_type == "call":
            payoff = np.maximum(ST - self.K, 0.0)
        else:
            payoff = np.maximum(self.K - ST, 0.0)

        discount = np.exp(-self.r * self.T)
        discounted_payoffs = discount * payoff

        price = float(np.mean(discounted_payoffs))
        stderr = float(np.std(discounted_payoffs, ddof=1) / np.sqrt(len(discounted_payoffs)))
        return price, stderr

    def price_asian(
        self,
        option_type: Literal["call", "put"] = "call",
        averaging_type: Literal["arithmetic", "geometric"] = "arithmetic",
        n_paths: int = 100_000,
        n_steps: int = 252,
        antithetic: bool = True,
    ) -> Tuple[float, float]:
        """
        Prices discrete Asian options based on path average asset price.
        """
        paths = self._generate_gbm_paths(n_paths=n_paths, n_steps=n_steps, antithetic=antithetic)
        # Exclude t=0 in monitoring schedule
        monitoring_paths = paths[:, 1:]

        if averaging_type == "arithmetic":
            A_T = np.mean(monitoring_paths, axis=1)
        elif averaging_type == "geometric":
            A_T = np.exp(np.mean(np.log(monitoring_paths), axis=1))
        else:
            raise ValueError(f"Unknown averaging type: {averaging_type}")

        if option_type == "call":
            payoff = np.maximum(A_T - self.K, 0.0)
        else:
            payoff = np.maximum(self.K - A_T, 0.0)

        discounted_payoff = np.exp(-self.r * self.T) * payoff
        price = float(np.mean(discounted_payoff))
        stderr = float(np.std(discounted_payoff, ddof=1) / np.sqrt(len(discounted_payoff)))
        return price, stderr

    def price_american_lsm(
        self,
        option_type: Literal["call", "put"] = "put",
        n_paths: int = 50_000,
        n_steps: int = 50,
        degree: int = 3,
    ) -> Tuple[float, float]:
        """
        Longstaff-Schwartz Least Squares Monte Carlo (LSM) Engine for American early exercise.
        Approximates continuation values via cross-sectional polynomial regression.
        """
        paths = self._generate_gbm_paths(n_paths=n_paths, n_steps=n_steps, antithetic=False)
        dt = self.T / n_steps
        discount = np.exp(-self.r * dt)

        # Compute intrinsic payoff matrix across the grid
        if option_type == "put":
            payoff = np.maximum(self.K - paths, 0.0)
        else:
            payoff = np.maximum(paths - self.K, 0.0)

        # Cashflow trajectory matrix initialized at terminal payoff
        cash_flows = payoff[:, -1].copy()

        # Backward induction
        for t in range(n_steps - 1, 0, -1):
            S_t = paths[:, t]
            h_t = payoff[:, t]

            # Consider only in-the-money paths to stabilize regression conditioning
            itm_mask = h_t > 0.0

            if np.count_nonzero(itm_mask) <= (degree + 1):
                # Insufficient ITM paths to fit basis functions; purely discount cash flows
                cash_flows = cash_flows * discount
                continue

            X = S_t[itm_mask]
            # Discounted future cashflow realization
            Y = cash_flows[itm_mask] * discount

            # Polynomial regression using weighted basis functions
            poly_basis = np.polynomial.polynomial.polyfit(X, Y, deg=degree)
            continuation_val = np.polynomial.polynomial.polyval(X, poly_basis)

            # Exercise policy decision
            exercise = h_t[itm_mask] > continuation_val

            # Update paths where exercise is optimal
            cash_flows = cash_flows * discount
            cash_flows[itm_mask] = np.where(exercise, h_t[itm_mask], cash_flows[itm_mask])

        option_price = float(np.mean(cash_flows * discount))
        stderr = float(np.std(cash_flows * discount, ddof=1) / np.sqrt(n_paths))
        return option_price, stderr
