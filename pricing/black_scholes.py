"""
Analytical Black-Scholes-Merton Engine.
Provides vectorized closed-form derivative valuation and exact Greeks.
"""

from dataclasses import dataclass
from typing import Dict, Literal
import numpy as np
from scipy.stats import norm


@dataclass(frozen=True)
class BlackScholesPricer:
    """
    Computes analytical prices and Greeks for European vanilla options.

    Attributes:
        S: Spot price of the underlying asset.
        K: Strike price.
        T: Time to expiration in years.
        r: Continuously compounded risk-free rate.
        sigma: Volatility of the underlying asset.
        q: Continuous dividend yield (default 0.0).
    """
    S: float
    K: float
    T: float
    r: float
    sigma: float
    q: float = 0.0

    def __post_init__(self):
        if self.S <= 0:
            raise ValueError(f"Spot S must be positive. Got {self.S}")
        if self.K <= 0:
            raise ValueError(f"Strike K must be positive. Got {self.K}")
        if self.T < 0:
            raise ValueError(f"Time T cannot be negative. Got {self.T}")
        if self.sigma < 0:
            raise ValueError(f"Volatility sigma cannot be negative. Got {self.sigma}")

    def _d1_d2(self) -> tuple[float, float]:
        """Calculates standard BSM d1 and d2 metrics."""
        if self.T == 0.0:
            return (np.nan, np.nan)
        
        vol_sqrt_t = self.sigma * np.sqrt(self.T)
        d1 = (
            np.log(self.S / self.K)
            + (self.r - self.q + 0.5 * self.sigma**2) * self.T
        ) / vol_sqrt_t
        d2 = d1 - vol_sqrt_t
        return d1, d2

    def price(self, option_type: Literal["call", "put"] = "call") -> float:
        """Computes European option price under Black-Scholes-Merton."""
        if self.T == 0.0:
            if option_type == "call":
                return max(self.S - self.K, 0.0)
            return max(self.K - self.S, 0.0)

        d1, d2 = self._d1_d2()
        df_r = np.exp(-self.r * self.T)
        df_q = np.exp(-self.q * self.T)

        if option_type == "call":
            return float(self.S * df_q * norm.cdf(d1) - self.K * df_r * norm.cdf(d2))
        elif option_type == "put":
            return float(self.K * df_r * norm.cdf(-d2) - self.S * df_q * norm.cdf(-d1))
        else:
            raise ValueError(f"Invalid option_type: {option_type}. Use 'call' or 'put'.")

    def greeks(self, option_type: Literal["call", "put"] = "call") -> Dict[str, float]:
        """
        Analytically derives first and second order sensitivities (Greeks).
        Returns Delta, Gamma, Vega, Theta, and Rho.
        """
        if self.T == 0.0:
            return {
                "delta": 1.0 if (option_type == "call" and self.S > self.K) else 0.0,
                "gamma": 0.0,
                "vega": 0.0,
                "theta": 0.0,
                "rho": 0.0,
            }

        d1, d2 = self._d1_d2()
        df_r = np.exp(-self.r * self.T)
        df_q = np.exp(-self.q * self.T)
        pdf_d1 = norm.pdf(d1)
        sqrt_t = np.sqrt(self.T)

        # Common Greeks
        gamma = (df_q * pdf_d1) / (self.S * self.sigma * sqrt_t)
        vega = self.S * df_q * sqrt_t * pdf_d1

        if option_type == "call":
            delta = df_q * norm.cdf(d1)
            theta = (
                -(self.S * df_q * self.sigma * pdf_d1) / (2 * sqrt_t)
                + self.q * self.S * df_q * norm.cdf(d1)
                - self.r * self.K * df_r * norm.cdf(d2)
            )
            rho = self.K * self.T * df_r * norm.cdf(d2)
        else:
            delta = -df_q * norm.cdf(-d1)
            theta = (
                -(self.S * df_q * self.sigma * pdf_d1) / (2 * sqrt_t)
                - self.q * self.S * df_q * norm.cdf(-d1)
                + self.r * self.K * df_r * norm.cdf(-d2)
            )
            rho = -self.K * self.T * df_r * norm.cdf(-d2)

        return {
            "delta": float(delta),
            "gamma": float(gamma),
            "vega": float(vega),
            "theta": float(theta),
            "rho": float(rho),
        }
