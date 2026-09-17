"""
Finite Difference PDE Solvers.
Implements the Crank-Nicolson implicit-explicit discretization with projected SOR
and Thomas tridiagonal solution for European and American option valuation.
"""

from typing import Literal, Tuple
import numpy as np


class CrankNicolsonPricer:
    """
    Finite Difference PDE solver utilizing the Crank-Nicolson scheme.
    Supports European and American early exercise valuation.
    """

    def __init__(
        self,
        S0: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
        S_max_mult: float = 4.0,
        M: int = 400,  # Asset steps
        N: int = 1000, # Time steps
    ):
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.q = q
        self.S_max = K * S_max_mult
        self.M = M
        self.N = N

        self.dS = self.S_max / self.M
        self.dt = self.T / self.N
        self.S_grid = np.linspace(0.0, self.S_max, self.M + 1)

    def _setup_tridiagonal_system(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Assembles tridiagonal matrices A and B for interior nodes:
        A * V^{n+1} = B * V^n + d
        """
        i = np.arange(1, self.M)
        alpha = 0.5 * (self.sigma**2 * (i**2) - (self.r - self.q) * i)
        beta = -(self.sigma**2 * (i**2) + self.r)
        gamma = 0.5 * (self.sigma**2 * (i**2) + (self.r - self.q) * i)

        # Matrix A (Left-hand side implicit step at n+1)
        # -theta * dt * alpha * V_{i-1}^{n+1} + (1 - theta*dt*beta) * V_i^{n+1} - theta * dt * gamma * V_{i+1}^{n+1}
        a_sub = -0.5 * self.dt * alpha[1:]
        a_diag = 1.0 - 0.5 * self.dt * beta
        a_sup = -0.5 * self.dt * gamma[:-1]

        # Matrix B (Right-hand side explicit step at n)
        b_sub = 0.5 * self.dt * alpha[1:]
        b_diag = 1.0 + 0.5 * self.dt * beta
        b_sup = 0.5 * self.dt * gamma[:-1]

        return a_sub, a_diag, a_sup, b_sub, b_diag, b_sup

    @staticmethod
    def _thomas_algorithm(a_sub: np.ndarray, a_diag: np.ndarray, a_sup: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        """
        Solves tridiagonal system Ax = rhs in O(M) complexity using the Thomas algorithm.
        """
        n = len(rhs)
        c_prime = np.zeros(n - 1)
        d_prime = np.zeros(n)

        c_prime[0] = a_sup[0] / a_diag[0]
        d_prime[0] = rhs[0] / a_diag[0]

        for i in range(1, n - 1):
            denom = a_diag[i] - a_sub[i - 1] * c_prime[i - 1]
            c_prime[i] = a_sup[i] / denom
            d_prime[i] = (rhs[i] - a_sub[i - 1] * d_prime[i - 1]) / denom

        denom = a_diag[-1] - a_sub[-1] * c_prime[-1]
        d_prime[-1] = (rhs[-1] - a_sub[-1] * d_prime[-2]) / denom

        x = np.zeros(n)
        x[-1] = d_prime[-1]
        for i in range(n - 2, -1, -1):
            x[i] = d_prime[i] - c_prime[i] * x[i + 1]

        return x

    def price(
        self,
        option_type: Literal["call", "put"] = "put",
        exercise_style: Literal["european", "american"] = "american",
    ) -> float:
        """
        Solves the Black-Scholes PDE backward in time from tau = 0 (t = T) to tau = T (t = 0).
        """
        # Terminal condition at tau = 0 (maturity)
        if option_type == "call":
            payoff = np.maximum(self.S_grid - self.K, 0.0)
        else:
            payoff = np.maximum(self.K - self.S_grid, 0.0)

        V = payoff.copy()
        a_sub, a_diag, a_sup, b_sub, b_diag, b_sup = self._setup_tridiagonal_system()

        i = np.arange(1, self.M)
        alpha = 0.5 * (self.sigma**2 * (i**2) - (self.r - self.q) * i)
        gamma = 0.5 * (self.sigma**2 * (i**2) + (self.r - self.q) * i)

        # Backward time stepping
        for step in range(self.N):
            tau = step * self.dt
            tau_next = (step + 1) * self.dt

            # Boundary conditions at S = 0 and S = S_max
            if option_type == "call":
                V_0_n = 0.0
                V_0_next = 0.0
                V_M_n = self.S_max * np.exp(-self.q * tau) - self.K * np.exp(-self.r * tau)
                V_M_next = self.S_max * np.exp(-self.q * tau_next) - self.K * np.exp(-self.r * tau_next)
            else:
                V_0_n = self.K * np.exp(-self.r * tau)
                V_0_next = self.K * np.exp(-self.r * tau_next)
                V_M_n = 0.0
                V_M_next = 0.0

            # Construct explicit RHS: B * V^n
            V_interior = V[1:-1]
            rhs = b_diag * V_interior
            rhs[:-1] += b_sup * V_interior[1:]
            rhs[1:] += b_sub * V_interior[:-1]

            # Adjust RHS with boundary values
            rhs[0] += 0.5 * self.dt * alpha[0] * (V_0_n + V_0_next)
            rhs[-1] += 0.5 * self.dt * gamma[-1] * (V_M_n + V_M_next)

            # Solve tridiagonal system
            V_new_interior = self._thomas_algorithm(a_sub, a_diag.copy(), a_sup, rhs)

            # American exercise boundary projection: V = max(V_continuation, Payoff)
            if exercise_style == "american":
                V_new_interior = np.maximum(V_new_interior, payoff[1:-1])

            V[1:-1] = V_new_interior
            V[0] = V_0_next
            V[-1] = V_M_next

        # Interpolate exact spot price S0
        return float(np.interp(self.S0, self.S_grid, V))
