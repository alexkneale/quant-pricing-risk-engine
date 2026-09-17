"""
Unit and Integration Tests for Derivative Pricing Engine.
Validates convergence properties across Analytical, PDE, and Monte Carlo engines.
"""

import sys
import pprint
pprint.pp(sys.path)

import pytest
import numpy as np
import os
print(os.getcwd())

import sys
from pathlib import Path

sys.path.append(str(Path.cwd().parent))

from pricing.black_scholes import BlackScholesPricer
from pricing.pde_solvers import CrankNicolsonPricer
from pricing.monte_carlo import MonteCarloEngine


@pytest.fixture
def market_parameters():
    return {
        "S0": 100.0,
        "K": 100.0,
        "T": 1.0,
        "r": 0.05,
        "sigma": 0.20,
        "q": 0.02,
    }


def test_analytical_put_call_parity(market_parameters):
    """Verifies that closed-form BSM satisfies Put-Call Parity: C - P = S*e^(-qT) - K*e^(-rT)."""
    p = market_parameters
    pricer = BlackScholesPricer(
        S=p["S0"], K=p["K"], T=p["T"], r=p["r"], sigma=p["sigma"], q=p["q"]
    )
    call_price = pricer.price("call")
    put_price = pricer.price("put")

    synthetic_diff = p["S0"] * np.exp(-p["q"] * p["T"]) - p["K"] * np.exp(-p["r"] * p["T"])
    assert np.isclose(call_price - put_price, synthetic_diff, atol=1e-7)


def test_analytical_greeks_finite_difference(market_parameters):
    """Cross-validates analytical Greeks against numerical finite-difference perturbations."""
    p = market_parameters
    pricer = BlackScholesPricer(
        S=p["S0"], K=p["K"], T=p["T"], r=p["r"], sigma=p["sigma"], q=p["q"]
    )
    exact_greeks = pricer.greeks("call")

    eps_S = 1e-4
    p_up = BlackScholesPricer(S=p["S0"] + eps_S, K=p["K"], T=p["T"], r=p["r"], sigma=p["sigma"], q=p["q"]).price("call")
    p_down = BlackScholesPricer(S=p["S0"] - eps_S, K=p["K"], T=p["T"], r=p["r"], sigma=p["sigma"], q=p["q"]).price("call")
    fd_delta = (p_up - p_down) / (2 * eps_S)
    fd_gamma = (p_up - 2 * pricer.price("call") + p_down) / (eps_S**2)

    assert np.isclose(exact_greeks["delta"], fd_delta, atol=1e-5)
    assert np.isclose(exact_greeks["gamma"], fd_gamma, atol=1e-4)


def test_pde_european_convergence_to_analytical(market_parameters):
    """Verifies that Crank-Nicolson PDE converges to the Black-Scholes analytical formula."""
    p = market_parameters
    bs = BlackScholesPricer(S=p["S0"], K=p["K"], T=p["T"], r=p["r"], sigma=p["sigma"], q=p["q"])
    analytical_put = bs.price("put")

    pde = CrankNicolsonPricer(
        S0=p["S0"], K=p["K"], T=p["T"], r=p["r"], sigma=p["sigma"], q=p["q"],
        M=500, N=1000
    )
    pde_put = pde.price(option_type="put", exercise_style="european")

    # Discretization error bounded within $0.05 on a $100 stock
    assert np.isclose(pde_put, analytical_put, atol=5e-2)


def test_american_early_exercise_premium(market_parameters):
    """Asserts that an American put is strictly >= European put under positive interest rates."""
    p = market_parameters
    pde_pricer = CrankNicolsonPricer(
        S0=p["S0"], K=p["K"], T=p["T"], r=p["r"], sigma=p["sigma"], q=p["q"],
        M=400, N=800
    )
    european_put = pde_pricer.price(option_type="put", exercise_style="european")
    american_put = pde_pricer.price(option_type="put", exercise_style="american")

    assert american_put >= european_put
    early_exercise_premium = american_put - european_put
    assert early_exercise_premium > 0.0


def test_monte_carlo_european_convergence(market_parameters):
    """Verifies that vectorized Monte Carlo converges within standard error tolerance."""
    p = market_parameters
    bs = BlackScholesPricer(S=p["S0"], K=p["K"], T=p["T"], r=p["r"], sigma=p["sigma"], q=p["q"])
    analytical_call = bs.price("call")

    mc = MonteCarloEngine(S0=p["S0"], K=p["K"], T=p["T"], r=p["r"], sigma=p["sigma"], q=p["q"], seed=101)
    mc_call, stderr = mc.price_european(option_type="call", n_paths=150_000, antithetic=True)

    # 99.7% confidence interval (3 sigma)
    assert np.abs(mc_call - analytical_call) <= 3 * stderr


def test_lsm_american_vs_pde(market_parameters):
    """Cross-validates Longstaff-Schwartz Monte Carlo against the Crank-Nicolson American solver."""
    p = market_parameters
    pde = CrankNicolsonPricer(S0=p["S0"], K=p["K"], T=p["T"], r=p["r"], sigma=p["sigma"], q=p["q"], M=300, N=500)
    pde_american_put = pde.price(option_type="put", exercise_style="american")

    mc = MonteCarloEngine(S0=p["S0"], K=p["K"], T=p["T"], r=p["r"], sigma=p["sigma"], q=p["q"], seed=7)
    lsm_put, stderr = mc.price_american_lsm(option_type="put", n_paths=80_000, n_steps=60, degree=3)

    # Validate that both solvers lie within reasonable engineering margin
    assert np.isclose(lsm_put, pde_american_put, atol=0.25)
