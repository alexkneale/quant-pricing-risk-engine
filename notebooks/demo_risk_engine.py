"""
End-to-End Demo: Portfolio Risk, Coherent Measures & Market-Neutral Optimization
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import yfinance as yf

from portfolio.var_es import PortfolioRiskEngine
from portfolio.risk_attribution import RiskAttributionEngine
from portfolio.hedger import BetaNeutralHedger


def calculate_yang_zhang_vol(df: pd.DataFrame, window: int = 30) -> float:
    """
    Computes annualized Yang-Zhang realized volatility over recent trading window.
    Assumes df has columns: 'Open', 'High', 'Low', 'Close'.
    """
    c_prev = df["Close"].shift(1)
    o = df["Open"]
    h = df["High"]
    l = df["Low"]
    c = df["Close"]

    k = 0.34 / (1.34 + (window + 1) / (window - 1))

    # Overnight jump variance
    v_overnight = (np.log(o / c_prev) ** 2).rolling(window=window).mean()
    # Open-to-close variance
    v_open_to_close = (np.log(c / o) ** 2).rolling(window=window).mean()
    # Rogers-Satchell intraday volatility
    rs = (
        np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o)
    ).rolling(window=window).mean()

    # Yang-Zhang composite variance
    yz_var = v_overnight + k * v_open_to_close + (1.0 - k) * rs
    # Annualize (252 trading days)
    yz_vol = np.sqrt(yz_var * 252.0).iloc[-1]
    return float(yz_vol)


def main():
    print("=" * 75)
    print("DEMO: FINANCIAL MATHEMATICS & RISK ATTRIBUTION ENGINE (MODULE 3)")
    print("=" * 75)

    # 1. Fetch Market & Portfolio Data
    tickers = ["AAPL", "MSFT", "NVDA", "JPM", "XOM"]
    market_ticker = "SPY"
    all_tickers = tickers + [market_ticker]

    print(f"\n[+] Downloading 2 years of daily data for {all_tickers}...")
    raw = yf.download(all_tickers, period="2y", interval="1d", group_by="ticker", auto_adjust=False)

    # Extract synchronous Close returns
    close_prices = pd.DataFrame({t: raw[t]["Close"] for t in all_tickers}).dropna()
    log_returns = np.log(close_prices / close_prices.shift(1)).dropna()

    asset_returns = log_returns[tickers]
    market_returns = log_returns[market_ticker]

    # Equal initial weighting
    N = len(tickers)
    w_initial = np.ones(N) / N

    print(f"[+] Calculating instantaneous Yang-Zhang Realized Volatilities (30-day window)...")
    yz_vols = np.array([calculate_yang_zhang_vol(raw[t]) for t in tickers])

    for t, vol in zip(tickers, yz_vols):
        print(f"    - {t} Yang-Zhang Vol (Annualized): {vol * 100:.2f}%")

    # 2. Assemble Sigma = D @ R @ D with Ledoit-Wolf Shrinkage
    cov_annualized = RiskAttributionEngine.build_covariance_from_realized(
        realized_vols=yz_vols,
        returns=asset_returns,
        shrinkage=True,
    )

    # Daily covariance for short-horizon daily VaR/ES
    cov_daily = cov_annualized / 252.0

    print("\n" + "-" * 75)
    print("1. VALUE AT RISK (VaR) & EXPECTED SHORTFALL (ES) - DAILY 99% CONFIDENCE")
    print("-" * 75)

    risk_engine = PortfolioRiskEngine(confidence_level=0.99)

    # Parametric Gaussian
    param_normal = risk_engine.parametric_risk(
        weights=w_initial,
        cov_matrix=cov_daily,
        distribution="gaussian"
    )
    # Parametric Student-t (dof=4)
    param_t = risk_engine.parametric_risk(
        weights=w_initial,
        cov_matrix=cov_daily,
        distribution="student_t",
        dof=4.0
    )
    # Historical Simulation
    hist_risk = risk_engine.historical_risk(weights=w_initial, returns=asset_returns)
    # Filtered Historical Simulation (FHS)
    print("[+] Running Filtered Historical Simulation (Fitting GARCH(1,1) per asset)...")
    fhs_risk = risk_engine.filtered_historical_simulation(
        weights=w_initial,
        returns=asset_returns,
        n_scenarios=5000,
        random_state=42
    )

    summary_df = pd.DataFrame({
        "Model": [
            "Parametric (Gaussian)",
            "Parametric (Student-t, dof=4)",
            "Historical Simulation",
            "Filtered Historical Sim (GARCH)"
        ],
        "Daily 99% VaR (%)": [
            param_normal["VaR"] * 100,
            param_t["VaR"] * 100,
            hist_risk["VaR"] * 100,
            fhs_risk["VaR"] * 100
        ],
        "Daily 99% ES (%)": [
            param_normal["ES"] * 100,
            param_t["ES"] * 100,
            hist_risk["ES"] * 100,
            fhs_risk["ES"] * 100
        ]
    })
    print(summary_df.to_string(index=False))

    print("\n" + "-" * 75)
    print("2. EULER RISK DECOMPOSITION (COMPONENT RISK ATTRIBUTION)")
    print("-" * 75)
    decomp_df = RiskAttributionEngine.euler_volatility_attribution(
        weights=w_initial,
        cov_matrix=cov_annualized
    )
    decomp_df.index = tickers
    print(decomp_df.to_string())

    print("\n" + "-" * 75)
    print("3. SYSTEMATIC VS. IDIOSYNCRATIC VARIANCE (CAPM SINGLE-INDEX)")
    print("-" * 75)
    factor_res = RiskAttributionEngine.factor_beta_decomposition(
        asset_returns=asset_returns,
        market_returns=market_returns,
        weights=w_initial
    )
    print(f"Portfolio Beta (SPY Benchmark) : {factor_res['Portfolio_Beta']:.4f}")
    print(f"Systematic Variance Ratio      : {factor_res['Pct_Systematic'] * 100:.2f}%")
    print(f"Idiosyncratic Variance Ratio   : {factor_res['Pct_Idiosyncratic'] * 100:.2f}%")

    print("\n" + "-" * 75)
    print("4. CONVEX OPTIMIZATION: MARKET-NEUTRAL HEDGER")
    print("-" * 75)
    hedger = BetaNeutralHedger(
        cov_matrix=cov_annualized,
        betas=factor_res["Asset_Betas"].values
    )

    # Solve for Beta = 0, Dollar Neutral (sum w = 0), weights between -0.5 and 0.5
    hedged_solution = hedger.optimize_beta_neutral(
        target_beta=0.0,
        dollar_neutral=True,
        weight_bounds=(-0.5, 0.5)
    )

    hedged_weights = pd.Series(hedged_solution["Optimal_Weights"], index=tickers)
    print("[+] Optimal Beta-Neutral Weights:")
    for ticker, w in hedged_weights.items():
        print(f"    - {ticker:5s}: {w * 100:>6.2f}%")

    print(f"\nNet Cash Exposure     : {hedged_solution['Net_Exposure']:.6f}")
    print(f"Gross Leverage        : {hedged_solution['Gross_Exposure'] * 100:.2f}%")
    print(f"Realized Portfolio Beta: {hedged_solution['Realized_Beta']:.6f} (Market Neutral)")
    print(f"Annualized Volatility : {hedged_solution['Optimized_Vol'] * 100:.2f}%")
    print("=" * 75)


if __name__ == "__main__":
    main()
