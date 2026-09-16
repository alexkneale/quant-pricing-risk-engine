import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from typing import Dict, Any


def test_stylized_facts(returns: np.ndarray, lags: int = 10) -> Dict[str, Any]:
    """Runs statistical battery verifying financial return stylized facts.
    returns: 1D array of daily log returns.
    """
    clean_returns = returns[~np.isnan(returns)]
    T = len(clean_returns)

    # 1. Distribution Moments
    mean_val = np.mean(clean_returns)
    std_val = np.std(clean_returns, ddof=1)
    skewness = stats.skew(clean_returns)
    kurtosis_excess = stats.kurtosis(clean_returns, fisher=True)  # Normal = 0

    # 2. Jarque-Bera Normality Test
    jb_stat, jb_p_value = stats.jarque_bera(clean_returns)

    # 3. Autocorrelation in raw returns (Market efficiency test)
    # Null hypothesis: No autocorrelation
    lb_returns = acorr_ljungbox(clean_returns, lags=[lags], return_df=True)
    raw_lb_stat = lb_returns['lb_stat'].values[0]
    raw_lb_pvalue = lb_returns['lb_pvalue'].values[0]

    # 4. Autocorrelation in squared returns (Volatility clustering test)
    # Rejection of Null confirms ARCH effects
    squared_returns = (clean_returns - mean_val) ** 2
    lb_sq_returns = acorr_ljungbox(squared_returns, lags=[lags], return_df=True)
    sq_lb_stat = lb_sq_returns['lb_stat'].values[0]
    sq_lb_pvalue = lb_sq_returns['lb_pvalue'].values[0]

    # 5. Leverage Effect Metric: Corr(r_t, |r_{t+1}|)
    r_t = clean_returns[:-1]
    abs_r_next = np.abs(clean_returns[1:])
    leverage_corr, _ = stats.pearsonr(r_t, abs_r_next)

    return {
        "sample_size": T,
        "mean_daily": mean_val,
        "annualized_vol": std_val * np.sqrt(252),
        "skewness": skewness,
        "excess_kurtosis": kurtosis_excess,
        "is_fat_tailed": kurtosis_excess > 1.0,
        "jarque_bera": {"stat": jb_stat, "p_value": jb_p_value, "normal_rejected": jb_p_value < 0.01},
        "raw_returns_autocorr": {"lb_stat": raw_lb_stat, "p_value": raw_lb_pvalue, "is_serially_correlated": raw_lb_pvalue < 0.05},
        "volatility_clustering": {"lb_stat": sq_lb_stat, "p_value": sq_lb_pvalue, "has_clustering": sq_lb_pvalue < 0.01},
        "leverage_effect_correlation": leverage_corr  # Expect negative sign
    }
