import numpy as np
import pandas as pd


def close_to_close_vol(df: pd.DataFrame, window: int = 21, annualize: bool = True) -> pd.Series:
    """Computes standard rolling close-to-close realized volatility."""
    log_ret = np.log(df['Close'] / df['Close'].shift(1))
    vol = log_ret.rolling(window=window).std()
    return vol * np.sqrt(252) if annualize else vol


def garman_klass_vol(df: pd.DataFrame, window: int = 21, annualize: bool = True) -> pd.Series:
    """Computes Garman-Klass realized volatility.
    Assumes zero drift and no overnight jump.
    """
    log_hl = np.log(df['High'] / df['Low'])
    log_co = np.log(df['Close'] / df['Open'])

    rs_term = 0.5 * (log_hl ** 2)
    co_term = (2 * np.log(2) - 1) * (log_co ** 2)
    daily_var = rs_term - co_term

    rolling_var = daily_var.rolling(window=window).mean()
    vol = np.sqrt(rolling_var)
    return vol * np.sqrt(252) if annualize else vol


def yang_zhang_vol(df: pd.DataFrame, window: int = 21, annualize: bool = True) -> pd.Series:
    """Computes Yang-Zhang realized volatility.
    Handles both overnight jumps and continuous intraday price drift.
    """
    # 1. Overnight jump: Close(t-1) to Open(t)
    o = np.log(df['Open'] / df['Close'].shift(1))
    
    # 2. Open to Close: Open(t) to Close(t)
    c = np.log(df['Close'] / df['Open'])
    
    # 3. High/Low relative to Open
    u = np.log(df['High'] / df['Open'])
    d = np.log(df['Low'] / df['Open'])

    # Rogers-Satchell intraday variance component
    rs = u * (u - c) + d * (d - c)

    # Rolling sample variances (ddof=1 for unbiased sample variance)
    var_o = o.rolling(window=window).var(ddof=1)
    var_c = c.rolling(window=window).var(ddof=1)
    var_rs = rs.rolling(window=window).mean()

    # Optimal weighting constant k
    k = 0.34 / (1.34 + (window + 1) / (window - 1))

    # Combined Yang-Zhang variance
    yz_var = var_o + k * var_c + (1.0 - k) * var_rs
    
    # Handle floating point inaccuracies near zero
    yz_var = yz_var.clip(lower=0.0)
    vol = np.sqrt(yz_var)

    return vol * np.sqrt(252) if annualize else vol