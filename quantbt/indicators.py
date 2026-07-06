"""
Vectorized technical indicators.

Every function takes a price/OHLC ``Series``/``DataFrame`` and returns a
``Series`` (or tuple of Series) aligned to the input index. Nothing here
peeks into the future — each value at time *t* uses only data up to *t*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(price: pd.Series, window: int) -> pd.Series:
    """Simple moving average."""
    return price.rolling(window, min_periods=window).mean()


def ema(price: pd.Series, span: int) -> pd.Series:
    """Exponential moving average."""
    return price.ewm(span=span, adjust=False).mean()


def rsi(price: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's Relative Strength Index in the 0..100 range."""
    delta = price.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    return out.fillna(50.0)


def bollinger(price: pd.Series, window: int = 20, n_std: float = 2.0):
    """Return (middle, upper, lower) Bollinger bands."""
    mid = sma(price, window)
    std = price.rolling(window, min_periods=window).std()
    upper = mid + n_std * std
    lower = mid - n_std * std
    return mid, upper, lower


def macd(price: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Return (macd_line, signal_line, histogram)."""
    macd_line = ema(price, fast) - ema(price, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Average True Range — a volatility measure used for position sizing."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / window, adjust=False).mean()


def rolling_zscore(price: pd.Series, window: int = 20) -> pd.Series:
    """Rolling z-score of price — the core of many mean-reversion signals."""
    mean = price.rolling(window, min_periods=window).mean()
    std = price.rolling(window, min_periods=window).std()
    return (price - mean) / std.replace(0.0, np.nan)


def donchian(high: pd.Series, low: pd.Series, window: int = 20):
    """Donchian channel (upper, lower) used for breakout systems."""
    upper = high.rolling(window, min_periods=window).max()
    lower = low.rolling(window, min_periods=window).min()
    return upper, lower
