"""
Performance and risk metrics.

All functions operate on a per-bar *net return* series. Annualisation uses
``periods_per_year`` (252 for daily bars) so the same code works for
intraday data by passing a different value.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def total_return(returns: pd.Series) -> float:
    return float((1.0 + returns).prod() - 1.0)


def cagr(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """Compound annual growth rate."""
    n = len(returns)
    if n == 0:
        return 0.0
    growth = (1.0 + returns).prod()
    if growth <= 0:
        return -1.0
    years = n / periods_per_year
    return float(growth ** (1.0 / years) - 1.0)


def annual_volatility(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    return float(returns.std(ddof=0) * np.sqrt(periods_per_year))


def sharpe(returns: pd.Series, rf: float = 0.0, periods_per_year: int = TRADING_DAYS) -> float:
    """Annualised Sharpe ratio."""
    excess = returns - rf / periods_per_year
    sd = excess.std(ddof=0)
    if sd == 0:
        return 0.0
    return float(excess.mean() / sd * np.sqrt(periods_per_year))


def sortino(returns: pd.Series, rf: float = 0.0, periods_per_year: int = TRADING_DAYS) -> float:
    """Like Sharpe but penalises only downside deviation."""
    excess = returns - rf / periods_per_year
    downside = excess[excess < 0]
    dd = downside.std(ddof=0)
    if dd == 0:
        return 0.0
    return float(excess.mean() / dd * np.sqrt(periods_per_year))


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Fractional drawdown from the running peak at each point in time."""
    equity = (1.0 + returns).cumprod()
    peak = equity.cummax()
    return equity / peak - 1.0


def max_drawdown(returns: pd.Series) -> float:
    dd = drawdown_series(returns)
    return float(dd.min()) if len(dd) else 0.0


def calmar(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """CAGR divided by the magnitude of the max drawdown."""
    mdd = abs(max_drawdown(returns))
    if mdd == 0:
        return 0.0
    return float(cagr(returns, periods_per_year) / mdd)


def hit_rate(returns: pd.Series) -> float:
    """Fraction of non-zero bars that were positive."""
    active = returns[returns != 0]
    if len(active) == 0:
        return 0.0
    return float((active > 0).mean())


def profit_factor(returns: pd.Series) -> float:
    """Gross profit divided by gross loss."""
    gains = returns[returns > 0].sum()
    losses = -returns[returns < 0].sum()
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def summary(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> dict:
    """One-shot dictionary of the metrics you actually report."""
    return {
        "total_return": total_return(returns),
        "cagr": cagr(returns, periods_per_year),
        "annual_vol": annual_volatility(returns, periods_per_year),
        "sharpe": sharpe(returns, periods_per_year=periods_per_year),
        "sortino": sortino(returns, periods_per_year=periods_per_year),
        "max_drawdown": max_drawdown(returns),
        "calmar": calmar(returns, periods_per_year),
        "hit_rate": hit_rate(returns),
        "profit_factor": profit_factor(returns),
    }


def format_summary(stats: dict) -> str:
    """Pretty, aligned text block for printing to a terminal."""
    pct = {"total_return", "cagr", "annual_vol", "max_drawdown"}
    lines = []
    for key, value in stats.items():
        label = key.replace("_", " ").title()
        if key in pct:
            lines.append(f"  {label:<16} {value * 100:>8.2f}%")
        else:
            lines.append(f"  {label:<16} {value:>9.3f}")
    return "\n".join(lines)
