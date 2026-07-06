"""
quantbt — a compact, vectorized backtesting and research library.

Designed for systematic-strategy research: generate or load OHLCV data,
build signals from technical indicators, run a cost-aware backtest, and
evaluate the result with proper risk metrics, walk-forward analysis and
Monte Carlo robustness checks.

Public API
----------
    from quantbt import Backtester, BacktestConfig
    from quantbt import metrics, indicators, signals
"""

from .engine import Backtester, BacktestConfig, BacktestResult
from .data import load_ohlcv, synthetic_ohlcv
from . import indicators, signals, metrics, portfolio, plotting

__version__ = "0.6.0"

__all__ = [
    "Backtester",
    "BacktestConfig",
    "BacktestResult",
    "load_ohlcv",
    "synthetic_ohlcv",
    "indicators",
    "signals",
    "metrics",
    "portfolio",
    "plotting",
]
