"""
The backtest engine.

``Backtester`` glues the pieces together: it takes an OHLCV frame and a
signal function, runs it through the execution model, and hands back a
:class:`BacktestResult` carrying the per-bar frame, the equity curve and
the summary statistics.

The engine is deliberately *vectorized* — no Python bar-by-bar loop — so a
multi-thousand-bar backtest runs in milliseconds, which matters once you
start sweeping parameters or running Monte Carlo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from . import metrics, portfolio
from .portfolio import CostModel

Signal = Callable[[pd.DataFrame], pd.Series]


@dataclass
class BacktestConfig:
    commission_bps: float = 2.0
    slippage_bps: float = 1.0
    initial_capital: float = 100_000.0
    periods_per_year: int = 252
    vol_target: float | None = None  # e.g. 0.15 to target 15% annual vol
    signal_lag: int = 1

    def cost_model(self) -> CostModel:
        return CostModel(self.commission_bps, self.slippage_bps)


@dataclass
class BacktestResult:
    frame: pd.DataFrame
    equity: pd.Series
    stats: dict
    config: BacktestConfig = field(repr=False)

    @property
    def net_ret(self) -> pd.Series:
        return self.frame["net_ret"]

    def report(self) -> str:
        head = (
            f"Backtest over {len(self.frame)} bars "
            f"({self.frame.index[0].date()} \u2192 {self.frame.index[-1].date()})"
        )
        return head + "\n" + metrics.format_summary(self.stats)

    def to_frame(self) -> pd.DataFrame:
        out = self.frame.copy()
        out["equity"] = self.equity
        out["drawdown"] = metrics.drawdown_series(self.net_ret)
        return out


class Backtester:
    """Run a single strategy over an OHLCV frame."""

    def __init__(self, config: BacktestConfig | None = None):
        self.config = config or BacktestConfig()

    def run(self, data: pd.DataFrame, signal: Signal) -> BacktestResult:
        required = {"open", "high", "low", "close"}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f"data is missing columns: {sorted(missing)}")

        target = signal(data).astype(float)
        exec_frame = portfolio.apply_execution(
            data["close"], target, self.config.cost_model(), lag=self.config.signal_lag
        )

        if self.config.vol_target is not None:
            scalar = portfolio.volatility_target(
                exec_frame["net_ret"], self.config.vol_target
            )
            exec_frame["position"] *= scalar
            exec_frame["gross_ret"] *= scalar
            exec_frame["net_ret"] = exec_frame["gross_ret"] - exec_frame["cost"] * scalar

        equity = portfolio.equity_curve(
            exec_frame["net_ret"], self.config.initial_capital
        )
        stats = metrics.summary(exec_frame["net_ret"], self.config.periods_per_year)
        return BacktestResult(exec_frame, equity, stats, self.config)

    def compare(self, data: pd.DataFrame, signals: dict[str, Signal]) -> pd.DataFrame:
        """Run several named strategies and return a stats table."""
        rows = {name: self.run(data, sig).stats for name, sig in signals.items()}
        return pd.DataFrame(rows).T
