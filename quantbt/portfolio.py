"""
Execution and portfolio accounting.

This module turns a *target position* series into a realised strategy
return stream. It models the two frictions that most naive backtests
ignore and that quietly turn a "profitable" system into a losing one:

* **Transaction costs** — a per-trade cost in basis points charged on the
  turnover (the change in position), covering brokerage + exchange fees.
* **Slippage** — an additional bps hit applied whenever the position
  changes, standing in for the gap between the signal price and the fill.

It also lags the signal by one bar so that a decision made on the close of
bar *t* is only acted on from bar *t+1* — removing look-ahead bias.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class CostModel:
    """Friction parameters, all in basis points (1 bp = 0.01%)."""

    commission_bps: float = 2.0
    slippage_bps: float = 1.0

    @property
    def per_turnover(self) -> float:
        return (self.commission_bps + self.slippage_bps) / 10_000.0


def apply_execution(
    close: pd.Series,
    target_position: pd.Series,
    costs: CostModel,
    lag: int = 1,
) -> pd.DataFrame:
    """Compute per-bar strategy returns net of costs.

    Returns a frame with columns:
        ``position``   — the position actually held during the bar.
        ``asset_ret``  — the underlying close-to-close return.
        ``gross_ret``  — position * asset_ret.
        ``cost``       — friction charged this bar.
        ``net_ret``    — gross_ret - cost.
    """
    asset_ret = close.pct_change().fillna(0.0)

    # Act on the signal only from the next bar — this is the single most
    # important line for avoiding look-ahead bias.
    held = target_position.shift(lag).fillna(0.0)

    turnover = held.diff().abs().fillna(held.abs())
    cost = turnover * costs.per_turnover

    gross = held * asset_ret
    net = gross - cost

    return pd.DataFrame(
        {
            "position": held,
            "asset_ret": asset_ret,
            "gross_ret": gross,
            "cost": cost,
            "net_ret": net,
        }
    )


def equity_curve(net_ret: pd.Series, initial: float = 1.0) -> pd.Series:
    """Compound a net-return series into an equity curve."""
    return initial * (1.0 + net_ret).cumprod()


def volatility_target(
    net_ret: pd.Series, target_annual_vol: float = 0.15, window: int = 20
) -> pd.Series:
    """Return a per-bar leverage scalar that targets a constant volatility.

    Scaling exposure by realised volatility is one of the cheapest ways to
    improve a strategy's risk-adjusted return; it caps leverage at 3x to
    stay realistic.
    """
    realised = net_ret.rolling(window, min_periods=window).std() * np.sqrt(252)
    scalar = (target_annual_vol / realised).clip(upper=3.0)
    return scalar.shift(1).fillna(1.0)
