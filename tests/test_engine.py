"""Tests for the engine, execution model and look-ahead safety."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from quantbt import Backtester, BacktestConfig, synthetic_ohlcv, signals
from quantbt.portfolio import CostModel, apply_execution


def _flat_signal(df):
    return pd.Series(0.0, index=df.index)


def test_flat_strategy_has_zero_return():
    data = synthetic_ohlcv(n=300, seed=1)
    res = Backtester().run(data, _flat_signal)
    assert res.net_ret.abs().sum() == pytest.approx(0.0)


def test_costs_reduce_returns():
    data = synthetic_ohlcv(n=500, seed=2)
    sig = signals.ma_crossover(10, 30)

    cheap = Backtester(BacktestConfig(commission_bps=0, slippage_bps=0)).run(data, sig)
    pricey = Backtester(BacktestConfig(commission_bps=10, slippage_bps=5)).run(data, sig)

    assert pricey.stats["total_return"] <= cheap.stats["total_return"]


def test_signal_is_lagged_no_lookahead():
    # If a signal could act on the same bar it's computed, a "perfect"
    # signal equal to the sign of the *current* return would be riskless.
    # With the mandatory 1-bar lag it must not be.
    data = synthetic_ohlcv(n=400, seed=5)
    ret = data["close"].pct_change().fillna(0.0)

    def clairvoyant(df):
        return np.sign(ret)  # knows today's return — cheating

    res = Backtester(BacktestConfig(commission_bps=0, slippage_bps=0)).run(
        data, clairvoyant
    )
    # Because of the lag, the position at t uses sign(ret_{t-1}), so it is
    # NOT guaranteed to be profitable. Sharpe should be finite and modest,
    # not the astronomically large value a look-ahead bug would produce.
    assert abs(res.stats["sharpe"]) < 8.0


def test_position_held_matches_lag():
    close = pd.Series(np.linspace(100, 110, 10))
    target = pd.Series([1.0] * 10)
    frame = apply_execution(close, target, CostModel(0, 0), lag=1)
    # First bar cannot be in-position because the signal is lagged.
    assert frame["position"].iloc[0] == 0.0
    assert frame["position"].iloc[1] == 1.0


def test_compare_returns_table():
    data = synthetic_ohlcv(n=600, seed=8)
    bt = Backtester()
    table = bt.compare(
        data,
        {"fast": signals.ma_crossover(5, 20), "slow": signals.ma_crossover(50, 200)},
    )
    assert list(table.index) == ["fast", "slow"]
    assert "sharpe" in table.columns
