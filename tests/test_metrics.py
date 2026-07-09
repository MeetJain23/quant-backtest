"""Unit tests for quantbt.metrics — the numbers everything else reports."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from quantbt import metrics


def test_total_return_compounds():
    r = pd.Series([0.10, 0.10])
    assert metrics.total_return(r) == pytest.approx(0.21)


def test_zero_returns_give_zero_sharpe():
    r = pd.Series([0.0, 0.0, 0.0])
    assert metrics.sharpe(r) == 0.0


def test_sharpe_sign_matches_mean():
    up = pd.Series([0.01, 0.02, 0.015, 0.005])
    down = -up
    assert metrics.sharpe(up) > 0
    assert metrics.sharpe(down) < 0


def test_max_drawdown_is_negative_or_zero():
    r = pd.Series([0.05, -0.10, 0.03, -0.20, 0.04])
    mdd = metrics.max_drawdown(r)
    assert mdd <= 0.0


def test_max_drawdown_known_value():
    # Up 10% then down to a trough: peak equity 1.10, trough 0.88 -> -20%.
    r = pd.Series([0.10, -0.20])
    assert metrics.max_drawdown(r) == pytest.approx(-0.20)


def test_hit_rate_ignores_flat_bars():
    r = pd.Series([0.01, 0.0, -0.01, 0.0, 0.02])
    # Active bars: +, -, +  -> 2/3.
    assert metrics.hit_rate(r) == pytest.approx(2 / 3)


def test_profit_factor_all_gains_is_inf():
    r = pd.Series([0.01, 0.02, 0.03])
    assert metrics.profit_factor(r) == float("inf")


def test_summary_has_all_keys():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0005, 0.01, 500))
    stats = metrics.summary(r)
    expected = {
        "total_return", "cagr", "annual_vol", "sharpe", "sortino",
        "max_drawdown", "calmar", "hit_rate", "profit_factor",
    }
    assert expected <= set(stats)


def test_sortino_at_least_sharpe_for_symmetric():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0.001, 0.01, 1000))
    # With positive drift, downside deviation <= total deviation.
    assert metrics.sortino(r) >= metrics.sharpe(r) - 1e-9
