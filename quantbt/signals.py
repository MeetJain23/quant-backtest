"""
Signal generators.

A *signal* maps an OHLCV frame to a target-position ``Series`` in
``{-1, 0, +1}`` (short / flat / long). The engine is responsible for
lagging the signal by one bar before it touches P&L, so signals here are
allowed to be computed on the close of bar *t*; they simply must not use
any information from *t+1* onward.

Combining signals is handled by :func:`combine`, which lets you build an
ensemble from several sub-signals with consensus / majority / weighted
voting — the kind of thing you'd do when blending independent alphas.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import pandas as pd

from . import indicators as ind

Signal = Callable[[pd.DataFrame], pd.Series]


def ma_crossover(fast: int = 20, slow: int = 50) -> Signal:
    """Long when the fast MA is above the slow MA, short otherwise."""

    def _signal(df: pd.DataFrame) -> pd.Series:
        f = ind.sma(df["close"], fast)
        s = ind.sma(df["close"], slow)
        pos = np.where(f > s, 1.0, -1.0)
        return pd.Series(pos, index=df.index).where(s.notna(), 0.0)

    return _signal


def rsi_reversion(window: int = 14, low: float = 30.0, high: float = 70.0) -> Signal:
    """Buy oversold, sell overbought, hold the position in between."""

    def _signal(df: pd.DataFrame) -> pd.Series:
        r = ind.rsi(df["close"], window)
        raw = pd.Series(np.nan, index=df.index)
        raw[r < low] = 1.0
        raw[r > high] = -1.0
        return raw.ffill().fillna(0.0)

    return _signal


def bollinger_reversion(window: int = 20, n_std: float = 2.0) -> Signal:
    """Fade band touches: long below the lower band, short above the upper."""

    def _signal(df: pd.DataFrame) -> pd.Series:
        _, upper, lower = ind.bollinger(df["close"], window, n_std)
        raw = pd.Series(np.nan, index=df.index)
        raw[df["close"] < lower] = 1.0
        raw[df["close"] > upper] = -1.0
        return raw.ffill().fillna(0.0)

    return _signal


def donchian_breakout(window: int = 20) -> Signal:
    """Classic turtle-style breakout of the N-bar high/low."""

    def _signal(df: pd.DataFrame) -> pd.Series:
        upper, lower = ind.donchian(df["high"], df["low"], window)
        raw = pd.Series(np.nan, index=df.index)
        raw[df["close"] >= upper.shift(1)] = 1.0
        raw[df["close"] <= lower.shift(1)] = -1.0
        return raw.ffill().fillna(0.0)

    return _signal


def macd_trend(fast: int = 12, slow: int = 26, signal: int = 9) -> Signal:
    """Long while the MACD histogram is positive."""

    def _signal(df: pd.DataFrame) -> pd.Series:
        _, _, hist = ind.macd(df["close"], fast, slow, signal)
        return pd.Series(np.where(hist > 0, 1.0, -1.0), index=df.index).where(
            hist.notna(), 0.0
        )

    return _signal


def combine(
    subsignals: Sequence[Signal],
    mode: str = "majority",
    weights: Sequence[float] | None = None,
) -> Signal:
    """Blend several signals into one.

    mode:
        ``consensus`` — take a position only when every sub-signal agrees.
        ``majority``  — sign of the (equal-weighted) vote.
        ``weighted``  — sign of the weighted vote (needs ``weights``).
    """
    if mode == "weighted" and weights is None:
        raise ValueError("weighted mode requires weights")

    def _signal(df: pd.DataFrame) -> pd.Series:
        votes = pd.concat([s(df) for s in subsignals], axis=1)
        if mode == "consensus":
            agree = votes.apply(lambda row: row.nunique() == 1, axis=1)
            return votes.iloc[:, 0].where(agree, 0.0)
        if mode == "majority":
            return np.sign(votes.sum(axis=1))
        if mode == "weighted":
            w = np.asarray(weights, dtype=float)
            return np.sign(votes.mul(w, axis=1).sum(axis=1))
        raise ValueError(f"unknown combine mode {mode!r}")

    return _signal
