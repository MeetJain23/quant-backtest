"""
Data layer for quantbt.

Two entry points:

* ``synthetic_ohlcv`` — build a reproducible OHLCV series from a
  geometric-Brownian-motion price path with an optional regime drift, so
  strategies can be developed and tested without any external data feed.
* ``load_ohlcv`` — read a CSV of real market data (e.g. an NSE/BSE export
  or a yfinance dump) into the canonical column layout the engine expects.

The canonical frame has a ``DatetimeIndex`` and the columns
``open, high, low, close, volume``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CANONICAL_COLUMNS = ["open", "high", "low", "close", "volume"]


def synthetic_ohlcv(
    n: int = 1000,
    start: str = "2020-01-01",
    freq: str = "B",
    s0: float = 100.0,
    mu: float = 0.08,
    sigma: float = 0.20,
    regime_shift: float | None = None,
    seed: int | None = 42,
) -> pd.DataFrame:
    """Generate a reproducible OHLCV frame from a GBM close path.

    Parameters
    ----------
    n : number of bars.
    start : first bar timestamp.
    freq : pandas offset alias ("B" = business days, "D", "h", ...).
    s0 : initial price.
    mu : annualised drift.
    sigma : annualised volatility.
    regime_shift : if given, drift flips sign halfway through, creating a
        trend-then-reversal regime that is useful for stress-testing
        trend-following vs mean-reversion strategies.
    seed : RNG seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / 252.0
    drift = np.full(n, mu)
    if regime_shift is not None:
        drift[n // 2:] = regime_shift

    shocks = rng.normal(0.0, 1.0, size=n)
    log_returns = (drift - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * shocks
    close = s0 * np.exp(np.cumsum(log_returns))

    # Build a plausible OHLC around the close path.
    intrabar = np.abs(rng.normal(0.0, sigma * np.sqrt(dt) * 0.6, size=n)) * close
    open_ = np.empty(n)
    open_[0] = s0
    open_[1:] = close[:-1]
    high = np.maximum(open_, close) + intrabar
    low = np.minimum(open_, close) - intrabar
    volume = rng.integers(1_000, 50_000, size=n).astype(float)

    index = pd.date_range(start=start, periods=n, freq=freq, name="date")
    frame = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    )
    return frame[CANONICAL_COLUMNS]


def load_ohlcv(path: str, date_col: str = "Date") -> pd.DataFrame:
    """Load a CSV into the canonical OHLCV layout.

    Column matching is case-insensitive, so exports that use ``Open``,
    ``OPEN`` or ``open`` all work. A missing ``volume`` column is filled
    with NaN rather than raising, since some feeds omit it.
    """
    raw = pd.read_csv(path)
    lower = {c.lower(): c for c in raw.columns}

    if date_col.lower() not in lower:
        raise ValueError(f"date column {date_col!r} not found in {list(raw.columns)}")
    raw[lower[date_col.lower()]] = pd.to_datetime(raw[lower[date_col.lower()]])
    raw = raw.set_index(lower[date_col.lower()]).sort_index()
    raw.index.name = "date"

    out = pd.DataFrame(index=raw.index)
    for col in CANONICAL_COLUMNS:
        if col in lower:
            out[col] = pd.to_numeric(raw[lower[col]], errors="coerce")
        elif col == "volume":
            out[col] = np.nan
        else:
            raise ValueError(f"required column {col!r} missing from {path}")
    return out.dropna(subset=["close"])


def train_test_split(frame: pd.DataFrame, train_frac: float = 0.7):
    """Chronological split — never shuffle time series."""
    if not 0 < train_frac < 1:
        raise ValueError("train_frac must be in (0, 1)")
    cut = int(len(frame) * train_frac)
    return frame.iloc[:cut].copy(), frame.iloc[cut:].copy()
