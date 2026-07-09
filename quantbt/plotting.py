"""
Matplotlib charts for backtest results.

Charts are the fastest way to catch a bug that the summary stats hide — a
suspiciously smooth equity curve usually means look-ahead has crept in.
Everything here saves to a file so it works headless (CI, servers).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
import pandas as pd

from . import metrics


def plot_equity(result, path: str = "equity.png", benchmark: pd.Series | None = None):
    """Equity curve with the underlying drawdown panel beneath it."""
    frame = result.to_frame()
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 7), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )

    ax1.plot(frame.index, frame["equity"], color="#2563eb", lw=1.6, label="Strategy")
    if benchmark is not None:
        scaled = result.config.initial_capital * (1 + benchmark).cumprod()
        ax1.plot(scaled.index, scaled, color="#94a3b8", lw=1.2, label="Buy & Hold")
    ax1.set_title("Equity Curve")
    ax1.set_ylabel("Portfolio value")
    ax1.legend(loc="upper left", frameon=False)
    ax1.grid(alpha=0.25)

    dd = frame["drawdown"] * 100
    ax2.fill_between(dd.index, dd, 0, color="#dc2626", alpha=0.4)
    ax2.set_ylabel("Drawdown %")
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_signals(data: pd.DataFrame, position: pd.Series, path: str = "signals.png"):
    """Price with long/short shading so you can eyeball the trades."""
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(data.index, data["close"], color="#0f172a", lw=1.0)

    longs = position > 0
    shorts = position < 0
    ax.fill_between(data.index, data["close"].min(), data["close"].max(),
                    where=longs, color="#22c55e", alpha=0.10, label="Long")
    ax.fill_between(data.index, data["close"].min(), data["close"].max(),
                    where=shorts, color="#ef4444", alpha=0.10, label="Short")
    ax.set_title("Price & Positioning")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_return_distribution(returns: pd.Series, path: str = "returns_hist.png"):
    """Histogram of daily returns — check the tails and skew."""
    active = returns[returns != 0]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(active * 100, bins=60, color="#6366f1", alpha=0.8)
    ax.axvline(0, color="#0f172a", lw=1)
    ax.set_title("Daily Return Distribution")
    ax.set_xlabel("Return %")
    ax.set_ylabel("Frequency")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
