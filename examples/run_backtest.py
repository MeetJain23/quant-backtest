"""
End-to-end example.

Generates a synthetic price series with a trend-then-reversal regime, runs
three strategies through the engine, prints a comparison table, and saves
the equity / signal / distribution charts for the best performer.

Run from the project root::

    python examples/run_backtest.py
"""

from __future__ import annotations

import os
import sys

# Allow running the script directly from the repo without installing.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from quantbt import Backtester, BacktestConfig, synthetic_ohlcv, plotting, metrics
from strategies import ma_crossover, mean_reversion, momentum


def main() -> None:
    data = synthetic_ohlcv(n=1500, regime_shift=-0.10, seed=7)

    config = BacktestConfig(
        commission_bps=2.0,
        slippage_bps=1.0,
        initial_capital=100_000.0,
        vol_target=0.15,
    )
    bt = Backtester(config)

    strategies = {
        "MA Crossover": ma_crossover.build(20, 50),
        "Mean Reversion": mean_reversion.build(),
        "Momentum": momentum.build(),
    }

    print("=" * 60)
    print("  quantbt — strategy comparison on synthetic data")
    print("=" * 60)

    table = bt.compare(data, strategies)
    with pd.option_context("display.float_format", lambda v: f"{v:7.3f}"):
        print(table[["cagr", "sharpe", "max_drawdown", "hit_rate", "profit_factor"]])

    best_name = table["sharpe"].idxmax()
    print(f"\nBest by Sharpe: {best_name}\n")

    result = bt.run(data, strategies[best_name])
    print(result.report())

    out_dir = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(out_dir, exist_ok=True)
    bench = data["close"].pct_change().fillna(0.0)
    plotting.plot_equity(result, os.path.join(out_dir, "equity.png"), benchmark=bench)
    plotting.plot_signals(data, result.frame["position"],
                          os.path.join(out_dir, "signals.png"))
    plotting.plot_return_distribution(result.net_ret,
                                      os.path.join(out_dir, "returns_hist.png"))
    print(f"\nCharts written to {out_dir}/")


if __name__ == "__main__":
    main()
