"""
Robustness research example.

Shows the workflow you'd actually trust before risking capital:

1. Walk-forward analysis with per-fold parameter re-optimisation, so the
   reported numbers are out-of-sample.
2. A block-bootstrap of the OOS returns to put a confidence interval
   around the Sharpe and worst-case drawdown.

Run::

    python examples/research_walkforward.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantbt import synthetic_ohlcv, metrics
from quantbt.walkforward import walk_forward
from quantbt.montecarlo import bootstrap_returns, confidence_interval
from strategies import ma_crossover


def main() -> None:
    data = synthetic_ohlcv(n=2500, mu=0.10, sigma=0.22, seed=3)

    print("Running walk-forward on MA crossover (re-optimised each fold)...\n")
    wf = walk_forward(
        data,
        factory=ma_crossover.build,
        param_grid=ma_crossover.PARAM_GRID,
        n_folds=6,
        train_frac=0.6,
        objective="sharpe",
    )

    print(wf.fold_stats.to_string(index=False))
    print("\nStitched out-of-sample performance:")
    print(metrics.format_summary(wf.summary))

    print("\nBootstrapping the OOS return stream (2000 paths)...")
    booted = bootstrap_returns(wf.oos_returns, n_paths=2000, block=5)
    lo_s, hi_s = confidence_interval(booted, "sharpe")
    lo_d, hi_d = confidence_interval(booted, "max_drawdown")
    print(f"  Sharpe 95% CI       [{lo_s:6.3f}, {hi_s:6.3f}]")
    print(f"  Max drawdown 95% CI [{lo_d*100:6.2f}%, {hi_d*100:6.2f}%]")


if __name__ == "__main__":
    main()
