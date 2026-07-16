"""
Walk-forward analysis.

A single in-sample backtest number is close to meaningless — it's trivial
to overfit parameters to one slice of history. Walk-forward analysis
splits the series into consecutive train/test windows, (optionally)
re-optimises parameters on each train window, and stitches together the
*out-of-sample* test returns. The concatenated OOS curve is a far more
honest estimate of how the strategy would have traded live.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from .engine import Backtester, BacktestConfig
from . import metrics

SignalFactory = Callable[..., Callable[[pd.DataFrame], pd.Series]]


@dataclass
class WalkForwardResult:
    oos_returns: pd.Series
    fold_stats: pd.DataFrame
    chosen_params: list[dict]

    @property
    def summary(self) -> dict:
        return metrics.summary(self.oos_returns)


def _param_grid(grid: dict[str, Sequence]) -> list[dict]:
    keys = list(grid)
    return [dict(zip(keys, combo)) for combo in product(*(grid[k] for k in keys))]


def walk_forward(
    data: pd.DataFrame,
    factory: SignalFactory,
    param_grid: dict[str, Sequence],
    n_folds: int = 5,
    train_frac: float = 0.6,
    config: BacktestConfig | None = None,
    objective: str = "sharpe",
) -> WalkForwardResult:
    """Rolling train/test walk-forward with per-fold re-optimisation.

    Each fold trains on ``train_frac`` of the window, picks the parameter
    set that maximises ``objective`` in-sample, then evaluates it on the
    held-out remainder. Only the held-out returns are kept.
    """
    bt = Backtester(config)
    grid = _param_grid(param_grid)
    fold_size = len(data) // n_folds

    oos_chunks: list[pd.Series] = []
    rows = []
    chosen: list[dict] = []

    for fold in range(n_folds):
        start = fold * fold_size
        end = len(data) if fold == n_folds - 1 else (fold + 1) * fold_size
        window = data.iloc[start:end]
        if len(window) < 30:
            continue

        cut = int(len(window) * train_frac)
        train, test = window.iloc[:cut], window.iloc[cut:]
        if len(test) < 5:
            continue

        best_score, best_params = -np.inf, grid[0]
        for params in grid:
            res = bt.run(train, factory(**params))
            score = res.stats.get(objective, -np.inf)
            if np.isfinite(score) and score > best_score:
                best_score, best_params = score, params

        test_res = bt.run(test, factory(**best_params))
        oos_chunks.append(test_res.net_ret)
        chosen.append(best_params)
        rows.append(
            {
                "fold": fold,
                "train_start": train.index[0].date(),
                "test_start": test.index[0].date(),
                "is_score": round(best_score, 3),
                "oos_sharpe": round(test_res.stats["sharpe"], 3),
                "oos_return": round(test_res.stats["total_return"], 4),
                **best_params,
            }
        )

    oos = pd.concat(oos_chunks) if oos_chunks else pd.Series(dtype=float)
    return WalkForwardResult(oos, pd.DataFrame(rows), chosen)
