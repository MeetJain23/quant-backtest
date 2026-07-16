"""
Monte Carlo robustness checks.

Two complementary bootstraps:

* :func:`bootstrap_returns` — resample the strategy's realised returns to
  build a distribution of terminal equity and drawdown. This answers
  "how much of my result was luck in the *ordering* of trades?".
* :func:`permutation_pvalue` — shuffle the sign of the position stream to
  test whether the strategy's edge is distinguishable from random
  positioning on the same price path (a rough significance test).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import metrics


def bootstrap_returns(
    returns: pd.Series,
    n_paths: int = 2000,
    block: int = 5,
    seed: int | None = 7,
) -> pd.DataFrame:
    """Stationary block-bootstrap of a return series.

    Sampling in blocks preserves short-term autocorrelation, which an
    i.i.d. bootstrap would destroy. Returns a frame of summary stats, one
    row per simulated path.
    """
    rng = np.random.default_rng(seed)
    values = returns.to_numpy()
    n = len(values)
    if n == 0:
        return pd.DataFrame()

    n_blocks = int(np.ceil(n / block))
    rows = []
    for _ in range(n_paths):
        starts = rng.integers(0, n, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block) % n for s in starts])[:n]
        sample = pd.Series(values[idx])
        rows.append(
            {
                "total_return": metrics.total_return(sample),
                "sharpe": metrics.sharpe(sample),
                "max_drawdown": metrics.max_drawdown(sample),
            }
        )
    return pd.DataFrame(rows)


def confidence_interval(bootstrapped: pd.DataFrame, metric: str, alpha: float = 0.05):
    """Percentile confidence interval for a bootstrapped metric."""
    lo = bootstrapped[metric].quantile(alpha / 2)
    hi = bootstrapped[metric].quantile(1 - alpha / 2)
    return float(lo), float(hi)


def permutation_pvalue(
    net_ret: pd.Series,
    position: pd.Series,
    asset_ret: pd.Series,
    n_perms: int = 1000,
    seed: int | None = 11,
) -> float:
    """One-sided p-value that the strategy Sharpe beats random positioning.

    Keeps the price path fixed and randomly permutes the held positions,
    recomputing the Sharpe each time. The p-value is the fraction of
    permutations whose Sharpe matches or exceeds the real one.
    """
    rng = np.random.default_rng(seed)
    observed = metrics.sharpe(net_ret)
    pos = position.to_numpy()
    ar = asset_ret.to_numpy()

    count = 0
    for _ in range(n_perms):
        shuffled = rng.permutation(pos)
        perm_ret = pd.Series(shuffled * ar)
        if metrics.sharpe(perm_ret) >= observed:
            count += 1
    return (count + 1) / (n_perms + 1)
