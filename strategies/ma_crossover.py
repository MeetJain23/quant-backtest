"""
Moving-average crossover — the canonical trend-following strategy.

Long when a fast SMA sits above a slow SMA. It's the "hello world" of
systematic trading: cheap to compute, easy to reason about, and a useful
baseline that any fancier alpha has to beat after costs.
"""

from __future__ import annotations

from quantbt import signals


def build(fast: int = 20, slow: int = 50):
    """Return a ready-to-run MA-crossover signal."""
    if fast >= slow:
        raise ValueError("fast window must be shorter than slow window")
    return signals.ma_crossover(fast=fast, slow=slow)


# A small parameter grid that walk-forward can search over.
PARAM_GRID = {
    "fast": [10, 20, 30],
    "slow": [50, 100, 200],
}
