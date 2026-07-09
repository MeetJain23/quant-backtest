"""
Mean-reversion ensemble.

Blends an RSI reversion signal with a Bollinger-band reversion signal and
requires them to agree (consensus) before taking a position. Demanding
agreement cuts the trade count sharply but tends to lift the hit rate — a
classic precision-vs-recall trade-off applied to signals.
"""

from __future__ import annotations

from quantbt import signals


def build(rsi_window: int = 14, bb_window: int = 20, n_std: float = 2.0):
    return signals.combine(
        [
            signals.rsi_reversion(window=rsi_window),
            signals.bollinger_reversion(window=bb_window, n_std=n_std),
        ],
        mode="consensus",
    )


PARAM_GRID = {
    "rsi_window": [7, 14, 21],
    "bb_window": [15, 20, 30],
}
