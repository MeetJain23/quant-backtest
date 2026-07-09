"""
Momentum / breakout ensemble.

Combines a Donchian channel breakout with a MACD trend filter using a
majority vote. The breakout supplies the entry impulse; the MACD filter
keeps you out of chop by requiring the broader trend to agree.
"""

from __future__ import annotations

from quantbt import signals


def build(breakout_window: int = 20, macd_fast: int = 12, macd_slow: int = 26):
    return signals.combine(
        [
            signals.donchian_breakout(window=breakout_window),
            signals.macd_trend(fast=macd_fast, slow=macd_slow),
        ],
        mode="majority",
    )


PARAM_GRID = {
    "breakout_window": [10, 20, 55],
    "macd_fast": [8, 12],
}
