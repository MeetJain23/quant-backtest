# quantbt

A compact, **vectorized backtesting and research library** for systematic
trading strategies. Built to do the boring-but-critical things most toy
backtests skip: transaction costs, slippage, look-ahead protection,
walk-forward validation, and Monte Carlo robustness checks.

No external data feed required — a reproducible synthetic OHLCV generator
lets you develop and stress-test strategies offline, and a CSV loader
drops in real NSE/BSE (or any) data when you're ready.

---

## Why this exists

It is trivially easy to build a backtest that looks profitable and is
completely fake. The three usual culprits:

1. **Look-ahead bias** — acting on information from the same bar the signal
   is computed on.
2. **Ignoring costs** — commission + slippage quietly turn many "edges"
   negative once you pay to trade.
3. **Overfitting** — one great in-sample number tuned to one slice of
   history tells you nothing about the future.

`quantbt` is structured so each of these is handled by default: signals are
lagged one bar before they touch P&L, a cost model charges every unit of
turnover, and the `walkforward` + `montecarlo` modules give you honest
out-of-sample numbers with confidence intervals.

---

## Install

```bash
git clone https://github.com/<you>/quantbt.git
cd quantbt
pip install -r requirements.txt
```

## Quick start

```python
from quantbt import Backtester, BacktestConfig, synthetic_ohlcv, signals

data = synthetic_ohlcv(n=1500, regime_shift=-0.10, seed=7)

config = BacktestConfig(commission_bps=2, slippage_bps=1, vol_target=0.15)
result = Backtester(config).run(data, signals.ma_crossover(20, 50))

print(result.report())
```

Run the bundled examples end-to-end:

```bash
python examples/run_backtest.py           # compare three strategies + charts
python examples/research_walkforward.py   # walk-forward + bootstrap CIs
pytest -q                                 # 14 tests
```

---

## Architecture

```
quantbt/
├── data.py          synthetic OHLCV generator + CSV loader + splits
├── indicators.py    SMA/EMA/RSI/MACD/Bollinger/ATR/Donchian/z-score
├── signals.py       strategy signals + ensemble voting (combine)
├── portfolio.py     execution: 1-bar lag, costs, slippage, vol targeting
├── metrics.py       CAGR, Sharpe, Sortino, Calmar, drawdown, hit rate...
├── engine.py        Backtester — vectorized run() + compare()
├── walkforward.py   rolling train/test with per-fold re-optimisation
├── montecarlo.py    block-bootstrap CIs + permutation significance test
└── plotting.py      equity curve, drawdown, positioning, return dist.
strategies/          ready-made strategies with parameter grids
examples/            runnable demos
tests/               pytest suite
```

The engine is fully vectorized (no per-bar Python loop), so a
multi-thousand-bar backtest runs in milliseconds — which is what makes the
parameter sweeps in walk-forward and the thousands of Monte Carlo paths
practical.

---

## Metrics reported

| Metric | Meaning |
|---|---|
| CAGR | compound annual growth rate |
| Sharpe / Sortino | risk-adjusted return (total vol / downside vol) |
| Max drawdown | worst peak-to-trough decline |
| Calmar | CAGR per unit of max drawdown |
| Hit rate | fraction of active bars that were positive |
| Profit factor | gross profit ÷ gross loss |

---

## Roadmap

- Multi-asset portfolio backtests with correlation-aware sizing
- Intraday bar support and session handling
- Pluggable data adapters (yfinance, broker exports)
- HTML tearsheet export

## License

MIT — see [LICENSE](LICENSE).
