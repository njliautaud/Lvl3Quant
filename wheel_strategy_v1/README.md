# wheel_strategy_v1

Income-focused wheel-strategy backtester driven by a Genetic Algorithm.
Implements HC #542: cash-secured puts on names we'd be happy to own; covered
calls on assigned shares. GA produces a Pareto front of (yield, drawdown,
assignment-rate) and we publish Conservative / Balanced / Aggressive picks.

## Components

**data/ingest_universe.py** — Builds the universe of optionable US equities.
Starts with a curated S&P500 list plus ~50 popular optionable mid-caps. Uses
yfinance to validate tickers and pull basic metadata; falls back to the
hardcoded curated list if yfinance is rate-limited.

**data/ingest_fundamentals.py** — Pulls a per-name fundamentals snapshot
(P/E, P/S, FCF yield, debt/equity, gross margin, ROIC, revenue growth, next
earnings date, dividend yield, short interest, beta) via yfinance.Ticker.info.
Cached to parquet so we don't re-hit yfinance on every backtest.

**data/ingest_prices.py** — Daily OHLCV from 2015-present for the universe.
Computes realized vol (20/60/252d), max drawdown, ATR%, gap stats, and last
earnings move. Cached to parquet.

**data/ingest_macro.py** — Macro/regime overlay. NAAIM exposure index,
VIX, VIX3M (term structure), 2s10s yield curve (FRED T10Y2Y), HY OAS (FRED
BAMLH0A0HYM2), DXY. Cached daily 2015-present.

**data/ingest_options.py** — Historical options chains are paid data. For the
backtest we SYNTHESIZE option prices via Black-Scholes using realized
volatility per name with an IV-rank feature derived from rolling 252d realized
vol percentile (a stand-in for true IV rank). Real options ingest is Phase 2.

**backtest/wheel_engine.py** — Simulates wheel mechanics. Sell CSP at target
delta/DTE -> expire worthless (keep premium) or get assigned (now hold 100
shares per contract). Sell CC on assigned shares -> expire worthless (keep
premium + shares) or called away (back to cash). Tracks P&L, premium
captured, assignment count, drawdown, sector concentration.

**backtest/costs.py** — Retail options costs ($0 commission on
Schwab/Robinhood, ~$0.03/contract regulatory fee, $0 assignment fee). All
constants documented inline.

**ga/chromosome.py** — Defines the GA genome: put_delta_target,
call_delta_target, dte_min, dte_max, profit_take_pct, roll_dte_trigger,
max_concurrent_names, sector_cap_pct, vix_max_gate, naaim_min_gate,
fund_score_floor.

**ga/fitness.py** — Fitness = annualized_premium_yield * Sortino /
max(1, max_drawdown_pct), with hard caps on assignment_rate (<= 0.40),
single_name allocation (<= 0.15), and sector allocation (<= chromosome cap).

**ga/run_ga.py** — pyGAD driver. Population 100, generations 50, tournament
selection, single-point crossover, gaussian mutation. Saves Pareto front of
(yield, drawdown, assignment_rate) to results/pareto.parquet.

**report/build_report.py** — Picks 3 representative configs (low-yield
Conservative, mid Balanced, high-yield Aggressive) from the Pareto front and
writes a markdown report with per-tier yield, max DD, worst month, Sortino,
average DTE / delta, total trades, assignment count, and 5 example trades.

## Run order

```
python3 data/ingest_universe.py
python3 data/ingest_prices.py
python3 data/ingest_fundamentals.py
python3 data/ingest_macro.py
python3 data/ingest_options.py     # builds IV-rank features
python3 ga/run_ga.py               # GA search
python3 report/build_report.py     # publish 3-tier report
```

## Constraints (HC #542 R1)

- Free data only (yfinance, FRED, NAAIM website CSV).
- Backtest window 2015-present (>= 10 years, multiple regimes).
- Fitness penalizes tail risk (Sortino + max_DD denominator).
- Hard caps on assignment frequency, sector concentration, max single-name
  allocation.
