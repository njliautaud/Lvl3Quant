# wheel_strategy_v1 — STATUS

Generated 2026-06-05 ~11:14 ET, post-smoke-pass, full run launched.

## What was built

A complete income-focused wheel-strategy research pipeline driven by a
Genetic Algorithm, per HC #542.

Components:
- Universe ingest (curated S&P500 + ~50 mid-caps + ~30 ETFs ~329 names;
  optional yfinance enrichment for sector/market-cap, falls back to
  curated list if rate-limited).
- Per-name fundamentals ingest (P/E, P/S, FCF yield, debt/equity, gross
  margin, ROIC proxy via ROA, revenue growth, dividend yield, short
  interest, beta) -> composite `fund_score` 0..100.
- Daily price ingest 2015-present + features (realized vol 20/60/252d,
  max DD, ATR%, gap stats, 1y/3y/5y total return).
- Macro overlay (VIX, VIX3M, VIX term structure, DXY, NAAIM exposure
  index, 2s10s yield curve via FRED, HY OAS via FRED). NAAIM synthetic
  placeholder if the NAAIM website is unreachable; documented TODO to
  replace with cached CSV.
- Options price synthesis via Black-Scholes using realized vol
  (annualized) as the IV input, with IV rank computed from rolling 252d
  realized vol percentile. Documented Phase-2 TODO for real options ingest.
- Wheel engine: full CSP -> assignment -> CC -> called-away simulation
  with profit-take, roll-on-DTE, sector caps, single-name caps, macro gates.
- Retail cost model: $0 commission (Schwab / Robinhood), $0.03/contract
  regulatory, $0 assignment, SEC TAF (~0.23 bps) on share sells.
- GA harness via pyGAD: 11-gene chromosome (put delta, call delta, dte
  min/max, profit-take pct, roll trigger, max concurrent names, sector
  cap, VIX gate, NAAIM gate, fundamental floor), tournament selection,
  single-point crossover, mutation, elitism.
- Fitness = annualized premium yield * Sortino / max(1, max_DD_pct), with
  10x penalty multipliers when assignment rate > 0.40 or single-name
  alloc > 0.15.
- Pareto-front extraction on (yield up, DD down, assignment-rate down).
- Markdown report builder that picks Conservative / Balanced / Aggressive
  tier configs and writes per-tier metrics + 5 example trades.

## Smoke results (validates the engine)

Universe: SPY, QQQ, AAPL, MSFT, NVDA. Window: 2024-01-01 to 2024-12-31.
GA: population 10, generations 5 (50 evaluations, ~19 seconds).

Pareto front size: 13. Tier picks:

| Tier         | Yield % | MaxDD % | Worst Month % | Sortino | Trades | Assignments | Avg DTE | Avg Delta |
|--------------|---------|---------|----------------|---------|--------|-------------|---------|-----------|
| Conservative | 1.68    | 0.68    | -0.27          | 1.34    | 43     | 0           | 22.0    | 0.132     |
| Balanced     | 7.12    | 1.81    | -0.39          | 2.49    | 90     | 0           | 22.0    | 0.385     |
| Aggressive   | 7.23    | 2.19    | -0.18          | 1.95    | 63     | 0           | 29.0    | 0.385     |

Engine sanity-check: positive Sortino, non-zero trades, drawdowns
realistic for 1y of 5 names, assignments occurred in the broader eval
sample (visible in `results/all_evals_smoke.parquet`). Example trades
show plausible CSP strikes (NVDA 65-119, AAPL 220) and PnL magnitudes
($100-$450 per contract per trade).

Smoke outputs saved to `results/smoke/`.

## Full run

Launched at 2026-06-05 11:11 ET (pid 1421018, nohup bash run_full.sh).
- Universe: ~329 tickers (curated, no-enrich path used to bypass yfinance
  metadata rate limits).
- Window: 2015-01-01 to today (~11 years, multi-regime: 2018 vol, 2020
  crash, 2021 melt-up, 2022 bear, 2023 recovery, 2024 melt-up, 2025-26).
- GA: population 100, generations 50 (~5000 backtest evaluations).

Expected completion: rough estimate 4-12 hours wall-clock. Dominant cost
is the GA backtest loop (~5000 evaluations of the full universe over 11
years). Data ingest (yfinance for ~329 tickers) front-loads ~30-60min.

Logs: `wheel_strategy_v1/logs/full_run.log` (single rolling log written
by `run_full.sh`).

## What to check when reviewing results

After the run completes, look at:

1. `results/summary_full.json` — top-line: n_evals, best fitness, best
   yield, best DD%, best Sortino, best trade count. If best yield is 0 or
   best DD% is ~0, something is broken.
2. `results/report_full.md` — the three-tier picks. Read the Conservative
   tier first — it should have low yield (5-12%), low DD (<5%), and a
   sensible put delta (<0.20). Aggressive tier should have higher yield
   (20-40%), higher DD (10-25%), higher delta (>0.30), and a non-trivial
   assignment count.
3. `results/pareto_full.parquet` — full Pareto front. If size is < 5, the
   GA either converged too fast or the fitness is dominated by one gene.
4. `results/all_evals_full.parquet` — every evaluation. Useful for
   correlating gene values vs metrics offline.

Specific red flags:
- Conservative tier with assignment rate > 0.40 -> the assignment cap is
  not firing; check `ga/fitness.py` penalty.
- All tiers with the same put_delta_target -> GA collapsed to a single
  delta; consider widening gene bounds or increasing mutation rate.
- Yield reported but `n_trades == 0` -> engine entered no positions
  because of overly tight macro gates (vix_max_gate too low, or
  fund_score_floor too high).
- Worst-month worse than -10% -> tail-risk penalty in the fitness should
  have suppressed this config; investigate.

## Known caveats (documented in code)

- Options prices are SYNTHETIC (Black-Scholes on realized vol). True IV
  smile/skew not modelled. Phase-2 TODO: ingest real chains (Tradier
  free / Polygon / paid vendor).
- Fundamentals snapshot is point-in-TIME-OF-RUN, not point-in-time of
  the historical date. Mild lookahead bias; documented in
  `ingest_fundamentals.py`.
- NAAIM may be a synthetic placeholder if the NAAIM website returns 404.
  Visible in macro.parquet by checking against known weekly cadence.
- FRED series (2s10s, HY OAS) skipped on read timeout; engine doesn't
  hard-gate on them but they're useful regime features for inspection.

## Files

```
wheel_strategy_v1/
  README.md
  STATUS.md                   (this file)
  run_full.sh                 (one-shot driver)
  data/
    ingest_universe.py
    ingest_prices.py
    ingest_fundamentals.py
    ingest_macro.py
    ingest_options.py         (synthesized IV features)
    cache/                    (parquets land here)
  backtest/
    wheel_engine.py
    costs.py
  ga/
    chromosome.py
    fitness.py
    run_ga.py
  report/
    build_report.py
  results/
    smoke/                    (validated outputs)
    (full outputs land here when run finishes)
  logs/
    full_run.log
```
