# macro_exposure_v1 — STATUS

Per DIRECTIVES.md HC #543 R2 (2026-06-05).

## Built

- Full pipeline scaffolded: data ingest (market / macro features / sentiment / breadth), backtest engine (daily exposure, $0 commission, 1bp slippage), GA chromosome (feature weights + thresholds + basket + leverage + cadence), pyGAD driver, tiered report builder.
- Reuses `wheel_strategy_v1/data/cache/macro.parquet` for NAAIM (no duplicate yfinance hit) and `wheel_strategy_v1/data/cache/prices.parquet` (or smoke variant) for breadth.

## Smoke test (2024 window, pop=10 gen=5)

- All 14 macro tickers fetched cleanly from yfinance.
- 55 GA evals in ~2 seconds. 19 configs on the Pareto front.
- Best config: CAGR 28%, max DD 14.6%, Sortino 1.85, turnover ~5/yr.
- Tier report generated with 3 tiers + equity-curve PNGs.

## Full run — NOT LAUNCHED YET

The wheel_strategy_v1 pipeline is currently mid-ingest, hitting yfinance for ~329 stocks (2015-present). Running both pipelines simultaneously would rate-limit both.

### How to launch the full run

1. Watch `wheel_strategy_v1/logs/full_run.log` for the line containing `ingest_fundamentals done` or the start of `ingest_options`. After that the heavy yfinance pulls are done.
2. Then:

```
cd /home/jupiter/Lvl3Quant/macro_exposure_v1
nohup ./run_full.sh > logs/nohup.log 2>&1 &
```

The full run does:
- ingest_market (2010-01-01 → today, 14 tickers, small load)
- ingest_macro_features (no network — derived from market.parquet)
- ingest_sentiment (reuses wheel NAAIM cache; falls back to live or synthetic)
- ingest_breadth (reads wheel prices.parquet — no network)
- run_ga.py --pop 80 --gen 40 (≈ 3200 evals, expect ~20–60 min on Jupiter CPU)
- build_report.py (markdown + PNGs)

Outputs land in `results/full/`.

## Open TODOs (non-blocking)

- AAII bull-bear: no free clean source — column stays NaN, weight gene contributes nothing.
- Put/call ratio: same — TODO real source.
- Short borrow fee for short legs: not modeled in v1. Add ~25–40 bps annualized before any live deployment.
- True 2s10s uses TNX – FVX (5y) as a proxy; swap to FRED DGS2 in the next pass if desired.
