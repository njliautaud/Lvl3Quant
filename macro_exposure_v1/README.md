# macro_exposure_v1

GA-driven macro-exposure timing strategy. Goes long / flat / short a basket of
broad-market ETFs (SPY, QQQ, IWM) based on macro regime signals. Daily-or-weekly
cadence. NOT scalping. Per DIRECTIVES.md HC #543 R2 (2026-06-05).

## Design

```
data/        ingest_market.py        -> data/cache/market.parquet     (daily OHLCV, 2010+)
             ingest_macro_features.py -> data/cache/macro_features.parquet
             ingest_sentiment.py     -> data/cache/sentiment.parquet  (NAAIM weekly, AAII TODO)
             ingest_breadth.py       -> data/cache/breadth.parquet    (% S&P > 200DMA)

backtest/    exposure_engine.py      -> daily PnL given allocation signal + basket weights
                                        Costs: $0 commission, 1bp slippage on rebalance days

ga/          chromosome.py           -> feature weights, thresholds, basket mix, leverage, cadence
             fitness.py              -> CAGR * Sortino / (1 + max_DD_pct)
                                        Penalty: -50% if max_DD > 25%, -25% if turnover > 12/yr
             run_ga.py               -> pyGAD, pop=80, gen=40

report/      build_report.py         -> 3 tier picks (Conservative / Balanced / Aggressive)
                                        markdown + equity curves
```

## Universe (small-set yfinance, ~15 tickers — safe to co-run later)

SPY, QQQ, IWM, ^VIX, ^VIX3M, ^VIX9D, GLD, GC=F, HG=F, DX-Y.NYB, ^TNX, ^IRX, ^FVX (5y), ^TYX (30y).

## Feature stack (GA-evolvable weights)

- NAAIM exposure (weekly, ffilled)
- VIX 20d percentile
- VIX term structure slope (VIX3M / VIX)
- VIX9D / VIX (front-end skew, TODO if VIX9D unavailable)
- AAII bull-bear (TODO — sentiment.parquet)
- SPY momentum 3m / 6m / 12m
- SPY distance to 50/100/200 DMA
- 52-week-high distance
- Yield curve 2s10s proxy (TNX - FVX, or TNX - IRX if 5y missing)
- DXY trend (20d slope)
- Gold/copper ratio (GLD / HG=F)
- % S&P 500 above 200DMA (breadth)
- ATR % of price (vol regime)

## Costs

- Commission: $0 (Alpaca / Robinhood / IBKR Lite — SPY/QQQ/IWM, per HC #541 R1)
- Slippage: 1 bp per rebalance leg traded (notional × 0.0001)
- No borrow fee modeled for shorts (TODO: add SPY short cost ~25–40 bps annualized when going to prod)

## Backtest window

2010-01-01 to today. Covers 2011 correction, 2013 taper, 2015 chop, 2018 vol,
2020 crash, 2022 bear, 2024 melt-up. Multi-regime by construction (HC #428 R1).

## Reuse from wheel_strategy_v1

- `wheel_strategy_v1/data/cache/macro.parquet` — VIX, VIX3M, DXY, NAAIM, 2s10s (when present)
- `wheel_strategy_v1/data/cache/prices.parquet` — 329 large-cap names for breadth %-above-200DMA

The wheel pipeline is running yfinance pulls for ~329 names right now. We do
NOT re-pull that data. We add a ~15-ticker incremental pull for the macro
indices and ETFs not already in the wheel cache.

## Run

```
# Smoke (already done at build time):
python3 data/ingest_market.py --smoke
python3 data/ingest_macro_features.py --smoke
python3 data/ingest_sentiment.py --smoke
python3 data/ingest_breadth.py --smoke
python3 ga/run_ga.py --smoke
python3 report/build_report.py --smoke

# Full (DO NOT LAUNCH until wheel_strategy_v1 ingest is done):
./run_full.sh
```
