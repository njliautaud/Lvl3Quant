# Wheel Expansion — QQQ & IWM, Tier2 Balanced Scalp

Generated: 2026-06-09 16:35:17
Window: 2018-01-01 to 2025-12-31
Starting cash: $20,000, leverage 1.0x, single-underlying

Config: put_delta 0.22, call_delta 0.22, DTE 30-45, profit-take 50%, roll DTE<10, VIX gate 32.0

Pricing: Black-Scholes with modeled ATM sigma (iv_features.parquet). 
Slippage: max(2.5% of premium, $0.03/share/leg). Commission: $0.65/contract/leg.

Regime classification: SPY close-to-close, threshold ±0.5σ.

Note on chain data: live-market option chains for 2018-2025 on QQQ/IWM 
are not in the local cache. This run uses delta-replicated synthetic 
chains (BS at the modeled ATM sigma) per the task fallback. Headline 
metrics here represent a first-look estimate, not vendor-priced fills.


## Headline Metrics

| Ticker | CAGR | Sharpe | Sortino | MaxDD | Calmar | WR | PF | Total Ret | Final $ |
|---|---|---|---|---|---|---|---|---|---|
| QQQ | 20.2% | 1.87 | 1.78 | -18.1% | 1.11 | 64.7% | 1.51 | 335.1% | $87,268 |
| IWM | 18.8% | 1.89 | 1.96 | -21.5% | 0.87 | 60.2% | 1.49 | 295.2% | $79,267 |

## Regime Gate (HC #428 R1)

| Ticker | n_green | n_red | n_flat | Sh green | Sh red | Sh flat | gap | pass |
|---|---|---|---|---|---|---|---|---|
| QQQ | 534 | 406 | 1070 | 15.19 | -11.95 | 6.39 | 1.79 | FAIL |
| IWM | 534 | 406 | 1070 | 15.64 | -11.77 | 5.05 | 1.75 | FAIL |

## Deploy Gates

| Ticker | Sharpe≥1.0 | Calmar≥1.5 | Regime gap≤0.50 | Day conc≤0.70 | n_days≥40 | DEPLOY READY |
|---|---|---|---|---|---|---|
| QQQ | PASS | FAIL | FAIL | PASS | PASS | **NO** |
| IWM | PASS | FAIL | FAIL | PASS | PASS | **NO** |

## Tail Event Stress

| Ticker | Event | MaxDD | VIX peak | Recover days | Cum ret |
|---|---|---|---|---|---|
| QQQ | COVID_2020 | -18.1% | 82.7 | 141 | -7.7% |
| QQQ | 2022_bear | -8.1% | 36.5 | 40 | 4.8% |
| QQQ | Aug_2024_carry | -7.4% | 38.6 | 60 | -1.4% |
| IWM | COVID_2020 | -21.5% | 82.7 | 245 | -15.3% |
| IWM | 2022_bear | -6.6% | 36.5 | 0 | 19.6% |
| IWM | Aug_2024_carry | -4.6% | 38.6 | 12 | 3.3% |

## Recommendation

- **QQQ: SHELVE** — fails: calmar_ge_1.5, regime_gap_le_0.50.
- **IWM: SHELVE** — fails: calmar_ge_1.5, regime_gap_le_0.50.

Neither ticker passes all deploy gates with synthetic-chain pricing. Before final reject, re-run with vendor option chains (Polygon/CBOE) so spreads/IV skew aren't approximated.

## Caveats

- Pricing is BS with modeled ATM sigma — no real bid/ask, no skew. 
Real fills will differ; expect ~5-15% premium haircut on QQQ/IWM at 30-45 DTE 22Δ.
- The original SPY t=7.45 alpha figure was from a MULTI-NAME tier ladder backtest (~329 names, full IV-rank gating). Comparing single-underlying ETF wheels to that 
number is apples-to-oranges; this report's purpose is a relative QQQ-vs-IWM-vs-SPY check.
- VIX gate at 32 means the strategy STOPS opening new positions during high-vol — 
this is what produces the regime gap (most red days have elevated VIX).