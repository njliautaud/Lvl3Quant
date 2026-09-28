# Wheel + Regime-Gated Entries v1 - SPY Tier2 Balanced Scalp

Generated: 2026-06-09 16:48:03
Window: 2018-01-01 to 2025-12-31
Starting cash: $50,000, leverage 1.0x, SPY only
Wheel cfg: put_delta 0.22, call_delta 0.22, DTE 30-45, profit-take 50%, roll DTE<=10, VIX gate 32.0

All variants gate NEW short-put ENTRIES only. Existing positions manage/exit normally regardless of gate state.

Entry-gate variants:
- **TREND_ONLY**: SPY > 50d MA
- **TREND_VOL**: SPY > 50d MA AND VIX < 20
- **TIGHT_TREND_VOL**: SPY > 20d MA AND SPY > 50d MA AND VIX < 18
- **RV_GATE**: 20d realized vol < 15% AND SPY > 50d MA
- **DD_FROM_HIGH**: SPY within 5% of trailing-30d high AND VIX < 25

## Headline Metrics

| Variant | In-market | CAGR | Sharpe | Sortino | MaxDD | Calmar | WR | PF | Final $ |
|---|---|---|---|---|---|---|---|---|---|
| BASELINE | 100% | 18.96% | 2.14 | 2.13 | -13.1% | 1.45 | 64.0% | 1.61 | $200,799 |
| TREND_ONLY | 70% | 16.22% | 1.98 | 1.70 | -14.3% | 1.13 | 54.4% | 1.62 | $166,313 |
| TREND_VOL | 53% | 7.84% | 1.22 | 0.90 | -14.3% | 0.55 | 45.2% | 1.42 | $91,433 |
| TIGHT_TREND_VOL | 42% | 6.97% | 1.14 | 0.76 | -14.5% | 0.48 | 39.4% | 1.44 | $85,716 |
| RV_GATE | 51% | 6.17% | 1.34 | 0.95 | -8.0% | 0.77 | 40.0% | 1.44 | $80,664 |
| DD_FROM_HIGH | 74% | 14.98% | 1.96 | 1.73 | -13.4% | 1.12 | 56.4% | 1.60 | $152,631 |

## CAGR vs BASELINE

| Variant | CAGR | Delta vs baseline | Entries taken | Entries blocked |
|---|---|---|---|---|
| BASELINE | 18.96% | +0.00pp | 284 | 0 |
| TREND_ONLY | 16.22% | -2.74pp | 247 | 302 |
| TREND_VOL | 7.84% | -11.12pp | 199 | 550 |
| TIGHT_TREND_VOL | 6.97% | -11.99pp | 177 | 740 |
| RV_GATE | 6.17% | -12.80pp | 179 | 710 |
| DD_FROM_HIGH | 14.98% | -3.98pp | 248 | 215 |

## Regime Gate (HC #428 R1) — green/red day Sharpe gap

| Variant | n_green | n_red | n_flat | Sh green | Sh red | Sh flat | gap | pass |
|---|---|---|---|---|---|---|---|---|
| BASELINE | 534 | 406 | 1070 | 15.99 | -13.18 | 7.41 | 1.82 | FAIL |
| TREND_ONLY | 534 | 406 | 1070 | 13.55 | -10.63 | 7.34 | 1.78 | FAIL |
| TREND_VOL | 534 | 406 | 1070 | 9.55 | -8.75 | 6.92 | 1.92 | FAIL |
| TIGHT_TREND_VOL | 534 | 406 | 1070 | 8.09 | -7.37 | 6.33 | 1.91 | FAIL |
| RV_GATE | 534 | 406 | 1070 | 10.83 | -9.06 | 6.46 | 1.84 | FAIL |
| DD_FROM_HIGH | 534 | 406 | 1070 | 13.40 | -10.84 | 7.50 | 1.81 | FAIL |

## Deploy Gates

| Variant | Sharpe>=1.0 | Calmar>=1.5 | Regime gap<=0.50 | Day conc<=0.70 | In-mkt>=40% | DEPLOY |
|---|---|---|---|---|---|---|
| BASELINE | PASS | FAIL | FAIL | PASS | PASS | **NO** |
| TREND_ONLY | PASS | FAIL | FAIL | PASS | PASS | **NO** |
| TREND_VOL | PASS | FAIL | FAIL | PASS | PASS | **NO** |
| TIGHT_TREND_VOL | PASS | FAIL | FAIL | PASS | PASS | **NO** |
| RV_GATE | PASS | FAIL | FAIL | PASS | PASS | **NO** |
| DD_FROM_HIGH | PASS | FAIL | FAIL | PASS | PASS | **NO** |

## Tail Event Stress

| Variant | Event | MaxDD | VIX peak | Recover days | Worst week | Cum ret |
|---|---|---|---|---|---|---|
| BASELINE | COVID_2020 | -13.1% | 82.7 | 119 | -7.2% | -3.8% |
| BASELINE | 2022_bear | -5.5% | 36.5 | 8 | -3.3% | 12.5% |
| BASELINE | Aug_2024_carry | -4.6% | 38.6 | 39 | -1.2% | 0.2% |
| TREND_ONLY | COVID_2020 | -14.3% | 82.7 | 119 | -7.8% | -4.1% |
| TREND_ONLY | 2022_bear | -5.3% | 36.5 | 57 | -3.6% | 6.0% |
| TREND_ONLY | Aug_2024_carry | -5.4% | 38.6 | 56 | -1.7% | -0.9% |
| TREND_VOL | COVID_2020 | -14.3% | 82.7 | 609 | -7.8% | -10.7% |
| TREND_VOL | 2022_bear | -4.1% | 36.5 | 162 | -2.6% | -2.5% |
| TREND_VOL | Aug_2024_carry | -4.5% | 38.6 | 56 | -1.4% | -0.8% |
| TIGHT_TREND_VOL | COVID_2020 | -14.5% | 82.7 | 665 | -7.9% | -10.8% |
| TIGHT_TREND_VOL | 2022_bear | -3.9% | 36.5 | 445 | -2.6% | -1.1% |
| TIGHT_TREND_VOL | Aug_2024_carry | -4.7% | 38.6 | 56 | -1.5% | -0.8% |
| RV_GATE | COVID_2020 | -8.0% | 82.7 | 443 | -4.4% | -6.0% |
| RV_GATE | 2022_bear | 0.0% | 36.5 | 427 | 0.0% | 0.0% |
| RV_GATE | Aug_2024_carry | -4.8% | 38.6 | 93 | -1.5% | -3.2% |
| DD_FROM_HIGH | COVID_2020 | -13.4% | 82.7 | 296 | -7.3% | -10.0% |
| DD_FROM_HIGH | 2022_bear | -4.2% | 36.5 | 41 | -2.1% | 6.6% |
| DD_FROM_HIGH | Aug_2024_carry | -6.1% | 38.6 | 50 | -1.9% | -0.9% |

## Recommendation

- **No variant passes all deploy gates.** Recommendation: keep `entries_paused: true` in wheel_paper_engine.py and do NOT add regime_gate logic. Investigate whether the wheel's directional short-vol exposure can be made regime-agnostic at all, or pivot to a different income strategy.

### Why each variant failed

- **TREND_ONLY** (in-mkt 70%, Sharpe 1.98, Calmar 1.13, regime gap 1.78): failed [calmar_ge_1.5, regime_gap_le_0.50]
- **TREND_VOL** (in-mkt 53%, Sharpe 1.22, Calmar 0.55, regime gap 1.92): failed [calmar_ge_1.5, regime_gap_le_0.50]
- **TIGHT_TREND_VOL** (in-mkt 42%, Sharpe 1.14, Calmar 0.48, regime gap 1.91): failed [calmar_ge_1.5, regime_gap_le_0.50]
- **RV_GATE** (in-mkt 51%, Sharpe 1.34, Calmar 0.77, regime gap 1.84): failed [calmar_ge_1.5, regime_gap_le_0.50]
- **DD_FROM_HIGH** (in-mkt 74%, Sharpe 1.96, Calmar 1.12, regime gap 1.81): failed [calmar_ge_1.5, regime_gap_le_0.50]

## Honest caveats

- BS with modeled ATM sigma, no skew. Same pricing limitations as the hedge-overlay study; gate decisions are based on freely-available SPY/VIX data so this transfers cleanly to live paper trading via yfinance.
- Gates use spot prices and indicators known at the open of each trading day. Realized-vol and 30d-high are computed on prior close, so no look-ahead.
- A regime gate that flattens green/red Sharpe gap by REDUCING TIME IN MARKET must still pass the 40% in-market floor to be deployable. A gate that's open 20% of the time produces a 'great Sharpe' on a tiny sample - not robust.
- Even if a variant passes the gates, BS without skew under-prices OTM puts; real chain premiums will be ~10-20% higher, slightly reducing realized P&L.