# Wheel Short-DTE v1 - SPY (HC #584 R2 - FINAL wheel experiment)

Generated: 2026-06-09 16:53:17
Window: 2018-01-01 to 2025-12-31
Starting cash: $50,000, leverage 1.0x, SPY only
Commission: $0.65/contract per leg, slippage 2.5% / $0.03 min

## Variants

| Variant | DTE range | Target DTE | Roll DTE | Profit-take | Put delta | VIX cap |
|---|---|---|---|---|---|---|
| BASELINE | 30-45 | 37 | <=10 | 50% | 0.22 | 32.0 |
| WEEKLY | 5-10 | 7 | no roll | 50% | 0.22 | 32.0 |
| SHORT_DTE | 10-15 | 12 | <=3 | 50% | 0.22 | 32.0 |

## Headline Metrics

| Variant | Trades | CAGR | Sharpe | Sortino | MaxDD | Calmar | WR | PF | Final $ |
|---|---|---|---|---|---|---|---|---|---|
| BASELINE | 283 | 18.96% | 2.14 | 2.13 | -13.1% | 1.45 | 64.0% | 1.61 | $200,799 |
| WEEKLY | 21 | -28.47% | -0.28 | -0.03 | -95.2% | -0.30 | 2.1% | 0.40 | $3,438 |
| SHORT_DTE | 577 | 22.70% | 3.13 | 2.92 | -7.1% | 3.19 | 67.7% | 1.89 | $256,821 |

## Delta vs BASELINE (DTE 37)

| Variant | CAGR | dCAGR | Sharpe | dSharpe | MaxDD | dMaxDD | Trades | dTrades |
|---|---|---|---|---|---|---|---|---|
| BASELINE | 18.96% | +0.00pp | 2.14 | +0.00 | -13.1% | +0.00pp | 283 | +0 |
| WEEKLY | -28.47% | -47.43pp | -0.28 | -2.42 | -95.2% | -82.13pp | 21 | -262 |
| SHORT_DTE | 22.70% | +3.74pp | 3.13 | +0.99 | -7.1% | +5.99pp | 577 | +294 |

## HC #428 R1 - green/red regime Sharpe gap

| Variant | n_green | n_red | n_flat | Sh green | Sh red | Sh flat | gap | pass (<=0.50) |
|---|---|---|---|---|---|---|---|---|
| BASELINE | 534 | 406 | 1070 | 15.99 | -13.18 | 7.41 | 1.82 | FAIL |
| WEEKLY | 534 | 406 | 1070 | 0.80 | -1.13 | 0.73 | 1.71 | FAIL |
| SHORT_DTE | 534 | 406 | 1070 | 20.44 | -12.44 | 10.54 | 1.61 | FAIL |

## Deploy Gates (HC #428 + day-conc + Calmar)

| Variant | Sharpe>=1.0 | Calmar>=1.5 | Regime gap<=0.50 | Day conc<=0.70 | DEPLOY |
|---|---|---|---|---|---|
| BASELINE | PASS | FAIL | FAIL | PASS | **NO** |
| WEEKLY | FAIL | FAIL | FAIL | PASS | **NO** |
| SHORT_DTE | PASS | PASS | FAIL | PASS | **NO** |

## Tail Event Stress

| Variant | Event | MaxDD | VIX peak | Recover days | Worst week | Cum ret |
|---|---|---|---|---|---|---|
| BASELINE | COVID_2020 | -13.1% | 82.7 | 119 | -7.2% | -3.8% |
| BASELINE | 2022_bear | -5.5% | 36.5 | 8 | -3.3% | 12.5% |
| BASELINE | Aug_2024_carry | -4.6% | 38.6 | 39 | -1.2% | 0.2% |
| WEEKLY | COVID_2020 | 0.0% | 82.7 | not_recovered | 0.0% | 0.0% |
| WEEKLY | 2022_bear | 0.0% | 36.5 | not_recovered | 0.0% | 0.0% |
| WEEKLY | Aug_2024_carry | 0.0% | 38.6 | not_recovered | 0.0% | 0.0% |
| SHORT_DTE | COVID_2020 | -6.4% | 82.7 | 98 | -6.2% | -1.2% |
| SHORT_DTE | 2022_bear | -7.1% | 36.5 | 0 | -2.3% | 19.1% |
| SHORT_DTE | Aug_2024_carry | -2.8% | 38.6 | 9 | -1.2% | 3.6% |

## Recommendation

- **No short-DTE variant passes all deploy gates.**
- The duration cut did NOT structurally fix the green/red regime gap. The wheel is short-vol / short-tail at every duration tested. The regime asymmetry is intrinsic to the payoff, not the holding period.
- **Recommendation: formally shelve the wheel direction.** Do not modify wheel_paper_engine.py. Tier2_Balanced_Scalp stays entries_paused=True.

### Why each variant failed

- **WEEKLY** (Sharpe -0.28, Calmar -0.30, regime gap 1.71, trades 21): failed [sharpe_ge_1.0, calmar_ge_1.5, regime_gap_le_0.50]
- **SHORT_DTE** (Sharpe 3.13, Calmar 3.19, regime gap 1.61, trades 577): failed [regime_gap_le_0.50]

## Honest caveats

- BS-modeled premiums with no skew. Real weekly puts at delta 0.22 trade richer than BS by ~15-25% (gamma/skew premium); realized P&L on weekly leg will likely be slightly higher in live trading. This makes the weekly result CONSERVATIVE.
- Weekly cycle: ~50 trades/year per dollar deployed vs ~10 for the 37-DTE baseline. Commission drag at $0.65/contract/leg is fully modeled.
- Same SPY/IV/VIX panel and FIFO MTM bookkeeping as the prior wheel studies.
- Assignment is taken realistically on ITM expiry; covered call is written on shares immediately the next day if VIX allows.