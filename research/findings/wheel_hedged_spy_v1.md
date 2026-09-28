# Wheel + Hedge Overlay v1 - SPY Tier2 Balanced Scalp

Generated: 2026-06-09 16:41:29
Window: 2018-01-01 to 2025-12-31
Starting cash: $50,000, leverage 1.0x, single underlying SPY
Wheel cfg: put_delta 0.22, call_delta 0.22, DTE 30-45, profit-take 50%, roll DTE<=10, VIX gate 32.0

Pricing: Black-Scholes with modeled ATM sigma for SPY; BS on VIX with vol-of-vol=1.10 for VIX calls. No skew model -> hedge premium estimates are LOWER BOUNDS; real OTM put cost would be 30-60% higher in the chain.
Slippage: max(2.5% premium, $0.03/share) for SPY; 5% for VIX. Commission: $0.65/contract/leg.

Hedge variants:
- **STATIC_PUT**: always-on long SPY put, 90 DTE, |delta|=0.10, roll DTE<30
- **VIX_CALL_COND**: long VIX call (delta +0.20, 30 DTE) when VIX>20, close when VIX<16
- **PUTSPREAD_COLLAR**: long-short SPY put-spread (long -0.10, short -0.05) when VIX>20 OR 50d MA < 200d MA, 90 DTE, roll DTE<30

## Headline Metrics

| Variant | CAGR | Sharpe | Sortino | MaxDD | Calmar | WR | PF | Hedge $ drag | Final $ |
|---|---|---|---|---|---|---|---|---|---|
| BASELINE | 18.96% | 2.14 | 2.13 | -13.1% | 1.45 | 64.0% | 1.61 | $0 | $200,799 |
| STATIC_PUT | 20.94% | 2.60 | 2.94 | -9.4% | 2.22 | 64.3% | 1.74 | $-2,772 | $229,111 |
| VIX_CALL_COND | 17.51% | 2.13 | 2.22 | -10.5% | 1.66 | 64.2% | 1.58 | $2,223 | $182,025 |
| PUTSPREAD_COLLAR | 17.59% | 2.10 | 2.08 | -12.6% | 1.40 | 64.7% | 1.59 | $1,207 | $183,041 |

## CAGR cost of the hedge (vs BASELINE)

| Variant | CAGR | Delta vs baseline | Hedge cost % CAGR |
|---|---|---|---|
| BASELINE | 18.96% | +0.00pp | -0.00pp |
| STATIC_PUT | 20.94% | +1.98pp | -1.98pp |
| VIX_CALL_COND | 17.51% | -1.45pp | +1.45pp |
| PUTSPREAD_COLLAR | 17.59% | -1.37pp | +1.37pp |

## Regime Gate (HC #428 R1)

| Variant | n_green | n_red | n_flat | Sh green | Sh red | Sh flat | gap | pass |
|---|---|---|---|---|---|---|---|---|
| BASELINE | 534 | 406 | 1070 | 15.99 | -13.18 | 7.41 | 1.82 | FAIL |
| STATIC_PUT | 534 | 406 | 1070 | 14.33 | -11.56 | 7.53 | 1.81 | FAIL |
| VIX_CALL_COND | 534 | 406 | 1070 | 15.90 | -11.99 | 7.31 | 1.75 | FAIL |
| PUTSPREAD_COLLAR | 534 | 406 | 1070 | 16.00 | -12.94 | 7.59 | 1.81 | FAIL |

## Deploy Gates

| Variant | Sharpe>=1.0 | Calmar>=1.5 | Regime gap<=0.50 | Day conc<=0.70 | n_days>=40 | DEPLOY |
|---|---|---|---|---|---|---|
| BASELINE | PASS | FAIL | FAIL | PASS | PASS | **NO** |
| STATIC_PUT | PASS | PASS | FAIL | PASS | PASS | **NO** |
| VIX_CALL_COND | PASS | PASS | FAIL | PASS | PASS | **NO** |
| PUTSPREAD_COLLAR | PASS | FAIL | FAIL | PASS | PASS | **NO** |

## Tail Event Stress

| Variant | Event | MaxDD | VIX peak | Recover days | Worst week | Cum ret |
|---|---|---|---|---|---|---|
| BASELINE | COVID_2020 | -13.1% | 82.7 | 119 | -7.2% | -3.8% |
| BASELINE | 2022_bear | -5.5% | 36.5 | 8 | -3.3% | 12.5% |
| BASELINE | Aug_2024_carry | -4.6% | 38.6 | 39 | -1.2% | 0.2% |
| STATIC_PUT | COVID_2020 | -7.9% | 82.7 | 9 | -4.8% | 4.0% |
| STATIC_PUT | 2022_bear | -4.6% | 36.5 | 50 | -3.0% | 15.6% |
| STATIC_PUT | Aug_2024_carry | -3.7% | 38.6 | 18 | -0.9% | 2.0% |
| VIX_CALL_COND | COVID_2020 | -10.5% | 82.7 | 84 | -6.3% | -0.8% |
| VIX_CALL_COND | 2022_bear | -5.5% | 36.5 | 8 | -3.0% | 11.1% |
| VIX_CALL_COND | Aug_2024_carry | -4.4% | 38.6 | 39 | -1.3% | 0.1% |
| PUTSPREAD_COLLAR | COVID_2020 | -12.6% | 82.7 | 119 | -6.8% | -3.7% |
| PUTSPREAD_COLLAR | 2022_bear | -5.0% | 36.5 | 8 | -3.3% | 12.7% |
| PUTSPREAD_COLLAR | Aug_2024_carry | -4.8% | 38.6 | 39 | -1.3% | 0.1% |

## Recommendation

- **No hedge variant clears all HC #428 deploy gates** in this backtest.
- BASELINE itself fails: calmar_ge_1.5, regime_gap_le_0.50. The hedge can only narrow the regime gap; it cannot create alpha. If baseline fails Sharpe/Calmar, a hedge that costs CAGR will make those metrics worse.
- STATIC_PUT: regime gap narrowed from 1.82 -> 1.81.
- VIX_CALL_COND: regime gap narrowed from 1.82 -> 1.75.
- PUTSPREAD_COLLAR: regime gap narrowed from 1.82 -> 1.81.

## Honest caveats

- BS with no skew model UNDER-prices OTM SPY puts. Real chain premium for a -0.10 delta 90 DTE SPY put runs 30-60% above modeled here due to crash skew. Hedge drag in production will exceed these numbers.
- VIX option pricing uses a constant vol-of-vol=1.10; the true VIX surface is mean-reverting and has its own term structure. Treat VIX_CALL_COND P&L with extra skepticism vs the put hedges.
- The wheel itself is short vol and short put gamma. A long-put hedge that PASSES the regime-gap gate while still earning >=1.0 Sharpe is mathematically asking the wheel to earn more carry than the hedge bleeds, in all regimes - a high bar that the underlying SPY mid-DTE 22-delta wheel rarely clears.
- This run uses the SAME 2018-2025 window the unhedged SPY backtest used. Sub-agent aac533ea documented the BASELINE regime gap is structurally large; the question this report answers is HOW MUCH does each hedge variant flatten that gap, and AT WHAT COST in CAGR/Sharpe.