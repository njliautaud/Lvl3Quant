# Market-Neutral SPY v1 (HC #585 R2 - pivot from short-vol carry)

Generated: 2026-06-09 16:59:36
Window: 2018-01-01 to 2025-12-31
Starting cash: $20,000  (HC #580 anchor; % returns are the primary read-out)
Costs: options $0.65/contract + 1.0% slippage; shares $0.005/share + 1.0 bp slippage
VIX gate: don't open if VIX > 30.0

## Why this experiment exists

- 5 wheel variants (baseline / QQQ+IWM / hedge overlay / regime-gated / short-DTE) ALL failed HC #428 R1.
- 3 directional carry rotations (ETF / tech sub-industry / blend) ALL failed HC #428 R1.
- Conclusion: short-vol payoffs at any duration are intrinsically regime-asymmetric. The carry IS the asymmetry.
- Per HC #585 R2: only genuinely market-neutral structures can clear the gate. Two candidates below.

## Strategy summary

- **DELTA_STRADDLE**: sell SPY ATM straddle, DTE 30, daily delta-hedge to neutral with shares. Close at 50% profit or DTE 5.
- **PUT_CALENDAR**: long 60 DTE SPY put @ ATM-5%, short 30 DTE SPY put @ same strike. Close at front-month expiry, reopen weekly.

## Headline metrics ($20K notional)

| Strategy | Trades | Hedge trades | CAGR | Sharpe | Sortino | MaxDD | Calmar | WR | PF | Final $ |
|---|---|---|---|---|---|---|---|---|---|---|
| DELTA_STRADDLE | 108 | 1973 | -3.51% | -0.19 | -0.15 | -50.4% | -0.07 | 61.6% | 0.94 | $15,021 |
| PUT_CALENDAR | 82 | 0 | 38.14% | 0.76 | 2.94 | -98.5% | 0.39 | 34.0% | 1.37 | $238,522 |
| SPY BH | n/a | n/a | 14.12% | 0.78 | 0.95 | -33.7% | 0.42 | 55.4% | 1.16 | $57,495 |

## HC #428 R1 - green/red regime Sharpe gap (THE gate)

| Strategy | n_green | n_red | n_flat | Sh green | Sh red | Sh flat | gap | pass (<=0.50) |
|---|---|---|---|---|---|---|---|---|
| DELTA_STRADDLE | 534 | 406 | 1070 | -2.47 | -6.36 | 9.35 | 0.61 | FAIL |
| PUT_CALENDAR | 534 | 406 | 1070 | -2.95 | 4.56 | -5.15 | 1.65 | FAIL |

## Deploy gates

| Strategy | Sharpe>=1.0 | Calmar>=1.0 | Regime gap<=0.50 | Day conc<=0.70 | DEPLOY |
|---|---|---|---|---|---|
| DELTA_STRADDLE | FAIL | FAIL | FAIL | PASS | **NO** |
| PUT_CALENDAR | FAIL | FAIL | FAIL | PASS | **NO** |

## Tail event stress (per strategy)

| Strategy | Event | MaxDD | VIX peak | Recover days | Worst week | Cum ret |
|---|---|---|---|---|---|---|
| DELTA_STRADDLE | Volmageddon_Feb2018 | -4.2% | 37.3 | 31 | -2.5% | 0.6% |
| DELTA_STRADDLE | COVID_Mar2020 | -6.2% | 82.7 | not_recovered | -3.7% | -3.5% |
| DELTA_STRADDLE | Aug2024_carry | -13.0% | 38.6 | not_recovered | -5.0% | -5.0% |
| DELTA_STRADDLE | 2022_bear | -14.9% | 36.5 | not_recovered | -3.7% | -9.4% |
| PUT_CALENDAR | Volmageddon_Feb2018 | -68.2% | 37.3 | 0 | -67.9% | 30.0% |
| PUT_CALENDAR | COVID_Mar2020 | -63.2% | 82.7 | not_recovered | -10.8% | 102.9% |
| PUT_CALENDAR | Aug2024_carry | -52.0% | 38.6 | not_recovered | -32.5% | 711.8% |
| PUT_CALENDAR | 2022_bear | -58.0% | 36.5 | not_recovered | -26.0% | 251.3% |

## Honest caveats

- BS pricing without skew under-prices OTM puts (calendar back-leg and straddle put leg).
  Real ATM straddles are typically richer than BS by 5-15% during normal regimes and 30-100% richer during stress (vol-smile + VRP).
- Delta-hedge frequency: this backtest rebalances ONCE PER DAY at the close. Live execution with intraday hedging would have higher gamma-slippage costs but better delta tracking.
- The straddle equity series will exhibit jumpy daily returns from discrete hedging error; the green/red gate is a strict test of whether residual delta survives the hedge.
- Calendar net debit is small relative to $20K account; sizing is conservative (~25% of equity per open).
- VIX-30 gate keeps both strategies OUT of the heart of Volmageddon and COVID.

## Diagnostic - why each failed

### DELTA_STRADDLE (Sharpe -0.19, regime gap 0.61, CAGR -3.5%)

The gap (0.61) is the *closest any short-vol-flavored strategy has come* to the
HC #428 R1 threshold of 0.50 across today's eight backtests. But the strategy
still loses money in both green AND red regimes (Sharpe -2.47 and -6.36
respectively) and only profits in flat regimes (Sharpe +9.35). The structural
issue:

- 1973 daily hedge trades across 108 straddle cycles -> ~18 hedge trades per
  cycle. At $0.005/share + 1bp slippage, the hedging-cost drag is on the order
  of 15-25% of premium collected per cycle.
- BS pricing without skew underprices the put leg; in reality VRP would be
  larger and the strategy would clear the spread cost. But the regime asymmetry
  (loses on both green AND red days, profits only on flat) is NOT a cost
  artifact - it's the gamma payoff: the position is short gamma, so any sharp
  move in either direction churns the hedge unfavourably.
- Conclusion: delta-neutral does NOT make the strategy regime-symmetric when
  realized vol >> 0. The exposure becomes gamma vs theta, not delta. To pass
  HC #428 R1, the strategy needs to be both delta-neutral AND gamma-neutral,
  which requires multi-strike structures (e.g. iron condor with offsetting
  gammas) - not a single straddle.

### PUT_CALENDAR (Sharpe 0.76, regime gap 1.65, CAGR 38% but MaxDD -98.5%)

This strategy looks great on headline CAGR (38%) but the -98.5% drawdown
disqualifies it instantly. The hidden problem:

- Long-vega exposure: when implied vol spikes (e.g. Volmageddon 2018, COVID
  2020, Aug 2024), the LONG back-leg gains more than the SHORT front-leg loses,
  so the gross book-value of the calendar gets huge. But this is offset by a
  short-vol leg that bleeds theta in calm markets.
- Volmageddon -68% drawdown in 5 weeks, recovered. COVID -63%. Aug 2024 -52%.
  These are all VIX-30+ regimes where the calendar held a stale position
  (we only check VIX on OPEN, not on existing positions).
- Profile is more long-vol than market-neutral, despite the "calendar"
  framing. Inherits the worst of both worlds: pays theta when calm, suffers
  vega when stressed.

## Recommendation

- **Neither strategy cleared all deploy gates.** Do NOT paper-deploy.
- DELTA_STRADDLE is the closest to passing the regime gate (0.61 vs 0.50
  threshold). Worth one more iteration with: (a) ALSO closing on VIX > 30
  intraday (not just gating opens), (b) smaller delta-band tolerance (e.g.
  rebalance only when |delta| > 10 shares, to cut hedge-cost drag), (c) iron
  condor variant to remove gamma exposure. Estimated 1-2 days additional work.
- PUT_CALENDAR has structural long-vol exposure on top of the short-theta
  carry. Shelve.
- Pivot candidates that require new data: (i) VIX futures roll-down (need
  futures term-structure data); (ii) dispersion trade (sell SPY straddle, buy
  basket of stock straddles - need single-name option chains, which the
  current data panel doesn't have).
- Honest read: **no genuinely market-neutral income strategy in the current
  data panel clears HC #428 R1 with the current backtest fidelity.** Either
  the iteration above produces a passing iron-condor variant, or the next
  research direction needs to be intraday/event-driven rather than carry.