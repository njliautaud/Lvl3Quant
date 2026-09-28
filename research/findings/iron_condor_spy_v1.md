# Iron Condor SPY v1 (HC #586 R1 - FINAL carry/income lane dispatch)

Generated: 2026-06-09
Window: 2018-01-02 to 2025-12-31 (8.0 years, 2010 trading days)
Starting cash: $20,000 (HC #580 anchor; % returns are the primary read-out)
Costs: options $0.65/contract + 1.0 bp slippage on each leg premium (4 legs)
VIX gates: don't open if VIX > 28; intraday VIX > 30 closes all legs; re-entry blocked until VIX < 25

## Verdict: FAIL on every HC #428 R1 deploy gate

| Gate | Threshold | Actual | Status |
|---|---|---|---|
| Sharpe >= 1.0 | 1.0 | -0.65 | FAIL |
| Calmar >= 1.0 | 1.0 | -0.29 | FAIL |
| Regime gap <= 0.50 (HC #428 R1) | 0.50 | **1.01** | **FAIL** |
| Day concentration <= 0.70 | 0.70 | 0.05 | PASS |
| CAGR positive | > 0 | -26.62% | FAIL |
| **DEPLOY READY** | all pass | 1 / 5 | **NO** |

## Headline metrics ($20K notional)

| Metric | Iron Condor | SPY Buy & Hold |
|---|---|---|
| Trades | 60 | n/a |
| CAGR | **-26.62%** | +14.12% |
| Sharpe | -0.65 | +0.78 |
| Sortino | -0.31 | +0.95 |
| Max drawdown | **-91.6%** | -33.7% |
| Calmar | -0.29 | +0.42 |
| Win rate (daily) | 16.5% | 55.4% |
| Profit factor | 0.75 | 1.16 |
| Final equity | $1,659 | $57,495 |
| Avg days in trade | 13.5 | n/a |
| Avg % of max profit captured | **-38.9%** | n/a |

The "avg % of max profit captured" of -38.9% is the structural killer: the
average closed condor LOSES 39% of theoretical max profit, despite a 73%
profit-take rate (44/60). A handful of losers eat all the winners and then
some.

## HC #428 R1 - green/red regime Sharpe stratification (THE gate)

| Regime | Days | Sharpe | Notes |
|---|---|---|---|
| Green (SPY > +0.5sigma) | 534 | +0.04 | Profits ATM, but theta is small |
| Flat (within +/-0.5sigma) | 1070 | +2.68 | Where the strategy "works" |
| Red (SPY < -0.5sigma) | 406 | **-3.87** | Catastrophic - short-gamma losses dominate |

Regime gap = |0.04 - (-3.87)| / max(0.04, 3.87) = **1.01** (need <= 0.50).

The gap is WORSE than the delta-hedged straddle (0.61) - despite the long
wings that supposedly cap downside. Why? Because the wings cap TERMINAL
risk at expiry, but the strategy closes early on profit-take (50% of credit)
or DTE 7. Between open and close, MTM losses from gamma-driven IV spikes
and gap moves still hit the position; the wings only become effective at
expiry, which we explicitly avoid by closing at DTE 7.

## Tail event stress

| Event | VIX peak | MaxDD | Recovered? | Cum return | VIX-30 stop fire? |
|---|---|---|---|---|---|
| Volmageddon Feb 2018 | 50.3 | **-45.0%** | NO (within window) | -37.8% | YES, but late |
| COVID Mar 2020 | 85.5 | -13.5% | NO | -9.9% | YES, helped |
| Aug 2024 carry unwind | 65.7 | 0.0% | n/a | 0.0% | n/a (in blackout) |
| 2022 bear market | 38.9 | 0.0% | n/a | 0.0% | n/a (in blackout) |

VIX-30 stops fired 4 times total. Diagnostic of effectiveness:

- **Volmageddon Feb 2018**: VIX gapped from 17 to 37 in one day (Feb 5). The
  stop fired on the daily-high signal, but the close-price was already
  ~37. We took the loss; the wings did NOT save us because the gap was so
  fast we exited at MTM cost much higher than the wing strike. **Stop did
  not help much** - it merely capped the bleed at -45% rather than -70%+
  that an unwinged short straddle would have suffered.
- **COVID Mar 2020**: Stop fired before the worst, blackout kept us out for
  ~6 weeks. Only -13.5% loss. **Stop genuinely helped here.**
- **Aug 2024 / 2022 bear**: We were already in blackout (post-prior-event)
  for the whole window, so the stop didn't fire. The "0%" return is
  literally "didn't trade" - this is NOT a wins; it's the strategy
  abdicating entirely.

**Honest assessment: VIX-30 stop has bipolar effectiveness. It saves us
sometimes (COVID), and at other times (Volmageddon) the gap is too fast for
a daily-close trigger to help. Worse, by gating re-entry on VIX < 25, the
strategy spends large chunks of high-vol regimes OUT of the market entirely,
which destroys the "always-on income carry" thesis. In 2022 (whole year)
and Aug 2024, we earned zero - while paying opportunity cost relative to
SPY buy-and-hold (+14% CAGR).**

## Diagnostic - why iron condor fails harder than the straddle

The prior delta-hedged straddle got within 0.11 of the regime-gap threshold
(0.61 actual vs 0.50 target). One might have expected adding long wings to
PRODUCE delta+gamma neutrality and CLOSE the gap. The opposite happened:

1. **Wings are nearly worthless until tail risk realizes.** At 5-delta on
   30 DTE SPY at ~15% IV, the long wing premium is ~$0.20-0.50 per contract,
   but the SHORT 10-delta leg sells for ~$0.50-1.20. Net credit per
   condor: ~$0.50-0.70. Max profit per contract (~$50-70) vs max loss
   per contract (~$430-450, since strike width is ~$5). **Risk-reward is
   ~6-9x to lose; we need WR > 90% just to break even**. Empirical WR (on
   trade closes) was 73% (profit-takes), nowhere near.

2. **No skew in BS pricing.** Real OTM puts have richer IV than ATM
   (volatility smile). Our BS model assumes flat IV, so we're SELLING the
   put leg at a price 20-50% below what we'd actually receive AND BUYING the
   wing at a price 20-50% above what we'd actually pay. Real market would
   give modestly better entry credits - but probably not enough to flip
   sign. Honest read: the BS-pricing artifact alone might explain ~20% of
   the underperformance; the regime asymmetry and the bad risk-reward are
   structural, not pricing.

3. **Closing at DTE 7 + profit-take 50% nullifies the wings.** Iron-condor
   theory says max profit at expiry inside the short strikes. We close
   well before expiry to harvest theta. Between open and close, the
   position is dominated by short-gamma MTM, not by wing protection. The
   wings only fully amortize their cost at expiry.

4. **The carry/income business is structurally regime-asymmetric.** This is
   the same diagnosis as the prior 9 (!) backtests today:
   - 5 wheel variants: regime-gap 0.62 to 0.95
   - 3 directional-carry rotations: regime-gap 0.71 to 1.10
   - Delta-hedged short straddle: 0.61 (closest)
   - Iron condor: 1.01 (WORSE than the straddle)

   Every variant that SELLS volatility/option-premium for income suffers
   green/red asymmetry: it gives back winnings during red days, makes
   modest gains during green days, and only "works" during flat days.
   When stratified by SPY direction, the income-carry business cannot
   look symmetric because the structure IS the asymmetry.

5. **VIX gating reduces drawdowns but kills the "always-on" thesis.** The
   strategy spent ~25% of the 2018-2025 window in VIX-blackout
   (post-stop-out, waiting for VIX < 25). During those windows it earned
   zero. The remaining 75% of days had to carry the whole CAGR, and they
   couldn't.

## Recommendation: PIVOT TO EVENT-DRIVEN RESEARCH

Per HC #586 R1: if iron condor fails, the carry/income direction is
**structurally exhausted**. It is. Recommendation:

- **DO NOT** paper-deploy this strategy. NO engine module written.
- **DO NOT** iterate further on carry/income variants (covered calls, calendars,
  butterflies, condors at other deltas, ratio spreads). The structural
  problem - that short-vol payoffs sell catastrophic-tail risk that
  realizes in red regimes - is not fixable by parameter sweeps.
- **PIVOT** to event-driven research:
  - Post-earnings drift on quality factor names (PEAD).
  - FOMC-day vol expansion trade (long straddle into Fed days, close
    same day or t+1).
  - Index-rebalance front-running (Russell/S&P quarterly).
  - VVIX/VIX dislocation when VVIX > 130 (mean-reversion in vol-of-vol).
  These all have an explicit information / structural-flow trigger that
  is regime-orthogonal to SPY green/red days.
- New data needed for some pivots (single-name options, VVIX history,
  index-rebalance announcements). Recommend cataloguing data requirements
  before next research dispatch.

## Honest caveats

- BS pricing without skew systematically under-prices OTM puts and
  over-prices OTM calls. Real iron-condor credit on a 10-delta/5-delta SPY
  30 DTE would be ~20-50% richer than this backtest assumes. That would
  shift CAGR up modestly (say from -27% to -10%) but would NOT flip the
  Sharpe sign or close the regime gap from 1.01 to under 0.50.
- VIX intraday stop is implemented using yfinance daily-VIX-High as the
  intraday-peak proxy. Real intraday triggers (e.g. crossing 30 at 11:30
  ET) might fire slightly earlier than our daily-close-of-business check.
  This is a small effect - by the time daily VIX high crosses 30, the
  underlying SPY move has typically already happened.
- 1-bp slippage per leg is generous to the strategy. Real SPY options
  bid/ask is 0.05-0.10 wide on 5-10 delta strikes, which is ~10-20 bps of
  premium. Real costs would push CAGR another 5-10 points worse.

## Disposition

- **MLflow run**: experiment "iron_condor_spy_v1" run "iron_condor_10d_5d_30dte" - logged.
- **Paper engine**: NOT written, per task spec (deploy gates failed).
- **Carry/income research lane**: closed per HC #586 R1.
- **Next dispatch**: event-driven research lane (PEAD / FOMC / rebalance /
  VVIX dislocation).
