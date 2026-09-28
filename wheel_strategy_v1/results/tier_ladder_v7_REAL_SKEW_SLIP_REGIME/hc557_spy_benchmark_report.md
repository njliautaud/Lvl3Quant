# HC #557 — Wheel vs. SPY/Margin-SPY Benchmark

**Verdict (blended ladder, 35% Conservative / 55% Balanced / 10% Turbo): WHEEL LOSES — JUST USE MARGIN SPY**

Window: 2020-01-01 to 2025-12-31. Cash starts at $100k per tier ($300k for blended ladder). All wheel numbers use realized cash, not mark-to-market.

## Headline Comparison

| Strategy | CAGR | Sharpe | Sortino | Max Drawdown |
|---|---|---|---|---|
| SPY (unlevered) | 14.82% | 0.77 | 0.95 | -33.72% |
| SPY 1.5x margin | 18.52% | 0.70 | 0.87 | -47.34% |
| SPY 2.0x margin | 20.99% | 0.67 | 0.83 | -58.86% |
| Tier1 Conservative | 3.33% | 0.85 | 0.18 | -5.72% |
| Tier2 Balanced | 10.54% | 1.02 | 0.41 | -11.92% |
| Tier3 Income | 1.01% | 0.14 | 0.05 | -45.24% |
| Tier4 Aggressive | 7.54% | 0.36 | 0.19 | -65.73% |
| Tier5 Turbo | 7.00% | 0.54 | 0.27 | -30.15% |
| Blended Ladder | 7.92% | 1.22 | 0.78 | -8.44% |

## Monthly Realized Income by Regime

Day counts in window — green up days: 391, red down days: 313, flat days: 804.

| Strategy | Green Months | Red Months | Flat Months | Worst-Bucket Test |
|---|---|---|---|---|
| Tier1 Conservative | $256 | $123 | $395 | FAIL |
| Tier2 Balanced | $1,210 | $317 | $1,440 | FAIL |
| Tier3 Income | $89 | $-1,186 | $581 | FAIL |
| Tier4 Aggressive | $688 | $-1,417 | $1,644 | FAIL |
| Tier5 Turbo | $615 | $463 | $828 | PASS |
| Blended Ladder | $2,450 | $792 | $3,039 | FAIL |

Gate rule: positive monthly income in every regime AND no single bucket's risk-adjusted return more than 2x the others.

## Per-Tier Verdicts

- **Tier1 Conservative** — WHEEL LOSES — JUST USE MARGIN SPY (Sharpe 0.85 vs margin-SPY 0.70; regime gate FAIL — sharpe_dispersion |worst|/|best|=0.36<0.50).
- **Tier2 Balanced** — WHEEL LOSES — JUST USE MARGIN SPY (Sharpe 1.02 vs margin-SPY 0.70; regime gate FAIL — sharpe_dispersion |worst|/|best|=0.14<0.50).
- **Tier3 Income** — WHEEL LOSES — JUST USE MARGIN SPY (Sharpe 0.14 vs margin-SPY 0.70; regime gate FAIL — red_monthly<=0 (-1186); sharpe_dispersion |worst|/|best|=0.06<0.50).
- **Tier4 Aggressive** — WHEEL LOSES — JUST USE MARGIN SPY (Sharpe 0.36 vs margin-SPY 0.70; regime gate FAIL — red_monthly<=0 (-1417); sharpe_dispersion |worst|/|best|=0.27<0.50).
- **Tier5 Turbo** — WHEEL LOSES — JUST USE MARGIN SPY (Sharpe 0.54 vs margin-SPY 0.70; regime gate PASS).
- **Blended Ladder** — WHEEL LOSES — JUST USE MARGIN SPY (Sharpe 1.22 vs margin-SPY 0.70; regime gate FAIL — sharpe_dispersion |worst|/|best|=0.23<0.50).

## Bottom Line

The blended wheel ladder does not beat a 1.5x margin SPY portfolio on risk-adjusted basis, or fails the regime-income gate. Under HC #557, this is a NEGATIVE FINDING — a margin SPY position is strictly better. Do not deploy the wheel as a stand-alone product unless the structure is materially changed.
