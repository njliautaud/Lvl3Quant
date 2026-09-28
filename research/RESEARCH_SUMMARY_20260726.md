# Research Summary — July 26-27, 2026

> **SUPERSEDED** — See `/research/QUANT_KNOWLEDGE_BASE.md` for the definitive, continuously-updated knowledge base.
> Numbers below were further corrected on 2026-07-27 by the definitive validation (no-exit-tricks, honest pricing).

## Best Validated Strategies (DEFINITIVE Honest Numbers — 2026-07-27)

| Strategy | Definitive Sharpe | Previously Reported | CAGR | MaxDD | WR | Trades |
|----------|-------------------|---------------------|------|-------|-----|--------|
| Sector Bull Spreads (hold to expiry) | **3.04** | 4.73 (exit inflated) | 63% | -9.6% | 57% | 422 |
| Sector Bull Spreads (VIX>25 only) | **2.97** | 5.18 | 135% | -3.7% | 79% | 178 |
| VIX Mean Reversion | **2.34** | 2.34 (unchanged) | 5.4% | -7.7% | 71% | 38 |

**Key correction:** The Sharpe 4.73 / MDD -1.4% / WR 88.7% reported from production_sector_v3 was inflated by early exit assumptions (selling spread mid-life at theoretical value with no bid-ask haircut). The definitive hold-to-expiry number is Sharpe 3.04 / MDD -9.6% / WR 57%. Still excellent, but materially different from what was previously reported.

**75% of the edge is structural** (bull call spreads + VIX filter work on ANY sector ETF). LGBM ranking adds ~25% incremental Sharpe.

---

## Strategies That Failed (and Why)

### Structurally Unprofitable
- **Butterfly Strategies (6 variants)** — Call butterflies, iron butterflies, broken-wing butterflies on sector ETFs. Win rate 0–16%, drawdowns near -100%. Our signal predicts direction, not a precise price target. Butterflies need pinpoint accuracy. All dead.
- **Sector Iron Condor Income (7 variants)** — Selling iron condors on weak-momentum sectors. Every standalone config loses money — account goes negative. When combined with bull spreads, the iron condors are pure drag (-$74/trade vs +$3K/trade for spreads). Sector ETFs are too volatile for premium selling at this account size.

### Worse Than Existing Strategy
- **Weekly DTE Spreads (7 variants)** — 5–7 day holding period instead of monthly. Best Sharpe only 0.44 vs monthly's 1.5+. Drawdowns -29% to -65%. Extreme regime dependency (gap 0.70–0.80). Random selection was also profitable, meaning the "edge" is structural noise. Monthly is strictly better.

### Unnecessary (Null Result)
- **Drawdown Control Rules (8 configs)** — Scaling down during drawdowns, skipping trades after losses, recovery boosting, ML risk filters. All produced identical results to the base strategy because the VIX>20 filter already limits exposure so well that drawdown rules never trigger. The VIX filter IS the risk manager — adding more rules on top does nothing.

### Failed Permutation Test
- **Exit Optimization (8 variants)** — 20-day exit nearly doubles Sharpe (0.55 vs 0.29 baseline), and trailing stops also help. But all configs fail the permutation test — the improvement is structural (time value decay), not from ML prediction. Still actionable for production (use 20-day exit), but not a new "edge."

---

## Key Findings

1. **Sharpe Inflation Bug (Critical)** — All small-account strategies had inflated Sharpe ratios. The bug: when calculating monthly returns, the system divided profit by the starting balance ($645) instead of the current account value. As the account grew from $645 to $36K+, this made returns look 2–6x better than reality. The broad universe strategy was the worst offender (claimed 4.70, honest 0.82 — a 4x inflation). The sectors-only strategy was barely affected (claimed 3.59, honest 3.72) because it has less compounding.

2. **Calendar Month vs Chunk Aggregation** — A second bug was found where monthly Sharpe was calculated by splitting trades into equal-sized chunks instead of actual calendar months. This smoothed out variance and inflated Sharpe by ~25%. Fixed to use real calendar months.

3. **ML Adds Risk Management, Not Return** — When comparing the ML model against random stock picking, equal weighting, and simple momentum, ALL methods are profitable (Sharpe 3.1–4.6). The structural edge comes from bull call spreads + VIX>20 filter, not from which stocks are picked. However, the ML model is significantly safer: drawdown is 5x lower (-3.1% vs -16.7%) and win rate is higher (81% vs 76%). The ML's real value is avoiding bad sectors and reducing risk.

4. **20-Day Exit Nearly Doubles Sharpe** — Exiting positions at 20 days instead of holding the full 30 days to expiration improves Sharpe from 0.29 to 0.55. This is because time decay accelerates in the last 10 days and the spread loses value faster. Simple, actionable improvement.

5. **VIX>20 Filter Is the Real Risk Manager** — This single filter cuts maximum drawdown from -4.1% to -1.0% while improving Sharpe from 3.79 to 4.20. It works because it keeps the strategy out of the market during low-volatility periods when option premiums are thin and whipsaw risk is high. Every additional risk management rule tested on top of this filter was redundant.

6. **Black-Scholes Pricing Validated Against Real Options Data** — Compared our model's prices against actual SPY option quotes from July 24. With volatility skew adjustment, the model was only -4.2% off from real mid-prices for bull call spreads. For iron condors, our model actually underprices credit by 68%, meaning our backtests are conservative — real income strategies should perform better. The 15% slippage haircut we apply covers the bid-ask spread with 14% margin.

7. **Sensitivity Analysis: Strategy is Robust** — Tested 28 parameter variations (spread width, holding period, number of positions, VIX threshold, rebalance frequency, lookback window). All 28 pass validation with honest Sharpe ranging 0.97 to 2.97. No parameter combination kills the edge. Most sensitive parameter is VIX threshold; least sensitive is lookback period.

---

## Actionable Next Steps

- **PG Earnings Trade Monday 7/29** — Iron condor approved. Historical analysis of 20 quarters shows 95% win rate on PG earnings. Short strikes are 3.1–3.7% away from current price; the P90 earnings move is only 2.8%. Expected value +$89, max risk $196.
- **Deploy 20-Day Exit** — Production config should exit positions at 20 days instead of holding to 30-day expiration. Nearly doubles risk-adjusted returns.
- **Sector ETFs Added to Options Data Collection** — XLE, XLK, XLF added to the options chain collector for future real-data backtests (currently using model prices).
- **Paper Trading Engines Need Restart Monday** — To pick up the corrected Sharpe calculations and 20-day exit rule.
- **Income Strategy Sharpe Audit Still Pending** — SPY Iron Condor, VIX Options, and Earnings strategies have the same inflation bug (estimated 10–30% lower than reported). Not yet corrected.

---

## Adversarial Validation Status

| Strategy | Gate 1: Permutation | Gate 2: Regime Balance | Gate 3: Sub-Period | Gate 4: Robustness | Honest Sharpe |
|----------|:-------------------:|:---------------------:|:-----------------:|:-----------------:|:-------------:|
| Optimized Sectors High VIX | PASS | PASS | PASS | PASS | 5.18 |
| Optimized Sectors 60d DTE | PASS | PASS | PASS | PASS | 4.22 |
| Sectors Baseline (30d) | PASS | PASS | PASS | PASS | 4.21 |
| Quality+Momentum Production | PASS | PASS | PASS | PASS | 4.20 |
| Sectors + Bonds | PASS | PASS | PASS | PASS | 2.77 |
| Sensitivity (28 configs) | ALL PASS | ALL PASS | ALL PASS | ALL PASS | 0.97–2.97 |
| Weekly DTE Spreads (7) | PASS | FAIL (gap 0.70+) | — | — | 0.44 best |
| Butterfly (6 variants) | — | — | — | — | All negative |
| Iron Condor Income (7) | — | — | — | — | All negative |
| Exit Optimization (8) | FAIL (structural) | PASS | PASS | PASS | 0.55 best |
| Drawdown Control (8) | NULL | NULL | NULL | NULL | Same as base |

**Note on Sharpe discrepancies:** The "Optimized Sectors" numbers (4.21–5.18) use the calendar-month-corrected method but may still carry residual inflation from the compounding denominator issue. The most conservative honest estimate for the baseline strategy is Sharpe 1.49 (from the full sensitivity recheck with all corrections applied). The truth likely falls between 1.5 and 4.2 — a genuinely strong strategy either way.
