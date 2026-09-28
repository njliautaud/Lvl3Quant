# Sector Rotation Timing & Optimal Entry Patterns — Deep Research Findings

**Date**: 2026-08-23
**Data**: 11 SPDR sector ETFs (XLK, XLF, XLE, XLV, XLI, XLC, XLY, XLP, XLB, XLU, XLRE) + SPY + VIX
**Daily data**: 2000-01-03 to 2026-08-21 (6,700 days; XLC from 2018, XLRE from 2015)
**Hourly data**: 2023-09-25 to 2026-08-21 (730 trading days, 5,072 bars)

---

## 1. Time-of-Day Return Patterns

### 1A. First Hour Dominance

The first trading hour (9:30-10:30 ET) produces statistically significant positive returns across most sectors (annualized Sharpe 0.59-1.52). XLF, XLI, XLC show significant first-vs-last hour differences (p<0.01).

### 1B. Regime-Stratified

First-hour alpha is entirely a low-VIX phenomenon (+9.92 bps low VIX vs -2.04 bps high VIX). Do NOT enter dips in the first hour during high-VIX regimes.

### 1C. Same-Day Dip Buying

Consistently negative. Entering at 9:30 on a dip day averages -38 bps to close. Multi-day holding (10d avg) is essential.

### 1D. Day-of-Week

No significant effects.

---

## 2. Sector Rotation Timing

### 2A. Leadership Persistence

Monthly sector leadership changes every ~4 days (median 2). XLK dominates leadership frequency. XLE shows longest streaks when it does lead.

### 2B. Mean Reversion vs Momentum

Neither shows robust alpha at sector level across 35 lookback/holding combinations. The dip strategy's edge is NOT simple momentum or mean reversion.

### 2C. Lagging-to-Leading Transition

Bottom-tercile sectors have 40.4% probability of reaching top tercile next quarter (vs 33% baseline). Mild mean-reversion at quarterly horizon.

### 2D. Rotation Cycle

Monthly ranks decorrelate within 1 month. Quarterly ranks persist ~6 weeks then mean-revert. Rotation cycle: 4-6 weeks persistence, then reversal.

---

## 3. Fundamental Factor Screens

No significant findings. PE, relative valuation, RS acceleration proxies show no meaningful predictive power for sector dip recovery. The strategy's edge is purely technical/structural.

---

## 4. Cross-Sector Correlation Dynamics (Strongest Findings)

### 4A. Correlation Spikes (p < 0.0001)

During 63d correlation spikes (>1.5 std above mean, 8.9% of days):
- Spike regime: +1.63% 10d recovery, 68% WR, n=847
- Normal regime: +0.69% 10d recovery, 59% WR, n=6295
- This is the most statistically significant finding in the entire study.

### 4B. VIX x Drawdown Interaction

- High VIX + Deep dip: +1.68%, 68% WR (BEST setup)
- Low VIX + Deep dip: +0.26%, 56% WR (WORST setup)

### 4C. Sector Dispersion

U-shaped: both low AND high dispersion produce good recoveries. Worst at moderate dispersion.

### 4D. High Correlation = Better Forward Returns

SPY 21d forward return by 63d correlation quintile:
- Q1 (low corr): -0.25%, 54% WR
- Q5 (high corr): +2.92%, 72% WR

---

## Suggested Strategy Enhancements

1. **Correlation regime sizing**: Increase allocation when 63d avg pairwise correlation > 0.73 (top decile)
2. **VIX-depth interaction filter**: Require VIX > median for deep dips (>5%); skip deep dips in low VIX
3. **Lagging sector preference**: When multiple sectors dip, prefer worse 63d relative performance
4. **Entry timing**: Avoid first-hour entries during elevated VIX; prefer afternoon entries (13:30-15:00)
