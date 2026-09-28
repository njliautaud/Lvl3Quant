# Adversarial Validation: Stock Predictor v3 Enhanced
**Date:** 2026-07-15  
**Reviewer:** Separate adversarial agent (HC #665 compliance)  
**Script:** `/home/jupiter/Lvl3Quant/scripts/growth_research/stock_prediction/stock_predictor_v3.py`  
**Output:** `/home/jupiter/Lvl3Quant/output/growth_research/stock_prediction/v3_enhanced/`

---

## SESSION_STATE Claims
- L/S Sharpe: 0.72  
- CAGR: 21.3%  
- 193 stocks, 53 features, 93 WF folds  
- LGBM + XGBoost ensemble  
- R1 regime gap: 0.0 (PASS)  
- Permutation p=0.0 (signal real)  
- Lift 1.14x at 80% confidence  

---

## FINDINGS

### CRITICAL BUG #1: R1 Regime Test Is Completely Broken (DISQUALIFYING)

**The reported R1 gap of 0.0 does NOT mean "regime-neutral signal." It means the regime classifier silently failed.**

**Root cause:** The evaluation function classifies market regimes by looking for `macro_spy_mom_63d` in the prediction DataFrame rows. But when saving predictions, the code only saves `['date', 'ticker', 'sector', 'proba_lgbm', 'proba_xgb', 'proba_ensemble', 'y_true', 'fold', target_col, fwd_excess_cols]`. The macro column is **not included**. So the evaluation falls through to `spy_mom = 0` for every single date, which maps to `'flat'`.

**Result:** 100% of the 374,365 prediction rows were classified as `regime='flat'`. The R1 test never compared green days vs red days. The code then reports `r1_gap=0.0` because there's only ONE regime class — division trivially yields zero.

**Corrected R1 test** (merging proper SPY momentum from master panel):
- Green days (SPY 63d mom > 2%): 1,624 trading days, precision@0.55 = 38.7%  
- Red days (SPY 63d mom < -2%): 433 trading days, precision@0.55 = 44.2%  
- Flat days: 273 trading days, precision@0.55 = 35.5%  
- **Real R1 gap = 0.123 → PASS (< 0.50 threshold)**

The signal IS regime-neutral by the HC criterion, but this was luck — the test infrastructure was broken. The "PASS" in SESSION_STATE was based on a completely vacuous test. Must fix the evaluation code and re-run.

---

### CRITICAL BUG #2: Sharpe Is Inflated by ~1.69x Due to Wrong Annualization

**The code annualizes Sharpe as `mean / std * sqrt(12)`, treating the L/S returns as monthly.**  
But the underlying returns are **60-day forward returns** — approximately quarterly, not monthly.

**Correct annualization:** `sqrt(252 / 60) = sqrt(4.2) = 2.05`, not `sqrt(12) = 3.46`.

| Metric | Reported | Corrected |
|--------|----------|-----------|
| Sharpe (60d_3pct target) | 0.715 | **0.424** |
| Sharpe (60d_5pct target) | 0.641 | **0.380** |
| Sharpe (90d_5pct target) | 1.103 | **0.452** |

The "L/S Sharpe 0.72" in SESSION_STATE is actually **~0.42** when correctly annualized.

---

### CRITICAL BUG #3: CAGR Is Inflated by 2.86x

**The code computes CAGR as `mean_return * 12`** — treating 60-day holding-period returns as if they were monthly returns. There are ~4.2 non-overlapping 60-day periods per year, not 12.

| Metric | Reported | Corrected |
|--------|----------|-----------|
| CAGR (60d_3pct target) | 21.3% | **9.9%** |
| CAGR (60d_5pct target) | 17.2% | **7.9%** |
| CAGR (90d_5pct target) | 27.0% | **7.8%** |

The "21.3% CAGR" is actually **~9.9%** in excess of SPY.

---

### FINDING #4: L/S Returns Are Still Partially Overlapping

The code "de-overlaps" by taking every 21st observation from a dataset of daily observations, but 60-day returns overlap by a factor of 60/21 = 2.86. The correct approach is to sample every 60th observation.

When sampling every 60th observation (true non-overlapping), the result is only 33 data points — too few for reliable Sharpe estimation. This is a fundamental limitation of 60-day return modeling with 7.8 years of data.

---

### FINDING #5: Beta Asymmetry in L/S Portfolio

The long leg (top decile by model probability) has mean beta = **1.058** vs the short leg at **0.893** — a 0.165 difference.

This means the L/S portfolio has residual net-long market exposure of ~0.165 beta. In bull markets (2020, 2024), this generates spurious "alpha." In bear markets (2022), it creates additional L/S drag beyond the short exposure.

The 2020 result (mean 60d excess of 5.67% for high-confidence signals) is significantly amplified by this beta asymmetry — the long leg benefited from the sharp COVID rebound, and the short leg also recovered but from a worse position.

**Year-by-year L/S (non-overlapping every 21d):**
| Year | N | Mean 60d L/S | Sharpe (code inflated) |
|------|---|--------------|------------------------|
| 2018 | 4 | +0.2% | 0.06 |
| 2019 | 12 | +1.6% | 0.75 |
| 2020 | 12 | +11.7% | 3.33 ← outlier year |
| 2021 | 12 | -2.9% | -1.31 |
| 2022 | 12 | -2.1% | -0.58 |
| 2023 | 11 | +0.2% | 0.05 |
| 2024 | 12 | +3.6% | 1.33 |
| 2025 | 12 | +0.4% | 0.10 |
| 2026 | 6 | +11.1% | 3.18 ← partial year, only 6 obs |

**2020 and 2026 are extreme outliers. Remove them and the strategy barely works.**

---

### FINDING #6: Survivorship Bias — Mild But Present

The universe is a hand-picked list of current or recent S&P 500 members with known good outcomes. True survivorship bias is limited because:

1. No stocks fully disappeared from the dataset (no bankruptcies in the universe)  
2. PXD (Pioneer Natural Resources, acquired by Exxon 2024) is absent from OOT data entirely — could not have been traded on its inclusion/exclusion
3. ABNB and DASH only entered mid-2021 (their IPO dates), which is correct — yfinance stops at IPO

However, the universe selection itself is **look-ahead biased**: the 193 stocks chosen are all names that are prominent in 2026. Companies that were in the S&P 500 in 2017-2020 but later failed (GE at old weight, INTC at higher weight, numerous retailers) are underrepresented. This is a structural survivorship bias that will inflate any long-biased metric.

---

### FINDING #7: Permutation p=0.0 Is Uninformative

With 374,000 observations and only 200 permutation trials, a z-score of 17.97 was reported. This means the signal is statistically distinguishable from random — but with 374K rows and a 0.5% effect, almost anything is statistically significant. The p-value only says "the model learned something"; it says nothing about economic significance.

---

### FINDING #8: Lift of 1.14x at 80% Confidence — Is It Tradeable?

At the 80% confidence threshold:
- **229 independent signal events per year** (60d gap between signals per ticker)
- **Mean 60d excess return: +2.44% gross**
- After ~4bps round-trip spread cost: **+2.40% net per position**

The signal IS positive after costs for large-cap stocks (commission-free per HC #694). However:

- You only get 229 non-overlapping opportunities per year
- Position sizing requires running many simultaneously — max diversification requires ~20-30 concurrent positions
- The signal is weak: 56.9% of high-confidence picks DO NOT deliver the expected excess return
- In 2021, 2022, 2023, and 2025, the strategy underperformed or was flat

**Verdict:** The signal passes a basic tradability test on paper but is not strong enough to build a business around. It's an overlay, not a strategy.

---

### FINDING #9: Model vs Simple Momentum

Corrected comparison (same methodology, proper annualization):

| Strategy | Sharpe (corrected) | CAGR excess (correct) |
|----------|--------------------|-----------------------|
| Model (60d_3pct) | 0.42 | 9.9% |
| Simple 63d Momentum | 0.15 | 3.7% |
| Random | -0.32 | -3.2% |

The model does meaningfully outperform simple momentum: +6.2% additional excess CAGR. This is real incremental value — the ensemble adds something beyond naive momentum. But the absolute performance (Sharpe 0.42) is not production-grade without further work.

---

## VERDICTS BY CLAIM

| Claim | Verdict | Notes |
|-------|---------|-------|
| L/S Sharpe 0.72 | **WRONG** | Actual corrected Sharpe ≈ 0.42 due to annualization bug |
| CAGR 21.3% | **WRONG** | Actual corrected CAGR ≈ 9.9% excess due to same bug |
| R1 gap = 0.0, PASS | **VACUOUS** | Regime classifier silently failed; real gap = 0.12, still PASSES |
| perm p = 0.0 | **TRUE BUT WEAK** | Real effect confirmed but z-score overstated at this N |
| Lift 1.14x at 80% | **APPROXIMATELY TRUE** | Confirmed; mean excess = +2.44% per 60d position |
| 193 stocks, 53 features, 93 folds | **TRUE** | Verified |
| Sliding window | **TRUE** | 252-day sliding train confirmed |
| 60d embargo | **TRUE** | Verified in fold structure |

---

## REQUIRED FIXES

1. **Fix regime classification in `evaluate_predictions()`** — include `macro_spy_mom_63d` in the prediction DataFrame saved to disk, OR look it up from the master panel during evaluation. Re-run evaluation.

2. **Fix Sharpe annualization** — change `* np.sqrt(12)` to `* np.sqrt(252 / hold_days)` where `hold_days=60`.

3. **Fix CAGR calculation** — change `mean_ret * 12` to `mean_ret * (252 / hold_days)` = `mean_ret * 4.2`.

4. **Re-run evaluation** with all three fixes. After correction: expected Sharpe ~0.40-0.45, expected CAGR ~9-10% excess.

5. **Add year-by-year breakdown to the standard report** — the current summary hides that 2020/2026 drive most of the positive attribution.

6. **Remove 2026 partial year from CAGR calculation** — only 6 observations in the non-overlapping sample, extremely high mean (11%), distorts overall.

---

## BOTTOM LINE

The signal is real. It beats random and beats simple momentum. But the headline numbers (Sharpe 0.72, CAGR 21.3%) are **both wrong due to the same annualization error** applied to a 60-day return horizon. True corrected numbers are approximately Sharpe 0.42 and 9.9% excess CAGR.

The R1 pass was vacuous — the regime test never ran. When run correctly, the signal does pass R1 (gap 0.12 < 0.50 threshold).

The strategy is a **promising overlay** — 229 tradeable signals per year, each with +2.4% expected edge at 80% confidence — but it's not a standalone strategy and the year-by-year stability is poor (negative in 2021, 2022, flat in 2023 and 2025).

Do not deploy as-is. Fix the three calculation bugs, re-evaluate with corrected numbers, and treat as a long-term position overlay rather than a high-frequency alpha source.
