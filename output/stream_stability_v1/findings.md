# HC #488 R2 + HC #486 R3: Stream Stability Analysis

## Executive Summary

**Verdict: KILL the sign-consistency gate on 4/29. CONDITIONAL KEEP on 4/28, but insufficient cross-day support.**

The stream-coherence metric (sign_consistency_K20) shows **strong correlation with P&L on 4/28** but **zero correlation on 4/29**, violating HC #488 R2 (regime-agnostic ≥40-day validation). On 4/28 alone, the gate delivers +114% lift in mean P&L (0.60 → 1.28 ticks) and +6.1% WR lift, but this is NOT cross-regime stable.

---

## A. Baseline Performance (Top-1% Long 5s)

| Metric | 4/28 | 4/29 |
|--------|------|------|
| Base mean ticks | 1.179 | 4.381 |
| Base WR | 62.7% | 51.3% |
| Sample size | 23,681 | 26,696 |
| **Regime** | Green (ES +0.75%) | Red (ES -0.85%) |

---

## B. Stream Coherence Gate: Top-Decile vs Baseline

**4/28 (Green Day):**
- Top-decile (sign_consistency_k20 = 1.0): **1.281 ticks, 63.6% WR** (20,147 trades)
- Bottom-decile (sign_consistency < 0.95): **0.599 ticks, 57.5% WR** (3,534 trades)
- **Lift: +0.683 ticks (+114% vs bottom), +6.1% WR**
- Trades retained: 85% of top-1% (20,147 / 23,681)

**4/29 (Red Day):**
- Top-decile + rest: **4.381 ticks, 51.3% WR** (all 26,696 trades)
- Bottom-decile: **4.381 ticks, 51.3% WR** (all 26,696 trades)
- **Lift: 0.0 ticks, 0.0% WR delta**
- No variance in sign_consistency_k20 > 0.95 (99.4% of trades at ceiling)

---

## C. Statistical Significance

| Test | 4/28 | 4/29 |
|------|------|------|
| Spearman ρ | +0.086 | -0.002 |
| p-value | 1e-10 (highly significant) | 0.80 (not significant) |
| Effect size | Weak but consistent | Null |

**Interpretation:**
- 4/28: sign_consistency ranks-orders P&L significantly (p < 0.001)
- 4/29: zero relationship; gate has no predictive power

---

## D. Stream Metrics: Best vs Worst Predictions

**4/28 Top-1% (N=23,681):**
| K (window) | Sign Consistency | Flip Rate (/s) |
|------------|------------------|----------------|
| K=4 | 0.998 | 0.00 |
| K=20 | 0.974 | 0.14 |
| K=40 | 0.933 | 0.33 |
| K=80 | 0.884 | 0.52 |

**4/29 Top-1% (N=26,696):**
| K (window) | Sign Consistency | Flip Rate (/s) |
|------------|------------------|----------------|
| K=4 | 0.999 | 0.01 |
| K=20 | 0.995 | 0.06 |
| K=40 | 0.988 | 0.12 |
| K=80 | 0.982 | 0.18 |

**Insight:** 4/29 predictions are MORE coherent overall (99.5% vs 97.4% at K=20) but do NOT produce better P&L. This breaks the HC #486 hypothesis: prediction coherence ≠ trading edge on red days.

---

## E. Cross-Day Stability

**HC #488 R2 Regime-Agnostic Gate Check:**

The sign-consistency gate FAILS regime stability (green vs red day):
- Green day (4/28): Gate adds +114% lift → **effective**
- Red day (4/29): Gate adds 0% lift → **ineffective**
- Ratio: |lift_green - lift_red| / max(|lift_green|, |lift_red|) = |114 - 0| / 114 = **100% ≫ 0.50 threshold**

**REJECTED per HC #488 R2.** This is a **regime-tailored gate**, not a regime-agnostic edge.

---

## F. Verdict: KILL

**Gate effectiveness:**
- ❌ **Fails HC #488 R2:** Regime-dependent (100% lift variance > 50% threshold)
- ❌ **Fails HC #486 R3:** Cross-day validation shows zero correlation on red day
- ❌ **Insufficient sample size:** Only 2 OOT days; need ≥40 days all-regime validation
- ⚠️ **4/28 correlation is real but fragile:** Single-day statistical significance without red-day replication = overfitting to green-day patterns

**Recommendation:** Do NOT stack sign_consistency_k20 on top of the base +0.26-tick alpha. The apparent 4/28 lift is likely regime-specific luck.

---

## G. Data Caveats

1. **OOT_STRIDE=5** means predictions downsampled 5x from raw MBO stream (~50-250ms cadence in downsampled space).
2. **Time window estimates** (K → seconds) are rough: K=20 ≈ 4-5s estimated at ~200ms/event. Actual cadence varies by market volume.
3. **Top-1% thresholds vary by day** (1.378 on 4/28 vs 1.653 on 4/29), reflecting regime shift in prediction magnitudes.
4. **No 4/27 fold_00 data** available; full 40-day analysis blocked by checkpoint availability.
5. **Red-day trades underperform baseline significantly** (4.38 ticks is WELL below the verified +0.26-tick long-only edge from HC #489). This suggests the top-1% long bucket may be reversed-polarity on red days or the model's confidence is miscalibrated for short horizons on down-days.

---

## Next Steps

**DO NOT USE this gate.** Instead:
- Return to base +0.26-tick long-5s alpha (HC #489 validated cell)
- Focus on execution optimization (FIFO sweep, fee/slippage subtraction) rather than prediction filtering
- If stream coherence matters, validate on FULL 40+ day OOT set with regime stratification
