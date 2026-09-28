# K=6 Meta-Gate v2 (Regime-Asymmetric Regression) — Findings

**Run date**: 2026-06-10
**Script**: `strategy/macro_picker/k6_meta_classifier_v2.py` (v1 untouched)
**Node**: Jupiter (CPU)
**MLflow experiment**: `k6_meta_classifier_v2` (id 463239843526605570) — parent run `af98e42126824735a4e6bb26ef8deddd` + 4 nested variant runs
**Outputs**: `output/macro_picker/k6_meta_classifier_v2/` (report.json, run.log, per-variant gated_book/oot_predictions/per_day parquets)
**Verdict**: **NEGATIVE on the deploy-gate objective — DO NOT DEPLOY.** No variant closes the HC #428 R1 regime gap (all ~1.73–1.75 vs threshold ≤0.50). One variant (h5_redonly) is mildly accretive on Sharpe/Calmar but does not change the regime verdict.

---

## 1. Setup

Implements v1 findings next-iteration candidates (1)-(4) as a 2x2 sweep:
- **(a)** LGBM REGRESSION on forward K=6 return; gate (cash) when E[ret] < −10 bps/day.
- **(b)** Red-only conditioning: gate eligible only when SPY t−1 return < −0.1% (leakage-free decision regime; day-t SPY used for EVAL stratification only).
- **(c)** 5d-horizon target: forward 5d mean daily return.
- **(d)** Fold filter train pos_rate < 0.15 (never triggered — early 2018-19 folds already excluded by target-NaN drop since the K6 book starts 2020; effective folds 21-22).

Harness identical to v1: SLIDING 24m/6m/3m (HC #0), same cached 18-feature matrix (features lagged 1d at build). Leakage audit: 5d train targets required to end strictly before train_end; OOT preds deduped keep-first.

## 2. Results (1505 OOT days, 2020-2025)

| Variant | Sharpe | Calmar | MaxDD | PF | WR% | Green Sh | Red Sh | Gap | R1 ≤0.50 | Skip days |
|---|---|---|---|---|---|---|---|---|---|---|
| Baseline ungated | 2.34 | 5.36 | −19.6% | 1.68 | 36.7 | +9.86 | −7.38 | 1.75 | FAIL | 0 |
| h1_alldays | 2.32 | 5.27 | −19.6% | 1.68 | 36.3 | +9.78 | −7.32 | 1.75 | FAIL | 15 |
| h1_redonly | 2.34 | 5.35 | −19.6% | 1.69 | 36.6 | +9.83 | −7.34 | 1.75 | FAIL | 6 |
| h5_alldays | 2.21 | 4.28 | −21.4% | 1.65 | 35.2 | +9.58 | −7.09 | 1.74 | FAIL | 44 |
| **h5_redonly** | **2.42** | **5.58** | −19.6% | **1.72** | 36.5 | +9.87 | −7.24 | **1.73** | FAIL | 16 |

## 3. Diagnosis

1. The regressor almost never predicts E[ret] < −10bps at 1d horizon (6-15 skip days of 1505) — macro features cannot forecast day-1 megacap losses with that confidence. The asymmetric gate is correctly conservative but therefore nearly a no-op.
2. h5_redonly is the only accretive cell (+0.08 Sharpe, +0.22 Calmar, PF 1.68→1.72) — skipping 16 red-regime days with predicted negative 5d drift removes more losers than winners. Real but tiny edge; per-fold IC is noisy (−0.12 to +0.32).
3. **The regime gap is structural to K=6 itself** (long-only megacap momentum is mechanically pro-green-day). No overlay that merely zeroes a handful of days can move the gap from 1.75 toward 0.50 — that would require hedging/shorting on red days, not skipping.

## 4. Disposition

- P-item closed NEGATIVE for the R1-gate objective. h5_redonly logged as a possible small-accretive overlay if K=6 ever passes R1 by other means (e.g. SPY-hedged book), not as a standalone fix.
- Live K=6 paper state UNTOUCHED.
