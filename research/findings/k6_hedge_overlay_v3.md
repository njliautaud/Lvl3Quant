# K=6 Hedge Overlay v3 — SPY Short Overlay (2026-06-10)

**Verdict: ALL 5 VARIANTS REJECTED for HC #428 R1 (gap ≤ 0.50). Hedging narrows the gap far more than v1/v2 abstention did, but cannot close it.**

## Setup
- Book: K=6 megacap momentum (`book_K6_mom60.parquet`, 1505 OOT days 2020-01→2025-12).
- LGBM 1d-horizon E[ret] from the cached v1 18-feature matrix (features lagged t-1), SLIDING 24m/6m/3m WF (HC #0), reusing v2 machinery.
- Hedge: short SPY sized by trailing 60d realized beta (shift(1), ≤ t-1 info, clipped [0,2], mean ≈ 1.05). Cost: 1bp per hedge-adjustment day.
- Degenerate-fold guard: v2 fold 3 (OOT 2020-10-01) reproduced — best_iter=1 constant predictions, NaN IC, caused by tiny early train set (n=165). v3 detects pred-std < 1e-12 and DROPS the fold (1 dropped). Other folds healthy.

## Results (baseline: Sharpe 2.34, Calmar 5.36, green +9.86 / red −7.38, gap 1.75)

| Variant | Sharpe | Calmar | MaxDD | green Sh | red Sh | gap | R1 | hedge cost |
|---|---|---|---|---|---|---|---|---|
| static_beta | 2.24 | 4.46 | −18.9% | +4.01 | +0.03 | **0.99** | FAIL | 11.05% total |
| cond_m10bps | 2.33 | 5.32 | −19.6% | +9.81 | −7.32 | 1.75 | FAIL | 0.25% |
| cond_0bps | 2.34 | 5.34 | −19.6% | +9.67 | −7.13 | 1.74 | FAIL | 0.79% |
| scaled_k25 | 2.33 | 5.32 | −19.6% | +9.80 | −7.32 | 1.75 | FAIL | 0.81% |
| scaled_k50 | 2.34 | 5.33 | −19.6% | +9.83 | −7.35 | 1.75 | FAIL | 0.82% |

Day-concentration ≤ 0.70 passes everywhere (HC #344).

## Diagnosis
1. **Conditional/scaled variants are no-ops.** The LGBM E[ret] is < 0 on only 59 of 1444 predicted days (the book's unconditional mean is strongly positive and the 1d IC is weak, ~0.05-0.1 mid-sample). The signal almost never triggers the hedge, so red-day exposure is untouched.
2. **Static full-beta hedge works as a hedge** — red-day Sharpe goes from −7.38 to +0.03 (beta fully neutralized, residual is pure stock-specific alpha) at modest cost (Sharpe 2.34→2.24, Calmar 5.36→4.46). But the residual gap is now **alpha asymmetry**: the K=6 momentum alpha itself only pays on green days (+4.01 vs +0.03). gap = 3.98/4.01 = 0.99.
3. **Implication:** the regime gap is not (only) beta — the *alpha* is regime-conditional. No SPY-sized overlay can fix that; passing R1 would require red-day positive alpha (e.g., a short-leg stock book or red-regime alternative sleeve), or relaxing R1 for beta-hedged residuals.

## Next candidates (not launched)
- Pair the beta-hedged book with a sleeve that earns on red days (e.g., short weakest-momentum megacaps) — long/short market-neutral K=6.
- Test whether static_beta's red Sharpe ≈ 0 with green ≈ +4 is acceptable under a revised gate interpretation (escalate: policy question, not engineering).

MLflow: experiment `k6_hedge_overlay_v3` (parent + 5 nested). Artifacts: `output/macro_picker/k6_hedge_overlay_v3/` (report.json, hedged books, OOT predictions). Live K=6 paper state untouched.
