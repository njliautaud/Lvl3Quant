# Jupiter All-Night v3.2 Edge Research — FINAL VERDICT
_Generated 2026-05-14 ~01:00 ET, after 8 sequential passes (orchestrator + passes 2-8)_

## TL;DR

**v3.2 has NO statistically significant tradable edge under realistic FIFO execution that can be attributed to the model's predictions.**

After exhaustive search across 8 research passes — confluence gates, time-of-day, vol regimes, FIFO ground-truth labels, meta-MLP, contextual bandits, RL bandits, per-head ablation, leave-one-day-out CV, and permutation testing — the headline candidate (**S_Top5% × confluence × golden-ToD × vol_mid**, n=262, Sharpe 3.85) **fails the permutation test (p=0.39)**. Random sign-shuffles produce essentially the same Sharpe.

The "edge" we found is a structural artifact of (a) the post-hoc filter selection on the same 5 OOT days and (b) passive-fill selection bias under FIFO mechanics — NOT from the model's signal.

**Recommendation**: Do NOT deploy v3.2 live. Wait for v3.3 fold 0 OOT (~09-11 ET morning) and the HC #337 extended-OOT (~10 days vs current 5).

---

## Pass-by-pass synopsis

### Orchestrator Tasks 1-10 (00:25-00:35 ET)
- Task 1 (unit-fix HC #345): 15 cells passive-cost-positive, 2 market-cost-positive of 40
- Task 2 (edge-decay): 1s edge strongest, decays to ~0 by 30s as expected
- Task 3 (ToD): no robust ToD pattern at single-task granularity
- Task 4 (confluence gates): best PnL t = -0.064 across 64 binary configs — none net positive
- Task 5 (vol regimes): no robust vol-regime split
- Task 6 (FIFO queue back-off): 8-config sweep, no winner
- Task 7 (aggressive cross): 10-config sweep, all losing
- Task 8 (meta MLP): mean AUC 0.518, no filter improvement
- Task 9 (contextual bandit): policy +0.53t vs oracle +3.25t — bandit underperforms even random
- Task 10 (per-head ablation): baseline AUC 0.511 across 32 heads — no head dominates

### Pass 2 (long-IOC drill)
- Hypothesis: pass-1's "+6t long-IOC profit" was real
- Result: ORACLE artifact. With realistic TP/SL, long-IOC LOSES -2 to -3t/trade
- ALL 28 TP/SL combos tested lose money
- Best Sharpe-toy = -7.89

### Pass 3 (realistic-exit + FIFO ground truth)
- H1 (long fixed-hold): initially looked +5t profitable
- **H2-H5 on FIFO labels**: short-side Top0.1% × agree_15 = +1.82t/fill, WR 82.1%, Sharpe 4.32
- H1 numbers turned out to be a unit bug (caught in pass 4)

### Pass 4 (validation)
- Caught the unit bug: target_log_ret_* fields are z-scored, not raw log returns
- Pass 3's H1 long-IOC numbers were inflated ~20000×
- Long-side claims withdrawn

### Pass 5 (calibrated re-validation)
- Used z2t_30s = 0.881 calibration anchored to sd_mfe30_ticks=4.992
- **All long bands NEGATIVE on FIFO ground truth** (95% CI)
- All short bands marginally positive but **CI spans zero**
- Per-day robustness: short edge concentrated on day 2 only

### Pass 6 (positive-pocket search)
- Stacked: Top0.5% × agree_15 × no_reversal × golden-ToD × vol_mid
- n=19, mean +1.29t, Sharpe 2.70, **95% CI [+0.29, +2.13] — first config with positive CI**
- 3 of 3 trade-days positive, 0 negative

### Pass 7 (robustness battery)
- LODO: ALL 19 fills fall on day 2 → not a strategy, one good day
- Permutation test on Top0.5% pocket: p=0.046 (borderline)
- ToD bucket sensitivity: shifting buckets ±30 min → Sharpe collapses to -1.71
- Drop-one ablation: dropping vol_mid → 33 fills Sharpe 1.37; dropping ToD → 54 fills Sharpe 0.72
- **Surprise**: loose Top5% same filters → 262 fills, Sharpe 3.85

### Pass 8 (Top5% loose-band validation) — the truth-teller
- T1 per-day: 239 of 262 fills on day 2 (91% concentration!)
- T3 ToD coverage: 197 of 262 fills in 11:18-12:00 bucket alone (75%)
- T5 half-split: first 2 days mean -0.55, Sharpe -0.53; last 3 days mean +0.69, Sharpe 4.10 → NOT time-stable
- **T4 permutation: observed Sharpe 3.85, null mean 3.66, p=0.39 → NOT statistically significant**
- T8: even Top5% RAW (no filter) gets Sharpe 3.51 — the "edge" is from the FIFO+ToD selection structure, not the model

---

## Why Pass 8 kills the candidate

Permutation test logic: shuffle the SIGN of pred_5s 1000× while keeping magnitudes fixed and the filter structure intact. If the model has real signal, observed Sharpe should fall in the right tail of the null distribution.

| Statistic | Value |
|---|---|
| Observed Sharpe | 3.85 |
| Null mean Sharpe | 3.66 |
| Null 95th percentile | 4.90 |
| Null 99th percentile | 5.35 |
| **p-value (one-tail)** | **0.39** |

39% of random label shuffles produce ≥ observed Sharpe. The model's signal is statistically indistinguishable from random under this filter setup.

---

## What we DID learn (positive findings, structural not predictive)

1. **Passive shorts during 11:18-12:00 ET were systematically profitable across these 5 days** — likely a market microstructure effect (lunch liquidity dry-up, dealer hedging unwind), but needs many more OOT days to confirm
2. **Wider FIFO stops (tp8sl5) outperform tighter (tp4sl3) for shorts** consistently — a real execution finding regardless of model
3. **Long-side has NO edge under FIFO** at any band/filter combination in 5 OOT days
4. **Adverse selection eats every passive fill** even at 60-70% directional WR
5. **target_log_ret_* fields in v3.2 NPZ are z-score normalized** (not raw log returns) — must be calibrated by sd_mfe30_ticks for any tick-based analysis

---

## Recommended next steps

1. **DO NOT deploy v3.2 live** — would waste capital
2. **Wait for v3.3 fold 0 OOT** (~09-11 ET, fold trainer must release lock)
3. **Run HC #337 extended-OOT on v3.2** (10+ days instead of 5) — if the 11:18-12:00 short pattern persists, it's a structural finding worth a static rules executor
4. **Re-run permutation tests on v3.3** when ready — the bar is null Sharpe << observed Sharpe with p < 0.01
5. **Stop optimizing v3.2** — diminishing returns; resources better spent on v3.3 evaluation and Razer live infrastructure (HC #328 4-component completion)

---

## Files generated

- `task_01_unit_fix/` ... `task_10_per_head_ablation/` (orchestrator)
- `pass2_long_ioc/` (long-IOC drill)
- `pass3_realistic/` (FIFO + time-based exits)
- `pass4_validate/` (caught unit bug)
- `pass5_calibrated/` (calibrated re-validation, all-bands honest table)
- `pass6_pocket/` (positive-pocket search, found Top0.5% Sharpe 2.70)
- `pass7_robustness/` (LODO killed Top0.5%, R4 found Top5% candidate)
- `pass8_top5/` (permutation test killed Top5% candidate)
- `ALLNIGHT_FINAL_SUMMARY.md` (this file)

Total compute: ~3 minutes Jupiter CPU across all passes.
