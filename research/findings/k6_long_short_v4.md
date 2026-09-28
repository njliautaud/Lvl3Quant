# K=6 Long/Short Market-Neutral v4 — Short Weakest Megacaps (2026-06-10)

**Verdict: ALL 6 VARIANTS REJECTED. The short leg has NEGATIVE standalone alpha — shorting bottom-momentum megacaps in 2020-25 loses money even on red days. Dollar-neutral variants narrow the gap only by destroying both sides (Sharpe ≤ 0.05). The K=6 alpha is long-only; R1 cannot be passed by shorting within this 8-name universe.**

## Setup
- Script: `strategy/macro_picker/k6_long_short_v4.py` (v1/v2/v3 untouched).
- Long leg: identical top-6 mom60 ridge picker, same SLIDING 24m/6m/3m WF (HC #0), same regime gate (SPY>50dMA & VIX<25, both legs), same baseline harness incl. overlapping-OOT fold-sum convention (~2x fold-ensemble; Sharpe/gap/day-conc scale-invariant). **Harness validation: n_short=0 reproduces the baseline book exactly (max daily diff 1.4e-17, corr 1.000000).**
- Short leg: bottom-N of the SAME 8-name universe, netted per name (8-name universe forces overlap: N=6 @ 1.0x ≈ long ranks 1-2 vs short ranks 7-8 at 1/6 each).
- Costs: baseline per-name model ($0.005/sh + 1bp slippage) on |Δ net weight| at every rebal, PLUS 25bps/yr borrow on gross short notional accrued daily (megacaps = GC, cheap to borrow). Borrow totaled 0.26-1.32% over 6yrs — immaterial.
- Leakage: rankings from ≤t−1 closes; beta-neutral sizing uses per-stock trailing 60d beta vs SPY, shift(1); day-t SPY for EVAL stratification only. Degenerate-score fold guard active (0 folds dropped — ridge never degenerate).

## Results (1505 OOT days 2020-2025; comparators: baseline 2.34/5.36/gap 1.75, v3 static-beta 2.24/4.46/gap 0.99)

| Variant | Sharpe | Sortino | Calmar | MaxDD | PF | green Sh | red Sh | gap | R1 |
|---|---|---|---|---|---|---|---|---|---|
| ls_n3_s050 | 1.52 | 1.82 | 1.70 | −21.4% | 1.39 | +7.08 | −5.63 | 1.79 | FAIL |
| ls_n3_s100 | −0.43 | −0.43 | −0.17 | −73.3% | 0.91 | −0.24 | −0.66 | 0.64 | FAIL |
| ls_n6_s050 | 2.00 | 2.45 | 3.61 | −12.5% | 1.55 | +8.53 | −6.43 | 1.75 | FAIL |
| ls_n6_s100 | 0.05 | 0.05 | −0.01 | −40.3% | 1.01 | +0.63 | −0.94 | 1.67 | FAIL |
| ls_n3_betaneutral (mean scale 0.98) | −0.30 | −0.30 | −0.14 | −68.3% | 0.93 | +0.33 | −1.10 | 1.30 | FAIL |
| ls_n6_betaneutral (mean scale 1.01) | 0.00 | 0.00 | −0.02 | −43.2% | 1.00 | +0.59 | −0.89 | 1.66 | FAIL |

Day-concentration ≤0.70 passes everywhere (≤0.011, HC #344).

## Diagnosis
1. **The short leg is negative alpha, not red-day insurance.** At 1.0x (dollar-neutral) the book collapses (Sharpe −0.43 to +0.05) and — critically — **red-day Sharpe stays NEGATIVE** (−0.66 to −0.94). The weakest-momentum megacaps do not fall on red days more than the strongest; in 2020-25 every name in this universe trended up, so shorts bleed in all regimes.
2. **0.5x sizing is just a worse beta hedge**: keeps gap ≈1.75-1.79 while paying away Sharpe (2.34→1.52-2.00). Strictly dominated by v3 static SPY hedge (2.24/gap 0.99).
3. **Beta-neutral sizing ≈ dollar-neutral** (long/short legs have near-equal trailing betas, mean scale ≈1.0) — same failure.
4. **Cross-sectional momentum spread within 8 megacaps carries no spreadable alpha**: top-vs-bottom L/S is roughly zero-PF (1.00-1.01 at N=6). The K=6 edge is the long beta+momentum tilt itself.

## Disposition
- P-item closed NEGATIVE. The "short weakest megacaps" route from the v3 findings is exhausted within this universe. Together v1-v4 establish: K=6's regime gap is irreducible by (1) abstention, (2) index hedging, (3) intra-universe shorting.
- Remaining honest options: (a) widen the short universe beyond the 8 megacaps (different strategy, new P-item), or (b) the standing policy question from v3 — whether static_beta's red Sharpe ≈ 0 / green +4.01 profile is acceptable under a revised R1 interpretation (escalated, user call).
- Live K=6 paper state UNTOUCHED.

MLflow: experiment `k6_long_short_v4` (id 317881362195307530, parent + 6 nested, regime_gap + passes_hc428_r1 logged). Artifacts: `output/macro_picker/k6_long_short_v4/` (report.json, per-variant book/rebal parquets, run.log). Wall time 90s.
