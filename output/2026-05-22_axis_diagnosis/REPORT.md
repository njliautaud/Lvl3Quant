# 2026-05-22 — Synthesis: 10 Axes Tested, Structural Diagnosis

**Generated**: 2026-05-22 ~14:15 ET
**Compliance**: HC #488 R4 (axis rotation), HC #428 (deploy gates), HC #74 (FIFO canonical), HC #486 (stream thesis).
**Scope**: Every research axis exercised today on the short_10s @ 0.55 meta-classifier survivor (the only result from prior work that produced positive per-trade P&L: +5.03 t/trade FIFO across 15 OOT days, 8/15 profit days = 53%).

---

## TL;DR for the operator

- The base v3.4.2 model has a real but **very thin** edge: pooled IC(pred10s, realized10s) = **+0.036** across 32 OOT days. Half a standard-deviation above noise.
- The +5.03 t/trade survivor is REAL on per-trade basis but cannot clear the 11/15 profit-days deploy bar.
- Nine post-hoc transformations of the survivor failed (filters × 7, sizing, horizon ensemble).
- One independent signal source (trade-tape imbalance) failed as alpha but produced a **liquidity-provision finding** that connects to why the survivor works at all.
- **15-day OOT sample is the structural ceiling.** No statistical machinery can manufacture deploy-grade confidence at this sample size.

**Recommended next action** (ranked by expected leverage / effort):
1. **Wait for Neptune retrain fold-0 OOT** (ETA ~16:30 ET) — bias-fixed labels + corrected NaN audit may lift IC above 0.036.
2. **Extend OOT collection** — need 30+ OOT days for any day-gate to survive forward-walk.
3. **Regression-target meta-retrain** — replace binary-winner target with realized-ticks-net; OFI ranked #1 in trade-classifier importance, magnitude information being thrown away.
4. **Investigate 30s head calibration bug** — broken across all current models; fixing it may add an independent gating dimension.
5. **Liquidity-provision execution model** — use the negative tape-imbalance IC as input to a "be the absorber" passive-limit policy (NOT a directional alpha).

---

## All 10 Axes — Verdict Table

| # | Axis | What was tested | Pooled net t/trade | Profit-days | Verdict |
|---|---|---|---|---|---|
| 0 | **Baseline (no gate)** | short_10s @ meta_prob ≥ 0.55, FIFO replay | **+5.03** | **8/15 (53%)** | **PARTIAL** (fails 11/15 gate) |
| 1 | Conformal wrapper (morning) | Calibrated intervals from v3.4.2 point preds, gate on interval width | n/a | n/a | REJECT — width informative only on longs |
| 2 | Microprice gate (morning) | Skip signal if microprice drifting against direction | random 49.6% | n/a | REJECT — decoupled from book |
| 3 | TOD × velocity stratification | Top 6 cells from label-level stratification, then FIFO replay | -0.07 to -0.27 vs label | 0/6 cells pass | REJECT — label-level edge dies in FIFO |
| 4 | Adaptive exit policy v1 | Stream-coherent in-trade exit (clean MBO replay, no leak) | -0.38 | 0/5 OOT days | REJECT — v0's claim was leak-driven |
| 5 | Day-classifier ES-only | LGBM + single-feature gate on pre-market features | LGBM AUC 0.18 (overfit); single-feature `trend_open_to_945 asc` LOO AUC 0.76 → forward-walk **+2.72 collapse to -0.22** | 17% forward | REJECT under forward-walk |
| 6 | Day-classifier cross-asset | Add VIX/NQ/YM/calendar; LOO AUC + logistic combo | VIX_change_5d AUC 0.85; under forward-walk: VIX_desc 0/15 folds, VIX_asc 9/15 (60%, below 10/15) | 1/3 forward | REJECT under forward-walk |
| 7 | Trade-level classifier | LGBM + LogReg on intraday features per trade; OOS τ-sweep | OOS AUC 0.535 ± 0.118 (fold range 0.40–0.67); LogReg 0.488 (disagrees) | best τ 71% (only 7 unique OOS days) | REJECT — fold-specific noise |
| 8 | Kelly / position sizing | 6 schemes (linear/quadratic/threshold/inverse-vol/Kelly) with bootstrap day-Sharpe CI | best +5.69 (B_linear) | **7/15 = 47% (WORSE)** | REJECT — same days, lower notional |
| 9 | Horizon ensemble | Vote across short_5s/10s/30s on raw v3.4.2 (32 days) | best V3 = **-0.13** | 9/32 (28%) | REJECT — also surfaced 30s-head bug |
| 10 | Trade tape imbalance | Aggressor sweeps + absorption + size buckets on raw MBO (48 days) | -0.47 (top-5%) | 5/47 (11%) | REJECT as alpha, INTERESTING as liquidity insight |

---

## Three Structural Diagnoses

### Diagnosis 1: Sample-Size Ceiling

- Meta-classifier OOT covers 15 days. `meta_prob` is only computed on 7 of those 15 days (the rest default to size=1 in sizing experiments). The "gate-able" sample is effectively 7 days.
- Every day-level classifier (axes 5, 6) found "best" features that select **zero folds** under leave-one-out forward-walk. Headline AUCs of 0.76–0.85 were post-hoc selection artifacts.
- Bonferroni-adjusted noise floor across 23 features on 15 samples ≈ AUC 0.78. We never cleared it under honest validation.
- **Implication**: any day-gate claim from this dataset requires 30+ OOT days or pre-registered feature/direction/K. The HC #428 R1 minimum of 40 days is binding for a reason — we cannot deploy at 15.

### Diagnosis 2: Thin-Alpha Ceiling

- Pooled IC(pred10s, realized10s) = **+0.036** on 32 OOT days. This is statistically real (LOO std ≈ 0) but economically marginal.
- 5s and 10s heads are 90% correlated — same signal, no ensemble benefit.
- **30s head is broken** (axis 9 diagnostic): mean prediction -0.86 vs 10s +0.04; negatively correlated -0.51 with 10s, -0.585 with 5s. Different baseline/scale than near-horizon heads.
- The +5.03 t/trade survivor exists because at IC 0.036 with 0.376-tick passive-limit cost and ~10s holds, the top-5% confidence bucket still has positive expected value. But this is a sharp-edge regime: small calibration drift erases it.
- **Implication**: more training compute on the same architecture/labels will not move IC much above 0.036. Need either (a) better features (OFI ranked #1 in trade-classifier feature importance — suggests current models underuse it), (b) different target (regression vs binary winner), or (c) different architecture (Razer attribution earlier confirmed alpha is LINEAR — DLinear matches CNN-Mamba — so smaller models may be sufficient and faster to iterate).

### Diagnosis 3: Liquidity-Provision Edge (NEW, from axis 10)

- Trade-tape aggressor flow has **negative IC -0.036** against forward 10s mid-move. Buyers get faded; sellers get bid back.
- SWEEP flow (large aggressors) carries 4x the information of small/absorption trades.
- The sign of this IC is the SAME sign as our survivor's edge: passive-limit fills (liquidity provision) earn the reversal that aggressors pay for.
- **Implication**: the +5.03 t/trade survivor is partially a *liquidity-provision* edge, not purely a *directional-alpha* edge. This explains why label-vs-FIFO gap was +1.5 t IN OUR FAVOR (fills came better than mid-label assumed): we got rewarded for absorbing aggressor flow.
- **Actionable**: build an entry policy that EXPLICITLY rewards liquidity provision — peg limits above the bid (joining the queue) when sweeps fire AGAINST our signal direction, take market orders only when sweeps fire WITH our signal. This is execution-axis, not alpha-axis, and is genuinely untested.

---

## What was NOT tested today (queue for next session if Neptune fold-0 doesn't reset priorities)

- **Variable exit on stream sign-flip** (pressure-based exit per HC #486 R5, applied to filled trades). NOT the rejected adaptive-exit which optimized TP.
- **Regression-target meta-retrain** (binary→ticks). Requires Neptune compute.
- **Liquidity-provision entry policy** (peg-vs-take based on sweep alignment).
- **Information-bottleneck on prediction stream** (HC #488 evaluation axis).
- **PCMCI causal-discovery feature selection** on meta_layer_v1 (model axis).
- **30s head calibration audit + fix** (training-config-axis).
- **Self-supervised pretraining on unlabeled MBO** (model-axis, larger compute).

---

## Compute / state at synthesis time

- **Neptune**: v3.4.2 hc477fix_v2 retrain, fold-0 ~40-50% complete. ETA OOT verdict ~16:30 ET. Bias-fix from HC #482, labels regenerated under HC #485 NaN audit.
- **Razer**: DLinear quantile (P10/P50/P90 pinball) fold 1 of 3. ETA full 3 folds ~17:00 ET. Tests whether explicit quantile heads produce informative interval width.
- **Jupiter**: free after synthesis. Per HC #483 R3, will dispatch ONE creative axis from the queue above when synthesis is done — leading candidate: liquidity-provision entry policy (cheap to test on existing per_trade_fifo data).

---

## File index

Today's outputs under `/home/jupiter/Lvl3Quant/output/`:
- `meta_classifier_v1_fifo/` — the +5.03 survivor
- `adaptive_exit_v1/` — axis 4 reject
- `conformal_wrapper_v1/` — axis 1 reject
- `microprice_adverse_selection_v1/` — axis 2 reject
- `tod_velocity_fifo_replay_v1/` — axis 3 reject
- `day_classifier_v1/` + `day_classifier_forward_walk_v1/` — axis 5 reject
- `cross_asset_day_classifier_v1/` + `cross_asset_forward_walk_v1/` — axis 6 reject
- `trade_classifier_v1/` — axis 7 reject
- `kelly_sizing_v1/` — axis 8 reject
- `horizon_ensemble_v1/` — axis 9 reject
- `trade_tape_imbalance_v1/` — axis 10 reject + liquidity insight
- `2026-05-22_axis_diagnosis/REPORT.md` — this synthesis
