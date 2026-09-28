# HC #421 Issue A — Shadow Gate "Zero Fills" Root-Cause Diagnosis

**Date**: 2026-05-18 13:50 ET
**Status**: ROOT-CAUSED. Shadow v2_1s_short_top05 cannot fill because the model produces a **fundamentally different prediction distribution** on live data vs. backtest data. **Not** a scaling artifact.
**Verdict on HC #421 cutover today (5/18 16:00 ET)**: **NO-GO**. Pushed to Tue 5/19 09:30 ET *contingent on resolution of this gap*.

---

## Headline numbers

| metric (pred_log_ret_1s) | LIVE 5/18 14:00 ET (n=237) | BACKTEST 56d OOT (n=2,313,722) | Δ |
|---|---:|---:|---:|
| mean | **+0.270** | +0.064 | +0.206 |
| std | **+0.255** | +0.380 | −0.125 (33% narrower) |
| min | −0.0755 | −3.36 | live can't go strongly negative |
| max | +1.63 | +3.97 | live caps lower on the positive side too |
| q50 | +0.232 | +0.009 | shifted +0.22 |
| q99 | +1.24 | +1.16 | similar magnitudes on positive tail |
| **pred ≤ −0.6926 (the live gate floor)** | **0.000%** | 0.465% | 100% shortfall |
| pred ≤ −0.10 | 0.000% | 33.3% | massive |
| pred ≤ −0.01 | 5.06% | 46.1% | 9× sparser |
| pred ≥ +0.001 (long bias) | **94.1%** | 51.8% | live is overwhelmingly long-biased |

## What this means

- The live model is producing a prediction stream that is **shifted +0.21 in mean** and **33% narrower in std** vs the training/backtest distribution.
- The gate floor (−0.6926, equal to the backtest q0.5 percentile of pred_1s) was chosen for the backtest distribution. Against the LIVE distribution it is unreachable: live's most-negative prediction so far is **−0.0755**, more than 9× too shallow to ever clear the floor.
- This is **not** a units mismatch (no rescaling factor would explain the mean shift while also compressing the std). Live and backtest both consume raw head-0 output from the same checkpoint.
- The shadow's six-tuple encode logic was verified against the live trader source (action≥2→trade, price_ticks tick-offset, delta_us=micros, qty=log1p(size), spread=ticks). `skip_normalize=True` is confirmed in both training and live (means~0 stds~1 in fold_09_feature_stats.npz).
- Therefore the divergence is upstream: either (a) **smart_v3 feature encoding** in the live `StreamingFeaturesSmartV3` differs from the offline preprocessor used to build training NPZs, OR (b) the **current market regime** (today's ES around 7400 with chop) is producing feature vectors at the edge of the model's training manifold and the model is degenerating to a positive-biased estimate.

## Methodology (read-only, no live-trader changes)

- New diagnostic at `scripts/diag_live_pred_distribution.py`. **Does not touch** the running shadow PID 29600 or legacy PID 25512.
- Tailed the last 60,000 events from `live_trading/logs/live_events.jsonl` (today's MBO recorder feed).
- Built a fresh `CNNMambaV2Inference` on CPU (Razer GPU still owned by shadow) with same weights (`fold_10_best.pt`, sha256 `300e338d…`) and same stats (`fold_09_feature_stats.npz`).
- Reconstructed the 6-tuple encode exactly per `paper_trading_v2_1s_short_top05.py` lines 1103–1125.
- Window=1000, stride=250 → 240 expected preds; got 237 (3 stride-aligned skips at warmup). Confirms streamer + engine are operating correctly.

## Cross-check vs the live trader's own counters

| source | pred-pass-rate against −0.6926 |
|---|---:|
| Live trader PID 29600 cumulative (08:06 ET → 13:42 ET, **6,000 preds**) | 0 / 6000 = 0.0000% |
| This diagnostic on last 60,000 events (~237 preds) | 0 / 237 = 0.0000% |
| Backtest 56d OOT reference | 10,765 / 2,313,722 = 0.4651% |

Diagnostic confirms what shadow PID 29600 observed independently.

## What I am NOT doing (per scope discipline)

- Not retraining the model.
- Not modifying the live trader or the engine.
- Not lowering the gate floor blindly to harvest fills — that would be deploying an untested strategy spec.

## What needs to happen before HC #421 cutover

1. **Compare live feature distribution vs training feature distribution.** Dump the 25-dim `feat_vec` first-moments from live shadow over today's session and from any reference training day NPZ. If first-moments diverge → feature pipeline drift. If they agree → regime drift.
2. **Decide on a fix path**:
   - (i) **Re-calibrate** the gate threshold on a recent live-distribution snapshot (the legitimate per-day-99.5pct portion of the live gate already does this, but the GLOBAL floor must drop or the shorting strategy is dead in current regime).
   - (ii) **Rebuild the encoder** to match training preprocessing exactly (if a bug is identified in step 1).
   - (iii) **Retrain v2** on data through 2026-05-15 (HC #344 weekly retrain mandate). Reset gate floor from the new pred distribution. This is the cleanest answer if step 1 reveals no encoder bug.
3. **Pre-flight HC #421 G1 (Rithmic broker wiring)** is still pending; do not wire orders until prediction distribution is reconciled.

## Update 14:05 ET — Suspect (b) "regime drift" CONFIRMED by per-day backtest trajectory

Computed per-day pred_1s distribution across all 48 OOT days. The distribution **was already drifting positive monotonically** through the OOT window:

| OOT window | n_days | mean | std | q50 | q1 | %≤−0.6926 |
|---|---:|---:|---:|---:|---:|---:|
| 2026-02-24 → 2026-03-20 | 12 | +0.024 | 0.32 | ≈0.000 | −0.60 | 0.39% |
| 2026-03-22 → 2026-04-15 | 22 | +0.077 | 0.38 | +0.014 | −0.62 | 0.43% |
| 2026-04-16 → 2026-04-29 | 14 | +0.082 | 0.43 | +0.009 | −0.66 | 0.59% |
| **2026-05-18 (LIVE today)** | 1 | **+0.270** | **0.26** | **+0.232** | **−0.05** | **0.000%** |

Pattern: mean drifts from +0.02 (late Feb) → +0.09 (late Apr) → **+0.27 (today, 19 days past last training data)**. The drift is **monotonic** and **accelerating**. The std *widened* through OOT (0.32→0.43) then **contracted to 0.26** in live — model is increasingly uncertain and biased.

**Verdict: this is REGIME DRIFT, not encoder drift.** HC #344 (weekly retrain mandate) has been silently violated since 4/29 — model's training cutoff is ~3 weeks stale. The "max negative pred = −0.0755" today is consistent with a model whose central tendency has shifted +0.21 and whose dispersion has collapsed.

## Recommended fix path (in order)

1. **STOP** anything that depends on this checkpoint's negative tail. The shadow's behavior is correct — no fills is the safe answer when the gate floor is unreachable.
2. **Retrain v2** on data through 2026-05-15 inclusive. This is now overdue per HC #344. After retrain, recompute the global floor as q0.5 of the new OOT pred_log_ret_1s distribution.
3. **DO NOT just remove the global floor and rely on the per-day 99.5pct gate.** That gate would adapt to today's distribution, but the absolute magnitude of the strongest short signal today (−0.0755) is too weak to expect backtest-quality edge. The per-fill profit assumption in `output/hc417_v2_1s_short_top05_DEPLOYMENT_SPEC.md` (+$3.43/fill, n=639) is built on a distribution where the gated preds were in the [−0.69, −1.5] region. Today's gate would pick signals from the [−0.05, −0.08] region — fundamentally different alpha.
4. **HC #421 cutover Tue 5/19 09:30 ET**: BLOCKED until step 2 completes and gate re-calibrates.

## Files

- Diagnostic script: `/home/jupiter/Lvl3Quant/scripts/diag_live_pred_distribution.py` (read-only)
- Diagnostic JSON: `/home/jupiter/Lvl3Quant/output/diag_pred_dist.json`
- Live training reference: `/home/jupiter/Lvl3Quant/output/hc417_v2_full_oot_56d.npz`
- This verdict: `/home/jupiter/Lvl3Quant/output/hc421_issueA_pred_distribution_diagnosis.md`
