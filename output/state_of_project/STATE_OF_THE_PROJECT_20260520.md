# State of the Project — 2026-05-20 02:55 ET

**Author**: autonomous session (post v3.3 OOT inference completion, post HC #444 cross-model probe).
**Friday deadline**: 2026-05-22 EOD. **Status**: HC #444 R2 grid exhausted, no canonical-profitable config found. Pivoting to HC #444 R4 fallback.

---

## TL;DR

After 3.5 months and an exhaustive search across signal models × confluence × gates × filters, **no configuration trades positively under canonical FIFO market-replay**. The label-level edge of the CNN-Mamba v2 signal is real (IC_1s = 0.222), but it does not survive realistic passive-execution costs on top of the model's actual horizon.

Friday delivers:
1. **Live-data-collection harness** — production MBO + v2/v3.4.2/v3.3 inference + paper fills under HC #441 PRIMARY geometry. No live cash deployment.
2. **Closest-to-profit candidates report** (this document) — best baselines and analysis of why each fails.
3. **Post-Friday research roadmap** — three signal/execution R&D directions to recover edge.

---

## What was tested under HC #444 R2 (now closed)

| Dimension | Tested | Result |
|---|---|---|
| Models | CNN-Mamba v2, v3.4.2-fixedmtl, v3.3-uncertainty-weighted, PatchTST | All single-model configs net-negative under canonical FIFO |
| Confluence | same-model multi-h (1s/5s/10s), cross-model 3-way (per-day) | Multi-h: PF 0.58-0.60, WR 27-28%. Cross-model per-day: zero configs pass |
| Gates | conf-band (top 0.5%/1%/5%/10%/20%), time-of-day buckets, queue position, pred-strength quintile | LGBM meta-classifier AUC ceiling = 0.55 (pre-trade features only); GPU MLP confirmed 0.53 AUC after leakage fix |
| Filters | hold-time, fill-type, queue-ahead, day-of-week, pred-strength, side | Single-direction filter best = `hold_time_bucket=1.5-5s` (selection-bias finding, not actionable for entry) |
| TP/SL/hold geometry | sweep TP ∈ {0.96, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 8.0}, SL ∈ {0.5, 1.0, 2.0, 3.0}, hold ∈ {0.5, 1.0, 1.5, 5.0, 10.0}s | Best single-model canonical config: net = −0.31 tk/fill (still losing) |
| Horizon match | h=5s + hold=5s + cancel=5s top0.5% short (HC #428 R2 compliant) | n=2625, PF=0.59, WR=19%, net=−0.29 tk/fill |
| Cross-model agreement | per-day v2 ∩ v3.4.2 ∩ v3.3 sign-aligned filter | ZERO of 25 configs achieve mean_tk_net > 0 + day_pct ≥ 60% |

**Per-signal cross-model agreement** (the only finer-grained version of the last cell) cannot rescue anything — it is a strictly stricter filter than per-day, so configs failing per-day fail per-signal by construction.

---

## Closest-to-profit candidates (audit trail for post-Friday R&D)

These are the LEAST-LOSING configs on the full OOT under each canonical engine. None are tradable. They are listed as starting points for post-Friday research.

| Rank | Config | Side | n | Net tk/fill (after cost) | Raw mean tk | PF | WR | day_pct | Notes |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `v342_lshort_5s_ensemble_50_50` | long | 3982 | −0.490 | −0.114 | 0.57 | 48.7% | 6% | 50/50 ensemble of v3.4.2 long-and-short heads on 5s horizon. Closest to flat. |
| 2 | `v342_long_1s_top0.5_tp1.0_sl0.5_h1.5_c1.0_passive_at_touch` | long | 1728 | −0.529 | −0.153 | 0.66 | 48.3% | 0% | v3.4.2 long top-0.5% with HC #428 R2-compliant geometry |
| 3 | `v342_long_5s_top0.5_for_ensemble` | long | 2645 | −0.532 | −0.156 | 0.66 | 48.1% | 6% | v3.4.2 long 5s horizon |
| 4 | `v2_short_1s_top0.5_baseline` | short | 2676 | −0.549 | −0.173 | 0.62 | 47.0% | 9% | v2 short top-0.5% — the canonical baseline |
| 5 | `hc443_h5_hold5_top0p5_short` | short | 2625 | −0.667 | −0.291 | 0.59 | 19.2% | 0% | HC #428 R2-compliant short variant |
| 6 | `hc443_band_top1_short` | short | 4189 | −0.667 | −0.291 | 0.55 | 25.1% | 6% | top 1% (wider band, more fills) |
| 7 | `hc443_chase_sl05_tp3_h15` | short | 2431 | −0.702 | −0.326 | 0.51 | 22.8% | 0% | HC #441 PRIMARY (was the "champion" until analytic-resolver bug, HC #442) |

Cost constant: ES_RT_COMMISSION_TICKS = 0.376 (AMP/Rithmic, no spread component because passive fills hit at touch).

**Why these all fail under canonical FIFO**:
1. Top-of-book passive fills at peak signal moments face severe adverse-selection (queue-ahead during the few ms the signal is most actionable means we get the bad fills, not the good ones).
2. Tight SL (0.5 ticks) kills positions that would otherwise mean-revert — the filter-strat analysis showed timeout-fills are net positive while SL-hit fills dominate the negative tail.
3. The model's actual MFE@1s p90 = 30 ticks, but realized mean MFE is much smaller; expanding TP catches more noise without proportional gain.
4. Long side is closest-to-flat but has its own adverse-selection issue: top-0.5% long signals fire near top-of-bar where bid-side liquidity is fast-fading.

---

## Why HC #441 looked profitable (and wasn't) — for historical record

HC #441 declared SL=0.50 TP=3.00 hold=1.5s as "champion" on 2026-05-19 21:30 ET with claimed net=+0.71 tk/fill, PF=2.54, 12/12 positive OOS days. **This was a software bug** in the analytic resolver, not real edge.

The analytic resolver built its trajectory cache with `cancel_s=10s` (violates HC #428 R2). When the canonical realtime_sl FIFO engine re-ran the same config with HC #428 R2-compliant `cancel_s=1s`, the same identified trades produced net=−0.21 tk/fill — opposite sign. The +0.71 result was an artifact of stale predictions surviving until trade exit, not the geometry working. HC #442 documents this in full.

This is documented to prevent any future session from re-attempting the same false-positive.

---

## Post-Friday Research Roadmap (three paths)

### Path A — Signal R&D (new model or new feature set)

**Hypothesis**: the current signal universe (CNN-Mamba family on `mbo_events_smart_v3` event-stream features) has a ceiling that adverse-selection wipes out under any tested geometry. New input modalities or architectures may break through.

**Concrete next steps**:
1. **Event-level order-flow imbalance features** — net buyer-vs-seller pressure at the sub-second scale, conditional on book imbalance. Not currently in the feature vector.
2. **Cross-asset features** — NQ/RTY lead-lag relative to ES at 100ms scale. We have the data; we have not extracted it.
3. **PatchTST + CNN-Mamba ensemble at the feature level (not output level)** — concatenate latent embeddings instead of averaging predictions.
4. **Reinforcement-learning execution model** — was previously discussed for Neptune but never launched because no profitable signal candidate existed. Path A produces the candidate.

### Path B — Microstructure-aware execution

**Hypothesis**: the signal IS profitable but only if executed without adverse selection. Currently we assume FIFO queue position; in reality the queue is gameable.

**Concrete next steps**:
1. **Queue-position simulation** — we have queue_ahead in fills CSVs but it's the queue depth at order placement, not at fill time. Track queue evolution from placement → fill.
2. **Hidden-liquidity inference** — estimate iceberg fills using trade-print clustering. If the top of book has hidden liquidity, our passive fills are systematically adversely-selected.
3. **Joiner-leader detection** — identify when our order joins a cascade vs initiates one. Joiners almost always lose.
4. **Order-type alternative — iceberg/hidden ourselves** — outside HC #74 canonical assumptions, requires regulatory check.

### Path C — Longer-horizon regime shift

**Hypothesis**: the 1s-horizon edge is too fast for any retail-grade execution path. The model's longer horizons (30s, 60s) have weaker IC (0.05-0.07) but may survive realistic execution because the holding period absorbs adverse-selection noise.

**Concrete next steps**:
1. **30s-horizon canonical sweep** — TP={4, 6, 8, 10}, SL={2, 3}, hold={30, 45, 60}s. Not tested under canonical FIFO yet.
2. **Multi-day position-style** — use the model's 5min horizon prediction as a directional bias for end-of-day position. Departures from intraday scalp regime.
3. **Vol-regime conditional sizing** — large position only when LGBM-Vol predicts high realized vol AND model agrees on direction. Both conditions filter most days, but those that survive may be the asymmetric-edge days.

---

## Cluster state at handoff (02:55 ET)

| Node | Status | Currently | Holding for |
|---|---|---|---|
| Neptune RTX 3090 | idle | v3.3 inference done | Path A model training (when chosen) |
| Razer RTX 3070 | live | MBO record + v2 inference + paper trader | Harness extension (add v3.4.2/v3.3 inference channels) |
| Jupiter CPU | idle | HC #444 cross-model analysis done | Closest-to-profit doc (this file) + harness extension code |
| Saturn | offline | — | — |

Artifacts in `output/hc444_cross_model_perday/` are PERMANENT per HC #443 R2. Do not delete.

---

## Honesty rule applied (HC #442 R3)

Every result in this document was produced by the canonical `realtime_sl` FIFO market-replay engine on full OOT (32+ days). No analytic-resolver numbers. No 3-day hand-picked windows. No P&L claims that aren't backed by full-OOT canonical replay.

The label-level edge (IC = 0.222 at 1s) is real and stated; it is NOT a tradeable P&L claim.

---

## Next action

Building the live-data-collection harness extension on Razer: extend the existing v2-only inference loop to also score each event with v3.4.2 and v3.3 in real time and log all three predictions + paper fills under HC #441 PRIMARY geometry. This becomes the post-Friday research corpus.

---

## HC #451 Data-Representation Overhaul (Added 2026-05-20 08:30 ET)

**User's reframing** (2026-05-20 08:11 ET): the model is not learning concepts like continuation, sweeps, support/resistance, and key-level interaction because we never gave it the data to learn them from. The current input — 1000 raw MBO events — is too narrow a window to distinguish "this is the breakout" from "this is a pullback inside the bigger breakout". The Mamba SSM can remember long-range events, but only if those events are SALIENT in the input and the loss rewards remembering them.

### The five changes to data representation

1. **Multi-resolution context channels** — alongside raw MBO events, synchronously inject 1-second, 10-second, 60-second, and 5-minute bar features (signed volume, imbalance, mid slope, realized vol) plus distance-to-key-levels (day VPOC, overnight high/low, prior close, recent 30-min H/L).

2. **Event-salience tags** — per-event boolean/scalar channels that flag what a human notices: sweep, suspected iceberg, large print, extreme imbalance, level touch, VPOC touch, momentum-persistence run. Not hard-coded alpha — salient hooks the network can selectively attend to.

3. **Key-level state (always on)** — at every event, scalar features for ticks-to-nearest-S/R, time-since-last-touch, volume traded at that level (so "thin" S/R differs from "thick"), session VWAP deviation, day-VPOC location.

4. **Cumulative pressure features** — running net signed volume / aggressive-buy-minus-sell / trade-count-imbalance over rolling 1s / 10s / 60s / 5min / 30min windows. So the model knows when it's been one-sided for 30 minutes, not just for 250ms.

5. **Pressure-style labels** — replace fixed-h single-horizon label with: multi-horizon vector (1/5/10/30/60s), direction-persistence binary, MFE-without-drawdown regression, smoothed-direction (causal EMA of forward returns). Per HC #450 R5, the smoothed label is the natural target for a pressure model.

### Why this is the right path

- HC #450 R4 confirmed empirically: current signal autocorrelation lag-1 ≈ 0.02, sign-flip ~2/sec. The model is predicting noise/snapshot, not pressure.
- HC #428 R2 mandates TP/hold/cancel bounded by the model's predictive horizon. If the predictive horizon is microscopic (which it is), no execution geometry can rescue P&L.
- The fix isn't a wider gate or a tighter SL — it's a model that predicts a smoother, longer-horizon, pressure-like quantity. That requires teaching the model from inputs and labels that ENCODE pressure.
- Eight months of training the same SSM on the same narrow window has stalled at IC_10s ≈ 0.10. The constraint is the data representation, not the architecture.

### Implementation order (Friday-safe)

| Step | Owner | Timing | Output |
|---|---|---|---|
| Friday harness + closest-to-profit report (HC #448 R2) | unchanged | 2026-05-22 EOD | live recording stack, candidate report |
| Build feature pipeline (R1–R4 channels) on top of existing tensor cache | Jupiter CPU | starts today, parallel to Friday work | augmented tensor cache, no model change yet |
| New label cache (R5 multi-horizon + persistence + MFE-MAE) | Jupiter CPU | starts today | augmented label cache |
| First retrain of CNN-Mamba v3.4.2-style net on augmented data | Neptune GPU | starts after Friday ships | new signal model |
| Diagnostic gate: HC #450 R4 smoothness diagnostic on new model | Jupiter CPU | within 24h of retrain finish | accept/reject decision |
| If lag-1..lag-40 autocorr ≥ 0.30: proceed to canonical FIFO replay | Jupiter CPU | post-diagnostic | P&L numbers |
| If autocorr still ≈ 0.02: pivot to longer-horizon labels or new arch | research call | escalate to user | new plan |

### Cross-references to HC #450

- HC #450 R5 (smoothed inference variant) is operationally subsumed by HC #451 R5 (smoothed label). Smoothing at training-time is strictly stronger than smoothing at inference-time.
- HC #450 R6 alpha-staleness candidates (PatchTST-on-trades, longer-horizon labels, multi-task signal, OFI pure-feature model) remain valid; HC #451 R5 multi-horizon + persistence label is the first one to execute.
