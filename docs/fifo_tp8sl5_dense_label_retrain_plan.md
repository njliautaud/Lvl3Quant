# Dense-Label Retrain Plan -- `fifo_tp8sl5_net` head (HC #396 follow-up)

**Status:** SCOPE ONLY -- do NOT launch without Neptune time approval. This
is a v3.3 training-config change that requires re-running fold_00 (at
minimum) to produce a comparable NPZ.

**Author:** HC #396 weekend lane (sizing follow-up agent), 2026-05-16.

---

## 1. Label sparsity diagnosis (fold 0, v3.3 60d champion NPZ)

Measured from `output/cnn_mamba_v3_3_uncertainty_weighted/fold_00_predictions.npz` (N=241,351 OOT rows across 5 days):

| head                | mask_frac | finite_frac | keep_frac | N_keep   | zero_among_kept | kept_abs_mean | kept_std |
|---------------------|-----------|-------------|-----------|----------|-----------------|---------------|----------|
| `fifo_tp8sl5_net`   | **0.2446**| 1.0000      | 0.2446    | **59,024**| 0.0000          | 5.35 ticks    | 6.48     |
| `fifo_tp4sl3_net`   | 0.2446    | 1.0000      | 0.2446    | 59,024   | 0.0000          | 3.25          | 3.42     |
| `log_ret_1s`        | 1.0000    | 1.0000      | 1.0000    | 241,351  | 0.3484          | 1.08          | 1.63     |
| `log_ret_10s`       | 1.0000    | 0.9958      | 0.9958    | 240,341  | 0.1214          | 3.41          | 4.84     |
| `p_up_10s`          | 1.0000    | 1.0000      | 0.5662 (zero collapse)| 241,351 | 0.566 | 0.43 | 0.50 |

**Bottleneck:** `fifo_tp8sl5_net`'s mask drops 75.5% of rows. This is ~4x
sparser than the log-ret heads. Every kept row has a non-zero, finite
target (no internal zero collapse) -- the entire information loss is at
the mask gate.

**Why the mask kills so many rows:** The FIFO bracket label logic
(`data/processed/mbo_events_smart_v3_fifo_labels`) requires that BOTH the
take-profit AND stop-loss legs of the bracket resolve within the hold
window. Rows near end-of-session, near end-of-day boundary, or where the
bracket would extend past the available future-window are masked out.
For `tp8sl5` (TP=8 ticks, SL=5 ticks) the bracket is wider than `tp4sl3`,
yet mask_frac is identical -- suggesting the dominant mask cause is the
HOLD-WINDOW boundary (need enough future ticks regardless of bracket
width), not the bracket width itself.

---

## 2. What "denser labels" means -- three concrete options

### Option A: Shorter hold window (recommended -- cheap and well-defined)
Reduce the max-hold-time used during label generation. If current is 30s,
try 10s. Cost: TP/SL hits become rarer near the wide bracket, but the
non-resolved exit-at-horizon outcome can still be recorded as net P&L
(not NaN). Expected mask_frac ~0.5-0.7 (2-3x denser).

**Concrete change:** `data/processed/mbo_events_smart_v3_fifo_labels/`
generation script -- find the `max_hold_ns` or equivalent param. Set to
10s instead of 30s. Re-run on the 60d window. Re-train fold 0 only.

### Option B: Non-NaN imputation -- exit-at-horizon fallback
For rows that don't hit TP or SL within hold window, currently set NaN/mask=False.
Instead, set the label to "net P&L at exit-horizon" (1s after entry or end-of-bracket).
This is the BANDIT exit convention used in operational replay anyway, so
labels would align better with downstream consumer.

**Concrete change:** In FIFO label generator, when neither TP nor SL hits,
emit `target = side_sign * (price_at_horizon - entry_price) - commission_ticks`
and `mask = True`. Risk: this adds many small-magnitude labels (the bracket
didn't trigger because nothing happened) -- could DILUTE signal.

### Option C: Relaxed TP/SL thresholds (least preferred)
Use TP=4, SL=3 (current `tp4sl3` head already does this). Already has same
mask_frac as `tp8sl5`, so this doesn't densify; not useful in isolation.
Could be combined with Option A.

**Recommendation:** Start with **Option A** alone (shorter hold window).
Compare val IC and replay metrics to the current model. Only proceed to
Option B if Option A doesn't materially lift IC.

---

## 3. Cost estimate (Neptune RTX 3090)

Reference: v3.3 60d champion training (`output/cnn_mamba_v3_3_uncertainty_weighted/`).
60-day sliding-window walk-forward, single fold ~3-4 hours wall clock at
batch=64, 4 epochs per fold on 3090.

For label regeneration: ~10-20 min on Jupiter CPU (FIFO label generator
is single-threaded numpy, but parallelizable per day).

For retrain (single fold for comparison):
- 1 fold * 4 epochs ~3-4h Neptune GPU
- 5 folds (full 60d sliding) ~16-20h Neptune GPU

For ablation (Option A only, fold 0 only): **~4 hours Neptune GPU + 20 min Jupiter CPU**.

---

## 4. Risk: does denser labels = noisier signal?

**Yes, potentially.** Option A's shorter hold window means more "incomplete"
bracket outcomes (label = whatever P&L happened to be at exit-horizon).
These outcomes are noisier because they reflect microstructure noise rather
than a clean take-profit/stop-loss resolution.

### Validation gates before promoting

1. **Concat val IC on `target_fifo_tp8sl5_net`** (rebuilt with new labels):
   must be ≥ 0.048 (current raw head val IC on old labels) and ideally ≥ 0.10
   (matching log_ret peers).
2. **Per-day IC stability:** new labels must NOT have IC stdev > 2x old labels.
   Noisy labels often manifest as higher per-fold variance.
3. **Replay-bench:** run new head through `v33_production_readiness_full_sweep`
   at same config (P95, long, passive_at_touch_plus_1, cw=40, hold=1s).
   Sharpe must be ≥ current 2646 (val slab) within 20% variance to be
   considered comparable.
4. **HC #344 check:** day_conc must NOT regress -- if it goes from 1.0 to
   >1.0 (impossible -- already at ceiling on val), check fold-1+ and full
   60d retrain instead.

---

## 5. Decision points before launch

- [ ] User approves Neptune ~4h GPU time for ablation (Option A, fold 0).
- [ ] FIFO label generator path identified in
      `data/processed/mbo_events_smart_v3_fifo_labels/` source. (Not searched
      this round -- needs a 5-min code dive.)
- [ ] No conflict with active Neptune training PIDs 416391 (v3.4.2 60d) and
      418732 (v3.3 chunk1 inference).

---

## 6. Why this matters

The previous HC #396 agent showed the meta-MLP stacker's apparent "4.8x IC
boost" was a single-head artifact driven by THIS head's anomalously weak
raw IC (0.048 -- vs 0.08-0.11 for peers). The root cause is label sparsity.
Fixing it at the source (denser labels) is strictly preferable to bolting
on a stacker downstream, because:

- A retrained head's improvement persists in operational replay (the
  stacker did not generalize -- 5/5 directional heads showed stacker IC
  WORSE than raw).
- A retrained head doesn't add inference cost (no second model to run
  online).
- A retrained head benefits other downstream consumers (the sizing
  calibrator from this task also uses `pred_fifo_tp8sl5_net` as an input
  feature -- if that input gets stronger, the calibrator gets stronger).

End plan.
