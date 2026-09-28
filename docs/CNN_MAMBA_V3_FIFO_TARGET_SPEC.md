# CNN-Mamba v3 — FIFO-Realized Target Trainer Spec

**Status**: DRAFT (2026-05-09 05:50 ET)
**Owner**: head-of-quant (autonomous)
**Authority**: HC #271(D) — pursue the alpha→execution gap (HC #261) at the SOURCE
**Target node**: Razer (RTX 3070, 8GB) — pivot away from SAC v2 which is the abandoned RL track

---

## Why v3?

CNN-Mamba v2 trains to predict mid-price moves at horizons {1s, 5s, 10s}. Predictions
have been validated as having genuine information (concat IC_1s = 0.222 on the 39-fold
walk-forward). But under HC #265 FIFO market replay, every rules-based execution
config tested loses money (HC #267 winner is +0.98 NET ticks/trade only by extreme
selectivity — top 0.1% short-only — and even that fails HC #271(A) regime concentration).

**The execution gap (HC #261)** says: midpoint α exists, but FIFO realized α does not
match it. The root cause is adverse selection — the market crosses our limit at exactly
the moments when mean reversion is about to pick us off. The signal *predicts* the
right direction, but the *executable* P&L is dominated by queue/spread/timing dynamics
that the v2 head never saw at training time.

**v3 thesis**: train the model with FIFO-realized P&L as an auxiliary target, jointly
with the existing midpoint horizons. This forces the encoder to allocate capacity to
features that are predictive of *executable* outcomes, not just of midpoint drift.

---

## Architecture

Same trunk as v2 (CNN feature extractor → Mamba state-space sequence model → fusion
projection). Only the **head** changes:

```
v2 head: 3 outputs       → pred_1s, pred_5s, pred_10s   (regression on Δmid in ticks)
v3 head: 4 outputs       → pred_1s, pred_5s, pred_10s, pred_fifo_net_ticks
                                                         ↑
                                                   NEW (regression on
                                                   FIFO-realized NET ticks
                                                   from immediate top-of-book
                                                   limit at signal_ts + commission)
```

The new output is **target-aligned with deployment** — exactly the quantity the user
cares about, computed by the same FIFO market replay that validates HC #265 results.

---

## Target construction (offline preprocessing)

For every signal in the training set:
1. Run the same FIFO market replay used by `validate_via_fifo_replay.py` to determine,
   *at signal_ts*, what would have happened if we placed a passive limit at the top of
   book on the predicted side, with cancel-window=2s, hold=30s, TP=8t / SL=5t (the
   HC #267 winning config).
2. Record `realized_net_ticks` (gross minus 0.376t commission, 0 if not filled).
3. Two strategies for non-fills:
   - **Strategy A (mask)**: mask the FIFO loss for non-filled signals (the head
     doesn't get gradient on those). Cleaner but throws away ~30-50% of signals.
   - **Strategy B (zero-target)**: treat non-fills as `realized_net_ticks = 0` (no
     trade = no P&L). Keeps all signals but introduces a strong "do nothing" prior.
   - **Decision**: start with **Strategy A** (mask). Evaluate B as ablation.

The training-set FIFO replay already exists for the 36 OOT dates as a byproduct of the
meta-LGBM Phase 1 work (`extract_fifo_labels_for_lgbm.py`). It needs to be extended to
the **in-sample period** (the 248 dates covering July 2025 – Feb 2026) — that's the
most expensive part of v3 prep.

**Cost estimate**: FIFO labeler runs ~3-4 min per date. 248 dates × 4 min = ~16 hours
on Jupiter (single-process). Parallelizable to ~4 hours with 4 workers. Add 248 × 50 MB
≈ 12 GB of FIFO label parquets. Manageable.

---

## Loss

```
loss = α₁ · MSE(pred_1s, Δmid_1s)
     + α₅ · MSE(pred_5s, Δmid_5s)
     + α₁₀· MSE(pred_10s, Δmid_10s)
     + β   · masked_MSE(pred_fifo_net_ticks, realized_net_ticks)
```

**Initial weights**: α₁=α₅=α₁₀=1.0, β=2.0. Rationale: FIFO-realized is the primary
deployment target; midpoint horizons are auxiliary tasks that regularize the encoder.

**Sweep plan**: β ∈ {0.5, 1, 2, 5} as a 4-cell ablation after the v3 baseline lands.

---

## Walk-forward

Same protocol as v2 (HC #0 sliding 30 train days, 1 OOT day, slide 1d). Re-train from
scratch — do NOT warm-start from v2 weights, because the new head has different
output dimensionality and the encoder may need to re-allocate capacity given β=2.

ETA per fold on RTX 3070: v2 trained at ~25 min/fold. v3 should be similar (slightly
slower due to the extra head and FIFO-target masking gather). Full WF (39 folds OOT)
= ~16 hours. Match the existing v2 OOT date set so direct comparison is possible.

---

## Comparison protocol (v3 vs v2)

After v3 training completes, run the FIFO market replay validator on v3 predictions
exactly the same way as v2. Required dominance criteria (HC #271(C) extended):

1. **Headline edge (top 0.1% short, tp8sl5, cancel=2s, hold=30s)**: v3 sum-NET-ticks
   must be > v2's +50.95t **AND** v3 must clear HC #271(A) (≥1 regime profitable AND
   ≤50% concentration AND ≥60% folds-positive).
2. **Concat IC**: v3 IC_1s > v2 IC_1s = 0.222 (or at minimum, v3 IC_1s ≥ 0.21 — i.e.
   the FIFO target shouldn't degrade the midpoint signal).
3. **Concat IC of pred_fifo_net_ticks vs realized**: must be > 0.05 with t-stat > 4
   on OOT (otherwise the new head is noise).
4. **Per-regime check**: v3 must dominate v2 in **at least 2 of 3** trend regimes.

If v3 fails 1+2 simultaneously, archive and revert. If it fails only 1, sweep β. If
it fails 2, the architecture isn't allocating capacity to the FIFO head — increase β
or add separate FIFO-feature inputs (queue position, depth imbalance) directly into
the head's pre-projection.

---

## Razer launch sequence (when ready)

```bash
# 1. Stop SAC v2
ssh claude@razer "schtasks /End /TN train_fifo_rl_sac_v2"
# (or kill PID 22472 directly)

# 2. Sync v3 trainer + FIFO labels to Razer
rsync -avz scripts/train_cnn_mamba_v3_fifo.py \
    claude@razer:C:/Users/claude/Lvl3Quant/scripts/

rsync -avz output/fifo_labels_full/ \
    claude@razer:C:/Users/claude/Lvl3Quant/output/fifo_labels_full/

# 3. Launch under launch_with_watchdog.sh (HC mandatory wrapper)
ssh claude@razer "bash ~/Lvl3Quant/scripts/launch_with_watchdog.sh \
    train_cnn_mamba_v3_fifo \
    'python -u scripts/train_cnn_mamba_v3_fifo.py \
        --train-days 30 --slide-days 1 \
        --beta-fifo 2.0 --mask-nonfills \
        --batch-size 256 --num-workers 8 \
        --mlflow-experiment cnn_mamba_v3_fifo \
        --mlflow-uri http://neptune-win:5000'"
```

---

## Pre-requisites (must complete before Razer launch)

- [ ] Extend FIFO labeler from 46 OOT dates to full 248 in-sample + OOT date set.
      Estimated cost: 4h on Jupiter with 4 parallel workers.
- [ ] Build `train_cnn_mamba_v3_fifo.py` (clone of v2 trainer with new head + loss).
      Estimated cost: 2h coding + smoke test.
- [ ] MLflow Tailscale URI must work from Razer (verify with curl).
- [ ] Disk space check on Razer (12 GB FIFO labels + ~4 GB v3 weights + buffers).

---

## Decision gate

**Promote to live trading only if**: v3 wins HC #271(C) dominance against v2 + parents
on the meta-LGBM gate's BEST operating point AND v3 clears HC #271(A) gate AND v3
predictions, when fed through the meta-LGBM gate, **lift Sortino by ≥ 30%** vs v2-fed
meta-LGBM gate at the same trade count. This is the "stacking actually pays" floor.

If v3 wins all three: it replaces v2 as the live signal model on Razer's paper trader.
If v3 only wins HC #271(C) but not the meta-LGBM lift: keep v2 in live, treat v3 as a
research artifact for the next iteration.
