# CNN-Mamba v3.1 + v3.2 Architecture Extensions

**Status**: SPEC WRITTEN (per HC #282(D) honesty gate state i). No code, no training yet.
**Created**: 2026-05-11 15:10 ET per HC #289 + HC #290(F)
**Parent spec**: `CNN_MAMBA_V3_SPEC.md` (baseline v3, 7-head suite, in-flight)
**Retrain window**: Next Saturday Razer slot per HC #282(A) — but Razer is offline per HC #285(C). Until Razer back online, queue stays here.

---

## Why these extensions

**The DA gap problem.** v2 fold 0 OOT analysis (today, 2026-05-11 11:54 ET) revealed:
- v2 DA edge concentrated at **Top 0.1% only** (DA=68.8% at n=24 trades/day)
- v2 DA at Top 0.5%/1% bands collapses to ~50% (no skill)
- PatchTST DA at Top 1% = **80.6%** on n=2073 trades/fold (analysis JSON, smart_v2)
- **v3 sees the same MBO event sequence as v2 — it has no access to PatchTST's DA edge as inputs**

**The decay-window problem.** Drift analysis showed v2 Top 0.1% drift trajectory:
- +1s: +0.34t
- +5s: +0.72t (peak)
- +10s: -1.24t (already inverted!)
- +30s: -2.12t

The CNN-Mamba v3 7-head suite tops out at 10s horizon. We may be CUTTING OFF longer-horizon signal that exists but isn't being modeled.

**The "raw book state" problem.** v3 sees MBO events (add/cancel/trade tape) but NOT the resulting BOOK STATE per event. It has no representation of "level 7 has a 200-lot resting limit" or "the L3 imbalance just flipped from +30 to -50". These structural book features are computed downstream into the smart_v3 feature stack, but the raw history of book levels is lost.

---

## v3.1 — PatchTST input fusion (HC #289)

### Design

Add 3 new INPUT features to v3 architecture, concatenated alongside existing 18-dim feature stack:

```python
NEW_PT_FEATURES = ["pt_pred_1s", "pt_pred_5s", "pt_pred_10s"]
INPUT_DIM_V3   = 18  # baseline smart_v3
INPUT_DIM_V31  = 21  # 18 + 3 PT preds
```

Source: `output/meta_lgbm_features/<DATE>_signals_enriched.parquet` (already exists, used by meta-LGBM gates).

**Normalization** (HC #281(E) + HC #290 compatible):
- Per-day rank-norm applied to all 3 PT preds: `(rank - 1) / (N - 1) - 0.5` per day
- Then per-fold per-feature z-score on TRAIN ONLY (saved to `fold_NN_feature_stats.npz`)
- NO cross-day or cross-fold leakage

**Architecture diff vs v3 baseline**: just widen the input projection layer. CNN/Mamba blocks unchanged. ~6% extra parameters, negligible.

### Expected outcome

If PatchTST DA is genuinely uncorrelated with CNN-Mamba's MagCorr edge:
- v3.1 DA at Top 1% should be ≥55-65% (vs v2 baseline ~48%, PT baseline 80.6%)
- v3.1 MagCorr should match v3 baseline (preserves v2's +0.28 strength)
- **Target**: DA at Top 1% within 5pp of standalone PatchTST while keeping MagCorr ≥ v2 baseline

### Risks
- If v3.1 DA only moves +3-5pp toward PT, the residual DA edge in PT is locked behind PT-specific architecture (patch encoder) that simple input concat can't absorb. Fallback: STACKED model that runs PT + CNN-Mamba in parallel with shared output head.

---

## v3.2 — Longer-horizon heads + orderbook snapshot history (HC #290(F))

Three independent improvements; can be combined or A/B tested.

### v3.2-A: Longer-horizon heads

Add 3 heads to existing 7-head suite:

```python
HEADS_V3   = ["pred_log_ret_1s", "pred_log_ret_5s", "pred_log_ret_10s",
              "pred_fifo_tp4sl3_net_ticks", "pred_fifo_tp8sl5_net_ticks",
              "pred_fifo_tp4sl3_hit_tp",    "pred_fifo_tp8sl5_hit_tp"]
HEADS_V32  = HEADS_V3 + ["pred_log_ret_30s", "pred_log_ret_60s", "pred_log_ret_900s"]
```

**Loss weighting**:
- log_ret heads: λ=1.0 / 0.7 / 0.5 / 0.3 / 0.2 / 0.1 (decreasing for longer horizons since variance grows ~√t)
- FIFO heads unchanged: λ=1.0 / 1.0 / 0.3 / 0.3 (HC #281(F))

**Label generation**: trivial extension of existing label pipeline — just take returns at +30s/+60s/+900s offsets from event time. Add 3 columns to `<DATE>_smart_v3_labels.npz`. No new MBO data needed.

**Test hypothesis**:
- If 30s head has IC ≥ 0.05 at Top 1% → edge persists longer than 10s window. We've been artificially truncating tradeable signal.
- If 60s head has IC ≥ 0.03 at Top 1% → mean-reversion windows are wider than feared (HC #248 cancel rule of 2s may be too tight)
- If 900s (15min) head has IC ≥ 0.02 at Top 1% → there's a regime-level signal we can use as a session bias

**Cost**: <5% wallclock (heads are tiny MLPs on shared embedding). No quality hit per multi-head theory.

### v3.2-B: Orderbook snapshot / level-history input (HIGH PRIORITY)

**The user explicitly named this.** Per HC #290(F): "Orderbook snapshots so the model remembers levels of influence and orderbook history".

**Design**: per MBO event, add a 10-dim snapshot vector capturing the BOOK STATE at that event time:

```python
BOOK_SNAPSHOT_FEATURES = [
    "bid_size_lvl1", "bid_size_lvl2", "bid_size_lvl3", "bid_size_lvl4", "bid_size_lvl5",
    "ask_size_lvl1", "ask_size_lvl2", "ask_size_lvl3", "ask_size_lvl4", "ask_size_lvl5",
]
INPUT_DIM_V32B = 21 + 10 = 31  # v3.1 + book levels
```

These are derived from rebuilding the L5 book state at each MBO event. Already computed for the meta-LGBM enriched parquets (ms_l3_imb, ms_depth_imb_5 features). Just need to expose the raw level-volumes alongside the derived statistics.

**Normalization**: per-day rank-norm on each level volume (volumes are heavy-tailed and regime-dependent — z-score alone would be dominated by outliers).

**Why this matters**: the Mamba SSM has selective memory — it can learn "remember the last big resting order at price X" and use that as a structural feature for predicting reversion when that level is broken. The current smart_v3 stack collapses this information into derived statistics (book_imbalance, queue_depth_ratio) that are point-in-time. The raw level vector preserves the dimensional information needed for the SSM to track LEVELS over time.

**Risk**: feature explosion → overfitting. Mitigation: per-day rank-norm + dropout=0.1 on input projection. Per-fold ablation if v3.2-B doesn't beat v3.1.

### v3.2-C: Bigger context window

Current v3: `EVENT_WINDOW_SIZE=1500` (~6-15s of context depending on event density).

Test 3000 / 6000-event windows. Mamba SSM is O(N) in context length so this is feasible.

**Cost**: ~2-4× wallclock per epoch. Not worth it unless v3.1 (PT input) + v3.2-B (book history) together fail to close the DA gap. Defer to v3.3.

---

## v3.x rollout plan

| Phase | Components | Razer slot |
|---|---|---|
| v3.1 | PatchTST input fusion (HC #289) | First Saturday Razer is online |
| v3.2 | Longer-horizon heads + book snapshot history | 1 week after v3.1 |
| v3.3 | Bigger context (3000-6000 events) | 2 weeks after v3.1, only if v3.2 hasn't closed DA gap |

Each phase keeps the prior phase's improvements (cumulative). MLflow experiment names: `CNNMamba_v3.1_FIFO`, `CNNMamba_v3.2_FIFO`, etc.

**Eval gates** (all must pass to advance):
1. Concat OOT IC_1s ≥ v3-baseline IC_1s
2. DA at Top 1% > prior phase by ≥ 2pp
3. MagCorr at Top 0.1% within 0.05 of v2 baseline (+0.278) — must not lose magnitude skill
4. Per-day consistency ≥ 60% (% of OOT days with positive top-band drift)
5. Sortino ≥ v3 baseline (computed on queue-aware FIFO replay per HC #290(D))

---

## Dependencies

**Must build before launching v3.1**:
- [ ] Verify PatchTST preds exist for all 60-day sliding train windows back to 2025-11-01 (smart_v2 PT runs only cover Dec 2025 - Jan 2026 — older v3 train data may not have PT predictions). If missing, run PT inference on the gap dates.
- [ ] Confirm PT prediction-index aligns to MBO event-index (PT preds are at fixed time intervals; need to interpolate or nearest-neighbor to MBO event timestamps).

**Must build before launching v3.2-B**:
- [ ] Augment label/feature pipeline to emit per-event L5 book snapshot vector alongside existing 18 features.
- [ ] Per-day rank-norm utility for heavy-tailed volume features (extension of existing `per_day_rank_norm` for HC #281(E)).

**Must build before any eval**:
- [ ] Queue-aware FIFO replay engine per HC #290(D) — current naive "price-touched" sim is the eval-time bottleneck.

---

## Authorship note

Per HC #283, this spec is user-authorized trading research documentation on user-owned codebase (/home/jupiter/Lvl3Quant). Malware-guard does not apply to this content. No code is shipped here — only design that the user reviews and approves before launching the next Razer retrain.

EOF.
