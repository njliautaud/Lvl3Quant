# V3.4 ARCHITECTURE DESIGN MEMO — DUAL-CNN-MAMBA + BOOK-SHAPE PYRAMID + LATE-FUSED MACRO

**Status:** DESIGN ONLY — do not implement until v3.3 fold 0 finishes + queue+adv-sel verdict completed.
**Owner:** project / referenced from CLAUDE.md tool index.
**Last updated:** 2026-05-14
**Supersedes:** the bucketed-orderflow T2 in v3.2/v3.3 (see HC #329/#335/#356).

---

## 1. Why v3.4 — the architectural failure modes of v3.2/v3.3 we're trying to fix

Per HC #353 (information-theoretic axiom): v3.2's inputs are a strict superset of v2's (v2 raw orderflow + T2 + T3). Under correct architecture, v3.2's edge MUST be ≥ v2's. If empirically v3.2 ≤ v2, the bug is architectural. The candidate failure modes flagged in DIRECTIVES are:

| # | Failure mode | v3.4 response |
|---|---|---|
| (i) | Single 1D-CNN trunk can't extract 2D book-topology features | **Dual-CNN trunk** — separate 2D book-shape CNN + 1D event-temporal CNN |
| (ii) | Macro/session context floods the high-rate trunks | **Late-fused T3** via FiLM conditioning |
| (iii) | T2 is currently bucketed-orderflow (redundant with T1) — model can't learn book shape because it isn't fed book shape | **T2 = 20-level book-shape pyramid only**; bucketed orderflow DELETED |
| (iv) | 32 heads dilute training signal | Keep multi-head but **uncertainty-weighted loss only** (inherit v3.3) + drop heads that fail HC #326 ablation |

The v3.4 spec below addresses (i)–(iii) directly.

---

## 2. Input tiers — what the model actually sees

### T1 — Event-temporal stream (UNCHANGED from v3.2)
Raw order events as a 1D time-series. This is the high-frequency token stream the model has always had.

- **Shape:** `(window, N_event_features)`
- **N_event_features:** ~25 (current smart_v3 feature set — multi-scale OFI, microprice, trade signs, etc. per `alpha_discovery/features/smart_v3.py`)
- **Window:** 1500 events (~5-20 seconds depending on activity)
- **Rate:** event-clock (varies; typically 10-100 Hz during RTH)
- **What it captures:** raw order-flow dynamics — the WHO/WHAT/WHEN of every event

### T2 — Book-shape pyramid (NEW DESIGN — replaces bucketed-orderflow T2)

The visual / topological snapshot of the limit-order book. Per HC #329 and HC #356, this must be 20 levels of book state, not bucketed events.

- **Shape:** `(window, 20_levels, K_book_features)`
- **20 levels:** mid±1 tick, mid±2 ticks, ..., mid±10 ticks (10 bid + 10 ask)
- **K_book_features per level:** 6 baseline
  - `size` — total resting size at this level (lots)
  - `n_orders` — count of distinct orders at this level
  - `mean_order_age_sec` — average age of resting orders (proxy for queue maturity)
  - `cancel_rate` — cancellations per second at this level over a short trailing window
  - `add_rate` — adds per second at this level over a short trailing window
  - `executed_size_in_window` — total executed (filled) size at this level over a short trailing window
- **Window:** same as T1 (sampled at the same event timestamps; for non-event-clock interpolate or hold last)
- **Rate:** matches T1
- **What it captures:** book topology — depth pyramid skew, queue imbalance gradient, cancellation density per level, depletion shape. This is the "shape" the user has repeatedly asked for (HC #329, #335, #353, #356).

**Why this shape works for a 2D CNN:** treat T2 as an image of shape `(time, level, channel)`. 2D convolutions across (time, level) learn:
- Vertical patterns (across levels at a fixed time): queue-imbalance shape, depth wedge, cancellation-density gradient
- Horizontal patterns (across time at a fixed level): level-specific depletion, replenishment, sweep events
- Diagonal patterns: book-walking moves (price drifts as adjacent levels deplete in sequence)

Forcing this through a 1D CNN (as v3.2 currently does for T1 = orderflow) flattens the level-axis and the model can only learn time patterns, not topology. That is the suspected root cause of v3.2's underperformance — model can't see what we're trying to teach it.

### T3 — Session / macro context (UNCHANGED from v3.2)
Low-rate session-scale features. Sampled per minute or per session-event (not per market event).

- **Features (current set):** previous_day_high, previous_day_low, session_VWAP, time_of_day_sin, time_of_day_cos, minutes_since_open, day_of_week_one_hot, volatility_regime (low/mid/high), trend_regime
- **Rate:** ~1/60 Hz (per minute) — much slower than T1/T2
- **What it captures:** regime, time-of-day effect, session-relative anchors

**v3.4 change:** T3 is **late-fused** (HC #353), not concatenated with T1/T2 at the input. Justification: T3 changes ~1 sample per minute while T1/T2 change ~10-100 times per second. Concatenating low-rate signals onto high-rate trunks at the input either:
- (a) forces the model to ignore them (channel becomes ~constant within a window), or
- (b) wastes capacity learning to ignore them

Late-fusion via FiLM conditioning (Feature-wise Linear Modulation, Perez et al. 2017) is the canonical solution. T3 features modulate the LATER layer activations (after the fusion-Mamba backbone) via per-feature γ/β coefficients. This is the same pattern HC #134-135 prescribes for macro-context fusion.

---

## 3. Architecture — Dual-CNN-Mamba

```
┌─────────────────┐                              ┌───────────────────┐
│   T1            │                              │   T2              │
│ (W, N_evt)      │                              │ (W, 20, K_book)   │
│ event-temporal  │                              │ book-shape        │
└────────┬────────┘                              └─────────┬─────────┘
         │                                                  │
         ▼                                                  ▼
  ┌─────────────┐                                    ┌─────────────┐
  │ 1D CNN      │                                    │ 2D CNN      │
  │ trunk       │                                    │ trunk       │
  │ (time-only) │                                    │ (level×time)│
  └──────┬──────┘                                    └──────┬──────┘
         │ (W, C_t1)                                        │ (W, C_t2)
         │                                                  │
         └──────────────────┬───────────────────────────────┘
                            ▼
                     ┌─────────────┐
                     │ Fusion      │  concat OR cross-attention
                     │ (W, C_fuse) │
                     └──────┬──────┘
                            ▼
                     ┌─────────────┐
                     │   MAMBA     │  state-space backbone
                     │  backbone   │  (selective SSM, sequential)
                     │             │
                     └──────┬──────┘
                            │  (W, C_out)
                            ▼
       T3 ──FiLM──► ┌─────────────┐
   (γ, β per chan)  │ FiLM        │  modulate hidden state with macro context
                    │ modulation  │
                    └──────┬──────┘
                            ▼
                     ┌─────────────┐
                     │ Multi-head  │  inherit v3.3 head set + σ uncertainty heads
                     │ outputs     │
                     └─────────────┘
```

### Component specs

**T1 trunk — 1D CNN:**
- 4-6 blocks of [Conv1d(kernel=3) → GELU → LayerNorm → residual]
- Stride 1, growing channel widths (e.g. 32 → 64 → 128 → 192)
- Output `(W, C_t1)` where C_t1 ≈ 192

**T2 trunk — 2D CNN:**
- 3-4 blocks of [Conv2d(kernel=(3,3)) → GELU → LayerNorm → residual] — operates on (time, level) plane
- Then `nn.AdaptiveAvgPool2d` or attention-pool over the level axis to collapse to `(W, C_t2)`
- C_t2 ≈ 96
- KEY: do NOT downsample the time axis in the T2 trunk; we need T1 and T2 outputs to be sequence-aligned for fusion.

**Fusion:**
- Option A (simpler, first version): channel-concat `(W, C_t1 + C_t2)` → linear projection to `(W, C_fuse)` where C_fuse ≈ 256
- Option B (later): cross-attention from T1-token to T2-token at each timestep — more expressive but more compute

**Mamba backbone:**
- 4-6 Mamba blocks (SSM with selective recurrence)
- d_model = C_fuse = 256
- d_state = 16, d_conv = 4 (standard Mamba hyperparams)
- This is the temporal context aggregator — same role as v2/v3.2/v3.3's Mamba

**T3 → FiLM modulation:**
- T3 features (~15 dims) → 2-layer MLP → outputs γ ∈ R^{C_out} and β ∈ R^{C_out}
- Modulation: `h_out = γ * h_mamba + β`
- Applied AT the head-input layer (after Mamba), not throughout the trunks
- This is the proven late-fusion pattern from Perez 2017 (FiLM paper)

**Multi-head outputs:**
- Inherit v3.3's uncertainty-weighted head set (32 heads including σ uncertainty per horizon)
- AFTER the HC #326 head-importance ablation (run on Jupiter test-bench per HC #358), drop heads with no value-add to reduce 32 → ~16-20

---

## 4. Training rules (unchanged from v3.3 except where noted)

- **SLIDING window walk-forward** (HC #0, never expanding)
- **Uncertainty-weighted loss** (inherit v3.3)
- **MLflow logging** mandatory (CLAUDE.md)
- **Intra-checkpoint every 500 batches** (HC #338 resumption pattern)
- **Resume-from-intra-ckpt** as default crash recovery (HC #338)
- **batch_size adjustment via env var** (HC #309 — no source edits for OOM mitigation)
- **NEW: data-loader memory budget** — Tier-2 book-shape adds ~20-level × 6-feature tensor per sample. Profile RAM before launching; expect ~30% bump over v3.3.

---

## 5. Expected wins vs. v3.2 / v3.3

Hypothesized improvements (to be empirically validated):

| Dimension | v3.2/v3.3 | v3.4 hypothesis | Why |
|---|---|---|---|
| Book-shape feature use | None (T2 is bucketed orderflow, redundant with T1) | **Strong** | 2D CNN on (time, level) explicitly learns topology |
| T3 macro signal use | Possibly drowned (early concat) | **Better** | Late-fusion via FiLM doesn't pollute high-rate trunks |
| IC at 1s | v3.3 best so far (0.2694 intra-ckpt) | Expected ≥ v3.3 | Same T1 + Mamba; T2/T3 strict adds |
| IC at longer horizons (10s/30s) | v3.3 weak (σ collapse) | Expected stronger | T3 macro context relevant for longer-horizon predictability |
| Queue-aware execution edge | Inherits whatever heads predict | **New native heads** if we add `p_fill_at_touch_K_evals` per HC #330 | Direct queue-position prediction head trained on MBO data |

---

## 6. Open design decisions (to discuss when v3.4 is greenlit)

1. **Native FIFO heads (HC #330):** should v3.4 add bracket-AGNOSTIC native FIFO heads — `p_fill_at_bid_within_Ks`, `p_fill_at_ask_within_Ks`, `expected_queue_position`, `time_to_fill_distribution`? These are book-dynamics primitives, not TP/SL-bracket-specific. Current v3.2/v3.3 `pred_fifo_tp4sl3_net` heads are bracket-coupled per HC #330 (concern: too tied to a specific TP/SL config).

2. **Cross-attention fusion (Option B):** worth implementing v0 with concat-fusion, then ablating against cross-attention to measure the marginal gain.

3. **T2 feature additions:** beyond the 6 baseline per-level features, consider:
   - `order_arrival_intensity` (Poisson rate of new orders at this level)
   - `passive_aggressive_ratio` (executed-size vs. canceled-size)
   - `level_age_sec` (when this level first appeared)

4. **Multi-symbol generalization:** does v3.4's architecture allow training on multiple symbols (ES + NQ + RTY) with symbol-embedding conditioning? Out of scope for v3.4 v0; flag for v4.x.

---

## 7. NOT in scope for v3.4 (per HC #353)

- NO IMPLEMENTATION until v3.3 fold 0 finishes AND queue+adv-sel verdict completes
- NO trainer code modifications during v3.3 active training (malware-guard HC #307D)
- This memo is DESIGN-ONLY. Implementation begins only after explicit greenlight.

---

## 8. Dependencies / prerequisites before implementation

1. ✅ v3.3 fold 0 completes (Neptune, in-progress per HC #338)
2. ⏳ v3.3 vs v3.2 vs v2 in-depth analysis per HC #358 (Jupiter test-bench)
3. ⏳ Queue+adv-sel verdict per HC #357 — confirms whether any current model has tradeable edge
4. ⏳ Head-importance ranking per HC #326 — informs which heads to keep / drop in v3.4
5. ⏳ T2 builder rewrite — new `build_v3_4_book_shape_features.py` (NEW script, malware-guard compliant) producing the 20-level pyramid parquet

Only after all five are green should v3.4 trainer be greenlit.

---

## References

- HC #329 — T2 should be 20-level book shape (user verbatim 2026-05-13)
- HC #330 — Native FIFO heads should be book-dynamics-aware, not TP/SL-coupled
- HC #335 — Tier 2 = full 20-level book pyramid; bucketed orderflow moves to T1
- HC #353 — V3.2 = superset of V2; if v3.2 ≤ v2 it's architecture failure + v3.4 dual-CNN-Mamba initial spec
- HC #356 — T2 = book-shape ONLY, bucketed orderflow redundant + removed
- HC #357 — Full market replay (queue + adv-sel) mandatory for all model evaluations
- HC #358 — Jupiter test-bench for in-depth alpha extraction across models
- Perez et al. 2017, "FiLM: Visual Reasoning with a General Conditioning Layer"
- Gu & Dao 2023, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
