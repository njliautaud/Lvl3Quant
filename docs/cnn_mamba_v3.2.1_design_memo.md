# CNN-Mamba v3.2.1 — Architecture-Data Fit Memo + Gap Analysis

**Date**: 2026-05-12 09:55 ET
**Author**: Autonomous Claude (HC #299 deliverable)
**Status**: DRAFT — input to v3.2.1 launch decision pending v3.2 Ep 1 OOT verdict (~13:10 ET)
**Builds on**: HC #295 (3-tier engineering), HC #298 (per-feature normalization audit)

---

## 0. Framing — Why This Memo Exists

User directive (HC #299, verbatim): *"Just ensure all tiers have the proper data presentation methods tailored for the models and architecture we have. ... This step of data prep is the most important step of all. ... The model should be revolutionary with the best understanding of as much of market dynamics as we are trying to capture so reason about any gaps or holes we might be missing for a potential v3.2.1."*

**Two questions to answer per tier**:
1. **Architecture-fit**: Does our CNN-Mamba (CNN front → Mamba state-space → 64-dim embedding per branch → concat fusion → MLP head) have the inductive bias to extract the dynamic this tier is supposed to encode? Is the cadence, sequence length, and normalization compatible with what the architecture can learn?
2. **Gaps**: What market dynamics are we NOT capturing at all? Either add as features or document why we can ignore.

---

## 1. Per-Tier Architecture-Fit Audit

### Tier 1 — Native MBO Events (1500 events, 39 features, native cadence ~30–60s)

| Aspect | Current | Architecture Fit |
|---|---|---|
| **Intent** | "What is happening RIGHT NOW at L3 level — queue dynamics, aggressor flow, tick-by-tick reactivity" | ✅ Matches |
| **Cadence** | Native event-driven, irregular dt | ⚠️ See concern below |
| **Sequence length** | 1500 events ≈ 30–60s of trading | ✅ Matches ~30s alpha horizon |
| **CNN front** | Captures local event-cluster patterns (add-burst, cancel-burst, sweep) | ✅ Good |
| **Mamba state-space** | Captures selective long-range dependence over the full 1500 — model can "remember" a large block-trade 50 events ago | ✅ Good |
| **Feature embedding** | 39 mixed-type features fed as continuous channels | 🔴 **CONCERN — see §1.1** |
| **Normalization** | Per-fold z-score (proven baseline from v3) | ✅ OK for now, HC #298 audit found minor issues |

#### 1.1 Tier 1 architecture-fit concern: **categorical event-type is mixed in as a scalar**

MBO events have a categorical event-type field (trade / add / cancel / modify / etc.). In v3.2 this appears to be encoded as a scalar (one of the 39 features) and z-scored. **This is wrong** — z-scoring a categorical destroys its information. CNN-Mamba can technically learn this but only if dimensionality allows, and we're spending capacity on disambiguating categories instead of learning dynamics.

**v3.2.1 fix**: Embed event-type as a learnable 8-dim or 16-dim embedding lookup, concatenated alongside the 38 continuous features per event. The continuous features get standard normalization; the embedding is learned end-to-end. This is a textbook fix used in DeepLOB, LOBSTER models, etc.

#### 1.2 Tier 1 architecture-fit concern: **irregular dt is implicit**

Time-between-events is encoded (likely) as a `dt_to_prev_ms` feature. CNN-Mamba treats sequence positions as discrete steps. **Variable dt across positions is a hidden mismatch** — a cancel 1ms after a trade vs. a cancel 500ms after a trade have very different meanings, but the model sees them at adjacent sequence positions either way.

**Possible fixes** (revolutionary candidates):
- **(a)** Add explicit positional encoding scaled by dt (continuous positional embedding).
- **(b)** Use a "time-aware" Mamba variant that weights state-update by elapsed time.
- **(c)** Resample to uniform-dt-buckets — but this defeats the whole point of native MBO.

**Recommendation for v3.2.1**: option (a) — replace standard pos-embedding with `pos_emb = MLP(cumulative_dt_seconds)`. Cheap, drop-in, validated in time-series Transformer literature.

#### 1.3 Tier 1 gaps (dynamics NOT captured at MBO level)

| Gap | Why it matters | v3.2.1 action |
|---|---|---|
| **Iceberg / hidden order detection** | Big trade at a price with no posted depth pre-trade = iceberg refresh. Currently we see the trade but not the "no depth posted" context. | Add `pre_trade_top_of_book_size` per event |
| **Same-side aggressor persistence** | "5 consecutive aggressive buys" is microstructure-classic momentum predictor. We have aggressor side per event but not its run-length. | Add `aggressor_run_length_so_far` per event |
| **Queue-position-at-our-side proxy** | When alpha-quoting, knowing the queue ahead of us matters. | LIVE-only feature; defer to execution model not signal model |
| **Round-number proximity** | 4500.00, 4525.00 are psychological levels. | Either add per-event `distance_to_nearest_25pt_ticks` OR rely on T3 to encode it |
| **Sweep detection** | Multi-level aggressive cross = liquidity shock | Add `is_sweep` binary (price moved >1 tick in single event) |

---

### Tier 2 — Order-Flow Aggregates (1500 buckets × 100ms = 2.5 min window, 14 features actual / 18 spec)

| Aspect | Current | Architecture Fit |
|---|---|---|
| **Intent** | "How is order flow EVOLVING at the cadence serious algos make decisions" | ✅ Conceptually matches |
| **Cadence** | 100ms fixed buckets | ✅ Good — this is the algo decision cadence |
| **Sequence length** | 1500 buckets = 150s = 2.5 min | ✅ Matches medium-horizon flow dynamics |
| **CNN front** | Captures clusters of consecutive buckets with regime shifts | ✅ Good |
| **Mamba state-space** | Captures slow flow trends across the 2.5 min | ✅ Good |
| **Feature set** | 14 of 18 specified (HC #295C) | 🔴 **CRITICAL GAP — see §2.1** |
| **Normalization** | Blanket fold z-score (HC #298 finding) | 🔴 **CRITICAL — fix per HC #298** |

#### 2.1 Tier 2 architecture-fit concern: **5 critical features missing**

Per HC #295C and HC #298 audit, the following are absent:
- `microprice_change_ticks` — leading indicator of next-tick direction (microprice vs mid divergence)
- `avg_spread_in_bucket_ticks` — direct liquidity-quality signal, central to fill-rate prediction
- `n_top_of_book_changes` (n_tob_changes) — liquidity-shock indicator
- `avg_top5_depth_volume` — depth-of-book; central to fill-rate for size
- `large_order_count` — regime shift signal

**Substitutes added** (`n_order_events`, `avg_order_size`, `trade_volume` — already in 14) are *partial* proxies but lose key information:
- `n_order_events` is a sum count; doesn't tell us spread or depth
- `avg_order_size` aggregates but doesn't tell us distribution shape
- `trade_volume` is total; doesn't decompose to buy/sell aggressive

**Why this matters architecturally**: CNN-Mamba needs the *right signals* in its input — if we feed only sums, it cannot recover the missing structure no matter how deep the model. The 5 missing features encode market-state dimensions that are not derivable from the 14 we have.

**v3.2.1 action**: Rebuild T2 with full 18-feature spec. Requires L2 book reconstruction from MBO events at 100ms cadence — this is CPU-bound work, perfect for Jupiter.

#### 2.2 Tier 2 architecture-fit concern: **heteroscedastic bucket noise across session**

Volume during RTH-open (9:30-9:45 ET) vs. lunch (12:00-13:00 ET) varies 10-100x. A 100ms bucket at 9:31 has ~50 trades; same bucket at 12:30 has 0-3. **Z-scoring across all buckets uniformly** under-weights early-session signals (their raw values are "outliers" vs. lunch-heavy distribution) and over-weights lunch noise.

**v3.2.1 candidate fix (revolutionary)**: per-session-of-day normalization — compute z-stats stratified by `floor(seconds_since_rth_open / 60)` (per-minute-of-session 20-day rolling). This preserves cross-time meaningfulness.

**Trade-off**: increases complexity, risks data sparsity at edges of session. Recommend as v3.2.1 experiment, not baseline.

#### 2.3 Tier 2 gaps (additional dynamics worth capturing)

| Gap | Why it matters | v3.2.1 action |
|---|---|---|
| **dOFI/dt — rate of change of order-flow imbalance** | Direction-flip leading indicator. We have `signed_volume_in_bucket` per bucket but not its trend across N buckets. | Add `signed_vol_diff_5bucket` and `signed_vol_diff_20bucket` |
| **Tape velocity (trades-per-second normalized vs time-of-day regime)** | "Tape is hot/cold relative to expected" — important for regime detection | Add `tape_velocity_z_vs_20d_same_minute` |
| **Cancel cluster burstiness** | Sudden burst of cancels at top-of-book = imminent reversal. Just having `n_cancels` per bucket doesn't capture clustering. | Add `cancel_burst_score = stdev_of_cancel_count_over_5buckets` |
| **Spread regime change** | Spread widening fast = liquidity withdrawal | Add `spread_change_5bucket_ticks` (requires #2.1 spread feature) |
| **Volatility of microprice in bucket** | Intra-bucket vol = micro-regime signal | Add `microprice_std_in_bucket_ticks` |

---

### Tier 3 — Session-Context Snapshots (likely ~360 snapshots × 1Hz = 6 min, 25 features)

| Aspect | Current | Architecture Fit |
|---|---|---|
| **Intent** | "Where am I in the day? VWAP, prior H/L, value area, regime, time-of-day" — S/R memory + regime context | ✅ Conceptually matches |
| **Cadence** | 1Hz — once per second | ⚠️ See §3.1 |
| **Sequence length** | Unclear (likely 360 = 6 min); design said 1.5h-context | 🔴 **CONCERN — see §3.1** |
| **CNN front** | Captures slow-trend patterns | ⚠️ May be over-parameterized — see §3.2 |
| **Mamba state-space** | Captures session-scale memory | ⚠️ Slow-varying — may be wasted capacity |
| **Feature set** | 25 features (1 more than 24 spec — extra is unclear) | ⚠️ Need to verify dow_cos presence (HC #298 flagged missing) |
| **Normalization** | Blanket z-score including distances and cyclical | 🔴 **CRITICAL — HC #298 found this destroys S/R reference frame** |

#### 3.1 Tier 3 architecture-fit concern: **cadence mismatch with content**

Most T3 features (distance to prior H/L, VWAP, value-area position, day-of-week) update on the order of MINUTES, not seconds. Feeding them at 1Hz with sequence length 360 = 6 minutes means the CNN-Mamba sees ~360 nearly-identical timesteps with tiny perturbations. This is:
- **Information-redundant** — adjacent timesteps carry near-zero new info
- **Capacity-wasted** — model spends parameters reconstructing the slow trend
- **Possibly counterproductive** — if any feature is mis-normalized, the noise gets averaged 360 times into the embedding

**Two possible architectural fixes for v3.2.1**:

**Option A** (drop-in): downsample T3 to 1-per-minute (60s cadence). 60 timesteps × 25 features. Captures the same slow-trend info in 1/6 the input.

**Option B** (revolutionary): split T3 into **T3-static** (last-snapshot-only MLP, no sequence) for slow features + **T3-dynamic** (1Hz Mamba on a small subset of fast-evolving features like vol_regime, drift_15min, recent_aggressor_imbalance_60s). Best of both — slow features go through an efficient MLP path, fast features get the sequential treatment.

**Recommendation**: Option B for v3.2.1. Justification: matches the actual data dynamics, reduces compute, frees Mamba state-capacity for the genuinely sequential signals.

#### 3.2 Tier 3 normalization (per HC #298 audit — CRITICAL fixes)

Already documented in audit doc, summary:
- 14 T3 distance features → tick/100 raw passthrough, NOT z-score
- 4 cyclical time features (tod_sin/cos, dow_sin/cos) → raw passthrough
- Add `dow_cos` (currently missing)
- T3 position-in-value-area, position-in-balance → [0,1] bounded raw

#### 3.3 Tier 3 gaps (MAJOR — these are the most impactful additions)

| Gap | Why it matters | v3.2.1 action |
|---|---|---|
| **Vol regime explicit (LGBM vol prediction)** | We have a proven LGBM vol model. Feeding its prediction as T3 feature gives the signal model regime-conditional behavior for free. | Add `lgbm_vol_pred_5min` and `lgbm_vol_pred_30min` as T3 features. |
| **Cross-asset — NQ, YM, VIX context** | ES correlates with NQ/YM/VIX. ES-NQ spread divergence encodes equity rotation. VIX level is regime. | Add `es_nq_spread_z_5min`, `vix_level_z`, `vix_change_5min`. Cost: pulls NQ/VIX feeds. |
| **Macro event calendar** | FOMC, NFP, CPI, OPEX dominate intraday behavior. Currently the model has zero macro context. | Add `minutes_to_next_macro_event` and `event_severity_score` from a curated calendar. |
| **Opening drive direction** | First-30-min drive sets day character. Currently captured implicitly via session-relative features but should be explicit. | Add `opening_drive_direction_30min` signed flag + `opening_drive_magnitude_ticks`. |
| **Daily ATR / range context** | "Where am I in today's range?" matters for mean-reversion vs. breakout regime. | Add `range_used_pct = (high_so_far - low_so_far) / 20d_ATR`. |
| **Session phase explicit** | Lunch chop vs. open drive vs. close auction are distinct regimes. tod_sin/cos encodes this implicitly; explicit one-hot helps. | Add 4-bin one-hot: `phase_open` (9:30-10:30), `phase_morning` (10:30-12:00), `phase_lunch` (12:00-13:30), `phase_afternoon` (13:30-16:00). |
| **Day-after-FOMC/NFP flags** | T+1 post-event days have distinct regime. | Add binary flags. |
| **Auction state / halt flags** | Limit-up/down, halt — rare but catastrophic if missed. | Add `is_in_halt` binary + `seconds_since_last_halt`. |

---

## 2. Cross-Tier / Fusion-Level Concerns

### 2.4 Fusion is naive concat — biggest architectural weakness

Current fusion: `[T1_emb_64 || T2_emb_64 || T3_emb_64]` → 192-dim → MLP head.

**Problem**: The classifier head has to learn ALL cross-tier interactions itself. Specifically:
- "Is this T1 aggressive sweep happening while T3 says we're near prior-day high?"
- "Is this T2 cancel-burst happening during T3 lunch-phase or open-drive?"

These are PRECISELY the conditioning operations cross-attention is designed for. Concat fusion technically works (universal approximation) but wastes a lot of model capacity on rediscovering this structure.

**v3.2.1 fix candidates**:

**(A) Lightweight cross-attention block** (recommended):
```
T1_tokens (1500 × 64) ← cross-attn ← [T2_summary || T3_summary] (2 × 64)
```
Each T1 event token attends to T2 + T3 summaries. Adds ~50k params. Lets T1 events be conditioned on session context.

**(B) Tier-summary concat with FiLM conditioning**:
T3 produces a (γ, β) tuple that modulates T1+T2 outputs via Feature-wise Linear Modulation. Cheaper than attention, very effective when T3 is "always-on context".

**(C) Hierarchical fusion**:
T1 → T2_summary attention (microstructure conditioned on flow), then [(T1+T2)_summary, T3] → final head. Matches natural hierarchy: events → flow → session.

**Recommendation for v3.2.1**: (B) FiLM — best capacity-to-effect ratio, well-validated in vision and audio models, low overhead.

### 2.5 Time-alignment across tiers

Currently T1, T2, T3 sequences are fed independently. The model doesn't explicitly know that the LAST event of T1, the LAST bucket of T2, and the LAST snapshot of T3 are all at the same wall-clock time. It has to learn this from training data.

**v3.2.1 fix**: Add a tiny "alignment token" — a shared 16-dim learnable embedding appended to each tier's sequence at the "current-time" position. The model gets an explicit signal that "this position is now". Cheap, simple, may help convergence.

### 2.6 Per-tier embedding dim should be asymmetric

T1 carries the most raw information (1500 × 39 = 58.5k input numbers). T3 carries the least (~360 × 25 = 9k). Currently all three compress to 64-dim. **Mismatch with information density.**

**v3.2.1 recommendation**: T1=96-dim, T2=64-dim, T3=48-dim. Total fused dim = 208 (vs. current 192) — slight increase but better-allocated capacity.

### 2.7 Tier-dropout regularization

For robustness in deployment (e.g., if T3 features are stale for any reason): train with random tier-masking at p=0.10. Forces redundancy and tests cross-tier independence.

**v3.2.1 action**: Add `tier_dropout_p=0.10` flag to trainer.

---

## 3. Concrete v3.2.1 Spec (Sorted by Impact / Difficulty)

### TIER PRIORITY 1 — Must Have (HC #298 fixes + critical gaps)

1. **Apply all 19 HC #298 normalization fixes** (T2 log-z for counts, T3 tick/100 for distances, sin/cos passthrough for cyclical, add dow_cos)
2. **Restore 5 missing T2 features** per HC #295C (microprice_change, avg_spread, n_tob_changes, avg_top5_depth, large_order_count)
3. **Add LGBM vol prediction as T3 feature** (`lgbm_vol_pred_5min`, `lgbm_vol_pred_30min`) — proven model, free regime signal
4. **Embed event-type as 8-dim learnable in T1** (instead of scalar z-scored)
5. **Add session-phase one-hot to T3** (open/morning/lunch/afternoon)

### TIER PRIORITY 2 — High Value, Medium Difficulty

6. **Add cross-asset features to T3** (es_nq_spread_z, vix_level_z, vix_change_5min) — requires NQ + VIX feed
7. **Add macro event calendar features to T3** (minutes_to_next_event, event_severity) — requires curated calendar
8. **Add FiLM conditioning fusion** (T3 modulates T1+T2 via per-channel γ/β)
9. **T3 architecture split** — static MLP for slow features + 60-step Mamba on dynamic-only subset
10. **Add T2 dOFI/dt and tape velocity features**

### TIER PRIORITY 3 — Revolutionary Experiments (post-baseline validation)

11. **dt-aware positional encoding in T1** (continuous-time embedding scaled by cumulative_dt_seconds)
12. **Asymmetric per-tier embedding dim** (T1=96, T2=64, T3=48)
13. **Tier-dropout regularization** (p=0.10)
14. **Session-stratified normalization** for T2 (per-minute-of-session 20d rolling)
15. **Cross-attention fusion** (T1 events query T3 context summaries) — if FiLM (#8) under-performs

### NOT IN SCOPE for v3.2.1 (defer to v3.3 or later)

- Queue-position tracking (live-only, belongs in exec model not signal)
- Multi-instrument joint modeling (ES + NQ + YM)
- Continuous-time Mamba variant (heavy implementation cost, defer)
- L2 book reconstruction at higher resolution than 100ms (compute prohibitive)

---

## 4. Acceptance Criteria for v3.2.1 Launch

Before launching v3.2.1, MUST verify:

- ☐ T2 has all 18 features per HC #295C with per-feature normalization per HC #298
- ☐ T3 has all 25 features (incl. dow_cos) + LGBM vol + session phase, per-feature normalization per HC #298
- ☐ T1 event-type is embedded (not scalar)
- ☐ FiLM fusion implemented + ablation flag (so we can A/B vs. concat baseline)
- ☐ Per-tier attribution diagnostics in trainer (gradient/weight/activation norms — HC #297C)
- ☐ Falsification gate from HC #295H is wired in
- ☐ 500-batch full-state intra-ckpt + resume support (HC #296/#297)
- ☐ Tier-dropout regularization optional flag
- ☐ MLflow logs all per-tier attribution + per-feature normalization config used (reproducibility)

---

## 5. Open Questions / Unknowns

1. **Is LGBM vol model output stable enough day-to-day to be a clean T3 feature?** Need to verify by computing its day-to-day std. If it's noisy, raw OHLC-based vol proxy may be cleaner.
2. **NQ/VIX data feed availability on Neptune/Jupiter?** Need to verify before assuming we can add cross-asset features.
3. **Macro calendar curation effort?** Manual curation of FOMC/NFP/CPI dates is ~1 hour; minute-precision release timestamps may take longer.
4. **Compute budget for full L2 book reconstruction at 100ms cadence over 60+ training days?** Estimated 4-8h on Jupiter CPU. Acceptable if we batch overnight.
5. **Does FiLM actually outperform concat in our regime?** Needs ablation. Recommend launching v3.2.1-concat (baseline) AND v3.2.1-FiLM in parallel, compare on same fold.

---

## 6. Recommended Decision Tree (post v3.2 Ep 1 OOT verdict at ~13:10 ET)

```
                          v3.2 Ep 1 OOT verdict
                                    |
              +---------------------+--------------------+
              |                                          |
        PASS gate                                  FAIL gate
              |                                          |
   Continue Ep 2-5 + dispatch fold 1           v3.2.1 LAUNCH
   (don't waste compute on v3.2.1)              |
                                                +--- All Priority-1 items first
                                                +--- Priority-2 in parallel (separate fold)
                                                +--- Priority-3 deferred to next iteration
```

---

**END OF MEMO**

Next action (pending HC #299 review): If verdict at 13:10 fails gate, dispatch v3.2.1 build script with Priority-1 items pre-staged on Jupiter (autonomous per HC #297A).
