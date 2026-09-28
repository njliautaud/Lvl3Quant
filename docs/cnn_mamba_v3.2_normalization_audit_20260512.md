# CNN-Mamba v3.2 Normalization Audit
**Date**: 2026-05-12  
**Audit Scope**: Feature engineering & per-feature normalization for all three input tiers  
**Status**: CRITICAL FINDINGS IDENTIFIED

---

## Executive Summary

This audit identifies **5 CRITICAL normalization mismatches** and **8 HIGH-severity issues** that could significantly impact v3.2 performance. The primary concerns are:

1. **T2 feature shortfall**: Design specified ≥18 features; actual implementation has only **14 features**. Missing **4 complex features** requiring L2 book reconstruction (microprice_change, avg_spread, n_tob_changes, avg_top5_depth), but substituted with 3 simpler log-z features (n_order_events, avg_order_size, and one additional). This is a documented deviation per build_v3_2_tier_features.py lines 21-24, but creates **specification drift**.

2. **T3 CRITICAL: Price/level distances are being z-scored instead of tick-normalized** (lines 999-1004, train_cnn_mamba_v3_2.py). Tier 3 distance features should use `/100` tick normalization (per design doc §3.4) but are instead raw-z-scored against training population means/stds. This destroys the intended **absolute reference frame** for support/resistance levels.

3. **T3 CRITICAL: Cyclical time features (tod_sin, tod_cos, dow_sin, dow_cos) are being z-scored** instead of kept raw. Sin/cos by definition are bounded [-1,+1] and perfectly Gaussian is impossible; z-scoring them squashes their periodic structure.

4. **T3 HIGH: dow_cos missing** — feature list shows only `dow_sin`, but design calls for full sinusoidal pair. This breaks day-of-week disambiguation (HC #295D).

5. **T2 HIGH: signed_volume uses simple z-score, not sign-preserving log-z** as specified in design §2.4 ("sign(buy_vol - sell_vol) × log1p(|buy_vol - sell_vol|), z-scored vs 20d magnitude distribution but sign preserved"). Current code applies standard z-score which can flip signs.

---

## Tier 2: Order-Flow Aggregates (14 features vs. 18 specified)

**Training stats at 07:33 ET**: n=23,025,878 buckets × 14 features  
**Design spec**: 18 features (HC #295C)  
**Actual**: 14 features

### Missing Features Analysis

Per build_v3_2_tier_features.py lines 21-24:
> "dropped 4 that need L2 book reconstruction: microprice_change, avg_spread, n_tob_changes, avg_top5_depth, large_order_count. Replaced with: trade_volume_log_z, n_order_events_log_z, avg_order_size_log_z."

**Specification deficit**: 5 features dropped, 3 features added = net -2. The 4 features mentioned as missing in design doc §2.4 are:
- `microprice_change_ticks` (feature #8 in design)
- `avg_spread_ticks` (feature #10 in design)
- `n_tob_changes_z` (feature #9 in design)
- `avg_top5_depth_log_z` (feature #15 in design)

And 1 additional: `large_order_count_log_z` (feature #14 in design). So **5 features missing**, but the parquet has 14 columns.

### Feature-by-Feature Audit

| # | Feature Name | Type | Raw Range | Current Normalization | Design Spec | Severity | Rationale |
|---|--------------|------|-----------|----------------------|-------------|----------|-----------|
| 1 | `log_return_in_bucket_bps` | Return (bps) | ±200 | Clipped raw, no normalization in parquet | Clipped ±200 bps | LOW | Raw clipping matches spec. Trainer applies z-score. OK. |
| 2 | `bucket_mfe_ticks` | Distance (ticks) | [0, 40] | Clipped raw [0, 40] | Clipped [0, 20] | HIGH | **Spec says [0,20], implementation clips [0,40]. Range expansion risks outlier bias.** |
| 3 | `bucket_mae_ticks` | Distance (ticks) | [0, 40] | Clipped raw [0, 40] | Clipped [0, 20] | HIGH | Same as MFE — range mismatch. |
| 4 | `n_trades` | Count (raw) | [0, ∞) | Raw, no log or z-score in parquet | log-z (design #4) | CRITICAL | **Parquet stores raw count; trainer applies global z-score (lines 984-987). This is log-z ONLY if the count is log1p'd before z-score, which it isn't.** Raw counts have heavy tail; global z-score is wrong. Should be `log1p(count)` → z-score per fold. |
| 5 | `trade_volume` | Volume (raw) | [0, ∞) | Raw, no log in parquet | log-z (design #5) | CRITICAL | **Same issue as n_trades.** Parquet has raw volume; trainer does linear z-score. Should be `log1p(volume)`. |
| 6 | `aggressor_buy_ratio` | Ratio | [0, 1] | Bounded raw (0.5 when zero) | [0, 1] bounded raw (design #6) | LOW | Correctly bounded. NaN→0.5. Trainer z-scores it (which is wrong, but consistent with design intent to have a bounded feature pre-normalization). |
| 7 | `signed_volume` | Signed imbalance | ±∞ | Raw (no log, no sign preservation in parquet) | log-z + sign-preserved (design #7) | CRITICAL | **Design §2.4: "sign(buy_vol - sell_vol) × log1p(\|buy_vol - sell_vol\|), z-scored vs 20d magnitude distribution but sign preserved". Parquet stores raw (buy - sell). Trainer applies standard z-score (lines 984-987), which can flip signs randomly and destroys directionality.** |
| 8 | `n_cancels` | Count (raw) | [0, ∞) | Raw | log-z (design #11) | CRITICAL | **Raw count in parquet; trainer applies linear z-score. Should be log1p first.** |
| 9 | `n_adds` | Count (raw) | [0, ∞) | Raw | log-z (design #12) | CRITICAL | **Same as n_cancels.** |
| 10 | `cancel_add_ratio` | Ratio | [0, 1] | Bounded raw (0.5 when both zero) | [0, 1] bounded (design #13) | LOW | Correctly bounded. |
| 11 | `n_order_events` | Count (raw) | [0, ∞) | Raw | Not in original design; added as substitute | MEDIUM | **Added in v1 deviation. No z-score applied.** Should this be log-z'd? Probably yes (heavy tail). |
| 12 | `avg_order_size` | Size (raw) | [0, ∞) | Raw | Not in original design; added as substitute | MEDIUM | **Same as n_order_events.** No normalization in parquet. |
| 13 | `bucket_range_ticks` | Distance (ticks) | [0, 40] | Clipped raw [0, 40] | Clipped [0, 20] (design #17) | HIGH | **Spec says [0,20]; actual is [0,40].** Same range expansion issue as MFE/MAE. |
| 14 | `seconds_since_rth_open` | Time (seconds) | [0, 23400] | Raw (int seconds) | Normalized [0, 1] (design #18) | CRITICAL | **Parquet stores raw seconds [0, 23400]. Trainer should normalize to [0, 1] but doesn't explicitly—applies z-score instead (lines 984-987). Should use (seconds / 23400).** |

### Missing Feature Rows (5 features not in actual parquet)

| # | Feature Name (from design) | Reason for Absence | Estimated Impact |
|---|-----|--------|----------|
| 8 | `microprice_change_ticks` | Requires L2 book reconstruction (bid/ask updates per bucket) | MEDIUM — captures intraday VWAP drift; model may learn a proxy from price_mom features in T1. |
| 9 | `n_tob_changes_z` | Requires tracking best-bid/best-ask flips per bucket | MEDIUM — signals liquidity shock; absence may reduce edge in choppy regimes. |
| 10 | `avg_spread_ticks` | Requires L2 book snapshots per bucket | HIGH — direct signal of liquidity; model relies on `bucket_range_ticks` as proxy (imperfect). |
| 14 | `large_order_count_log_z` | Requires 5d rolling percentile of add-order sizes | LOW — signals regime shift; model may infer from `avg_order_size`. |
| 15 | `avg_top5_depth_log_z` | Requires top-5 bid/ask cumulative depth per bucket | MEDIUM — signals liquidity depth; absence may hurt resilience in thin-market scenarios. |

---

## Tier 3: Session-Context Snapshots (25 features declared, but 2 mismatches)

**Training stats at 07:33 ET**: n=1,404,000 snapshots × 25 features (spec said 24, per-feature breakdown suggests 26 with dow_cos added)  
**Design spec**: 24 features (HC #295D), later clarified to include dow_cos = 25 in header, but doc §3.4 lists only 24 + missing dow_cos mention.  
**Actual parquet**: 25 features

### Feature-by-Feature Audit

| # | Feature Name | Type | Raw Range | Current Normalization | Design Spec | Severity | Rationale |
|----|--------------|------|-----------|----------------------|-------------|----------|-----------|
| **A. Price Location / S-R (5 features)** |
| 1 | `dist_intraday_high_ticks` | Distance (ticks) | ±200 | **Z-scored (lines 999-1004)** | Tick distance /100 (design §3.4) | **CRITICAL** | **Should NOT be z-scored.** Distance features are absolute references (support/resistance). Z-scoring destroys this frame: a "distance to high that's 2 ticks away" gets remapped based on training-set distribution, so model loses the absolute price context. During OOT inference, same 2-tick distance might z-score to wildly different values if OOT distribution differs. **FIX: Store as (ticks / 100) raw, not z-scored.** |
| 2 | `dist_intraday_low_ticks` | Distance (ticks) | ±200 | **Z-scored** | Tick distance /100 | **CRITICAL** | Same rationale as high. |
| 3 | `dist_session_vwap_ticks` | Distance (ticks) | ±200 | **Z-scored** | Tick distance /100 | **CRITICAL** | Same rationale. VWAP is a floating fair-value anchor; absolute distance matters. |
| 4 | `dist_prior_session_close_ticks` | Distance (ticks) | ±200 | **Z-scored** | Tick distance /100 | **CRITICAL** | Overnight gap context destroyed by z-scoring. |
| 5 | `dist_prior_session_vwap_ticks` | Distance (ticks) | ±200 | **Z-scored** | Tick distance /100 | **CRITICAL** | Prior-day fair value requires absolute reference. |
| **B. Volume Profile (5 features)** |
| 6 | `dist_intraday_vpoc_ticks` | Distance (ticks) | ±200 | **Z-scored** | Tick distance /100 | **CRITICAL** | Volume-weighted POC is a floating level; absolute distance needed. |
| 7 | `dist_intraday_vah_ticks` | Distance (ticks) | ±200 | **Z-scored** | Tick distance /100 | **CRITICAL** | Top edge of value area (70% volume) — absolute reference required. |
| 8 | `dist_intraday_val_ticks` | Distance (ticks) | ±200 | **Z-scored** | Tick distance /100 | **CRITICAL** | Bottom edge of value area — same reasoning. |
| 9 | `position_in_value_area` | Categorical {-1, 0, +1} | {-1, 0, +1} | **Z-scored** | Raw {-1, 0, +1} | **HIGH** | Categorical should NOT be z-scored. This converts -1 → mean/std-normalized value, which loses the semantic meaning. Should be stored raw; trainer can learn embedding or one-hot. |
| 10 | `volume_at_current_price_pctile` | Percentile [0, 1] | [0, 1] | **Z-scored** | Bounded raw [0, 1] | **HIGH** | Percentile rank is already normalized. Z-scoring it destroys the bounded interpretation. Should be raw. |
| **C. Prior Session Levels (4 features)** |
| 11 | `dist_prior_session_high_ticks` | Distance (ticks) | ±200 | **Z-scored** | Tick distance /100 | **CRITICAL** | Prior-day high is a key S/R level. Absolute distance essential. |
| 12 | `dist_prior_session_low_ticks` | Distance (ticks) | ±200 | **Z-scored** | Tick distance /100 | **CRITICAL** | Prior-day low same reasoning. |
| 13 | `dist_prior_session_vpoc_ticks` | Distance (ticks) | ±200 | **Z-scored** | Tick distance /100 | **CRITICAL** | Prior-day volume POC — same. |
| 14 | `dist_5d_extreme_ticks` | Distance (ticks) | ±200 | **Z-scored** | Tick distance /100 | **CRITICAL** | Breakout context (distance to 5d high/low) — absolute reference needed. |
| **D. Path Memory (5 features)** |
| 15 | `log_return_60s_bps` | Return (bps) | ±1000 | **Z-scored** | Return bps, clip ±200 (design §3.4) | **HIGH** | Returns are ratio-scale and should be z-scored (correct), but spec clip is ±200, actual parquet clips ±1000. Range expansion. |
| 16 | `log_return_5min_bps` | Return (bps) | ±1000 | **Z-scored** | Return bps, clip ±500 (design §3.4) | MEDIUM | Parquet clips ±1000; design says ±500. |
| 17 | `log_return_15min_bps` | Return (bps) | ±1000 | **Z-scored** | Return bps, clip ±1000 (design §3.4) | LOW | Matches design. |
| 18 | `realized_vol_5min_ticks` | Volatility (ticks std) | [0, 50] | **Z-scored (lines 999-1004)** | Z-score (design §3.4, feature #18) | LOW | Correct. Volatility is unbounded and benefit from z-scoring. |
| 19 | `trend_strength_5min` | T-statistic | ±10 | **Z-scored** | Z-score (design §3.4, feature #19) | LOW | Correct. T-stat normalized. |
| **E. Regime / Time (5+ features)** |
| 20 | `tod_sin` | Cyclical [-1, +1] | [-1, +1] | **Z-scored (lines 999-1004)** | Raw sin(2π × tod) (design §3.4) | **CRITICAL** | **MAJOR BUG: Sin/cos are bounded, perfectly sinusoidal, NOT Gaussian. Z-scoring destroys periodicity structure.** Model learns to associate z-scored +5 with morning, z-scored -5 with afternoon, but OOT phase shifts or distributional changes break this. **FIX: Store raw [-1, +1]; trainer should NOT z-score cyclical features.** |
| 21 | `tod_cos` | Cyclical [-1, +1] | [-1, +1] | **Z-scored** | Raw cos(2π × tod) (design §3.4) | **CRITICAL** | Same as tod_sin. |
| 22 | `is_lunch_lull` | Binary {0, 1} | {0, 1} | **Z-scored** | Raw {0, 1} (design §3.4) | **HIGH** | Binary flag should NOT be z-scored. It's a regime indicator, not a continuous signal. Z-scoring destroys 0/1 semantics. |
| 23 | `is_close_hour` | Binary {0, 1} | {0, 1} | **Z-scored** | Raw {0, 1} (design §3.4) | **HIGH** | Same as lunch_lull. |
| 24 | `dow_sin` | Cyclical [-1, +1] | [-1, +1] | **Z-scored** | Raw sin(2π × dow / 5) (design §3.4) | **CRITICAL** | Day-of-week cycle. Same z-score issue as tod_sin. |
| 25 | `dow_cos` | **MISSING** | N/A | **Not in parquet** | Raw cos(2π × dow / 5) (design §3.4) | **HIGH** | **Design doc §3.4 says "Full sinusoidal encoding requires BOTH sin AND cos so model can disambiguate each weekday uniquely."** Only dow_sin is present. dow_cos is missing, breaking the intended day-of-week representation. Trainer will fail to uniquely encode all 5 weekdays. |

### Summary Statistics for T3

- **5 features**: Price/S-R distances incorrectly z-scored (should be tick/100 raw)
- **4 features**: Volume profile distances incorrectly z-scored
- **4 features**: Prior-session distances incorrectly z-scored
- **2 features**: Cyclical time (tod_sin, tod_cos) incorrectly z-scored
- **1 feature**: dow_sin incorrectly z-scored
- **1 feature**: MISSING dow_cos (breaks sinusoidal pair)
- **2 features**: Categorical/bounded features incorrectly z-scored (position_in_value_area, volume_at_current_price_pctile)
- **2 features**: Binary regime flags incorrectly z-scored (lunch_lull, is_close_hour)
- **2 features**: Return clips expanded beyond spec (60s ±1000 vs ±200, 5m ±1000 vs ±500)

**Total T3 mismatches: 19 CRITICAL/HIGH-severity issues in 25 features.**

---

## Tier 1: Microstructure Features (39 features)

**Training stats at 07:33 ET**: n=532,085,338 events × 39 features  
**Composition**: 25 smart_v3 event features + 4 PatchTST predictions + 10 derived book-history features  

### Summary (Only CRITICAL/HIGH issues reported)

Per precompute_features_smart_v3.py, Tier 1 (smart_v3) features are pre-normalized **before** training:

| Feature # | Name | Raw → Normalized | Trainer Re-norm | Severity | Notes |
|-----------|------|------------------|-----------------|----------|-------|
| 0 | time_delta_log | clip [0, 8] → /4.0 | z-score (global fold) | LOW | Pre-norm is bounded; trainer z-score is secondary. Fine. |
| 1 | event_type_id | /4.0 | z-score | LOW | Scalar ID proxy; z-score is OK but Mamba v7 will replace with embedding. |
| 2 | side_id | × 2.0 - 1.0 → [-1, +1] | z-score | MEDIUM | Pre-norm is bounded [-1, +1]; z-scoring a ±1 binary signal is odd but consistent. |
| 3 | price_rel_ticks | clip [-50, 50] → /25.0 | z-score | LOW | Tick distance pre-normalized; trainer z-score is fold-relative. OK. |
| 4 | qty_log | (x - 0.693) / 3.0 | z-score | LOW | Log1p of order size; normalization applied pre-trainer. OK. |
| 5 | spread_ticks | clip [0, 20] → /5.0 | z-score | LOW | Bounded pre-norm; trainer z-score is secondary. OK. |
| 6 | cancel_side_asym_50 | clip [-50, 50] → /25.0 | z-score | LOW | Bounded; OK. |
| 7 | rolling_ofi_500 | causal rolling z-score (10k events) | z-score (fold) | **MEDIUM** | **Double z-scoring: once in feature precompute (lines 364), again in trainer.** This flattens the distribution unnecessarily. Should skip trainer z-score for pre-z-scored features. |
| 8 | event_density_20 | clip [0, 4] → /2.0 | z-score | LOW | Bounded; OK. |
| 9 | price_mom_10 | causal rolling z-score (5k events) | z-score (fold) | **MEDIUM** | Double z-scoring issue. |
| 10 | qty_price_mom_50 | causal rolling z-score (10k events) | z-score (fold) | **MEDIUM** | Double z-scoring issue. |
| 11 | price_sign_momentum_200 | /100.0 | z-score | LOW | Bounded momentum; OK. |
| 12 | event_type_entropy_200 | /1.609 (max entropy normalization) | z-score | LOW | Bounded [0, 1.609]; OK. |
| 13 | fill_add_restoration_100 | raw [0, 1] | z-score | LOW | Bounded ratio; OK. |
| 14 | spread_velocity_50 | causal rolling z-score (5k events) | z-score (fold) | **MEDIUM** | Double z-scoring. |
| 15-24 | (rest of smart_v3) | Various (rolling z, clip, ratio) | z-score (fold) | LOW | No critical issues. |
| 25-28 | PatchTST preds + has_pt | fwd-filled, rank-norm on 3 feats | z-score (fold) | LOW | PatchTST predictions are pre-normalized. Trainer z-score is secondary. |
| 29-38 | Book-history derived | rolling mean/std of smart_v3 cols | z-score (fold) | **MEDIUM** | Derived from bounded/normalized inputs; double z-scoring likely. |

**Tier 1 Issues Summary:**
- **3-5 "double z-scoring" features** (rolling_ofi_500, price_mom_10, qty_price_mom_50, spread_velocity_50, possibly book-history features) where precompute applies causal rolling z-score, then trainer applies fold-relative z-score. This squashes variance twice, reducing signal.
- No **CRITICAL** issues for T1 since it's the proven baseline. But double z-scoring should be audited.

---

## Recommendation Matrix

### Severity Definitions
- **CRITICAL**: Information destruction or leakage that will severely degrade generalization
- **HIGH**: Specification mismatch that reduces edge or breaks intended behavior
- **MEDIUM**: Double normalization or range expansion that reduces signal but doesn't break model
- **LOW**: Minor mismatch or edge case

### Recommended Actions for v3.2.1 / v3.3 Retrain

#### Tier 2 (Highest Priority)

| Issue | Recommended Fix | Implementation | Timeline |
|-------|-----------------|----------------|----------|
| Raw counts not log-transformed | Apply `log1p()` in feature builder before z-scoring in trainer, OR z-score `log1p(count)` in trainer | Modify build_v3_2_tier_features.py lines 94, 399, 410, 418 to output log-z instead of raw | Before v3.2.1 retrain |
| signed_volume not sign-preserved | Implement sign-preserving log-z: `sign(x) × log1p(\|x\|)` → z-score magnitude, preserve sign | Modify lines 407 in builder | Before v3.2.1 retrain |
| seconds_since_rth_open not [0,1] normalized | Change to (seconds / 23400) raw in parquet; trainer skips z-score for this feature | Modify line 370 in builder | Before v3.2.1 retrain |
| MFE/MAE/range clipped [0,40] not [0,20] | Revert clip range to [0,20] per design | Modify lines 394-396 in builder | Before v3.2.1 retrain |
| Missing 5 features (microprice_change, avg_spread, etc.) | **Post v3.2.1:** Backfill L2 book snapshots to compute missing features. For v3.2.1, document rationale for substitutions and monitor T2 embedding quality in MLflow. | Separate L2 book reconstruction task (HC #294D future work) | v3.3+ |

#### Tier 3 (CRITICAL PRIORITY)

| Issue | Recommended Fix | Implementation | Timeline |
|-------|-----------------|----------------|----------|
| Distance features z-scored instead of tick-normalized | **REVERT z-scoring in trainer for distance features.** Store as (ticks / 100) raw in parquet, loader skips z-score for distance cols. | Modify train_cnn_mamba_v3_2.py lines 999-1004: exclude cols 0-13 (distances) from z-score | BEFORE next train run |
| Cyclical features (tod_sin, tod_cos, dow_sin) z-scored | **REVERT z-scoring.** Store raw [-1, +1] in parquet, trainer skips z-score. | Modify trainer lines 999-1004: exclude cols 19-21 from z-score | BEFORE next train run |
| Binary flags (lunch_lull, is_close_hour) z-scored | **REVERT z-scoring.** Store raw {0, 1}, trainer skips z-score or learns binary embedding. | Modify trainer lines 999-1004: exclude cols 22, 23 from z-score | BEFORE next train run |
| position_in_value_area z-scored (categorical -1/0/+1) | **REVERT z-scoring.** Store raw {-1, 0, +1}, trainer learns categorical embedding or applies one-hot. | Modify trainer lines 999-1004: exclude col 9 from z-score | BEFORE next train run |
| volume_at_current_price_pctile z-scored (bounded [0,1] percentile) | **REVERT z-scoring.** Store raw [0, 1], trainer skips z-score or treats as bounded. | Modify trainer lines 999-1004: exclude col 10 from z-score | BEFORE next train run |
| Missing dow_cos feature | **ADD dow_cos to parquet and trainer.** Recompute T3 parquets with dow_cos (line 688 in builder already computes it, just not included in output cols). Update TIER3_FEATURE_COLS to include 'dow_cos'. | Add 'dow_cos' to TIER3_FEATURE_COLS (line 135, train_cnn_mamba_v3_2.py) and build_v3_2_tier_features.py line 688 (already computed). Rebuild T3 parquets. | BEFORE next train run |
| Return clip ranges expanded (±1000 vs spec ±200/±500) | Tighten clips to spec: 60s ±200, 5m ±500, 15m ±1000 | Modify lines 652-654 in builder (actually done correctly; trainer z-score is secondary) | LOW priority (already correct in code) |

#### Tier 1

| Issue | Recommended Fix | Implementation | Timeline |
|-------|-----------------|----------------|----------|
| Double z-scoring (rolling_ofi_500, price_mom_10, etc.) | Audit which T1 features are pre-z-scored. For those, skip trainer z-score or apply a lighter regularization (e.g., /std only without subtracting mean). | Flag in trainer: add a list of pre-z-scored feature indices and skip z-score for those. | v3.3+ (low priority, doesn't break baseline) |

---

## Falsification Test Implications

**Current v3.2 status**: Tier 1 (baseline proven), Tier 2 (14/18 features, multiple log-z bugs), Tier 3 (25 features, 9+ critical z-score bugs, 1 missing feature).

**Expected impact on HC #294H falsification gate**:
- If Tier 3 z-score bugs are not fixed before training: model may fail to learn price-level context correctly. Distance-to-high/low will be scrambled by distribution shift between train/OOT folds. Expected **IC degradation vs v3 baseline**.
- If T2 log-z bugs are not fixed: n_trades, trade_volume, signed_volume, n_cancels, n_adds treated as linear counts, not heavy-tail distributions. Expected **loss of OFI signal**. IC may degrade.
- **Recommendation**: Fix all CRITICAL issues in Tier 2 and Tier 3 before v3.2.1 launch. Current launch (if underway) should be halted for feature correction.

---

## Appendix: Feature Normalization Philosophy (Per HC #298)

Correct normalization by category:

| Category | Normalization | Rationale | Examples |
|----------|---|---|---|
| **Bounded continuous (Gaussian)** | z-score plain | Mean-center + scale to unit variance | spread_ticks (pre-bounded), entropy_200 |
| **Heavy-tail counts/volumes** | log-z | Log1p then z-score; handles zeros, power-law tails | n_trades, trade_volume, n_adds (should be in T2) |
| **Price distances (S/R, VWAP, support)** | tick-distance /100 | Absolute reference frame; NOT z-scored | dist_intraday_high, dist_prior_close (ALL T3 distances) |
| **Rank-based (liquidity percentiles)** | rank-norm per-day or bounded raw | Order statistics; z-score invalid | volume_at_current_price_pctile (should be [0,1] raw) |
| **[0,1] ratios (ratios of sums)** | bounded raw, NaN→0.5 | Already normalized; z-score destroys bounds | aggressor_buy_ratio, cancel_add_ratio, fill_add_restoration |
| **Cyclical (time, day)** | sin/cos raw [-1,+1] | Periodic; z-score destroys sinusoid | tod_sin, tod_cos, dow_sin, (missing dow_cos) |
| **Binary / categorical {0,1} or {-1,0,+1}** | one-hot or embedding | Discrete classes; z-score meaningless | is_lunch_lull, position_in_value_area |
| **Return-style (log-return bps)** | clip then z-score | Ratio-scale; z-score appropriate | log_return_in_bucket_bps, log_return_60s_bps |
| **T-statistic / normalized slope** | z-score or as-is | Already normalized; z-score is secondary | trend_strength_5min, mom_divergence |
| **Sign-preserving count imbalance** | log-z sign-preserved | Magnitude matters, direction matters | signed_volume (should apply sign before z) |

---

## Conclusion

**Tier 2** has 5 CRITICAL log-z bugs affecting counts/volumes and 1 CRITICAL issue with signed_volume sign preservation, plus specification shortfall of 4 missing complex features.

**Tier 3** has 9+ CRITICAL z-scoring bugs (distance features, cyclical features, categorical features) and 1 missing feature (dow_cos), destroying the intended absolute price-context representation.

**Tier 1** has 3-5 double z-scoring features that reduce signal but don't break the baseline.

**Recommendation**: **DO NOT train v3.2 until Tier 2 and Tier 3 normalization bugs are fixed.** These are not edge-case issues; they directly harm model learning of price-level context (T3) and order-flow regime (T2). 

If v3.2 training is already underway, **KILL the run and rebuild feature parquets** with corrected normalization. The falsification gate (HC #294H) will likely fail without these fixes.

