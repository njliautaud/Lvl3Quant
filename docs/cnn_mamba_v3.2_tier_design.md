# CNN-Mamba v3.2 — Per-Tier Feature Design

**Status**: ACTIVE DESIGN (2026-05-11 19:45 ET)
**Authorizing HC**: #294 (v3.2 long-context arch) + #295 (proper Tier 2 + Tier 3 engineering, verbatim user grant)
**Supersedes**: lazy bucket-mean Tier 2 + zero-placeholder Tier 3 from first v3.2 launch (MLflow run `72e67852df4d43bb8aa230f375e4d7ad` — KILLED 2026-05-11 ~19:42 ET)

---

## 0. Why three tiers at all

The market operates on multiple timescales simultaneously. A model that only sees one timescale is forced to entangle short-horizon noise with long-horizon regime in a single representation. v3.2 separates them into three parallel Mamba-SSM branches and fuses their embeddings before the heads:

| Tier | Cadence | Window length | What it sees | What it should learn |
|------|---------|---------------|--------------|---------------------|
| 1 | Native MBO event | 1500 events (~30-60s) | Queue dynamics, order arrival/cancel, top-of-book microstructure | "What's the very-short-horizon book pressure?" |
| 2 | 100ms bucket | 1500 buckets (~2.5min) | How order flow EVOLVES at algo cadence | "Is the book absorbing or aggressive? Large prints? Vol expanding?" |
| 3 | 1Hz snapshot | 1500 snaps (~25min) | Session regime, S/R levels, path memory | "Am I extended from VWAP? Near prior-day high? In lunch chop? Trending?" |

**Mamba SSM scales linearly in sequence length**, so 1500 × 3 tiers ≈ 24 GB on the RTX 3090 (verified during first launch — GPU sat at 5.9/24 GB at bs=16, plenty of headroom).

The fused trunk concatenates 3× 64-dim tier embeddings → 192 dim → small MLP → 32 heads.

---

## 1. Tier 1 — Microstructure (KEPT AS-IS for now)

User said: *"Obv tier 1 we have decent"*. Keeping current v3.2 Tier 1 design:

- **39 features per native MBO event**:
  - 25 smart_v3 features (existing — book imbalance, microprice, trade aggression, queue features, etc.)
  - 4 PatchTST predictions (pred_1s, pred_5s, pred_10s, plus a forward-fill indicator), rank-normed, has_pt mask for missing days
  - 10 derived book-history features (rolling 5/20-event imbalance, persistence, queue, pressure, std)
- **Mamba branch**: same backbone as v3 (57 tensors warmstart from `cnn_mamba_v3_smart_v3_fifo/fold_00_best.pt`)
- **Output**: 64-dim embedding

**Future work (not this launch)**: replace the 10 derived rolling-stats features with raw top-10 book snapshot features (HC #294D). That's a separate backfill task.

---

## 2. Tier 2 — Medium-Resolution Order Flow (FULL REDESIGN)

### 2.1 Motivation

Tier 2 sits between native-event microstructure (Tier 1) and session-scale memory (Tier 3). Its cadence (100ms = 10 Hz) matches the timescale at which serious execution algorithms make decisions. A naive `bucket_mean(Tier 1)` loses the within-bucket order-flow dynamics — it averages cancels and adds together, smooths over aggressive prints, hides volatility expansion. We replace it with **engineered order-flow statistics** computed natively at 100ms resolution.

### 2.2 Source data

The same MBO event stream that feeds Tier 1. Each MBO event has:
- `ts_event` (nanosecond ET timestamp)
- `action` (Add / Cancel / Modify / Trade)
- `side` (Bid / Ask)
- `price`, `size`
- `bid_px_00..09`, `bid_sz_00..09`, `ask_px_00..09`, `ask_sz_00..09` (top-10 book snapshot at event time)

For each 100ms bucket aligned to RTH (9:30:00.000 → 9:30:00.100 → …), aggregate events with timestamp `[bucket_start, bucket_start + 100ms)`.

### 2.3 Window & history

- **Window**: 1500 buckets × 100ms = 150 seconds = 2.5 minutes of past order flow (strict <T)
- **Lookback for z-score baselines**: rolling 20 trading days (~2.34 M buckets) for log-z stats; rolling 5 days for the large-order percentile threshold
- **Strict less-than**: at evaluation time T, buckets are `[T-2.5min, T)` — bucket containing T itself excluded to prevent leakage

### 2.4 Feature list (18 features per bucket)

#### A. Price action (3)

| # | Feature | Formula | Range | Normalization |
|---|---------|---------|-------|---------------|
| 1 | `log_return_in_bucket` | `log(close_micro/open_micro)` | typically ±0.0001 | multiply by 10000 → bps; clip ±20 |
| 2 | `bucket_mfe_ticks` | `(bucket_high - bucket_open) / 0.25` | [0, ∞) | clip [0, 20] |
| 3 | `bucket_mae_ticks` | `(bucket_open - bucket_low) / 0.25` | [0, ∞) | clip [0, 20] |

Open/close = microprice (bid×ask_size + ask×bid_size)/(bid_size+ask_size) at first/last book update in bucket. High/low = max/min of microprice across all updates in bucket.

#### B. Trade flow (3)

| # | Feature | Formula | Normalization |
|---|---------|---------|---------------|
| 4 | `n_trades_log_z` | `log1p(n_trades_in_bucket)` then z-score vs rolling 20d | clip ±5 |
| 5 | `volume_log_z` | `log1p(sum(trade_size))` then z-score vs rolling 20d | clip ±5 |
| 6 | `aggressor_buy_ratio` | `buyer_initiated_volume / total_volume`; if `total_volume == 0` → 0.5 | already ∈[0,1] |

Aggressor side: use Lee-Ready or trade-at-ask = buy / trade-at-bid = sell. (Lee-Ready preferred — handles mid-quote trades.)

#### C. Order flow imbalance (2)

| # | Feature | Formula | Normalization |
|---|---------|---------|---------------|
| 7 | `signed_volume_log_z` | `sign(buy_vol - sell_vol) × log1p(|buy_vol - sell_vol|)`, z-scored vs 20d magnitude distribution but sign preserved | clip ±5 |
| 8 | `microprice_change_ticks` | `(close_micro - open_micro) / 0.25` | clip ±5 |

#### D. Book dynamics (5)

| # | Feature | Formula | Normalization |
|---|---------|---------|---------------|
| 9 | `n_tob_changes_z` | count of top-of-book level changes in bucket, z-scored | clip ±5 |
| 10 | `avg_spread_ticks` | mean `(ask_px_00 - bid_px_00) / 0.25` across event ticks in bucket | clip [0, 10] |
| 11 | `n_cancels_log_z` | `log1p(n_cancel_events)`, z-scored vs 20d | clip ±5 |
| 12 | `n_adds_log_z` | `log1p(n_add_events)`, z-scored vs 20d | clip ±5 |
| 13 | `cancel_add_ratio` | `n_cancels / (n_cancels + n_adds)`; if both 0 → 0.5 | already ∈[0,1] |

#### E. Size regime (2)

| # | Feature | Formula | Normalization |
|---|---------|---------|---------------|
| 14 | `large_order_count_log_z` | count of orders with size > 95th-percentile of rolling 5d add-order sizes, `log1p`, z-scored | clip ±5 |
| 15 | `avg_top5_depth_log_z` | mean of `(sum(bid_sz_00..04) + sum(ask_sz_00..04))/10` across event ticks in bucket, `log1p`, z-scored | clip ±5 |

#### F. Volatility (3) — note: 3 not 2 (added bucket_volume separately to capture trade-side; vol features are 2)

Wait — recount. F is 2 features as designed. Total: 3+3+2+5+2+2 = 17. Add one more:

| # | Feature | Formula | Normalization |
|---|---------|---------|---------------|
| 16 | `rolling_5bucket_vol_z` | std of `log_return_in_bucket` over last 5 buckets, z-scored vs rolling 20d distribution | clip ±5 |
| 17 | `bucket_range_ticks` | `(bucket_high - bucket_low) / 0.25` | clip [0, 20] |
| 18 | `seconds_since_session_open` | normalized [0, 1] across RTH | already bounded |

Total: **18 features**. Good.

### 2.5 What the Mamba branch should learn

The branch trains end-to-end on the alpha-first heads. Inductive bias from features:

- **Absorption regime**: high `n_cancels_log_z` + flat `microprice_change_ticks` + low `signed_volume_log_z` → passive liquidity is absorbing aggressive prints → reversion likely
- **Aggressive regime**: high `signed_volume_log_z` + matching-sign `microprice_change_ticks` + high `n_trades_log_z` + low `avg_spread_ticks` (tightening) → trend continuation likely
- **Vol expansion**: rising `rolling_5bucket_vol_z` + widening `bucket_range_ticks` → larger expected MFE/MAE; sizing/bracket should adapt
- **Liquidity shock**: spike in `large_order_count_log_z` + jump in `avg_spread_ticks` → uncertain, lower confidence

### 2.6 Output of the branch

Mamba SSM with 4 layers, d_model=128, → final-token pooling → linear → **64-dim Tier 2 embedding**.

---

## 3. Tier 3 — Session-Scale Memory (FULL REDESIGN — replacing zero placeholders)

### 3.1 Motivation

Without session-scale context, the model can't know whether the current 30s of microstructure is happening at the prior-day high (likely rejection) vs deep inside the value area (likely continuation). Tier 3 gives the model **price-path memory** + **level context** + **session regime**.

This is the tier the user specifically named in HC #294: *"s r levels prior session high low accumulation and distribution levels order book features alongside our smart v3 or v4"*.

### 3.2 Source data

- Prior 5 sessions of daily MBO data (for prior-session H/L/C/VWAP/VPOC + 5d extremes)
- Current intraday MBO data, cumulative from RTH open to strict-less-than-T (for intraday H/L, VWAP, volume profile)
- Calendar (for time-of-day, day-of-week)

### 3.3 Window & history

- **Window**: 1500 snapshots × 1Hz = 1500 seconds = 25 minutes of past 1-second snapshots (strict <T)
- **History needed**:
  - 20 prior trading days for vol/trend z-score baselines
  - 5 prior trading days for `dist_5d_extreme`
  - 1 prior trading day (CLOSED, fully aggregated) for `prior_session_*` features
  - Current intraday session cumulative for `intraday_*` features

### 3.4 Feature list (24 features per 1-second snapshot)

#### A. Price location / Support-Resistance (5)

All distances are in ticks, clipped to ±200, divided by 100 → range roughly [-2, +2].

| # | Feature | Reference | Why it matters |
|---|---------|-----------|----------------|
| 1 | `dist_intraday_high_ticks_norm` | (current_mid - intraday_high) / 0.25 | extension from session high — fade target |
| 2 | `dist_intraday_low_ticks_norm` | (current_mid - intraday_low) / 0.25 | extension from session low — bounce target |
| 3 | `dist_session_vwap_ticks_norm` | (current_mid - session_VWAP) / 0.25 | mean-reversion anchor |
| 4 | `dist_prior_session_close_ticks_norm` | (current_mid - prior_close) / 0.25 | overnight gap context |
| 5 | `dist_prior_session_vwap_ticks_norm` | (current_mid - prior_VWAP) / 0.25 | prior-day fair-value anchor |

`current_mid` = (best_bid + best_ask) / 2 at snapshot time.

#### B. Volume profile (5)

Volume profile = histogram of cumulative traded volume by price level. Built on a per-session basis with bin width = 1 tick (0.25 points).

| # | Feature | Definition | Normalization |
|---|---------|------------|---------------|
| 6 | `dist_intraday_vpoc_ticks_norm` | dist to intraday VPOC (highest-volume price bin so far today) | clip ±200, /100 |
| 7 | `dist_intraday_vah_ticks_norm` | dist to intraday VAH (top edge of 70% value area) | clip ±200, /100 |
| 8 | `dist_intraday_val_ticks_norm` | dist to intraday VAL (bottom edge of 70% value area) | clip ±200, /100 |
| 9 | `position_in_value_area` | -1 if below VAL, +1 if above VAH, 0 if inside | discrete {-1, 0, +1} |
| 10 | `volume_at_current_price_pctile` | percentile rank of cumulative volume at current 1-tick bin within today's distribution | ∈[0, 1] |

#### C. Prior session levels (4)

| # | Feature | Definition | Normalization |
|---|---------|------------|---------------|
| 11 | `dist_prior_session_high_ticks_norm` | dist to prior session high | clip ±200, /100 |
| 12 | `dist_prior_session_low_ticks_norm` | dist to prior session low | clip ±200, /100 |
| 13 | `dist_prior_session_vpoc_ticks_norm` | dist to prior session VPOC | clip ±200, /100 |
| 14 | `dist_5d_extreme_ticks_norm` | signed distance to nearest of {5d-high, 5d-low}; positive if current is above 5d-high or below 5d-low, negative if inside the 5d range, magnitude = ticks to nearest extreme | clip ±200, /100 |

#### D. Path memory (5)

The Mamba branch can in principle learn returns from raw mid prices, but providing them explicitly as features both shortens training and lets the same features participate in the engineered-feature stack the model fuses.

| # | Feature | Formula | Normalization |
|---|---------|---------|---------------|
| 15 | `log_return_60s` | log(mid_T / mid_{T-60s}) | × 10000 → bps; clip ±200 |
| 16 | `log_return_5min` | log(mid_T / mid_{T-5min}) | × 10000 → bps; clip ±500 |
| 17 | `log_return_15min` | log(mid_T / mid_{T-15min}) | × 10000 → bps; clip ±1000 |
| 18 | `realized_vol_5min_z` | std of 1-second log returns over last 5min, z-scored vs rolling 20d of same statistic | clip ±5 |
| 19 | `trend_strength_5min_z` | slope of OLS(log_mid_t vs t) over last 5min × duration, z-scored vs 20d | clip ±5 |

#### E. Regime / time (5)

| # | Feature | Formula | Range |
|---|---------|---------|-------|
| 20 | `tod_sin` | sin(2π × seconds_in_RTH / 23400) | [-1, +1] |
| 21 | `tod_cos` | cos(2π × seconds_in_RTH / 23400) | [-1, +1] |
| 22 | `is_lunch_lull` | 1 if 11:30 ≤ ET ≤ 13:30 else 0 | {0, 1} |
| 23 | `is_close_hour` | 1 if last 30 min of RTH else 0 | {0, 1} |
| 24 | `dow_sin` | sin(2π × dow_index / 5) where Mon=0, Fri=4 | [-1, +1] |

(Could add `dow_cos` for symmetry but day-of-week cyclicality is loose; one sinusoid sufficient.)

### 3.5 What the Mamba branch should learn

- **Extension fade**: large `|dist_intraday_high|` or `|dist_intraday_low|` + opposing `trend_strength_5min` → reversion edge
- **Trend continuation**: `position_in_value_area = +1` + `trend_strength_5min_z > 1` + `dist_5d_extreme_ticks_norm > 0` (breakout above 5d high) → momentum continuation
- **Chop avoidance**: `is_lunch_lull = 1` + `realized_vol_5min_z < -0.5` + `|dist_session_vwap| < 0.1` → low-edge regime, downstream strategy should reduce confidence threshold
- **Gap fill**: large `dist_prior_session_close` early in session + opposing `log_return_5min` → mean-revert toward prior close

### 3.6 Output of the branch

Mamba SSM with 4 layers, d_model=128, → final-token pooling → linear → **64-dim Tier 3 embedding**.

---

## 4. Normalization discipline (ALL tiers)

### 4.1 Rolling statistics

All z-scores use a **rolling 20 trading day** window of the same statistic, computed ONLY from train-fold dates. Per fold:

1. At fold setup, identify the 60 train days
2. For each feature requiring z-scoring, compute the per-day mean and std over the train window
3. Save to `feature_stats_fold_NN.json`
4. Apply the same mean/std unchanged to OOT days

This prevents leakage — OOT-day statistics never influence the normalization constants.

### 4.2 Clipping

Every normalized feature is clipped to a documented range (typically ±5 for z-scores, ±2 for distance/100, [0, 20] for tick ranges). Hard clip, not winsorization — keeps gradient stable.

### 4.3 NaN/Inf handling

- Distance features when reference is undefined (e.g., prior_session_VPOC if prior session had no trades) → **0** (model interprets as "no level")
- Ratio features when denominator is 0 → **0.5** (model interprets as "neutral")
- Log-z when underlying count is 0 → **z = (log1p(0) - mean) / std = -mean/std** (negative → "below average")
- Any unexpected NaN/Inf → log warning + replace with 0; raise alert if rate >0.1% of rows

### 4.4 Per-fold feature stats

Each fold gets its own `feature_stats.json`. Format:
```json
{
  "fold_id": 0,
  "train_dates": ["2025-12-22", ..., "2026-02-17"],
  "tier1": {"feature_name": {"mean": ..., "std": ...}, ...},
  "tier2": {"feature_name": {"mean": ..., "std": ...}, ...},
  "tier3": {"feature_name": {"mean": ..., "std": ...}, ...}
}
```

---

## 5. Output schema (parquet layout)

### 5.1 Tier 2 parquet

Path: `/home/jupiter/Lvl3Quant/data/derived/tier2_orderflow_features_v1.parquet/date=YYYY-MM-DD/part-*.parquet`

Schema:
- `ts_event` (int64 ns UTC)
- `date` (string YYYY-MM-DD ET)
- `bucket_idx` (int32, 0-indexed within RTH)
- 18 feature columns (float32)

Rows per day: 6.5h × 3600s × 10 buckets/s = **234,000 rows/day**.

### 5.2 Tier 3 parquet

Path: `/home/jupiter/Lvl3Quant/data/derived/tier3_session_features_v1.parquet/date=YYYY-MM-DD/part-*.parquet`

Schema:
- `ts_event` (int64 ns UTC)
- `date` (string YYYY-MM-DD ET)
- `second_idx` (int32, 0-indexed within RTH)
- 24 feature columns (float32)

Rows per day: 23,400 (1/sec × RTH).

### 5.3 Coverage

- Backfill: 2025-12-01 → today (2026-05-11). ~110 trading days.
- Storage estimate: Tier 2 ~ 110 × 234k × 18 × 4B ≈ 1.85 GB; Tier 3 ~ 110 × 23.4k × 24 × 4B ≈ 0.25 GB. Both negligible vs MBO event data.

---

## 6. Dataloader changes (v3.2)

The current dataloader builds Tier 2 by bucket-mean of Tier 1 and Tier 3 from zeros. Replace with:

```python
def get_tier2(self, ts_event_T, lookback=1500):
    """Load 1500 most recent 100ms buckets strictly before ts_event_T."""
    date_T = pd.Timestamp(ts_event_T, unit='ns', tz='UTC').tz_convert('America/New_York').date()
    df = pq.read_table(
        TIER2_PARQUET_ROOT,
        filters=[('date', '=', str(date_T))],
        columns=['ts_event'] + TIER2_FEATURE_COLS
    ).to_pandas()
    df = df[df['ts_event'] < ts_event_T]
    df = df.iloc[-lookback:]
    if len(df) < lookback:
        # pad front with zeros (early in session)
        pad = lookback - len(df)
        padding = np.zeros((pad, len(TIER2_FEATURE_COLS)), dtype=np.float32)
        return np.concatenate([padding, df[TIER2_FEATURE_COLS].values.astype(np.float32)])
    return df[TIER2_FEATURE_COLS].values.astype(np.float32)
```

Same pattern for Tier 3 with 1500 × 1s lookback.

Fallback: if parquet missing for `date_T`, log once and return zeros (model degrades gracefully but warning surfaces).

---

## 7. Falsification gate (unchanged from HC #294H)

v3.2 Ep 1 OOT must show at least ONE of:
1. ≥+0.01 IC improvement on any of {IC_5s, IC_10s, IC_30s} vs v3 baseline (0.128 / 0.089 / TBD)
2. ≥+0.05 Spearman correlation between predicted vs realized MFE/MAE at 30s horizon
3. Non-degenerate quantile calibration on log_ret_10s quantiles (10/50/90 within ±5% of empirical)

Else: KILL at Ep 1, postmortem, decide whether to revert architecture changes.

---

## 8. Implementation plan

1. **Feature builder** (`alpha_discovery/features/build_v3_2_tier_features.py`):
   - Reads MBO parquets day-by-day in time order (must process days sequentially for prior-day state)
   - Per day: computes Tier 2 (vectorized over 100ms buckets) and Tier 3 (vectorized over 1Hz snapshots)
   - Writes daily parquet partitions
   - Parallelism: at the bucket-compute level within a day, NOT across days (dependencies on prior-day state)
   - Per-fold `feature_stats.json` computed in a separate pass once all daily parquets exist

2. **Dataloader patch** (`alpha_discovery/deep_models/train_cnn_mamba_v3_2.py`):
   - Replace `_build_tier2_from_tier1_bucket_mean` with `_load_tier2_from_parquet`
   - Replace `_build_tier3_zeros` with `_load_tier3_from_parquet`
   - Apply per-fold `feature_stats.json` normalization on the fly (already-normalized values in parquet × per-fold rescale is fine; or pre-apply normalization at parquet build time and just load — simpler and faster)

3. **Per-fold feature stats**: post-process pass over the daily parquets, computes per-feature mean/std on each fold's train window, writes JSON to fold output dir

4. **Verification**:
   - For one validation day (e.g., 2026-04-15), assert: all 18 Tier 2 features have non-zero variance, no Inf/NaN, distributions roughly look like documented (e.g., `tod_sin` traces a sinusoid)
   - For Tier 3: same checks + spot-check `dist_intraday_vpoc` matches a manual computation

5. **Relaunch v3.2**:
   - Same fold structure, same warmstart from v3 fold_00_best.pt
   - New MLflow experiment name suffix: `_real_features`
   - Verify Ep 1 batch losses within 5 min of launch (should look similar to first launch — Tier 1 is identical; Tier 2 + 3 are new but small fraction of total params)

---

## 9. Open questions for v3.3+ (NOT this launch)

- Replace Tier 1 derived rolling-stats with raw top-10 book snapshots (HC #294D)
- Add 60s and 5min direction labels to alpha_labels so the long-horizon heads get gradient (currently masked)
- Consider a 4th tier at 30s cadence covering 12h (overnight session memory)
- Volume-profile features could be extended to multi-session (full week, full month) — but diminishing returns vs prior-day already captured

---

## 10. Sign-off

This design captures the engineering reasoning the user explicitly requested in HC #295. All 24 Tier 3 features and 18 Tier 2 features have documented purpose, source data, normalization, history, and model interpretation. No zero placeholders. No bucket-mean lazy proxies. No leakage paths.

Implementation begins after this doc is committed.
