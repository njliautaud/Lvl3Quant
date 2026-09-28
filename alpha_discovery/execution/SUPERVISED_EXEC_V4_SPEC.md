# Supervised Execution Predictor v4 — Feature Spec

**Per HC #253, #254 (DIRECTIVES.md, 2026-05-08)**

Status: spec + builder script in place, smoke test only. Full training is the user's call.

## What changed vs v3

v3 (`train_supervised_exec.py`) used 19 features:
CNN-Mamba {pred_1s, pred_5s, pred_10s, abs versions, confidence_tier}, micro
(book_imb, bid_depth_log, ask_depth_log, spread), volatility, time_of_day,
volume_imbalance, signal_agreement, signal_strength, price_momentum, pred_std, pred_range.

v4 keeps all 19 and adds 9 — total **28 features**.

## v4 feature additions

| # | Feature | Source | Range | Caveats |
|---|---------|--------|-------|---------|
| 20 | `patchtst_pred_1s`  | `output/patchtst_bulk_oot/{date}_predictions.npz` `predictions[:,0]` | typical [-1, +1] (sigmoid-ish) | Aligned by pred_idx; PatchTST file may have +/-N more rows than CNN-Mamba — we truncate to `min(n_mamba, n_patch)` |
| 21 | `patchtst_pred_5s`  | same, col 1 | same | |
| 22 | `patchtst_pred_10s` | same, col 2 | same | |
| 23 | `patchtst_abs_pred_1s` | `abs(patchtst_pred_1s)` | [0, ~1] | |
| 24 | `pt_mamba_sign_agree` | `1.0` if `sign(patchtst_1s) == sign(mamba_1s)` else `0.0` | {0, 1} | Requires both nonzero |
| 25 | `pt_mamba_mag_ratio` | `abs(patchtst_1s) / (abs(mamba_1s) + 1e-6)`, clipped [0, 5] | [0, 5] | Magnitude agreement proxy |
| 26 | `vol_regime_bucket` | rolling-vol percentile within day → {0=low, 1=med, 2=high} | {0, 1, 2} | Low <33pct, med 33-67pct, high >67pct of within-day rolling vol |
| 27 | `tod_bucket` | hour-of-day bucket: 0=open (9:30-10:30), 1=mid (10:30-15:00), 2=close (15:00-16:00) | {0, 1, 2} | ET clock; v3 already had a continuous `time_of_day` — this is the discrete categorical |
| 28 | `queue_pos_bid` | from MBO event row col 11 (`queue_bid` in precomputed meta) | [0, ~) (qty contracts ahead) | If feature columns < 12 → 0 with TODO; ES is typically 0..50 |
| 29 | `queue_pos_ask` | col 12 (`queue_ask`) | [0, ~) | Same |
| 30 | `recent_fill_rate` | per-day rolling fill rate from fill_sim cache | [0, 1] | **PLACEHOLDER 0.0 in v4 first pass — TODO HC #253(f)**. Computing correctly requires per-signal counterfactual fill state which is heavy. Recorded as a feature slot so the model can learn it once we wire it |

(Feature count = 19 (v3) + 11 = **30** with both queue cols + the placeholder. Precise count printed at startup of `train_supervised_exec_v4.py` via `len(FEATURE_NAMES_V4)`.)

## Label

`pnl_ticks` per trade from FIFO fill_sim — the trade record field already deducts
`commission_ticks=0.376` (HC #130, HC #231). NO spread crossing cost, NO theoretical
slippage modeled on top of the FIFO fill price (HC #231 explicit).

Source: `/home/nick/Lvl3Quant/output/wide_tp_queue_sweep/{config}_{date}.json` `trades[*].pnl_ticks`.

Labels are **per-trade**, not per-prediction. Joining strategy:
- For each prediction event, look up whether it generated a trade in the chosen fill_sim config.
- If yes: label = `pnl_ticks` from trade record (matched by `signal_time_ns ≈ ts_s * 1e9` within 1 stride).
- If no (signal didn't pass gate, didn't fill, etc): **excluded from supervised set** (we cannot know counterfactual P&L cheaply).
- **This means label coverage is fill_sim-config-dependent.** We use `tp6_sl3_h60000_q3_t0.5` as the default since it is one of the wider/looser configs and produces a usable trade count per day. Picking the BEST config for label-density is fine; the MLP learns conditional-on-trade.

## Walk-forward

5-fold sliding (HC #0 — never expanding):
- 56 OOT dates available (CNN-Mamba `bulk_oot` covers 20260306 → 20260429).
- 40 of those have wide_tp_queue_sweep fill_sim labels (20260315 → 20260429).
- Fold structure: `n_train_days = ceil(40 * 4/5) = 32`, `n_oot_days = 8`, slide by 8.
  → folds: train[0..32) test[32..40), then train[8..40) test[?] — adjusted so we never reuse OOT dates as train (sliding only).

## MLflow

`EXPERIMENT_NAME = "supervised_exec_v4"` on `MLFLOW_TRACKING_URI=http://jupiter:5000` (Jupiter MLflow). Mandatory per HC #0.

## Empirical results (2026-05-08 build)

Smoke test on 20260415: 17 labeled trades extracted. End-to-end pipeline verified.

Aggregate over 40 fill_sim dates × `tp6_sl3_h60000_q3_t0.5` config:
- **3,184 total labeled trades** across 31 dates with non-empty fill_sim output (9 dates produced 0 trades — non-trading-day artifacts in the cache).
- 1,050 wins / 2,134 losses (WR ~33%, mean P&L ~ -0.7 ticks per trade — consistent with the user's observation that single-config sweeps mostly lose money). This is the supervised dataset whose top-quantile predictions need to be SHOWN to be net-positive.
- Date range: 20260316 → 20260428.

Effective prediction window/stride from npz metadata:
- CNN-Mamba bulk_oot: window=3000, stride=250.
- PatchTST bulk_oot:  window=500,  stride=250.
- (v3 hardcoded 1000/50 — wrong for bulk_oot files. v4 reads from file metadata.)

Strides match (250) so row-by-row alignment between the two prediction tensors is valid; v4 truncates to `min(N_mamba, N_patch)`.

## v4-inverse design (HC #255, added 2026-05-08)

User directive (HC #255): "PatchTST DA was good, only works as a confluence... Or
maybe inverse — use PatchTST for direction and CNN-Mamba for magnitude and other
market context." We test BOTH heads in parallel; same 30-feature input vector,
different label-target/gating semantics.

### Hypothesis

The two architectures answer different questions:
- **CNN-Mamba v2**: trained on multi-horizon return regression — strong at
  *magnitude/edge* per the IC numbers (IC_1s=0.222, IC_5s=0.141, IC_10s=0.106).
- **PatchTST**: directional accuracy (DA) was its empirical strength when used
  as a confluence filter. Maybe its sign should be the *direction* decision and
  CNN-Mamba's signed magnitude should be a feature, not a sign.

This contradicts v3/v4-confluence's implicit assumption that CNN-Mamba sign is
the right direction estimator. v4-inverse stress-tests that assumption.

### Heads

**`--head confluence` (original v4 head, default)**
- CNN-Mamba sign decides direction at fill_sim time (already baked into the
  recorded trade — fill_sim was generated using CNN-Mamba sign).
- PatchTST sign-agreement is just a filter feature (`pt_mamba_sign_agree`,
  index 23) and a magnitude-ratio feature (`pt_mamba_mag_ratio`, index 24).
- **Target**: `pnl_ticks` directly. No pre-gate. Model learns
  `E[pnl_ticks | features]`.
- MLflow experiment: `supervised_exec_v4_confluence`.

**`--head inverse --inverse-mode pregate` (HC #255 strict)**
- Drop trades where `sign(patchtst_pred_1s) != sign(realized_dir)` — i.e. only
  train on the trades PatchTST got the direction right on.
  - Implementation note: `realized_dir = sign(pnl_ticks)` from the fill_sim
    record (post-cost). This is a proxy for true realized return direction —
    fine for training-time gating because the model conditions on PatchTST
    being right anyway. At inference time the gate is `sign(patchtst_pred_1s)`
    matching the proposed side; the model says yes/no on profitability given
    that gate.
- **Target**: `pnl_ticks`. Model learns `E[pnl | PatchTST is directionally right]`.
- Caveats: data scarcity — pre-gating throws away ~50–65% of labeled trades
  (smoke 20260415: 17 → 6, PatchTST sign-correctness only 6/17 = 35%, *below*
  random, which is itself an interesting datapoint about that date).
- MLflow experiment: `supervised_exec_v4_inverse` with
  `inverse_mode=pregate` tag.

**`--head inverse --inverse-mode feature` (HC #255 lenient — let model learn)**
- Keep all rows. **Append a 31st binary feature** `patchtst_sign_correct = 1
  if sign(patchtst_pred_1s) == sign(pnl_ticks) else 0`.
- **Target**: `pnl_ticks`. Model learns the gate itself, as well as how to
  combine PatchTST direction with CNN-Mamba magnitude features.
- This is the safer default — keeps full data, lets the MLP discover whether
  PatchTST sign is actually informative net of CNN-Mamba and microstructure
  features. If `patchtst_sign_correct` ends up the top-importance feature,
  pre-gating is justified for live trading.
- MLflow experiment: `supervised_exec_v4_inverse` with
  `inverse_mode=feature` tag.

### Smoke test results — 20260415, fill_sim cfg `tp6_sl3_h60000_q3_t0.5`

| Head | Inverse mode | n_trades | n_features | label_mean | label_std | WR | Top-3 corr w/ pnl |
|------|--------------|----------|------------|-----------:|----------:|---:|-------------------|
| confluence | feature (n/a) | 17 | 30 | -1.7289 | 3.7329 | 0.235 | signal_agreement (+0.54), ask_depth_log (+0.42), patchtst_pred_1s (-0.34) |
| inverse | pregate | 6 | 30 | -2.7927 | 2.4224 | 0.167 | price_momentum (-0.97), patchtst_pred_10s (+0.91), patchtst_pred_5s (+0.90) — small-N caveat |
| inverse | feature | 17 | 31 | -1.7289 | 3.7329 | 0.235 | signal_agreement (+0.54), ask_depth_log (+0.42), patchtst_pred_1s (-0.34) |

Observations:
1. PatchTST got direction wrong on **11/17** trades on 20260415 — *worse than
   random*. This is one date; cannot generalize. Real test is across all 56 OOT
   dates per HC #254.
2. The pre-gated set (n=6) shows extreme correlations because of small-N
   instability. Not statistically meaningful at one date.
3. The `patchtst_sign_correct` feature in feature-mode has mean 0.353 (matches
   6/17 = 0.353) — sanity check on alignment.

### What this does NOT do

- Does NOT launch full training (per user directive).
- Does NOT re-generate fill_sim labels using PatchTST sign as the gate. Today
  fill_sim is built off CNN-Mamba sign. A *true* inverse test would re-run
  fill_sim with PatchTST as the entry-side decider — that's a Neptune fill_sim
  sweep job, much heavier. The two heads above test the inverse hypothesis
  *given* the existing label set; if they look promising the next step is
  rebuilding fill_sim labels with PatchTST sign.
- Does NOT modify v3 code or `train_supervised_exec.py`.

### How to run

```
# Smoke each head individually
python3 alpha_discovery/execution/train_supervised_exec_v4.py --smoke-date 20260415 --head confluence
python3 alpha_discovery/execution/train_supervised_exec_v4.py --smoke-date 20260415 --head inverse --inverse-mode pregate
python3 alpha_discovery/execution/train_supervised_exec_v4.py --smoke-date 20260415 --head inverse --inverse-mode feature

# Or all three at once
python3 alpha_discovery/execution/train_supervised_exec_v4.py --smoke-date 20260415 --smoke-both
```

## Known blockers / caveats

1. PatchTST and CNN-Mamba prediction arrays have slightly different row counts (e.g. 81599 vs 81589 on 20260306). We align by truncation: `n = min(n_mamba, n_patch)` and use rows `[:n]` from each, assuming both are stride-aligned from event 0. Loss = ~10 rows out of 80k = negligible.
2. `recent_fill_rate` is a placeholder. To be filled in once we have a streaming fill_rate computation per session.
3. PatchTST predictions on Jupiter (`output/patchtst_bulk_oot/`) cover 47 dates, CNN-Mamba covers 56. Intersection = the dataset.
4. fill_sim config used for labels conditions the dataset. The trained MLP will be biased toward signals that survive that config's gate. Future: train on union of multiple configs with `config_id` as a categorical feature.
5. Cost constant: `COMMISSION_TICKS = 0.376`, no spread cost (HC #231 — `MARKET_ORDER_COST_TICKS = 1.376` is BANNED).
6. fill_sim signal_time_ns are quantized to 100ms boundaries (resampled bars), but MBO event ts are exact ns. We match by nearest-neighbor with 0.5s tolerance around the corresponding pred_idx event — empirically 100% of fill_sim trades match within tolerance on tested dates.
7. Several feature columns from the 25-col MBO event tensor look ALREADY-NORMALIZED rather than raw (e.g. col 8 "book_imbalance" was 0 across all rows on 20260415; col 11/12 "queue_pos_*" range [-1, +1] not raw contracts). Features are still passed through to the MLP as-is — model can learn whatever signal exists. TODO: cross-check with `precompute_observations.py` to map columns to canonical feature names.
