# HC #424 §3 Fallback — v3.3 fold_00 Execution-Gate Verdict (2026-05-18)

## Source

- **v3.3 NPZ**: `/home/jupiter/Lvl3Quant/output/hc424_jupiter_exec_research/inputs/v3_3_fold00/fold_00_predictions.npz`
- **SHA256 (v3.3)**: `b455f30373555ae71d991a8962d9f307dadba4c60ecd22c23eb723d9e38e06b0`
- **Origin (v3.3)**: already on Jupiter at `output/cnn_mamba_v3_3_uncertainty_weighted/fold_00_predictions.npz` — **no SCP needed**, Neptune untouched.
- **v3.4.2 ep1 NPZ**: `/home/jupiter/Lvl3Quant/output/hc424_jupiter_exec_research/inputs/fold_00_ep1_oot.npz`
- **SHA256 (v3.4.2 ep1)**: `8cf4d7d1c479cef07cd60720ceb2cc0ee682e3cd37c21cddb631695fa33fc994`
- **n_samples**: 241,351 across 20260223–20260227 (same 5-day OOT window as v3.4.2 ep3)
- **MLflow experiment**: `hc424_jupiter_exec_research_v33` (id 130480352858023723)
  - v3.3 fold_00 run: `8f516457a64c4e7a84570f7df5a0a3fc`
  - v3.4.2 ep1 run: `f47f0691f5384156a1ffd11dea596d82`

## IC Verification (v3.3 fold_00)
| Horizon | Measured | CLAUDE.md champion claim | Delta |
|---|---|---|---|
| IC_1s  | **0.2625** | 0.222 | +0.040 (better) |
| IC_5s  | **0.1303** | 0.141 | -0.011 (slightly under) |
| IC_10s | **0.0865** | 0.106 | -0.020 (under) |

v3.3 fold_00 numbers are close to the champion claim. IC_1s is actually higher than the documented 0.222. Verdict valid.

## Methodology (identical to v3.4.2 ep3 sub-agent — HC #424 §3)
- 30-head multi-head LGBM (HC #422 R8 / HC #423 §4 feature set)
- Single-head LGBM baseline (`pred_log_ret_1s` only)
- Raw signal baseline (HC #74 rules)
- HC #74 FIFO market replay labels — `tp4sl3_long/short_net_ticks`, per-day NPZ in `data/processed/mbo_events_smart_v3_fifo_labels/`
- HC #392 cost — passive limit = 0.376 ticks (commission only); market = 1.376
- HC #344 day_conc gate — pass if max-day < 0.50
- HC #0 sliding + HC #393 holdout — per-day last 20%
- Train: 193,079 rows. Holdout: 48,272 rows.
- HC #397B canonical columns reported on holdout.

## Canonical Replay Verdict — Long Side (passive limit, FIFO-realized fills)

### v3.3 fold_00 — best LONG slices
| Method | Top % | n_fills | tpf | Sharpe√N | Sortino√N | PF | WR | day_conc | pass_hc344 |
|---|---|---|---|---|---|---|---|---|---|
| raw_signal_long | 1% | 186 | **-0.892** | -3.71 | -13.18 | 0.57 | 38.7% | 0.371 | pass |
| raw_signal_long | 2% | 386 | -0.975 | -5.92 | -18.57 | 0.54 | 36.5% | 0.358 | pass |
| singlehead_lgbm_long | 1% | 188 | -1.406 | -6.23 | -19.60 | 0.40 | 30.3% | 0.250 | pass |
| multihead_lgbm_long | 1% | 287 | -1.435 | -8.95 | -17.60 | 0.32 | 27.9% | 0.969 | FAIL |
| rules_baseline_long | thr=0 | 8961 | -0.856 | -25.49 | -71.85 | 0.57 | 38.8% | 0.257 | pass |

### v3.3 fold_00 — best SHORT slices
| Method | Top % | n_fills | tpf | Sharpe√N | Sortino√N | PF | WR | day_conc | pass_hc344 |
|---|---|---|---|---|---|---|---|---|---|
| **raw_signal_short** | **1%** | **190** | **-0.534** | **-2.29** | **-6.67** | **0.70** | **45.3%** | 0.421 | **pass** |
| singlehead_lgbm_short | 1% | 227 | -0.827 | -3.65 | -7.56 | 0.59 | 40.1% | 0.300 | pass |
| singlehead_lgbm_short | 2% | 468 | -0.828 | -5.33 | -12.14 | 0.59 | 39.7% | 0.314 | pass |
| raw_signal_short | 2% | 379 | -0.868 | -5.30 | -15.91 | 0.57 | 39.1% | 0.383 | pass |
| multihead_lgbm_short | 1% | 944 | -1.612 | -15.58 | -47.64 | 0.36 | 30.2% | 0.809 | FAIL |
| rules_baseline_short | thr=0 | 9250 | -0.948 | -28.50 | -72.79 | 0.54 | 37.8% | 0.320 | pass |

Best deployable-looking slice (v3.3 fold_00 raw_signal_short top-1%): **tpf=-0.534, PF=0.70, WR=45.3%**.

## Threshold-Gate Behaviour (multi/single LGBM thr > 0)

For all three NPZs (v3.3, v3.4.2 ep1, v3.4.2 ep3), thresholds {0.0, 0.5, 1.0} produce **0 fills** because the LGBM predicted-net distribution sits below the 0.376-tick cost line — the LGBM correctly learned that "in expectation, on average, the realized cost-adjusted net is negative" so it refuses to gate IN any trade. This matches the v3.4.2 ep3 behaviour (see verdict_v1.md). Top-percentile sweeps were used for ranking diagnostic.

## Three-Way Comparison Table — Best Slice Per Side

### LONG (top 1–2% raw signal; least-bad slice per NPZ)
| NPZ | Best Slice | n_fills | tpf | PF | WR | Sharpe√N | day_conc | pass_hc344 | Beat cost? |
|---|---|---|---|---|---|---|---|---|---|
| v3.3 fold_00 | raw top-1% long | 186 | **-0.892** | 0.57 | 38.7% | -3.71 | 0.371 | pass | **NO** |
| v3.4.2 ep1   | raw top-5% long | 928 | -0.762 | 0.63 | 40.8% | -6.96 | 0.303 | pass | **NO** |
| v3.4.2 ep3   | single-head top-2% long | 410 | -0.520 | 0.71 | 43.4% | -3.25 | 0.263 | pass | **NO** |

### SHORT (top 1–2% raw signal or single-head; least-bad)
| NPZ | Best Slice | n_fills | tpf | PF | WR | Sharpe√N | day_conc | pass_hc344 | Beat cost? |
|---|---|---|---|---|---|---|---|---|---|
| **v3.3 fold_00** | **raw top-1% short** | 190 | **-0.534** | **0.70** | **45.3%** | -2.29 | 0.421 | pass | **NO** |
| v3.4.2 ep1   | raw top-5% short | 929 | -0.706 | 0.61 | 39.8% | -7.03 | 0.364 | pass | **NO** |
| v3.4.2 ep3   | raw top-1% short | 191 | -0.739 | 0.60 | 40.3% | -3.29 | 0.330 | pass | **NO** |

## Rules Baseline Comparison (HC #74, thr=0, all fills)
| NPZ | Long tpf | Long n | Short tpf | Short n |
|---|---|---|---|---|
| v3.3 fold_00 | -0.856 | 8961 | -0.948 | 9250 |
| v3.4.2 ep1   | -0.862 | 8843 | -0.942 | 9436 |
| v3.4.2 ep3   | -0.877 | 9138 | -0.960 | 9092 |

All three signal versions produce nearly-identical, deeply-losing rules baselines on this 5-day OOT window. The signals themselves do not provide a sufficient edge to clear cost when traded indiscriminately.

## Feature Importance (multi-head LGBM, v3.3)

**Long side** (top by split count):
1. pred_log_ret_60s_q50 — 8
2. pred_pred_mfe_60s_ticks — 6
3. pred_pred_realized_vol_30s_ticks — 6
4. pred_p_reversal_60s — 5
5. pred_fifo_tp4sl3_hit_tp — 5

**Short side**:
1. pred_log_ret_60s_q50 — 9
2. pred_pred_mae_60s_ticks — 7
3. pred_p_reversal_30s — 6
4. pred_pred_mfe_60s_ticks — 5
5. pred_log_ret_60s — 4

Multi-head LGBM keeps clustering on 60s-horizon features (q50, mfe, mae, p_reversal) — exactly the horizons where v3.3 IC has decayed by 30s+. The model uses them because the FIFO replay target is itself a ~30s-window outcome (tp4sl3), but the predictive content at that horizon is too weak relative to the variance the LGBM has to explain.

## Bottom-Line Verdict: NO-GO across all three signal versions

> **v3.3 fold_00, v3.4.2 ep1, and v3.4.2 ep3 ALL fail to produce a deployable execution-gate
> on the 20260223–20260227 OOT week.** Every method, every percentile slice, every side:
> ticks-per-fill is **negative**, profit-factor < 0.71, win-rate < 46%, Sharpe deeply
> negative. None clears the 0.376-tick passive-limit cost floor.

### Comparative ranking (least bad → worst)
1. **v3.4.2 ep3 single-head top-2% long** (tpf=-0.520) — still loses 0.52 t/fill before commission. NOT deployable.
2. **v3.3 raw top-1% short** (tpf=-0.534, PF=0.70, WR=45.3%) — the best risk-adjusted slice across all three but only 190 fills over 5 days = ~38 trades/day max.
3. v3.4.2 ep1 raw top-5% short (tpf=-0.706, n=929) — wider sample but worse tpf.
4. v3.4.2 ep3 raw top-1% short (tpf=-0.739) — worst of the three single-head shorts.

### Per-version characterisation
- **v3.4.2 ep3** has slightly better LONG single-head ranking (tpf=-0.520 vs v3.3's -1.406 single-head top-1%), reflecting its higher IC_1s (0.274 vs 0.262).
- **v3.3 fold_00** has the BEST SHORT raw-signal extreme tail (tpf=-0.534), consistent with HC #421's finding that v3.3 retained a short-side edge.
- **v3.4.2 ep1** is between the two on both sides — under-fit relative to ep3 on long ranking, but no advantage on short either.

## Root Cause (NO-GO)

This is a **regime / OOT-week problem, NOT a signal-quality problem**:
1. On the 5-day window 20260223–20260227, the FIFO market replay realises **negative gross net** before commission (long net mean -0.109, short net mean -0.129 ticks). Half of all signals fill, and the average filled trade is already losing.
2. After 0.376-tick passive-limit commission, average net is ≈ -0.5 ticks regardless of signal.
3. The cost floor of 0.376 cannot be cleared even by the top-1% slice of the best-ranking method.
4. CLAUDE.md says "top 10% confidence short signals: +1.56 ticks avg move, 60.5% WR" — that was a different OOT window. **This OOT week does not reproduce that result for any of the three signal versions.**
5. v3.3 → v3.4.2 ep3 IC regression (IC_5s 0.141→0.128, IC_10s 0.106→0.090) is real but **secondary**: even v3.3 with its better IC fails the same window.

## Recommended Next Step (NOT auto-launched — decision for next session)

The data says: **the bottleneck is the OOT-week regime, not the model.** Three actionable directions, ranked by what would change the verdict:

1. **TRY A DIFFERENT OOT WINDOW** before declaring the architecture broken. The 20260223–20260227 week may be a low-edge regime (post-Feb-OPEX, pre-FOMC) where ALL three models fail. Run the same LGBM gate against v3.3 on a more recent OOT week (e.g. 20260310-onwards) to test if the negative tpf is a window artefact. **Cost: 1 LGBM rerun on Jupiter CPU (~3 min) + locating a fresh fold_NN_oot.npz.**
2. **REGIME GATE** — add a market-state filter before signal selection (e.g. realised vol > threshold, or absolute trend > threshold). The 5-day window may be a low-vol drift regime where mean-reversion dominates and short-term momentum signals (CNN-Mamba-style) systematically underperform. Likely highest-effort change.
3. **WIDER COST-COVERAGE INSTRUMENT** — switch from ES (1-tick spread, 0.376 commission floor) to MES or to a different contract with better cost structure relative to signal strength. Out-of-scope per HC #424 but worth noting.

**Do NOT retrain.** The IC numbers across v3.3 / v3.4.2 ep1 / v3.4.2 ep3 are all within a normal range — the signal is fine, the OOT replay window is hostile. Retraining will not change this.

**Do NOT deploy any gate from this experiment.** Every single percentile slice across all three NPZs fails the cost floor.

## Outputs
- `output/hc424_jupiter_exec_research/lgbm_gate_v33_fold00_threshold_verdicts.csv`
- `output/hc424_jupiter_exec_research/lgbm_gate_v33_fold00_percentile_verdicts.csv`
- `output/hc424_jupiter_exec_research/lgbm_gate_v33_fold00_feature_importance.json`
- `output/hc424_jupiter_exec_research/lgbm_gate_v342_ep1_threshold_verdicts.csv`
- `output/hc424_jupiter_exec_research/lgbm_gate_v342_ep1_percentile_verdicts.csv`
- `output/hc424_jupiter_exec_research/lgbm_gate_v342_ep1_feature_importance.json`
- `scripts/smart_exec/hc424_lgbm_gate_v33_fold00.py`
- MLflow: http://jupiter:5000/#/experiments/130480352858023723
