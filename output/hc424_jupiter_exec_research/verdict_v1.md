# HC #424 §3 — Jupiter CPU Execution Research Verdict v1 (2026-05-18)

## Source
- **NPZ**: `/home/jupiter/Lvl3Quant/output/hc424_jupiter_exec_research/inputs/v3_4_2_ep3/fold_00_ep3_oot.npz`
- **SHA256**: `bf513ec9eef5ffad47e841d30893222934f3537396fc75c59062e351e43b9838`
- **Origin**: SCP from `nick@neptune:/home/nick/Lvl3Quant/output/cnn_mamba_v3_4_2_fixedmtl/fold_00_ep3_oot.npz`
- **Samples**: 241,351 across 5 OOT trading days (20260223–20260227)
- **Reported IC** (in NPZ): IC_1s=0.2744, IC_5s=0.1277, IC_10s=0.0897, IC_30s=0.0559
- **MFE/MAE corr**: 0.292 / 0.276

## Execution Layer Choice
**(A) LGBM gate** — chosen for speed of first canonical-replay verdict on Jupiter CPU.
LGBM regressor predicting `tp4sl3_long/short_net_ticks` (FIFO market replay target) from the
full 30-head multi-head feature vector. Compared against:
- **Single-head LGBM** baseline (pred_log_ret_1s only)
- **Raw signal** baseline (HC #74 rules: take direction implied by pred_log_ret_1s sign)

## Configuration
- Cost model HC #392: passive limit = 0.376 ticks (commission only), market = 1.376
- HC #344 day_conc gate: pass if < 0.50
- HC #74 FIFO market replay outcomes (`tp4sl3_*_net_ticks` from per-day FIFO label NPZs)
- HC #0 sliding: per-day last-20% holdout (HC #393 holdout discipline)
- Train: 193,079 samples (first 80% of each OOT day)
- Holdout: 48,272 samples (last 20% of each day)

## Multi-Head Features Used (30 heads — HC #422 R8 / HC #423 §4)
log_ret horizons (1s/5s/10s/30s/60s/5min) + p_up + 3-point quantiles for 10s/30s/60s
+ pred_mfe/mae_30s/60s + time_to_mfe + p_reversal_15/30/60s + realized_vol_30s_ticks
+ pred_fifo_tp4sl3_net + pred_fifo_tp4sl3_hit_tp. **30 features total.**

## Canonical Replay Verdict (HC #397B columns)

### Long side — passive limit, FIFO-realized fills
| Method | Top % | n_fills | tpf | Sharpe√N | Sortino√N | PF | WR | day_conc | pass_hc344 |
|---|---|---|---|---|---|---|---|---|---|
| multi_head_lgbm | 5%  | 3596 | **-0.847** | -15.32 | -54.09 | 0.59 | 40.2% | 0.591 | FAIL |
| multi_head_lgbm | 2%  | 381  | -1.236 | -7.22 | -23.93 | 0.47 | 35.2% | 0.480 | pass |
| single_head_lgbm | 2% | 410  | **-0.520** | -3.25 | -8.58 | 0.71 | 43.4% | 0.263 | pass |
| single_head_lgbm | 5% | 965  | -0.685 | -6.61 | -18.57 | 0.64 | 40.9% | 0.283 | pass |
| raw_signal | 2%      | 386  | -0.915 | -5.67 | -16.30 | 0.55 | 38.3% | 0.290 | pass |

### Short side — passive limit, FIFO-realized fills
| Method | Top % | n_fills | tpf | Sharpe√N | Sortino√N | PF | WR | day_conc | pass_hc344 |
|---|---|---|---|---|---|---|---|---|---|
| multi_head_lgbm | 1% | 186  | -0.873 | -3.59 | -15.32 | 0.58 | 40.3% | 0.355 | pass |
| multi_head_lgbm | 2% | 368  | -0.912 | -5.43 | -16.69 | 0.56 | 38.9% | 0.312 | pass |
| raw_signal | 1%     | 191  | **-0.739** | -3.29 | -8.24 | 0.60 | 40.3% | 0.330 | pass |
| single_head_lgbm | 1% | 238 | -1.225 | -5.80 | -13.46 | 0.45 | 35.3% | 0.282 | pass |

### Rules baseline (HC #74, all-fills @ thr 0)
- Long: 9138 fills, **tpf=-0.877**, Sharpe=-26.34, PF=0.56, WR=38.4%, day_conc=0.264 (HC #344 pass)
- Short: 9092 fills, **tpf=-0.960**, Sharpe=-28.75, PF=0.53, WR=37.5%, day_conc=0.312 (HC #344 pass)

## Bottom Line

**v3.4.2 ep3 predictions DO NOT support a profitable execution gate on the 20260223–20260227 OOT window.**

Every method, every percentile slice, every side: ticks-per-fill is **negative**, profit-factor < 0.7,
win-rate < 41%, Sharpe is deeply negative. The LGBM multi-head gate is **NOT a meaningful improvement
over the rules baseline** — and in many slices it underperforms even the single-head baseline because
LGBM ranks higher the trades where realized FIFO is worse (likely because the LGBM picks up the
"high vol = high MFE prediction = high realized net" axis, which correlates with high variance, not edge).

The single-head LGBM at top-2% LONG (tpf=-0.520, PF=0.71, WR=43.4%) is the *least bad* result —
but still loses 0.52 ticks/fill, far from the 0.376 ticks needed to clear commission with passive limits.
None of the multi-head Sharpe values exceeds the rules baseline. **No deployable execution layer found
in this experiment.**

### Comparison vs HC #421 baselines
HC #421 cited v3.3 fold_00 single-head exec rules baseline as marginally profitable
(top-decile short ~+0.34 net). The v3.4.2 ep3 OOT window appears to be a **regime where the
multi-head v3.4.2 model has lost the short-side edge** v3.3 had — consistent with HC #422's
mixed-IC finding. IC_5s/10s dropped from v3.3's 0.141/0.106 to v3.4.2 ep3's 0.128/0.090.

## Feature Importance (multi-head LGBM, top features by split count)

**Short side (more useful):**
1. pred_fifo_tp4sl3_hit_tp — 34 splits (the model's OWN tp/sl head — directly learned)
2. pred_p_up_60s — 33
3. pred_log_ret_60s_q10 — 28
4. pred_log_ret_60s_q90 — 23
5. pred_log_ret_5s — 18

**Long side (less useful — flat importance):**
1. pred_log_ret_1s — 5
2. pred_log_ret_5min — 5
3. pred_log_ret_60s_q50 — 5
4. pred_p_reversal_60s — 5

The flat importance on long side indicates LGBM found very little signal to exploit there.
On short side it leaned heavily on the model's own tp/sl-hit head, which is reasonable but
clearly insufficient.

## Next Steps (recommendations)

1. **DO NOT deploy** any of these gates. None passes a deployment bar.
2. The bottleneck is the v3.4.2 ep3 **signal quality on this OOT week**, not the gate.
   Compare: v3.3 fold_00 was IC_5s=0.141 / IC_10s=0.106; v3.4.2 ep3 is 0.128 / 0.090 →
   measurable regression on the 5s/10s horizons that drive execution edge.
3. **Try v3.3 fold_00 predictions for comparison** as a fallback per HC #424 §3.
   If v3.3 produces a usable gate on the same dates, this confirms v3.4.2 is the issue,
   not the gate architecture or OOT regime.
4. **Try v3.4.2 ep1 NPZ instead of ep3** — HC #422 noted ep1 had the IC_1s win first;
   different epochs may rank features differently.
5. Defer (C) MLP gate and (B) PPO until a stronger source NPZ is available — they will
   inherit the same signal limitations.

## MLflow Runs
- Experiment: `hc424_jupiter_exec_research_v342` (id 906502745046598612)
- Run 1 (absolute-threshold gate): `7fc05d9c60724f498d12990e40f824a3`
  - URL: http://jupiter:5000/#/experiments/906502745046598612/runs/7fc05d9c60724f498d12990e40f824a3
- Run 2 (percentile gate diagnostic): `4d8b1d940c964a76a8e9b3c0e894ed82`
  - URL: http://jupiter:5000/#/experiments/906502745046598612/runs/4d8b1d940c964a76a8e9b3c0e894ed82

## Artifacts
- `/home/jupiter/Lvl3Quant/output/hc424_jupiter_exec_research/v3_4_2_ep3_npz_schema.md`
- `/home/jupiter/Lvl3Quant/output/hc424_jupiter_exec_research/lgbm_gate_v342_ep3_verdicts.csv`
- `/home/jupiter/Lvl3Quant/output/hc424_jupiter_exec_research/lgbm_gate_v342_ep3_pctl_verdicts.csv`
- `/home/jupiter/Lvl3Quant/output/hc424_jupiter_exec_research/lgbm_gate_v342_ep3_feature_importance.json`
- `/home/jupiter/Lvl3Quant/output/hc424_jupiter_exec_research/lgbm_gate_v342_ep3_pctl_feature_importance.json`
- Scripts: `/home/jupiter/Lvl3Quant/scripts/smart_exec/hc424_lgbm_gate_v342_ep3.py` + `_v2.py`
