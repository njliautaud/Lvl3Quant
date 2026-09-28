# HC #396 — Meta-MLP Stacker Validation Report

Date: 2026-05-16
Author: weekend lane agent
Source NPZ: `output/cnn_mamba_v3_3_uncertainty_weighted/fold_00_predictions.npz`

## TL;DR — verdict

The "4.8x IC boost" the prior agent claimed for the stacker on `fifo_tp8sl5_net` is **head-specific, not general**, and the PnL evidence is **mixed, not convincing**.

1. In the full canonical market replay (HC #377 5-component, val slab only — fair held-out comparison), the stacker on `fifo_tp8sl5_net`:
   - **Wins**: ticks/fill +0.99 vs +0.61 raw (+62%), profit factor 4.67 vs 4.13, win rate 75.0% vs 69.3%, **day_conc 0.63 vs 1.00** (raw had everything bunched on one day; stacker spreads risk across days).
   - **Loses**: Sharpe 1612 vs 2646 (raw). Higher per-fill profit but higher variance.
   - **HC #344**: BOTH FAIL (day_conc>0.20 — but raw is full 1.00, stacker is 0.63).
2. Across the top-5 directional heads (by abs Sharpe in the production-readiness sweep), the stacker val IC is **WORSE than raw head IC in 5/5 cases** (lift 0.55–0.85x). The fifo_tp8sl5_net case is an outlier driven by an unusually weak raw head (val IC 0.048, much lower than its peers at 0.08–0.11).
3. The stacker's main consistent win is **val R²** (5/6 heads): when raw heads are mis-scaled (raw R² huge negative — e.g. p_up_5s raw R² = −3.92), the stacker re-calibrates to ~0. So the stacker behaves more like a **calibration layer** than a signal-boost layer.

## Task 1 — Canonical market replay (raw head vs stacker)

Operational config (best raw `fifo_tp8sl5_net` from `output/v3_3_production_readiness_20260516/sweep_results_full.csv`):
- band P95 (top 5% by pred), side=long, order_type=passive_at_touch_plus_1, cancel_window=40 evals, hold_s=1.0, exit_horizon=1s, regime=all.

VAL SLAB ONLY (last 20% of mask-valid rows; orig idx [172496, 236864] of n_total=241351):

| Signal | n_signals | n_filled | fill_rate | ticks/fill | Sharpe | Sortino | PF | WR (%) | day_conc | pass_hc344 |
|---|---|---|---|---|---|---|---|---|---|---|
| RAW head | 631 | 75 | 0.119 | +0.611 | 2646 | 6606 | 4.13 | 69.33 | 1.00 | FAIL |
| META stacker | 591 | 68 | 0.115 | **+0.992** | 1612 | 4601 | **4.67** | **75.0** | **0.626** | FAIL |

Full CSV: `output/meta_mlp_v3_3/stacker_vs_raw_replay.csv` (includes full-slab biased numbers for context).

Verdict: **mixed**. Stacker delivers higher per-trade profitability and dramatically better day diversification but at lower Sharpe (higher variance). The 0.626 day_conc is a real win — raw head's PnL on val is fully concentrated on one day (1.00), which is not a trustworthy signal in production.

## Task 2 — Stacker generalization across top-5 directional heads

Selection: top 5 directional heads by best abs(Sharpe) in production-readiness sweep, regime=all, n_filled≥30, with positive raw-replay Sharpe (sign-flipping the negative-Sharpe heads is a separate question):

| target_head | stacker val IC | raw val IC | Δ IC | lift | stacker val R² | raw val R² | MLflow run |
|---|---|---|---|---|---|---|---|
| fifo_tp8sl5_net (control) | **+0.230** | +0.048 | +0.182 | **4.79x** | +0.022 | −0.057 | 22812ff2... |
| log_ret_30s_q90 | −0.016 | +0.099 | −0.115 | −0.16x | −0.029 | −0.542 | a57202dd... |
| log_ret_10s_q90 | +0.089 | +0.110 | −0.021 | 0.81x | +0.002 | −0.554 | 8587c6b7... |
| p_up_10s | +0.049 | +0.088 | −0.039 | 0.56x | +0.006 | −2.890 | d57c0d86... |
| log_ret_30s | +0.054 | +0.079 | −0.026 | 0.68x | −0.006 | +0.005 | 8adb0d43... |
| p_up_5s | +0.090 | +0.106 | −0.015 | 0.85x | +0.012 | −3.920 | 5366fc95... |

All 6 runs logged to MLflow experiment `meta_mlp_v3_3_stacker`.

Each stacker trained on first 80% chronologically; val on last 20%. Same architecture as control (`[34→128→64→1]`, dropout 0.1, MSE, AdamW lr=1e-3, 20 epochs). Each head's stacker uses the OTHER 31 heads + 3 book-context proxies (`pred_pred_realized_vol_30s_ticks`, `pred_p_up_30s − 0.5`, spread=1.0).

Interpretation:
- The IC boost on `fifo_tp8sl5_net` is a single-head artifact, not a general phenomenon. Likely because (a) raw `fifo_tp8sl5_net` head has the weakest val IC of all 6 (0.048), making the bar low, and (b) its mask is much sparser (47k train rows vs 193k for log-ret/p_up heads), so the underlying head is undertrained relative to peers.
- For well-trained heads (log_ret_*, p_up_*), the raw head already captures the directional signal better than the stacker can reconstruct from its peers. Stackers add noise relative to the raw signal on rank-correlation.
- Stackers DO consistently improve val R² (magnitude calibration). Raw p_up_5s/p_up_10s heads have huge negative R² → they are mis-scaled. The stacker output sits in a regression-friendly scale.

## Recommendation for next weekend job

1. **Do NOT productionize the meta-MLP stacker as a signal-replacement** for the trained heads — generalization evidence is negative on rank-correlation across 5/5 well-trained heads.
2. **Investigate the stacker as a CALIBRATION layer**: use raw heads' rank (for selection) but stacker's magnitude (for sizing). This separates the "what to trade" (raw IC wins) from "how big" (stacker R² wins).
3. **Re-train the raw `fifo_tp8sl5_net` head with more data / class balancing** before drawing further conclusions about that head's bracket-net signal. The 47k-row mask is the bottleneck — a re-trained head might already match the stacker's IC without a separate model.
4. **Confluence pass on the stacker outputs**: the production-readiness sweep produced 0 confluence pairs (no heads passed HC #344). Re-run the sweep with stacker-calibrated heads to see if confluence pairs emerge under tighter day_conc.

Time budget consumed: ~45 min.

Deliverables paths:
- `/home/jupiter/Lvl3Quant/output/meta_mlp_v3_3/stacker_vs_raw_replay.csv`
- `/home/jupiter/Lvl3Quant/output/meta_mlp_v3_3/stacker_vs_raw_replay.md`
- `/home/jupiter/Lvl3Quant/output/meta_mlp_v3_3/stacker_topK_heads_comparison.csv`
- `/home/jupiter/Lvl3Quant/output/meta_mlp_v3_3/HC396_stacker_validation_report.md` (this file)
- `/home/jupiter/Lvl3Quant/scripts/v3_3_research/v33_stacker_vs_raw_replay.py`
- Razer: `C:\Users\claude\Lvl3Quant\scripts\meta_mlp_v3_3\train_multi.py`, `launch_multi.ps1`
- MLflow runs: 5 new in experiment `meta_mlp_v3_3_stacker` (6 total).
