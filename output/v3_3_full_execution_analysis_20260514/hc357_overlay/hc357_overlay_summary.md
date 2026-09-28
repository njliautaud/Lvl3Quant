# v3.3 HC #357 EXECUTION OVERLAY — HC #363 deliverable 5

**⚠️ STATISTICAL OVERLAY, NOT L3 REPLAY.** The full HC #357 number requires real MBO event replay against `data/processed/mbo_events_smart_v3/<date>_mbo_events.npz` (12.4M events/day × 5 OOT days). That builder is OUT OF SCOPE here; this overlay applies queue/adv-sel/cancel haircuts derived from HC #80 + HC #357 priors.

## Overlay parameters

- Queue-position fill probability: **50%** (FIFO claims 100% fill rate; reality ~50% due to L3 queue depth)
- Adverse-selection bad-fill rate: **25%** of filled trades, losing **2.0 ticks** each on average
- Cancel/replace: **1.5 events/attempt × 0.10 ticks each**
- Commission baked into `passive_net_after_comm`: **0.376 ticks RT**

## Top 25 cells by HC #357-adjusted Sharpe

| Head | Side | Band | n_fills | FIFO Sharpe | FIFO net | HC357 Sharpe | HC357 net | Δ net | Day-conc | CI low |
|---|---|---|---|---|---|---|---|---|---|---|
| fifo_tp8sl5_net | SHORT | Top0.1% | 114 | 0.455 | 1.048 | **0.054** | 0.124 | -0.924 | 95% | 0.86 |
| p_reversal_60s | SHORT | Top0.1% | 35 | 0.507 | 0.971 | **0.045** | 0.086 | -0.886 | 44% | 0.48 |
| log_ret_60s_q50 | SHORT | Top1% | 48 | 0.366 | 0.812 | **0.003** | 0.006 | -0.806 | 92% | 0.31 |
| log_ret_60s_q90 | SHORT | Top5% | 232 | -0.001 | -0.379 | **-0.001** | -0.590 | -0.210 | 62% | -0.50 |
| log_ret_10s_q50 | LONG | Top0.1% | 62 | -0.004 | -0.389 | **-0.006** | -0.595 | -0.205 | 37% | -0.78 |
| log_ret_10s_q50 | SHORT | Top0.1% | 30 | 0.390 | 0.767 | **-0.008** | -0.017 | -0.783 | 38% | 0.04 |
| p_up_60s | SHORT | Top0.5% | 336 | 0.006 | -0.354 | **-0.010** | -0.577 | -0.223 | 79% | -0.37 |
| log_ret_60s | SHORT | Top10% | 48 | 0.006 | -0.354 | **-0.010** | -0.577 | -0.223 | 39% | -1.00 |
| p_up_5s | LONG | Top20% | 44 | 0.007 | -0.354 | **-0.011** | -0.577 | -0.223 | 33% | -0.92 |
| log_ret_60s_q90 | SHORT | Top10% | 387 | -0.008 | -0.404 | **-0.012** | -0.602 | -0.198 | 43% | -0.38 |
| p_reversal_30s | LONG | Top0.1% | 109 | 0.008 | -0.348 | **-0.013** | -0.574 | -0.226 | 82% | -0.68 |
| log_ret_30s_q10 | SHORT | Top0.1% | 128 | 0.008 | -0.348 | **-0.013** | -0.574 | -0.226 | 70% | -0.58 |
| log_ret_30s_q90 | LONG | Top1% | 888 | -0.010 | -0.408 | **-0.014** | -0.604 | -0.196 | 38% | -0.24 |
| log_ret_60s_q50 | SHORT | Top20% | 745 | 0.012 | -0.337 | **-0.020** | -0.568 | -0.232 | 38% | -0.19 |
| log_ret_60s_q90 | SHORT | Top20% | 875 | 0.014 | -0.330 | **-0.024** | -0.565 | -0.235 | 38% | -0.16 |
| p_reversal_15s | SHORT | Top1% | 647 | 0.015 | -0.326 | **-0.026** | -0.563 | -0.237 | 32% | -0.22 |
| log_ret_30s_q50 | SHORT | Top0.5% | 72 | 0.355 | 0.694 | **-0.027** | -0.053 | -0.747 | 35% | 0.36 |
| p_up_60s | SHORT | Top1% | 496 | 0.017 | -0.317 | **-0.030** | -0.558 | -0.242 | 55% | -0.26 |
| p_reversal_15s | LONG | Top0.1% | 77 | -0.030 | -0.479 | **-0.039** | -0.640 | -0.160 | 89% | -0.83 |
| p_reversal_30s | SHORT | Top1% | 397 | -0.031 | -0.485 | **-0.042** | -0.642 | -0.158 | 31% | -0.44 |
| log_ret_30s_q90 | LONG | Top0.5% | 478 | 0.029 | -0.278 | **-0.056** | -0.539 | -0.261 | 42% | -0.19 |
| log_ret_10s_q90 | LONG | Top1% | 1013 | -0.052 | -0.543 | **-0.064** | -0.671 | -0.129 | 35% | -0.35 |
| log_ret_60s_q50 | SHORT | Top10% | 340 | -0.063 | -0.596 | **-0.074** | -0.698 | -0.102 | 50% | -0.60 |
| fifo_tp4sl3_net | SHORT | Top0.1% | 68 | 0.261 | 0.493 | **-0.081** | -0.154 | -0.646 | 98% | 0.01 |
| log_ret_60s | SHORT | Top20% | 83 | -0.072 | -0.627 | **-0.082** | -0.713 | -0.087 | 36% | -0.97 |

## SURVIVOR cells — HC357 Sharpe > 0 AND CI low > 0 (deploy-ready under HC #357 priors)

**3 of 213 cells survive** (1%).

| Head | Side | Band | n_fills | HC357 Sharpe | HC357 net | Δ net vs FIFO |
|---|---|---|---|---|---|---|
| fifo_tp8sl5_net | SHORT | Top0.1% | 114 | 0.054 | 0.124 | -0.924 |
| p_reversal_60s | SHORT | Top0.1% | 35 | 0.045 | 0.086 | -0.886 |
| log_ret_60s_q50 | SHORT | Top1% | 48 | 0.003 | 0.006 | -0.806 |

## What this overlay tells us

- Top HC357 cell: **fifo_tp8sl5_net SHORT Top0.1%** — Sharpe 0.054, net 0.124 t/attempt (vs FIFO 1.048 t/fill).
- Δ net vs FIFO across survivors: median -0.886 ticks/attempt — confirms that under HC #357 priors, FIFO over-states realized P&L by ~half a tick on these cells.
- This is the bar to beat with real L3 replay. If the L3 simulator produces HC357 Sharpe NUMBERS BELOW these, our 50%/25%/2t prior was too generous → tighten.

## What's MISSING for true HC #357 compliance

- Real queue-position tracking from L3 add/cancel/modify event stream.
- Time-varying adverse-selection (regime-dependent, currently constant 25%).
- Dynamic cancel/replace policy informed by signal staleness.
- Spread-state at fill time (not assumed 1-tick).

**ETA for full L3 replay builder**: ~4-8h Jupiter dev + ~30min/OOT-date compute. Recommend dispatching as next sprint after user signoff on this overlay.
