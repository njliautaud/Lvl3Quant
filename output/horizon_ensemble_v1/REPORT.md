# Horizon Ensemble V1 — Report

Generated: 2026-05-22T17:50:50.384396Z

## Setup

- Source: `output/cnn_mamba_v3_4_2_fixedmtl/oot_47day_perdate`
- Usable OOT dates: **32**
- Horizons: ['5s', '10s', '30s'], trade horizon = 10s
- Fire threshold: top 5.0% per-day per-horizon on `-pred_log_ret_h` (short-side semantics)
- Cost: passive-limit RT commission = 0.376 ticks
- **Caveat (HC #74)**: label-level P&L on `target_log_ret_10s`, NOT FIFO market replay. Prior label-vs-FIFO gap on this signal was ~+1.5 ticks IN OUR FAVOR, so label-level under-estimates.

## Diagnostics

### Pooled horizon-pair Spearman correlations

| pair | rho |
|---|---|
| 5s_vs_10s | 0.8967 |
| 5s_vs_30s | -0.5850 |
| 10s_vs_30s | -0.5082 |

5s and 10s are highly correlated (~0.9); 30s is near-independent — ensemble has at least one independent axis.

### Pooled IC (pred10s vs realized10s): +0.0356

## Variant comparison

| variant | verdict | net_ticks | trades | fire% | pdays/total | day_sharpe | day_conc |
|---|---|---|---|---|---|---|---|
| baseline_10s | **REJECT** | -0.1235 | 87565 | 5.55% | 8/32 (0.25) | -0.251 | 0.121 |
| V1_unanimous | **REJECT** | -1.6837 | 39 | 0.00% | 3/32 (0.09) | +0.054 | 0.794 |
| V2_majority | **REJECT** | -0.1316 | 38950 | 2.47% | 9/32 (0.28) | -0.275 | 0.175 |
| V3_any_with_confirm | **REJECT** | -0.1313 | 38945 | 2.47% | 9/32 (0.28) | -0.274 | 0.175 |

## Regime stratification (per variant)


### baseline_10s

| regime | n_days | net_mean | profit_days | sharpe |
|---|---|---|---|---|
| green | 14 | -0.2768 | 2 | -0.9607 |
| red | 10 | +0.1507 | 4 | +0.2725 |
| flat | 8 | -0.1197 | 2 | -0.4839 |

### V1_unanimous

| regime | n_days | net_mean | profit_days | sharpe |
|---|---|---|---|---|
| green | 1 | +1.6240 | 1 | +nan |
| red | 2 | -1.1707 | 1 | -0.3085 |
| flat | 1 | +1.2907 | 1 | +nan |

### V2_majority

| regime | n_days | net_mean | profit_days | sharpe |
|---|---|---|---|---|
| green | 14 | -0.5223 | 2 | -0.6535 |
| red | 10 | +0.2629 | 5 | +0.3305 |
| flat | 8 | -0.2739 | 2 | -0.6323 |

### V3_any_with_confirm

| regime | n_days | net_mean | profit_days | sharpe |
|---|---|---|---|---|
| green | 14 | -0.5222 | 2 | -0.6533 |
| red | 10 | +0.2635 | 5 | +0.3312 |
| flat | 8 | -0.2739 | 2 | -0.6323 |

## Verdict summary

- **Best ensemble variant**: `V3_any_with_confirm` (REJECT, net=-0.1313, day_sharpe=-0.274)
- **Baseline (10s solo)**: net=-0.1235, day_sharpe=-0.251
- **Best vs baseline delta (net_ticks)**: -0.0077

**NEXT STEP**: all variants REJECT at label-level. Horizon ensembling does not rescue the short-side signal on raw CNN-Mamba v3.4.2 predictions. Pivot to next axis.
