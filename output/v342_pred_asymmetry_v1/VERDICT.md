# v3.4.2 PRED ASYMMETRY — VERDICT

Run date: 2026-06-05 08:43

OOT days analyzed: 32  
Regime mix: green=15, red=11, flat=2, unknown=4


## 1. Top-5% sign distribution by head (pooled across all days)

| head | n | %long | %short |
|------|---|-------|--------|
| log_ret_1s | 93949 | 9.2% | 90.8% |
| log_ret_5s | 82512 | 0.0% | 100.0% |
| log_ret_10s | 89894 | 1.0% | 99.0% |
| log_ret_30s | 79121 | 98.2% | 1.8% |

## 2. Sign skew across quantiles (5s and 30s heads)

| quantile | 5s %long | 5s %short | 30s %long | 30s %short |
|----------|----------|-----------|-----------|------------|
| top-1% | 0.0% | 100.0% | 94.3% | 5.7% |
| top-5% | 0.0% | 100.0% | 98.2% | 1.8% |
| top-10% | 2.0% | 98.0% | 75.9% | 24.1% |
| top-20% | 1.5% | 98.5% | 42.4% | 57.6% |
| top-50% | 35.6% | 64.4% | 43.5% | 56.5% |
| all | 61.6% | 38.4% | 30.7% | 69.3% |

## 3. Head agreement at top-5% (by |pred_5s|)

- n = 82512
- 5s sign: 0.0% pos, 100.0% neg
- 30s sign: 100.0% pos, 0.0% neg
- 1s sign: 0.0% pos, 100.0% neg
- 10s sign: 0.0% pos, 100.0% neg
- 5s ↔ 30s agree: **0.0%**
- 1s ↔ 5s agree: 100.0%
- 1s ↔ 30s agree: 0.0%
- 1s/5s/30s ALL agree: **0.0%**

## 4. Regime-stratified top-5% sign (5s head)

| regime | n | %long | %short |
|--------|---|-------|--------|
| green | 55848 | 0.0% | 100.0% |
| red | 32978 | 3.2% | 96.8% |
| flat | 6553 | 1.8% | 98.2% |

## 4b. Regime-stratified top-5% sign (30s head)

| regime | n | %long | %short |
|--------|---|-------|--------|
| green | 43647 | 98.5% | 1.5% |
| red | 31025 | 65.3% | 34.7% |
| flat | 6039 | 4.7% | 95.3% |

## 5. Calibration at top-5% (pooled)

| head | n | sign_acc | realized_mean | when_long_pred | when_short_pred | binary_acc_vs_p_up |
|------|---|----------|---------------|----------------|-----------------|--------------------|
| log_ret_1s | 93949 | 0.449 | -0.1868 | -0.0123 | -0.2045 | nan |
| log_ret_5s | 82512 | 0.481 | -0.1100 | -0.2500 | -0.1100 | 0.582 |
| log_ret_10s | 89894 | 0.490 | -0.1383 | 0.1071 | -0.1409 | 0.567 |
| log_ret_30s | 79121 | 0.469 | 0.2128 | 0.2316 | -0.8143 | 0.467 |

## VERDICT (plain English)


### Base-rate edge test (CRITICAL):
If the model just always predicts the majority direction in top-5%, its accuracy = max(base_rate_up, 1 - base_rate_up). Edge over that baseline is what matters.
  - 5s: binary_acc=0.582, base_rate_up=0.418, pred_up_rate=0.007%, edge_above_constant_baseline=-0.000
  - 10s: binary_acc=0.567, base_rate_up=0.432, pred_up_rate=1.049%, edge_above_constant_baseline=-0.001
  - 30s: binary_acc=0.467, base_rate_up=0.465, pred_up_rate=98.208%, edge_above_constant_baseline=-0.068
- **Asymmetry:** **SEVERE asymmetry on 5s head**: top-5% predictions are 100% short / 0% long.
- **Regime driver:** Top-5% is short-skewed in BOTH green (100% short) and red (97% short) regimes → asymmetry is **INTRINSIC to the model**, NOT regime-driven.
- **5s vs 30s heads:** **5s and 30s heads STRONGLY DISAGREE** at top-5%: only 0.0% agreement on sign. These heads encode different (likely conflicting) horizons. DO NOT fuse naively. Pick one or build an explicit multi-h model.
- **Calibration (5s, top-5%):** sign accuracy at top-5% is 48.1% — **NO edge / coin flip** | realized return is negative for BOTH long and short predictions — model is wrong on longs, right on shorts (one-sided edge)

### Recommendation

- **REPLACE v3.4.2 for execution research.** Top-confidence predictions are heavily one-sided AND sign accuracy is at coin-flip. This is consistent with the model latching onto a training-window artifact (likely a downtrending training period), not a transferable edge.
- **30s head usage:** do NOT fuse with 5s. Drop from execution path or use only as a veto when |pred_30s| is large and disagrees.