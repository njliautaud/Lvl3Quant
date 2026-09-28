# SPY Cross-Asset Execution — Phase A Report

_Generated 2026-06-05 00:45:37 — HC #534 / #527 R1+R2._


## Data Inventory

| Date | ES preds | In SPY-window | SPY open | SPY close | Regime | ES intra-IC(1s) |
|------|----------|---------------|----------|-----------|--------|------------------|
| 20260302 | 59,677 | 57,742 | 678.65 | 686.33 | green | 0.0975 |
| 20260303 | 79,396 | 76,120 | 675.02 | 680.32 | green | 0.0836 |
| 20260304 | 45,464 | 43,144 | 681.61 | 685.19 | green | 0.0904 |
| 20260305 | 74,559 | 72,315 | 682.04 | 681.42 | flat | 0.0996 |
| 20260306 | 81,526 | 75,723 | 673.41 | 672.46 | red | 0.0881 |
| 20260309 | 69,288 | 66,526 | 666.43 | 678.30 | green | 0.0694 |
| 20260310 | 65,674 | 64,277 | 677.68 | 677.00 | red | 0.0782 |
| 20260311 | 58,570 | 57,519 | 677.55 | 676.35 | red | 0.0885 |
| 20260312 | 62,423 | 61,008 | 671.12 | 666.05 | red | 0.0859 |

## ES Prediction Coverage Gap

All 9 days have ES v3.4.2 predictions available — no gap.
Prediction-to-timestamp mapping: ES MBO event index = 999 + i*250 (stride=250, window=1000).
Max truncation from end: 86 predictions (negligible).

## Cross-Asset IC: ES Prediction vs SPY Forward Drift

IC computed Pearson, head-horizon matched (pred_1s vs SPY drift over 1s, etc.).
Drift measured in bps of SPY mid-price.

### Concat IC (all valid samples across 9 days)

| Horizon | Lag (ms) | Weighted-Mean IC | N samples | N days | SE (across days) |
|---------|----------|------------------|-----------|--------|------------------|
| 1s | 0 | 0.0039 | 574,374 | 9 | 0.0045 |
| 1s | 50 | 0.0074 | 574,374 | 9 | 0.0044 |
| 1s | 100 | 0.0109 | 574,374 | 9 | 0.0045 |
| 1s | 200 | 0.0182 | 574,374 | 9 | 0.0044 |
| 1s | 500 | 0.0206 | 574,374 | 9 | 0.0024 |
| 1s | 1000 | 0.0070 | 574,374 | 9 | 0.0012 |
| 5s | 0 | 0.0050 | 574,374 | 9 | 0.0029 |
| 5s | 50 | 0.0065 | 574,374 | 9 | 0.0028 |
| 5s | 100 | 0.0079 | 574,374 | 9 | 0.0028 |
| 5s | 200 | 0.0111 | 574,374 | 9 | 0.0026 |
| 5s | 500 | 0.0111 | 574,374 | 9 | 0.0021 |
| 5s | 1000 | 0.0040 | 574,374 | 9 | 0.0015 |
| 10s | 0 | 0.0081 | 574,374 | 9 | 0.0025 |
| 10s | 50 | 0.0090 | 574,374 | 9 | 0.0024 |
| 10s | 100 | 0.0101 | 574,374 | 9 | 0.0024 |
| 10s | 200 | 0.0121 | 574,374 | 9 | 0.0022 |
| 10s | 500 | 0.0120 | 574,374 | 9 | 0.0016 |
| 10s | 1000 | 0.0072 | 574,374 | 9 | 0.0015 |
| 30s | 0 | 0.0105 | 574,374 | 9 | 0.0063 |
| 30s | 50 | 0.0101 | 574,374 | 9 | 0.0063 |
| 30s | 100 | 0.0093 | 574,374 | 9 | 0.0062 |
| 30s | 200 | 0.0087 | 574,374 | 9 | 0.0064 |
| 30s | 500 | 0.0087 | 574,374 | 9 | 0.0064 |
| 30s | 1000 | 0.0117 | 574,374 | 9 | 0.0062 |

### Per-Day IC (lag=0, head-horizon matched)

| Date | Regime | IC(1s) | IC(5s) | IC(10s) | IC(30s) |
|------|--------|--------|--------|---------|---------|
| 20260302 | green | 0.0086 | -0.0014 | 0.0162 | 0.0285 |
| 20260303 | green | 0.0122 | 0.0167 | 0.0160 | 0.0032 |
| 20260304 | green | -0.0008 | 0.0041 | 0.0119 | 0.0462 |
| 20260305 | flat | 0.0112 | 0.0120 | 0.0088 | 0.0069 |
| 20260306 | red | 0.0277 | 0.0152 | 0.0133 | -0.0127 |
| 20260309 | green | -0.0151 | -0.0005 | -0.0028 | -0.0009 |
| 20260310 | red | 0.0012 | 0.0026 | 0.0096 | 0.0299 |
| 20260311 | red | -0.0090 | -0.0013 | 0.0020 | 0.0163 |
| 20260312 | red | -0.0104 | -0.0088 | -0.0030 | -0.0029 |

## Cross-Asset Execution PnL: Trade SPY on ES Signals

Confidence threshold = top X% by |pred|. Side filter = short only / long only.
Mode 'market' = cross 1 tick total round-trip + SEC + TAF. Mode 'passive' = no spread cost (best case).
Costs: SPY $0 commission (Alpaca), SEC fee 8e-6 on sell notional, TAF $0.0000166/share.
PnL reported in BPS of SPY notional. Sharpe = per-trade, NOT annualized.

### Champion config: head=1s, lag=0, by (top_pct, side, mode)

| Top% | Side | Mode | Days | N trades | Mean net (bps) | Sharpe | Sortino | PF | WR |
|------|------|------|------|----------|----------------|--------|---------|----|----|
| 5% | short | market | 9 | 35072 | -0.24 | -0.306 | -0.392 | 0.41 | 33.50% |
| 5% | short | passive | 9 | 35072 | -0.09 | -0.116 | -0.147 | 0.72 | 42.79% |
| 5% | long | market | 9 | 7270 | -0.18 | -0.205 | -0.284 | 0.62 | 36.88% |
| 5% | long | passive | 9 | 7270 | -0.03 | -0.014 | -0.004 | 1.02 | 46.43% |
| 10% | short | market | 9 | 70759 | -0.22 | -0.291 | -0.380 | 0.44 | 33.93% |
| 10% | short | passive | 9 | 70759 | -0.07 | -0.096 | -0.124 | 0.76 | 43.44% |
| 10% | long | market | 9 | 10907 | -0.19 | -0.194 | -0.274 | 0.63 | 37.09% |
| 10% | long | passive | 9 | 10907 | -0.04 | -0.006 | 0.006 | 1.03 | 46.46% |
| 20% | short | market | 9 | 105384 | -0.23 | -0.291 | -0.381 | 0.44 | 34.12% |
| 20% | short | passive | 9 | 105384 | -0.08 | -0.100 | -0.131 | 0.75 | 43.44% |
| 20% | long | market | 9 | 19292 | -0.18 | -0.204 | -0.289 | 0.59 | 36.06% |
| 20% | long | passive | 9 | 19292 | -0.03 | -0.010 | -0.001 | 1.00 | 45.92% |

### Best Sharpe per (lag, horizon, mode) — top10% short

| Horizon | Lag (ms) | Mode | Days | N trades | Mean net (bps) | Sharpe |
|---------|----------|------|------|----------|----------------|--------|
| 1s | 0 | market | 9 | 70759 | -0.22 | -0.291 |
| 1s | 0 | passive | 9 | 70759 | -0.07 | -0.096 |
| 1s | 50 | market | 9 | 70759 | -0.22 | -0.289 |
| 1s | 50 | passive | 9 | 70759 | -0.07 | -0.090 |
| 1s | 100 | market | 9 | 70759 | -0.21 | -0.285 |
| 1s | 100 | passive | 9 | 70759 | -0.06 | -0.081 |
| 1s | 200 | market | 9 | 70759 | -0.20 | -0.280 |
| 1s | 200 | passive | 9 | 70759 | -0.05 | -0.069 |
| 1s | 500 | market | 9 | 70759 | -0.20 | -0.281 |
| 1s | 500 | passive | 9 | 70759 | -0.05 | -0.069 |
| 1s | 1000 | market | 9 | 70759 | -0.22 | -0.301 |
| 1s | 1000 | passive | 9 | 70759 | -0.07 | -0.094 |
| 5s | 0 | market | 9 | 78991 | -0.24 | -0.155 |
| 5s | 0 | passive | 9 | 78991 | -0.10 | -0.061 |
| 5s | 50 | market | 9 | 78991 | -0.24 | -0.152 |
| 5s | 50 | passive | 9 | 78991 | -0.09 | -0.058 |
| 5s | 100 | market | 9 | 78991 | -0.23 | -0.149 |
| 5s | 100 | passive | 9 | 78991 | -0.09 | -0.054 |
| 5s | 200 | market | 9 | 78991 | -0.22 | -0.144 |
| 5s | 200 | passive | 9 | 78991 | -0.07 | -0.048 |
| 5s | 500 | market | 9 | 78991 | -0.22 | -0.142 |
| 5s | 500 | passive | 9 | 78991 | -0.07 | -0.047 |
| 5s | 1000 | market | 9 | 78991 | -0.25 | -0.158 |
| 5s | 1000 | passive | 9 | 78991 | -0.10 | -0.062 |
| 10s | 0 | market | 9 | 51542 | -0.28 | -0.131 |
| 10s | 0 | passive | 9 | 51542 | -0.14 | -0.061 |
| 10s | 50 | market | 9 | 51542 | -0.28 | -0.130 |
| 10s | 50 | passive | 9 | 51542 | -0.13 | -0.059 |
| 10s | 100 | market | 9 | 51542 | -0.27 | -0.126 |
| 10s | 100 | passive | 9 | 51542 | -0.13 | -0.055 |
| 10s | 200 | market | 9 | 51542 | -0.26 | -0.123 |
| 10s | 200 | passive | 9 | 51542 | -0.12 | -0.052 |
| 10s | 500 | market | 9 | 51542 | -0.26 | -0.118 |
| 10s | 500 | passive | 9 | 51542 | -0.11 | -0.047 |
| 10s | 1000 | market | 9 | 51542 | -0.26 | -0.121 |
| 10s | 1000 | passive | 9 | 51542 | -0.12 | -0.050 |
| 30s | 0 | market | 9 | 16914 | -0.11 | 0.068 |
| 30s | 0 | passive | 9 | 16914 | +0.04 | 0.108 |
| 30s | 50 | market | 9 | 16914 | -0.11 | 0.071 |
| 30s | 50 | passive | 9 | 16914 | +0.04 | 0.111 |
| 30s | 100 | market | 9 | 16914 | -0.11 | 0.065 |
| 30s | 100 | passive | 9 | 16914 | +0.04 | 0.105 |
| 30s | 200 | market | 9 | 16914 | -0.10 | 0.073 |
| 30s | 200 | passive | 9 | 16914 | +0.04 | 0.113 |
| 30s | 500 | market | 9 | 16914 | -0.09 | 0.075 |
| 30s | 500 | passive | 9 | 16914 | +0.05 | 0.116 |
| 30s | 1000 | market | 9 | 16914 | -0.08 | 0.100 |
| 30s | 1000 | passive | 9 | 16914 | +0.07 | 0.141 |

## HC #428 R1 — Regime Stratification Gate

Champion config = head_1s + lag_0 + top10% short + market exec.

- Green days Sharpe avg = -0.297 (n_days=4)
- Red days Sharpe avg = -0.288 (n_days=4)
- Flat days Sharpe avg = -0.283 (n_days=1)
- HC #428 R1 verdict: **PASS — green=-0.297 red=-0.288 gap=3.02%**

## HC #428 R2 — MFE-within-Horizon Gate

This Phase A does NOT impose explicit TP/SL — we use the full forward-drift PnL.
R2 gate (TP ≤ p90 of realized MFE within horizon h) is not directly applicable at this stage.
Note: forward drift over horizon h IS the closest to a 'no TP/SL' P&L — this is BY DESIGN
the maximum bounded result. Any future production config layering TP/SL on top must
re-test against R2.

## Conclusion


Champion (h=1s, lag=0, top10% short, market): 70759 trades across 9 days, net mean = -0.22 bps/trade, per-trade Sharpe = -0.291.

Best config sweep: **h=30s lag=1000ms mode=passive** with Sharpe = 0.141, net mean = +0.07 bps/trade.


Maximum weighted-mean cross-asset IC across all (lag, horizon) cells: **0.0206**.
For comparison, ES intra-asset IC(1s) on these days averages ~0.0868.

**Hypothesis verdict:** the cross-asset hypothesis **does NOT survive** Phase A. Predictive linkage between ES and SPY is detectable but materially weaker than ES intra-asset signal (IC ~5–10x lower in raw magnitude). After SPY costs, net Sharpe is non-positive. Phase B is justified only if a more efficient signal extraction (residualization, lag-specific models, or microstructure adapters) can lift IC by >=2x.