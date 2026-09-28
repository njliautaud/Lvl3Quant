# ES->SPY Lead-Lag IC Study -- REPORT (v3.4.2)

**Model**: CNN-Mamba **v3.4.2** (production champion per HC #529)  
**Weights**: `fold_03_intra_ckpt.pt` from `cnn_mamba_v3_4_2_hc477fix_v2`  
**Window/stride**: 1500/250 events (T1) + T2 100ms-buckets + T3 1Hz  
**Primary directional head**: log_ret_5s (concat IC ~0.108-0.141 OOT)  
**Lags (ms)**: [0, 50, 100, 200, 500, 1000]  
**Horizons (s)**: [1, 5, 10]  
**Days analyzed**: 9/9 (20260302, 20260303, 20260304, 20260305, 20260306, 20260309, 20260310, 20260311, 20260312)

## Headline
- **Decision**: NO-GO
- **Best (lag, horizon)**: `lag500_h1s` with concat IC = `0.0012`
- **Threshold**: concat IC >= 0.05 AND >= 5 days positive at that threshold

## Concat IC matrix (lag x horizon)

| lag\h | 1s | 5s | 10s |
|---|---|---|---|
| 0ms | -0.0018 | +0.0002 | -0.0016 |
| 50ms | +0.0003 | +0.0008 | -0.0010 |
| 100ms | +0.0000 | +0.0005 | -0.0011 |
| 200ms | +0.0007 | +0.0006 | -0.0014 |
| 500ms | +0.0012 | +0.0006 | -0.0023 |
| 1000ms | +0.0008 | +0.0001 | -0.0028 |

## Days passing threshold (count >= 0.05)

- `lag0_h1s`: 0/9 days >= 0.05
- `lag0_h5s`: 0/9 days >= 0.05
- `lag0_h10s`: 0/9 days >= 0.05
- `lag50_h1s`: 0/9 days >= 0.05
- `lag50_h5s`: 0/9 days >= 0.05
- `lag50_h10s`: 0/9 days >= 0.05
- `lag100_h1s`: 0/9 days >= 0.05
- `lag100_h5s`: 0/9 days >= 0.05
- `lag100_h10s`: 0/9 days >= 0.05
- `lag200_h1s`: 0/9 days >= 0.05
- `lag200_h5s`: 0/9 days >= 0.05
- `lag200_h10s`: 0/9 days >= 0.05
- `lag500_h1s`: 0/9 days >= 0.05
- `lag500_h5s`: 0/9 days >= 0.05
- `lag500_h10s`: 0/9 days >= 0.05
- `lag1000_h1s`: 0/9 days >= 0.05
- `lag1000_h5s`: 0/9 days >= 0.05
- `lag1000_h10s`: 0/9 days >= 0.05

## Per-day IC at best (lag, horizon)

| date | IC | n samples |
|---|---|---|
| 20260302 | -0.0056 | 57,494 |
| 20260303 | -0.0109 | 75,925 |
| 20260304 | +0.0125 | 42,982 |
| 20260305 | -0.0096 | 72,117 |
| 20260306 | +0.0208 | 75,481 |
| 20260309 | -0.0008 | 66,285 |
| 20260310 | +0.0075 | 64,028 |
| 20260311 | -0.0113 | 57,367 |
| 20260312 | +0.0090 | 60,786 |

## Why
No (lag, horizon) cleared the 0.05 concat-IC AND >=5-days-positive bar. Best was `lag500_h1s` at +0.0012. NO-GO on SPY-execution-only at the lags/horizons tested.

## Run notes
- ES events: `/home/nick/Lvl3Quant/data/processed/mbo_events_smart_v3`
- SPY raw DBN: `/home/nick/Lvl3Quant/data/raw/spy_mbo`
- v3.4.2 weights: `/home/nick/Lvl3Quant/output/cnn_mamba_v3_4_2_hc477fix_v2/fold_03_intra_ckpt.pt`
- MLflow run: `9d15449518ca4a00bf665a8661a89fa8`
