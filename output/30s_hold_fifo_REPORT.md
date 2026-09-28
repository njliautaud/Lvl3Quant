# 30s-Hold FIFO Grading Report (existing preds, wider TP/SL)

Date: 2026-05-28
Hypothesis: hold longer (30s), TP at p90 MFE(30s)≈12t, SL near p50 MAE(30s)≈3t.
Same 17 OOT dates as morning v7_fifo_branches.
Cost: passive limit = 0.376t (commission only).
MLflow experiment: 30s_hold_fifo_existing_preds

## cell1_v7_top1_TP12_SL3

Config: TP=12.0t SL=3.0t hold=30.0s cancel=2.0s

| Metric | Value |
|--------|-------|
| N trades | 4,604 |
| N days | 15 |
| Positive days | 0/15 |
| Net ticks/trade | -0.4059 |
| WR | 32.5% |
| PF | 0.812 |
| Sharpe (ann) | -19.578 |
| Sortino (ann) | -19.578 |
| TP / SL / MaxHold / EOD | 8.2% / 62.3% / 29.5% / 0.0% |
| Avg time-to-fill | 501 ms |
| Avg time-in-trade | 15.22 s |
| Regime Sharpe (G / R / skew) | -18.20 / -16.62 / 0.09 |
| **Verdict** | **REJECT** — fails acceptance gates |

## cell2_v7_top05_TP12_SL3

Config: TP=12.0t SL=3.0t hold=30.0s cancel=2.0s

| Metric | Value |
|--------|-------|
| N trades | 2,296 |
| N days | 15 |
| Positive days | 4/15 |
| Net ticks/trade | -0.2682 |
| WR | 34.2% |
| PF | 0.873 |
| Sharpe (ann) | -10.464 |
| Sortino (ann) | -13.562 |
| TP / SL / MaxHold / EOD | 8.5% / 60.8% / 30.7% / 0.0% |
| Avg time-to-fill | 500 ms |
| Avg time-in-trade | 15.42 s |
| Regime Sharpe (G / R / skew) | -9.93 / -7.00 / 0.29 |
| **Verdict** | **REJECT** — fails acceptance gates |

## cell3_v2h10_top1_TP12_SL3

Config: TP=12.0t SL=3.0t hold=30.0s cancel=2.0s

| Metric | Value |
|--------|-------|
| N trades | 5,288 |
| N days | 13 |
| Positive days | 0/13 |
| Net ticks/trade | -0.5553 |
| WR | 32.9% |
| PF | 0.739 |
| Sharpe (ann) | -25.408 |
| Sortino (ann) | -25.408 |
| TP / SL / MaxHold / EOD | 6.5% / 60.9% / 32.6% / 0.0% |
| Avg time-to-fill | 354 ms |
| Avg time-in-trade | 16.06 s |
| Regime Sharpe (G / R / skew) | -23.70 / -21.95 / 0.07 |
| **Verdict** | **REJECT** — fails acceptance gates |

## cell4_v2h10_short_top1_TP12_SL3

Config: TP=12.0t SL=3.0t hold=30.0s cancel=2.0s

| Metric | Value |
|--------|-------|
| N trades | 1,630 |
| N days | 14 |
| Positive days | 1/14 |
| Net ticks/trade | -0.8334 |
| WR | 28.8% |
| PF | 0.639 |
| Sharpe (ann) | -23.523 |
| Sortino (ann) | -28.905 |
| TP / SL / MaxHold / EOD | 6.4% / 66.9% / 26.7% / 0.0% |
| Avg time-to-fill | 499 ms |
| Avg time-in-trade | 14.18 s |
| Regime Sharpe (G / R / skew) | -24.60 / -20.51 / 0.17 |
| **Verdict** | **REJECT** — fails acceptance gates |

## cell5_v7_top1_TP8_SL2

Config: TP=8.0t SL=2.0t hold=30.0s cancel=2.0s

| Metric | Value |
|--------|-------|
| N trades | 4,604 |
| N days | 15 |
| Positive days | 1/15 |
| Net ticks/trade | -0.4695 |
| WR | 25.2% |
| PF | 0.733 |
| Sharpe (ann) | -24.981 |
| Sortino (ann) | -26.866 |
| TP / SL / MaxHold / EOD | 13.2% / 73.7% / 13.1% / 0.0% |
| Avg time-to-fill | 501 ms |
| Avg time-in-trade | 9.43 s |
| Regime Sharpe (G / R / skew) | -26.74 / -14.05 / 0.47 |
| **Verdict** | **REJECT** — fails acceptance gates |

## cell6_v2h10_top1_TP8_SL2

Config: TP=8.0t SL=2.0t hold=30.0s cancel=2.0s

| Metric | Value |
|--------|-------|
| N trades | 5,288 |
| N days | 13 |
| Positive days | 0/13 |
| Net ticks/trade | -0.6112 |
| WR | 24.5% |
| PF | 0.654 |
| Sharpe (ann) | -30.948 |
| Sortino (ann) | -30.948 |
| TP / SL / MaxHold / EOD | 10.8% / 73.8% / 15.4% / 0.0% |
| Avg time-to-fill | 354 ms |
| Avg time-in-trade | 10.30 s |
| Regime Sharpe (G / R / skew) | -29.42 / -29.17 / 0.01 |
| **Verdict** | **REJECT** — fails acceptance gates |


## HC #491 R2 — Fill Verification
```
Source v7 preds:  /home/jupiter/Lvl3Quant/output/meta_v7_prod/concat_oot_predictions.npz
Source v2 dir:    /home/jupiter/Lvl3Quant/output/cnn_mamba_v2_bulk_oot_v2 (h=10s column)
MBO event dir:    /home/jupiter/Lvl3Quant/data/processed/mbo_events_smart_v3

--- cell1_v7_top1_TP12_SL3 ---
Total fills: 4,604
TP_rate=8.2% SL_rate=62.3% MH_rate=29.5% EOD_rate=0.0%
First 3 fills:
  date=20260403 dir=short net=-3.376t type=sl hold=0.04s wait=0.26s strength=0.9912
  date=20260403 dir=long net=+0.624t type=max_hold hold=30.08s wait=0.37s strength=1.0343
  date=20260403 dir=long net=-0.376t type=max_hold hold=32.45s wait=0.68s strength=0.9387

--- cell2_v7_top05_TP12_SL3 ---
Total fills: 2,296
TP_rate=8.5% SL_rate=60.8% MH_rate=30.7% EOD_rate=0.0%
First 3 fills:
  date=20260403 dir=long net=+0.624t type=max_hold hold=30.08s wait=0.37s strength=1.0343
  date=20260412 dir=long net=-3.376t type=sl hold=12.29s wait=1.50s strength=1.0706
  date=20260412 dir=long net=-3.376t type=sl hold=4.47s wait=0.83s strength=0.9988

--- cell3_v2h10_top1_TP12_SL3 ---
Total fills: 5,288
TP_rate=6.5% SL_rate=60.9% MH_rate=32.6% EOD_rate=0.0%
First 3 fills:
  date=20260410 dir=long net=+2.624t type=max_hold hold=39.70s wait=0.65s strength=1.6608
  date=20260410 dir=long net=+1.124t type=max_hold hold=30.36s wait=2.45s strength=2.1008
  date=20260410 dir=long net=-0.876t type=max_hold hold=32.74s wait=1.99s strength=1.5834

--- cell4_v2h10_short_top1_TP12_SL3 ---
Total fills: 1,630
TP_rate=6.4% SL_rate=66.9% MH_rate=26.7% EOD_rate=0.0%
First 3 fills:
  date=20260405 dir=short net=-3.376t type=sl hold=9.85s wait=0.10s strength=0.9730
  date=20260405 dir=short net=-3.376t type=sl hold=0.99s wait=1.28s strength=0.9468
  date=20260410 dir=short net=-3.376t type=sl hold=18.63s wait=1.92s strength=0.9882

--- cell5_v7_top1_TP8_SL2 ---
Total fills: 4,604
TP_rate=13.2% SL_rate=73.7% MH_rate=13.1% EOD_rate=0.0%
First 3 fills:
  date=20260403 dir=short net=-2.376t type=sl hold=0.04s wait=0.26s strength=0.9912
  date=20260403 dir=long net=+7.624t type=tp hold=15.74s wait=0.37s strength=1.0343
  date=20260403 dir=long net=-2.376t type=sl hold=18.03s wait=0.68s strength=0.9387

--- cell6_v2h10_top1_TP8_SL2 ---
Total fills: 5,288
TP_rate=10.8% SL_rate=73.8% MH_rate=15.4% EOD_rate=0.0%
First 3 fills:
  date=20260410 dir=long net=+2.624t type=max_hold hold=39.70s wait=0.65s strength=1.6608
  date=20260410 dir=long net=+1.124t type=max_hold hold=30.36s wait=2.45s strength=2.1008
  date=20260410 dir=long net=-2.376t type=sl hold=20.08s wait=1.99s strength=1.5834
```

## Honest Failure-Mode Statement

All 6 cells REJECT. The p90 MFE = ~13t observation within 30s is a marginal-distribution artifact: in the actual price path the adverse excursion (MAE) generally comes BEFORE the favorable one, so SL is hit before TP. The signal cannot be captured with a fixed TP/SL bracket at this horizon. Re-running the morning data with a wider window does NOT change the path-dependent reality. Next axis must be the model (longer-horizon target with conditional MAE-first filter) or the exit logic (dynamic trailing TP, adverse-excursion-conditional exit).