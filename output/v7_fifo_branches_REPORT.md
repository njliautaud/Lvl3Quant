# HC #491 R1 — v7 FIFO Branch Evaluation Report

Date: 2026-05-28
Predictions: v7 prod concat (concat_oot_predictions.npz)
17 OOT dates with local DBN tape
Config: TP=2.0t SL=1.0t hold≤1.5s
Base harness cancel default: 30s | v7 grade override: 1.0s | Branch B override: 0.25s
Cost: passive limit = 0.376t (commission only), no crossing cost

## Baseline (v7 base, both sides, top 5%, cancel=1.0s)
| Metric | Value |
|--------|-------|
| Net ticks/trade | -0.621 |
| WR | 29.5% |
| PF | 0.334 |
| Positive days | 0/17 |
| Sharpe (ann) | -25.3 |
| Verdict | HARD REJECT |

## Branch A (short-only, 5%, 1s cancel)

| Metric | Value |
|--------|-------|
| N trades | 9,497 |
| N days | 17 |
| Positive days | 0/17 |
| Net ticks/trade | -0.6281 |
| WR | 29.1% |
| PF | 0.3308 |
| Sharpe (ann) | -24.9331 |
| Sortino (ann) | -24.9331 |
| Regime check | Sharpe_green=-28.90, Sharpe_red=-13.21, skew=0.54 → FAIL |

### Per-day breakdown
| Date | N trades | PnL ticks | WR | Positive |
|------|----------|-----------|-----|---------|
| 20260401 | 929 | -570.80 | 29.3% | NO |
| 20260402 | 1132 | -675.63 | 27.4% | NO |
| 20260403 | 11 | -3.64 | 36.4% | NO |
| 20260405 | 12 | -2.01 | 50.0% | NO |
| 20260406 | 632 | -337.63 | 30.9% | NO |
| 20260407 | 1129 | -632.50 | 29.2% | NO |
| 20260408 | 935 | -572.56 | 28.2% | NO |
| 20260409 | 677 | -478.05 | 27.3% | NO |
| 20260410 | 523 | -355.65 | 29.8% | NO |
| 20260412 | 16 | -9.52 | 31.2% | NO |
| 20260413 | 571 | -307.70 | 34.7% | NO |
| 20260414 | 438 | -343.19 | 26.0% | NO |
| 20260415 | 515 | -356.64 | 31.5% | NO |
| 20260416 | 588 | -413.59 | 28.7% | NO |
| 20260417 | 758 | -468.01 | 30.3% | NO |
| 20260419 | 10 | -10.76 | 10.0% | NO |
| 20260420 | 621 | -427.50 | 26.9% | NO |

**Verdict: REJECT — n=9497, 0/17 positive days, net=-0.628t/trade, WR=29.1%, PF=0.331, Sharpe=-24.93 | Sharpe_green=-28.90, Sharpe_red=-13.21, skew=0.54 → FAIL**

## Branch B (both, 5%, 0.25s cancel)

| Metric | Value |
|--------|-------|
| N trades | 10,910 |
| N days | 17 |
| Positive days | 0/17 |
| Net ticks/trade | -0.6163 |
| WR | 29.4% |
| PF | 0.3434 |
| Sharpe (ann) | -24.5113 |
| Sortino (ann) | -24.5113 |
| Regime check | Sharpe_green=-27.99, Sharpe_red=-13.43, skew=0.52 → FAIL |

### Per-day breakdown
| Date | N trades | PnL ticks | WR | Positive |
|------|----------|-----------|-----|---------|
| 20260401 | 1035 | -623.66 | 29.2% | NO |
| 20260402 | 1329 | -814.20 | 26.9% | NO |
| 20260403 | 15 | -11.64 | 6.7% | NO |
| 20260405 | 8 | -3.51 | 37.5% | NO |
| 20260406 | 696 | -360.20 | 30.9% | NO |
| 20260407 | 1348 | -749.85 | 29.0% | NO |
| 20260408 | 1081 | -640.46 | 29.4% | NO |
| 20260409 | 806 | -504.06 | 29.8% | NO |
| 20260410 | 578 | -399.33 | 29.6% | NO |
| 20260412 | 19 | -9.64 | 36.8% | NO |
| 20260413 | 662 | -407.41 | 32.2% | NO |
| 20260414 | 500 | -364.00 | 28.0% | NO |
| 20260415 | 561 | -389.44 | 29.4% | NO |
| 20260416 | 665 | -469.04 | 28.6% | NO |
| 20260417 | 842 | -509.09 | 31.2% | NO |
| 20260419 | 13 | -7.89 | 23.1% | NO |
| 20260420 | 752 | -460.25 | 29.8% | NO |

**Verdict: REJECT — n=10910, 0/17 positive days, net=-0.616t/trade, WR=29.4%, PF=0.343, Sharpe=-24.51 | Sharpe_green=-27.99, Sharpe_red=-13.43, skew=0.52 → FAIL**

## Branch C1 (short, top1%, 1s cancel)

| Metric | Value |
|--------|-------|
| N trades | 1,913 |
| N days | 14 |
| Positive days | 1/14 |
| Net ticks/trade | -0.6332 |
| WR | 28.9% |
| PF | 0.3257 |
| Sharpe (ann) | -33.8613 |
| Sortino (ann) | -41.2862 |
| Regime check | Sharpe_green=-38.35, Sharpe_red=-18.25, skew=0.52 → FAIL |

### Per-day breakdown
| Date | N trades | PnL ticks | WR | Positive |
|------|----------|-----------|-----|---------|
| 20260401 | 286 | -149.04 | 32.2% | NO |
| 20260402 | 304 | -174.80 | 28.0% | NO |
| 20260403 | 2 | +0.25 | 50.0% | YES |
| 20260406 | 112 | -70.11 | 25.9% | NO |
| 20260407 | 118 | -60.87 | 31.4% | NO |
| 20260408 | 149 | -103.02 | 25.5% | NO |
| 20260409 | 136 | -89.14 | 30.9% | NO |
| 20260410 | 125 | -99.00 | 24.8% | NO |
| 20260413 | 104 | -77.10 | 28.8% | NO |
| 20260414 | 90 | -63.34 | 27.8% | NO |
| 20260415 | 111 | -79.24 | 31.5% | NO |
| 20260416 | 113 | -77.49 | 28.3% | NO |
| 20260417 | 131 | -80.76 | 30.5% | NO |
| 20260420 | 132 | -87.63 | 27.3% | NO |

**Verdict: REJECT — n=1913, 1/14 positive days, net=-0.633t/trade, WR=28.9%, PF=0.326, Sharpe=-33.86 | Sharpe_green=-38.35, Sharpe_red=-18.25, skew=0.52 → FAIL**

## Branch C2 (short, top2%, 1s cancel)

| Metric | Value |
|--------|-------|
| N trades | 3,783 |
| N days | 16 |
| Positive days | 2/16 |
| Net ticks/trade | -0.6328 |
| WR | 29.2% |
| PF | 0.3250 |
| Sharpe (ann) | -24.6672 |
| Sortino (ann) | -29.0051 |
| Regime check | Sharpe_green=-28.81, Sharpe_red=-13.76, skew=0.52 → FAIL |

### Per-day breakdown
| Date | N trades | PnL ticks | WR | Positive |
|------|----------|-----------|-----|---------|
| 20260401 | 554 | -311.80 | 31.4% | NO |
| 20260402 | 573 | -332.95 | 28.3% | NO |
| 20260403 | 3 | +1.87 | 66.7% | YES |
| 20260405 | 2 | -2.75 | 0.0% | NO |
| 20260406 | 208 | -127.71 | 26.9% | NO |
| 20260407 | 295 | -189.42 | 27.8% | NO |
| 20260408 | 330 | -208.58 | 27.0% | NO |
| 20260409 | 272 | -192.27 | 28.7% | NO |
| 20260410 | 227 | -185.85 | 24.2% | NO |
| 20260413 | 194 | -99.44 | 37.1% | NO |
| 20260414 | 179 | -134.30 | 25.7% | NO |
| 20260415 | 237 | -164.61 | 31.6% | NO |
| 20260416 | 213 | -133.09 | 31.0% | NO |
| 20260417 | 259 | -165.38 | 30.1% | NO |
| 20260419 | 2 | +0.25 | 50.0% | YES |
| 20260420 | 235 | -147.86 | 29.4% | NO |

**Verdict: REJECT — n=3783, 2/16 positive days, net=-0.633t/trade, WR=29.2%, PF=0.325, Sharpe=-24.67 | Sharpe_green=-28.81, Sharpe_red=-13.76, skew=0.52 → FAIL**

## Branch D (short, top2%, 0.25s cancel)

| Metric | Value |
|--------|-------|
| N trades | 2,130 |
| N days | 16 |
| Positive days | 0/16 |
| Net ticks/trade | -0.6258 |
| WR | 29.4% |
| PF | 0.3365 |
| Sharpe (ann) | -23.5990 |
| Sortino (ann) | -23.5990 |
| Regime check | Sharpe_green=-26.87, Sharpe_red=-14.04, skew=0.48 → PASS |

### Per-day breakdown
| Date | N trades | PnL ticks | WR | Positive |
|------|----------|-----------|-----|---------|
| 20260401 | 307 | -174.93 | 30.6% | NO |
| 20260402 | 339 | -202.96 | 27.1% | NO |
| 20260403 | 1 | -1.38 | 0.0% | NO |
| 20260405 | 1 | -1.38 | 0.0% | NO |
| 20260406 | 114 | -58.86 | 30.7% | NO |
| 20260407 | 165 | -93.54 | 29.7% | NO |
| 20260408 | 197 | -116.57 | 27.4% | NO |
| 20260409 | 157 | -122.03 | 25.5% | NO |
| 20260410 | 127 | -102.75 | 26.0% | NO |
| 20260413 | 111 | -71.24 | 32.4% | NO |
| 20260414 | 99 | -73.72 | 27.3% | NO |
| 20260415 | 133 | -90.51 | 33.1% | NO |
| 20260416 | 110 | -71.86 | 30.0% | NO |
| 20260417 | 136 | -73.64 | 33.8% | NO |
| 20260419 | 1 | -1.38 | 0.0% | NO |
| 20260420 | 132 | -76.13 | 32.6% | NO |

**Verdict: REJECT — n=2130, 0/16 positive days, net=-0.626t/trade, WR=29.4%, PF=0.337, Sharpe=-23.60 | Sharpe_green=-26.87, Sharpe_red=-14.04, skew=0.48 → PASS**


## HC #491 R2 — Fill Verification
```

--- Verify Branch A (short-only, 5%, 1s cancel) ---
Total fills: 9,497 (non-zero: 9,497)
First 3 fills:
  date=20260405 dir=short net=-1.376t type=sl strength=0.8427
  date=20260405 dir=short net=-1.376t type=sl strength=0.9159
  date=20260405 dir=short net=-1.376t type=sl strength=0.7866

--- Verify Branch B (both, 5%, 0.25s cancel) ---
Total fills: 10,910 (non-zero: 10,910)
First 3 fills:
  date=20260405 dir=short net=-1.376t type=sl strength=0.8427
  date=20260405 dir=short net=+0.624t type=max_hold strength=0.8219
  date=20260405 dir=short net=+0.624t type=max_hold strength=0.8051

--- Verify Branch C1 (short, top1%, 1s cancel) ---
Total fills: 1,913 (non-zero: 1,913)
First 3 fills:
  date=20260403 dir=short net=+1.624t type=tp strength=0.9912
  date=20260403 dir=short net=-1.376t type=sl strength=0.9261
  date=20260410 dir=short net=-1.376t type=sl strength=1.0639

--- Verify Branch C2 (short, top2%, 1s cancel) ---
Total fills: 3,783 (non-zero: 3,783)
First 3 fills:
  date=20260405 dir=short net=-1.376t type=sl strength=0.9159
  date=20260405 dir=short net=-1.376t type=sl strength=0.9064
  date=20260403 dir=short net=+1.624t type=tp strength=0.8670

--- Verify Branch D (short, top2%, 0.25s cancel) ---
Total fills: 2,130 (non-zero: 2,130)
First 3 fills:
  date=20260405 dir=short net=-1.376t type=sl strength=0.9064
  date=20260403 dir=short net=-1.376t type=sl strength=0.9261
  date=20260406 dir=short net=+1.624t type=tp strength=0.8663
```

## Summary & Recommendation

- **A (short-only, 5%, 1s cancel)**: REJECT — n=9497, 0/17 positive days, net=-0.628t/trade, WR=29.1%, PF=0.331, Sharpe=-24.93
- **B (both, 5%, 0.25s cancel)**: REJECT — n=10910, 0/17 positive days, net=-0.616t/trade, WR=29.4%, PF=0.343, Sharpe=-24.51
- **C1 (short, top1%, 1s cancel)**: REJECT — n=1913, 1/14 positive days, net=-0.633t/trade, WR=28.9%, PF=0.326, Sharpe=-33.86
- **C2 (short, top2%, 1s cancel)**: REJECT — n=3783, 2/16 positive days, net=-0.633t/trade, WR=29.2%, PF=0.325, Sharpe=-24.67
- **D (short, top2%, 0.25s cancel)**: REJECT — n=2130, 0/16 positive days, net=-0.626t/trade, WR=29.4%, PF=0.337, Sharpe=-23.60

### All branches REJECTED.

Root cause: adverse selection under FIFO persists across all variants. The 1s prediction horizon is too short for limit order queue-wait dynamics. Even short-only, tight-cancel, and top-1% confidence fail to overcome the systematic adverse fill bias (-0.55 slippage ticks average in base).

**Next axis per HC #488 R2 (axis rotation):**

**MODEL AXIS** — retrain on longer horizon (5s or 10s target), which gives the limit order time to sit in queue WITHOUT being adversely selected. The 5s CNN-Mamba v2 predictions (preds[:,1]) show IC=0.141 and MFE extends to 30s — far better alignment with passive limit fill mechanics. Test: re-run the same harness using preds[:,1] (5s horizon) with hold≤7.5s, cancel≤5s, TP≤p90 MFE(5s). This is the highest-probability next move given: (a) signal still exists at 5s, (b) queue wait ~370ms avg is a small fraction of 5s, (c) short edge is strongest at 5s per decay analysis.