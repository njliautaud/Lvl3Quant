# macro_regime_rotation_v1 — regime-conditioned sector rotation
_Run timestamp: 20260609_172010_

## TL;DR
- **Strategy CAGR**: 9.5% | **Sharpe**: 0.69 | **Sortino**: 0.88 | **MaxDD**: -21.6% | **Calmar**: 0.44 | **PF**: 1.13 | **WR**: 54.0%
- **SPY B&H**: CAGR 12.5% | Sharpe 0.70 | MaxDD -34.1%
- **Unconditional baseline (no regime gating)**: CAGR 9.0% | Sharpe 0.54 | MaxDD -33.1%

## CRITICAL STRUCTURAL FINDING (HC #428 R1 vs long-only)
SPY-self gap = **1.92**. 60/40 SPY/TLT gap = **1.93**. Our regime-rotation gap = **1.87**.
Any long-only equity strategy partitioned by SPY green/red days has Sh_green ~+15 and
Sh_red ~-15 mechanically — you're long, you make money when market is up, lose when down.
HC #428 R1 (gap ≤ 0.50) is structurally **un-passable for any directional long-only book**.
It implicitly requires a long/short or actively hedged design. The regime-rotation IS
slightly less directional than SPY (gap 1.87 < SPY's 1.92), but it cannot satisfy 0.50.
This means HC #428 R1 should EITHER be reinterpreted for long-only (e.g. apply to per-DAY
*excess* return over SPY) OR this lane must be redesigned as long/short.

## HC #428 R1 — Regime-Agnostic Gate
- **Green-day Sharpe**: 13.86 (n=992)
- **Red-day Sharpe**:   -12.03 (n=797)
- **Flat-day Sharpe**:  0.17 (n=222)
- **Gap = |Sh_g - Sh_r| / max = 1.868** (need ≤ 0.50)
- **HC #428 R1**: FAIL

## Baseline HC #428 R1 (for comparison)
- Green Sharpe 11.53 | Red Sharpe -11.06 | Gap 1.959 | FAIL

## Day Concentration
- Strategy: 0.0043 (need ≤ 0.70)

## Regime Classifier Spot Checks
- **COVID Mar-Apr 2020**: {'RECESSION': np.int64(33)}
- **2022 bear**: {'LATE': np.int64(122), 'RECESSION': np.int64(99), 'MID': np.int64(28), 'EARLY': np.int64(11)}
- **Mid 2023 → 2024**: {'MID': np.int64(276), 'LATE': np.int64(81), 'RECESSION': np.int64(23), 'EARLY': np.int64(12)}
- Regime transitions: **106** over 8.0y → **13.3/yr**

## Per-Regime Performance (strategy)
| Regime | n_days | Sharpe (ann) | Mean daily | Cum return |
|---|---|---|---|---|
| EARLY | 129 | -0.33 | -0.021% | -3.3% |
| MID | 1002 | 1.79 | 0.115% | 201.0% |
| LATE | 484 | 0.80 | 0.038% | 18.5% |
| RECESSION | 396 | -2.28 | -0.125% | -40.0% |

## Regime Transition Timeline
| Start | End | Regime | Days |
|---|---|---|---|
| 2022-03-16 | 2022-04-22 | LATE | 37 |
| 2022-04-22 | 2022-05-27 | RECESSION | 35 |
| 2022-05-27 | 2022-06-13 | EARLY | 17 |
| 2022-06-13 | 2022-07-01 | RECESSION | 18 |
| 2022-07-01 | 2022-08-01 | MID | 31 |
| 2022-08-01 | 2022-09-23 | LATE | 53 |
| 2022-09-23 | 2022-10-24 | RECESSION | 31 |
| 2022-10-24 | 2022-11-02 | MID | 9 |
| 2022-11-02 | 2022-12-01 | LATE | 29 |
| 2022-12-01 | 2023-02-01 | RECESSION | 62 |
| 2023-02-01 | 2023-03-13 | LATE | 40 |
| 2023-03-13 | 2023-06-05 | MID | 84 |
| 2023-06-05 | 2023-08-17 | LATE | 73 |
| 2023-08-17 | 2023-11-24 | MID | 99 |
| 2023-11-24 | 2023-12-12 | EARLY | 18 |
| 2023-12-12 | 2024-01-01 | MID | 20 |
| 2024-01-01 | 2024-02-01 | RECESSION | 31 |
| 2024-02-01 | 2024-04-04 | MID | 63 |
| 2024-04-04 | 2024-04-23 | LATE | 19 |
| 2024-04-23 | 2024-05-06 | MID | 13 |
| 2024-05-06 | 2024-05-28 | LATE | 22 |
| 2024-05-28 | 2024-06-04 | MID | 7 |
| 2024-06-04 | 2024-06-25 | LATE | 21 |
| 2024-06-25 | 2024-12-13 | MID | 171 |
| 2024-12-13 | 2024-12-19 | LATE | 6 |
| 2024-12-19 | 2025-04-03 | MID | 105 |
| 2025-04-03 | 2025-04-25 | RECESSION | 22 |
| 2025-04-25 | 2025-06-10 | MID | 46 |
| 2025-06-10 | 2025-07-01 | EARLY | 21 |
| 2025-07-01 | 2025-07-21 | MID | 20 |
| 2025-07-21 | 2025-08-01 | LATE | 11 |
| 2025-08-01 | 2025-08-21 | MID | 20 |
| 2025-08-21 | 2025-08-29 | EARLY | 8 |
| 2025-08-29 | 2025-10-10 | MID | 42 |
| 2025-10-10 | 2025-10-31 | RECESSION | 21 |
| 2025-10-31 | 2025-11-07 | MID | 7 |
| 2025-11-07 | 2025-11-26 | LATE | 19 |
| 2025-11-26 | 2026-03-09 | MID | 103 |
| 2026-03-09 | 2026-03-31 | LATE | 22 |
| 2026-03-31 | 2026-06-05 | MID | 66 |

## Verdict & Failure-Mode Diagnosis
- **HC #428 R1**: FAIL (gap 1.868)
- **Day-concentration ≤ 0.70**: PASS
- **Sharpe vs SPY**: strategy 0.69 vs SPY 0.70 → loses to SPY
- **Strategy vs Baseline (regime gating value)**: strat Sharpe 0.69 vs unconditional 0.54

### Failure mode
- Per-regime Sharpe spread: EARLY=-0.33, MID=1.79, LATE=0.80, RECESSION=-2.28
- If RECESSION Sharpe is negative AND classifier called COVID correctly → **within-regime allocation wrong** (defensives still bled).
- If RECESSION Sharpe is OK but green/red gap is huge → **classifier mis-timing transitions** (caught crisis too late or too early).
- If MID Sharpe dominates everything → strategy is ~MID-only and not really regime-aware; the tilts are not differentiated enough.

## Files
- Summary JSON: `/home/jupiter/Lvl3Quant/output/macro_regime/regime_rotation_v1_20260609_172010/summary.json`
- Daily returns: `/home/jupiter/Lvl3Quant/output/macro_regime/regime_rotation_v1_20260609_172010/daily_returns.parquet`
- Regime timeline: `/home/jupiter/Lvl3Quant/output/macro_regime/regime_rotation_v1_20260609_172010/regime_timeline.parquet`
- Holdings log: `/home/jupiter/Lvl3Quant/output/macro_regime/regime_rotation_v1_20260609_172010/holdings_log.parquet`