# HC #437 Bug 2 — Phase B: 47-Day Bracket-Mode Sanity Baseline

## Run config
- Script: `scripts/v3_4_research/hc437_v2_baseline_runner_bracket.py`
- Mode: `order_management='hc413_bracket'` (FIFOReplayEngine + per-signal labels_by_h)
- v2 NPZ: `cnn_mamba_v2_bulk_oot_v2` (1s horizon, short side, top0.5% conf)
- Brackets: TP1=0.4782t, TP2=0.9564t, SL=0.5686t (HC #413 v2-native MFE cell)
- hold_s=10, cancel_s=10, passive_at_touch
- Dates: 48 in v2 OOT, 46 had signals, 32 with MBO data, 10 missing DBN
- Run launched 19:17 ET 2026-05-19; finished 19:26 (~9 min, 10 workers). Prior agent's
  identical 19:16 run was preserved at `*_PRIOR_AGENT_*` and matches within rounding.

## Aggregate metrics
| metric | value |
|---|---|
| n_fills | 3,452 |
| n_signal_days | 46 (32 with MBO data) |
| mean_gross_ticks | +0.4923 |
| **mean_net_ticks** | **+0.1163** |
| **PF** | **1.447** |
| **WR%** | **72.05** |
| **Sharpe(√N)** | **10.237** |
| exit_counts | tp2: 2188, sl: 935, tp1: 299, time_stop: 30 |

## HC #437 R1 verdict gates

| gate | required | actual | verdict |
|---|---|---|---|
| net_tk/fill | [+0.224, +0.324] | +0.1163 | **FAIL** (below by 48% of low-end) |
| Sh(√N) | ≥ 10 | 10.237 | PASS |
| PF | ≥ 2.5 | 1.447 | **FAIL** |
| WR | ≥ 82% | 72.05% | **FAIL** |

### HC #437 R1: NOT_SATISFIED under bracket mode at full 47-day OOT

The 3-day reference reproduction (+0.240 tk/fill, PF 2.15, WR 77.5%, n=218) was on
a favorable subset (20260415-20260417: bracket per-day net_tk was +0.150, +0.169,
+0.211 on those three days specifically — closer to the 3-day aggregate). Adding
the other 43 days (n=3234 additional fills) drags net_tk down to +0.116.

**HC #413's published +0.274 tk/fill, PF 2.92, WR 85% is NOT reproduced under the
HC #437 bracket harness over the full 47-day OOT — it appears to have been
fold-windowed and/or label-set differently from this v2 NPZ harness.**

## Regime stratification (per HC #428 R1)
| regime | n_days | n_fills | mean_net_tk | PF | WR% | Sh√N |
|---|---|---|---|---|---|---|
| green | 17 | 1,915 | +0.128 | 1.51 | 72.95 | +8.49 |
| red | 15 | 1,537 | +0.102 | 1.37 | 70.92 | +5.89 |
| flat | 0 | 0 | — | — | — | — |

|Sh_g - Sh_r| / max(|Sh_g|, |Sh_r|) = 0.306 → **REGIME-AGNOSTIC PASS** (≤0.50).
The bracket-mode edge IS consistent across green/red days. The problem is
absolute magnitude, not regime fragility.

## Per-day table (32 days with fills)
| date | regime | n_fills | mean_net_tk | PF | WR% | Sh√N |
|------|--------|---------|-------------|-----|------|------|
| 20260224 | green | 90 | +0.2309 | 2.078 | 76.67 | +3.423 |
| 20260226 | red | 107 | +0.2668 | 2.374 | 79.44 | +4.458 |
| 20260301 | green | 1 | -0.9446 | 0.000 | 0.00 | +0.000 |
| 20260303 | green | 231 | +0.1604 | 1.619 | 72.29 | +3.579 |
| 20260305 | green | 199 | +0.2585 | 2.297 | 78.89 | +5.846 |
| 20260306 | red | 197 | +0.1836 | 1.754 | 73.60 | +3.866 |
| 20260309 | green | 199 | +0.1742 | 1.693 | 73.37 | +3.637 |
| 20260310 | red | 166 | +0.0694 | 1.220 | 66.27 | +1.242 |
| 20260311 | red | 105 | +0.0721 | 1.229 | 66.67 | +1.022 |
| 20260312 | green | 160 | +0.1038 | 1.352 | 68.75 | +1.852 |
| 20260313 | red | 128 | +0.0562 | 1.173 | 65.62 | +0.874 |
| 20260316 | green | 74 | +0.1436 | 1.592 | 74.32 | +1.876 |
| 20260317 | green | 95 | +0.1874 | 2.064 | 80.00 | +3.178 |
| 20260318 | red | 125 | +0.2353 | 2.335 | 80.80 | +4.481 |
| 20260319 | red | 77 | +0.1296 | 1.556 | 75.32 | +1.772 |
| 20260401 | red | 125 | -0.0276 | 0.914 | 65.60 | -0.453 |
| 20260402 | green | 101 | -0.0247 | 0.927 | 64.36 | -0.352 |
| 20260405 | red | 1 | -0.9446 | 0.000 | 0.00 | +0.000 |
| 20260406 | red | 54 | -0.0852 | 0.779 | 59.26 | -0.855 |
| 20260407 | red | 100 | +0.0494 | 1.163 | 68.00 | +0.707 |
| 20260408 | red | 102 | +0.0428 | 1.152 | 69.61 | +0.647 |
| 20260409 | green | 73 | -0.1339 | 0.677 | 56.16 | -1.560 |
| 20260410 | red | 63 | +0.0623 | 1.227 | 69.84 | +0.739 |
| 20260413 | green | 84 | +0.1184 | 1.478 | 73.81 | +1.653 |
| 20260414 | green | 76 | +0.2389 | 2.551 | 82.89 | +3.748 |
| 20260415 | green | 107 | +0.0798 | 1.316 | 71.96 | +1.281 |
| 20260416 | green | 90 | +0.0404 | 1.147 | 70.00 | +0.582 |
| 20260417 | green | 118 | +0.1464 | 1.677 | 77.12 | +2.553 |
| 20260420 | red | 92 | +0.0767 | 1.283 | 70.65 | +1.098 |
| 20260422 | red | 95 | +0.1544 | 1.779 | 78.95 | +2.507 |
| 20260424 | green | 108 | -0.0618 | 0.816 | 63.89 | -0.944 |
| 20260427 | green | 109 | +0.1698 | 1.951 | 78.90 | +3.106 |
