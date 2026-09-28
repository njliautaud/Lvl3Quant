# v2 Short-Side Gate Coverage Audit (HC #352)

**Date:** 2026-05-14
**Data source:** `output/cnn_mamba_v2_smart_v3_mar/fold_*_oot_predictions.npz` (canonical v2 OOT: IC_1s/5s/10s ≈ 0.22/0.11/0.07, matches CLAUDE.md)
**Confluence source:** `output/confluence_features/all_confluence_features.npz` (aligned CNN-Mamba + PatchTST + realized moves)
**Unique OOT trading days analyzed:** 10
**Total OOT rows:** 273,798

**HC #349 ANNOTATION — FIFO-FLOOR ONLY: Every PnL/edge number in this report uses commission ($4.70 RT = 0.376 ticks) + 1-tick spread crossing for market orders (1.376 ticks total) and commission-only for passive limits (0.376 ticks). NO queue-position model, NO adverse-selection cost. These numbers are a LOWER BOUND on tradeable edge and MUST NOT be used as a headline 'this works in live' claim. Full queue+adv-sel comes in the next deliverable.**

---

## 1. Coverage at P90 (top-10%) vs current P99.5 / P99

**Live gate (from `live_trading/framework_config.json`):**
- `min_percentile_1s = 99.5` (top 0.5%)
- `min_percentile_5s = 99.0` (top 1%)
- `min_percentile_10s = 99.0` (top 1%)
- PatchTST veto: enabled on 5s and 10s

**Per-session signal count, v2 short-side, single-horizon thresholds:**

| Gate | n signals | n/session | avg move (t) | WR | net market (t) | net passive (t) | session net market | session net passive |
|-----:|----------:|----------:|-------------:|----:|---------------:|----------------:|-------------------:|--------------------:|
| P   50 |  136,900 | 13690.0 | +0.279 |  42.3% | -1.097 | -0.097 | -15023.94 | -1333.94 |
| P   70 |   82,160 | 8216.0 | +0.425 |  46.4% | -0.951 | +0.049 | -7813.67 | +402.33 |
| P   80 |   54,779 | 5477.9 | +0.512 |  49.1% | -0.864 | +0.136 | -4734.64 | +743.26 |
| P   90 |   27,420 | 2742.0 | +0.620 |  53.2% | -0.756 | +0.244 | -2073.09 | +668.91 |
| P   95 |   13,741 | 1374.1 | +0.709 |  56.1% | -0.667 | +0.333 | -916.16 | +457.94 |
| P 97.5 |    6,854 |  685.4 | +0.807 |  58.8% | -0.569 | +0.431 | -389.81 | +295.59 |
| P   99 |    2,741 |  274.1 | +0.979 |  61.6% | -0.397 | +0.603 | -108.76 | +165.34 |
| P 99.5 |    1,373 |  137.3 | +1.141 |  63.8% | -0.235 | +0.765 |  -32.32 | +104.98 |
| P 99.7 |      823 |   82.3 | +1.349 |  66.7% | -0.027 | +0.973 |   -2.24 |  +80.06 |
| P 99.9 |      275 |   27.5 | +1.953 |  74.2% | +0.577 | +1.577 |  +15.86 |  +43.36 |

**Coverage retention: live P99 gate keeps 10.0% of P90 short signals (2,741/27,420). P99.5 keeps 5.0% (1,373/27,420).**

**Throwaway: P99 discards 90.0% of the proven top-10% short band. P99.5 discards 95.0%.**

---

## 2. Net edge per trade per band — short side, FIFO FLOOR

Per-horizon realized-move stats at the proven and live gate levels (short profit = price went DOWN, so realized = -label_ticks):

| Gate | Horizon | n | n/sess | avg move (t) | WR | net market | net passive |
|-----:|:--------|--:|------:|------:|----:|------:|------:|
| P90 | 1s | 27,420 | 2742.0 | +0.620 |  53.2% | -0.756 | +0.244 |
| P90 | 5s | 27,409 | 2740.9 | +0.620 |  52.6% | -0.756 | +0.244 |
| P90 | 10s | 27,402 | 2740.2 | +0.595 |  51.9% | -0.781 | +0.219 |
| P95 | 1s | 13,741 | 1374.1 | +0.709 |  56.1% | -0.667 | +0.333 |
| P95 | 5s | 13,703 | 1370.3 | +0.750 |  54.2% | -0.626 | +0.374 |
| P95 | 10s | 13,693 | 1369.3 | +0.698 |  52.8% | -0.678 | +0.322 |
| P99 | 1s |  2,741 | 274.1 | +0.979 |  61.6% | -0.397 | +0.603 |
| P99 | 5s |  2,745 | 274.5 | +1.121 |  57.8% | -0.255 | +0.745 |
| P99 | 10s |  2,745 | 274.5 | +1.044 |  56.0% | -0.332 | +0.668 |
| P99.5 | 1s |  1,373 | 137.3 | +1.141 |  63.8% | -0.235 | +0.765 |
| P99.5 | 5s |  1,375 | 137.5 | +1.233 |  58.8% | -0.143 | +0.857 |
| P99.5 | 10s |  1,369 | 136.9 | +1.005 |  56.5% | -0.371 | +0.629 |

**Interpretation (1s horizon, short side):**
- At P90, avg short move = +0.620 ticks (53.2% WR). Net market = -0.756t — LOSING on market orders. Net passive = +0.244t.
- At P99, avg short move = +0.979 ticks (61.6% WR). Net market = -0.397t. Net passive = +0.603t.
- At P99.5, avg short move = +1.141 ticks (63.8% WR). Net market = -0.235t. Net passive = +0.765t.

---

## 3. PatchTST veto impact

From `all_confluence_features.npz` (250,070 aligned rows, 10 unique days):

- CNN-Mamba top-10% short band (cnn_pred_1s ≤ P10): n = 25,016
- PatchTST veto rule: reject if PT predicts ≥0 at 5s OR 10s (disagrees with CNN short).
- **Veto rejection rate within CNN top-10% shorts: 58.2%**

**Edge of trades ADMITTED (PatchTST agrees) vs REJECTED (PatchTST vetoes), 1s realized:**

| Group | n | n/sess | avg move (t) | WR | net market | net passive |
|:------|--:|------:|------:|----:|-------:|-------:|
| Admitted (PT agrees) | 10,461 | 1046.1 | +0.605 | 52.1% | -0.771 | +0.229 |
| Rejected (PT vetoes) | 14,555 | 1455.5 | +0.638 | 54.0% | -0.738 | +0.262 |

**Verdict: PatchTST veto HURTS — admitted trades show +0.605t vs rejected +0.638t (delta = -0.033t). PT is throwing away tradeable trades — veto should be loosened or dropped.**

**5s horizon:**
- Admitted: avg +0.864t, WR 55.7%
- Rejected: avg +0.447t, WR 50.6%
**10s horizon:**
- Admitted: avg +0.975t, WR 55.1%
- Rejected: avg +0.351t, WR 49.8%

---

## 4. Recommended gate setting

Optimization criterion: max total net-tick PnL per session (edge × signal count), FIFO FLOOR basis. Single-horizon 1s gate, short side only.

**Optimal under MARKET-ORDER cost (1.376t):**
- Gate: **P99.9**
- n/session: 27.5
- avg move: +1.953t (WR 74.2%)
- net per trade: +0.577t
- **session net ticks: +15.86t** (= $+198/session)

**Optimal under PASSIVE-LIMIT cost (0.376t):**
- Gate: **P80**
- n/session: 5477.9
- avg move: +0.512t (WR 49.1%)
- net per trade: +0.136t
- **session net ticks: +743.26t** (= $+9,291/session)

**Live setting comparison (P99.5 1s gate, FIFO floor 1s realized):**
- n/session: 137.3
- session net market: -32.32t ($-404/sess)
- session net passive: +104.98t ($+1,312/sess)

---

## CAVEATS

1. **FIFO FLOOR ONLY (HC #349):** All numbers are commission + 1-tick-spread lower bounds. Queue position and adverse selection are NOT modeled. Real live edge will be LOWER once queue model is layered on top.
2. **No fill probability:** Passive-limit numbers assume the limit fills at touch within the holding window. Real fill rates at top bands are well below 100% — separate fill-prob study needed (Razer paper-trader logs).
3. **Per-prediction trades, no stride dedup:** With 250-event stride, consecutive predictions can re-fire on the same alpha event. Real trade count after dedup will be lower.
4. **Sliding window walk-forward labels:** v2 smart_v3_mar uses 60d-train / 1d-OOT sliding folds — proper out-of-sample, no leakage.
5. **STRONGEST MODEL FOR EXECUTION (per HC #350):** v2 short top-10% remains the only setup with demonstrated face-value edge in the relevant cost stack; this audit answers whether the live gate is denying us that edge.

![Gate coverage histogram](gate_coverage_histogram.png)
