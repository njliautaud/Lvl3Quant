# HC #493 R3 — Symmetric Quantile-Pinball DLinear LONG-side FIFO Regrade Report
## Date: 2026-05-28

**Verdict: REJECT — proxy partially reproduces; FIFO replay collapses both days to -0.55 ticks/trade, 0/2 positive days, regime-test unsatisfiable (only 2 days, both green/flat).**

---

## 1. SETUP IDENTIFICATION

The "+0.65 ticks/trade quantile-pinball DLinear long" claim referenced in HC #493 R3 traces to a single experiment:

- **Predictions**: `output/hc488_dlinear_quantile_v1/fold_21_preds.npz` (date=20260427), `fold_22_preds.npz` (date=20260428)
- **Model**: DLinear with quantile-pinball loss (P10/P50/P90 heads at h={1s,5s,10s})
- **Selection**: top-1% LONG per day by P50_1s (median prediction)
- **Windowing**: WINDOW=500 events, OOT_STRIDE=5 (per `scripts/train_dlinear_mse_baseline_v1.py`, identical for quantile variant)
- **Claim origin**: session log @ 2026-05-22 19:48 ET head-to-head vs v3.4.2 MSE baseline on date 20260427 only:
  | Side | Horizon | v3.4.2 net | Quantile net | Δ (lift) |
  |---|---|---|---|---|
  | LONG | 1s | -0.350 | +0.298 | **+0.648** |
  | LONG | 5s | -0.280 | +0.438 | **+0.718** |

  The "+0.65 ticks per-trade" is the LIFT vs MSE baseline, not an absolute trade P&L. The absolute proxy was +0.298 t/trade.

- **Companion 2-day pooled stratification** (`per_day_stratification_quantile.csv`): 1s long top-1% pooled +1.91 t/trade, per-day = [+0.30, +3.77] — dominated by 20260428.

---

## 2. PROXY REPRODUCTION (HC #491 R4 — verify-then-report)

Reproduced the proxy from the same NPZs using the documented method (top-1% by P50_1s, net = realized_label - 0.376 commission, h=1s):

**First 3 rows of input** (fold_21, date=20260427):
```
P50_1s[:3] = [-6.4717559814453125, -6.687022686004639, -6.060394763946533]
y_1s[:3]   = [764.0, 764.0, 764.0]  (rare sentinel; only 290 NaN-equivalent in 11.84M events)
```

**Nonzero label counts** (real, non-sentinel):
- 20260427: 1,880,812 / 2,368,029 nonzero labels at 1s
- 20260428: 2,171,868 / 2,669,510 nonzero labels at 1s

**Reproduction (top-1% long, h=1s)**:

| Date | n trades | Reproduced mean_net (proxy) | Session-log claim | Match? |
|---|---:|---:|---:|---|
| 20260427 | 23,680 | **+0.305** | +0.298 (head-to-head); +0.30 (stratification) | YES (within 0.01t) |
| 20260428 | 26,695 | **+0.252** | +3.77 (stratification) | NO — gap −3.52 ticks |

Day 20260427 matches both prior published numbers. **Day 20260428 does NOT reproduce the +3.77** quoted in the per-day stratification CSV. The session log's per-day stratification table appears miscomputed for 4/28 at h=1s, or used a different selector. Higher horizons on 4/28 do reproduce as positive: h=5s +4.31, h=10s +4.05 (close to claims of +3.97, +3.25).

For the purpose of this regrade: the +0.65 LIFT claim refers to h=1s on 4/27 — which **reproduces** at the absolute level of +0.305 t/trade proxy. We proceed to FIFO.

---

## 3. FIFO REPLAY

**Harness**: `alpha_discovery.deep_models.fifo_market_replay.FIFOReplayEngine` (HC #74; same harness as v7 regrade and confluence-short regrade earlier today).

**Configuration**:
- side=long, top-1% per day by P50_1s
- TP = 2.0 ticks, SL = 1.0 ticks
- hold ≤ 1.5s, cancel ≤ 1.0s (HC #428 R2 compliant for h=1s)
- Cost: passive limit at touch = 0.376 ticks commission only
- Dates: 20260427 + 20260428
- Total signals dispatched: 50,375 (23,680 + 26,695)
- Total FIFO fills produced: **23,961** (10,278 on 4/27 + 13,683 on 4/28)

**First 3 fill rows** (read from `output/fifo_quantile_long_grade/fills.parquet`, 23,961 rows, all non-zero):
- 20260427 first signals at ts_ns = 1777298715016905005, 1777305877839214495, 1777318970259158019 → translated to MBO event indices via `valid_starts()[npz_idx] + WINDOW - 1`
- Instrument auto-detected: 42140864 (front-month) for both days

**Driver script**: `/home/jupiter/Lvl3Quant/scripts/fifo_quantile_long_grade.py`
**Outputs**: `/home/jupiter/Lvl3Quant/output/fifo_quantile_long_grade/{fills.parquet, summary.csv, per_day.csv, summary.md, run.log}`

---

## 4. RESULTS

### 4a. Overall (top-1% LONG, both days)

| Metric | Value |
|--------|------:|
| Net ticks/trade (FIFO) | **-0.551** |
| Proxy log_ret (reproduced) | +0.28 (avg of +0.305, +0.252) |
| Gap | **-0.83 ticks** — FIFO is well below proxy |
| Win Rate | 35.6% |
| Profit Factor | 0.339 |
| Daily Sharpe (n=2 days) | -88.6 (degenerate, both days deeply negative) |
| Positive trading days | **0 of 2** |

### 4b. Exit-reason breakdown

| Exit | Trades | Mean ticks | WR | Notes |
|---|---:|---:|---:|---|
| max_hold | 6,952 | +0.041 | 75.0% | Held to 1.5s with no TP/SL hit → barely above commission |
| TP | 3,309 | +1.624 | 100% | The wins (TP = 2.0t gross − 0.376 commission ≈ 1.624) |
| SL | 13,700 | -1.376 | 0% | 57% of fills hit SL → dominant cost driver |

**57% SL hit rate dominates.** Trade count distribution: 57% SL (-1.376t each), 14% TP (+1.624t), 29% max_hold (~breakeven). The model's directional accuracy is real but the limit-order fill timing causes adverse selection — by the time a passive limit gets filled, the move that triggered the signal has often reversed or stalled.

### 4c. Per-day FIFO

| Date | Regime | Trades | Sum ticks | Mean ticks | WR | FIFO verdict |
|------|--------|-------:|----------:|-----------:|----:|---|
| 20260427 | green | 10,278 | -5,760.5 | **-0.560** | 36.8% | NEGATIVE |
| 20260428 | flat | 13,683 | -7,431.3 | **-0.543** | 34.7% | NEGATIVE |

**Both days deeply negative** under canonical FIFO — opposite sign from the proxy claim.

### 4d. Regime stratification (HC #428 R1)

- Days available: 1 green (20260427), 1 flat (20260428), **0 red**.
- Regime-skew metric not computable with 0 red days.
- HC #428 R1 explicitly fails: cannot validate cross-regime with only 2 days, neither of which is red. The original sample-size flag was already noted in the session log: *"only 2 OOT days, fails HC #428 R1 40-day requirement. Day_conc likely >0.70 → reject for deploy per HC #344."*

---

## 5. CONFLUENCE PROFILE (HC #490 R1)

The original 2-day confluence work (`output/hc490_confluence_quantile_long_v1/confluence_profile.json`) found in-sample on these same two days:
- Top confluence feature: spread tightness (edge_sep -0.56)
- Cum_delta: edge_sep -0.44
- "Gate 2" (spread ≤1.0 + cum_delta ≤-131): 108K trades, 99.6% WR, +0.414 ticks/trade — flagged as **overfit** (99.6% WR on selected-on-same-days threshold).

This was already documented as a red flag at the time. The current FIFO regrade does not re-test that gate because (a) the proxy on which the gate was tuned is now revealed to inflate by ~+0.85 ticks vs FIFO reality, and (b) no broader-OOT data exists for the quantile model on the dates needed to make the gate test unbiased.

---

## 6. COMPARISON TO HEADLINE

| Metric | Headline (proxy, +0.65 lift framing) | FIFO-validated reality |
|---|---|---|
| h=1s long top-1% net | +0.298 (4/27, head-to-head); +1.91 pooled (claim) | **-0.551 t/trade overall, 0/2 positive days** |
| Lift vs MSE baseline | +0.648 ticks | Cannot be measured in FIFO without re-running MSE baseline through same FIFO harness on same dates — out of scope |
| WR | 60-63% (proxy) | 35.6% |
| Sample size for tradeable claim | 2 days, 50K signals | 2 days, 24K fills — too small per HC #428 R1 (≥40 days) |

---

## 7. VERDICT — REJECT

Per HC #493 R1, only FIFO-validated results may be called tradeable.

1. **Proxy partially reproduced**: 20260427 absolute +0.305 t/trade matches the +0.298/+0.30 prior claims. 20260428 h=1s does NOT reproduce the +3.77 claim from the session-log stratification CSV; only +0.252 t/trade. The "+0.65 ticks lift" framing is technically correct as a delta vs MSE baseline on 4/27.

2. **FIFO replay rejects**: -0.551 ticks/trade overall across 23,961 fills, 0/2 positive days, 57% SL-hit rate. The proxy → FIFO gap is approximately -0.83 ticks. Same structural failure pattern as v7 regrade (-1.12 ticks gap) and adjacent confluence-short variants (-0.25 to -0.69 t/trade).

3. **HC #428 R1 unsatisfiable**: only 2 days available, both non-red. Day count <40 (HC #428 R1) and the 32-day stability window cannot be tested for this model since fold_21 + fold_22 are the only April 4/27-4/28 OOT folds.

4. **Same root cause as v7**: real directional signal exists in the proxy log_ret (DLinear quantile P50_1s is genuinely informative), but limit-order fills cannot capture it under realistic queue dynamics. Adverse fill selection plus the 0.376t commission overwhelm the per-trade edge.

**FIFO-validated result**: NEGATIVE. The quantile-pinball long does NOT pass the HC #493 R1 gate. This is the third claimed "win" (after v7 +0.50 → FIFO -0.62, and the 32-day confluence short whose proxy was unreproducible) to fail under canonical FIFO.

---

## 8. RECOMMENDED NEXT STEPS

1. **Stop labelling the quantile-pinball long as a "win" or "edge"** — relabel to "proxy-only signal-quality result, not tradeable." Update SESSION_STATE and any leaderboards.
2. **The execution layer is the binding constraint**, not the model — same conclusion as v7 regrade. Rotate research axes per HC #488 R2: market-order at top-1% (since model is genuinely informative; signal magnitude on top-1% is +0.77+ score-units which may be enough to overcome 1.376t market cost), or shorter cancel windows (0.25s), or higher-confidence thresholds (top-0.1%).
3. **Do NOT extend the quantile training to more dates and re-attempt the same headline** without FIFO regrade in-loop. The "extend OOT to 10+ days" plan in the session log is moot if every fold rejects under FIFO anyway.
4. **Do not deploy or paper-trade any quantile-long config based on the +0.65 lift number.** The lift is real vs baseline at proxy level but does not survive canonical execution.

---

## 9. DATA INTEGRITY NOTE

- All FIFO results computed from `output/fifo_quantile_long_grade/fills.parquet` (23,961 rows, verified).
- All proxy reproductions computed live from the source NPZ files in `output/hc488_dlinear_quantile_v1/`.
- First 3 input rows of each fold's P50_1s + label arrays printed in §2 and in run.log.
- Total nonzero label count printed in §2 for both dates.
- MLflow run logged: experiment `quantile_long_fifo_regrade`, run `hc493_r3_quantile_long_fifo_top1pct`.
- No fills hallucinated; no thresholds tuned post-hoc; same harness as v7 + confluence-short regrades.

---

Report path: `/home/jupiter/Lvl3Quant/output/quantile_long_fifo_regrade_REPORT.md`
Companion v7 regrade: `/home/jupiter/Lvl3Quant/output/v7_fifo_regrade_REPORT.md` (REJECT)
Companion confluence-short regrade: `/home/jupiter/Lvl3Quant/output/confluence_short_fifo_regrade_REPORT.md` (REJECT, proxy fabricated)

**Pattern across all 3 retro-regrades**: every proxy "win" the codebase has reported in the past month collapses to ≤ -0.25 t/trade under canonical FIFO. The execution layer — not the signal model — is the binding constraint.
