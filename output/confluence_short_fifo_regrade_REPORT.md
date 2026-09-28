# HC #493 R3 — 32-Day Confluence-Gated SHORT FIFO Regrade Report
## Date: 2026-05-28

**Verdict: REJECT (no salvageable setup; the proxy headline that triggered the regrade is itself unreproducible — every FIFO-graded confluence-short variant on record is negative).**

---

## 1. SETUP IDENTIFICATION (the "32-day confluence short")

The "32-day confluence-gated SHORT" claim cited in HC #493 R3 traces to a single artifact set:

- **Path**: `output/confluence_5s_short_top1/`
- **Model**: v3.4.2 CNN-Mamba 5s SHORT predictions
- **Selection**: top-1% confidence shorts
- **Confluence gate**: OFI/spread/flow bucket `wide_ofi_pos_vol_q3_flow_pos`
- **Claim headline**: 125 trades over 32 OOT dates, **+0.768 ticks/event** (proxy), 56.8% WR, green WR 52.6% / red WR 63.3%

## 2. PROXY-REPRODUCTION GAP (already documented 2026-05-22)

Before any FIFO work, the claim must reproduce on its own proxy. It does not.

`output/surviving_5s_short_fifo_v1/REPRODUCTION_CHECK.txt` (2026-05-22) tested 6 reasonable interpretations of the `wide_ofi_pos_vol_q3_flow_pos` bucket against real OFI features from `mbo_events_smart_v3_ofi_features/` for the same 34 OOT dates. **None reproduced** the +0.768 t/event headline within the ±0.15-tick tolerance mandated by HC #74:

| Config | n_trades | mean_net | WR | delta vs +0.768 |
|---|---:|---:|---:|---:|
| A: ofi_book_1s>0, vol_q3, flow>0 | 20 | -0.676 | 60.0% | 1.444 |
| B: ofi_agg_1s>0, vol_q3, flow>0 | 109 | -0.270 | 51.4% | 1.038 |
| C: ofi_book_5s>0, vol_q3, flow5s>0 | 17 | +1.448 | 76.5% | 0.680 |
| D: wide spread only | 839 | -0.312 | 46.0% | 1.080 |
| E: no filter (top-1% short baseline) | 7,373 | **-0.059** | 49.0% | 0.827 |
| F: ofi_book_1s>0, flow<0 | 83 | -0.599 | 45.8% | 1.367 |

**Root cause confirmed in source archeology**:
- No Python script in the codebase writes `output/confluence_5s_short_top1/gates.txt`.
- The two related scripts (`scripts/analyze_timeofday_confluence.py`, `scripts/reconstruct_timeofday_simple.py`) explicitly state they use hardcoded synthetic values with the 0.768 constant baked in.
- The baseline 5s-short top-1% (7,373 trades, real data) is **-0.059 ticks proxy net** — *negative*. A sub-bucket of a negative baseline cannot plausibly yield +0.768.

Per HC #74 and HC #491 R4: **FIFO replay may not proceed until proxy reproduction is within ±0.15 ticks.** It cannot — the proxy is fabricated.

## 3. CANNOT FABRICATE FIFO NUMBERS ON A FABRICATED PROXY

Per the verify-then-report rule, I refuse to invent a "FIFO regrade" by selecting one of the 6 candidate interpretations and reporting its FIFO as if it were the canonical setup. That would launder a fabricated proxy claim through a real harness. The setup definition is not recoverable from disk.

**This is the gap, reported honestly.**

## 4. ADJACENT FIFO-VALIDATED CONFLUENCE-SHORT EVIDENCE (the real near-substitutes)

Two FIFO-graded confluence-short bodies of work exist and bear on the same research thesis:

### 4a. `output/stream_backtest_v2/surviving_canonical_fifo_REPORT.md` (16 days, 9 head-agreement configs)
Generated 2026-05-21 via canonical FIFO market replay (`hc432_fifo_full_market_replay`) on v3.4.2 multi-head predictions. The 9 configs are the survivors of HC #428 regime + HC #344 day-conc gates over the 32-day stability window.

Verify-then-report — first 3 SHORT fills (read directly from `surviving_canonical_fifo_fills.parquet`, 20,905 short-fill rows, all non-zero):
```
20260224 short hold=0.000s sl  net=-3.376t qa=37  strength=1.0  pair01_logret1s+pup5s
20260224 short hold=0.055s sl  net=-3.376t qa=70  strength=1.0  pair01_logret1s+pup5s
20260224 short hold=1.040s sl  net=-3.376t qa=49  strength=1.0  pair01_logret1s+pup5s
```

Per-config SHORT FIFO net (re-computed from the parquet just now):

| Config | n_short | days | net_t/trade | WR | sum ticks |
|---|---:|---:|---:|---:|---:|
| pair01_logret1s+pup5s | 9,635 | 10 | -0.392 | 42.7% | -3,778.8 |
| pair07_logret10s+logret60sq50 | 1,220 | 13 | -0.593 | 40.2% | -723.2 |
| pair08_logret5s+pup5s | 5,181 | 15 | -0.345 | 43.5% | -1,789.1 |
| trip01_pup5s+logret1s+logret10s | 66 | 8 | -0.694 | 37.9% | -45.8 |
| trip03_logret5s+pup5s+logret1s | 4,292 | 10 | -0.366 | 43.2% | -1,569.3 |
| trip04_pup5s+pup10s+logret1s | 331 | 9 | -0.246 | 45.3% | -81.5 |
| trip07_logret60s+logret30sq50+fifotp8sl5 | 58 | 11 | +0.607 | 56.9% | +35.2 |
| trip09_logret60s+logret10sq50+fifotp8sl5 | 61 | 11 | +0.526 | 55.7% | +32.1 |
| trip10_logret60s+logret30sq50+fifotp8sl5_top10 | 61 | 11 | +0.640 | 57.4% | +39.1 |

**HC #428 R1 gate (verbatim from `surviving_canonical_fifo_REPORT.md`)**: all 9 configs **FAIL** the regime-skew + day-conc gate. The three "positive" triplets (trip07/09/10) have sample sizes of 58–61 trades over 11 days with regime skew 0.91–0.93 (red Sharpe ~+0.44, green Sharpe ~+0.03) — pure red-day artifacts.

### 4b. `output/confluence_fifo_replay_v1/REPORT.md` (32 days, CNN-Mamba v2 × PatchTST confluence)
Generated 2026-05-23. Same FIFO harness, 32 OOT days, 1.388M aligned events. Tests BOTH-top-N% direction agreement (the closest existing analog to "32-day confluence short" by sample size + day count).

| Bracket | Selection | Filled | Net FIFO t/trade | Daily Sharpe | Profitable days |
|---|---|---:|---:|---:|---:|
| TP4SL3 | Both top 1% | 74 | **-0.396** | -2.14 | 32% |
| TP4SL3 | Both top 2% | 243 | **-0.269** | -4.64 | 21% |
| TP4SL3 | Both top 5% | 1,257 | **-0.614** | -6.83 | 14% |
| TP8SL5 | Both top 1% | 74 | -1.455 | -4.43 | 20% |
| TP8SL5 | Both top 2% | 243 | -0.946 | -3.60 | 25% |

All variants UNPROFITABLE under canonical FIFO. The +1.187 ticks proxy headline at "Both top 2% short" (from `confluence_stacking_v1`, 42-day window) collapses to **-0.269 ticks net FIFO**. Gap of ~-1.46 ticks — same failure pattern as v7.

## 5. REGIME-AGNOSTIC + ≥40-DAY CHECK (HC #428 R1)

- The **claimed** 32-day setup (`wide_ofi_pos_vol_q3_flow_pos`) cannot be FIFO-graded — fabricated proxy, no script provenance. 32 < 40 even if reproducible.
- The **closest FIFO-graded analog** (head-agreement confluence shorts): 9 configs over 8–15 days each. Regime gap 0.91–1.87 → FAIL HC #428 R1 across the board. Day count 8–15 → FAIL the ≥40-day rule.
- Document-the-gap path (the only honest path): no confluence-short setup currently passes HC #428 R1 + R2 + HC #493 R1 simultaneously.

## 6. COMPARISON TO PROXY HEADLINE

| Headline (proxy, fabricated/synthesized) | FIFO-validated reality |
|---|---|
| `wide_ofi_pos_vol_q3_flow_pos` 125 trades, +0.768 t/trade, 56.8% WR, Sharpe 3.07 | **Cannot FIFO-grade** — proxy unreproducible (2026-05-22 abort) |
| `confluence_stacking_v1` Both top 2% short +0.311 hybrid-cost, daily Sharpe +9.91 | **-0.269 t/trade FIFO** (243 fills, daily Sharpe -4.64, 21% positive days) |
| 9 surviving head-agreement configs (32-day stability) | All FIFO net negative on full-volume configs (-0.25 to -0.69 t/trade); positive-triplet configs are 58-61 trades on 11 red-heavy days with regime skew >0.90 |

## 7. VERDICT — REJECT (PROXY UNVERIFIABLE)

Per HC #493 R1, only FIFO-validated results may be called tradeable.

1. The 32-day confluence SHORT headline (+0.768 t/trade, 125 trades, `wide_ofi_pos_vol_q3_flow_pos`) **cannot be FIFO-graded** because the underlying proxy is fabricated. No script in the codebase produces the bucket cuts; the headline was synthesized with a hardcoded 0.768 constant. Six reasonable interpretations of the bucket name all fail to reproduce, with the closest at -0.270 ticks (wrong sign, wrong magnitude).
2. Every adjacent FIFO-graded confluence-SHORT variant on disk — head-agreement pairs/triplets (16 days), CNN-Mamba × PatchTST top-N% (32 days) — is **net negative** under canonical FIFO market replay, mirroring the v7 production regrade result from 07:35 ET this morning.
3. HC #428 R1 fails on every variant (regime skew >0.50, or sample size too small to evaluate, or day count <40).

This is the same structural failure observed in the v7 regrade earlier today: model-direction signal exists in proxy log_ret, but limit orders cannot capture it under realistic queue dynamics. Adverse selection on fills (-0.5 to -0.6 t/trade) plus 0.376t commission overwhelms the directional edge at every confidence threshold and every confluence variant tested to date.

**FIFO-validated result**: NONE. No confluence-short setup currently passes the HC #493 R1 gate.

## 8. PER-DAY SHARPE DISTRIBUTION (proxy headline)

The 32-day per-day breakdown for `wide_ofi_pos_vol_q3_flow_pos` is **not on disk** — never persisted by any reproducible script. The only per-day numbers in `confluence_5s_short_top1/` are the bucket-level aggregates (green: 76 trades / 52.6% WR / +0.144; red: 49 trades / 63.3% WR / +1.736). These per-bucket numbers are themselves the output of the same fabricated 0.768 generator.

For the 32-day FIFO analog (`confluence_fifo_replay_v1`, Both top 2% short, n=243): daily Sharpe -4.64, only 21% of trading days profitable. Distribution unavailable here without re-parsing the harness output — defer to the harness's per-day summary if needed.

## 9. RECOMMENDED NEXT STEPS

1. **Stop treating the +0.768 headline as a candidate.** Mark `confluence_5s_short_top1/` as fabricated in any future leaderboard. Add a `FABRICATED_DO_NOT_USE` marker file to that directory.
2. **Build a real bucket-confluence FIFO pipeline** if confluence research is to continue: a saved, reproducible script that (a) loads real OFI features for the 32+ OOT dates, (b) defines the bucket logic in code (not in a filename), (c) emits per-trade timestamps the FIFO harness can ingest, (d) reports proxy and FIFO side-by-side with the gap explicit.
3. **The execution problem is the structural blocker**, not the selection scheme — same conclusion as v7 regrade. Rotate to the model/horizon axis (per HC #488 R2) as the v7 follow-up branches are already doing (5s CNN-Mamba raw + market-order variants).
4. **Do not deploy or paper-trade any "32-day confluence short" config.** It does not exist as a measured setup.

## 10. DATA INTEGRITY NOTE

- Per-config short FIFO metrics in §4a re-computed from `output/stream_backtest_v2/surviving_canonical_fifo_fills.parquet` (20,905 short fills, non-zero count = 20,905). First 3 rows printed verbatim above.
- §4b table reproduced verbatim from `output/confluence_fifo_replay_v1/REPORT.md`.
- §2 reproduction findings reproduced verbatim from `output/surviving_5s_short_fifo_v1/REPRODUCTION_CHECK.txt`.
- No fills were generated by this regrade — there is nothing to generate fills from. Per task constraint: "If the harness or setup is missing → REPORT THE GAP HONESTLY in the report file and STOP. Do not hallucinate fills or numbers."

---
Report path: `/home/jupiter/Lvl3Quant/output/confluence_short_fifo_regrade_REPORT.md`
Companion v7 regrade: `/home/jupiter/Lvl3Quant/output/v7_fifo_regrade_REPORT.md` (REJECT, same root cause)
Companion v7 branches: `/home/jupiter/Lvl3Quant/output/v7_fifo_branches_REPORT.md` (5/5 REJECT)
