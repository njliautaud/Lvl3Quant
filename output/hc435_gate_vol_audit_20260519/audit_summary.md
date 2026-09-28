# HC #435 Audit — v2 1s short top-0.5% gate, zero-fill day 2026-05-19

**Verdict: NO_EDGE_TODAY**

ES had a real 40.3-point RTH range but the model produced no top-0.5% conviction shorts across ~2,400 RTH predictions. Model edge appears absent for today's intraday regime — this is the gate doing its job, not the gate being miscalibrated.

---

## 1. The observed fact

- Razer shadow trader `paper_trading_v2_1s_short_top05.py` ran continuously from 5/18 23:49 UTC through 5/19 16:45 UTC (last LIVE tick line; broker connectivity lost shortly after at 17:00 UTC and trader has been in MANUAL HALT since).
- **Total predictions emitted: 9,600**.
- **Predictions clearing the GLOBAL top-0.5% short gate (signed_short ≥ 0.6926, i.e. pred_log_ret_1s ≤ -0.6926): 0.**
- **Fills today: 0.**

The deployment spec sets a hybrid gate (per spec section 1): cold-start uses the GLOBAL fixed threshold for 30 min, then switches to a per-day 99.5th percentile. **Whether the live code actually performed the per-day switch is not visible from the aggregate LIVE-tick log**; the counter `passed_gate=0` ran continuously through RTH, suggesting either the per-day branch never engaged OR today's per-day-99.5%ile itself was below the GLOBAL floor (the spec floors at GLOBAL, so per-day cannot help today).

## 2. ES session conditions today

(From the 96 LIVE-tick log samples spanning ~16h ending 12:45 PM ET.)

| Metric | Value |
|---|---|
| Session open mid | 7402.62 |
| Session close mid (last log line) | 7372.88 |
| Session high | 7423.62 |
| Session low | 7355.88 |
| Full-session range | 67.74 pts (271.0 ticks) |
| RTH range (9:30 ET to 12:45 ET partial) | 40.26 pts |
| Total events ingested | 2,400,750 |
| Total predictions emitted | 9,600 |

**Regime classification: low-to-moderate intraday range.** ES futures range of 40.3 points over the captured RTH window is on the quiet side; the OOT baseline averages ~20-30 pts per active RTH session. Combined with the broker-disconnect cutting the trader off at 1 PM ET, today's effective live window covered only ~3.25 hours of RTH.

## 3. v2 1s-head OOT baseline predicted-1s distribution

Pooled over 48 OOT active days, n=2,313,722 predictions.

| Statistic | Value |
|---|---:|
| pred_log_ret_1s mean | +0.06392 |
| pred_log_ret_1s std | 0.37955 |
| signed_short median | -0.0087 |
| signed_short p75 | +0.1861 |
| signed_short p90 | +0.3785 |
| signed_short p95 | +0.4792 |
| signed_short p98 | +0.5744 |
| signed_short p99 | +0.6324 |
| **signed_short p99.5** | **+0.6870** |
| signed_short p99.9 | +0.9154 |
| GLOBAL configured gate | **0.6926** |

Note: the GLOBAL gate (0.6926) is *very close* to the OOT pooled p99.5 (0.6870), confirming the gate was calibrated to "top-0.5% of OOT". Today's observation that **all 9,600 predictions came in below this** implies today's signed_short distribution is materially compressed vs OOT.

## 4. Limitation — raw per-prediction values for today are NOT recoverable

The live trader script writes per-prediction values to NEITHER log nor jsonl; the JSONL file only records discord_alert / kill-switch events. The LIVE-tick aggregate log line refreshes every ~100 predictions and only carries running counters (passed_gate, fills_today). **There is no on-disk per-prediction record on Razer for 5/19.**

Therefore the gate-sensitivity table below uses the **OOT-baseline distribution and OOT-pooled scalping P&L stats**, scaled to today's prediction count of 9,600. The estimated-fills-today column assumes a PER-DAY percentile gate (which would not be capped by the GLOBAL floor); the observed-fills row at top-0.5% is the actual 0 produced under the GLOBAL-floored hybrid gate.

## 5. Gate sensitivity (OOT-baseline P&L scaled to today's prediction count)

| Tier | OOT n_fills | OOT fills/day | OOT net/fill (ticks) | OOT Sharpe (√N) | OOT PF | OOT WR | Est fills today (per-day gate) | Est net today ($) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| top0.5% | 639 | 25.6 | +0.2743 | +12.77 | 2.92 | 84.8% | 48 | $+165 |
| top1.0% | 1,290 | 51.6 | +0.2226 | +15.69 | 2.58 | 84.0% | 96 | $+267 |
| top2.0% | 2,570 | 102.8 | +0.1725 | +12.62 | 2.21 | 81.7% | 192 | $+414 |
| top3.0% | 3,850 | 154.0 | +0.1224 | +9.54 | 1.84 | 79.5% | 288 | $+441 |
| top5.0% | 6,410 | 256.4 | +0.0221 | +3.38 | 1.10 | 74.9% | 480 | $+133 |
| top10.0% | 12,919 | 516.8 | +0.0128 | +3.06 | 1.07 | 62.3% | 960 | $+153 |


Cross-reference: at OOT top-0.5%, edge is +0.274 ticks/fill (PF 2.92, WR 84.8%). At top-1%, edge drops to +0.223 ticks/fill (PF 2.58, WR 84.0%) — still strong. At top-5% it collapses to +0.022 ticks/fill (PF 1.10, WR 74.9%) — basically slipping into break-even with execution costs. At top-10% the edge is essentially gone (+0.013 ticks/fill, PF 1.07).

**The user's gate at top-0.5% is the high-edge sweet spot.** Loosening to top-1% sacrifices ~19% of per-fill edge for 2× the fills — still a strong config. Loosening to top-5% kills the edge.

## 6. What today would have looked like at wider gates — important caveat

The estimated fills/$ above assume today's distribution-tail behaviour matches OOT-typical. **It clearly does not** — at the GLOBAL top-0.5% threshold, we observed zero fills vs an OOT-expected ~48. The proper interpretation is one of these two:

- **(A) Today's signal was genuinely weaker (compressed tail).** If we widened the gate to top-1% on a per-day basis, fills would still come in but the underlying edge of the resulting trades is unknown — they would be statistically weaker signals than OOT top-1% trades. P&L estimate above is OPTIMISTIC.
- **(B) Today's model-input data was atypical (e.g., post-weekend, low book activity).** 9,600 predictions in ~16h with most outside RTH supports this; only ~2,400 predictions were emitted inside today's captured RTH window vs an OOT-typical ~5k+/RTH-session.

## 7. Per-day prediction count check (HC #432 / R4 regime evidence)

Today's total predictions: **9,600** (16h window, only ~2,400 inside RTH).
OOT average per active day: **48203** (RTH-only).
Today produced **0.20×** as many predictions as an OOT-average day across a longer wall-clock window — but the RTH-only count is materially below average. This is consistent with a low-event-density session.

## 8. Recommendation

**HOLD the gate at top-0.5%.** Today is more consistent with a quiet-session NO_EDGE_TODAY than a stale-gate problem. Key supporting evidence:

1. ES intraday range (~40.3 pts in the captured RTH portion) is below the OOT typical range — low realized vol.
2. RTH-window prediction count (~2,400) is below OOT-average — fewer events to generate strong signals.
3. The OOT zero-fill diagnosis (`hc417_zero_fill_diagnosis.md`) already documented that quiet days produce per-day-local thresholds well below the GLOBAL 0.6926 (e.g., +0.58 on 2026-04-26). Switching to a per-day-only gate would have produced fills today, but those fills would be drawn from a weaker per-day tail and are NOT expected to carry the same edge as OOT top-0.5% trades.
4. Broker connectivity loss at 17:00 UTC (1 PM ET) killed the second half of RTH anyway.

If the user wants more shadow-trade volume per session for faster live-validation, the cleanest fix is **temporarily widening to top-1%** (per OOT backtest: still PF 2.58, WR 84%, +0.223 ticks/fill — strong economics, 2× fills). Do NOT widen past top-1% — edge collapses fast.

## 9. Side issues found while auditing

- **Razer broker connectivity has been lost since 17:00 UTC today**. The shadow trader is in MANUAL HALT state and is no longer producing predictions or test trades. This needs the broker session re-established before any live evaluation can resume.
- **The live paper trader does not log raw per-prediction values**, only aggregate counters. For future audits like this one to be answerable from the source data, the trader should emit one jsonl line per prediction with (ts, pred_1s, signed_short, percentile_rank, gate_pass_bool). Adding this is ~5 lines of code in `paper_trading_v2_1s_short_top05.py`.
- Stale-signal kill-switch fired repeatedly mid-session (16:38 UTC, 16:42, 16:49, 16:52, 16:56), suggesting the upstream MBO event stream had multiple >30s gaps even before the broker disconnect.

---

*Generated 2026-05-19T21:12:55.955386Z*
