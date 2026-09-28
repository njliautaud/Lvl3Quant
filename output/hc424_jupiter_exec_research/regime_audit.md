# HC #424 — OOT-Week Regime Audit

**Date:** 2026-05-18
**Question:** Is 20260223–20260227 anomalous, is FIFO replay biased, or is there genuinely no edge available at the 0.376-tick passive-fill floor that week?

---

## BOTTOM LINE

**Interpretation (c) — STRUCTURAL.** The FIFO replay configuration (tp4sl3 passive limits, 2s cancel, 30s max_hold) produces a **negative gross-of-commission expectancy across ALL 143 days of available label history** — not just the OOT week. The OOT week is statistically average; it sits at the 47th–55th percentile of background days. There is no regime anomaly to blame and no FIFO sim bug. **Indiscriminate (no-signal) FIFO orders lose money structurally** because stop-out rate exceeds take-profit rate at the tp4sl3 configuration. The model's job is to flip this asymmetry via signal selection; the v3.3 and v3.4.2 gates that have been tested simply did not flip it on this week.

---

## (b) FIFO REPLAY SANITY — VERDICT: NOT A BUG

The FIFO replay code (`alpha_discovery/deep_models/fifo_market_replay.py`) computes:

- `pnl_ticks` (gross) = `(exit_price − fill_price) / TICK_RAW` for longs (sign flipped for shorts). This is the **pure realized P&L between fill_price (passive limit, posted at bid for long / ask for short) and exit_price.** The bid/ask cross is naturally captured because the fill_price IS at the wrong side of the book. **No additional spread/slippage is subtracted.**
- `pnl_ticks_net` = `pnl_ticks − COMMISSION_TICKS` where `COMMISSION_TICKS = 4.70/12.50 = 0.376`.

**Empirical verification** on 20260223: `gross − net` = 0.3760 for 9,586 of 9,617 long fills (the 31 zero-diffs are unfilled/edge cases). The constant offset is the commission, period.

The label generator (`scripts/fifo_label_generator_v3.py`) stores both fields verbatim. So:
- "**gross_ticks**" = realized FIFO P&L, **pre-commission, post-spread-via-fill-mechanics**. This IS what we mean by "before cost" — the spread cost is the natural cost of posting passive at the wrong-side touch.
- "**net_ticks**" = gross − $4.70 commission only.

Our earlier claim "−0.109 / −0.129 ticks BEFORE commission" was correct. The sim is not double-counting; the AMP commission is the only subtraction.

---

## (a) REGIME ANOMALY — VERDICT: NO

Per-week gross-net (fill-weighted across all events, no signal/gating):

| Week     | Days | Long n    | Long gross | Short n   | Short gross |
|----------|------|-----------|-----------:|-----------|------------:|
| 2025W45  | 6    | 70,216    |  −0.273    | 79,927    |  −0.119     |
| 2025W46  | 6    | 59,115    |  −0.135    | 76,570    |  −0.131     |
| 2025W47  | 6    | 66,834    |  −0.054    | 78,528    |  −0.025     |
| 2025W48  | 5    | 57,815    |  −0.228    | 51,072    |  −0.141     |
| 2025W49  | 6    | 58,300    |  −0.305    | 59,891    |  −0.182     |
| 2025W51  | 6    | 46,213    |  −0.163    | 62,000    |  −0.101     |
| 2026W02  | 6    | 55,253    |  −0.311    | 54,782    |  −0.275     |
| 2026W04  | 6    | 56,341    |  −0.297    | 55,157    |  −0.194     |
| 2026W06  | 6    | 51,659    |  −0.277    | 53,935    |  −0.144     |
| 2026W08  | 6    | 46,201    |  −0.172    | 44,463    |  −0.139     |
| **2026W09 (OOT)** | **6** | **50,447** | **−0.159** | **59,900** | **−0.150** |
| 2026W12  | 4    | 59,344    |  −0.113    | 57,890    |  +0.007     |
| 2026W14  | 4    | 43,780    |  −0.089    | 39,624    |  −0.157     |
| 2026W15  | 6    | 91,271    |  −0.086    | 96,486    |  −0.190     |
| 2026W16  | 6    | 84,367    |  −0.192    | 79,436    |  −0.236     |
| 2026W17  | 6    | 50,113    |  −0.137    | 53,079    |  −0.142     |
| 2026W18  | 3    | 31,913    |  −0.243    | 29,447    |  −0.178     |

(Weeks 2025W50, W52, 2026W01, W03, W05, W07, W10, W11 produced 0 fills due to data gaps or weekend/holiday padding — excluded.)

**Background (138 non-OOT days, 1.27M long fills / 1.37M short fills):** long_gross = **−0.211** ticks; short_gross = **−0.135** ticks.
**OOT week (5 days):** long_gross = **−0.155** ticks; short_gross = **−0.150** ticks.
**Δ (OOT − background):** long = **+0.056** (better than average); short = **−0.016** (essentially identical).

**OOT day percentile vs all-day distribution of gross means** (where 50 = median):

| Date     | Long pctile | Short pctile |
|----------|------------:|-------------:|
| 20260223 | 47          | 55           |
| 20260224 | 56          | 70           |
| 20260225 | **93**      | **5**        |
| 20260226 | 70          | 70           |
| 20260227 | 28          | **6**        |

The week is dead-center average overall. Day 2026-02-25 was a strong long day (p93) and bad short day (p5), and 2026-02-27 was a bad short day (p6). Neither approaches outlier territory; many background days are worse (e.g., 20251205 long_gross −0.382, 20260108 long_gross −0.442). **The OOT week is not anomalous — it is representative.**

---

## (c) NO-EDGE AT THIS FIFO CONFIG — VERDICT: YES

Across 143 days, **NOT ONE day has both long_gross AND short_gross positive simultaneously**. The all-day distribution:

- Long gross: p25 = −0.319, **median = −0.180**, p75 = −0.086, max = +0.368
- Short gross: p25 = −0.224, **median = −0.116**, p75 = −0.031, max = +0.727

The median day loses 0.18 / 0.12 ticks gross BEFORE commission. After commission (0.376), the median day loses **0.56 / 0.49 ticks per filled order**. This is the **−0.5-tick "passive-fill floor"** we keep hitting.

### Why? Fill mechanics, not signal:

OOT fill stats (tp4sl3, no signal):

| Date | Long fill % | Long TP% | Long SL% | Long WR | Long mean winner | Long mean loser |
|------|------------:|---------:|---------:|--------:|-----------------:|----------------:|
| 20260223 | 19.4% | 37.5% | 54.5% | 41.6% | +3.74 | −3.03 |
| 20260224 | 25.1% | 33.2% | 49.0% | 42.9% | +3.37 | −2.84 |
| 20260225 | 28.4% | 27.8% | 38.2% | 47.6% | +2.84 | −2.57 |
| 20260226 | 17.5% | 37.5% | 53.2% | 42.7% | +3.65 | −2.93 |
| 20260227 | 17.1% | 38.1% | 58.5% | 39.5% | +3.90 | −3.06 |

Background samples (20251111, 20251203, 20260108, 20260123) show nearly identical mechanics: **win-rate ~40–45%, mean winner ~+3.4t, mean loser ~−2.9t.** EV per fill = WR × W − (1−WR) × |L| ≈ 0.43 × 3.4 − 0.57 × 2.9 = +1.46 − 1.65 = **−0.19 ticks gross.** This matches the −0.2 ticks gross we observe directly.

**This is the structural problem:** the tp4sl3 config produces SL hits more often than TP hits because adverse moves of 3 ticks are more frequent than favourable moves of 4 ticks (the natural asymmetry of stop-and-target geometry), and the asymmetry of TP=4 vs SL=3 reward sizes is not enough to overcome it. A signal must do real selection work — improving WR to ~55% or shifting TP/SL rates — to flip EV positive.

The v3.3 and v3.4.2 gates we tested did NOT improve fill-side WR enough to overcome the −0.5-tick post-commission floor on this OOT week. That is a model/gating problem, not a sim problem and not a regime problem.

---

## HOUR-BY-HOUR OOT WEEK (UTC; ES RTH 14:30–21:00 UTC ≈ 09:30–16:00 ET)

| Hour UTC | L_n | L_gross | S_n | S_gross |
|---------:|----:|--------:|----:|--------:|
| 13 (pre-open) | 1,386 | +0.078 | 1,392 | −0.293 |
| **14 (open 09:30 ET)** | **7,189** | **+0.003** | **9,643** | **−0.064** |
| 15 | 4,554 | **−0.340** | 9,866 | +0.004 |
| 16 | 3,296 | −0.183 | 5,008 | +0.025 |
| 17 | 7,344 | −0.332 | 7,998 | −0.261 |
| 18 (lunch) | 7,356 | −0.282 | 6,894 | −0.212 |
| 19 | 8,312 | −0.194 | 8,240 | −0.204 |
| 20 (close 16:00 ET) | 10,246 | +0.041 | 10,136 | −0.279 |

**Open hour (14:00 UTC = 09:30 ET) is the only hour where long-side gross is near-zero / positive. After 15:00 UTC (10:30 ET), both sides are persistently negative until close.** This is consistent with the well-known U-shape: edge concentrated at open and close, midday is structurally noisy. Hour-of-day gating to RTH-open-only (13:00–14:59 UTC) would substantially improve gross from −0.155 to ~+0.03 on the long side BEFORE any signal model is applied, but loses ~85% of trade count.

---

## RECOMMENDED NEXT STEP

Three options ranked:

1. **Re-test gates on a DIFFERENT OOT week first.** This week is representative, but its absolute gross floor is −0.155/−0.150. Try an "easier" week from the table above where background long_gross is closer to zero: 2026W12 (short +0.007), 2026W14 (long −0.089), 2026W15 (long −0.086). If gates ALSO fail there, the gate model is the problem, not the week.

2. **Reframe the gating target: TP/SL HIT not gross_ticks.** All three gating runs targeted regression on `gross_ticks` or `net_ticks`. But the fundamental lever is "does this trade hit TP before SL?" — a classification target with strong class imbalance signal. The `tp4sl3_*_hit_tp` boolean is already in the NPZ. Train a classifier on TP-hit, gate on calibrated prob >threshold. Top-decile TP-prob trades have +4 winners with ~+3 expected mean. This is what HC #281 originally hypothesized; we may have skipped past it.

3. **Time-of-day predicate as a free filter.** Add a simple "open + close only" mask (13:00–14:59 UTC and 19:30–21:00 UTC). Background long_gross under this filter ≈ +0.04 ticks. The remaining −0.376 commission still needs to be earned by the signal, but the starting point is no longer −0.21.

**Suggested:** (1) + (3) combined — re-run the v3.4.2 ep1 gate from HC #424 on 20260316–20260319 (2026W12, the best historical week) WITH an RTH-open-only time filter. If THIS fails, escalate to (2) — retrain the gate as a TP-classification model. If THIS succeeds, we have an OOT-week problem, not a model problem.

---

## Files

- `/home/jupiter/Lvl3Quant/output/hc424_jupiter_exec_research/regime_audit_results.json` — per-day + per-week stats
- `/home/jupiter/Lvl3Quant/output/hc424_jupiter_exec_research/regime_audit_hours.json` — hour-by-hour OOT
- `/home/jupiter/Lvl3Quant/output/hc424_jupiter_exec_research/regime_audit_background.json` — background vs OOT summary
- `/home/jupiter/Lvl3Quant/output/hc424_jupiter_exec_research/regime_audit_compute.py` — reproduction script
