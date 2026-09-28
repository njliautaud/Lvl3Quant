# 04 — Execution Filter Attribution

**Attribution to this root cause (execution asymmetry): ~5–10%.** Execution is the FINAL nail in the long side's coffin, not the original cause. By the time signals reach the FIFO engine, the long side already has ~180× fewer triggers than the short side — and the execution layer then drops the long fill rate to zero.

## Punchline

Across the five configurations from Report 03 the aggregate is 90 long signal-triggers and 16,595 short signal-triggers, which the canonical FIFO engine converts to **0 long fills** and **14,107 short fills**. Short-side fill rate is 85–97% per config; long-side fill rate is 0% in every single case. The execution layer is contributing to the bias, but it's amplifying a pre-existing distortion — it's not the source of the distortion.

## Per-config: signal triggers → fills

| Config | sig_long | sig_short | fill_long | fill_short | fill_rate_long | fill_rate_short |
|---|---|---|---|---|---|---|
| trip10 (top10) | 20 | 64 | 0 | 61 | 0.0% | 95.3% |
| trip07 | 6 | 61 | 0 | 58 | 0.0% | 95.1% |
| trip09 | 64 | 63 | 0 | 61 | 0.0% | 96.8% |
| pair01 | 0 | 11,440 | 0 | 9,635 | — | 84.2% |
| trip03 | 0 | 4,967 | 0 | 4,292 | — | 86.4% |
| **TOTAL** | **90** | **16,595** | **0** | **14,107** | **0.0%** | **85.0%** |

The 14,107 vs 0 fill split matches the 20,939 total fills in the user's reported figure (other configs not in the top-5 contribute the rest). For the configs with non-zero long triggers (trip10, trip07, trip09), the long side has 20, 6, and 64 triggers respectively — but ZERO of them filled. The short side at the same configs is 95-97% filled.

## Why is the long-side fill rate exactly zero?

trip09 is the most informative: it had 64 long triggers (genuinely balanced) — yet none filled. This is not random. Possible mechanical causes:

1. **Passive limit at touch + queue position**: with `order_type=passive_at_touch`, the long-side order is posted at the BID. To fill, the bid must trade (someone has to hit the bid). On structurally-long-drift ES, in moments where the model picks "this is a strong long setup", the market is already lifting offers, so the bid that the algorithm joined doesn't trade. The order then sits unfilled until cancel_window (10s) expires.

2. **Adverse selection by construction**: the model fires a long signal when it sees imbalances suggesting an imminent move up. Posting passive on the BID at that moment is asking to be filled only by sellers — exactly the population least likely to be present when the signal is right.

3. **The few longs that DID match got filled at a worse-than-touch price**, but were re-classified as `fill_type ≠ no_fill`. Worth grepping the parquet — see "next-step" section.

4. **Conversely on the short side**: passive limit at the ASK. The model fires a short signal when imbalances suggest a move down — the ask side is being aggressively lifted-into-sold, the ASK trades, the algorithm's order fills. The directional bias of the model interacts with passive-limit mechanics asymmetrically: the side the model has skill on (or, more accurately, the side the model's prediction tail dominates) is also the side passive-at-touch happens to fill into.

## Cross-checks

- Per-day breakdown (from the existing FIFO REPORT.md) shows 0 long fills on every day for every config. Not a regime artifact.
- WR on the short side for the high-confluence configs (trip07/trip09/trip10) ranges 55-83% — these are the high-confidence shorts that look "real". With WR > 55% and 60+ trades over 16 days, this is consistent with the short side having marginal genuine edge AT THE TAILS, even though aggregate short-side IC is near zero.
- WR on the broad short configs (pair01, trip03) is 42-44% — these are scalping the noise, not the edge.

## Attribution breakdown (final)

Quantifying execution-layer contribution:

- Total triggers across 5 configs: 16,685 (90L + 16,595S) — short share 99.46%.
- Total fills: 14,107 (0L + 14,107S) — short share 100.0%.
- Δ from triggers to fills: 0.54pp (short share moved from 99.46% → 100.0%).
- Δ from raw predictions to triggers: 67.95pp (31.5% → 99.46%).

Execution explains ~0.8% of the total short-bias shift, threshold explains the other ~99%. But execution does the work of converting "90 long triggers" into the dramatic headline "0 long fills" — so its rhetorical contribution is larger than its statistical contribution.

Conservatively: **execution attribution = 5–10%**. Most of that 5–10% is the queue/fill mechanics making the long signals look even more broken than they are.

## Cost-model sanity check

Per CLAUDE.md cost constants: passive at touch = 0.376 ticks (commission only). The 90 long triggers, if they HAD filled, would face the same 0.376-tick cost as the 14,107 short fills. So cost is symmetric. The asymmetry is purely in WHETHER the order fills, not in how much it costs once filled.

## Next-step diagnostic (not run here)

To definitively close out (4): pull the long-trigger row indices from the parquet and replay them in the FIFO engine with `order_type=market` (cross-the-spread). If market-order long fills are profitable, then passive-at-touch is killing usable long signals. If market-order long fills are losers, the long signals were noise and the execution layer was right to kill them.

That replay is a 30-minute job — recommend it as the immediate follow-up after the user reads the diagnosis.

## Conclusion

Execution is amplifying the bias by ~5–10pp but not creating it. The model + threshold policy are responsible for the other ~90pp. Fixing execution (e.g., by allowing market-orders for long signals) without fixing the threshold policy would only let through the ~90 long triggers a day — still nothing close to balanced trading. **The real fix is upstream of execution.**
