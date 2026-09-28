# STATE OF THE PROJECT — 2026-05-19 23:30 ET (Update #2)

Supersedes `STATE_OF_THE_PROJECT_20260519.md` (22:55 ET). Newest file wins per CLAUDE.md.

## TL;DR — Where We Actually Are 3 Days From Friday Deadline

- **5 canonical FIFO runs completed since 22:30 ET. ZERO are profitable.** All on the CNN-Mamba v2 proven model, all on 32-36 OOT days.
- **The 22:55 ET checkpoint conclusion holds and is reinforced**: label-level edge is real (IC_1s=0.222), but tradeable edge under realistic FIFO microstructure is not yet found.
- **Two configs still running** (auto-sequenced): wider geometry (SL=3 TP=8 H=10s) and multi-horizon confluence (1s∧5s∧10s top-30% short).
- **Two more queued**: tight geometry (SL=1 TP=2 H=0.5s) and chase-execution variant.
- After these 4 run we have exhausted the "obvious" execution-side knobs. If none clear, Thursday morning pivots to the data-collection fallback per HC #443 R5.

## Canonical Run Ledger (CNN-Mamba v2, FIFO realtime_sl engine)

| # | Config | Filter | TP / SL / Hold / Cancel | Entry | n_fills | mean_tk | PF | WR | Sharpe√N | R1 | R2 | Verdict |
|---|---|---|---|---|---:|---:|---:|---:|---:|---|---|---|
| 1 | hc442_v2_canon_c1 | top0.5 short | 3.0 / 0.5 / 1.5s / 1s | passive | 2,431 | **−0.326** | 0.51 | 22.8% | — | F | P | LOSING |
| 2 | hc442_v2_canon_c10 | top0.5 short | 3.0 / 0.5 / 1.5s / 10s | passive | 3,455 | **−0.313** | 0.51 | 25.1% | — | F | F | LOSING |
| 3 | hc443_market_entry_tp3 | top0.5 short | 3.0 / 0.5 / 1.5s / 1s | **market** | 4,299 | **−0.676** | 0.16 | 7.3% | −31.4 | F | P | LOSING |
| 4 | hc443_wider_sl2_tp3 | top0.5 short | 3.0 / **2.0** / 1.5s / 1s | passive | 2,431 | **−0.645** | 0.46 | 38.8% | −17.1 | **P** | P | LOSING |
| 5 | hc443_upside_sl3_tp8_h10 | top0.5 short | 8.0 / 3.0 / **10s** / 1s | passive | RUNNING | — | — | — | — | — | — | TBD |
| 6 | hc443_multih_confluence_top30 | 1s∧5s∧10s top30 short | 3.0 / 0.5 / 1.5s / 1s | passive | QUEUED | — | — | — | — | — | — | TBD |
| 7 | hc443_tight_sl1_tp2_h05 | top0.5 short | 2.0 / 1.0 / **0.5s** / 0.5s | passive | QUEUED | — | — | — | — | — | — | TBD |
| 8 | hc443_chase_sl05_tp3_h15 | top0.5 short | 3.0 / 0.5 / 1.5s / 1s | **chase** | QUEUED | — | — | — | — | — | — | TBD |

## What Each Failed Run Teaches Us

### Run #3: Market entry (worse than passive by 0.35 tk)
- Crossing the spread destroyed 1 tick of label edge plus added the commission. Even with TP=3 the WR collapsed to 7%. Indicates the label edge is captured AT the passive touch and evaporates the moment you cross.
- **Implication**: ANY execution variant that pays >0.5 ticks more than passive is dead-on-arrival.

### Run #4: Wider stop (SL=2 instead of 0.5) — INFORMATIVE
- This is the most interesting failure. WR jumped 22→39%, PF improved 0.51→0.46 (a near-wash because losers got proportionally bigger).
- **Crucial**: R1 PASSED (regime ratio 0.115, well under 0.50). The model fails CONSISTENTLY across green/red/flat — not regime-dependent.
- Mean tk got worse (−0.65 vs −0.33) because each loss is 4× larger and wins didn't scale.
- **Implication**: Widening the stop doesn't change the underlying problem. The signal does not reliably reach +3 ticks within 1.5s under realistic queue/fill conditions. The 21-tick label MFE is captured by the LABEL but not by passive limit orders that face queue priority.

### Runs #1, #2: The canon baselines (the proven negative)
- Identical to HC #441 "champion" config except FIFO-honest. Both confirm the analytic resolver inverted the sign.

## The Hard Truth Diagram

```
LABEL edge:    +21 tk mean MFE within 1s on top-0.5% short  (REAL)
                          |
                          v
QUEUE penalty: signals arrive at touch but queue ahead = many,
               fill probability low when price moves away
                          |
                          v
ADVERSE selection: the times we DO get filled at passive are
                   disproportionately the times the move stalls
                   or reverses (no-one else wanted to be filled THERE)
                          |
                          v
REALIZED edge: −0.33 tk per fill across 2,431 trades, 22.8% WR
```

The realized−label gap is the entire problem. None of runs #1-#4 closed it. Runs #5-#8 are our remaining 4 attempts before pivoting to the data-collection fallback.

## What's Still Running

- **PID 1558478**: hc443_upside_sl3_tp8_h10 (~4 min in, processing April dates)
- **PID 1559131**: sequencer that auto-launches multi-h, tight, chase after upside finishes
- All four remaining runs will complete by ~04:00 ET if each takes ~30 min

## What's Definitively Killed

| Path | Why dead |
|---|---|
| HC #441 R2-STRICT analytic | Bug, canonical disproves |
| Market-order entry | Spread crossing kills label edge (run #3) |
| Wider SL alone | Same trades, just bigger losers (run #4) |
| Cancel window >h (HC #428 R2 violation) | Survivorship bias, c10 still loses (run #2) |
| HC #437 baseline TP=0.96 SL=0.57 | Pre-known loser (−0.27 tk) |

## What's Still Live (in this 4-run wave)

1. **Wider geometry with long hold** — gives the trade 10s to develop. Risk: HC #428 R2 violation (hold > 1.5 × 1s = 1.5s). But the model's edge persists ~30s per the decay analysis, so it may be defensible as a multi-h fusion.
2. **Multi-horizon confluence** — require top-30% short at 1s AND 5s AND 10s. Fewer trades, hopefully each survives adverse selection. Conceptually closest to HC #428 R2-compliant "multi-h fusion" exemption.
3. **Tight geometry** — 0.5s hold, SL=1, TP=2. Smallest target should be easiest to capture if any move happens; smallest losses if not.
4. **Chase execution** — re-quotes the limit as the book moves. Higher fill rate than passive_at_touch but more risk of buying the top.

## Decision Tree for Thursday Morning (When These 4 Land)

```
IF any config has  mean_tk > 0 AND PF > 1.1 AND R1 + R2 + Sharpe>0  -> CHAMPION FOUND
  -> run regime-stratified 21/15 holdout split
  -> if survives: build deploy script + runbook for Friday cutover
  -> if fails OOS holdout: treat as overfit, fall through to FALLBACK

IF best mean_tk in (-0.1, 0]  AND PF in (0.85, 1.0]               -> MARGINAL
  -> sweep tighter conf bands (top0.1, top0.05) on best config
  -> sweep entry policies: post-only, peg-to-mid+1tick, etc.
  -> deliverable: "we have a near-breakeven config worth paper-trading"

IF best mean_tk < -0.1                                             -> FALLBACK
  -> ship live data-collection harness as Friday deliverable
  -> record what model WOULD HAVE TRADED at every signal across 1 week of live data
  -> deliverable: "production-aware system collecting evidence, not yet a P&L system"
```

## Files Saved For Friday Handoff (per HC #443 R3)

- `/output/state_of_project/STATE_OF_THE_PROJECT_20260519.md` — initial checkpoint (22:55 ET)
- `/output/state_of_project/STATE_OF_THE_PROJECT_20260519_2330ET.md` — this file
- `/output/hc432_v342_47day_validation/hc442_v2_canon_c1_*` — canonical c1 results
- `/output/hc432_v342_47day_validation/hc442_v2_canon_c10_*` — canonical c10 results
- `/output/hc432_v342_47day_validation/hc443_market_entry_tp3_*` — market entry results
- `/output/hc432_v342_47day_validation/hc443_wider_sl2_tp3_*` — wider stop results
- `/scripts/v3_4_research/hc443_multih_confluence_canonical.py` — multi-h confluence runner
- `/tmp/hc443_sequencer.sh` — autonomous sequencer (PID 1559131)
- `/logs/hc443_sequencer.log` — sequencer activity log
- `/logs/hc443_*.log` — individual run logs

## For Friday-Morning Claude (or 04:00 Claude when the sequencer finishes)

1. Read THIS file first. Then check `STATE_OF_THE_PROJECT_*.md` newer than this one if any.
2. Read `/logs/hc443_sequencer.log` to see which of runs #5-#8 completed successfully.
3. For each completed run, check `/output/hc432_v342_47day_validation/{config}_verdict.md` for the canonical verdict.
4. Apply the decision tree above.
5. **Do not** rely on any analytic-resolver number. Every P&L claim must come from a `_fifo_fills.csv` file.
6. **Do not** trust the HC #441 +0.71 tk/fill verdict; it is invalidated.
7. If escalating to user via Discord, follow CLAUDE.md banned-pattern rules: plain English, no paths, no PIDs, no HC#s as primary content, ≤8 lines.

## Discord Message Already Sent Tonight

User has been told (in plain English) that:
- The earlier +0.71 verdict was wrong (analytic bug).
- Honest canonical results are negative.
- We are running the last-hurrah variants overnight.
- If nothing clears, Friday delivers a live-recorder fallback.

No new Discord message planned until at least 2 of the 4 in-flight runs produce verdicts (avoids the premature-verdict failure mode that happened twice tonight).
