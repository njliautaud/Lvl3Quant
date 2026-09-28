# HC #437 Bug 2 — Final Verdict (2026-05-19, Jupiter resume)

## TL;DR

**HC #437 Bug 2 hypothesis CONFIRMED.** HC #413's published v2 1s short top0.5%
edge (+0.274 tk/fill, PF 2.92, WR 85%) is a *methodology artifact*. It is the
result of evaluating TP/SL ONLY at horizon checkpoints (1s/5s/10s) against the
realized log-return — which structurally underweights drawdowns that an actual
exchange-side bracket order would stop into.

Under the realistic intra-event price-tracking exit (HC #432 `realtime_sl` mode,
which matches live limit-bracket order behavior), the same signal set with the
same TP/SL magnitudes produces **-0.274 tk/fill, PF 0.48, WR 44%, Sh√N -21.3**.

The realtime-vs-bracket gap is **+0.391 tk/fill** — entirely from intra-event
SL hits the bracket harness does not see.

## HC #437 R1 status by mode

| mode | net_tk | PF | WR | Sh√N | R1 verdict |
|---|---|---|---|---|---|
| hc413_bracket | +0.116 | 1.45 | 72.0 | +10.24 | **NOT_SATISFIED** (net/PF/WR below floor) |
| realtime_sl   | -0.274 | 0.48 | 44.0 | -21.32 | **NOT_SATISFIED** (no edge) |

Neither mode produces the +0.274 / PF 2.92 / WR 85% HC #413 reference at full
47-day OOT scale. The 3-day bracket reproduction (+0.240, PF 2.15, WR 77.5%,
n=218) was a favorable subset; magnitudes regress sharply on the other 43 days.

## Recommendation on production baseline

**Default to `realtime_sl` mode as the production methodology.** This matches
live trading reality: a working bracket order at the exchange will stop on
intra-event price excursions, not at horizon checkpoints. The bracket harness
is useful for back-testing diagnostic comparison ONLY (it answers "what's the
horizon-checkpoint expected edge of this signal?"), not for go/no-go on
live capital deployment.

Concretely:
- All HC #432/HC #436 v3.4.x acceptance reports built on `realtime_sl` are
  the methodologically correct ones. Their FAIL verdicts STAND.
- Any analysis/leaderboard that uses `hc413_bracket` or
  `scripts/hc413_scalping_backtester/*` exit semantics overstates edge by
  ~0.4 tk/fill and must be flagged in the leaderboard with a "BRACKET-MODE"
  badge. Numbers from these files cannot be used to size live positions.

## Phase D status

Phase D (re-evaluate v3.4.x candidate configs under both modes) was **GATED on
Phase B satisfying R1**, per the task brief. R1 was NOT satisfied under either
mode at 47-day scale. Phase D is therefore **NOT EXECUTED** in this session.

Per the bracket-vs-realtime gap, we already know the most likely outcome of
running Phase D: any v3.4.x config currently labeled PASS-BRACKET (HC #413
methodology) will collapse to FAIL under realtime_sl by ~0.4 tk/fill. None of
today's HC #432 v3.4.x FAIL configs is likely to flip to PASS by reverting to
bracket exits — they were already FAIL on realtime_sl which is the harder bar,
but bracket gives a +0.4tk bonus, so it MAY tip a borderline FAIL to PASS-BRACKET.
That's a paper edge, not a deployable edge. Skipping the run is the correct call.

## Top 3 v3.4.x candidates that PASS-BOTH (HC #436 end-of-week goal)

**Zero candidates currently satisfy PASS-BOTH.** The realtime_sl floor is not
being met by any v3.4.x config tested through 2026-05-19. The end-of-week goal
of "3 production-ready PASS-BOTH configs" requires either:

1. A signal model improvement that lifts edge by ≥0.4 tk/fill (so it survives
   the bracket → realtime delta), OR
2. An exit-side improvement (smarter SL placement that doesn't get clipped
   intraevent — e.g. wider SL with smaller size, or queue-aware SL), OR
3. A confluence/filter that screens out the signals that experience the worst
   intra-event SL excursions (the 1934 SL fills in realtime_sl mode are
   what's killing PF; if we can filter those before entry, edge returns).

These are research directions, not configurations to launch tonight.

## Files produced this session

- `bracket_mode_runcmd.md` — canonical bracket-mode run command + Phase A verification
- `sanity_47day_bracket_mode.md` — Phase B full 47-day bracket result + R1 verdict + regime strat + per-day table
- `two_mode_comparison.md` — Phase C bracket vs realtime_sl side-by-side + per-day delta + regime strat
- `bug2_final_verdict.md` — this file (Phase E)
- `v2_short_1s_top0.5_bracket_HC437_47day_bracket_fifo_{fills.csv,summary.json}` — 47-day bracket raw fills
- `v2_short_1s_top0.5_bracket_HC437_47day_PRIOR_AGENT_*` — preserved copy of prior agent's identical run
- `bracket_47day_per_day.csv` / `realtime_sl_47day_per_day.csv` — per-day metric tables
- `regime_lookup_v2drift.{json,csv}` — green/red/flat classification per day (drift-from-NPZ-labels proxy; no ES daily close file was available on this node)
- `regime_stratified.json` — per-mode green/red regime metrics
- `bracket_47day_progress.json` — combined progress checkpoint

Realtime_sl matched-params output lives at
`output/hc432_v342_47day_validation/v2_short_1s_top0.5_realtime_sl_HC437_47day_bracketparams_*`.
