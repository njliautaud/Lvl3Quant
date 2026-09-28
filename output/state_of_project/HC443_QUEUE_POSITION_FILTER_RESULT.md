# HC #443 — Queue-Position Filter Probe Result

Generated 2026-05-19 23:55 ET. Probe ran on existing canonical `_fifo_fills.csv` outputs (10 configs × queue_ahead buckets).

## TL;DR

**No queue-position bucket is profitable on any tested config.** Every `(config, q_bucket)` cell with ≥30 fills returns a NEGATIVE mean tk/fill and PF < 1.

The least-bad cell is `hc442_primary_cancel10s_match × q=1-5` at -0.12 tk/fill, PF 0.81, WR 27%, sharpe -1.93. Still losing.

## What this rules out

`queue_ahead` in the fills CSV is computed at order-placement time (line 817 of `fifo_market_replay.py`: `book.qty_at(passive_side, entry_price)`). It is a **pre-trade-known quantity** — a candidate filter that could be applied in live trading ("don't enter if queue ahead > N").

Result: filtering on queue depth at signal time cannot make any of the tested configs profitable. This contradicts the natural hypothesis that "we are losing because we are queued behind toxic flow". Even at q=0/q=1-5 (front of queue), we lose -0.31 tk/fill on the canonical c1 config.

## Implication for HC #443 layered roadmap

Queue-position filtering is **NOT** the missing ingredient. The other levers from HC #442 R2 still untested are:
- Multi-horizon confluence (1s ∧ 5s ∧ 10s agreement) — in flight via phase-3 sequencer
- PatchTST confluence — not yet attempted canonically
- Vol-regime gating — not yet attempted canonically
- Meta-classifier learned on per-signal features — not yet attempted

If h=5s/hold=5s (in flight) also fails, the case for HC #443 FALLBACK (live-data-collection harness as Friday deliverable) gets stronger.

## Data sources

- 10 canonical fills CSVs in `output/hc432_v342_47day_validation/hc44*_fifo_fills.csv`
- Stratification output: `output/hc443_filter_strat/all_slices.csv` (queue_ahead_bucket rows)

## Per-config queue bucket summary

| config | q=0 | q=1-5 | q=6-20 | q=21-100 | q=100+ |
|---|---|---|---|---|---|
| hc442_v2_canon_c1 (PRIMARY) | — | -0.31 | -0.22 | -0.38 | -0.68 |
| hc442_v2_canon_c10 (analytic match) | — | -0.31 | -0.24 | -0.35 | -0.53 |
| hc443_band_top5_short | — | -0.22 | -0.22 | -0.32 | -0.54 |
| hc443_market_entry_tp3 | -0.68 | — | — | — | — |
| hc443_wider_sl2_tp3 | — | -0.65 | -0.43 | -0.74 | -1.64 |
| hc443_tight_sl1_tp2_h05 | — | -0.57 | -0.51 | -0.72 | -0.97 |
| hc443_upside_sl3_tp8_h10 | — | -0.38 | -0.68 | -0.80 | -1.63 |

All values are mean net ticks per fill. All negative. All PF < 1.

## Conclusion

Queue-position is not the missing filter. Continuing to wait for h=5s/hold=5s sweep and phase-3 multi-h confluence verdicts before committing to the fallback path.
