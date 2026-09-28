# HC #437 — Sanity-reproduction outcome (Phase 3 attempt)

**Date**: 2026-05-19  •  **Agent**: Jupiter

## Phase 3 results summary

Patched `hc432_v2_baseline_runner.py` to source predictions from
`cnn_mamba_v2_bulk_oot_v2/` (correct-window=1000) instead of
`cnn_mamba_v2_all_oot/` (corrupted-window=3000).

### 3-date repro (20260224, 20260226, 20260301 — first 3 OOT dates that the runner's
`list_dates()` returns)

| run | NPZ source | TP / SL / cancel | n_fills | mean_tk/fill | Sharpe√N | PF | WR |
|---|---|---|---:|---:|---:|---:|---:|
| ORIGINAL (broken) | all_oot (w=3000) | 1.0 / 0.5 / 1.0 | (34 dates batch) | -0.173 | -12.12 | 0.62 | 47% |
| PATCHED-source-only | bulk_oot_v2 (w=1000) | 1.0 / 0.5 / 1.0 | 160 | **-0.270** | -4.63 | 0.48 | 41% |
| PATCHED + HC413-TP | bulk_oot_v2 (w=1000) | 0.4782 / 0.5686 / 1.0 | 160 | **-0.519** | -12.73 | 0.07 | 41% |
| HC #413 baseline (target) | wrapped (bulk_oot_v2 + label fix) | TP1=0.48/TP2=0.96/SL=0.57 horizon-checkpoint | 639 (47d) | **+0.274** | +12.77 | 2.92 | 85% |

Patching the source NPZ alone did NOT reproduce HC #413. A second methodology gap exists.

## Why patching the NPZ source isn't enough

Signal-only verification (no fill simulation, just realized-label mean of selected
top-0.5% short signals, full 47-OOT):

| source | mean IC_1s | N-weighted realized SHORT pnl (ticks) |
|---|---:|---:|
| all_oot (broken) | +0.013 | -0.157 |
| bulk_oot_v2 (correct) | +0.163 | **+0.730** |

So the SIGNAL has +0.73 tk/fill of raw edge under the correct NPZ. But the FIFO replay
engine destroys it down to -0.27 tk/fill (or worse with tighter TP). Why?

**Methodology gap (HC #432 FIFOReplayEngine vs HC #413 backtester):**

| dimension | HC #432 engine | HC #413 backtester |
|---|---|---|
| Exit evaluation | Intra-event (every DBN tick) | Horizon checkpoints (1s/5s/10s endpoints only) |
| TP/SL semantics | Single TP, single SL ticks | TP1 (scalp-half) → TP2 (full MFE) → SL (capped at MAE) brackets |
| Path dependency | Yes — SL fires on any transient adverse tick | No — only end-of-horizon value compared |
| Fill model | True FIFO queue position from MBO replay | Pre-computed FIFO labels (tp4sl3_short_filled), deflated by 0.5 mean-of-queue heuristic |
| Cost model | Engine accounts for spread + commission | Hard-coded 0.376 tk passive commission |
| Cancel window | 1s default in HC #432 config | 10s (40 evals × 250ms stride) default |

The HC #432 engine is strictly more conservative on the exit side. The 1-second signal
has typical MFE ~0.48 ticks but transient adverse moves of 0.5+ ticks happen frequently
within the 1.5s hold window — so SL fires before TP fires (or before the 1s horizon
"would" mean-revert under HC #413's checkpoint methodology). 94 SL hits vs 64 TP hits
in the 3-date repro = exactly this signature.

## Conclusion for HC #437 R1

**HC #413 cannot be reproduced as-is by hc432_v2_baseline_runner.py** — even after the
NPZ-source fix — because the two pipelines use FUNDAMENTALLY DIFFERENT exit
methodologies. The HC #413 backtester's TP1/TP2/SL-horizon-checkpoint model
systematically overstates edge relative to a true FIFO replay because it doesn't model
intra-horizon path drawdowns.

**This is a known-and-significant gap.** It means the HC #413 "+0.274 tk/fill" number
is itself OVERSTATED relative to what a realistic execution stack would achieve. The
HC #432 FIFO replay's "-0.17 tk/fill" overstated the OTHER WAY because it was running
on a corrupted-IC signal.

The true honest baseline number for v2_1s_short_top0.5% would be something in between:
the signal HAS +0.73 ticks/fill of raw edge, but realistic FIFO replay with intra-event
SL evaluation gives back a significant portion. A correct sanity baseline would need
a FIFOReplayEngine variant that:
  1. Sources from `cnn_mamba_v2_bulk_oot_v2/` (corrected)
  2. Implements HC #413's TP1/TP2/SL bracket (so live ops actually match backtest)
  3. Acknowledges that intra-event SL is harsher than checkpoint SL

## Recommended next step (out of scope of HC #437 today)

Port HC #413's bracket-exit logic into the FIFOReplayEngine as a new
`order_management='hc413_bracket'` mode, then re-run the v2 sanity. Until then,
HC #432's verdicts on v3.4.x configs are **suspect** for the same reason — they're
running a strictly-more-conservative exit model than HC #413/HC #415 used to validate
those configs originally.

## What was actually fixed today

- `scripts/v3_4_research/hc432_v2_baseline_runner.py:V2_OOT_DIR` now points at
  `cnn_mamba_v2_bulk_oot_v2/`. This restores the signal IC from 0.013 to 0.163 on the
  47-day OOT.
- The fill simulation gap remains — escalated to user-level decision in `verdict.md`.

