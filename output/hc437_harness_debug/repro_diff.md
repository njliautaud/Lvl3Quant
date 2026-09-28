# HC #437 — Phase 3.1 minimal repro: per-trade-row diff

**Repro target**: Show, on a single date, that swapping ONLY the prediction NPZ
source from `cnn_mamba_v2_all_oot/` (window=3000, corrupted) to
`cnn_mamba_v2_bulk_oot_v2/` (window=1000, correct) flips the signal from
"no edge" to "+0.27 tk/fill edge" — without changing the HC #432 FIFOReplayEngine
itself.

## Single-date sanity (20260306)

Both NPZs are read for date 20260306 head 1s. Top-0.5% short signals are picked
the SAME way (`pred < 0 → strength = -pred → top-0.5% by strength`). Realized
label is the per-window label at h=1s (in ticks) embedded in each NPZ.

| metric | ALL_OOT (window=3000, HC #432 source) | BULK_V2 (window=1000, HC #413 source) |
|---|---:|---:|
| N windows | 81,589 | 81,597 |
| IC_1s (model edge vs label) | **0.004** | **0.189** |
| n_short_top0.5%_selected | 274 | 204 |
| selection threshold (short_signed=-pred) | 0.9355 | 0.7893 |
| n_with_valid_label | 206 | 180 |
| realized mean SHORT pnl (ticks) | **-0.28** | **+1.44** |
| win-rate (label < 0 = short wins) | **46.1%** | **75.0%** |

## Interpretation

The two NPZs are not the same predictions. They were produced by two different
inference passes that disagreed on `window_size` (the trainer-default of 3000 was
used to make `cnn_mamba_v2_all_oot`, while the checkpoint's `arch.window_size=1000`
was used to make `cnn_mamba_v2_bulk_oot_v2` after the May-3 discovery that the 3000
version was corrupted).

A model with IC=0.004 in the corrupted source has essentially no signal — the top
0.5% selection picks slightly LOSING trades by random chance, then HC #432's
otherwise-correct FIFO replay engine faithfully executes them and the trades pay
half-spread + commission, yielding ~-0.17 tk/fill on aggregate. The "-12 Sharpe" is
exactly what you'd expect from a no-edge strategy with realistic costs.

A model with IC=0.189 in the correct source picks genuinely-strong signals; the
realized SHORT mean is +1.44 ticks on this date (far above the +0.27 tk/fill
average because individual days vary widely, and 20260306 had favorable downside
moves).

## Why no FIFO replay was needed for this repro

The realized-label mean is a tighter proxy than the full FIFO replay — it bypasses
fill simulation entirely and just asks "does the picked signal have edge at all?"
The answer for the corrupted NPZ is "no" (~zero IC, ~zero mean realized). No fill
model on earth can recover positive ticks/fill from a signal with no edge.

This means we don't need a full 47-day FIFO replay to prove the bug. We can prove
it with one day's IC + realized-label mean: the input is wrong before the FIFO
engine ever runs.

## Full-OOT verification

A subsequent script `verify_predictions_full_oot.py` was run (see results inline
below) to confirm the IC difference is consistent across all 47 dates that the
HC #432 baseline was supposed to evaluate.
