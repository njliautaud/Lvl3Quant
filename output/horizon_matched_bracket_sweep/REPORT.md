# Horizon-Matched Bracket Sweep — HC #432 R2 binding
Generated: 2026-05-22 04:44 ET. Wall: 180.3s.

Input fills: `output/hc475_ab/symmetric_gate_fills.parquet`
Replay: FIFO market replay on raw MBO TRADE events (HC #74 binding, no midpoint).
Commission: 0.376 ticks round-trip deducted from gross.

## Grid construction

- Horizons (h): 1s, 5s, 10s, 30s
- TP candidates: 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0 ticks
- SL candidates: 0.25, 0.5, 0.75, 1.0 ticks
- hold_seconds = min(1.5*h, 60); cancel_window_seconds = 0.5*h
- Filters applied at grid construction:
  - SL <= TP/2 (favorable R:R)
  - TP <= p90(realized MFE within h) per side (HC #432 R2)

## p90(MFE_within_h) realized (ticks) — per side

| h | long p90 | short p90 | combined p90 |
|---|---|---|---|
| 1s | 3.000 | 3.000 | 3.000 |
| 5s | 7.000 | 6.000 | 7.000 |
| 10s | 10.000 | 9.000 | 10.000 |
| 30s | 16.000 | 15.000 | 16.000 |

## Headline: 0 cells pass ALL 5 gates

**Status: NONE — no cell passes all 5 gates.**

**Root cause:** Only 7/104 cells are profitable at all (all TP=2.0t / SL=0.25t,
i.e. extremely asymmetric R:R that rides ~1-in-3 winners through 8:1 R:R). All
7 profitable cells fail:
- HC #474 WR floor (max WR = 29.7%, far below 0.55) AND
- HC #344 day-conc (best day_conc = 0.70, most cells > 0.92)

The best Sharpe achievable on these fills under any horizon-matched bracket
is +0.042 (h=10s, TP=2t, SL=0.25t, long-side). This is well below the +2.0
Sharpe override that HC #474 would accept to waive the WR floor. The
symmetric_gate fills' underlying signal **does not generate sufficient edge
to support a passive bracket strategy at any of the 4 horizons** when
HC #432 R2 constraints (TP <= p90(MFE_h), hold <= 1.5h) are honored.

## Top 10 by Sharpe among gate-passing cells

(none)

## Top 10 by Sharpe overall (any gate status)

| h | TP | SL | side | n | Sharpe | PF | WR | mean_ticks | gates_passed |
|---|---|---|---|---|---|---|---|---|---|
| 10s | 2.0 | 0.25 | long | 24393 | +0.042 | 1.10 | 29.7% | +0.043 | 2/4 |
| 30s | 2.0 | 0.25 | long | 24393 | +0.041 | 1.10 | 29.7% | +0.043 | 2/4 |
| 5s | 2.0 | 0.25 | long | 24393 | +0.041 | 1.10 | 29.7% | +0.042 | 2/4 |
| 10s | 2.0 | 0.25 | short | 6693 | +0.024 | 1.06 | 28.9% | +0.025 | 2/4 |
| 30s | 2.0 | 0.25 | short | 6693 | +0.024 | 1.06 | 28.9% | +0.025 | 2/4 |
| 5s | 2.0 | 0.25 | short | 6693 | +0.024 | 1.05 | 28.9% | +0.024 | 1/4 |
| 1s | 2.0 | 0.25 | long | 24393 | +0.000 | 1.00 | 29.3% | +0.000 | 1/4 |
| 5s | 1.0 | 0.25 | short | 6693 | -0.006 | 0.99 | 49.8% | -0.004 | 1/4 |
| 30s | 1.0 | 0.25 | short | 6693 | -0.006 | 0.99 | 49.8% | -0.004 | 1/4 |
| 10s | 1.0 | 0.25 | short | 6693 | -0.006 | 0.99 | 49.8% | -0.004 | 1/4 |

## Failure-mode breakdown

- Total cells executed: 104
- Pass HC #428 R1 (regime-agnostic): 95/104
- Pass HC #432 R2 (MFE-within-horizon): 104/104
- Pass HC #344 (day-conc <= 0.70 AND profitable): 0/104
- Pass HC #474 WR floor (or Sharpe/PF override): 0/104
- Pass ALL 5 gates: 0/104

## Best per-horizon

| h | best Sharpe (any gate) | best cell (TP/SL/side) | passes_all_gates |
|---|---|---|---|
| 1s | +0.000 | TP=2.0 SL=0.25 long | 0 |
| 5s | +0.041 | TP=2.0 SL=0.25 long | 0 |
| 10s | +0.042 | TP=2.0 SL=0.25 long | 0 |
| 30s | +0.041 | TP=2.0 SL=0.25 long | 0 |

## HC #475 R1 long/short balance note

All cells in this sweep are per-side by construction (the symmetric_gate fills are
heavily long-biased — 24,393 long / 6,693 short out of 31,086 = 78.5% long /
21.5% short). Per-side cells satisfy HC #475 R1 only if the opposite-side cell
under same params also passes; check pairs in `bracket_grid_results.parquet`.

## Caveats

- Entry already happened in the fill record. We replay the EXIT only — TP/SL/hold
  applied to the trade-price path starting at `ts_entry_ns`. The `cancel_window_seconds`
  parameter is recorded but does not modify replay (entry already filled in upstream).
- Tie-break on TP/SL same tick: TP wins (favorable for trade, consistent across cells).
- Commission deducted as flat 0.376 ticks (round-trip) per trade.
- Regime label from `output/regime_labels/oot_dates_regime.parquet` (trend_label up/down/flat).
