# Adaptive Exit v1 — Look-Ahead Leak Removed
Generated: 2026-05-22 03:45 ET. Wall: 105.1s.

**Compliance**: HC #469 R4/R5(f) — v1 replaces v0's linear-interpolation toward
the known exit price with EXACT MBO trade-stream replay. Every feature at tick k
is computable from data ≤ tick k only.

## What changed vs v0

| Aspect | v0 | v1 |
|---|---|---|
| In-trade MFE/MAE | LINEAR INTERP toward `net_total_ticks` (LEAK) | Running max/min of actual trade-price diffs |
| Current net at tick k | `net_total * (k/n)` (LEAK) | `(price[k] - entry_price) * sign / tick` |
| Trade-price source | None — synthesized | Raw Databento MBO trade events (action=='T') |
| Walk-forward | 70/30 single split | Rolling 10-day train / 1-day OOT × ≤5 folds |
| Exit-now P&L | linear interp toward exit | actual price at sampled tick − RT commission |

## Headline (v1 OOT replay)

| Metric | Value |
|---|---|
| n_trades (OOT) | 5166 |
| n_oot_days | 5 |
| net ticks / trade (adaptive) | -0.3768 |
| Sharpe (per-trade) | -1.4816 |
| PF | 0.039 |
| WR | 1.4% |
| positive OOT days | 0 / 5 |

## v0 vs v1 comparison

| Policy | mean net_ticks/trade | Sharpe |
|---|---|---|
| v0 (leak-driven, claimed) | +0.6134 | +0.4573 |
| v1 (clean replay) | -0.3768 | -1.4816 |
| Baseline (hold to realized exit) | -0.1488 | -0.0429 |

## VERDICT: **REJECT — v0 was leak-driven; clean replay does not reproduce the +0.61**

### Gate detail
- net_ticks_per_trade > +0.10: NO (got -0.3768)
- Sharpe > 0.30: NO (got -1.4816)
- positive-day share > 60%: 0/5 = 0.0%

### Data scope note

Input fills: 31,086 from `output/hc475_ab/symmetric_gate_fills.parquet` across 15 OOT trading days (2026-02-23 … 2026-03-13).
Rebuilt in-trade rows: 551,295 (≤20 sample ticks per trade).
The spec called for 47 OOT days; the just-landed symmetric-gate fills span only 15 days because the upstream 47-day NPZ's OOT slice for the 5 surviving configs has 16 dates (one with zero fills). Walk-forward adapted to 10 train / 1 OOT × 5 folds.

## Next steps if PASS
- Wire v1 policy into the live paper-trader as a candidate exit policy.
- A/B vs the fixed 30s hold + 10s cancel baseline on Razer live stack (paper).

## Next steps if REJECT
- Adaptive exit is NOT a real edge — v0's +0.61 was the leak. Keep the 30s/10s baseline.
- Document the failure under HC #469 R5(f) and move execution research focus elsewhere.
