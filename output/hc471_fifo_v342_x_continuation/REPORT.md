# HC #471 R4 — DOES +3.40 TICKS NET SURVIVE CANONICAL FIFO REPLAY?

Setup: 10s pred top-1% × continuation top-20%. Per-day quantiles.
FIFO config: TP=4, SL=3, hold=30s, cancel=10s, passive_at_touch.
Days: 17. Wall: 1444s.

## (a) NAIVE fill-at-touch (the +3.40 headline assumption)
  n_trades=479
  gross_mean=-0.319 ticks
  net_passive_mean=-0.695 ticks
  net_passive_total=-333.10 ticks
  hit_rate=45.1%

## (b) CANONICAL FIFO with realistic queue position
  n_signals=351
  n_filled=351
  fill_rate=100.0%
  net_ticks_mean=-0.457 ticks
  net_ticks_total=-160.48 ticks
  hit_rate=42.7%

## VERDICT
  Naive→FIFO gap = -0.238 ticks (lost to queue dynamics).
  ❌ NEGATIVE under canonical FIFO. Queue position kills the edge.