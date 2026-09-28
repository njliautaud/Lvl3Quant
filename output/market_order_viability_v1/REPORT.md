# Market Order Viability Analysis — CNN-Mamba v2 OOT
Generated: 2026-05-23 05:03

## Setup
- Model: CNN-Mamba v2 (bulk_oot_v2 per-date predictions, window=1000, stride=250)
- OOT dates: 46 days (20260224 to 20260427)
- Total events: 2,305,333
- Market order RT cost: 1.376 ticks ($17.20)
- Horizons analyzed: 1s, 5s, 10s

## Key Findings

### NO profitable market-order configurations found.

The best configurations are:
- SHORT 5s top 1%: net -0.418 ticks/trade (avg realized 0.958, need >1.376), WR 45.0%, 191 events/day
- SHORT 10s top 1%: net -0.418 ticks/trade (avg realized 0.958, need >1.376), WR 46.0%, 199 events/day
- SHORT 5s top 2%: net -0.486 ticks/trade (avg realized 0.890, need >1.376), WR 44.3%, 381 events/day
- SHORT 10s top 2%: net -0.486 ticks/trade (avg realized 0.890, need >1.376), WR 45.0%, 399 events/day
- SHORT 5s top 3%: net -0.512 ticks/trade (avg realized 0.864, need >1.376), WR 43.4%, 572 events/day

## Break-Even Analysis

Market order cost = 1.376 ticks. Need avg realized move > 1.376 ticks to profit.

Best short: 5s top 1% — avg 0.958 ticks, gap to breakeven: -0.418 ticks
Best long: 10s top 1% — avg 0.686 ticks, gap to breakeven: -0.690 ticks

## Implications

Pure market orders are NOT viable even at extreme confidence.
Alternative paths:
1. Passive limit orders with adverse-selection-aware placement
2. Hybrid: passive entry + market exit (or vice versa)
3. Smart execution: RL/MLP to learn optimal order type per signal
4. Multi-signal confluence to boost confidence further