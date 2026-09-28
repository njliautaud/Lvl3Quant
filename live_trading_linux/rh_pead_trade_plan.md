# PEAD Trade Plan — Robinhood Agentic Account

## Data Backing
- PEAD Drift study: 211 trades, Sharpe 0.785, WR 60.7%, PF 1.52, p=0.015
- 10-15% gaps: 64.1% WR, +1.4% avg drift over 2 days
- 15%+ gaps: 81% WR, +4.3% avg drift over 2 days

## Trade 1: NFLX (Jul 16 PM earnings)
- **Ticker stats:** 13 PEAD trades, 84.6% WR, +4.1% avg drift
- **Trigger:** Gap > 7% on Jul 17 AM open
- **Action:** If gap UP → bull call spread. If gap DOWN → bear put spread.
- **Strikes:** $5 wide spread around the gap-adjusted price
- **Expiration:** Jul 24 (gives buffer past 2-day hold)
- **Budget:** $X max (1 contract)
- **Exit:** Market close Jul 21 (Monday) or Jul 22 (Tuesday)
- **Skip if:** Gap < 7%

## Trade 2: INTC (Jul 23 PM earnings)
- **Ticker stats:** 17 PEAD trades, 70.6% WR, +1.7% avg drift
- **Trigger:** Gap > 7% on Jul 24 AM open
- **Action:** Same as NFLX — direction follows gap
- **Expiration:** Jul 31
- **Budget:** $X max
- **Exit:** 2 days after entry

## Avoid
- TSLA: 50% WR (no edge)
- GOOGL: negative avg drift
- BA: -2.3% avg (wrong direction)
- NVDA: negative avg drift

## Risk Management
- Max $X per trade (34% of account)
- Max 2 concurrent trades
- Hard exits at planned hold period — no holding to expiry
- Skip if gap < 7% (data shows no edge below that threshold)
