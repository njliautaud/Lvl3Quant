# Wheel Strategy Live Deployment Checklist
## Generated 2026-07-08 — Updated when paper track record reaches milestones

---

## PHASE 0: Paper Track Record (IN PROGRESS — Day 2 of 30)

### Minimum 30-Day Paper Requirements Before ANY Live Capital
- [ ] 30 calendar days of paper trading (currently: 2 days)
- [ ] At least 50 completed round-trip trades across all variants
- [ ] Win rate > 55% sustained over 30 days
- [ ] No single-day loss > 5% of NAV
- [ ] Drawdown trigger activates and recovers correctly at least once
- [ ] Earnings filter correctly avoids all earnings weeks (verify vs actual earnings calendar)
- [ ] Paper engine stability: no crashes, no state corruption, no missed market days

### Strategy Selection Gate (at Day 30)
- [ ] Compare all 7 paper engine variants on:
  - Realized Sharpe (annualized from daily mark-to-market)
  - Max drawdown (peak to trough)
  - Win rate on completed trades
  - Average days to profit-take
  - Worst single-day P&L
- [ ] Select top 2 variants for initial live deployment
- [ ] REJECT any variant with paper Sharpe < 0.5 annualized

---

## PHASE 1: Broker Setup (DO BEFORE Day 30)

### Recommended Broker: Interactive Brokers (IBKR)
- [ ] Account type: Individual margin account
- [ ] Enable options trading (Level 3: spreads required for BPS/IC)
- [ ] Enable real-time market data for:
  - US Equities (Level 2 optional but recommended)
  - US Equity Options (OPRA data)
  - Index options (VIX for regime gate)
- [ ] API access enabled (TWS API or IB Gateway)
- [ ] Paper trading account activated for final validation
- [ ] Verify commission rates: target ≤ $0.65/contract

### Alternative Broker Options
- Tastytrade: Lower options commissions ($0.50/contract), good for beginners
- TD Ameritrade/Schwab: $0.65/contract, good platform
- NOTE: Rithmic (current ES broker) does NOT support equity options well

### Account Sizing
- Initial deployment capital: $25,000 - $50,000 (quarter-Kelly per Monte Carlo)
- Max margin utilization: 50% of NAV (leave 50% cash reserve)
- Per-name cap: 3% of NAV ($750 - $1,500 per position on $25K-$50K)
- Assignment reserve: enough cash to take assignment on 3 names simultaneously

---

## PHASE 2: Strategy Configuration for Live

### Recommended First Strategy: CSP Diversified 20-Name
**Why first**: Lowest MaxDD (-13.1%), simplest execution (1 leg), regime-robust

Config:
```
PUT_DELTA = 0.25
DTE_TARGET = 14 days
PROFIT_TAKE = 50%
MARGIN_CAP = 20% (start conservative, scale to 30% after 60 days)
PER_NAME_CAP = 3%
BEAR_GATE = SPY < 50d SMA
EQUITY_CURVE_BRAKE = 3% DD from 60d peak
DD_TRIGGER = -5% trailing 3-day return
EARNINGS_BUFFER = 2 days before/after
```

### Scale-Up Path
1. **Month 1-2**: CSP Diversified only, 20% margin cap
2. **Month 3**: If Sharpe > 1.0 on live, scale to 30% margin cap
3. **Month 4**: Add BPS alongside CSP (separate capital allocation)
4. **Month 6**: If IC paper track record is strong, consider adding IC

---

## PHASE 3: Automation & Monitoring

### Required Infrastructure
- [ ] Order execution script connected to broker API
- [ ] Real-time portfolio monitoring (positions, margin, P&L)
- [ ] Daily automated report (already built: `scripts/wheel_daily_report.py`)
- [ ] Alert system for:
  - Position assigned (need to decide: sell CC or close assignment)
  - Margin utilization > 35%
  - Single-day loss > 2%
  - Bear gate activation (SPY crosses below 50d SMA)
  - Drawdown trigger activated
  - Earnings approaching for held position

### Risk Controls (HARD LIMITS — never override)
- MAX gross margin: 40% of account equity (50% absolute ceiling)
- MAX per-name: 5% of equity
- MAX sector concentration: 20% of equity
- HALT new positions if: trailing 5-day return < -8%
- CLOSE ALL if: account equity drops below 70% of initial capital

---

## PHASE 4: Ongoing Operations

### Daily Routine (automated where possible)
1. Pre-market (8:30 ET): Check overnight gaps for held positions
2. Market open (9:30 ET): Check assignment notifications
3. Mid-day (12:00 ET): Check profit-take levels, roll approaching expirations
4. Market close (16:00 ET): Daily P&L log, margin check
5. After-hours: Run daily report script

### Weekly Routine
1. Friday close: Review expiring positions, plan next week's entries
2. Weekend: Compare paper vs live performance (should converge)
3. Check upcoming earnings calendar for basket names

### Monthly Routine
1. Full performance review: Sharpe, Sortino, MaxDD, WR vs paper track record
2. Basket rebalancing: remove underperformers, add new candidates
3. Cost analysis: actual fills vs BS-model expectations
4. Risk parameter review: adjust margin cap based on realized vol

---

## CRITICAL WARNINGS

1. **DO NOT skip paper phase.** BS-modeled pricing is optimistic. Real fills will be worse.
2. **Start small.** First month should be ≤ 20% margin utilization regardless of backtest results.
3. **Assignment happens.** Budget for it. Have cash to take delivery AND sell covered calls.
4. **Earnings will surprise you.** The 2-day buffer is minimum. Consider 1 week for mega-caps.
5. **Ex-dividend dates cause early assignment.** Paper engines don't model this. For live, add ex-div date filter (avoid selling CSPs expiring through ex-div week).
6. **Correlation spikes in crashes.** The diversified basket will not protect in a market-wide crash.
   COVID MaxDD was -45.9% for BPS, -13.1% for diversified CSP. Plan for the worst.
6. **Paper ≠ Live.** Slippage on real options fills (especially IC with 4 legs) will be 2-5x the BS model.
   If paper IC Sharpe is 3.0, expect live Sharpe 1.5-2.0 at best.

---

## METRICS TO TRACK (Deployment Dashboard)

| Metric | Target | Red Flag |
|--------|--------|----------|
| Annualized Sharpe | > 1.5 | < 0.8 |
| Win Rate | > 58% | < 50% |
| Max Single-Day Loss | < -2% | > -4% |
| Max Drawdown | < -15% | > -25% |
| Margin Utilization | 20-30% | > 40% |
| Avg Days to Profit-Take | 5-10 days | > 14 days |
| Assignment Rate | < 5% | > 15% |
| Slippage vs Model | < 5% | > 15% |
