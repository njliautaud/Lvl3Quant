# Fear + SmartMoney + Dip Cross-Type Confluence Paper Trading Engine

## Overview
This is a paper trading engine implementing a three-signal confluence strategy designed to identify low-risk entry opportunities in large-cap stocks when multiple market conditions align.

## Strategy Signals

### Signal 1: Market Fear (VIX-based)
- **Trigger**: `VIX > 1.15 × VIX_60day_rolling_mean`
- **Rationale**: Elevated market fear creates asymmetric reward-to-risk for mean-reversion trades
- **Requirement**: Must be ACTIVE for any entries (acts as gating signal)

### Signal 2: Smart Money (Volume/Range Compression)
- **Trigger**: `Daily HL Range < 1.0 × 60-day Average HL Range`
- **Rationale**: Narrowing ranges before moves indicate institutional accumulation/distribution
- **Detection**: Real-time daily bar analysis vs 60-day rolling average

### Signal 3: Stock Dipping (Momentum)
- **Trigger**: `RSI(14) < 40`
- **Rationale**: Oversold conditions increase probability of mean-reversion
- **Detection**: 14-period Relative Strength Index on daily closes

### Confluence Requirement
**All three signals must fire within a 5-day window** (signal 1 provides the 5-day window)

## Universe & Position Sizing
**Stocks**: AAPL, MSFT, GOOGL, AMZN, META, NVDA, JPM, UNH, LLY, AVGO, AMD (11 mega-caps)

**Position Size**: $300 USD per entry
**Max Concurrent Positions**: 2 (prevents concentration risk)

## Exit Rules (in order of priority)

| Exit Condition | Details |
|---|---|
| **Take Profit** | +10% unrealized gain |
| **Stop Loss** | -15% unrealized loss |
| **Max Hold** | 21 calendar days held (regardless of P&L) |
| **Confluence Broken** | VIX drops below 60-day average (signal 1 expires) |

All exits are evaluated daily during the scheduled run.

## Implementation Details

### Engine Location
```
/home/jupiter/Lvl3Quant/paper_engines/cross_type_confluence_paper.py
```

### State Management
- **State File**: `/home/jupiter/Lvl3Quant/paper_engines/state/cross_type_confluence_state.json`
- **Log File**: `/home/jupiter/Lvl3Quant/paper_engines/logs/cross_type_confluence_paper.log`

State persists:
- Open positions (symbol, entry date/price, share count, signals fired)
- Closed trades (entry/exit details, realized P&L, exit reason)
- Last run date (prevents duplicate daily runs)
- VIX signal active flag (for edge cases)

### Cron Schedule
**Daily at 4:50 PM ET (16:50) on weekdays**
```cron
50 16 * * 1-5 cd /home/jupiter/Lvl3Quant && /usr/bin/python3 paper_engines/cross_type_confluence_paper.py >> paper_engines/logs/cross_type_confluence_paper.log 2>&1
```

## Daily Workflow

Each scheduled run executes in this order:

1. **Load persisted state** (open positions, closed trades)
2. **Fetch VIX data** (70-day lookback, compute 60-day rolling mean)
3. **Evaluate fear signal** (VIX > 1.15 × mean?)
4. **Check open positions for exits**:
   - Evaluate TP/SL/max hold/confluence rules
   - Close any triggered positions
   - Record realized P&L
5. **Find new entry signals** (if VIX signal active and room available):
   - Scan all 11 stocks for SmartMoney + Dip signals
   - Execute up to 2 new positions (respecting max concurrent cap)
6. **Print daily summary** (VIX status, open positions, unrealized P&L, closed trade stats)
7. **Save state** (persist for next run)

## Output & Reporting

### Console Summary (printed & logged)
```
================================================================================
DAILY SUMMARY: 2026-08-03
================================================================================
VIX: 15.99 | 60d Avg: 17.40 | Fear Signal: INACTIVE

Open Positions: 0/2
  (None)

Closed Trades: 0
================================================================================
```

### Detailed Logging
- **Level**: DEBUG (full signal details) → INFO (trading events) → ERROR (failures)
- **Timestamp**: All log entries include ISO timestamp + severity
- **Traceability**: Every entry/exit reason logged with signal details

### Trade History
Run `engine.print_trade_history()` to see all closed trades with details:
```
AAPL   | Entry: 2026-08-01 @ $175.50 | Exit: 2026-08-05 @ $192.65 | P&L: $48.99 (+9.34%) | Reason: TP | Shares: 1.71
```

## Dependencies
- `yfinance` — VIX and stock OHLCV data
- `pandas` — Data manipulation (rolling calcs)
- `numpy` — RSI computation
- `dataclasses` — Position/trade serialization
- Python 3.8+

## Running Manually
```bash
cd /home/jupiter/Lvl3Quant
python3 paper_engines/cross_type_confluence_paper.py
```

First run initializes state file. Subsequent runs on the same day skip processing.

## Monitoring
- **Check state**: `cat /home/jupiter/Lvl3Quant/paper_engines/state/cross_type_confluence_state.json`
- **Tail logs**: `tail -f /home/jupiter/Lvl3Quant/paper_engines/logs/cross_type_confluence_paper.log`
- **Verify cron**: `crontab -l | grep cross_type_confluence`

## Key Thresholds
| Parameter | Value | Rationale |
|---|---|---|
| VIX multiplier | 1.15 | 15% above mean = elevated fear without extremes |
| HL range threshold | 1.0 | Current range < average = compression/volume conviction |
| RSI period | 14 | Standard momentum oscillator |
| RSI trigger | < 40 | Oversold without near-bottom extremes |
| TP % | +10% | Captures mean-reversion edge without greed |
| SL % | -15% | 1.5:1 risk/reward ratio |
| Max hold | 21 days | Prevents signal decay & theta bleed |
| Lookback (data) | 70 days | 60 days for rolling calcs + 10 day buffer |

## Design Principles
1. **Multi-factor confluence** reduces false signals vs single indicators
2. **Daily execution** captures intraday opportunities without overtrading
3. **Fixed position size** ensures consistent risk per trade
4. **Hard stops** (TP/SL/hold) prevent emotional decisions
5. **Persistent state** enables autonomous operation without manual re-entry
6. **Rich logging** enables post-trade analysis and signal debugging

## Future Enhancements
- Sector weighting to avoid correlated positions
- Volatility-based position sizing (tighter when VIX spikes further)
- Multi-timeframe confirmation (1h/4h RSI for confirmation)
- Walk-forward backtesting on 5-year data
- Ensemble with other confluence engines

## Troubleshooting

### "Cannot proceed without VIX data"
- Check internet connectivity
- Verify yfinance is installed: `pip install yfinance`
- VIX market hours: 09:30-16:15 ET (similar to stock market)

### No entries appearing despite signals
- Check VIX signal first: log should show "Fear Signal: ACTIVE"
- Verify position count < 2: check state file
- Confirm RSI < 40 and HL range compression in debug logs

### State file corruption
- Delete: `rm /home/jupiter/Lvl3Quant/paper_engines/state/cross_type_confluence_state.json`
- Next run reinitializes fresh (loses trade history, keep backup logs)

---

**Created**: 2026-08-02  
**Last Updated**: 2026-08-03  
**Status**: ACTIVE (cron scheduled)
