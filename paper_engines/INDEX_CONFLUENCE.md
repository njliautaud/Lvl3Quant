# Fear + SmartMoney + Dip Cross-Type Confluence Paper Engine

## Quick Start
```bash
# View today's status
cat /home/jupiter/Lvl3Quant/paper_engines/state/cross_type_confluence_state.json

# Tail the live log
tail -f /home/jupiter/Lvl3Quant/paper_engines/logs/cross_type_confluence_paper.log

# Manual run (safe - idempotent)
cd /home/jupiter/Lvl3Quant && python3 paper_engines/cross_type_confluence_paper.py
```

## Documentation Files

| File | Purpose | Size |
|------|---------|------|
| `cross_type_confluence_paper.py` | Main engine implementation | 19 KB |
| `CROSS_TYPE_CONFLUENCE_README.md` | Full strategy guide with examples | 6.8 KB |
| `CONFLUENCE_IMPLEMENTATION_SUMMARY.txt` | Detailed implementation report | 11 KB |
| `CONFLUENCE_QUICK_REFERENCE.txt` | Single-page cheat sheet | 13 KB |
| `INDEX_CONFLUENCE.md` | This file - navigation index | - |

## Strategy at a Glance

**Three-Signal Confluence:**
1. **Fear (VIX Gate)**: VIX > 1.15 × 60d_mean → Activates entry window
2. **Smart Money**: Daily HL range < 1.0 × 60d avg HL → Institutional accumulation
3. **Dip**: RSI(14) < 40 → Oversold bounce opportunity

**Position Management:**
- Entry: $300 USD per trade
- Max concurrent: 2 positions
- Exit: TP (+10%), SL (-15%), Max hold (21d), Confluence break

**Universe:** AAPL, MSFT, GOOGL, AMZN, META, NVDA, JPM, UNH, LLY, AVGO, AMD

## File Structure

```
/home/jupiter/Lvl3Quant/paper_engines/
├── cross_type_confluence_paper.py       [MAIN ENGINE]
├── state/
│   └── cross_type_confluence_state.json  [PERSISTENT STATE]
├── logs/
│   └── cross_type_confluence_paper.log   [EXECUTION LOG]
├── CROSS_TYPE_CONFLUENCE_README.md       [FULL DOCS]
├── CONFLUENCE_IMPLEMENTATION_SUMMARY.txt [REPORT]
├── CONFLUENCE_QUICK_REFERENCE.txt        [CHEAT SHEET]
└── INDEX_CONFLUENCE.md                   [THIS FILE]
```

## Scheduled Execution

**Cron Schedule:** `50 16 * * 1-5` (4:50 PM ET, weekdays)

The engine runs automatically every weekday at 4:50 PM ET. Each run:
1. Loads persisted state
2. Fetches current VIX and stock data
3. Evaluates exit conditions for open positions
4. Scans for new entry signals (if fear gate active)
5. Logs daily summary and saves state

**Runs are idempotent** - safe to manually re-run on the same day.

## Key Thresholds

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| VIX multiplier | 1.15 | 15% above mean (elevated without extreme) |
| HL range | 1.0x | Current must be below average (compression signal) |
| RSI period | 14 | Standard momentum oscillator |
| RSI trigger | < 40 | Oversold without near-bottom extremes |
| TP | +10% | Captures mean-reversion edge |
| SL | -15% | 1.5:1 risk/reward ratio |
| Max hold | 21 days | Prevents theta decay and signal staleness |

## State Management

The engine maintains persistent state across runs:

**cross_type_confluence_state.json:**
- `last_run_date` - Prevents duplicate daily executions
- `vix_signal_active` - Current fear gate status
- `positions` - Dict of open positions (symbol → Position object)
- `closed_trades` - List of realized trades with P&L

**Position object:**
- Symbol, entry date/price, shares, entry value ($300)
- Signals fired (fear, smart_money, dip flags)

**ClosedTrade object:**
- Entry/exit dates and prices
- Realized P&L and P&L%
- Exit reason (TP, SL, hold_expired, vix_dropped)

## Monitoring

**Real-time monitoring:**
```bash
# Watch logs as they arrive
tail -f /home/jupiter/Lvl3Quant/paper_engines/logs/cross_type_confluence_paper.log

# Check current state
cat /home/jupiter/Lvl3Quant/paper_engines/state/cross_type_confluence_state.json | jq .

# Verify cron is scheduled
crontab -l | grep cross_type_confluence
```

**What to look for:**
- `VIX: 15.99 | Fear Signal: INACTIVE` → Engine waiting for fear gate
- `ENTRY SIGNAL | Price: $175.50` → New entry found (will open if room available)
- `TP HIT | Entry: $175.50 | Current: $192.65 | Gain: +10.00%` → Take profit triggered
- `Confluence Broken (VIX dropped)` → VIX signal expired, position closed

## Manual Operations

**Manually run the engine:**
```bash
cd /home/jupiter/Lvl3Quant
python3 paper_engines/cross_type_confluence_paper.py
```
Safe to run anytime - checks `last_run_date` to prevent double-processing.

**Reset state (loses trade history!):**
```bash
rm /home/jupiter/Lvl3Quant/paper_engines/state/cross_type_confluence_state.json
```
Next run creates fresh state file.

**View closed trade history:**
```python
from paper_engines.cross_type_confluence_paper import PaperTradingEngine
engine = PaperTradingEngine()
engine.print_trade_history()
```

## Customization

Edit constants at the top of `cross_type_confluence_paper.py`:

```python
VIX_MULTIPLIER = 1.15              # Adjust fear threshold
RSI_THRESHOLD = 40                 # Adjust dip sensitivity
HL_RANGE_THRESHOLD = 1.0           # Adjust compression signal
TP_PERCENT = 0.10                  # Adjust take profit
SL_PERCENT = 0.15                  # Adjust stop loss
MAX_HOLD_DAYS = 21                 # Adjust max hold period
POSITION_SIZE = 300.0              # Adjust entry size (USD)
MAX_CONCURRENT_POSITIONS = 2       # Adjust max open positions
```

## Performance Tracking

After trades close, review:

```json
{
  "closed_trades": [
    {
      "symbol": "AAPL",
      "entry_date": "2026-08-01",
      "entry_price": 175.50,
      "exit_date": "2026-08-05",
      "exit_price": 192.65,
      "pnl": 48.99,
      "pnl_percent": 0.0934,
      "exit_reason": "TP"
    }
  ]
}
```

Track key metrics:
- **Win rate** = wins / total trades
- **Avg win** = sum(positive PnL) / win count
- **Avg loss** = sum(negative PnL) / loss count
- **Profit factor** = total wins / abs(total losses)
- **Cumulative P&L** = sum of all pnl values

## Troubleshooting

| Issue | Solution |
|-------|----------|
| No entries appearing | Check VIX signal in logs (may be INACTIVE) |
| Cron not running | Verify with `crontab -l | grep cross_type` |
| yfinance errors | `pip install --upgrade yfinance` |
| State file corruption | `rm state/*.json` to reset (loses history) |
| Wrong signal thresholds | Edit constants and re-run manually |
| Log file too large | Log rotates automatically; can safely delete old logs |

## Dependencies

**Python packages:**
- `yfinance` - Market data
- `pandas` - Time series calculations
- `numpy` - RSI computation
- `dataclasses` - Data serialization (built-in, Python 3.7+)

**Verify with:**
```bash
python3 -c "import yfinance, pandas, numpy; print('OK')"
```

## Architecture Overview

```
┌─────────────────────────────────────┐
│ Cron Job (4:50 PM ET weekdays)      │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ Load Persisted State (JSON)         │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ Fetch VIX & Stock OHLCV Data        │
│ (yfinance, 70-day lookback)         │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ Evaluate Fear Gate (VIX > 1.15*mean)│
└──────────────┬──────────────────────┘
               │
               ├─ ACTIVE ──► Scan universe
               │ (Open positions checked)
               │             │
               │             ├─ TP Hit?   ─► Close (realized P&L)
               │             ├─ SL Hit?   ─► Close (realized P&L)
               │             ├─ Max Hold? ─► Close (realized P&L)
               │             ├─ Confluence? ─► Close (realized P&L)
               │             │
               │             └─ Find new entries
               │               (SmartMoney + Dip)
               │               │
               │               └─ Open if room
               │
               └─ INACTIVE ─► Close any open (confluence break)
                              Exit summary
                              Save state
```

## Version History

| Date | Version | Status | Notes |
|------|---------|--------|-------|
| 2026-08-02 | 1.0 | ACTIVE | Initial creation & testing |
| 2026-08-03 | 1.0 | ACTIVE | Cron scheduled, docs complete |

## Support

For issues or enhancements:
1. Check logs first: `tail -f paper_engines/logs/cross_type_confluence_paper.log`
2. Verify state file: `cat paper_engines/state/cross_type_confluence_state.json`
3. Review README: `cat paper_engines/CROSS_TYPE_CONFLUENCE_README.md`
4. Check quick reference: `cat paper_engines/CONFLUENCE_QUICK_REFERENCE.txt`

---

**Engine Status:** READY FOR PRODUCTION  
**Last Updated:** 2026-08-03  
**Cron Status:** SCHEDULED (50 16 * * 1-5)
