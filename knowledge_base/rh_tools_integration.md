# Robinhood MCP Tools Integration Guide

**Last updated:** 2026-07-28  
**Purpose:** Document available Robinhood MCP tools and integration patterns for the agentic signal aggregator and paper engines.

---

## 1. Available Scan IDs

### Saved Scans (from Robinhood Legend/Beacon)

The user has created several saved scans in Robinhood that can be executed via `run_scan`:

| Scan Name | Scan ID | Purpose | Use Case |
|-----------|---------|---------|----------|
| Upcoming Earnings | *Retrieved via `get_scans`* | Identifies stocks reporting earnings in next N days | Flag high-risk trading windows for option positions |
| High Options Volume/IV | `<SCAN_ID>` | Unusual options activity + elevated implied vol | Confluence: confirms volatility expansion signals |
| Custom (LGBM top picks) | *To be created* | Can create scan from current signal output | Real-time validation of aggregator picks |

**How to use:**
```python
# Get all available scans
scans = get_scans()  # Lists all user's scans with IDs, filters, status

# Execute a specific scan (returns live results)
results = run_scan(scan_id="<SCAN_ID>")
# Returns: scan title, matching instruments, columns, sort order, live market data
```

---

## 2. Real-Time Market Data Tools

### `get_equity_quotes`
Fetch current bid/ask and last trade for one or more symbols (max 20 per call).

**Use in aggregator:**
- Replace yfinance calls where speed/reliability matters
- Validate signal ticker prices before generating recommendations
- Check liquidity (bid/ask spread < N bps)

**Example:**
```python
quotes = get_equity_quotes(symbols=["XLK", "XLE", "XLF"])
# Returns: bid, ask, last, timestamp, prior close
```

### `get_equity_historicals`
OHLCV bars for equity symbols across a time range. Auto-selects interval if not specified.

**Use in aggregator:**
- Compute technical indicators (RSI, momentum, moving averages) if yfinance unavailable
- Validate signal recency (bar data should match market_data timestamps)

**Example:**
```python
bars = get_equity_historicals(
    symbols=["XLK"],
    start_time="2026-07-23T00:00:00Z",  # RFC3339 UTC
    interval="day"
)
# Returns: open, high, low, close, volume per bar
```

---

## 3. Technical Indicators

### `get_equity_technical_indicators`
Compute indicators on symbol's OHLCV bars. Supports:
- **RSI** (default period 14)
- **MACD** (fast 12, slow 26, signal 9)
- **Bollinger Bands** (period 20, num_std 2)
- **ATR** (period 14)
- **ADX** (period 10, trend strength)
- **SMA/EMA** (moving averages)
- **Keltner Channels**, **Supertrend**, **Pivot Points**, **VWAP**, **OBV**

**Use in aggregator:**
- Cross-reference LGBM signals with technical confluence
- Example: LGBM bull signal on XLK + RSI(14) > 60 + above SMA(50) = HIGH confidence
- Example: IV Run-Up straddle setup + Bollinger Bands squeeze = optimal entry

**Example:**
```python
rsi = get_equity_technical_indicators(
    symbol="XLK",
    type="rsi",
    period=14,
    interval="day",  # REQUIRED — no auto-select for indicators
    start_time="2026-07-01T00:00:00Z",
    output="latest"  # Only return most recent value
)
# Returns: series of RSI values; with output="latest", just the final bar

macd = get_equity_technical_indicators(
    symbol="XLE",
    type="macd",
    fast_period=12,
    slow_period=26,
    signal_period=9,
    interval="day",
    start_time="2026-07-01T00:00:00Z"
)
# Returns: MACD line, Signal line, Histogram per bar
```

**Integration pattern:**
```python
# Current aggregator: "strong_momentum" if mom_21d > 3% AND rel_str > 1%
# ENHANCED: Add technical confirmation
from robinhood_mcp import get_equity_technical_indicators

signal = {"ticker": "XLK", "direction": "bull", "confidence": 0.65}

# Fetch RSI + MACD for confirmation
rsi_val = get_equity_technical_indicators(
    symbol="XLK", type="rsi", period=14,
    interval="day", start_time=...
)["rsi"][-1]  # Latest RSI

if rsi_val > 60 and signal["direction"] == "bull":
    signal["confidence"] *= 1.15  # Boost by 15% (cap at 1.0)
    signal["confirming_sources"].append("rsi_bullish")

elif rsi_val < 40 and signal["direction"] == "bear":
    signal["confidence"] *= 1.15
    signal["confirming_sources"].append("rsi_bearish")
```

---

## 4. Options Chain & Liquidity Checks

### `get_option_chains`
List all expirations and contracts available for an underlying.

**Use before placing any option trade:**
- Verify option exists for recommended strike/expiry
- Check `settle_on_open` flag (true for index options that settle at open)
- Confirm chain is actively traded (not frozen/halted)

**Example:**
```python
chains = get_option_chains(underlying_symbol="XLK")
# Returns: list of expiration dates, underlying ID, chains
# Each chain has: id, symbol, settle_on_open, expirations, state
```

### `get_option_instruments`
Fetch specific option contracts (calls/puts) for a strike/expiry.

**Use to:**
- Get `option_id` needed for order placement
- Filter by type (call/put), strike, expiry, state (active/expired)

**Example:**
```python
contracts = get_option_instruments(
    chain_symbol="XLK",
    expiration_dates="2026-08-15",
    strike_price="185.0",
    type="call"
)
# Returns: call/put contracts with UUIDs, bid/ask, etc.
```

### `get_option_quotes`
Real-time bid/ask and implied vol for one or more option contracts.

**Use to:**
- Validate liquidity before trade: bid/ask width < 0.50 (0.5 ticks)
- Check OI (open interest) — ideally >= 50 contracts for liquidity
- Compare implied vol across strategies

**Example:**
```python
quotes = get_option_quotes(instrument_ids=[option_id_1, option_id_2])
# Returns: bid, ask, midpoint, IV, volume, open interest per contract
```

### `get_option_positions`
List all open/closed option positions in an account.

**Use to:**
- Verify position before exercise/closing
- Check open P&L
- Prevent duplicate entries (if system tries to open same option twice)

---

## 5. Order Simulation & Execution

### `review_equity_order` / `review_option_order`
**Simulate an order without placing it.** Returns:
- Current market quote
- Pre-trade alerts (buying power, PDT, halt, etc.)
- Estimated fees + collateral (for options)
- Estimated execution price

**Use in aggregator:**
- Before calling `place_equity_order` or `place_option_order`
- Validate order parameters (side, type, quantity, price)
- Alert if buying power insufficient or trade violates PDT/halts

**Example:**
```python
review = review_option_order(
    account_number="12345",
    legs=[{
        "option_id": option_uuid,
        "side": "buy",
        "position_effect": "open"
    }],
    quantity=1,
    type="limit",
    price="2.50",
    chain_symbol="XLK",
    underlying_type="equity"
)
# Returns: current_quote, alerts[], fees, collateral_required, notes
```

### `place_option_order`
**REAL MONEY.** Only call after user explicitly confirms reviewed order.

**Mandatory workflow:**
1. Call `review_option_order` (shows alerts + estimated cost)
2. User acknowledges alert
3. Call `place_option_order` with SAME parameters
4. Poll `get_option_orders` to verify fill

---

## 6. Integration Checklist for Aggregator

The aggregator currently:
- ✅ Loads 13 signal sources (LGBM, momentum, RSI, relative strength, etc.)
- ✅ Computes confidence from source confluence
- ✅ Recommends single-leg options (Level 2 account)
- ✅ Applies earnings filter (yfinance)

**Next enhancements (in priority order):**

### Phase 1: Real-Time Validation (Quick)
- [ ] Replace yfinance VIX fetch with `get_equity_quotes(symbols=["^VIX"])` (more reliable)
- [ ] Use `get_equity_quotes` to validate all signaled tickers exist + current price
- [ ] Add liquidity check: fetch bid/ask spread, reject if > 2% (illiquid)

### Phase 2: Technical Confluence (Medium)
- [ ] Fetch RSI(14) + MACD for each bullish signal
- [ ] Boost confidence +15% if RSI/MACD align with direction
- [ ] Flag RSI extremes (>70 or <30) as potential reversal risk
- [ ] Add Bollinger Bands squeeze detection (vol compression = breakout setup)

### Phase 3: Options Liquidity (Medium)
- [ ] After generating option recommendation:
  - Fetch `get_option_chains(underlying_symbol=ticker)`
  - Call `get_option_instruments` for recommended strike/expiry
  - Fetch `get_option_quotes` to check bid/ask width + OI
  - Reject if OI < 50 or spread > $0.50 (0.5 ticks)

### Phase 4: Pre-Trade Simulation (Advanced)
- [ ] Before any execution, call `review_option_order` to simulate
- [ ] Check alerts (buying power, PDT, halt)
- [ ] Log estimated collateral + fees
- [ ] User reviews + confirms via Discord
- [ ] Call `place_option_order` with ref_id for idempotency

### Phase 5: Earnings Calendar Integration (Advanced)
- [ ] Replace yfinance earnings checks with `get_earnings_calendar`
  - More reliable, covers more edge cases (earnings dates vs. times)
  - Supports market-cap filtering (only large-caps if needed)

---

## 7. Error Handling Patterns

### Rate Limiting
- Robinhood MCP has per-minute rate limits
- **Solution:** Cache results for 5 min, batch requests, implement backoff

```python
from functools import lru_cache
from datetime import datetime, timedelta

@lru_cache(maxsize=128)
def get_quotes_cached(symbols_tuple, max_age_sec=300):
    return get_equity_quotes(list(symbols_tuple))
```

### Authentication Failures
- If `get_accounts` returns empty or 401, user's Robinhood session expired
- **Solution:** Log error, alert user via Discord, skip RH tools

```python
try:
    accounts = get_accounts()
except Exception as e:
    log.error(f"Robinhood auth failed: {e}")
    alert_discord("RH tools unavailable — using cached signals only")
    # Fall back to yfinance + local data
```

### Missing Data (Chain/Contract Not Found)
- If `get_option_instruments` returns empty, strike/expiry doesn't exist
- **Solution:** Adjust strike (move ITM/OTM) or use next expiration

```python
contracts = get_option_instruments(
    chain_symbol="XLK", expiration_dates="2026-08-15",
    strike_price="185.0", type="call"
)
if not contracts:
    # Recommended strike doesn't exist, try one step OTM
    strike_otm = 185.0 + 5.0  # Next 5-dollar strike
    contracts = get_option_instruments(
        chain_symbol="XLK", expiration_dates="2026-08-15",
        strike_price=str(strike_otm), type="call"
    )
```

---

## 8. Cron Integration

**Aggregator runs daily at 9:30 AM ET** (via cron).

### How to wire RH tools into daily run:
```bash
# In /etc/cron.d/ or PM2 startup hook:

# 1. At 9:25 AM, ensure Robinhood session is active
# 2. Run aggregator (it will use RH tools if available, fall back to yfinance)
# 3. At 9:35 AM, post signal summary to Discord

30 9 * * 1-5  python3 /home/jupiter/Lvl3Quant/paper_engines/agentic_signal_aggregator.py
35 9 * * 1-5  curl -X POST $WEBHOOK_URL -d "$(cat /home/jupiter/Lvl3Quant/state/agentic_signals.json)"
```

### Monitoring
- Check `/var/log/cron` for job status
- Verify output file updated: `ls -lt /home/jupiter/Lvl3Quant/state/agentic_signals.json`
- If RH tools unavailable (e.g., yfinance fallback used), log will show warnings

---

## 9. Example: Full Integration Loop

```python
#!/usr/bin/env python3
"""
Example: Enhanced agentic signal aggregator with RH MCP tools.
This shows the full integration: fetch, validate, enhance, recommend.
"""

from robinhood_mcp import (
    get_equity_quotes, get_equity_technical_indicators,
    get_option_chains, get_option_instruments, get_option_quotes,
    review_option_order, run_scan
)
import json

# 1. Run High Options Flow scan (predefined, user's MCP ID)
HIGH_OI_SCAN_ID = "<SCAN_ID>"
scan_results = run_scan(HIGH_OI_SCAN_ID)
high_vol_tickers = [row["ticker"] for row in scan_results["results"]]

# 2. Fetch latest signals from aggregator
with open("state/agentic_signals.json") as f:
    signals = json.load(f)

# 3. For each signal, enhance with RH data
for sig in signals["actionable_signals"]:
    ticker = sig["ticker"]
    
    # A. Fetch current price + bid/ask
    quotes = get_equity_quotes(symbols=[ticker])
    quote = quotes[0]
    sig["current_price"] = quote["last"]
    sig["bid_ask_spread_pct"] = (quote["ask"] - quote["bid"]) / quote["last"] * 100
    
    # B. Fetch technical indicators
    rsi = get_equity_technical_indicators(
        symbol=ticker, type="rsi", period=14,
        interval="day", start_time="2026-07-01T00:00:00Z",
        output="latest"
    )
    sig["rsi_14"] = rsi["rsi"][-1]
    
    # C. If ticker in high-options-flow scan, boost confidence
    if ticker in high_vol_tickers:
        sig["confidence_score"] *= 1.1  # 10% boost
        sig["confirming_sources"].append("high_options_volume_scan")
    
    # D. Fetch option liquidity
    chains = get_option_chains(underlying_symbol=ticker)
    # Filter to recommended expiry
    recommended_expiry = sig["recommended_expiry"]
    matching_chain = [c for c in chains if recommended_expiry in c["expirations"]]
    
    if matching_chain:
        # Get option contracts for recommended strike
        contracts = get_option_instruments(
            chain_symbol=ticker,
            expiration_dates=recommended_expiry,
            strike_price=str(sig["recommended_strike"]),
            type=sig["recommended_option"]
        )
        
        if contracts:
            opt_id = contracts[0]["id"]
            quotes_opt = get_option_quotes([opt_id])
            opt_quote = quotes_opt[0]
            
            sig["option_bid"] = opt_quote["bid"]
            sig["option_ask"] = opt_quote["ask"]
            sig["option_iv"] = opt_quote["implied_vol"]
            sig["option_oi"] = opt_quote["open_interest"]
            
            # Liquidity check: reject if OI < 50 or spread > $0.50
            if opt_quote["open_interest"] < 50:
                sig["tradable"] = False
                sig["reject_reason"] = "Low open interest"
            elif (opt_quote["ask"] - opt_quote["bid"]) > 0.50:
                sig["tradable"] = False
                sig["reject_reason"] = "Wide bid/ask spread"
            else:
                sig["tradable"] = True
        else:
            sig["tradable"] = False
            sig["reject_reason"] = "No option contracts found"
    else:
        sig["tradable"] = False
        sig["reject_reason"] = "Expiry not available"
    
    # E. Simulate order
    if sig.get("tradable"):
        review = review_option_order(
            account_number="12345",
            legs=[{"option_id": opt_id, "side": "buy", "position_effect": "open"}],
            quantity=1,
            type="limit",
            price=str((opt_quote["bid"] + opt_quote["ask"]) / 2),
            chain_symbol=ticker,
            underlying_type="equity"
        )
        sig["review_status"] = review  # Captures alerts, fees, etc.

# 4. Output enhanced signals
with open("state/agentic_signals_enhanced.json", "w") as f:
    json.dump(signals, f, indent=2)

print(f"Enhanced {len(signals['actionable_signals'])} signals with RH data")
```

---

## 10. Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| `get_equity_quotes` returns empty | Ticker not found | Verify ticker symbol (case-sensitive: "XLK" not "xlk") |
| Technical indicators timeout | start_time too far back + interval too fine | Reduce time range or coarsen interval (use "day" not "minute") |
| `get_option_instruments` returns 0 | Strike/expiry doesn't exist | Adjust strike to next available, move to front-month expiry |
| `review_option_order` shows low buying power | Account too small | Reduce quantity or choose cheaper strike (OTM) |
| Rate limit errors | Too many requests in 60 sec | Cache results for 5 min, batch queries, add 1-sec delays |

---

## See Also

- **Robinhood MCP Tool Docs:** `/home/jupiter/Lvl3Quant/docs/ref_trading.md`
- **Signal Aggregator Source:** `/home/jupiter/Lvl3Quant/paper_engines/agentic_signal_aggregator.py`
- **Paper Engine States:** `/home/jupiter/Lvl3Quant/state/`
- **Live Execution System:** `/home/jupiter/Lvl3Quant/trading_agents/` (uses agentic_signals.json)
