#!/usr/bin/env python3
"""
Put-Call Ratio Contrarian Paper Engine (AVO-evolved v23)
=========================================================
Sector rotation with VIX fear overlay. Buy top-momentum sector ETF every 5 days.
Layer 2 adds a second position when VIX z-score spikes above threshold and VIX >= 18.

Strategy source: AVO run put_call_contrarian-20260827-071946, step 23.

Cron: 15 16 * * 1-5  (4:15 PM ET weekdays, after market close)
State: paper_engines/state/put_call_contrarian_state.json
"""
import json
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytz

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
except ImportError:
    print("ERROR: yfinance required. pip install yfinance")
    sys.exit(1)

ET = pytz.timezone("US/Eastern")
BASE = Path("/home/jupiter/Lvl3Quant")
STATE_DIR = BASE / "paper_engines" / "state"
STATE_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = STATE_DIR / "put_call_contrarian_state.json"
LOG_DIR = BASE / "paper_engines" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "put_call_contrarian.log"

# ── Strategy Parameters (from AVO v23 strategy.py -- verbatim) ──

OFFENSIVE_ETFS = ['XLK', 'XLY', 'XLI', 'XLF', 'XLE', 'XLB']
DEFENSIVE_ETFS = ['XLU', 'XLP', 'GLD']
ALL_ETFS = OFFENSIVE_ETFS + DEFENSIVE_ETFS
BENCHMARK = 'SPY'

# Fear gauge
FEAR_LOOKBACK = 15
VIX_WEIGHT = 0.50
VOLUME_WEIGHT = 0.50
FEAR_THRESHOLD = 0.6

# Signal
MOMENTUM_LOOKBACK = 20
MOMENTUM_MIN_PCT = 0.01
MOMENTUM_MIN_PCT_LOWVOL = 0.03
REBALANCE_DAYS = 5
LOWVIX_SKIP_LAYER2 = 18.0

# Position sizing
INITIAL_CAPITAL = 10_000.0
MAX_PER_TRADE = 500.0
MAX_CONCURRENT = 2
SLIPPAGE_PCT = 0.0001

# Trade management
MAX_HOLD_DAYS = 5
TRAILING_STOP_PCT = -0.006
TAKE_PROFIT_PCT = 0.03
EARLY_EXIT_LOSS = -0.005

# Data lookback for indicators
DATA_LOOKBACK_DAYS = 120  # calendar days to fetch


# ── Logging ──

def log(msg):
    ts = datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S ET")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


# ── State Management ──

def load_state():
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {
        "capital": INITIAL_CAPITAL,
        "equity": INITIAL_CAPITAL,
        "positions": [],
        "trades": [],
        "daily_pnl": [],
        "last_rebalance_date": None,
        "days_since_rebalance": 999,  # force first rebalance
        "created": datetime.now(ET).isoformat(),
    }


def save_state(state):
    state["updated"] = datetime.now(ET).isoformat()
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


# ── Data Fetching ──

def fetch_data():
    """Fetch price, volume, and VIX data via yfinance."""
    end = datetime.now(ET)
    start = end - timedelta(days=DATA_LOOKBACK_DAYS)

    tickers = ALL_ETFS + [BENCHMARK, '^VIX']
    log(f"Fetching data for {len(tickers)} tickers from {start.date()} to {end.date()}")

    data = yf.download(tickers, start=start.strftime("%Y-%m-%d"),
                       end=end.strftime("%Y-%m-%d"), progress=False, auto_adjust=True)

    if data.empty:
        log("ERROR: No data returned from yfinance")
        return None, None, None, None, None

    close = data['Close'] if 'Close' in data.columns else data
    volume = data['Volume'] if 'Volume' in data.columns else None

    # Extract individual series
    vix = close['^VIX'] if '^VIX' in close.columns else None
    spy = close[BENCHMARK] if BENCHMARK in close.columns else None

    # ETF prices only
    etf_cols = [c for c in ALL_ETFS if c in close.columns]
    prices = close[etf_cols]

    etf_vol_cols = [c for c in ALL_ETFS if volume is not None and c in volume.columns]
    etf_volume = volume[etf_vol_cols] if etf_vol_cols else pd.DataFrame(index=prices.index)

    return prices, etf_volume, spy, vix, close


# ── Strategy Logic (from AVO v23) ──

def compute_fear_gauge(vix_s, volume, spy_returns):
    """Composite fear gauge from VIX z-score + panic volume."""
    components = []
    weights = []

    if vix_s is not None and len(vix_s.dropna()) > FEAR_LOOKBACK:
        vix_clean = vix_s.ffill().fillna(20.0)
        vix_mean = vix_clean.rolling(FEAR_LOOKBACK, min_periods=10).mean()
        vix_std = vix_clean.rolling(FEAR_LOOKBACK, min_periods=10).std()
        vix_z = (vix_clean - vix_mean) / vix_std.replace(0, 1)
        components.append(vix_z)
        weights.append(VIX_WEIGHT)

    if volume is not None and spy_returns is not None:
        off_cols = [c for c in OFFENSIVE_ETFS if c in volume.columns]
        if off_cols:
            total_vol = volume[off_cols].sum(axis=1)
        else:
            total_vol = pd.Series(0, index=vix_s.index if vix_s is not None else pd.DatetimeIndex([]))
        vol_clean = total_vol.ffill().fillna(0)
        vol_mean = vol_clean.rolling(FEAR_LOOKBACK, min_periods=10).mean()
        vol_std = vol_clean.rolling(FEAR_LOOKBACK, min_periods=10).std()
        vol_z = (vol_clean - vol_mean) / vol_std.replace(0, 1)
        down_day = (spy_returns < 0).astype(float)
        panic_vol = vol_z * down_day
        components.append(panic_vol)
        weights.append(VOLUME_WEIGHT)

    if not components:
        idx = vix_s.index if vix_s is not None else pd.DatetimeIndex([])
        return pd.Series(0.0, index=idx)

    total_w = sum(weights)
    weights = [w / total_w for w in weights]

    idx = components[0].index
    for c in components[1:]:
        idx = idx.intersection(c.index)

    composite = pd.Series(0.0, index=idx)
    for comp, w in zip(components, weights):
        aligned = comp.reindex(idx).fillna(0)
        composite += w * aligned

    return composite


def get_top_momentum(prices, date, lookback, universe, n=1, min_pct=0.0):
    """Top N sectors by momentum over lookback period."""
    idx_loc = prices.index.get_loc(date)
    if idx_loc < lookback:
        return []

    candidates = []
    for etf in universe:
        if etf not in prices.columns:
            continue
        p_now = prices.iloc[idx_loc][etf]
        p_then = prices.iloc[idx_loc - lookback][etf]
        if pd.isna(p_now) or pd.isna(p_then) or p_then <= 0:
            continue
        rs = (p_now - p_then) / p_then
        if rs > min_pct:
            candidates.append((etf, rs))

    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[:n]


def check_signals_today(prices, volume, spy, vix, today_idx):
    """Check if we should open new positions today."""
    signals = []

    spy_returns = spy.pct_change()
    vix_aligned = vix.reindex(prices.index).ffill().fillna(20.0)
    fear = compute_fear_gauge(vix_aligned, volume, spy_returns)
    fear = fear.reindex(prices.index).fillna(0)

    curr_vix = vix_aligned.iloc[today_idx]
    if pd.isna(curr_vix):
        curr_vix = 20.0
    fg = fear.iloc[today_idx] if today_idx < len(fear) else 0.0
    today_date = prices.index[today_idx]

    # Layer 1: periodic momentum rebalance
    min_mom = MOMENTUM_MIN_PCT_LOWVOL if curr_vix < LOWVIX_SKIP_LAYER2 else MOMENTUM_MIN_PCT
    top = get_top_momentum(prices, today_date, MOMENTUM_LOOKBACK,
                           ALL_ETFS, n=1, min_pct=min_mom)
    if top:
        signals.append({
            "ticker": top[0][0],
            "momentum": round(top[0][1], 4),
            "layer": 1,
            "vix": round(float(curr_vix), 2),
            "fear_gauge": round(float(fg), 3),
        })

    # Layer 2: fear overlay (only when VIX >= threshold)
    if curr_vix >= LOWVIX_SKIP_LAYER2 and not pd.isna(fg) and fg > FEAR_THRESHOLD:
        top2 = get_top_momentum(prices, today_date, MOMENTUM_LOOKBACK,
                                ALL_ETFS, n=2, min_pct=-0.05)
        if len(top2) >= 2:
            signals.append({
                "ticker": top2[1][0],
                "momentum": round(top2[1][1], 4),
                "layer": 2,
                "vix": round(float(curr_vix), 2),
                "fear_gauge": round(float(fg), 3),
            })
        elif len(top2) == 1 and (not signals or top2[0][0] != signals[0]["ticker"]):
            signals.append({
                "ticker": top2[0][0],
                "momentum": round(top2[0][1], 4),
                "layer": 2,
                "vix": round(float(curr_vix), 2),
                "fear_gauge": round(float(fg), 3),
            })

    return signals


def should_exit(pos, current_price):
    """Check exit conditions for a position."""
    entry_price = pos["entry_price"]
    entry_date = datetime.fromisoformat(pos["entry_date"]).date() if isinstance(pos["entry_date"], str) else pos["entry_date"]
    today = datetime.now(ET).date()
    days_held = np.busday_count(
        np.datetime64(entry_date, 'D'),
        np.datetime64(today, 'D'))

    pnl_pct = (current_price - entry_price) / entry_price

    hwm = pos.get("high_water_mark", entry_price)
    if current_price > hwm:
        pos["high_water_mark"] = current_price
        hwm = current_price

    drawdown_from_high = (current_price - hwm) / hwm if hwm > 0 else 0

    reason = None

    if days_held >= MAX_HOLD_DAYS:
        reason = f"max_hold ({days_held}d)"
    elif pnl_pct >= TAKE_PROFIT_PCT:
        reason = f"take_profit ({pnl_pct:+.2%})"
    elif drawdown_from_high <= TRAILING_STOP_PCT:
        reason = f"trailing_stop (dd={drawdown_from_high:.2%})"
    elif days_held >= 1 and pnl_pct < EARLY_EXIT_LOSS:
        reason = f"early_exit_loss ({pnl_pct:+.2%} after {days_held}d)"

    return reason


# ── Main Engine ──

def run():
    """Run one daily cycle of the paper engine."""
    log("=" * 60)
    log("Put-Call Ratio Contrarian Paper Engine -- daily run")
    log("=" * 60)

    state = load_state()
    now = datetime.now(ET)
    today_str = now.strftime("%Y-%m-%d")

    # Check if market was open today (skip weekends)
    if now.weekday() >= 5:
        log(f"Weekend ({now.strftime('%A')}), skipping.")
        return

    # Check if we already ran today
    if state.get("daily_pnl") and state["daily_pnl"][-1].get("date") == today_str:
        log(f"Already ran today ({today_str}), skipping.")
        return

    # Fetch market data
    result = fetch_data()
    if result[0] is None:
        log("ERROR: Failed to fetch data, aborting.")
        return
    prices, volume, spy, vix, all_close = result

    if len(prices) < MOMENTUM_LOOKBACK + 20:
        log(f"ERROR: Not enough data ({len(prices)} rows, need {MOMENTUM_LOOKBACK + 20})")
        return

    today_idx = len(prices) - 1
    today_date = prices.index[today_idx]
    log(f"Latest data date: {today_date.strftime('%Y-%m-%d')}")

    # ── Step 1: Check exits on existing positions ──
    exits_today = []
    remaining_positions = []
    realized_pnl = 0.0

    for pos in state["positions"]:
        ticker = pos["ticker"]
        if ticker not in prices.columns:
            log(f"WARNING: {ticker} not in price data, keeping position")
            remaining_positions.append(pos)
            continue

        current_price = float(prices[ticker].iloc[today_idx])
        if pd.isna(current_price):
            log(f"WARNING: NaN price for {ticker}, keeping position")
            remaining_positions.append(pos)
            continue

        exit_reason = should_exit(pos, current_price)
        if exit_reason:
            # Execute exit
            exit_price = current_price * (1 - SLIPPAGE_PCT)  # slippage on sell
            shares = pos["shares"]
            trade_pnl = (exit_price - pos["entry_price"]) * shares
            realized_pnl += trade_pnl

            trade_record = {
                "ticker": ticker,
                "side": "sell",
                "entry_date": pos["entry_date"],
                "entry_price": pos["entry_price"],
                "exit_date": today_str,
                "exit_price": round(exit_price, 2),
                "shares": shares,
                "pnl": round(trade_pnl, 2),
                "pnl_pct": round((exit_price - pos["entry_price"]) / pos["entry_price"], 4),
                "reason": exit_reason,
                "layer": pos.get("layer", 1),
            }
            state["trades"].append(trade_record)
            exits_today.append(trade_record)
            state["capital"] += exit_price * shares

            log(f"EXIT {ticker}: {exit_reason} | PnL: ${trade_pnl:+.2f} "
                f"({trade_record['pnl_pct']:+.2%}) | {shares} shares @ ${exit_price:.2f}")
        else:
            # Update HWM
            if current_price > pos.get("high_water_mark", pos["entry_price"]):
                pos["high_water_mark"] = round(current_price, 2)
            remaining_positions.append(pos)

    state["positions"] = remaining_positions

    # ── Step 2: Check for new signals (rebalance check) ──
    entries_today = []
    days_since = state.get("days_since_rebalance", 999)

    if days_since >= REBALANCE_DAYS and len(state["positions"]) < MAX_CONCURRENT:
        signals = check_signals_today(prices, volume, spy, vix, today_idx)

        # Filter out tickers we already hold
        held_tickers = {p["ticker"] for p in state["positions"]}
        signals = [s for s in signals if s["ticker"] not in held_tickers]

        slots_available = MAX_CONCURRENT - len(state["positions"])
        signals = signals[:slots_available]

        for sig in signals:
            ticker = sig["ticker"]
            current_price = float(prices[ticker].iloc[today_idx])
            entry_price = current_price * (1 + SLIPPAGE_PCT)  # slippage on buy

            # Position sizing: $500 per trade (or remaining capital)
            trade_capital = min(MAX_PER_TRADE, state["capital"] * 0.95)
            if trade_capital < 50:
                log(f"Insufficient capital (${state['capital']:.2f}), skipping {ticker}")
                continue

            shares = int(trade_capital / entry_price)
            if shares < 1:
                log(f"Price too high for {ticker} (${entry_price:.2f}), skipping")
                continue

            cost = shares * entry_price
            state["capital"] -= cost

            position = {
                "ticker": ticker,
                "entry_date": today_str,
                "entry_price": round(entry_price, 2),
                "shares": shares,
                "cost": round(cost, 2),
                "high_water_mark": round(entry_price, 2),
                "layer": sig["layer"],
                "momentum": sig["momentum"],
                "vix_at_entry": sig["vix"],
                "fear_at_entry": sig["fear_gauge"],
            }
            state["positions"].append(position)
            entries_today.append(position)

            log(f"ENTRY {ticker} (L{sig['layer']}): {shares} shares @ ${entry_price:.2f} "
                f"(${cost:.2f}) | mom={sig['momentum']:+.2%} VIX={sig['vix']}")

        if signals:
            state["days_since_rebalance"] = 0
            state["last_rebalance_date"] = today_str
        else:
            state["days_since_rebalance"] = days_since + 1
    else:
        state["days_since_rebalance"] = days_since + 1

    # ── Step 3: Mark-to-market ──
    unrealized_pnl = 0.0
    for pos in state["positions"]:
        ticker = pos["ticker"]
        if ticker in prices.columns:
            current_price = float(prices[ticker].iloc[today_idx])
            if not pd.isna(current_price):
                unrealized_pnl += (current_price - pos["entry_price"]) * pos["shares"]

    # Equity = cash + position value
    position_value = sum(
        float(prices[p["ticker"]].iloc[today_idx]) * p["shares"]
        for p in state["positions"]
        if p["ticker"] in prices.columns and not pd.isna(prices[p["ticker"]].iloc[today_idx])
    )
    state["equity"] = round(state["capital"] + position_value, 2)

    # ── Step 4: Daily P&L record ──
    prev_equity = state["daily_pnl"][-1]["equity"] if state["daily_pnl"] else INITIAL_CAPITAL
    daily_change = state["equity"] - prev_equity

    daily_record = {
        "date": today_str,
        "equity": state["equity"],
        "capital": round(state["capital"], 2),
        "daily_pnl": round(daily_change, 2),
        "realized_pnl": round(realized_pnl, 2),
        "unrealized_pnl": round(unrealized_pnl, 2),
        "positions_held": len(state["positions"]),
        "entries": len(entries_today),
        "exits": len(exits_today),
    }
    state["daily_pnl"].append(daily_record)

    # Keep last 252 daily records
    if len(state["daily_pnl"]) > 252:
        state["daily_pnl"] = state["daily_pnl"][-252:]

    # ── Step 5: Summary stats ──
    total_trades = len(state["trades"])
    if total_trades > 0:
        wins = sum(1 for t in state["trades"] if t["pnl"] > 0)
        total_pnl = sum(t["pnl"] for t in state["trades"])
        avg_pnl = total_pnl / total_trades
        win_rate = wins / total_trades
    else:
        total_pnl = 0
        avg_pnl = 0
        win_rate = 0

    cum_return = (state["equity"] - INITIAL_CAPITAL) / INITIAL_CAPITAL

    log(f"--- Daily Summary ---")
    log(f"Equity: ${state['equity']:,.2f} ({cum_return:+.2%} total return)")
    log(f"Cash: ${state['capital']:,.2f} | Positions: {len(state['positions'])}")
    log(f"Today: {len(entries_today)} entries, {len(exits_today)} exits, PnL: ${daily_change:+.2f}")
    log(f"All-time: {total_trades} trades, WR: {win_rate:.0%}, Avg PnL: ${avg_pnl:+.2f}")

    if state["positions"]:
        log(f"Open positions:")
        for p in state["positions"]:
            ticker = p["ticker"]
            if ticker in prices.columns:
                curr = float(prices[ticker].iloc[today_idx])
                pos_pnl = (curr - p["entry_price"]) * p["shares"]
                pos_pct = (curr - p["entry_price"]) / p["entry_price"]
                log(f"  {ticker} (L{p.get('layer',1)}): {p['shares']} sh @ ${p['entry_price']:.2f} "
                    f"-> ${curr:.2f} ({pos_pct:+.2%}, ${pos_pnl:+.2f})")

    save_state(state)
    log(f"State saved to {STATE_FILE}")
    log("=" * 60)


if __name__ == "__main__":
    run()
