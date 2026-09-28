#!/usr/bin/env python3
"""
Breadth Momentum Regime Paper Engine
======================================
Paper trades sector ETFs based on market breadth signals.
- Buys defensives on breadth deterioration
- Buys offensives on breadth thrust (low-vol only)
- VIX>25 = defensives only
- Hold period: ~3-5 days, $2000 per trade, max 3 concurrent

Strategy source: strategies/breadth_momentum_regime_avo_v20_final.py

Cron: 55 16 * * 1-5  (4:55 PM ET, after market close)
State: paper_engines/state/breadth_momentum_regime_state.json
"""
import json
import subprocess
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytz
import yfinance as yf

warnings.filterwarnings("ignore")

# Add strategies to path
sys.path.insert(0, str(Path("/home/jupiter/Lvl3Quant")))

from strategies.breadth_momentum_regime_avo_v20_final import (
    SECTOR_ETFS, OFFENSIVE_SECTORS, DEFENSIVE_SECTORS, HIGH_VOL_SECTORS,
    MAX_HOLD_DAYS, MAX_PER_TRADE, MAX_CONCURRENT, SLIPPAGE_PCT,
    VIX_HIGH_THRESHOLD, TRAILING_STOP_LOW_VOL, TRAILING_STOP_HIGH_VOL,
    TAKE_PROFIT_PCT, UNDERWATER_EXIT_DAYS,
    generate_signals, should_exit, compute_breadth,
)

ET = pytz.timezone("US/Eastern")
BASE = Path("/home/jupiter/Lvl3Quant")
STATE_DIR = BASE / "paper_engines" / "state"
STATE_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = STATE_DIR / "breadth_momentum_regime_state.json"
LOG_DIR = BASE / "paper_engines" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "breadth_momentum_regime.log"
CALLBACK_SCRIPT = BASE / "scripts" / "run_engine_with_callback.sh"

# ── Paper Engine Parameters ──
INITIAL_CAPITAL = 10_000.0
DATA_LOOKBACK_DAYS = 120  # need 50d SMA + buffer

# All tickers we need to fetch
ALL_TICKERS = list(set(SECTOR_ETFS + ['SPY', 'RSP', '^VIX']))


# ── Engine Infrastructure ──

def log(msg: str):
    ts = datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S")
    line = f"{ts} [BREADTH-MOMENTUM-REGIME] {msg}"
    print(line)
    try:
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except (json.JSONDecodeError, KeyError):
            pass
    return {
        "positions": [],
        "closed_trades": [],
        "equity_curve": [],
        "capital": INITIAL_CAPITAL,
        "equity": INITIAL_CAPITAL,
        "peak_equity": INITIAL_CAPITAL,
        "max_drawdown": 0.0,
        "total_pnl": 0.0,
        "wins": 0,
        "losses": 0,
        "n_trades": 0,
        "gross_profit": 0.0,
        "gross_loss": 0.0,
        "created": datetime.now(ET).isoformat(),
        "last_run_date": None,
        "last_updated": None,
    }


def save_state(state: dict):
    state["last_updated"] = datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S")
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, default=str))
    tmp.rename(STATE_FILE)


def fire_callback(trade_info: dict):
    """Fire signal callback for trade taken."""
    if not CALLBACK_SCRIPT.exists():
        log("WARN: Callback script not found, skipping")
        return
    try:
        import os
        env_vars = {
            "ENGINE_NAME": "breadth_momentum_regime",
            "TRADE_TICKER": trade_info.get("ticker", ""),
            "TRADE_DIRECTION": trade_info.get("direction", "long"),
            "TRADE_ACTION": trade_info.get("action", ""),
            "TRADE_PRICE": str(trade_info.get("price", 0)),
            "TRADE_PNL": str(trade_info.get("pnl", 0)),
        }
        full_env = {**os.environ, **env_vars}
        subprocess.Popen(
            [str(CALLBACK_SCRIPT)],
            env=full_env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as e:
        log(f"WARN: Callback failed: {e}")


def fetch_data():
    """Fetch sector ETFs + SPY + RSP + VIX daily data via yfinance."""
    end = datetime.now(ET)
    start = end - timedelta(days=DATA_LOOKBACK_DAYS)

    tickers = list(dict.fromkeys(ALL_TICKERS))

    try:
        raw = yf.download(
            tickers,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
            threads=True,
            timeout=30,
        )
    except Exception as e:
        log(f"ERROR: yfinance download failed: {e}")
        return None, None, None, None

    if raw.empty:
        log("ERROR: yfinance returned empty data")
        return None, None, None, None

    mi = isinstance(raw.columns, pd.MultiIndex)
    close = raw["Close"] if mi else raw
    if isinstance(close.columns, pd.MultiIndex):
        close.columns = close.columns.get_level_values(-1)

    # Extract VIX as a series
    vix_series = None
    if "^VIX" in close.columns:
        vix_series = close["^VIX"].dropna()

    # SPY series
    spy_series = None
    if "SPY" in close.columns:
        spy_series = close["SPY"].dropna()

    # RSP series (equal-weight S&P 500)
    rsp_series = None
    if "RSP" in close.columns:
        rsp_series = close["RSP"].dropna()

    # Sector price DataFrame (only sector ETFs)
    sector_cols = [c for c in SECTOR_ETFS if c in close.columns]
    sector_prices = close[sector_cols].dropna(how="all") if sector_cols else pd.DataFrame()

    return sector_prices, spy_series, vix_series, rsp_series


def run():
    now = datetime.now(ET)
    today = now.date()
    today_str = today.isoformat()

    # Skip weekends
    if today.weekday() >= 5:
        log("Weekend -- skipping.")
        return

    log(f"=== Breadth Momentum Regime Paper Engine -- {today_str} ===")
    state = load_state()

    # Skip if already ran today
    if state.get("last_run_date") == today_str:
        log("Already ran today -- skipping.")
        print_summary(state)
        return

    # Fetch data
    sector_prices, spy_series, vix_series, rsp_series = fetch_data()
    if sector_prices is None or spy_series is None:
        log("ERROR: No price data. Aborting.")
        save_state(state)
        return

    if len(sector_prices) < 55:
        log(f"ERROR: Insufficient data ({len(sector_prices)} rows, need 55+). Aborting.")
        save_state(state)
        return

    # Check for new data
    last_data_date = sector_prices.index[-1]
    if hasattr(last_data_date, 'date'):
        last_data_date = last_data_date.date()
    data_age = (today - last_data_date).days
    if data_age > 3:
        log(f"WARNING: Latest data is {data_age} days old ({last_data_date}). Possible holiday/no new data.")

    vix_now = float(vix_series.iloc[-1]) if vix_series is not None and len(vix_series) > 0 else None
    vix_str = f"{vix_now:.2f}" if vix_now else "N/A"
    high_vol = vix_now is not None and vix_now > VIX_HIGH_THRESHOLD
    regime = "HIGH-VOL" if high_vol else "LOW-VOL"
    log(f"VIX: {vix_str} | Regime: {regime}")

    # Compute breadth
    breadth = compute_breadth(sector_prices)
    if len(breadth) > 0:
        last_breadth = breadth.iloc[-1]
        pct_20d = last_breadth.get('pct_above_20d', 'N/A')
        pct_50d = last_breadth.get('pct_above_50d', 'N/A')
        if not pd.isna(pct_20d) and not pd.isna(pct_50d):
            log(f"Breadth: {pct_20d:.0%} above 20d SMA, {pct_50d:.0%} above 50d SMA")

    # Get current prices for each sector
    current_prices = {}
    for etf in SECTOR_ETFS:
        if etf in sector_prices.columns:
            val = sector_prices[etf].dropna()
            if len(val) > 0:
                current_prices[etf] = float(val.iloc[-1])

    # --- 1. Check exits on open positions ---
    portfolio_dd = 0.0
    if state["peak_equity"] > 0:
        portfolio_dd = (state["equity"] - state["peak_equity"]) / state["peak_equity"]

    still_open = []
    exits_today = []
    for pos in state["positions"]:
        ticker = pos["ticker"]
        if ticker not in current_prices:
            log(f"  WARN: No price for {ticker}, keeping position open")
            still_open.append(pos)
            continue

        price = current_prices[ticker]

        # Update high water mark
        if price > pos.get("hwm", pos["entry_price"]):
            pos["hwm"] = price

        exit_flag = should_exit(pos, price, today_str, portfolio_dd)

        if exit_flag:
            exit_price = price * (1 - SLIPPAGE_PCT)
            shares = pos["shares"]
            pnl = (exit_price - pos["entry_price"]) * shares
            pnl_pct = (exit_price / pos["entry_price"] - 1) * 100
            days_held = np.busday_count(
                np.datetime64(pos["entry_date"], 'D'),
                np.datetime64(today_str, 'D'))

            state["capital"] += shares * exit_price
            state["total_pnl"] += pnl
            state["n_trades"] += 1
            if pnl >= 0:
                state["wins"] += 1
                state["gross_profit"] += pnl
            else:
                state["losses"] += 1
                state["gross_loss"] += abs(pnl)

            trade_record = {
                "ticker": ticker,
                "entry_date": pos["entry_date"],
                "exit_date": today_str,
                "entry_price": pos["entry_price"],
                "exit_price": round(exit_price, 4),
                "shares": shares,
                "pnl": round(pnl, 2),
                "pnl_pct": round(pnl_pct, 2),
                "days_held": int(days_held),
                "direction": pos.get("direction", "unknown"),
                "regime_at_entry": pos.get("regime_at_entry", "unknown"),
            }
            state["closed_trades"].append(trade_record)
            state["closed_trades"] = state["closed_trades"][-200:]
            exits_today.append(trade_record)

            log(f"  EXIT {ticker}: ${pnl:+.2f} ({pnl_pct:+.2f}%) after {days_held}d")

            fire_callback({
                "ticker": ticker, "action": "exit", "direction": "sell",
                "price": exit_price, "pnl": round(pnl, 2),
            })
        else:
            still_open.append(pos)

    state["positions"] = still_open

    # --- 2. Generate signals and check for new entries ---
    entries_today = []
    n_open = len(state["positions"])

    if n_open < MAX_CONCURRENT:
        # Align data for signal generation
        spy_aligned = spy_series.reindex(sector_prices.index).ffill()
        vix_aligned = vix_series.reindex(sector_prices.index).ffill().fillna(20.0) if vix_series is not None else pd.Series(20.0, index=sector_prices.index)

        signals = generate_signals(
            sector_prices, spy_aligned, vix_aligned,
            rsp=rsp_series.reindex(sector_prices.index).ffill() if rsp_series is not None else None,
            breadth=breadth,
        )

        # Get today's signals (last row)
        last_idx = signals.index[-1]
        today_signals = signals.loc[last_idx]
        firing = [etf for etf in SECTOR_ETFS if today_signals.get(etf, False)]

        if firing:
            log(f"  Signals firing: {', '.join(firing)}")

        held_tickers = {p["ticker"] for p in state["positions"]}

        for etf in firing:
            if n_open >= MAX_CONCURRENT:
                log(f"  SKIP {etf}: max concurrent ({MAX_CONCURRENT}) reached")
                break
            if etf in held_tickers:
                log(f"  SKIP {etf}: already holding")
                continue
            if etf not in current_prices:
                continue

            price = current_prices[etf]
            entry_price = price * (1 + SLIPPAGE_PCT)
            position_size = min(MAX_PER_TRADE, state["capital"] * 0.95)

            if position_size < 50:
                log(f"  SKIP {etf}: insufficient capital (${state['capital']:.0f})")
                continue

            shares = position_size / entry_price
            cost = shares * entry_price
            state["capital"] -= cost

            # Determine direction
            if etf in OFFENSIVE_SECTORS:
                direction = "offensive"
            elif etf in DEFENSIVE_SECTORS or etf in HIGH_VOL_SECTORS:
                direction = "defensive"
            else:
                direction = "unknown"

            pos = {
                "ticker": etf,
                "shares": round(shares, 6),
                "entry_price": round(entry_price, 4),
                "entry_date": today_str,
                "hwm": round(entry_price, 4),
                "direction": direction,
                "vix_at_entry": round(vix_now, 2) if vix_now else None,
                "regime_at_entry": regime,
            }
            state["positions"].append(pos)
            held_tickers.add(etf)
            n_open += 1
            entries_today.append(pos)

            log(f"  ENTRY {etf} ({direction}): {shares:.4f} shares @ ${entry_price:.2f} (regime={regime})")

            fire_callback({
                "ticker": etf, "action": "entry", "direction": "long",
                "price": entry_price, "pnl": 0,
            })
    else:
        log(f"  Max concurrent positions ({MAX_CONCURRENT}) -- skipping signal scan")

    # --- 3. Mark to market ---
    portfolio_value = state["capital"]
    for pos in state["positions"]:
        ticker = pos["ticker"]
        if ticker in current_prices:
            portfolio_value += pos["shares"] * current_prices[ticker]
        else:
            portfolio_value += pos["shares"] * pos["entry_price"]

    state["equity"] = round(portfolio_value, 2)

    if state["equity"] > state["peak_equity"]:
        state["peak_equity"] = state["equity"]
    current_dd = 0.0
    if state["peak_equity"] > 0:
        current_dd = (state["equity"] - state["peak_equity"]) / state["peak_equity"]
    if current_dd < state["max_drawdown"]:
        state["max_drawdown"] = round(current_dd, 6)

    state["equity_curve"].append({
        "date": today_str,
        "equity": state["equity"],
        "positions": len(state["positions"]),
    })
    state["equity_curve"] = state["equity_curve"][-500:]

    state["last_run_date"] = today_str
    save_state(state)

    # --- 4. Print summary ---
    print_summary(state, entries_today, exits_today)

    log("Done.")


def print_summary(state, entries_today=None, exits_today=None):
    """Print clean status summary to stdout."""
    entries_today = entries_today or []
    exits_today = exits_today or []

    total_trades = state["n_trades"]
    wr = state["wins"] / total_trades * 100 if total_trades > 0 else 0
    pf = state["gross_profit"] / state["gross_loss"] if state["gross_loss"] > 0 else float('inf')
    avg_win = state["gross_profit"] / state["wins"] if state["wins"] > 0 else 0
    avg_loss = state["gross_loss"] / state["losses"] if state["losses"] > 0 else 0
    return_pct = (state["equity"] / INITIAL_CAPITAL - 1) * 100
    dd_pct = state["max_drawdown"] * 100

    print("\n" + "=" * 60)
    print("  BREADTH MOMENTUM REGIME -- Paper Trading Status")
    print("=" * 60)
    print(f"  Equity:     ${state['equity']:,.2f}  ({return_pct:+.2f}%)")
    print(f"  Cash:       ${state['capital']:,.2f}")
    print(f"  Max DD:     {dd_pct:.2f}%")
    print(f"  Total PnL:  ${state['total_pnl']:+,.2f}")
    print("-" * 60)
    print(f"  Trades:     {total_trades}  |  W/L: {state['wins']}/{state['losses']}  |  WR: {wr:.1f}%")
    print(f"  PF: {pf:.2f}  |  Avg Win: ${avg_win:.2f}  |  Avg Loss: ${avg_loss:.2f}")
    print("-" * 60)

    if state["positions"]:
        print("  Open Positions:")
        for pos in state["positions"]:
            print(f"    {pos['ticker']:5s}  {pos['shares']:.2f} sh @ ${pos['entry_price']:.2f}  "
                  f"(entered {pos['entry_date']}, {pos.get('direction', '?')}/{pos.get('regime_at_entry', '?')})")
    else:
        print("  No open positions.")

    if entries_today:
        print(f"\n  Today's Entries: {', '.join(p['ticker'] for p in entries_today)}")
    if exits_today:
        exit_strs = [f"{t['ticker']} ${t['pnl']:+.2f}" for t in exits_today]
        print(f"  Today's Exits:  {', '.join(exit_strs)}")

    print("=" * 60 + "\n")

    log(f"Equity: ${state['equity']:.2f} | Positions: {len(state['positions'])} | "
        f"Trades: {total_trades} | WR: {wr:.0f}% | PF: {pf:.2f} | PnL: ${state['total_pnl']:+.2f} | DD: {dd_pct:.2f}%")


if __name__ == "__main__":
    run()
