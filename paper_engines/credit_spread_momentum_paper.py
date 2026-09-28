#!/usr/bin/env python3
"""
Credit Spread Momentum Paper Engine
======================================
Paper trades sector ETFs based on HYG/LQD credit spread z-score regime.
Bull credit (tightening) → offensive ETFs. Bear credit (widening) → defensive ETFs.

Strategy source: strategies/credit_spread_momentum_avo_v10_final.py
AVO score: 9.60 (ALL-TIME RECORD), lockbox Sharpe 7.92, 111 trades
8/8 folds positive, geomean Sharpe 5.26, regime gap 17.5%

Cron: 48 16 * * 1-5  (4:48 PM ET, after market close)
State: paper_engines/state/credit_spread_momentum_state.json
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

sys.path.insert(0, str(Path("/home/jupiter/Lvl3Quant")))

from strategies.credit_spread_momentum_avo_v16_final import (
    BULL_ASSETS, BEAR_ASSETS, NEUTRAL_ASSETS,
    MAX_HOLD_DAYS, MAX_PER_TRADE, MAX_CONCURRENT, SLIPPAGE_PCT,
    TAKE_PROFIT_PCT, STOP_LOSS_PCT, TRAILING_STOP_PCT,
    generate_signals, should_exit,
)

ET = pytz.timezone("US/Eastern")
BASE = Path("/home/jupiter/Lvl3Quant")
STATE_DIR = BASE / "paper_engines" / "state"
STATE_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = STATE_DIR / "credit_spread_momentum_state.json"
LOG_DIR = BASE / "paper_engines" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "credit_spread_momentum.log"
CALLBACK_SCRIPT = BASE / "scripts" / "run_engine_with_callback.sh"

INITIAL_CAPITAL = 10_000.0
DATA_LOOKBACK_DAYS = 120

ALL_TICKERS = list(set(BULL_ASSETS + BEAR_ASSETS + NEUTRAL_ASSETS +
                       ['SPY', 'HYG', 'LQD', '^VIX']))


def log(msg: str):
    ts = datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S")
    line = f"{ts} [CREDIT-SPREAD-MOM] {msg}"
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
    if not CALLBACK_SCRIPT.exists():
        return
    try:
        import os
        env_vars = {
            "ENGINE_NAME": "credit_spread_momentum",
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
        return None, None, None

    if raw.empty:
        log("ERROR: yfinance returned empty data")
        return None, None, None

    mi = isinstance(raw.columns, pd.MultiIndex)
    close = raw["Close"] if mi else raw
    if isinstance(close.columns, pd.MultiIndex):
        close.columns = close.columns.get_level_values(-1)

    vix_series = close["^VIX"].dropna() if "^VIX" in close.columns else None
    spy_series = close["SPY"].dropna() if "SPY" in close.columns else None

    # Include all tradeable + signal tickers
    all_cols = list(set(BULL_ASSETS + BEAR_ASSETS + NEUTRAL_ASSETS + ['HYG', 'LQD']))
    avail_cols = [c for c in all_cols if c in close.columns]
    prices = close[avail_cols].dropna(how="all") if avail_cols else pd.DataFrame()

    return prices, spy_series, vix_series


def print_summary(state):
    eq = state["equity"]
    pnl = state["total_pnl"]
    pnl_pct = (eq / INITIAL_CAPITAL - 1) * 100
    wr = state["wins"] / max(state["n_trades"], 1) * 100
    pf = state["gross_profit"] / max(abs(state["gross_loss"]), 0.01)
    n_open = len(state["positions"])
    log(f"  Equity: ${eq:,.2f} ({pnl_pct:+.1f}%) | PnL: ${pnl:+,.2f} | "
        f"Trades: {state['n_trades']} | WR: {wr:.0f}% | PF: {pf:.2f} | "
        f"MDD: {state['max_drawdown']:.1%} | Open: {n_open}")


def run():
    now = datetime.now(ET)
    today = now.date()
    today_str = today.isoformat()

    if today.weekday() >= 5:
        log("Weekend -- skipping.")
        return

    log(f"=== Credit Spread Momentum Paper Engine -- {today_str} ===")
    state = load_state()

    if state.get("last_run_date") == today_str:
        log("Already ran today -- skipping.")
        print_summary(state)
        return

    prices, spy_series, vix_series = fetch_data()
    if prices is None or spy_series is None:
        log("ERROR: No price data. Aborting.")
        save_state(state)
        return

    if len(prices) < 70:
        log(f"ERROR: Insufficient data ({len(prices)} rows, need 70+). Aborting.")
        save_state(state)
        return

    vix_now = float(vix_series.iloc[-1]) if vix_series is not None and len(vix_series) > 0 else None
    vix_str = f"{vix_now:.2f}" if vix_now else "N/A"
    log(f"VIX: {vix_str}")

    # Get current prices for all tradeable tickers
    tradeable = list(set(BULL_ASSETS + BEAR_ASSETS + NEUTRAL_ASSETS))
    current_prices = {}
    for etf in tradeable:
        if etf in prices.columns:
            val = prices[etf].dropna()
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
                state["gross_loss"] += pnl

            trade_record = {
                "ticker": ticker,
                "entry_date": pos["entry_date"],
                "entry_price": pos["entry_price"],
                "exit_date": today_str,
                "exit_price": round(exit_price, 4),
                "shares": shares,
                "pnl": round(pnl, 2),
                "pnl_pct": round(pnl_pct, 2),
                "days_held": int(days_held),
                "direction": pos.get("direction", "long"),
            }
            state["closed_trades"].append(trade_record)
            exits_today.append(trade_record)

            log(f"  EXIT {ticker}: ${exit_price:.2f} | PnL: ${pnl:+.2f} ({pnl_pct:+.1f}%) | "
                f"Held: {days_held}d | Direction: {pos.get('direction', 'long')}")

            fire_callback({
                "ticker": ticker,
                "direction": pos.get("direction", "long"),
                "action": "exit",
                "price": exit_price,
                "pnl": pnl,
            })
        else:
            still_open.append(pos)

    state["positions"] = still_open

    # --- 2. Generate signals and enter new positions ---
    n_open = len(state["positions"])
    if n_open < MAX_CONCURRENT:
        signals = generate_signals(prices, spy_series, vix_series)

        if signals is not None and not signals.empty:
            last_row = signals.iloc[-1]
            active_signals = last_row[last_row > 0].index.tolist()

            # Filter out tickers we already hold
            held_tickers = {p["ticker"] for p in state["positions"]}
            active_signals = [s for s in active_signals if s not in held_tickers]

            # HC #807 R2: no re-entry within 5 trading days of a TP hit
            recent_tp_tickers = set()
            for ct in state["closed_trades"][-20:]:
                if ct.get("pnl", 0) > 0:
                    exit_date = datetime.strptime(ct["exit_date"], "%Y-%m-%d").date()
                    bdays = np.busday_count(
                        np.datetime64(exit_date, 'D'),
                        np.datetime64(today_str, 'D'))
                    if bdays <= 5:
                        recent_tp_tickers.add(ct["ticker"])
            active_signals = [s for s in active_signals if s not in recent_tp_tickers]

            for ticker in active_signals[:MAX_CONCURRENT - n_open]:
                if ticker not in current_prices:
                    continue
                price = current_prices[ticker]
                entry_price = price * (1 + SLIPPAGE_PCT)
                shares = int(MAX_PER_TRADE / entry_price)
                if shares <= 0:
                    continue

                cost = shares * entry_price
                if cost > state["capital"]:
                    shares = int(state["capital"] / entry_price)
                    if shares <= 0:
                        continue
                    cost = shares * entry_price

                state["capital"] -= cost

                # Determine direction based on credit regime
                if ticker in BULL_ASSETS:
                    direction = "bull_credit"
                elif ticker in BEAR_ASSETS:
                    direction = "bear_credit"
                else:
                    direction = "neutral"

                pos = {
                    "ticker": ticker,
                    "entry_date": today_str,
                    "entry_price": round(entry_price, 4),
                    "entry_price_adj": round(entry_price, 4),
                    "shares": shares,
                    "hwm": entry_price,
                    "high_water_mark": entry_price,
                    "direction": direction,
                }
                state["positions"].append(pos)
                n_open += 1

                log(f"  ENTRY {ticker}: ${entry_price:.2f} x {shares} shares = "
                    f"${cost:.2f} | Direction: {direction}")

                fire_callback({
                    "ticker": ticker,
                    "direction": direction,
                    "action": "entry",
                    "price": entry_price,
                })

    # --- 3. Update equity ---
    position_value = sum(
        p["shares"] * current_prices.get(p["ticker"], p["entry_price"])
        for p in state["positions"]
    )
    state["equity"] = state["capital"] + position_value
    if state["equity"] > state["peak_equity"]:
        state["peak_equity"] = state["equity"]
    dd = (state["equity"] - state["peak_equity"]) / state["peak_equity"]
    if dd < state["max_drawdown"]:
        state["max_drawdown"] = dd

    state["equity_curve"].append({
        "date": today_str,
        "equity": round(state["equity"], 2),
        "n_open": len(state["positions"]),
    })
    # Keep last 200 entries
    state["equity_curve"] = state["equity_curve"][-200:]

    state["last_run_date"] = today_str
    save_state(state)
    print_summary(state)
    log("=== Done ===")


if __name__ == "__main__":
    run()
