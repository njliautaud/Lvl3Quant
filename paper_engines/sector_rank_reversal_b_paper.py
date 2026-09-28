#!/usr/bin/env python3
"""
Sector Rank Reversal B Paper Engine
=====================================
Validated strategy — passed 6/6 adversarial validation.
Sharpe 2.16, 510 trades, WR 56.9%.

Strategy Rules:
  - Rank all 11 sector ETFs by 20-day return
  - Buy the bottom 2 ranked sectors when they show positive 3-day momentum
  - Hold 5 days, equal weight
  - Starting capital: $645

Cron: 25 20 * * 1-5 (4:25 PM ET)
State: /home/jupiter/Lvl3Quant/state/sector_rank_reversal_b_paper_state.json
"""
import json
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytz
import yfinance as yf

warnings.filterwarnings("ignore")

ET = pytz.timezone("US/Eastern")
BASE = Path("/home/jupiter/Lvl3Quant")
STATE_DIR = BASE / "state"
STATE_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = STATE_DIR / "sector_rank_reversal_b_paper_state.json"
LOG_DIR = BASE / "paper_engines" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "sector_rank_reversal_b.log"

# ── Strategy Parameters ──
INITIAL_CAPITAL = 645.0
HOLD_DAYS = 5
LOOKBACK_RANK = 20   # 20-day return for ranking
MOMENTUM_DAYS = 3    # 3-day momentum check
BOTTOM_N = 2         # Buy bottom 2 ranked sectors
SPREAD_BPS = 2

SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLI", "XLP", "XLU", "XLRE", "XLB", "XLC"]


def log(msg: str):
    ts = datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S")
    line = f"{ts} [RANK-REVERSAL-B] {msg}"
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
        "signals": [],
        "capital": INITIAL_CAPITAL,
        "equity": INITIAL_CAPITAL,
        "total_pnl": 0,
        "wins": 0,
        "losses": 0,
        "n_trades": 0,
        "created": datetime.now(ET).isoformat(),
        "last_updated": None,
    }


def save_state(state: dict):
    state["last_updated"] = datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S.%f")
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, default=str))
    tmp.rename(STATE_FILE)


def fetch_data() -> dict:
    """Fetch sector ETF prices. Returns {ticker: {prices: Series, current: float}}."""
    end = datetime.now(ET)
    start = end - timedelta(days=45)  # Need ~25 trading days

    try:
        raw = yf.download(SECTOR_ETFS, start=start.strftime("%Y-%m-%d"),
                          end=end.strftime("%Y-%m-%d"),
                          auto_adjust=True, progress=False, threads=True, timeout=30)
    except Exception as e:
        log(f"ERROR: yfinance download failed: {e}")
        return {}

    if raw.empty:
        log("ERROR: yfinance returned empty data")
        return {}

    mi = isinstance(raw.columns, pd.MultiIndex)
    close = raw["Close"] if mi else raw
    if isinstance(close.columns, pd.MultiIndex):
        close.columns = close.columns.get_level_values(-1)

    price_data = {}
    for etf in SECTOR_ETFS:
        if etf in close.columns:
            series = close[etf].dropna()
            if len(series) >= LOOKBACK_RANK + 2:
                price_data[etf] = {
                    "prices": series,
                    "current": float(series.iloc[-1]),
                }

    return price_data


def run():
    now = datetime.now(ET)
    today = now.date()
    today_str = today.isoformat()

    if today.weekday() >= 5:
        log("Weekend — skipping.")
        return

    log(f"=== Sector Rank Reversal B Engine — {today_str} ===")
    state = load_state()

    # Fetch data
    price_data = fetch_data()
    if len(price_data) < 6:
        log(f"ERROR: Only {len(price_data)} sectors with data. Need at least 6. Aborting.")
        save_state(state)
        return

    # --- 1. Compute rankings ---
    rankings = {}
    momentum_3d = {}
    for etf, data in price_data.items():
        prices = data["prices"]
        # 20-day return
        if len(prices) >= LOOKBACK_RANK + 1:
            ret_20d = float(prices.iloc[-1] / prices.iloc[-(LOOKBACK_RANK + 1)] - 1)
            rankings[etf] = ret_20d

        # 3-day momentum
        if len(prices) >= 4:
            ret_3d = float(prices.iloc[-1] / prices.iloc[-4] - 1)
            momentum_3d[etf] = ret_3d

    # Sort by 20-day return (ascending = worst performers first)
    sorted_sectors = sorted(rankings.items(), key=lambda x: x[1])
    log(f"20d Return Rankings (worst to best):")
    for i, (etf, ret) in enumerate(sorted_sectors):
        mom = momentum_3d.get(etf, 0)
        tag = " <-- BOTTOM" if i < BOTTOM_N else ""
        log(f"  {i+1}. {etf}: 20d={ret*100:+.2f}%, 3d_mom={mom*100:+.2f}%{tag}")

    # --- 2. Close expired positions ---
    still_open = []
    for pos in state["positions"]:
        entry_date = datetime.fromisoformat(pos["entry_date"]).date()
        days_held = (today - entry_date).days
        ticker = pos["ticker"]

        if days_held >= HOLD_DAYS:
            if ticker in price_data:
                exit_price = price_data[ticker]["current"] * (1 - SPREAD_BPS / 10_000)
                pnl = (exit_price - pos["entry_price"]) * pos["shares"]
                pnl_pct = (exit_price / pos["entry_price"] - 1) * 100

                state["capital"] += pos["shares"] * exit_price
                state["total_pnl"] += pnl
                state["n_trades"] += 1
                if pnl >= 0:
                    state["wins"] += 1
                else:
                    state["losses"] += 1

                state["closed_trades"].append({
                    "ticker": ticker,
                    "entry_date": pos["entry_date"],
                    "exit_date": today_str,
                    "entry_price": pos["entry_price"],
                    "exit_price": round(exit_price, 4),
                    "pnl": round(pnl, 2),
                    "pnl_pct": round(pnl_pct, 2),
                    "days_held": days_held,
                    "rank_at_entry": pos.get("rank_at_entry"),
                })
                state["closed_trades"] = state["closed_trades"][-100:]
                log(f"  CLOSED {ticker}: ${pnl:+.2f} ({pnl_pct:+.2f}%) after {days_held}d")
            else:
                log(f"  WARN: Cannot close {ticker} — no price. Keeping.")
                still_open.append(pos)
                continue
        else:
            still_open.append(pos)

    state["positions"] = still_open

    # --- 3. Identify bottom 2 with positive 3-day momentum ---
    bottom_n = sorted_sectors[:BOTTOM_N]
    qualifying = []
    new_signals = []

    for etf, ret_20d in sorted_sectors:
        mom = momentum_3d.get(etf, 0)
        rank_idx = [e for e, _ in sorted_sectors].index(etf) + 1
        is_bottom = rank_idx <= BOTTOM_N
        has_mom = mom > 0

        signal = {
            "ticker": etf,
            "direction": "long" if (is_bottom and has_mom) else "none",
            "strength": round(max(0, 1.0 - rank_idx / len(sorted_sectors)), 2),
            "ret_20d": round(ret_20d * 100, 2),
            "mom_3d": round(mom * 100, 2),
            "rank": rank_idx,
        }
        new_signals.append(signal)

        if is_bottom and has_mom:
            qualifying.append(etf)

    state["signals"] = new_signals

    # --- 4. Open new positions for qualifying sectors ---
    held_tickers = [p["ticker"] for p in state["positions"]]
    for etf in qualifying:
        if etf in held_tickers:
            log(f"  SKIP {etf}: already holding")
            continue

        if etf not in price_data:
            continue

        buy_price = price_data[etf]["current"] * (1 + SPREAD_BPS / 10_000)
        per_pos = state["capital"] * 0.40  # 40% per position (max 2 positions)
        if per_pos < 20:
            log(f"  SKIP {etf}: insufficient capital (${state['capital']:.0f})")
            continue

        shares = per_pos / buy_price
        cost = shares * buy_price
        state["capital"] -= cost

        rank_idx = [e for e, _ in sorted_sectors].index(etf) + 1
        pos = {
            "ticker": etf,
            "shares": round(shares, 6),
            "entry_price": round(buy_price, 4),
            "entry_date": now.isoformat(),
            "rank_at_entry": rank_idx,
            "ret_20d_at_entry": round(rankings.get(etf, 0) * 100, 2),
            "mom_3d_at_entry": round(momentum_3d.get(etf, 0) * 100, 2),
        }
        state["positions"].append(pos)
        log(f"  OPENED {etf}: rank={rank_idx}, 20d={rankings[etf]*100:+.1f}%, "
            f"3d_mom={momentum_3d.get(etf,0)*100:+.1f}%, {shares:.4f} shares @ ${buy_price:.2f}")

    # --- 5. Mark to market ---
    portfolio_value = state["capital"]
    for pos in state["positions"]:
        ticker = pos["ticker"]
        if ticker in price_data:
            portfolio_value += pos["shares"] * price_data[ticker]["current"]
        else:
            portfolio_value += pos["shares"] * pos["entry_price"]
    state["equity"] = round(portfolio_value, 2)

    total = state["wins"] + state["losses"]
    wr = state["wins"] / total * 100 if total > 0 else 0
    log(f"Equity: ${state['equity']:.2f} | Positions: {len(state['positions'])} | "
        f"Trades: {total} | WR: {wr:.0f}% | PnL: ${state['total_pnl']:+.2f}")

    save_state(state)
    log("Done.")


if __name__ == "__main__":
    run()
