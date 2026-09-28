#!/usr/bin/env python3
"""
Scanner Outcome Tracker
========================
Tracks outcomes of play scanner recommendations to measure real win rate.

For each "recommended" trade in agentic_trade_log.json:
  - Pulls actual price data for the underlying at entry date
  - Tracks price movement over the trade plan's hold period
  - Marks as WIN if underlying moved favorably by target amount
  - Marks as LOSS if stop was hit or time expired unfavorably
  - Updates trade log with actual outcomes

Runs daily at 4:30 PM ET via PM2 cron: "30 20 * * 1-5"
"""
import json
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytz
import yfinance as yf

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)

ET = pytz.timezone("US/Eastern")
STATE_DIR = Path("/home/jupiter/Lvl3Quant/state")
TRADE_LOG = STATE_DIR / "agentic_trade_log.json"
OUTCOME_HISTORY = STATE_DIR / "scanner_outcome_history.csv"


def load_trade_log() -> dict:
    if TRADE_LOG.exists():
        return json.loads(TRADE_LOG.read_text())
    return {"trades": [], "stats": {"total": 0, "wins": 0, "losses": 0, "open": 0}}


def save_trade_log(data: dict):
    TRADE_LOG.write_text(json.dumps(data, indent=2))


def evaluate_trade(trade: dict) -> dict:
    """Evaluate a recommended trade against actual price movement."""
    ticker = trade["ticker"]
    entry_date = trade["entry_date"]
    direction = trade.get("direction", "call")
    strike = trade.get("strike", 0)

    # Default hold period: 14 trading days (3 weeks)
    max_hold_days = 14

    try:
        # Get price data from entry date to now (or max hold period)
        start = pd.Timestamp(entry_date) - pd.Timedelta(days=1)
        end = min(
            pd.Timestamp(entry_date) + pd.Timedelta(days=max_hold_days * 1.5),
            pd.Timestamp(datetime.now(ET).date())
        )

        if end <= start + pd.Timedelta(days=1):
            return {"status": "too_early", "days_held": 0}

        tk = yf.Ticker(ticker)
        hist = tk.history(start=start.strftime("%Y-%m-%d"),
                          end=end.strftime("%Y-%m-%d"))

        if hist.empty or len(hist) < 2:
            return {"status": "no_data"}

        # Find entry price (close on entry date or next available)
        entry_dt = pd.Timestamp(entry_date)
        hist.index = hist.index.tz_localize(None)
        after_entry = hist[hist.index >= entry_dt]
        if after_entry.empty:
            return {"status": "no_data_after_entry"}

        entry_price = after_entry.iloc[0]["Close"]
        current_price = after_entry.iloc[-1]["Close"]
        days_held = len(after_entry)

        # Calculate price movement
        if direction in ("call", "long"):
            pct_move = (current_price - entry_price) / entry_price * 100
            max_favorable = ((after_entry["High"].max() - entry_price) / entry_price) * 100
            max_adverse = ((after_entry["Low"].min() - entry_price) / entry_price) * 100
        else:  # put / short
            pct_move = (entry_price - current_price) / entry_price * 100
            max_favorable = ((entry_price - after_entry["Low"].min()) / entry_price) * 100
            max_adverse = ((entry_price - after_entry["High"].max()) / entry_price) * 100

        # Win/loss logic for options:
        # Options need ~2-3% underlying move for 40-80% option gain
        # Stop is typically -40% option value (~1.5% adverse underlying move)
        WIN_THRESHOLD = 2.0   # 2% favorable underlying = ~60-100% option gain
        LOSS_THRESHOLD = -2.0  # 2% adverse = ~40-60% option loss

        # Check if trade should be closed
        expired = days_held >= max_hold_days
        hit_target = max_favorable >= WIN_THRESHOLD
        hit_stop = max_adverse <= LOSS_THRESHOLD

        if hit_target:
            outcome = "WIN"
            reason = f"underlying moved +{max_favorable:.1f}% favorably"
        elif hit_stop and not hit_target:
            outcome = "LOSS"
            reason = f"underlying moved {max_adverse:.1f}% adversely"
        elif expired:
            outcome = "WIN" if pct_move > 0.5 else "LOSS"
            reason = f"expired at {pct_move:+.1f}% move"
        else:
            outcome = "OPEN"
            reason = f"day {days_held}, currently {pct_move:+.1f}%"

        return {
            "status": outcome,
            "entry_price": round(entry_price, 2),
            "current_price": round(current_price, 2),
            "pct_move": round(pct_move, 2),
            "max_favorable_pct": round(max_favorable, 2),
            "max_adverse_pct": round(max_adverse, 2),
            "days_held": days_held,
            "reason": reason,
        }

    except Exception as e:
        return {"status": "error", "reason": str(e)}


def run_tracker():
    now = datetime.now(ET)
    print(f"Scanner Outcome Tracker — {now.strftime('%Y-%m-%d %H:%M ET')}")
    print("=" * 55)

    data = load_trade_log()
    trades = data.get("trades", [])

    if not trades:
        print("No trades to track.")
        return

    wins = 0
    losses = 0
    open_count = 0
    outcomes = []

    for trade in trades:
        if trade.get("status") in ("WIN", "LOSS"):
            # Already resolved
            if trade["status"] == "WIN":
                wins += 1
            else:
                losses += 1
            continue

        print(f"\n  Evaluating #{trade['id']} {trade['ticker']} {trade['direction']} ${trade.get('strike', '?')}...")
        result = evaluate_trade(trade)

        if result["status"] in ("WIN", "LOSS"):
            trade["status"] = result["status"]
            trade["exit_date"] = now.strftime("%Y-%m-%d")
            trade["exit_pnl"] = result.get("pct_move", 0)
            trade["outcome_detail"] = result
            if result["status"] == "WIN":
                wins += 1
            else:
                losses += 1
            print(f"    → {result['status']}: {result['reason']}")
            outcomes.append({
                "id": trade["id"],
                "ticker": trade["ticker"],
                "direction": trade["direction"],
                "entry_date": trade["entry_date"],
                "exit_date": trade.get("exit_date"),
                "status": result["status"],
                "pct_move": result.get("pct_move"),
                "max_favorable": result.get("max_favorable_pct"),
                "max_adverse": result.get("max_adverse_pct"),
                "days_held": result.get("days_held"),
            })
        elif result["status"] == "OPEN":
            open_count += 1
            trade["outcome_detail"] = result
            print(f"    → OPEN: {result['reason']}")
        else:
            open_count += 1
            print(f"    → {result['status']}: {result.get('reason', 'waiting')}")

    # Update stats
    data["stats"] = {
        "total": len(trades),
        "wins": wins,
        "losses": losses,
        "open": open_count,
        "win_rate_pct": round(wins / max(wins + losses, 1) * 100, 1),
    }

    save_trade_log(data)

    # Append outcomes to CSV history
    if outcomes:
        df = pd.DataFrame(outcomes)
        if OUTCOME_HISTORY.exists():
            df.to_csv(OUTCOME_HISTORY, mode="a", header=False, index=False)
        else:
            df.to_csv(OUTCOME_HISTORY, index=False)

    print(f"\n  Summary: {wins}W / {losses}L / {open_count} open")
    if wins + losses > 0:
        wr = wins / (wins + losses) * 100
        print(f"  Win rate: {wr:.0f}%")
    print("Done.")


if __name__ == "__main__":
    run_tracker()
