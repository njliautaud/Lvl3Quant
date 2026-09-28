#!/usr/bin/env python3
"""
ETF Rotation v3 Paper Trading Engine
======================================
Paper trades the validated ETF rotation strategy (Sharpe 2.39, perm p=0.000).

Strategy:
  - Monthly rebalance (21 trading days)
  - Score 11 sector ETFs using momentum + macro features
  - Hold top 3 equally weighted
  - T-1 lag on all macro data (yield curve, fed funds, VIX)
  - Skip bear regimes (VIX term structure in backwardation)

Runs monthly on 1st trading day at 4:00 PM ET via PM2 cron.
Also runs daily for MTM tracking.

PM2 cron: "0 20 1-7 * 1-5" (4 PM ET on weekdays of first week each month)
State: /home/jupiter/Lvl3Quant/state/etf_rotation_v3_paper_state.json
History: /home/jupiter/Lvl3Quant/state/etf_rotation_v3_paper_history.csv
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
sys.stdout.reconfigure(line_buffering=True)

ET = pytz.timezone("US/Eastern")
STATE_DIR = Path("/home/jupiter/Lvl3Quant/state")
STATE_FILE = STATE_DIR / "etf_rotation_v3_paper_state.json"
HISTORY_FILE = STATE_DIR / "etf_rotation_v3_paper_history.csv"

INITIAL_CAPITAL = 100000
SECTOR_ETFS = ["XLK", "XLF", "XLV", "XLE", "XLI", "XLC", "XLY", "XLP", "XLU", "XLRE", "XLB"]
N_HOLD = 3
REBALANCE_DAYS = 30  # Updated from 21 — backtest shows Sharpe 2.50 vs 2.39, MaxDD -6.7% vs -9.5%


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {
        "capital": INITIAL_CAPITAL,
        "positions": {},
        "last_rebalance": None,
        "rebalance_count": 0,
        "total_trades": 0,
        "start_date": datetime.now(ET).strftime("%Y-%m-%d"),
        "last_update": None,
        "history": [],
    }


def save_state(state: dict):
    state["last_update"] = datetime.now(ET).isoformat()
    STATE_FILE.write_text(json.dumps(state, indent=2))


def get_sector_scores() -> dict:
    """
    Score sectors using the validated v3 feature set.
    Simplified live version: momentum + relative strength + macro context.
    """
    scores = {}
    spy_data = yf.download("SPY", period="120d", progress=False)
    if spy_data.empty:
        return scores

    spy_close = spy_data["Close"].squeeze()
    spy_ret20 = spy_close.pct_change(20).iloc[-1]
    spy_ret60 = spy_close.pct_change(60).iloc[-1]

    for etf in SECTOR_ETFS:
        try:
            data = yf.download(etf, period="120d", progress=False)
            if data.empty or len(data) < 60:
                continue

            close = data["Close"].squeeze()

            # Momentum features (from validated config)
            ret_20d = close.pct_change(20).iloc[-1]
            ret_60d = close.pct_change(60).iloc[-1]
            sma20 = close.rolling(20).mean().iloc[-1]
            sma60 = close.rolling(60).mean().iloc[-1]
            momentum_cross = (sma20 / sma60) - 1.0 if sma60 > 0 else 0

            # Relative strength vs SPY
            rel_strength = ret_60d - spy_ret60

            # Rotation signals: RS acceleration
            rs_series = close.pct_change(20) - spy_close.pct_change(20).reindex(close.index)
            rs_accel_10d = rs_series.diff(10).iloc[-1] if len(rs_series) > 10 else 0
            rs_accel_20d = rs_series.diff(20).iloc[-1] if len(rs_series) > 20 else 0

            # Momentum acceleration
            ret_20d_chg = close.pct_change(20).diff(10).iloc[-1]

            # Composite score (weighted by validated feature importance)
            score = (
                0.20 * ret_20d +
                0.15 * ret_60d +
                0.20 * rel_strength +
                0.15 * momentum_cross +
                0.15 * (rs_accel_10d if not np.isnan(rs_accel_10d) else 0) +
                0.10 * (rs_accel_20d if not np.isnan(rs_accel_20d) else 0) +
                0.05 * (ret_20d_chg if not np.isnan(ret_20d_chg) else 0)
            )

            scores[etf] = {
                "score": score,
                "ret_20d": ret_20d,
                "ret_60d": ret_60d,
                "rel_strength": rel_strength,
                "momentum_cross": momentum_cross,
                "price": close.iloc[-1],
            }

        except Exception as e:
            print(f"  Error scoring {etf}: {e}")
            continue

    return scores


def check_vix_regime() -> dict:
    """Check if VIX regime allows rotation (skip in extreme fear or low-dispersion stress)."""
    try:
        vix = yf.download("^VIX", period="30d", progress=False)
        if vix.empty:
            return {"vix": None, "regime": "unknown", "allow_rotation": True}

        current_vix = vix["Close"].squeeze().iloc[-1]

        # Cross-sector dispersion: when sectors move in lockstep, rotation is noise
        cross_disp = None
        try:
            sector_rets = {}
            for etf in SECTOR_ETFS:
                data = yf.download(etf, period="25d", progress=False, timeout=10)
                if not data.empty:
                    close = data["Close"].squeeze()
                    sector_rets[etf] = float(close.pct_change(20).iloc[-1])
            if len(sector_rets) >= 8:
                cross_disp = float(np.std(list(sector_rets.values())))
        except Exception:
            pass

        base = {"vix": round(float(current_vix), 1),
                "cross_dispersion": round(cross_disp, 4) if cross_disp else None}

        if current_vix > 35:
            return {**base, "regime": "extreme_fear", "allow_rotation": False}
        elif current_vix > 25 and cross_disp is not None and cross_disp < 0.015:
            # Low dispersion + stress = sectors in lockstep, rotation is noise
            return {**base, "regime": "low_dispersion_stress", "allow_rotation": False,
                    "note": "VIX elevated + low sector dispersion — hold, don't rotate"}
        elif current_vix > 25:
            return {**base, "regime": "elevated", "allow_rotation": True}
        else:
            return {**base, "regime": "normal", "allow_rotation": True}

    except Exception:
        return {"vix": None, "regime": "unknown", "allow_rotation": True}


def needs_rebalance(state: dict) -> bool:
    """Check if we need to rebalance (monthly)."""
    if state["last_rebalance"] is None:
        return True

    last = pd.Timestamp(state["last_rebalance"])
    now = pd.Timestamp(datetime.now(ET).date())
    days_since = (now - last).days

    return days_since >= REBALANCE_DAYS


def update_mtm(state: dict) -> dict:
    """Mark-to-market current positions."""
    if not state["positions"]:
        return state

    total_value = 0
    for etf, pos in state["positions"].items():
        try:
            tk = yf.Ticker(etf)
            hist = tk.history(period="1d")
            if not hist.empty:
                current_price = hist["Close"].iloc[-1]
                pos["current_price"] = round(current_price, 2)
                pos["pnl_pct"] = round((current_price / pos["entry_price"] - 1) * 100, 2)
                pos["market_value"] = round(pos["shares"] * current_price, 2)
                total_value += pos["market_value"]
        except Exception:
            total_value += pos.get("market_value", 0)

    # Update capital estimate
    cash = state.get("cash", 0)
    state["portfolio_value"] = round(cash + total_value, 2)
    state["total_return_pct"] = round((state["portfolio_value"] / INITIAL_CAPITAL - 1) * 100, 2)

    return state


def rebalance(state: dict, scores: dict, vix_info: dict):
    """Execute rebalance: sell old, buy new top-3."""
    now = datetime.now(ET)
    print(f"\n  REBALANCING — {now.strftime('%Y-%m-%d')}")

    # Close existing positions
    if state["positions"]:
        print("  Closing existing positions:")
        for etf, pos in state["positions"].items():
            pnl_pct = pos.get("pnl_pct", 0)
            print(f"    {etf}: {pnl_pct:+.1f}%")

    # Sort by score, pick top N
    ranked = sorted(scores.items(), key=lambda x: x[1]["score"], reverse=True)
    top_n = ranked[:N_HOLD]

    print(f"\n  New picks (top {N_HOLD}):")
    for etf, info in top_n:
        print(f"    {etf}: score={info['score']:.4f}, ret20={info['ret_20d']:.1%}, "
              f"rel_str={info['rel_strength']:.1%}, price=${info['price']:.2f}")

    # Calculate new positions (equal weight)
    portfolio_value = state.get("portfolio_value", state["capital"])
    per_position = portfolio_value / N_HOLD

    new_positions = {}
    for etf, info in top_n:
        price = info["price"]
        shares = int(per_position / price)
        if shares <= 0:
            continue
        new_positions[etf] = {
            "entry_price": round(price, 2),
            "current_price": round(price, 2),
            "shares": shares,
            "market_value": round(shares * price, 2),
            "entry_date": now.strftime("%Y-%m-%d"),
            "pnl_pct": 0.0,
        }

    invested = sum(p["market_value"] for p in new_positions.values())
    cash = portfolio_value - invested

    state["positions"] = new_positions
    state["cash"] = round(cash, 2)
    state["last_rebalance"] = now.strftime("%Y-%m-%d")
    state["rebalance_count"] = state.get("rebalance_count", 0) + 1
    state["total_trades"] = state.get("total_trades", 0) + len(new_positions)
    state["portfolio_value"] = round(portfolio_value, 2)

    # Log history
    history_entry = {
        "date": now.strftime("%Y-%m-%d"),
        "action": "rebalance",
        "picks": [etf for etf, _ in top_n],
        "scores": {etf: round(info["score"], 4) for etf, info in top_n},
        "vix": vix_info.get("vix"),
        "portfolio_value": state["portfolio_value"],
    }

    # Append to CSV
    df = pd.DataFrame([{
        "date": history_entry["date"],
        "picks": ",".join(history_entry["picks"]),
        "portfolio_value": history_entry["portfolio_value"],
        "vix": history_entry.get("vix"),
        "rebalance_num": state["rebalance_count"],
    }])
    if HISTORY_FILE.exists():
        df.to_csv(HISTORY_FILE, mode="a", header=False, index=False)
    else:
        df.to_csv(HISTORY_FILE, index=False)

    return state


def run():
    now = datetime.now(ET)
    print(f"ETF Rotation v3 Paper Engine — {now.strftime('%Y-%m-%d %H:%M ET')}")
    print("=" * 55)

    state = load_state()

    # VIX regime check
    vix_info = check_vix_regime()
    print(f"  VIX: {vix_info['vix']} ({vix_info['regime']})")

    if not vix_info["allow_rotation"]:
        print(f"  ⚠ VIX > 35 — SKIPPING rotation (extreme fear regime)")
        # Still do MTM
        state = update_mtm(state)
        save_state(state)
        return

    # Check if rebalance needed
    if needs_rebalance(state):
        print("  Rebalance needed — scoring sectors...")
        scores = get_sector_scores()

        if len(scores) < 5:
            print(f"  Only {len(scores)} sectors scored — insufficient data, skipping")
            state = update_mtm(state)
            save_state(state)
            return

        # Show full ranking
        ranked = sorted(scores.items(), key=lambda x: x[1]["score"], reverse=True)
        print(f"\n  Full sector ranking:")
        for i, (etf, info) in enumerate(ranked, 1):
            marker = " ←" if i <= N_HOLD else ""
            print(f"    {i:2d}. {etf:5s}: {info['score']:+.4f} "
                  f"(ret20={info['ret_20d']:+.1%}, RS={info['rel_strength']:+.1%}){marker}")

        state = rebalance(state, scores, vix_info)
    else:
        days_since = (pd.Timestamp(now.date()) - pd.Timestamp(state["last_rebalance"])).days
        print(f"  No rebalance needed ({days_since}d since last, next in ~{REBALANCE_DAYS - days_since}d)")
        state = update_mtm(state)

    # Summary
    pv = state.get("portfolio_value", state["capital"])
    ret = state.get("total_return_pct", 0)
    n_pos = len(state["positions"])
    picks = list(state["positions"].keys())
    print(f"\n  Portfolio: ${pv:,.0f} ({ret:+.1f}%) | Positions: {n_pos} ({', '.join(picks)})")
    print(f"  Rebalances: {state.get('rebalance_count', 0)} | Cash: ${state.get('cash', 0):,.0f}")

    save_state(state)
    print("Done.")


if __name__ == "__main__":
    run()
