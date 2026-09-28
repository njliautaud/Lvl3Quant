#!/usr/bin/env python3
"""
Strategy Rotation v2 F (Dynamic Contrarian) — Paper Engine
Validated: 5/5 gates + 4/5 adversarial (Sharpe 1.886, regime gap 0.012)

Logic (enhanced version of Strategy Rotation A):
  - VIX > 25 AND VIX declining (5d): VIX fade → buy SPY
  - SPY > 200-SMA (bull): earnings momentum → long QQQ
  - SPY < 200-SMA (bear): dynamic contrarian → buy SPY when
    weekly return < -1.5 × VIX/100. Hold 8d during high-vol bears.
  - Otherwise: cash

Key improvement over A: dynamic VIX-adjusted contrarian threshold
and extended hold period in high-vol bears. Regime gap 0.012 (near-perfect).

Daily execution at 4:40 PM ET via cron.
Starting capital: $645.
"""

import json, datetime as dt, warnings, sys, logging
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────
STATE_FILE = Path("/home/jupiter/Lvl3Quant/state/strategy_rotation_v2f_paper_state.json")
LOG_FILE = Path("/home/jupiter/Lvl3Quant/paper_engines/logs/strategy_rotation_v2f_paper.log")
STARTING_CAPITAL = 645.0
SLIPPAGE_PCT = 0.0002  # 2 bps

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


def load_state():
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {
        "created": dt.datetime.now().isoformat(),
        "capital": STARTING_CAPITAL,
        "position": None,
        "mode": "idle",
        "contrarian_hold_remaining": 0,
        "trades": [],
        "daily_equity": [],
    }


def save_state(state):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def get_market_data():
    """Fetch recent market data for regime detection."""
    end = dt.datetime.now()
    start = end - dt.timedelta(days=400)

    tickers = ["SPY", "QQQ", "^VIX"]
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"]
    else:
        close = data

    if isinstance(close.columns, pd.MultiIndex):
        close.columns = close.columns.get_level_values(-1)

    df = pd.DataFrame()
    df["SPY"] = close["SPY"]
    df["QQQ"] = close["QQQ"]
    df["VIX"] = close.get("^VIX", close.get("VIX", pd.Series(dtype=float)))
    df = df.ffill().dropna()

    df["SMA200"] = df["SPY"].rolling(200).mean()
    df["SPY_ret5"] = df["SPY"].pct_change(5)
    df["VIX_chg5"] = df["VIX"].pct_change(5)
    df["bull"] = (df["SPY"] > df["SMA200"]).astype(int)

    return df


def determine_regime(df):
    """Determine current regime with DYNAMIC contrarian threshold."""
    latest = df.iloc[-1]
    vix = latest["VIX"]
    bull = bool(latest["bull"])
    spy_ret5 = latest["SPY_ret5"]
    vix_chg5 = latest["VIX_chg5"]
    spy_price = latest["SPY"]
    qqq_price = latest["QQQ"]
    sma200 = latest["SMA200"]

    # Dynamic contrarian threshold: -1.5 × VIX/100
    contrarian_threshold = -1.5 * vix / 100

    if vix > 25 and vix_chg5 < 0:
        mode = "spy_vix_fade"
        reason = f"VIX={vix:.1f}>25 declining → VIX fade (SPY)"
    elif bull:
        mode = "qqq_long"
        reason = f"Bull (SPY ${spy_price:.0f} > SMA ${sma200:.0f}) → QQQ long"
    elif spy_ret5 <= contrarian_threshold:
        mode = "spy_contrarian"
        reason = f"Bear + dip ({spy_ret5:.1%} wkly ≤ {contrarian_threshold:.1%} threshold) → Dynamic contrarian"
    else:
        mode = "bear_waiting"
        reason = f"Bear, no trigger (wkly {spy_ret5:.1%}, need ≤{contrarian_threshold:.1%})"

    # Determine hold period: 8d in high-vol (VIX > 20), 5d otherwise
    hold_days = 7 if vix > 20 else 4  # -1 because entry day counts

    return {
        "mode": mode, "reason": reason,
        "vix": round(float(vix), 2), "bull": bull,
        "spy_price": round(float(spy_price), 2),
        "qqq_price": round(float(qqq_price), 2),
        "sma200": round(float(sma200), 2),
        "spy_ret5": round(float(spy_ret5), 4),
        "contrarian_threshold": round(float(contrarian_threshold), 4),
        "hold_days": hold_days,
    }


def execute_paper(state, regime, df):
    """Execute paper trades based on regime."""
    today = dt.datetime.now().strftime("%Y-%m-%d")
    latest = df.iloc[-1]
    mode = regime["mode"]
    position = state["position"]

    # --- Handle contrarian countdown ---
    if state["contrarian_hold_remaining"] > 0:
        state["contrarian_hold_remaining"] -= 1
        if state["contrarian_hold_remaining"] == 0 and position:
            exit_price = float(latest["SPY"]) * (1 - SLIPPAGE_PCT)
            pnl = (exit_price - position["entry_price"]) * position["shares"]
            ret_pct = (exit_price / position["entry_price"] - 1) * 100
            state["capital"] = position["entry_price"] * position["shares"] + pnl
            state["trades"].append({
                "ticker": "SPY", "type": "dynamic_contrarian",
                "entry_date": position["entry_date"], "exit_date": today,
                "entry_price": position["entry_price"],
                "exit_price": round(exit_price, 2),
                "shares": position["shares"],
                "pnl": round(pnl, 2), "return_pct": round(ret_pct, 2),
            })
            log.info(f"CLOSED contrarian SPY: {ret_pct:+.2f}% (${pnl:+.2f})")
            state["position"] = None
            position = None
        elif state["contrarian_hold_remaining"] > 0:
            log.info(f"Contrarian hold: {state['contrarian_hold_remaining']}d left")

    # --- Mode transition: close position if regime changed ---
    if position and mode != state.get("mode", "idle") and state["contrarian_hold_remaining"] == 0:
        ticker = position["ticker"]
        price_key = ticker if ticker in ["SPY", "QQQ"] else "SPY"
        exit_price = float(latest[price_key]) * (1 - SLIPPAGE_PCT)
        pnl = (exit_price - position["entry_price"]) * position["shares"]
        ret_pct = (exit_price / position["entry_price"] - 1) * 100
        state["capital"] = position["entry_price"] * position["shares"] + pnl
        state["trades"].append({
            "ticker": ticker, "type": state.get("mode", "unknown"),
            "entry_date": position["entry_date"], "exit_date": today,
            "entry_price": position["entry_price"],
            "exit_price": round(exit_price, 2),
            "shares": position["shares"],
            "pnl": round(pnl, 2), "return_pct": round(ret_pct, 2),
        })
        log.info(f"CLOSED {ticker} (regime change): {ret_pct:+.2f}% (${pnl:+.2f})")
        state["position"] = None
        position = None

    # --- Open new position if needed ---
    if position is None and state["contrarian_hold_remaining"] == 0:
        if mode in ("qqq_long", "spy_vix_fade", "spy_contrarian"):
            ticker = "QQQ" if mode == "qqq_long" else "SPY"
            entry_price = float(latest[ticker]) * (1 + SLIPPAGE_PCT)
            shares = state["capital"] / entry_price
            state["position"] = {
                "ticker": ticker, "shares": round(shares, 4),
                "entry_price": round(entry_price, 2),
                "entry_date": today,
            }
            state["capital"] = 0
            if mode == "spy_contrarian":
                state["contrarian_hold_remaining"] = regime["hold_days"]
            log.info(f"OPENED {ticker} ({mode}): {shares:.4f} sh @ ${entry_price:.2f}")

    state["mode"] = mode

    # --- Compute equity ---
    if state["position"]:
        pos = state["position"]
        current = float(latest[pos["ticker"]]) if pos["ticker"] in ["SPY", "QQQ"] else pos["entry_price"]
        unrealized = (current - pos["entry_price"]) * pos["shares"]
        equity = state["capital"] + pos["entry_price"] * pos["shares"] + unrealized
    else:
        equity = state["capital"]

    state["daily_equity"].append({"date": today, "equity": round(equity, 2), "mode": mode})
    if len(state["daily_equity"]) > 90:
        state["daily_equity"] = state["daily_equity"][-90:]

    return state


def main():
    log.info("=" * 60)
    log.info("STRATEGY ROTATION v2 F (Dynamic Contrarian) — Paper Engine")

    state = load_state()

    try:
        df = get_market_data()
    except Exception as e:
        log.error(f"Market data fetch failed: {e}")
        save_state(state)
        return

    if len(df) < 200:
        log.warning(f"Only {len(df)} days — need 200 for SMA. Skipping.")
        save_state(state)
        return

    regime = determine_regime(df)
    log.info(f"Regime: {regime['reason']}")
    log.info(f"  SPY ${regime['spy_price']} | QQQ ${regime['qqq_price']} | VIX {regime['vix']}")

    state = execute_paper(state, regime, df)

    equity_now = state["daily_equity"][-1]["equity"] if state["daily_equity"] else state["capital"]
    total_ret = (equity_now / STARTING_CAPITAL - 1) * 100
    n_closed = len(state["trades"])
    wins = sum(1 for t in state["trades"] if t["pnl"] > 0)
    wr = (wins / n_closed * 100) if n_closed > 0 else 0
    total_pnl = sum(t["pnl"] for t in state["trades"])

    log.info(f"Equity: ${equity_now:.2f} ({total_ret:+.1f}%) | Closed: {n_closed} | WR: {wr:.0f}% | PnL: ${total_pnl:+.2f}")

    if state["position"]:
        p = state["position"]
        log.info(f"  Open: {p['ticker']} {p['shares']:.2f}sh @ ${p['entry_price']} (since {p['entry_date']})")
    else:
        log.info("  Open: None (cash)")

    save_state(state)
    log.info("State saved. Done.")


if __name__ == "__main__":
    main()
