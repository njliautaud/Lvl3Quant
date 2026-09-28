#!/usr/bin/env python3
"""
Signal Aggregator A — Paper Engine (CHAMPION STRATEGY)
=======================================================
Validated: 5/5 gates + 5/5 adversarial
Sharpe 2.70, regime gap 0.179, MDD -9.9%, 121 trades, $645→$3,952

Logic: 5 binary signals → composite score 0-5.
  Score ≥ 3 → long QQQ | Score < 3 → cash

Signals:
  1. REGIME:    SPY > 200-SMA
  2. VIX CALM:  VIX < 20 OR (was >25 recently and declining)
  3. MOMENTUM:  QQQ 20d return > 0
  4. VOLUME:    Any sector ETF w/ 5+ consecutive above-avg volume days
  5. BREADTH:   SPY 20d ret > 0 AND RSP keeping up (>50% of SPY ret)

Daily 4:30 PM ET via cron. Capital: $645.
"""

import json, datetime as dt, warnings, sys, logging
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

STATE_FILE = Path("/home/jupiter/Lvl3Quant/state/signal_aggregator_paper_state.json")
LOG_FILE = Path("/home/jupiter/Lvl3Quant/paper_engines/logs/signal_aggregator_paper.log")
STARTING_CAPITAL = 645.0
SLIPPAGE_PCT = 0.0002

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

SECTOR_ETFS = ["XLK", "XLC", "XLY", "XLE", "XLF", "XLV"]
ALL_TICKERS = ["SPY", "QQQ", "RSP", "^VIX"] + SECTOR_ETFS


def load_state():
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {
        "created": dt.datetime.now().isoformat(),
        "capital": STARTING_CAPITAL,
        "position": None,
        "mode": "cash",
        "trades": [],
        "daily_equity": [],
        "signal_history": [],
    }


def save_state(state):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def get_data():
    end = dt.datetime.now()
    start = end - dt.timedelta(days=300)
    data = yf.download(ALL_TICKERS, start=start, end=end, auto_adjust=True, progress=False)

    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"].copy()
        volume = data["Volume"].copy()
    else:
        close = data
        volume = pd.DataFrame()

    if isinstance(close.columns, pd.MultiIndex):
        close.columns = close.columns.get_level_values(-1)
    if isinstance(volume.columns, pd.MultiIndex):
        volume.columns = volume.columns.get_level_values(-1)

    close = close.dropna(subset=["SPY"])
    volume = volume.loc[close.index]
    return close, volume


def compute_signals(close, volume):
    """Compute 5 binary signals and return dict with latest values."""

    # 1. REGIME: SPY > 200-SMA
    sma200 = close["SPY"].rolling(200).mean()
    regime = int(close["SPY"].iloc[-1] > sma200.iloc[-1]) if not pd.isna(sma200.iloc[-1]) else 0

    # 2. VIX CALM
    vix_col = "^VIX" if "^VIX" in close.columns else "VIX"
    vix_now = None
    vix_calm = 0
    if vix_col in close.columns:
        vix = close[vix_col]
        vix_now = float(vix.iloc[-1])
        vix_below_20 = vix_now < 20
        vix_was_high = float(vix.rolling(10).max().iloc[-1]) > 25
        vix_declining = vix_now < float(vix.iloc[-6]) if len(vix) > 5 else False
        vix_calm = int(vix_below_20 or (vix_was_high and vix_declining))

    # 3. MOMENTUM: QQQ 20d return > 0
    qqq_ret_20d = float(close["QQQ"].pct_change(20).iloc[-1])
    momentum = int(qqq_ret_20d > 0) if not pd.isna(qqq_ret_20d) else 0

    # 4. VOLUME SURGE: any sector ETF with 5+ consecutive above-avg volume days
    vol_surge = 0
    for etf in SECTOR_ETFS:
        if etf in volume.columns:
            vol_avg = volume[etf].rolling(20).mean()
            above = volume[etf] > vol_avg
            streak = 0
            for i in range(len(above) - 1, max(len(above) - 20, -1), -1):
                if above.iloc[i]:
                    streak += 1
                else:
                    break
            if streak >= 5:
                vol_surge = 1
                break

    # 5. BREADTH: SPY 20d ret > 0 AND RSP keeping up
    spy_ret_20d = float(close["SPY"].pct_change(20).iloc[-1])
    rsp_ret_20d = float(close["RSP"].pct_change(20).iloc[-1]) if "RSP" in close.columns else 0
    breadth = int(spy_ret_20d > 0 and rsp_ret_20d > spy_ret_20d * 0.5) if not pd.isna(spy_ret_20d) else 0

    score = regime + vix_calm + momentum + vol_surge + breadth

    return {
        "regime": regime, "vix_calm": vix_calm, "momentum": momentum,
        "volume_surge": vol_surge, "breadth": breadth, "score": score,
        "vix": round(vix_now, 2) if vix_now else None,
        "spy": round(float(close["SPY"].iloc[-1]), 2),
        "qqq": round(float(close["QQQ"].iloc[-1]), 2),
        "sma200": round(float(sma200.iloc[-1]), 2) if not pd.isna(sma200.iloc[-1]) else None,
        "qqq_20d_ret": round(qqq_ret_20d * 100, 2) if not pd.isna(qqq_ret_20d) else None,
    }


def execute(state, signals):
    today = dt.datetime.now().strftime("%Y-%m-%d")
    score = signals["score"]
    should_long = score >= 3
    position = state["position"]

    state["signal_history"].append({
        "date": today, "score": score,
        "signals": {k: signals[k] for k in ["regime", "vix_calm", "momentum", "volume_surge", "breadth"]},
    })
    if len(state["signal_history"]) > 60:
        state["signal_history"] = state["signal_history"][-60:]

    # Close if going to cash
    if position and not should_long:
        exit_price = signals["qqq"] * (1 - SLIPPAGE_PCT)
        pnl = (exit_price - position["entry_price"]) * position["shares"]
        ret_pct = (exit_price / position["entry_price"] - 1) * 100
        state["capital"] = position["entry_price"] * position["shares"] + pnl
        state["trades"].append({
            "entry_date": position["entry_date"], "exit_date": today,
            "entry_price": position["entry_price"], "exit_price": round(exit_price, 2),
            "shares": position["shares"], "pnl": round(pnl, 2),
            "return_pct": round(ret_pct, 2), "exit_score": score,
        })
        log.info(f"CLOSED QQQ: {ret_pct:+.2f}% (${pnl:+.2f}) — score→{score}")
        state["position"] = None
        state["mode"] = "cash"

    # Open if going long
    if not state["position"] and should_long:
        entry_price = signals["qqq"] * (1 + SLIPPAGE_PCT)
        shares = state["capital"] / entry_price
        state["position"] = {
            "ticker": "QQQ", "shares": round(shares, 4),
            "entry_price": round(entry_price, 2), "entry_date": today,
        }
        state["capital"] = 0
        state["mode"] = "long"
        log.info(f"OPENED QQQ: {shares:.4f}sh @ ${entry_price:.2f} — score={score}")

    # Equity
    if state["position"]:
        p = state["position"]
        unrealized = (signals["qqq"] - p["entry_price"]) * p["shares"]
        equity = p["entry_price"] * p["shares"] + unrealized
    else:
        equity = state["capital"]

    state["daily_equity"].append({"date": today, "equity": round(equity, 2), "score": score})
    if len(state["daily_equity"]) > 90:
        state["daily_equity"] = state["daily_equity"][-90:]

    return state


def main():
    log.info("=" * 60)
    log.info("SIGNAL AGGREGATOR A — Paper Engine (CHAMPION)")

    state = load_state()

    try:
        close, volume = get_data()
    except Exception as e:
        log.error(f"Data fetch failed: {e}")
        save_state(state)
        return

    if len(close) < 200:
        log.warning(f"Only {len(close)}d — need 200. Skipping.")
        save_state(state)
        return

    signals = compute_signals(close, volume)

    sig_str = " | ".join(f"{k}={'✅' if v else '❌'}" for k, v in [
        ("Regime", signals["regime"]), ("VIX", signals["vix_calm"]),
        ("Mom", signals["momentum"]), ("Vol", signals["volume_surge"]),
        ("Breadth", signals["breadth"]),
    ])
    log.info(f"Signals: {sig_str} → Score {signals['score']}/5")
    log.info(f"  SPY ${signals['spy']} | QQQ ${signals['qqq']} | VIX {signals['vix']}")
    log.info(f"  → {'LONG QQQ' if signals['score'] >= 3 else 'CASH'}")

    state = execute(state, signals)

    equity = state["daily_equity"][-1]["equity"] if state["daily_equity"] else state["capital"]
    total_ret = (equity / STARTING_CAPITAL - 1) * 100
    n_closed = len(state["trades"])
    wins = sum(1 for t in state["trades"] if t["pnl"] > 0)
    wr = (wins / n_closed * 100) if n_closed > 0 else 0

    log.info(f"Equity: ${equity:.2f} ({total_ret:+.1f}%) | Trades: {n_closed} | WR: {wr:.0f}%")
    if state["position"]:
        p = state["position"]
        log.info(f"  Open: QQQ {p['shares']:.4f}sh @ ${p['entry_price']} (since {p['entry_date']})")
    else:
        log.info(f"  Cash: ${state['capital']:.2f}")

    save_state(state)
    log.info("Done.")


if __name__ == "__main__":
    main()
