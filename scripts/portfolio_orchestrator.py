#!/usr/bin/env python3
"""
Portfolio Orchestrator v1 — Unified Strategy → Execution Bridge
================================================================
Bridges validated strategies to Robinhood execution.

Architecture:
  1. Signal Layer:   Computes signals for all validated strategies
  2. Risk Layer:     Kill switch, position limits, drawdown circuit breaker
  3. Allocation:     Capital allocation across strategies
  4. Execution:      Generates orders for Robinhood (human-readable output)

Validated Strategies (adversarial-tested):
  - Signal Aggregator A: 5/5 gates + 5/5 adversarial (CHAMPION)
  - Strategy Rotation v2 F: 5/5 gates + 4/5 adversarial (BACKUP — correlated)
  - Extreme Idio C: 5/5 gates + 4/5 adversarial (BACKUP — bear-fragile)

Kill Switch Conditions:
  - VIX > 25 (elevated fear)
  - SPY below 50-SMA (short-term downtrend)
  - Both active → FULL STOP (100% cash)
  - One active → HALF SIZE

Run daily at 4:30 PM ET via cron.
"""

import os
import json
import logging
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────
STATE_DIR = Path("/home/jupiter/Lvl3Quant/state/portfolio_orchestrator")
STATE_FILE = STATE_DIR / "state.json"
LOG_FILE = STATE_DIR / "orchestrator.log"
SIGNAL_AGG_STATE = Path("/home/jupiter/Lvl3Quant/state/signal_aggregator_paper_state.json")
SCORECARD_FILE = Path("/home/jupiter/Lvl3Quant/output/asymmetric_scorecard/current_state.json")

STATE_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("orchestrator")

# ── Constants ──────────────────────────────────────────────────────────
ACCOUNT_NUMBER = os.environ.get("ROBINHOOD_ACCOUNT_NUMBER", "")
STARTING_CAPITAL = 667.73
MAX_POSITION_PCT = 0.95        # never invest >95% of capital
MIN_TRADE_AMOUNT = 5.0         # don't trade if order < $5
DRAWDOWN_CIRCUIT_BREAKER = -0.15  # pause at -15% from peak

SECTOR_ETFS = ["XLK", "XLC", "XLY", "XLE", "XLF", "XLV"]
ALL_TICKERS = ["SPY", "QQQ", "RSP", "^VIX", "^VIX3M"] + SECTOR_ETFS


# ═════════════════════════════════════════════════════════════════════════
# STATE MANAGEMENT
# ═════════════════════════════════════════════════════════════════════════

def load_state():
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {
        "created": datetime.now().isoformat(),
        "capital": STARTING_CAPITAL,
        "positions": [],       # [{ticker, shares, entry_price, entry_date, strategy}]
        "pending_orders": [],  # [{action, ticker, shares, reason}]
        "trade_history": [],
        "daily_equity": [],
        "peak_equity": STARTING_CAPITAL,
        "kill_switch": {"active": False, "reasons": []},
        "last_run": None,
        "mode": "paper",       # "paper" or "live" — start paper, user promotes to live
    }


def save_state(state):
    state["last_run"] = datetime.now().isoformat()
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


# ═════════════════════════════════════════════════════════════════════════
# DATA LAYER
# ═════════════════════════════════════════════════════════════════════════

def fetch_market_data():
    """Download 300 days of data for all needed tickers."""
    end = datetime.now()
    start = end - timedelta(days=400)
    data = yf.download(
        ALL_TICKERS, start=start, end=end,
        auto_adjust=True, progress=False, threads=True,
    )

    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"].copy()
        volume = data["Volume"].copy()
    else:
        close = data
        volume = pd.DataFrame()

    # Flatten any remaining MultiIndex
    if isinstance(close.columns, pd.MultiIndex):
        close.columns = close.columns.get_level_values(-1)
    if isinstance(volume.columns, pd.MultiIndex):
        volume.columns = volume.columns.get_level_values(-1)

    close = close.dropna(subset=["SPY"])
    volume = volume.reindex(close.index)
    return close, volume


# ═════════════════════════════════════════════════════════════════════════
# SIGNAL LAYER — Signal Aggregator A (5/5 adversarial champion)
# ═════════════════════════════════════════════════════════════════════════

def _rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def compute_signal_aggregator(close, volume):
    """
    Signal Aggregator A: 5 binary signals → composite score 0-5.
    Score ≥ 3 → long QQQ. Score < 3 → cash.
    """
    signals = {}

    # 1. REGIME: SPY > 200-SMA
    sma200 = close["SPY"].rolling(200).mean()
    regime = int(close["SPY"].iloc[-1] > sma200.iloc[-1]) if not pd.isna(sma200.iloc[-1]) else 0
    signals["regime"] = regime

    # 2. VIX CALM: VIX < 20 OR (was >25 recently and declining)
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
    signals["vix_calm"] = vix_calm

    # 3. MOMENTUM: QQQ 20d return > 0
    qqq_ret_20d = float(close["QQQ"].pct_change(20).iloc[-1])
    momentum = int(qqq_ret_20d > 0) if not pd.isna(qqq_ret_20d) else 0
    signals["momentum"] = momentum

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
    signals["volume_surge"] = vol_surge

    # 5. BREADTH: SPY 20d ret > 0 AND RSP keeping up (>50% of SPY ret)
    spy_ret_20d = float(close["SPY"].pct_change(20).iloc[-1])
    rsp_ret_20d = float(close["RSP"].pct_change(20).iloc[-1]) if "RSP" in close.columns else 0
    breadth = int(spy_ret_20d > 0 and rsp_ret_20d > spy_ret_20d * 0.5) if not pd.isna(spy_ret_20d) else 0
    signals["breadth"] = breadth

    signals["score"] = regime + vix_calm + momentum + vol_surge + breadth
    signals["action"] = "LONG_QQQ" if signals["score"] >= 3 else "CASH"

    # Context data
    signals["spy_price"] = round(float(close["SPY"].iloc[-1]), 2)
    signals["qqq_price"] = round(float(close["QQQ"].iloc[-1]), 2)
    signals["spy_200sma"] = round(float(sma200.iloc[-1]), 2) if not pd.isna(sma200.iloc[-1]) else None
    signals["spy_50sma"] = round(float(close["SPY"].rolling(50).mean().iloc[-1]), 2) if len(close) > 50 else None
    signals["vix"] = round(vix_now, 2) if vix_now else None
    signals["qqq_20d_ret_pct"] = round(qqq_ret_20d * 100, 2) if not pd.isna(qqq_ret_20d) else None

    return signals


# ═════════════════════════════════════════════════════════════════════════
# RISK LAYER
# ═════════════════════════════════════════════════════════════════════════

def evaluate_kill_switch(signals):
    """
    Kill switch conditions:
      - VIX > 25: elevated fear → FULL STOP
      - SPY below 50-SMA: short-term downtrend → HALF SIZE
      - Both → FULL STOP
      - Drawdown > 15% from peak → FULL STOP
    """
    reasons = []
    vix = signals.get("vix", 18)
    spy = signals.get("spy_price", 0)
    spy_50sma = signals.get("spy_50sma", 0)

    if vix and vix > 25:
        reasons.append(f"VIX elevated ({vix:.1f} > 25)")

    if spy and spy_50sma and spy < spy_50sma:
        reasons.append(f"SPY below 50-SMA ({spy:.2f} < {spy_50sma:.2f})")

    active = len(reasons) > 0
    severity = "FULL_STOP" if len(reasons) >= 2 or (vix and vix > 30) else ("HALF_SIZE" if reasons else "CLEAR")

    return {
        "active": active,
        "severity": severity,
        "reasons": reasons,
        "size_multiplier": 0.0 if severity == "FULL_STOP" else (0.5 if severity == "HALF_SIZE" else 1.0),
    }


def check_drawdown_circuit_breaker(state, current_equity):
    """Pause trading if equity drops >15% from peak."""
    peak = state.get("peak_equity", STARTING_CAPITAL)
    if current_equity > peak:
        state["peak_equity"] = current_equity
        peak = current_equity

    drawdown = (current_equity - peak) / peak
    if drawdown < DRAWDOWN_CIRCUIT_BREAKER:
        return True, drawdown
    return False, drawdown


# ═════════════════════════════════════════════════════════════════════════
# ALLOCATION LAYER
# ═════════════════════════════════════════════════════════════════════════

def compute_allocation(state, signals, kill_switch):
    """
    Determine target portfolio based on signals and risk constraints.
    Returns list of target positions and orders needed.
    """
    orders = []
    today = datetime.now().strftime("%Y-%m-%d")

    # Current equity
    current_equity = state["capital"]
    for pos in state["positions"]:
        if pos["ticker"] == "QQQ":
            current_equity += pos["shares"] * signals["qqq_price"]

    # Check drawdown circuit breaker
    dd_active, dd_pct = check_drawdown_circuit_breaker(state, current_equity)
    if dd_active:
        log.warning(f"DRAWDOWN CIRCUIT BREAKER ACTIVE: {dd_pct:.1%} from peak")
        # Close all positions
        for pos in state["positions"]:
            orders.append({
                "action": "SELL",
                "ticker": pos["ticker"],
                "shares": pos["shares"],
                "reason": f"Drawdown circuit breaker ({dd_pct:.1%})",
            })
        return orders, current_equity, dd_pct

    # Kill switch sizing
    size_mult = kill_switch["size_multiplier"]
    if size_mult == 0:
        # Full stop — close everything
        for pos in state["positions"]:
            orders.append({
                "action": "SELL",
                "ticker": pos["ticker"],
                "shares": pos["shares"],
                "reason": f"Kill switch: {', '.join(kill_switch['reasons'])}",
            })
        return orders, current_equity, dd_pct

    # Signal Aggregator A — primary strategy
    sig_action = signals["action"]
    target_invested = sig_action == "LONG_QQQ"
    currently_invested = any(p["ticker"] == "QQQ" for p in state["positions"])

    if target_invested and not currently_invested:
        # BUY QQQ
        invest_amount = current_equity * MAX_POSITION_PCT * size_mult
        if invest_amount >= MIN_TRADE_AMOUNT:
            qqq_price = signals["qqq_price"]
            # Robinhood supports fractional shares
            shares = round(invest_amount / qqq_price, 6)
            orders.append({
                "action": "BUY",
                "ticker": "QQQ",
                "shares": shares,
                "dollar_amount": round(invest_amount, 2),
                "strategy": "signal_aggregator_a",
                "reason": f"Score {signals['score']}/5 ≥ 3 → LONG QQQ",
                "signal_details": {
                    "regime": signals["regime"],
                    "vix_calm": signals["vix_calm"],
                    "momentum": signals["momentum"],
                    "volume_surge": signals["volume_surge"],
                    "breadth": signals["breadth"],
                },
            })

    elif not target_invested and currently_invested:
        # SELL QQQ
        for pos in state["positions"]:
            if pos["ticker"] == "QQQ":
                orders.append({
                    "action": "SELL",
                    "ticker": "QQQ",
                    "shares": pos["shares"],
                    "strategy": "signal_aggregator_a",
                    "reason": f"Score {signals['score']}/5 < 3 → CASH",
                })

    return orders, current_equity, dd_pct


# ═════════════════════════════════════════════════════════════════════════
# EXECUTION LAYER
# ═════════════════════════════════════════════════════════════════════════

def execute_paper(state, orders, signals):
    """Execute orders in paper mode — simulate fills at market price."""
    today = datetime.now().strftime("%Y-%m-%d")

    for order in orders:
        ticker = order["ticker"]
        price = signals.get(f"{ticker.lower()}_price", signals.get("qqq_price", 0))

        if order["action"] == "BUY":
            entry_price = price * 1.0002  # slippage
            shares = order["shares"]
            cost = entry_price * shares
            state["capital"] -= cost
            state["positions"].append({
                "ticker": ticker,
                "shares": shares,
                "entry_price": round(entry_price, 2),
                "entry_date": today,
                "strategy": order.get("strategy", "unknown"),
            })
            log.info(f"PAPER BUY {shares:.4f} {ticker} @ ${entry_price:.2f} = ${cost:.2f}")

        elif order["action"] == "SELL":
            exit_price = price * 0.9998  # slippage
            shares = order["shares"]
            # Find matching position
            for i, pos in enumerate(state["positions"]):
                if pos["ticker"] == ticker:
                    pnl = (exit_price - pos["entry_price"]) * shares
                    ret_pct = (exit_price / pos["entry_price"] - 1) * 100
                    state["capital"] += exit_price * shares
                    state["trade_history"].append({
                        "entry_date": pos["entry_date"],
                        "exit_date": today,
                        "ticker": ticker,
                        "shares": shares,
                        "entry_price": pos["entry_price"],
                        "exit_price": round(exit_price, 2),
                        "pnl": round(pnl, 2),
                        "return_pct": round(ret_pct, 2),
                        "strategy": pos.get("strategy", "unknown"),
                        "reason": order.get("reason", ""),
                    })
                    state["positions"].pop(i)
                    log.info(f"PAPER SELL {shares:.4f} {ticker} @ ${exit_price:.2f} = {ret_pct:+.2f}% (${pnl:+.2f})")
                    break

    return state


def generate_live_orders(orders, signals):
    """
    Generate human-readable order instructions for Robinhood execution.
    In live mode, these would be passed to the Robinhood MCP tools.
    """
    instructions = []
    for order in orders:
        ticker = order["ticker"]
        price = signals.get(f"{ticker.lower()}_price", signals.get("qqq_price", 0))

        if order["action"] == "BUY":
            instructions.append({
                "type": "market_buy",
                "ticker": ticker,
                "dollar_amount": order.get("dollar_amount", round(order["shares"] * price, 2)),
                "shares": order["shares"],
                "reason": order["reason"],
                "confidence": "HIGH" if signals.get("score", 0) >= 4 else "MEDIUM",
            })
        elif order["action"] == "SELL":
            instructions.append({
                "type": "market_sell",
                "ticker": ticker,
                "shares": order["shares"],
                "reason": order["reason"],
            })

    return instructions


# ═════════════════════════════════════════════════════════════════════════
# REPORTING
# ═════════════════════════════════════════════════════════════════════════

def compute_performance(state, current_equity):
    """Compute portfolio performance metrics."""
    trades = state.get("trade_history", [])
    equity_history = state.get("daily_equity", [])

    metrics = {
        "current_equity": round(current_equity, 2),
        "total_return_pct": round((current_equity / STARTING_CAPITAL - 1) * 100, 2),
        "total_trades": len(trades),
        "mode": state.get("mode", "paper"),
    }

    if trades:
        wins = [t for t in trades if t["pnl"] > 0]
        losses = [t for t in trades if t["pnl"] <= 0]
        metrics["win_rate"] = round(len(wins) / len(trades) * 100, 1) if trades else 0
        metrics["total_pnl"] = round(sum(t["pnl"] for t in trades), 2)
        metrics["avg_trade_pnl"] = round(metrics["total_pnl"] / len(trades), 2)
        gross_win = sum(t["pnl"] for t in wins) if wins else 0
        gross_loss = abs(sum(t["pnl"] for t in losses)) if losses else 0
        metrics["profit_factor"] = round(gross_win / gross_loss, 2) if gross_loss > 0 else float("inf")

    if len(equity_history) >= 5:
        eq_series = pd.Series([e["equity"] for e in equity_history])
        rets = eq_series.pct_change().dropna()
        if len(rets) > 0 and rets.std() > 0:
            metrics["sharpe"] = round((rets.mean() / rets.std()) * np.sqrt(252), 2)
            downside = rets[rets < 0].std()
            metrics["sortino"] = round((rets.mean() / downside) * np.sqrt(252), 2) if downside > 0 else None

    return metrics


def generate_report(state, signals, kill_switch, orders, current_equity, dd_pct):
    """Generate a clean status report."""
    today = datetime.now().strftime("%Y-%m-%d %H:%M")
    perf = compute_performance(state, current_equity)

    report = []
    report.append(f"{'='*60}")
    report.append(f"PORTFOLIO ORCHESTRATOR — {today}")
    report.append(f"{'='*60}")
    report.append(f"")
    report.append(f"ACCOUNT: ${current_equity:.2f} ({perf['total_return_pct']:+.2f}% total)")
    report.append(f"MODE: {state.get('mode', 'paper').upper()}")
    report.append(f"")

    # Kill switch
    if kill_switch["active"]:
        report.append(f"⚠️  KILL SWITCH: {kill_switch['severity']}")
        for r in kill_switch["reasons"]:
            report.append(f"    → {r}")
    else:
        report.append(f"✅ Kill switch: CLEAR")
    report.append(f"")

    # Market
    report.append(f"MARKET:")
    report.append(f"  SPY: ${signals['spy_price']}  (200-SMA: ${signals.get('spy_200sma', '?')})")
    report.append(f"  QQQ: ${signals['qqq_price']}  (20d ret: {signals.get('qqq_20d_ret_pct', '?')}%)")
    report.append(f"  VIX: {signals.get('vix', '?')}")
    report.append(f"")

    # Signal Aggregator
    report.append(f"SIGNAL AGGREGATOR A (score {signals['score']}/5 → {signals['action']}):")
    for s in ["regime", "vix_calm", "momentum", "volume_surge", "breadth"]:
        icon = "🟢" if signals[s] else "🔴"
        report.append(f"  {icon} {s}: {signals[s]}")
    report.append(f"")

    # Positions
    report.append(f"POSITIONS:")
    if state["positions"]:
        for pos in state["positions"]:
            price = signals.get("qqq_price", 0)
            unrealized = (price - pos["entry_price"]) * pos["shares"]
            ret = (price / pos["entry_price"] - 1) * 100
            report.append(f"  {pos['ticker']}: {pos['shares']:.4f}sh @ ${pos['entry_price']:.2f} → ${price:.2f} ({ret:+.2f}%, ${unrealized:+.2f})")
    else:
        report.append(f"  [ALL CASH]")
    report.append(f"")

    # Orders
    if orders:
        report.append(f"ORDERS:")
        for o in orders:
            report.append(f"  {o['action']} {o.get('shares', '?')} {o['ticker']} — {o.get('reason', '')}")
    else:
        report.append(f"ORDERS: None (no action needed)")
    report.append(f"")

    # Performance
    if perf.get("total_trades", 0) > 0:
        report.append(f"PERFORMANCE ({perf['total_trades']} trades):")
        report.append(f"  Win Rate: {perf.get('win_rate', '?')}%")
        report.append(f"  Profit Factor: {perf.get('profit_factor', '?')}")
        report.append(f"  Sharpe: {perf.get('sharpe', 'N/A')}")
        report.append(f"  Drawdown: {dd_pct:.1%}")
    report.append(f"{'='*60}")

    return "\n".join(report)


# ═════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════

def main():
    log.info("=" * 60)
    log.info("PORTFOLIO ORCHESTRATOR v1 — Starting")

    state = load_state()

    # Fetch data
    try:
        close, volume = fetch_market_data()
        log.info(f"Market data: {len(close)} days ({close.index[-1].date()})")
    except Exception as e:
        log.error(f"Data fetch failed: {e}")
        save_state(state)
        return

    if len(close) < 200:
        log.warning(f"Only {len(close)} days of data — need 200. Skipping.")
        save_state(state)
        return

    # Compute signals
    signals = compute_signal_aggregator(close, volume)
    log.info(f"Signal Agg A: score={signals['score']}/5 → {signals['action']}")

    # Kill switch
    kill_switch = evaluate_kill_switch(signals)
    state["kill_switch"] = kill_switch
    if kill_switch["active"]:
        log.warning(f"Kill switch {kill_switch['severity']}: {kill_switch['reasons']}")

    # Allocation
    orders, current_equity, dd_pct = compute_allocation(state, signals, kill_switch)

    # Execute (paper mode)
    if orders:
        if state.get("mode", "paper") == "paper":
            state = execute_paper(state, orders, signals)
        else:
            live_orders = generate_live_orders(orders, signals)
            log.info(f"LIVE ORDERS (not auto-executing): {json.dumps(live_orders, indent=2)}")

    # Update equity tracking
    today = datetime.now().strftime("%Y-%m-%d")
    state["daily_equity"].append({
        "date": today,
        "equity": round(current_equity, 2),
        "score": signals["score"],
        "kill_switch": kill_switch["severity"],
    })
    if len(state["daily_equity"]) > 365:
        state["daily_equity"] = state["daily_equity"][-365:]

    # Report
    report = generate_report(state, signals, kill_switch, orders, current_equity, dd_pct)
    print(report)

    # Save
    save_state(state)
    log.info("State saved. Done.")

    return {
        "signals": signals,
        "kill_switch": kill_switch,
        "orders": orders,
        "equity": current_equity,
        "report": report,
    }


if __name__ == "__main__":
    result = main()
