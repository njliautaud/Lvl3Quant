#!/usr/bin/env python3
"""Earnings Surprise Momentum Scanner — daily live scanner for RH agentic account.

Run daily during earnings season:
    python3 scripts/earnings_momentum_scanner.py

Detects post-earnings gap-ups >3%, generates BUY/EXIT signals,
tracks positions, and enforces a VIX+SPY kill switch.
"""

import json, os, sys, datetime as dt
import yfinance as yf
import numpy as np

STATE_PATH = "/home/jupiter/Lvl3Quant/state/earnings_momentum_state.json"
ACCOUNT_CASH = 670.0
MAX_PER_POSITION = 200.0
ALLOC_FRAC = 0.33
MAX_CONCURRENT = 3
HOLD_DAYS = 40
GAP_THRESHOLD = 0.03  # 3%

UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD",
    "NFLX", "CRM", "PLTR", "SOFI", "HOOD", "SNAP", "PINS", "COIN",
    "RBLX", "UBER", "LYFT", "DDOG", "TTD", "SHOP", "NET", "ROKU",
    "RDDT", "RIVN", "MSTR", "FSLR", "MPWR", "DXCM", "ILMN", "SYK",
    "REGN", "BMY", "CROX", "ABBV", "MA", "KKR",
]

# ── State management ──────────────────────────────────────────────

def load_state():
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            return json.load(f)
    return {"positions": [], "history": []}

def save_state(state):
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2, default=lambda x: int(x) if hasattr(x, 'item') else float(x) if hasattr(x, 'is_integer') else str(x))

# ── Kill switch ───────────────────────────────────────────────────

def check_kill_switch():
    """Return (killed: bool, vix: float, spy_price: float, spy_sma50: float)."""
    try:
        vix = yf.Ticker("^VIX").history(period="2d")
        vix_val = float(vix["Close"].iloc[-1]) if len(vix) else 0.0
    except Exception:
        vix_val = 0.0

    try:
        spy = yf.Ticker("SPY").history(period="70d")
        spy_price = float(spy["Close"].iloc[-1]) if len(spy) else 0.0
        spy_sma50 = float(spy["Close"].tail(50).mean()) if len(spy) >= 50 else spy_price
    except Exception:
        spy_price, spy_sma50 = 0.0, 0.0

    killed = vix_val > 20 and spy_price < spy_sma50
    return killed, vix_val, spy_price, spy_sma50

# ── Gap detection ─────────────────────────────────────────────────

def detect_gaps(lookback_days=5):
    """Scan universe for >3% gap-ups in last 2 trading days."""
    signals = []
    for sym in UNIVERSE:
        try:
            hist = yf.Ticker(sym).history(period=f"{lookback_days}d")
            if len(hist) < 3:
                continue
            # Check last 2 trading days for gap
            for i in range(-2, 0):
                if abs(i) >= len(hist):
                    continue
                prev_close = float(hist["Close"].iloc[i - 1])
                day_open = float(hist["Open"].iloc[i])
                if prev_close <= 0:
                    continue
                gap_pct = (day_open - prev_close) / prev_close
                if gap_pct >= GAP_THRESHOLD:
                    # Check 20d momentum (price vs 20d ago)
                    long_hist = yf.Ticker(sym).history(period="30d")
                    mom_20d = 0.0
                    if len(long_hist) >= 20:
                        mom_20d = (float(long_hist["Close"].iloc[-1]) - float(long_hist["Close"].iloc[-20])) / float(long_hist["Close"].iloc[-20])

                    gap_date = str(hist.index[i].date())
                    current_price = float(hist["Close"].iloc[-1])
                    strength = "STRONG BUY" if gap_pct > GAP_THRESHOLD and mom_20d > 0 else "BUY"
                    signals.append({
                        "symbol": sym,
                        "gap_pct": round(gap_pct * 100, 2),
                        "gap_date": gap_date,
                        "current_price": round(current_price, 2),
                        "mom_20d": round(mom_20d * 100, 2),
                        "strength": strength,
                    })
                    break  # one signal per ticker
        except Exception as e:
            print(f"  WARN: {sym} fetch failed: {e}")
    return signals

# ── Position sizing ───────────────────────────────────────────────

def size_position(price, cash_available):
    budget = min(cash_available * ALLOC_FRAC, MAX_PER_POSITION)
    shares = int(budget // price) if price > 0 else 0
    # Allow 1 share if price fits in cash even if above per-position budget
    if shares == 0 and 0 < price <= cash_available:
        shares = 1
    return max(shares, 0)

# ── Main ──────────────────────────────────────────────────────────

def main():
    today = dt.date.today().isoformat()
    state = load_state()
    positions = state.get("positions", [])

    print(f"={'=' * 58}")
    print(f"  EARNINGS MOMENTUM SCANNER  |  {today}")
    print(f"={'=' * 58}\n")

    # ── Kill switch ──
    killed, vix, spy, sma50 = check_kill_switch()
    status = "ACTIVE (entries blocked)" if killed else "OFF (entries allowed)"
    print(f"KILL SWITCH: {status}")
    print(f"  VIX={vix:.1f}  SPY={spy:.2f}  SMA50={sma50:.2f}\n")

    # ── Check exits on existing positions ──
    exits = []
    remaining = []
    for pos in positions:
        entry_date = dt.date.fromisoformat(pos["entry_date"])
        days_held = np.busday_count(entry_date, dt.date.today())
        if days_held >= HOLD_DAYS:
            try:
                cur = float(yf.Ticker(pos["symbol"]).history(period="2d")["Close"].iloc[-1])
            except Exception:
                cur = pos["entry_price"]
            pnl_pct = (cur - pos["entry_price"]) / pos["entry_price"] * 100
            exits.append({**pos, "exit_price": round(cur, 2), "pnl_pct": round(pnl_pct, 2), "days_held": days_held})
        else:
            pos["days_remaining"] = HOLD_DAYS - days_held
            remaining.append(pos)

    # ── Print exits ──
    if exits:
        print("EXIT SIGNALS (40-day hold reached):")
        for e in exits:
            print(f"  SELL {e['symbol']}  entry=${e['entry_price']}  now=${e['exit_price']}  P&L={e['pnl_pct']:+.1f}%  held {e['days_held']}d")
            state.setdefault("history", []).append({**e, "exit_date": today})
        print()

    # ── Print open positions ──
    if remaining:
        print(f"OPEN POSITIONS ({len(remaining)}/{MAX_CONCURRENT}):")
        for p in remaining:
            try:
                cur = float(yf.Ticker(p["symbol"]).history(period="2d")["Close"].iloc[-1])
            except Exception:
                cur = p["entry_price"]
            pnl = (cur - p["entry_price"]) / p["entry_price"] * 100
            print(f"  {p['symbol']:6s}  {p['shares']} sh @ ${p['entry_price']:.2f}  now ${cur:.2f}  {pnl:+.1f}%  {p['days_remaining']}d left")
        print()

    # ── Scan for new signals ──
    print("Scanning for earnings gaps (>3%) in last 2 trading days...")
    signals = detect_gaps()

    # Filter out symbols we already hold
    held_syms = {p["symbol"] for p in remaining}
    signals = [s for s in signals if s["symbol"] not in held_syms]

    if not signals:
        print("  No new earnings gap signals found.\n")
    else:
        print(f"\n  SIGNALS DETECTED:")
        for s in signals:
            print(f"    {s['strength']:10s}  {s['symbol']:6s}  gap={s['gap_pct']:+.1f}%  20d_mom={s['mom_20d']:+.1f}%  price=${s['current_price']:.2f}")

    # ── Generate entries (if kill switch off and slots available) ──
    slots = MAX_CONCURRENT - len(remaining)
    cash = ACCOUNT_CASH - sum(p["entry_price"] * p["shares"] for p in remaining)

    new_entries = []
    if killed:
        print("\n  Kill switch ACTIVE — no new entries.\n")
    elif slots <= 0:
        print(f"\n  Max {MAX_CONCURRENT} concurrent positions reached — no new entries.\n")
    elif signals:
        # Prioritize STRONG BUY, then largest gap
        signals.sort(key=lambda s: (s["strength"] != "STRONG BUY", -s["gap_pct"]))
        for s in signals[:slots]:
            shares = size_position(s["current_price"], cash)
            if shares == 0:
                continue
            cost = shares * s["current_price"]
            entry = {
                "symbol": s["symbol"],
                "shares": shares,
                "entry_price": s["current_price"],
                "entry_date": today,
                "gap_pct": s["gap_pct"],
                "strength": s["strength"],
            }
            new_entries.append(entry)
            remaining.append({**entry, "days_remaining": HOLD_DAYS})
            cash -= cost

        if new_entries:
            print(f"\n  NEW ENTRIES:")
            for e in new_entries:
                print(f"    BUY {e['shares']} {e['symbol']} @ ${e['entry_price']:.2f} (${e['shares'] * e['entry_price']:.0f})")
            print()

    # ── Track missed signals (blocked by BP or slots) ──
    if signals and (killed or slots <= 0 or not new_entries):
        missed = state.get("missed_signals", [])
        for s in signals:
            if s["symbol"] not in {e["symbol"] for e in new_entries}:
                missed.append({**s, "scan_date": today, "reason": "kill_switch" if killed else "max_slots" if slots <= 0 else "no_bp"})
        state["missed_signals"] = missed[-50:]  # keep last 50

    # ── Save state ──
    state["positions"] = remaining
    state["last_scan"] = today
    state["kill_switch"] = {"active": killed, "vix": vix, "spy": spy, "sma50": sma50}
    save_state(state)
    print(f"State saved to {STATE_PATH}")

if __name__ == "__main__":
    main()
