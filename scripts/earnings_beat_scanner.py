#!/usr/bin/env python3
"""
Earnings Beat Scanner - Daily Signal Generator

Strategy: Buy shares + sector ETF after earnings beats (gap up >3%),
hold ~60 trading days. Validated Sharpe 1.5-1.8, perm p=0.001.

Kill switch: VIX > 20 AND SPY < 50-SMA -> no entries.
Either condition alone -> reduced size (50%).

Usage:
    python3 scripts/earnings_beat_scanner.py
    python3 scripts/earnings_beat_scanner.py --account-value 1000
    python3 scripts/earnings_beat_scanner.py --dry-run
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

try:
    import yfinance as yf
except ImportError:
    print("ERROR: yfinance not installed. Run: pip install yfinance")
    sys.exit(1)

# ── Logging ───────────────────────────────────────────────────────────────

LOG_DIR = "/home/jupiter/Lvl3Quant/logs"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "earnings_beat_scanner.log")),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("earnings_beat_scanner")


# ── Configuration ─────────────────────────────────────────────────────────

UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD",
    "NFLX", "CRM", "PLTR", "SOFI", "HOOD", "SNAP", "PINS", "UBER",
    "LYFT", "COIN", "RBLX", "DDOG", "TTD", "SHOP", "NET", "ROKU",
]

# Stock -> Sector ETF mapping
SECTOR_MAP = {
    # Tech -> XLK
    "AAPL": "XLK", "MSFT": "XLK", "NVDA": "XLK", "AMD": "XLK",
    "CRM": "XLK", "DDOG": "XLK", "NET": "XLK", "TTD": "XLK",
    "SHOP": "XLK", "PLTR": "XLK",
    # Communications -> XLC
    "GOOGL": "XLC", "META": "XLC", "SNAP": "XLC", "PINS": "XLC",
    "RBLX": "XLC", "ROKU": "XLC", "NFLX": "XLC",
    # Consumer Discretionary -> XLY
    "AMZN": "XLY", "TSLA": "XLY", "UBER": "XLY", "LYFT": "XLY",
    # Financials -> XLF
    "SOFI": "XLF", "HOOD": "XLF", "COIN": "XLF",
}

TOTAL_CAPITAL = 645.0             # Total capital allocated to strategy
MAX_CONCURRENT = 5                # Max concurrent position pairs
GAP_THRESHOLD_PCT = 3.0           # Minimum gap-up % to qualify
HOLD_DAYS = 60                    # Trading days to hold
LOOKBACK_DAYS = 5                 # Calendar days to check for recent gaps
VIX_KILL_LEVEL = 20.0
CAUTION_SIZE_MULT = 0.50

STATE_FILE = "/home/jupiter/Lvl3Quant/state/earnings_beat_scanner.json"


# ── Sector ETF price cache ────────────────────────────────────────────────

_etf_price_cache: dict = {}


def get_etf_price(etf_ticker: str) -> float | None:
    """Get current price for a sector ETF, with caching."""
    if etf_ticker in _etf_price_cache:
        return _etf_price_cache[etf_ticker]
    try:
        t = yf.Ticker(etf_ticker)
        hist = t.history(period="5d")
        if not hist.empty:
            price = round(float(hist["Close"].iloc[-1]), 2)
            _etf_price_cache[etf_ticker] = price
            return price
    except Exception as e:
        log.warning("Failed to fetch %s price: %s", etf_ticker, e)
    return None


# ── Market Regime ─────────────────────────────────────────────────────────

def get_market_regime() -> dict:
    """Check VIX level and SPY vs 50-day SMA for kill switch."""
    regime = {
        "vix_level": None,
        "vix_above_threshold": False,
        "spy_price": None,
        "spy_sma50": None,
        "spy_below_sma50": False,
        "kill_switch_active": False,
        "caution": False,
        "size_multiplier": 1.0,
    }
    try:
        vix = yf.Ticker("^VIX")
        vix_hist = vix.history(period="5d")
        if not vix_hist.empty:
            regime["vix_level"] = round(float(vix_hist["Close"].iloc[-1]), 2)
            regime["vix_above_threshold"] = regime["vix_level"] > VIX_KILL_LEVEL

        spy = yf.Ticker("SPY")
        spy_hist = spy.history(period="75d")
        if len(spy_hist) >= 50:
            regime["spy_price"] = round(float(spy_hist["Close"].iloc[-1]), 2)
            regime["spy_sma50"] = round(float(spy_hist["Close"].tail(50).mean()), 2)
            regime["spy_below_sma50"] = regime["spy_price"] < regime["spy_sma50"]

        if regime["vix_above_threshold"] and regime["spy_below_sma50"]:
            regime["kill_switch_active"] = True
            regime["size_multiplier"] = 0.0
        elif regime["vix_above_threshold"] or regime["spy_below_sma50"]:
            regime["caution"] = True
            regime["size_multiplier"] = CAUTION_SIZE_MULT
    except Exception as e:
        log.warning("Could not fetch market regime data: %s", e)
    return regime


# ── Earnings Gap Detection ────────────────────────────────────────────────

def check_earnings_gap(ticker: str) -> dict | None:
    """
    Check if a stock had a gap-up >3% in the last few trading days.
    Uses price gap detection as earnings-beat proxy.
    Also tries to confirm via yfinance earnings dates.
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="10d")
        if len(hist) < 3:
            return None

        # Try to confirm recent earnings via yfinance
        had_recent_earnings = False
        earnings_date_str = None

        try:
            edates = stock.earnings_dates
            if edates is not None and not edates.empty:
                now = datetime.now()
                cutoff = now - timedelta(days=LOOKBACK_DAYS)
                for idx in edates.index:
                    dt = idx.to_pydatetime().replace(tzinfo=None) if hasattr(idx, "to_pydatetime") else idx
                    if cutoff <= dt <= now:
                        earnings_date_str = str(dt.date())
                        had_recent_earnings = True
                        break
        except Exception:
            pass

        # Scan last 3 trading days for qualifying gaps
        best_gap = None
        for i in range(1, min(4, len(hist))):
            prev_close = float(hist["Close"].iloc[-(i + 1)])
            day_open = float(hist["Open"].iloc[-i])
            day_close = float(hist["Close"].iloc[-i])
            gap_date = hist.index[-i]

            gap_pct = ((day_open - prev_close) / prev_close) * 100

            if gap_pct >= GAP_THRESHOLD_PCT:
                if best_gap is None or gap_pct > best_gap["gap_pct"]:
                    best_gap = {
                        "ticker": ticker,
                        "gap_date": str(gap_date.date()) if hasattr(gap_date, "date") else str(gap_date)[:10],
                        "prev_close": round(prev_close, 2),
                        "open_price": round(day_open, 2),
                        "close_price": round(day_close, 2),
                        "gap_pct": round(gap_pct, 2),
                        "current_price": round(float(hist["Close"].iloc[-1]), 2),
                        "confirmed_earnings": had_recent_earnings,
                        "earnings_date": earnings_date_str,
                        "sector_etf": SECTOR_MAP.get(ticker, "XLK"),
                    }
        return best_gap

    except Exception as e:
        log.warning("Error checking %s: %s", ticker, e)
        return None


# ── Next Expected Earnings ────────────────────────────────────────────────

def get_next_earnings_dates() -> list[dict]:
    """Get upcoming earnings dates for universe stocks."""
    upcoming = []
    now = datetime.now()
    for ticker in UNIVERSE:
        try:
            stock = yf.Ticker(ticker)
            edates = stock.earnings_dates
            if edates is None or edates.empty:
                continue
            for idx in edates.index:
                dt = idx.to_pydatetime().replace(tzinfo=None) if hasattr(idx, "to_pydatetime") else idx
                if dt > now:
                    upcoming.append({
                        "ticker": ticker,
                        "earnings_date": str(dt.date()),
                        "days_away": (dt.date() - now.date()).days,
                    })
                    break  # only next date per ticker
        except Exception:
            continue
    upcoming.sort(key=lambda x: x["days_away"])
    return upcoming


# ── State Management ──────────────────────────────────────────────────────

def load_state() -> dict:
    """Load scanner state from disk."""
    try:
        with open(STATE_FILE, "r") as f:
            state = json.load(f)
            # Ensure required keys
            state.setdefault("active_positions", [])
            state.setdefault("signal_history", [])
            state.setdefault("last_scan_date", None)
            return state
    except (FileNotFoundError, json.JSONDecodeError):
        return {"active_positions": [], "signal_history": [], "last_scan_date": None}


def save_state(state: dict, dry_run: bool = False):
    """Save scanner state to disk."""
    if dry_run:
        log.info("[DRY RUN] State not saved.")
        return
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)
    log.info("State saved.")


def get_active_positions(state: dict) -> list[dict]:
    """Return positions that haven't reached exit date."""
    today = datetime.now().date()
    return [
        p for p in state.get("active_positions", [])
        if datetime.strptime(p["target_exit_date"], "%Y-%m-%d").date() > today
    ]


def compute_exit_date(entry_date_str: str) -> str:
    """60 trading days ~ 84 calendar days."""
    entry = datetime.strptime(entry_date_str, "%Y-%m-%d")
    calendar_days = int(HOLD_DAYS * 7 / 5)
    return (entry + timedelta(days=calendar_days)).strftime("%Y-%m-%d")


def compute_days_remaining(exit_date_str: str) -> int:
    """Trading days remaining (approximate)."""
    exit_dt = datetime.strptime(exit_date_str, "%Y-%m-%d").date()
    today = datetime.now().date()
    cal_days = (exit_dt - today).days
    return max(0, int(cal_days * 5 / 7))


# ── Position P&L ──────────────────────────────────────────────────────────

def update_position_pnl(positions: list[dict]) -> list[dict]:
    """Fetch current prices and compute P&L for active positions."""
    for pos in positions:
        try:
            ticker = pos["ticker"]
            stock = yf.Ticker(ticker)
            hist = stock.history(period="2d")
            if not hist.empty:
                current = round(float(hist["Close"].iloc[-1]), 2)
                entry = pos["entry_price"]
                pnl_pct = round(((current - entry) / entry) * 100, 2)
                pnl_dollar = round((current - entry) * pos.get("shares", 0), 2)
                pos["current_price"] = current
                pos["pnl_pct"] = pnl_pct
                pos["pnl_dollar"] = pnl_dollar
                pos["days_remaining"] = compute_days_remaining(pos["target_exit_date"])
        except Exception as e:
            log.warning("Could not update P&L for %s: %s", pos.get("ticker"), e)
    return positions


# ── Main Scanner ──────────────────────────────────────────────────────────

def run_scanner(account_value: float = TOTAL_CAPITAL, dry_run: bool = False):
    """Main scanner logic. Returns (signals, regime)."""

    print("=" * 70)
    print(f"  EARNINGS BEAT SCANNER -- {datetime.now().strftime('%Y-%m-%d %H:%M ET')}")
    print("=" * 70)
    print()

    # ── 1. Market regime / kill switch ──
    log.info("Checking market regime...")
    regime = get_market_regime()

    print(f"  VIX:  {regime['vix_level']}  {'ELEVATED' if regime['vix_above_threshold'] else 'OK'}")
    print(f"  SPY:  {regime['spy_price']} vs 50-SMA {regime['spy_sma50']}  "
          f"{'BELOW' if regime['spy_below_sma50'] else 'ABOVE'}")

    if regime["kill_switch_active"]:
        print()
        print("  *** KILL SWITCH ACTIVE -- VIX > 20 AND SPY < 50-SMA ***")
        print("  *** NO NEW ENTRIES ***")
    elif regime["caution"]:
        print(f"  CAUTION -- one condition active, size reduced to {int(regime['size_multiplier']*100)}%")
    else:
        print("  Regime: CLEAR for entries")
    print()

    # ── 2. Load state, show active positions with P&L ──
    state = load_state()
    active = get_active_positions(state)

    # Position size: $645 / (2 * max_concurrent) -- half stock, half ETF
    per_leg_size = account_value / (2 * MAX_CONCURRENT)

    print(f"  Active positions: {len(active)} / {MAX_CONCURRENT} pairs")
    print(f"  Per-leg size: ${per_leg_size:.0f} (stock) + ${per_leg_size:.0f} (sector ETF)")
    print()

    if active:
        log.info("Updating P&L for %d active positions...", len(active))
        active = update_position_pnl(active)

        print("  ACTIVE POSITIONS:")
        print("  " + "-" * 66)
        print(f"  {'Ticker':<8} {'Entry':>8} {'Current':>8} {'P&L%':>7} {'P&L$':>8} {'Days Left':>10} {'Exit By':<12}")
        print("  " + "-" * 66)

        total_pnl = 0.0
        for p in active:
            pnl_pct = p.get("pnl_pct", 0)
            pnl_dollar = p.get("pnl_dollar", 0)
            total_pnl += pnl_dollar
            days_rem = p.get("days_remaining", "?")
            marker = "+" if pnl_pct >= 0 else ""
            etf_note = f" ({p.get('sector_etf', '')})" if p.get("is_etf_leg") else ""
            print(f"  {p['ticker']:<8} {p['entry_price']:>8.2f} {p.get('current_price', 0):>8.2f} "
                  f"{marker}{pnl_pct:>6.1f}% {pnl_dollar:>+7.2f} {days_rem:>10} {p['target_exit_date']:<12}{etf_note}")

        print("  " + "-" * 66)
        print(f"  Total unrealized P&L: ${total_pnl:+.2f}")
        print()

    open_slots = MAX_CONCURRENT - len([p for p in active if not p.get("is_etf_leg")])
    # Count only stock legs for slot counting (each pair = 1 slot)
    stock_positions = [p for p in active if not p.get("is_etf_leg")]
    open_slots = MAX_CONCURRENT - len(stock_positions)

    if open_slots <= 0:
        print("  No open slots for new pairs.")
    else:
        print(f"  Open slots: {open_slots}")
    print()

    # ── 3. Scan universe for earnings gaps ──
    log.info("Scanning %d stocks for earnings gaps (>%.0f%%)...", len(UNIVERSE), GAP_THRESHOLD_PCT)
    print(f"  Scanning {len(UNIVERSE)} stocks for gaps > {GAP_THRESHOLD_PCT}%...")
    print()

    candidates = []
    already_holding = {p["ticker"] for p in active}

    for ticker in UNIVERSE:
        if ticker in already_holding:
            log.info("  %s: SKIP (already holding)", ticker)
            continue

        gap_info = check_earnings_gap(ticker)
        if gap_info is None:
            log.info("  %s: no qualifying gap", ticker)
            continue

        label = f"GAP +{gap_info['gap_pct']}%"
        if gap_info["confirmed_earnings"]:
            label += " (earnings confirmed)"
        label += f" -> {gap_info['sector_etf']}"

        print(f"    {ticker}: {label}")
        candidates.append(gap_info)

    candidates.sort(key=lambda x: x["gap_pct"], reverse=True)
    print()

    # ── 4. Generate signals ──
    signals = []
    new_positions = []

    for cand in candidates:
        ticker = cand["ticker"]
        etf = cand["sector_etf"]

        if regime["kill_switch_active"]:
            signals.append({
                "ticker": ticker, "signal": "BLOCKED",
                "gap_pct": cand["gap_pct"],
                "reason": "Kill switch active (VIX > 20 AND SPY < 50-SMA)",
            })
            continue

        if open_slots <= 0:
            signals.append({
                "ticker": ticker, "signal": "SKIP",
                "gap_pct": cand["gap_pct"],
                "reason": "No open position slots",
            })
            continue

        # Position sizing: $645 / (2 * max_concurrent) per leg
        stock_alloc = per_leg_size * regime["size_multiplier"]
        etf_alloc = per_leg_size * regime["size_multiplier"]

        if stock_alloc < 10:
            signals.append({
                "ticker": ticker, "signal": "SKIP",
                "gap_pct": cand["gap_pct"],
                "reason": "Position size too small after regime adjustment",
            })
            continue

        stock_price = cand["current_price"]
        stock_shares = max(1, int(stock_alloc / stock_price))
        stock_cost = round(stock_shares * stock_price, 2)

        etf_price = get_etf_price(etf)
        etf_shares = 0
        etf_cost = 0.0
        if etf_price and etf_price > 0:
            etf_shares = max(1, int(etf_alloc / etf_price))
            etf_cost = round(etf_shares * etf_price, 2)

        entry_date = cand["gap_date"]
        exit_date = compute_exit_date(entry_date)

        signal = {
            "ticker": ticker,
            "signal": "ENTER_PAIR",
            "gap_pct": cand["gap_pct"],
            "gap_date": entry_date,
            "confirmed_earnings": cand["confirmed_earnings"],
            "stock": {
                "ticker": ticker,
                "price": stock_price,
                "shares": stock_shares,
                "cost": stock_cost,
            },
            "sector_etf": {
                "ticker": etf,
                "price": etf_price,
                "shares": etf_shares,
                "cost": etf_cost,
            },
            "total_cost": round(stock_cost + etf_cost, 2),
            "entry_date": entry_date,
            "target_exit_date": exit_date,
            "regime_note": "CAUTION -- reduced size" if regime["caution"] else "normal",
        }
        signals.append(signal)

        # Build position records for state
        new_positions.append({
            "ticker": ticker,
            "entry_date": entry_date,
            "entry_price": stock_price,
            "shares": stock_shares,
            "position_value": stock_cost,
            "target_exit_date": exit_date,
            "sector_etf": etf,
            "gap_pct": cand["gap_pct"],
            "is_etf_leg": False,
        })
        if etf_price:
            new_positions.append({
                "ticker": etf,
                "entry_date": entry_date,
                "entry_price": etf_price,
                "shares": etf_shares,
                "position_value": etf_cost,
                "target_exit_date": exit_date,
                "sector_etf": etf,
                "gap_pct": cand["gap_pct"],
                "is_etf_leg": True,
                "parent_stock": ticker,
            })

        open_slots -= 1

    # ── Print signal summary ──
    print("-" * 70)

    enter_signals = [s for s in signals if s["signal"] == "ENTER_PAIR"]
    other_signals = [s for s in signals if s["signal"] != "ENTER_PAIR"]

    if enter_signals:
        print(f"\n  NEW ENTRY SIGNALS ({len(enter_signals)} pairs):")
        print()
        for s in enter_signals:
            stk = s["stock"]
            etf = s["sector_etf"]
            print(f"  >>> BUY PAIR: {stk['ticker']} + {etf['ticker']}")
            print(f"      Earnings gap: +{s['gap_pct']}% on {s['gap_date']}"
                  f"{'  (confirmed)' if s['confirmed_earnings'] else '  (gap-based)'}")
            print(f"      Stock:  {stk['shares']} x {stk['ticker']} @ ${stk['price']:.2f} = ${stk['cost']:.2f}")
            if etf["price"]:
                print(f"      ETF:    {etf['shares']} x {etf['ticker']} @ ${etf['price']:.2f} = ${etf['cost']:.2f}")
            else:
                print(f"      ETF:    {etf['ticker']} -- price unavailable, skip ETF leg")
            print(f"      Total:  ${s['total_cost']:.2f}")
            print(f"      Hold:   ~{HOLD_DAYS} trading days -> exit by {s['target_exit_date']}")
            if s["regime_note"] != "normal":
                print(f"      Note:   {s['regime_note']}")
            print()
    else:
        print("\n  NO NEW ENTRY SIGNALS TODAY")
        if regime["kill_switch_active"]:
            print("  Reason: Kill switch is active")
        elif not candidates:
            print("  Reason: No qualifying earnings gaps in universe")
        elif open_slots <= 0:
            print("  Reason: All position slots full")
        print()

    if other_signals:
        print(f"  Blocked/Skipped ({len(other_signals)}):")
        for s in other_signals:
            print(f"    {s['ticker']}: {s['signal']} -- {s.get('reason', '')}")
        print()

    print("-" * 70)
    print()

    # ── 5. Next expected earnings reporters ──
    log.info("Fetching next expected earnings dates...")
    print("  NEXT EXPECTED EARNINGS (our universe):")
    upcoming = get_next_earnings_dates()
    if upcoming:
        for u in upcoming[:10]:  # Show next 10
            etf = SECTOR_MAP.get(u["ticker"], "?")
            print(f"    {u['ticker']:<7} {u['earnings_date']}  ({u['days_away']}d away)  -> {etf}")
    else:
        print("    No upcoming earnings dates found via yfinance")
    print()

    # ── 6. Save state ──
    if not dry_run and new_positions:
        state["active_positions"].extend(new_positions)

    # Clean expired positions
    today = datetime.now().date()
    state["active_positions"] = [
        p for p in state.get("active_positions", [])
        if datetime.strptime(p["target_exit_date"], "%Y-%m-%d").date() > today
    ]

    # Update P&L snapshots on active positions
    for p in state["active_positions"]:
        for ap in active:
            if ap["ticker"] == p["ticker"] and ap.get("entry_date") == p.get("entry_date"):
                p["current_price"] = ap.get("current_price")
                p["pnl_pct"] = ap.get("pnl_pct")
                p["pnl_dollar"] = ap.get("pnl_dollar")

    scan_record = {
        "scan_timestamp": datetime.now().isoformat(),
        "account_value": account_value,
        "regime": regime,
        "n_active": len([p for p in state["active_positions"] if not p.get("is_etf_leg")]),
        "candidates_found": len(candidates),
        "entries_generated": len(enter_signals),
    }
    state.setdefault("signal_history", []).append(scan_record)
    state["signal_history"] = state["signal_history"][-90:]
    state["last_scan_date"] = datetime.now().strftime("%Y-%m-%d")

    save_state(state, dry_run=dry_run)

    print("=" * 70)
    print()
    return signals, regime


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Earnings Beat Scanner")
    parser.add_argument("--account-value", type=float, default=TOTAL_CAPITAL,
                        help=f"Total capital for strategy (default: ${TOTAL_CAPITAL})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show signals but don't save state")
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON")
    args = parser.parse_args()

    signals, regime = run_scanner(args.account_value, dry_run=args.dry_run)

    if args.json:
        print(json.dumps({
            "timestamp": datetime.now().isoformat(),
            "regime": regime,
            "signals": signals,
        }, indent=2, default=str))

    enter_count = sum(1 for s in signals if s["signal"] == "ENTER_PAIR")
    sys.exit(0 if enter_count > 0 else 1)


if __name__ == "__main__":
    main()
