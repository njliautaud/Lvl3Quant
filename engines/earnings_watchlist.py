#!/usr/bin/env python3
"""
Earnings Watchlist — Post-Earnings Bounce Strategy Preparation
==============================================================
Identifies stocks reporting earnings in the next 7 days and evaluates them
as potential post-earnings bounce candidates.

Signal: 8%+ drop in 2 days after earnings → buy day 3, hold 10 days
Validated: 62.7% WR, PF 1.70, Sharpe 1.50

Also checks LIVE candidates (stocks that already reported and dropped 8%+).

Output: /home/jupiter/Lvl3Quant/state/earnings_watchlist.json
"""
import json
import sys
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings("ignore")

# ── Universe (from play_scanner_v2.py) ────────────────────────────────────
SECTOR_ETFS = ["XLK", "XLF", "XLV", "XLE", "XLI", "XLC", "XLY", "XLP", "XLU", "XLRE", "XLB"]
INDEX_ETFS = ["SPY", "QQQ", "IWM", "GLD", "TLT", "SLV"]
TOP_30_SP500 = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "BRK-B", "LLY", "AVGO", "JPM",
    "TSLA", "UNH", "V", "XOM", "MA", "COST", "PG", "JNJ", "HD", "ABBV",
    "MRK", "WMT", "BAC", "CRM", "NFLX", "AMD", "ORCL", "KO", "PEP", "TMO",
]
BUDGET_FRIENDLY = [
    "F", "INTC", "SNAP", "PLTR", "SOFI", "RIVN", "NIO", "MARA", "COIN",
    "T", "VZ", "PFE", "CSCO", "GM", "UBER", "ROKU", "DKNG", "HOOD",
]
ALL_TICKERS = SECTOR_ETFS + INDEX_ETFS + TOP_30_SP500 + BUDGET_FRIENDLY

# ── Config ────────────────────────────────────────────────────────────────
STATE_DIR = Path("/home/jupiter/Lvl3Quant/state")
STATE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = STATE_DIR / "earnings_watchlist.json"
TODAY = datetime.now().date()
BOUNCE_DROP_THRESHOLD = -0.08  # 8% drop triggers bounce signal
MAX_CONTRACT_COST = 300  # Budget constraint
LOOKBACK_WINDOW = 7  # Days ahead to watch for earnings
HOLD_DAYS = 10  # Strategy hold period


def get_earnings_date(ticker_obj):
    """Get upcoming and recent earnings dates from yfinance."""
    try:
        cal = ticker_obj.calendar
        if cal is not None:
            if isinstance(cal, dict):
                # Newer yfinance returns dict
                ed = cal.get("Earnings Date")
                if ed:
                    if isinstance(ed, list):
                        return [pd.Timestamp(d).date() for d in ed]
                    return [pd.Timestamp(ed).date()]
            elif isinstance(cal, pd.DataFrame):
                if "Earnings Date" in cal.columns:
                    dates = cal["Earnings Date"].tolist()
                    return [pd.Timestamp(d).date() for d in dates if pd.notna(d)]
                elif "Earnings Date" in cal.index:
                    val = cal.loc["Earnings Date"]
                    if hasattr(val, "tolist"):
                        return [pd.Timestamp(d).date() for d in val.tolist() if pd.notna(d)]
                    return [pd.Timestamp(val).date()]
    except Exception:
        pass

    # Fallback: try earnings_dates attribute
    try:
        ed = ticker_obj.earnings_dates
        if ed is not None and len(ed) > 0:
            dates = [d.date() if hasattr(d, 'date') else pd.Timestamp(d).date()
                     for d in ed.index]
            return dates
    except Exception:
        pass

    return []


def historical_earnings_reactions(ticker_sym, hist_df):
    """Analyze last 4 earnings reactions from price history.
    Returns list of 2-day post-earnings moves (as fraction)."""
    try:
        tk = yf.Ticker(ticker_sym)
        try:
            ed = tk.earnings_dates
        except Exception:
            return []
        if ed is None or len(ed) == 0:
            return []

        past_dates = [d.date() if hasattr(d, 'date') else pd.Timestamp(d).date()
                      for d in ed.index
                      if (d.date() if hasattr(d, 'date') else pd.Timestamp(d).date()) < TODAY]
        past_dates = sorted(past_dates, reverse=True)[:4]

        if not past_dates or hist_df is None or hist_df.empty:
            return []

        hist_dates = hist_df.index
        reactions = []
        for edate in past_dates:
            # Find the trading day on or just after earnings
            mask = hist_dates >= pd.Timestamp(edate)
            if mask.sum() < 3:
                continue
            post_days = hist_dates[mask][:3]  # earnings day + 2 trading days after
            if len(post_days) < 3:
                continue
            close_at_earnings = hist_df.loc[post_days[0], "Close"]
            close_2d_later = hist_df.loc[post_days[2], "Close"]
            if close_at_earnings > 0:
                move = (close_2d_later - close_at_earnings) / close_at_earnings
                reactions.append(round(float(move), 4))
        return reactions
    except Exception:
        return []


def check_options_affordable(ticker_obj, current_price):
    """Check if ATM call options are affordable (< $300 per contract).
    Returns (affordable: bool, cheapest_call_price: float or None, iv: float or None)."""
    try:
        expirations = ticker_obj.options
        if not expirations:
            return False, None, None

        # Find nearest expiration 10-30 days out
        target_exp = None
        for exp_str in expirations:
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
            dte = (exp_date - TODAY).days
            if 7 <= dte <= 45:
                target_exp = exp_str
                break

        if not target_exp:
            # Take first available
            target_exp = expirations[0]

        chain = ticker_obj.option_chain(target_exp)
        calls = chain.calls

        if calls.empty:
            return False, None, None

        # Find ATM call (closest strike to current price)
        calls = calls.copy()
        calls["dist"] = abs(calls["strike"] - current_price)
        atm = calls.loc[calls["dist"].idxmin()]

        # Price per contract = lastPrice * 100
        call_price = float(atm.get("lastPrice", 0))
        if call_price == 0:
            call_price = float(atm.get("ask", 0))
        contract_cost = call_price * 100
        iv = float(atm.get("impliedVolatility", 0)) if "impliedVolatility" in atm.index else None

        affordable = contract_cost <= MAX_CONTRACT_COST and contract_cost > 0
        return affordable, round(contract_cost, 2) if contract_cost > 0 else None, round(iv, 3) if iv else None

    except Exception:
        return False, None, None


def check_live_candidates(tickers_to_check):
    """Check stocks that recently reported earnings for 8%+ drops.
    Specifically checks TSLA and GOOGL which reported 7/22."""
    live_candidates = []

    for sym in tickers_to_check:
        try:
            print(f"  Checking live candidate: {sym}...")
            tk = yf.Ticker(sym)
            # Get recent price history
            hist = tk.history(period="10d")
            if hist.empty or len(hist) < 3:
                continue

            # Find the earnings date (we know 7/22 for TSLA/GOOGL)
            earnings_date = None
            try:
                ed = tk.earnings_dates
                if ed is not None and len(ed) > 0:
                    past = [d.date() if hasattr(d, 'date') else pd.Timestamp(d).date()
                            for d in ed.index
                            if (d.date() if hasattr(d, 'date') else pd.Timestamp(d).date()) <= TODAY]
                    past = sorted(past, reverse=True)
                    if past:
                        earnings_date = past[0]
            except Exception:
                pass

            # Get prices around earnings
            hist_dates = [d.date() for d in hist.index]
            current_price = float(hist["Close"].iloc[-1])

            # Calculate drop from earnings date close
            # For TSLA/GOOGL we expect earnings_date around 7/22
            earnings_close = None
            drop_pct = None

            if earnings_date:
                # Find the close on or nearest before earnings date
                for i, d in enumerate(hist_dates):
                    if d >= earnings_date:
                        if i > 0:
                            earnings_close = float(hist["Close"].iloc[i-1])  # Pre-earnings close
                        else:
                            earnings_close = float(hist["Close"].iloc[i])
                        break

            if earnings_close and earnings_close > 0:
                drop_pct = (current_price - earnings_close) / earnings_close

            # Also calculate 2-day drop from post-earnings first close
            post_earnings_close = None
            two_day_drop = None
            if earnings_date:
                post_mask = [i for i, d in enumerate(hist_dates) if d > earnings_date]
                if len(post_mask) >= 1:
                    post_earnings_close = float(hist["Close"].iloc[post_mask[0]])
                if len(post_mask) >= 2:
                    day2_close = float(hist["Close"].iloc[post_mask[1]])
                    if post_earnings_close and post_earnings_close > 0:
                        two_day_drop = (day2_close - earnings_close) / earnings_close if earnings_close else None

            # Check if this qualifies as a bounce candidate
            is_bounce_candidate = False
            if drop_pct is not None and drop_pct <= BOUNCE_DROP_THRESHOLD:
                is_bounce_candidate = True

            # Options affordability
            affordable, contract_cost, iv = check_options_affordable(tk, current_price)

            candidate = {
                "ticker": sym,
                "earnings_date": str(earnings_date) if earnings_date else "unknown",
                "current_price": round(current_price, 2),
                "pre_earnings_close": round(earnings_close, 2) if earnings_close else None,
                "post_earnings_first_close": round(post_earnings_close, 2) if post_earnings_close else None,
                "total_drop_pct": round(drop_pct * 100, 2) if drop_pct is not None else None,
                "two_day_drop_pct": round(two_day_drop * 100, 2) if two_day_drop is not None else None,
                "is_bounce_candidate": is_bounce_candidate,
                "options_affordable": affordable,
                "atm_call_cost": contract_cost,
                "implied_vol": iv,
                "days_since_earnings": (TODAY - earnings_date).days if earnings_date else None,
            }
            live_candidates.append(candidate)

            status = "BOUNCE CANDIDATE" if is_bounce_candidate else "watching"
            drop_str = f"{drop_pct*100:+.1f}%" if drop_pct else "N/A"
            print(f"    {sym}: {drop_str} from pre-earnings close, {status}")

        except Exception as e:
            print(f"    {sym}: Error - {e}")

    return live_candidates


def scan_upcoming_earnings():
    """Scan all tickers for earnings in the next 7 days."""
    upcoming = []
    errors = []
    cutoff = TODAY + timedelta(days=LOOKBACK_WINDOW)

    print(f"\nScanning {len(ALL_TICKERS)} tickers for earnings in next {LOOKBACK_WINDOW} days...")
    print(f"Window: {TODAY} to {cutoff}\n")

    for i, sym in enumerate(ALL_TICKERS):
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(ALL_TICKERS)}...")

        try:
            tk = yf.Ticker(sym)
            earnings_dates = get_earnings_date(tk)

            if not earnings_dates:
                continue

            # Check if any earnings date falls within our window
            relevant_dates = [d for d in earnings_dates
                              if TODAY <= d <= cutoff]

            if not relevant_dates:
                continue

            earnings_dt = min(relevant_dates)
            days_until = (earnings_dt - TODAY).days

            # Get current price
            hist = tk.history(period="6mo")
            if hist.empty:
                continue
            current_price = float(hist["Close"].iloc[-1])

            # Historical earnings reactions
            reactions = historical_earnings_reactions(sym, hist)
            big_drops = [r for r in reactions if r <= BOUNCE_DROP_THRESHOLD]
            avg_2d_move = round(np.mean(reactions) * 100, 2) if reactions else None
            drop_frequency = f"{len(big_drops)}/{len(reactions)}" if reactions else "N/A"

            # Options affordability
            affordable, contract_cost, iv = check_options_affordable(tk, current_price)

            entry = {
                "ticker": sym,
                "earnings_date": str(earnings_dt),
                "days_until_earnings": days_until,
                "current_price": round(current_price, 2),
                "historical_reactions": reactions,
                "avg_2d_move_pct": avg_2d_move,
                "big_drop_frequency": drop_frequency,
                "has_historical_big_drops": len(big_drops) > 0,
                "options_affordable": affordable,
                "atm_call_cost": contract_cost,
                "implied_vol": iv,
            }
            upcoming.append(entry)

            opt_str = f"${contract_cost}" if contract_cost else "N/A"
            print(f"  ** {sym} reports {earnings_dt} ({days_until}d) | "
                  f"Price: ${current_price:.2f} | "
                  f"Avg 2d move: {avg_2d_move}% | "
                  f"Big drops: {drop_frequency} | "
                  f"ATM call: {opt_str}")

            time.sleep(0.2)  # Rate limit

        except Exception as e:
            errors.append(f"{sym}: {e}")

    return upcoming, errors


def print_summary(upcoming, live_candidates):
    """Print a clean summary table."""
    print("\n" + "=" * 80)
    print("EARNINGS WATCHLIST — POST-EARNINGS BOUNCE STRATEGY")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M ET')}")
    print(f"Signal: 8%+ drop in 2d after earnings → buy day 3, hold {HOLD_DAYS}d")
    print(f"Validated: 62.7% WR, PF 1.70, Sharpe 1.50")
    print(f"Budget: ${MAX_CONTRACT_COST} max per trade")
    print("=" * 80)

    # Live candidates section
    if live_candidates:
        print("\n── LIVE POST-EARNINGS BOUNCE CANDIDATES ──")
        print(f"{'Ticker':<8} {'Earnings':<12} {'Drop%':<10} {'Price':<10} {'ATM Call':<12} {'IV':<8} {'Status':<20}")
        print("-" * 80)
        for c in live_candidates:
            drop = f"{c['total_drop_pct']:+.1f}%" if c['total_drop_pct'] else "N/A"
            price = f"${c['current_price']:.2f}"
            cost = f"${c['atm_call_cost']:.0f}" if c['atm_call_cost'] else "N/A"
            iv = f"{c['implied_vol']:.1%}" if c['implied_vol'] else "N/A"
            status = "BUY SIGNAL" if c['is_bounce_candidate'] else "watching"
            aff = " [AFFORDABLE]" if c['options_affordable'] else " [TOO EXPENSIVE]"
            if c['is_bounce_candidate']:
                status += aff
            print(f"{c['ticker']:<8} {c['earnings_date']:<12} {drop:<10} {price:<10} {cost:<12} {iv:<8} {status}")

            if c.get('two_day_drop_pct') is not None:
                print(f"         2-day drop from pre-earnings close: {c['two_day_drop_pct']:+.1f}%")
            if c.get('days_since_earnings') is not None:
                print(f"         Days since earnings: {c['days_since_earnings']}")

    # Upcoming earnings section
    if upcoming:
        print("\n── UPCOMING EARNINGS (NEXT 7 DAYS) ──")
        print(f"{'Ticker':<8} {'Date':<12} {'Days':<6} {'Price':<10} {'Avg 2d':<10} {'Drops':<10} {'ATM Call':<12} {'Affordable':<10}")
        print("-" * 80)
        # Sort by days until earnings
        for e in sorted(upcoming, key=lambda x: x['days_until_earnings']):
            avg = f"{e['avg_2d_move_pct']:+.1f}%" if e['avg_2d_move_pct'] is not None else "N/A"
            cost = f"${e['atm_call_cost']:.0f}" if e['atm_call_cost'] else "N/A"
            aff = "YES" if e['options_affordable'] else "NO"
            print(f"{e['ticker']:<8} {e['earnings_date']:<12} {e['days_until_earnings']:<6} "
                  f"${e['current_price']:<9.2f} {avg:<10} {e['big_drop_frequency']:<10} "
                  f"{cost:<12} {aff}")

        # Highlight high-probability bounce candidates
        high_prob = [e for e in upcoming if e['has_historical_big_drops'] and e['options_affordable']]
        if high_prob:
            print("\n  HIGH-PROBABILITY BOUNCE WATCHLIST (history of 8%+ drops + affordable options):")
            for e in high_prob:
                print(f"    {e['ticker']} — reports {e['earnings_date']}, "
                      f"avg 2d move {e['avg_2d_move_pct']}%, "
                      f"ATM call ${e['atm_call_cost']:.0f}")
    else:
        print("\n  No stocks in our universe reporting earnings in the next 7 days.")

    print("\n" + "=" * 80)


def main():
    print("Earnings Watchlist Builder")
    print(f"Date: {TODAY}")
    print(f"Universe: {len(ALL_TICKERS)} tickers")
    print(f"Budget: ${MAX_CONTRACT_COST} max per trade\n")

    # 1. Check LIVE candidates — TSLA and GOOGL reported 7/22
    print("── CHECKING LIVE POST-EARNINGS CANDIDATES ──")
    live_candidates = check_live_candidates(["TSLA", "GOOGL"])

    # 2. Scan all tickers for upcoming earnings
    upcoming, errors = scan_upcoming_earnings()

    # 3. Also check if any other tickers in our universe recently reported
    #    and dropped 8%+ (last 5 trading days)
    print("\n── SCANNING FOR OTHER RECENT EARNINGS DROPS ──")
    recent_live = []
    already_checked = {"TSLA", "GOOGL"}
    for sym in ALL_TICKERS:
        if sym in already_checked:
            continue
        try:
            tk = yf.Ticker(sym)
            dates = get_earnings_date(tk)
            recent = [d for d in dates if (TODAY - timedelta(days=5)) <= d <= TODAY]
            if recent:
                recent_live.append(sym)
        except Exception:
            pass

    if recent_live:
        print(f"  Found {len(recent_live)} additional recent reporters: {recent_live}")
        additional = check_live_candidates(recent_live)
        live_candidates.extend(additional)
    else:
        print("  No additional recent earnings reporters found in universe.")

    # 4. Print summary
    print_summary(upcoming, live_candidates)

    # 5. Save JSON watchlist
    output = {
        "generated": datetime.now().isoformat(),
        "date": str(TODAY),
        "strategy": {
            "name": "Post-Earnings Bounce",
            "signal": "8%+ drop in 2 days after earnings",
            "action": "Buy day 3, hold 10 days",
            "validated_wr": 0.627,
            "validated_pf": 1.70,
            "validated_sharpe": 1.50,
        },
        "budget": {
            "account_size": 645,
            "max_per_trade": MAX_CONTRACT_COST,
        },
        "live_candidates": live_candidates,
        "upcoming_earnings": upcoming,
        "errors": errors[:10],  # Cap error list
    }

    OUTPUT_FILE.write_text(json.dumps(output, indent=2, default=str))
    print(f"\nWatchlist saved to {OUTPUT_FILE}")

    # Summary stats
    bounce_ready = [c for c in live_candidates if c['is_bounce_candidate']]
    affordable_bounces = [c for c in bounce_ready if c['options_affordable']]
    print(f"\nSUMMARY:")
    print(f"  Live bounce candidates: {len(bounce_ready)}")
    print(f"  Affordable bounce plays: {len(affordable_bounces)}")
    print(f"  Upcoming earnings (7d): {len(upcoming)}")
    if errors:
        print(f"  Errors: {len(errors)} tickers failed")


if __name__ == "__main__":
    main()
