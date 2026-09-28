#!/usr/bin/env python3
"""
Earnings Gap Scanner — Daily watchlist generator for 10%+ post-earnings gaps.
==============================================================================

Strategy backing: Earnings Gap Buyer v1 — Sharpe 5.55, WR 63%, PF 2.61, perm p=0.000
When a stock gaps 10%+ after earnings, buying the gap direction on next open has edge.

This script:
  1. Finds stocks with earnings in the next 5 trading days
  2. Checks each stock's historical earnings move magnitude (last 4-8 quarters)
  3. Flags stocks likely to produce 10%+ gaps based on historical behavior
  4. Outputs a ranked watchlist to JSON

Run as daily cron at 8 PM ET (night before earnings):
  0 20 * * 1-5 /usr/bin/python3 /home/jupiter/Lvl3Quant/scripts/growth_research/earnings_gap_scanner.py

Dependencies: yfinance, pandas, numpy (all standard in our env)
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo

warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/jupiter/Lvl3Quant/output/growth_research'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'earnings_watchlist.json')
os.makedirs(OUTPUT_DIR, exist_ok=True)

ET = ZoneInfo('America/New_York')

# ---------------------------------------------------------------------------
# Known high-volatility earnings movers — stocks that have historically
# produced 10%+ gaps on earnings. This is our priority scan list.
# Curated from historical data: stocks with at least one 10%+ earnings move
# in the last 2 years, sorted by frequency of large moves.
# ---------------------------------------------------------------------------
PRIORITY_TICKERS = [
    # Mega-cap tech (frequent 10%+ movers)
    'META', 'TSLA', 'NFLX', 'GOOGL', 'AMZN', 'AAPL', 'MSFT', 'NVDA',
    # Semis (volatile earnings)
    'AMD', 'INTC', 'MRVL', 'MU', 'AVGO', 'QCOM', 'SMCI', 'ARM', 'LRCX', 'AMAT',
    # High-beta tech / growth
    'SNAP', 'PINS', 'ROKU', 'SHOP', 'SQ', 'PYPL', 'COIN', 'PLTR', 'CRWD', 'NET',
    'DDOG', 'SNOW', 'ZS', 'PANW', 'OKTA', 'MDB', 'TTD', 'ABNB', 'UBER', 'LYFT',
    'DASH', 'RBLX', 'AFRM', 'HOOD', 'SOFI', 'UPST', 'BILL', 'HUBS',
    # Biotech / pharma (binary events)
    'MRNA', 'BNTX', 'BIIB', 'REGN', 'VRTX', 'ALGN', 'DXCM', 'ISRG', 'ILMN',
    'SGEN', 'BMRN', 'NBIX', 'EXAS', 'RARE', 'SRPT',
    # Consumer / retail (volatile reporters)
    'LULU', 'ETSY', 'W', 'CHWY', 'DKS', 'DECK', 'ON', 'ENPH', 'SEDG', 'FSLR',
    # EV / energy transition
    'RIVN', 'LCID', 'NIO', 'LI', 'XPEV', 'QS',
    # Financials with volatile reports
    'GS', 'MS', 'SCHW', 'IBKR',
    # Other known movers
    'CMG', 'NFLX', 'SPOT', 'ZM', 'DOCU', 'TWLO', 'WDAY', 'NOW', 'CRM',
    # Large-cap that occasionally gap big
    'UNH', 'CAT', 'BA', 'DE', 'FDX', 'UPS',
]
# Deduplicate
PRIORITY_TICKERS = list(dict.fromkeys(PRIORITY_TICKERS))


def get_upcoming_earnings(tickers: list[str], days_ahead: int = 5) -> dict:
    """
    Check which tickers have earnings in the next N trading days.
    Returns dict: {ticker: earnings_date}
    """
    today = datetime.now(ET).date()
    cutoff = today + timedelta(days=days_ahead + 2)  # pad for weekends

    upcoming = {}
    failed = []

    for ticker in tickers:
        try:
            t = yf.Ticker(ticker)
            # yfinance exposes earnings dates via the calendar property
            cal = t.calendar
            if cal is None or (isinstance(cal, pd.DataFrame) and cal.empty):
                continue

            # calendar can be a dict or DataFrame depending on yfinance version
            if isinstance(cal, dict):
                earn_date = cal.get('Earnings Date')
                if earn_date is None:
                    continue
                # Can be a list of dates or single date
                if isinstance(earn_date, list):
                    earn_date = earn_date[0]
                if hasattr(earn_date, 'date'):
                    earn_date = earn_date.date()
                elif isinstance(earn_date, str):
                    earn_date = datetime.strptime(earn_date[:10], '%Y-%m-%d').date()
            elif isinstance(cal, pd.DataFrame):
                if 'Earnings Date' in cal.columns:
                    earn_date = cal['Earnings Date'].iloc[0]
                elif 'Earnings Date' in cal.index:
                    earn_date = cal.loc['Earnings Date'].iloc[0]
                else:
                    continue
                if hasattr(earn_date, 'date'):
                    earn_date = earn_date.date()
                elif isinstance(earn_date, str):
                    earn_date = datetime.strptime(earn_date[:10], '%Y-%m-%d').date()
            else:
                continue

            if today <= earn_date <= cutoff:
                upcoming[ticker] = earn_date
        except Exception as e:
            failed.append((ticker, str(e)))
            continue

    if failed:
        print(f"  [info] Failed to fetch calendar for {len(failed)} tickers (rate limits / no data)")

    return upcoming


def get_historical_earnings_moves(ticker: str, n_quarters: int = 8) -> dict | None:
    """
    Calculate historical post-earnings gap magnitudes for a ticker.
    Uses quarterly earnings dates and compares prev close to next open.

    Returns dict with stats or None if insufficient data.
    """
    try:
        t = yf.Ticker(ticker)

        # Get earnings history (dates of past earnings)
        earnings_hist = t.earnings_dates
        if earnings_hist is None or len(earnings_hist) == 0:
            return None

        # Get price history (2 years should cover 8 quarters)
        hist = t.history(period='2y', interval='1d')
        if hist is None or len(hist) < 60:
            return None

        # earnings_dates index has timezone-aware timestamps
        # Convert to dates for matching
        earnings_dates = []
        for dt_idx in earnings_hist.index[:n_quarters]:
            if hasattr(dt_idx, 'date'):
                earnings_dates.append(dt_idx.date())
            else:
                earnings_dates.append(dt_idx)

        # For each earnings date, compute the gap (prev close -> post-earnings open)
        gaps = []
        hist_dates = [d.date() if hasattr(d, 'date') else d for d in hist.index]

        for edate in earnings_dates:
            # Find the trading day of/after earnings and the day before
            # Earnings can be AMC (after market close) or BMO (before market open)
            # For AMC: gap shows on next trading day's open
            # For BMO: gap shows on that day's open
            # We look for the gap between the closest prior close and next open

            # Find closest trading dates around earnings
            dates_before = [d for d in hist_dates if d <= edate]
            dates_after = [d for d in hist_dates if d >= edate]

            if len(dates_before) < 1 or len(dates_after) < 1:
                continue

            # Check if earnings date itself is a trading day
            if edate in hist_dates:
                edate_idx = hist_dates.index(edate)
                # Check gap on earnings day (BMO case)
                if edate_idx > 0:
                    prev_close = hist['Close'].iloc[edate_idx - 1]
                    eday_open = hist['Open'].iloc[edate_idx]
                    gap_eday = ((eday_open - prev_close) / prev_close) * 100

                    # Check gap on day after (AMC case)
                    if edate_idx + 1 < len(hist):
                        eday_close = hist['Close'].iloc[edate_idx]
                        next_open = hist['Open'].iloc[edate_idx + 1]
                        gap_next = ((next_open - eday_close) / eday_close) * 100
                    else:
                        gap_next = 0

                    # The actual earnings gap is whichever is larger in absolute terms
                    # (one will be ~0 if timing is clear, both nonzero if uncertain)
                    if abs(gap_eday) >= abs(gap_next):
                        gaps.append(gap_eday)
                    else:
                        gaps.append(gap_next)
            else:
                # Earnings on non-trading day — gap shows on next trading day
                next_td = min(dates_after)
                next_idx = hist_dates.index(next_td)
                if next_idx > 0:
                    prev_close = hist['Close'].iloc[next_idx - 1]
                    next_open = hist['Open'].iloc[next_idx]
                    gap = ((next_open - prev_close) / prev_close) * 100
                    gaps.append(gap)

        if len(gaps) < 2:
            return None

        gaps_arr = np.array(gaps)
        abs_gaps = np.abs(gaps_arr)

        return {
            'ticker': ticker,
            'n_quarters_analyzed': len(gaps),
            'gaps': [round(g, 2) for g in gaps],
            'avg_abs_gap_pct': round(float(np.mean(abs_gaps)), 2),
            'max_abs_gap_pct': round(float(np.max(abs_gaps)), 2),
            'min_abs_gap_pct': round(float(np.min(abs_gaps)), 2),
            'median_abs_gap_pct': round(float(np.median(abs_gaps)), 2),
            'n_gaps_above_10pct': int(np.sum(abs_gaps >= 10)),
            'n_gaps_above_7pct': int(np.sum(abs_gaps >= 7)),
            'n_gaps_above_5pct': int(np.sum(abs_gaps >= 5)),
            'pct_above_10': round(float(np.mean(abs_gaps >= 10) * 100), 1),
            'pct_above_7': round(float(np.mean(abs_gaps >= 7) * 100), 1),
            'last_gap_pct': round(float(gaps[0]), 2),
        }

    except Exception as e:
        return None


def score_ticker(move_stats: dict) -> float:
    """
    Score a ticker's likelihood of producing a 10%+ gap.
    Higher = more likely to trigger our strategy.
    """
    score = 0.0

    # Direct evidence: has produced 10%+ gaps before
    score += move_stats['n_gaps_above_10pct'] * 30

    # Near-misses: 7%+ gaps suggest capability
    score += (move_stats['n_gaps_above_7pct'] - move_stats['n_gaps_above_10pct']) * 15

    # Average magnitude
    avg = move_stats['avg_abs_gap_pct']
    if avg >= 10:
        score += 40
    elif avg >= 7:
        score += 25
    elif avg >= 5:
        score += 10

    # Max gap (shows tail potential)
    if move_stats['max_abs_gap_pct'] >= 15:
        score += 20
    elif move_stats['max_abs_gap_pct'] >= 10:
        score += 10

    # Consistency: what fraction of quarters produce big moves
    score += move_stats['pct_above_10'] * 0.5
    score += move_stats['pct_above_7'] * 0.3

    return round(score, 1)


def get_current_price(ticker: str) -> float | None:
    """Get current/last price for position sizing context."""
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period='2d')
        if len(hist) > 0:
            return round(float(hist['Close'].iloc[-1]), 2)
    except:
        pass
    return None


def run_scanner():
    """Main scanner logic."""
    now = datetime.now(ET)
    print(f"{'=' * 70}")
    print(f"EARNINGS GAP SCANNER — {now.strftime('%Y-%m-%d %H:%M ET')}")
    print(f"Strategy: Buy 10%+ gaps in direction of gap (Sharpe 5.55, WR 63%)")
    print(f"{'=' * 70}")
    print()

    # Step 1: Find upcoming earnings among priority tickers
    print(f"[1/3] Scanning {len(PRIORITY_TICKERS)} high-volatility tickers for upcoming earnings...")
    upcoming = get_upcoming_earnings(PRIORITY_TICKERS, days_ahead=5)

    if not upcoming:
        print("  No upcoming earnings found in priority list for next 5 days.")
        print("  (This can happen on off-cycle weeks or due to yfinance rate limits)")
        result = {
            'scan_time': now.isoformat(),
            'scan_date': now.strftime('%Y-%m-%d'),
            'watchlist': [],
            'summary': 'No upcoming earnings found among priority tickers.',
        }
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\n  Empty watchlist saved to {OUTPUT_FILE}")
        return result

    print(f"  Found {len(upcoming)} tickers with earnings in next 5 days:")
    for ticker, edate in sorted(upcoming.items(), key=lambda x: x[1]):
        days_until = (edate - now.date()).days
        label = "TOMORROW" if days_until <= 1 else f"in {days_until} days"
        print(f"    {ticker:6s} — earnings {edate} ({label})")
    print()

    # Step 2: Analyze historical moves for each
    print(f"[2/3] Analyzing historical earnings moves...")
    watchlist = []

    for ticker, edate in sorted(upcoming.items(), key=lambda x: x[1]):
        stats = get_historical_earnings_moves(ticker)
        if stats is None:
            print(f"  {ticker:6s} — insufficient historical data, skipping")
            continue

        score = score_ticker(stats)
        price = get_current_price(ticker)
        days_until = (edate - now.date()).days

        entry = {
            'ticker': ticker,
            'earnings_date': edate.isoformat(),
            'days_until_earnings': days_until,
            'score': score,
            'current_price': price,
            'historical_moves': stats,
            'alert_level': (
                'HIGH' if score >= 50 else
                'MEDIUM' if score >= 25 else
                'LOW'
            ),
        }
        watchlist.append(entry)

        alert_tag = entry['alert_level']
        print(f"  {ticker:6s} [{alert_tag:6s}] score={score:5.1f}  "
              f"avg_gap={stats['avg_abs_gap_pct']:5.1f}%  "
              f"max_gap={stats['max_abs_gap_pct']:5.1f}%  "
              f"10%+ hits={stats['n_gaps_above_10pct']}/{stats['n_quarters_analyzed']}  "
              f"price=${price or 0:.0f}")

    # Sort by score descending
    watchlist.sort(key=lambda x: x['score'], reverse=True)
    print()

    # Step 3: Output summary
    high_alerts = [w for w in watchlist if w['alert_level'] == 'HIGH']
    medium_alerts = [w for w in watchlist if w['alert_level'] == 'MEDIUM']
    tomorrow = [w for w in watchlist if w['days_until_earnings'] <= 1]

    print(f"[3/3] SUMMARY")
    print(f"  Total upcoming: {len(watchlist)}")
    print(f"  HIGH priority (score >= 50): {len(high_alerts)}")
    print(f"  MEDIUM priority (score >= 25): {len(medium_alerts)}")
    print(f"  Reporting TOMORROW: {len(tomorrow)}")
    print()

    if high_alerts:
        print("  === HIGH PRIORITY — Set gap check crons for these ===")
        for w in high_alerts:
            print(f"    {w['ticker']:6s} — earnings {w['earnings_date']}, "
                  f"avg gap {w['historical_moves']['avg_abs_gap_pct']:.1f}%, "
                  f"last gap {w['historical_moves']['last_gap_pct']:+.1f}%, "
                  f"price ${w['current_price'] or 0:.0f}")
        print()

    if tomorrow:
        print("  === REPORTING TOMORROW — Monitor at 9:35 AM ET ===")
        for w in tomorrow:
            print(f"    {w['ticker']:6s} — check gap at open, threshold 10%")
        print()

    # Build the output
    result = {
        'scan_time': now.isoformat(),
        'scan_date': now.strftime('%Y-%m-%d'),
        'strategy_reference': {
            'name': 'Earnings Gap Buyer v1',
            'sharpe': 5.55,
            'win_rate': 0.632,
            'profit_factor': 2.61,
            'perm_p_value': 0.000,
            'gap_threshold_pct': 10.0,
            'avg_return_per_trade_pct': 1.53,
            'n_backtested_trades': 57,
        },
        'watchlist': watchlist,
        'high_priority': [w['ticker'] for w in high_alerts],
        'tomorrow_earnings': [w['ticker'] for w in tomorrow],
        'summary': (
            f"{len(watchlist)} stocks with upcoming earnings scanned. "
            f"{len(high_alerts)} HIGH priority (historically 10%+ movers). "
            f"{len(tomorrow)} reporting tomorrow."
        ),
    }

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(result, f, indent=2, default=str)

    print(f"  Watchlist saved to {OUTPUT_FILE}")
    return result


if __name__ == '__main__':
    result = run_scanner()
