#!/usr/bin/env python3
"""
LEAPS + PMCC Screener
======================
Finds optimal candidates for Poor Man's Covered Calls:
  1. Buy deep ITM LEAPS call (0.80+ delta, 12-24 month expiry)
  2. Sell short-dated OTM calls against it (30-45 DTE, 0.20-0.30 delta)

Screens for:
  - Strong momentum (top quartile from momentum scanner)
  - Liquid options (tight spreads, high OI)
  - Good IV rank (want to SELL high IV on short leg)
  - Capital efficiency (LEAPS cost vs stock cost)
  - Weekly options available (more income opportunities)

Target account: Agentic (~$441), so we need LOW-COST LEAPS.

Usage:
  python growth/leaps_screener.py [--max-cost 500] [--min-momentum 0.10]
"""

import argparse
import json
import os
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")


def get_candidate_tickers(min_momentum: float = 0.0):
    """
    Get tickers from latest momentum scan, or fall back to a curated list
    of popular LEAPS candidates.
    """
    output_dir = Path(__file__).parent / "output"

    # Try to load latest momentum scan
    csvs = sorted(output_dir.glob("momentum_scan_*.csv"))
    if csvs:
        latest = pd.read_csv(csvs[-1])
        # Top quartile by momentum
        candidates = latest[latest["momentum_12_1"] >= min_momentum]
        tickers = candidates["ticker"].tolist()
        print(f"[INFO] Loaded {len(tickers)} candidates from momentum scan "
              f"(momentum >= {min_momentum:.0%})")
        return tickers

    # Fallback: popular LEAPS candidates with liquid options
    print("[INFO] No momentum scan found. Using curated candidate list.")
    return [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "AMD", "TSLA",
        "SPY", "QQQ", "IWM", "SOFI", "PLTR", "COIN", "SQ", "SHOP",
        "NET", "SNOW", "CRWD", "DDOG", "MDB", "UBER", "ABNB", "LI",
        "NIO", "RIVN", "LCID", "F", "GM", "BAC", "JPM", "GS",
        "XOM", "CVX", "COST", "WMT", "HD", "LOW", "DIS", "NFLX",
        "V", "MA", "PYPL", "CRM", "ORCL", "INTC", "MU", "QCOM",
    ]


def analyze_leaps_candidate(ticker: str, max_leaps_cost: float = 500.0):
    """
    Analyze a single ticker for LEAPS + PMCC suitability.

    Returns dict with analysis or None if not suitable.
    """
    try:
        import time
        time.sleep(0.5)  # Rate limit protection

        stock = yf.Ticker(ticker)

        # Get price from history (more reliable than .info which gets rate-limited)
        try:
            hist = stock.history(period="5d")
            if hist.empty:
                return None
            current_price = hist["Close"].iloc[-1]
        except Exception:
            return None

        # Get option expiration dates
        try:
            expirations = stock.options
        except Exception:
            return None

        if not expirations:
            return None

        # Find LEAPS expiration (12-24 months out)
        today = datetime.now().date()
        leaps_exp = None
        short_exp = None

        for exp_str in expirations:
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
            days_out = (exp_date - today).days

            # LEAPS: 300-730 days out (10-24 months)
            if 300 <= days_out <= 730 and leaps_exp is None:
                leaps_exp = exp_str

            # Short leg: 25-50 days out
            if 25 <= days_out <= 50 and short_exp is None:
                short_exp = exp_str

        if leaps_exp is None:
            return None  # No LEAPS available

        # Get LEAPS chain
        try:
            leaps_chain = stock.option_chain(leaps_exp)
        except Exception:
            return None

        calls = leaps_chain.calls

        if calls.empty:
            return None

        # Find deep ITM LEAPS (strike ~20-30% below current price for ~0.80 delta)
        target_strike = current_price * 0.75  # 25% below = deep ITM
        calls["strike_diff"] = abs(calls["strike"] - target_strike)
        best_leaps = calls.nsmallest(3, "strike_diff")

        if best_leaps.empty:
            return None

        leaps_option = best_leaps.iloc[0]
        leaps_mid = (leaps_option.get("bid", 0) + leaps_option.get("ask", 0)) / 2
        leaps_cost = leaps_mid * 100  # Cost per contract

        if leaps_cost <= 0 or leaps_cost > max_leaps_cost:
            return None

        # Capital efficiency: LEAPS cost vs 100 shares cost
        shares_cost = current_price * 100
        capital_efficiency = 1 - (leaps_cost / shares_cost)

        # Intrinsic value of LEAPS
        intrinsic = max(0, current_price - leaps_option["strike"]) * 100
        extrinsic = leaps_cost - intrinsic
        extrinsic_pct = extrinsic / leaps_cost if leaps_cost > 0 else 0

        # Short call analysis (if available)
        short_income = 0
        short_strike = None
        monthly_yield = 0

        if short_exp is not None:
            try:
                short_chain = stock.option_chain(short_exp)
                short_calls = short_chain.calls

                # Target: OTM call with delta ~0.20-0.30 (strike ~5-10% above current)
                target_short_strike = current_price * 1.05
                short_calls["strike_diff"] = abs(short_calls["strike"] - target_short_strike)
                best_short = short_calls.nsmallest(3, "strike_diff")

                if not best_short.empty:
                    short_opt = best_short.iloc[0]
                    short_mid = (short_opt.get("bid", 0) + short_opt.get("ask", 0)) / 2
                    short_income = short_mid * 100
                    short_strike = short_opt["strike"]
                    # Annualized yield on LEAPS cost
                    days_to_short_exp = (datetime.strptime(short_exp, "%Y-%m-%d").date() - today).days
                    if days_to_short_exp > 0 and leaps_cost > 0:
                        monthly_yield = (short_income / leaps_cost) * (30 / days_to_short_exp)
            except Exception:
                pass

        # IV rank approximation (compare current IV to range)
        iv = leaps_option.get("impliedVolatility", 0)

        # Has weekly options?
        has_weeklies = len([e for e in expirations
                          if 7 <= (datetime.strptime(e, "%Y-%m-%d").date() - today).days <= 14]) > 0

        return {
            "ticker": ticker,
            "price": round(current_price, 2),
            "shares_cost": round(shares_cost, 0),
            "leaps_exp": leaps_exp,
            "leaps_strike": leaps_option["strike"],
            "leaps_cost": round(leaps_cost, 0),
            "capital_saved_pct": round(capital_efficiency * 100, 1),
            "intrinsic_value": round(intrinsic, 0),
            "extrinsic_pct": round(extrinsic_pct * 100, 1),
            "short_exp": short_exp,
            "short_strike": short_strike,
            "short_income": round(short_income, 0),
            "monthly_yield_pct": round(monthly_yield * 100, 2),
            "annualized_yield_pct": round(monthly_yield * 12 * 100, 1),
            "iv": round(iv * 100, 1) if iv else None,
            "has_weeklies": has_weeklies,
            "leaps_oi": int(leaps_option.get("openInterest", 0)),
            "leaps_volume": int(leaps_option.get("volume", 0)) if pd.notna(leaps_option.get("volume")) else 0,
        }

    except Exception as e:
        return None


def run_screener(max_cost: float = 500.0, min_momentum: float = 0.0,
                 output_dir: str = None):
    """Run the LEAPS + PMCC screener."""
    print(f"\n{'='*70}")
    print(f"  LEAPS + PMCC SCREENER — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Max LEAPS cost: ${max_cost:,.0f}")
    print(f"{'='*70}\n")

    tickers = get_candidate_tickers(min_momentum=min_momentum)

    print(f"[INFO] Screening {len(tickers)} candidates for LEAPS opportunities...")
    results = []

    for i, ticker in enumerate(tickers):
        result = analyze_leaps_candidate(ticker, max_leaps_cost=max_cost)
        if result:
            results.append(result)
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(tickers)} ({len(results)} candidates found)")

    if not results:
        print("[WARN] No LEAPS candidates found within cost constraints.")
        return None

    df = pd.DataFrame(results)

    # Sort by annualized yield (best income first)
    df = df.sort_values("annualized_yield_pct", ascending=False).reset_index(drop=True)

    # Display results
    print(f"\n{'='*80}")
    print(f"  TOP LEAPS + PMCC CANDIDATES (sorted by annualized yield)")
    print(f"  Max LEAPS cost: ${max_cost:,.0f} | Found: {len(df)} candidates")
    print(f"{'='*80}")
    print(f"{'Ticker':<7} {'Price':>7} {'LEAPS$':>7} {'Saved%':>6} {'ShortInc':>8} "
          f"{'Mo Yld%':>7} {'Ann Yld%':>8} {'IV%':>5} {'Wkly':>5} {'LEAPS Exp':>11}")
    print("-" * 80)

    for _, row in df.head(25).iterrows():
        wkly = "Y" if row["has_weeklies"] else "N"
        iv_str = f"{row['iv']:.0f}" if row["iv"] else "N/A"
        print(f"{row['ticker']:<7} {row['price']:7.2f} {row['leaps_cost']:7.0f} "
              f"{row['capital_saved_pct']:5.1f}% {row['short_income']:8.0f} "
              f"{row['monthly_yield_pct']:6.2f}% {row['annualized_yield_pct']:7.1f}% "
              f"{iv_str:>5} {wkly:>5} {row['leaps_exp']:>11}")

    # Capital-constrained picks (for Agentic account ~$441)
    affordable = df[df["leaps_cost"] <= max_cost]
    if not affordable.empty:
        print(f"\n{'='*80}")
        print(f"  AFFORDABLE PICKS (LEAPS cost <= ${max_cost:,.0f})")
        print(f"{'='*80}")
        for _, row in affordable.head(10).iterrows():
            print(f"  {row['ticker']:<7} LEAPS: ${row['leaps_cost']:,.0f} | "
                  f"Sell monthly: ${row['short_income']:,.0f} | "
                  f"Yield: {row['annualized_yield_pct']:.1f}%/yr | "
                  f"Save {row['capital_saved_pct']:.0f}% vs shares")

    # Save outputs
    if output_dir is None:
        output_dir = str(Path(__file__).parent / "output")
    os.makedirs(output_dir, exist_ok=True)

    date_str = datetime.now().strftime("%Y%m%d")

    csv_path = os.path.join(output_dir, f"leaps_scan_{date_str}.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[INFO] Full results saved: {csv_path}")

    summary = {
        "scan_date": datetime.now().isoformat(),
        "max_leaps_cost": max_cost,
        "candidates_found": len(df),
        "affordable_count": len(affordable),
        "top_picks": df.head(10).to_dict("records"),
        "affordable_picks": affordable.head(5).to_dict("records") if not affordable.empty else [],
    }
    json_path = os.path.join(output_dir, f"leaps_summary_{date_str}.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LEAPS + PMCC Screener")
    parser.add_argument("--max-cost", type=float, default=500,
                        help="Maximum LEAPS contract cost ($)")
    parser.add_argument("--min-momentum", type=float, default=0.0,
                        help="Minimum 12-1 momentum to consider")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    run_screener(max_cost=args.max_cost, min_momentum=args.min_momentum,
                 output_dir=args.output_dir)
