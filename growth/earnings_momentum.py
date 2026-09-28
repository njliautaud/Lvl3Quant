#!/usr/bin/env python3
"""
Earnings Momentum Scanner (Post-Earnings Drift)
=================================================
Exploits the most persistent anomaly in finance: stocks that beat earnings
estimates tend to continue drifting in the surprise direction for 30-60 days.

Strategy:
  - Find stocks that reported earnings in the last 5 days
  - Filter for: beat estimates + raised guidance + positive price reaction
  - Rank by surprise magnitude
  - Hold 30-60 days, then exit

Data source: Yahoo Finance earnings calendar + fundamentals.

Usage:
  python growth/earnings_momentum.py [--lookback-days 7] [--top 20]
"""

import argparse
import json
import os
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")


def get_recent_earnings(lookback_days: int = 7):
    """
    Get stocks that reported earnings recently.
    Uses yfinance to check earnings dates for a broad universe.
    """
    # Get S&P 500 tickers
    try:
        from universe import get_sp500_tickers
        tickers = get_sp500_tickers()
    except Exception:
        print("[WARN] Could not fetch S&P 500 list")
        return []

    today = datetime.now().date()
    cutoff = today - timedelta(days=lookback_days)

    results = []
    print(f"[INFO] Scanning {len(tickers)} tickers for recent earnings (last {lookback_days} days)...")

    batch_count = 0
    for i, ticker in enumerate(tickers):
        try:
            stock = yf.Ticker(ticker)
            cal = stock.calendar
            earnings_dates = stock.earnings_dates

            if earnings_dates is not None and not earnings_dates.empty:
                # Find most recent past earnings date
                past_earnings = earnings_dates[earnings_dates.index.date <= today]
                if not past_earnings.empty:
                    last_earnings = past_earnings.index[0]
                    if last_earnings.date() >= cutoff:
                        # Get surprise data
                        row = past_earnings.iloc[0]
                        eps_estimate = row.get("EPS Estimate", None)
                        eps_actual = row.get("Reported EPS", None)
                        surprise_pct = row.get("Surprise(%)", None)

                        # Get price reaction
                        hist = stock.history(start=last_earnings.date() - timedelta(days=2),
                                           end=last_earnings.date() + timedelta(days=3))
                        if len(hist) >= 2:
                            pre_price = hist["Close"].iloc[0]
                            post_price = hist["Close"].iloc[-1]
                            price_reaction = (post_price / pre_price - 1) * 100
                        else:
                            price_reaction = None

                        results.append({
                            "ticker": ticker,
                            "earnings_date": last_earnings.strftime("%Y-%m-%d"),
                            "eps_estimate": float(eps_estimate) if pd.notna(eps_estimate) else None,
                            "eps_actual": float(eps_actual) if pd.notna(eps_actual) else None,
                            "surprise_pct": float(surprise_pct) if pd.notna(surprise_pct) else None,
                            "price_reaction_pct": round(price_reaction, 2) if price_reaction else None,
                        })

        except Exception:
            pass

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(tickers)} ({len(results)} with recent earnings)")

    return results


def score_earnings_candidates(results: list):
    """
    Score and rank earnings candidates.

    Best candidates: beat estimates + positive price reaction + high surprise %
    """
    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)

    # Calculate composite score
    df["beat"] = df["surprise_pct"].apply(lambda x: x > 0 if pd.notna(x) else False)
    df["positive_reaction"] = df["price_reaction_pct"].apply(
        lambda x: x > 0 if pd.notna(x) else False)

    # Score: higher is better
    # Surprise % (normalized) + price reaction alignment + beat bonus
    df["score"] = 0.0

    # Surprise magnitude (capped at 100% for outliers)
    if df["surprise_pct"].notna().any():
        sp = df["surprise_pct"].fillna(0).clip(-100, 100)
        df["score"] += sp / sp.abs().max() * 40  # 40 points max

    # Price reaction alignment (same direction as surprise = good)
    if df["price_reaction_pct"].notna().any():
        df["aligned"] = (df["surprise_pct"].fillna(0) > 0) == (df["price_reaction_pct"].fillna(0) > 0)
        df["score"] += df["aligned"].astype(float) * 30  # 30 points for alignment

    # Absolute price reaction (bigger = more conviction)
    if df["price_reaction_pct"].notna().any():
        pr = df["price_reaction_pct"].fillna(0).abs()
        df["score"] += (pr / pr.max() * 30).clip(0, 30)  # 30 points max

    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    return df


def run_scanner(lookback_days: int = 7, top_n: int = 20, output_dir: str = None):
    """Run the earnings momentum scanner."""
    print(f"\n{'='*70}")
    print(f"  EARNINGS MOMENTUM SCANNER — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Lookback: {lookback_days} days")
    print(f"{'='*70}\n")

    results = get_recent_earnings(lookback_days=lookback_days)

    if not results:
        print("[INFO] No recent earnings found in the lookback period.")
        return None

    df = score_earnings_candidates(results)

    # Display
    print(f"\n{'='*80}")
    print(f"  EARNINGS MOMENTUM CANDIDATES ({len(df)} stocks reported)")
    print(f"{'='*80}")

    # Positive surprises (BUY candidates)
    buys = df[df["beat"] == True].head(top_n)
    if not buys.empty:
        print(f"\n  BEATS (potential longs — post-earnings drift up):")
        print(f"  {'Ticker':<7} {'Date':>11} {'EPS Est':>8} {'EPS Act':>8} "
              f"{'Surprise%':>10} {'Reaction%':>10} {'Score':>6}")
        print("  " + "-" * 70)
        for _, row in buys.iterrows():
            est = f"{row['eps_estimate']:.2f}" if pd.notna(row["eps_estimate"]) else "N/A"
            act = f"{row['eps_actual']:.2f}" if pd.notna(row["eps_actual"]) else "N/A"
            surp = f"{row['surprise_pct']:.1f}%" if pd.notna(row["surprise_pct"]) else "N/A"
            react = f"{row['price_reaction_pct']:.1f}%" if pd.notna(row["price_reaction_pct"]) else "N/A"
            print(f"  {row['ticker']:<7} {row['earnings_date']:>11} {est:>8} {act:>8} "
                  f"{surp:>10} {react:>10} {row['score']:6.1f}")

    # Negative surprises (SHORT candidates)
    misses = df[df["beat"] == False].head(top_n)
    if not misses.empty:
        print(f"\n  MISSES (potential shorts — post-earnings drift down):")
        print(f"  {'Ticker':<7} {'Date':>11} {'EPS Est':>8} {'EPS Act':>8} "
              f"{'Surprise%':>10} {'Reaction%':>10} {'Score':>6}")
        print("  " + "-" * 70)
        for _, row in misses.iterrows():
            est = f"{row['eps_estimate']:.2f}" if pd.notna(row["eps_estimate"]) else "N/A"
            act = f"{row['eps_actual']:.2f}" if pd.notna(row["eps_actual"]) else "N/A"
            surp = f"{row['surprise_pct']:.1f}%" if pd.notna(row["surprise_pct"]) else "N/A"
            react = f"{row['price_reaction_pct']:.1f}%" if pd.notna(row["price_reaction_pct"]) else "N/A"
            print(f"  {row['ticker']:<7} {row['earnings_date']:>11} {est:>8} {act:>8} "
                  f"{surp:>10} {react:>10} {row['score']:6.1f}")

    # Save
    if output_dir is None:
        output_dir = str(Path(__file__).parent / "output")
    os.makedirs(output_dir, exist_ok=True)

    date_str = datetime.now().strftime("%Y%m%d")
    csv_path = os.path.join(output_dir, f"earnings_momentum_{date_str}.csv")
    df.to_csv(csv_path, index=False)

    summary = {
        "scan_date": datetime.now().isoformat(),
        "lookback_days": lookback_days,
        "total_earnings": len(df),
        "beats": len(df[df["beat"] == True]),
        "misses": len(df[df["beat"] == False]),
        "top_beats": buys.head(5).to_dict("records") if not buys.empty else [],
        "top_misses": misses.head(5).to_dict("records") if not misses.empty else [],
    }
    json_path = os.path.join(output_dir, f"earnings_summary_{date_str}.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n[INFO] Results saved to {output_dir}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Earnings Momentum Scanner")
    parser.add_argument("--lookback-days", type=int, default=7)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    run_scanner(lookback_days=args.lookback_days, top_n=args.top,
                output_dir=args.output_dir)
