#!/usr/bin/env python3
"""
Systematic Momentum Scanner
============================
Ranks stocks by 12-1 month momentum (Jegadeesh & Titman, 1993).
The "skip the most recent month" is critical — it avoids short-term reversal.

Outputs:
  - Top decile (buy candidates)
  - Bottom decile (avoid / short candidates)
  - Full ranked universe as CSV

Universe: S&P 500 + NASDAQ-100 (deduplicated)
Data: Yahoo Finance (free, no API key needed)
Schedule: Run weekly (Sunday night or Monday pre-market)

Usage:
  python growth/momentum_scanner.py [--top N] [--min-price 5] [--min-volume 500000]
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


# ── Universe ──────────────────────────────────────────────────────────────────

from universe import build_universe


# ── Momentum Calculation ─────────────────────────────────────────────────────

def calc_momentum_12_1(prices: pd.DataFrame) -> pd.Series:
    """
    12-1 month momentum: return from 12 months ago to 1 month ago.
    Skipping the most recent month avoids short-term reversal effect.

    prices: DataFrame with tickers as columns, dates as index (adjusted close)
    Returns: Series of momentum scores indexed by ticker
    """
    if len(prices) < 252:
        print(f"[WARN] Only {len(prices)} trading days of data (need 252 for 12mo)")

    # 252 trading days ≈ 12 months, 21 trading days ≈ 1 month
    t_12m = min(252, len(prices) - 1)
    t_1m = min(21, len(prices) - 1)

    price_now = prices.iloc[-1]
    price_1m_ago = prices.iloc[-t_1m]
    price_12m_ago = prices.iloc[-t_12m]

    # Momentum = return from 12m ago to 1m ago (skip recent month)
    momentum = (price_1m_ago / price_12m_ago) - 1.0

    return momentum


def calc_momentum_6_1(prices: pd.DataFrame) -> pd.Series:
    """6-1 month momentum (shorter lookback, useful for faster-moving stocks)."""
    t_6m = min(126, len(prices) - 1)
    t_1m = min(21, len(prices) - 1)

    price_1m_ago = prices.iloc[-t_1m]
    price_6m_ago = prices.iloc[-t_6m]

    return (price_1m_ago / price_6m_ago) - 1.0


def calc_volatility(prices: pd.DataFrame, window: int = 63) -> pd.Series:
    """Annualized volatility over trailing window (default 3 months)."""
    returns = prices.pct_change().iloc[-window:]
    return returns.std() * np.sqrt(252)


def calc_risk_adjusted_momentum(momentum: pd.Series, volatility: pd.Series) -> pd.Series:
    """Momentum divided by volatility — penalizes high-vol momentum."""
    return momentum / volatility.replace(0, np.nan)


# ── Filters ──────────────────────────────────────────────────────────────────

def apply_filters(df: pd.DataFrame, min_price: float = 5.0, min_volume: float = 500_000) -> pd.DataFrame:
    """Filter out penny stocks, illiquid names, and missing data."""
    before = len(df)
    df = df.dropna(subset=["momentum_12_1"])
    df = df[df["last_price"] >= min_price]
    df = df[df["avg_volume"] >= min_volume]
    after = len(df)
    print(f"[INFO] Filters: {before} -> {after} stocks (removed {before - after})")
    return df


# ── Main Scanner ─────────────────────────────────────────────────────────────

def run_scanner(top_n: int = 20, min_price: float = 5.0, min_volume: float = 500_000,
                output_dir: str = None):
    """Run the full momentum scan."""
    print(f"\n{'='*70}")
    print(f"  MOMENTUM SCANNER — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*70}\n")

    # 1. Build universe
    tickers = build_universe()
    if not tickers:
        print("[ERROR] No tickers found. Check internet connection.")
        return None

    # 2. Download price data (13 months to compute 12-1 momentum)
    print(f"[INFO] Downloading 13 months of price data for {len(tickers)} tickers...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=400)  # ~13 months with buffer

    # Download in batches to avoid timeout/rate limits
    import time
    batch_size = 50  # Smaller batches to avoid rate limits
    all_prices = {}
    all_volumes = {}

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        batch_str = " ".join(batch)
        try:
            data = yf.download(batch_str, start=start_date, end=end_date,
                             progress=False, threads=True, group_by="ticker")
            if not data.empty:
                if isinstance(data.columns, pd.MultiIndex):
                    # group_by="ticker" gives (TICKER, OHLCV) MultiIndex
                    for ticker in batch:
                        try:
                            if ticker in data.columns.get_level_values(0):
                                ticker_data = data[ticker]
                                close_col = "Adj Close" if "Adj Close" in ticker_data.columns else "Close"
                                series = ticker_data[close_col].dropna()
                                if len(series) > 100:
                                    all_prices[ticker] = series
                                    all_volumes[ticker] = ticker_data["Volume"].dropna()
                        except Exception:
                            pass
                else:
                    # Single ticker returned (shouldn't happen with batches)
                    if len(batch) == 1:
                        close_col = "Adj Close" if "Adj Close" in data.columns else "Close"
                        all_prices[batch[0]] = data[close_col].dropna()
                        all_volumes[batch[0]] = data["Volume"].dropna()

            got = sum(1 for t in batch if t in all_prices)
            print(f"  Batch {i//batch_size + 1}/{(len(tickers)-1)//batch_size + 1}: "
                  f"got {got}/{len(batch)} tickers")
        except Exception as e:
            print(f"  [WARN] Batch {i//batch_size + 1} failed: {e}")

        # Small delay to avoid rate limiting
        if i + batch_size < len(tickers):
            time.sleep(1)

    prices_df = pd.DataFrame(all_prices)
    volumes_df = pd.DataFrame(all_volumes)

    print(f"[INFO] Got price data for {len(prices_df.columns)} / {len(tickers)} tickers")

    # 3. Calculate momentum metrics
    print("[INFO] Calculating momentum scores...")
    mom_12_1 = calc_momentum_12_1(prices_df)
    mom_6_1 = calc_momentum_6_1(prices_df)
    vol = calc_volatility(prices_df)
    risk_adj_mom = calc_risk_adjusted_momentum(mom_12_1, vol)

    # 4. Build results DataFrame
    ticker_list = list(prices_df.columns)
    avg_vol = volumes_df.iloc[-63:].mean() if len(volumes_df) >= 63 else volumes_df.mean()
    ret_1m = (prices_df.iloc[-1] / prices_df.iloc[-21] - 1) if len(prices_df) >= 21 else pd.Series(np.nan, index=prices_df.columns)

    results = pd.DataFrame({
        "ticker": ticker_list,
        "last_price": [prices_df[t].iloc[-1] for t in ticker_list],
        "avg_volume": [avg_vol.get(t, np.nan) for t in ticker_list],
        "momentum_12_1": [mom_12_1.get(t, np.nan) for t in ticker_list],
        "momentum_6_1": [mom_6_1.get(t, np.nan) for t in ticker_list],
        "volatility_3m": [vol.get(t, np.nan) for t in ticker_list],
        "risk_adj_momentum": [risk_adj_mom.get(t, np.nan) for t in ticker_list],
        "return_1m": [ret_1m.get(t, np.nan) for t in ticker_list],
    })

    # 5. Apply filters
    results = apply_filters(results, min_price=min_price, min_volume=min_volume)

    # 6. Rank by momentum
    results = results.sort_values("momentum_12_1", ascending=False).reset_index(drop=True)
    results["rank"] = range(1, len(results) + 1)
    n_stocks = len(results)
    results["decile"] = pd.qcut(results["rank"], 10, labels=False, duplicates="drop") + 1

    # 7. Output
    top = results.head(top_n)
    bottom = results.tail(top_n)

    print(f"\n{'='*70}")
    print(f"  TOP {top_n} MOMENTUM STOCKS (BUY CANDIDATES)")
    print(f"{'='*70}")
    print(f"{'Rank':>4} {'Ticker':<7} {'Price':>8} {'Mom 12-1':>10} {'Mom 6-1':>9} "
          f"{'RiskAdj':>8} {'Vol 3m':>7} {'Ret 1m':>8}")
    print("-" * 70)
    for _, row in top.iterrows():
        print(f"{int(row['rank']):4d} {row['ticker']:<7} {row['last_price']:8.2f} "
              f"{row['momentum_12_1']:10.1%} {row['momentum_6_1']:9.1%} "
              f"{row['risk_adj_momentum']:8.2f} {row['volatility_3m']:7.1%} "
              f"{row['return_1m']:8.1%}")

    print(f"\n{'='*70}")
    print(f"  BOTTOM {top_n} MOMENTUM STOCKS (AVOID / SHORT CANDIDATES)")
    print(f"{'='*70}")
    print(f"{'Rank':>4} {'Ticker':<7} {'Price':>8} {'Mom 12-1':>10} {'Mom 6-1':>9} "
          f"{'RiskAdj':>8} {'Vol 3m':>7} {'Ret 1m':>8}")
    print("-" * 70)
    for _, row in bottom.iterrows():
        print(f"{int(row['rank']):4d} {row['ticker']:<7} {row['last_price']:8.2f} "
              f"{row['momentum_12_1']:10.1%} {row['momentum_6_1']:9.1%} "
              f"{row['risk_adj_momentum']:8.2f} {row['volatility_3m']:7.1%} "
              f"{row['return_1m']:8.1%}")

    # 8. Save outputs
    if output_dir is None:
        output_dir = str(Path(__file__).parent / "output")
    os.makedirs(output_dir, exist_ok=True)

    date_str = datetime.now().strftime("%Y%m%d")

    # Full ranked universe
    csv_path = os.path.join(output_dir, f"momentum_scan_{date_str}.csv")
    results.to_csv(csv_path, index=False)
    print(f"\n[INFO] Full rankings saved: {csv_path}")

    # Summary JSON
    summary = {
        "scan_date": datetime.now().isoformat(),
        "universe_size": n_stocks,
        "top_decile": top[["ticker", "last_price", "momentum_12_1",
                           "risk_adj_momentum"]].to_dict("records"),
        "bottom_decile": bottom[["ticker", "last_price", "momentum_12_1",
                                  "risk_adj_momentum"]].to_dict("records"),
        "median_momentum": float(results["momentum_12_1"].median()),
        "mean_momentum": float(results["momentum_12_1"].mean()),
    }
    json_path = os.path.join(output_dir, f"momentum_summary_{date_str}.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"[INFO] Summary saved: {json_path}")

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Systematic Momentum Scanner")
    parser.add_argument("--top", type=int, default=20, help="Number of top/bottom stocks to display")
    parser.add_argument("--min-price", type=float, default=5.0, help="Minimum stock price filter")
    parser.add_argument("--min-volume", type=float, default=500_000, help="Minimum avg daily volume")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    run_scanner(top_n=args.top, min_price=args.min_price,
                min_volume=args.min_volume, output_dir=args.output_dir)
