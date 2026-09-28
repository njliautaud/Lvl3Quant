#!/usr/bin/env python3
"""
Insider Buy Signal Backtest
Measures forward returns after insider purchases for our universe.
Validates whether insider buying predicts future returns before we trade it.
"""

import json
import logging
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import requests

# Paths
BASE_DIR = Path("/home/jupiter/Lvl3Quant")
FILINGS_DIR = BASE_DIR / "data" / "insider_filings"
LOG_DIR = BASE_DIR / "logs" / "insider_signals"
RESULTS_FILE = FILINGS_DIR / "backtest_results.json"

LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "backtest.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# Forward return horizons (trading days)
HORIZONS = [5, 10, 20, 60]

# Yahoo Finance rate limit
YF_DELAY = 0.5


def load_purchases() -> list[dict]:
    """Load insider purchase transactions."""
    purchases_file = FILINGS_DIR / "purchases_only.json"
    if purchases_file.exists():
        with open(purchases_file) as f:
            return json.load(f)

    # Fallback to master file
    master_file = FILINGS_DIR / "all_transactions.json"
    if master_file.exists():
        with open(master_file) as f:
            all_txns = json.load(f)
        return [t for t in all_txns if t.get("transaction_code") == "P"]

    return []


def fetch_historical_prices(ticker: str, start_date: str, end_date: str) -> dict[str, float] | None:
    """
    Fetch daily closing prices from Yahoo Finance.
    Returns dict mapping date string -> close price.
    """
    try:
        # Convert dates to timestamps
        start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
        end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp())

        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        params = {
            "period1": start_ts,
            "period2": end_ts,
            "interval": "1d",
        }
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, params=params, headers=headers, timeout=15)
        time.sleep(YF_DELAY)

        if not resp.ok:
            log.warning(f"Yahoo Finance returned {resp.status_code} for {ticker}")
            return None

        data = resp.json()
        result = data.get("chart", {}).get("result", [])
        if not result:
            return None

        timestamps = result[0].get("timestamp", [])
        closes = result[0].get("indicators", {}).get("quote", [{}])[0].get("close", [])
        # Also get highs for 20-day high calculation
        highs = result[0].get("indicators", {}).get("quote", [{}])[0].get("high", [])

        prices = {}
        high_prices = {}
        for ts, close, high in zip(timestamps, closes, highs):
            if close is not None:
                date_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
                prices[date_str] = close
                if high is not None:
                    high_prices[date_str] = high

        return {"close": prices, "high": high_prices}

    except Exception as e:
        log.warning(f"Price fetch failed for {ticker}: {e}")
        return None


def compute_forward_returns(
    prices: dict[str, float], entry_date: str, horizons: list[int]
) -> dict[int, float | None]:
    """
    Compute forward returns from entry_date for given horizons.
    horizons are in TRADING days.
    """
    sorted_dates = sorted(prices.keys())
    if entry_date not in prices:
        # Find nearest trading day after entry_date
        for d in sorted_dates:
            if d >= entry_date:
                entry_date = d
                break
        else:
            return {h: None for h in horizons}

    entry_idx = sorted_dates.index(entry_date)
    entry_price = prices[entry_date]

    returns = {}
    for h in horizons:
        target_idx = entry_idx + h
        if target_idx < len(sorted_dates):
            target_price = prices[sorted_dates[target_idx]]
            returns[h] = (target_price - entry_price) / entry_price * 100
        else:
            returns[h] = None

    return returns


def compute_dip_at_purchase(high_prices: dict[str, float], close_prices: dict[str, float],
                             purchase_date: str, lookback: int = 20) -> float | None:
    """Compute how far the stock is below its 20-day high at purchase time."""
    sorted_dates = sorted(close_prices.keys())

    # Find purchase date index
    if purchase_date not in close_prices:
        for d in sorted_dates:
            if d >= purchase_date:
                purchase_date = d
                break
        else:
            return None

    idx = sorted_dates.index(purchase_date)
    start_idx = max(0, idx - lookback)

    window_highs = []
    for i in range(start_idx, idx + 1):
        d = sorted_dates[i]
        if d in high_prices:
            window_highs.append(high_prices[d])

    if not window_highs:
        return None

    high_20d = max(window_highs)
    current = close_prices[purchase_date]
    return ((high_20d - current) / high_20d) * 100


def generate_random_baseline(prices: dict[str, float], n_samples: int, horizons: list[int]) -> dict:
    """Generate baseline returns from random entry dates for comparison."""
    sorted_dates = sorted(prices.keys())
    if len(sorted_dates) < max(horizons) + 20:
        return {}

    # Only sample from dates where we can compute all horizons
    valid_range = sorted_dates[20:-max(horizons)]
    if len(valid_range) < n_samples:
        n_samples = len(valid_range)

    rng = np.random.RandomState(42)
    sample_indices = rng.choice(len(valid_range), size=n_samples, replace=True)

    baseline_returns = {h: [] for h in horizons}
    for idx in sample_indices:
        entry_date = valid_range[idx]
        fwd = compute_forward_returns(prices, entry_date, horizons)
        for h in horizons:
            if fwd[h] is not None:
                baseline_returns[h].append(fwd[h])

    return {h: {"mean": np.mean(vals), "median": np.median(vals), "n": len(vals)}
            for h, vals in baseline_returns.items() if vals}


def run():
    """Run the backtest."""
    log.info("=== Insider Buy Signal Backtest ===")

    purchases = load_purchases()
    if not purchases:
        log.error("No purchase data found. Run insider_filing_fetcher.py first.")
        return

    log.info(f"Loaded {len(purchases)} insider purchases")

    # Group by ticker
    by_ticker = defaultdict(list)
    for p in purchases:
        by_ticker[p["ticker"]].append(p)

    log.info(f"Purchases span {len(by_ticker)} tickers")

    # Results accumulators
    results_by_role = defaultdict(lambda: {h: [] for h in HORIZONS})
    results_by_size = {"small": {h: [] for h in HORIZONS},
                       "medium": {h: [] for h in HORIZONS},
                       "large": {h: [] for h in HORIZONS}}
    results_by_dip = {"no_dip": {h: [] for h in HORIZONS},
                      "moderate_dip": {h: [] for h in HORIZONS},
                      "deep_dip": {h: [] for h in HORIZONS}}
    results_overall = {h: [] for h in HORIZONS}
    baseline_overall = {h: [] for h in HORIZONS}
    all_events = []

    # Process each ticker
    for ticker_idx, (ticker, ticker_purchases) in enumerate(sorted(by_ticker.items())):
        log.info(f"[{ticker_idx+1}/{len(by_ticker)}] Processing {ticker} ({len(ticker_purchases)} purchases)")

        # Determine date range needed
        earliest = min(p["transaction_date"] for p in ticker_purchases)
        latest = max(p["transaction_date"] for p in ticker_purchases)

        # Need prices from 20 days before earliest to 60 trading days after latest
        start_dt = datetime.strptime(earliest, "%Y-%m-%d") - timedelta(days=40)
        end_dt = datetime.strptime(latest, "%Y-%m-%d") + timedelta(days=100)
        # Cap end date to today
        end_dt = min(end_dt, datetime.now())

        price_data = fetch_historical_prices(
            ticker,
            start_dt.strftime("%Y-%m-%d"),
            end_dt.strftime("%Y-%m-%d"),
        )

        if not price_data:
            log.warning(f"  No price data for {ticker}, skipping")
            continue

        close_prices = price_data["close"]
        high_prices = price_data["high"]

        if len(close_prices) < 30:
            log.warning(f"  Insufficient price data for {ticker}")
            continue

        # Generate baseline for this ticker
        baseline = generate_random_baseline(close_prices, min(200, len(close_prices)), HORIZONS)
        for h in HORIZONS:
            if h in baseline and baseline[h]["n"] > 0:
                # Add baseline samples
                baseline_overall[h].append(baseline[h]["mean"])

        # Process each purchase
        for p in ticker_purchases:
            txn_date = p["transaction_date"]
            role = p.get("role_bucket", "Other")
            value = p.get("total_value", 0)

            # Compute forward returns
            fwd = compute_forward_returns(close_prices, txn_date, HORIZONS)

            # Compute dip depth at purchase
            dip = compute_dip_at_purchase(high_prices, close_prices, txn_date)

            event = {
                "ticker": ticker,
                "insider": p.get("insider_name", ""),
                "role": role,
                "value": value,
                "date": txn_date,
                "dip_pct": round(dip, 2) if dip else 0,
                "fwd_returns": {str(h): round(r, 4) if r is not None else None
                                for h, r in fwd.items()},
            }
            all_events.append(event)

            # Accumulate by role
            for h in HORIZONS:
                if fwd[h] is not None:
                    results_by_role[role][h].append(fwd[h])
                    results_overall[h].append(fwd[h])

            # Accumulate by purchase size
            if value < 100_000:
                size_bucket = "small"
            elif value < 500_000:
                size_bucket = "medium"
            else:
                size_bucket = "large"
            for h in HORIZONS:
                if fwd[h] is not None:
                    results_by_size[size_bucket][h].append(fwd[h])

            # Accumulate by dip depth
            if dip is not None:
                if dip < 5:
                    dip_bucket = "no_dip"
                elif dip < 15:
                    dip_bucket = "moderate_dip"
                else:
                    dip_bucket = "deep_dip"
                for h in HORIZONS:
                    if fwd[h] is not None:
                        results_by_dip[dip_bucket][h].append(fwd[h])

    # Compile results
    def summarize(returns_dict):
        summary = {}
        for key, horizons_data in returns_dict.items():
            if isinstance(horizons_data, dict):
                summary[key] = {}
                for h, vals in horizons_data.items():
                    if vals:
                        summary[key][f"{h}d"] = {
                            "mean_return_pct": round(np.mean(vals), 3),
                            "median_return_pct": round(np.median(vals), 3),
                            "win_rate_pct": round(np.mean([1 for v in vals if v > 0]) * 100, 1),
                            "std_pct": round(np.std(vals), 3),
                            "sharpe": round(np.mean(vals) / np.std(vals), 3) if np.std(vals) > 0 else 0,
                            "n_events": len(vals),
                        }
        return summary

    output = {
        "run_date": datetime.now().isoformat(),
        "total_purchases_analyzed": len(all_events),
        "tickers_analyzed": len(by_ticker),
        "horizons": HORIZONS,
        "overall": {},
        "by_role": summarize(results_by_role),
        "by_purchase_size": summarize(results_by_size),
        "by_dip_depth": summarize(results_by_dip),
        "baseline_comparison": {},
        "events": all_events,
    }

    # Overall summary
    for h in HORIZONS:
        vals = results_overall[h]
        if vals:
            output["overall"][f"{h}d"] = {
                "mean_return_pct": round(np.mean(vals), 3),
                "median_return_pct": round(np.median(vals), 3),
                "win_rate_pct": round(np.mean([1 for v in vals if v > 0]) * 100, 1),
                "std_pct": round(np.std(vals), 3),
                "sharpe": round(np.mean(vals) / np.std(vals), 3) if np.std(vals) > 0 else 0,
                "n_events": len(vals),
            }

    # Baseline comparison
    for h in HORIZONS:
        insider_vals = results_overall[h]
        baseline_vals = baseline_overall[h]
        if insider_vals and baseline_vals:
            insider_mean = np.mean(insider_vals)
            baseline_mean = np.mean(baseline_vals)
            output["baseline_comparison"][f"{h}d"] = {
                "insider_mean_pct": round(insider_mean, 3),
                "baseline_mean_pct": round(baseline_mean, 3),
                "excess_return_pct": round(insider_mean - baseline_mean, 3),
                "insider_n": len(insider_vals),
            }

    # Save results
    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2, default=str)
    log.info(f"Results saved to {RESULTS_FILE}")

    # Print report
    print("\n" + "=" * 70)
    print("INSIDER BUY SIGNAL BACKTEST RESULTS")
    print("=" * 70)

    print(f"\nTotal purchases analyzed: {len(all_events)}")
    print(f"Tickers covered: {len(by_ticker)}")

    print("\n--- OVERALL FORWARD RETURNS AFTER INSIDER BUYS ---")
    print(f"{'Horizon':<10} {'Mean%':<10} {'Median%':<10} {'WinRate%':<10} {'Sharpe':<10} {'N':<8}")
    for h in HORIZONS:
        if f"{h}d" in output["overall"]:
            d = output["overall"][f"{h}d"]
            print(f"{h}d{'':<7} {d['mean_return_pct']:>8.3f}  {d['median_return_pct']:>8.3f}  "
                  f"{d['win_rate_pct']:>8.1f}  {d['sharpe']:>8.3f}  {d['n_events']:<8}")

    print("\n--- vs BASELINE (random entry) ---")
    print(f"{'Horizon':<10} {'Insider%':<10} {'Baseline%':<10} {'Excess%':<10}")
    for h in HORIZONS:
        if f"{h}d" in output["baseline_comparison"]:
            d = output["baseline_comparison"][f"{h}d"]
            print(f"{h}d{'':<7} {d['insider_mean_pct']:>8.3f}  {d['baseline_mean_pct']:>8.3f}  "
                  f"{d['excess_return_pct']:>+8.3f}")

    print("\n--- BY INSIDER ROLE ---")
    for role in ["CEO", "CFO", "Director", "VP", "Other"]:
        if role in output["by_role"]:
            print(f"\n  {role}:")
            print(f"  {'Horizon':<10} {'Mean%':<10} {'WinRate%':<10} {'N':<8}")
            for h in HORIZONS:
                key = f"{h}d"
                if key in output["by_role"][role]:
                    d = output["by_role"][role][key]
                    print(f"  {h}d{'':<7} {d['mean_return_pct']:>8.3f}  {d['win_rate_pct']:>8.1f}  {d['n_events']:<8}")

    print("\n--- BY PURCHASE SIZE ---")
    for size in ["small", "medium", "large"]:
        if size in output["by_purchase_size"]:
            label = {"small": "<$100K", "medium": "$100K-$500K", "large": ">$500K"}[size]
            print(f"\n  {label}:")
            print(f"  {'Horizon':<10} {'Mean%':<10} {'WinRate%':<10} {'N':<8}")
            for h in HORIZONS:
                key = f"{h}d"
                if key in output["by_purchase_size"][size]:
                    d = output["by_purchase_size"][size][key]
                    print(f"  {h}d{'':<7} {d['mean_return_pct']:>8.3f}  {d['win_rate_pct']:>8.1f}  {d['n_events']:<8}")

    print("\n--- BY DIP DEPTH AT PURCHASE ---")
    for dip in ["no_dip", "moderate_dip", "deep_dip"]:
        if dip in output["by_dip_depth"]:
            label = {"no_dip": "<5% from high", "moderate_dip": "5-15% from high",
                     "deep_dip": ">15% from high"}[dip]
            print(f"\n  {label}:")
            print(f"  {'Horizon':<10} {'Mean%':<10} {'WinRate%':<10} {'N':<8}")
            for h in HORIZONS:
                key = f"{h}d"
                if key in output["by_dip_depth"][dip]:
                    d = output["by_dip_depth"][dip][key]
                    print(f"  {h}d{'':<7} {d['mean_return_pct']:>8.3f}  {d['win_rate_pct']:>8.1f}  {d['n_events']:<8}")

    print("\n" + "=" * 70)
    return output


if __name__ == "__main__":
    run()
