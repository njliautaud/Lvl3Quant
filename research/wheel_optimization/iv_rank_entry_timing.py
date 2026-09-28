#!/usr/bin/env python3
"""
IV Rank Entry Timing Research for Wheel Strategy.

Tests the hypothesis: selling puts when IV rank is elevated (>50th percentile)
captures more premium with similar or lower assignment risk.

Research questions:
  1. Does entering CSPs only when IV rank > X improve Sharpe?
  2. What is the optimal IV rank threshold for each ticker category?
  3. How does IV rank interact with DTE selection?
  4. Should we use IV rank or IV percentile?

Uses yfinance for historical vol data and estimates IV rank from
realized vol history as a proxy.

Usage:
    python iv_rank_entry_timing.py --tickers AAPL,AMD,TSLA --months 24
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path("/home/jupiter/teleclaude-main")
sys.path.insert(0, str(REPO_ROOT))


def compute_iv_rank_series(prices: np.ndarray, lookback: int = 252, window: int = 20) -> np.ndarray:
    """
    Compute IV rank proxy from historical volatility.

    IV rank = (current_HV - min_HV_1yr) / (max_HV_1yr - min_HV_1yr)

    Returns array of same length as prices, with NaN for insufficient data.
    """
    n = len(prices)
    iv_rank = np.full(n, np.nan)

    for i in range(lookback + window, n):
        # Current HV (20-day)
        rets = np.diff(np.log(prices[i-window:i+1]))
        current_hv = np.std(rets) * np.sqrt(252)

        # 1-year HV history (rolling 20-day HV for past year)
        hv_history = []
        for j in range(i - lookback, i - window + 1):
            r = np.diff(np.log(prices[j:j+window+1]))
            if len(r) >= window:
                hv_history.append(np.std(r) * np.sqrt(252))

        if len(hv_history) < 10:
            continue

        min_hv = min(hv_history)
        max_hv = max(hv_history)

        if max_hv - min_hv > 0.001:
            iv_rank[i] = (current_hv - min_hv) / (max_hv - min_hv)
        else:
            iv_rank[i] = 0.5

    return iv_rank


def simulate_csp_with_iv_filter(
    prices: np.ndarray,
    dates: list,
    iv_rank: np.ndarray,
    iv_threshold: float,
    csp_delta: float = 0.25,
    target_dte: int = 14,
    early_close: float = 0.75,
    capital: float = 100_000,
) -> dict[str, Any]:
    """
    Simulate CSP selling with IV rank filter.

    Only enters new CSPs when IV rank > threshold.
    Tracks premium collected, assignment rate, and P&L.
    """
    n = len(prices)
    equity = capital
    equity_curve = [equity]
    trades = []
    active_position = None
    total_premium = 0.0
    assignments = 0
    expirations = 0
    trades_entered = 0
    skipped_due_to_iv = 0

    i = max(252 + 20, 0)  # Start after enough history for IV rank

    while i < n:
        price = prices[i]

        # Check if we have an active position
        if active_position is not None:
            pos = active_position
            days_held = i - pos["entry_idx"]

            if days_held >= pos["dte"]:
                # Expiration day
                if price < pos["strike"]:
                    # Assigned — stock below strike
                    loss = (pos["strike"] - price) * 100 * pos["contracts"]
                    net = pos["premium"] - loss
                    equity += net
                    assignments += 1
                    trades.append({
                        "type": "CSP_ASSIGNED",
                        "entry_date": str(dates[pos["entry_idx"]]),
                        "exit_date": str(dates[i]),
                        "premium": pos["premium"],
                        "loss": loss,
                        "net": net,
                        "iv_rank_at_entry": pos["iv_rank"],
                    })
                else:
                    # Expired worthless — keep premium
                    equity += pos["premium"]
                    expirations += 1
                    trades.append({
                        "type": "CSP_EXPIRED",
                        "entry_date": str(dates[pos["entry_idx"]]),
                        "exit_date": str(dates[i]),
                        "premium": pos["premium"],
                        "net": pos["premium"],
                        "iv_rank_at_entry": pos["iv_rank"],
                    })
                total_premium += pos["premium"]
                active_position = None
            else:
                # Check for early close
                remaining_dte = pos["dte"] - days_held
                time_decay_factor = remaining_dte / pos["dte"]
                moneyness = price / pos["strike"]

                if moneyness > 1.0 + 0.05:  # Well OTM, option lost most value
                    current_value_est = pos["premium"] * time_decay_factor * 0.3
                    profit_pct = 1.0 - (current_value_est / pos["premium"])
                    if profit_pct >= early_close:
                        # Early close
                        net = pos["premium"] * early_close
                        equity += net
                        expirations += 1
                        total_premium += net
                        trades.append({
                            "type": "CSP_EARLY_CLOSE",
                            "entry_date": str(dates[pos["entry_idx"]]),
                            "exit_date": str(dates[i]),
                            "premium": net,
                            "net": net,
                            "iv_rank_at_entry": pos["iv_rank"],
                        })
                        active_position = None

        # Try to enter new position if none active
        if active_position is None and i < n - target_dte:
            current_iv_rank = iv_rank[i]

            if np.isnan(current_iv_rank) or current_iv_rank < iv_threshold:
                skipped_due_to_iv += 1
                i += 1
                equity_curve.append(equity)
                continue

            # Calculate strike (OTM by delta)
            hv = max(0.10, np.std(np.diff(np.log(prices[max(0,i-20):i+1]))) * np.sqrt(252))
            # Approximate strike from delta (simplified)
            strike = price * (1.0 - csp_delta * hv * np.sqrt(target_dte / 252))
            strike = round(strike, 0)

            # Premium estimate (simplified Black-Scholes proxy)
            premium_per_share = price * hv * np.sqrt(target_dte / 365) * 0.4 * (1 + current_iv_rank * 0.3)

            # Position sizing
            contracts = max(1, int(equity * 0.25 / (strike * 100)))

            active_position = {
                "entry_idx": i,
                "strike": strike,
                "dte": target_dte,
                "contracts": contracts,
                "premium": premium_per_share * 100 * contracts,
                "iv_rank": current_iv_rank,
            }
            trades_entered += 1

        equity_curve.append(equity)
        i += 1

    # Compute metrics
    arr = np.array(equity_curve)
    total_ret = (arr[-1] / arr[0]) - 1.0
    daily_rets = np.diff(arr) / arr[:-1]
    daily_rets = daily_rets[np.isfinite(daily_rets)]

    sharpe = 0.0
    sortino = 0.0
    if len(daily_rets) > 1:
        mean_r = np.mean(daily_rets)
        std_r = np.std(daily_rets, ddof=1)
        if std_r > 0:
            sharpe = mean_r / std_r * np.sqrt(252)
        down = daily_rets[daily_rets < 0]
        dd_std = np.std(down, ddof=1) if len(down) > 1 else std_r
        if dd_std > 0:
            sortino = mean_r / dd_std * np.sqrt(252)

    peak = np.maximum.accumulate(arr)
    max_dd = np.min((arr - peak) / peak) * 100

    return {
        "iv_threshold": iv_threshold,
        "csp_delta": csp_delta,
        "target_dte": target_dte,
        "total_return_pct": round(total_ret * 100, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_dd_pct": round(max_dd, 2),
        "trades_entered": trades_entered,
        "assignments": assignments,
        "expirations": expirations,
        "skipped_due_to_iv": skipped_due_to_iv,
        "total_premium": round(total_premium, 2),
        "assignment_rate": round(assignments / max(1, trades_entered) * 100, 1),
        "win_rate": round(expirations / max(1, trades_entered) * 100, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", type=str, default="AAPL,AMD,TSLA,BAC,PLTR")
    parser.add_argument("--months", type=int, default=24)
    args = parser.parse_args()

    if not HAS_YF:
        print("ERROR: yfinance required. Install with: pip install yfinance")
        return

    tickers = args.tickers.split(",")
    iv_thresholds = [0.0, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]

    all_results = {}
    print(f"Testing IV rank entry timing for {tickers}")
    print(f"IV thresholds: {iv_thresholds}")
    print(f"Window: {args.months} months\n")

    for ticker in tickers:
        print(f"\n--- {ticker} ---")
        try:
            end = date.today()
            start = end - timedelta(days=(args.months + 14) * 30)
            df = yf.download(ticker, start=str(start), end=str(end), progress=False, auto_adjust=True)
            if df.empty:
                print(f"  No data for {ticker}")
                continue

            if hasattr(df.columns, 'levels') and len(df.columns.levels) > 1:
                df.columns = df.columns.get_level_values(0)

            prices = df["Close"].values.astype(float)
            dates_list = df.index.tolist()

            # Compute IV rank
            iv_rank = compute_iv_rank_series(prices)

            results_for_ticker = []
            for threshold in iv_thresholds:
                result = simulate_csp_with_iv_filter(
                    prices=prices,
                    dates=dates_list,
                    iv_rank=iv_rank,
                    iv_threshold=threshold,
                )
                result["ticker"] = ticker
                results_for_ticker.append(result)

                print(f"  IVR>{threshold:.0%}: Sharpe={result['sharpe']:+.3f}  "
                      f"Return={result['total_return_pct']:+.1f}%  "
                      f"DD={result['max_dd_pct']:.1f}%  "
                      f"Trades={result['trades_entered']}  "
                      f"AssignRate={result['assignment_rate']:.0f}%")

            all_results[ticker] = results_for_ticker

        except Exception as e:
            print(f"  Error: {e}")

    # Save
    out_file = SCRIPT_DIR / "iv_rank_entry_results.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_file}")

    # Summary
    print("\n" + "=" * 70)
    print("IV RANK ENTRY TIMING — SUMMARY")
    print("=" * 70)
    print("\nOptimal IV rank threshold per ticker (by Sharpe):")
    for ticker, results in all_results.items():
        best = max(results, key=lambda x: x.get("sharpe", 0))
        baseline = results[0]  # threshold=0 is baseline
        delta_sharpe = best["sharpe"] - baseline["sharpe"]
        print(f"  {ticker:>6s}: best IVR>{best['iv_threshold']:.0%}  "
              f"Sharpe={best['sharpe']:+.3f} (vs baseline {baseline['sharpe']:+.3f}, "
              f"delta={delta_sharpe:+.3f})")


if __name__ == "__main__":
    main()
