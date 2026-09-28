#!/usr/bin/env python3
"""
Industry/Sector Performance Monitor
====================================
Downloads sector ETF prices, calculates momentum, relative strength,
trend status, and regime classification. Also runs a historical analysis
of sector leadership patterns and subsequent SPY returns.

Output: output/growth_research/industry_monitor/latest_snapshot.json
"""

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SECTOR_ETFS = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLE": "Energy",
    "XLV": "Health Care",
    "XLI": "Industrials",
    "XLY": "Consumer Disc.",
    "XLC": "Comm. Services",
    "XLP": "Consumer Staples",
    "XLRE": "Real Estate",
    "XLB": "Materials",
    "XLU": "Utilities",
}
BENCHMARK = "SPY"
ALL_TICKERS = list(SECTOR_ETFS.keys()) + [BENCHMARK]

MOMENTUM_WINDOWS = {
    "1w": 5,
    "1m": 21,
    "3m": 63,
    "6m": 126,
}

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "output" / "growth_research" / "industry_monitor"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------
def download_data(lookback_years: int = 3) -> pd.DataFrame:
    """Download adjusted close prices for all tickers."""
    end = datetime.now()
    start = end - timedelta(days=lookback_years * 365)
    print(f"Downloading {len(ALL_TICKERS)} tickers from {start.date()} to {end.date()} ...")
    data = yf.download(ALL_TICKERS, start=start, end=end, auto_adjust=True, progress=False)
    # yfinance returns multi-level columns; grab Close
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data[["Close"]].rename(columns={"Close": ALL_TICKERS[0]})
    prices = prices.dropna(how="all")
    print(f"  Got {len(prices)} trading days, {prices.shape[1]} tickers")
    return prices


# ---------------------------------------------------------------------------
# Momentum & relative strength
# ---------------------------------------------------------------------------
def calc_momentum(prices: pd.DataFrame) -> dict:
    """Calculate momentum returns for each ticker over standard windows."""
    results = {}
    latest = prices.iloc[-1]
    for label, days in MOMENTUM_WINDOWS.items():
        if len(prices) < days:
            continue
        past = prices.iloc[-days - 1]  # price 'days' ago
        ret = (latest / past - 1) * 100  # percent
        results[label] = ret
    return results


def calc_relative_strength(momentum: dict, benchmark: str = BENCHMARK) -> dict:
    """Sector return minus SPY return for each window."""
    rs = {}
    for window, rets in momentum.items():
        spy_ret = rets.get(benchmark, 0)
        rs[window] = {t: round(r - spy_ret, 2) for t, r in rets.items() if t != benchmark}
    return rs


# ---------------------------------------------------------------------------
# Volatility & trend
# ---------------------------------------------------------------------------
def calc_rolling_vol(prices: pd.DataFrame, window: int = 20) -> pd.Series:
    """Annualized rolling volatility (20-day)."""
    log_ret = np.log(prices / prices.shift(1))
    vol = log_ret.rolling(window).std() * np.sqrt(252) * 100  # annualized %
    return vol.iloc[-1]


def calc_trend(prices: pd.DataFrame) -> dict:
    """Above/below 50-day and 200-day SMA for each ticker."""
    sma50 = prices.rolling(50).mean().iloc[-1]
    sma200 = prices.rolling(200).mean().iloc[-1]
    latest = prices.iloc[-1]
    trend = {}
    for t in prices.columns:
        above_50 = bool(latest[t] > sma50[t]) if pd.notna(sma50[t]) else None
        above_200 = bool(latest[t] > sma200[t]) if pd.notna(sma200[t]) else None
        trend[t] = {"above_50sma": above_50, "above_200sma": above_200}
    return trend


# ---------------------------------------------------------------------------
# Regime classification
# ---------------------------------------------------------------------------
def classify_regime(trend: dict, sector_tickers: list) -> str:
    """
    Broad Bull:  >7 sectors above 200 SMA
    Mixed:       4-7 sectors above 200 SMA
    Broad Bear:  <4 sectors above 200 SMA
    """
    count = sum(1 for t in sector_tickers if trend.get(t, {}).get("above_200sma", False))
    if count > 7:
        regime = "BROAD BULL"
    elif count >= 4:
        regime = "MIXED"
    else:
        regime = "BROAD BEAR"
    return regime, count


# ---------------------------------------------------------------------------
# Leadership / lagging identification
# ---------------------------------------------------------------------------
def identify_leaders_laggers(momentum: dict, sector_tickers: list, window: str = "1m"):
    """Rank sectors by momentum for a given window, return leaders and laggers."""
    rets = momentum.get(window, {})
    sector_rets = {t: rets[t] for t in sector_tickers if t in rets}
    ranked = sorted(sector_rets.items(), key=lambda x: x[1], reverse=True)
    leaders = ranked[:3]
    laggers = ranked[-3:]
    return leaders, laggers


# ---------------------------------------------------------------------------
# Historical analysis: sector leadership -> subsequent SPY returns
# ---------------------------------------------------------------------------
def historical_leadership_analysis(prices: pd.DataFrame, sector_tickers: list) -> dict:
    """
    For each month in history, identify the top-3 leading sectors (by 1m momentum).
    Then measure SPY's forward returns over 1w, 2w, 3w, 4w.
    Group by which sectors were leading and compute average forward SPY return.
    """
    results = {"by_leader_sector": {}, "regime_forward_returns": {}}

    if len(prices) < 252:
        return results

    spy = prices[BENCHMARK]

    # Monthly sampling points (every 21 days)
    sample_indices = list(range(126, len(prices) - 21, 21))  # start after 6m warmup

    forward_windows = {"1w": 5, "2w": 10, "3w": 15, "4w": 20}

    # Track: for each sector, when it's a top-3 leader, what are SPY forward returns?
    sector_forward = {t: {fw: [] for fw in forward_windows} for t in sector_tickers}
    regime_forward = {"BROAD BULL": {fw: [] for fw in forward_windows},
                      "MIXED": {fw: [] for fw in forward_windows},
                      "BROAD BEAR": {fw: [] for fw in forward_windows}}

    for idx in sample_indices:
        # 1-month momentum at this point
        if idx < 21:
            continue
        current = prices.iloc[idx]
        past_1m = prices.iloc[idx - 21]

        mom = {}
        for t in sector_tickers:
            if pd.notna(current[t]) and pd.notna(past_1m[t]) and past_1m[t] > 0:
                mom[t] = (current[t] / past_1m[t] - 1) * 100

        if len(mom) < 6:
            continue

        ranked = sorted(mom.items(), key=lambda x: x[1], reverse=True)
        top3 = [r[0] for r in ranked[:3]]

        # 200 SMA regime at this point
        if idx >= 200:
            sma200 = prices.iloc[idx - 200:idx].mean()
            count_above = sum(1 for t in sector_tickers
                           if pd.notna(current[t]) and pd.notna(sma200[t]) and current[t] > sma200[t])
            if count_above > 7:
                regime = "BROAD BULL"
            elif count_above >= 4:
                regime = "MIXED"
            else:
                regime = "BROAD BEAR"
        else:
            regime = "MIXED"

        # Forward SPY returns
        for fw_label, fw_days in forward_windows.items():
            future_idx = idx + fw_days
            if future_idx < len(spy) and pd.notna(spy.iloc[future_idx]) and pd.notna(spy.iloc[idx]) and spy.iloc[idx] > 0:
                fwd_ret = (spy.iloc[future_idx] / spy.iloc[idx] - 1) * 100

                for t in top3:
                    sector_forward[t][fw_label].append(fwd_ret)

                regime_forward[regime][fw_label].append(fwd_ret)

    # Aggregate
    for t in sector_tickers:
        sector_stats = {}
        for fw, vals in sector_forward[t].items():
            if vals:
                sector_stats[fw] = {
                    "avg_spy_return_pct": round(np.mean(vals), 3),
                    "median_spy_return_pct": round(np.median(vals), 3),
                    "win_rate_pct": round(sum(1 for v in vals if v > 0) / len(vals) * 100, 1),
                    "n_obs": len(vals),
                }
        results["by_leader_sector"][f"{t} ({SECTOR_ETFS.get(t, t)})"] = sector_stats

    for regime, fws in regime_forward.items():
        regime_stats = {}
        for fw, vals in fws.items():
            if vals:
                regime_stats[fw] = {
                    "avg_spy_return_pct": round(np.mean(vals), 3),
                    "median_spy_return_pct": round(np.median(vals), 3),
                    "win_rate_pct": round(sum(1 for v in vals if v > 0) / len(vals) * 100, 1),
                    "n_obs": len(vals),
                }
        results["regime_forward_returns"][regime] = regime_stats

    return results


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
def print_summary(sector_tickers, momentum, rel_strength, vol, trend, regime, regime_count,
                  leaders, laggers, names):
    """Print a clean console summary."""
    print("\n" + "=" * 90)
    print(f"  SECTOR PERFORMANCE MONITOR  |  {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  Regime: {regime} ({regime_count}/11 above 200 SMA)")
    print("=" * 90)

    # Header
    header = f"{'Ticker':<6} {'Sector':<18} {'1w%':>6} {'1m%':>6} {'3m%':>7} {'6m%':>7} {'RS_1m':>6} {'Vol20':>6} {'50SMA':>6} {'200SMA':>7}"
    print(header)
    print("-" * 90)

    # Sort by 1m momentum descending
    mom_1m = momentum.get("1m", {})
    sorted_tickers = sorted(sector_tickers, key=lambda t: mom_1m.get(t, -999), reverse=True)

    for t in sorted_tickers:
        name = names.get(t, t)[:17]
        w1 = momentum.get("1w", {}).get(t, float("nan"))
        m1 = momentum.get("1m", {}).get(t, float("nan"))
        m3 = momentum.get("3m", {}).get(t, float("nan"))
        m6 = momentum.get("6m", {}).get(t, float("nan"))
        rs = rel_strength.get("1m", {}).get(t, float("nan"))
        v = vol.get(t, float("nan"))
        t50 = "UP" if trend.get(t, {}).get("above_50sma") else "DOWN"
        t200 = "UP" if trend.get(t, {}).get("above_200sma") else "DOWN"

        print(f"{t:<6} {name:<18} {w1:>6.1f} {m1:>6.1f} {m3:>7.1f} {m6:>7.1f} {rs:>6.1f} {v:>6.1f} {t50:>6} {t200:>7}")

    # SPY benchmark
    print("-" * 90)
    t = BENCHMARK
    w1 = momentum.get("1w", {}).get(t, float("nan"))
    m1 = momentum.get("1m", {}).get(t, float("nan"))
    m3 = momentum.get("3m", {}).get(t, float("nan"))
    m6 = momentum.get("6m", {}).get(t, float("nan"))
    v = vol.get(t, float("nan"))
    t50 = "UP" if trend.get(t, {}).get("above_50sma") else "DOWN"
    t200 = "UP" if trend.get(t, {}).get("above_200sma") else "DOWN"
    print(f"{t:<6} {'S&P 500':<18} {w1:>6.1f} {m1:>6.1f} {m3:>7.1f} {m6:>7.1f} {'--':>6} {v:>6.1f} {t50:>6} {t200:>7}")

    print("\n  LEADERS (1m):", ", ".join(f"{t} ({names.get(t,'')}: {r:+.1f}%)" for t, r in leaders))
    print("  LAGGERS (1m):", ", ".join(f"{t} ({names.get(t,'')}: {r:+.1f}%)" for t, r in laggers))
    print()


def print_historical_summary(hist: dict):
    """Print key findings from historical leadership analysis."""
    print("=" * 90)
    print("  HISTORICAL ANALYSIS: When sector X leads (1m), what happens to SPY next?")
    print("=" * 90)

    # Regime forward returns
    regime_data = hist.get("regime_forward_returns", {})
    if regime_data:
        print(f"\n  {'Regime':<14} {'1w avg%':>8} {'2w avg%':>8} {'3w avg%':>8} {'4w avg%':>8} {'4w WR%':>7} {'N':>5}")
        print("  " + "-" * 60)
        for regime in ["BROAD BULL", "MIXED", "BROAD BEAR"]:
            rd = regime_data.get(regime, {})
            w1 = rd.get("1w", {}).get("avg_spy_return_pct", float("nan"))
            w2 = rd.get("2w", {}).get("avg_spy_return_pct", float("nan"))
            w3 = rd.get("3w", {}).get("avg_spy_return_pct", float("nan"))
            w4 = rd.get("4w", {}).get("avg_spy_return_pct", float("nan"))
            wr = rd.get("4w", {}).get("win_rate_pct", float("nan"))
            n = rd.get("4w", {}).get("n_obs", 0)
            print(f"  {regime:<14} {w1:>+8.2f} {w2:>+8.2f} {w3:>+8.2f} {w4:>+8.2f} {wr:>7.1f} {n:>5}")

    # Sector leadership -> SPY returns (show top insights)
    sector_data = hist.get("by_leader_sector", {})
    if sector_data:
        print(f"\n  When this sector leads (top-3 by 1m mom), SPY 4-week forward return:")
        print(f"  {'Sector':<30} {'avg%':>7} {'med%':>7} {'WR%':>6} {'N':>5}")
        print("  " + "-" * 58)
        # Sort by 4w avg return
        items = []
        for sector, stats in sector_data.items():
            w4 = stats.get("4w", {})
            if w4:
                items.append((sector, w4))
        items.sort(key=lambda x: x[1].get("avg_spy_return_pct", 0), reverse=True)
        for sector, w4 in items:
            avg = w4.get("avg_spy_return_pct", 0)
            med = w4.get("median_spy_return_pct", 0)
            wr = w4.get("win_rate_pct", 0)
            n = w4.get("n_obs", 0)
            print(f"  {sector:<30} {avg:>+7.2f} {med:>+7.2f} {wr:>6.1f} {n:>5}")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    sector_tickers = list(SECTOR_ETFS.keys())

    # Download
    prices = download_data(lookback_years=3)

    # Momentum
    momentum_raw = calc_momentum(prices)
    momentum = {}
    for window, rets in momentum_raw.items():
        momentum[window] = {t: round(float(rets[t]), 2) if pd.notna(rets[t]) else None for t in rets.index}

    # Relative strength
    rel_strength = calc_relative_strength(momentum)

    # Volatility
    vol_raw = calc_rolling_vol(prices)
    vol = {t: round(float(vol_raw[t]), 2) if pd.notna(vol_raw[t]) else None for t in vol_raw.index}

    # Trend
    trend = calc_trend(prices)

    # Regime
    regime, regime_count = classify_regime(trend, sector_tickers)

    # Leaders / laggers
    leaders, laggers = identify_leaders_laggers(momentum, sector_tickers, "1m")

    # Historical analysis
    hist = historical_leadership_analysis(prices, sector_tickers)

    # Print summary
    print_summary(sector_tickers, momentum, rel_strength, vol, trend, regime, regime_count,
                  leaders, laggers, SECTOR_ETFS)
    print_historical_summary(hist)

    # Build snapshot
    snapshot = {
        "timestamp": datetime.now().isoformat(),
        "regime": regime,
        "sectors_above_200sma": regime_count,
        "sectors": {},
        "benchmark": {
            "ticker": BENCHMARK,
            "momentum": {w: momentum[w].get(BENCHMARK) for w in momentum},
            "volatility_20d_ann": vol.get(BENCHMARK),
            "trend": trend.get(BENCHMARK),
        },
        "leaders_1m": [{"ticker": t, "name": SECTOR_ETFS.get(t), "return_pct": r} for t, r in leaders],
        "laggers_1m": [{"ticker": t, "name": SECTOR_ETFS.get(t), "return_pct": r} for t, r in laggers],
        "historical_analysis": hist,
    }

    for t in sector_tickers:
        snapshot["sectors"][t] = {
            "name": SECTOR_ETFS[t],
            "momentum": {w: momentum[w].get(t) for w in momentum},
            "relative_strength_vs_spy": {w: rel_strength[w].get(t) for w in rel_strength},
            "volatility_20d_ann": vol.get(t),
            "trend": trend.get(t),
        }

    # Save
    out_path = OUTPUT_DIR / "latest_snapshot.json"
    with open(out_path, "w") as f:
        json.dump(snapshot, f, indent=2, default=str)
    print(f"Snapshot saved to {out_path}")

    return snapshot


if __name__ == "__main__":
    main()
