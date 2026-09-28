#!/usr/bin/env python3
"""
Trend-Following Scanner (Multi-Asset)
=======================================
Simple but powerful: be long when price > 200-day MA, flat when below.
Applied across equities, bonds, commodities, currencies via ETFs.

This is CRISIS ALPHA — makes money when buy-and-hold gets crushed.
Historically 12-18% CAGR with much lower drawdowns than buy-and-hold.

Scoring:
  - Trend strength: distance from 200MA (normalized)
  - Multi-timeframe confirmation: 50MA vs 200MA alignment
  - Momentum: rate of change of the trend
  - Volume confirmation: above-average volume on trend days

Usage:
  python growth/trend_scanner.py
"""

import json
import os
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")


# ── Multi-Asset ETF Universe ─────────────────────────────────────────────────

ASSET_CLASSES = {
    "US Large Cap": ["SPY", "QQQ", "IVV", "VOO"],
    "US Mid Cap": ["IJH", "MDY", "VO"],
    "US Small Cap": ["IWM", "IJR", "VB"],
    "International Dev": ["EFA", "VEA", "IEFA"],
    "Emerging Markets": ["EEM", "VWO", "IEMG"],
    "US Bonds": ["TLT", "IEF", "AGG", "BND", "SHY"],
    "TIPS": ["TIP", "SCHP"],
    "Corporate Bonds": ["LQD", "HYG", "JNK"],
    "Commodities": ["GLD", "SLV", "USO", "DBA", "DBC"],
    "Real Estate": ["VNQ", "IYR", "XLRE"],
    "Sectors": ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC"],
    "Crypto": ["BITO"],
    "Volatility": ["VIXY"],
    "Currency": ["UUP", "FXE", "FXY", "FXB"],
}


def get_all_etfs():
    """Flatten asset class dict into unique ETF list."""
    etfs = set()
    for class_etfs in ASSET_CLASSES.values():
        etfs.update(class_etfs)
    return sorted(etfs)


def get_asset_class(ticker: str) -> str:
    """Look up which asset class a ticker belongs to."""
    for cls, tickers in ASSET_CLASSES.items():
        if ticker in tickers:
            return cls
    return "Unknown"


def analyze_trend(ticker: str, hist: pd.DataFrame) -> dict:
    """
    Analyze trend status for a single ETF.

    Returns dict with trend metrics or None.
    """
    if hist is None or len(hist) < 200:
        return None

    close = hist["Close"]
    volume = hist["Volume"]

    # Moving averages
    ma_200 = close.rolling(200).mean()
    ma_50 = close.rolling(50).mean()
    ma_20 = close.rolling(20).mean()

    current_price = close.iloc[-1]
    current_ma200 = ma_200.iloc[-1]
    current_ma50 = ma_50.iloc[-1]
    current_ma20 = ma_20.iloc[-1]

    if pd.isna(current_ma200):
        return None

    # Trend signal
    above_200 = current_price > current_ma200
    above_50 = current_price > current_ma50
    ma50_above_200 = current_ma50 > current_ma200  # Golden cross

    # Trend strength: % distance from 200MA
    pct_from_200 = (current_price / current_ma200 - 1) * 100

    # Momentum: 200MA slope (annualized rate of change)
    if len(ma_200.dropna()) >= 21:
        ma200_slope = (ma_200.iloc[-1] / ma_200.iloc[-21] - 1) * 12 * 100
    else:
        ma200_slope = 0

    # Volume confirmation
    avg_vol_50 = volume.iloc[-50:].mean()
    recent_vol = volume.iloc[-5:].mean()
    vol_ratio = recent_vol / avg_vol_50 if avg_vol_50 > 0 else 1.0

    # Multi-timeframe score
    # +1 for each bullish signal, -1 for each bearish
    signals = 0
    signals += 1 if above_200 else -1
    signals += 1 if above_50 else -1
    signals += 1 if ma50_above_200 else -1
    signals += 1 if current_price > current_ma20 else -1

    # Returns
    ret_1m = (close.iloc[-1] / close.iloc[-21] - 1) * 100 if len(close) > 21 else 0
    ret_3m = (close.iloc[-1] / close.iloc[-63] - 1) * 100 if len(close) > 63 else 0
    ret_6m = (close.iloc[-1] / close.iloc[-126] - 1) * 100 if len(close) > 126 else 0
    ret_1y = (close.iloc[-1] / close.iloc[-252] - 1) * 100 if len(close) > 252 else 0

    # Drawdown from peak
    peak = close.iloc[-252:].max()
    dd_from_peak = (current_price / peak - 1) * 100

    return {
        "ticker": ticker,
        "asset_class": get_asset_class(ticker),
        "price": round(current_price, 2),
        "ma_200": round(current_ma200, 2),
        "pct_from_200ma": round(pct_from_200, 2),
        "above_200ma": above_200,
        "above_50ma": above_50,
        "golden_cross": ma50_above_200,
        "trend_signals": signals,  # -4 to +4
        "trend_label": "STRONG UP" if signals >= 3 else
                       "UP" if signals >= 1 else
                       "FLAT" if signals == 0 else
                       "DOWN" if signals >= -2 else "STRONG DOWN",
        "ma200_slope_ann": round(ma200_slope, 2),
        "vol_ratio": round(vol_ratio, 2),
        "ret_1m": round(ret_1m, 2),
        "ret_3m": round(ret_3m, 2),
        "ret_6m": round(ret_6m, 2),
        "ret_1y": round(ret_1y, 2),
        "dd_from_peak": round(dd_from_peak, 2),
    }


def run_scanner(output_dir: str = None):
    """Run the trend-following scanner across all asset classes."""
    print(f"\n{'='*70}")
    print(f"  TREND-FOLLOWING SCANNER — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*70}\n")

    etfs = get_all_etfs()
    print(f"[INFO] Scanning {len(etfs)} ETFs across {len(ASSET_CLASSES)} asset classes...")

    # Download all at once
    tickers_str = " ".join(etfs)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=400)

    data = yf.download(tickers_str, start=start_date, end=end_date,
                       progress=False, threads=True)

    results = []
    for ticker in etfs:
        try:
            if isinstance(data.columns, pd.MultiIndex):
                hist = pd.DataFrame({
                    "Close": data["Close"][ticker] if ticker in data["Close"].columns else None,
                    "Volume": data["Volume"][ticker] if ticker in data["Volume"].columns else None,
                }).dropna()
            else:
                hist = data[["Close", "Volume"]].dropna()

            if len(hist) >= 200:
                result = analyze_trend(ticker, hist)
                if result:
                    results.append(result)
        except Exception:
            pass

    if not results:
        print("[ERROR] No results. Check internet connection.")
        return None

    df = pd.DataFrame(results)
    df = df.sort_values("trend_signals", ascending=False).reset_index(drop=True)

    # Display by asset class
    print(f"\n{'='*90}")
    print(f"  TREND STATUS BY ASSET CLASS")
    print(f"{'='*90}")

    for asset_class in sorted(ASSET_CLASSES.keys()):
        class_df = df[df["asset_class"] == asset_class].sort_values(
            "trend_signals", ascending=False)
        if class_df.empty:
            continue

        print(f"\n  {asset_class}:")
        print(f"  {'Ticker':<6} {'Price':>8} {'%200MA':>7} {'Trend':>10} {'Sig':>4} "
              f"{'1M%':>6} {'3M%':>6} {'6M%':>6} {'1Y%':>6} {'DD%':>6}")
        print("  " + "-" * 78)

        for _, row in class_df.iterrows():
            trend_icon = "+" if row["trend_signals"] > 0 else "-" if row["trend_signals"] < 0 else "="
            print(f"  {row['ticker']:<6} {row['price']:8.2f} {row['pct_from_200ma']:+6.1f}% "
                  f"{row['trend_label']:>10} {row['trend_signals']:>+3d}{trend_icon} "
                  f"{row['ret_1m']:+5.1f}% {row['ret_3m']:+5.1f}% "
                  f"{row['ret_6m']:+5.1f}% {row['ret_1y']:+5.1f}% {row['dd_from_peak']:+5.1f}%")

    # Summary
    strong_up = df[df["trend_signals"] >= 3]
    up = df[df["trend_signals"].between(1, 2)]
    down = df[df["trend_signals"] <= -1]

    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  Strong uptrend: {len(strong_up)} ETFs")
    print(f"  Uptrend:        {len(up)} ETFs")
    print(f"  Downtrend:      {len(down)} ETFs")

    if not strong_up.empty:
        print(f"\n  BE LONG (strong uptrend, all signals aligned):")
        for _, row in strong_up.iterrows():
            print(f"    {row['ticker']:<6} ({row['asset_class']}) — "
                  f"{row['pct_from_200ma']:+.1f}% above 200MA, "
                  f"1Y return: {row['ret_1y']:+.1f}%")

    if not down.empty:
        print(f"\n  STAY FLAT / SHORT (downtrend):")
        for _, row in down.head(10).iterrows():
            print(f"    {row['ticker']:<6} ({row['asset_class']}) — "
                  f"{row['pct_from_200ma']:+.1f}% from 200MA, "
                  f"1Y return: {row['ret_1y']:+.1f}%")

    # Save
    if output_dir is None:
        output_dir = str(Path(__file__).parent / "output")
    os.makedirs(output_dir, exist_ok=True)

    date_str = datetime.now().strftime("%Y%m%d")
    csv_path = os.path.join(output_dir, f"trend_scan_{date_str}.csv")
    df.to_csv(csv_path, index=False)

    summary = {
        "scan_date": datetime.now().isoformat(),
        "etfs_scanned": len(df),
        "strong_uptrend": strong_up["ticker"].tolist(),
        "downtrend": down["ticker"].tolist(),
        "asset_class_summary": {
            cls: {
                "bullish": int((df[df["asset_class"] == cls]["trend_signals"] > 0).sum()),
                "bearish": int((df[df["asset_class"] == cls]["trend_signals"] < 0).sum()),
            }
            for cls in ASSET_CLASSES.keys()
            if cls in df["asset_class"].values
        }
    }
    json_path = os.path.join(output_dir, f"trend_summary_{date_str}.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n[INFO] Results saved to {output_dir}")
    return df


if __name__ == "__main__":
    run_scanner()
