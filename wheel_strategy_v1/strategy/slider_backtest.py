#!/usr/bin/env python3
"""
slider_backtest.py — Backtest all 10 slider levels and produce a comparison table.
==================================================================================

Runs the wheel engine at each aggressiveness level (1-10) over the full
available history, then outputs a side-by-side comparison of:
  - CAGR, Sharpe, Sortino, MaxDD, WR, PF
  - Weekly return stats (mean, median, std)
  - Assignment frequency
  - Regime-stratified Sharpe (green/red/flat days)

Usage:
    python3 slider_backtest.py                     # all 10 levels
    python3 slider_backtest.py --levels 3 5 7      # specific levels
    python3 slider_backtest.py --levels 7 --detail  # detailed single-level report

Author: Claude (2026-07-14)
"""
import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from strategy.slider import slider, slider_ladder, SliderConfig
from backtest.wheel_engine import WheelConfig, run_wheel

CACHE = ROOT / "data" / "cache"
OUTPUT_DIR = ROOT / "output" / "slider_backtest"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """Load price, IV, fundamentals, macro data from cache."""
    prices_files = ["prices.parquet", "prices_expanded.parquet", "prices_v3_expansion.parquet"]
    dfs = []
    for pf in prices_files:
        path = CACHE / pf
        if path.exists():
            dfs.append(pd.read_parquet(path))
    if not dfs:
        raise FileNotFoundError(f"No price data found in {CACHE}")
    prices = pd.concat(dfs, ignore_index=True)
    prices = prices.drop_duplicates(subset=["ticker", "date"])
    prices["date"] = pd.to_datetime(prices["date"], utc=True).dt.tz_localize(None)

    # IV features
    iv_path = CACHE / "iv_features_real_blend.parquet"
    if not iv_path.exists():
        iv_path = CACHE / "iv_features.parquet"
    iv_features = pd.read_parquet(iv_path) if iv_path.exists() else pd.DataFrame()
    if "date" in iv_features.columns:
        iv_features["date"] = pd.to_datetime(iv_features["date"], utc=True).dt.tz_localize(None)

    # Fundamentals
    fund_path = CACHE / "fundamentals.parquet"
    fundamentals = pd.read_parquet(fund_path) if fund_path.exists() else pd.DataFrame()

    # Macro (VIX, NAAIM, SPY)
    macro_path = CACHE / "macro.parquet"
    macro = pd.read_parquet(macro_path) if macro_path.exists() else pd.DataFrame()
    if "date" in macro.columns:
        macro["date"] = pd.to_datetime(macro["date"], utc=True).dt.tz_localize(None)

    # SPY for regime classification
    spy = prices[prices["ticker"] == "SPY"][["date", "close"]].copy()
    spy = spy.sort_values("date").drop_duplicates("date")

    return prices, iv_features, fundamentals, macro, spy


def classify_regime(spy: pd.DataFrame) -> pd.DataFrame:
    """Classify each day as green/red/flat based on SPY close-to-close."""
    spy = spy.copy().sort_values("date")
    spy["ret"] = spy["close"].pct_change()
    spy["regime"] = "flat"
    spy.loc[spy["ret"] > 0.001, "regime"] = "green"
    spy.loc[spy["ret"] < -0.001, "regime"] = "red"
    return spy[["date", "regime"]].dropna()


def compute_metrics(equity_curve: list, trades: list, spy_regime: pd.DataFrame) -> dict:
    """Compute risk-adjusted metrics from equity curve and trade ledger."""
    if not equity_curve or len(equity_curve) < 10:
        return {"error": "insufficient data"}

    dates = [e[0] for e in equity_curve]
    equity = np.array([e[1] for e in equity_curve], dtype=float)

    # Daily returns
    rets = np.diff(equity) / equity[:-1]
    rets = rets[np.isfinite(rets)]

    if len(rets) < 5:
        return {"error": "insufficient returns"}

    # Annualized metrics
    ann_factor = 252
    mean_daily = np.mean(rets)
    std_daily = np.std(rets, ddof=1) if len(rets) > 1 else 1e-9
    downside_rets = rets[rets < 0]
    downside_std = np.std(downside_rets, ddof=1) if len(downside_rets) > 1 else 1e-9

    sharpe = (mean_daily / std_daily) * np.sqrt(ann_factor) if std_daily > 0 else 0
    sortino = (mean_daily / downside_std) * np.sqrt(ann_factor) if downside_std > 0 else 0

    # CAGR
    years = len(rets) / ann_factor
    total_ret = equity[-1] / equity[0]
    cagr = (total_ret ** (1 / max(years, 0.01))) - 1 if total_ret > 0 else -1.0

    # Max drawdown
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd = float(np.min(dd))

    # Trade metrics
    if trades:
        pnls = [t.get("realized_pnl", 0) for t in trades if "realized_pnl" in t]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        wr = len(wins) / len(pnls) if pnls else 0
        pf = sum(wins) / abs(sum(losses)) if losses else float("inf")
        assignments = sum(1 for t in trades if t.get("event") == "assigned")
    else:
        wr, pf, assignments = 0, 0, 0
        pnls = []

    # Weekly returns
    eq_df = pd.DataFrame({"date": dates[:len(rets)+1], "equity": equity[:len(rets)+1]})
    eq_df["date"] = pd.to_datetime(eq_df["date"])
    eq_df = eq_df.set_index("date")
    weekly = eq_df["equity"].resample("W").last().dropna()
    weekly_rets = weekly.pct_change().dropna()
    weekly_mean = float(weekly_rets.mean()) if len(weekly_rets) > 0 else 0
    weekly_median = float(weekly_rets.median()) if len(weekly_rets) > 0 else 0
    weekly_std = float(weekly_rets.std()) if len(weekly_rets) > 0 else 0

    # Regime-stratified Sharpe
    regime_sharpe = {}
    if not spy_regime.empty:
        ret_df = pd.DataFrame({"date": dates[1:len(rets)+1], "ret": rets})
        ret_df["date"] = pd.to_datetime(ret_df["date"])
        merged = ret_df.merge(spy_regime, on="date", how="left")
        for regime in ["green", "red", "flat"]:
            sub = merged[merged["regime"] == regime]["ret"]
            if len(sub) > 5:
                s = (sub.mean() / sub.std()) * np.sqrt(252) if sub.std() > 0 else 0
                regime_sharpe[regime] = round(s, 3)
            else:
                regime_sharpe[regime] = None

    # Regime gap check (HC #428 R1)
    regime_gap = None
    if regime_sharpe.get("green") is not None and regime_sharpe.get("red") is not None:
        sg, sr = abs(regime_sharpe["green"]), abs(regime_sharpe["red"])
        if max(sg, sr) > 0:
            regime_gap = abs(regime_sharpe["green"] - regime_sharpe["red"]) / max(sg, sr)

    return {
        "cagr": round(cagr * 100, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_dd": round(max_dd * 100, 2),
        "win_rate": round(wr * 100, 1),
        "profit_factor": round(pf, 2) if pf < 100 else ">100",
        "total_trades": len(pnls),
        "assignments": assignments,
        "weekly_mean_pct": round(weekly_mean * 100, 3),
        "weekly_median_pct": round(weekly_median * 100, 3),
        "weekly_std_pct": round(weekly_std * 100, 3),
        "regime_sharpe": regime_sharpe,
        "regime_gap": round(regime_gap, 3) if regime_gap is not None else None,
        "r1_pass": regime_gap is not None and regime_gap <= 0.50,
        "years": round(years, 2),
        "final_equity": round(float(equity[-1]), 2),
    }


def run_slider_level(level: int, prices: pd.DataFrame, iv_features: pd.DataFrame,
                     fundamentals: pd.DataFrame, macro: pd.DataFrame,
                     spy_regime: pd.DataFrame) -> dict:
    """Run a full backtest at a given slider level."""
    cfg = slider(level)
    wcfg = cfg.to_wheel_config()

    # Filter universe to tickers that have price data
    available = set(prices["ticker"].unique())
    universe_tickers = cfg.get_universe(available)

    if not universe_tickers:
        return {"level": level, "name": cfg.level_name, "error": "no tickers available"}

    print(f"\n  Level {level} ({cfg.level_name}): {len(universe_tickers)} tickers, "
          f"delta={cfg.put_delta_target:.2f}, DTE={cfg.dte_min}-{cfg.dte_max}")

    # Build universe df with sector info
    if "sector" in fundamentals.columns:
        uni_df = fundamentals[fundamentals["ticker"].isin(universe_tickers)][["ticker", "sector"]].copy()
    else:
        uni_df = pd.DataFrame({"ticker": universe_tickers, "sector": "Unknown"})
    # Ensure all tickers in universe_tickers appear
    missing = set(universe_tickers) - set(uni_df["ticker"])
    if missing:
        uni_df = pd.concat([uni_df, pd.DataFrame({"ticker": list(missing), "sector": "Unknown"})],
                           ignore_index=True)

    t0 = time.time()
    try:
        result = run_wheel(
            cfg=wcfg,
            prices=prices[prices["ticker"].isin(universe_tickers + ["SPY"])],
            iv=iv_features,
            fundamentals=fundamentals,
            macro=macro,
            universe=uni_df,
        )
    except Exception as e:
        print(f"    ERROR: {e}")
        traceback.print_exc()
        return {"level": level, "name": cfg.level_name, "error": str(e)}

    elapsed = time.time() - t0

    # Extract equity curve and trades
    equity_curve = result.get("equity_curve", [])
    trades = result.get("ledger", [])

    metrics = compute_metrics(equity_curve, trades, spy_regime)
    metrics["level"] = level
    metrics["name"] = cfg.level_name
    metrics["weekly_target"] = cfg.weekly_target
    metrics["delta"] = cfg.put_delta_target
    metrics["dte_range"] = f"{cfg.dte_min}-{cfg.dte_max}"
    metrics["universe_size"] = len(universe_tickers)
    metrics["elapsed_sec"] = round(elapsed, 1)

    # Save equity curve
    eq_path = OUTPUT_DIR / f"equity_level_{level}.parquet"
    if equity_curve:
        eq_df = pd.DataFrame(equity_curve, columns=["date", "equity"])
        eq_df.to_parquet(eq_path, index=False)

    print(f"    Done in {elapsed:.0f}s — CAGR {metrics.get('cagr', '?')}%, "
          f"Sharpe {metrics.get('sharpe', '?')}, MaxDD {metrics.get('max_dd', '?')}%")

    return metrics


def print_comparison_table(results: list):
    """Print a clean side-by-side comparison table."""
    print("\n" + "=" * 110)
    print(f"{'WHEEL AGGRESSIVENESS SLIDER — BACKTEST COMPARISON':^110s}")
    print("=" * 110)

    header = (f"{'Lvl':>3s} {'Name':>20s} {'Delta':>6s} {'DTE':>7s} "
              f"{'CAGR%':>7s} {'Sharpe':>7s} {'Sortino':>8s} {'MaxDD%':>7s} "
              f"{'WR%':>5s} {'PF':>6s} {'Trades':>6s} {'Wk%':>7s} {'R1':>4s}")
    print(header)
    print("-" * 110)

    for r in results:
        if "error" in r:
            print(f"  {r['level']:2d} {r['name']:>20s}  — ERROR: {r['error']}")
            continue

        r1 = "PASS" if r.get("r1_pass") else "FAIL"
        pf_str = f"{r['profit_factor']:>6.2f}" if isinstance(r['profit_factor'], (int, float)) else f"{r['profit_factor']:>6s}"
        print(f"  {r['level']:2d} {r['name']:>20s} {r['delta']:>6.2f} {r['dte_range']:>7s} "
              f"{r['cagr']:>7.1f} {r['sharpe']:>7.3f} {r['sortino']:>8.3f} {r['max_dd']:>7.1f} "
              f"{r['win_rate']:>5.1f} {pf_str} {r['total_trades']:>6d} {r['weekly_mean_pct']:>7.3f} {r1:>4s}")

    print("=" * 110)
    print("\nR1 = Regime-agnostic gate (|Sharpe_green - Sharpe_red| / max < 0.50)")
    print("WR% = Win Rate, PF = Profit Factor, Wk% = Mean Weekly Return")


def main():
    parser = argparse.ArgumentParser(description="Wheel slider backtest")
    parser.add_argument("--levels", type=int, nargs="+", default=list(range(1, 11)),
                        help="Slider levels to backtest (default: 1-10)")
    parser.add_argument("--detail", action="store_true",
                        help="Show detailed report for each level")
    args = parser.parse_args()

    print("Loading data...")
    prices, iv_features, fundamentals, macro, spy = load_data()
    spy_regime = classify_regime(spy)
    print(f"  Prices: {len(prices)} rows, {prices['ticker'].nunique()} tickers")
    print(f"  Date range: {prices['date'].min().date()} to {prices['date'].max().date()}")

    results = []
    for level in sorted(args.levels):
        r = run_slider_level(level, prices, iv_features, fundamentals, macro, spy_regime)
        results.append(r)

        if args.detail and "error" not in r:
            cfg = slider(level)
            print(cfg.summary())
            print(f"  Regime Sharpe: {r.get('regime_sharpe', {})}")
            print(f"  Regime Gap: {r.get('regime_gap', 'N/A')}")
            print(f"  Assignments: {r.get('assignments', 0)}")

    # Print comparison
    print_comparison_table(results)

    # Save results JSON
    out_path = OUTPUT_DIR / "slider_comparison.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
