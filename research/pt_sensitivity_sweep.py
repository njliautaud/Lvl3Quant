#!/usr/bin/env python3
"""
PT Sensitivity Sweep — Controlled V5 Config with R1 Gate
==========================================================

Fix all parameters at V5 optimal:
  - Delta: 0.30 put/call
  - DTE: 7-14 (target 10)
  - IV rank floor: 0.20
  - Sector filter: Cannabis, Consumer Cyclical, Basic Materials, Healthcare
  - VIX max gate: 35
  - NAAIM min: 20

Only vary profit_take_pct from 25% to 85% in 5% steps (13 configs).

For each, compute:
  - Sharpe, Sortino, CAGR, MaxDD, WR, PF
  - R1 regime gap (green/red/flat stratification)
  - Trade count and assignment rate

This gives definitive answer on optimal PT for V5 wheel.

Author: Claude (2026-07-10)
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path("/home/jupiter/Lvl3Quant")
WS_ROOT = ROOT / "wheel_strategy_v1"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(WS_ROOT))

OUTPUT = ROOT / "output" / "pt_sensitivity"
OUTPUT.mkdir(parents=True, exist_ok=True)

TRADING_DAYS = 252
RISK_FREE = 0.04
RF_DAILY = RISK_FREE / TRADING_DAYS


def load_data():
    """Load all required data for wheel engine."""
    cache = WS_ROOT / "data" / "cache"

    print("Loading prices...", flush=True)
    prices = pd.read_parquet(cache / "prices.parquet")
    prices["date"] = pd.to_datetime(prices["date"], utc=True).dt.tz_localize(None)
    print(f"  prices.parquet: {len(prices)} rows, {prices['ticker'].nunique()} tickers", flush=True)

    print("Loading IV...", flush=True)
    iv_path = cache / "iv_cache.parquet"
    if iv_path.exists():
        iv = pd.read_parquet(iv_path)
        iv["date"] = pd.to_datetime(iv["date"])
        # Clip extreme sigma values
        iv["sigma"] = iv["sigma"].clip(0.05, 3.0)
        print(f"  IV cache loaded: {len(iv)} rows", flush=True)
    else:
        print("  No IV cache, using rv_20...", flush=True)
        iv = prices[["date", "ticker"]].copy()
        iv["sigma"] = prices["rv_20"].fillna(0.30)
        iv["iv_rank"] = 0.50

    print("Loading macro...", flush=True)
    macro = pd.read_parquet(cache / "macro.parquet")

    print("Loading fundamentals...", flush=True)
    fundamentals = pd.read_parquet(cache / "fundamentals.parquet")
    fundamentals["fund_score"] = 0.50
    fundamentals["dividend_yield"] = 0.02

    print("Loading universe...", flush=True)
    universe = pd.read_parquet(cache / "universe.parquet")

    # Apply V5 sector filter
    excluded = {"Cannabis", "Consumer Cyclical", "Basic Materials", "Healthcare"}
    universe = universe[~universe["sector"].isin(excluded)]
    valid_tickers = set(universe["ticker"].unique())
    prices = prices[prices["ticker"].isin(valid_tickers)]
    print(f"Universe after sector filter: {len(universe)} tickers, "
          f"{len(prices)} price rows", flush=True)

    return prices, iv, macro, fundamentals, universe


def compute_metrics(equity_curve, spy_returns=None):
    """Compute risk-adjusted metrics from equity curve (DataFrame or list)."""
    if isinstance(equity_curve, pd.DataFrame):
        if len(equity_curve) < 10:
            return {}
        navs = equity_curve["equity"].values
    else:
        if len(equity_curve) < 10:
            return {}
        navs = np.array([e[1] for e in equity_curve])

    rets = np.diff(navs) / navs[:-1]
    rets = rets[np.isfinite(rets)]

    if len(rets) < 5:
        return {}

    ann_ret = np.mean(rets) * TRADING_DAYS
    ann_vol = np.std(rets) * np.sqrt(TRADING_DAYS)
    sharpe = (ann_ret - RISK_FREE) / ann_vol if ann_vol > 0 else 0

    downside = rets[rets < 0]
    downside_std = np.std(downside) * np.sqrt(TRADING_DAYS) if len(downside) > 0 else ann_vol
    sortino = (ann_ret - RISK_FREE) / downside_std if downside_std > 0 else 0

    # Cumulative returns for CAGR and MaxDD
    cum = np.cumprod(1 + rets)
    total_ret = cum[-1] - 1
    years = len(rets) / TRADING_DAYS
    cagr = (cum[-1]) ** (1 / years) - 1 if years > 0 else 0
    max_dd = np.min(cum / np.maximum.accumulate(cum) - 1)
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    # Win rate
    wr = np.mean(rets > 0) if len(rets) > 0 else 0

    # Profit factor
    gains = rets[rets > 0].sum()
    losses = abs(rets[rets < 0].sum())
    pf = gains / losses if losses > 0 else float("inf")

    return {
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "cagr_pct": float(cagr * 100),
        "max_dd_pct": float(max_dd * 100),
        "calmar": float(calmar),
        "win_rate_pct": float(wr * 100),
        "profit_factor": float(pf),
        "total_return_pct": float(total_ret * 100),
        "n_days": len(rets),
        "years": float(years),
        "final_equity": float(navs[-1]),
    }


def compute_r1_gap(equity_curve, spy_data):
    """Compute R1 regime gap from equity curve."""
    # Handle DataFrame equity curve
    if isinstance(equity_curve, pd.DataFrame):
        if len(equity_curve) < 40:
            return {"gap": np.nan, "sharpe_green": np.nan, "sharpe_red": np.nan}
        dates = pd.to_datetime(equity_curve["date"])
        navs = equity_curve["equity"].values
    else:
        if len(equity_curve) < 40:
            return {"gap": np.nan, "sharpe_green": np.nan, "sharpe_red": np.nan}
        dates = pd.to_datetime([e[0] for e in equity_curve])
        navs = np.array([e[1] for e in equity_curve])

    rets = pd.Series(np.diff(navs) / navs[:-1], index=dates[1:])
    rets = rets[np.isfinite(rets)]

    # Classify each day using SPY
    if spy_data is None or spy_data.empty:
        return {"gap": np.nan, "sharpe_green": np.nan, "sharpe_red": np.nan}

    spy_rets = spy_data.set_index("date")["close"].pct_change()
    sigma_thresh = 0.005  # ~0.5% daily

    green_dates = spy_rets[spy_rets > sigma_thresh].index
    red_dates = spy_rets[spy_rets < -sigma_thresh].index

    green_rets = rets[rets.index.isin(green_dates)]
    red_rets = rets[rets.index.isin(red_dates)]

    if len(green_rets) < 10 or len(red_rets) < 10:
        return {"gap": np.nan, "sharpe_green": np.nan, "sharpe_red": np.nan,
                "n_green": len(green_rets), "n_red": len(red_rets)}

    sg = (green_rets.mean() * TRADING_DAYS - RISK_FREE) / (green_rets.std() * np.sqrt(TRADING_DAYS))
    sr = (red_rets.mean() * TRADING_DAYS - RISK_FREE) / (red_rets.std() * np.sqrt(TRADING_DAYS))

    denom = max(abs(sg), abs(sr))
    gap = abs(sg - sr) / denom if denom > 0 else 0

    return {
        "gap": float(gap),
        "sharpe_green": float(sg),
        "sharpe_red": float(sr),
        "n_green": len(green_rets),
        "n_red": len(red_rets),
        "r1_pass": gap <= 0.50,
    }


def run_sweep():
    """Main: run wheel engine with different PT levels."""
    from backtest.wheel_engine import WheelConfig, run_wheel

    prices, iv, macro, fundamentals, universe = load_data()

    # Load SPY for regime classification (from FULL prices, not filtered)
    spy_path = WS_ROOT / "data" / "cache" / "prices.parquet"
    all_prices = pd.read_parquet(spy_path)
    all_prices["date"] = pd.to_datetime(all_prices["date"], utc=True).dt.tz_localize(None)
    spy_data = all_prices[all_prices["ticker"] == "SPY"][["date", "close"]].copy()
    print(f"SPY data: {len(spy_data)} rows", flush=True)
    del all_prices  # Free memory

    # V5 base config (everything fixed except PT)
    base_cfg = {
        "put_delta_target": 0.30,
        "call_delta_target": 0.30,
        "dte_min": 7,
        "dte_max": 14,
        "roll_dte_trigger": 2,
        "max_concurrent_names": 20,
        "sector_cap_pct": 0.30,
        "vix_max_gate": 35.0,
        "naaim_min_gate": 20.0,
        "fund_score_floor": 0.0,
    }

    pt_levels = np.arange(0.25, 0.90, 0.05)
    results = []

    print(f"\nRunning PT sweep: {len(pt_levels)} configs", flush=True)
    print(f"{'PT':>6} {'Sharpe':>8} {'Sort':>8} {'CAGR':>8} {'MaxDD':>8} "
          f"{'WR':>6} {'Gap':>8} {'R1':>6} {'Time':>6}", flush=True)
    print("-" * 70, flush=True)

    for pt in pt_levels:
        t0 = time.time()

        cfg = WheelConfig(
            profit_take_pct=float(pt),
            **base_cfg,
        )

        try:
            result = run_wheel(
                cfg, prices, iv, macro, fundamentals, universe,
                starting_cash=100_000.0,
                verbose=False,
            )

            eq = result.get("equity_curve", pd.DataFrame())
            metrics = compute_metrics(eq)
            r1 = compute_r1_gap(eq, spy_data)

            elapsed = time.time() - t0

            n_csp = result.get("csp_opened", 0)
            n_assign = result.get("assignment_count", 0)
            assign_rate = n_assign / n_csp if n_csp > 0 else 0

            row = {
                "pt_pct": float(pt),
                **metrics,
                **r1,
                "n_trades": n_csp,
                "assignment_rate": assign_rate,
                "elapsed_sec": elapsed,
            }
            results.append(row)

            r1_str = "PASS" if r1.get("r1_pass", False) else "FAIL"
            print(f"{pt:>5.0%} {metrics.get('sharpe', 0):>8.2f} "
                  f"{metrics.get('sortino', 0):>8.2f} "
                  f"{metrics.get('cagr_pct', 0):>7.1f}% "
                  f"{metrics.get('max_dd_pct', 0):>7.1f}% "
                  f"{metrics.get('win_rate_pct', 0):>5.0f}% "
                  f"{r1.get('gap', 0):>8.3f} "
                  f"{r1_str:>6} "
                  f"{elapsed:>5.0f}s", flush=True)

        except Exception as e:
            elapsed = time.time() - t0
            print(f"{pt:>5.0%} ERROR: {e} ({elapsed:.0f}s)", flush=True)
            results.append({"pt_pct": float(pt), "error": str(e)})

    # Summary
    print(f"\n{'=' * 70}")
    print("PT SENSITIVITY SWEEP — V5 CONFIG")
    print(f"{'=' * 70}")

    valid = [r for r in results if "sharpe" in r]
    if valid:
        best_sharpe = max(valid, key=lambda x: x.get("sharpe", 0))
        best_sortino = max(valid, key=lambda x: x.get("sortino", 0))
        r1_passing = [r for r in valid if r.get("r1_pass", False)]

        print(f"\nBest Sharpe: PT={best_sharpe['pt_pct']:.0%} → Sharpe {best_sharpe['sharpe']:.2f}")
        print(f"Best Sortino: PT={best_sortino['pt_pct']:.0%} → Sortino {best_sortino['sortino']:.2f}")
        print(f"R1 passing configs: {len(r1_passing)} / {len(valid)}")

        if r1_passing:
            best_r1 = max(r1_passing, key=lambda x: x.get("sharpe", 0))
            print(f"Best R1-passing: PT={best_r1['pt_pct']:.0%} → "
                  f"Sharpe {best_r1['sharpe']:.2f}, gap {best_r1['gap']:.3f}")

    # Save
    import json
    output_file = OUTPUT / "pt_sweep_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    run_sweep()
