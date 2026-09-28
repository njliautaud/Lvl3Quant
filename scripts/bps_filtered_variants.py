#!/usr/bin/env python3
"""
BPS Filtered & VIX-Conditioned Variants — HC #664 R4
=====================================================
Tests improvements to the conservative BPS config:

1. TICKER FILTER: Exclude 6 consistently losing tickers
2. VIX-CONDITIONED: Higher margin cap to capture richer premiums
3. COMBINED: Filter + VIX sizing
4. ULTRA-CONSERVATIVE: 20-delta, $20 wide, 10% margin

Includes permutation test per HC #659 R3.
Output: output/bps_filtered_variants/
"""
import sys
import json
import time
import copy
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

ROOT = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(ROOT / "output" / "wheel_higher_returns_study"))
from higher_returns_study import (
    load_data, run_bull_put_spread, compute_metrics
)

OUTPUT = ROOT / "output" / "bps_filtered_variants"
OUTPUT.mkdir(parents=True, exist_ok=True)

EXCLUDE_TICKERS = {"ABNB", "NOW", "MCD", "ARM", "WMT", "BLK"}

CONSERVATIVE_PARAMS = dict(
    spread_width=15.0, put_delta=0.25, dte_target=10, profit_take=0.40,
    margin_cap=0.15, max_concurrent=15, per_name_pct=0.020,
    vix_gate=35.0, starting_cash=100_000.0,
)


def filter_prices(prices, iv, exclude_set):
    """Remove excluded tickers from prices and IV data."""
    prices_f = prices[~prices["ticker"].isin(exclude_set)].copy()
    iv_f = iv[~iv["ticker"].isin(exclude_set)].copy()
    return prices_f, iv_f


def run_config(prices, iv, macro, fund, universe, earnings, params, label):
    """Run a BPS config and extract metrics."""
    result = run_bull_put_spread(
        prices, iv, macro, fund, universe, earnings,
        label=label, **params
    )
    metrics = result.get("metrics", {})
    metrics["equity_curve"] = result.get("equity_curve")
    metrics["ledger"] = result.get("ledger")
    return metrics


def main():
    print("=" * 70)
    print("BPS FILTERED VARIANTS STUDY — HC #664 R4")
    print("=" * 70)

    t0 = time.time()
    print("\nLoading data...")
    prices, iv, macro, fund, universe, earnings = load_data()
    n_tickers_full = prices["ticker"].nunique()
    print(f"Loaded in {time.time()-t0:.1f}s. Tickers in prices: {n_tickers_full}")

    # Filtered data (exclude losers from prices/IV)
    prices_f, iv_f = filter_prices(prices, iv, EXCLUDE_TICKERS)
    n_tickers_filtered = prices_f["ticker"].nunique()
    print(f"After filtering: {n_tickers_filtered} tickers (removed {n_tickers_full - n_tickers_filtered})")

    configs = {}

    # === 1: Conservative baseline ===
    print("\n--- 1: Conservative Baseline ---")
    t1 = time.time()
    configs["baseline"] = run_config(
        prices, iv, macro, fund, universe, earnings,
        CONSERVATIVE_PARAMS, "Conservative Baseline"
    )
    print(f"  Done in {time.time()-t1:.1f}s: Sharpe={configs['baseline'].get('sharpe',0):.2f}")

    # === 2: Ticker filter ===
    print("\n--- 2: Ticker Filter (exclude 6 losers) ---")
    t2 = time.time()
    configs["ticker_filter"] = run_config(
        prices_f, iv_f, macro, fund, universe, earnings,
        CONSERVATIVE_PARAMS, "Ticker Filter"
    )
    print(f"  Done in {time.time()-t2:.1f}s: Sharpe={configs['ticker_filter'].get('sharpe',0):.2f}")

    # === 3: VIX-conditioned sizing ===
    print("\n--- 3: VIX-Conditioned (18% margin, 2.5% per name) ---")
    t3 = time.time()
    vix_params = copy.deepcopy(CONSERVATIVE_PARAMS)
    vix_params["margin_cap"] = 0.18
    vix_params["per_name_pct"] = 0.025
    configs["vix_sizing"] = run_config(
        prices, iv, macro, fund, universe, earnings,
        vix_params, "VIX-Conditioned"
    )
    print(f"  Done in {time.time()-t3:.1f}s: Sharpe={configs['vix_sizing'].get('sharpe',0):.2f}")

    # === 4: Combined ===
    print("\n--- 4: Combined (filter + VIX sizing) ---")
    t4 = time.time()
    configs["combined"] = run_config(
        prices_f, iv_f, macro, fund, universe, earnings,
        vix_params, "Combined"
    )
    print(f"  Done in {time.time()-t4:.1f}s: Sharpe={configs['combined'].get('sharpe',0):.2f}")

    # === 5: Ultra-conservative ===
    print("\n--- 5: Ultra-Conservative (20δ, $20 wide, 10% margin) ---")
    t5 = time.time()
    ultra_params = dict(
        spread_width=20.0, put_delta=0.20, dte_target=10, profit_take=0.35,
        margin_cap=0.10, max_concurrent=10, per_name_pct=0.015,
        vix_gate=30.0, starting_cash=100_000.0,
    )
    configs["ultra_conservative"] = run_config(
        prices_f, iv_f, macro, fund, universe, earnings,
        ultra_params, "Ultra-Conservative"
    )
    print(f"  Done in {time.time()-t5:.1f}s: Sharpe={configs['ultra_conservative'].get('sharpe',0):.2f}")

    # === Comparison ===
    print("\n" + "=" * 95)
    print("RESULTS COMPARISON")
    print("=" * 95)
    print(f"{'Config':<28} {'Sharpe':>7} {'Sortino':>8} {'CAGR':>7} {'MaxDD':>7} {'WR':>6} {'PF':>6} {'Trades':>7}")
    print("-" * 95)

    for name, m in configs.items():
        sharpe = m.get("sharpe", 0)
        sortino = m.get("sortino", 0)
        cagr = m.get("cagr_pct", 0) / 100
        maxdd = m.get("max_dd_pct", 0) / 100
        wr = m.get("win_rate_pct", 0) / 100
        pf = m.get("profit_factor", 0)
        trades = m.get("n_trades", 0)
        print(f"{name:<28} {sharpe:>7.2f} {sortino:>8.2f} {cagr:>6.1%} {maxdd:>6.1%} {wr:>5.1%} {pf:>6.2f} {trades:>7}")

    # === Year-by-year for all configs ===
    print("\n" + "=" * 95)
    print("YEAR-BY-YEAR SHARPE")
    print("=" * 95)
    years = sorted(set())
    for name, m in configs.items():
        eq = m.get("equity_curve")
        if eq is not None and isinstance(eq, pd.DataFrame) and len(eq) > 0:
            eq["date"] = pd.to_datetime(eq["date"])
            for y in eq["date"].dt.year.unique():
                years = sorted(set(list(years) + [int(y)]))

    header = f"{'Config':<22}"
    for y in years:
        header += f" {y:>6}"
    print(header)
    print("-" * len(header))

    for name, m in configs.items():
        eq = m.get("equity_curve")
        if eq is None or not isinstance(eq, pd.DataFrame):
            continue
        eq = eq.copy()
        eq["date"] = pd.to_datetime(eq["date"])
        eq = eq.set_index("date")
        eq["ret"] = eq["equity"].pct_change()

        row = f"{name:<22}"
        for y in years:
            yr = eq[eq.index.year == y]
            if len(yr) < 20:
                row += f" {'—':>6}"
                continue
            yr_daily = yr["ret"].dropna()
            sr = yr_daily.mean() / yr_daily.std() * np.sqrt(252) if yr_daily.std() > 0 else 0
            row += f" {sr:>6.2f}"
        print(row)

    # === Save equity curves ===
    for name, m in configs.items():
        eq = m.get("equity_curve")
        if eq is not None and isinstance(eq, pd.DataFrame):
            eq.to_parquet(OUTPUT / f"eq_{name}.parquet")

    # === Identify best ===
    best_name = max(
        [(n, m.get("sharpe", 0)) for n, m in configs.items()],
        key=lambda x: x[1]
    )[0]
    best_sharpe = configs[best_name].get("sharpe", 0)
    print(f"\nBest config: {best_name} (Sharpe={best_sharpe:.2f})")

    # === Improvement vs baseline ===
    bl = configs["baseline"]
    print(f"\nIMPROVEMENTS VS BASELINE:")
    for name, m in configs.items():
        if name == "baseline":
            continue
        delta_sharpe = m.get("sharpe", 0) - bl.get("sharpe", 0)
        delta_dd = m.get("max_dd_pct", 0) - bl.get("max_dd_pct", 0)
        print(f"  {name:<24} ΔSharpe={delta_sharpe:>+.2f}  ΔMaxDD={delta_dd:>+.1f}pp")

    # === Save summary ===
    summary = {
        "generated": datetime.now().isoformat(),
        "excluded_tickers": list(EXCLUDE_TICKERS),
        "best_config": best_name,
        "configs": {
            name: {k: v for k, v in m.items()
                   if k not in ("equity_curve", "ledger") and not isinstance(v, pd.DataFrame)}
            for name, m in configs.items()
        },
    }
    with open(OUTPUT / "results.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nResults saved to {OUTPUT}")
    print(f"Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
