#!/usr/bin/env python3
"""
Options Pricing Validator
=========================
Validates our theoretical Black-Scholes pricing (from options_pricer.py) against
real market option chain data for the sector rotation strategy.

Strategy context:
  - Bull call spreads: long K1 call (2-3% OTM) + short K2 call (K1 * 1.03)
  - Bear put spreads: long K1 put (2-3% OTM below) + short K2 put (K1 * 0.97)
  - DTE target: 14-28 days
  - 15% haircut applied to BS fair value

Outputs:
  - Pricing error statistics by ticker and trade type
  - Haircut adequacy analysis
  - Results saved to output/growth_research/options_pricing_validation/
"""
from __future__ import annotations

import os
import sys
import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path("/home/jupiter/Lvl3Quant")
CHAINS_DIR = ROOT / "data" / "options_chains"
OUTPUT_DIR = ROOT / "output" / "growth_research" / "options_pricing_validation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Add project root so we can import the pricer
sys.path.insert(0, str(ROOT))
from research.tools.options_pricer import (
    price_bull_call_spread,
    price_bear_put_spread,
    bs_call_price,
    bs_put_price,
    estimate_iv,
    DEFAULT_HAIRCUT,
    RISK_FREE_RATE,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SECTOR_ETFS = [
    "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC",
]
DTE_MIN = 14
DTE_MAX = 28
OTM_MIN = 0.02   # 2% OTM
OTM_MAX = 0.03   # 3% OTM
SPREAD_WIDTH = 0.03  # 3% spread width
HAIRCUT = DEFAULT_HAIRCUT  # 15%

# If sector ETFs have bad IV, we can also validate on liquid single-name tickers
FALLBACK_TICKERS = [
    "SPY", "QQQ", "IWM", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA",
]


def find_spread_legs(
    df: pd.DataFrame,
    spot: float,
    option_type: str,
    dte_target: int,
) -> list[dict]:
    """
    Find matching spread legs from actual option chain data.

    For bull call spreads:
      - Long K1 call at 2-3% OTM (above spot)
      - Short K2 call at K1 * 1.03 (3% spread width)

    For bear put spreads:
      - Long K1 put at 2-3% OTM (below spot)
      - Short K2 put at K1 * 0.97 (3% spread width)

    Returns list of dicts with spread details.
    """
    sub = df[(df["option_type"] == option_type) & (df["dte_days"] == dte_target)].copy()
    if sub.empty:
        return []

    strikes = sorted(sub["strike"].unique())
    if len(strikes) < 2:
        return []

    results = []

    if option_type == "call":
        # Bull call spread: K1 is 2-3% OTM above spot
        k1_min = spot * (1 + OTM_MIN)
        k1_max = spot * (1 + OTM_MAX)
        k1_candidates = [s for s in strikes if k1_min <= s <= k1_max]

        for k1 in k1_candidates:
            # K2 = K1 * 1.03 (3% spread width) -- find closest available strike
            k2_target = k1 * (1 + SPREAD_WIDTH)
            k2 = min(strikes, key=lambda s: abs(s - k2_target))
            # Accept if within 1% of target
            if abs(k2 - k2_target) / k2_target > 0.01 or k2 <= k1:
                continue

            leg1 = sub[sub["strike"] == k1].iloc[0]
            leg2 = sub[sub["strike"] == k2].iloc[0]

            # Need at least mid price to be usable
            if leg1["mid"] <= 0 or leg2["mid"] <= 0:
                continue

            # Market spread cost = long call premium - short call premium
            market_mid = leg1["mid"] - leg2["mid"]
            market_bid_cost = leg1["ask"] - leg2["bid"] if leg1["ask"] > 0 and leg2["bid"] > 0 else None
            market_ask_cost = leg1["bid"] - leg2["ask"] if leg1["bid"] > 0 and leg2["ask"] > 0 else None

            if market_mid <= 0:
                continue

            results.append({
                "trade_type": "bull_call",
                "K1": k1,
                "K2": k2,
                "dte": dte_target,
                "spot": spot,
                "market_mid_cost": market_mid,
                "market_worst_cost": market_bid_cost,  # what you'd pay (buy at ask, sell at bid)
                "market_best_cost": market_ask_cost,    # best case
                "leg1_mid": leg1["mid"],
                "leg2_mid": leg2["mid"],
                "leg1_bid": leg1["bid"],
                "leg1_ask": leg1["ask"],
                "leg2_bid": leg2["bid"],
                "leg2_ask": leg2["ask"],
                "leg1_iv": leg1["iv"],
                "leg2_iv": leg2["iv"],
            })

    elif option_type == "put":
        # Bear put spread: K1 (long put) is 2-3% OTM below spot
        k1_min = spot * (1 - OTM_MAX)
        k1_max = spot * (1 - OTM_MIN)
        k1_candidates = [s for s in strikes if k1_min <= s <= k1_max]

        for k1 in k1_candidates:
            # K2 = K1 * 0.97 (short put, 3% below long put)
            k2_target = k1 * (1 - SPREAD_WIDTH)
            k2 = min(strikes, key=lambda s: abs(s - k2_target))
            if abs(k2 - k2_target) / k2_target > 0.01 or k2 >= k1:
                continue

            leg1 = sub[sub["strike"] == k1].iloc[0]
            leg2 = sub[sub["strike"] == k2].iloc[0]

            if leg1["mid"] <= 0 or leg2["mid"] <= 0:
                continue

            # Market spread cost = long put (K1, higher) - short put (K2, lower)
            market_mid = leg1["mid"] - leg2["mid"]
            market_bid_cost = leg1["ask"] - leg2["bid"] if leg1["ask"] > 0 and leg2["bid"] > 0 else None
            market_ask_cost = leg1["bid"] - leg2["ask"] if leg1["bid"] > 0 and leg2["ask"] > 0 else None

            if market_mid <= 0:
                continue

            results.append({
                "trade_type": "bear_put",
                "K1": k1,          # higher strike (long put)
                "K2": k2,          # lower strike (short put)
                "dte": dte_target,
                "spot": spot,
                "market_mid_cost": market_mid,
                "market_worst_cost": market_bid_cost,
                "market_best_cost": market_ask_cost,
                "leg1_mid": leg1["mid"],
                "leg2_mid": leg2["mid"],
                "leg1_bid": leg1["bid"],
                "leg1_ask": leg1["ask"],
                "leg2_bid": leg2["bid"],
                "leg2_ask": leg2["ask"],
                "leg1_iv": leg1["iv"],
                "leg2_iv": leg2["iv"],
            })

    return results


def compute_theoretical_price(spread: dict, use_market_iv: bool = False) -> dict:
    """
    Compute the theoretical BS spread price using our pricer module.

    We compute in two modes:
      1. ATR-estimated IV (what our strategy actually uses) -- but we need ATR
         which we don't have, so we back-compute a reasonable ATR from market IV
      2. Market IV directly (to isolate BS model error from IV estimation error)

    Since we don't have ATR in this context, we use market IV when available
    and estimate a reasonable ATR-based IV otherwise.
    """
    spot = spread["spot"]
    k1 = spread["K1"]
    k2 = spread["K2"]
    dte = spread["dte"]
    T = dte / 365.0

    result = {}

    # --- Mode 1: Use market IV (isolates BS formula error) ---
    avg_iv = (spread["leg1_iv"] + spread["leg2_iv"]) / 2.0
    if avg_iv > 0.01:  # sanity check: IV > 1%
        if spread["trade_type"] == "bull_call":
            # Use individual IVs for each leg (captures skew)
            c1 = bs_call_price(spot, k1, T, RISK_FREE_RATE, spread["leg1_iv"])
            c2 = bs_call_price(spot, k2, T, RISK_FREE_RATE, spread["leg2_iv"])
            fair_value_skew = max(c1 - c2, 0.001)

            # Also flat IV version
            c1_flat = bs_call_price(spot, k1, T, RISK_FREE_RATE, avg_iv)
            c2_flat = bs_call_price(spot, k2, T, RISK_FREE_RATE, avg_iv)
            fair_value_flat = max(c1_flat - c2_flat, 0.001)

            result["bs_fair_skew"] = fair_value_skew
            result["bs_fair_flat"] = fair_value_flat
        else:
            p1 = bs_put_price(spot, k1, T, RISK_FREE_RATE, spread["leg1_iv"])
            p2 = bs_put_price(spot, k2, T, RISK_FREE_RATE, spread["leg2_iv"])
            fair_value_skew = max(p1 - p2, 0.001)

            p1_flat = bs_put_price(spot, k1, T, RISK_FREE_RATE, avg_iv)
            p2_flat = bs_put_price(spot, k2, T, RISK_FREE_RATE, avg_iv)
            fair_value_flat = max(p1_flat - p2_flat, 0.001)

            result["bs_fair_skew"] = fair_value_skew
            result["bs_fair_flat"] = fair_value_flat

        result["market_iv_used"] = True
        result["avg_iv"] = avg_iv
    else:
        result["market_iv_used"] = False
        result["avg_iv"] = avg_iv

    # --- Mode 2: ATR-estimated IV (simulates what our strategy does) ---
    # Estimate ATR from a typical ETF vol profile:
    # typical sector ETF ATR ~ 1-3% of price for 14-day ATR
    # Use 2% as default, adjustable
    atr_pct = 0.02
    atr_est = spot * atr_pct
    vix_est = 20.0  # default VIX assumption

    if spread["trade_type"] == "bull_call":
        entry_cost_atr, max_profit_atr = price_bull_call_spread(
            S=spot, K1=k1, K2=k2, dte=dte, atr=atr_est, vix=vix_est,
            haircut=0.0,  # get fair value without haircut first
        )
        entry_cost_atr_hc, _ = price_bull_call_spread(
            S=spot, K1=k1, K2=k2, dte=dte, atr=atr_est, vix=vix_est,
            haircut=HAIRCUT,
        )
    else:
        # Bear put spread: in our pricer, K1 < K2
        k_low = min(k1, k2)
        k_high = max(k1, k2)
        entry_cost_atr, max_profit_atr = price_bear_put_spread(
            S=spot, K1=k_low, K2=k_high, dte=dte, atr=atr_est, vix=vix_est,
            haircut=0.0,
        )
        entry_cost_atr_hc, _ = price_bear_put_spread(
            S=spot, K1=k_low, K2=k_high, dte=dte, atr=atr_est, vix=vix_est,
            haircut=HAIRCUT,
        )

    result["bs_fair_atr"] = entry_cost_atr
    result["bs_haircut_atr"] = entry_cost_atr_hc
    result["atr_iv_est"] = estimate_iv(atr_est, spot, vix_est)

    return result


def analyze_one_file(filepath: Path, ticker: str, date_str: str) -> list[dict]:
    """Process one parquet file, find all valid spreads, compute errors."""
    try:
        df = pd.read_parquet(filepath)
    except Exception as e:
        print(f"  [WARN] Could not read {filepath}: {e}")
        return []

    if df.empty:
        return []

    spot = df["underlying_price"].iloc[0]
    available_dtes = sorted(df["dte_days"].unique())
    target_dtes = [d for d in available_dtes if DTE_MIN <= d <= DTE_MAX]

    if not target_dtes:
        return []

    all_spreads = []
    for dte in target_dtes:
        # Bull call spreads
        bull_spreads = find_spread_legs(df, spot, "call", dte)
        for s in bull_spreads:
            s["ticker"] = ticker
            s["date"] = date_str
        all_spreads.extend(bull_spreads)

        # Bear put spreads
        bear_spreads = find_spread_legs(df, spot, "put", dte)
        for s in bear_spreads:
            s["ticker"] = ticker
            s["date"] = date_str
        all_spreads.extend(bear_spreads)

    # Compute theoretical prices for each spread
    results = []
    for spread in all_spreads:
        theo = compute_theoretical_price(spread)
        row = {**spread, **theo}

        # Compute pricing errors
        market_cost = row["market_mid_cost"]

        # Error vs market mid using market IV (BS formula accuracy)
        if row.get("market_iv_used") and row.get("bs_fair_skew"):
            row["error_bs_skew"] = (row["bs_fair_skew"] - market_cost) / market_cost
            row["error_bs_flat"] = (row["bs_fair_flat"] - market_cost) / market_cost
            # With haircut applied
            row["bs_skew_with_hc"] = row["bs_fair_skew"] * (1 + HAIRCUT)
            row["error_skew_hc"] = (row["bs_skew_with_hc"] - market_cost) / market_cost

        # Error vs market mid using ATR-estimated IV
        row["error_atr"] = (row["bs_fair_atr"] - market_cost) / market_cost
        row["error_atr_hc"] = (row["bs_haircut_atr"] - market_cost) / market_cost

        # Haircut adequacy: does theoretical + haircut >= market worst cost?
        if row.get("market_worst_cost") and row["market_worst_cost"] > 0:
            row["haircut_covers_worst"] = row.get("bs_skew_with_hc", row["bs_haircut_atr"]) >= row["market_worst_cost"]
        else:
            row["haircut_covers_worst"] = None

        results.append(row)

    return results


def run_validation():
    """Main validation loop over all available data."""
    print("=" * 80)
    print("OPTIONS PRICING VALIDATOR")
    print(f"Comparing BS theoretical pricing vs. real market option chains")
    print(f"Haircut: {HAIRCUT*100:.0f}%  |  DTE range: {DTE_MIN}-{DTE_MAX}")
    print(f"OTM: {OTM_MIN*100:.0f}-{OTM_MAX*100:.0f}%  |  Spread width: {SPREAD_WIDTH*100:.0f}%")
    print("=" * 80)

    all_results = []

    # Walk through all date directories
    date_dirs = sorted([
        d for d in CHAINS_DIR.iterdir()
        if d.is_dir() and d.name.startswith("20")
    ])

    for date_dir in date_dirs:
        date_str = date_dir.name

        # Try sector ETFs first, then fallback tickers
        tickers_to_check = SECTOR_ETFS + FALLBACK_TICKERS
        tickers_found = []

        for ticker in tickers_to_check:
            fpath = date_dir / f"{ticker}.parquet"
            if fpath.exists():
                tickers_found.append((ticker, fpath))

        if not tickers_found:
            continue

        print(f"\n--- {date_str} ({len(tickers_found)} tickers) ---")

        for ticker, fpath in tickers_found:
            results = analyze_one_file(fpath, ticker, date_str)
            if results:
                print(f"  {ticker}: {len(results)} valid spreads")
                all_results.extend(results)

    if not all_results:
        print("\n[ERROR] No valid spreads found. Check data quality.")
        return

    # Convert to DataFrame for analysis
    df = pd.DataFrame(all_results)
    print(f"\nTotal valid spreads analyzed: {len(df)}")

    # ---------------------------------------------------------------------------
    # Summary Statistics
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PRICING ERROR ANALYSIS")
    print("=" * 80)

    # Determine which error columns we have
    has_market_iv = "error_bs_skew" in df.columns and df["error_bs_skew"].notna().any()

    # --- 1. Overall Error Stats ---
    print("\n--- Overall Pricing Error (theoretical vs market mid) ---")
    print(f"{'Metric':<35} {'ATR-IV':<15} {'ATR+HC':<15}", end="")
    if has_market_iv:
        print(f"{'MktIV-Skew':<15} {'MktIV+HC':<15}", end="")
    print()

    for label, col_atr, col_atr_hc, col_skew, col_skew_hc in [
        ("Mean error", "error_atr", "error_atr_hc", "error_bs_skew", "error_skew_hc"),
        ("Median error", "error_atr", "error_atr_hc", "error_bs_skew", "error_skew_hc"),
    ]:
        if label.startswith("Mean"):
            v_atr = df[col_atr].mean()
            v_atr_hc = df[col_atr_hc].mean()
        else:
            v_atr = df[col_atr].median()
            v_atr_hc = df[col_atr_hc].median()

        print(f"  {label:<33} {v_atr:>+12.1%}   {v_atr_hc:>+12.1%}", end="")
        if has_market_iv:
            vals = df[col_skew].dropna()
            vals_hc = df[col_skew_hc].dropna()
            if label.startswith("Mean"):
                print(f"   {vals.mean():>+12.1%}   {vals_hc.mean():>+12.1%}", end="")
            else:
                print(f"   {vals.median():>+12.1%}   {vals_hc.median():>+12.1%}", end="")
        print()

    # P90 absolute error
    print(f"  {'P90 |error|':<33} {df['error_atr'].abs().quantile(0.9):>12.1%}   "
          f"{df['error_atr_hc'].abs().quantile(0.9):>12.1%}", end="")
    if has_market_iv:
        print(f"   {df['error_bs_skew'].dropna().abs().quantile(0.9):>12.1%}   "
              f"{df['error_skew_hc'].dropna().abs().quantile(0.9):>12.1%}", end="")
    print()

    # --- 2. Haircut Adequacy ---
    print("\n--- Haircut Adequacy (15%) ---")
    # Overestimate = theoretical + haircut > market = conservative (good)
    # Underestimate = theoretical + haircut < market = too cheap (bad)
    overest_atr = (df["error_atr_hc"] > 0).sum()
    underest_atr = (df["error_atr_hc"] < 0).sum()
    total = len(df)
    print(f"  ATR-IV + haircut OVERESTIMATES cost:   {overest_atr}/{total} ({overest_atr/total:.1%}) [conservative=GOOD]")
    print(f"  ATR-IV + haircut UNDERESTIMATES cost:   {underest_atr}/{total} ({underest_atr/total:.1%}) [too aggressive=BAD]")

    if has_market_iv:
        vals = df["error_skew_hc"].dropna()
        overest = (vals > 0).sum()
        underest = (vals < 0).sum()
        t = len(vals)
        print(f"  MktIV + haircut OVERESTIMATES cost:    {overest}/{t} ({overest/t:.1%}) [conservative=GOOD]")
        print(f"  MktIV + haircut UNDERESTIMATES cost:   {underest}/{t} ({underest/t:.1%}) [too aggressive=BAD]")

    # --- 3. By Ticker ---
    print("\n--- Pricing Error by Ticker ---")
    print(f"  {'Ticker':<8} {'N':<5} {'Mean Err(ATR)':<15} {'Mean Err(ATR+HC)':<18} {'Overest%':<10}")
    for ticker, grp in df.groupby("ticker"):
        n = len(grp)
        me = grp["error_atr"].mean()
        me_hc = grp["error_atr_hc"].mean()
        oe = (grp["error_atr_hc"] > 0).mean()
        print(f"  {ticker:<8} {n:<5} {me:>+12.1%}    {me_hc:>+15.1%}    {oe:>7.0%}")

    # --- 4. By Trade Type ---
    print("\n--- Pricing Error by Trade Type ---")
    print(f"  {'Type':<12} {'N':<5} {'Mean Err(ATR)':<15} {'Mean Err(ATR+HC)':<18} {'Overest%':<10}")
    for ttype, grp in df.groupby("trade_type"):
        n = len(grp)
        me = grp["error_atr"].mean()
        me_hc = grp["error_atr_hc"].mean()
        oe = (grp["error_atr_hc"] > 0).mean()
        print(f"  {ttype:<12} {n:<5} {me:>+12.1%}    {me_hc:>+15.1%}    {oe:>7.0%}")

    # --- 5. Market Cost Distribution ---
    print("\n--- Market Spread Cost Distribution (per share, mid) ---")
    print(f"  {'Type':<12} {'Mean':<10} {'Median':<10} {'P10':<10} {'P90':<10} {'Min':<10} {'Max':<10}")
    for ttype, grp in df.groupby("trade_type"):
        costs = grp["market_mid_cost"]
        print(f"  {ttype:<12} ${costs.mean():<9.3f}${costs.median():<9.3f}"
              f"${costs.quantile(0.1):<9.3f}${costs.quantile(0.9):<9.3f}"
              f"${costs.min():<9.3f}${costs.max():<9.3f}")

    # --- 6. IV Comparison (market vs ATR-estimated) ---
    if has_market_iv:
        print("\n--- IV Comparison: Market vs ATR-Estimated ---")
        print(f"  {'Ticker':<8} {'Mkt IV (avg)':<14} {'ATR IV est':<14} {'IV Ratio':<10}")
        for ticker, grp in df.groupby("ticker"):
            mkt_iv = grp["avg_iv"].mean()
            atr_iv = grp["atr_iv_est"].mean()
            if mkt_iv > 0:
                ratio = atr_iv / mkt_iv
                print(f"  {ticker:<8} {mkt_iv:>11.1%}    {atr_iv:>11.1%}    {ratio:>7.2f}x")

    # --- 7. Detailed per-spread table (first 20) ---
    print("\n--- Sample Spreads (first 20) ---")
    cols_show = ["ticker", "date", "trade_type", "K1", "K2", "dte", "spot",
                 "market_mid_cost", "bs_fair_atr", "bs_haircut_atr", "error_atr", "error_atr_hc"]
    if has_market_iv:
        cols_show.extend(["bs_fair_skew", "error_bs_skew"])

    available_cols = [c for c in cols_show if c in df.columns]
    sample = df[available_cols].head(20)

    # Format numeric columns
    fmt_df = sample.copy()
    for c in fmt_df.columns:
        if "error" in c:
            fmt_df[c] = fmt_df[c].apply(lambda x: f"{x:+.1%}" if pd.notna(x) else "N/A")
        elif c in ("market_mid_cost", "bs_fair_atr", "bs_haircut_atr", "bs_fair_skew"):
            fmt_df[c] = fmt_df[c].apply(lambda x: f"${x:.3f}" if pd.notna(x) else "N/A")
        elif c in ("K1", "K2", "spot"):
            fmt_df[c] = fmt_df[c].apply(lambda x: f"{x:.2f}")

    print(fmt_df.to_string(index=False))

    # ---------------------------------------------------------------------------
    # Save Results
    # ---------------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Full results CSV
    csv_path = OUTPUT_DIR / f"validation_results_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nFull results saved to: {csv_path}")

    # Summary JSON
    summary = {
        "timestamp": timestamp,
        "total_spreads": int(len(df)),
        "dates_analyzed": sorted(df["date"].unique().tolist()),
        "tickers_analyzed": sorted(df["ticker"].unique().tolist()),
        "haircut_pct": HAIRCUT * 100,
        "atr_iv_pricing": {
            "mean_error": float(df["error_atr"].mean()),
            "median_error": float(df["error_atr"].median()),
            "p90_abs_error": float(df["error_atr"].abs().quantile(0.9)),
            "with_haircut": {
                "mean_error": float(df["error_atr_hc"].mean()),
                "median_error": float(df["error_atr_hc"].median()),
                "overestimate_pct": float((df["error_atr_hc"] > 0).mean() * 100),
                "underestimate_pct": float((df["error_atr_hc"] < 0).mean() * 100),
            },
        },
        "by_trade_type": {},
        "by_ticker": {},
    }

    if has_market_iv:
        skew_vals = df["error_bs_skew"].dropna()
        hc_vals = df["error_skew_hc"].dropna()
        summary["market_iv_pricing"] = {
            "mean_error": float(skew_vals.mean()),
            "median_error": float(skew_vals.median()),
            "p90_abs_error": float(skew_vals.abs().quantile(0.9)),
            "with_haircut": {
                "mean_error": float(hc_vals.mean()),
                "overestimate_pct": float((hc_vals > 0).mean() * 100),
                "underestimate_pct": float((hc_vals < 0).mean() * 100),
            },
        }

    for ttype, grp in df.groupby("trade_type"):
        summary["by_trade_type"][ttype] = {
            "count": int(len(grp)),
            "mean_error_atr": float(grp["error_atr"].mean()),
            "mean_error_atr_hc": float(grp["error_atr_hc"].mean()),
            "overestimate_pct": float((grp["error_atr_hc"] > 0).mean() * 100),
            "avg_market_cost": float(grp["market_mid_cost"].mean()),
        }

    for ticker, grp in df.groupby("ticker"):
        summary["by_ticker"][ticker] = {
            "count": int(len(grp)),
            "mean_error_atr": float(grp["error_atr"].mean()),
            "mean_error_atr_hc": float(grp["error_atr_hc"].mean()),
            "overestimate_pct": float((grp["error_atr_hc"] > 0).mean() * 100),
        }

    json_path = OUTPUT_DIR / f"validation_summary_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary JSON saved to: {json_path}")

    # Key verdict
    print("\n" + "=" * 80)
    overest_rate = (df["error_atr_hc"] > 0).mean()
    mean_err = df["error_atr_hc"].mean()
    if overest_rate >= 0.80:
        verdict = "CONSERVATIVE -- 15% haircut overestimates cost in most cases. Strategy cost assumptions are safe."
    elif overest_rate >= 0.50:
        verdict = "ADEQUATE -- 15% haircut covers cost in majority of cases but has some leakage."
    else:
        verdict = "INSUFFICIENT -- 15% haircut underestimates cost too often. Consider increasing haircut."
    print(f"VERDICT: {verdict}")
    print(f"  Overestimate rate: {overest_rate:.0%}  |  Mean error with haircut: {mean_err:+.1%}")
    print("=" * 80)


if __name__ == "__main__":
    run_validation()
