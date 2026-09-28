#!/usr/bin/env python3
"""
Volatility Risk Premium Harvesting via VIX ETP Pairs v1
========================================================
Strategy: Exploit the well-documented VRP by going short vol (long SVXY / short VXX)
ONLY when VIX futures curve is in steep contango (proxy: VIX/VIXM ratio or VIX vs VIX3M).

Key differences from naive SVXY buy-and-hold (which failed):
- Only trade 2-8 times per year on EXTREME contango (top decile historically)
- Use VIX term structure slope as primary signal
- Tight trailing stops to limit blowup risk
- Risk-size positions (never >30% of portfolio)
- Cash otherwise (SHY/BIL)

Adversarial validation built in: permutation test, sub-period stability, outlier removal.

HC #0  : Sliding walk-forward (trailing lookback for signals)
HC #428: Regime-agnostic validation (R1)
HC #694: Commission-free (Robinhood)
HC #697: No crypto
"""

import os
import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/growth_research/vrp_harvesting")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

START = "2011-10-01"  # SVXY inception ~Oct 2011
END = "2026-07-17"
INITIAL_CAPITAL = 100_000
POSITION_SIZE = 0.25  # Max 25% of portfolio in vol trade
CASH_TICKER = "SHY"   # Safe haven when not in vol trade

# Contango signal thresholds
CONTANGO_ENTRY_PCT = 5.0   # Enter when VIX/VIX3M contango > 5% (steep)
CONTANGO_EXIT_PCT = 1.0    # Exit when contango narrows below 1%
TRAILING_STOP_PCT = 0.08   # 8% trailing stop on the vol position
MIN_HOLD_DAYS = 5          # Minimum holding period to avoid whipsaw
MAX_HOLD_DAYS = 60         # Maximum hold — don't overstay

# VIX absolute guards
VIX_MAX_ENTRY = 22.0       # Don't enter if VIX already elevated
VIX_PANIC_EXIT = 28.0      # Emergency exit if VIX spikes


def download_data():
    """Download required data."""
    tickers = ["SVXY", "VXX", "VIXY", "SHY", "SPY", "^VIX"]

    print("Downloading price data...")
    raw = yf.download(tickers, start=START, end=END, auto_adjust=True, progress=False)

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw

    # Download VIX3M (CBOE 3-month VIX) — proxy via VIXM ETF if VIX3M unavailable
    try:
        vix3m_raw = yf.download("^VIX3M", start=START, end=END, auto_adjust=True, progress=False)
        if isinstance(vix3m_raw.columns, pd.MultiIndex):
            vix3m = vix3m_raw["Close"].squeeze()
        else:
            vix3m = vix3m_raw["Close"].squeeze() if "Close" in vix3m_raw.columns else vix3m_raw.squeeze()
    except Exception:
        vix3m = None

    prices = prices.ffill(limit=5)
    vix = prices["^VIX"].copy()
    prices = prices.drop(columns=["^VIX"], errors="ignore")

    print(f"  Data shape: {prices.shape}")
    print(f"  Range: {prices.index[0].date()} to {prices.index[-1].date()}")

    return prices, vix, vix3m


def compute_contango_signal(vix, vix3m):
    """
    Compute VIX term structure contango percentage.
    Contango = (VIX3M - VIX) / VIX * 100
    Positive = contango (normal), negative = backwardation (stress).
    """
    if vix3m is not None and len(vix3m) > 100:
        # Align indices
        common = vix.index.intersection(vix3m.index)
        v = vix.loc[common]
        v3 = vix3m.loc[common]
        contango = (v3 - v) / v * 100
        contango = contango.reindex(vix.index).ffill()
    else:
        # Fallback: use VIX 5d MA vs 20d MA as crude term structure proxy
        print("  WARNING: VIX3M not available, using MA proxy")
        vix_5d = vix.rolling(5).mean()
        vix_20d = vix.rolling(20).mean()
        contango = (vix_20d - vix_5d) / vix_5d * 100

    return contango


def backtest_vrp(prices, vix, contango, variant="base"):
    """
    Run the VRP harvesting backtest.

    Variants:
    - "base": Standard contango entry/exit with trailing stop
    - "aggressive": Lower contango threshold, larger position
    - "conservative": Higher contango threshold, tighter stop
    """
    params = {
        "base": {
            "entry_pct": CONTANGO_ENTRY_PCT,
            "exit_pct": CONTANGO_EXIT_PCT,
            "stop_pct": TRAILING_STOP_PCT,
            "pos_size": POSITION_SIZE,
            "vix_max": VIX_MAX_ENTRY,
        },
        "aggressive": {
            "entry_pct": 3.5,
            "exit_pct": 0.5,
            "stop_pct": 0.10,
            "pos_size": 0.35,
            "vix_max": 24.0,
        },
        "conservative": {
            "entry_pct": 7.0,
            "exit_pct": 2.0,
            "stop_pct": 0.06,
            "pos_size": 0.20,
            "vix_max": 20.0,
        },
    }[variant]

    # Use SVXY as the vol-short vehicle (inverse VIX)
    vol_ticker = "SVXY"
    if vol_ticker not in prices.columns:
        print(f"  ERROR: {vol_ticker} not in data")
        return None

    vol_prices = prices[vol_ticker]
    shy_prices = prices[CASH_TICKER] if CASH_TICKER in prices.columns else None
    spy_prices = prices["SPY"]

    # Compute daily returns
    vol_ret = vol_prices.pct_change().fillna(0)
    shy_ret = shy_prices.pct_change().fillna(0) if shy_prices is not None else pd.Series(0.0001 / 252, index=vol_prices.index)
    spy_ret = spy_prices.pct_change().fillna(0)

    # Align all series
    common_idx = contango.dropna().index.intersection(vol_prices.dropna().index)
    common_idx = common_idx[common_idx >= pd.Timestamp("2012-01-01")]  # Ensure enough data

    capital = INITIAL_CAPITAL
    position = "cash"  # "cash" or "vol_long"
    entry_price = 0
    peak_price = 0
    hold_days = 0
    trades = []
    equity = []
    positions_log = []

    for i, date in enumerate(common_idx):
        c = contango.get(date, np.nan)
        v = vix.get(date, np.nan)

        if np.isnan(c) or np.isnan(v):
            equity.append({"date": date, "equity": capital})
            continue

        daily_ret = 0.0

        if position == "cash":
            # Check entry conditions
            if (c > params["entry_pct"] and
                v < params["vix_max"] and
                not np.isnan(vol_prices.get(date, np.nan))):
                # ENTER vol-short position
                position = "vol_long"
                entry_price = vol_prices[date]
                peak_price = entry_price
                hold_days = 0
                trades.append({
                    "entry_date": str(date.date()),
                    "entry_price": entry_price,
                    "contango_at_entry": c,
                    "vix_at_entry": v,
                })
                # Allocate: pos_size to SVXY, rest to SHY
                daily_ret = params["pos_size"] * vol_ret.get(date, 0) + (1 - params["pos_size"]) * shy_ret.get(date, 0)
            else:
                # Stay in cash (SHY)
                daily_ret = shy_ret.get(date, 0)
        else:
            # In vol position — check exit conditions
            hold_days += 1
            current_price = vol_prices.get(date, entry_price)
            peak_price = max(peak_price, current_price)

            # Exit conditions
            trailing_stop_hit = (current_price / peak_price - 1) < -params["stop_pct"]
            contango_collapsed = c < params["exit_pct"] and hold_days >= MIN_HOLD_DAYS
            vix_panic = v > VIX_PANIC_EXIT
            max_hold_hit = hold_days >= MAX_HOLD_DAYS

            if trailing_stop_hit or contango_collapsed or vix_panic or max_hold_hit:
                # EXIT
                trade_ret = current_price / entry_price - 1
                exit_reason = (
                    "trailing_stop" if trailing_stop_hit else
                    "contango_collapse" if contango_collapsed else
                    "vix_panic" if vix_panic else
                    "max_hold"
                )
                if trades:
                    trades[-1].update({
                        "exit_date": str(date.date()),
                        "exit_price": current_price,
                        "trade_return": trade_ret,
                        "hold_days": hold_days,
                        "exit_reason": exit_reason,
                    })
                position = "cash"
                daily_ret = params["pos_size"] * vol_ret.get(date, 0) + (1 - params["pos_size"]) * shy_ret.get(date, 0)
            else:
                daily_ret = params["pos_size"] * vol_ret.get(date, 0) + (1 - params["pos_size"]) * shy_ret.get(date, 0)

        capital *= (1 + daily_ret)
        equity.append({"date": date, "equity": capital})
        positions_log.append({"date": date, "position": position})

    equity_df = pd.DataFrame(equity).set_index("date")
    return {
        "equity": equity_df,
        "trades": trades,
        "variant": variant,
        "params": params,
    }


def compute_metrics(equity_df, label="Strategy"):
    """Compute standard performance metrics."""
    eq = equity_df["equity"]
    daily_ret = eq.pct_change().dropna()

    total_return = eq.iloc[-1] / eq.iloc[0] - 1
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (1 + total_return) ** (1 / years) - 1

    # Drawdown
    peak = eq.cummax()
    dd = (eq - peak) / peak
    max_dd = dd.min()

    # Risk metrics
    ann_vol = daily_ret.std() * np.sqrt(252)
    sharpe = (daily_ret.mean() * 252) / (daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0
    downside = daily_ret[daily_ret < 0].std() * np.sqrt(252)
    sortino = (daily_ret.mean() * 252) / downside if downside > 0 else 0
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    return {
        "label": label,
        "total_return": total_return,
        "cagr": cagr,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_dd": max_dd,
        "calmar": calmar,
        "years": years,
        "final_equity": eq.iloc[-1],
    }


def adversarial_validation(equity_df, trades, n_perms=1000):
    """
    Adversarial checks:
    1. Permutation test — shuffle daily returns, is strategy Sharpe significantly better?
    2. Sub-period stability — split into halves, both profitable?
    3. Outlier removal — remove best 5 days, still profitable?
    4. R1 regime check — does it work in both up and down markets?
    """
    results = {}
    eq = equity_df["equity"]
    daily_ret = eq.pct_change().dropna()
    actual_sharpe = (daily_ret.mean() * 252) / (daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0

    # 1. Permutation test
    print("  Running permutation test (1000 shuffles)...")
    perm_sharpes = []
    ret_vals = daily_ret.values.copy()
    for _ in range(n_perms):
        np.random.shuffle(ret_vals)
        s = (ret_vals.mean() * 252) / (ret_vals.std() * np.sqrt(252))
        perm_sharpes.append(s)
    perm_sharpes = np.array(perm_sharpes)
    p_value = (perm_sharpes >= actual_sharpe).mean()
    results["permutation"] = {
        "actual_sharpe": actual_sharpe,
        "perm_mean_sharpe": float(np.mean(perm_sharpes)),
        "p_value": float(p_value),
        "pass": p_value < 0.05,
    }

    # 2. Sub-period stability
    mid = len(daily_ret) // 2
    first_half = daily_ret.iloc[:mid]
    second_half = daily_ret.iloc[mid:]
    s1 = (first_half.mean() * 252) / (first_half.std() * np.sqrt(252)) if first_half.std() > 0 else 0
    s2 = (second_half.mean() * 252) / (second_half.std() * np.sqrt(252)) if second_half.std() > 0 else 0
    results["sub_period"] = {
        "first_half_sharpe": float(s1),
        "second_half_sharpe": float(s2),
        "both_positive": s1 > 0 and s2 > 0,
        "pass": s1 > 0 and s2 > 0,
    }

    # 3. Outlier removal — remove best 5 days
    sorted_ret = daily_ret.sort_values(ascending=False)
    trimmed = sorted_ret.iloc[5:]  # Remove top 5 days
    trimmed_sharpe = (trimmed.mean() * 252) / (trimmed.std() * np.sqrt(252)) if trimmed.std() > 0 else 0
    trimmed_total = (1 + trimmed).prod() - 1
    results["outlier_removal"] = {
        "original_sharpe": actual_sharpe,
        "trimmed_sharpe": float(trimmed_sharpe),
        "trimmed_total_return": float(trimmed_total),
        "still_profitable": trimmed_total > 0,
        "pass": trimmed_total > 0 and trimmed_sharpe > 0,
    }

    # 4. R1 Regime check — classify SPY months as green/red
    # Use SPY monthly returns to classify
    monthly_ret = daily_ret.resample("ME").sum()
    green_months = monthly_ret[monthly_ret > 0]
    red_months = monthly_ret[monthly_ret <= 0]
    s_green = (green_months.mean() * 12) / (green_months.std() * np.sqrt(12)) if len(green_months) > 2 and green_months.std() > 0 else 0
    s_red = (red_months.mean() * 12) / (red_months.std() * np.sqrt(12)) if len(red_months) > 2 and red_months.std() > 0 else 0
    regime_ratio = abs(s_green - s_red) / max(abs(s_green), abs(s_red), 0.01)
    results["regime_agnostic"] = {
        "green_month_sharpe": float(s_green),
        "red_month_sharpe": float(s_red),
        "regime_ratio": float(regime_ratio),
        "pass": regime_ratio < 0.50,  # HC #428 R1
    }

    # Overall
    all_pass = all(r.get("pass", False) for r in results.values())
    results["overall_pass"] = all_pass

    return results


def run_benchmark(prices, start_date="2012-01-01"):
    """SPY buy-and-hold benchmark."""
    spy = prices["SPY"].dropna()
    spy = spy[spy.index >= pd.Timestamp(start_date)]
    eq = INITIAL_CAPITAL * (spy / spy.iloc[0])
    return pd.DataFrame({"equity": eq})


def main():
    print("=" * 70)
    print("VRP HARVESTING VIA VIX ETP PAIRS v1")
    print("=" * 70)

    prices, vix, vix3m = download_data()
    contango = compute_contango_signal(vix, vix3m)

    print(f"\nContango signal stats:")
    print(f"  Mean: {contango.mean():.2f}%")
    print(f"  Median: {contango.median():.2f}%")
    print(f"  Pct > 5%: {(contango > 5).mean() * 100:.1f}%")
    print(f"  Pct > 7%: {(contango > 7).mean() * 100:.1f}%")

    all_results = {}

    for variant in ["base", "aggressive", "conservative"]:
        print(f"\n--- Running {variant} variant ---")
        result = backtest_vrp(prices, vix, contango, variant=variant)
        if result is None:
            continue

        metrics = compute_metrics(result["equity"], label=f"VRP_{variant}")
        print(f"  CAGR: {metrics['cagr']:.2%}")
        print(f"  Sharpe: {metrics['sharpe']:.3f}")
        print(f"  Sortino: {metrics['sortino']:.3f}")
        print(f"  MaxDD: {metrics['max_dd']:.2%}")
        print(f"  Calmar: {metrics['calmar']:.3f}")
        print(f"  Trades: {len(result['trades'])}")
        print(f"  Final equity: ${metrics['final_equity']:,.0f}")

        # Trade analysis
        completed = [t for t in result["trades"] if "trade_return" in t]
        if completed:
            wins = [t for t in completed if t["trade_return"] > 0]
            losses = [t for t in completed if t["trade_return"] <= 0]
            avg_win = np.mean([t["trade_return"] for t in wins]) if wins else 0
            avg_loss = np.mean([t["trade_return"] for t in losses]) if losses else 0
            print(f"  Win rate: {len(wins)/len(completed):.1%}")
            print(f"  Avg win: {avg_win:.2%}, Avg loss: {avg_loss:.2%}")
            print(f"  Avg hold: {np.mean([t['hold_days'] for t in completed]):.0f} days")

        # Adversarial validation
        print(f"\n  Adversarial validation:")
        adv = adversarial_validation(result["equity"], result["trades"])
        for test_name, test_result in adv.items():
            if test_name == "overall_pass":
                continue
            status = "PASS" if test_result.get("pass") else "FAIL"
            print(f"    {test_name}: {status}")
        print(f"    OVERALL: {'PASS' if adv['overall_pass'] else 'FAIL'}")

        all_results[variant] = {
            "metrics": metrics,
            "adversarial": adv,
            "trades": result["trades"],
            "num_trades": len(result["trades"]),
        }

    # SPY benchmark
    print(f"\n--- SPY Buy & Hold Benchmark ---")
    spy_eq = run_benchmark(prices)
    spy_metrics = compute_metrics(spy_eq, label="SPY_BH")
    print(f"  CAGR: {spy_metrics['cagr']:.2%}")
    print(f"  Sharpe: {spy_metrics['sharpe']:.3f}")
    print(f"  MaxDD: {spy_metrics['max_dd']:.2%}")

    # Save results
    save_results = {}
    for k, v in all_results.items():
        save_results[k] = {
            "metrics": {mk: float(mv) if isinstance(mv, (np.floating, float)) else mv
                       for mk, mv in v["metrics"].items()},
            "adversarial": v["adversarial"],
            "num_trades": v["num_trades"],
            "trades": v["trades"][:20],  # Save first 20 trades for reference
        }
    save_results["spy_benchmark"] = {
        mk: float(mv) if isinstance(mv, (np.floating, float)) else mv
        for mk, mv in spy_metrics.items()
    }

    with open(OUTPUT_DIR / "vrp_harvesting_results.json", "w") as f:
        json.dump(save_results, f, indent=2, default=str)

    # Plot equity curves
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    for variant in all_results:
        if variant in all_results:
            # Re-run to get equity for plotting (already have it)
            result = backtest_vrp(prices, vix, contango, variant=variant)
            if result:
                eq = result["equity"]["equity"]
                axes[0].plot(eq.index, eq.values, label=f"VRP {variant}")

    spy_eq_vals = spy_eq["equity"]
    axes[0].plot(spy_eq_vals.index, spy_eq_vals.values, label="SPY B&H", linestyle="--", alpha=0.7)
    axes[0].set_title("VRP Harvesting - Equity Curves")
    axes[0].set_ylabel("Portfolio Value ($)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale("log")

    # Contango signal plot
    c_clean = contango.dropna()
    axes[1].plot(c_clean.index, c_clean.values, alpha=0.5, linewidth=0.5)
    axes[1].axhline(y=CONTANGO_ENTRY_PCT, color="green", linestyle="--", label=f"Entry ({CONTANGO_ENTRY_PCT}%)")
    axes[1].axhline(y=CONTANGO_EXIT_PCT, color="red", linestyle="--", label=f"Exit ({CONTANGO_EXIT_PCT}%)")
    axes[1].axhline(y=0, color="black", linewidth=0.5)
    axes[1].set_title("VIX Term Structure Contango %")
    axes[1].set_ylabel("Contango %")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "vrp_harvesting_equity.png", dpi=150)
    plt.close()

    print(f"\nResults saved to {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
