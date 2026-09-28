#!/usr/bin/env python3
"""
Tax Loss Harvesting Rebound Backtest — January Effect
=====================================================
Identifies growth stocks sold for tax losses in Q4 and buys them late December,
expecting a January rebound as selling pressure ends.

6 Variants:
  A) Classic TLH Rebound (top-5 Q4 losers >15% decline, Dec 20 - Jan 31)
  B) Extended Hold (Dec 20 - Feb 28)
  C) Severity Filter (>25% Q4 decline)
  D) Volume Confirmation (Dec volume >1.5x avg)
  E) Small-Cap Focus (sub-$30 names only)
  F) Options Leverage (ATM calls, Feb expiry)

Walk-forward OOT: Jan 2022 - Jul 2026
5-gate validation: Sharpe>0.5, perm p<0.05, regime gap<0.5, MaxDD>-50%, >=20 trades
"""

import json
import warnings
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────────────
ACCOUNT_SIZE = 669.0
SLIPPAGE_PCT = 0.0002  # 0.02% for shares
OPTION_PREMIUM_PCT = 0.03  # 3% of stock price for ~6wk ATM call
OPTION_COMMISSION = 0.65
OPTION_BIDASK_HAIRCUT = 0.05  # 5%
N_PERMUTATIONS = 1000

UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD", "NFLX", "CRM",
    "SNOW", "PLTR", "SOFI", "HOOD", "SNAP", "PINS", "COIN", "RBLX", "RIVN", "UBER",
    "LYFT", "ROKU", "NET", "DDOG", "TTD", "SHOP", "SE", "MELI", "NU", "SQ"
]

SMALL_CAP_NAMES = ["SOFI", "SNAP", "HOOD", "RBLX", "RIVN", "PLTR", "PINS", "COIN", "ROKU"]

# Test years: entry in Dec of year Y, exit in Jan (or Feb) of year Y+1
# OOT window: Jan 2022 - Jul 2026 means we need entries in Dec 2021 through Dec 2025
TEST_YEARS = [2021, 2022, 2023, 2024, 2025]


def download_data():
    """Download price and volume data for all tickers + SPY."""
    tickers = UNIVERSE + ["SPY"]
    print(f"Downloading data for {len(tickers)} tickers...")
    # Need data from Jul 2021 (for Q4 lookback) through Jul 2026
    data = yf.download(tickers, start="2021-06-01", end="2026-07-31",
                       auto_adjust=True, progress=False, group_by="ticker")
    return data


def get_ticker_df(data, ticker):
    """Extract single ticker DataFrame from grouped download."""
    try:
        if isinstance(data.columns, pd.MultiIndex):
            df = data[ticker].copy()
        else:
            df = data.copy()
        df = df.dropna(subset=["Close"])
        return df
    except (KeyError, TypeError):
        return pd.DataFrame()


def compute_spy_regime(spy_df):
    """Bull = SPY > 200-SMA, Bear = SPY < 200-SMA."""
    spy_df = spy_df.copy()
    spy_df["SMA200"] = spy_df["Close"].rolling(200).mean()
    spy_df["regime"] = np.where(spy_df["Close"] > spy_df["SMA200"], "bull", "bear")
    return spy_df


def get_q4_performance(data, ticker, year):
    """Compute Oct 1 - Dec 19 return for a ticker in a given year."""
    df = get_ticker_df(data, ticker)
    if df.empty:
        return None, None, None

    q4_start = f"{year}-10-01"
    q4_end = f"{year}-12-19"

    mask = (df.index >= q4_start) & (df.index <= q4_end)
    q4 = df.loc[mask]

    if len(q4) < 20:
        return None, None, None

    ret = (q4["Close"].iloc[-1] / q4["Close"].iloc[0]) - 1.0
    # Average daily volume in Q4
    avg_vol = q4["Volume"].mean() if "Volume" in q4.columns else None
    dec_price = q4["Close"].iloc[-1]

    return ret, avg_vol, dec_price


def get_dec_volume_ratio(data, ticker, year):
    """Ratio of Dec volume to trailing 6-month average volume."""
    df = get_ticker_df(data, ticker)
    if df.empty:
        return None

    dec_start = f"{year}-12-01"
    dec_end = f"{year}-12-31"
    trailing_start = f"{year}-06-01"
    trailing_end = f"{year}-11-30"

    dec_vol = df.loc[(df.index >= dec_start) & (df.index <= dec_end), "Volume"]
    trail_vol = df.loc[(df.index >= trailing_start) & (df.index <= trailing_end), "Volume"]

    if len(dec_vol) < 5 or len(trail_vol) < 20:
        return None

    ratio = dec_vol.mean() / trail_vol.mean() if trail_vol.mean() > 0 else None
    return ratio


def get_entry_exit_prices(data, ticker, year, exit_month_end):
    """
    Entry: first trading day on or after Dec 20 of `year`.
    Exit: last trading day of exit_month_end (e.g., '2023-01-31' or '2023-02-28').
    Returns (entry_date, entry_price, exit_date, exit_price) or Nones.
    """
    df = get_ticker_df(data, ticker)
    if df.empty:
        return None, None, None, None

    # Entry: Dec 20 of year
    entry_target = f"{year}-12-20"
    entry_candidates = df.loc[(df.index >= entry_target) & (df.index <= f"{year}-12-31")]
    if entry_candidates.empty:
        return None, None, None, None

    entry_date = entry_candidates.index[0]
    entry_price = entry_candidates["Close"].iloc[0]

    # Exit: last day of exit month
    exit_candidates = df.loc[(df.index >= f"{year+1}-01-01") & (df.index <= exit_month_end)]
    if exit_candidates.empty:
        return None, None, None, None

    exit_date = exit_candidates.index[-1]
    exit_price = exit_candidates["Close"].iloc[-1]

    return entry_date, entry_price, exit_date, exit_price


def select_stocks_variant(data, year, variant):
    """Select stocks based on variant rules. Returns list of (ticker, q4_return)."""
    candidates = []

    if variant == "E":
        universe = SMALL_CAP_NAMES
    else:
        universe = UNIVERSE

    for ticker in universe:
        q4_ret, avg_vol, dec_price = get_q4_performance(data, ticker, year)
        if q4_ret is None:
            continue

        # Decline threshold
        if variant == "C":
            threshold = -0.25
        else:
            threshold = -0.15

        if q4_ret > threshold:
            continue

        # Small-cap price filter
        if variant == "E" and dec_price is not None and dec_price > 30:
            continue

        # Volume confirmation
        if variant == "D":
            vol_ratio = get_dec_volume_ratio(data, ticker, year)
            if vol_ratio is None or vol_ratio < 1.5:
                continue

        candidates.append((ticker, q4_ret))

    # Sort by worst performance (most negative first)
    candidates.sort(key=lambda x: x[1])

    # Take top 5 worst performers (or fewer if not enough qualify)
    return candidates[:5]


def run_variant(data, spy_regime_df, variant):
    """Run a single variant backtest across all test years."""
    trades = []

    # Exit month depends on variant
    if variant == "B":
        exit_month_offset = 2  # Feb
    else:
        exit_month_offset = 1  # Jan

    for year in TEST_YEARS:
        selected = select_stocks_variant(data, year, variant)
        if not selected:
            continue

        n_stocks = len(selected)
        alloc_per_stock = ACCOUNT_SIZE / n_stocks

        for ticker, q4_ret in selected:
            # Determine exit date
            exit_year = year + 1
            if exit_month_offset == 1:
                exit_month_end = f"{exit_year}-01-31"
            else:
                # Feb: handle leap years
                if exit_year % 4 == 0 and (exit_year % 100 != 0 or exit_year % 400 == 0):
                    exit_month_end = f"{exit_year}-02-29"
                else:
                    exit_month_end = f"{exit_year}-02-28"

            entry_date, entry_price, exit_date, exit_price = get_entry_exit_prices(
                data, ticker, year, exit_month_end
            )
            if entry_price is None or exit_price is None:
                continue

            if variant == "F":
                # Options variant
                premium = entry_price * OPTION_PREMIUM_PCT
                premium_with_haircut = premium * (1 + OPTION_BIDASK_HAIRCUT)
                # How many contracts can we buy?
                cost_per_contract = premium_with_haircut * 100 + OPTION_COMMISSION
                n_contracts = max(1, int(alloc_per_stock / cost_per_contract))
                total_cost = n_contracts * cost_per_contract

                # P&L: intrinsic value at exit minus premium paid
                stock_move = exit_price - entry_price
                if stock_move > 0:
                    intrinsic = stock_move * 100 * n_contracts
                    exit_commission = OPTION_COMMISSION * n_contracts
                    pnl = intrinsic - total_cost - exit_commission
                else:
                    # Options expire worthless (or close for residual)
                    pnl = -total_cost

                pnl_pct = pnl / alloc_per_stock
            else:
                # Shares variant
                shares = int(alloc_per_stock / entry_price) if entry_price > 0 else 0
                if shares == 0:
                    # Even fractional
                    shares = alloc_per_stock / entry_price

                entry_cost = entry_price * (1 + SLIPPAGE_PCT)
                exit_proceeds = exit_price * (1 - SLIPPAGE_PCT)

                pnl = (exit_proceeds - entry_cost) * shares
                pnl_pct = (exit_proceeds / entry_cost) - 1.0

            # Determine regime at entry
            entry_ts = pd.Timestamp(entry_date)
            regime_mask = spy_regime_df.index <= entry_ts
            if regime_mask.any():
                regime = spy_regime_df.loc[regime_mask, "regime"].iloc[-1]
            else:
                regime = "unknown"

            trades.append({
                "variant": variant,
                "year": year,
                "ticker": ticker,
                "q4_return": round(q4_ret * 100, 2),
                "entry_date": str(entry_date.date()) if hasattr(entry_date, 'date') else str(entry_date),
                "exit_date": str(exit_date.date()) if hasattr(exit_date, 'date') else str(exit_date),
                "entry_price": round(float(entry_price), 2),
                "exit_price": round(float(exit_price), 2),
                "pnl": round(float(pnl), 2),
                "pnl_pct": round(float(pnl_pct) * 100, 2),
                "regime": regime,
            })

    return trades


def compute_metrics(trades):
    """Compute performance metrics from trade list."""
    if not trades:
        return {
            "n_trades": 0, "sharpe": 0, "sortino": 0, "profit_factor": 0,
            "win_rate": 0, "total_pnl": 0, "total_pnl_pct": 0,
            "max_drawdown_pct": 0, "avg_pnl": 0, "avg_pnl_pct": 0,
        }

    pnls = [t["pnl"] for t in trades]
    pnl_pcts = [t["pnl_pct"] / 100.0 for t in trades]

    n = len(pnls)
    total_pnl = sum(pnls)
    avg_pnl = np.mean(pnls)
    avg_pnl_pct = np.mean(pnl_pcts)

    winners = [p for p in pnls if p > 0]
    losers = [p for p in pnls if p <= 0]
    win_rate = len(winners) / n if n > 0 else 0

    gross_profit = sum(winners) if winners else 0
    gross_loss = abs(sum(losers)) if losers else 0.001
    profit_factor = gross_profit / gross_loss

    # Sharpe: annualize assuming ~5 trades/year (seasonal strategy)
    trades_per_year = n / len(TEST_YEARS) if len(TEST_YEARS) > 0 else 1
    std_pnl_pct = np.std(pnl_pcts) if len(pnl_pcts) > 1 else 0.001
    sharpe = (avg_pnl_pct / std_pnl_pct) * np.sqrt(trades_per_year) if std_pnl_pct > 0 else 0

    # Sortino
    downside = [p for p in pnl_pcts if p < 0]
    downside_std = np.std(downside) if len(downside) > 1 else 0.001
    sortino = (avg_pnl_pct / downside_std) * np.sqrt(trades_per_year) if downside_std > 0 else 0

    # Max drawdown on cumulative equity
    equity = np.cumsum(pnls) + ACCOUNT_SIZE
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd = float(dd.min()) if len(dd) > 0 else 0

    return {
        "n_trades": n,
        "sharpe": round(float(sharpe), 3),
        "sortino": round(float(sortino), 3),
        "profit_factor": round(float(profit_factor), 3),
        "win_rate": round(float(win_rate) * 100, 1),
        "total_pnl": round(float(total_pnl), 2),
        "total_pnl_pct": round(float(total_pnl / ACCOUNT_SIZE) * 100, 2),
        "max_drawdown_pct": round(float(max_dd) * 100, 2),
        "avg_pnl": round(float(avg_pnl), 2),
        "avg_pnl_pct": round(float(avg_pnl_pct) * 100, 2),
    }


def compute_regime_metrics(trades):
    """Compute per-regime Sharpe to check regime gap."""
    bull_trades = [t for t in trades if t["regime"] == "bull"]
    bear_trades = [t for t in trades if t["regime"] == "bear"]

    bull_m = compute_metrics(bull_trades)
    bear_m = compute_metrics(bear_trades)

    bull_sharpe = bull_m["sharpe"]
    bear_sharpe = bear_m["sharpe"]

    denom = max(abs(bull_sharpe), abs(bear_sharpe), 0.001)
    regime_gap = abs(bull_sharpe - bear_sharpe) / denom

    return {
        "bull_sharpe": bull_sharpe,
        "bear_sharpe": bear_sharpe,
        "bull_trades": bull_m["n_trades"],
        "bear_trades": bear_m["n_trades"],
        "regime_gap": round(float(regime_gap), 3),
    }


def permutation_test(trades, n_perms=N_PERMUTATIONS):
    """
    Shuffle stock selection randomly to test if observed returns are better than chance.
    We shuffle the pnl_pct values and compare mean to observed.
    """
    if len(trades) < 5:
        return 1.0

    observed_mean = np.mean([t["pnl_pct"] for t in trades])
    pnl_pcts = np.array([t["pnl_pct"] for t in trades])

    rng = np.random.default_rng(42)
    count_better = 0

    for _ in range(n_perms):
        # Randomly flip signs (null hypothesis: no directional edge)
        signs = rng.choice([-1, 1], size=len(pnl_pcts))
        shuffled_mean = np.mean(pnl_pcts * signs)
        if shuffled_mean >= observed_mean:
            count_better += 1

    p_value = (count_better + 1) / (n_perms + 1)
    return round(float(p_value), 4)


def validate_5gate(metrics, regime_metrics, p_value):
    """Apply 5-gate validation."""
    gates = {
        "sharpe_gt_0.5": metrics["sharpe"] > 0.5,
        "perm_p_lt_0.05": p_value < 0.05,
        "regime_gap_lt_0.5": regime_metrics["regime_gap"] < 0.5,
        "max_dd_gt_neg50": metrics["max_drawdown_pct"] > -50,
        "min_20_trades": metrics["n_trades"] >= 20,
    }
    gates["all_passed"] = all(gates.values())
    return gates


def main():
    print("=" * 70)
    print("TAX LOSS HARVESTING REBOUND BACKTEST")
    print("=" * 70)

    # Download data
    data = download_data()

    # SPY regime
    spy_df = get_ticker_df(data, "SPY")
    spy_regime_df = compute_spy_regime(spy_df)

    variant_names = {
        "A": "Classic TLH Rebound (top-5, >15% decline, Dec20-Jan31)",
        "B": "Extended Hold (Dec20-Feb28)",
        "C": "Severity Filter (>25% decline)",
        "D": "Volume Confirmation (Dec vol >1.5x avg)",
        "E": "Small-Cap Focus (<$30 names)",
        "F": "Options Leverage (ATM calls, Feb expiry)",
    }

    results = {}

    for variant in ["A", "B", "C", "D", "E", "F"]:
        print(f"\n{'─' * 60}")
        print(f"Variant {variant}: {variant_names[variant]}")
        print(f"{'─' * 60}")

        trades = run_variant(data, spy_regime_df, variant)
        metrics = compute_metrics(trades)
        regime = compute_regime_metrics(trades)

        print(f"  Trades: {metrics['n_trades']}")
        print(f"  Total P&L: ${metrics['total_pnl']:.2f} ({metrics['total_pnl_pct']:.1f}%)")
        print(f"  Win Rate: {metrics['win_rate']:.1f}%")
        print(f"  Sharpe: {metrics['sharpe']:.3f}")
        print(f"  Sortino: {metrics['sortino']:.3f}")
        print(f"  Profit Factor: {metrics['profit_factor']:.3f}")
        print(f"  Max DD: {metrics['max_drawdown_pct']:.1f}%")
        print(f"  Regime: Bull Sharpe={regime['bull_sharpe']:.3f} ({regime['bull_trades']}t), "
              f"Bear Sharpe={regime['bear_sharpe']:.3f} ({regime['bear_trades']}t), "
              f"Gap={regime['regime_gap']:.3f}")

        # Permutation test
        print(f"  Running {N_PERMUTATIONS} permutations...")
        p_value = permutation_test(trades)
        print(f"  Permutation p-value: {p_value:.4f}")

        # 5-gate validation
        gates = validate_5gate(metrics, regime, p_value)
        print(f"  5-Gate Validation:")
        for gate, passed in gates.items():
            status = "PASS" if passed else "FAIL"
            print(f"    {gate}: {status}")

        # Trade details by year
        print(f"\n  Trades by year:")
        for year in TEST_YEARS:
            year_trades = [t for t in trades if t["year"] == year]
            if year_trades:
                year_pnl = sum(t["pnl"] for t in year_trades)
                tickers = [t["ticker"] for t in year_trades]
                print(f"    {year}->{year+1}: {len(year_trades)} trades, "
                      f"P&L=${year_pnl:.2f}, stocks={tickers}")

        results[f"variant_{variant}"] = {
            "name": variant_names[variant],
            "metrics": metrics,
            "regime": regime,
            "p_value": p_value,
            "gates": gates,
            "trades": trades,
        }

    # ── Summary Table ─────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Var':<4} {'Name':<45} {'Trades':>6} {'P&L':>8} {'Sharpe':>7} {'WR':>6} {'PF':>6} {'DD':>7} {'5G':>4}")
    print(f"{'─' * 4} {'─' * 45} {'─' * 6} {'─' * 8} {'─' * 7} {'─' * 6} {'─' * 6} {'─' * 7} {'─' * 4}")

    for v in ["A", "B", "C", "D", "E", "F"]:
        r = results[f"variant_{v}"]
        m = r["metrics"]
        passed = "YES" if r["gates"]["all_passed"] else "NO"
        print(f"  {v}  {r['name']:<45} {m['n_trades']:>5} {m['total_pnl']:>8.2f} "
              f"{m['sharpe']:>7.3f} {m['win_rate']:>5.1f}% {m['profit_factor']:>5.2f} "
              f"{m['max_drawdown_pct']:>6.1f}% {passed:>4}")

    # ── Best variant ──────────────────────────────────────────────────────
    passing = {k: v for k, v in results.items() if v["gates"]["all_passed"]}
    if passing:
        best = max(passing.items(), key=lambda x: x[1]["metrics"]["sharpe"])
        print(f"\nBest passing variant: {best[0]} (Sharpe={best[1]['metrics']['sharpe']:.3f})")
    else:
        # Best by Sharpe even if not passing
        best = max(results.items(), key=lambda x: x[1]["metrics"]["sharpe"])
        print(f"\nNo variant passed all 5 gates. Best Sharpe: {best[0]} ({best[1]['metrics']['sharpe']:.3f})")
        # Show which gates failed
        for gate, passed in best[1]["gates"].items():
            if not passed and gate != "all_passed":
                print(f"  FAILED: {gate}")

    # ── Save results ──────────────────────────────────────────────────────
    output_path = Path("/home/jupiter/Lvl3Quant/data/tax_loss_rebound_results.json")

    # Strip trades for JSON (keep summary)
    save_results = {}
    for k, v in results.items():
        save_results[k] = {
            "name": v["name"],
            "metrics": v["metrics"],
            "regime": v["regime"],
            "p_value": v["p_value"],
            "gates": v["gates"],
            "trades": v["trades"],
        }

    save_results["metadata"] = {
        "account_size": ACCOUNT_SIZE,
        "universe_size": len(UNIVERSE),
        "test_years": TEST_YEARS,
        "oot_window": "Jan 2022 - Jul 2026",
        "slippage_pct": SLIPPAGE_PCT,
        "option_premium_pct": OPTION_PREMIUM_PCT,
        "n_permutations": N_PERMUTATIONS,
        "run_timestamp": datetime.now().isoformat(),
    }

    output_path.write_text(json.dumps(save_results, indent=2, default=str))
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
