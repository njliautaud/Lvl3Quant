#!/usr/bin/env python3
"""
Institutional Rebalancing Flow — Sector ETF Options Signal Backtest
===================================================================
Hypothesis: Month-end/quarter-end institutional rebalancing creates
predictable 3-5 day mean-reversion flows in sector ETFs.

Variants:
  A. Simple worst-2 buy / best-2 sell at every month-end
  B. Only when divergence > 1.5 stdev
  C. Quarter-end only
  D. Month-end base + quarter-end amplified
  E. Divergence-weighted sizing
  F. With momentum confirmation (5-day turn)

5-Gate System: Sharpe>0.5, p<0.05, regime_gap<0.50, MDD<50%, trades>=30
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from typing import Dict, List, Tuple
import json
import os

# ─── Configuration ───────────────────────────────────────────────────
SECTOR_ETFS = ["XLE", "XLU", "XLF", "XLK", "XLY", "XLP", "XLB", "XLI", "XLV", "XLRE", "XLC"]
BENCHMARK = "SPY"
START_DATE = "2017-06-01"  # extra buffer for lookback
END_DATE = "2026-08-18"
TRADE_START = "2018-01-01"  # actual trade period starts here

LOOKBACK_DAYS = 21       # trailing relative return window
ENTRY_OFFSET = -5        # enter T-5 before month-end
EXIT_OFFSET = 2          # exit T+2 after month-end
N_PERMUTATIONS = 1000
SEED = 42

# 5-gate thresholds
SHARPE_GATE = 0.5
PVALUE_GATE = 0.05
REGIME_GAP_GATE = 0.50
MDD_GATE = 0.50
MIN_TRADES = 30


def download_data() -> pd.DataFrame:
    """Download daily close data for all tickers."""
    tickers = SECTOR_ETFS + [BENCHMARK]
    print(f"Downloading {len(tickers)} tickers from {START_DATE} to {END_DATE}...")

    data = yf.download(tickers, start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)

    # Handle multi-level columns from yfinance
    if isinstance(data.columns, pd.MultiIndex):
        closes = data["Close"]
    else:
        closes = data

    # Drop any ticker with >20% missing
    thresh = len(closes) * 0.8
    closes = closes.dropna(axis=1, thresh=int(thresh))
    closes = closes.ffill().bfill()

    missing = [t for t in tickers if t not in closes.columns]
    if missing:
        print(f"  WARNING: Missing tickers (excluded): {missing}")

    print(f"  Got {len(closes)} trading days, {len(closes.columns)} tickers")
    return closes


def compute_relative_returns(closes: pd.DataFrame) -> pd.DataFrame:
    """Compute trailing 21-day return of each sector relative to SPY."""
    spy_ret = closes[BENCHMARK].pct_change(LOOKBACK_DAYS)

    rel_returns = pd.DataFrame(index=closes.index)
    for etf in SECTOR_ETFS:
        if etf in closes.columns:
            sector_ret = closes[etf].pct_change(LOOKBACK_DAYS)
            rel_returns[etf] = sector_ret - spy_ret

    return rel_returns


def get_rebalancing_dates(closes: pd.DataFrame) -> List[dict]:
    """Identify month-end rebalancing windows with entry/exit dates."""
    dates = closes.index
    # Keep only dates >= TRADE_START
    dates = dates[dates >= TRADE_START]

    windows = []
    # Group by year-month
    for ym in dates.to_period("M").unique():
        month_dates = dates[dates.to_period("M") == ym]
        if len(month_dates) < 10:
            continue

        month_end = month_dates[-1]
        is_quarter_end = ym.month in [3, 6, 9, 12]

        # Entry: T-5 from month-end
        entry_idx_pos = len(month_dates) + ENTRY_OFFSET  # -5 from end
        if entry_idx_pos < 0:
            entry_idx_pos = 0
        entry_date = month_dates[entry_idx_pos]

        # Exit: T+2 after month-end — find in next month
        me_loc = closes.index.get_loc(month_end)
        exit_loc = me_loc + EXIT_OFFSET
        if exit_loc >= len(closes.index):
            continue
        exit_date = closes.index[exit_loc]

        windows.append({
            "entry_date": entry_date,
            "month_end": month_end,
            "exit_date": exit_date,
            "is_quarter_end": is_quarter_end,
            "year_month": str(ym),
        })

    return windows


def compute_trade_return(closes: pd.DataFrame, etf: str, entry_date, exit_date, direction: str) -> float:
    """Compute return for a single trade. direction='long' or 'short'."""
    try:
        entry_price = closes.loc[entry_date, etf]
        exit_price = closes.loc[exit_date, etf]
        ret = (exit_price / entry_price) - 1.0
        if direction == "short":
            ret = -ret
        return ret
    except (KeyError, TypeError):
        return np.nan


def spy_regime(closes: pd.DataFrame, date) -> str:
    """Classify the day's SPY regime: green if SPY up over the trade window."""
    loc = closes.index.get_loc(date)
    if loc < 5:
        return "unknown"
    spy_5d = closes[BENCHMARK].iloc[loc] / closes[BENCHMARK].iloc[loc - 5] - 1
    return "green" if spy_5d > 0 else "red"


def run_variant_a(closes, rel_returns, windows):
    """Variant A: Simple worst-2 buy / best-2 sell at every month-end."""
    trades = []
    for w in windows:
        entry = w["entry_date"]
        exit_d = w["exit_date"]

        if entry not in rel_returns.index:
            continue

        rels = rel_returns.loc[entry].dropna().sort_values()
        if len(rels) < 4:
            continue

        # Worst 2 → buy
        for etf in rels.index[:2]:
            ret = compute_trade_return(closes, etf, entry, exit_d, "long")
            if not np.isnan(ret):
                trades.append({
                    "entry_date": entry, "exit_date": exit_d, "etf": etf,
                    "direction": "long", "return": ret,
                    "regime": spy_regime(closes, entry),
                    "rel_perf": rels[etf], "is_qe": w["is_quarter_end"],
                })

        # Best 2 → sell/put
        for etf in rels.index[-2:]:
            ret = compute_trade_return(closes, etf, entry, exit_d, "short")
            if not np.isnan(ret):
                trades.append({
                    "entry_date": entry, "exit_date": exit_d, "etf": etf,
                    "direction": "short", "return": ret,
                    "regime": spy_regime(closes, entry),
                    "rel_perf": rels[etf], "is_qe": w["is_quarter_end"],
                })

    return pd.DataFrame(trades)


def run_variant_b(closes, rel_returns, windows):
    """Variant B: Only trade when divergence > 1.5 stdev."""
    # Compute rolling stdev of relative returns
    rel_std = rel_returns.rolling(63, min_periods=21).std()

    trades = []
    for w in windows:
        entry = w["entry_date"]
        exit_d = w["exit_date"]

        if entry not in rel_returns.index or entry not in rel_std.index:
            continue

        rels = rel_returns.loc[entry].dropna()
        stds = rel_std.loc[entry].dropna()

        common = rels.index.intersection(stds.index)
        if len(common) < 4:
            continue

        rels = rels[common]
        stds = stds[common]

        z_scores = rels / stds.replace(0, np.nan)
        z_scores = z_scores.dropna().sort_values()

        # Only trade extreme divergences
        for etf in z_scores.index:
            if z_scores[etf] < -1.5:
                ret = compute_trade_return(closes, etf, entry, exit_d, "long")
                if not np.isnan(ret):
                    trades.append({
                        "entry_date": entry, "exit_date": exit_d, "etf": etf,
                        "direction": "long", "return": ret,
                        "regime": spy_regime(closes, entry),
                        "rel_perf": rels[etf], "z_score": z_scores[etf],
                        "is_qe": w["is_quarter_end"],
                    })
            elif z_scores[etf] > 1.5:
                ret = compute_trade_return(closes, etf, entry, exit_d, "short")
                if not np.isnan(ret):
                    trades.append({
                        "entry_date": entry, "exit_date": exit_d, "etf": etf,
                        "direction": "short", "return": ret,
                        "regime": spy_regime(closes, entry),
                        "rel_perf": rels[etf], "z_score": z_scores[etf],
                        "is_qe": w["is_quarter_end"],
                    })

    return pd.DataFrame(trades)


def run_variant_c(closes, rel_returns, windows):
    """Variant C: Quarter-end only."""
    qe_windows = [w for w in windows if w["is_quarter_end"]]

    trades = []
    for w in qe_windows:
        entry = w["entry_date"]
        exit_d = w["exit_date"]

        if entry not in rel_returns.index:
            continue

        rels = rel_returns.loc[entry].dropna().sort_values()
        if len(rels) < 4:
            continue

        # Top/bottom 3 at quarter-end (stronger signal, trade more)
        for etf in rels.index[:3]:
            ret = compute_trade_return(closes, etf, entry, exit_d, "long")
            if not np.isnan(ret):
                trades.append({
                    "entry_date": entry, "exit_date": exit_d, "etf": etf,
                    "direction": "long", "return": ret,
                    "regime": spy_regime(closes, entry),
                    "rel_perf": rels[etf], "is_qe": True,
                })

        for etf in rels.index[-3:]:
            ret = compute_trade_return(closes, etf, entry, exit_d, "short")
            if not np.isnan(ret):
                trades.append({
                    "entry_date": entry, "exit_date": exit_d, "etf": etf,
                    "direction": "short", "return": ret,
                    "regime": spy_regime(closes, entry),
                    "rel_perf": rels[etf], "is_qe": True,
                })

    return pd.DataFrame(trades)


def run_variant_d(closes, rel_returns, windows):
    """Variant D: Month-end base (2 sectors) + quarter-end amplified (3 sectors)."""
    trades = []
    for w in windows:
        entry = w["entry_date"]
        exit_d = w["exit_date"]

        if entry not in rel_returns.index:
            continue

        rels = rel_returns.loc[entry].dropna().sort_values()
        if len(rels) < 4:
            continue

        n = 3 if w["is_quarter_end"] else 2
        weight = 1.5 if w["is_quarter_end"] else 1.0

        for etf in rels.index[:n]:
            ret = compute_trade_return(closes, etf, entry, exit_d, "long")
            if not np.isnan(ret):
                trades.append({
                    "entry_date": entry, "exit_date": exit_d, "etf": etf,
                    "direction": "long", "return": ret * weight,
                    "regime": spy_regime(closes, entry),
                    "rel_perf": rels[etf], "is_qe": w["is_quarter_end"],
                    "weight": weight,
                })

        for etf in rels.index[-n:]:
            ret = compute_trade_return(closes, etf, entry, exit_d, "short")
            if not np.isnan(ret):
                trades.append({
                    "entry_date": entry, "exit_date": exit_d, "etf": etf,
                    "direction": "short", "return": ret * weight,
                    "regime": spy_regime(closes, entry),
                    "rel_perf": rels[etf], "is_qe": w["is_quarter_end"],
                    "weight": weight,
                })

    return pd.DataFrame(trades)


def run_variant_e(closes, rel_returns, windows):
    """Variant E: Divergence-weighted sizing (position size ~ magnitude of divergence)."""
    rel_std = rel_returns.rolling(63, min_periods=21).std()

    trades = []
    for w in windows:
        entry = w["entry_date"]
        exit_d = w["exit_date"]

        if entry not in rel_returns.index or entry not in rel_std.index:
            continue

        rels = rel_returns.loc[entry].dropna()
        stds = rel_std.loc[entry].dropna()
        common = rels.index.intersection(stds.index)
        if len(common) < 4:
            continue

        rels = rels[common]
        stds = stds[common]
        z_scores = (rels / stds.replace(0, np.nan)).dropna().sort_values()

        # Bottom 2: buy, weighted by |z|
        for etf in z_scores.index[:2]:
            weight = min(abs(z_scores[etf]), 3.0)  # cap at 3x
            ret = compute_trade_return(closes, etf, entry, exit_d, "long")
            if not np.isnan(ret):
                trades.append({
                    "entry_date": entry, "exit_date": exit_d, "etf": etf,
                    "direction": "long", "return": ret * weight,
                    "regime": spy_regime(closes, entry),
                    "rel_perf": rels[etf], "z_score": z_scores[etf],
                    "weight": weight, "is_qe": w["is_quarter_end"],
                })

        # Top 2: sell, weighted by |z|
        for etf in z_scores.index[-2:]:
            weight = min(abs(z_scores[etf]), 3.0)
            ret = compute_trade_return(closes, etf, entry, exit_d, "short")
            if not np.isnan(ret):
                trades.append({
                    "entry_date": entry, "exit_date": exit_d, "etf": etf,
                    "direction": "short", "return": ret * weight,
                    "regime": spy_regime(closes, entry),
                    "rel_perf": rels[etf], "z_score": z_scores[etf],
                    "weight": weight, "is_qe": w["is_quarter_end"],
                })

    return pd.DataFrame(trades)


def run_variant_f(closes, rel_returns, windows):
    """Variant F: Momentum confirmation — only buy underperformers showing 5-day turn."""
    mom_5d = closes.pct_change(5)

    trades = []
    for w in windows:
        entry = w["entry_date"]
        exit_d = w["exit_date"]

        if entry not in rel_returns.index or entry not in mom_5d.index:
            continue

        rels = rel_returns.loc[entry].dropna().sort_values()
        if len(rels) < 4:
            continue

        # Buy worst performers ONLY if 5-day momentum turned positive
        for etf in rels.index[:3]:
            if etf in mom_5d.columns and mom_5d.loc[entry, etf] > 0:
                ret = compute_trade_return(closes, etf, entry, exit_d, "long")
                if not np.isnan(ret):
                    trades.append({
                        "entry_date": entry, "exit_date": exit_d, "etf": etf,
                        "direction": "long", "return": ret,
                        "regime": spy_regime(closes, entry),
                        "rel_perf": rels[etf], "mom_5d": mom_5d.loc[entry, etf],
                        "is_qe": w["is_quarter_end"],
                    })

        # Sell best performers ONLY if 5-day momentum turned negative
        for etf in rels.index[-3:]:
            if etf in mom_5d.columns and mom_5d.loc[entry, etf] < 0:
                ret = compute_trade_return(closes, etf, entry, exit_d, "short")
                if not np.isnan(ret):
                    trades.append({
                        "entry_date": entry, "exit_date": exit_d, "etf": etf,
                        "direction": "short", "return": ret,
                        "regime": spy_regime(closes, entry),
                        "rel_perf": rels[etf], "mom_5d": mom_5d.loc[entry, etf],
                        "is_qe": w["is_quarter_end"],
                    })

    return pd.DataFrame(trades)


def compute_metrics(trades_df: pd.DataFrame) -> dict:
    """Compute strategy metrics from a trades DataFrame."""
    if trades_df.empty or len(trades_df) < 5:
        return {
            "n_trades": len(trades_df), "sharpe": 0, "win_rate": 0,
            "profit_factor": 0, "max_dd": 1.0, "mean_ret": 0,
            "total_ret": 0, "sharpe_green": 0, "sharpe_red": 0,
            "regime_gap": 1.0, "p_value": 1.0,
        }

    returns = trades_df["return"].values
    n = len(returns)

    # Basic stats
    mean_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1) if n > 1 else 1e-9
    total_ret = np.sum(returns)

    # Annualized Sharpe (assume ~12 trades/year for monthly, scale accordingly)
    trades_per_year = max(n / ((trades_df["entry_date"].max() - trades_df["entry_date"].min()).days / 365.25), 1)
    sharpe = (mean_ret / max(std_ret, 1e-9)) * np.sqrt(trades_per_year)

    # Win rate
    wins = np.sum(returns > 0)
    win_rate = wins / n

    # Profit factor
    gross_profit = np.sum(returns[returns > 0])
    gross_loss = abs(np.sum(returns[returns < 0]))
    profit_factor = gross_profit / max(gross_loss, 1e-9)

    # Max drawdown on cumulative equity
    cum = np.cumsum(returns)
    running_max = np.maximum.accumulate(cum)
    drawdowns = running_max - cum
    max_dd = np.max(drawdowns) if len(drawdowns) > 0 else 0
    # Normalize DD as fraction of peak equity
    peak = np.max(running_max) if np.max(running_max) > 0 else 1.0
    max_dd_pct = max_dd / peak if peak > 0 else 0

    # Regime stratification
    green_trades = trades_df[trades_df["regime"] == "green"]
    red_trades = trades_df[trades_df["regime"] == "red"]

    def regime_sharpe(df):
        if len(df) < 3:
            return 0
        r = df["return"].values
        m, s = np.mean(r), np.std(r, ddof=1)
        if s < 1e-9:
            return 0
        tpy = max(len(r) / max(((df["entry_date"].max() - df["entry_date"].min()).days / 365.25), 0.5), 1)
        return (m / s) * np.sqrt(tpy)

    sharpe_green = regime_sharpe(green_trades)
    sharpe_red = regime_sharpe(red_trades)

    max_abs = max(abs(sharpe_green), abs(sharpe_red), 1e-9)
    regime_gap = abs(sharpe_green - sharpe_red) / max_abs

    return {
        "n_trades": n,
        "sharpe": round(sharpe, 3),
        "win_rate": round(win_rate, 3),
        "profit_factor": round(profit_factor, 3),
        "max_dd_pct": round(max_dd_pct, 3),
        "mean_ret": round(mean_ret * 100, 4),  # in %
        "total_ret": round(total_ret * 100, 2),  # in %
        "sharpe_green": round(sharpe_green, 3),
        "sharpe_red": round(sharpe_red, 3),
        "regime_gap": round(regime_gap, 3),
        "n_green": len(green_trades),
        "n_red": len(red_trades),
        "trades_per_year": round(trades_per_year, 1),
    }


def permutation_test(returns: np.ndarray, n_perms: int = N_PERMUTATIONS) -> float:
    """Permutation test: shuffle returns, compute fraction with Sharpe >= observed."""
    if len(returns) < 5:
        return 1.0

    rng = np.random.RandomState(SEED)
    observed_mean = np.mean(returns)
    observed_std = np.std(returns, ddof=1)
    if observed_std < 1e-9:
        return 1.0
    observed_sharpe = observed_mean / observed_std

    count_ge = 0
    for _ in range(n_perms):
        shuffled = rng.permutation(returns)
        # Randomly flip signs (null: no directional edge)
        signs = rng.choice([-1, 1], size=len(returns))
        shuffled = returns * signs
        m = np.mean(shuffled)
        s = np.std(shuffled, ddof=1)
        if s > 1e-9:
            if m / s >= observed_sharpe:
                count_ge += 1

    return count_ge / n_perms


def apply_five_gates(metrics: dict) -> dict:
    """Apply the 5-gate filter system."""
    gates = {
        "sharpe_pass": metrics["sharpe"] > SHARPE_GATE,
        "pvalue_pass": metrics.get("p_value", 1.0) < PVALUE_GATE,
        "regime_pass": metrics["regime_gap"] < REGIME_GAP_GATE,
        "mdd_pass": metrics["max_dd_pct"] < MDD_GATE,
        "trades_pass": metrics["n_trades"] >= MIN_TRADES,
    }
    gates["all_pass"] = all(gates.values())
    return gates


def long_short_breakdown(trades_df: pd.DataFrame) -> dict:
    """Separate long/short performance."""
    result = {}
    for direction in ["long", "short"]:
        sub = trades_df[trades_df["direction"] == direction]
        if len(sub) > 0:
            rets = sub["return"].values
            result[direction] = {
                "n": len(sub),
                "mean_ret_pct": round(np.mean(rets) * 100, 4),
                "win_rate": round(np.sum(rets > 0) / len(rets), 3),
                "total_ret_pct": round(np.sum(rets) * 100, 2),
            }
        else:
            result[direction] = {"n": 0, "mean_ret_pct": 0, "win_rate": 0, "total_ret_pct": 0}
    return result


def print_results(name: str, trades_df: pd.DataFrame, metrics: dict, gates: dict, ls_breakdown: dict):
    """Pretty-print results for one variant."""
    status = "PASS" if gates["all_pass"] else "FAIL"
    print(f"\n{'='*70}")
    print(f"  Variant {name}  [{status}]")
    print(f"{'='*70}")
    print(f"  Trades: {metrics['n_trades']}  ({metrics.get('trades_per_year', 0)}/yr)")
    print(f"  Sharpe: {metrics['sharpe']:.3f}   Win Rate: {metrics['win_rate']:.1%}   PF: {metrics['profit_factor']:.2f}")
    print(f"  Mean Return: {metrics['mean_ret']:.4f}%   Total Return: {metrics['total_ret']:.2f}%")
    print(f"  Max Drawdown: {metrics['max_dd_pct']:.1%}")
    print(f"  Regime — Green Sharpe: {metrics['sharpe_green']:.3f}  Red Sharpe: {metrics['sharpe_red']:.3f}  Gap: {metrics['regime_gap']:.3f}")
    print(f"  P-value (permutation): {metrics.get('p_value', 'N/A')}")
    print(f"  --- Long/Short Breakdown ---")
    for d in ["long", "short"]:
        b = ls_breakdown[d]
        print(f"    {d.upper()}: n={b['n']}  mean={b['mean_ret_pct']:.4f}%  WR={b['win_rate']:.1%}  total={b['total_ret_pct']:.2f}%")
    print(f"  --- 5-Gate Results ---")
    for g, v in gates.items():
        if g != "all_pass":
            sym = "OK" if v else "XX"
            print(f"    [{sym}] {g}")
    print(f"  OVERALL: {'*** PASS ***' if gates['all_pass'] else 'FAIL'}")


def main():
    print("=" * 70)
    print("  INSTITUTIONAL REBALANCING FLOW — SECTOR ETF BACKTEST")
    print("  Period: 2018-01 to 2026-08 | Sectors: 11 ETFs vs SPY")
    print("=" * 70)

    # 1. Download data
    closes = download_data()

    # 2. Compute relative returns
    rel_returns = compute_relative_returns(closes)

    # 3. Get rebalancing windows
    windows = get_rebalancing_dates(closes)
    print(f"\nIdentified {len(windows)} month-end windows ({sum(1 for w in windows if w['is_quarter_end'])} quarter-ends)")

    # 4. Run all variants
    variants = {
        "A — Simple worst-2/best-2 monthly": run_variant_a,
        "B — Divergence > 1.5 stdev": run_variant_b,
        "C — Quarter-end only": run_variant_c,
        "D — Monthly + QE amplified": run_variant_d,
        "E — Divergence-weighted": run_variant_e,
        "F — Momentum confirmation": run_variant_f,
    }

    all_results = {}

    for name, func in variants.items():
        print(f"\n>>> Running {name}...")
        trades_df = func(closes, rel_returns, windows)

        if trades_df.empty:
            print(f"  No trades generated. SKIP.")
            continue

        # Compute metrics
        metrics = compute_metrics(trades_df)

        # Permutation test
        print(f"  Running {N_PERMUTATIONS} permutations...")
        metrics["p_value"] = round(permutation_test(trades_df["return"].values), 4)

        # 5-gate
        gates = apply_five_gates(metrics)

        # Long/short breakdown
        ls_breakdown = long_short_breakdown(trades_df)

        # Print
        print_results(name, trades_df, metrics, gates, ls_breakdown)

        # Store
        all_results[name] = {
            "metrics": metrics,
            "gates": gates,
            "ls_breakdown": ls_breakdown,
            "n_trades": len(trades_df),
        }

    # ─── Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SUMMARY — ALL VARIANTS")
    print("=" * 70)
    print(f"  {'Variant':<40} {'Sharpe':>7} {'WR':>7} {'PF':>7} {'p-val':>7} {'Trades':>7} {'Pass':>6}")
    print(f"  {'-'*40} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*6}")

    any_pass = False
    for name, res in all_results.items():
        m = res["metrics"]
        status = "PASS" if res["gates"]["all_pass"] else "FAIL"
        if res["gates"]["all_pass"]:
            any_pass = True
        print(f"  {name:<40} {m['sharpe']:>7.3f} {m['win_rate']:>6.1%} {m['profit_factor']:>7.2f} {m.get('p_value', 1.0):>7.4f} {m['n_trades']:>7} {status:>6}")

    if not any_pass:
        print("\n  ** NO VARIANT PASSED ALL 5 GATES **")
        print("  Institutional rebalancing flow does NOT appear to be a tradeable edge")
        print("  in sector ETFs over 2018-2026.")
    else:
        print("\n  ** PASSING VARIANTS FOUND — investigate further **")

    # ─── Year-by-year for best variant ───────────────────────────
    if all_results:
        best_name = max(all_results, key=lambda k: all_results[k]["metrics"]["sharpe"])
        best_trades_func = variants[best_name]
        best_trades = best_trades_func(closes, rel_returns, windows)

        if not best_trades.empty:
            best_trades["year"] = pd.to_datetime(best_trades["entry_date"]).dt.year
            print(f"\n  Year-by-year for best variant: {best_name}")
            print(f"  {'Year':>6} {'Trades':>7} {'Mean%':>8} {'WR':>7} {'Total%':>8}")
            for yr, grp in best_trades.groupby("year"):
                rets = grp["return"].values
                print(f"  {yr:>6} {len(grp):>7} {np.mean(rets)*100:>8.3f} {np.sum(rets>0)/len(rets):>6.1%} {np.sum(rets)*100:>8.2f}")

    # ─── Sector-level analysis for best variant ──────────────────
    if all_results and not best_trades.empty:
        print(f"\n  Sector breakdown for best variant: {best_name}")
        print(f"  {'ETF':>6} {'Trades':>7} {'Mean%':>8} {'WR':>7}")
        for etf, grp in best_trades.groupby("etf"):
            rets = grp["return"].values
            print(f"  {etf:>6} {len(grp):>7} {np.mean(rets)*100:>8.3f} {np.sum(rets>0)/len(rets):>6.1%}")

    print("\n" + "=" * 70)
    print("  DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
