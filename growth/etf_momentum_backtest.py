#!/usr/bin/env python3
"""
ETF Cross-Sectional Momentum Backtest
=======================================
ETFs don't suffer survivorship bias — the universe is stable and liquid.
Tests whether buying strongest-momentum ETFs adds value over equal-weight or trend-only.

Strategy variants:
  A) Pure momentum: buy top-K by 12-1 month momentum
  B) Momentum + 200MA filter: only buy if above 200MA AND has momentum
  C) Momentum + dynamic exits: exit to cash if 10-day momentum flips negative
  D) Momentum + 200MA + dynamic exits

Walk-forward: sliding 252-day lookback, 21-day hold, 2009-01-01 to present.
Benchmarks: SPY B&H, equal-weight all ETFs, trend-only (200MA).
Permutation test: 100 shuffles to verify momentum selection beats random.

Usage:
  python3 growth/etf_momentum_backtest.py
"""

import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Constants ────────────────────────────────────────────────────────────────

ETF_UNIVERSE = [
    # US Sectors (11 SPDRs)
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC",
    # Size
    "SPY", "QQQ", "IWM",
    # International
    "EFA", "EEM",
    # Bonds
    "TLT", "IEF", "AGG",
    # Commodities
    "GLD", "DBA",
    # Real Estate
    "VNQ",
]

LOOKBACK_DAYS = 252       # 12 months
SKIP_RECENT = 21          # skip most recent month (12-1 month momentum)
HOLD_PERIOD = 21          # monthly rebalance
MOM_EXIT_DAYS = 10        # dynamic exit: 10-day momentum flip
MA_PERIOD = 200           # 200-day MA for regime filter
COST_PER_TRADE = 0.0005   # 0.05% per trade (ETFs very liquid)
PORTFOLIO_SIZES = [3, 5, 7]

DATA_START = "2007-06-01"  # enough history for 252-day lookback + 200MA before 2009
BACKTEST_START = "2009-01-01"

N_PERMUTATIONS = 100


# ── Data Download ────────────────────────────────────────────────────────────

def download_data(tickers: list, start: str, end: str = None) -> pd.DataFrame:
    """Download adjusted close prices for all ETFs."""
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")

    print(f"[INFO] Downloading data for {len(tickers)} ETFs from {start} to {end}...")

    batch_str = " ".join(tickers)
    data = yf.download(batch_str, start=start, end=end, progress=False, threads=True, group_by="ticker")

    all_close = {}
    if isinstance(data.columns, pd.MultiIndex):
        for ticker in tickers:
            try:
                if ticker in data.columns.get_level_values(0):
                    td = data[ticker]
                    close_col = "Adj Close" if "Adj Close" in td.columns else "Close"
                    c = td[close_col].dropna()
                    if len(c) > 200:
                        all_close[ticker] = c
            except Exception:
                pass
    else:
        close_col = "Adj Close" if "Adj Close" in data.columns else "Close"
        all_close[tickers[0]] = data[close_col].dropna()

    close_df = pd.DataFrame(all_close)
    print(f"[INFO] Got {len(close_df.columns)} ETFs, {len(close_df)} days "
          f"({close_df.index[0].date()} to {close_df.index[-1].date()})")

    missing = set(tickers) - set(close_df.columns)
    if missing:
        print(f"[WARN] Missing ETFs (not enough history): {sorted(missing)}")

    return close_df


# ── Precompute Signals (vectorized) ─────────────────────────────────────────

def precompute_signals(prices: pd.DataFrame) -> dict:
    """Precompute all signals to avoid per-day recalculation."""
    print("[INFO] Precomputing signals...")

    # Daily returns
    daily_rets = prices.pct_change().fillna(0.0)

    # 12-1 month momentum: return from t-252 to t-21
    mom_12_1 = prices.shift(SKIP_RECENT) / prices.shift(LOOKBACK_DAYS) - 1.0

    # 200-day moving average
    ma200 = prices.rolling(MA_PERIOD).mean()

    # Above 200MA flag
    above_ma = prices > ma200

    # 10-day momentum for dynamic exits
    mom_10d = prices / prices.shift(MOM_EXIT_DAYS) - 1.0

    return {
        "daily_rets": daily_rets,
        "mom_12_1": mom_12_1,
        "ma200": ma200,
        "above_ma": above_ma,
        "mom_10d": mom_10d,
    }


# ── Backtest Engine (vectorized-signal version) ─────────────────────────────

def run_backtest(prices: pd.DataFrame, signals: dict, top_k: int,
                 use_ma_filter: bool = False, use_dynamic_exit: bool = False,
                 random_select: bool = False, rng: np.random.Generator = None) -> dict:
    """Run walk-forward momentum backtest using precomputed signals."""

    daily_rets = signals["daily_rets"]
    mom_12_1 = signals["mom_12_1"]
    above_ma = signals["above_ma"]
    mom_10d = signals["mom_10d"]

    # Find backtest start index
    backtest_start_idx = None
    for i in range(len(prices)):
        if prices.index[i] >= pd.Timestamp(BACKTEST_START):
            backtest_start_idx = i
            break

    if backtest_start_idx is None or backtest_start_idx < LOOKBACK_DAYS:
        raise ValueError("Not enough data before backtest start")

    n_days = len(prices)
    tickers = list(prices.columns)
    n_tickers = len(tickers)

    # Convert to numpy for speed
    rets_np = daily_rets.values  # (n_days, n_tickers)
    mom_np = mom_12_1.values
    above_ma_np = above_ma.values
    mom10_np = mom_10d.values

    port_returns = np.zeros(n_days - backtest_start_idx)
    holdings = set()
    last_rebalance_idx = backtest_start_idx - HOLD_PERIOD
    n_trades = 0

    for day_offset, idx in enumerate(range(backtest_start_idx, n_days)):

        # ── Dynamic exits ──
        if use_dynamic_exit and holdings:
            exits = set()
            for ti in holdings:
                if not np.isnan(mom10_np[idx, ti]) and mom10_np[idx, ti] < 0:
                    exits.add(ti)
            if exits:
                n_trades += len(exits)
                holdings -= exits

        # ── Monthly rebalance ──
        if idx - last_rebalance_idx >= HOLD_PERIOD:
            last_rebalance_idx = idx

            # Get valid momentum scores
            mom_row = mom_np[idx]
            valid_mask = ~np.isnan(mom_row)

            if use_ma_filter:
                ama_row = above_ma_np[idx]
                valid_mask &= ama_row  # only above-200MA ETFs

            valid_indices = np.where(valid_mask)[0]

            if random_select and rng is not None:
                if len(valid_indices) >= top_k:
                    selected = set(rng.choice(valid_indices, size=top_k, replace=False))
                else:
                    selected = set(valid_indices)
            else:
                if len(valid_indices) >= top_k:
                    mom_valid = mom_row[valid_indices]
                    top_idx = np.argsort(mom_valid)[-top_k:]
                    selected = set(valid_indices[top_idx])
                else:
                    selected = set(valid_indices)

            # Count turnover
            n_trades += len(holdings - selected) + len(selected - holdings)
            holdings = selected

        # ── Daily portfolio return ──
        if holdings:
            w = 1.0 / len(holdings)
            day_ret = 0.0
            for ti in holdings:
                r = rets_np[idx, ti]
                if not np.isnan(r):
                    day_ret += w * r
            port_returns[day_offset] = day_ret

    # Deduct transaction costs
    total_cost = n_trades * COST_PER_TRADE
    if len(port_returns) > 0 and n_trades > 0:
        port_returns -= total_cost / len(port_returns)

    dates = list(prices.index[backtest_start_idx:])

    return {
        "daily_returns": port_returns,
        "dates": dates,
        "n_trades": n_trades,
        "total_cost_pct": round(total_cost * 100, 3),
    }


# ── Benchmarks ───────────────────────────────────────────────────────────────

def run_benchmarks(prices: pd.DataFrame, signals: dict, dates: list) -> dict:
    """Run all benchmarks aligned to strategy dates."""
    daily_rets = signals["daily_rets"]
    above_ma = signals["above_ma"]

    benchmarks = {}

    # SPY buy-and-hold
    if "SPY" in prices.columns:
        spy_col = list(prices.columns).index("SPY")
        spy_rets = np.array([daily_rets.values[prices.index.get_loc(d), spy_col]
                             if d in prices.index else 0.0 for d in dates])
        benchmarks["SPY_buy_and_hold"] = spy_rets

    # Equal-weight all ETFs
    ew_rets = np.array([daily_rets.loc[d].dropna().mean() if d in daily_rets.index else 0.0
                        for d in dates])
    benchmarks["equal_weight_all"] = ew_rets

    # Trend-only (200MA): hold each ETF if above 200MA, equal weight
    trend_rets = []
    for d in dates:
        if d in daily_rets.index and d in above_ma.index:
            eligible = above_ma.loc[d]
            eligible_tickers = eligible[eligible].index
            day_r = daily_rets.loc[d][eligible_tickers].dropna()
            trend_rets.append(day_r.mean() if len(day_r) > 0 else 0.0)
        else:
            trend_rets.append(0.0)
    benchmarks["trend_200ma_only"] = np.array(trend_rets)

    return benchmarks


# ── Metrics ──────────────────────────────────────────────────────────────────

def calc_metrics(daily_returns: np.ndarray, label: str = "") -> dict:
    """Calculate risk-adjusted performance metrics."""
    if len(daily_returns) == 0:
        return {}

    total_days = len(daily_returns)
    years = total_days / 252

    cum = np.cumprod(1 + daily_returns)
    total_return = cum[-1] - 1.0
    cagr = (cum[-1]) ** (1 / years) - 1.0 if years > 0 else 0.0

    mean_daily = np.mean(daily_returns)
    std_daily = np.std(daily_returns, ddof=1) if len(daily_returns) > 1 else 1e-9
    sharpe = (mean_daily / std_daily) * np.sqrt(252) if std_daily > 1e-9 else 0.0

    downside = daily_returns[daily_returns < 0]
    downside_std = np.std(downside, ddof=1) if len(downside) > 1 else 1e-9
    sortino = (mean_daily / downside_std) * np.sqrt(252) if downside_std > 1e-9 else 0.0

    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    max_dd = np.min(dd)

    calmar = cagr / abs(max_dd) if abs(max_dd) > 1e-9 else 0.0

    gains = daily_returns[daily_returns > 0].sum()
    losses = abs(daily_returns[daily_returns < 0].sum())
    pf = gains / losses if losses > 1e-9 else float("inf")

    monthly_chunks = [daily_returns[i:i+21] for i in range(0, len(daily_returns), 21)]
    monthly_rets = [np.prod(1 + chunk) - 1 for chunk in monthly_chunks if len(chunk) >= 15]
    win_months = sum(1 for r in monthly_rets if r > 0)
    wr = win_months / len(monthly_rets) if monthly_rets else 0.0

    return {
        "label": label,
        "total_return_pct": round(total_return * 100, 2),
        "cagr_pct": round(cagr * 100, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_dd_pct": round(max_dd * 100, 2),
        "calmar": round(calmar, 3),
        "profit_factor": round(pf, 3),
        "win_rate_monthly": round(wr * 100, 1),
        "n_years": round(years, 1),
    }


def regime_analysis(daily_returns: np.ndarray, dates: list, spy_prices: pd.Series) -> dict:
    """Analyze performance in green vs red months (by SPY)."""
    spy_monthly = spy_prices.resample("ME").last().pct_change().dropna()

    ret_series = pd.Series(daily_returns, index=dates)
    monthly_bt = ret_series.resample("ME").apply(lambda x: np.prod(1 + x) - 1)

    green_rets = []
    red_rets = []

    for month in monthly_bt.index:
        if month in spy_monthly.index:
            if spy_monthly[month] >= 0:
                green_rets.append(monthly_bt[month])
            else:
                red_rets.append(monthly_bt[month])

    def monthly_sharpe(rets):
        if len(rets) < 3:
            return 0.0
        arr = np.array(rets)
        s = np.std(arr, ddof=1)
        return (np.mean(arr) / s) * np.sqrt(12) if s > 1e-9 else 0.0

    return {
        "green_months": len(green_rets),
        "red_months": len(red_rets),
        "green_avg_ret_pct": round(np.mean(green_rets) * 100, 2) if green_rets else 0.0,
        "red_avg_ret_pct": round(np.mean(red_rets) * 100, 2) if red_rets else 0.0,
        "green_sharpe": round(monthly_sharpe(green_rets), 3),
        "red_sharpe": round(monthly_sharpe(red_rets), 3),
        "green_win_rate": round(sum(1 for r in green_rets if r > 0) / len(green_rets) * 100, 1) if green_rets else 0.0,
        "red_win_rate": round(sum(1 for r in red_rets if r > 0) / len(red_rets) * 100, 1) if red_rets else 0.0,
    }


# ── Permutation Test ─────────────────────────────────────────────────────────

def permutation_test(prices: pd.DataFrame, signals: dict, top_k: int,
                     actual_sharpe: float, use_ma_filter: bool,
                     use_dynamic_exit: bool, n_perms: int = N_PERMUTATIONS) -> dict:
    """Test if momentum selection beats random ETF selection."""
    print(f"  [PERM] Running {n_perms} permutations for K={top_k}...")
    perm_sharpes = []
    rng = np.random.default_rng(42)

    for i in range(n_perms):
        result = run_backtest(prices, signals, top_k, use_ma_filter=use_ma_filter,
                              use_dynamic_exit=use_dynamic_exit,
                              random_select=True, rng=rng)
        m = calc_metrics(result["daily_returns"])
        perm_sharpes.append(m["sharpe"])
        if (i + 1) % 25 == 0:
            print(f"    {i+1}/{n_perms} done...")

    perm_sharpes = np.array(perm_sharpes)
    p_value = np.mean(perm_sharpes >= actual_sharpe)

    return {
        "actual_sharpe": round(actual_sharpe, 3),
        "perm_mean_sharpe": round(float(np.mean(perm_sharpes)), 3),
        "perm_std_sharpe": round(float(np.std(perm_sharpes)), 3),
        "perm_p5_sharpe": round(float(np.percentile(perm_sharpes, 5)), 3),
        "perm_p95_sharpe": round(float(np.percentile(perm_sharpes, 95)), 3),
        "p_value": round(float(p_value), 3),
        "n_permutations": n_perms,
        "significant_at_05": bool(p_value < 0.05),
        "significant_at_10": bool(p_value < 0.10),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("ETF CROSS-SECTIONAL MOMENTUM BACKTEST")
    print("=" * 70)
    print(f"Universe: {len(ETF_UNIVERSE)} ETFs")
    print(f"Period: {BACKTEST_START} to present")
    print(f"Walk-forward: {LOOKBACK_DAYS}d lookback, {HOLD_PERIOD}d hold, sliding")
    print(f"Transaction cost: {COST_PER_TRADE*100:.2f}% per trade")
    print()

    # Download data
    prices = download_data(ETF_UNIVERSE, DATA_START)

    # Flatten any MultiIndex columns
    if isinstance(prices.columns, pd.MultiIndex):
        prices.columns = prices.columns.get_level_values(0)

    spy_prices = prices["SPY"].copy()

    # Precompute all signals
    signals = precompute_signals(prices)

    # Track all results
    all_results = {}

    # ── Strategy variants ──
    variants = [
        ("pure_momentum", False, False),
        ("momentum_200ma", True, False),
        ("momentum_dynamic_exit", False, True),
        ("momentum_200ma_dynamic_exit", True, True),
    ]

    for k in PORTFOLIO_SIZES:
        print(f"\n{'─' * 50}")
        print(f"PORTFOLIO SIZE: Top {k} ETFs")
        print(f"{'─' * 50}")

        for variant_name, use_ma, use_exit in variants:
            label = f"{variant_name}_K{k}"
            print(f"\n[RUN] {label}...")

            result = run_backtest(prices, signals, k, use_ma_filter=use_ma, use_dynamic_exit=use_exit)
            metrics = calc_metrics(result["daily_returns"], label)
            regime = regime_analysis(result["daily_returns"], result["dates"], spy_prices)

            metrics["n_trades"] = result["n_trades"]
            metrics["total_cost_pct"] = result["total_cost_pct"]
            metrics["regime"] = regime

            all_results[label] = metrics

            print(f"  Sharpe={metrics['sharpe']:.3f}  Sortino={metrics['sortino']:.3f}  "
                  f"CAGR={metrics['cagr_pct']:.1f}%  MaxDD={metrics['max_dd_pct']:.1f}%  "
                  f"PF={metrics['profit_factor']:.2f}  WR={metrics['win_rate_monthly']:.1f}%  "
                  f"Trades={result['n_trades']}")

    # ── Benchmarks ──
    print(f"\n{'─' * 50}")
    print("BENCHMARKS")
    print(f"{'─' * 50}")

    # Use dates from first strategy run
    ref_key = list(all_results.keys())[0]
    ref_result = run_backtest(prices, signals, PORTFOLIO_SIZES[0])
    dates = ref_result["dates"]

    benchmarks = run_benchmarks(prices, signals, dates)
    for bench_name, bench_rets in benchmarks.items():
        bench_metrics = calc_metrics(bench_rets, bench_name)
        bench_regime = regime_analysis(bench_rets, dates, spy_prices)
        bench_metrics["regime"] = bench_regime
        all_results[bench_name] = bench_metrics
        print(f"\n[BENCH] {bench_name}:")
        print(f"  Sharpe={bench_metrics['sharpe']:.3f}  CAGR={bench_metrics['cagr_pct']:.1f}%  "
              f"MaxDD={bench_metrics['max_dd_pct']:.1f}%")

    # ── Permutation Tests ──
    print(f"\n{'─' * 50}")
    print("PERMUTATION TESTS (momentum vs random ETF selection)")
    print(f"{'─' * 50}")

    perm_results = {}
    for k in PORTFOLIO_SIZES:
        for variant_name, use_ma, use_exit in [("pure_momentum", False, False),
                                                 ("momentum_200ma", True, False)]:
            label = f"{variant_name}_K{k}"
            actual_sharpe = all_results[label]["sharpe"]
            perm = permutation_test(prices, signals, k, actual_sharpe, use_ma, use_exit)
            perm_results[label] = perm
            sig = "YES" if perm["significant_at_05"] else ("MARGINAL" if perm["significant_at_10"] else "NO")
            print(f"\n  {label}: actual Sharpe={actual_sharpe:.3f} vs "
                  f"random mean={perm['perm_mean_sharpe']:.3f} (p={perm['p_value']:.3f}) → {sig}")

    # ── Summary ──
    print(f"\n\n{'=' * 70}")
    print("SUMMARY: ETF MOMENTUM BACKTEST RESULTS")
    print(f"{'=' * 70}")

    bench_names = {"SPY_buy_and_hold", "equal_weight_all", "trend_200ma_only"}
    strat_keys = [k for k in all_results.keys() if k not in bench_names]
    strat_keys.sort(key=lambda k: all_results[k].get("sharpe", 0), reverse=True)

    print(f"\n{'Strategy':<40} {'Sharpe':>7} {'Sortino':>8} {'CAGR%':>7} {'MaxDD%':>7} {'PF':>6} {'WR%':>6}")
    print("-" * 82)

    for k in strat_keys:
        m = all_results[k]
        print(f"{k:<40} {m['sharpe']:>7.3f} {m['sortino']:>8.3f} {m['cagr_pct']:>7.1f} "
              f"{m['max_dd_pct']:>7.1f} {m['profit_factor']:>6.2f} {m['win_rate_monthly']:>6.1f}")

    print("-" * 82)
    for bench in sorted(bench_names):
        if bench in all_results:
            m = all_results[bench]
            print(f"{bench:<40} {m['sharpe']:>7.3f} {m['sortino']:>8.3f} {m['cagr_pct']:>7.1f} "
                  f"{m['max_dd_pct']:>7.1f} {m['profit_factor']:>6.2f} {m['win_rate_monthly']:>6.1f}")

    # Regime analysis for best strategy
    best_key = strat_keys[0]
    best_regime = all_results[best_key]["regime"]
    print(f"\nBest strategy: {best_key}")
    print(f"  Green months: avg {best_regime['green_avg_ret_pct']:.2f}% (WR {best_regime['green_win_rate']:.0f}%, "
          f"Sharpe {best_regime['green_sharpe']:.2f})")
    print(f"  Red months:   avg {best_regime['red_avg_ret_pct']:.2f}% (WR {best_regime['red_win_rate']:.0f}%, "
          f"Sharpe {best_regime['red_sharpe']:.2f})")

    # Permutation summary
    print(f"\nPermutation test summary:")
    for label, perm in perm_results.items():
        verdict = "SIGNIFICANT" if perm["significant_at_05"] else ("MARGINAL" if perm["significant_at_10"] else "NOT SIGNIFICANT")
        print(f"  {label}: p={perm['p_value']:.3f} → {verdict}")

    # Key finding
    print(f"\nKEY FINDING:")
    best_sharpe = all_results[best_key]["sharpe"]
    spy_sharpe = all_results.get("SPY_buy_and_hold", {}).get("sharpe", 0)
    trend_sharpe = all_results.get("trend_200ma_only", {}).get("sharpe", 0)
    ew_sharpe = all_results.get("equal_weight_all", {}).get("sharpe", 0)

    if best_sharpe > spy_sharpe and best_sharpe > ew_sharpe:
        any_sig = any(p["significant_at_05"] for p in perm_results.values())
        if any_sig:
            print(f"  ETF momentum ADDS VALUE. Best={best_key} Sharpe {best_sharpe:.3f} vs SPY {spy_sharpe:.3f}")
            print(f"  Permutation test confirms selection skill (not just diversification).")
        else:
            print(f"  ETF momentum has higher Sharpe ({best_sharpe:.3f} vs SPY {spy_sharpe:.3f})")
            print(f"  BUT permutation test shows NO significant selection skill — could be diversification benefit.")
    else:
        print(f"  ETF momentum does NOT add meaningful value over benchmarks.")
        print(f"  Best strat Sharpe={best_sharpe:.3f} vs SPY={spy_sharpe:.3f}, EW={ew_sharpe:.3f}, Trend={trend_sharpe:.3f}")

    # ── Save results ──
    output = {
        "metadata": {
            "run_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "universe": sorted(list(prices.columns)),
            "n_etfs": len(prices.columns),
            "backtest_start": BACKTEST_START,
            "backtest_end": prices.index[-1].strftime("%Y-%m-%d"),
            "lookback_days": LOOKBACK_DAYS,
            "hold_period": HOLD_PERIOD,
            "cost_per_trade_pct": COST_PER_TRADE * 100,
        },
        "strategy_results": {k: v for k, v in all_results.items()},
        "permutation_tests": perm_results,
        "key_finding": f"Best: {best_key} Sharpe={best_sharpe:.3f}",
    }

    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    outfile = output_dir / f"etf_momentum_backtest_{datetime.now().strftime('%Y%m%d')}.json"
    with open(outfile, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n[SAVED] {outfile}")


if __name__ == "__main__":
    main()
