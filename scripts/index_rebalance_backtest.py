#!/usr/bin/env python3
"""
INDEX REBALANCE / RECONSTITUTION TRADING — Structural Edge Backtest
===================================================================
Exploits predictable demand from index fund rebalancing:
  A) Quarter-end momentum (top-5 performers → buy before QE)
  B) Month-end reversion (worst-5 in 20d → buy before ME)
  C) Quarter-end reversion (worst-5 in 60d → buy before QE)
  D) Window dressing (top-5 → buy 3d before QE, sell on QE)
  E) January effect (small-cap underperformers in Dec → hold Jan)
  F) Turn-of-month (buy SPY 1d before ME, sell 3d into new month)

OOT: Jan 2022 – Jul 2026
Account: $645, slippage: 0.02%
5-gate validation framework.
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

# ── Configuration ──────────────────────────────────────────────────────────
INITIAL_CAPITAL = 645.0
SLIPPAGE_PCT = 0.0002  # 0.02%
OOT_START = "2022-01-01"
OOT_END = "2026-07-29"
PERM_ITERATIONS = 1000

STOCKS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
    "JPM", "BAC", "WFC", "JNJ", "PFE", "UNH", "ABBV", "MRK",
    "HD", "MCD", "DIS", "NFLX", "CRM", "ADBE", "AMD", "INTC",
    "PYPL", "UBER", "ABNB", "PLTR", "COIN", "RIVN", "SOFI",
]
REGIME_TICKERS = ["SPY", "IWM"]
ALL_TICKERS = list(set(STOCKS + REGIME_TICKERS))

OUTPUT_PATH = Path("/home/jupiter/Lvl3Quant/data/index_rebalance_results.json")


# ── Data Download ──────────────────────────────────────────────────────────
def download_data():
    """Download all price data from yfinance."""
    print("Downloading price data...")
    # Need lookback before OOT start for momentum/regime calculations
    start = "2021-06-01"
    end = OOT_END

    data = yf.download(ALL_TICKERS, start=start, end=end, auto_adjust=True, progress=False)
    if data.empty:
        print("ERROR: No data downloaded")
        sys.exit(1)

    # Handle multi-level columns from yfinance
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"]
    else:
        close = data

    close = close.ffill()
    print(f"  Data shape: {close.shape}, range: {close.index[0].date()} to {close.index[-1].date()}")
    return close


# ── Calendar Helpers ───────────────────────────────────────────────────────
def get_quarter_end_dates(dates):
    """Get trading dates closest to quarter-end (Mar, Jun, Sep, Dec)."""
    qe_months = {3, 6, 9, 12}
    result = []
    for year in range(dates[0].year, dates[-1].year + 1):
        for month in qe_months:
            # Find the 3rd Friday (options expiry / rebalance date)
            # Approximate: last business day of quarter
            if month == 12:
                qe = pd.Timestamp(year, 12, 31)
            else:
                qe = pd.Timestamp(year, month + 1, 1) - pd.Timedelta(days=1)
            # Find closest trading day at or before
            mask = dates <= qe
            if mask.any():
                result.append(dates[mask][-1])
    return sorted(set(result))


def get_month_end_dates(dates):
    """Get last trading day of each month."""
    df = pd.Series(dates, index=dates)
    return list(df.groupby([df.index.year, df.index.month]).last().values)


def offset_trading_day(dates, ref_date, offset):
    """Get trading day that is `offset` days before/after ref_date.
    Negative offset = before, positive = after."""
    idx = dates.get_loc(ref_date, method="ffill") if ref_date not in dates else dates.get_loc(ref_date)
    target = idx + offset
    if 0 <= target < len(dates):
        return dates[target]
    return None


# ── Regime Classification ─────────────────────────────────────────────────
def classify_regime(close_df):
    """Bull = SPY > 200-SMA, Bear = SPY < 200-SMA."""
    spy = close_df["SPY"].dropna()
    sma200 = spy.rolling(200).mean()
    regime = pd.Series("bull", index=spy.index)
    regime[spy < sma200] = "bear"
    return regime


# ── Trade Simulator ────────────────────────────────────────────────────────
def simulate_trades(trades, close_df):
    """Simulate trades with equal-weight allocation and slippage.
    Each trade: dict with keys: ticker, entry_date, exit_date.
    Returns list of completed trade dicts with pnl info."""
    results = []
    for t in trades:
        ticker = t["ticker"]
        entry_date = t["entry_date"]
        exit_date = t["exit_date"]

        if ticker not in close_df.columns:
            continue
        if entry_date not in close_df.index or exit_date not in close_df.index:
            continue

        entry_price = close_df.loc[entry_date, ticker]
        exit_price = close_df.loc[exit_date, ticker]

        if pd.isna(entry_price) or pd.isna(exit_price) or entry_price <= 0:
            continue

        # Apply slippage both ways
        entry_price_adj = entry_price * (1 + SLIPPAGE_PCT)
        exit_price_adj = exit_price * (1 - SLIPPAGE_PCT)

        ret = (exit_price_adj / entry_price_adj) - 1.0
        results.append({
            "ticker": ticker,
            "entry_date": str(entry_date.date()),
            "exit_date": str(exit_date.date()),
            "entry_price": round(float(entry_price), 2),
            "exit_price": round(float(exit_price), 2),
            "return": round(float(ret), 6),
            "hold_days": (exit_date - entry_date).days,
        })
    return results


# ── Strategy Variants ──────────────────────────────────────────────────────
def variant_a_quarter_momentum(close_df, dates):
    """A) Top-5 60d performers → buy 5 days before quarter end, hold 10 days."""
    trades = []
    qe_dates = get_quarter_end_dates(dates)
    stock_cols = [s for s in STOCKS if s in close_df.columns]

    for qe in qe_dates:
        if qe < pd.Timestamp(OOT_START) or qe > pd.Timestamp(OOT_END):
            continue

        entry = offset_trading_day(dates, qe, -5)
        exit_d = offset_trading_day(dates, qe, 5)  # 10 trading days total
        lookback_start = offset_trading_day(dates, qe, -65)

        if entry is None or exit_d is None or lookback_start is None:
            continue

        # Compute 60-day returns
        rets = {}
        for s in stock_cols:
            p_start = close_df.loc[lookback_start, s] if lookback_start in close_df.index else np.nan
            p_end = close_df.loc[entry, s] if entry in close_df.index else np.nan
            if pd.notna(p_start) and pd.notna(p_end) and p_start > 0:
                rets[s] = (p_end / p_start) - 1
        if len(rets) < 5:
            continue

        # Top 5 performers
        sorted_stocks = sorted(rets.items(), key=lambda x: x[1], reverse=True)[:5]
        for ticker, _ in sorted_stocks:
            trades.append({"ticker": ticker, "entry_date": entry, "exit_date": exit_d})

    return trades


def variant_b_monthend_reversion(close_df, dates):
    """B) Worst-5 in prior 20d → buy 3 days before month end, hold 5 days."""
    trades = []
    me_dates = get_month_end_dates(dates)
    stock_cols = [s for s in STOCKS if s in close_df.columns]

    for me in me_dates:
        me = pd.Timestamp(me)
        if me < pd.Timestamp(OOT_START) or me > pd.Timestamp(OOT_END):
            continue

        entry = offset_trading_day(dates, me, -3)
        exit_d = offset_trading_day(dates, me, 2)  # hold ~5 days
        lookback_start = offset_trading_day(dates, me, -23)

        if entry is None or exit_d is None or lookback_start is None:
            continue

        # 20-day returns
        rets = {}
        for s in stock_cols:
            p_start = close_df.loc[lookback_start, s] if lookback_start in close_df.index else np.nan
            p_end = close_df.loc[entry, s] if entry in close_df.index else np.nan
            if pd.notna(p_start) and pd.notna(p_end) and p_start > 0:
                rets[s] = (p_end / p_start) - 1
        if len(rets) < 5:
            continue

        # Worst 5
        sorted_stocks = sorted(rets.items(), key=lambda x: x[1])[:5]
        for ticker, _ in sorted_stocks:
            trades.append({"ticker": ticker, "entry_date": entry, "exit_date": exit_d})

    return trades


def variant_c_quarter_reversion(close_df, dates):
    """C) Worst-5 in 60d → buy 5 days before quarter end, hold 10 days."""
    trades = []
    qe_dates = get_quarter_end_dates(dates)
    stock_cols = [s for s in STOCKS if s in close_df.columns]

    for qe in qe_dates:
        if qe < pd.Timestamp(OOT_START) or qe > pd.Timestamp(OOT_END):
            continue

        entry = offset_trading_day(dates, qe, -5)
        exit_d = offset_trading_day(dates, qe, 5)
        lookback_start = offset_trading_day(dates, qe, -65)

        if entry is None or exit_d is None or lookback_start is None:
            continue

        rets = {}
        for s in stock_cols:
            p_start = close_df.loc[lookback_start, s] if lookback_start in close_df.index else np.nan
            p_end = close_df.loc[entry, s] if entry in close_df.index else np.nan
            if pd.notna(p_start) and pd.notna(p_end) and p_start > 0:
                rets[s] = (p_end / p_start) - 1
        if len(rets) < 5:
            continue

        sorted_stocks = sorted(rets.items(), key=lambda x: x[1])[:5]
        for ticker, _ in sorted_stocks:
            trades.append({"ticker": ticker, "entry_date": entry, "exit_date": exit_d})

    return trades


def variant_d_window_dressing(close_df, dates):
    """D) Top-5 performers → buy 3d before QE, sell on QE day."""
    trades = []
    qe_dates = get_quarter_end_dates(dates)
    stock_cols = [s for s in STOCKS if s in close_df.columns]

    for qe in qe_dates:
        if qe < pd.Timestamp(OOT_START) or qe > pd.Timestamp(OOT_END):
            continue

        entry = offset_trading_day(dates, qe, -3)
        exit_d = qe
        lookback_start = offset_trading_day(dates, qe, -63)

        if entry is None or lookback_start is None:
            continue

        rets = {}
        for s in stock_cols:
            p_start = close_df.loc[lookback_start, s] if lookback_start in close_df.index else np.nan
            p_end = close_df.loc[entry, s] if entry in close_df.index else np.nan
            if pd.notna(p_start) and pd.notna(p_end) and p_start > 0:
                rets[s] = (p_end / p_start) - 1
        if len(rets) < 5:
            continue

        sorted_stocks = sorted(rets.items(), key=lambda x: x[1], reverse=True)[:5]
        for ticker, _ in sorted_stocks:
            trades.append({"ticker": ticker, "entry_date": entry, "exit_date": exit_d})

    return trades


def variant_e_january_effect(close_df, dates):
    """E) Buy small-cap underperformers in late Dec, hold through Jan."""
    trades = []
    # Use IWM-relative underperformers as "small-cap losers"
    stock_cols = [s for s in STOCKS if s in close_df.columns]

    for year in range(pd.Timestamp(OOT_START).year, pd.Timestamp(OOT_END).year + 1):
        # Entry: ~Dec 20 (or closest trading day)
        entry_target = pd.Timestamp(year - 1, 12, 20)
        exit_target = pd.Timestamp(year, 1, 31)

        entry = offset_trading_day(dates, entry_target, 0) if entry_target in dates else None
        if entry is None:
            # Find closest trading day on/after Dec 20
            mask = (dates >= entry_target) & (dates <= pd.Timestamp(year - 1, 12, 31))
            if mask.any():
                entry = dates[mask][0]
            else:
                continue

        # Exit: last trading day of January
        mask_jan = (dates >= pd.Timestamp(year, 1, 1)) & (dates <= exit_target)
        if not mask_jan.any():
            continue
        exit_d = dates[mask_jan][-1]

        if entry < pd.Timestamp(OOT_START) or exit_d > pd.Timestamp(OOT_END):
            continue

        # Look at prior-year performance, pick worst 5
        lookback_start = offset_trading_day(dates, entry, -252)
        if lookback_start is None:
            continue

        rets = {}
        for s in stock_cols:
            p_start = close_df.loc[lookback_start, s] if lookback_start in close_df.index else np.nan
            p_end = close_df.loc[entry, s] if entry in close_df.index else np.nan
            if pd.notna(p_start) and pd.notna(p_end) and p_start > 0:
                rets[s] = (p_end / p_start) - 1
        if len(rets) < 5:
            continue

        sorted_stocks = sorted(rets.items(), key=lambda x: x[1])[:5]
        for ticker, _ in sorted_stocks:
            trades.append({"ticker": ticker, "entry_date": entry, "exit_date": exit_d})

    return trades


def variant_f_turn_of_month(close_df, dates):
    """F) Buy SPY 1d before month end, sell 3d into new month."""
    trades = []
    me_dates = get_month_end_dates(dates)

    for me in me_dates:
        me = pd.Timestamp(me)
        if me < pd.Timestamp(OOT_START) or me > pd.Timestamp(OOT_END):
            continue

        entry = offset_trading_day(dates, me, -1)
        exit_d = offset_trading_day(dates, me, 3)

        if entry is None or exit_d is None:
            continue
        if "SPY" not in close_df.columns:
            continue

        trades.append({"ticker": "SPY", "entry_date": entry, "exit_date": exit_d})

    return trades


# ── Evaluation ─────────────────────────────────────────────────────────────
def build_equity_curve(trade_results, initial_capital):
    """Build equity curve from trade returns with equal-weight sizing."""
    if not trade_results:
        return pd.Series([initial_capital]), initial_capital, 0.0

    # Sort by entry date
    trades_sorted = sorted(trade_results, key=lambda x: x["entry_date"])

    # Group trades by entry date for batch allocation
    from collections import defaultdict
    groups = defaultdict(list)
    for t in trades_sorted:
        groups[t["entry_date"]].append(t)

    capital = initial_capital
    equity_points = [(trades_sorted[0]["entry_date"], capital)]

    for date_key in sorted(groups.keys()):
        batch = groups[date_key]
        n = len(batch)
        alloc_per = capital / max(n, 1)
        batch_pnl = 0
        for t in batch:
            batch_pnl += alloc_per * t["return"]
        capital += batch_pnl
        equity_points.append((batch[-1]["exit_date"], capital))

    equity = pd.Series([e[1] for e in equity_points], index=[e[0] for e in equity_points])
    return equity, capital, (capital / initial_capital - 1) * 100


def compute_metrics(trade_results, initial_capital):
    """Compute Sharpe, Sortino, MaxDD, PF, WR from trade returns."""
    if not trade_results:
        return {"sharpe": 0, "sortino": 0, "pf": 0, "wr": 0, "max_dd": 0,
                "n_trades": 0, "total_return_pct": 0, "avg_return_pct": 0}

    rets = np.array([t["return"] for t in trade_results])
    n = len(rets)
    avg = np.mean(rets)
    std = np.std(rets, ddof=1) if n > 1 else 1e-9

    # Annualize: assume ~5 day avg hold → ~50 trades/year
    avg_hold = np.mean([t["hold_days"] for t in trade_results])
    trades_per_year = max(252 / max(avg_hold, 1), 1)
    ann_factor = np.sqrt(trades_per_year)

    sharpe = (avg / std * ann_factor) if std > 1e-9 else 0

    downside = rets[rets < 0]
    downside_std = np.std(downside, ddof=1) if len(downside) > 1 else 1e-9
    sortino = (avg / downside_std * ann_factor) if downside_std > 1e-9 else 0

    wins = rets[rets > 0]
    losses = rets[rets < 0]
    gross_profit = np.sum(wins) if len(wins) > 0 else 0
    gross_loss = abs(np.sum(losses)) if len(losses) > 0 else 1e-9
    pf = gross_profit / gross_loss if gross_loss > 1e-9 else float("inf")

    wr = np.mean(rets > 0) * 100

    # Max drawdown from cumulative returns
    cum = np.cumprod(1 + rets)
    running_max = np.maximum.accumulate(cum)
    dd = (cum - running_max) / running_max
    max_dd = np.min(dd) * 100

    equity, final_cap, total_ret = build_equity_curve(trade_results, initial_capital)

    return {
        "sharpe": round(float(sharpe), 3),
        "sortino": round(float(sortino), 3),
        "pf": round(float(pf), 3),
        "wr": round(float(wr), 1),
        "max_dd": round(float(max_dd), 1),
        "n_trades": n,
        "total_return_pct": round(total_ret, 2),
        "avg_return_pct": round(float(avg * 100), 3),
        "final_capital": round(float(final_cap), 2),
        "avg_hold_days": round(float(avg_hold), 1),
    }


def permutation_test(trade_results, actual_sharpe, n_iter=PERM_ITERATIONS):
    """Shuffle trade timing to test if edge is structural."""
    if len(trade_results) < 5:
        return 1.0

    rets = np.array([t["return"] for t in trade_results])
    avg_hold = np.mean([t["hold_days"] for t in trade_results])
    trades_per_year = max(252 / max(avg_hold, 1), 1)
    ann_factor = np.sqrt(trades_per_year)

    count_better = 0
    for _ in range(n_iter):
        np.random.shuffle(rets)
        std = np.std(rets, ddof=1)
        if std > 1e-9:
            perm_sharpe = np.mean(rets) / std * ann_factor
        else:
            perm_sharpe = 0
        if perm_sharpe >= actual_sharpe:
            count_better += 1

    return count_better / n_iter


def regime_analysis(trade_results, regime_series):
    """Split trades by regime (bull/bear), compute Sharpe for each."""
    if not trade_results:
        return 0, 0, 1.0

    bull_rets = []
    bear_rets = []
    for t in trade_results:
        entry = pd.Timestamp(t["entry_date"])
        if entry in regime_series.index:
            r = regime_series.loc[entry]
        else:
            # Find nearest
            idx = regime_series.index.get_indexer([entry], method="ffill")[0]
            r = regime_series.iloc[idx] if idx >= 0 else "bull"

        if r == "bull":
            bull_rets.append(t["return"])
        else:
            bear_rets.append(t["return"])

    def _sharpe(arr):
        if len(arr) < 2:
            return 0
        std = np.std(arr, ddof=1)
        return np.mean(arr) / std if std > 1e-9 else 0

    s_bull = _sharpe(bull_rets)
    s_bear = _sharpe(bear_rets)
    denom = max(abs(s_bull), abs(s_bear), 1e-9)
    gap = abs(s_bull - s_bear) / denom

    return round(s_bull, 3), round(s_bear, 3), round(gap, 3)


# ── 5-Gate Validation ──────────────────────────────────────────────────────
def validate_5gate(metrics, perm_p, regime_gap):
    """Apply 5-gate framework. Returns dict of gate results."""
    gates = {
        "G1_sharpe_gt_0.5": metrics["sharpe"] > 0.5,
        "G2_perm_p_lt_0.05": perm_p < 0.05,
        "G3_regime_gap_lt_0.5": regime_gap < 0.5,
        "G4_maxdd_gt_neg50": metrics["max_dd"] > -50,
        "G5_min_20_trades": metrics["n_trades"] >= 20,
    }
    gates["ALL_PASS"] = all(gates.values())
    return gates


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    np.random.seed(42)

    close_df = download_data()
    dates = close_df.index

    # Regime classification
    regime = classify_regime(close_df)

    # Define variants
    variants = {
        "A_quarter_momentum": ("Top-5 60d performers, buy 5d before QE, hold 10d", variant_a_quarter_momentum),
        "B_monthend_reversion": ("Worst-5 20d performers, buy 3d before ME, hold 5d", variant_b_monthend_reversion),
        "C_quarter_reversion": ("Worst-5 60d performers, buy 5d before QE, hold 10d", variant_c_quarter_reversion),
        "D_window_dressing": ("Top-5 performers, buy 3d before QE, sell on QE", variant_d_window_dressing),
        "E_january_effect": ("Worst-5 annual performers, buy late Dec, hold Jan", variant_e_january_effect),
        "F_turn_of_month": ("Buy SPY 1d before ME, sell 3d into new month", variant_f_turn_of_month),
    }

    all_results = {}
    print("\n" + "=" * 80)
    print("INDEX REBALANCE / RECONSTITUTION TRADING — 6 VARIANTS")
    print("=" * 80)
    print(f"OOT: {OOT_START} to {OOT_END} | Capital: ${INITIAL_CAPITAL} | Slippage: {SLIPPAGE_PCT*100:.2f}%")
    print("=" * 80)

    for name, (desc, func) in variants.items():
        print(f"\n{'─' * 70}")
        print(f"VARIANT {name}")
        print(f"  {desc}")
        print(f"{'─' * 70}")

        # Generate trades
        raw_trades = func(close_df, dates)
        trade_results = simulate_trades(raw_trades, close_df)

        # Metrics
        metrics = compute_metrics(trade_results, INITIAL_CAPITAL)

        # Permutation test
        perm_p = permutation_test(trade_results, metrics["sharpe"])

        # Regime analysis
        s_bull, s_bear, r_gap = regime_analysis(trade_results, regime)

        # 5-gate validation
        gates = validate_5gate(metrics, perm_p, r_gap)

        print(f"  Trades: {metrics['n_trades']} | Avg hold: {metrics['avg_hold_days']}d")
        print(f"  Total return: {metrics['total_return_pct']:+.2f}% | Final capital: ${metrics['final_capital']:.2f}")
        print(f"  Avg return/trade: {metrics['avg_return_pct']:+.3f}%")
        print(f"  Sharpe: {metrics['sharpe']:.3f} | Sortino: {metrics['sortino']:.3f} | PF: {metrics['pf']:.3f} | WR: {metrics['wr']:.1f}%")
        print(f"  MaxDD: {metrics['max_dd']:.1f}%")
        print(f"  Regime — Bull Sharpe: {s_bull:.3f} | Bear Sharpe: {s_bear:.3f} | Gap: {r_gap:.3f}")
        print(f"  Permutation p-value: {perm_p:.4f}")
        print()
        print(f"  5-GATE VALIDATION:")
        for gname, passed in gates.items():
            status = "PASS" if passed else "FAIL"
            print(f"    [{status}] {gname}")
        verdict = "PASS" if gates["ALL_PASS"] else "FAIL"
        print(f"  ──> OVERALL: {verdict}")

        all_results[name] = {
            "description": desc,
            "metrics": metrics,
            "perm_p": round(perm_p, 4),
            "regime_bull_sharpe": s_bull,
            "regime_bear_sharpe": s_bear,
            "regime_gap": r_gap,
            "gates": {k: bool(v) for k, v in gates.items()},
            "sample_trades": trade_results[:5] if trade_results else [],
        }

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Variant':<28} {'Sharpe':>7} {'Sortino':>8} {'PF':>6} {'WR':>6} {'MaxDD':>7} {'#Tr':>5} {'Ret%':>8} {'Pass':>5}")
    print("-" * 80)
    for name, res in all_results.items():
        m = res["metrics"]
        v = "YES" if res["gates"]["ALL_PASS"] else "NO"
        print(f"{name:<28} {m['sharpe']:>7.3f} {m['sortino']:>8.3f} {m['pf']:>6.2f} {m['wr']:>5.1f}% {m['max_dd']:>6.1f}% {m['n_trades']:>5} {m['total_return_pct']:>+7.2f}% {v:>5}")

    passed = [n for n, r in all_results.items() if r["gates"]["ALL_PASS"]]
    print(f"\nVariants passing all 5 gates: {len(passed)}/{len(all_results)}")
    if passed:
        print(f"  Passed: {', '.join(passed)}")
    else:
        print("  None passed all 5 gates.")

    # Save results
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump({
            "strategy": "Index Rebalance / Reconstitution Trading",
            "oot_period": f"{OOT_START} to {OOT_END}",
            "initial_capital": INITIAL_CAPITAL,
            "slippage_pct": SLIPPAGE_PCT,
            "n_variants": len(all_results),
            "n_passed": len(passed),
            "variants": all_results,
        }, f, indent=2)
    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
