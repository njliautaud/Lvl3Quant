#!/usr/bin/env python3
"""
Earnings Season Contagion — Sector ETF Options Signal Backtest
==============================================================
Hypothesis: Mega-cap earnings surprises set the tone for their sector ETF
over the following 1-5 trading days. A big beat → buy calls on sector ETF;
a big miss → buy puts on sector ETF.

Methodology:
  1. Identify bellwether earnings dates via yfinance
  2. Compute earnings-day return (close-to-close)
  3. If return > +2% → positive surprise → go LONG sector ETF next day, hold 3d
     If return < -2% → negative surprise → go SHORT sector ETF next day, hold 3d
  4. Also test same-day entry and multi-bellwether confluence
  5. Measure residual edge AFTER initial sector ETF reaction

5-Gate acceptance:
  G1: Sharpe > 0.5
  G2: permutation p < 0.05
  G3: regime gap < 0.50
  G4: MaxDD < 50%
  G5: trades >= 30
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from collections import defaultdict
import sys

# ── Configuration ──────────────────────────────────────────────────────────────

BELLWETHER_MAP = {
    "XLK": ["AAPL", "MSFT", "NVDA"],
    "XLC": ["GOOGL", "META"],
    "XLF": ["JPM", "BAC", "GS", "MS"],
    "XLE": ["XOM", "CVX"],
    "XLY": ["AMZN", "TSLA", "HD"],
    "XLV": ["UNH", "JNJ", "PFE"],
    "XLI": ["CAT", "BA", "HON"],
    "XLP": ["PG", "KO", "WMT"],
}

SURPRISE_THRESHOLD = 0.02  # 2% move = surprise
HOLD_DAYS = 3
TEST_START = "2020-01-01"
TEST_END = "2026-08-15"
N_PERMUTATIONS = 1000
SEED = 42

# ── Data Download ──────────────────────────────────────────────────────────────

def download_prices(tickers, start, end):
    """Download adjusted close prices for all tickers."""
    all_tickers = list(set(tickers))
    print(f"Downloading price data for {len(all_tickers)} tickers...")
    data = yf.download(all_tickers, start=start, end=end, auto_adjust=True, progress=False)
    if "Close" in data.columns or (isinstance(data.columns, pd.MultiIndex) and "Close" in data.columns.get_level_values(0)):
        if isinstance(data.columns, pd.MultiIndex):
            closes = data["Close"]
        else:
            closes = data[["Close"]]
            closes.columns = all_tickers
    else:
        closes = data
    return closes


def get_earnings_dates_for_ticker(ticker_str):
    """Get historical earnings dates from yfinance."""
    tk = yf.Ticker(ticker_str)
    try:
        # Try earnings_dates first (has historical data)
        ed = tk.earnings_dates
        if ed is not None and len(ed) > 0:
            dates = ed.index.tz_localize(None) if ed.index.tz else ed.index
            return sorted(dates.tolist())
    except Exception:
        pass

    try:
        # Fallback: quarterly earnings from earnings_history
        eh = tk.earnings_history
        if eh is not None and len(eh) > 0:
            if isinstance(eh, pd.DataFrame) and "Earnings Date" in eh.columns:
                return sorted(eh["Earnings Date"].tolist())
    except Exception:
        pass

    try:
        # Fallback: get_earnings_dates
        ed = tk.get_earnings_dates(limit=50)
        if ed is not None and len(ed) > 0:
            dates = ed.index.tz_localize(None) if ed.index.tz else ed.index
            return sorted(dates.tolist())
    except Exception:
        pass

    return []


# ── Signal Generation ──────────────────────────────────────────────────────────

def generate_signals(closes, bellwether_map, threshold, hold_days):
    """
    For each bellwether earnings date:
      - Compute earnings day return (close-to-close)
      - If |return| > threshold → signal on sector ETF
      - Entry: next trading day open (approximated as close)
      - Exit: hold_days later close

    Returns DataFrame of trades.
    """
    all_tickers = []
    for etf, bws in bellwether_map.items():
        all_tickers.append(etf)
        all_tickers.extend(bws)
    all_tickers = list(set(all_tickers))

    # Collect all earnings dates
    print("\nFetching earnings dates...")
    earnings_dates = {}
    for etf, bellwethers in bellwether_map.items():
        for bw in bellwethers:
            print(f"  {bw}...", end=" ", flush=True)
            dates = get_earnings_dates_for_ticker(bw)
            # Filter to test period
            start_dt = pd.Timestamp(TEST_START)
            end_dt = pd.Timestamp(TEST_END)
            dates = [d for d in dates if start_dt <= pd.Timestamp(d) <= end_dt]
            earnings_dates[(etf, bw)] = dates
            print(f"{len(dates)} dates")

    # Generate trades
    trades = []
    trading_days = closes.index

    for (etf, bw), dates in earnings_dates.items():
        if etf not in closes.columns or bw not in closes.columns:
            continue

        for edate in dates:
            edate = pd.Timestamp(edate).normalize()

            # Find the earnings day in trading calendar
            mask = trading_days >= edate
            if mask.sum() == 0:
                continue
            earn_day_idx = trading_days[mask][0]
            earn_day_loc = trading_days.get_loc(earn_day_idx)

            # Need previous day for return calc
            if earn_day_loc < 1:
                continue

            prev_day = trading_days[earn_day_loc - 1]

            # Bellwether earnings day return
            bw_close_prev = closes.loc[prev_day, bw]
            bw_close_earn = closes.loc[earn_day_idx, bw]
            if pd.isna(bw_close_prev) or pd.isna(bw_close_earn) or bw_close_prev == 0:
                continue
            bw_ret = (bw_close_earn - bw_close_prev) / bw_close_prev

            # Check surprise threshold
            if abs(bw_ret) < threshold:
                continue

            direction = 1 if bw_ret > 0 else -1

            # ── NEXT-DAY ENTRY ──
            entry_loc = earn_day_loc + 1
            exit_loc = earn_day_loc + 1 + hold_days

            if exit_loc >= len(trading_days):
                continue

            entry_day = trading_days[entry_loc]
            exit_day = trading_days[exit_loc]

            etf_entry = closes.loc[entry_day, etf]
            etf_exit = closes.loc[exit_day, etf]

            if pd.isna(etf_entry) or pd.isna(etf_exit) or etf_entry == 0:
                continue

            etf_ret = (etf_exit - etf_entry) / etf_entry
            trade_ret = direction * etf_ret

            # Also capture same-day sector ETF move (to measure residual)
            etf_close_prev = closes.loc[prev_day, etf]
            etf_close_earn = closes.loc[earn_day_idx, etf]
            if pd.isna(etf_close_prev) or pd.isna(etf_close_earn) or etf_close_prev == 0:
                etf_earn_day_ret = np.nan
            else:
                etf_earn_day_ret = (etf_close_earn - etf_close_prev) / etf_close_prev

            trades.append({
                "etf": etf,
                "bellwether": bw,
                "earn_date": earn_day_idx,
                "entry_date": entry_day,
                "exit_date": exit_day,
                "bw_earn_ret": bw_ret,
                "direction": direction,
                "etf_earn_day_ret": etf_earn_day_ret,
                "etf_trade_ret": etf_ret,
                "trade_ret": trade_ret,
                "entry_type": "next_day",
            })

            # ── SAME-DAY ENTRY (buy sector ETF at close on earnings day) ──
            sd_exit_loc = earn_day_loc + hold_days
            if sd_exit_loc >= len(trading_days):
                continue

            sd_exit_day = trading_days[sd_exit_loc]
            sd_etf_entry = closes.loc[earn_day_idx, etf]
            sd_etf_exit = closes.loc[sd_exit_day, etf]

            if pd.isna(sd_etf_entry) or pd.isna(sd_etf_exit) or sd_etf_entry == 0:
                continue

            sd_etf_ret = (sd_etf_exit - sd_etf_entry) / sd_etf_entry
            sd_trade_ret = direction * sd_etf_ret

            trades.append({
                "etf": etf,
                "bellwether": bw,
                "earn_date": earn_day_idx,
                "entry_date": earn_day_idx,
                "exit_date": sd_exit_day,
                "bw_earn_ret": bw_ret,
                "direction": direction,
                "etf_earn_day_ret": etf_earn_day_ret,
                "etf_trade_ret": sd_etf_ret,
                "trade_ret": sd_trade_ret,
                "entry_type": "same_day",
            })

    return pd.DataFrame(trades)


# ── Analytics ──────────────────────────────────────────────────────────────────

def compute_metrics(returns, annual_factor=252/3):
    """Compute Sharpe, Sortino, PF, WR, MaxDD from a series of trade returns."""
    if len(returns) == 0:
        return {}
    returns = np.array(returns)
    n = len(returns)
    mean_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1) if n > 1 else 1e-9

    # Sharpe (annualized — ~84 trades/year if 252/3)
    sharpe = (mean_ret / std_ret) * np.sqrt(annual_factor) if std_ret > 1e-12 else 0

    # Sortino
    downside = returns[returns < 0]
    down_std = np.std(downside, ddof=1) if len(downside) > 1 else 1e-9
    sortino = (mean_ret / down_std) * np.sqrt(annual_factor) if down_std > 1e-12 else 0

    # PF
    gross_wins = returns[returns > 0].sum()
    gross_losses = abs(returns[returns < 0].sum())
    pf = gross_wins / gross_losses if gross_losses > 1e-12 else np.inf

    # WR
    wr = (returns > 0).sum() / n

    # MaxDD (on cumulative equity curve)
    cum = np.cumsum(returns)
    running_max = np.maximum.accumulate(cum)
    dd = cum - running_max
    max_dd = abs(dd.min()) if len(dd) > 0 else 0

    # Avg return
    avg_ret = mean_ret

    return {
        "n_trades": n,
        "avg_ret_pct": avg_ret * 100,
        "sharpe": sharpe,
        "sortino": sortino,
        "pf": pf,
        "wr_pct": wr * 100,
        "max_dd_pct": max_dd * 100,
        "total_ret_pct": cum[-1] * 100 if len(cum) > 0 else 0,
    }


def permutation_test(returns, n_perms=1000, seed=42):
    """Shuffle direction labels to test statistical significance."""
    rng = np.random.RandomState(seed)
    actual_mean = np.mean(returns)
    count_better = 0
    for _ in range(n_perms):
        # Randomly flip signs
        signs = rng.choice([-1, 1], size=len(returns))
        perm_mean = np.mean(returns * signs)
        if perm_mean >= actual_mean:
            count_better += 1
    p_value = count_better / n_perms
    return p_value


def regime_stratify(trades_df, spy_closes):
    """Split trades by whether SPY was green or red on the entry day."""
    spy_rets = spy_closes.pct_change()

    green_rets = []
    red_rets = []

    for _, row in trades_df.iterrows():
        entry = row["entry_date"]
        if entry in spy_rets.index:
            spy_r = spy_rets.loc[entry]
            if spy_r >= 0:
                green_rets.append(row["trade_ret"])
            else:
                red_rets.append(row["trade_ret"])

    return np.array(green_rets), np.array(red_rets)


def five_gate_check(metrics, p_value, regime_gap):
    """Check 5-gate acceptance criteria."""
    gates = {
        "G1_Sharpe>0.5": metrics.get("sharpe", 0) > 0.5,
        "G2_p<0.05": p_value < 0.05,
        "G3_regime_gap<0.50": regime_gap < 0.50,
        "G4_MDD<50%": metrics.get("max_dd_pct", 100) < 50,
        "G5_trades>=30": metrics.get("n_trades", 0) >= 30,
    }
    return gates


# ── Multi-bellwether confluence ────────────────────────────────────────────────

def confluence_signals(trades_df):
    """
    When multiple bellwethers from the same sector report in the same week
    and agree on direction, the signal should be stronger.
    """
    # Group by etf + week
    trades_df = trades_df.copy()
    trades_df["week"] = trades_df["earn_date"].dt.isocalendar().week.values
    trades_df["year"] = trades_df["earn_date"].dt.year

    confluence_trades = []

    for (etf, year, week), group in trades_df.groupby(["etf", "year", "week"]):
        if len(group) < 2:
            continue
        # Check if all bellwethers agree on direction
        directions = group["direction"].values
        if np.all(directions == directions[0]):
            # Use average trade return (they share the same ETF move mostly)
            avg_ret = group["trade_ret"].mean()
            confluence_trades.append({
                "etf": etf,
                "n_bellwethers": len(group),
                "direction": directions[0],
                "trade_ret": avg_ret,
                "earn_date": group["earn_date"].iloc[0],
                "entry_date": group["entry_date"].iloc[0],
            })

    return pd.DataFrame(confluence_trades)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("EARNINGS CONTAGION BACKTEST")
    print("=" * 70)

    # Collect all tickers
    all_tickers = set()
    for etf, bws in BELLWETHER_MAP.items():
        all_tickers.add(etf)
        all_tickers.update(bws)
    all_tickers.add("SPY")

    # Download prices
    closes = download_prices(list(all_tickers), TEST_START, TEST_END)
    spy_closes = closes["SPY"] if "SPY" in closes.columns else None

    # Generate signals
    trades_df = generate_signals(closes, BELLWETHER_MAP, SURPRISE_THRESHOLD, HOLD_DAYS)

    if len(trades_df) == 0:
        print("\nNO TRADES GENERATED. Cannot continue.")
        sys.exit(1)

    print(f"\nTotal signals generated: {len(trades_df)}")

    # ── Analyze by entry type ──────────────────────────────────────────────
    for entry_type in ["next_day", "same_day"]:
        subset = trades_df[trades_df["entry_type"] == entry_type]
        if len(subset) == 0:
            continue

        print(f"\n{'='*70}")
        print(f"ENTRY TYPE: {entry_type.upper()} | {HOLD_DAYS}-day hold | {SURPRISE_THRESHOLD*100:.0f}% threshold")
        print(f"{'='*70}")

        rets = subset["trade_ret"].values
        metrics = compute_metrics(rets)

        print(f"\n  Trades:     {metrics['n_trades']}")
        print(f"  Avg Return: {metrics['avg_ret_pct']:+.3f}%")
        print(f"  Win Rate:   {metrics['wr_pct']:.1f}%")
        print(f"  Sharpe:     {metrics['sharpe']:.3f}")
        print(f"  Sortino:    {metrics['sortino']:.3f}")
        print(f"  PF:         {metrics['pf']:.3f}")
        print(f"  Max DD:     {metrics['max_dd_pct']:.2f}%")
        print(f"  Total Ret:  {metrics['total_ret_pct']:+.2f}%")

        # Direction breakdown
        longs = subset[subset["direction"] == 1]
        shorts = subset[subset["direction"] == -1]
        if len(longs) > 0:
            long_wr = (longs["trade_ret"] > 0).mean() * 100
            long_avg = longs["trade_ret"].mean() * 100
            print(f"\n  LONG trades:  {len(longs)}, WR={long_wr:.1f}%, avg={long_avg:+.3f}%")
        if len(shorts) > 0:
            short_wr = (shorts["trade_ret"] > 0).mean() * 100
            short_avg = shorts["trade_ret"].mean() * 100
            print(f"  SHORT trades: {len(shorts)}, WR={short_wr:.1f}%, avg={short_avg:+.3f}%")

        # Per-sector breakdown
        print(f"\n  Per-sector breakdown:")
        for etf in sorted(subset["etf"].unique()):
            sec = subset[subset["etf"] == etf]
            sec_m = compute_metrics(sec["trade_ret"].values)
            print(f"    {etf}: {sec_m['n_trades']} trades, WR={sec_m['wr_pct']:.1f}%, "
                  f"Sharpe={sec_m['sharpe']:.2f}, avg={sec_m['avg_ret_pct']:+.3f}%")

        # Residual edge analysis
        print(f"\n  Residual edge (does ETF continue moving AFTER earnings day)?")
        valid_resid = subset.dropna(subset=["etf_earn_day_ret"])
        if len(valid_resid) > 0:
            earn_day_move = (valid_resid["direction"] * valid_resid["etf_earn_day_ret"]).mean() * 100
            post_move = valid_resid["trade_ret"].mean() * 100
            print(f"    Avg sector ETF move on earnings day (aligned): {earn_day_move:+.3f}%")
            print(f"    Avg post-earnings trade return:                {post_move:+.3f}%")
            if entry_type == "next_day":
                print(f"    → Next-day entry captures RESIDUAL momentum only")
            else:
                print(f"    → Same-day entry includes the initial reaction")

        # Permutation test
        p_value = permutation_test(rets, N_PERMUTATIONS, SEED)
        print(f"\n  Permutation test ({N_PERMUTATIONS} shuffles): p = {p_value:.4f}")

        # Regime stratification
        if spy_closes is not None:
            green_rets, red_rets = regime_stratify(subset, spy_closes)
            if len(green_rets) > 5 and len(red_rets) > 5:
                green_m = compute_metrics(green_rets)
                red_m = compute_metrics(red_rets)
                regime_gap = abs(green_m["sharpe"] - red_m["sharpe"]) / max(abs(green_m["sharpe"]), abs(red_m["sharpe"]), 1e-9)
                print(f"\n  Regime stratification:")
                print(f"    GREEN days: {len(green_rets)} trades, Sharpe={green_m['sharpe']:.3f}, WR={green_m['wr_pct']:.1f}%")
                print(f"    RED days:   {len(red_rets)} trades, Sharpe={red_m['sharpe']:.3f}, WR={red_m['wr_pct']:.1f}%")
                print(f"    Regime gap: {regime_gap:.3f}")
            else:
                regime_gap = 0
                print(f"\n  Regime stratification: insufficient data (green={len(green_rets)}, red={len(red_rets)})")
        else:
            regime_gap = 0

        # 5-gate check
        gates = five_gate_check(metrics, p_value, regime_gap)
        n_pass = sum(gates.values())
        print(f"\n  5-GATE CHECK:")
        for gate, passed in gates.items():
            status = "PASS" if passed else "FAIL"
            print(f"    {gate}: {status}")
        print(f"  Result: {n_pass}/5 gates passed → {'ACCEPT' if n_pass == 5 else 'REJECT'}")

    # ── Confluence analysis ────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("CONFLUENCE ANALYSIS: Multiple bellwethers same week, same direction")
    print(f"{'='*70}")

    for entry_type in ["next_day", "same_day"]:
        subset = trades_df[trades_df["entry_type"] == entry_type]
        conf_df = confluence_signals(subset)
        if len(conf_df) == 0:
            print(f"\n  {entry_type}: No confluence signals found")
            continue

        conf_rets = conf_df["trade_ret"].values
        conf_m = compute_metrics(conf_rets)
        print(f"\n  {entry_type.upper()} confluence ({len(conf_df)} signals):")
        print(f"    Avg Return: {conf_m['avg_ret_pct']:+.3f}%")
        print(f"    WR:         {conf_m['wr_pct']:.1f}%")
        print(f"    Sharpe:     {conf_m['sharpe']:.3f}")
        print(f"    PF:         {conf_m['pf']:.3f}")

        # Compare to non-confluence
        all_rets = subset["trade_ret"].values
        all_m = compute_metrics(all_rets)
        print(f"    vs All signals: Sharpe={all_m['sharpe']:.3f}, WR={all_m['wr_pct']:.1f}%")
        print(f"    Confluence lift: {conf_m['avg_ret_pct'] - all_m['avg_ret_pct']:+.3f}% per trade")

    # ── Surprise magnitude analysis ────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SURPRISE MAGNITUDE: Does bigger surprise = bigger edge?")
    print(f"{'='*70}")

    nd = trades_df[trades_df["entry_type"] == "next_day"].copy()
    if len(nd) > 0:
        nd["abs_bw_ret"] = nd["bw_earn_ret"].abs()
        for lo, hi, label in [(0.02, 0.05, "2-5%"), (0.05, 0.10, "5-10%"), (0.10, 1.0, "10%+")]:
            bucket = nd[(nd["abs_bw_ret"] >= lo) & (nd["abs_bw_ret"] < hi)]
            if len(bucket) >= 5:
                bm = compute_metrics(bucket["trade_ret"].values)
                print(f"  {label} surprise: {bm['n_trades']} trades, "
                      f"WR={bm['wr_pct']:.1f}%, avg={bm['avg_ret_pct']:+.3f}%, "
                      f"Sharpe={bm['sharpe']:.3f}")

    # ── Year-by-year stability ─────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("YEAR-BY-YEAR STABILITY (next-day entry)")
    print(f"{'='*70}")

    if len(nd) > 0:
        nd["year"] = nd["earn_date"].dt.year
        for year in sorted(nd["year"].unique()):
            yr = nd[nd["year"] == year]
            ym = compute_metrics(yr["trade_ret"].values)
            print(f"  {year}: {ym['n_trades']:3d} trades, WR={ym['wr_pct']:.1f}%, "
                  f"avg={ym['avg_ret_pct']:+.3f}%, Sharpe={ym['sharpe']:.3f}")

    print(f"\n{'='*70}")
    print("BACKTEST COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
