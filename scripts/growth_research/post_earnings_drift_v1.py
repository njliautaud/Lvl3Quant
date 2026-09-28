#!/usr/bin/env python3
"""
Post-Earnings Announcement Drift (PEAD) Backtest v1
=====================================================
One of the most persistent anomalies in finance: stocks that gap on earnings
continue to drift in the same direction for 20-60 trading days.

Strategy:
  - Universe: 50 large-cap stocks
  - Signal: earnings-day gap (open-to-prior-close) > threshold
  - Long: gap > +threshold → buy at T+1 open, hold N days
  - Short: gap < -threshold → short at T+1 open, hold N days
  - Thresholds tested: 2%, 3%, 5%
  - Hold periods tested: 10, 20, 40, 60 days
  - Portfolio: equal-weight, max 10 concurrent positions
  - Capital: $100K fixed, NO DCA (HC #713)
  - Execution: T+1 open (next day after earnings)

Adversarial (HC #705):
  - Permutation test: shuffle which earnings events are traded
  - Sub-period consistency (3 blocks)
  - Survivorship bias: exclude top 5 performers and retest
"""

import json
import os
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# dependency check
for pkg in ["yfinance", "scipy"]:
    try:
        __import__(pkg)
    except ImportError:
        os.system(f"{sys.executable} -m pip install {pkg} -q")

import yfinance as yf
from scipy import stats

# Force unbuffered output
import functools
print = functools.partial(print, flush=True)

# ---------- paths ----------
OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/post_earnings_drift_v1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- constants ----------
INITIAL_CAPITAL = 100_000
MAX_POSITIONS = 10
START_DATE = "2019-01-01"
END_DATE = "2026-07-01"

UNIVERSE = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "AMD", "NFLX", "CRM",
    "AVGO", "COST", "JPM", "V", "MA", "UNH", "HD", "PG", "JNJ", "ABBV",
    "LLY", "MRK", "PFE", "WMT", "KO", "PEP", "MCD", "DIS", "INTC", "CSCO",
    "QCOM", "TXN", "ADBE", "PYPL", "SQ", "SHOP", "NOW", "ZS", "CRWD", "SNOW",
    "BA", "CAT", "DE", "GS", "MS", "BLK", "AXP", "ORCL", "IBM", "ACN",
]

GAP_THRESHOLDS = [0.02, 0.03, 0.05]
HOLD_PERIODS = [10, 20, 40, 60]


# ============================================================
#  DATA DOWNLOAD
# ============================================================
def download_prices(tickers: List[str], start: str, end: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Download open and close prices for all tickers."""
    print(f"  Downloading price data for {len(tickers)} tickers...")
    # Add buffer for hold periods
    start_dt = pd.Timestamp(start) - pd.Timedelta(days=30)
    end_dt = pd.Timestamp(end) + pd.Timedelta(days=90)  # buffer for hold periods

    raw = yf.download(
        " ".join(tickers + ["SPY"]),
        start=start_dt.strftime("%Y-%m-%d"),
        end=end_dt.strftime("%Y-%m-%d"),
        progress=False,
        group_by="ticker",
    )

    opens = pd.DataFrame()
    closes = pd.DataFrame()

    for t in tickers + ["SPY"]:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                o = raw[(t, "Open")]
                c = raw[(t, "Close")]
            else:
                o = raw["Open"]
                c = raw["Close"]
            opens[t] = o
            closes[t] = c
        except (KeyError, Exception):
            pass

    print(f"  Got data for {len(closes.columns)} tickers, {len(closes)} trading days")
    print(f"  Date range: {closes.index.min().date()} to {closes.index.max().date()}")
    return opens, closes


def detect_earnings_gaps(
    opens: pd.DataFrame, closes: pd.DataFrame, tickers: List[str]
) -> pd.DataFrame:
    """
    Detect earnings-day gaps using price action (vectorized).

    An earnings gap is when the open is significantly different from prior close.
    We detect gaps > 2% as potential earnings events.
    """
    print("  Detecting earnings gaps from price data...")
    all_frames = []

    for ticker in tickers:
        if ticker not in closes.columns or ticker not in opens.columns:
            continue

        c = closes[ticker].dropna()
        o = opens[ticker].dropna()
        common_idx = c.index.intersection(o.index)
        if len(common_idx) < 30:
            continue

        c = c.loc[common_idx]
        o = o.loc[common_idx]

        prior_close = c.shift(1)
        gap_pct = (o - prior_close) / prior_close

        # Vectorized filter for |gap| >= 2%
        mask = gap_pct.abs() >= 0.02
        if mask.sum() == 0:
            continue

        sub = pd.DataFrame({
            "ticker": ticker,
            "date": gap_pct.index[mask],
            "gap_pct": gap_pct[mask].values,
            "prior_close": prior_close[mask].values,
            "open": o[mask].values,
        })
        sub["direction"] = np.where(sub["gap_pct"] > 0, "long", "short")
        all_frames.append(sub)

    if not all_frames:
        return pd.DataFrame()

    df = pd.concat(all_frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Deduplicate: for each ticker, keep only 1 event per 60 days
    # (filters out non-earnings gaps somewhat)
    deduped = []
    ticker_last: Dict[str, pd.Timestamp] = {}
    for _, row in df.iterrows():
        key = row["ticker"]
        dt = row["date"]
        if key in ticker_last and (dt - ticker_last[key]).days < 60:
            continue
        ticker_last[key] = dt
        deduped.append(row)

    df = pd.DataFrame(deduped).reset_index(drop=True)
    print(f"  Found {len(df)} earnings-gap events across {df['ticker'].nunique()} tickers")
    print(f"  Long gaps: {(df['direction']=='long').sum()}, Short gaps: {(df['direction']=='short').sum()}")
    return df


# ============================================================
#  BACKTEST ENGINE
# ============================================================
def run_backtest(
    events: pd.DataFrame,
    opens: pd.DataFrame,
    closes: pd.DataFrame,
    gap_threshold: float = 0.02,
    hold_days: int = 20,
    long_only: bool = False,
    max_positions: int = MAX_POSITIONS,
    excluded_tickers: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run PEAD backtest.

    Returns:
        trades: DataFrame of individual trades
        equity: DataFrame with daily equity curve
    """
    # Filter events by threshold
    filtered = events[events["gap_pct"].abs() >= gap_threshold].copy()
    if excluded_tickers:
        filtered = filtered[~filtered["ticker"].isin(excluded_tickers)]
    if long_only:
        filtered = filtered[filtered["direction"] == "long"]

    if filtered.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Sort by date
    filtered = filtered.sort_values("date").reset_index(drop=True)

    # Get all trading days
    all_dates = closes.index.sort_values()
    date_to_idx = {d: i for i, d in enumerate(all_dates)}

    # Pre-compute entry/exit indices and prices for all events (vectorized prep)
    event_dates = filtered["date"].values
    event_tickers = filtered["ticker"].values
    event_gaps = filtered["gap_pct"].values
    event_dirs = filtered["direction"].values

    trades = []
    active_exit_indices = []  # list of exit_idx for active positions
    active_entry_indices = []  # list of entry_idx for active positions
    position_size = INITIAL_CAPITAL / max_positions

    for i in range(len(filtered)):
        event_date = pd.Timestamp(event_dates[i])
        event_idx = date_to_idx.get(event_date)
        if event_idx is None or event_idx + 1 >= len(all_dates):
            continue

        entry_idx = event_idx + 1
        exit_idx = entry_idx + hold_days
        if exit_idx >= len(all_dates):
            continue

        entry_date = all_dates[entry_idx]
        exit_date = all_dates[exit_idx]
        ticker = event_tickers[i]

        if ticker not in opens.columns:
            continue

        entry_price = opens[ticker].iloc[entry_idx] if entry_idx < len(opens) else np.nan
        if pd.isna(entry_price) or entry_price <= 0:
            continue

        exit_price = closes[ticker].iloc[exit_idx] if exit_idx < len(closes) else np.nan
        if pd.isna(exit_price) or exit_price <= 0:
            continue

        # Count active positions (those whose exit_idx > entry_idx)
        n_active = sum(1 for ei in active_exit_indices if ei > entry_idx)
        if n_active >= max_positions:
            continue

        direction = event_dirs[i]
        if direction == "long":
            ret = (exit_price - entry_price) / entry_price
        else:
            ret = (entry_price - exit_price) / entry_price

        trades.append({
            "ticker": ticker,
            "direction": direction,
            "event_date": event_date,
            "entry_date": entry_date,
            "exit_date": exit_date,
            "gap_pct": event_gaps[i],
            "entry_price": entry_price,
            "exit_price": exit_price,
            "return": ret,
            "pnl": ret * position_size,
            "hold_days": hold_days,
        })

        active_exit_indices.append(exit_idx)

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        return trades_df, pd.DataFrame()

    # Build equity curve
    trades_df = trades_df.sort_values("entry_date").reset_index(drop=True)

    # Simple equity: cumulative PnL
    trades_df["cum_pnl"] = trades_df["pnl"].cumsum()
    trades_df["equity"] = INITIAL_CAPITAL + trades_df["cum_pnl"]

    # Build daily equity curve
    equity_curve = build_daily_equity(trades_df, opens, closes, all_dates)

    return trades_df, equity_curve


def build_daily_equity(
    trades: pd.DataFrame,
    opens: pd.DataFrame,
    closes: pd.DataFrame,
    all_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Build a daily equity curve from trades."""
    if trades.empty:
        return pd.DataFrame()

    min_date = trades["entry_date"].min()
    max_date = trades["exit_date"].max()

    mask = (all_dates >= min_date) & (all_dates <= max_date)
    dates = all_dates[mask]

    equity = INITIAL_CAPITAL
    daily = []

    for dt in dates:
        # Mark-to-market all active positions
        day_pnl = 0
        active = trades[
            (trades["entry_date"] <= dt) & (trades["exit_date"] >= dt)
        ]

        for _, t in active.iterrows():
            pos_size = INITIAL_CAPITAL / MAX_POSITIONS
            ticker = t["ticker"]
            entry_px = t["entry_price"]

            if ticker in closes.columns:
                current_px = closes[ticker].get(dt)
                if pd.notna(current_px) and entry_px > 0:
                    if t["direction"] == "long":
                        day_ret = (current_px - entry_px) / entry_px
                    else:
                        day_ret = (entry_px - current_px) / entry_px
                    day_pnl += day_ret * pos_size

        daily.append({"date": dt, "equity": INITIAL_CAPITAL + day_pnl})

    # Actually build cumulative from realized trades
    # Simpler approach: use exit-day PnL accumulation
    daily_df = pd.DataFrame(daily)
    if daily_df.empty:
        return daily_df

    daily_df = daily_df.set_index("date")

    # Overlay realized PnL
    realized = 0
    for dt in daily_df.index:
        exiting = trades[trades["exit_date"] == dt]
        realized += exiting["pnl"].sum()

    # Reconstruct properly using trade-level returns
    # Group trades by exit date and sum PnL
    exit_pnl = trades.groupby("exit_date")["pnl"].sum()

    cum_pnl = pd.Series(0.0, index=daily_df.index)
    running = 0.0
    for dt in cum_pnl.index:
        if dt in exit_pnl.index:
            running += exit_pnl[dt]
        cum_pnl[dt] = running

    daily_df["equity"] = INITIAL_CAPITAL + cum_pnl
    daily_df["daily_return"] = daily_df["equity"].pct_change().fillna(0)

    return daily_df


# ============================================================
#  METRICS
# ============================================================
def compute_metrics(trades: pd.DataFrame, equity: pd.DataFrame = None) -> Dict:
    """Compute comprehensive performance metrics."""
    if trades.empty:
        return {"n_trades": 0}

    returns = trades["return"].values
    pnl = trades["pnl"].values

    n = len(returns)
    winners = returns[returns > 0]
    losers = returns[returns <= 0]

    avg_win = np.mean(winners) if len(winners) > 0 else 0
    avg_loss = np.mean(np.abs(losers)) if len(losers) > 0 else 0
    win_rate = len(winners) / n if n > 0 else 0
    profit_factor = (np.sum(winners) / np.sum(np.abs(losers))) if len(losers) > 0 and np.sum(np.abs(losers)) > 0 else float("inf")

    # Annualize from trade returns
    # Estimate trades per year
    date_range = (trades["exit_date"].max() - trades["entry_date"].min()).days
    years = max(date_range / 365.25, 0.5)
    trades_per_year = n / years

    mean_ret = np.mean(returns)
    std_ret = np.std(returns) if n > 1 else 1e-9

    # Sharpe: annualize assuming trades_per_year independent trades
    sharpe = (mean_ret / std_ret) * np.sqrt(trades_per_year) if std_ret > 0 else 0

    # Sortino
    downside = returns[returns < 0]
    downside_std = np.std(downside) if len(downside) > 1 else 1e-9
    sortino = (mean_ret / downside_std) * np.sqrt(trades_per_year) if downside_std > 0 else 0

    # CAGR from equity curve
    total_pnl = np.sum(pnl)
    total_return = total_pnl / INITIAL_CAPITAL
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

    # Max drawdown from equity curve
    if equity is not None and not equity.empty and "equity" in equity.columns:
        eq = equity["equity"].values
        peak = np.maximum.accumulate(eq)
        dd = (eq - peak) / peak
        max_dd = np.min(dd)
    else:
        cum = INITIAL_CAPITAL + np.cumsum(pnl)
        peak = np.maximum.accumulate(cum)
        dd = (cum - peak) / peak
        max_dd = np.min(dd) if len(dd) > 0 else 0

    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    return {
        "n_trades": n,
        "win_rate": win_rate,
        "avg_return": mean_ret,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "sharpe": sharpe,
        "sortino": sortino,
        "cagr": cagr,
        "max_dd": max_dd,
        "calmar": calmar,
        "total_pnl": total_pnl,
        "total_return": total_return,
        "years": years,
    }


def compute_monthly_returns(trades: pd.DataFrame) -> pd.DataFrame:
    """Compute monthly return table."""
    if trades.empty:
        return pd.DataFrame()

    trades = trades.copy()
    trades["month"] = trades["exit_date"].dt.to_period("M")
    monthly = trades.groupby("month").agg(
        pnl=("pnl", "sum"),
        n_trades=("return", "count"),
        avg_return=("return", "mean"),
        win_rate=("return", lambda x: (x > 0).mean()),
    )
    monthly["cum_pnl"] = monthly["pnl"].cumsum()
    return monthly


# ============================================================
#  ADVERSARIAL TESTS (HC #705)
# ============================================================
def permutation_test(
    events: pd.DataFrame,
    opens: pd.DataFrame,
    closes: pd.DataFrame,
    gap_threshold: float,
    hold_days: int,
    n_permutations: int = 100,
) -> Dict:
    """Shuffle which earnings events are traded and compare."""
    print(f"    Running permutation test ({n_permutations} shuffles)...")

    # Get actual result
    actual_trades, _ = run_backtest(events, opens, closes, gap_threshold, hold_days)
    if actual_trades.empty:
        return {"actual_sharpe": 0, "p_value": 1.0}

    actual_mean = actual_trades["return"].mean()
    actual_n = len(actual_trades)

    # Permutation: randomly select same number of events
    all_events = events[events["gap_pct"].abs() >= gap_threshold].copy()
    perm_means = []

    for i in range(n_permutations):
        # Randomly sample same number of events (with replacement from pool)
        if len(all_events) <= actual_n:
            sampled = all_events.sample(n=len(all_events), replace=True, random_state=i)
        else:
            sampled = all_events.sample(n=actual_n, replace=False, random_state=i)

        # Randomly flip directions
        sampled = sampled.copy()
        flip_mask = np.random.RandomState(i + 10000).choice([True, False], size=len(sampled))
        sampled.loc[flip_mask, "direction"] = sampled.loc[flip_mask, "direction"].map(
            {"long": "short", "short": "long"}
        )

        t, _ = run_backtest(sampled, opens, closes, gap_threshold, hold_days, max_positions=MAX_POSITIONS)
        if not t.empty:
            perm_means.append(t["return"].mean())

    if not perm_means:
        return {"actual_mean": actual_mean, "p_value": 1.0}

    perm_means = np.array(perm_means)
    p_value = np.mean(perm_means >= actual_mean)

    return {
        "actual_mean": actual_mean,
        "perm_mean": np.mean(perm_means),
        "perm_std": np.std(perm_means),
        "p_value": p_value,
        "n_permutations": n_permutations,
    }


def subperiod_consistency(
    events: pd.DataFrame,
    opens: pd.DataFrame,
    closes: pd.DataFrame,
    gap_threshold: float,
    hold_days: int,
    n_blocks: int = 3,
) -> List[Dict]:
    """Split data into n_blocks time periods and test each."""
    print(f"    Sub-period consistency ({n_blocks} blocks)...")

    filtered = events[events["gap_pct"].abs() >= gap_threshold].copy()
    if filtered.empty:
        return []

    min_dt = filtered["date"].min()
    max_dt = filtered["date"].max()
    total_days = (max_dt - min_dt).days
    block_days = total_days // n_blocks

    results = []
    for i in range(n_blocks):
        block_start = min_dt + pd.Timedelta(days=i * block_days)
        block_end = min_dt + pd.Timedelta(days=(i + 1) * block_days)
        if i == n_blocks - 1:
            block_end = max_dt + pd.Timedelta(days=1)

        block_events = filtered[
            (filtered["date"] >= block_start) & (filtered["date"] < block_end)
        ]

        trades, equity = run_backtest(block_events, opens, closes, gap_threshold, hold_days)
        metrics = compute_metrics(trades, equity)
        metrics["period"] = f"{block_start.date()} to {block_end.date()}"
        metrics["block"] = i + 1
        results.append(metrics)

    return results


def survivorship_bias_check(
    events: pd.DataFrame,
    opens: pd.DataFrame,
    closes: pd.DataFrame,
    gap_threshold: float,
    hold_days: int,
    n_exclude: int = 5,
) -> Dict:
    """Exclude top N performers and retest."""
    print(f"    Survivorship bias check (excluding top {n_exclude} performers)...")

    # First run full to find top performers
    trades, _ = run_backtest(events, opens, closes, gap_threshold, hold_days)
    if trades.empty:
        return {}

    # Find top N tickers by total PnL
    ticker_pnl = trades.groupby("ticker")["pnl"].sum().sort_values(ascending=False)
    top_tickers = ticker_pnl.head(n_exclude).index.tolist()

    print(f"      Top {n_exclude} performers: {top_tickers}")
    print(f"      Their total PnL: ${ticker_pnl.head(n_exclude).sum():,.0f}")

    # Rerun excluding them
    trades_ex, equity_ex = run_backtest(
        events, opens, closes, gap_threshold, hold_days, excluded_tickers=top_tickers
    )
    metrics_ex = compute_metrics(trades_ex, equity_ex)
    metrics_ex["excluded"] = top_tickers
    metrics_ex["excluded_pnl"] = float(ticker_pnl.head(n_exclude).sum())

    return metrics_ex


# ============================================================
#  SPY BENCHMARK
# ============================================================
def spy_buyhold(closes: pd.DataFrame, start: str, end: str) -> Dict:
    """Compute SPY buy-hold metrics for comparison."""
    if "SPY" not in closes.columns:
        return {}

    spy = closes["SPY"].dropna()
    spy = spy.loc[start:end]
    if len(spy) < 30:
        return {}

    daily_ret = spy.pct_change().dropna()
    years = len(daily_ret) / 252

    total_ret = (spy.iloc[-1] / spy.iloc[0]) - 1
    cagr = (1 + total_ret) ** (1 / years) - 1

    sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(252)
    downside = daily_ret[daily_ret < 0]
    sortino = daily_ret.mean() / downside.std() * np.sqrt(252) if len(downside) > 1 else 0

    peak = np.maximum.accumulate(spy.values)
    dd = (spy.values - peak) / peak
    max_dd = np.min(dd)

    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    return {
        "sharpe": sharpe,
        "sortino": sortino,
        "cagr": cagr,
        "max_dd": max_dd,
        "calmar": calmar,
        "total_return": total_ret,
    }


# ============================================================
#  MAIN
# ============================================================
def main():
    print("=" * 70)
    print("POST-EARNINGS ANNOUNCEMENT DRIFT (PEAD) BACKTEST v1")
    print("=" * 70)
    print(f"Universe: {len(UNIVERSE)} large-cap stocks")
    print(f"Period: {START_DATE} to {END_DATE}")
    print(f"Capital: ${INITIAL_CAPITAL:,}  |  Max positions: {MAX_POSITIONS}")
    print(f"Gap thresholds: {GAP_THRESHOLDS}")
    print(f"Hold periods: {HOLD_PERIODS}")
    print()

    # ---- Phase 1: Download data ----
    print("[1/5] Downloading price data...")
    opens, closes = download_prices(UNIVERSE, START_DATE, END_DATE)

    # ---- Phase 2: Detect earnings gaps ----
    print("\n[2/5] Detecting earnings-day gaps...")
    events = detect_earnings_gaps(opens, closes, UNIVERSE)
    if events.empty:
        print("FATAL: No earnings gaps detected. Exiting.")
        return

    events.to_csv(OUTPUT_DIR / "earnings_events.csv", index=False)

    # ---- Phase 3: Run backtests for all combinations ----
    print("\n[3/5] Running backtests...")
    all_results = {}

    for gap_th in GAP_THRESHOLDS:
        for hold in HOLD_PERIODS:
            for mode in ["long_short", "long_only"]:
                long_only = mode == "long_only"
                key = f"gap{int(gap_th*100)}pct_hold{hold}d_{mode}"
                print(f"\n  --- {key} ---")

                trades, equity = run_backtest(
                    events, opens, closes, gap_th, hold, long_only=long_only
                )
                metrics = compute_metrics(trades, equity)

                if metrics["n_trades"] > 0:
                    print(f"    Trades: {metrics['n_trades']}")
                    print(f"    Win rate: {metrics['win_rate']*100:.1f}%")
                    print(f"    Avg return: {metrics['avg_return']*100:+.2f}%")
                    print(f"    Profit factor: {metrics['profit_factor']:.2f}")
                    print(f"    Sharpe: {metrics['sharpe']:.2f}")
                    print(f"    Sortino: {metrics['sortino']:.2f}")
                    print(f"    CAGR: {metrics['cagr']*100:.1f}%")
                    print(f"    Max DD: {metrics['max_dd']*100:.1f}%")
                    print(f"    Calmar: {metrics['calmar']:.2f}")
                    print(f"    Total PnL: ${metrics['total_pnl']:+,.0f}")

                    # Monthly returns
                    monthly = compute_monthly_returns(trades)
                    if not monthly.empty:
                        pos_months = (monthly["pnl"] > 0).sum()
                        total_months = len(monthly)
                        print(f"    Positive months: {pos_months}/{total_months} ({pos_months/total_months*100:.0f}%)")

                    # Save trades
                    trades.to_csv(OUTPUT_DIR / f"trades_{key}.csv", index=False)
                else:
                    print(f"    No trades generated")

                all_results[key] = metrics

    # ---- Phase 4: SPY Benchmark ----
    print("\n\n[4/5] SPY Buy-Hold Benchmark...")
    spy_metrics = spy_buyhold(closes, START_DATE, END_DATE)
    if spy_metrics:
        print(f"  SPY Sharpe: {spy_metrics['sharpe']:.2f}")
        print(f"  SPY Sortino: {spy_metrics['sortino']:.2f}")
        print(f"  SPY CAGR: {spy_metrics['cagr']*100:.1f}%")
        print(f"  SPY Max DD: {spy_metrics['max_dd']*100:.1f}%")
        print(f"  SPY Total Return: {spy_metrics['total_return']*100:.1f}%")

    # ---- Phase 5: Adversarial Testing (HC #705) ----
    print("\n\n[5/5] Adversarial Testing (HC #705)...")

    # Pick best config for adversarial testing
    best_key = None
    best_sharpe = -999
    for k, v in all_results.items():
        if v.get("n_trades", 0) >= 20 and v.get("sharpe", -999) > best_sharpe:
            best_sharpe = v["sharpe"]
            best_key = k

    adversarial = {}
    if best_key:
        print(f"\n  Best config for adversarial: {best_key} (Sharpe={best_sharpe:.2f})")

        # Parse best config
        parts = best_key.split("_")
        best_gap = int(parts[0].replace("gap", "").replace("pct", "")) / 100
        best_hold = int(parts[1].replace("hold", "").replace("d", ""))

        # 5a. Permutation test
        perm = permutation_test(events, opens, closes, best_gap, best_hold, n_permutations=100)
        adversarial["permutation"] = perm
        print(f"    Permutation p-value: {perm.get('p_value', 'N/A')}")
        if perm.get("p_value", 1.0) < 0.05:
            print(f"    ✓ SIGNIFICANT at 5% level")
        else:
            print(f"    ✗ NOT significant at 5% level")

        # 5b. Sub-period consistency
        subperiods = subperiod_consistency(events, opens, closes, best_gap, best_hold)
        adversarial["subperiods"] = subperiods
        for sp in subperiods:
            print(f"    Block {sp['block']} ({sp['period']}): "
                  f"Sharpe={sp.get('sharpe', 0):.2f}, "
                  f"WR={sp.get('win_rate', 0)*100:.0f}%, "
                  f"Trades={sp.get('n_trades', 0)}")

        # Consistency check
        sharpes = [sp.get("sharpe", 0) for sp in subperiods if sp.get("n_trades", 0) >= 5]
        if len(sharpes) >= 2:
            all_positive = all(s > 0 for s in sharpes)
            print(f"    All sub-periods positive Sharpe: {'YES' if all_positive else 'NO'}")

        # 5c. Survivorship bias
        surv = survivorship_bias_check(events, opens, closes, best_gap, best_hold)
        adversarial["survivorship"] = surv
        if surv.get("n_trades", 0) > 0:
            print(f"    Ex-top5: Sharpe={surv['sharpe']:.2f}, "
                  f"WR={surv['win_rate']*100:.0f}%, "
                  f"CAGR={surv['cagr']*100:.1f}%")
            print(f"    Excluded tickers: {surv.get('excluded', [])}")
            print(f"    Excluded PnL: ${surv.get('excluded_pnl', 0):+,.0f}")
    else:
        print("  No config with >= 20 trades found for adversarial testing")

    # ============================================================
    #  FINAL SUMMARY
    # ============================================================
    print("\n")
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    # Summary table
    print(f"\n{'Config':<35} {'Trades':>6} {'WR':>6} {'Sharpe':>7} {'Sortino':>8} {'CAGR':>7} {'MaxDD':>7} {'PF':>6} {'PnL':>10}")
    print("-" * 100)

    for k, v in sorted(all_results.items()):
        if v.get("n_trades", 0) == 0:
            continue
        print(f"{k:<35} {v['n_trades']:>6} {v['win_rate']*100:>5.1f}% {v['sharpe']:>7.2f} {v['sortino']:>8.2f} "
              f"{v['cagr']*100:>6.1f}% {v['max_dd']*100:>6.1f}% {v['profit_factor']:>6.2f} ${v['total_pnl']:>+9,.0f}")

    if spy_metrics:
        print(f"\n{'SPY Buy-Hold':<35} {'---':>6} {'---':>6} {spy_metrics['sharpe']:>7.2f} {spy_metrics['sortino']:>8.2f} "
              f"{spy_metrics['cagr']*100:>6.1f}% {spy_metrics['max_dd']*100:>6.1f}% {'---':>6} {'---':>10}")

    # Adversarial summary
    if adversarial:
        print("\n--- Adversarial Summary ---")
        if "permutation" in adversarial:
            p = adversarial["permutation"]
            print(f"Permutation test p-value: {p.get('p_value', 'N/A'):.3f} "
                  f"({'PASS' if p.get('p_value', 1) < 0.05 else 'FAIL'})")
        if "subperiods" in adversarial:
            sharpes = [sp.get("sharpe", 0) for sp in adversarial["subperiods"] if sp.get("n_trades", 0) >= 5]
            n_pos = sum(1 for s in sharpes if s > 0)
            print(f"Sub-period consistency: {n_pos}/{len(sharpes)} blocks with positive Sharpe")
        if "survivorship" in adversarial and adversarial["survivorship"].get("n_trades", 0) > 0:
            s = adversarial["survivorship"]
            print(f"Survivorship (ex-top5): Sharpe={s['sharpe']:.2f}, CAGR={s['cagr']*100:.1f}%")

    # Save all results
    output = {
        "config": {
            "universe_size": len(UNIVERSE),
            "start_date": START_DATE,
            "end_date": END_DATE,
            "initial_capital": INITIAL_CAPITAL,
            "max_positions": MAX_POSITIONS,
            "gap_thresholds": GAP_THRESHOLDS,
            "hold_periods": HOLD_PERIODS,
        },
        "n_events": len(events),
        "results": {k: {kk: float(vv) if isinstance(vv, (np.floating, np.integer)) else vv
                        for kk, vv in v.items()} for k, v in all_results.items()},
        "spy_benchmark": spy_metrics,
        "adversarial": {
            k: (v if not isinstance(v, list) else [
                {kk: float(vv) if isinstance(vv, (np.floating, np.integer)) else vv
                 for kk, vv in item.items()} for item in v
            ]) if not isinstance(v, dict) else {
                kk: float(vv) if isinstance(vv, (np.floating, np.integer)) else vv
                for kk, vv in v.items()
            }
            for k, v in adversarial.items()
        },
        "run_timestamp": datetime.now().isoformat(),
    }

    with open(OUTPUT_DIR / "results_summary.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to {OUTPUT_DIR}")
    print("DONE.")


if __name__ == "__main__":
    main()
