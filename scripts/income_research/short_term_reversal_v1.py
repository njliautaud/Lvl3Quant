#!/usr/bin/env python3
"""
Short-Term Reversal Strategy Backtest v1
=========================================
Buy stocks that dropped the most over the trailing week, sell after a bounce.
At 1-week horizons, returns tend to REVERSE -- one of the strongest anomalies
in academic finance (Jegadeesh 1990, Lehmann 1990).

8 configurations tested with full quality gates:
  - Permutation test (p<0.05, 1000 iterations)
  - R1 Regime test (green/red week SPY, Sharpe gap < 0.50)
  - Sub-period consistency (first-half vs second-half both positive)
  - Outlier robustness (trim top/bottom 5%, still profitable)
"""

import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

warnings.filterwarnings("ignore")

# ── Universe ──────────────────────────────────────────────────────────────────
UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM", "V", "MA",
    "UNH", "JNJ", "PG", "HD", "KO", "PEP", "COST", "MCD", "AVGO", "LLY",
    "ABBV", "MRK", "CVX", "XOM", "WMT", "BAC", "GS", "MS", "CRM", "NFLX",
]

START = "2014-01-01"
END = "2026-07-01"
OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/short_reversal_v1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_PERM = 1000
np.random.seed(42)


# ── Data Download ─────────────────────────────────────────────────────────────
def download_data():
    """Download adjusted close prices for universe + SPY + ^VIX."""
    tickers = UNIVERSE + ["SPY", "^VIX"]
    print(f"Downloading {len(tickers)} tickers from {START} to {END}...")

    # Download in small batches to avoid timeout
    all_close = {}
    batch_size = 5
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        print(f"  Batch {i // batch_size + 1}: {batch}")
        try:
            data = yf.download(batch, start=START, end=END, auto_adjust=True,
                               progress=False, threads=False, timeout=30)
            if isinstance(data.columns, pd.MultiIndex):
                close = data["Close"]
            else:
                close = data
            if isinstance(close, pd.Series):
                all_close[batch[0]] = close
            else:
                for col in close.columns:
                    all_close[col] = close[col]
        except Exception as e:
            print(f"    WARNING: Failed to download {batch}: {e}")

    close_df = pd.DataFrame(all_close)
    print(f"  Got {len(close_df)} trading days, {close_df.shape[1]} tickers")
    return close_df


# ── RSI Calculation ───────────────────────────────────────────────────────────
def compute_rsi(series, period=14):
    """Standard RSI."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


# ── Core Backtest Engine ─────────────────────────────────────────────────────
def run_backtest(close_df, config):
    """
    Run a single backtest configuration.

    Parameters
    ----------
    close_df : DataFrame of adjusted close prices (index=date, cols=tickers)
    config : dict with keys:
        name, n_stocks, lookback_days, hold_days,
        min_drop (optional), max_vix (optional), max_rsi (optional)

    Returns
    -------
    dict with results + quality gates
    """
    name = config["name"]
    n_stocks = config["n_stocks"]
    lookback = config["lookback_days"]
    hold = config["hold_days"]
    min_drop = config.get("min_drop", None)       # e.g., -0.03
    max_vix = config.get("max_vix", None)          # e.g., 25
    max_rsi = config.get("max_rsi", None)          # e.g., 40

    stock_cols = [c for c in close_df.columns if c not in ["SPY", "^VIX"]]
    spy = close_df["SPY"]
    vix = close_df.get("^VIX")

    # Compute trailing returns for ranking
    trailing_ret = close_df[stock_cols].pct_change(lookback)

    # Compute RSI if needed
    rsi_df = None
    if max_rsi is not None:
        rsi_df = pd.DataFrame({col: compute_rsi(close_df[col]) for col in stock_cols})

    # Get weekly rebalance dates (Fridays or last trading day of each week)
    dates = close_df.index
    week_groups = dates.to_series().dt.isocalendar()
    week_groups = week_groups[["year", "week"]].astype(str).agg("-".join, axis=1)
    # Last trading day of each week
    rebal_dates = dates.to_series().groupby(week_groups).last()
    rebal_dates = rebal_dates.sort_values().values
    # Filter to dates where we have enough lookback
    rebal_dates = [d for d in rebal_dates if d >= dates[lookback + 5]]

    period_returns = []
    period_dates = []

    for i, rebal_date in enumerate(rebal_dates):
        rebal_idx = dates.get_loc(rebal_date)

        # Check if we have enough forward days
        if rebal_idx + hold >= len(dates):
            break

        # Get trailing returns on rebalance date
        tr = trailing_ret.loc[rebal_date, stock_cols].dropna()
        if len(tr) < n_stocks:
            continue

        # VIX filter
        if max_vix is not None and vix is not None:
            try:
                current_vix = vix.loc[rebal_date]
                if pd.isna(current_vix) or current_vix >= max_vix:
                    continue
            except KeyError:
                continue

        # Rank and select bottom N
        ranked = tr.sort_values()
        candidates = ranked.head(n_stocks)

        # Min drop filter
        if min_drop is not None:
            candidates = candidates[candidates < min_drop]
            if len(candidates) == 0:
                continue

        # RSI filter
        if max_rsi is not None and rsi_df is not None:
            rsi_vals = rsi_df.loc[rebal_date, candidates.index].dropna()
            candidates = candidates.loc[candidates.index.intersection(
                rsi_vals[rsi_vals < max_rsi].index
            )]
            if len(candidates) == 0:
                continue

        # Compute forward return (hold period)
        entry_price = close_df.loc[rebal_date, candidates.index]
        exit_date = dates[rebal_idx + hold]
        exit_price = close_df.loc[exit_date, candidates.index]

        fwd_rets = (exit_price / entry_price - 1.0).dropna()
        if len(fwd_rets) == 0:
            continue

        basket_ret = fwd_rets.mean()
        period_returns.append(basket_ret)
        period_dates.append(rebal_date)

    if len(period_returns) < 10:
        return _empty_result(name)

    returns = np.array(period_returns)
    dates_arr = np.array(period_dates)

    # ── Compute metrics ───────────────────────────────────────────────────
    mean_ret = returns.mean()
    std_ret = returns.std()
    n_periods = len(returns)
    win_rate = (returns > 0).mean()

    # Annualize (approx periods per year based on hold)
    periods_per_year = 252 / hold
    ann_mean = mean_ret * periods_per_year
    ann_std = std_ret * np.sqrt(periods_per_year)
    sharpe = ann_mean / ann_std if ann_std > 0 else 0.0

    # Sortino
    downside = returns[returns < 0]
    downside_std = downside.std() * np.sqrt(periods_per_year) if len(downside) > 1 else ann_std
    sortino = ann_mean / downside_std if downside_std > 0 else 0.0

    # Profit factor
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    profit_factor = gains / losses if losses > 0 else float("inf")

    # Max drawdown (cumulative)
    cum = (1 + returns).cumprod()
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    max_dd = dd.min()

    # CAGR
    total_years = (pd.Timestamp(dates_arr[-1]) - pd.Timestamp(dates_arr[0])).days / 365.25
    total_return = cum[-1]
    cagr = (total_return ** (1 / total_years) - 1) if total_years > 0 else 0.0

    # ── Quality Gates ─────────────────────────────────────────────────────

    # 1. Permutation test
    perm_means = _permutation_test(close_df, stock_cols, config, rebal_dates, n_perm=N_PERM)
    if perm_means is not None and len(perm_means) > 0:
        p_value = (perm_means >= mean_ret).mean()
    else:
        p_value = 1.0
    perm_pass = p_value < 0.05

    # 2. R1 Regime test (prior-week SPY return)
    spy_weekly_ret = spy.pct_change(5)
    green_rets, red_rets = [], []
    for d, r in zip(dates_arr, returns):
        try:
            spy_r = spy_weekly_ret.loc[d]
            if pd.isna(spy_r):
                continue
            if spy_r >= 0:
                green_rets.append(r)
            else:
                red_rets.append(r)
        except KeyError:
            continue

    green_rets = np.array(green_rets)
    red_rets = np.array(red_rets)

    def _regime_sharpe(rets):
        if len(rets) < 5:
            return 0.0
        m = rets.mean() * periods_per_year
        s = rets.std() * np.sqrt(periods_per_year)
        return m / s if s > 0 else 0.0

    sharpe_green = _regime_sharpe(green_rets)
    sharpe_red = _regime_sharpe(red_rets)
    max_sharpe = max(abs(sharpe_green), abs(sharpe_red))
    regime_gap = abs(sharpe_green - sharpe_red) / max_sharpe if max_sharpe > 0 else 0.0
    regime_pass = regime_gap <= 0.50

    # 3. Sub-period consistency
    mid = len(returns) // 2
    first_half_mean = returns[:mid].mean()
    second_half_mean = returns[mid:].mean()
    subperiod_pass = first_half_mean > 0 and second_half_mean > 0

    # 4. Outlier robustness (trim top/bottom 5%)
    p5, p95 = np.percentile(returns, [5, 95])
    trimmed = returns[(returns >= p5) & (returns <= p95)]
    trimmed_mean = trimmed.mean() if len(trimmed) > 0 else 0.0
    outlier_pass = trimmed_mean > 0

    all_pass = perm_pass and regime_pass and subperiod_pass and outlier_pass

    result = {
        "config_name": name,
        "n_periods": int(n_periods),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "win_rate": round(win_rate, 4),
        "profit_factor": round(profit_factor, 3) if profit_factor != float("inf") else 999.0,
        "mean_return_pct": round(mean_ret * 100, 4),
        "max_dd_pct": round(max_dd * 100, 2),
        "cagr_pct": round(cagr * 100, 2),
        "quality_gates": {
            "permutation_p_value": round(p_value, 4),
            "permutation_pass": perm_pass,
            "regime_sharpe_green": round(sharpe_green, 3),
            "regime_sharpe_red": round(sharpe_red, 3),
            "regime_gap": round(regime_gap, 3),
            "regime_pass": regime_pass,
            "first_half_mean_pct": round(first_half_mean * 100, 4),
            "second_half_mean_pct": round(second_half_mean * 100, 4),
            "subperiod_pass": subperiod_pass,
            "trimmed_mean_pct": round(trimmed_mean * 100, 4),
            "outlier_pass": outlier_pass,
            "all_pass": all_pass,
        },
    }

    print(f"  {name}: Sharpe={sharpe:.3f}, WR={win_rate:.1%}, "
          f"PF={profit_factor:.2f}, CAGR={cagr*100:.2f}%, "
          f"Perm p={p_value:.4f}, Regime gap={regime_gap:.3f}, "
          f"All pass={all_pass}")

    return result


def _permutation_test(close_df, stock_cols, config, rebal_dates, n_perm=1000):
    """
    Permutation test: randomly select N stocks each week instead of bottom-N.
    Returns array of mean returns from random selection.
    """
    n_stocks = config["n_stocks"]
    hold = config["hold_days"]
    lookback = config["lookback_days"]
    min_drop = config.get("min_drop", None)
    max_vix = config.get("max_vix", None)
    max_rsi = config.get("max_rsi", None)

    dates = close_df.index
    vix = close_df.get("^VIX")
    trailing_ret = close_df[stock_cols].pct_change(lookback)

    rsi_df = None
    if max_rsi is not None:
        rsi_df = pd.DataFrame({col: compute_rsi(close_df[col]) for col in stock_cols})

    # Pre-compute valid rebalance dates and available stocks
    valid_periods = []
    for rebal_date in rebal_dates:
        rebal_idx = dates.get_loc(rebal_date)
        if rebal_idx + hold >= len(dates):
            break

        tr = trailing_ret.loc[rebal_date, stock_cols].dropna()
        if len(tr) < n_stocks:
            continue

        if max_vix is not None and vix is not None:
            try:
                current_vix = vix.loc[rebal_date]
                if pd.isna(current_vix) or current_vix >= max_vix:
                    continue
            except KeyError:
                continue

        # Get available stocks (those with valid data)
        available = tr.index.tolist()

        # For RSI filter, only keep stocks meeting criteria
        if max_rsi is not None and rsi_df is not None:
            rsi_vals = rsi_df.loc[rebal_date, available].dropna()
            available = rsi_vals[rsi_vals < max_rsi].index.tolist()

        # For min_drop, only keep stocks that dropped enough
        if min_drop is not None:
            dropped = tr[tr < min_drop]
            available = [s for s in available if s in dropped.index]

        if len(available) < 1:
            continue

        # Forward returns for all available stocks
        entry = close_df.loc[rebal_date, available]
        exit_date = dates[rebal_idx + hold]
        exit_p = close_df.loc[exit_date, available]
        fwd = (exit_p / entry - 1.0).dropna()

        if len(fwd) == 0:
            continue

        valid_periods.append(fwd)

    if len(valid_periods) < 10:
        return None

    perm_means = np.zeros(n_perm)
    actual_n = min(n_stocks, min(len(vp) for vp in valid_periods))

    for i in range(n_perm):
        period_rets = []
        for fwd in valid_periods:
            # Randomly pick n_stocks (or fewer if not enough)
            pick_n = min(actual_n, len(fwd))
            chosen = np.random.choice(len(fwd), size=pick_n, replace=False)
            period_rets.append(fwd.iloc[chosen].mean())
        perm_means[i] = np.mean(period_rets)

    return perm_means


def _empty_result(name):
    return {
        "config_name": name,
        "n_periods": 0,
        "sharpe": 0.0,
        "sortino": 0.0,
        "win_rate": 0.0,
        "profit_factor": 0.0,
        "mean_return_pct": 0.0,
        "max_dd_pct": 0.0,
        "cagr_pct": 0.0,
        "quality_gates": {
            "permutation_p_value": 1.0,
            "permutation_pass": False,
            "regime_sharpe_green": 0.0,
            "regime_sharpe_red": 0.0,
            "regime_gap": 0.0,
            "regime_pass": False,
            "first_half_mean_pct": 0.0,
            "second_half_mean_pct": 0.0,
            "subperiod_pass": False,
            "trimmed_mean_pct": 0.0,
            "outlier_pass": False,
            "all_pass": False,
        },
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 80)
    print("SHORT-TERM REVERSAL STRATEGY BACKTEST v1")
    print("=" * 80)
    print(f"Universe: {len(UNIVERSE)} large-cap stocks")
    print(f"Period: {START} to {END}")
    print(f"Permutation iterations: {N_PERM}")
    print()

    close_df = download_data()

    # Drop any stocks with too much missing data
    missing_pct = close_df[UNIVERSE].isna().mean()
    bad = missing_pct[missing_pct > 0.3].index.tolist()
    if bad:
        print(f"  WARNING: Dropping {bad} due to >30% missing data")
        for b in bad:
            if b in UNIVERSE:
                UNIVERSE.remove(b)

    # Forward-fill small gaps
    close_df = close_df.ffill(limit=5)

    configs = [
        {"name": "C1: Bottom3 by 5d, hold 5d",
         "n_stocks": 3, "lookback_days": 5, "hold_days": 5},
        {"name": "C2: Bottom5 by 5d, hold 5d",
         "n_stocks": 5, "lookback_days": 5, "hold_days": 5},
        {"name": "C3: Bottom3 by 5d, hold 10d",
         "n_stocks": 3, "lookback_days": 5, "hold_days": 10},
        {"name": "C4: Bottom5 by 5d, hold 10d",
         "n_stocks": 5, "lookback_days": 5, "hold_days": 10},
        {"name": "C5: Bottom3 by 5d (>3% drop), hold 5d",
         "n_stocks": 3, "lookback_days": 5, "hold_days": 5, "min_drop": -0.03},
        {"name": "C6: Bottom3 by 10d, hold 5d",
         "n_stocks": 3, "lookback_days": 10, "hold_days": 5},
        {"name": "C7: Bottom5 by 5d, hold 5d, VIX<25",
         "n_stocks": 5, "lookback_days": 5, "hold_days": 5, "max_vix": 25},
        {"name": "C8: Bottom3 by 5d, hold 5d, RSI<40",
         "n_stocks": 3, "lookback_days": 5, "hold_days": 5, "max_rsi": 40},
    ]

    results = []
    for cfg in configs:
        print(f"\nRunning: {cfg['name']}")
        res = run_backtest(close_df, cfg)
        results.append(res)

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Config':<45} {'Sharpe':>7} {'Sortino':>8} {'WR':>6} {'PF':>6} "
          f"{'CAGR%':>7} {'MaxDD%':>7} {'Pass':>5}")
    print("-" * 100)
    for r in results:
        qg = r["quality_gates"]
        pass_str = "YES" if qg["all_pass"] else "NO"
        print(f"{r['config_name']:<45} {r['sharpe']:>7.3f} {r['sortino']:>8.3f} "
              f"{r['win_rate']:>5.1%} {r['profit_factor']:>6.2f} "
              f"{r['cagr_pct']:>6.2f}% {r['max_dd_pct']:>6.2f}% {pass_str:>5}")

    print("\nQuality Gate Details:")
    for r in results:
        qg = r["quality_gates"]
        flags = []
        if not qg["permutation_pass"]:
            flags.append(f"Perm FAIL (p={qg['permutation_p_value']:.4f})")
        if not qg["regime_pass"]:
            flags.append(f"Regime FAIL (gap={qg['regime_gap']:.3f})")
        if not qg["subperiod_pass"]:
            flags.append(f"SubPeriod FAIL (H1={qg['first_half_mean_pct']:.4f}%, H2={qg['second_half_mean_pct']:.4f}%)")
        if not qg["outlier_pass"]:
            flags.append("Outlier FAIL")
        status = ", ".join(flags) if flags else "ALL PASS"
        print(f"  {r['config_name']}: {status}")

    # ── Save ──────────────────────────────────────────────────────────────
    out_path = OUTPUT_DIR / "backtest_report.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
