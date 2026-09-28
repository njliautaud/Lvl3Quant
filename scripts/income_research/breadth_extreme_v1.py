#!/usr/bin/env python3
"""
Breadth Extreme Mean-Reversion Strategy Backtest
=================================================
Buy SPY when market breadth drops to extreme lows, expecting mean-reversion.

Uses a sample of large-cap S&P 500 components to compute breadth
(% above 200-day MA, % above 50-day MA) as a proxy for full index breadth.

Quality gates (inline):
  1. Permutation test (p < 0.05)
  2. Regime test (R1): green vs red day Sharpe gap < 0.50
  3. Sub-period consistency: both halves profitable
  4. Outlier robustness: profitable after trimming top/bottom 5%
"""

import json
import os
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

warnings.filterwarnings("ignore")

# ── Configuration ───────────────────────────────────────────────────────────
OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/breadth_extreme_v1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = "2009-01-01"  # extra buffer for 200-day MA warmup
END_DATE = "2026-07-01"
ANALYSIS_START = "2010-01-01"  # actual analysis starts here (after MA warmup)

N_PERMUTATIONS = 1000
REGIME_GAP_THRESHOLD = 0.50
TRIM_PCT = 0.05

# Representative large-cap sample (80 tickers spanning all GICS sectors)
SAMPLE_TICKERS = [
    # Tech
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AVGO", "ADBE", "CRM", "CSCO", "ORCL",
    "INTC", "AMD", "QCOM", "TXN", "IBM",
    # Health Care
    "UNH", "JNJ", "LLY", "PFE", "ABBV", "MRK", "TMO", "ABT", "BMY", "AMGN",
    # Financials
    "JPM", "BAC", "WFC", "GS", "MS", "BLK", "SCHW", "AXP", "C", "USB",
    # Consumer Discretionary
    "AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "LOW", "TJX",
    # Consumer Staples
    "PG", "KO", "PEP", "COST", "WMT", "CL", "PM",
    # Industrials
    "CAT", "HON", "UNP", "RTX", "BA", "GE", "DE", "MMM", "LMT",
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG",
    # Utilities
    "NEE", "DUK", "SO", "D",
    # Materials
    "LIN", "APD", "SHW", "ECL",
    # Real Estate
    "PLD", "AMT", "CCI", "SPG",
    # Communication
    "GOOG", "DIS", "CMCSA", "NFLX", "T", "VZ",
]

CONFIGS = [
    {"name": "200ma_lt30_hold5",   "ma": 200, "threshold": 30, "hold": 5,  "ma2": None, "threshold2": None},
    {"name": "200ma_lt30_hold10",  "ma": 200, "threshold": 30, "hold": 10, "ma2": None, "threshold2": None},
    {"name": "200ma_lt30_hold20",  "ma": 200, "threshold": 30, "hold": 20, "ma2": None, "threshold2": None},
    {"name": "200ma_lt20_hold10",  "ma": 200, "threshold": 20, "hold": 10, "ma2": None, "threshold2": None},
    {"name": "200ma_lt20_hold20",  "ma": 200, "threshold": 20, "hold": 20, "ma2": None, "threshold2": None},
    {"name": "50ma_lt20_hold5",    "ma": 50,  "threshold": 20, "hold": 5,  "ma2": None, "threshold2": None},
    {"name": "50ma_lt20_hold10",   "ma": 50,  "threshold": 20, "hold": 10, "ma2": None, "threshold2": None},
    {"name": "50ma_lt15_hold5",    "ma": 50,  "threshold": 15, "hold": 5,  "ma2": None, "threshold2": None},
    {"name": "combined_200lt30_50lt25_hold10", "ma": 200, "threshold": 30, "hold": 10, "ma2": 50, "threshold2": 25},
]


def download_data():
    """Download SPY and sample component data."""
    print("Downloading SPY data...")
    spy = yf.download("SPY", start=START_DATE, end=END_DATE, progress=False, auto_adjust=True)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    spy_close = spy["Close"].copy()
    spy_close.index = spy_close.index.tz_localize(None)

    print(f"Downloading {len(SAMPLE_TICKERS)} component tickers...")
    # Download in batch
    raw = yf.download(SAMPLE_TICKERS, start=START_DATE, end=END_DATE, progress=False, auto_adjust=True)

    # Extract Close prices
    if isinstance(raw.columns, pd.MultiIndex):
        closes = raw["Close"].copy()
    else:
        closes = raw[["Close"]].copy()

    closes.index = closes.index.tz_localize(None)

    # Drop tickers with < 80% data coverage
    min_rows = int(len(closes) * 0.80)
    good_tickers = closes.columns[closes.notna().sum() >= min_rows]
    closes = closes[good_tickers]
    print(f"  {len(good_tickers)} tickers with sufficient data")

    return spy_close, closes


def compute_breadth(closes: pd.DataFrame) -> pd.DataFrame:
    """Compute % of stocks above N-day MA for various windows."""
    breadth = pd.DataFrame(index=closes.index)

    for window in [50, 200]:
        ma = closes.rolling(window=window, min_periods=window).mean()
        above = (closes > ma).astype(float)
        # pct of non-NaN stocks that are above MA
        n_valid = closes.notna().astype(float)
        pct_above = (above.sum(axis=1) / n_valid.sum(axis=1)) * 100
        breadth[f"pct_above_{window}ma"] = pct_above

    return breadth


def generate_signals(breadth: pd.DataFrame, config: dict) -> pd.Series:
    """Generate entry signals based on breadth thresholds."""
    col = f"pct_above_{config['ma']}ma"
    signal = breadth[col] < config["threshold"]

    if config["ma2"] is not None:
        col2 = f"pct_above_{config['ma2']}ma"
        signal2 = breadth[col2] < config["threshold2"]
        signal = signal & signal2

    return signal


def compute_trades(signal: pd.Series, spy_close: pd.Series, hold_days: int,
                   analysis_start: str) -> pd.DataFrame:
    """Convert signals to non-overlapping trades with forward returns."""
    # Align indices
    common_idx = signal.index.intersection(spy_close.index)
    signal = signal.loc[common_idx]
    spy = spy_close.loc[common_idx]

    # Filter to analysis period
    mask = signal.index >= pd.Timestamp(analysis_start)
    signal = signal[mask]
    spy = spy[spy.index >= pd.Timestamp(analysis_start)]

    entries = signal[signal].index.tolist()
    if not entries:
        return pd.DataFrame()

    trades = []
    last_exit = pd.Timestamp("1900-01-01")

    for entry_date in entries:
        if entry_date <= last_exit:
            continue  # skip overlapping

        # Find exit date (hold_days trading days later)
        entry_loc = spy.index.get_loc(entry_date)
        exit_loc = min(entry_loc + hold_days, len(spy) - 1)
        exit_date = spy.index[exit_loc]

        entry_price = spy.iloc[entry_loc]
        exit_price = spy.iloc[exit_loc]
        ret = (exit_price / entry_price) - 1.0

        # Prior-day SPY return for regime classification
        if entry_loc > 0:
            prior_ret = (spy.iloc[entry_loc] / spy.iloc[entry_loc - 1]) - 1.0
        else:
            prior_ret = 0.0

        trades.append({
            "entry_date": entry_date,
            "exit_date": exit_date,
            "entry_price": float(entry_price),
            "exit_price": float(exit_price),
            "return": float(ret),
            "prior_day_ret": float(prior_ret),
        })
        last_exit = exit_date

    return pd.DataFrame(trades)


def sharpe_from_returns(returns: np.ndarray, annualize: float = 252.0) -> float:
    """Annualized Sharpe ratio from trade returns."""
    if len(returns) < 2 or np.std(returns) == 0:
        return 0.0
    return float(np.mean(returns) / np.std(returns) * np.sqrt(annualize / max(np.mean([1]), 1)))


def sortino_from_returns(returns: np.ndarray) -> float:
    """Sortino ratio from trade returns."""
    if len(returns) < 2:
        return 0.0
    downside = returns[returns < 0]
    if len(downside) == 0 or np.std(downside) == 0:
        return float("inf") if np.mean(returns) > 0 else 0.0
    return float(np.mean(returns) / np.std(downside) * np.sqrt(252))


def profit_factor(returns: np.ndarray) -> float:
    """Gross profits / gross losses."""
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def max_drawdown(returns: np.ndarray) -> float:
    """Max drawdown from cumulative returns."""
    cum = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    return float(dd.min()) * 100  # as percentage


# ── Quality Gates ───────────────────────────────────────────────────────────

def permutation_test(trade_returns: np.ndarray, spy_close: pd.Series,
                     hold_days: int, n_perms: int = N_PERMUTATIONS) -> float:
    """
    Compare real mean return vs random-entry distribution.
    Sample random start dates, compute hold_days forward returns, repeat n_perms times.
    Returns p-value (fraction of random means >= real mean).
    """
    real_mean = np.mean(trade_returns)
    n_trades = len(trade_returns)

    spy_arr = spy_close.values
    valid_start_max = len(spy_arr) - hold_days - 1

    if valid_start_max < n_trades:
        return 1.0  # not enough data

    rng = np.random.default_rng(42)
    count_ge = 0

    for _ in range(n_perms):
        starts = rng.integers(0, valid_start_max, size=n_trades)
        rand_rets = (spy_arr[starts + hold_days] / spy_arr[starts]) - 1.0
        if np.mean(rand_rets) >= real_mean:
            count_ge += 1

    return count_ge / n_perms


def regime_test(trades_df: pd.DataFrame) -> float:
    """
    R1 regime gap test. Green = prior day SPY up, Red = prior day SPY down.
    Returns gap ratio. REJECT if > 0.50.
    """
    green = trades_df[trades_df["prior_day_ret"] >= 0]["return"].values
    red = trades_df[trades_df["prior_day_ret"] < 0]["return"].values

    if len(green) < 3 or len(red) < 3:
        return 0.0  # insufficient data, pass by default

    sharpe_g = sharpe_from_returns(green)
    sharpe_r = sharpe_from_returns(red)

    denom = max(abs(sharpe_g), abs(sharpe_r))
    if denom == 0:
        return 0.0
    return abs(sharpe_g - sharpe_r) / denom


def sub_period_test(trades_df: pd.DataFrame) -> bool:
    """Both halves of trades (by date) must have positive mean return."""
    if len(trades_df) < 4:
        return True  # not enough to split meaningfully

    mid = len(trades_df) // 2
    first_half = trades_df.iloc[:mid]["return"].mean()
    second_half = trades_df.iloc[mid:]["return"].mean()
    return first_half > 0 and second_half > 0


def outlier_robustness_test(returns: np.ndarray) -> bool:
    """After removing top/bottom 5% of returns, mean must still be positive."""
    if len(returns) < 10:
        return np.mean(returns) > 0

    lower = np.percentile(returns, TRIM_PCT * 100)
    upper = np.percentile(returns, (1 - TRIM_PCT) * 100)
    trimmed = returns[(returns >= lower) & (returns <= upper)]

    if len(trimmed) == 0:
        return False
    return float(np.mean(trimmed)) > 0


# ── Main ────────────────────────────────────────────────────────────────────

def run_backtest():
    spy_close, closes = download_data()
    breadth = compute_breadth(closes)

    # Filter SPY to analysis period for permutation baseline
    spy_analysis = spy_close[spy_close.index >= pd.Timestamp(ANALYSIS_START)]

    print(f"\nBreadth data range: {breadth.index[0].date()} to {breadth.index[-1].date()}")
    print(f"SPY data points: {len(spy_analysis)}")
    print(f"Breadth stats (analysis period):")
    for col in breadth.columns:
        b = breadth.loc[breadth.index >= pd.Timestamp(ANALYSIS_START), col]
        print(f"  {col}: mean={b.mean():.1f}%, min={b.min():.1f}%, "
              f"p10={b.quantile(0.10):.1f}%, p25={b.quantile(0.25):.1f}%")

    results = []

    for cfg in CONFIGS:
        print(f"\n{'='*70}")
        print(f"Config: {cfg['name']}")
        print(f"  MA={cfg['ma']}, threshold<{cfg['threshold']}%, hold={cfg['hold']}d", end="")
        if cfg["ma2"]:
            print(f" + MA2={cfg['ma2']}, threshold2<{cfg['threshold2']}%", end="")
        print()

        signal = generate_signals(breadth, cfg)
        trades_df = compute_trades(signal, spy_close, cfg["hold"], ANALYSIS_START)

        if len(trades_df) < 3:
            print(f"  SKIP: only {len(trades_df)} trades")
            results.append({
                "config_name": cfg["name"],
                "n_trades": len(trades_df),
                "sharpe": 0, "sortino": 0, "win_rate": 0,
                "profit_factor": 0, "mean_return_pct": 0, "max_dd_pct": 0,
                "quality_gates": {
                    "permutation_p": 1.0, "regime_gap": 0.0,
                    "sub_period_pass": False, "outlier_robust": False, "ALL_PASS": False,
                },
            })
            continue

        rets = trades_df["return"].values
        n = len(rets)
        sr = sharpe_from_returns(rets)
        so = sortino_from_returns(rets)
        wr = float((rets > 0).sum() / n)
        pf = profit_factor(rets)
        mean_ret = float(np.mean(rets)) * 100
        mdd = max_drawdown(rets)

        print(f"  Trades: {n}")
        print(f"  Mean return: {mean_ret:+.2f}%")
        print(f"  Win rate: {wr:.1%}")
        print(f"  Sharpe: {sr:.2f}  Sortino: {so:.2f}  PF: {pf:.2f}")
        print(f"  Max DD: {mdd:.2f}%")

        # Quality gates
        print("  Quality gates:")

        perm_p = permutation_test(rets, spy_analysis, cfg["hold"])
        perm_pass = perm_p < 0.05
        print(f"    Permutation p-value: {perm_p:.3f} {'PASS' if perm_pass else 'FAIL'}")

        rgap = regime_test(trades_df)
        regime_pass = rgap <= REGIME_GAP_THRESHOLD
        print(f"    Regime gap: {rgap:.3f} {'PASS' if regime_pass else 'FAIL'}")

        sp_pass = sub_period_test(trades_df)
        print(f"    Sub-period consistency: {'PASS' if sp_pass else 'FAIL'}")

        ol_pass = outlier_robustness_test(rets)
        print(f"    Outlier robustness: {'PASS' if ol_pass else 'FAIL'}")

        all_pass = perm_pass and regime_pass and sp_pass and ol_pass
        print(f"    >>> ALL PASS: {all_pass}")

        # Trade date range
        print(f"  Date range: {trades_df['entry_date'].iloc[0].date()} to "
              f"{trades_df['entry_date'].iloc[-1].date()}")

        results.append({
            "config_name": cfg["name"],
            "n_trades": int(n),
            "sharpe": round(sr, 4),
            "sortino": round(so, 4),
            "win_rate": round(wr, 4),
            "profit_factor": round(pf, 4),
            "mean_return_pct": round(mean_ret, 4),
            "max_dd_pct": round(mdd, 4),
            "quality_gates": {
                "permutation_p": round(perm_p, 4),
                "regime_gap": round(rgap, 4),
                "sub_period_pass": sp_pass,
                "outlier_robust": ol_pass,
                "ALL_PASS": all_pass,
            },
        })

    # Save results
    report = {
        "strategy": "breadth_extreme_mean_reversion",
        "run_date": datetime.now().isoformat(),
        "analysis_period": f"{ANALYSIS_START} to {END_DATE}",
        "breadth_proxy": f"{len(SAMPLE_TICKERS)} large-cap S&P 500 components",
        "permutation_iterations": N_PERMUTATIONS,
        "configs": results,
    }

    out_path = OUTPUT_DIR / "backtest_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    # Summary table
    print(f"\n{'='*90}")
    print(f"{'Config':<42} {'N':>4} {'Mean%':>7} {'WR':>6} {'Sharpe':>7} {'PF':>6} {'Perm_p':>7} {'PASS':>5}")
    print(f"{'-'*90}")
    for r in results:
        qg = r["quality_gates"]
        print(f"{r['config_name']:<42} {r['n_trades']:>4} {r['mean_return_pct']:>+7.2f} "
              f"{r['win_rate']:>6.1%} {r['sharpe']:>7.2f} {r['profit_factor']:>6.2f} "
              f"{qg['permutation_p']:>7.3f} {'YES' if qg['ALL_PASS'] else 'NO':>5}")

    passing = [r for r in results if r["quality_gates"]["ALL_PASS"]]
    print(f"\n{len(passing)}/{len(results)} configs pass ALL quality gates.")

    return report


if __name__ == "__main__":
    run_backtest()
