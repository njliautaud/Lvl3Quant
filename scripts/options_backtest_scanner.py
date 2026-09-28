#!/usr/bin/env python3
"""
Options Signal Backtest Scanner
===============================
Backtests technical entry signals for single-leg options on sector ETFs.
Walk-forward validated (80/20 train/test split). No lookahead bias.

Usage:
    python options_backtest_scanner.py          # Run full backtest + save results

    # Import scanner for live use:
    from options_backtest_scanner import live_scan
    results = live_scan()  # Returns scored ETFs with active signals

Author: Claude Opus 4.6 for Lvl3Quant
Date: 2026-09-22
"""

import json
import warnings
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
from tabulate import tabulate

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================================
# CONFIGURATION
# ============================================================================

ETFS = [
    "XLF", "XLE", "XLU", "XLP", "XLK", "XLC", "XLI", "XLB",
    "XLV", "XLRE", "XLY", "IWM", "QQQ", "SPY",
]

LOOKBACK_YEARS = 2
TRAIN_FRAC = 0.80  # Walk-forward: train on first 80%, test on last 20%
FORWARD_WINDOWS = [1, 3, 5, 10]  # Days to measure forward returns
MIN_OBSERVATIONS = 5  # Minimum signal fires to consider statistically meaningful
CONFIDENCE_LEVEL = 0.95

OUTPUT_PATH = Path("/home/jupiter/Lvl3Quant/data/signal_backtest_results.json")


# ============================================================================
# TECHNICAL INDICATOR COMPUTATION
# ============================================================================

def compute_rsi(series: pd.Series, period: int) -> pd.Series:
    """Wilder's RSI computation."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD line, signal line, and histogram."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_bollinger(close: pd.Series, period: int = 20, num_std: float = 2.0):
    """Bollinger Bands."""
    sma = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    return upper, sma, lower


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range."""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    return atr


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all technical indicators. No lookahead -- all use past data only."""
    df = df.copy()

    # RSI
    df["rsi_5"] = compute_rsi(df["Close"], 5)
    df["rsi_14"] = compute_rsi(df["Close"], 14)

    # MACD
    df["macd_line"], df["macd_signal"], df["macd_hist"] = compute_macd(df["Close"])

    # Bollinger Bands
    df["bb_upper"], df["bb_mid"], df["bb_lower"] = compute_bollinger(df["Close"])

    # ATR
    df["atr_14"] = compute_atr(df["High"], df["Low"], df["Close"], 14)

    # 20-day SMA
    df["sma_20"] = df["Close"].rolling(window=20).mean()

    # Volume ratio (today vs 20d avg) -- uses today's volume (available at close)
    df["vol_sma_20"] = df["Volume"].rolling(window=20).mean()
    df["vol_ratio"] = df["Volume"] / df["vol_sma_20"].replace(0, np.nan)

    # Daily return
    df["daily_return"] = df["Close"].pct_change()

    # Green/red day
    df["is_green"] = df["Close"] > df["Open"]

    # Consecutive red days (lookback, no lookahead)
    red_days = (~df["is_green"]).astype(int)
    # Count consecutive red days ending at previous day
    consec_red = pd.Series(0, index=df.index, dtype=int)
    for i in range(1, len(df)):
        if red_days.iloc[i - 1] == 1:
            consec_red.iloc[i] = consec_red.iloc[i - 1] + 1
        else:
            consec_red.iloc[i] = 0
    df["consec_red"] = consec_red

    # Consecutive green days (for put signals)
    green_days = df["is_green"].astype(int)
    consec_green = pd.Series(0, index=df.index, dtype=int)
    for i in range(1, len(df)):
        if green_days.iloc[i - 1] == 1:
            consec_green.iloc[i] = consec_green.iloc[i - 1] + 1
        else:
            consec_green.iloc[i] = 0
    df["consec_green"] = consec_green

    # MACD histogram sign change tracking (consecutive negative days before today)
    macd_neg = (df["macd_hist"] < 0).astype(int)
    consec_macd_neg = pd.Series(0, index=df.index, dtype=int)
    for i in range(1, len(df)):
        if macd_neg.iloc[i - 1] == 1:
            consec_macd_neg.iloc[i] = consec_macd_neg.iloc[i - 1] + 1
        else:
            consec_macd_neg.iloc[i] = 0
    df["consec_macd_neg"] = consec_macd_neg

    # Consecutive positive MACD histogram (for put signals)
    macd_pos = (df["macd_hist"] > 0).astype(int)
    consec_macd_pos = pd.Series(0, index=df.index, dtype=int)
    for i in range(1, len(df)):
        if macd_pos.iloc[i - 1] == 1:
            consec_macd_pos.iloc[i] = consec_macd_pos.iloc[i - 1] + 1
        else:
            consec_macd_pos.iloc[i] = 0
    df["consec_macd_pos"] = consec_macd_pos

    # Forward returns (these are LABELS, not features -- used only for evaluation)
    for w in FORWARD_WINDOWS:
        df[f"fwd_ret_{w}d"] = df["Close"].shift(-w) / df["Close"] - 1

    return df


# ============================================================================
# SIGNAL DEFINITIONS
# ============================================================================

def signal_call_rsi_oversold(df: pd.DataFrame) -> pd.Series:
    """CALL: RSI(14) < 30 -- deeply oversold bounce."""
    return df["rsi_14"] < 30


def signal_call_rsi_momentum_shift(df: pd.DataFrame) -> pd.Series:
    """CALL: RSI(14) < 35 AND RSI(5) crossing above RSI(14).
    Crossing = RSI(5) > RSI(14) today AND RSI(5) <= RSI(14) yesterday.
    """
    rsi5_above = df["rsi_5"] > df["rsi_14"]
    rsi5_was_below = df["rsi_5"].shift(1) <= df["rsi_14"].shift(1)
    return (df["rsi_14"] < 35) & rsi5_above & rsi5_was_below


def signal_call_bb_rsi(df: pd.DataFrame) -> pd.Series:
    """CALL: Price touches lower BB AND RSI(14) < 40."""
    return (df["Low"] <= df["bb_lower"]) & (df["rsi_14"] < 40)


def signal_call_macd_turn(df: pd.DataFrame) -> pd.Series:
    """CALL: MACD histogram turns positive after being negative >= 3 consecutive days."""
    return (df["macd_hist"] > 0) & (df["consec_macd_neg"] >= 3)


def signal_call_volume_reversal(df: pd.DataFrame) -> pd.Series:
    """CALL: Volume spike (>1.5x 20d avg) on green day after >= 3 red days."""
    return (df["vol_ratio"] > 1.5) & df["is_green"] & (df["consec_red"] >= 3)


def signal_put_rsi_overbought(df: pd.DataFrame) -> pd.Series:
    """PUT: RSI(14) > 70 -- deeply overbought."""
    return df["rsi_14"] > 70


def signal_put_rsi_momentum_shift(df: pd.DataFrame) -> pd.Series:
    """PUT: RSI(14) > 65 AND RSI(5) crossing below RSI(14)."""
    rsi5_below = df["rsi_5"] < df["rsi_14"]
    rsi5_was_above = df["rsi_5"].shift(1) >= df["rsi_14"].shift(1)
    return (df["rsi_14"] > 65) & rsi5_below & rsi5_was_above


def signal_put_bb_rsi(df: pd.DataFrame) -> pd.Series:
    """PUT: Price touches upper BB AND RSI(14) > 60."""
    return (df["High"] >= df["bb_upper"]) & (df["rsi_14"] > 60)


def signal_put_macd_turn(df: pd.DataFrame) -> pd.Series:
    """PUT: MACD histogram turns negative after being positive >= 3 consecutive days."""
    return (df["macd_hist"] < 0) & (df["consec_macd_pos"] >= 3)


def signal_put_volume_reversal(df: pd.DataFrame) -> pd.Series:
    """PUT: Volume spike (>1.5x 20d avg) on red day after >= 3 green days."""
    return (df["vol_ratio"] > 1.5) & (~df["is_green"]) & (df["consec_green"] >= 3)


CALL_SIGNALS = {
    "call_rsi_oversold":       signal_call_rsi_oversold,
    "call_rsi_momentum_shift": signal_call_rsi_momentum_shift,
    "call_bb_rsi":             signal_call_bb_rsi,
    "call_macd_turn":          signal_call_macd_turn,
    "call_volume_reversal":    signal_call_volume_reversal,
}

PUT_SIGNALS = {
    "put_rsi_overbought":      signal_put_rsi_overbought,
    "put_rsi_momentum_shift":  signal_put_rsi_momentum_shift,
    "put_bb_rsi":              signal_put_bb_rsi,
    "put_macd_turn":           signal_put_macd_turn,
    "put_volume_reversal":     signal_put_volume_reversal,
}

ALL_SIGNALS = {**CALL_SIGNALS, **PUT_SIGNALS}


# ============================================================================
# DATA DOWNLOAD
# ============================================================================

def download_data(tickers: List[str], years: int = 2) -> Dict[str, pd.DataFrame]:
    """Download OHLCV data from yfinance. Returns dict of {ticker: DataFrame}."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365 + 30)  # Extra buffer for indicator warmup

    print(f"Downloading {len(tickers)} ETFs from {start_date.date()} to {end_date.date()}...")

    data = {}
    failed = []
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
            if df is not None and len(df) > 60:
                # Flatten MultiIndex columns if present
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                data[ticker] = df
                print(f"  {ticker}: {len(df)} days")
            else:
                failed.append(ticker)
                print(f"  {ticker}: INSUFFICIENT DATA ({len(df) if df is not None else 0} days)")
        except Exception as e:
            failed.append(ticker)
            print(f"  {ticker}: FAILED ({e})")

    if failed:
        print(f"\nWarning: Failed to download: {failed}")

    return data


# ============================================================================
# BACKTEST ENGINE
# ============================================================================

def evaluate_signal(
    mask: pd.Series,
    df: pd.DataFrame,
    direction: str,  # "call" or "put"
    split_idx: int,
) -> Dict:
    """
    Evaluate a signal's performance.

    For CALL signals, we measure positive forward returns.
    For PUT signals, we measure negative forward returns (we profit from drops).

    Args:
        mask: Boolean series where True = signal fires
        df: DataFrame with indicators and forward returns
        direction: "call" or "put"
        split_idx: Index position separating train from test

    Returns:
        Dict with performance stats for train and test periods.
    """
    # Direction multiplier: calls profit from up, puts from down
    mult = 1.0 if direction == "call" else -1.0

    results = {}
    for period_name, start, end in [
        ("train", 0, split_idx),
        ("test", split_idx, len(df)),
    ]:
        period_mask = mask.copy()
        period_mask.iloc[:start] = False
        period_mask.iloc[end:] = False

        n_signals = period_mask.sum()

        if n_signals < MIN_OBSERVATIONS:
            results[period_name] = {
                "n_signals": int(n_signals),
                "insufficient_data": True,
            }
            continue

        period_results = {"n_signals": int(n_signals), "insufficient_data": False}

        for w in FORWARD_WINDOWS:
            col = f"fwd_ret_{w}d"
            returns = df.loc[period_mask, col].dropna() * mult

            if len(returns) < MIN_OBSERVATIONS:
                period_results[f"{w}d"] = {"n": int(len(returns)), "insufficient_data": True}
                continue

            n = len(returns)
            mean_ret = float(returns.mean())
            std_ret = float(returns.std())
            win_rate = float((returns > 0).mean())

            # Sharpe (annualized, assuming daily signals)
            sharpe = (mean_ret / std_ret * np.sqrt(252 / w)) if std_ret > 0 else 0.0

            # Confidence interval on mean return
            se = std_ret / np.sqrt(n)
            t_crit = stats.t.ppf((1 + CONFIDENCE_LEVEL) / 2, df=n - 1)
            ci_lower = mean_ret - t_crit * se
            ci_upper = mean_ret + t_crit * se

            # Edge quality = (win_rate - 0.5) * avg_return * sqrt(n)
            # Positive when signal has consistent directional edge
            edge_quality = (win_rate - 0.5) * mean_ret * np.sqrt(n)

            # T-test: is mean return significantly > 0?
            t_stat, p_value = stats.ttest_1samp(returns, 0)

            # Profit factor
            gross_profit = float(returns[returns > 0].sum()) if (returns > 0).any() else 0.0
            gross_loss = float(abs(returns[returns < 0].sum())) if (returns < 0).any() else 0.0001
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

            # Max drawdown of cumulative signal returns
            cum = (1 + returns).cumprod()
            running_max = cum.cummax()
            dd = (cum / running_max - 1)
            max_dd = float(dd.min())

            period_results[f"{w}d"] = {
                "n": int(n),
                "mean_return_pct": round(mean_ret * 100, 4),
                "std_return_pct": round(std_ret * 100, 4),
                "win_rate": round(win_rate, 4),
                "sharpe": round(sharpe, 3),
                "profit_factor": round(profit_factor, 3),
                "edge_quality": round(edge_quality, 6),
                "ci_lower_pct": round(ci_lower * 100, 4),
                "ci_upper_pct": round(ci_upper * 100, 4),
                "t_stat": round(float(t_stat), 3),
                "p_value": round(float(p_value), 4),
                "max_drawdown_pct": round(max_dd * 100, 4),
                "insufficient_data": False,
            }

        results[period_name] = period_results

    return results


def run_backtest(data: Dict[str, pd.DataFrame]) -> Dict:
    """
    Run the full backtest across all ETFs and signals.
    Walk-forward: train on first 80% of data, test on last 20%.
    """
    all_results = {}

    for ticker, raw_df in data.items():
        print(f"\n{'='*60}")
        print(f"Backtesting {ticker} ({len(raw_df)} days)")
        print(f"{'='*60}")

        df = compute_indicators(raw_df)

        # Drop warmup period (first 30 rows have NaN indicators)
        df = df.iloc[30:].copy()
        df = df.reset_index(drop=True)

        # Walk-forward split
        split_idx = int(len(df) * TRAIN_FRAC)
        train_end_date = df.index[split_idx] if isinstance(df.index, pd.DatetimeIndex) else f"row_{split_idx}"

        print(f"  Train: rows 0-{split_idx} | Test: rows {split_idx}-{len(df)}")

        ticker_results = {}

        for signal_name, signal_fn in ALL_SIGNALS.items():
            direction = "call" if signal_name.startswith("call_") else "put"

            try:
                mask = signal_fn(df)
                # Ensure no NaN in mask
                mask = mask.fillna(False)

                result = evaluate_signal(mask, df, direction, split_idx)
                ticker_results[signal_name] = result

                # Print summary for test period
                test = result.get("test", {})
                n = test.get("n_signals", 0)
                if not test.get("insufficient_data", True) and "5d" in test:
                    s5 = test["5d"]
                    if not s5.get("insufficient_data", True):
                        print(f"  {signal_name:30s} | n={n:3d} | "
                              f"WR={s5['win_rate']:.1%} | "
                              f"Ret={s5['mean_return_pct']:+.2f}% | "
                              f"Sharpe={s5['sharpe']:+.2f} | "
                              f"p={s5['p_value']:.3f}")
                    else:
                        print(f"  {signal_name:30s} | n={n:3d} | 5d: insufficient data")
                else:
                    print(f"  {signal_name:30s} | n={n:3d} | insufficient test data")
            except Exception as e:
                print(f"  {signal_name:30s} | ERROR: {e}")
                ticker_results[signal_name] = {"error": str(e)}

        all_results[ticker] = ticker_results

    return all_results


# ============================================================================
# RANKING AND ANALYSIS
# ============================================================================

def rank_signals(results: Dict) -> pd.DataFrame:
    """
    Rank all signal-ETF combinations by edge quality on TEST data.
    Only includes signals with sufficient test data.
    """
    rows = []

    for ticker, signals in results.items():
        for signal_name, signal_data in signals.items():
            if "error" in signal_data:
                continue

            test = signal_data.get("test", {})
            train = signal_data.get("train", {})

            if test.get("insufficient_data", True):
                continue

            direction = "CALL" if signal_name.startswith("call_") else "PUT"

            for w in FORWARD_WINDOWS:
                key = f"{w}d"
                test_stats = test.get(key, {})
                train_stats = train.get(key, {})

                if test_stats.get("insufficient_data", True):
                    continue
                if train_stats.get("insufficient_data", True):
                    continue

                # Check train/test consistency (avoid overfitting)
                train_sharpe = train_stats.get("sharpe", 0)
                test_sharpe = test_stats.get("sharpe", 0)

                # Both must be same sign for the edge to be "real"
                consistent = (train_sharpe > 0 and test_sharpe > 0) or \
                             (train_sharpe < 0 and test_sharpe < 0)

                rows.append({
                    "ticker": ticker,
                    "signal": signal_name,
                    "direction": direction,
                    "horizon": f"{w}d",
                    "test_n": test_stats["n"],
                    "test_wr": test_stats["win_rate"],
                    "test_ret_pct": test_stats["mean_return_pct"],
                    "test_sharpe": test_sharpe,
                    "test_pf": test_stats["profit_factor"],
                    "test_edge_quality": test_stats["edge_quality"],
                    "test_p_value": test_stats["p_value"],
                    "test_ci_lower": test_stats["ci_lower_pct"],
                    "test_ci_upper": test_stats["ci_upper_pct"],
                    "train_sharpe": train_sharpe,
                    "train_wr": train_stats.get("win_rate", 0),
                    "train_ret_pct": train_stats.get("mean_return_pct", 0),
                    "train_test_consistent": consistent,
                    "max_dd_pct": test_stats["max_drawdown_pct"],
                })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values("test_edge_quality", ascending=False)
    return df


def compute_signal_aggregate_stats(rankings: pd.DataFrame) -> pd.DataFrame:
    """Aggregate stats per signal across all ETFs for a given horizon."""
    if rankings.empty:
        return pd.DataFrame()

    agg = rankings.groupby(["signal", "direction", "horizon"]).agg(
        total_n=("test_n", "sum"),
        mean_wr=("test_wr", "mean"),
        mean_ret=("test_ret_pct", "mean"),
        mean_sharpe=("test_sharpe", "mean"),
        mean_edge=("test_edge_quality", "mean"),
        pct_consistent=("train_test_consistent", "mean"),
        n_etfs=("ticker", "nunique"),
        median_p=("test_p_value", "median"),
    ).reset_index()

    agg = agg.sort_values("mean_edge", ascending=False)
    return agg


def compute_recommended_thresholds(rankings: pd.DataFrame) -> Dict:
    """
    Based on backtest results, compute recommended minimum thresholds
    for each signal to filter for higher-edge occurrences.
    """
    thresholds = {}

    # For each signal, find the best-performing horizon
    for signal in rankings["signal"].unique():
        sig_data = rankings[rankings["signal"] == signal]

        # Only consider consistent signals
        consistent = sig_data[sig_data["train_test_consistent"]]

        if consistent.empty:
            thresholds[signal] = {
                "status": "NO_CONSISTENT_EDGE",
                "recommendation": "Avoid -- train/test performance diverges",
            }
            continue

        # Best horizon by edge quality
        best = consistent.loc[consistent["test_edge_quality"].idxmax()]

        thresholds[signal] = {
            "status": "TRADEABLE" if best["test_p_value"] < 0.10 else "MARGINAL",
            "best_horizon": best["horizon"],
            "best_etf": best["ticker"],
            "test_win_rate": round(float(best["test_wr"]), 3),
            "test_sharpe": round(float(best["test_sharpe"]), 3),
            "test_return_pct": round(float(best["test_ret_pct"]), 4),
            "test_p_value": round(float(best["test_p_value"]), 4),
            "recommendation": (
                f"Use on {best['ticker']} at {best['horizon']} horizon. "
                f"WR={best['test_wr']:.1%}, Sharpe={best['test_sharpe']:.2f}. "
                f"{'Statistically significant.' if best['test_p_value'] < 0.05 else 'Marginally significant -- use with other confluence.' if best['test_p_value'] < 0.10 else 'NOT significant -- use only as confluence with other signals.'}"
            ),
        }

    return thresholds


# ============================================================================
# LIVE SCANNER
# ============================================================================

def live_scan(tickers: Optional[List[str]] = None,
              historical_results_path: Optional[str] = None) -> Dict:
    """
    Live scanner: downloads current data, checks which signals are firing NOW,
    and scores each ETF based on backtested edge quality.

    Args:
        tickers: List of tickers to scan (default: all 14 ETFs)
        historical_results_path: Path to saved backtest results JSON

    Returns:
        Dict with scored ETFs and active signals
    """
    if tickers is None:
        tickers = ETFS

    # Load historical edge weights
    results_path = historical_results_path or str(OUTPUT_PATH)
    edge_weights = {}
    try:
        with open(results_path) as f:
            historical = json.load(f)
            if "rankings_5d" in historical:
                for entry in historical["rankings_5d"]:
                    key = f"{entry['ticker']}_{entry['signal']}"
                    edge_weights[key] = entry.get("test_edge_quality", 0)
    except FileNotFoundError:
        print("Warning: No historical results found. Run full backtest first.")
        print("  Signals will fire but won't have edge-quality weights.")

    # Download recent data (need ~60 days for indicator warmup)
    scan_results = {}

    for ticker in tickers:
        try:
            df = yf.download(ticker, period="120d", progress=False, auto_adjust=True)
            if df is None or len(df) < 40:
                scan_results[ticker] = {"error": "Insufficient data"}
                continue

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = compute_indicators(df)

            # Check each signal on the LATEST bar
            active_signals = []
            total_edge_score = 0.0

            for signal_name, signal_fn in ALL_SIGNALS.items():
                mask = signal_fn(df).fillna(False)

                if mask.iloc[-1]:
                    direction = "CALL" if signal_name.startswith("call_") else "PUT"
                    edge_key = f"{ticker}_{signal_name}"
                    edge_wt = edge_weights.get(edge_key, 0)

                    active_signals.append({
                        "signal": signal_name,
                        "direction": direction,
                        "edge_weight": round(edge_wt, 6),
                    })
                    total_edge_score += edge_wt

            # Current indicator values for context
            last = df.iloc[-1]
            current_indicators = {
                "close": round(float(last["Close"]), 2),
                "rsi_5": round(float(last["rsi_5"]), 1) if pd.notna(last["rsi_5"]) else None,
                "rsi_14": round(float(last["rsi_14"]), 1) if pd.notna(last["rsi_14"]) else None,
                "macd_hist": round(float(last["macd_hist"]), 4) if pd.notna(last["macd_hist"]) else None,
                "bb_position": round(
                    float((last["Close"] - last["bb_lower"]) / (last["bb_upper"] - last["bb_lower"])), 3
                ) if pd.notna(last["bb_upper"]) and (last["bb_upper"] - last["bb_lower"]) > 0 else None,
                "vol_ratio": round(float(last["vol_ratio"]), 2) if pd.notna(last["vol_ratio"]) else None,
                "atr_14": round(float(last["atr_14"]), 3) if pd.notna(last["atr_14"]) else None,
            }

            scan_results[ticker] = {
                "active_signals": active_signals,
                "n_active": len(active_signals),
                "total_edge_score": round(total_edge_score, 6),
                "indicators": current_indicators,
            }
        except Exception as e:
            scan_results[ticker] = {"error": str(e)}

    # Sort by edge score
    scored_tickers = [
        (t, r) for t, r in scan_results.items()
        if "error" not in r and r["n_active"] > 0
    ]
    scored_tickers.sort(key=lambda x: x[1]["total_edge_score"], reverse=True)

    return {
        "scan_time": datetime.now().isoformat(),
        "active_opportunities": [
            {"ticker": t, **r} for t, r in scored_tickers
        ],
        "all_tickers": scan_results,
    }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("=" * 70)
    print("OPTIONS SIGNAL BACKTEST SCANNER")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"ETFs: {', '.join(ETFS)}")
    print(f"Lookback: {LOOKBACK_YEARS} years | Train/Test: {TRAIN_FRAC:.0%}/{1-TRAIN_FRAC:.0%}")
    print("=" * 70)

    # 1. Download data
    data = download_data(ETFS, LOOKBACK_YEARS)

    if not data:
        print("\nFATAL: No data downloaded. Check network/yfinance.")
        sys.exit(1)

    # 2. Run backtest
    results = run_backtest(data)

    # 3. Rank signals
    rankings = rank_signals(results)

    if rankings.empty:
        print("\nNo signals had sufficient data for ranking.")
        sys.exit(1)

    # 4. Aggregate stats
    agg_stats = compute_signal_aggregate_stats(rankings)

    # 5. Recommended thresholds
    thresholds = compute_recommended_thresholds(rankings)

    # ======================================================================
    # PRINT SUMMARY TABLES
    # ======================================================================

    print("\n" + "=" * 70)
    print("TOP 25 SIGNAL-ETF COMBINATIONS (TEST DATA, SORTED BY EDGE QUALITY)")
    print("=" * 70)

    # Filter to 5d horizon for the main table (most relevant for options)
    top_5d = rankings[rankings["horizon"] == "5d"].head(25)

    if not top_5d.empty:
        display_cols = [
            "ticker", "signal", "direction", "test_n", "test_wr",
            "test_ret_pct", "test_sharpe", "test_pf", "test_p_value",
            "train_test_consistent",
        ]
        headers = [
            "Ticker", "Signal", "Dir", "N", "WR",
            "Ret%", "Sharpe", "PF", "p-val", "Consistent",
        ]
        table_data = []
        for _, row in top_5d[display_cols].iterrows():
            table_data.append([
                row["ticker"],
                row["signal"].replace("call_", "C:").replace("put_", "P:"),
                row["direction"],
                int(row["test_n"]),
                f"{row['test_wr']:.1%}",
                f"{row['test_ret_pct']:+.2f}",
                f"{row['test_sharpe']:+.2f}",
                f"{row['test_pf']:.2f}",
                f"{row['test_p_value']:.3f}",
                "Yes" if row["train_test_consistent"] else "NO",
            ])
        print(tabulate(table_data, headers=headers, tablefmt="simple"))

    print("\n" + "=" * 70)
    print("AGGREGATE SIGNAL PERFORMANCE (ACROSS ALL ETFS, 5-DAY HORIZON)")
    print("=" * 70)

    agg_5d = agg_stats[agg_stats["horizon"] == "5d"]
    if not agg_5d.empty:
        agg_table = []
        for _, row in agg_5d.iterrows():
            agg_table.append([
                row["signal"].replace("call_", "C:").replace("put_", "P:"),
                row["direction"],
                int(row["total_n"]),
                int(row["n_etfs"]),
                f"{row['mean_wr']:.1%}",
                f"{row['mean_ret']:+.2f}%",
                f"{row['mean_sharpe']:+.2f}",
                f"{row['mean_edge']:+.4f}",
                f"{row['pct_consistent']:.0%}",
                f"{row['median_p']:.3f}",
            ])
        agg_headers = [
            "Signal", "Dir", "TotalN", "#ETFs", "AvgWR",
            "AvgRet", "AvgSharpe", "AvgEdge", "%Consist", "MedP",
        ]
        print(tabulate(agg_table, headers=agg_headers, tablefmt="simple"))

    print("\n" + "=" * 70)
    print("SIGNAL RECOMMENDATIONS")
    print("=" * 70)

    for sig_name, rec in sorted(thresholds.items()):
        status_marker = {
            "TRADEABLE": "[OK]",
            "MARGINAL": "[~~]",
            "NO_CONSISTENT_EDGE": "[XX]",
        }.get(rec["status"], "[??]")
        print(f"\n{status_marker} {sig_name}")
        print(f"    {rec['recommendation']}")

    # ======================================================================
    # TRAIN/TEST CONSISTENCY CHECK
    # ======================================================================

    print("\n" + "=" * 70)
    print("TRAIN vs TEST CONSISTENCY (5-DAY HORIZON)")
    print("=" * 70)

    consist_5d = rankings[rankings["horizon"] == "5d"].copy()
    if not consist_5d.empty:
        n_total = len(consist_5d)
        n_consistent = consist_5d["train_test_consistent"].sum()
        n_profitable_test = (consist_5d["test_sharpe"] > 0).sum()
        n_sig = (consist_5d["test_p_value"] < 0.05).sum()

        print(f"  Total signal-ETF combinations:  {n_total}")
        print(f"  Train/test direction consistent: {n_consistent} ({n_consistent/n_total:.0%})")
        print(f"  Profitable on test data:         {n_profitable_test} ({n_profitable_test/n_total:.0%})")
        print(f"  Statistically significant (p<.05): {n_sig} ({n_sig/n_total:.0%})")

        # Highlight the REAL edges: consistent + significant
        real_edges = consist_5d[
            consist_5d["train_test_consistent"] &
            (consist_5d["test_p_value"] < 0.10) &
            (consist_5d["test_sharpe"] > 0)
        ]

        if not real_edges.empty:
            print(f"\n  REAL EDGES (consistent + p<0.10 + positive Sharpe): {len(real_edges)}")
            for _, row in real_edges.head(15).iterrows():
                print(f"    {row['ticker']:5s} {row['signal']:30s} "
                      f"WR={row['test_wr']:.1%} Ret={row['test_ret_pct']:+.2f}% "
                      f"Sharpe={row['test_sharpe']:+.2f} p={row['test_p_value']:.3f}")
        else:
            print("\n  No combinations pass all three filters.")
            print("  Loosening to p<0.20...")
            loose = consist_5d[
                consist_5d["train_test_consistent"] &
                (consist_5d["test_p_value"] < 0.20) &
                (consist_5d["test_sharpe"] > 0)
            ]
            if not loose.empty:
                for _, row in loose.head(10).iterrows():
                    print(f"    {row['ticker']:5s} {row['signal']:30s} "
                          f"WR={row['test_wr']:.1%} Ret={row['test_ret_pct']:+.2f}% "
                          f"Sharpe={row['test_sharpe']:+.2f} p={row['test_p_value']:.3f}")

    # ======================================================================
    # SAVE RESULTS
    # ======================================================================

    # Prepare rankings for JSON serialization
    rankings_json = {}
    for horizon in ["1d", "3d", "5d", "10d"]:
        h_data = rankings[rankings["horizon"] == horizon].to_dict(orient="records")
        rankings_json[f"rankings_{horizon}"] = h_data

    output = {
        "metadata": {
            "run_date": datetime.now().isoformat(),
            "etfs": ETFS,
            "lookback_years": LOOKBACK_YEARS,
            "train_frac": TRAIN_FRAC,
            "n_etfs_downloaded": len(data),
            "forward_windows": FORWARD_WINDOWS,
        },
        "per_etf_results": results,
        **rankings_json,
        "aggregate_stats": agg_stats.to_dict(orient="records") if not agg_stats.empty else [],
        "signal_recommendations": thresholds,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to {OUTPUT_PATH}")

    # ======================================================================
    # QUICK LIVE SCAN
    # ======================================================================

    print("\n" + "=" * 70)
    print("LIVE SCAN (CURRENT SIGNALS FIRING)")
    print("=" * 70)

    scan = live_scan()

    if scan["active_opportunities"]:
        scan_table = []
        for opp in scan["active_opportunities"]:
            signals_str = ", ".join(
                f"{s['direction']}:{s['signal'].split('_', 1)[1]}"
                for s in opp["active_signals"]
            )
            scan_table.append([
                opp["ticker"],
                opp["n_active"],
                f"{opp['total_edge_score']:+.4f}",
                opp.get("indicators", {}).get("rsi_14", ""),
                opp.get("indicators", {}).get("bb_position", ""),
                signals_str[:60],
            ])
        scan_headers = ["Ticker", "#Sig", "EdgeScore", "RSI14", "BB%", "Active Signals"]
        print(tabulate(scan_table, headers=scan_headers, tablefmt="simple"))
    else:
        print("  No signals currently firing on any ETF.")

    print("\nDone.")
    return output


if __name__ == "__main__":
    main()
