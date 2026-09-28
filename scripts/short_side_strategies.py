#!/usr/bin/env python3
"""
short_side_strategies.py — Short-side equity strategy backtest suite.

Tests 6 short-side strategies on S&P 500 top-100 liquid stocks.
Our MBO analysis showed short side has significantly better edge than long.
Most of our 80+ backtests were long-only (equity beta). This tests short alpha.

Strategies:
  A. RSI extreme short (parabolic exhaustion)
  B. Bollinger band short (parabolic exhaustion)
  C. Gap-up fade (gap exhaustion, non-earnings)
  D. Earnings gap exhaustion (over-reaction short)
  E. Momentum reversal (top-decile recent winners)
  F. Volume climax short (climax buying)

5-gate validation:
  1. Sharpe > 0.5
  2. Permutation p < 0.05 (500 iter)
  3. Regime gap < 0.5
  4. MaxDD > -50%
  5. >= 20 trades
"""

import json
import sys
import warnings
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.utils.yfinance_safe import safe_download

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*LARGE MOVE.*")
warnings.filterwarnings("ignore", message=".*SPLIT ARTIFACT.*")
warnings.filterwarnings("ignore", message=".*PRICE DISCONTINUITY.*")

# ─── Configuration ───────────────────────────────────────────────────────────
OOT_START = "2022-01-01"
OOT_END = "2026-07-25"
DATA_START = "2020-06-01"  # Need lookback for indicators (RSI, BB, SMA200 etc.)
STARTING_CAPITAL = 645.0
BORROW_COST_BPS_PER_DAY = 0.5  # 0.5 bps/day, conservative for S&P 500
SLIPPAGE_BPS = 5.0  # 5 bps per side
PERMUTATION_ITERS = 500

# Top 100 S&P 500 by market cap (as of mid-2026, stable large caps)
SP500_TOP100 = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "BRK-B", "LLY", "AVGO", "JPM",
    "TSLA", "UNH", "XOM", "V", "MA", "PG", "JNJ", "COST", "HD", "ABBV",
    "MRK", "NFLX", "CRM", "BAC", "AMD", "CVX", "KO", "ORCL", "PEP", "WMT",
    "LIN", "TMO", "ACN", "MCD", "CSCO", "ABT", "ADBE", "WFC", "PM", "IBM",
    "GE", "CAT", "ISRG", "INTU", "VZ", "QCOM", "TXN", "NOW", "CMCSA", "GS",
    "DHR", "AMGN", "NEE", "PFE", "T", "MS", "SPGI", "RTX", "LOW", "HON",
    "UNP", "BLK", "ELV", "BKNG", "SYK", "AMAT", "ADP", "DE", "MDLZ", "GILD",
    "SCHW", "MMC", "TJX", "LRCX", "VRTX", "AXP", "CB", "CI", "ADI", "REGN",
    "SBUX", "MO", "CME", "PLD", "BSX", "ZTS", "PANW", "SNPS", "KLAC", "TMUS",
    "FI", "EQIX", "ICE", "SO", "DUK", "CL", "SHW", "MCO", "CDNS", "EOG",
]


# ─── Helper Functions ────────────────────────────────────────────────────────

def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Standard RSI calculation."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_bollinger(close: pd.Series, period: int = 20, num_std: float = 2.5):
    """Return (upper, middle, lower) Bollinger Bands."""
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    return mid + num_std * std, mid, mid - num_std * std


def compute_sma(close: pd.Series, period: int) -> pd.Series:
    return close.rolling(period).mean()


def get_regime(spy_close: pd.Series) -> pd.Series:
    """Bull = SPY > 200-SMA, Bear = SPY < 200-SMA."""
    sma200 = compute_sma(spy_close, 200)
    regime = pd.Series("bull", index=spy_close.index)
    regime[spy_close < sma200] = "bear"
    return regime


def apply_costs(pnl_pct: float, hold_days: int) -> float:
    """Apply slippage (entry+exit) and borrow cost to a short trade return."""
    slippage_cost = 2 * SLIPPAGE_BPS / 10000  # entry + exit
    borrow_cost = hold_days * BORROW_COST_BPS_PER_DAY / 10000
    return pnl_pct - slippage_cost - borrow_cost


def backtest_trades(trades: List[Dict], starting_capital: float = STARTING_CAPITAL) -> Dict:
    """
    Given a list of trades with 'date', 'pnl_pct', 'regime', compute metrics.
    Each trade is equal-weight (full capital allocation, no compounding for simplicity,
    but we track equity curve).
    """
    if len(trades) == 0:
        return {
            "n_trades": 0, "sharpe": 0, "sortino": 0, "profit_factor": 0,
            "win_rate": 0, "max_drawdown_pct": 0, "total_return_pct": 0,
            "sharpe_bull": 0, "sharpe_bear": 0, "regime_gap": 0,
        }

    # Sort by date
    trades = sorted(trades, key=lambda t: t["date"])
    returns = np.array([t["pnl_pct"] for t in trades])

    # Equity curve (simple: each trade uses full capital, returns compound)
    equity = [starting_capital]
    for r in returns:
        equity.append(equity[-1] * (1 + r))
    equity = np.array(equity)

    # Max drawdown
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd = dd.min()

    # Sharpe (annualize assuming ~20 trades/year as baseline, use daily-ish)
    if len(returns) > 1 and returns.std() > 0:
        # Annualize: assume average hold is ~10 days, so ~25 trades/year
        trades_per_year = min(252 / 10, len(returns) / 4.5)  # rough
        sharpe = (returns.mean() / returns.std()) * np.sqrt(trades_per_year)
    else:
        sharpe = 0.0

    # Sortino
    downside = returns[returns < 0]
    if len(downside) > 1:
        downside_std = downside.std()
        if downside_std > 0:
            trades_per_year = min(252 / 10, len(returns) / 4.5)
            sortino = (returns.mean() / downside_std) * np.sqrt(trades_per_year)
        else:
            sortino = np.inf if returns.mean() > 0 else 0
    else:
        sortino = np.inf if returns.mean() > 0 else 0

    # Profit factor
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else (np.inf if gross_profit > 0 else 0)

    # Win rate
    wr = (returns > 0).sum() / len(returns)

    # Regime split
    bull_returns = np.array([t["pnl_pct"] for t in trades if t["regime"] == "bull"])
    bear_returns = np.array([t["pnl_pct"] for t in trades if t["regime"] == "bear"])

    def _sharpe(arr):
        if len(arr) < 3 or arr.std() == 0:
            return 0.0
        tpy = min(252 / 10, len(arr) / 4.5)
        return (arr.mean() / arr.std()) * np.sqrt(tpy)

    sharpe_bull = _sharpe(bull_returns)
    sharpe_bear = _sharpe(bear_returns)

    max_abs = max(abs(sharpe_bull), abs(sharpe_bear), 1e-9)
    regime_gap = abs(sharpe_bull - sharpe_bear) / max_abs

    return {
        "n_trades": len(returns),
        "sharpe": round(sharpe, 3),
        "sortino": round(min(sortino, 99.0), 3),
        "profit_factor": round(min(pf, 99.0), 3),
        "win_rate": round(wr, 4),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "total_return_pct": round((equity[-1] / equity[0] - 1) * 100, 2),
        "sharpe_bull": round(sharpe_bull, 3),
        "sharpe_bear": round(sharpe_bear, 3),
        "regime_gap": round(regime_gap, 3),
        "n_bull_trades": len(bull_returns),
        "n_bear_trades": len(bear_returns),
        "avg_return_pct": round(returns.mean() * 100, 4),
        "median_return_pct": round(np.median(returns) * 100, 4),
    }


def permutation_test(returns: np.ndarray, n_iter: int = PERMUTATION_ITERS) -> float:
    """Permutation test: fraction of random shuffles with mean >= observed."""
    if len(returns) < 5:
        return 1.0
    observed = returns.mean()
    count = 0
    rng = np.random.RandomState(42)
    for _ in range(n_iter):
        shuffled = returns.copy()
        rng.shuffle(shuffled)
        if shuffled.mean() >= observed:
            count += 1
    return count / n_iter


def validate_5gates(metrics: Dict, returns: np.ndarray) -> Dict:
    """Run 5-gate validation."""
    perm_p = permutation_test(returns)
    gates = {
        "1_sharpe_gt_05": metrics["sharpe"] > 0.5,
        "2_perm_p_lt_005": perm_p < 0.05,
        "3_regime_gap_lt_05": metrics["regime_gap"] < 0.5,
        "4_maxdd_gt_neg50": metrics["max_drawdown_pct"] > -50.0,
        "5_min_20_trades": metrics["n_trades"] >= 20,
    }
    return {
        "gates": gates,
        "perm_p_value": round(perm_p, 4),
        "gates_passed": sum(gates.values()),
        "all_passed": all(gates.values()),
    }


# ─── Strategy Implementations ────────────────────────────────────────────────

def strategy_a_rsi_extreme(close_df: pd.DataFrame, regime_series: pd.Series) -> List[Dict]:
    """
    A. RSI Extreme Short: Short when RSI(14) > 80, hold 10 days.
    Regime-aware: full size in bull (shorting into strength), half in bear.
    """
    trades = []
    oot_start = pd.Timestamp(OOT_START)
    oot_end = pd.Timestamp(OOT_END)
    hold_days = 10

    for ticker in close_df.columns:
        px = close_df[ticker].dropna()
        if len(px) < 50:
            continue
        rsi = compute_rsi(px, 14)

        # Find entry signals in OOT
        for i in range(len(px)):
            dt = px.index[i]
            if dt < oot_start or dt > oot_end:
                continue
            if pd.isna(rsi.iloc[i]) or rsi.iloc[i] <= 80:
                continue

            # Entry
            entry_price = px.iloc[i]
            exit_idx = min(i + hold_days, len(px) - 1)
            exit_price = px.iloc[exit_idx]
            exit_date = px.index[exit_idx]

            if entry_price <= 0 or exit_price <= 0:
                continue

            # SHORT P&L: positive when stock goes down
            raw_pnl_pct = (entry_price - exit_price) / entry_price
            actual_hold = (exit_date - dt).days
            pnl_pct = apply_costs(raw_pnl_pct, max(actual_hold, 1))

            # Regime sizing
            regime = regime_series.get(dt, "bull")
            if regime == "bear":
                pnl_pct *= 0.5  # half size in bear

            trades.append({
                "date": str(dt.date()),
                "ticker": ticker,
                "entry_price": round(entry_price, 2),
                "exit_price": round(exit_price, 2),
                "pnl_pct": pnl_pct,
                "regime": regime,
                "rsi": round(rsi.iloc[i], 1),
            })

    return trades


def strategy_b_bollinger_short(close_df: pd.DataFrame, regime_series: pd.Series) -> List[Dict]:
    """
    B. Bollinger Band Short: Short when price > upper BB(20, 2.5).
    Close when price < BB midline or 15 days max.
    """
    trades = []
    oot_start = pd.Timestamp(OOT_START)
    oot_end = pd.Timestamp(OOT_END)
    max_hold = 15

    for ticker in close_df.columns:
        px = close_df[ticker].dropna()
        if len(px) < 30:
            continue
        upper, mid, _ = compute_bollinger(px, 20, 2.5)

        for i in range(len(px)):
            dt = px.index[i]
            if dt < oot_start or dt > oot_end:
                continue
            if pd.isna(upper.iloc[i]) or px.iloc[i] <= upper.iloc[i]:
                continue

            entry_price = px.iloc[i]

            # Find exit: price < midline or max_hold
            exit_idx = i + 1
            while exit_idx < min(i + max_hold, len(px)):
                if px.iloc[exit_idx] < mid.iloc[exit_idx]:
                    break
                exit_idx += 1
            exit_idx = min(exit_idx, len(px) - 1)
            exit_price = px.iloc[exit_idx]
            exit_date = px.index[exit_idx]

            if entry_price <= 0 or exit_price <= 0:
                continue

            raw_pnl_pct = (entry_price - exit_price) / entry_price
            actual_hold = (exit_date - dt).days
            pnl_pct = apply_costs(raw_pnl_pct, max(actual_hold, 1))

            regime = regime_series.get(dt, "bull")
            trades.append({
                "date": str(dt.date()),
                "ticker": ticker,
                "entry_price": round(entry_price, 2),
                "exit_price": round(exit_price, 2),
                "pnl_pct": pnl_pct,
                "regime": regime,
            })

    return trades


def strategy_c_gap_fade(close_df: pd.DataFrame, open_df: pd.DataFrame,
                         regime_series: pd.Series) -> List[Dict]:
    """
    C. Gap-Up Fade: Short stocks that gap up >5% on non-earnings days, hold 5 days.
    We can't easily filter earnings dates without an earnings calendar, so we
    skip gaps >15% (likely earnings) and only take 5-15% gaps.
    """
    trades = []
    oot_start = pd.Timestamp(OOT_START)
    oot_end = pd.Timestamp(OOT_END)
    hold_days = 5

    for ticker in close_df.columns:
        px_close = close_df[ticker].dropna()
        px_open = open_df[ticker].dropna()
        # Align
        common = px_close.index.intersection(px_open.index)
        if len(common) < 10:
            continue
        px_close = px_close.loc[common]
        px_open = px_open.loc[common]

        for i in range(1, len(common)):
            dt = common[i]
            if dt < oot_start or dt > oot_end:
                continue

            prev_close = px_close.iloc[i - 1]
            today_open = px_open.iloc[i]

            if prev_close <= 0:
                continue

            gap_pct = (today_open - prev_close) / prev_close

            # Non-earnings proxy: take 5-15% gaps (>15% likely earnings)
            if gap_pct < 0.05 or gap_pct > 0.15:
                continue

            entry_price = today_open  # Short at the open
            exit_idx = min(i + hold_days, len(common) - 1)
            exit_price = px_close.iloc[exit_idx]
            exit_date = common[exit_idx]

            if entry_price <= 0 or exit_price <= 0:
                continue

            raw_pnl_pct = (entry_price - exit_price) / entry_price
            actual_hold = (exit_date - dt).days
            pnl_pct = apply_costs(raw_pnl_pct, max(actual_hold, 1))

            regime = regime_series.get(dt, "bull")
            trades.append({
                "date": str(dt.date()),
                "ticker": ticker,
                "entry_price": round(entry_price, 2),
                "exit_price": round(exit_price, 2),
                "gap_pct": round(gap_pct * 100, 2),
                "pnl_pct": pnl_pct,
                "regime": regime,
            })

    return trades


def strategy_d_earnings_gap_exhaustion(close_df: pd.DataFrame, open_df: pd.DataFrame,
                                        regime_series: pd.Series) -> List[Dict]:
    """
    D. Earnings Gap Exhaustion: Short stocks that gap up >10% (earnings proxy), hold 10 days.
    Gaps >10% are very likely earnings-driven. Over-reaction tends to revert.
    """
    trades = []
    oot_start = pd.Timestamp(OOT_START)
    oot_end = pd.Timestamp(OOT_END)
    hold_days = 10

    for ticker in close_df.columns:
        px_close = close_df[ticker].dropna()
        px_open = open_df[ticker].dropna()
        common = px_close.index.intersection(px_open.index)
        if len(common) < 10:
            continue
        px_close = px_close.loc[common]
        px_open = px_open.loc[common]

        for i in range(1, len(common)):
            dt = common[i]
            if dt < oot_start or dt > oot_end:
                continue

            prev_close = px_close.iloc[i - 1]
            today_open = px_open.iloc[i]

            if prev_close <= 0:
                continue

            gap_pct = (today_open - prev_close) / prev_close

            # Large gaps (>10%) — likely earnings over-reaction
            if gap_pct < 0.10:
                continue

            entry_price = today_open
            exit_idx = min(i + hold_days, len(common) - 1)
            exit_price = px_close.iloc[exit_idx]
            exit_date = common[exit_idx]

            if entry_price <= 0 or exit_price <= 0:
                continue

            raw_pnl_pct = (entry_price - exit_price) / entry_price
            actual_hold = (exit_date - dt).days
            pnl_pct = apply_costs(raw_pnl_pct, max(actual_hold, 1))

            regime = regime_series.get(dt, "bull")
            trades.append({
                "date": str(dt.date()),
                "ticker": ticker,
                "entry_price": round(entry_price, 2),
                "exit_price": round(exit_price, 2),
                "gap_pct": round(gap_pct * 100, 2),
                "pnl_pct": pnl_pct,
                "regime": regime,
            })

    return trades


def strategy_e_momentum_reversal(close_df: pd.DataFrame, regime_series: pd.Series) -> List[Dict]:
    """
    E. Momentum Reversal: Short stocks in top decile of 20-day returns, hold 20 days.
    Classic short-term reversal — biggest recent winners tend to give back gains.
    Cross-sectional: rank all stocks each day, short top 10%.
    """
    trades = []
    oot_start = pd.Timestamp(OOT_START)
    oot_end = pd.Timestamp(OOT_END)
    hold_days = 20
    lookback = 20

    # Compute 20-day returns for all stocks
    ret20 = close_df.pct_change(lookback)

    # Only look at dates in OOT
    oot_dates = ret20.index[(ret20.index >= oot_start) & (ret20.index <= oot_end)]

    # Sample every 20 days (non-overlapping trades)
    sampled_dates = oot_dates[::hold_days]

    for dt in sampled_dates:
        row = ret20.loc[dt].dropna()
        if len(row) < 20:
            continue

        # Top decile = top 10%
        threshold = row.quantile(0.90)
        top_decile = row[row >= threshold].index.tolist()

        for ticker in top_decile:
            entry_price = close_df.loc[dt, ticker]
            if pd.isna(entry_price) or entry_price <= 0:
                continue

            # Find exit
            future_dates = close_df.index[close_df.index > dt]
            if len(future_dates) < hold_days:
                continue
            exit_date = future_dates[min(hold_days - 1, len(future_dates) - 1)]
            exit_price = close_df.loc[exit_date, ticker]
            if pd.isna(exit_price) or exit_price <= 0:
                continue

            raw_pnl_pct = (entry_price - exit_price) / entry_price
            actual_hold = (exit_date - dt).days
            pnl_pct = apply_costs(raw_pnl_pct, max(actual_hold, 1))

            regime = regime_series.get(dt, "bull")
            trades.append({
                "date": str(dt.date()),
                "ticker": ticker,
                "entry_price": round(entry_price, 2),
                "exit_price": round(exit_price, 2),
                "mom_20d": round(ret20.loc[dt, ticker] * 100, 2),
                "pnl_pct": pnl_pct,
                "regime": regime,
            })

    return trades


def strategy_f_volume_climax(close_df: pd.DataFrame, volume_df: pd.DataFrame,
                              regime_series: pd.Series) -> List[Dict]:
    """
    F. Volume Climax Short: Short when stock is up >3% on >3x average volume.
    Climax buying exhaustion. Hold 5 days.
    """
    trades = []
    oot_start = pd.Timestamp(OOT_START)
    oot_end = pd.Timestamp(OOT_END)
    hold_days = 5
    vol_lookback = 20

    for ticker in close_df.columns:
        px = close_df[ticker].dropna()
        vol = volume_df[ticker].dropna() if ticker in volume_df.columns else pd.Series(dtype=float)
        common = px.index.intersection(vol.index)
        if len(common) < vol_lookback + 5:
            continue
        px = px.loc[common]
        vol = vol.loc[common]

        avg_vol = vol.rolling(vol_lookback).mean()
        daily_ret = px.pct_change()

        for i in range(vol_lookback + 1, len(common)):
            dt = common[i]
            if dt < oot_start or dt > oot_end:
                continue

            if pd.isna(daily_ret.iloc[i]) or pd.isna(avg_vol.iloc[i]):
                continue
            if avg_vol.iloc[i] <= 0:
                continue

            # Up >3% on >3x average volume
            if daily_ret.iloc[i] <= 0.03:
                continue
            if vol.iloc[i] <= 3.0 * avg_vol.iloc[i]:
                continue

            entry_price = px.iloc[i]
            exit_idx = min(i + hold_days, len(common) - 1)
            exit_price = px.iloc[exit_idx]
            exit_date = common[exit_idx]

            if entry_price <= 0 or exit_price <= 0:
                continue

            raw_pnl_pct = (entry_price - exit_price) / entry_price
            actual_hold = (exit_date - dt).days
            pnl_pct = apply_costs(raw_pnl_pct, max(actual_hold, 1))

            regime = regime_series.get(dt, "bull")
            vol_ratio = vol.iloc[i] / avg_vol.iloc[i]
            trades.append({
                "date": str(dt.date()),
                "ticker": ticker,
                "entry_price": round(entry_price, 2),
                "exit_price": round(exit_price, 2),
                "daily_ret_pct": round(daily_ret.iloc[i] * 100, 2),
                "vol_ratio": round(vol_ratio, 1),
                "pnl_pct": pnl_pct,
                "regime": regime,
            })

    return trades


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("SHORT-SIDE STRATEGY BACKTEST SUITE")
    print(f"Universe: {len(SP500_TOP100)} S&P 500 stocks | OOT: {OOT_START} to {OOT_END}")
    print(f"Starting capital: ${STARTING_CAPITAL} | Borrow: {BORROW_COST_BPS_PER_DAY} bps/day | Slippage: {SLIPPAGE_BPS} bps")
    print("=" * 80)

    # ── Download data ────────────────────────────────────────────────────────
    print("\n[1/3] Downloading price data...")
    t0 = time.time()

    # Download in batches to avoid yfinance issues
    batch_size = 25
    all_close = {}
    all_open = {}
    all_volume = {}

    for batch_start in range(0, len(SP500_TOP100), batch_size):
        batch = SP500_TOP100[batch_start:batch_start + batch_size]
        print(f"  Downloading batch {batch_start // batch_size + 1}/{(len(SP500_TOP100) + batch_size - 1) // batch_size}: {batch[0]}...{batch[-1]}")
        try:
            close, open_, volume = safe_download(batch, DATA_START, OOT_END, progress=False)
            for col in close.columns:
                all_close[col] = close[col]
            for col in open_.columns:
                all_open[col] = open_[col]
            for col in volume.columns:
                all_volume[col] = volume[col]
        except Exception as e:
            print(f"  WARNING: Batch failed: {e}")
            continue

    close_df = pd.DataFrame(all_close)
    open_df = pd.DataFrame(all_open)
    volume_df = pd.DataFrame(all_volume)

    print(f"  Downloaded {len(close_df.columns)} tickers, {len(close_df)} trading days in {time.time()-t0:.0f}s")

    # ── SPY for regime ───────────────────────────────────────────────────────
    print("  Downloading SPY for regime detection...")
    spy_close, _, _ = safe_download(["SPY"], DATA_START, OOT_END, progress=False)
    spy_close = spy_close["SPY"]
    regime_series = get_regime(spy_close)

    # ── Run strategies ───────────────────────────────────────────────────────
    print("\n[2/3] Running strategies...")

    strategies = {
        "A_RSI_Extreme_Short": {
            "description": "Short RSI(14)>80, hold 10d, regime-aware sizing",
            "func": lambda: strategy_a_rsi_extreme(close_df, regime_series),
        },
        "B_Bollinger_Band_Short": {
            "description": "Short above BB(20,2.5), exit at midline or 15d max",
            "func": lambda: strategy_b_bollinger_short(close_df, regime_series),
        },
        "C_Gap_Up_Fade": {
            "description": "Short 5-15% gap-ups (non-earnings proxy), hold 5d",
            "func": lambda: strategy_c_gap_fade(close_df, open_df, regime_series),
        },
        "D_Earnings_Gap_Exhaustion": {
            "description": "Short >10% gap-ups (earnings proxy), hold 10d",
            "func": lambda: strategy_d_earnings_gap_exhaustion(close_df, open_df, regime_series),
        },
        "E_Momentum_Reversal": {
            "description": "Short top-decile 20d winners, hold 20d, cross-sectional",
            "func": lambda: strategy_e_momentum_reversal(close_df, regime_series),
        },
        "F_Volume_Climax_Short": {
            "description": "Short >3% up on >3x volume, hold 5d",
            "func": lambda: strategy_f_volume_climax(close_df, volume_df, regime_series),
        },
    }

    results = {}

    for name, spec in strategies.items():
        print(f"\n  Running {name}...")
        t1 = time.time()
        trades = spec["func"]()
        metrics = backtest_trades(trades)
        returns = np.array([t["pnl_pct"] for t in trades])
        validation = validate_5gates(metrics, returns)

        elapsed = time.time() - t1
        print(f"    {metrics['n_trades']} trades | Sharpe={metrics['sharpe']:.2f} | "
              f"Sortino={metrics['sortino']:.2f} | PF={metrics['profit_factor']:.2f} | "
              f"WR={metrics['win_rate']:.1%} | MaxDD={metrics['max_drawdown_pct']:.1f}% | "
              f"RegimeGap={metrics['regime_gap']:.2f} | "
              f"Gates={validation['gates_passed']}/5 | {elapsed:.1f}s")

        # Verdict
        if validation["all_passed"]:
            verdict = "PASS — genuine short-side edge"
        elif validation["gates_passed"] >= 3:
            verdict = "PARTIAL — some merit, needs refinement"
        else:
            verdict = "FAIL — no reliable edge"

        results[name] = {
            "description": spec["description"],
            "metrics": metrics,
            "validation": validation,
            "verdict": verdict,
            "sample_trades": trades[:5] if trades else [],
            "trade_count_by_year": {},
        }

        # Trade count by year
        for t in trades:
            year = t["date"][:4]
            results[name]["trade_count_by_year"][year] = \
                results[name]["trade_count_by_year"].get(year, 0) + 1

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SUMMARY — SHORT-SIDE STRATEGY RESULTS")
    print("=" * 80)
    print(f"{'Strategy':<30} {'Trades':>6} {'Sharpe':>7} {'Sortino':>8} {'PF':>6} "
          f"{'WR':>6} {'MaxDD':>7} {'RGap':>5} {'Gates':>5} {'Verdict':<30}")
    print("-" * 120)

    passing = []
    for name, r in results.items():
        m = r["metrics"]
        v = r["validation"]
        print(f"{name:<30} {m['n_trades']:>6} {m['sharpe']:>7.2f} {m['sortino']:>8.2f} "
              f"{m['profit_factor']:>6.2f} {m['win_rate']:>5.1%} {m['max_drawdown_pct']:>6.1f}% "
              f"{m['regime_gap']:>5.2f} {v['gates_passed']:>3}/5  {r['verdict']:<30}")
        if v["all_passed"]:
            passing.append(name)

    print(f"\nStrategies passing all 5 gates: {len(passing)}")
    for p in passing:
        print(f"  -> {p}")

    if not passing:
        print("\nNo strategy passed all 5 gates. Best candidates for refinement:")
        ranked = sorted(results.items(), key=lambda x: x[1]["validation"]["gates_passed"], reverse=True)
        for name, r in ranked[:3]:
            failed = [k for k, v in r["validation"]["gates"].items() if not v]
            print(f"  {name}: {r['validation']['gates_passed']}/5 gates — failed: {', '.join(failed)}")

    # ── Save results ─────────────────────────────────────────────────────────
    print("\n[3/3] Saving results...")

    output = {
        "strategy_suite": "short_side_strategies",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "oot_period": f"{OOT_START} to {OOT_END}",
            "universe_size": len(close_df.columns),
            "starting_capital": STARTING_CAPITAL,
            "borrow_cost_bps_per_day": BORROW_COST_BPS_PER_DAY,
            "slippage_bps": SLIPPAGE_BPS,
            "permutation_iterations": PERMUTATION_ITERS,
        },
        "strategies": results,
        "summary": {
            "total_strategies": len(results),
            "passing_all_gates": len(passing),
            "passing_names": passing,
            "best_sharpe": max((r["metrics"]["sharpe"] for r in results.values()), default=0),
            "best_strategy": max(results.keys(), key=lambda k: results[k]["metrics"]["sharpe"]) if results else None,
        },
    }

    output_path = Path("/home/jupiter/Lvl3Quant/data/short_side_strategies_results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"  Results saved to {output_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
