#!/usr/bin/env python3
"""
Spin-Off / Special Situation Event Backtest
============================================
Tests 6 variants of forced-selling recovery strategies using known spin-offs,
recent IPOs, and proxy signals for institutional forced selling.

Academic basis: Joel Greenblatt's "You Can Be a Stock Picking Genius" — spin-offs
consistently outperform for 12-24 months due to structural forced selling by index funds.

OOT: Jan 2022 – Jul 2026 | Account: $645 | Slippage: 0.05% each way
"""

import json
import warnings
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ─── Configuration ───────────────────────────────────────────────────────────

ACCOUNT_SIZE = 645.0
SLIPPAGE_PCT = 0.0005  # 0.05% each way
OOT_START = "2022-01-01"
OOT_END = "2026-07-29"
PERMUTATION_ITERS = 1000
RANDOM_SEED = 42

# Known spin-offs and recent IPOs
SPINOFF_TICKERS = [
    "KVUE", "GEV", "SOLV", "VLTO", "GEHC", "KD", "CTVA", "DOW",
    "OTIS", "CARR", "PLTR", "RIVN", "COIN", "RBLX", "HOOD", "ABNB",
    "DASH", "UBER", "LYFT",
]

# Sector ETFs for variant F
SECTOR_ETFS = {
    "Technology": "XLK", "Healthcare": "XLV", "Financials": "XLF",
    "Consumer Discretionary": "XLY", "Industrials": "XLI",
    "Energy": "XLE", "Materials": "XLB", "Communication": "XLC",
    "Utilities": "XLU", "Real Estate": "XLRE", "Consumer Staples": "XLP",
}

# Map tickers to rough sectors for variant F
TICKER_SECTORS = {
    "KVUE": "Healthcare", "SOLV": "Healthcare", "GEHC": "Healthcare",
    "GEV": "Industrials", "VLTO": "Industrials", "OTIS": "Industrials", "CARR": "Industrials",
    "KD": "Technology", "PLTR": "Technology", "COIN": "Technology",
    "RBLX": "Technology", "HOOD": "Financials",
    "CTVA": "Materials", "DOW": "Materials",
    "ABNB": "Consumer Discretionary", "DASH": "Consumer Discretionary",
    "UBER": "Consumer Discretionary", "LYFT": "Consumer Discretionary",
    "RIVN": "Consumer Discretionary",
}

# ─── Data Download ───────────────────────────────────────────────────────────

def download_data(tickers, start, end):
    """Download daily OHLCV data for all tickers."""
    all_data = {}
    # Also download sector ETFs
    all_tickers = list(set(tickers + list(SECTOR_ETFS.values()) + ["SPY"]))
    print(f"Downloading {len(all_tickers)} tickers...")
    for ticker in all_tickers:
        try:
            df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            if df is not None and len(df) > 20:
                # Flatten MultiIndex columns if present
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                all_data[ticker] = df
        except Exception as e:
            print(f"  Warning: {ticker} download failed: {e}")
    print(f"  Downloaded {len(all_data)} tickers successfully")
    return all_data


# ─── Trade Execution Helpers ─────────────────────────────────────────────────

def apply_slippage(price, direction="buy"):
    """Apply slippage to execution price."""
    if direction == "buy":
        return price * (1 + SLIPPAGE_PCT)
    else:
        return price * (1 - SLIPPAGE_PCT)


def compute_metrics(returns, trades_df):
    """Compute Sharpe, Sortino, PF, WR, MaxDD, trade count."""
    if len(returns) == 0 or len(trades_df) == 0:
        return {
            "sharpe": 0, "sortino": 0, "pf": 0, "wr": 0,
            "max_dd_pct": -100, "n_trades": 0, "total_return_pct": 0,
            "avg_return_pct": 0, "median_hold_days": 0,
        }

    # Daily returns series for Sharpe/Sortino
    mean_r = np.mean(returns)
    std_r = np.std(returns, ddof=1) if len(returns) > 1 else 1e-9
    downside = returns[returns < 0]
    downside_std = np.std(downside, ddof=1) if len(downside) > 1 else 1e-9

    sharpe = (mean_r / std_r) * np.sqrt(252) if std_r > 0 else 0
    sortino = (mean_r / downside_std) * np.sqrt(252) if downside_std > 0 else 0

    # PF from trades
    wins = trades_df[trades_df["pnl"] > 0]["pnl"].sum()
    losses = abs(trades_df[trades_df["pnl"] < 0]["pnl"].sum())
    pf = wins / losses if losses > 0 else (999 if wins > 0 else 0)

    wr = (trades_df["pnl"] > 0).mean() * 100

    # MaxDD from cumulative equity
    cum = (1 + pd.Series(returns)).cumprod()
    rolling_max = cum.cummax()
    dd = (cum - rolling_max) / rolling_max
    max_dd = dd.min() * 100

    total_ret = (cum.iloc[-1] - 1) * 100 if len(cum) > 0 else 0

    return {
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "pf": round(pf, 3),
        "wr": round(wr, 1),
        "max_dd_pct": round(max_dd, 2),
        "n_trades": len(trades_df),
        "total_return_pct": round(total_ret, 2),
        "avg_return_pct": round(trades_df["return_pct"].mean(), 2),
        "median_hold_days": int(trades_df["hold_days"].median()) if len(trades_df) > 0 else 0,
    }


def permutation_test(returns, n_iters=PERMUTATION_ITERS):
    """Permutation test: shuffle returns, compute p-value for observed Sharpe."""
    if len(returns) < 10:
        return 1.0
    rng = np.random.RandomState(RANDOM_SEED)
    observed_sharpe = np.mean(returns) / (np.std(returns, ddof=1) + 1e-12) * np.sqrt(252)
    count_ge = 0
    for _ in range(n_iters):
        shuffled = rng.permutation(returns)
        s = np.mean(shuffled) / (np.std(shuffled, ddof=1) + 1e-12) * np.sqrt(252)
        if s >= observed_sharpe:
            count_ge += 1
    return count_ge / n_iters


def regime_analysis(trades_df, spy_data):
    """Split trades by SPY regime (green/red periods) and compute gap."""
    if len(trades_df) < 4 or spy_data is None or len(spy_data) == 0:
        return 0.99  # fail if insufficient data

    spy_returns = spy_data["Close"].pct_change().dropna()
    # Rolling 20-day SPY return to classify regime
    spy_rolling = spy_returns.rolling(20).mean()

    green_pnls, red_pnls = [], []
    for _, trade in trades_df.iterrows():
        entry_date = trade["entry_date"]
        if isinstance(entry_date, str):
            entry_date = pd.Timestamp(entry_date)
        # Find nearest SPY date
        nearest_dates = spy_rolling.index[spy_rolling.index <= entry_date]
        if len(nearest_dates) == 0:
            continue
        spy_regime = spy_rolling.loc[nearest_dates[-1]]
        if pd.isna(spy_regime):
            continue
        if spy_regime > 0:
            green_pnls.append(trade["return_pct"])
        else:
            red_pnls.append(trade["return_pct"])

    if len(green_pnls) < 2 or len(red_pnls) < 2:
        return 0.99  # can't assess

    sharpe_green = np.mean(green_pnls) / (np.std(green_pnls, ddof=1) + 1e-12)
    sharpe_red = np.mean(red_pnls) / (np.std(red_pnls, ddof=1) + 1e-12)

    gap = abs(sharpe_green - sharpe_red) / (max(abs(sharpe_green), abs(sharpe_red)) + 1e-12)
    return round(gap, 3)


# ─── Strategy Variants ───────────────────────────────────────────────────────

def variant_a_post_ipo_recovery(data, tickers):
    """
    Variant A: Post-IPO Recovery
    Buy stocks 30 days after their listing if down 20%+ from first-week high. Hold 60 days.
    """
    trades = []
    for ticker in tickers:
        if ticker not in data:
            continue
        df = data[ticker]
        if len(df) < 90:
            continue
        # First available date = proxy for listing date
        listing_date = df.index[0]
        first_week = df.iloc[:5]
        first_week_high = first_week["High"].max()

        # Check price at day 30
        if len(df) < 30:
            continue
        day30_idx = min(29, len(df) - 1)
        price_day30 = df.iloc[day30_idx]["Close"]
        drop_pct = (price_day30 - first_week_high) / first_week_high

        if drop_pct <= -0.20:  # Down 20%+
            entry_idx = day30_idx
            entry_price = apply_slippage(df.iloc[entry_idx]["Close"], "buy")
            exit_idx = min(entry_idx + 60, len(df) - 1)
            exit_price = apply_slippage(df.iloc[exit_idx]["Close"], "sell")

            shares = max(1, int(ACCOUNT_SIZE / entry_price))
            pnl = (exit_price - entry_price) * shares
            ret_pct = (exit_price - entry_price) / entry_price * 100

            trades.append({
                "ticker": ticker,
                "entry_date": str(df.index[entry_idx].date()),
                "exit_date": str(df.index[exit_idx].date()),
                "entry_price": round(entry_price, 2),
                "exit_price": round(exit_price, 2),
                "pnl": round(pnl, 2),
                "return_pct": round(ret_pct, 2),
                "hold_days": exit_idx - entry_idx,
                "signal": f"Down {drop_pct*100:.0f}% from first-week high at day 30",
            })

        # Also scan for additional drops within OOT period
        # Look for any 20%+ drop from 20-day high, then buy and hold 60 days
        highs_20d = df["High"].rolling(20).max()
        for i in range(30, len(df) - 60, 20):  # step by 20 to avoid overlapping
            if pd.isna(highs_20d.iloc[i]):
                continue
            current = df.iloc[i]["Close"]
            drop = (current - highs_20d.iloc[i]) / highs_20d.iloc[i]
            if drop <= -0.20:
                entry_price = apply_slippage(current, "buy")
                exit_idx = i + 60
                exit_price = apply_slippage(df.iloc[exit_idx]["Close"], "sell")

                shares = max(1, int(ACCOUNT_SIZE / entry_price))
                pnl = (exit_price - entry_price) * shares
                ret_pct = (exit_price - entry_price) / entry_price * 100

                trades.append({
                    "ticker": ticker,
                    "entry_date": str(df.index[i].date()),
                    "exit_date": str(df.index[exit_idx].date()),
                    "entry_price": round(entry_price, 2),
                    "exit_price": round(exit_price, 2),
                    "pnl": round(pnl, 2),
                    "return_pct": round(ret_pct, 2),
                    "hold_days": 60,
                    "signal": f"Down {drop*100:.0f}% from 20d high",
                })

    return pd.DataFrame(trades)


def variant_b_forced_selling_recovery(data, tickers):
    """
    Variant B: Forced Selling Recovery
    Stock drops 25-40% in 5 days on < 2x avg volume. Buy and hold 40 days.
    """
    trades = []
    for ticker in tickers:
        if ticker not in data:
            continue
        df = data[ticker]
        if len(df) < 60:
            continue

        avg_vol = df["Volume"].rolling(50).mean()

        for i in range(50, len(df) - 40):
            # 5-day return
            ret_5d = (df.iloc[i]["Close"] - df.iloc[i-5]["Close"]) / df.iloc[i-5]["Close"]
            # Average volume over those 5 days
            vol_5d = df.iloc[i-4:i+1]["Volume"].mean()
            avg = avg_vol.iloc[i]

            if pd.isna(avg) or avg == 0:
                continue

            vol_ratio = vol_5d / avg

            # Drop 25-40%, volume < 2x normal (not earnings/news driven)
            if -0.40 <= ret_5d <= -0.25 and vol_ratio < 2.0:
                entry_price = apply_slippage(df.iloc[i]["Close"], "buy")
                exit_idx = min(i + 40, len(df) - 1)
                exit_price = apply_slippage(df.iloc[exit_idx]["Close"], "sell")

                shares = max(1, int(ACCOUNT_SIZE / entry_price))
                pnl = (exit_price - entry_price) * shares
                ret_pct = (exit_price - entry_price) / entry_price * 100

                trades.append({
                    "ticker": ticker,
                    "entry_date": str(df.index[i].date()),
                    "exit_date": str(df.index[exit_idx].date()),
                    "entry_price": round(entry_price, 2),
                    "exit_price": round(exit_price, 2),
                    "pnl": round(pnl, 2),
                    "return_pct": round(ret_pct, 2),
                    "hold_days": exit_idx - i,
                    "signal": f"5d drop {ret_5d*100:.1f}%, vol ratio {vol_ratio:.1f}x",
                })
                # Skip ahead to avoid overlapping trades on same ticker
                # (handled by step in loop — but let's add a cooldown)

    # Deduplicate: no overlapping trades per ticker
    if trades:
        trades_df = pd.DataFrame(trades)
        deduped = []
        for ticker in trades_df["ticker"].unique():
            ticker_trades = trades_df[trades_df["ticker"] == ticker].sort_values("entry_date")
            last_exit = None
            for _, row in ticker_trades.iterrows():
                if last_exit is None or row["entry_date"] > last_exit:
                    deduped.append(row.to_dict())
                    last_exit = row["exit_date"]
        return pd.DataFrame(deduped)
    return pd.DataFrame(trades)


def variant_c_new_listing_momentum(data, tickers):
    """
    Variant C: New Listing Momentum
    Buy stocks 20%+ above their first-week low at day 30. Hold 40 days.
    """
    trades = []
    for ticker in tickers:
        if ticker not in data:
            continue
        df = data[ticker]
        if len(df) < 70:
            continue

        first_week = df.iloc[:5]
        first_week_low = first_week["Low"].min()

        # Check at day 30
        day30_idx = min(29, len(df) - 41)
        price_day30 = df.iloc[day30_idx]["Close"]
        gain_pct = (price_day30 - first_week_low) / first_week_low

        if gain_pct >= 0.20:
            entry_price = apply_slippage(df.iloc[day30_idx]["Close"], "buy")
            exit_idx = day30_idx + 40
            if exit_idx >= len(df):
                exit_idx = len(df) - 1
            exit_price = apply_slippage(df.iloc[exit_idx]["Close"], "sell")

            shares = max(1, int(ACCOUNT_SIZE / entry_price))
            pnl = (exit_price - entry_price) * shares
            ret_pct = (exit_price - entry_price) / entry_price * 100

            trades.append({
                "ticker": ticker,
                "entry_date": str(df.index[day30_idx].date()),
                "exit_date": str(df.index[exit_idx].date()),
                "entry_price": round(entry_price, 2),
                "exit_price": round(exit_price, 2),
                "pnl": round(pnl, 2),
                "return_pct": round(ret_pct, 2),
                "hold_days": exit_idx - day30_idx,
                "signal": f"Up {gain_pct*100:.0f}% from first-week low at day 30",
            })

    return pd.DataFrame(trades)


def variant_d_spinoff_basket(data, tickers):
    """
    Variant D: Spin-Off Basket
    Equal-weight basket of known spin-offs, rebalanced monthly.
    Benchmark against SPY.
    """
    # Use only tickers with data
    available = [t for t in tickers if t in data and len(data[t]) > 60]
    if len(available) < 3:
        return pd.DataFrame()

    # Build monthly rebalanced basket
    trades = []
    # Get common date range
    all_dates = None
    for ticker in available:
        idx = data[ticker].index
        if all_dates is None:
            all_dates = idx
        else:
            all_dates = all_dates.intersection(idx)

    if len(all_dates) < 60:
        return pd.DataFrame()

    # Monthly rebalance dates
    monthly = pd.Series(all_dates).groupby([all_dates.year, all_dates.month]).first().values
    monthly = pd.DatetimeIndex(monthly)

    for i in range(len(monthly) - 1):
        start_date = monthly[i]
        end_date = monthly[i + 1]

        for ticker in available:
            df = data[ticker]
            mask = (df.index >= start_date) & (df.index < end_date)
            period = df.loc[mask]
            if len(period) < 2:
                continue

            weight = 1.0 / len(available)
            alloc = ACCOUNT_SIZE * weight
            entry_price = apply_slippage(period.iloc[0]["Close"], "buy")
            exit_price = apply_slippage(period.iloc[-1]["Close"], "sell")
            shares = max(1, int(alloc / entry_price)) if entry_price > 0 else 0
            if shares == 0:
                continue

            pnl = (exit_price - entry_price) * shares
            ret_pct = (exit_price - entry_price) / entry_price * 100

            trades.append({
                "ticker": ticker,
                "entry_date": str(period.index[0].date()),
                "exit_date": str(period.index[-1].date()),
                "entry_price": round(entry_price, 2),
                "exit_price": round(exit_price, 2),
                "pnl": round(pnl, 2),
                "return_pct": round(ret_pct, 2),
                "hold_days": len(period),
                "signal": "Monthly rebalance basket",
            })

    return pd.DataFrame(trades)


def variant_e_contrarian_extreme(data, tickers):
    """
    Variant E: Contrarian Extreme
    Stocks down 50%+ from 52-week high with recent volume normalization.
    Buy and hold 60 days.
    """
    trades = []
    # Expand universe: use all tickers
    for ticker in tickers:
        if ticker not in data:
            continue
        df = data[ticker]
        if len(df) < 260:
            continue

        high_252 = df["High"].rolling(252).max()
        avg_vol = df["Volume"].rolling(50).mean()
        recent_vol = df["Volume"].rolling(5).mean()

        for i in range(260, len(df) - 60, 10):  # step by 10 to reduce overlaps
            h = high_252.iloc[i]
            c = df.iloc[i]["Close"]
            if pd.isna(h) or h == 0:
                continue
            drawdown = (c - h) / h

            av = avg_vol.iloc[i]
            rv = recent_vol.iloc[i]
            if pd.isna(av) or pd.isna(rv) or av == 0:
                continue
            vol_norm = rv / av

            # Down 50%+ from 52w high, volume normalizing (< 1.5x)
            if drawdown <= -0.50 and vol_norm < 1.5:
                entry_price = apply_slippage(c, "buy")
                exit_idx = i + 60
                exit_price = apply_slippage(df.iloc[exit_idx]["Close"], "sell")

                shares = max(1, int(ACCOUNT_SIZE / entry_price))
                pnl = (exit_price - entry_price) * shares
                ret_pct = (exit_price - entry_price) / entry_price * 100

                trades.append({
                    "ticker": ticker,
                    "entry_date": str(df.index[i].date()),
                    "exit_date": str(df.index[exit_idx].date()),
                    "entry_price": round(entry_price, 2),
                    "exit_price": round(exit_price, 2),
                    "pnl": round(pnl, 2),
                    "return_pct": round(ret_pct, 2),
                    "hold_days": 60,
                    "signal": f"Down {drawdown*100:.0f}% from 52w high, vol norm {vol_norm:.1f}x",
                })

    # Deduplicate per ticker
    if trades:
        trades_df = pd.DataFrame(trades)
        deduped = []
        for ticker in trades_df["ticker"].unique():
            ticker_trades = trades_df[trades_df["ticker"] == ticker].sort_values("entry_date")
            last_exit = None
            for _, row in ticker_trades.iterrows():
                if last_exit is None or row["entry_date"] > last_exit:
                    deduped.append(row.to_dict())
                    last_exit = row["exit_date"]
        return pd.DataFrame(deduped)
    return pd.DataFrame(trades)


def variant_f_sector_rotation(data, tickers):
    """
    Variant F: Sector Rotation After Forced Selling
    When 3+ stocks in a sector drop 20%+ in 20 days, buy sector ETF for 20 days.
    """
    trades = []

    # Build sector groups
    sectors = {}
    for ticker in tickers:
        if ticker in TICKER_SECTORS and ticker in data:
            sector = TICKER_SECTORS[ticker]
            if sector not in sectors:
                sectors[sector] = []
            sectors[sector].append(ticker)

    # For each sector, check for forced selling events
    for sector, sector_tickers in sectors.items():
        etf = SECTOR_ETFS.get(sector)
        if etf is None or etf not in data:
            continue
        etf_data = data[etf]

        # Get common dates
        for check_idx in range(20, len(etf_data) - 20, 5):
            check_date = etf_data.index[check_idx]
            start_window = etf_data.index[max(0, check_idx - 20)]

            # Count stocks with 20%+ drop in last 20 days
            drop_count = 0
            for ticker in sector_tickers:
                if ticker not in data:
                    continue
                df = data[ticker]
                mask_start = df.index[df.index <= start_window]
                mask_end = df.index[df.index <= check_date]
                if len(mask_start) == 0 or len(mask_end) == 0:
                    continue
                p_start = df.loc[mask_start[-1], "Close"]
                p_end = df.loc[mask_end[-1], "Close"]
                if p_start > 0:
                    ret = (p_end - p_start) / p_start
                    if ret <= -0.20:
                        drop_count += 1

            if drop_count >= 3:
                entry_price = apply_slippage(etf_data.iloc[check_idx]["Close"], "buy")
                exit_idx = min(check_idx + 20, len(etf_data) - 1)
                exit_price = apply_slippage(etf_data.iloc[exit_idx]["Close"], "sell")

                shares = max(1, int(ACCOUNT_SIZE / entry_price))
                pnl = (exit_price - entry_price) * shares
                ret_pct = (exit_price - entry_price) / entry_price * 100

                trades.append({
                    "ticker": etf,
                    "entry_date": str(etf_data.index[check_idx].date()),
                    "exit_date": str(etf_data.index[exit_idx].date()),
                    "entry_price": round(entry_price, 2),
                    "exit_price": round(exit_price, 2),
                    "pnl": round(pnl, 2),
                    "return_pct": round(ret_pct, 2),
                    "hold_days": exit_idx - check_idx,
                    "signal": f"{drop_count} stocks in {sector} down 20%+",
                })

    # Deduplicate
    if trades:
        trades_df = pd.DataFrame(trades)
        deduped = []
        for ticker in trades_df["ticker"].unique():
            ticker_trades = trades_df[trades_df["ticker"] == ticker].sort_values("entry_date")
            last_exit = None
            for _, row in ticker_trades.iterrows():
                if last_exit is None or row["entry_date"] > last_exit:
                    deduped.append(row.to_dict())
                    last_exit = row["exit_date"]
        return pd.DataFrame(deduped)
    return pd.DataFrame(trades)


# ─── 5-Gate Validation ──────────────────────────────────────────────────────

def validate_5gates(metrics, perm_p, regime_gap):
    """Apply 5-gate validation framework."""
    gates = {
        "G1_Sharpe_gt_0.5": {"pass": metrics["sharpe"] > 0.5, "value": metrics["sharpe"], "threshold": "> 0.5"},
        "G2_Permutation_p_lt_0.05": {"pass": perm_p < 0.05, "value": round(perm_p, 4), "threshold": "< 0.05"},
        "G3_Regime_gap_lt_0.5": {"pass": regime_gap < 0.5, "value": regime_gap, "threshold": "< 0.5"},
        "G4_MaxDD_gt_neg50": {"pass": metrics["max_dd_pct"] > -50, "value": metrics["max_dd_pct"], "threshold": "> -50%"},
        "G5_Trades_gte_20": {"pass": metrics["n_trades"] >= 20, "value": metrics["n_trades"], "threshold": ">= 20"},
    }
    gates_passed = sum(1 for g in gates.values() if g["pass"])
    return gates, gates_passed


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("SPIN-OFF / SPECIAL SITUATION EVENT BACKTEST")
    print(f"OOT: {OOT_START} to {OOT_END} | Account: ${ACCOUNT_SIZE} | Slippage: {SLIPPAGE_PCT*100:.2f}% each way")
    print("=" * 80)

    # Download data
    data = download_data(SPINOFF_TICKERS, OOT_START, OOT_END)
    spy_data = data.get("SPY")

    # Run all variants
    variants = {
        "A_PostIPO_Recovery": variant_a_post_ipo_recovery,
        "B_Forced_Selling_Recovery": variant_b_forced_selling_recovery,
        "C_New_Listing_Momentum": variant_c_new_listing_momentum,
        "D_Spinoff_Basket": variant_d_spinoff_basket,
        "E_Contrarian_Extreme": variant_e_contrarian_extreme,
        "F_Sector_Rotation": variant_f_sector_rotation,
    }

    results = {}

    for name, func in variants.items():
        print(f"\n{'─' * 70}")
        print(f"Variant {name}")
        print(f"{'─' * 70}")

        trades_df = func(data, SPINOFF_TICKERS)

        if trades_df is None or len(trades_df) == 0:
            print(f"  No trades generated.")
            results[name] = {
                "metrics": {"sharpe": 0, "sortino": 0, "pf": 0, "wr": 0,
                           "max_dd_pct": 0, "n_trades": 0, "total_return_pct": 0},
                "gates": {},
                "gates_passed": 0,
                "trades": [],
            }
            continue

        # Compute daily returns from trades
        returns = trades_df["return_pct"].values / 100.0
        metrics = compute_metrics(returns, trades_df)

        # Permutation test
        perm_p = permutation_test(returns)

        # Regime analysis
        regime_gap = regime_analysis(trades_df, spy_data)

        # 5-gate validation
        gates, gates_passed = validate_5gates(metrics, perm_p, regime_gap)

        # Print results
        print(f"  Trades: {metrics['n_trades']} | Sharpe: {metrics['sharpe']} | "
              f"Sortino: {metrics['sortino']} | PF: {metrics['pf']} | WR: {metrics['wr']}%")
        print(f"  Total Return: {metrics['total_return_pct']}% | MaxDD: {metrics['max_dd_pct']}% | "
              f"Avg Return/Trade: {metrics['avg_return_pct']}%")
        print(f"  Median Hold: {metrics['median_hold_days']} days")
        print(f"  Permutation p-value: {perm_p:.4f} | Regime Gap: {regime_gap}")
        print()
        print(f"  5-GATE VALIDATION: {'PASS' if gates_passed == 5 else 'FAIL'} ({gates_passed}/5)")
        for gname, gdata in gates.items():
            status = "PASS" if gdata["pass"] else "FAIL"
            print(f"    [{status}] {gname}: {gdata['value']} ({gdata['threshold']})")

        # Top trades
        if len(trades_df) > 0:
            print(f"\n  Top 5 trades:")
            top = trades_df.nlargest(5, "pnl")
            for _, t in top.iterrows():
                print(f"    {t['ticker']} {t['entry_date']}→{t['exit_date']}: "
                      f"${t['pnl']:+.2f} ({t['return_pct']:+.1f}%) | {t['signal']}")

            print(f"\n  Bottom 5 trades:")
            bottom = trades_df.nsmallest(5, "pnl")
            for _, t in bottom.iterrows():
                print(f"    {t['ticker']} {t['entry_date']}→{t['exit_date']}: "
                      f"${t['pnl']:+.2f} ({t['return_pct']:+.1f}%) | {t['signal']}")

        results[name] = {
            "metrics": metrics,
            "gates": {k: {"pass": v["pass"], "value": v["value"], "threshold": v["threshold"]}
                     for k, v in gates.items()},
            "gates_passed": gates_passed,
            "perm_p": round(perm_p, 4),
            "regime_gap": regime_gap,
            "trades": trades_df.to_dict(orient="records") if len(trades_df) <= 200 else
                      trades_df.head(200).to_dict(orient="records"),
        }

    # ─── Summary ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SUMMARY — ALL VARIANTS")
    print("=" * 80)
    print(f"{'Variant':<30} {'Trades':>6} {'Sharpe':>7} {'Sortino':>8} {'PF':>6} "
          f"{'WR%':>5} {'TotRet%':>8} {'MaxDD%':>7} {'Gates':>5}")
    print("-" * 90)

    any_passed = False
    for name, res in results.items():
        m = res["metrics"]
        gp = res["gates_passed"]
        tag = " <<<" if gp == 5 else ""
        if gp == 5:
            any_passed = True
        print(f"  {name:<28} {m['n_trades']:>6} {m['sharpe']:>7.3f} {m['sortino']:>8.3f} "
              f"{m['pf']:>6.2f} {m['wr']:>5.1f} {m['total_return_pct']:>8.2f} "
              f"{m['max_dd_pct']:>7.2f} {gp}/5{tag}")

    if not any_passed:
        print("\n  ** No variant passed all 5 gates. **")
    else:
        print("\n  ** Variants marked <<< passed all 5 gates. **")

    # ─── Save Results ────────────────────────────────────────────────────────
    output_path = Path("/home/jupiter/Lvl3Quant/data/spinoff_event_results.json")
    output = {
        "strategy": "Spin-Off / Special Situation Event Trading",
        "oot_period": f"{OOT_START} to {OOT_END}",
        "account_size": ACCOUNT_SIZE,
        "slippage_pct": SLIPPAGE_PCT,
        "run_timestamp": dt.datetime.now().isoformat(),
        "variants": results,
    }
    output_path.write_text(json.dumps(output, indent=2, default=str))
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
