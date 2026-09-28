#!/usr/bin/env python3
"""
Put-Call Ratio Sector Divergence Backtest
=========================================
Hypothesis: When a sector ETF's sentiment proxy diverges significantly from
the broad market (SPY), it signals unusual positioning that can be traded.

Sub-hypotheses:
  H1 (Mean-reversion): Extreme bearish sector divergence → BUY (contrarian)
  H2 (Momentum): Extreme bullish sector divergence → BUY (follow smart money)
  H2b (Fade bullish): Extreme bullish sector divergence → SELL

Sentiment proxy: Since direct put-call ratio data isn't available via yfinance,
we use a composite of:
  - Down-volume ratio: fraction of recent days with negative returns
  - Volatility asymmetry: realized vol on down days vs up days
  - Range compression: ATR on down days / ATR on up days

The divergence z-score measures how much more bearish (or bullish) a sector
is relative to SPY over a rolling window.

Author: Claude Opus 4.6
Date: 2026-08-18
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import warnings
import sys
import os

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================
SECTOR_ETFS = ["XLE", "XLU", "XLF", "XLK", "XLY", "XLP", "XLB", "XLI", "XLV", "XLRE", "XLC"]
MARKET_TICKER = "SPY"
START_DATE = "2019-01-01"
END_DATE = "2026-08-15"

LOOKBACK = 20          # Rolling window for sentiment proxy (trading days)
ZSCORE_LOOKBACK = 60   # Rolling window for z-score normalization
HOLD_DAYS = 4          # 3-5 day hold (using 4 as midpoint)
ZSCORE_THRESHOLD = 1.5 # Entry threshold

# 5-gate thresholds
MIN_SHARPE = 0.5
MAX_PVALUE = 0.05
MAX_REGIME_GAP = 0.50
MAX_DRAWDOWN = 0.50
MIN_TRADES = 30

N_PERMUTATIONS = 1000
SEED = 42

np.random.seed(SEED)


# ============================================================================
# DATA FETCHING
# ============================================================================
def fetch_data():
    """Download OHLCV data for all tickers."""
    tickers = SECTOR_ETFS + [MARKET_TICKER]
    print(f"Fetching data for {len(tickers)} tickers: {START_DATE} to {END_DATE}")

    data = {}
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False, auto_adjust=True)
            if df.empty:
                print(f"  WARNING: No data for {ticker}")
                continue
            # Flatten multi-level columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            data[ticker] = df
            print(f"  {ticker}: {len(df)} days ({df.index[0].date()} to {df.index[-1].date()})")
        except Exception as e:
            print(f"  ERROR fetching {ticker}: {e}")

    return data


# ============================================================================
# SENTIMENT PROXY COMPUTATION
# ============================================================================
def compute_sentiment_proxy(df, lookback=LOOKBACK):
    """
    Compute a bearishness proxy from price/volume data.

    Components:
      1. Down-day ratio: fraction of days with negative returns
      2. Down-vol ratio: realized vol on down days / overall vol
      3. Down-range ratio: avg range on down days / avg range on up days

    Higher value = more bearish sentiment.
    """
    ret = df["Close"].pct_change()
    is_down = (ret < 0).astype(float)
    daily_range = (df["High"] - df["Low"]) / df["Close"]

    # Component 1: Down-day ratio (rolling)
    down_ratio = is_down.rolling(lookback, min_periods=lookback).mean()

    # Component 2: Down-volatility ratio
    # Realized vol on down days vs all days
    down_ret_sq = (ret * is_down) ** 2
    all_ret_sq = ret ** 2
    down_vol = down_ret_sq.rolling(lookback, min_periods=lookback).sum().apply(np.sqrt)
    all_vol = all_ret_sq.rolling(lookback, min_periods=lookback).sum().apply(np.sqrt)
    vol_ratio = down_vol / all_vol.replace(0, np.nan)

    # Component 3: Down-range ratio
    down_range = (daily_range * is_down).rolling(lookback, min_periods=lookback).sum()
    up_range = (daily_range * (1 - is_down)).rolling(lookback, min_periods=lookback).sum()
    range_ratio = down_range / up_range.replace(0, np.nan)

    # Composite: equal-weight average of z-scored components
    proxy = (down_ratio + vol_ratio + range_ratio) / 3.0
    return proxy


def compute_divergence_zscore(sector_proxy, market_proxy, zscore_lookback=ZSCORE_LOOKBACK):
    """
    Compute the z-score of (sector_proxy - market_proxy).

    Positive z-score = sector is MORE bearish than market.
    Negative z-score = sector is MORE bullish than market.
    """
    divergence = sector_proxy - market_proxy
    roll_mean = divergence.rolling(zscore_lookback, min_periods=zscore_lookback).mean()
    roll_std = divergence.rolling(zscore_lookback, min_periods=zscore_lookback).std()
    zscore = (divergence - roll_mean) / roll_std.replace(0, np.nan)
    return zscore


# ============================================================================
# SIGNAL GENERATION & BACKTESTING
# ============================================================================
def generate_signals(zscore_series, close_series, spy_close, hold_days=HOLD_DAYS,
                     threshold=ZSCORE_THRESHOLD, signal_type="contrarian_buy"):
    """
    Generate trade signals from z-score divergence.

    signal_type:
      - "contrarian_buy": z > threshold → BUY (mean-reversion on excessive fear)
      - "momentum_buy":   z < -threshold → BUY (follow bullish divergence)
      - "fade_bullish":   z < -threshold → SELL (fade bullish excess)
    """
    trades = []
    i = 0
    dates = zscore_series.index
    n = len(dates)

    while i < n - hold_days:
        z = zscore_series.iloc[i]
        if np.isnan(z):
            i += 1
            continue

        trigger = False
        direction = 1  # 1 = long, -1 = short

        if signal_type == "contrarian_buy" and z > threshold:
            trigger = True
            direction = 1
        elif signal_type == "momentum_buy" and z < -threshold:
            trigger = True
            direction = 1
        elif signal_type == "fade_bullish" and z < -threshold:
            trigger = True
            direction = -1

        if trigger:
            entry_date = dates[i]
            exit_idx = min(i + hold_days, n - 1)
            exit_date = dates[exit_idx]

            entry_price = close_series.iloc[i]
            exit_price = close_series.iloc[exit_idx]

            if np.isnan(entry_price) or np.isnan(exit_price) or entry_price == 0:
                i += 1
                continue

            raw_return = (exit_price / entry_price - 1.0) * direction

            # Classify regime: was SPY green or red on entry day?
            spy_ret = spy_close.pct_change()
            if entry_date in spy_ret.index and not np.isnan(spy_ret.loc[entry_date]):
                regime = "green" if spy_ret.loc[entry_date] >= 0 else "red"
            else:
                regime = "unknown"

            trades.append({
                "entry_date": entry_date,
                "exit_date": exit_date,
                "direction": "LONG" if direction == 1 else "SHORT",
                "entry_price": entry_price,
                "exit_price": exit_price,
                "return": raw_return,
                "zscore": z,
                "regime": regime,
            })

            # Skip hold_days to avoid overlapping trades in same sector
            i += hold_days
        else:
            i += 1

    return trades


def compute_metrics(trades_df):
    """Compute Sharpe, win rate, profit factor, max drawdown from trade returns."""
    if len(trades_df) == 0:
        return {"sharpe": 0, "win_rate": 0, "profit_factor": 0, "max_dd": 1.0,
                "n_trades": 0, "mean_return": 0, "total_return": 0}

    rets = trades_df["return"].values
    n = len(rets)

    mean_ret = np.mean(rets)
    std_ret = np.std(rets, ddof=1) if n > 1 else 1e-9

    # Annualize: ~252/hold_days trades per year if fully invested
    trades_per_year = 252 / HOLD_DAYS
    sharpe = (mean_ret / std_ret) * np.sqrt(trades_per_year) if std_ret > 0 else 0

    wins = rets[rets > 0]
    losses = rets[rets <= 0]
    win_rate = len(wins) / n if n > 0 else 0

    gross_profit = np.sum(wins) if len(wins) > 0 else 0
    gross_loss = np.abs(np.sum(losses)) if len(losses) > 0 else 1e-9
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Max drawdown from cumulative equity curve
    cum = np.cumsum(rets)
    running_max = np.maximum.accumulate(cum)
    drawdowns = running_max - cum
    max_dd = np.max(drawdowns) if len(drawdowns) > 0 else 0

    # Express max DD as fraction of peak equity (assume starting equity = 1)
    equity = 1.0 + cum
    peak_equity = np.maximum.accumulate(equity)
    dd_pct = np.max((peak_equity - equity) / peak_equity) if len(equity) > 0 else 0

    total_return = np.sum(rets)

    return {
        "sharpe": round(sharpe, 3),
        "win_rate": round(win_rate, 4),
        "profit_factor": round(profit_factor, 3),
        "max_dd": round(dd_pct, 4),
        "n_trades": n,
        "mean_return": round(mean_ret * 100, 4),  # in %
        "total_return": round(total_return * 100, 2),  # in %
    }


def permutation_test(trades_df, all_fwd_returns, n_perms=N_PERMUTATIONS):
    """
    Permutation test: compare observed Sharpe against random entry timing.

    For each permutation, sample len(trades) random forward returns from the
    full universe of possible entries (all_fwd_returns). This tests whether
    the signal's timing adds value vs random entry.

    p-value = fraction of permuted Sharpes >= observed Sharpe.
    """
    if len(trades_df) < 5:
        return 1.0

    n_trades = len(trades_df)
    observed_sharpe = compute_metrics(trades_df)["sharpe"]

    # Pool of all possible forward returns
    pool = all_fwd_returns[np.isfinite(all_fwd_returns)]
    if len(pool) < n_trades:
        return 1.0

    count_ge = 0
    for _ in range(n_perms):
        sampled = np.random.choice(pool, size=n_trades, replace=True)
        mean_s = np.mean(sampled)
        std_s = np.std(sampled, ddof=1)
        if std_s > 0:
            perm_sharpe = (mean_s / std_s) * np.sqrt(252 / HOLD_DAYS)
        else:
            perm_sharpe = 0
        if perm_sharpe >= observed_sharpe:
            count_ge += 1

    return count_ge / n_perms


def regime_analysis(trades_df):
    """Compute Sharpe by regime (green/red SPY days) and regime gap."""
    green = trades_df[trades_df["regime"] == "green"]
    red = trades_df[trades_df["regime"] == "red"]

    green_metrics = compute_metrics(green)
    red_metrics = compute_metrics(red)

    s_green = green_metrics["sharpe"]
    s_red = red_metrics["sharpe"]
    denom = max(abs(s_green), abs(s_red), 1e-9)
    regime_gap = abs(s_green - s_red) / denom

    return {
        "green_sharpe": s_green,
        "red_sharpe": s_red,
        "green_n": green_metrics["n_trades"],
        "red_n": red_metrics["n_trades"],
        "regime_gap": round(regime_gap, 3),
    }


def apply_five_gates(metrics, pvalue, regime_info):
    """Apply the 5-gate acceptance system."""
    gates = {
        "G1_Sharpe": (metrics["sharpe"] > MIN_SHARPE, f"{metrics['sharpe']:.3f} > {MIN_SHARPE}"),
        "G2_Pvalue": (pvalue < MAX_PVALUE, f"{pvalue:.4f} < {MAX_PVALUE}"),
        "G3_RegimeGap": (regime_info["regime_gap"] < MAX_REGIME_GAP,
                         f"{regime_info['regime_gap']:.3f} < {MAX_REGIME_GAP}"),
        "G4_MaxDD": (metrics["max_dd"] < MAX_DRAWDOWN, f"{metrics['max_dd']:.4f} < {MAX_DRAWDOWN}"),
        "G5_MinTrades": (metrics["n_trades"] >= MIN_TRADES, f"{metrics['n_trades']} >= {MIN_TRADES}"),
    }
    all_pass = all(v[0] for v in gates.values())
    return gates, all_pass


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 80)
    print("PUT-CALL RATIO SECTOR DIVERGENCE BACKTEST")
    print("=" * 80)
    print()

    # 1. Fetch data
    data = fetch_data()
    if MARKET_TICKER not in data:
        print("FATAL: Could not fetch SPY data")
        sys.exit(1)

    spy_df = data[MARKET_TICKER]
    spy_proxy = compute_sentiment_proxy(spy_df)
    spy_close = spy_df["Close"]

    # 2. Run all three signal types across all sectors
    signal_types = ["contrarian_buy", "momentum_buy", "fade_bullish"]
    results = {}

    for sig_type in signal_types:
        all_trades = []

        for sector in SECTOR_ETFS:
            if sector not in data:
                continue

            sector_df = data[sector]
            sector_proxy = compute_sentiment_proxy(sector_df)

            # Align dates
            common_idx = sector_proxy.index.intersection(spy_proxy.index)
            if len(common_idx) < ZSCORE_LOOKBACK + LOOKBACK + 10:
                continue

            sp = sector_proxy.reindex(common_idx)
            mp = spy_proxy.reindex(common_idx)
            zscore = compute_divergence_zscore(sp, mp)

            sector_close = sector_df["Close"].reindex(common_idx)

            trades = generate_signals(
                zscore, sector_close, spy_close,
                hold_days=HOLD_DAYS, threshold=ZSCORE_THRESHOLD,
                signal_type=sig_type
            )

            for t in trades:
                t["sector"] = sector

            all_trades.extend(trades)

        if not all_trades:
            results[sig_type] = {"trades_df": pd.DataFrame(), "metrics": compute_metrics(pd.DataFrame())}
            continue

        trades_df = pd.DataFrame(all_trades)
        trades_df = trades_df.sort_values("entry_date").reset_index(drop=True)
        results[sig_type] = {"trades_df": trades_df}

    # 2b. Compute pool of ALL possible forward returns (for permutation test)
    all_fwd_returns = []
    for sector in SECTOR_ETFS:
        if sector not in data:
            continue
        close = data[sector]["Close"]
        fwd = close.shift(-HOLD_DAYS) / close - 1.0
        all_fwd_returns.append(fwd.dropna().values)
    all_fwd_returns = np.concatenate(all_fwd_returns) if all_fwd_returns else np.array([])

    # 3. Compute metrics and run tests
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    for sig_type in signal_types:
        trades_df = results[sig_type]["trades_df"]
        print(f"\n{'─' * 70}")
        print(f"SIGNAL: {sig_type.upper()}")
        print(f"  Z-score threshold: {ZSCORE_THRESHOLD}, Hold: {HOLD_DAYS} days, Lookback: {LOOKBACK}d")
        print(f"{'─' * 70}")

        if len(trades_df) == 0:
            print("  No trades generated.")
            continue

        metrics = compute_metrics(trades_df)
        results[sig_type]["metrics"] = metrics

        print(f"\n  Trade count:    {metrics['n_trades']}")
        print(f"  Mean return:    {metrics['mean_return']:.4f}%")
        print(f"  Total return:   {metrics['total_return']:.2f}%")
        print(f"  Sharpe ratio:   {metrics['sharpe']:.3f}")
        print(f"  Win rate:       {metrics['win_rate']:.1%}")
        print(f"  Profit factor:  {metrics['profit_factor']:.3f}")
        print(f"  Max drawdown:   {metrics['max_dd']:.2%}")

        # Sector breakdown
        print(f"\n  Trades per sector:")
        sector_counts = trades_df["sector"].value_counts()
        for sec, cnt in sector_counts.items():
            sec_rets = trades_df[trades_df["sector"] == sec]["return"]
            print(f"    {sec}: {cnt} trades, mean ret {sec_rets.mean()*100:.3f}%, "
                  f"WR {(sec_rets > 0).mean():.1%}")

        # Regime analysis
        regime_info = regime_analysis(trades_df)
        print(f"\n  Regime stratification:")
        print(f"    Green days: Sharpe={regime_info['green_sharpe']:.3f} ({regime_info['green_n']} trades)")
        print(f"    Red days:   Sharpe={regime_info['red_sharpe']:.3f} ({regime_info['red_n']} trades)")
        print(f"    Regime gap: {regime_info['regime_gap']:.3f}")

        # Permutation test
        # For short signals, use negated forward returns as pool
        if sig_type == "fade_bullish":
            perm_pool = -all_fwd_returns
        else:
            perm_pool = all_fwd_returns
        print(f"\n  Running permutation test ({N_PERMUTATIONS} shuffles)...", end=" ", flush=True)
        pvalue = permutation_test(trades_df, perm_pool)
        print(f"p-value = {pvalue:.4f}")

        # 5-gate system
        gates, all_pass = apply_five_gates(metrics, pvalue, regime_info)
        print(f"\n  5-GATE EVALUATION:")
        for gate_name, (passed, detail) in gates.items():
            status = "PASS" if passed else "FAIL"
            print(f"    [{status}] {gate_name}: {detail}")

        verdict = "PASS - VIABLE STRATEGY" if all_pass else "FAIL - NOT VIABLE"
        print(f"\n  VERDICT: {verdict}")

    # 4. Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for sig_type in signal_types:
        m = results[sig_type].get("metrics", {})
        n = m.get("n_trades", 0)
        sharpe = m.get("sharpe", 0)
        wr = m.get("win_rate", 0)
        print(f"  {sig_type:20s}: {n:4d} trades, Sharpe={sharpe:+.3f}, WR={wr:.1%}")

    print("\n" + "=" * 80)
    print("HONEST ASSESSMENT")
    print("=" * 80)
    any_pass = any(
        results[s].get("metrics", {}).get("n_trades", 0) >= MIN_TRADES and
        results[s].get("metrics", {}).get("sharpe", 0) > MIN_SHARPE
        for s in signal_types
    )
    if any_pass:
        print("  At least one signal type shows promise. Check gates carefully.")
        print("  Remember: this uses a PROXY for put-call ratio, not actual options flow data.")
        print("  Real edge would require actual CBOE put-call ratio or options volume data.")
    else:
        print("  No signal type passed the Sharpe threshold.")
        print("  This is expected -- most ideas fail. The sentiment proxy from price/volume")
        print("  alone is likely too noisy to capture the information that actual put-call")
        print("  ratio data would provide. The hypothesis may still be valid with better data")
        print("  (CBOE options volume, dealer positioning, GEX), but the freely available")
        print("  proxy tested here does not produce tradeable edge.")

    print("\nDone.")


if __name__ == "__main__":
    main()
