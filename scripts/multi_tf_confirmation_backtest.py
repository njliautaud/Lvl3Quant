#!/usr/bin/env python3
"""
Multi-Timeframe Oversold Confirmation Backtest
Tests whether requiring oversold conditions on MULTIPLE timeframes simultaneously
improves mean-reversion quality on quality stocks.

Variants:
  A: Daily + Weekly RSI Oversold (tight thresholds)
  B: Triple Timeframe (daily + weekly + monthly RSI)
  C: Daily Oversold + Weekly Downtrend Exhaustion (RSI inflection)
  D: Bollinger Band Multi-TF (daily + weekly BB breach)
  E: Moving Average Multi-TF (below 20/50 SMA, above 200 SMA, RSI<35)
  F: Rate of Change Multi-TF (cascading ROC oversold)
"""

import json
import datetime
import warnings
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ── Parameters ──────────────────────────────────────────────────────────────
UNIVERSE = [
    "AAPL", "MSFT", "AVGO", "JPM", "JNJ", "PG", "KO", "PEP", "HD", "COST",
    "UNH", "LLY", "V", "MA", "ABBV", "MRK", "WMT", "AMZN", "GOOGL", "META",
]
CAPITAL = 669.0
MAX_PER_TRADE = 200.0
MAX_CONCURRENT = 3
SLIPPAGE_BPS = 2
BACKTEST_START = "2022-01-01"
BACKTEST_END = "2026-07-31"
DOWNLOAD_START = "2021-01-01"  # extra lookback for monthly indicators
N_PERMUTATIONS = 1000
SEED = 42

# ── Data Download ───────────────────────────────────────────────────────────
print("Downloading price data...")
tickers_to_dl = UNIVERSE + ["SPY"]
raw = yf.download(tickers_to_dl, start=DOWNLOAD_START, end=BACKTEST_END,
                  auto_adjust=True, progress=False)

# Handle multi-level columns from yfinance
close_df = raw["Close"].copy()
high_df = raw["High"].copy()
low_df = raw["Low"].copy()

# SPY for regime classification
spy_close = close_df["SPY"].dropna()

# Drop SPY from trading universe
for df in [close_df, high_df, low_df]:
    if "SPY" in df.columns:
        df.drop(columns=["SPY"], inplace=True)


# ── Indicator Helpers ───────────────────────────────────────────────────────
def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def bollinger_bands(series, period=20, num_std=2):
    sma = series.rolling(period).mean()
    std = series.rolling(period).std()
    lower = sma - num_std * std
    upper = sma + num_std * std
    return sma, upper, lower


def resample_weekly(series):
    return series.resample("W-FRI").last().dropna()


def resample_monthly(series):
    return series.resample("ME").last().dropna()


def pct_below_recent_high(close, high, window=20):
    rolling_high = high.rolling(window).max()
    return (close - rolling_high) / rolling_high


# ── Precompute Indicators ──────────────────────────────────────────────────
print("Computing indicators...")
indicators = {}
for sym in UNIVERSE:
    c = close_df[sym].dropna()
    h = high_df[sym].dropna()
    # Align
    idx = c.index.intersection(h.index)
    c = c.loc[idx]
    h = h.loc[idx]

    d = {}
    d["close"] = c
    d["high"] = h
    d["rsi_14"] = rsi(c, 14)
    d["pct_below_high"] = pct_below_recent_high(c, h, 20)

    # Weekly RSI
    wk_close = resample_weekly(c)
    d["weekly_rsi"] = rsi(wk_close, 14)

    # Monthly RSI
    mo_close = resample_monthly(c)
    d["monthly_rsi"] = rsi(mo_close, 14)

    # Bollinger Bands daily
    bb_sma, bb_upper, bb_lower = bollinger_bands(c, 20, 2)
    d["bb_lower"] = bb_lower

    # Bollinger Bands weekly
    wk_bb_sma, wk_bb_upper, wk_bb_lower = bollinger_bands(wk_close, 20, 2)
    d["weekly_bb_lower"] = wk_bb_lower

    # SMAs
    d["sma_20"] = c.rolling(20).mean()
    d["sma_50"] = c.rolling(50).mean()
    d["sma_200"] = c.rolling(200).mean()

    # ROC
    d["roc_5"] = c.pct_change(5) * 100
    d["roc_10"] = c.pct_change(10) * 100
    d["roc_20"] = c.pct_change(20) * 100

    indicators[sym] = d


# ── Helper: map daily date to most recent weekly/monthly value ──────────────
def get_weekly_val(weekly_series, date):
    """Get most recent weekly value on or before date."""
    valid = weekly_series.loc[:date]
    if len(valid) == 0:
        return np.nan
    return valid.iloc[-1]


def weekly_rsi_inflecting(weekly_rsi_series, date):
    """Check if weekly RSI was declining 3+ weeks and is now rising."""
    valid = weekly_rsi_series.loc[:date]
    if len(valid) < 5:
        return False
    recent = valid.iloc[-5:]
    # Check that the last 3+ values before the most recent were declining
    # and the most recent is higher than the one before it
    vals = recent.values
    # Was declining: vals[-4] > vals[-3] > vals[-2] (at least 3 declining)
    declining = (vals[-4] > vals[-3]) and (vals[-3] > vals[-2])
    # Now rising: vals[-1] > vals[-2]
    rising = vals[-1] > vals[-2]
    return declining and rising


# ── Signal Generation ───────────────────────────────────────────────────────
def generate_signals_A(indicators, sym, dates):
    """Daily + Weekly RSI Oversold. RSI(14)<35 daily, weekly RSI<40, >5% below 20d high. Hold 10."""
    d = indicators[sym]
    signals = []
    for date in dates:
        if date not in d["rsi_14"].index:
            continue
        daily_rsi = d["rsi_14"].loc[date]
        weekly_rsi = get_weekly_val(d["weekly_rsi"], date)
        pct_below = d["pct_below_high"].loc[date]
        if daily_rsi < 35 and weekly_rsi < 40 and pct_below < -0.05:
            signals.append((date, sym, 10))
    return signals


def generate_signals_B(indicators, sym, dates):
    """Triple TF: daily RSI<40, weekly RSI<45, monthly RSI<50, >5% below high. Hold 15."""
    d = indicators[sym]
    signals = []
    for date in dates:
        if date not in d["rsi_14"].index:
            continue
        daily_rsi = d["rsi_14"].loc[date]
        weekly_rsi = get_weekly_val(d["weekly_rsi"], date)
        monthly_rsi = get_weekly_val(d["monthly_rsi"], date)
        pct_below = d["pct_below_high"].loc[date]
        if daily_rsi < 40 and weekly_rsi < 45 and monthly_rsi < 50 and pct_below < -0.05:
            signals.append((date, sym, 15))
    return signals


def generate_signals_C(indicators, sym, dates):
    """Daily RSI<35, weekly RSI inflection (3+ weeks decline then rise), >7% below high. Hold 10."""
    d = indicators[sym]
    signals = []
    for date in dates:
        if date not in d["rsi_14"].index:
            continue
        daily_rsi = d["rsi_14"].loc[date]
        pct_below = d["pct_below_high"].loc[date]
        inflecting = weekly_rsi_inflecting(d["weekly_rsi"], date)
        if daily_rsi < 35 and inflecting and pct_below < -0.07:
            signals.append((date, sym, 10))
    return signals


def generate_signals_D(indicators, sym, dates):
    """BB Multi-TF: daily close < daily lower BB AND weekly close < weekly lower BB, >5% below high. Hold 10."""
    d = indicators[sym]
    signals = []
    for date in dates:
        if date not in d["close"].index:
            continue
        price = d["close"].loc[date]
        daily_bb_low = d["bb_lower"].loc[date] if date in d["bb_lower"].index else np.nan
        weekly_bb_low = get_weekly_val(d["weekly_bb_lower"], date)
        pct_below = d["pct_below_high"].loc[date]
        if (not np.isnan(daily_bb_low) and not np.isnan(weekly_bb_low)
                and price < daily_bb_low and price < weekly_bb_low
                and pct_below < -0.05):
            signals.append((date, sym, 10))
    return signals


def generate_signals_E(indicators, sym, dates):
    """MA Multi-TF: below 20d SMA, below 50d SMA, above 200d SMA, RSI<35. Hold 10."""
    d = indicators[sym]
    signals = []
    for date in dates:
        if date not in d["close"].index:
            continue
        price = d["close"].loc[date]
        sma20 = d["sma_20"].loc[date] if date in d["sma_20"].index else np.nan
        sma50 = d["sma_50"].loc[date] if date in d["sma_50"].index else np.nan
        sma200 = d["sma_200"].loc[date] if date in d["sma_200"].index else np.nan
        daily_rsi = d["rsi_14"].loc[date]
        if (not np.isnan(sma20) and not np.isnan(sma50) and not np.isnan(sma200)
                and price < sma20 and price < sma50 and price > sma200
                and daily_rsi < 35):
            signals.append((date, sym, 10))
    return signals


def generate_signals_F(indicators, sym, dates):
    """ROC Multi-TF: 5d ROC < -5%, 10d ROC < -8%, 20d ROC < -10%. Hold 10."""
    d = indicators[sym]
    signals = []
    for date in dates:
        if date not in d["roc_5"].index:
            continue
        roc5 = d["roc_5"].loc[date]
        roc10 = d["roc_10"].loc[date]
        roc20 = d["roc_20"].loc[date]
        if (not np.isnan(roc5) and not np.isnan(roc10) and not np.isnan(roc20)
                and roc5 < -5 and roc10 < -8 and roc20 < -10):
            signals.append((date, sym, 10))
    return signals


VARIANTS = {
    "A": ("Daily+Weekly RSI Oversold (tight)", generate_signals_A),
    "B": ("Triple TF (D+W+M RSI)", generate_signals_B),
    "C": ("Daily Oversold + Weekly RSI Inflection", generate_signals_C),
    "D": ("Bollinger Band Multi-TF", generate_signals_D),
    "E": ("MA Multi-TF (below 20/50, above 200, RSI<35)", generate_signals_E),
    "F": ("ROC Multi-TF (cascading oversold)", generate_signals_F),
}


# ── Backtest Engine ─────────────────────────────────────────────────────────
def run_backtest(all_signals, close_df, capital, max_per_trade, max_concurrent, slippage_bps):
    """
    Simple backtest: process signals chronologically, manage position limits.
    Returns list of trade dicts.
    """
    # Sort signals by date
    all_signals.sort(key=lambda x: x[0])
    trades = []
    open_positions = []  # list of (exit_date, sym, entry_price, shares)
    cash = capital

    all_dates = close_df.index.sort_values()
    bt_start = pd.Timestamp(BACKTEST_START)

    for (entry_date, sym, hold_days) in all_signals:
        if entry_date < bt_start:
            continue

        # Close expired positions
        new_open = []
        for (exit_d, s, ep, sh) in open_positions:
            if entry_date >= exit_d:
                # Position already exited
                if exit_d in close_df.index and s in close_df.columns:
                    exit_price = close_df.loc[exit_d, s]
                    if not np.isnan(exit_price):
                        exit_price_adj = exit_price * (1 - slippage_bps / 10000)
                        pnl = (exit_price_adj - ep) * sh
                        cash += exit_price_adj * sh
                        ret = (exit_price_adj / ep) - 1
                        trades.append({
                            "sym": s, "entry_date": str(ep_date),
                            "exit_date": str(exit_d),
                            "entry_price": round(ep, 4),
                            "exit_price": round(exit_price_adj, 4),
                            "shares": sh, "pnl": round(pnl, 2),
                            "return": round(ret, 6),
                        })
            else:
                new_open.append((exit_d, s, ep, sh))
        open_positions = new_open

        # Check concurrent limit
        # Also need to properly close positions that have passed their exit date
        active = [(ed, s, ep, sh) for (ed, s, ep, sh) in open_positions if entry_date < ed]
        open_positions = active

        if len(open_positions) >= max_concurrent:
            continue

        # Check we have price data
        if sym not in close_df.columns or entry_date not in close_df.index:
            continue
        entry_price = close_df.loc[entry_date, sym]
        if np.isnan(entry_price):
            continue

        # Apply slippage to entry
        entry_price_adj = entry_price * (1 + slippage_bps / 10000)

        # Position sizing
        alloc = min(max_per_trade, cash)
        if alloc < 1:
            continue
        shares = int(alloc / entry_price_adj)
        if shares < 1:
            continue

        cost = shares * entry_price_adj
        cash -= cost

        # Find exit date (hold_days trading days later)
        entry_idx = all_dates.get_loc(entry_date)
        exit_idx = min(entry_idx + hold_days, len(all_dates) - 1)
        exit_date = all_dates[exit_idx]

        ep_date = entry_date  # store for trade record
        open_positions.append((exit_date, sym, entry_price_adj, shares))

    # Close remaining positions at their exit dates
    for (exit_d, s, ep, sh) in open_positions:
        if exit_d in close_df.index and s in close_df.columns:
            exit_price = close_df.loc[exit_d, s]
            if not np.isnan(exit_price):
                exit_price_adj = exit_price * (1 - slippage_bps / 10000)
                pnl = (exit_price_adj - ep) * sh
                ret = (exit_price_adj / ep) - 1
                trades.append({
                    "sym": s, "entry_date": "N/A",
                    "exit_date": str(exit_d),
                    "entry_price": round(ep, 4),
                    "exit_price": round(exit_price_adj, 4),
                    "shares": sh, "pnl": round(pnl, 2),
                    "return": round(ret, 6),
                })

    return trades


def run_backtest_v2(all_signals, close_df, capital, max_per_trade, max_concurrent, slippage_bps):
    """
    Cleaner backtest implementation with proper position tracking.
    """
    all_signals.sort(key=lambda x: x[0])
    trades = []
    # Track open positions as list of dicts
    positions = []
    cash = capital
    all_dates = sorted(close_df.index)
    bt_start = pd.Timestamp(BACKTEST_START)

    for (entry_date, sym, hold_days) in all_signals:
        if entry_date < bt_start:
            continue

        # Close any positions whose exit date has arrived
        still_open = []
        for pos in positions:
            if entry_date >= pos["exit_date"]:
                ex_d = pos["exit_date"]
                if ex_d in close_df.index and pos["sym"] in close_df.columns:
                    ex_price = close_df.loc[ex_d, pos["sym"]]
                    if not np.isnan(ex_price):
                        ex_price_adj = ex_price * (1 - slippage_bps / 10000)
                        pnl = (ex_price_adj - pos["entry_price"]) * pos["shares"]
                        ret = (ex_price_adj / pos["entry_price"]) - 1
                        cash += ex_price_adj * pos["shares"]
                        trades.append({
                            "sym": pos["sym"],
                            "entry_date": str(pos["entry_date"].date()),
                            "exit_date": str(ex_d.date()),
                            "entry_price": round(pos["entry_price"], 4),
                            "exit_price": round(ex_price_adj, 4),
                            "shares": pos["shares"],
                            "pnl": round(pnl, 2),
                            "return": round(ret, 6),
                        })
                    else:
                        cash += pos["entry_price"] * pos["shares"]  # fallback
                else:
                    cash += pos["entry_price"] * pos["shares"]
            else:
                still_open.append(pos)
        positions = still_open

        # Concurrent limit
        if len(positions) >= max_concurrent:
            continue

        # Price check
        if sym not in close_df.columns or entry_date not in close_df.index:
            continue
        entry_price = close_df.loc[entry_date, sym]
        if np.isnan(entry_price):
            continue

        entry_price_adj = entry_price * (1 + slippage_bps / 10000)

        # Position sizing
        alloc = min(max_per_trade, cash)
        if alloc < 1:
            continue
        shares = int(alloc / entry_price_adj)
        if shares < 1:
            continue

        cost = shares * entry_price_adj
        cash -= cost

        # Compute exit date
        try:
            entry_loc = close_df.index.get_loc(entry_date)
        except KeyError:
            cash += cost
            continue
        exit_loc = min(entry_loc + hold_days, len(close_df.index) - 1)
        exit_date = close_df.index[exit_loc]

        positions.append({
            "sym": sym,
            "entry_date": entry_date,
            "exit_date": exit_date,
            "entry_price": entry_price_adj,
            "shares": shares,
        })

    # Close remaining
    for pos in positions:
        ex_d = pos["exit_date"]
        if ex_d in close_df.index and pos["sym"] in close_df.columns:
            ex_price = close_df.loc[ex_d, pos["sym"]]
            if not np.isnan(ex_price):
                ex_price_adj = ex_price * (1 - slippage_bps / 10000)
                pnl = (ex_price_adj - pos["entry_price"]) * pos["shares"]
                ret = (ex_price_adj / pos["entry_price"]) - 1
                trades.append({
                    "sym": pos["sym"],
                    "entry_date": str(pos["entry_date"].date()),
                    "exit_date": str(ex_d.date()),
                    "entry_price": round(pos["entry_price"], 4),
                    "exit_price": round(ex_price_adj, 4),
                    "shares": pos["shares"],
                    "pnl": round(pnl, 2),
                    "return": round(ret, 6),
                })

    return trades


# ── Metrics ─────────────────────────────────────────────────────────────────
def compute_metrics(trades, capital):
    if not trades:
        return {"n_trades": 0, "sharpe": 0, "total_return_pct": 0, "max_dd_pct": 0,
                "win_rate": 0, "avg_return": 0, "profit_factor": 0, "sortino": 0}

    returns = np.array([t["return"] for t in trades])
    pnls = np.array([t["pnl"] for t in trades])

    n = len(returns)
    avg_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1) if n > 1 else 1e-9
    sharpe = (avg_ret / std_ret) * np.sqrt(252 / 10) if std_ret > 1e-9 else 0  # annualized approx

    # Sortino
    downside = returns[returns < 0]
    down_std = np.std(downside, ddof=1) if len(downside) > 1 else 1e-9
    sortino = (avg_ret / down_std) * np.sqrt(252 / 10) if down_std > 1e-9 else 0

    # Win rate
    wins = np.sum(returns > 0)
    wr = wins / n if n > 0 else 0

    # Profit factor
    gross_profit = np.sum(pnls[pnls > 0])
    gross_loss = np.abs(np.sum(pnls[pnls < 0]))
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Total return
    total_pnl = np.sum(pnls)
    total_ret_pct = (total_pnl / capital) * 100

    # Max drawdown (equity curve)
    eq = capital + np.cumsum(pnls)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    max_dd = np.min(dd) * 100 if len(dd) > 0 else 0

    return {
        "n_trades": n,
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "total_return_pct": round(total_ret_pct, 2),
        "total_pnl": round(total_pnl, 2),
        "max_dd_pct": round(max_dd, 2),
        "win_rate": round(wr, 4),
        "avg_return_pct": round(avg_ret * 100, 4),
        "profit_factor": round(pf, 3) if pf != float("inf") else 999.0,
    }


# ── Permutation Test ────────────────────────────────────────────────────────
def permutation_test(trades, all_signals, close_df, capital, max_per_trade,
                     max_concurrent, slippage_bps, n_perms=1000):
    """Shuffle entry dates among the same universe to test if edge is real."""
    if not trades:
        return 1.0

    rng = np.random.RandomState(SEED)
    actual_sharpe = compute_metrics(trades, capital)["sharpe"]

    # Collect all valid trading dates
    bt_dates = close_df.loc[BACKTEST_START:].index.tolist()
    n_signals = len(all_signals)

    count_better = 0
    for _ in range(n_perms):
        # Random entry dates, same symbols and hold periods
        shuffled = []
        for (_, sym, hold) in all_signals:
            rand_date = bt_dates[rng.randint(0, len(bt_dates))]
            shuffled.append((rand_date, sym, hold))

        perm_trades = run_backtest_v2(shuffled, close_df, capital, max_per_trade,
                                      max_concurrent, slippage_bps)
        perm_sharpe = compute_metrics(perm_trades, capital)["sharpe"]
        if perm_sharpe >= actual_sharpe:
            count_better += 1

    return round(count_better / n_perms, 4)


# ── Regime Analysis ─────────────────────────────────────────────────────────
def regime_analysis(trades, spy_close):
    """Split trades by SPY regime (bull/bear based on 20-day return at entry)."""
    if not trades:
        return {"regime_gap": 0, "bull_sharpe": 0, "bear_sharpe": 0}

    spy_ret_20d = spy_close.pct_change(20)

    bull_returns = []
    bear_returns = []

    for t in trades:
        entry_str = t["entry_date"]
        if entry_str == "N/A":
            continue
        entry_d = pd.Timestamp(entry_str)
        # Find nearest SPY date
        valid = spy_ret_20d.loc[:entry_d]
        if len(valid) == 0:
            continue
        regime_val = valid.iloc[-1]
        if np.isnan(regime_val):
            continue

        if regime_val >= 0:
            bull_returns.append(t["return"])
        else:
            bear_returns.append(t["return"])

    def sharpe_from_returns(rets):
        if len(rets) < 2:
            return 0
        arr = np.array(rets)
        m = np.mean(arr)
        s = np.std(arr, ddof=1)
        return (m / s) * np.sqrt(252 / 10) if s > 1e-9 else 0

    bull_sharpe = sharpe_from_returns(bull_returns)
    bear_sharpe = sharpe_from_returns(bear_returns)

    max_abs = max(abs(bull_sharpe), abs(bear_sharpe))
    regime_gap = abs(bull_sharpe - bear_sharpe) / max_abs if max_abs > 1e-9 else 0

    return {
        "bull_sharpe": round(bull_sharpe, 3),
        "bear_sharpe": round(bear_sharpe, 3),
        "regime_gap": round(regime_gap, 3),
        "bull_trades": len(bull_returns),
        "bear_trades": len(bear_returns),
    }


# ── 5-Gate Validation ──────────────────────────────────────────────────────
def validate_5gate(metrics, perm_p, regime_info):
    gates = {}
    gates["sharpe_gt_0.5"] = metrics["sharpe"] > 0.5
    gates["perm_p_lt_0.05"] = perm_p < 0.05
    gates["regime_gap_lt_0.5"] = regime_info["regime_gap"] < 0.5
    gates["max_dd_gt_neg50"] = metrics["max_dd_pct"] > -50
    gates["min_20_trades"] = metrics["n_trades"] >= 20
    gates["all_passed"] = all(gates.values())
    return gates


# ── Main Loop ───────────────────────────────────────────────────────────────
print("=" * 80)
print("MULTI-TIMEFRAME OVERSOLD CONFIRMATION BACKTEST")
print("=" * 80)

bt_dates = close_df.loc[BACKTEST_START:].index.tolist()
results = {}

for variant_key, (variant_name, signal_fn) in VARIANTS.items():
    print(f"\n{'─' * 60}")
    print(f"Variant {variant_key}: {variant_name}")
    print(f"{'─' * 60}")

    # Generate signals for all stocks
    all_signals = []
    for sym in UNIVERSE:
        sigs = signal_fn(indicators, sym, bt_dates)
        all_signals.extend(sigs)

    print(f"  Raw signals: {len(all_signals)}")

    # Run backtest
    trades = run_backtest_v2(all_signals, close_df, CAPITAL, MAX_PER_TRADE,
                             MAX_CONCURRENT, SLIPPAGE_BPS)
    print(f"  Executed trades: {len(trades)}")

    # Metrics
    metrics = compute_metrics(trades, CAPITAL)
    print(f"  Sharpe: {metrics['sharpe']}  Sortino: {metrics['sortino']}  "
          f"WR: {metrics['win_rate']:.1%}  PF: {metrics['profit_factor']}")
    print(f"  Total Return: {metrics['total_return_pct']:.1f}%  "
          f"Max DD: {metrics['max_dd_pct']:.1f}%  Avg Ret: {metrics['avg_return_pct']:.3f}%")

    # Regime analysis
    regime_info = regime_analysis(trades, spy_close)
    print(f"  Regime: Bull Sharpe={regime_info['bull_sharpe']}  "
          f"Bear Sharpe={regime_info['bear_sharpe']}  Gap={regime_info['regime_gap']}")

    # Permutation test (skip if <5 trades — won't be meaningful)
    if len(trades) >= 5:
        print(f"  Running {N_PERMUTATIONS} permutations...")
        perm_p = permutation_test(trades, all_signals, close_df, CAPITAL,
                                   MAX_PER_TRADE, MAX_CONCURRENT, SLIPPAGE_BPS, N_PERMUTATIONS)
    else:
        perm_p = 1.0
    print(f"  Permutation p-value: {perm_p}")

    # 5-Gate
    gates = validate_5gate(metrics, perm_p, regime_info)
    gate_str = " | ".join([f"{k}={'PASS' if v else 'FAIL'}" for k, v in gates.items() if k != "all_passed"])
    print(f"  Gates: {gate_str}")
    print(f"  >>> {'ALL GATES PASSED' if gates['all_passed'] else 'FAILED'} <<<")

    results[variant_key] = {
        "name": variant_name,
        "metrics": metrics,
        "regime": regime_info,
        "perm_p_value": perm_p,
        "gates": {k: bool(v) for k, v in gates.items()},
        "sample_trades": trades[:5] if trades else [],
        "all_trades_count": len(trades),
    }

# ── Summary ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"{'Var':<4} {'Name':<45} {'N':>4} {'Sharpe':>7} {'Sortino':>8} {'WR':>6} "
      f"{'PF':>6} {'Ret%':>7} {'DD%':>7} {'Perm-p':>7} {'Pass':>5}")
print("-" * 110)

for k in sorted(results.keys()):
    r = results[k]
    m = r["metrics"]
    passed = "YES" if r["gates"]["all_passed"] else "NO"
    print(f"  {k:<3} {r['name']:<45} {m['n_trades']:>4} {m['sharpe']:>7.3f} "
          f"{m['sortino']:>8.3f} {m['win_rate']:>5.1%} {m['profit_factor']:>6.2f} "
          f"{m['total_return_pct']:>6.1f}% {m['max_dd_pct']:>6.1f}% "
          f"{r['perm_p_value']:>7.4f} {passed:>5}")

# ── Save Results ────────────────────────────────────────────────────────────
output_path = "/home/jupiter/Lvl3Quant/data/multi_tf_confirmation_results.json"
output = {
    "strategy": "Multi-Timeframe Oversold Confirmation",
    "run_date": str(datetime.datetime.now()),
    "parameters": {
        "capital": CAPITAL,
        "max_per_trade": MAX_PER_TRADE,
        "max_concurrent": MAX_CONCURRENT,
        "slippage_bps": SLIPPAGE_BPS,
        "period": f"{BACKTEST_START} to {BACKTEST_END}",
        "universe": UNIVERSE,
        "n_permutations": N_PERMUTATIONS,
    },
    "variants": results,
    "passed_variants": [k for k, v in results.items() if v["gates"]["all_passed"]],
}

with open(output_path, "w") as f:
    json.dump(output, f, indent=2, default=str)

print(f"\nResults saved to {output_path}")
print("Done.")
