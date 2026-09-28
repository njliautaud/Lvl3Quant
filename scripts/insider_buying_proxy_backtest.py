#!/usr/bin/env python3
"""
Insider Buying MOMENTUM Proxy Backtest
======================================
Proxies insider/institutional buying via volume + price action signals.
Core thesis: unusual volume + positive price action outside earnings = informed accumulation.

Variants:
  A) Basic: vol>2x avg + close>open, hold 20d
  B) Strong Signal: vol>3x avg + close up>2%, hold 20d
  C) Cluster Signal: 3+ universe stocks trigger same day → buy SPY/QQQ, hold 20d
  D) Options Play: ATM calls 14-DTE, exit 10d or +50%/-30%
  E) Momentum Combo: high-vol up day + above 50-SMA + 20d momentum positive, hold 20d
  F) Contrarian Combo: high-vol up day after stock down >10% in 20d, hold 20d

Walk-forward OOT: Jan 2022 – Jul 2026
5-gate validation: Sharpe>0.5, perm p<0.05, regime gap<0.5, MaxDD>-50%, >=20 trades
$645 account, 0.02% slippage, $0 commission shares. Options: $0.65/contract, 3% ATM premium, 5% bid-ask.
"""

import json, os, sys, warnings, datetime as dt
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

warnings.filterwarnings("ignore")

# ── CONFIG ──────────────────────────────────────────────────────────────────
UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD", "NFLX", "CRM",
    "PLTR", "SOFI", "HOOD", "SNAP", "PINS", "UBER", "LYFT", "COIN", "RBLX", "DDOG",
    "TTD", "SHOP", "NET", "ROKU", "SE", "MELI", "NU",
]
ETF_TICKERS = ["SPY", "QQQ"]
BENCHMARK = "SPY"
INITIAL_CAPITAL = 645.0
SLIPPAGE_PCT = 0.0002  # 0.02%
COMMISSION_SHARES = 0.0
COMMISSION_OPTIONS = 0.65  # per contract
OPTION_PREMIUM_PCT = 0.03  # 3% of stock price for ATM 14-DTE
OPTION_BIDASK_PCT = 0.05  # 5% bid-ask spread on options
MAX_CONCURRENT = 3
OOT_START = "2022-01-01"
OOT_END = "2026-07-28"
DATA_START = "2020-06-01"  # lookback for 200-SMA etc
PERM_ITERS = 1000
RESULTS_PATH = "/home/jupiter/Lvl3Quant/data/insider_buying_proxy_results.json"
HOLD_DAYS = 20  # default holding period (trading days)

# ── DATA DOWNLOAD ───────────────────────────────────────────────────────────
print("Downloading price data...")
all_tickers = list(set(UNIVERSE + ETF_TICKERS + [BENCHMARK]))
data = {}
for t in all_tickers:
    try:
        df = yf.download(t, start=DATA_START, end=OOT_END, progress=False, auto_adjust=True)
        if df is not None and len(df) > 200:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            data[t] = df
            print(f"  {t}: {len(df)} bars")
        else:
            print(f"  {t}: insufficient data, skipping")
    except Exception as e:
        print(f"  {t}: download error {e}")

spy = data.get(BENCHMARK)
if spy is None:
    print("FATAL: Cannot download SPY"); sys.exit(1)

spy_sma200 = spy["Close"].rolling(200).mean()


# ── EARNINGS EXCLUSION ──────────────────────────────────────────────────────
def detect_earnings_windows(df, gap_thresh=0.03, window=5):
    """
    Detect earnings-like events: days with >3% gap (open vs prior close).
    Exclude ±window trading days around these.
    Returns boolean Series: True = EXCLUDE (near earnings).
    """
    prior_close = df["Close"].shift(1)
    gap = (df["Open"] - prior_close).abs() / prior_close
    big_gap_dates = df.index[gap > gap_thresh]

    exclude = pd.Series(False, index=df.index)
    for gd in big_gap_dates:
        loc = df.index.get_loc(gd)
        start = max(0, loc - window)
        end = min(len(df.index) - 1, loc + window)
        exclude.iloc[start:end+1] = True
    return exclude


# ── PRE-COMPUTE FEATURES ───────────────────────────────────────────────────
features = {}
earnings_exclude = {}

for ticker in UNIVERSE:
    if ticker not in data:
        continue
    df = data[ticker]
    f = pd.DataFrame(index=df.index)
    f["close"] = df["Close"]
    f["open"] = df["Open"]
    f["volume"] = df["Volume"]
    f["vol_avg20"] = df["Volume"].rolling(20).mean()
    f["close_gt_open"] = df["Close"] > df["Open"]
    f["daily_ret"] = df["Close"].pct_change()
    f["sma50"] = df["Close"].rolling(50).mean()
    f["mom20"] = df["Close"].pct_change(20)
    f["ret20d_lookback"] = df["Close"].pct_change(20)  # for contrarian: how much down in last 20d
    features[ticker] = f
    earnings_exclude[ticker] = detect_earnings_windows(df)


# ── SIGNAL GENERATORS ───────────────────────────────────────────────────────

def signal_basic(ticker):
    """A) Vol > 2x 20d avg, close > open, NOT near earnings."""
    if ticker not in features:
        return pd.Series(dtype=bool)
    f = features[ticker]
    ex = earnings_exclude[ticker]
    sig = (f["volume"] > 2 * f["vol_avg20"]) & f["close_gt_open"] & (~ex)
    return sig.fillna(False)


def signal_strong(ticker):
    """B) Vol > 3x avg AND close up >2%."""
    if ticker not in features:
        return pd.Series(dtype=bool)
    f = features[ticker]
    ex = earnings_exclude[ticker]
    sig = (f["volume"] > 3 * f["vol_avg20"]) & (f["daily_ret"] > 0.02) & (~ex)
    return sig.fillna(False)


def signal_momentum_combo(ticker):
    """E) High-vol up day + above 50-SMA + 20d momentum positive."""
    if ticker not in features:
        return pd.Series(dtype=bool)
    f = features[ticker]
    ex = earnings_exclude[ticker]
    basic = (f["volume"] > 2 * f["vol_avg20"]) & f["close_gt_open"] & (~ex)
    above_sma = f["close"] > f["sma50"]
    mom_pos = f["mom20"] > 0
    sig = basic & above_sma & mom_pos
    return sig.fillna(False)


def signal_contrarian(ticker):
    """F) High-vol up day AFTER stock down >10% in prior 20 days."""
    if ticker not in features:
        return pd.Series(dtype=bool)
    f = features[ticker]
    ex = earnings_exclude[ticker]
    basic = (f["volume"] > 2 * f["vol_avg20"]) & f["close_gt_open"] & (~ex)
    was_down = f["ret20d_lookback"] < -0.10
    sig = basic & was_down
    return sig.fillna(False)


# ── BACKTESTER (SHARES) ────────────────────────────────────────────────────

def run_shares_backtest(variant_name, signal_func, hold_td=HOLD_DAYS, use_etf=False, cluster_min=0):
    """
    Run walk-forward backtest for shares-based variants.
    hold_td: holding period in trading days.
    use_etf: if True, buy SPY instead of individual stock.
    cluster_min: if >0, only trigger when >= cluster_min stocks signal on same day.
    """
    # Collect all signals
    all_signals = []  # (date, ticker_to_buy, entry_price)
    all_dates = spy.index[(spy.index >= OOT_START) & (spy.index <= OOT_END)]

    if cluster_min > 0:
        # Cluster mode: count signals per day, buy ETF when enough fire
        daily_counts = {}
        for ticker in UNIVERSE:
            sig = signal_func(ticker)
            oot_mask = (sig.index >= OOT_START) & (sig.index <= OOT_END)
            sig_dates = sig.index[oot_mask & sig]
            for d in sig_dates:
                daily_counts[d] = daily_counts.get(d, 0) + 1

        # Use SPY as the buy target
        buy_ticker = "SPY"
        if buy_ticker not in data:
            return None
        buy_df = data[buy_ticker]
        for d, count in sorted(daily_counts.items()):
            if count >= cluster_min and d in buy_df.index:
                price = float(buy_df.loc[d, "Close"])
                if isinstance(price, pd.Series):
                    price = price.iloc[0]
                all_signals.append((d, buy_ticker, price))
    else:
        for ticker in UNIVERSE:
            sig = signal_func(ticker)
            df = data.get(ticker)
            if df is None:
                continue
            oot_mask = (sig.index >= OOT_START) & (sig.index <= OOT_END)
            sig_dates = sig.index[oot_mask & sig]
            for d in sig_dates:
                buy_ticker = "SPY" if use_etf else ticker
                buy_df = data.get(buy_ticker)
                if buy_df is None or d not in buy_df.index:
                    continue
                price = float(buy_df.loc[d, "Close"])
                if isinstance(price, pd.Series):
                    price = price.iloc[0]
                all_signals.append((d, buy_ticker, price))

    if not all_signals:
        return None

    all_signals.sort(key=lambda x: x[0])

    # Simulate
    capital = INITIAL_CAPITAL
    positions = []  # (entry_date, ticker, entry_price, shares, exit_idx)
    trades = []
    equity_curve = {}

    # Build trading day index
    td_index = all_dates.tolist()
    td_lookup = {d: i for i, d in enumerate(td_index)}

    sig_idx = 0
    for day_i, today in enumerate(td_index):
        # Close expired positions
        new_positions = []
        for pos in positions:
            entry_date, ticker, entry_price, shares, exit_idx = pos
            if day_i >= exit_idx:
                buy_df = data.get(ticker)
                if buy_df is not None and today in buy_df.index:
                    exit_price = float(buy_df.loc[today, "Close"])
                    if isinstance(exit_price, pd.Series):
                        exit_price = exit_price.iloc[0]
                else:
                    exit_price = entry_price

                exit_price_slip = exit_price * (1 - SLIPPAGE_PCT)
                entry_cost = entry_price * (1 + SLIPPAGE_PCT)
                pnl = shares * (exit_price_slip - entry_cost)
                capital += shares * exit_price_slip
                trades.append({
                    "ticker": ticker,
                    "entry_date": str(entry_date.date()) if hasattr(entry_date, 'date') else str(entry_date),
                    "exit_date": str(today.date()) if hasattr(today, 'date') else str(today),
                    "entry_price": round(entry_price, 2),
                    "exit_price": round(exit_price, 2),
                    "shares": round(shares, 4),
                    "pnl": round(pnl, 2),
                    "return_pct": round(pnl / (shares * entry_price) * 100, 2),
                    "hold_days": (today - entry_date).days,
                })
            else:
                new_positions.append(pos)
        positions = new_positions

        # Open new positions
        while sig_idx < len(all_signals) and all_signals[sig_idx][0] <= today:
            sig_date, sig_ticker, sig_price = all_signals[sig_idx]
            sig_idx += 1

            if sig_date != today:
                continue
            if len(positions) >= MAX_CONCURRENT:
                continue

            held_tickers = {p[1] for p in positions}
            if sig_ticker in held_tickers:
                continue

            pos_size = capital / MAX_CONCURRENT
            if pos_size < 10:
                continue

            entry_cost = sig_price * (1 + SLIPPAGE_PCT)
            shares = pos_size / entry_cost
            exit_idx = min(day_i + hold_td, len(td_index) - 1)

            capital -= shares * entry_cost
            positions.append((today, sig_ticker, sig_price, shares, exit_idx))

        # Mark-to-market
        mtm = capital
        for pos in positions:
            entry_date, ticker, entry_price, shares, exit_idx = pos
            buy_df = data.get(ticker)
            if buy_df is not None and today in buy_df.index:
                curr_price = float(buy_df.loc[today, "Close"])
                if isinstance(curr_price, pd.Series):
                    curr_price = curr_price.iloc[0]
            else:
                curr_price = entry_price
            mtm += shares * curr_price
        equity_curve[today] = mtm

    # Close remaining
    last_date = td_index[-1]
    for pos in positions:
        entry_date, ticker, entry_price, shares, exit_idx = pos
        buy_df = data.get(ticker)
        if buy_df is not None and last_date in buy_df.index:
            exit_price = float(buy_df.loc[last_date, "Close"])
            if isinstance(exit_price, pd.Series):
                exit_price = exit_price.iloc[0]
        else:
            exit_price = entry_price
        exit_price_slip = exit_price * (1 - SLIPPAGE_PCT)
        entry_cost = entry_price * (1 + SLIPPAGE_PCT)
        pnl = shares * (exit_price_slip - entry_cost)
        trades.append({
            "ticker": ticker,
            "entry_date": str(entry_date.date()) if hasattr(entry_date, 'date') else str(entry_date),
            "exit_date": str(last_date.date()) if hasattr(last_date, 'date') else str(last_date),
            "entry_price": round(entry_price, 2),
            "exit_price": round(exit_price, 2),
            "shares": round(shares, 4),
            "pnl": round(pnl, 2),
            "return_pct": round(pnl / (shares * entry_price) * 100, 2),
            "hold_days": (last_date - entry_date).days,
        })

    return compute_metrics(variant_name, trades, equity_curve)


# ── BACKTESTER (OPTIONS — VARIANT D) ───────────────────────────────────────

def run_options_backtest(variant_name):
    """
    D) Buy ATM calls (14-DTE) on high-vol up days.
    Exit after 10 trading days or at +50%/-30%.
    Premium = 3% of stock price. Bid-ask = 5%. Commission = $0.65/contract.
    """
    all_dates = spy.index[(spy.index >= OOT_START) & (spy.index <= OOT_END)]
    td_index = all_dates.tolist()
    td_lookup = {d: i for i, d in enumerate(td_index)}

    # Collect signals using basic signal
    all_signals = []
    for ticker in UNIVERSE:
        sig = signal_basic(ticker)
        df = data.get(ticker)
        if df is None:
            continue
        oot_mask = (sig.index >= OOT_START) & (sig.index <= OOT_END)
        sig_dates = sig.index[oot_mask & sig]
        for d in sig_dates:
            price = float(df.loc[d, "Close"])
            if isinstance(price, pd.Series):
                price = price.iloc[0]
            all_signals.append((d, ticker, price))

    all_signals.sort(key=lambda x: x[0])

    capital = INITIAL_CAPITAL
    positions = []  # (entry_date, ticker, stock_price_at_entry, n_contracts, premium_paid, exit_idx)
    trades = []
    equity_curve = {}

    sig_idx = 0
    for day_i, today in enumerate(td_index):
        # Check positions for exit
        new_positions = []
        for pos in positions:
            entry_date, ticker, stock_entry, n_contracts, premium_paid, exit_idx, entry_day_i = pos
            df = data.get(ticker)
            if df is None or today not in df.index:
                new_positions.append(pos)
                continue

            curr_price = float(df.loc[today, "Close"])
            if isinstance(curr_price, pd.Series):
                curr_price = curr_price.iloc[0]

            # Simplified option pricing: intrinsic + time value decay
            days_held = day_i - entry_day_i
            days_to_expiry = 14 - days_held  # approximate trading days
            if days_to_expiry < 0:
                days_to_expiry = 0

            # ATM call value approximation
            intrinsic = max(0, curr_price - stock_entry)
            # Time value decays linearly (rough)
            initial_time_value = stock_entry * OPTION_PREMIUM_PCT - max(0, 0)  # at entry it's all time value
            time_value = initial_time_value * (days_to_expiry / 14) if days_to_expiry > 0 else 0
            option_value = (intrinsic + time_value) * 100  # per contract, 100 shares

            # Apply bid-ask on exit
            exit_value = option_value * (1 - OPTION_BIDASK_PCT / 2)
            total_exit = exit_value * n_contracts - COMMISSION_OPTIONS * n_contracts

            pnl = total_exit - premium_paid
            ret_pct = pnl / premium_paid * 100 if premium_paid > 0 else 0

            # Exit conditions: 10 trading days, +50%, or -30%
            should_exit = (day_i >= exit_idx) or (ret_pct >= 50) or (ret_pct <= -30)

            if should_exit:
                capital += max(0, total_exit)
                trades.append({
                    "ticker": ticker,
                    "entry_date": str(entry_date.date()) if hasattr(entry_date, 'date') else str(entry_date),
                    "exit_date": str(today.date()) if hasattr(today, 'date') else str(today),
                    "entry_price": round(stock_entry, 2),
                    "exit_price": round(curr_price, 2),
                    "contracts": n_contracts,
                    "premium_paid": round(premium_paid, 2),
                    "exit_value": round(max(0, total_exit), 2),
                    "pnl": round(pnl, 2),
                    "return_pct": round(ret_pct, 2),
                    "hold_days": (today - entry_date).days,
                    "exit_reason": "time" if day_i >= exit_idx else ("profit" if ret_pct >= 50 else "stop"),
                })
            else:
                new_positions.append(pos)
        positions = new_positions

        # Open new positions
        while sig_idx < len(all_signals) and all_signals[sig_idx][0] <= today:
            sig_date, sig_ticker, sig_price = all_signals[sig_idx]
            sig_idx += 1

            if sig_date != today:
                continue
            if len(positions) >= MAX_CONCURRENT:
                continue
            held = {p[1] for p in positions}
            if sig_ticker in held:
                continue

            pos_size = capital / MAX_CONCURRENT
            if pos_size < 20:
                continue

            # ATM call: premium = 3% of stock price per share, *100 per contract
            premium_per_contract = sig_price * OPTION_PREMIUM_PCT * 100
            # Add bid-ask spread on entry
            entry_premium = premium_per_contract * (1 + OPTION_BIDASK_PCT / 2)
            entry_total = entry_premium + COMMISSION_OPTIONS  # per contract

            n_contracts = max(1, int(pos_size / entry_total))
            total_cost = n_contracts * entry_total

            if total_cost > capital:
                n_contracts = max(1, int(capital / entry_total))
                total_cost = n_contracts * entry_total

            if total_cost > capital or n_contracts < 1:
                continue

            capital -= total_cost
            exit_idx = day_i + 10  # 10 trading days
            positions.append((today, sig_ticker, sig_price, n_contracts, total_cost, exit_idx, day_i))

        # Mark-to-market
        mtm = capital
        for pos in positions:
            entry_date, ticker, stock_entry, n_contracts, premium_paid, exit_idx, entry_day_i = pos
            df = data.get(ticker)
            if df is not None and today in df.index:
                curr_price = float(df.loc[today, "Close"])
                if isinstance(curr_price, pd.Series):
                    curr_price = curr_price.iloc[0]
            else:
                curr_price = stock_entry
            days_held = day_i - entry_day_i
            days_to_expiry = max(0, 14 - days_held)
            intrinsic = max(0, curr_price - stock_entry)
            initial_tv = stock_entry * OPTION_PREMIUM_PCT
            time_value = initial_tv * (days_to_expiry / 14) if days_to_expiry > 0 else 0
            option_val = (intrinsic + time_value) * 100 * n_contracts
            mtm += option_val * (1 - OPTION_BIDASK_PCT / 2)
        equity_curve[today] = mtm

    # Close remaining
    last_date = td_index[-1]
    last_day_i = len(td_index) - 1
    for pos in positions:
        entry_date, ticker, stock_entry, n_contracts, premium_paid, exit_idx, entry_day_i = pos
        df = data.get(ticker)
        if df is not None and last_date in df.index:
            curr_price = float(df.loc[last_date, "Close"])
            if isinstance(curr_price, pd.Series):
                curr_price = curr_price.iloc[0]
        else:
            curr_price = stock_entry
        days_held = last_day_i - entry_day_i
        days_to_expiry = max(0, 14 - days_held)
        intrinsic = max(0, curr_price - stock_entry)
        initial_tv = stock_entry * OPTION_PREMIUM_PCT
        time_value = initial_tv * (days_to_expiry / 14) if days_to_expiry > 0 else 0
        option_value = (intrinsic + time_value) * 100 * n_contracts * (1 - OPTION_BIDASK_PCT / 2)
        total_exit = option_value - COMMISSION_OPTIONS * n_contracts
        pnl = max(0, total_exit) - premium_paid
        trades.append({
            "ticker": ticker,
            "entry_date": str(entry_date.date()) if hasattr(entry_date, 'date') else str(entry_date),
            "exit_date": str(last_date.date()) if hasattr(last_date, 'date') else str(last_date),
            "entry_price": round(stock_entry, 2),
            "exit_price": round(curr_price, 2),
            "contracts": n_contracts,
            "premium_paid": round(premium_paid, 2),
            "pnl": round(pnl, 2),
            "return_pct": round(pnl / premium_paid * 100 if premium_paid > 0 else 0, 2),
            "hold_days": (last_date - entry_date).days,
            "exit_reason": "end_of_test",
        })

    return compute_metrics(variant_name, trades, equity_curve)


# ── METRICS COMPUTATION ────────────────────────────────────────────────────

def compute_metrics(variant_name, trades, equity_curve):
    """Compute all metrics and 5-gate validation."""
    if not trades:
        return None

    eq = pd.Series(equity_curve).sort_index()
    daily_ret = eq.pct_change().dropna()

    n_trades = len(trades)
    winners = [t for t in trades if t["pnl"] > 0]
    losers = [t for t in trades if t["pnl"] <= 0]
    win_rate = len(winners) / n_trades if n_trades > 0 else 0
    total_pnl = sum(t["pnl"] for t in trades)
    final_equity = eq.iloc[-1] if len(eq) > 0 else INITIAL_CAPITAL

    gross_profit = sum(t["pnl"] for t in winners) if winners else 0
    gross_loss = abs(sum(t["pnl"] for t in losers)) if losers else 1e-9
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Annualized Sharpe
    if len(daily_ret) > 1 and daily_ret.std() > 0:
        sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252)
    else:
        sharpe = 0.0

    # Sortino
    downside = daily_ret[daily_ret < 0]
    if len(downside) > 1 and downside.std() > 0:
        sortino = (daily_ret.mean() / downside.std()) * np.sqrt(252)
    else:
        sortino = 0.0

    # Max drawdown
    peak = eq.cummax()
    dd = (eq - peak) / peak
    max_dd = dd.min()

    # Regime analysis
    bull_rets, bear_rets = [], []
    for t in trades:
        entry_d = pd.Timestamp(t["entry_date"])
        if entry_d in spy_sma200.index:
            spy_close = spy.loc[entry_d, "Close"]
            sma_val = spy_sma200.loc[entry_d]
            if isinstance(spy_close, pd.Series):
                spy_close = spy_close.iloc[0]
            if isinstance(sma_val, pd.Series):
                sma_val = sma_val.iloc[0]
            if pd.notna(sma_val):
                if spy_close > sma_val:
                    bull_rets.append(t["return_pct"])
                else:
                    bear_rets.append(t["return_pct"])

    bull_sharpe = bear_sharpe = 0.0
    # Annualize per-trade Sharpe using hold period
    ann_factor = np.sqrt(252 / HOLD_DAYS)
    if len(bull_rets) > 1:
        br = np.array(bull_rets)
        if br.std() > 0:
            bull_sharpe = (br.mean() / br.std()) * ann_factor
    if len(bear_rets) > 1:
        br = np.array(bear_rets)
        if br.std() > 0:
            bear_sharpe = (br.mean() / br.std()) * ann_factor

    regime_gap = abs(bull_sharpe - bear_sharpe) / max(abs(bull_sharpe), abs(bear_sharpe), 1e-9)

    # Permutation test: shuffle signs of trade returns
    trade_rets = np.array([t["return_pct"] for t in trades])
    observed_mean = trade_rets.mean()
    rng = np.random.default_rng(42)
    perm_count = 0
    for _ in range(PERM_ITERS):
        signs = rng.choice([-1, 1], size=len(trade_rets))
        shuffled_mean = (trade_rets * signs).mean()
        if shuffled_mean >= observed_mean:
            perm_count += 1
    perm_p = perm_count / PERM_ITERS

    # 5-gate validation
    gates = {
        "sharpe_gt_0.5": bool(sharpe > 0.5),
        "perm_p_lt_0.05": bool(perm_p < 0.05),
        "regime_gap_lt_0.5": bool(regime_gap < 0.5),
        "maxdd_gt_neg50pct": bool(max_dd > -0.50),
        "trades_gte_20": bool(n_trades >= 20),
    }
    gates_passed = sum(gates.values())

    result = {
        "variant": variant_name,
        "n_trades": n_trades,
        "win_rate": round(win_rate, 4),
        "total_pnl": round(total_pnl, 2),
        "final_equity": round(final_equity, 2),
        "total_return_pct": round((final_equity - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "profit_factor": round(min(profit_factor, 999.0), 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "avg_trade_return_pct": round(float(trade_rets.mean()), 2),
        "median_trade_return_pct": round(float(np.median(trade_rets)), 2),
        "bull_trades": len(bull_rets),
        "bear_trades": len(bear_rets),
        "bull_sharpe": round(bull_sharpe, 3),
        "bear_sharpe": round(bear_sharpe, 3),
        "regime_gap": round(regime_gap, 4),
        "perm_p_value": round(perm_p, 4),
        "gates": gates,
        "gates_passed": f"{gates_passed}/5",
        "verdict": "PASS" if gates_passed == 5 else "FAIL",
        "best_trades": sorted(trades, key=lambda x: x["pnl"], reverse=True)[:3],
        "worst_trades": sorted(trades, key=lambda x: x["pnl"])[:3],
    }
    return result


# ── RUN ALL VARIANTS ────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("INSIDER BUYING MOMENTUM PROXY BACKTEST")
print(f"OOT: {OOT_START} to {OOT_END} | Capital: ${INITIAL_CAPITAL}")
print(f"Universe: {len(UNIVERSE)} growth/tech stocks")
print("Proxy: unusual volume + positive price action (excl. earnings)")
print("=" * 70)

results = []

# A) Basic
print(f"\n{'─' * 50}")
print("Running A) Basic: vol>2x avg + close>open, hold 20d...")
res = run_shares_backtest("A_Basic_HighVol_Up", signal_basic, hold_td=20)
if res:
    results.append(res)
    print(f"  {res['verdict']} | {res['gates_passed']} gates | Trades={res['n_trades']} WR={res['win_rate']:.1%} "
          f"Sharpe={res['sharpe']:.2f} Sortino={res['sortino']:.2f} PF={res['profit_factor']:.2f} "
          f"MaxDD={res['max_drawdown_pct']:.1f}% Return={res['total_return_pct']:.1f}%")
else:
    print("  NO TRADES")
    results.append({"variant": "A_Basic_HighVol_Up", "n_trades": 0, "verdict": "FAIL — no trades"})

# B) Strong Signal
print(f"\n{'─' * 50}")
print("Running B) Strong Signal: vol>3x avg + close up>2%, hold 20d...")
res = run_shares_backtest("B_Strong_Signal", signal_strong, hold_td=20)
if res:
    results.append(res)
    print(f"  {res['verdict']} | {res['gates_passed']} gates | Trades={res['n_trades']} WR={res['win_rate']:.1%} "
          f"Sharpe={res['sharpe']:.2f} Sortino={res['sortino']:.2f} PF={res['profit_factor']:.2f} "
          f"MaxDD={res['max_drawdown_pct']:.1f}% Return={res['total_return_pct']:.1f}%")
else:
    print("  NO TRADES")
    results.append({"variant": "B_Strong_Signal", "n_trades": 0, "verdict": "FAIL — no trades"})

# C) Cluster Signal
print(f"\n{'─' * 50}")
print("Running C) Cluster Signal: 3+ stocks trigger same day → buy SPY, hold 20d...")
res = run_shares_backtest("C_Cluster_Signal", signal_basic, hold_td=20, cluster_min=3)
if res:
    results.append(res)
    print(f"  {res['verdict']} | {res['gates_passed']} gates | Trades={res['n_trades']} WR={res['win_rate']:.1%} "
          f"Sharpe={res['sharpe']:.2f} Sortino={res['sortino']:.2f} PF={res['profit_factor']:.2f} "
          f"MaxDD={res['max_drawdown_pct']:.1f}% Return={res['total_return_pct']:.1f}%")
else:
    print("  NO TRADES")
    results.append({"variant": "C_Cluster_Signal", "n_trades": 0, "verdict": "FAIL — no trades"})

# D) Options Play
print(f"\n{'─' * 50}")
print("Running D) Options Play: ATM calls 14-DTE, exit 10d or +50%/-30%...")
res = run_options_backtest("D_Options_Play")
if res:
    results.append(res)
    print(f"  {res['verdict']} | {res['gates_passed']} gates | Trades={res['n_trades']} WR={res['win_rate']:.1%} "
          f"Sharpe={res['sharpe']:.2f} Sortino={res['sortino']:.2f} PF={res['profit_factor']:.2f} "
          f"MaxDD={res['max_drawdown_pct']:.1f}% Return={res['total_return_pct']:.1f}%")
else:
    print("  NO TRADES")
    results.append({"variant": "D_Options_Play", "n_trades": 0, "verdict": "FAIL — no trades"})

# E) Momentum Combo
print(f"\n{'─' * 50}")
print("Running E) Momentum Combo: high-vol up + above 50-SMA + 20d mom positive, hold 20d...")
res = run_shares_backtest("E_Momentum_Combo", signal_momentum_combo, hold_td=20)
if res:
    results.append(res)
    print(f"  {res['verdict']} | {res['gates_passed']} gates | Trades={res['n_trades']} WR={res['win_rate']:.1%} "
          f"Sharpe={res['sharpe']:.2f} Sortino={res['sortino']:.2f} PF={res['profit_factor']:.2f} "
          f"MaxDD={res['max_drawdown_pct']:.1f}% Return={res['total_return_pct']:.1f}%")
else:
    print("  NO TRADES")
    results.append({"variant": "E_Momentum_Combo", "n_trades": 0, "verdict": "FAIL — no trades"})

# F) Contrarian Combo
print(f"\n{'─' * 50}")
print("Running F) Contrarian Combo: high-vol up day after >10% drawdown, hold 20d...")
res = run_shares_backtest("F_Contrarian_Combo", signal_contrarian, hold_td=20)
if res:
    results.append(res)
    print(f"  {res['verdict']} | {res['gates_passed']} gates | Trades={res['n_trades']} WR={res['win_rate']:.1%} "
          f"Sharpe={res['sharpe']:.2f} Sortino={res['sortino']:.2f} PF={res['profit_factor']:.2f} "
          f"MaxDD={res['max_drawdown_pct']:.1f}% Return={res['total_return_pct']:.1f}%")
else:
    print("  NO TRADES")
    results.append({"variant": "F_Contrarian_Combo", "n_trades": 0, "verdict": "FAIL — no trades"})

# ── SUMMARY TABLE ───────────────────────────────────────────────────────────
print("\n" + "=" * 120)
print("SUMMARY TABLE")
print("=" * 120)
print(f"{'Variant':<28} {'Trades':>6} {'WR':>6} {'Sharpe':>7} {'Sortino':>8} "
      f"{'PF':>6} {'MaxDD':>7} {'Return':>8} {'Perm-p':>7} {'RegGap':>7} {'Gates':>6} {'Verdict':>7}")
print("─" * 120)

for r in results:
    if r.get("n_trades", 0) > 0 and "sharpe" in r:
        print(f"{r['variant']:<28} {r['n_trades']:>6} {r['win_rate']:>5.1%} "
              f"{r['sharpe']:>7.2f} {r['sortino']:>8.2f} {r['profit_factor']:>6.2f} "
              f"{r['max_drawdown_pct']:>6.1f}% {r['total_return_pct']:>7.1f}% "
              f"{r['perm_p_value']:>7.4f} {r['regime_gap']:>7.3f} {r['gates_passed']:>6} {r['verdict']:>7}")
    else:
        print(f"{r['variant']:<28} {'—':>6} {'—':>6} {'—':>7} {'—':>8} "
              f"{'—':>6} {'—':>7} {'—':>8} {'—':>7} {'—':>7} {'—':>6} {'FAIL':>7}")

# ── CHAMPION SELECTION ──────────────────────────────────────────────────────
passing = [r for r in results if r.get("verdict") == "PASS"]
if passing:
    champion = max(passing, key=lambda x: x["sharpe"])
    print(f"\nCHAMPION: {champion['variant']} — Sharpe {champion['sharpe']:.2f}, "
          f"Sortino {champion['sortino']:.2f}, {champion['n_trades']} trades, "
          f"WR {champion['win_rate']:.1%}, PF {champion['profit_factor']:.2f}, "
          f"Return {champion['total_return_pct']:.1f}%")
else:
    valid = [r for r in results if r.get("n_trades", 0) > 0 and "sharpe" in r]
    if valid:
        champion = max(valid, key=lambda x: x["sharpe"])
        print(f"\nNO VARIANT PASSED ALL 5 GATES.")
        print(f"Best performer: {champion['variant']} — Sharpe {champion['sharpe']:.2f}, "
              f"{champion['gates_passed']} gates passed")
    else:
        champion = None
        print("\nNO VARIANTS PRODUCED TRADES.")

# ── SAVE RESULTS ────────────────────────────────────────────────────────────
output = {
    "strategy": "Insider Buying Momentum Proxy — Growth/Tech Stocks",
    "description": "Proxies insider/institutional buying via unusual volume + positive price action outside earnings windows",
    "run_date": str(dt.datetime.now()),
    "oot_period": f"{OOT_START} to {OOT_END}",
    "initial_capital": INITIAL_CAPITAL,
    "universe": UNIVERSE,
    "slippage_pct": SLIPPAGE_PCT,
    "commission_shares": COMMISSION_SHARES,
    "commission_options": COMMISSION_OPTIONS,
    "option_premium_pct": OPTION_PREMIUM_PCT,
    "option_bidask_pct": OPTION_BIDASK_PCT,
    "max_concurrent_positions": MAX_CONCURRENT,
    "hold_days": HOLD_DAYS,
    "permutation_iterations": PERM_ITERS,
    "variant_descriptions": {
        "A_Basic_HighVol_Up": "Vol > 2x 20d avg + close > open, hold 20 trading days",
        "B_Strong_Signal": "Vol > 3x avg + close up >2%, hold 20 trading days",
        "C_Cluster_Signal": "3+ stocks trigger same day → buy SPY, hold 20 trading days",
        "D_Options_Play": "ATM calls 14-DTE on basic signal, exit 10d or +50%/-30%",
        "E_Momentum_Combo": "Basic signal + above 50-SMA + 20d momentum positive, hold 20d",
        "F_Contrarian_Combo": "Basic signal after stock down >10% in prior 20d, hold 20d",
    },
    "earnings_exclusion": "Excluded ±5 trading days around gaps >3% (earnings proxy)",
    "variants": results,
    "champion": champion["variant"] if champion else None,
    "gates_explanation": {
        "sharpe_gt_0.5": "Annualized Sharpe ratio > 0.5",
        "perm_p_lt_0.05": "Permutation test p-value < 0.05 (edge is real, not random)",
        "regime_gap_lt_0.5": "|Bull Sharpe - Bear Sharpe| / max < 0.5 (regime-agnostic)",
        "maxdd_gt_neg50pct": "Max drawdown > -50% (survivable for $645 account)",
        "trades_gte_20": "At least 20 trades (statistical significance)",
    },
}

os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
with open(RESULTS_PATH, "w") as f:
    json.dump(output, f, indent=2, default=str)
print(f"\nResults saved to {RESULTS_PATH}")
print("DONE.")
