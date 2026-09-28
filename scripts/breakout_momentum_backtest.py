#!/usr/bin/env python3
"""
Breakout Momentum on Quality Stocks — 6 Variants (A-F)
With 5-Gate Validation: Sharpe, Permutation, Regime, Drawdown, Trade Count
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────
UNIVERSE = [
    "AAPL", "MSFT", "AVGO", "JPM", "JNJ", "PG", "KO", "PEP", "HD", "COST",
    "UNH", "LLY", "V", "MA", "ABBV", "MRK", "WMT", "AMZN", "GOOGL", "META",
]
CAPITAL = 645.0
MAX_PER_TRADE = 200.0
MAX_CONCURRENT = 3
SLIPPAGE_BPS = 2  # 2 basis points
START = "2022-01-01"
END = "2026-07-31"
PERMUTATION_N = 1000
RESULTS_PATH = Path("/home/jupiter/Lvl3Quant/data/breakout_momentum_results.json")


# ── Data Download ──────────────────────────────────────────────────────────
def download_data():
    """Download OHLCV for universe + SPY."""
    tickers = UNIVERSE + ["SPY"]
    print(f"Downloading {len(tickers)} tickers from {START} to {END} ...")
    data = yf.download(tickers, start=START, end=END, auto_adjust=True, progress=False)
    # yf.download returns MultiIndex columns (Price, Ticker)
    close = data["Close"]
    high = data["High"]
    low = data["Low"]
    opn = data["Open"]
    volume = data["Volume"]
    print(f"  Downloaded {len(close)} trading days, {close.shape[1]} tickers")
    return close, high, low, opn, volume


# ── Technical Indicators ───────────────────────────────────────────────────
def sma(series, n):
    return series.rolling(n).mean()

def ema(series, n):
    return series.ewm(span=n, adjust=False).mean()

def bollinger_upper(close_s, n=20, k=2):
    mid = sma(close_s, n)
    std = close_s.rolling(n).std()
    return mid + k * std

def rsi(close_s, n=14):
    delta = close_s.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = (-delta.clip(upper=0)).rolling(n).mean()
    rs = gain / loss
    return 100 - 100 / (1 + rs)

def macd(close_s, fast=12, slow=26, signal=9):
    macd_line = ema(close_s, fast) - ema(close_s, slow)
    signal_line = ema(macd_line, signal)
    return macd_line, signal_line


# ── Signal Generation ─────────────────────────────────────────────────────
def generate_signals(variant, close, high, low, opn, volume):
    """
    Return a DataFrame of {ticker: signal} where 1 = buy signal on that day.
    """
    signals = pd.DataFrame(0, index=close.index, columns=UNIVERSE)

    for tk in UNIVERSE:
        c = close[tk].dropna()
        h = high[tk].dropna()
        l = low[tk].dropna()
        o = opn[tk].dropna()
        v = volume[tk].dropna()
        # Align to common index
        idx = c.index.intersection(v.index).intersection(o.index)
        c, h, l, o, v = c.loc[idx], h.loc[idx], l.loc[idx], o.loc[idx], v.loc[idx]

        avg_vol_20 = v.rolling(20).mean()

        if variant == "A":
            # 20-day high breakout + volume surge 1.5x
            high_20 = h.rolling(20).max().shift(1)
            sig = (c > high_20) & (v > 1.5 * avg_vol_20)

        elif variant == "B":
            # 50-day high breakout + volume surge 1.5x
            high_50 = h.rolling(50).max().shift(1)
            sig = (c > high_50) & (v > 1.5 * avg_vol_20)

        elif variant == "C":
            # Bollinger Band breakout + RSI > 60
            bb_up = bollinger_upper(c, 20, 2)
            rsi_val = rsi(c, 14)
            # Rising RSI: today > yesterday
            rsi_rising = rsi_val > rsi_val.shift(1)
            sig = (c > bb_up) & (rsi_val > 60) & rsi_rising

        elif variant == "D":
            # Volume spike 2x + price > 10-SMA > 20-SMA
            sma10 = sma(c, 10)
            sma20 = sma(c, 20)
            sig = (v > 2 * avg_vol_20) & (c > sma10) & (sma10 > sma20)

        elif variant == "E":
            # Gap-up >2% on open + volume > 1.5x avg + close above open
            gap_pct = (o - c.shift(1)) / c.shift(1)
            sig = (gap_pct > 0.02) & (v > 1.5 * avg_vol_20) & (c > o)

        elif variant == "F":
            # MACD golden cross + price > 50-SMA
            macd_line, signal_line = macd(c)
            sma50 = sma(c, 50)
            cross = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))
            sig = cross & (c > sma50)

        else:
            raise ValueError(f"Unknown variant {variant}")

        signals.loc[sig[sig].index.intersection(signals.index), tk] = 1

    return signals


# ── Backtest Engine ────────────────────────────────────────────────────────
def get_hold_days(variant):
    hold_map = {"A": 10, "B": 15, "C": 10, "D": 10, "E": 5, "F": 15}
    return hold_map[variant]

def run_backtest(signals, close, variant):
    """
    Simple event-driven backtest with position limits.
    Returns trade_returns (list of per-trade pct returns) and daily_equity Series.
    """
    hold_days = get_hold_days(variant)
    dates = signals.index.tolist()

    positions = []  # list of (ticker, entry_date_idx, entry_price, shares)
    trade_returns = []
    equity = CAPITAL
    equity_curve = []

    for i, date in enumerate(dates):
        # Check exits
        new_positions = []
        for tk, entry_idx, entry_price, shares in positions:
            days_held = i - entry_idx
            if days_held >= hold_days:
                exit_price = close.loc[date, tk]
                if pd.isna(exit_price):
                    new_positions.append((tk, entry_idx, entry_price, shares))
                    continue
                # Apply slippage on exit
                exit_price *= (1 - SLIPPAGE_BPS / 10000)
                pnl = (exit_price - entry_price) * shares
                ret = (exit_price / entry_price) - 1
                trade_returns.append(ret)
                equity += pnl
            else:
                new_positions.append((tk, entry_idx, entry_price, shares))
        positions = new_positions

        # Check entries
        if len(positions) < MAX_CONCURRENT:
            day_signals = signals.loc[date]
            candidates = day_signals[day_signals == 1].index.tolist()
            np.random.shuffle(candidates)
            for tk in candidates:
                if len(positions) >= MAX_CONCURRENT:
                    break
                # Don't double up on same ticker
                if any(p[0] == tk for p in positions):
                    continue
                price = close.loc[date, tk]
                if pd.isna(price) or price <= 0:
                    continue
                # Apply slippage on entry
                price *= (1 + SLIPPAGE_BPS / 10000)
                alloc = min(MAX_PER_TRADE, equity / MAX_CONCURRENT)
                if alloc <= 0:
                    continue
                shares = alloc / price  # fractional shares allowed
                positions.append((tk, i, price, shares))

        # Mark-to-market
        mtm = equity
        for tk, entry_idx, entry_price, shares in positions:
            current = close.loc[date, tk]
            if pd.isna(current):
                continue
            mtm += (current - entry_price) * shares
        equity_curve.append(mtm)

    # Force close remaining positions at last available price
    last_date = dates[-1]
    for tk, entry_idx, entry_price, shares in positions:
        exit_price = close.loc[last_date, tk]
        if pd.isna(exit_price):
            continue
        exit_price *= (1 - SLIPPAGE_BPS / 10000)
        pnl = (exit_price - entry_price) * shares
        ret = (exit_price / entry_price) - 1
        trade_returns.append(ret)
        equity += pnl

    equity_series = pd.Series(equity_curve, index=dates)
    return trade_returns, equity_series


# ── Metrics ────────────────────────────────────────────────────────────────
def compute_metrics(trade_returns, equity_series):
    tr = np.array(trade_returns)
    if len(tr) == 0:
        return {k: 0.0 for k in [
            "sharpe", "sortino", "win_rate", "profit_factor",
            "max_drawdown", "total_return", "num_trades"
        ]}

    # Daily returns from equity curve
    daily_ret = equity_series.pct_change().dropna()
    ann_factor = np.sqrt(252)

    mean_d = daily_ret.mean()
    std_d = daily_ret.std()
    sharpe = (mean_d / std_d * ann_factor) if std_d > 0 else 0.0

    downside = daily_ret[daily_ret < 0].std()
    sortino = (mean_d / downside * ann_factor) if downside > 0 else 0.0

    win_rate = np.mean(tr > 0)

    gross_profit = tr[tr > 0].sum() if (tr > 0).any() else 0.0
    gross_loss = abs(tr[tr < 0].sum()) if (tr < 0).any() else 1e-9
    profit_factor = gross_profit / gross_loss

    # Max drawdown
    cum = equity_series.values
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    max_dd = dd.min()

    total_return = (equity_series.iloc[-1] / equity_series.iloc[0]) - 1

    return {
        "sharpe": round(float(sharpe), 4),
        "sortino": round(float(sortino), 4),
        "win_rate": round(float(win_rate), 4),
        "profit_factor": round(float(profit_factor), 4),
        "max_drawdown": round(float(max_dd), 4),
        "total_return": round(float(total_return), 4),
        "num_trades": int(len(tr)),
    }


# ── Regime Analysis ───────────────────────────────────────────────────────
def regime_analysis(trade_returns_by_date, spy_close, equity_series):
    """
    Classify each trade's entry date as bull/bear by SPY vs 200-SMA.
    Return bull_sharpe, bear_sharpe, regime_gap.
    """
    spy_sma200 = sma(spy_close, 200)
    bull_mask = spy_close > spy_sma200

    # Build daily return series split by regime
    daily_ret = equity_series.pct_change().dropna()
    bull_days = []
    bear_days = []

    for date, ret in daily_ret.items():
        if date in bull_mask.index and not pd.isna(bull_mask.loc[date]):
            if bull_mask.loc[date]:
                bull_days.append(ret)
            else:
                bear_days.append(ret)

    ann = np.sqrt(252)

    def _sharpe(rets):
        if len(rets) < 10:
            return 0.0
        r = np.array(rets)
        return float(r.mean() / r.std() * ann) if r.std() > 0 else 0.0

    bull_sharpe = _sharpe(bull_days)
    bear_sharpe = _sharpe(bear_days)

    denom = max(abs(bull_sharpe), abs(bear_sharpe), 1e-9)
    regime_gap = abs(bull_sharpe - bear_sharpe) / denom

    return round(bull_sharpe, 4), round(bear_sharpe, 4), round(regime_gap, 4)


# ── Permutation Test ───────────────────────────────────────────────────────
def permutation_test(signals, close, variant, actual_sharpe, n_perms=PERMUTATION_N):
    """
    Fast permutation test: compute forward returns for all signal entries,
    then shuffle assignment and recompute Sharpe.
    """
    hold_days = get_hold_days(variant)

    # Collect all (date_idx, forward_return) pairs from actual signals
    dates = signals.index.tolist()
    all_forward_returns = []
    for tk in UNIVERSE:
        sig_dates = signals.index[signals[tk] == 1].tolist()
        for d in sig_dates:
            idx = dates.index(d)
            exit_idx = min(idx + hold_days, len(dates) - 1)
            entry_p = close.loc[d, tk]
            exit_p = close.loc[dates[exit_idx], tk]
            if pd.isna(entry_p) or pd.isna(exit_p) or entry_p <= 0:
                continue
            entry_p *= (1 + SLIPPAGE_BPS / 10000)
            exit_p *= (1 - SLIPPAGE_BPS / 10000)
            all_forward_returns.append(exit_p / entry_p - 1)

    all_forward_returns = np.array(all_forward_returns)
    if len(all_forward_returns) < 5:
        return 1.0

    # Also precompute random forward returns from all possible entries
    all_possible_returns = []
    for tk in UNIVERSE:
        c = close[tk].dropna()
        for i in range(len(c) - hold_days):
            d = c.index[i]
            d_exit = c.index[min(i + hold_days, len(c) - 1)]
            entry_p = c.iloc[i] * (1 + SLIPPAGE_BPS / 10000)
            exit_p = c.loc[d_exit] * (1 - SLIPPAGE_BPS / 10000)
            if entry_p > 0:
                all_possible_returns.append(exit_p / entry_p - 1)
    all_possible_returns = np.array(all_possible_returns)

    n_trades = len(all_forward_returns)
    actual_mean = all_forward_returns.mean()
    actual_std = all_forward_returns.std()
    actual_trade_sharpe = actual_mean / actual_std if actual_std > 0 else 0

    count_better = 0
    for _ in range(n_perms):
        sample = np.random.choice(all_possible_returns, size=n_trades, replace=False)
        s_mean = sample.mean()
        s_std = sample.std()
        s_sharpe = s_mean / s_std if s_std > 0 else 0
        if s_sharpe >= actual_trade_sharpe:
            count_better += 1

    p_value = count_better / n_perms
    return round(p_value, 4)


# ── 5-Gate Validation ──────────────────────────────────────────────────────
def five_gate(metrics, bull_sharpe, bear_sharpe, regime_gap, perm_p):
    gates = {
        "sharpe_gt_0.5": metrics["sharpe"] > 0.5,
        "permutation_p_lt_0.05": perm_p < 0.05,
        "regime_gap_lt_0.5": regime_gap < 0.5,
        "max_dd_gt_neg50pct": metrics["max_drawdown"] > -0.50,
        "min_20_trades": metrics["num_trades"] >= 20,
    }
    gates["all_passed"] = all(gates.values())
    return gates


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    np.random.seed(42)
    close, high, low, opn, volume = download_data()

    spy_close = close["SPY"]

    variants = {
        "A": "20-day high breakout",
        "B": "50-day high breakout",
        "C": "Bollinger Band breakout",
        "D": "Volume spike + trend",
        "E": "Gap-up continuation",
        "F": "MACD golden cross",
    }

    results = {}

    for var_id, var_name in variants.items():
        print(f"\n{'='*60}")
        print(f"Variant {var_id}: {var_name}")
        print(f"{'='*60}")

        signals = generate_signals(var_id, close, high, low, opn, volume)
        total_signals = signals.sum().sum()
        print(f"  Total raw signals: {int(total_signals)}")

        trade_returns, equity_series = run_backtest(signals, close, var_id)
        metrics = compute_metrics(trade_returns, equity_series)
        print(f"  Trades: {metrics['num_trades']}, Sharpe: {metrics['sharpe']}, "
              f"Sortino: {metrics['sortino']}, WR: {metrics['win_rate']:.1%}")
        print(f"  PF: {metrics['profit_factor']}, MaxDD: {metrics['max_drawdown']:.1%}, "
              f"Return: {metrics['total_return']:.1%}")

        bull_s, bear_s, rgap = regime_analysis(trade_returns, spy_close, equity_series)
        print(f"  Bull Sharpe: {bull_s}, Bear Sharpe: {bear_s}, Regime Gap: {rgap}")

        print(f"  Running {PERMUTATION_N} permutations ...")
        perm_p = permutation_test(signals, close, var_id, metrics["sharpe"])
        print(f"  Permutation p-value: {perm_p}")

        gates = five_gate(metrics, bull_s, bear_s, rgap, perm_p)
        gate_status = "PASS" if gates["all_passed"] else "FAIL"
        print(f"  5-Gate: {gate_status}")
        for g, v in gates.items():
            if g != "all_passed":
                print(f"    {g}: {'PASS' if v else 'FAIL'}")

        results[f"variant_{var_id}"] = {
            "name": var_name,
            **metrics,
            "bull_sharpe": bull_s,
            "bear_sharpe": bear_s,
            "regime_gap": rgap,
            "permutation_p": perm_p,
            "five_gate": gates,
        }

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for k, v in results.items():
        status = "PASS" if v["five_gate"]["all_passed"] else "FAIL"
        print(f"  {k} ({v['name']}): Sharpe={v['sharpe']}, "
              f"Trades={v['num_trades']}, 5-Gate={status}")

    # Save results
    results["metadata"] = {
        "strategy": "Breakout Momentum on Quality Stocks",
        "universe": UNIVERSE,
        "capital": CAPITAL,
        "max_per_trade": MAX_PER_TRADE,
        "max_concurrent": MAX_CONCURRENT,
        "slippage_bps": SLIPPAGE_BPS,
        "period": f"{START} to {END}",
        "permutations": PERMUTATION_N,
        "run_timestamp": datetime.now().isoformat(),
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
