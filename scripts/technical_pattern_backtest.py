#!/usr/bin/env python3
"""
Technical Chart Pattern Backtest on Quality Stocks
Variants A-F: Double bottom, Higher low recovery, Support bounce,
              Mean reversion momentum shift, BB squeeze breakout, Volume climax reversal
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────────
UNIVERSE = [
    "AAPL", "MSFT", "AVGO", "JPM", "JNJ", "PG", "KO", "PEP", "HD", "COST",
    "UNH", "LLY", "V", "MA", "ABBV", "MRK", "WMT", "AMZN", "GOOGL", "META",
]
START = "2021-06-01"  # extra lookback for indicators
TRADE_START = "2022-01-01"
END = "2026-07-31"
CAPITAL = 669.0
MAX_PER_TRADE = 200.0
MAX_CONCURRENT = 3
SLIPPAGE_BPS = 2
HOLD_DAYS = 10
RESULTS_PATH = "/home/jupiter/Lvl3Quant/data/technical_pattern_results.json"

# ── Download data ───────────────────────────────────────────────────────
print("Downloading price data...")
raw = yf.download(UNIVERSE, start=START, end=END, auto_adjust=True, progress=False)
# Handle multi-level columns from yfinance
close_df = raw["Close"]
high_df = raw["High"]
low_df = raw["Low"]
open_df = raw["Open"]
volume_df = raw["Volume"]

# Also download SPY for regime classification
spy = yf.download("SPY", start=START, end=END, auto_adjust=True, progress=False)
spy_close = spy["Close"].squeeze()
spy_sma200 = spy_close.rolling(200).mean()
# Regime: bull if SPY > 200-SMA, bear otherwise
regime = (spy_close > spy_sma200).astype(int)  # 1=bull, 0=bear
regime.index = regime.index.tz_localize(None) if regime.index.tz else regime.index

# Ensure tz-naive indices
for df in [close_df, high_df, low_df, open_df, volume_df]:
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

dates = close_df.index
trade_start_idx = dates.get_indexer([pd.Timestamp(TRADE_START)], method="bfill")[0]

print(f"Data: {dates[0].date()} to {dates[-1].date()}, {len(dates)} days, {len(UNIVERSE)} stocks")


# ── Helper: compute indicators ──────────────────────────────────────────
def compute_indicators(ticker):
    """Return a DataFrame of indicators for one ticker."""
    c = close_df[ticker].dropna()
    h = high_df[ticker].dropna()
    l = low_df[ticker].dropna()
    o = open_df[ticker].dropna()
    v = volume_df[ticker].dropna()
    # Align all
    idx = c.index.intersection(h.index).intersection(l.index).intersection(o.index).intersection(v.index)
    df = pd.DataFrame({
        "close": c.reindex(idx), "high": h.reindex(idx),
        "low": l.reindex(idx), "open": o.reindex(idx), "volume": v.reindex(idx)
    })
    df["sma20"] = df["close"].rolling(20).mean()
    df["sma50"] = df["close"].rolling(50).mean()
    df["vol_avg20"] = df["volume"].rolling(20).mean()
    df["ret"] = df["close"].pct_change()
    df["rolling_min20"] = df["low"].rolling(20).min()

    # MACD
    ema12 = df["close"].ewm(span=12).mean()
    ema26 = df["close"].ewm(span=26).mean()
    macd_line = ema12 - ema26
    macd_signal = macd_line.ewm(span=9).mean()
    df["macd_hist"] = macd_line - macd_signal

    # Bollinger Bands
    df["bb_mid"] = df["close"].rolling(20).mean()
    bb_std = df["close"].rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + 2 * bb_std
    df["bb_lower"] = df["bb_mid"] - 2 * bb_std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
    df["bb_width_min20"] = df["bb_width"].rolling(20).min()

    # RSI
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    return df


# ── Signal generators ──────────────────────────────────────────────────
def signals_double_bottom(df):
    """A: Double bottom — two lows within 3%, 10-30 days apart, buy on breakout above peak."""
    signals = []
    lows = df["low"].values
    highs = df["high"].values
    closes = df["close"].values
    idx = df.index
    n = len(df)
    for i in range(60, n):
        # Look for first low in [i-40, i-10]
        for j in range(max(0, i - 40), max(0, i - 9)):
            low1 = lows[j]
            # Check if j is a local min (lowest in ±3 window)
            j_start = max(0, j - 3)
            j_end = min(n, j + 4)
            if low1 != lows[j_start:j_end].min():
                continue
            sep = i - j
            if sep < 10 or sep > 30:
                continue
            low2 = lows[i]
            # Check if i is a local min
            i_start = max(0, i - 3)
            i_end = min(n, i + 4)
            if low2 != lows[i_start:i_end].min():
                continue
            # Two lows within 3%
            if abs(low1 - low2) / max(low1, low2) > 0.03:
                continue
            # Peak between them
            peak = highs[j:i + 1].max()
            # Buy signal: close breaks above peak
            if closes[i] > peak:
                signals.append(idx[i])
                break  # one signal per day
    return signals


def signals_higher_low(df):
    """B: Higher low recovery — low (20-day min), then higher low 5-20 days later."""
    signals = []
    lows = df["low"].values
    rolling_min = df["rolling_min20"].values
    idx = df.index
    n = len(df)
    for i in range(30, n):
        # Is today a local low? (within 1% of 20-day rolling min)
        if np.isnan(rolling_min[i]):
            continue
        if lows[i] > rolling_min[i] * 1.01:
            continue
        # Look back 5-20 days for a LOWER low
        for j in range(max(0, i - 20), max(0, i - 4)):
            if lows[j] < lows[i] * 0.99:  # first low must be at least 1% lower
                # Today is the higher low → buy signal
                signals.append(idx[i])
                break
    return signals


def signals_support_bounce(df):
    """C: Support bounce — price touches 50-SMA from above after being >3% above it."""
    signals = []
    closes = df["close"].values
    sma50 = df["sma50"].values
    idx = df.index
    n = len(df)
    for i in range(55, n):
        if np.isnan(sma50[i]):
            continue
        # Currently within 1% of SMA50
        dist = (closes[i] - sma50[i]) / sma50[i]
        if dist < -0.01 or dist > 0.01:
            continue
        # Was >3% above in last 5-15 days
        was_above = False
        for j in range(max(0, i - 15), i):
            if not np.isnan(sma50[j]):
                if (closes[j] - sma50[j]) / sma50[j] > 0.03:
                    was_above = True
                    break
        if was_above:
            signals.append(idx[i])
    return signals


def signals_mean_rev_momentum(df):
    """D: Mean reversion momentum shift — MACD hist turns positive while price >5% below 20-SMA."""
    signals = []
    closes = df["close"].values
    sma20 = df["sma20"].values
    macd_hist = df["macd_hist"].values
    idx = df.index
    n = len(df)
    for i in range(26, n):
        if np.isnan(sma20[i]) or np.isnan(macd_hist[i]) or i < 1:
            continue
        if np.isnan(macd_hist[i - 1]):
            continue
        # MACD histogram crosses from negative to positive
        if macd_hist[i] > 0 and macd_hist[i - 1] <= 0:
            # Price >5% below 20-SMA
            dist = (closes[i] - sma20[i]) / sma20[i]
            if dist < -0.05:
                signals.append(idx[i])
    return signals


def signals_bb_squeeze(df):
    """E: BB squeeze breakout — bandwidth at 20-day low, price breaks above mid, RSI > 50."""
    signals = []
    closes = df["close"].values
    bb_width = df["bb_width"].values
    bb_width_min = df["bb_width_min20"].values
    bb_mid = df["bb_mid"].values
    rsi = df["rsi"].values
    idx = df.index
    n = len(df)
    for i in range(40, n):
        if any(np.isnan(x) for x in [bb_width[i], bb_width_min[i], bb_mid[i], rsi[i]]):
            continue
        # Squeeze: bandwidth at 20-day min (or within 5% of it)
        if bb_width[i] > bb_width_min[i] * 1.05:
            continue
        # Price breaks above middle band
        if closes[i] <= bb_mid[i]:
            continue
        # RSI > 50
        if rsi[i] <= 50:
            continue
        # Previous day was below or at mid band
        if i > 0 and closes[i - 1] > bb_mid[i - 1]:
            continue
        signals.append(idx[i])
    return signals


def signals_volume_climax(df):
    """F: Volume climax reversal — drop >3% on 2x avg volume, next day green."""
    signals = []
    closes = df["close"].values
    opens = df["open"].values
    vol = df["volume"].values
    vol_avg = df["vol_avg20"].values
    ret = df["ret"].values
    idx = df.index
    n = len(df)
    for i in range(21, n - 1):
        if np.isnan(vol_avg[i]) or np.isnan(ret[i]):
            continue
        # Drop >3% on 2x volume
        if ret[i] < -0.03 and vol[i] > 2 * vol_avg[i]:
            # Next day is green
            if i + 1 < n and closes[i + 1] > opens[i + 1]:
                signals.append(idx[i + 1])  # buy on the green day
    return signals


STRATEGIES = {
    "A_double_bottom": signals_double_bottom,
    "B_higher_low": signals_higher_low,
    "C_support_bounce": signals_support_bounce,
    "D_mean_rev_momentum": signals_mean_rev_momentum,
    "E_bb_squeeze": signals_bb_squeeze,
    "F_volume_climax": signals_volume_climax,
}


# ── Backtesting engine ─────────────────────────────────────────────────
def run_backtest(strategy_name, signal_func):
    """Run backtest for one strategy across all tickers."""
    all_trades = []
    trade_start_dt = pd.Timestamp(TRADE_START)

    for ticker in UNIVERSE:
        try:
            df = compute_indicators(ticker)
        except Exception:
            continue
        signal_dates = signal_func(df)
        c = df["close"]

        for entry_date in signal_dates:
            if entry_date < trade_start_dt:
                continue
            if entry_date not in c.index:
                continue
            entry_idx = c.index.get_loc(entry_date)
            exit_idx = min(entry_idx + HOLD_DAYS, len(c) - 1)
            if exit_idx <= entry_idx:
                continue

            entry_price = c.iloc[entry_idx]
            exit_price = c.iloc[exit_idx]
            exit_date = c.index[exit_idx]

            # Slippage
            entry_price *= (1 + SLIPPAGE_BPS / 10000)
            exit_price *= (1 - SLIPPAGE_BPS / 10000)

            # Position sizing
            shares = min(MAX_PER_TRADE, CAPITAL) / entry_price
            pnl = (exit_price - entry_price) * shares
            ret = (exit_price / entry_price) - 1

            all_trades.append({
                "ticker": ticker,
                "entry_date": str(entry_date.date()),
                "exit_date": str(exit_date.date()),
                "entry_price": round(float(entry_price), 2),
                "exit_price": round(float(exit_price), 2),
                "shares": round(float(shares), 4),
                "pnl": round(float(pnl), 2),
                "ret": round(float(ret), 6),
            })

    # Sort by entry date
    all_trades.sort(key=lambda x: x["entry_date"])

    # Apply concurrency limit: skip trades if MAX_CONCURRENT already open
    filtered = []
    for t in all_trades:
        entry_dt = pd.Timestamp(t["entry_date"])
        exit_dt = pd.Timestamp(t["exit_date"])
        active = sum(
            1 for ft in filtered
            if pd.Timestamp(ft["entry_date"]) <= entry_dt <= pd.Timestamp(ft["exit_date"])
        )
        if active < MAX_CONCURRENT:
            filtered.append(t)

    return filtered


def compute_metrics(trades):
    """Compute performance metrics from trade list."""
    if not trades:
        return None
    rets = np.array([t["ret"] for t in trades])
    pnls = np.array([t["pnl"] for t in trades])
    n = len(rets)

    total_pnl = float(pnls.sum())
    win_rate = float((rets > 0).mean())
    avg_ret = float(rets.mean())
    std_ret = float(rets.std()) if n > 1 else 0.001

    # Annualize: ~252/HOLD_DAYS trades per year per slot
    trades_per_year = 252 / HOLD_DAYS
    sharpe = (avg_ret / std_ret) * np.sqrt(trades_per_year) if std_ret > 0 else 0

    # Sortino
    downside = rets[rets < 0]
    downside_std = float(downside.std()) if len(downside) > 1 else 0.001
    sortino = (avg_ret / downside_std) * np.sqrt(trades_per_year) if downside_std > 0 else 0

    # Profit factor
    gross_profit = float(pnls[pnls > 0].sum()) if (pnls > 0).any() else 0
    gross_loss = float(abs(pnls[pnls < 0].sum())) if (pnls < 0).any() else 0.01
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Max drawdown on equity curve
    equity = np.cumsum(pnls) + CAPITAL
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd = float(dd.min())

    # Regime split
    bull_rets = []
    bear_rets = []
    for t in trades:
        d = pd.Timestamp(t["entry_date"])
        if d in regime.index:
            if regime.loc[d] == 1:
                bull_rets.append(t["ret"])
            else:
                bear_rets.append(t["ret"])
        else:
            # Find nearest
            nearest = regime.index.get_indexer([d], method="nearest")[0]
            if regime.iloc[nearest] == 1:
                bull_rets.append(t["ret"])
            else:
                bear_rets.append(t["ret"])

    bull_sharpe = 0
    bear_sharpe = 0
    if len(bull_rets) > 2:
        br = np.array(bull_rets)
        bull_sharpe = (br.mean() / br.std()) * np.sqrt(trades_per_year) if br.std() > 0 else 0
    if len(bear_rets) > 2:
        br2 = np.array(bear_rets)
        bear_sharpe = (br2.mean() / br2.std()) * np.sqrt(trades_per_year) if br2.std() > 0 else 0

    regime_gap = abs(bull_sharpe - bear_sharpe) / max(abs(bull_sharpe), abs(bear_sharpe), 0.001)

    return {
        "n_trades": n,
        "total_pnl": round(total_pnl, 2),
        "win_rate": round(win_rate, 4),
        "avg_ret": round(avg_ret, 6),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "profit_factor": round(profit_factor, 3),
        "max_drawdown": round(max_dd, 4),
        "bull_sharpe": round(bull_sharpe, 3),
        "bear_sharpe": round(bear_sharpe, 3),
        "regime_gap": round(regime_gap, 4),
        "bull_trades": len(bull_rets),
        "bear_trades": len(bear_rets),
    }


def permutation_test(trades, n_perms=1000):
    """Shuffle entry dates across the same ticker's history and compare Sharpe."""
    if len(trades) < 5:
        return 1.0
    actual_rets = np.array([t["ret"] for t in trades])
    actual_sharpe = actual_rets.mean() / actual_rets.std() if actual_rets.std() > 0 else 0

    # Build pool of possible returns per ticker
    ticker_returns = {}
    for ticker in UNIVERSE:
        try:
            c = close_df[ticker].dropna()
            c = c[c.index >= pd.Timestamp(TRADE_START)]
            if len(c) < HOLD_DAYS + 5:
                continue
            fwd = c.pct_change(HOLD_DAYS).shift(-HOLD_DAYS).dropna()
            ticker_returns[ticker] = fwd.values
        except Exception:
            continue

    if not ticker_returns:
        return 1.0

    all_pool = np.concatenate(list(ticker_returns.values()))
    n_trades = len(trades)
    count_better = 0

    rng = np.random.default_rng(42)
    for _ in range(n_perms):
        perm_rets = rng.choice(all_pool, size=n_trades, replace=True)
        perm_sharpe = perm_rets.mean() / perm_rets.std() if perm_rets.std() > 0 else 0
        if perm_sharpe >= actual_sharpe:
            count_better += 1

    return count_better / n_perms


# ── Run all strategies ──────────────────────────────────────────────────
results = {}
for name, func in STRATEGIES.items():
    print(f"\n{'='*60}")
    print(f"Strategy: {name}")
    print(f"{'='*60}")

    trades = run_backtest(name, func)
    metrics = compute_metrics(trades)

    if metrics is None:
        print("  No trades generated.")
        results[name] = {"status": "NO_TRADES", "gates": {}}
        continue

    print(f"  Trades: {metrics['n_trades']}")
    print(f"  Total PnL: ${metrics['total_pnl']:.2f}")
    print(f"  Win Rate: {metrics['win_rate']:.1%}")
    print(f"  Sharpe: {metrics['sharpe']:.3f}")
    print(f"  Sortino: {metrics['sortino']:.3f}")
    print(f"  Profit Factor: {metrics['profit_factor']:.3f}")
    print(f"  Max DD: {metrics['max_drawdown']:.2%}")
    print(f"  Bull Sharpe: {metrics['bull_sharpe']:.3f} ({metrics['bull_trades']} trades)")
    print(f"  Bear Sharpe: {metrics['bear_sharpe']:.3f} ({metrics['bear_trades']} trades)")
    print(f"  Regime Gap: {metrics['regime_gap']:.4f}")

    # Permutation test
    print("  Running permutation test (1000 shuffles)...")
    p_value = permutation_test(trades)
    print(f"  Permutation p-value: {p_value:.4f}")

    # 5-gate validation
    gates = {
        "sharpe_gt_0.5": metrics["sharpe"] > 0.5,
        "permutation_p_lt_0.05": p_value < 0.05,
        "regime_gap_lt_0.5": metrics["regime_gap"] < 0.5,
        "max_dd_gt_neg50pct": metrics["max_drawdown"] > -0.50,
        "min_20_trades": metrics["n_trades"] >= 20,
    }
    passed = sum(gates.values())
    status = "PASS" if all(gates.values()) else "FAIL"

    print(f"  Gates: {passed}/5 — {status}")
    for g, v in gates.items():
        print(f"    {'✓' if v else '✗'} {g}: {v}")

    results[name] = {
        "status": status,
        "metrics": metrics,
        "p_value": round(p_value, 4),
        "gates": gates,
        "gates_passed": f"{passed}/5",
        "sample_trades": trades[:5],
        "n_trades_total_before_concurrency_filter": "see logs",
    }

# ── Summary ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"{'Strategy':<25} {'Trades':>6} {'Sharpe':>7} {'Sortino':>8} {'WR':>6} {'PF':>6} {'MaxDD':>7} {'Gates':>6} {'Status':>6}")
print("-" * 70)
for name, r in results.items():
    if r["status"] == "NO_TRADES":
        print(f"{name:<25} {'—':>6} {'—':>7} {'—':>8} {'—':>6} {'—':>6} {'—':>7} {'0/5':>6} {'FAIL':>6}")
    else:
        m = r["metrics"]
        print(f"{name:<25} {m['n_trades']:>6} {m['sharpe']:>7.3f} {m['sortino']:>8.3f} "
              f"{m['win_rate']:>5.1%} {m['profit_factor']:>6.2f} {m['max_drawdown']:>6.2%} "
              f"{r['gates_passed']:>6} {r['status']:>6}")

# ── Save results ────────────────────────────────────────────────────────
output = {
    "backtest": "technical_chart_patterns",
    "run_date": datetime.now().isoformat(),
    "universe": UNIVERSE,
    "period": f"{TRADE_START} to {END}",
    "capital": CAPITAL,
    "max_per_trade": MAX_PER_TRADE,
    "max_concurrent": MAX_CONCURRENT,
    "slippage_bps": SLIPPAGE_BPS,
    "hold_days": HOLD_DAYS,
    "strategies": results,
}

with open(RESULTS_PATH, "w") as f:
    json.dump(output, f, indent=2, default=str)

print(f"\nResults saved to {RESULTS_PATH}")
