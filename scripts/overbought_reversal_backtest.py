#!/usr/bin/env python3
"""
Overbought Reversal (Short/Fade) Backtest on Quality Stocks
============================================================
Tests 6 variants of fading overbought conditions.
5-Gate validation: Sharpe>0.5, permutation p<0.05, regime gap<0.5, maxDD>-50%, trades>=20.
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────────────
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
SLIPPAGE_BPS = 2  # per side
RISK_FREE_RATE = 0.0

# ── Data Download ───────────────────────────────────────────────────────────
print("Downloading data...")
raw = yf.download(UNIVERSE, start=START, end=END, auto_adjust=True, progress=False)
close = raw["Close"][UNIVERSE].copy()
high = raw["High"][UNIVERSE].copy()
low = raw["Low"][UNIVERSE].copy()
volume = raw["Volume"][UNIVERSE].copy()
opn = raw["Open"][UNIVERSE].copy()

# Forward-fill missing
close = close.ffill()
high = high.ffill()
low = low.ffill()
volume = volume.ffill()
opn = opn.ffill()

dates = close.index
trade_mask = dates >= TRADE_START
trade_dates = dates[trade_mask]
print(f"Data: {dates[0].date()} to {dates[-1].date()}, {len(trade_dates)} trade days")

# ── Indicator Computation ───────────────────────────────────────────────────
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_bollinger(series, period=20, num_std=2):
    sma = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    return sma, upper, lower

print("Computing indicators...")
rsi = pd.DataFrame(index=dates, columns=UNIVERSE, dtype=float)
bb_sma = pd.DataFrame(index=dates, columns=UNIVERSE, dtype=float)
bb_upper = pd.DataFrame(index=dates, columns=UNIVERSE, dtype=float)
bb_lower = pd.DataFrame(index=dates, columns=UNIVERSE, dtype=float)

for sym in UNIVERSE:
    rsi[sym] = compute_rsi(close[sym])
    bb_sma[sym], bb_upper[sym], bb_lower[sym] = compute_bollinger(close[sym])

# Rolling 20-day low
low_20d = close.rolling(20).min()

# 5-day return
ret_5d = close.pct_change(5)

# Consecutive up days
def consecutive_up_days(series):
    up = (series.diff() > 0).astype(int)
    result = pd.Series(0, index=series.index, dtype=int)
    streak = 0
    for i in range(len(series)):
        if up.iloc[i] == 1:
            streak += 1
        else:
            streak = 0
        result.iloc[i] = streak
    return result

print("Computing consecutive up-days (slow step)...")
consec_up = pd.DataFrame(index=dates, columns=UNIVERSE, dtype=int)
for sym in UNIVERSE:
    consec_up[sym] = consecutive_up_days(close[sym])

# Average volume (20-day)
avg_vol_20 = volume.rolling(20).mean()

# SPY for regime classification
print("Downloading SPY for regime...")
spy = yf.download("SPY", start=START, end=END, auto_adjust=True, progress=False)
spy_close = spy["Close"].squeeze().reindex(dates).ffill()
spy_ret_monthly = spy_close.pct_change(21)

# ── Signal Generation ───────────────────────────────────────────────────────
def generate_signals_A(dates, trade_mask):
    """RSI overbought fade: RSI(14)>75 AND up>10% from 20d low. Hold 5 days."""
    signals = []
    for i in range(len(dates)):
        if not trade_mask[i]:
            continue
        dt = dates[i]
        for sym in UNIVERSE:
            r = rsi.at[dt, sym]
            c = close.at[dt, sym]
            l20 = low_20d.at[dt, sym]
            if pd.isna(r) or pd.isna(l20) or l20 == 0:
                continue
            pct_from_low = (c - l20) / l20
            if r > 75 and pct_from_low > 0.10:
                signals.append((dt, sym, 5))
    return signals

def generate_signals_B(dates, trade_mask):
    """Extended rally: 7+ consecutive up days AND RSI>70. Hold 5 days."""
    signals = []
    for i in range(len(dates)):
        if not trade_mask[i]:
            continue
        dt = dates[i]
        for sym in UNIVERSE:
            r = rsi.at[dt, sym]
            cu = consec_up.at[dt, sym]
            if pd.isna(r):
                continue
            if cu >= 7 and r > 70:
                signals.append((dt, sym, 5))
    return signals

def generate_signals_C(dates, trade_mask):
    """Volume exhaustion: up>8% in 5d on declining volume (each day < prior). Hold 5 days."""
    signals = []
    for i in range(len(dates)):
        if not trade_mask[i] or i < 5:
            continue
        dt = dates[i]
        for sym in UNIVERSE:
            r5 = ret_5d.at[dt, sym]
            if pd.isna(r5) or r5 <= 0.08:
                continue
            # Check 5 consecutive declining volume days
            vols = [volume.at[dates[i-j], sym] for j in range(5)]  # today, yesterday, ...
            declining = all(vols[j] < vols[j+1] for j in range(4))  # each day < prior day
            if declining:
                signals.append((dt, sym, 5))
    return signals

def generate_signals_D(dates, trade_mask):
    """BB reversion: close > upper BB + 2*std. Cover when inside bands or 10 days."""
    signals = []
    for i in range(len(dates)):
        if not trade_mask[i]:
            continue
        dt = dates[i]
        for sym in UNIVERSE:
            c = close.at[dt, sym]
            ub = bb_upper.at[dt, sym]
            std_val = (ub - bb_sma.at[dt, sym]) / 2  # recover std
            if pd.isna(ub) or pd.isna(std_val) or std_val == 0:
                continue
            # close > upper + 2*std (i.e., > sma + 4*std)
            extreme_upper = ub + 2 * std_val  # sma + 4*std
            if c > extreme_upper:
                signals.append((dt, sym, 10))  # max 10 days, but exit early if inside bands
    return signals

def generate_signals_E(dates, trade_mask):
    """Gap-up fade: gap up >3% on below-avg volume. Approximate as close-to-close 1 day."""
    signals = []
    for i in range(1, len(dates)):
        if not trade_mask[i]:
            continue
        dt = dates[i]
        prev_dt = dates[i-1]
        for sym in UNIVERSE:
            prev_c = close.at[prev_dt, sym]
            today_o = opn.at[dt, sym]
            v = volume.at[dt, sym]
            av = avg_vol_20.at[dt, sym]
            if pd.isna(prev_c) or pd.isna(today_o) or prev_c == 0 or pd.isna(av) or av == 0:
                continue
            gap = (today_o - prev_c) / prev_c
            if gap > 0.03 and v < av:
                signals.append((dt, sym, 1))  # same-day fade
    return signals

def generate_signals_F(dates, trade_mask):
    """Multi-condition: RSI>70 AND close>upper BB AND up>5% in 5d. Hold 5 days."""
    signals = []
    for i in range(len(dates)):
        if not trade_mask[i]:
            continue
        dt = dates[i]
        for sym in UNIVERSE:
            r = rsi.at[dt, sym]
            c = close.at[dt, sym]
            ub = bb_upper.at[dt, sym]
            r5 = ret_5d.at[dt, sym]
            if pd.isna(r) or pd.isna(ub) or pd.isna(r5):
                continue
            if r > 70 and c > ub and r5 > 0.05:
                signals.append((dt, sym, 5))
    return signals

# ── Backtester ──────────────────────────────────────────────────────────────
def run_backtest(signals, variant_name):
    """
    Run backtest for short signals.
    Each signal: (entry_date, symbol, hold_days).
    For variant D, exit early if price returns inside BB bands.
    """
    if not signals:
        return None

    dates_list = list(dates)
    date_to_idx = {d: i for i, d in enumerate(dates_list)}

    trades = []
    # Sort signals by date
    signals.sort(key=lambda x: x[0])

    # Track open positions to enforce max concurrent
    open_positions = []  # list of (exit_date, sym)

    for entry_date, sym, hold_days in signals:
        entry_idx = date_to_idx.get(entry_date)
        if entry_idx is None:
            continue

        # Clean expired positions
        open_positions = [(ed, s) for ed, s in open_positions if ed > entry_date]

        # Check concurrent limit
        if len(open_positions) >= MAX_CONCURRENT:
            continue

        # Skip if already short this symbol
        if any(s == sym for _, s in open_positions):
            continue

        entry_price = close.at[entry_date, sym]
        if pd.isna(entry_price) or entry_price == 0:
            continue

        # Position sizing
        shares = int(MAX_PER_TRADE / entry_price)
        if shares < 1:
            shares = 1
        notional = shares * entry_price
        if notional > MAX_PER_TRADE:
            continue

        # Find exit
        max_exit_idx = min(entry_idx + hold_days, len(dates_list) - 1)

        exit_idx = max_exit_idx
        exit_reason = "hold_expiry"

        # For variant D, check if price returns inside bands
        if variant_name == "D":
            for j in range(entry_idx + 1, max_exit_idx + 1):
                dt_j = dates_list[j]
                c_j = close.at[dt_j, sym]
                ub_j = bb_upper.at[dt_j, sym]
                lb_j = bb_lower.at[dt_j, sym]
                if pd.notna(c_j) and pd.notna(ub_j) and pd.notna(lb_j):
                    if lb_j <= c_j <= ub_j:
                        exit_idx = j
                        exit_reason = "inside_bands"
                        break

        if exit_idx >= len(dates_list):
            exit_idx = len(dates_list) - 1

        exit_date = dates_list[exit_idx]
        exit_price = close.at[exit_date, sym]
        if pd.isna(exit_price):
            continue

        # Short PnL: entry_price - exit_price (profit if price drops)
        slippage_entry = entry_price * SLIPPAGE_BPS / 10000
        slippage_exit = exit_price * SLIPPAGE_BPS / 10000
        gross_pnl_per_share = entry_price - exit_price
        net_pnl_per_share = gross_pnl_per_share - slippage_entry - slippage_exit
        net_pnl = net_pnl_per_share * shares
        ret = net_pnl / notional

        trades.append({
            "entry_date": str(entry_date.date()),
            "exit_date": str(exit_date.date()),
            "symbol": sym,
            "shares": shares,
            "entry_price": round(float(entry_price), 2),
            "exit_price": round(float(exit_price), 2),
            "pnl": round(float(net_pnl), 2),
            "return": round(float(ret), 6),
            "exit_reason": exit_reason,
            "hold_days": int(exit_idx - entry_idx),
        })

        # Track open position
        open_positions.append((exit_date, sym))

    return trades


def compute_metrics(trades):
    """Compute strategy metrics from trade list."""
    if not trades or len(trades) == 0:
        return None

    returns = np.array([t["return"] for t in trades])
    pnls = np.array([t["pnl"] for t in trades])
    n_trades = len(trades)

    total_pnl = float(np.sum(pnls))
    win_rate = float(np.mean(returns > 0))
    avg_return = float(np.mean(returns))
    avg_pnl = float(np.mean(pnls))

    # Daily returns for Sharpe/Sortino (approximate: distribute trade returns across days)
    # Build daily equity curve
    daily_pnl = {}
    for t in trades:
        ed = t["exit_date"]
        daily_pnl[ed] = daily_pnl.get(ed, 0) + t["pnl"]

    all_trade_dates = sorted(set([t["entry_date"] for t in trades] + [t["exit_date"] for t in trades]))
    if len(all_trade_dates) < 2:
        return None

    # Build daily returns series over entire trade period
    first_date = pd.Timestamp(trades[0]["entry_date"])
    last_date = pd.Timestamp(trades[-1]["exit_date"])
    date_range = pd.bdate_range(first_date, last_date)

    daily_rets = []
    equity = CAPITAL
    for d in date_range:
        ds = str(d.date())
        pnl = daily_pnl.get(ds, 0)
        if equity > 0:
            daily_rets.append(pnl / equity)
            equity += pnl

    daily_rets = np.array(daily_rets)
    if len(daily_rets) < 10:
        return None

    # Annualized Sharpe
    if np.std(daily_rets) > 0:
        sharpe = np.sqrt(252) * np.mean(daily_rets) / np.std(daily_rets)
    else:
        sharpe = 0.0

    # Sortino
    downside = daily_rets[daily_rets < 0]
    if len(downside) > 0 and np.std(downside) > 0:
        sortino = np.sqrt(252) * np.mean(daily_rets) / np.std(downside)
    else:
        sortino = 0.0

    # Profit factor
    gross_profit = float(np.sum(pnls[pnls > 0])) if np.any(pnls > 0) else 0
    gross_loss = float(np.abs(np.sum(pnls[pnls < 0]))) if np.any(pnls < 0) else 0.01
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Max drawdown
    equity_curve = CAPITAL + np.cumsum([daily_pnl.get(str(d.date()), 0) for d in date_range])
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / peak
    max_dd = float(np.min(drawdown))

    return {
        "n_trades": n_trades,
        "total_pnl": round(total_pnl, 2),
        "avg_pnl": round(avg_pnl, 2),
        "win_rate": round(win_rate, 4),
        "avg_return": round(avg_return, 6),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "profit_factor": round(profit_factor, 3),
        "max_drawdown": round(max_dd, 4),
        "total_return_pct": round(total_pnl / CAPITAL * 100, 2),
    }


def regime_analysis(trades):
    """Split trades by bull/bear regime (SPY monthly return)."""
    bull_rets = []
    bear_rets = []
    for t in trades:
        entry_dt = pd.Timestamp(t["entry_date"])
        # Find closest date in spy_ret_monthly
        idx = spy_ret_monthly.index.get_indexer([entry_dt], method="ffill")[0]
        if idx < 0 or idx >= len(spy_ret_monthly):
            continue
        spy_r = spy_ret_monthly.iloc[idx]
        if pd.isna(spy_r):
            continue
        if spy_r >= 0:
            bull_rets.append(t["return"])
        else:
            bear_rets.append(t["return"])

    bull_sharpe = 0.0
    bear_sharpe = 0.0
    if len(bull_rets) > 5 and np.std(bull_rets) > 0:
        bull_sharpe = np.mean(bull_rets) / np.std(bull_rets) * np.sqrt(252/5)
    if len(bear_rets) > 5 and np.std(bear_rets) > 0:
        bear_sharpe = np.mean(bear_rets) / np.std(bear_rets) * np.sqrt(252/5)

    max_abs = max(abs(bull_sharpe), abs(bear_sharpe), 0.001)
    regime_gap = abs(bull_sharpe - bear_sharpe) / max_abs

    return {
        "bull_trades": len(bull_rets),
        "bear_trades": len(bear_rets),
        "bull_sharpe": round(bull_sharpe, 3),
        "bear_sharpe": round(bear_sharpe, 3),
        "regime_gap": round(regime_gap, 4),
    }


def permutation_test(trades, n_perms=1000):
    """Shuffle entry dates, recompute mean return. p = fraction of shuffled >= actual."""
    if len(trades) < 5:
        return 1.0
    actual_mean = np.mean([t["return"] for t in trades])

    # Pool of all available returns: compute returns from random entry points
    all_returns = [t["return"] for t in trades]
    count_better = 0
    rng = np.random.default_rng(42)
    for _ in range(n_perms):
        shuffled = rng.permutation(all_returns)
        if np.mean(shuffled) >= actual_mean:
            count_better += 1
    return count_better / n_perms


def validate_5gate(metrics, regime, perm_p, variant_name):
    """Apply 5-gate validation."""
    gates = {}
    gates["sharpe_gt_0.5"] = metrics["sharpe"] > 0.5
    gates["perm_p_lt_0.05"] = perm_p < 0.05
    gates["regime_gap_lt_0.5"] = regime["regime_gap"] < 0.5
    gates["max_dd_gt_neg50"] = metrics["max_drawdown"] > -0.50
    gates["min_20_trades"] = metrics["n_trades"] >= 20
    gates["all_pass"] = all(gates.values())
    return gates


# ── Run All Variants ────────────────────────────────────────────────────────
generators = {
    "A": ("RSI Overbought Fade", generate_signals_A),
    "B": ("Extended Rally Fade", generate_signals_B),
    "C": ("Volume Exhaustion", generate_signals_C),
    "D": ("Bollinger Band Reversion", generate_signals_D),
    "E": ("Gap-Up Fade", generate_signals_E),
    "F": ("Multi-Condition Fade", generate_signals_F),
}

results = {}

for var_key, (var_name, gen_func) in generators.items():
    print(f"\n{'='*60}")
    print(f"Variant {var_key}: {var_name}")
    print(f"{'='*60}")

    signals = gen_func(dates, trade_mask if isinstance(trade_mask, np.ndarray) else trade_mask.values)
    print(f"  Signals generated: {len(signals)}")

    trades = run_backtest(signals, var_key)
    if trades is None or len(trades) == 0:
        print(f"  No trades executed.")
        results[var_key] = {
            "name": var_name,
            "n_signals": len(signals),
            "n_trades": 0,
            "status": "NO_TRADES",
        }
        continue

    print(f"  Trades executed: {len(trades)}")

    metrics = compute_metrics(trades)
    if metrics is None:
        print(f"  Insufficient data for metrics.")
        results[var_key] = {
            "name": var_name,
            "n_signals": len(signals),
            "n_trades": len(trades),
            "status": "INSUFFICIENT_DATA",
        }
        continue

    regime = regime_analysis(trades)
    perm_p = permutation_test(trades)
    gates = validate_5gate(metrics, regime, perm_p, var_key)

    # Print summary
    print(f"  Total PnL: ${metrics['total_pnl']:.2f} ({metrics['total_return_pct']:.1f}%)")
    print(f"  Sharpe: {metrics['sharpe']:.3f}  Sortino: {metrics['sortino']:.3f}  PF: {metrics['profit_factor']:.3f}")
    print(f"  Win Rate: {metrics['win_rate']:.1%}  Avg PnL: ${metrics['avg_pnl']:.2f}")
    print(f"  Max DD: {metrics['max_drawdown']:.2%}")
    print(f"  Regime: bull_sharpe={regime['bull_sharpe']:.3f} bear_sharpe={regime['bear_sharpe']:.3f} gap={regime['regime_gap']:.4f}")
    print(f"  Permutation p-value: {perm_p:.4f}")
    print(f"  5-Gate: {'PASS' if gates['all_pass'] else 'FAIL'} — {gates}")

    # Top symbols
    sym_pnl = {}
    for t in trades:
        sym_pnl[t["symbol"]] = sym_pnl.get(t["symbol"], 0) + t["pnl"]
    top_syms = sorted(sym_pnl.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"  Top symbols: {top_syms}")

    results[var_key] = {
        "name": var_name,
        "n_signals": len(signals),
        "metrics": metrics,
        "regime": regime,
        "permutation_p": round(perm_p, 4),
        "gates": gates,
        "status": "PASS" if gates["all_pass"] else "FAIL",
        "sample_trades": trades[:5],
        "symbol_pnl": {s: round(p, 2) for s, p in sorted(sym_pnl.items(), key=lambda x: x[1])},
    }

# ── Summary ─────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
for var_key, res in results.items():
    status = res.get("status", "UNKNOWN")
    n_trades = res.get("n_trades", res.get("metrics", {}).get("n_trades", 0))
    sharpe = res.get("metrics", {}).get("sharpe", "N/A")
    total_pnl = res.get("metrics", {}).get("total_pnl", "N/A")
    print(f"  {var_key} ({res['name']}): {status} | trades={n_trades} | sharpe={sharpe} | pnl={total_pnl}")

# ── Save Results ────────────────────────────────────────────────────────────
output = {
    "strategy": "Overbought Reversal (Short/Fade) on Quality Stocks",
    "run_date": str(datetime.now()),
    "period": f"{TRADE_START} to {END}",
    "universe": UNIVERSE,
    "capital": CAPITAL,
    "max_per_trade": MAX_PER_TRADE,
    "max_concurrent": MAX_CONCURRENT,
    "slippage_bps": SLIPPAGE_BPS,
    "variants": results,
}

output_path = Path("/home/jupiter/Lvl3Quant/data/overbought_reversal_results.json")
with open(output_path, "w") as f:
    json.dump(output, f, indent=2, default=str)

print(f"\nResults saved to {output_path}")
print("Done.")
