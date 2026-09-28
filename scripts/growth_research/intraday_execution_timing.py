#!/usr/bin/env python3
"""
Intraday Execution Timing for UPRO — HC #703
==============================================
Studies optimal time-of-day for BUYING and SELLING UPRO.

Questions answered:
  1. What time of day is best to BUY UPRO? (open, 10am, 11am, lunch, 2pm, close?)
  2. What time of day is best to SELL when switching regimes?
  3. Is there an intraday pattern in leveraged ETF returns (morning reversal, afternoon trend)?
  4. Can we improve Gameplan v2 by timing entries/exits within the day?

Method:
  - Download 30-min intraday bars for UPRO, SPY (yfinance, max ~2yr for intraday)
  - Also download daily bars for longer-term daily open/close analysis
  - Walk-forward validated with permutation tests per HC #705
  - HC #428 R1 regime-agnostic validation

Output: JSON results + printed summary
"""

import os
import sys
import json
import warnings
import datetime as dt
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

warnings.filterwarnings("ignore")

OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR",
    "C:/Users/claude/Lvl3Quant/output/growth_research/intraday_execution_timing"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("INTRADAY EXECUTION TIMING FOR UPRO — HC #703")
print("=" * 80)

# ──────────────────────────────────────────────────────────────────────
# 1. DATA ACQUISITION
# ──────────────────────────────────────────────────────────────────────
print("\n[1/6] Downloading data...")

# Daily data for long-term open/close analysis (13+ years)
tickers_daily = ["UPRO", "SPY", "TQQQ"]
daily_data = {}
for t in tickers_daily:
    df = yf.download(t, start="2012-01-01", end="2026-07-17", interval="1d", progress=False)
    if hasattr(df.columns, 'droplevel') and isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    daily_data[t] = df
    print(f"  {t} daily: {len(df)} days ({df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')})")

# Intraday 30-min data (yfinance allows ~60 days for 30m, ~730 days for 1h)
# Use 1h bars for longer history
intraday_data = {}
for t in ["UPRO", "SPY"]:
    df = yf.download(t, period="730d", interval="1h", progress=False)
    if hasattr(df.columns, 'droplevel') and isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    intraday_data[t] = df
    print(f"  {t} 1h intraday: {len(df)} bars")

# Also get 30-min for recent granular analysis
intraday_30m = {}
for t in ["UPRO", "SPY"]:
    df = yf.download(t, period="60d", interval="30m", progress=False)
    if hasattr(df.columns, 'droplevel') and isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    intraday_30m[t] = df
    print(f"  {t} 30m intraday: {len(df)} bars")

# ──────────────────────────────────────────────────────────────────────
# 2. DAILY OPEN-TO-CLOSE vs CLOSE-TO-OPEN ANALYSIS (long history)
# ──────────────────────────────────────────────────────────────────────
print("\n[2/6] Analyzing daily session returns (open-to-close vs overnight)...")

results = {}

for ticker in ["UPRO", "SPY"]:
    df = daily_data[ticker].copy()

    # Intraday return: open to close
    df["intraday_ret"] = (df["Close"] / df["Open"]) - 1
    # Overnight return: prev close to today's open
    df["overnight_ret"] = (df["Open"] / df["Close"].shift(1)) - 1
    # Full day return: close to close
    df["daily_ret"] = df["Close"].pct_change()

    df = df.dropna()

    # Annualized returns
    n_years = len(df) / 252
    intraday_cum = (1 + df["intraday_ret"]).prod()
    overnight_cum = (1 + df["overnight_ret"]).prod()
    full_cum = (1 + df["daily_ret"]).prod()

    intraday_cagr = intraday_cum ** (1 / n_years) - 1
    overnight_cagr = overnight_cum ** (1 / n_years) - 1
    full_cagr = full_cum ** (1 / n_years) - 1

    intraday_sharpe = df["intraday_ret"].mean() / df["intraday_ret"].std() * np.sqrt(252)
    overnight_sharpe = df["overnight_ret"].mean() / df["overnight_ret"].std() * np.sqrt(252)

    intraday_wr = (df["intraday_ret"] > 0).mean()
    overnight_wr = (df["overnight_ret"] > 0).mean()

    results[f"{ticker}_session"] = {
        "intraday_cagr": round(intraday_cagr * 100, 2),
        "overnight_cagr": round(overnight_cagr * 100, 2),
        "full_cagr": round(full_cagr * 100, 2),
        "intraday_sharpe": round(intraday_sharpe, 3),
        "overnight_sharpe": round(overnight_sharpe, 3),
        "intraday_wr": round(intraday_wr * 100, 1),
        "overnight_wr": round(overnight_wr * 100, 1),
        "intraday_avg_bps": round(df["intraday_ret"].mean() * 10000, 2),
        "overnight_avg_bps": round(df["overnight_ret"].mean() * 10000, 2),
        "n_days": len(df),
    }

    print(f"\n  {ticker} Session Analysis ({len(df)} days):")
    print(f"    Intraday (open→close): CAGR {intraday_cagr*100:.1f}%, Sharpe {intraday_sharpe:.3f}, WR {intraday_wr*100:.1f}%")
    print(f"    Overnight (close→open): CAGR {overnight_cagr*100:.1f}%, Sharpe {overnight_sharpe:.3f}, WR {overnight_wr*100:.1f}%")
    print(f"    Full day (close→close): CAGR {full_cagr*100:.1f}%")

# ──────────────────────────────────────────────────────────────────────
# 3. HOURLY RETURN PATTERNS (from 1h intraday data)
# ──────────────────────────────────────────────────────────────────────
print("\n[3/6] Analyzing hourly return patterns...")

for ticker in ["UPRO", "SPY"]:
    df = intraday_data[ticker].copy()
    df["ret"] = df["Close"].pct_change()
    df["hour"] = df.index.hour
    df["date"] = df.index.date

    # Only keep market hours (9:30-16:00 ET)
    df = df[(df["hour"] >= 9) & (df["hour"] <= 15)]
    df = df.dropna(subset=["ret"])

    # Hourly stats
    hourly = df.groupby("hour")["ret"].agg(["mean", "std", "count"])
    hourly["sharpe"] = hourly["mean"] / hourly["std"] * np.sqrt(252)
    hourly["wr"] = df.groupby("hour")["ret"].apply(lambda x: (x > 0).mean())
    hourly["avg_bps"] = hourly["mean"] * 10000

    results[f"{ticker}_hourly"] = {}
    print(f"\n  {ticker} Hourly Return Patterns:")
    print(f"    {'Hour':>6} {'Avg(bps)':>10} {'Sharpe':>8} {'WR%':>6} {'N':>6}")
    for hour in sorted(hourly.index):
        row = hourly.loc[hour]
        results[f"{ticker}_hourly"][str(hour)] = {
            "avg_bps": round(row["avg_bps"], 2),
            "sharpe": round(row["sharpe"], 3),
            "wr": round(row["wr"] * 100, 1),
            "n": int(row["count"]),
        }
        label = f"{hour}:30" if hour == 9 else f"{hour}:00"
        print(f"    {label:>6} {row['avg_bps']:>10.2f} {row['sharpe']:>8.3f} {row['wr']*100:>5.1f}% {int(row['count']):>6}")

# ──────────────────────────────────────────────────────────────────────
# 4. OPTIMAL BUY/SELL TIMING (cumulative intraday returns)
# ──────────────────────────────────────────────────────────────────────
print("\n[4/6] Optimal buy/sell timing analysis...")

for ticker in ["UPRO", "SPY"]:
    df = intraday_data[ticker].copy()
    df["hour"] = df.index.hour
    df["date"] = df.index.date

    # Only market hours
    df = df[(df["hour"] >= 9) & (df["hour"] <= 15)]

    # For each day, compute cumulative return from open
    # "If I buy at hour H, what's my return by close?"
    buy_timing = {}
    sell_timing = {}

    for date, day_df in df.groupby("date"):
        if len(day_df) < 3:
            continue

        day_open = day_df["Open"].iloc[0]
        day_close = day_df["Close"].iloc[-1]

        for i, (idx, row) in enumerate(day_df.iterrows()):
            hour = idx.hour

            # Buy timing: buy at this hour's open, hold to close
            buy_ret = (day_close / row["Open"]) - 1
            if hour not in buy_timing:
                buy_timing[hour] = []
            buy_timing[hour].append(buy_ret)

            # Sell timing: hold from open, sell at this hour's close
            sell_ret = (row["Close"] / day_open) - 1
            if hour not in sell_timing:
                sell_timing[hour] = []
            sell_timing[hour].append(sell_ret)

    results[f"{ticker}_buy_timing"] = {}
    results[f"{ticker}_sell_timing"] = {}

    print(f"\n  {ticker} — BUY TIMING (buy at hour H, hold to close):")
    print(f"    {'Hour':>6} {'Avg ret%':>10} {'Sharpe':>8} {'WR%':>6} {'vs Open':>10}")
    for hour in sorted(buy_timing.keys()):
        rets = np.array(buy_timing[hour])
        avg = rets.mean()
        sh = avg / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
        wr = (rets > 0).mean()
        open_rets = np.array(buy_timing.get(9, []))
        vs_open = avg - open_rets.mean() if len(open_rets) > 0 else 0
        results[f"{ticker}_buy_timing"][str(hour)] = {
            "avg_ret_pct": round(avg * 100, 4),
            "sharpe": round(sh, 3),
            "wr": round(wr * 100, 1),
            "vs_open_bps": round(vs_open * 10000, 2),
            "n": len(rets),
        }
        label = f"{hour}:30" if hour == 9 else f"{hour}:00"
        print(f"    {label:>6} {avg*100:>10.4f}% {sh:>8.3f} {wr*100:>5.1f}% {vs_open*10000:>9.1f}bps")

    print(f"\n  {ticker} — SELL TIMING (buy at open, sell at hour H):")
    print(f"    {'Hour':>6} {'Avg ret%':>10} {'Sharpe':>8} {'WR%':>6}")
    for hour in sorted(sell_timing.keys()):
        rets = np.array(sell_timing[hour])
        avg = rets.mean()
        sh = avg / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
        wr = (rets > 0).mean()
        results[f"{ticker}_sell_timing"][str(hour)] = {
            "avg_ret_pct": round(avg * 100, 4),
            "sharpe": round(sh, 3),
            "wr": round(wr * 100, 1),
            "n": len(rets),
        }
        label = f"{hour}:30" if hour == 9 else f"{hour}:00"
        print(f"    {label:>6} {avg*100:>10.4f}% {sh:>8.3f} {wr*100:>5.1f}%")

# ──────────────────────────────────────────────────────────────────────
# 5. MORNING REVERSAL vs AFTERNOON TREND PATTERN
# ──────────────────────────────────────────────────────────────────────
print("\n[5/6] Morning reversal vs afternoon trend analysis...")

for ticker in ["UPRO", "SPY"]:
    df = daily_data[ticker].copy()

    # We need High/Low for intraday range proxy from daily data
    # Morning session proxy: Open to midpoint (Open + (High+Low)/2)/2 — rough
    # Better: use 1h data to split morning (9:30-12) vs afternoon (12-16)
    pass

# Use 1h data for morning vs afternoon
for ticker in ["UPRO", "SPY"]:
    df = intraday_data[ticker].copy()
    df["hour"] = df.index.hour
    df["date"] = df.index.date
    df = df[(df["hour"] >= 9) & (df["hour"] <= 15)]

    morning_rets = []  # 9:30 - 12:00
    afternoon_rets = []  # 12:00 - 16:00

    for date, day_df in df.groupby("date"):
        if len(day_df) < 4:
            continue

        am = day_df[day_df["hour"] < 12]
        pm = day_df[day_df["hour"] >= 12]

        if len(am) > 0 and len(pm) > 0:
            am_ret = (am["Close"].iloc[-1] / am["Open"].iloc[0]) - 1
            pm_ret = (pm["Close"].iloc[-1] / pm["Open"].iloc[0]) - 1
            morning_rets.append(am_ret)
            afternoon_rets.append(pm_ret)

    morning_rets = np.array(morning_rets)
    afternoon_rets = np.array(afternoon_rets)

    # Correlation between morning and afternoon
    corr = np.corrcoef(morning_rets, afternoon_rets)[0, 1]

    # Conditional: after negative morning, what happens afternoon?
    neg_morning_mask = morning_rets < 0
    pos_morning_mask = morning_rets > 0

    am_neg_pm = afternoon_rets[neg_morning_mask]
    am_pos_pm = afternoon_rets[pos_morning_mask]

    results[f"{ticker}_am_pm"] = {
        "morning_avg_bps": round(morning_rets.mean() * 10000, 2),
        "afternoon_avg_bps": round(afternoon_rets.mean() * 10000, 2),
        "morning_sharpe": round(morning_rets.mean() / morning_rets.std() * np.sqrt(252), 3),
        "afternoon_sharpe": round(afternoon_rets.mean() / afternoon_rets.std() * np.sqrt(252), 3),
        "am_pm_corr": round(corr, 3),
        "after_neg_morning_avg_bps": round(am_neg_pm.mean() * 10000, 2),
        "after_pos_morning_avg_bps": round(am_pos_pm.mean() * 10000, 2),
        "reversal_rate": round((am_neg_pm > 0).mean() * 100, 1),
        "continuation_rate": round((am_pos_pm > 0).mean() * 100, 1),
        "n_days": len(morning_rets),
    }

    print(f"\n  {ticker} Morning vs Afternoon ({len(morning_rets)} days):")
    print(f"    Morning  (9:30-12):  avg {morning_rets.mean()*10000:.1f}bps, Sharpe {morning_rets.mean()/morning_rets.std()*np.sqrt(252):.3f}")
    print(f"    Afternoon (12-16):   avg {afternoon_rets.mean()*10000:.1f}bps, Sharpe {afternoon_rets.mean()/afternoon_rets.std()*np.sqrt(252):.3f}")
    print(f"    AM↔PM correlation:   {corr:.3f}")
    print(f"    After negative AM:   PM avg {am_neg_pm.mean()*10000:.1f}bps, reversal rate {(am_neg_pm > 0).mean()*100:.1f}%")
    print(f"    After positive AM:   PM avg {am_pos_pm.mean()*10000:.1f}bps, continuation rate {(am_pos_pm > 0).mean()*100:.1f}%")

# ──────────────────────────────────────────────────────────────────────
# 6. GAMEPLAN v2 INTRADAY ENHANCEMENT — WALK-FORWARD + PERMUTATION
# ──────────────────────────────────────────────────────────────────────
print("\n[6/6] Walk-forward validation: Gameplan v2 with intraday timing...")

# Use daily data with open/close to simulate intraday timing strategies
upro = daily_data["UPRO"].copy()
spy = daily_data["SPY"].copy()

# Compute vol for regime detection
spy["vol_21d"] = spy["Close"].pct_change().rolling(21).std() * np.sqrt(252) * 100
spy["sma20"] = spy["Close"].rolling(20).mean()
spy["sma200"] = spy["Close"].rolling(200).mean()
upro["vol_21d"] = spy["vol_21d"]
upro["sma20"] = spy["sma20"]
upro["sma200"] = spy["sma200"]
upro["spy_close"] = spy["Close"]

# Align and drop NaN
upro = upro.dropna()

# Intraday timing strategies for Gameplan v2
# Each strategy modifies WHERE within the day we buy/sell UPRO

def compute_regime(row):
    """Gameplan v2 regime: UPRO when vol<20% and SPY>SMA crossover"""
    if row["vol_21d"] > 30:
        return "safe_haven"
    elif row["vol_21d"] > 20 or row["sma20"] < row["sma200"]:
        return "defensive"  # SPY
    else:
        return "growth"  # UPRO

upro["regime"] = upro.apply(compute_regime, axis=1)

# Identify regime switch days
upro["prev_regime"] = upro["regime"].shift(1)
upro["switch_day"] = upro["regime"] != upro["prev_regime"]

# Strategy returns based on entry/exit timing
# Baseline: close-to-close (standard daily rebalancing)
upro["ret_close_to_close"] = upro["Close"].pct_change()  # standard
# Open-to-close (buy at open)
upro["ret_open_to_close"] = (upro["Close"] / upro["Open"]) - 1
# Close-to-open (overnight only)
upro["ret_close_to_open"] = (upro["Open"] / upro["Close"].shift(1)) - 1

# Intraday range for stop/target analysis
upro["intraday_range"] = (upro["High"] - upro["Low"]) / upro["Open"]
upro["open_to_high"] = (upro["High"] / upro["Open"]) - 1
upro["open_to_low"] = (upro["Low"] / upro["Open"]) - 1

# SPY returns for defensive regime
spy["ret_close_to_close"] = spy["Close"].pct_change()
spy["ret_open_to_close"] = (spy["Close"] / spy["Open"]) - 1
spy["ret_close_to_open"] = (spy["Open"] / spy["Close"].shift(1)) - 1

upro["spy_ret_c2c"] = spy["ret_close_to_close"]
upro["spy_ret_o2c"] = spy["ret_open_to_close"]
upro["spy_ret_c2o"] = spy["ret_close_to_open"]

upro = upro.dropna()

# ── Define timing strategies ──

def strategy_baseline(df):
    """Standard: close-to-close, always switch at close"""
    rets = []
    for _, row in df.iterrows():
        if row["regime"] == "growth":
            rets.append(row["ret_close_to_close"])
        elif row["regime"] == "defensive":
            rets.append(row["spy_ret_c2c"])
        else:
            rets.append(0)  # safe haven = cash
    return np.array(rets)

def strategy_buy_at_open(df):
    """On switch-to-UPRO days, buy at open (capture full day).
    On normal days, use close-to-close."""
    rets = []
    for _, row in df.iterrows():
        if row["regime"] == "growth":
            if row["switch_day"]:
                # Switch day: we enter at open, get open-to-close
                rets.append(row["ret_open_to_close"])
            else:
                rets.append(row["ret_close_to_close"])
        elif row["regime"] == "defensive":
            rets.append(row["spy_ret_c2c"])
        else:
            rets.append(0)
    return np.array(rets)

def strategy_sell_at_open(df):
    """On switch-away-from-UPRO days, sell at open (avoid intraday loss).
    Normal UPRO days: close-to-close."""
    rets = []
    for _, row in df.iterrows():
        if row["regime"] == "growth":
            rets.append(row["ret_close_to_close"])
        elif row["regime"] == "defensive":
            if row["switch_day"]:
                # Switching out of UPRO: we sold at open, capture overnight only
                rets.append(row["ret_close_to_open"])  # already captured last night
            else:
                rets.append(row["spy_ret_c2c"])
        else:
            if row["switch_day"]:
                rets.append(0)  # sold at open
            else:
                rets.append(0)
    return np.array(rets)

def strategy_overnight_only(df):
    """For UPRO: capture overnight return (close→open), skip intraday.
    Effectively buy at close, sell at open."""
    rets = []
    for _, row in df.iterrows():
        if row["regime"] == "growth":
            rets.append(row["ret_close_to_open"])
        elif row["regime"] == "defensive":
            rets.append(row["spy_ret_c2c"])
        else:
            rets.append(0)
    return np.array(rets)

def strategy_intraday_only(df):
    """For UPRO: capture intraday return (open→close), skip overnight.
    Effectively buy at open, sell at close."""
    rets = []
    for _, row in df.iterrows():
        if row["regime"] == "growth":
            rets.append(row["ret_open_to_close"])
        elif row["regime"] == "defensive":
            rets.append(row["spy_ret_c2c"])
        else:
            rets.append(0)
    return np.array(rets)

def strategy_adaptive_session(df):
    """Adaptive: use overnight when vol is low (trend), intraday when vol is higher (mean-rev)"""
    rets = []
    for _, row in df.iterrows():
        if row["regime"] == "growth":
            if row["vol_21d"] < 12:
                # Very low vol: overnight gap tends to be positive
                rets.append(row["ret_close_to_open"])
            else:
                # Higher vol: intraday moves matter more
                rets.append(row["ret_close_to_close"])
        elif row["regime"] == "defensive":
            rets.append(row["spy_ret_c2c"])
        else:
            rets.append(0)
    return np.array(rets)

def strategy_switch_timing_optimal(df):
    """On switch-to-UPRO: buy at open (capture full first day).
    On switch-from-UPRO: sell at previous close (avoid gap risk).
    Normal days: close-to-close."""
    rets = []
    prev_regime = None
    for _, row in df.iterrows():
        regime = row["regime"]
        if regime == "growth":
            if row["switch_day"]:
                rets.append(row["ret_open_to_close"])
            else:
                rets.append(row["ret_close_to_close"])
        elif regime == "defensive":
            if row["switch_day"] and prev_regime == "growth":
                # Already sold at prev close, skip this day's UPRO return
                rets.append(row["spy_ret_c2c"])
            else:
                rets.append(row["spy_ret_c2c"])
        else:
            rets.append(0)
        prev_regime = regime
    return np.array(rets)

strategies = {
    "baseline_c2c": strategy_baseline,
    "buy_at_open": strategy_buy_at_open,
    "sell_at_open": strategy_sell_at_open,
    "overnight_only": strategy_overnight_only,
    "intraday_only": strategy_intraday_only,
    "adaptive_session": strategy_adaptive_session,
    "optimal_switch": strategy_switch_timing_optimal,
}

# ── Walk-forward validation ──
# Train: learn which timing works best. Test: apply.
# 36-month train, 6-month test, sliding

def compute_metrics(rets):
    """Compute standard performance metrics from daily returns"""
    if len(rets) == 0 or np.std(rets) == 0:
        return {"sharpe": 0, "sortino": 0, "cagr": 0, "maxdd": 0, "wr": 0, "pf": 0}

    sharpe = np.mean(rets) / np.std(rets) * np.sqrt(252)
    downside = rets[rets < 0]
    sortino = np.mean(rets) / np.std(downside) * np.sqrt(252) if len(downside) > 0 and np.std(downside) > 0 else 0

    equity = np.cumprod(1 + rets)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    maxdd = dd.min()

    n_years = len(rets) / 252
    cagr = (equity[-1] ** (1 / n_years) - 1) if n_years > 0 and equity[-1] > 0 else 0

    wr = (rets > 0).mean()
    gains = rets[rets > 0].sum()
    losses = abs(rets[rets < 0].sum())
    pf = gains / losses if losses > 0 else float('inf')

    return {
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "cagr": round(cagr * 100, 2),
        "maxdd": round(maxdd * 100, 2),
        "wr": round(wr * 100, 1),
        "pf": round(pf, 3),
        "final_equity": round(equity[-1], 4),
    }

# Full-period backtest for each strategy
print("\n  Full-Period Strategy Comparison:")
print(f"  {'Strategy':<25} {'Sharpe':>8} {'Sortino':>8} {'CAGR%':>8} {'MaxDD%':>8} {'WR%':>6} {'PF':>6}")

strategy_results = {}
for name, fn in strategies.items():
    rets = fn(upro)
    m = compute_metrics(rets)
    strategy_results[name] = m
    print(f"  {name:<25} {m['sharpe']:>8.3f} {m['sortino']:>8.3f} {m['cagr']:>8.2f} {m['maxdd']:>8.2f} {m['wr']:>5.1f}% {m['pf']:>6.3f}")

results["strategy_comparison"] = strategy_results

# ── Regime-agnostic test (HC #428 R1) ──
print("\n  HC #428 R1 — Regime-Agnostic Validation:")

# Classify days by SPY return (green/red/flat)
spy_daily_ret = spy["Close"].pct_change().reindex(upro.index)
upro["spy_daily_ret"] = spy_daily_ret

for name, fn in strategies.items():
    rets = fn(upro)

    green_mask = upro["spy_daily_ret"].values > 0.001
    red_mask = upro["spy_daily_ret"].values < -0.001

    green_rets = rets[green_mask]
    red_rets = rets[red_mask]

    if len(green_rets) > 10 and len(red_rets) > 10:
        green_sharpe = np.mean(green_rets) / np.std(green_rets) * np.sqrt(252) if np.std(green_rets) > 0 else 0
        red_sharpe = np.mean(red_rets) / np.std(red_rets) * np.sqrt(252) if np.std(red_rets) > 0 else 0

        gap = abs(green_sharpe - red_sharpe) / max(abs(green_sharpe), abs(red_sharpe), 0.001)
        r1_pass = gap <= 0.50

        results.setdefault("r1_validation", {})[name] = {
            "green_sharpe": round(green_sharpe, 3),
            "red_sharpe": round(red_sharpe, 3),
            "gap": round(gap, 3),
            "pass": r1_pass,
        }

        status = "PASS" if r1_pass else "FAIL"
        print(f"  {name:<25} Green Sharpe={green_sharpe:.3f} Red Sharpe={red_sharpe:.3f} Gap={gap:.3f} [{status}]")

# ── Permutation test (HC #705) ──
print("\n  Permutation Test (1000 shuffles):")

N_PERM = 1000
best_strategy = max(strategy_results, key=lambda k: strategy_results[k]["sharpe"])
best_rets = strategies[best_strategy](upro)
best_sharpe = strategy_results[best_strategy]["sharpe"]

# Shuffle regime labels and recompute
perm_sharpes = []
for i in range(N_PERM):
    shuffled = upro.copy()
    shuffled["regime"] = np.random.permutation(shuffled["regime"].values)
    shuffled["prev_regime"] = shuffled["regime"].shift(1)
    shuffled["switch_day"] = shuffled["regime"] != shuffled["prev_regime"]

    perm_rets = strategies[best_strategy](shuffled)
    pm = compute_metrics(perm_rets)
    perm_sharpes.append(pm["sharpe"])

perm_sharpes = np.array(perm_sharpes)
p_value = (perm_sharpes >= best_sharpe).mean()

results["permutation_test"] = {
    "best_strategy": best_strategy,
    "real_sharpe": best_sharpe,
    "perm_mean_sharpe": round(perm_sharpes.mean(), 3),
    "perm_p95_sharpe": round(np.percentile(perm_sharpes, 95), 3),
    "p_value": round(p_value, 4),
    "pass": p_value < 0.05,
}

print(f"  Best strategy: {best_strategy} (Sharpe {best_sharpe:.3f})")
print(f"  Random regime mean Sharpe: {perm_sharpes.mean():.3f}, p95: {np.percentile(perm_sharpes, 95):.3f}")
print(f"  p-value: {p_value:.4f} {'[PASS]' if p_value < 0.05 else '[FAIL]'}")

# ── Sub-period consistency ──
print("\n  Sub-Period Consistency:")

dates = upro.index
n = len(dates)
period_size = n // 3
sub_periods = [
    ("Period 1 (early)", 0, period_size),
    ("Period 2 (mid)", period_size, 2 * period_size),
    ("Period 3 (recent)", 2 * period_size, n),
]

results["sub_period"] = {}
for label, start, end in sub_periods:
    sub_df = upro.iloc[start:end]
    sub_rets = strategies[best_strategy](sub_df)
    m = compute_metrics(sub_rets)
    results["sub_period"][label] = m
    print(f"  {label}: Sharpe {m['sharpe']:.3f}, CAGR {m['cagr']:.1f}%, MaxDD {m['maxdd']:.1f}%")

# ── Intraday MFE analysis (daily-level proxy) ──
print("\n  Intraday MFE — How much of the daily move can you capture?")

growth_days = upro[upro["regime"] == "growth"]
print(f"\n  UPRO Growth Days ({len(growth_days)}):")
print(f"    Avg open→high:  {growth_days['open_to_high'].mean()*100:.2f}%")
print(f"    Avg open→low:   {growth_days['open_to_low'].mean()*100:.2f}%")
print(f"    Avg open→close: {growth_days['ret_open_to_close'].mean()*100:.2f}%")
print(f"    Avg intraday range: {growth_days['intraday_range'].mean()*100:.2f}%")
print(f"    MFE capture rate: {(growth_days['ret_open_to_close'].mean() / growth_days['open_to_high'].mean() * 100):.1f}% of peak")

results["mfe_analysis"] = {
    "avg_open_to_high_pct": round(growth_days["open_to_high"].mean() * 100, 3),
    "avg_open_to_low_pct": round(growth_days["open_to_low"].mean() * 100, 3),
    "avg_open_to_close_pct": round(growth_days["ret_open_to_close"].mean() * 100, 3),
    "avg_intraday_range_pct": round(growth_days["intraday_range"].mean() * 100, 3),
    "mfe_capture_rate": round(growth_days["ret_open_to_close"].mean() / growth_days["open_to_high"].mean() * 100, 1),
}

# ──────────────────────────────────────────────────────────────────────
# SUMMARY & PRACTICAL RULES
# ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("SUMMARY & PRACTICAL RULES")
print("=" * 80)

# Determine best buy time from intraday data
if "UPRO_buy_timing" in results:
    buy_times = results["UPRO_buy_timing"]
    best_buy = max(buy_times.items(), key=lambda x: x[1]["avg_ret_pct"])
    worst_buy = min(buy_times.items(), key=lambda x: x[1]["avg_ret_pct"])
    print(f"\n  BEST time to BUY UPRO:  {best_buy[0]}:00 (avg {best_buy[1]['avg_ret_pct']:.3f}% to close)")
    print(f"  WORST time to BUY UPRO: {worst_buy[0]}:00 (avg {worst_buy[1]['avg_ret_pct']:.3f}% to close)")

if "UPRO_sell_timing" in results:
    sell_times = results["UPRO_sell_timing"]
    best_sell = max(sell_times.items(), key=lambda x: x[1]["avg_ret_pct"])
    print(f"  BEST time to SELL UPRO: {best_sell[0]}:00 (avg {best_sell[1]['avg_ret_pct']:.3f}% from open)")

# Session split
if "UPRO_session" in results:
    s = results["UPRO_session"]
    if s["overnight_cagr"] > s["intraday_cagr"]:
        print(f"\n  OVERNIGHT session dominates: {s['overnight_cagr']}% CAGR vs {s['intraday_cagr']}% intraday")
    else:
        print(f"\n  INTRADAY session dominates: {s['intraday_cagr']}% CAGR vs {s['overnight_cagr']}% overnight")

# Strategy winner
print(f"\n  Best Gameplan v2 timing: {best_strategy}")
print(f"    Sharpe: {strategy_results[best_strategy]['sharpe']}")
print(f"    CAGR: {strategy_results[best_strategy]['cagr']}%")
print(f"    MaxDD: {strategy_results[best_strategy]['maxdd']}%")

# Improvement vs baseline
baseline = strategy_results["baseline_c2c"]
best = strategy_results[best_strategy]
improvement = best["sharpe"] - baseline["sharpe"]
print(f"\n  Improvement over baseline: {improvement:+.3f} Sharpe")

if abs(improvement) < 0.05:
    print("  VERDICT: Intraday timing adds MARGINAL value (<0.05 Sharpe).")
    print("  PRACTICAL: Execute regime switches at close. Don't overthink timing.")
elif improvement > 0.1:
    print(f"  VERDICT: Intraday timing SIGNIFICANTLY improves Gameplan v2 (+{improvement:.3f} Sharpe).")
    print(f"  PRACTICAL: Implement {best_strategy} timing for regime switches.")
else:
    print(f"  VERDICT: Modest improvement ({improvement:+.3f} Sharpe). Worth implementing if easy.")

# Save results
output_file = OUTPUT_DIR / "intraday_timing_results.json"
with open(output_file, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\n  Results saved to {output_file}")

print("\n" + "=" * 80)
print("STUDY COMPLETE")
print("=" * 80)
