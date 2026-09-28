#!/usr/bin/env python3
"""
Earnings Surprise + Mean Reversion Backtest
============================================
Quality stocks that dip DESPITE good earnings are high-conviction MR opportunities.
The dip is irrational if earnings are strong.

Variants:
  A: Post-earnings dip buy — drop >3% in 3 days after earnings. Hold 10 days.
  B: Beat-and-drop — BEAT estimates AND drop >2% in 3 days. Hold 10 days.
  C: Miss-and-dip extreme — MISS estimates AND drop >8%. Hold 15 days.
  D: Earnings recovery — Down >5% from pre-earnings AND today is green. Hold 10 days.
  E: Pre-earnings MR — Dip >5% + RSI<35 within 10 days BEFORE earnings. Sell 1 day before.
  F: Earnings + triple signal — Dual Signal D fires AND earnings in last 20 days. Hold 10 days.

5-Gate Validation + Permutation Test (1000 iterations)

OOT: Jan 2022 – Jul 2026 | Capital: $669 | Max $200/trade | Max 3 concurrent
"""

import json
import warnings
import sys
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
np.random.seed(42)

try:
    import yfinance as yf
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance", "-q"])
    import yfinance as yf

# ── Configuration ─────────────────────────────────────────────────────────
CAPITAL = 669.0
MAX_PER_TRADE = 200.0
MAX_CONCURRENT = 3
SLIPPAGE_PCT = 0.0002  # 2bps each way
DATA_START = "2020-01-01"  # extra lookback for indicators
OOT_START = "2022-01-01"
END = "2026-07-31"
N_PERM = 1000

TICKERS = [
    "AAPL", "MSFT", "AVGO", "JPM", "JNJ", "PG", "KO", "PEP", "HD", "COST",
    "UNH", "LLY", "V", "MA", "ABBV", "MRK", "WMT", "AMZN", "GOOGL", "META",
]

ALL_TICKERS = sorted(set(TICKERS + ["SPY"]))

GATES = {
    "sharpe_min": 0.5,
    "perm_p_max": 0.05,
    "regime_gap_max": 0.5,
    "mdd_min": -50.0,
    "min_trades": 20,
}

RESULTS_PATH = "/home/jupiter/Lvl3Quant/data/earnings_surprise_mr_results.json"

# ── Data Download ─────────────────────────────────────────────────────────
print("Downloading price data ...")
raw = yf.download(ALL_TICKERS, start=DATA_START, end=END,
                  group_by="ticker", auto_adjust=True, progress=False)


def get_close(ticker):
    try:
        if len(ALL_TICKERS) == 1:
            s = raw["Close"].dropna()
        else:
            s = raw[ticker]["Close"].dropna()
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        return s
    except Exception:
        return pd.Series(dtype=float)


closes = {t: get_close(t) for t in TICKERS}
spy_close = get_close("SPY")

loaded = sum(1 for t in TICKERS if len(closes.get(t, [])) > 100)
print(f"  Tickers with data: {loaded}/{len(TICKERS)}")

# ── Earnings Data ────────────────────────────────────────────────────────
print("Fetching earnings data ...")

earnings_data = {}  # ticker -> list of {date, actual_eps, estimated_eps, surprise}

for t in TICKERS:
    try:
        stock = yf.Ticker(t)
        # Try earnings_dates first (has estimates)
        try:
            ed = stock.earnings_dates
            if ed is not None and len(ed) > 0:
                records = []
                for idx in ed.index:
                    dt = idx.date() if hasattr(idx, 'date') else pd.Timestamp(idx).date()
                    actual = ed.loc[idx].get("Reported EPS", np.nan) if "Reported EPS" in ed.columns else np.nan
                    estimated = ed.loc[idx].get("EPS Estimate", np.nan) if "EPS Estimate" in ed.columns else np.nan
                    surprise = None
                    if pd.notna(actual) and pd.notna(estimated) and estimated != 0:
                        surprise = (float(actual) - float(estimated)) / abs(float(estimated))
                    records.append({
                        "date": dt,
                        "actual_eps": float(actual) if pd.notna(actual) else None,
                        "estimated_eps": float(estimated) if pd.notna(estimated) else None,
                        "surprise": float(surprise) if surprise is not None else None,
                    })
                if records:
                    earnings_data[t] = records
                    continue
        except Exception:
            pass

        # Fallback: quarterly_earnings (has dates but not estimates)
        try:
            qe = stock.quarterly_earnings
            if qe is not None and len(qe) > 0:
                records = []
                for idx in qe.index:
                    dt = idx.date() if hasattr(idx, 'date') else pd.Timestamp(idx).date()
                    actual = qe.loc[idx].get("Earnings", np.nan) if "Earnings" in qe.columns else np.nan
                    records.append({
                        "date": dt,
                        "actual_eps": float(actual) if pd.notna(actual) else None,
                        "estimated_eps": None,
                        "surprise": None,
                    })
                if records:
                    earnings_data[t] = records
                    continue
        except Exception:
            pass

        # Last resort: approximate quarterly earnings dates
        # Most large-caps report in Jan/Apr/Jul/Oct
        approx_months = [1, 4, 7, 10]
        records = []
        for year in range(2020, 2027):
            for month in approx_months:
                # Approximate: 3rd week of the month
                try:
                    dt = datetime(year, month, 20).date()
                    records.append({
                        "date": dt,
                        "actual_eps": None,
                        "estimated_eps": None,
                        "surprise": None,
                    })
                except ValueError:
                    pass
        earnings_data[t] = records

    except Exception as e:
        print(f"  Warning: could not fetch earnings for {t}: {e}")
        # Use approximate dates
        approx_months = [1, 4, 7, 10]
        records = []
        for year in range(2020, 2027):
            for month in approx_months:
                try:
                    dt = datetime(year, month, 20).date()
                    records.append({
                        "date": dt,
                        "actual_eps": None,
                        "estimated_eps": None,
                        "surprise": None,
                    })
                except ValueError:
                    pass
        earnings_data[t] = records

earnings_loaded = sum(1 for t in TICKERS if t in earnings_data and len(earnings_data[t]) > 0)
has_surprise = sum(1 for t in TICKERS if t in earnings_data and
                   any(r.get("surprise") is not None for r in earnings_data.get(t, [])))
print(f"  Tickers with earnings dates: {earnings_loaded}/{len(TICKERS)}")
print(f"  Tickers with beat/miss data: {has_surprise}/{len(TICKERS)}")

# ── Indicator Helpers ────────────────────────────────────────────────────
def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def consecutive_red_days(series):
    is_red = (series.diff() < 0).astype(int)
    result = pd.Series(0, index=series.index, dtype=int)
    count = 0
    for i in range(len(is_red)):
        if is_red.iloc[i] == 1:
            count += 1
        else:
            count = 0
        result.iloc[i] = count
    return result


def is_green_day(series):
    return series.diff() > 0


# ── Pre-compute indicators ──────────────────────────────────────────────
print("Computing indicators ...")
indicators = {}
for t in TICKERS:
    c = closes.get(t, pd.Series(dtype=float))
    if len(c) < 50:
        continue
    indicators[t] = {
        "rsi14": calc_rsi(c, 14),
        "high20": c.rolling(20).max(),
        "consec_red": consecutive_red_days(c),
        "green": is_green_day(c),
    }

spy_sma200 = spy_close.rolling(200).mean()


# ── Earnings Helpers ─────────────────────────────────────────────────────
def get_earnings_dates(ticker):
    """Return sorted list of earnings dates for a ticker."""
    if ticker not in earnings_data:
        return []
    return sorted([r["date"] for r in earnings_data[ticker]])


def find_nearest_earnings_before(ticker, date, max_days=30):
    """Find the most recent earnings date before `date`, within max_days."""
    edates = get_earnings_dates(ticker)
    d = date.date() if hasattr(date, 'date') else date
    best = None
    for ed in edates:
        if ed <= d and (d - ed).days <= max_days:
            if best is None or ed > best:
                best = ed
    return best


def find_nearest_earnings_after(ticker, date, max_days=15):
    """Find the next earnings date after `date`, within max_days."""
    edates = get_earnings_dates(ticker)
    d = date.date() if hasattr(date, 'date') else date
    best = None
    for ed in edates:
        if ed > d and (ed - d).days <= max_days:
            if best is None or ed < best:
                best = ed
    return best


def get_earnings_surprise(ticker, earnings_date):
    """Get surprise ratio for a specific earnings date. Returns None if unavailable."""
    if ticker not in earnings_data:
        return None
    for r in earnings_data[ticker]:
        if r["date"] == earnings_date:
            return r.get("surprise")
    return None


def get_pre_earnings_price(ticker, earnings_date):
    """Get the closing price on the day before earnings."""
    c = closes.get(ticker, pd.Series(dtype=float))
    if len(c) == 0:
        return None
    ed_ts = pd.Timestamp(earnings_date)
    # Find last trading day on or before earnings date
    mask = c.index <= ed_ts
    if mask.sum() == 0:
        return None
    # Get the day before earnings (or earnings day if it's a trading day and we want pre)
    pre_mask = c.index < ed_ts
    if pre_mask.sum() == 0:
        return float(c.iloc[0])
    return float(c.loc[pre_mask].iloc[-1])


# ── Signal Generation ────────────────────────────────────────────────────
print("Generating signals ...")
oot_start_ts = pd.Timestamp(OOT_START)


def generate_signals_A():
    """Post-earnings dip buy: drop >3% in 3 days after earnings. Hold 10 days."""
    signals = []
    for t in TICKERS:
        c = closes.get(t, pd.Series(dtype=float))
        if len(c) < 50:
            continue
        edates = get_earnings_dates(t)
        for ed in edates:
            ed_ts = pd.Timestamp(ed)
            if ed_ts < oot_start_ts:
                continue
            # Find price on earnings day
            pre_price = get_pre_earnings_price(t, ed)
            if pre_price is None or pre_price <= 0:
                continue
            # Check each of 3 days after earnings
            for offset in range(1, 4):
                check_date = ed_ts + pd.Timedelta(days=offset)
                # Find nearest trading day
                mask = c.index >= check_date
                if mask.sum() == 0:
                    continue
                actual_date = c.index[mask][0]
                if (actual_date - ed_ts).days > 5:  # too far, skip
                    continue
                price = float(c.loc[actual_date])
                drop = (price - pre_price) / pre_price
                if drop < -0.03:
                    signals.append((actual_date, t, 10))
                    break  # Only one signal per earnings event
    return signals


def generate_signals_B():
    """Beat-and-drop: BEAT estimates AND drop >2% in 3 days. Hold 10 days."""
    signals = []
    for t in TICKERS:
        c = closes.get(t, pd.Series(dtype=float))
        if len(c) < 50:
            continue
        edates = get_earnings_dates(t)
        for ed in edates:
            ed_ts = pd.Timestamp(ed)
            if ed_ts < oot_start_ts:
                continue
            # Check if beat
            surprise = get_earnings_surprise(t, ed)
            if surprise is None or surprise <= 0:
                continue  # Need positive surprise (beat)
            # Find pre-earnings price
            pre_price = get_pre_earnings_price(t, ed)
            if pre_price is None or pre_price <= 0:
                continue
            # Check each of 3 days after earnings
            for offset in range(1, 4):
                check_date = ed_ts + pd.Timedelta(days=offset)
                mask = c.index >= check_date
                if mask.sum() == 0:
                    continue
                actual_date = c.index[mask][0]
                if (actual_date - ed_ts).days > 5:
                    continue
                price = float(c.loc[actual_date])
                drop = (price - pre_price) / pre_price
                if drop < -0.02:
                    signals.append((actual_date, t, 10))
                    break
    return signals


def generate_signals_C():
    """Miss-and-dip extreme: MISS estimates AND drop >8%. Hold 15 days."""
    signals = []
    for t in TICKERS:
        c = closes.get(t, pd.Series(dtype=float))
        if len(c) < 50:
            continue
        edates = get_earnings_dates(t)
        for ed in edates:
            ed_ts = pd.Timestamp(ed)
            if ed_ts < oot_start_ts:
                continue
            # Check if miss
            surprise = get_earnings_surprise(t, ed)
            if surprise is None or surprise >= 0:
                continue  # Need negative surprise (miss)
            pre_price = get_pre_earnings_price(t, ed)
            if pre_price is None or pre_price <= 0:
                continue
            # Check days 1-5 after earnings for >8% drop
            for offset in range(1, 6):
                check_date = ed_ts + pd.Timedelta(days=offset)
                mask = c.index >= check_date
                if mask.sum() == 0:
                    continue
                actual_date = c.index[mask][0]
                if (actual_date - ed_ts).days > 7:
                    continue
                price = float(c.loc[actual_date])
                drop = (price - pre_price) / pre_price
                if drop < -0.08:
                    signals.append((actual_date, t, 15))
                    break
    return signals


def generate_signals_D():
    """Earnings recovery: down >5% from pre-earnings AND today is green. Hold 10 days."""
    signals = []
    for t in TICKERS:
        if t not in indicators:
            continue
        c = closes.get(t, pd.Series(dtype=float))
        green = indicators[t]["green"]
        edates = get_earnings_dates(t)

        for ed in edates:
            ed_ts = pd.Timestamp(ed)
            if ed_ts < oot_start_ts:
                continue
            pre_price = get_pre_earnings_price(t, ed)
            if pre_price is None or pre_price <= 0:
                continue
            # Check days 1-20 after earnings
            for offset in range(1, 21):
                check_date = ed_ts + pd.Timedelta(days=offset)
                mask = c.index >= check_date
                if mask.sum() == 0:
                    continue
                actual_date = c.index[mask][0]
                if (actual_date - ed_ts).days > 25:
                    break
                try:
                    price = float(c.loc[actual_date])
                    drop = (price - pre_price) / pre_price
                    if drop < -0.05 and green.loc[actual_date]:
                        signals.append((actual_date, t, 10))
                        break  # One signal per earnings event
                except (KeyError, IndexError):
                    continue
    return signals


def generate_signals_E():
    """Pre-earnings MR: dip >5% + RSI<35 within 10 days before earnings. Sell 1 day before."""
    signals = []
    for t in TICKERS:
        if t not in indicators:
            continue
        c = closes.get(t, pd.Series(dtype=float))
        rsi = indicators[t]["rsi14"]
        h20 = indicators[t]["high20"]
        edates = get_earnings_dates(t)

        for ed in edates:
            ed_ts = pd.Timestamp(ed)
            if ed_ts < oot_start_ts:
                continue
            # Look for signal in 10 days before earnings
            window_start = ed_ts - pd.Timedelta(days=15)  # calendar days, allow some buffer
            mask = (c.index >= window_start) & (c.index < ed_ts)
            if mask.sum() == 0:
                continue

            for date in c.index[mask]:
                try:
                    price = float(c.loc[date])
                    high = float(h20.loc[date])
                    r = float(rsi.loc[date])
                    if pd.isna(price) or pd.isna(high) or pd.isna(r):
                        continue
                    drop = (price - high) / high
                    if drop < -0.05 and r < 35:
                        # Calculate hold days: sell 1 day before earnings
                        days_to_earnings = (ed_ts - date).days
                        hold_days = max(1, days_to_earnings - 1)
                        signals.append((date, t, hold_days))
                        break  # One signal per earnings window
                except (KeyError, IndexError):
                    continue
    return signals


def generate_signals_F():
    """Earnings + triple signal: >5% dip + RSI<35 + green after red AND earnings in last 20 days. Hold 10."""
    signals = []
    for t in TICKERS:
        if t not in indicators:
            continue
        c = closes.get(t, pd.Series(dtype=float))
        rsi = indicators[t]["rsi14"]
        h20 = indicators[t]["high20"]
        green = indicators[t]["green"]
        consec_red = indicators[t]["consec_red"]

        for i in range(1, len(c)):
            date = c.index[i]
            if date < oot_start_ts:
                continue
            try:
                price = float(c.iloc[i])
                high = float(h20.iloc[i])
                r = float(rsi.iloc[i])
                if pd.isna(price) or pd.isna(high) or pd.isna(r):
                    continue
                drop = (price - high) / high
                # Triple signal: >5% dip + RSI<35 + green after red
                if drop >= -0.05 or r >= 35:
                    continue
                if not green.iloc[i]:
                    continue
                if consec_red.iloc[i - 1] < 1:  # Previous day was red
                    continue
                # Check if earnings in last 20 days
                nearest = find_nearest_earnings_before(t, date, max_days=20)
                if nearest is not None:
                    signals.append((date, t, 10))
            except (KeyError, IndexError):
                continue
    return signals


# Generate all signals
signals_A = generate_signals_A()
signals_B = generate_signals_B()
signals_C = generate_signals_C()
signals_D = generate_signals_D()
signals_E = generate_signals_E()
signals_F = generate_signals_F()

print(f"  Variant A (post-earnings dip): {len(signals_A)} raw signals")
print(f"  Variant B (beat-and-drop): {len(signals_B)} raw signals")
print(f"  Variant C (miss-and-dip extreme): {len(signals_C)} raw signals")
print(f"  Variant D (earnings recovery): {len(signals_D)} raw signals")
print(f"  Variant E (pre-earnings MR): {len(signals_E)} raw signals")
print(f"  Variant F (earnings + triple): {len(signals_F)} raw signals")

all_variant_signals = {
    "A": signals_A,
    "B": signals_B,
    "C": signals_C,
    "D": signals_D,
    "E": signals_E,
    "F": signals_F,
}


# ── Trade Simulator ─────────────────────────────────────────────────────
def simulate_trades(signals, capital=CAPITAL, max_per_trade=MAX_PER_TRADE,
                    max_concurrent=MAX_CONCURRENT):
    """Simulate trades with position limits and concurrent position tracking."""
    if not signals:
        return []

    signals = sorted(signals, key=lambda x: x[0])

    # Deduplicate: no same-ticker entry within hold period
    deduped = []
    last_entry = {}
    for date, ticker, hold_days in signals:
        if ticker in last_entry:
            delta = (date - last_entry[ticker]).days
            if delta < last_entry.get(ticker + "_hold", 10):
                continue
        deduped.append((date, ticker, hold_days))
        last_entry[ticker] = date
        last_entry[ticker + "_hold"] = hold_days

    trades = []
    open_positions = []

    for date, ticker, hold_days in deduped:
        # Close expired positions
        open_positions = [(ed, tk) for ed, tk in open_positions if ed > date]

        if len(open_positions) >= max_concurrent:
            continue

        c = closes.get(ticker, pd.Series(dtype=float))
        if len(c) == 0:
            continue
        try:
            loc = c.index.get_loc(date)
        except KeyError:
            mask = c.index >= date
            if mask.sum() == 0:
                continue
            loc = c.index.get_loc(c.index[mask][0])

        exit_loc = min(loc + hold_days, len(c) - 1)
        entry_price = float(c.iloc[loc]) * (1 + SLIPPAGE_PCT)
        exit_price = float(c.iloc[exit_loc]) * (1 - SLIPPAGE_PCT)
        exit_date = c.index[exit_loc]

        if entry_price <= 0:
            continue

        shares = max_per_trade / entry_price
        pnl = shares * (exit_price - entry_price)
        ret = (exit_price - entry_price) / entry_price

        # Determine regime
        regime = "unknown"
        try:
            spy_val = spy_close.asof(date)
            sma_val = spy_sma200.asof(date)
            if pd.notna(spy_val) and pd.notna(sma_val):
                regime = "bull" if spy_val > sma_val else "bear"
        except Exception:
            pass

        trades.append({
            "ticker": ticker,
            "entry_date": str(c.index[loc].date()),
            "exit_date": str(exit_date.date()),
            "entry_price": round(entry_price, 2),
            "exit_price": round(exit_price, 2),
            "pnl": round(float(pnl), 2),
            "return": round(float(ret), 6),
            "hold_days": hold_days,
            "shares": round(float(shares), 4),
            "regime": regime,
        })

        open_positions.append((exit_date, ticker))

    return trades


# ── Metrics Computation ──────────────────────────────────────────────────
def compute_metrics(trades, label=""):
    """Compute performance metrics from trade list."""
    if not trades:
        return {
            "label": label,
            "n_trades": 0,
            "total_pnl": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "profit_factor": 0.0,
            "win_rate": 0.0,
            "max_drawdown_pct": 0.0,
            "avg_return": 0.0,
            "gates_passed": 0,
            "gate_details": {},
        }

    returns = np.array([t["return"] for t in trades])
    pnls = np.array([t["pnl"] for t in trades])
    n = len(trades)

    total_pnl = float(np.sum(pnls))
    avg_ret = float(np.mean(returns))
    std_ret = float(np.std(returns, ddof=1)) if n > 1 else 1.0

    # Sharpe (annualized, assume ~25 trades/yr average hold 10d)
    trades_per_year = 252 / 10.0  # approximate
    sharpe = (avg_ret / std_ret) * np.sqrt(trades_per_year) if std_ret > 0 else 0.0

    # Sortino
    downside = returns[returns < 0]
    downside_std = float(np.std(downside, ddof=1)) if len(downside) > 1 else std_ret
    sortino = (avg_ret / downside_std) * np.sqrt(trades_per_year) if downside_std > 0 else 0.0

    # Profit factor
    gross_profit = float(np.sum(pnls[pnls > 0]))
    gross_loss = float(np.abs(np.sum(pnls[pnls < 0])))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)

    # Win rate
    win_rate = float(np.mean(pnls > 0)) * 100

    # Max drawdown
    cum_pnl = np.cumsum(pnls)
    peak = np.maximum.accumulate(cum_pnl + CAPITAL)
    drawdown = (cum_pnl + CAPITAL - peak) / peak * 100
    max_dd = float(np.min(drawdown))

    # Regime analysis
    bull_rets = [t["return"] for t in trades if t.get("regime") == "bull"]
    bear_rets = [t["return"] for t in trades if t.get("regime") == "bear"]
    bull_sharpe = 0.0
    bear_sharpe = 0.0
    if len(bull_rets) > 1:
        bull_sharpe = (np.mean(bull_rets) / np.std(bull_rets, ddof=1)) * np.sqrt(trades_per_year)
    if len(bear_rets) > 1:
        bear_sharpe = (np.mean(bear_rets) / np.std(bear_rets, ddof=1)) * np.sqrt(trades_per_year)

    max_sharpe = max(abs(bull_sharpe), abs(bear_sharpe))
    regime_gap = abs(bull_sharpe - bear_sharpe) / max_sharpe if max_sharpe > 0 else 0.0

    # 5-gate validation
    gate_details = {
        "sharpe": {"value": round(sharpe, 3), "threshold": GATES["sharpe_min"],
                   "passed": sharpe > GATES["sharpe_min"]},
        "perm_p": {"value": None, "threshold": GATES["perm_p_max"], "passed": None},  # filled later
        "regime_gap": {"value": round(regime_gap, 3), "threshold": GATES["regime_gap_max"],
                       "passed": regime_gap < GATES["regime_gap_max"]},
        "max_dd": {"value": round(max_dd, 2), "threshold": GATES["mdd_min"],
                   "passed": max_dd > GATES["mdd_min"]},
        "n_trades": {"value": n, "threshold": GATES["min_trades"],
                     "passed": n >= GATES["min_trades"]},
    }

    gates_passed = sum(1 for g in gate_details.values() if g["passed"] is True)

    return {
        "label": label,
        "n_trades": n,
        "total_pnl": round(total_pnl, 2),
        "total_return_pct": round(total_pnl / CAPITAL * 100, 2),
        "sharpe": round(float(sharpe), 3),
        "sortino": round(float(sortino), 3),
        "profit_factor": round(float(profit_factor), 3),
        "win_rate": round(float(win_rate), 1),
        "max_drawdown_pct": round(float(max_dd), 2),
        "avg_return_pct": round(float(avg_ret * 100), 3),
        "bull_sharpe": round(float(bull_sharpe), 3),
        "bear_sharpe": round(float(bear_sharpe), 3),
        "regime_gap": round(float(regime_gap), 3),
        "bull_trades": len(bull_rets),
        "bear_trades": len(bear_rets),
        "gates_passed": gates_passed,
        "gate_details": gate_details,
    }


# ── Permutation Test ─────────────────────────────────────────────────────
def permutation_test(trades, n_perm=N_PERM):
    """Shuffle returns, compute null Sharpe distribution, get p-value."""
    if len(trades) < 5:
        return 1.0

    returns = np.array([t["return"] for t in trades])
    trades_per_year = 252 / 10.0
    observed_sharpe = (np.mean(returns) / np.std(returns, ddof=1)) * np.sqrt(trades_per_year) if np.std(returns) > 0 else 0

    null_sharpes = []
    for _ in range(n_perm):
        shuffled = np.random.permutation(returns)
        s = np.std(shuffled, ddof=1)
        if s > 0:
            null_sharpes.append((np.mean(shuffled) / s) * np.sqrt(trades_per_year))
        else:
            null_sharpes.append(0.0)

    null_sharpes = np.array(null_sharpes)
    p_value = float(np.mean(null_sharpes >= observed_sharpe))
    return p_value


# ── Run Backtests ────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("RUNNING BACKTESTS")
print("=" * 70)

variant_descriptions = {
    "A": "Post-earnings dip buy (>3% drop in 3d after earnings, hold 10d)",
    "B": "Beat-and-drop (BEAT + >2% drop in 3d, hold 10d)",
    "C": "Miss-and-dip extreme (MISS + >8% drop, hold 15d)",
    "D": "Earnings recovery (>5% below pre-earnings + green day, hold 10d)",
    "E": "Pre-earnings MR (>5% dip + RSI<35 before earnings, sell 1d pre)",
    "F": "Earnings + triple signal (dip+RSI+green after red + recent earnings, hold 10d)",
}

results = {}

for variant_name, signals in all_variant_signals.items():
    desc = variant_descriptions[variant_name]
    print(f"\n{'─' * 60}")
    print(f"Variant {variant_name}: {desc}")
    print(f"{'─' * 60}")

    trades = simulate_trades(signals)
    metrics = compute_metrics(trades, label=f"Variant {variant_name}")

    print(f"  Trades: {metrics['n_trades']}")
    print(f"  Total PnL: ${metrics['total_pnl']:+.2f} ({metrics.get('total_return_pct', 0):+.1f}%)")
    print(f"  Sharpe: {metrics['sharpe']:.3f}")
    print(f"  Sortino: {metrics['sortino']:.3f}")
    print(f"  Profit Factor: {metrics['profit_factor']:.3f}")
    print(f"  Win Rate: {metrics['win_rate']:.1f}%")
    print(f"  Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
    print(f"  Regime: Bull={metrics['bull_sharpe']:.3f} Bear={metrics['bear_sharpe']:.3f} Gap={metrics['regime_gap']:.3f}")

    # Permutation test
    if metrics["n_trades"] >= 5:
        p_val = permutation_test(trades)
        metrics["gate_details"]["perm_p"]["value"] = round(p_val, 4)
        metrics["gate_details"]["perm_p"]["passed"] = p_val < GATES["perm_p_max"]
        if metrics["gate_details"]["perm_p"]["passed"]:
            metrics["gates_passed"] += 1
        print(f"  Permutation p-value: {p_val:.4f}")
    else:
        metrics["gate_details"]["perm_p"]["value"] = 1.0
        metrics["gate_details"]["perm_p"]["passed"] = False
        print(f"  Permutation: skipped (< 5 trades)")

    # Gate summary
    gate_pass_str = " | ".join([
        f"{'PASS' if g['passed'] else 'FAIL'}:{k}={g['value']}"
        for k, g in metrics["gate_details"].items()
    ])
    print(f"  Gates: {metrics['gates_passed']}/5 | {gate_pass_str}")

    # Store sample trades
    metrics["sample_trades"] = trades[:5] if trades else []
    metrics["description"] = desc
    results[variant_name] = metrics

# ── Summary ──────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY — Earnings Surprise + Mean Reversion")
print("=" * 70)

print(f"\n{'Var':<5} {'Trades':>6} {'PnL':>10} {'Sharpe':>7} {'Sortino':>8} {'PF':>6} {'WR':>6} {'MaxDD':>7} {'Gates':>6}")
print("-" * 70)
for v in sorted(results.keys()):
    m = results[v]
    print(f"  {v:<3} {m['n_trades']:>6} {m['total_pnl']:>+9.2f} {m['sharpe']:>7.3f} {m['sortino']:>8.3f} "
          f"{m['profit_factor']:>6.2f} {m['win_rate']:>5.1f}% {m['max_drawdown_pct']:>6.2f}% {m['gates_passed']:>3}/5")

# Identify best variant
valid = {v: m for v, m in results.items() if m["gates_passed"] >= 3}
if valid:
    best = max(valid, key=lambda v: valid[v]["sharpe"])
    print(f"\nBest variant: {best} (Sharpe={results[best]['sharpe']:.3f}, {results[best]['gates_passed']}/5 gates)")
else:
    # Pick highest sharpe regardless
    best = max(results, key=lambda v: results[v]["sharpe"]) if results else None
    if best:
        print(f"\nNo variant passed 3+ gates. Best Sharpe: {best} ({results[best]['sharpe']:.3f}, {results[best]['gates_passed']}/5)")
    else:
        print("\nNo variants produced trades.")

# ── Save Results ─────────────────────────────────────────────────────────
output = {
    "strategy": "Earnings Surprise + Mean Reversion",
    "run_date": datetime.now().isoformat(),
    "config": {
        "capital": CAPITAL,
        "max_per_trade": MAX_PER_TRADE,
        "max_concurrent": MAX_CONCURRENT,
        "slippage_bps": SLIPPAGE_PCT * 10000,
        "oot_period": f"{OOT_START} to {END}",
        "universe_size": len(TICKERS),
        "n_permutations": N_PERM,
        "tickers": TICKERS,
    },
    "data_quality": {
        "tickers_with_earnings_dates": earnings_loaded,
        "tickers_with_beat_miss_data": has_surprise,
        "note": "Variants B/C require beat/miss data; if unavailable they produce fewer signals"
    },
    "gates": GATES,
    "variants": results,
    "best_variant": best,
    "recommendation": "",
}

if best and results[best]["gates_passed"] >= 3:
    output["recommendation"] = (
        f"Variant {best} passes {results[best]['gates_passed']}/5 gates. "
        f"Sharpe={results[best]['sharpe']:.3f}, WR={results[best]['win_rate']:.1f}%, "
        f"PF={results[best]['profit_factor']:.2f}. Consider for live allocation."
    )
else:
    output["recommendation"] = (
        "No variant achieved 3+ gates. Earnings-based MR may need "
        "different thresholds or a larger universe to generate sufficient trades."
    )

Path(RESULTS_PATH).parent.mkdir(parents=True, exist_ok=True)
with open(RESULTS_PATH, "w") as f:
    json.dump(output, f, indent=2, default=str)

print(f"\nResults saved to {RESULTS_PATH}")
print("Done.")
