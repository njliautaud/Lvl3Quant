#!/usr/bin/env python3
"""
Adversarial Validation: Dual Signal QMR Variant D
==================================================
6 adversarial checks to stress-test whether Variant D's edge is real.

Variant D Strategy:
  Buy when BOTH conditions are met simultaneously:
    1. Stock drops >5% from 20-day high AND RSI(14) < 35 (QMR signal)
    2. Stock has had 3+ consecutive red days AND today is the first green day (recovery confirmation)
  Hold 10 days. Max $200/trade, max 3 concurrent positions.

Original results: Sharpe 1.767, perm p=0.005, regime gap 0.083, MDD -5.96%, 110 trades.

Tests:
  1) Inverse Signal — buy when OPPOSITE conditions hold
  2) Random Entry Timing — 1000 random entry sets, percentile rank
  3) Sub-Period Stability — 4 sub-periods, all must be positive
  4) Remove Top 3 Tickers — Sharpe drop must be < 50%
  5) Parameter Sensitivity — sweep thresholds, report % with Sharpe > 0.3
  6) Cost Sensitivity — test at 5/10/20/50bps, report breakeven

OOT: Jan 2022 - Jul 2026. $645 capital. 0.02% slippage baseline.
"""

import json
import warnings
import sys
import itertools
from datetime import datetime
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
CAPITAL = 645.0
MAX_PER_TRADE = 200.0
MAX_CONCURRENT = 3
SLIPPAGE_PCT = 0.0002  # 0.02% each way
DATA_START = "2020-01-01"  # extra lookback for indicators
OOT_START = "2022-01-01"
END = "2026-07-31"
N_PERM = 1000
HOLD_DAYS = 10

TICKERS = [
    "AAPL", "MSFT", "AVGO", "JPM", "JNJ", "PG", "KO", "PEP", "HD", "COST",
    "UNH", "LLY", "V", "MA", "ABBV", "MRK", "WMT", "AMZN", "GOOGL", "META",
]

ALL_TICKERS = sorted(set(TICKERS + ["SPY"]))

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


closes = {t: get_close(t) for t in ALL_TICKERS}
spy_close = closes.get("SPY", pd.Series(dtype=float))

loaded = sum(1 for t in TICKERS if len(closes.get(t, [])) > 100)
print(f"  Tickers with data: {loaded}/{len(TICKERS)}")


# ── Indicator Helpers ─────────────────────────────────────────────────────
def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def rolling_high(series, period=20):
    return series.rolling(period).max()


def rolling_low(series, period=20):
    return series.rolling(period).min()


def consecutive_red_days(series):
    """Count of consecutive red (close < prior close) days ending at each date."""
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


def consecutive_green_days(series):
    """Count of consecutive green (close > prior close) days ending at each date."""
    is_green = (series.diff() > 0).astype(int)
    result = pd.Series(0, index=series.index, dtype=int)
    count = 0
    for i in range(len(is_green)):
        if is_green.iloc[i] == 1:
            count += 1
        else:
            count = 0
        result.iloc[i] = count
    return result


def is_green_day(series):
    return series.diff() > 0


# ── Pre-compute indicators ───────────────────────────────────────────────
print("Computing indicators ...")
indicators = {}
for t in TICKERS:
    c = closes.get(t, pd.Series(dtype=float))
    if len(c) < 50:
        continue
    indicators[t] = {
        "rsi14": calc_rsi(c, 14),
        "high20": rolling_high(c, 20),
        "low20": rolling_low(c, 20),
        "consec_red": consecutive_red_days(c),
        "consec_green": consecutive_green_days(c),
        "green": is_green_day(c),
    }

spy_sma200 = spy_close.rolling(200).mean()

oot_start_ts = pd.Timestamp(OOT_START)


# ── Signal Generation ─────────────────────────────────────────────────────
def generate_variant_d_signals(tickers=None, drawdown_pct=0.05, rsi_thresh=35,
                                consec_red_min=3, slippage=SLIPPAGE_PCT):
    """
    Variant D: BOTH conditions must be met simultaneously:
      1. Stock drops >drawdown_pct from 20-day high AND RSI(14) < rsi_thresh
      2. Stock has had consec_red_min+ consecutive red days AND today is first green day
    """
    if tickers is None:
        tickers = TICKERS
    signals = []
    for t in tickers:
        if t not in indicators:
            continue
        c = closes[t]
        rsi = indicators[t]["rsi14"]
        h20 = indicators[t]["high20"]
        consec = indicators[t]["consec_red"]
        green = indicators[t]["green"]

        for i in range(1, len(c)):
            date = c.index[i]
            if date < oot_start_ts:
                continue
            try:
                price = c.iloc[i]
                high = h20.loc[date]
                r = rsi.loc[date]
                if pd.isna(price) or pd.isna(high) or pd.isna(r) or high == 0:
                    continue

                # Condition 1: QMR signal
                drop_pct = (price - high) / high
                if drop_pct >= -drawdown_pct or r >= rsi_thresh:
                    continue

                # Condition 2: Recovery confirmation
                if not green.iloc[i]:
                    continue
                if i < 1:
                    continue
                if consec.iloc[i - 1] < consec_red_min:
                    continue

                signals.append((date, t))
            except (KeyError, IndexError):
                continue
    return signals


def generate_inverse_signals():
    """
    Inverse: Buy when OPPOSITE conditions hold:
      - Stock is near 20-day high (within 2%) AND RSI(14) > 65 (strong momentum)
      - No 3+ consecutive red days recently (last 5 bars all had consec_red < 3)
      - Today is NOT a green day after red streak (just any day with momentum)
    """
    signals = []
    for t in TICKERS:
        if t not in indicators:
            continue
        c = closes[t]
        rsi = indicators[t]["rsi14"]
        h20 = indicators[t]["high20"]
        consec = indicators[t]["consec_red"]

        for i in range(1, len(c)):
            date = c.index[i]
            if date < oot_start_ts:
                continue
            try:
                price = c.iloc[i]
                high = h20.loc[date]
                r = rsi.loc[date]
                if pd.isna(price) or pd.isna(high) or pd.isna(r) or high == 0:
                    continue

                # Opposite of drawdown: near highs (within 2%)
                gap_pct = (price - high) / high
                if gap_pct < -0.02:
                    continue

                # Opposite of RSI < 35: RSI > 65
                if r <= 65:
                    continue

                # Opposite of 3+ consecutive red days: no recent red streak
                if consec.iloc[max(0, i-1)] >= 3:
                    continue

                signals.append((date, t))
            except (KeyError, IndexError):
                continue
    return signals


# ── Trade Simulator ──────────────────────────────────────────────────────
def simulate_trades(signals, capital=CAPITAL, max_per_trade=MAX_PER_TRADE,
                    max_concurrent=MAX_CONCURRENT, hold_days=HOLD_DAYS,
                    slippage=SLIPPAGE_PCT):
    """Simulate trades with position limits and concurrent position tracking."""
    if not signals:
        return []

    signals = sorted(signals, key=lambda x: x[0])

    # Deduplicate: no same-ticker entry within hold period
    deduped = []
    last_entry = {}
    for date, ticker in signals:
        if ticker in last_entry:
            delta = (date - last_entry[ticker]).days
            if delta < hold_days:
                continue
        deduped.append((date, ticker))
        last_entry[ticker] = date

    trades = []
    open_positions = []

    for date, ticker in deduped:
        # Close expired positions
        open_positions = [(ed, tk) for ed, tk in open_positions if ed > date]

        # Check concurrent limit
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
        entry_price = float(c.iloc[loc]) * (1 + slippage)
        exit_price = float(c.iloc[exit_loc]) * (1 - slippage)
        exit_date = c.index[exit_loc]

        if entry_price <= 0:
            continue

        shares = max_per_trade / entry_price
        pnl = shares * (exit_price - entry_price)
        ret = (exit_price - entry_price) / entry_price

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
        })

        open_positions.append((exit_date, ticker))

    return trades


# ── Metrics Calculation ──────────────────────────────────────────────────
def calc_metrics(trades, capital=CAPITAL):
    if not trades:
        return {
            "total_return_pct": 0, "sharpe": 0, "sortino": 0,
            "max_drawdown_pct": 0, "win_rate": 0, "profit_factor": 0,
            "n_trades": 0, "total_pnl": 0, "avg_return_pct": 0,
        }

    returns = np.array([t["return"] for t in trades])
    pnls = np.array([t["pnl"] for t in trades])
    total_pnl = pnls.sum()

    wins = (pnls > 0).sum()
    win_rate = wins / len(pnls)

    gross_profit = pnls[pnls > 0].sum() if (pnls > 0).any() else 0
    gross_loss = abs(pnls[pnls < 0].sum()) if (pnls < 0).any() else 1e-9
    profit_factor = gross_profit / gross_loss

    oot_years = 4.5  # Jan 2022 - Jul 2026
    if len(returns) > 1 and returns.std() > 0:
        trades_per_year = max(len(returns) / oot_years, 1)
        sharpe = (returns.mean() / returns.std()) * np.sqrt(trades_per_year)
    else:
        sharpe = 0.0

    downside = returns[returns < 0]
    if len(downside) > 1 and downside.std() > 0:
        trades_per_year = max(len(returns) / oot_years, 1)
        sortino = (returns.mean() / downside.std()) * np.sqrt(trades_per_year)
    else:
        sortino = sharpe * 1.5 if sharpe > 0 else 0.0

    cum_pnl = np.cumsum(pnls)
    peak = np.maximum.accumulate(cum_pnl + capital)
    drawdowns = (cum_pnl + capital - peak) / peak
    max_dd = drawdowns.min() * 100 if len(drawdowns) > 0 else 0

    return {
        "total_return_pct": round(total_pnl / capital * 100, 2),
        "sharpe": round(float(sharpe), 3),
        "sortino": round(float(sortino), 3),
        "max_drawdown_pct": round(float(max_dd), 2),
        "win_rate": round(float(win_rate), 4),
        "profit_factor": round(float(profit_factor), 3),
        "n_trades": len(trades),
        "total_pnl": round(float(total_pnl), 2),
        "avg_return_pct": round(float(returns.mean() * 100), 3),
    }


def regime_stratified_sharpe(trades):
    if not trades:
        return {"bull_sharpe": 0, "bear_sharpe": 0, "regime_gap": 0,
                "bull_trades": 0, "bear_trades": 0}

    bull_rets, bear_rets = [], []
    for t in trades:
        entry = pd.Timestamp(t["entry_date"])
        try:
            spy_val = spy_close.asof(entry)
            sma_val = spy_sma200.asof(entry)
            if pd.isna(spy_val) or pd.isna(sma_val):
                continue
            if spy_val > sma_val:
                bull_rets.append(t["return"])
            else:
                bear_rets.append(t["return"])
        except Exception:
            continue

    def _sharpe(rets):
        if len(rets) < 2:
            return 0.0
        arr = np.array(rets)
        if arr.std() == 0:
            return 0.0
        tpy = max(len(arr) / 4.5, 1)
        return float((arr.mean() / arr.std()) * np.sqrt(tpy))

    bull_s = _sharpe(bull_rets)
    bear_s = _sharpe(bear_rets)
    max_abs = max(abs(bull_s), abs(bear_s), 1e-9)
    gap = abs(bull_s - bear_s) / max_abs

    return {
        "bull_sharpe": round(bull_s, 3),
        "bear_sharpe": round(bear_s, 3),
        "regime_gap": round(gap, 3),
        "bull_trades": len(bull_rets),
        "bear_trades": len(bear_rets),
    }


# ── Generate baseline signals and trades ──────────────────────────────────
print("\nGenerating Variant D baseline signals ...")
baseline_signals = generate_variant_d_signals()
baseline_trades = simulate_trades(baseline_signals)
baseline_metrics = calc_metrics(baseline_trades)
baseline_regime = regime_stratified_sharpe(baseline_trades)

print(f"  Baseline: {baseline_metrics['n_trades']} trades, "
      f"Sharpe {baseline_metrics['sharpe']:.3f}, "
      f"WR {baseline_metrics['win_rate']:.1%}, "
      f"MDD {baseline_metrics['max_drawdown_pct']:.2f}%")

results = {
    "strategy": "Dual Signal QMR Variant D",
    "run_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
    "config": {
        "capital": CAPITAL,
        "max_per_trade": MAX_PER_TRADE,
        "max_concurrent": MAX_CONCURRENT,
        "slippage_pct": SLIPPAGE_PCT,
        "oot_period": f"{OOT_START} to {END}",
        "hold_days": HOLD_DAYS,
        "universe_size": len(TICKERS),
    },
    "baseline": {
        "metrics": baseline_metrics,
        "regime": baseline_regime,
    },
    "tests": {},
}


# ══════════════════════════════════════════════════════════════════════════
# TEST 1: INVERSE SIGNAL
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("TEST 1: INVERSE SIGNAL")
print("="*70)
print("  Buy when: near 20d high (within 2%), RSI>65, no red streaks")

inverse_signals = generate_inverse_signals()
inverse_trades = simulate_trades(inverse_signals)
inverse_metrics = calc_metrics(inverse_trades)

print(f"  Inverse signals generated: {len(inverse_signals)}")
print(f"  Inverse trades: {inverse_metrics['n_trades']}")
print(f"  Inverse Sharpe: {inverse_metrics['sharpe']:.3f}")
print(f"  Baseline Sharpe: {baseline_metrics['sharpe']:.3f}")

# If inverse works well, it means edge is just from the universe (buy-and-hold)
inverse_pass = inverse_metrics["sharpe"] < baseline_metrics["sharpe"] * 0.5
test1_verdict = "PASS" if inverse_pass else "FAIL"
print(f"  Inverse Sharpe < 50% of baseline? {inverse_pass}")
print(f"  >>> TEST 1: {test1_verdict}")

results["tests"]["1_inverse_signal"] = {
    "description": "Buy when opposite conditions hold (near highs, RSI>65, no red streaks)",
    "inverse_sharpe": inverse_metrics["sharpe"],
    "inverse_trades": inverse_metrics["n_trades"],
    "inverse_return_pct": inverse_metrics["total_return_pct"],
    "baseline_sharpe": baseline_metrics["sharpe"],
    "ratio": round(inverse_metrics["sharpe"] / max(baseline_metrics["sharpe"], 0.001), 3),
    "pass": inverse_pass,
    "verdict": test1_verdict,
}


# ══════════════════════════════════════════════════════════════════════════
# TEST 2: RANDOM ENTRY TIMING
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("TEST 2: RANDOM ENTRY TIMING (1000 iterations)")
print("="*70)

real_mean_return = np.mean([t["return"] for t in baseline_trades]) if baseline_trades else 0

# Build pool of valid OOT dates per ticker
ticker_oot_dates = {}
for t in TICKERS:
    c = closes.get(t, pd.Series(dtype=float))
    valid = c.index[c.index >= oot_start_ts]
    if len(valid) > HOLD_DAYS + 1:
        ticker_oot_dates[t] = valid

# Count trades per ticker in baseline for proportional random sampling
ticker_trade_counts = defaultdict(int)
for tr in baseline_trades:
    ticker_trade_counts[tr["ticker"]] += 1

random_sharpes = []
count_better = 0

for iteration in range(N_PERM):
    random_signals = []
    for ticker, count in ticker_trade_counts.items():
        if ticker not in ticker_oot_dates:
            continue
        dates = ticker_oot_dates[ticker]
        max_idx = max(0, len(dates) - HOLD_DAYS - 1)
        if max_idx < 1:
            continue
        rand_indices = np.random.randint(0, max_idx, size=count)
        for idx in rand_indices:
            random_signals.append((dates[idx], ticker))

    random_trades = simulate_trades(random_signals)
    random_metrics = calc_metrics(random_trades)
    random_sharpes.append(random_metrics["sharpe"])

    if random_metrics["sharpe"] >= baseline_metrics["sharpe"]:
        count_better += 1

percentile = 100 * (1 - count_better / N_PERM)
random_sharpes = np.array(random_sharpes)
median_random = np.median(random_sharpes)
p95_random = np.percentile(random_sharpes, 95)

print(f"  Real Sharpe: {baseline_metrics['sharpe']:.3f}")
print(f"  Random median Sharpe: {median_random:.3f}")
print(f"  Random 95th percentile Sharpe: {p95_random:.3f}")
print(f"  Percentile rank: {percentile:.1f}%")
print(f"  p-value: {count_better/N_PERM:.4f}")

random_pass = percentile >= 95.0  # real strategy must be in top 5%
test2_verdict = "PASS" if random_pass else "FAIL"
print(f"  >>> TEST 2: {test2_verdict}")

results["tests"]["2_random_entry_timing"] = {
    "description": "1000 random entry timing iterations, report percentile",
    "real_sharpe": baseline_metrics["sharpe"],
    "random_median_sharpe": round(float(median_random), 3),
    "random_p95_sharpe": round(float(p95_random), 3),
    "percentile": round(percentile, 1),
    "p_value": round(count_better / N_PERM, 4),
    "pass": random_pass,
    "verdict": test2_verdict,
}


# ══════════════════════════════════════════════════════════════════════════
# TEST 3: SUB-PERIOD STABILITY
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("TEST 3: SUB-PERIOD STABILITY")
print("="*70)

sub_periods = [
    ("Jan22-Jul23", "2022-01-01", "2023-07-01"),
    ("Jul23-Jan24", "2023-07-01", "2024-01-01"),  # shorter period
    ("Jan24-Jul25", "2024-01-01", "2025-07-01"),
    ("Jul25-Jul26", "2025-07-01", "2026-07-31"),
]

sub_results = {}
all_positive = True

for label, sp_start, sp_end in sub_periods:
    sp_start_ts = pd.Timestamp(sp_start)
    sp_end_ts = pd.Timestamp(sp_end)

    sp_trades = [t for t in baseline_trades
                 if sp_start_ts <= pd.Timestamp(t["entry_date"]) < sp_end_ts]
    sp_metrics = calc_metrics(sp_trades)

    is_positive = sp_metrics["total_pnl"] > 0
    if not is_positive:
        all_positive = False

    print(f"  {label}: {sp_metrics['n_trades']} trades, "
          f"Sharpe {sp_metrics['sharpe']:.3f}, "
          f"Return {sp_metrics['total_return_pct']:.1f}%, "
          f"PnL ${sp_metrics['total_pnl']:.2f} "
          f"{'OK' if is_positive else 'NEGATIVE'}")

    sub_results[label] = {
        "n_trades": sp_metrics["n_trades"],
        "sharpe": sp_metrics["sharpe"],
        "total_return_pct": sp_metrics["total_return_pct"],
        "total_pnl": sp_metrics["total_pnl"],
        "win_rate": sp_metrics["win_rate"],
        "positive": is_positive,
    }

test3_verdict = "PASS" if all_positive else "FAIL"
print(f"  All sub-periods positive? {all_positive}")
print(f"  >>> TEST 3: {test3_verdict}")

results["tests"]["3_sub_period_stability"] = {
    "description": "4 sub-periods must all be positive",
    "sub_periods": sub_results,
    "all_positive": all_positive,
    "pass": all_positive,
    "verdict": test3_verdict,
}


# ══════════════════════════════════════════════════════════════════════════
# TEST 4: REMOVE TOP 3 TICKERS
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("TEST 4: REMOVE TOP 3 TICKERS")
print("="*70)

# Find top 3 tickers by PnL contribution
ticker_pnl = defaultdict(float)
for t in baseline_trades:
    ticker_pnl[t["ticker"]] += t["pnl"]

sorted_tickers = sorted(ticker_pnl.items(), key=lambda x: x[1], reverse=True)
top3 = [tk for tk, _ in sorted_tickers[:3]]
top3_pnl = sum(pnl for _, pnl in sorted_tickers[:3])

print(f"  Top 3 tickers by PnL: {', '.join(f'{tk} (${pnl:.2f})' for tk, pnl in sorted_tickers[:3])}")

# Re-run without top 3
reduced_tickers = [t for t in TICKERS if t not in top3]
reduced_signals = generate_variant_d_signals(tickers=reduced_tickers)
reduced_trades = simulate_trades(reduced_signals)
reduced_metrics = calc_metrics(reduced_trades)

sharpe_drop_pct = 0
if baseline_metrics["sharpe"] > 0:
    sharpe_drop_pct = (baseline_metrics["sharpe"] - reduced_metrics["sharpe"]) / baseline_metrics["sharpe"] * 100

print(f"  Baseline Sharpe: {baseline_metrics['sharpe']:.3f} ({baseline_metrics['n_trades']} trades)")
print(f"  Without top 3 Sharpe: {reduced_metrics['sharpe']:.3f} ({reduced_metrics['n_trades']} trades)")
print(f"  Sharpe drop: {sharpe_drop_pct:.1f}%")

test4_pass = sharpe_drop_pct < 50.0
test4_verdict = "PASS" if test4_pass else "FAIL"
print(f"  Drop < 50%? {test4_pass}")
print(f"  >>> TEST 4: {test4_verdict}")

results["tests"]["4_remove_top3_tickers"] = {
    "description": "Remove best 3 tickers by PnL. Sharpe drop must be < 50%",
    "top3_removed": top3,
    "top3_pnl": round(top3_pnl, 2),
    "baseline_sharpe": baseline_metrics["sharpe"],
    "reduced_sharpe": reduced_metrics["sharpe"],
    "reduced_trades": reduced_metrics["n_trades"],
    "sharpe_drop_pct": round(sharpe_drop_pct, 1),
    "pass": test4_pass,
    "verdict": test4_verdict,
}


# ══════════════════════════════════════════════════════════════════════════
# TEST 5: PARAMETER SENSITIVITY
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("TEST 5: PARAMETER SENSITIVITY")
print("="*70)

drawdown_thresholds = [0.03, 0.05, 0.07, 0.10]
rsi_thresholds = [25, 30, 35, 40]
consec_red_values = [2, 3, 4]
hold_periods = [5, 10, 15, 20]

total_combos = len(drawdown_thresholds) * len(rsi_thresholds) * len(consec_red_values) * len(hold_periods)
print(f"  Testing {total_combos} parameter combinations ...")

param_results = []
sharpe_above_03 = 0

for dd, rsi_t, cr, hp in itertools.product(drawdown_thresholds, rsi_thresholds,
                                             consec_red_values, hold_periods):
    sigs = generate_variant_d_signals(drawdown_pct=dd, rsi_thresh=rsi_t, consec_red_min=cr)
    trades = simulate_trades(sigs, hold_days=hp)
    metrics = calc_metrics(trades)

    is_baseline = (dd == 0.05 and rsi_t == 35 and cr == 3 and hp == 10)

    param_results.append({
        "drawdown_pct": dd,
        "rsi_thresh": rsi_t,
        "consec_red": cr,
        "hold_days": hp,
        "sharpe": metrics["sharpe"],
        "n_trades": metrics["n_trades"],
        "total_return_pct": metrics["total_return_pct"],
        "is_baseline": is_baseline,
    })

    if metrics["sharpe"] > 0.3:
        sharpe_above_03 += 1

pct_above_03 = sharpe_above_03 / total_combos * 100

# Find best and worst
sorted_params = sorted(param_results, key=lambda x: x["sharpe"], reverse=True)
best = sorted_params[0]
worst = sorted_params[-1]

print(f"  Total combinations: {total_combos}")
print(f"  Combinations with Sharpe > 0.3: {sharpe_above_03} ({pct_above_03:.1f}%)")
print(f"  Best: dd={best['drawdown_pct']}, rsi={best['rsi_thresh']}, "
      f"red={best['consec_red']}, hold={best['hold_days']} -> Sharpe {best['sharpe']:.3f}")
print(f"  Worst: dd={worst['drawdown_pct']}, rsi={worst['rsi_thresh']}, "
      f"red={worst['consec_red']}, hold={worst['hold_days']} -> Sharpe {worst['sharpe']:.3f}")

# Robustness: at least 30% of parameter space should have Sharpe > 0.3
test5_pass = pct_above_03 >= 30.0
test5_verdict = "PASS" if test5_pass else "FAIL"
print(f"  >= 30% of space with Sharpe > 0.3? {test5_pass}")
print(f"  >>> TEST 5: {test5_verdict}")

results["tests"]["5_parameter_sensitivity"] = {
    "description": "Sweep drawdown (3/5/7/10%), RSI (25/30/35/40), consec red (2/3/4), hold (5/10/15/20). Report % with Sharpe > 0.3",
    "total_combinations": total_combos,
    "sharpe_above_0_3": sharpe_above_03,
    "pct_above_0_3": round(pct_above_03, 1),
    "best_params": best,
    "worst_params": worst,
    "baseline_params": {"drawdown_pct": 0.05, "rsi_thresh": 35, "consec_red": 3, "hold_days": 10},
    "pass": test5_pass,
    "verdict": test5_verdict,
}


# ══════════════════════════════════════════════════════════════════════════
# TEST 6: COST SENSITIVITY
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("TEST 6: COST SENSITIVITY")
print("="*70)

cost_levels_bps = [5, 10, 20, 50]
cost_results = {}
breakeven_bps = None

for bps in cost_levels_bps:
    slip = bps / 10000.0  # convert bps to decimal
    trades = simulate_trades(baseline_signals, slippage=slip)
    metrics = calc_metrics(trades)

    cost_results[f"{bps}bps"] = {
        "slippage_bps": bps,
        "sharpe": metrics["sharpe"],
        "total_return_pct": metrics["total_return_pct"],
        "total_pnl": metrics["total_pnl"],
        "n_trades": metrics["n_trades"],
        "profitable": metrics["total_pnl"] > 0,
    }

    print(f"  {bps:>3}bps: Sharpe {metrics['sharpe']:.3f}, "
          f"Return {metrics['total_return_pct']:.1f}%, "
          f"PnL ${metrics['total_pnl']:.2f}, "
          f"{'Profitable' if metrics['total_pnl'] > 0 else 'UNPROFITABLE'}")

# Find breakeven via binary search
lo, hi = 0, 200  # bps range
for _ in range(20):
    mid = (lo + hi) / 2
    slip = mid / 10000.0
    trades = simulate_trades(baseline_signals, slippage=slip)
    metrics = calc_metrics(trades)
    if metrics["total_pnl"] > 0:
        lo = mid
    else:
        hi = mid
breakeven_bps = round((lo + hi) / 2, 1)

print(f"  Breakeven slippage: ~{breakeven_bps} bps")

test6_pass = breakeven_bps > 10.0  # must survive at least 10bps
test6_verdict = "PASS" if test6_pass else "FAIL"
print(f"  Breakeven > 10bps? {test6_pass}")
print(f"  >>> TEST 6: {test6_verdict}")

results["tests"]["6_cost_sensitivity"] = {
    "description": "Test at 5/10/20/50bps slippage. Report breakeven",
    "cost_levels": cost_results,
    "breakeven_bps": breakeven_bps,
    "pass": test6_pass,
    "verdict": test6_verdict,
}


# ══════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("ADVERSARIAL VALIDATION SUMMARY — Dual Signal QMR Variant D")
print("="*70)

n_pass = sum(1 for t in results["tests"].values() if t["pass"])
n_tests = len(results["tests"])

for test_name, test_data in results["tests"].items():
    print(f"  {test_name}: {test_data['verdict']}")

print(f"\n  OVERALL: {n_pass}/{n_tests} tests passed")

overall_pass = n_pass >= 5  # 5 of 6 must pass
results["summary"] = {
    "tests_passed": n_pass,
    "tests_total": n_tests,
    "overall_verdict": "PASS" if overall_pass else "FAIL",
    "baseline_sharpe": baseline_metrics["sharpe"],
    "baseline_trades": baseline_metrics["n_trades"],
}

print(f"  OVERALL VERDICT: {'PASS' if overall_pass else 'FAIL'}")

# ── Save Results ─────────────────────────────────────────────────────────
output_path = Path("/home/jupiter/Lvl3Quant/data/dual_signal_d_adversarial.json")
output_path.write_text(json.dumps(results, indent=2, default=str))
print(f"\nResults saved to {output_path}")
print("DONE.")
