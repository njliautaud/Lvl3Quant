#!/usr/bin/env python3
"""
RSI Divergence + Volume Decline on Quality Stocks (Variant C) — 6-Test Adversarial Validation

Entry logic:
  1. Local price low: day where Close is lower than surrounding 5 days each side
  2. Current local low is LOWER than previous local low (new low)
  3. RSI(14) at current low is HIGHER than RSI at previous low (bullish divergence)
  4. Volume declining: current volume < 20-day average volume (selling exhaustion)
  5. Buy at close on divergence day, hold 10 days

Universe: 20 large-cap quality stocks
Capital: $645, max $200/trade, max 3 concurrent, slippage 2bps
Period: 2022-01-01 to 2026-07-31
"""

import json
import warnings
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ── Constants ──────────────────────────────────────────────────────────────
TICKERS = [
    "AAPL", "MSFT", "AVGO", "JPM", "JNJ", "PG", "KO", "PEP", "HD", "COST",
    "UNH", "LLY", "V", "MA", "ABBV", "MRK", "WMT", "AMZN", "GOOGL", "META",
]
START = "2022-01-01"
END = "2026-07-31"
CAPITAL = 645.0
MAX_PER_TRADE = 200.0
MAX_CONCURRENT = 3
DEFAULT_SLIPPAGE_BPS = 2
RSI_PERIOD = 14
LOCAL_LOW_WINDOW = 5  # days each side
HOLD_DAYS = 10
VOL_LOOKBACK = 20
VOL_THRESHOLD = 1.0  # volume < VOL_THRESHOLD * 20d avg
RANDOM_ITERS = 1000

# ── Data Download ──────────────────────────────────────────────────────────
def download_data():
    """Download OHLCV for all tickers + SPY."""
    all_tickers = TICKERS + ["SPY"]
    data = {}
    for t in all_tickers:
        try:
            df = yf.download(t, start=START, end=END, progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if len(df) > 100:
                data[t] = df
        except Exception as e:
            print(f"  Warning: failed to download {t}: {e}")
    return data


# ── RSI Calculation ────────────────────────────────────────────────────────
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


# ── Detect Local Lows ─────────────────────────────────────────────────────
def find_local_lows(close, window=5):
    """Return indices where close is lower than surrounding `window` days each side."""
    lows = []
    for i in range(window, len(close) - window):
        val = close.iloc[i]
        left = close.iloc[i - window:i].min()
        right = close.iloc[i + 1:i + window + 1].min()
        if val <= left and val <= right:
            lows.append(i)
    return lows


# ── Signal Generation ─────────────────────────────────────────────────────
def generate_signals(df, local_low_window=LOCAL_LOW_WINDOW, vol_lookback=VOL_LOOKBACK,
                     vol_threshold=VOL_THRESHOLD, rsi_period=RSI_PERIOD):
    """Generate bullish RSI divergence + volume decline signals."""
    close = df["Close"].copy()
    volume = df["Volume"].copy()
    rsi = compute_rsi(close, rsi_period)
    vol_ma = volume.rolling(vol_lookback).mean()

    local_low_idxs = find_local_lows(close, local_low_window)
    signals = []

    for j in range(1, len(local_low_idxs)):
        prev_i = local_low_idxs[j - 1]
        curr_i = local_low_idxs[j]

        # Price makes lower low
        if close.iloc[curr_i] >= close.iloc[prev_i]:
            continue
        # RSI makes higher low (bullish divergence)
        if rsi.iloc[curr_i] <= rsi.iloc[prev_i]:
            continue
        # Volume declining
        if pd.isna(vol_ma.iloc[curr_i]) or volume.iloc[curr_i] >= vol_threshold * vol_ma.iloc[curr_i]:
            continue

        signals.append(curr_i)

    return signals


def generate_inverse_signals(df):
    """Inverse: price higher high + RSI lower high + volume increasing."""
    close = df["Close"].copy()
    volume = df["Volume"].copy()
    rsi = compute_rsi(close, RSI_PERIOD)
    vol_ma = volume.rolling(VOL_LOOKBACK).mean()

    # Find local highs
    highs = []
    for i in range(LOCAL_LOW_WINDOW, len(close) - LOCAL_LOW_WINDOW):
        val = close.iloc[i]
        left = close.iloc[i - LOCAL_LOW_WINDOW:i].max()
        right = close.iloc[i + 1:i + LOCAL_LOW_WINDOW + 1].max()
        if val >= left and val >= right:
            highs.append(i)

    signals = []
    for j in range(1, len(highs)):
        prev_i = highs[j - 1]
        curr_i = highs[j]
        # Higher high
        if close.iloc[curr_i] <= close.iloc[prev_i]:
            continue
        # RSI lower high (bearish divergence)
        if rsi.iloc[curr_i] >= rsi.iloc[prev_i]:
            continue
        # Volume increasing
        if pd.isna(vol_ma.iloc[curr_i]) or volume.iloc[curr_i] <= vol_ma.iloc[curr_i]:
            continue
        signals.append(curr_i)

    return signals


# ── Backtest Engine ────────────────────────────────────────────────────────
def backtest(data, signal_func, hold_days=HOLD_DAYS, slippage_bps=DEFAULT_SLIPPAGE_BPS,
             tickers=None, capital=CAPITAL, max_per_trade=MAX_PER_TRADE,
             max_concurrent=MAX_CONCURRENT, signal_kwargs=None):
    """Run backtest across tickers. Returns trades list and equity curve."""
    if tickers is None:
        tickers = TICKERS
    if signal_kwargs is None:
        signal_kwargs = {}

    all_trades = []
    for ticker in tickers:
        if ticker not in data:
            continue
        df = data[ticker]
        sig_idxs = signal_func(df, **signal_kwargs) if signal_kwargs else signal_func(df)
        close = df["Close"]
        dates = df.index

        for idx in sig_idxs:
            if idx + hold_days >= len(close):
                continue
            entry_date = dates[idx]
            exit_date = dates[min(idx + hold_days, len(close) - 1)]
            entry_price = close.iloc[idx]
            exit_price = close.iloc[min(idx + hold_days, len(close) - 1)]

            slip = slippage_bps / 10000.0
            entry_cost = entry_price * (1 + slip)
            exit_val = exit_price * (1 - slip)
            ret = (exit_val / entry_cost) - 1

            all_trades.append({
                "ticker": ticker,
                "entry_date": entry_date,
                "exit_date": exit_date,
                "entry_price": float(entry_price),
                "exit_price": float(exit_price),
                "return": float(ret),
            })

    # Sort by entry date
    all_trades.sort(key=lambda x: x["entry_date"])

    # Apply concurrent position limit
    filtered = []
    active_exits = []
    for t in all_trades:
        active_exits = [e for e in active_exits if e > t["entry_date"]]
        if len(active_exits) < max_concurrent:
            shares = min(max_per_trade, capital / max(1, max_concurrent)) / t["entry_price"]
            t["shares"] = float(shares)
            t["pnl"] = float(t["return"] * shares * t["entry_price"])
            filtered.append(t)
            active_exits.append(t["exit_date"])

    return filtered


def compute_metrics(trades, capital=CAPITAL):
    """Compute Sharpe, WR, PF, MDD, total return from trade list."""
    if not trades:
        return {"sharpe": 0, "win_rate": 0, "profit_factor": 0, "max_dd": 0,
                "total_return": 0, "n_trades": 0, "total_pnl": 0}

    pnls = [t["pnl"] for t in trades]
    returns = [t["return"] for t in trades]

    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    wr = len(wins) / len(pnls) if pnls else 0
    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 1e-9
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Daily returns for Sharpe
    if len(returns) > 1:
        avg_ret = np.mean(returns)
        std_ret = np.std(returns, ddof=1)
        # Annualize: assume ~24 trades/year based on ~96 trades over 4.5 years
        trades_per_year = len(trades) / 4.5
        sharpe = (avg_ret / std_ret) * np.sqrt(trades_per_year) if std_ret > 0 else 0
    else:
        sharpe = 0

    # MDD on equity curve
    equity = [capital]
    for p in pnls:
        equity.append(equity[-1] + p)
    equity = np.array(equity)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    mdd = float(dd.min())

    return {
        "sharpe": round(float(sharpe), 3),
        "win_rate": round(float(wr), 3),
        "profit_factor": round(float(pf), 3),
        "max_dd": round(float(mdd), 4),
        "total_return": round(float(sum(pnls) / capital), 4),
        "n_trades": len(trades),
        "total_pnl": round(float(sum(pnls)), 2),
    }


def compute_regime_sharpes(trades, spy_df):
    """Compute Sharpe in bull vs bear regimes (SPY > 200-SMA = bull)."""
    spy_close = spy_df["Close"]
    spy_sma200 = spy_close.rolling(200).mean()

    bull_trades = []
    bear_trades = []
    for t in trades:
        d = t["entry_date"]
        if d in spy_sma200.index:
            idx = spy_sma200.index.get_indexer([d], method="ffill")[0]
            if idx >= 0 and not pd.isna(spy_sma200.iloc[idx]):
                if spy_close.iloc[idx] > spy_sma200.iloc[idx]:
                    bull_trades.append(t)
                else:
                    bear_trades.append(t)

    bull_m = compute_metrics(bull_trades)
    bear_m = compute_metrics(bear_trades)
    return bull_m["sharpe"], bear_m["sharpe"]


# ── Test 1: Inverse Signal ────────────────────────────────────────────────
def test_inverse(data):
    print("\n=== Test 1: Inverse Signal ===")
    real_trades = backtest(data, generate_signals)
    real_m = compute_metrics(real_trades)

    inv_trades = backtest(data, generate_inverse_signals)
    inv_m = compute_metrics(inv_trades)

    ratio = inv_m["sharpe"] / real_m["sharpe"] if real_m["sharpe"] != 0 else float("inf")
    passed = ratio < 0.50

    print(f"  Real Sharpe: {real_m['sharpe']}, Inverse Sharpe: {inv_m['sharpe']}")
    print(f"  Ratio: {ratio:.3f}, PASS threshold: < 0.50")
    print(f"  Result: {'PASS' if passed else 'FAIL'}")

    return {
        "test": "inverse_signal",
        "real_sharpe": real_m["sharpe"],
        "inverse_sharpe": inv_m["sharpe"],
        "ratio": round(ratio, 4),
        "threshold": 0.50,
        "passed": passed,
        "real_metrics": real_m,
        "inverse_metrics": inv_m,
    }


# ── Test 2: Random Timing ─────────────────────────────────────────────────
def test_random_timing(data):
    print("\n=== Test 2: Random Timing (1000 iterations) ===")
    real_trades = backtest(data, generate_signals)
    real_m = compute_metrics(real_trades)

    # Count trades per ticker
    ticker_counts = {}
    for t in real_trades:
        ticker_counts[t["ticker"]] = ticker_counts.get(t["ticker"], 0) + 1

    rng = np.random.RandomState(42)
    random_sharpes = []

    for i in range(RANDOM_ITERS):
        rand_trades = []
        for ticker, count in ticker_counts.items():
            if ticker not in data:
                continue
            df = data[ticker]
            close = df["Close"]
            n = len(close)
            if n < HOLD_DAYS + 10:
                continue

            idxs = rng.choice(n - HOLD_DAYS, size=min(count, n - HOLD_DAYS), replace=False)
            for idx in idxs:
                entry_price = close.iloc[idx]
                exit_price = close.iloc[idx + HOLD_DAYS]
                slip = DEFAULT_SLIPPAGE_BPS / 10000.0
                ret = (exit_price * (1 - slip)) / (entry_price * (1 + slip)) - 1
                shares = min(MAX_PER_TRADE, CAPITAL / MAX_CONCURRENT) / entry_price
                rand_trades.append({
                    "ticker": ticker,
                    "entry_date": df.index[idx],
                    "exit_date": df.index[idx + HOLD_DAYS],
                    "entry_price": float(entry_price),
                    "exit_price": float(exit_price),
                    "return": float(ret),
                    "shares": float(shares),
                    "pnl": float(ret * shares * entry_price),
                })

        rm = compute_metrics(rand_trades)
        random_sharpes.append(rm["sharpe"])

    random_sharpes = np.array(random_sharpes)
    p_value = float(np.mean(random_sharpes >= real_m["sharpe"]))
    passed = p_value < 0.05

    print(f"  Real Sharpe: {real_m['sharpe']}")
    print(f"  Random Sharpe: mean={np.mean(random_sharpes):.3f}, std={np.std(random_sharpes):.3f}")
    print(f"  p-value: {p_value:.4f}, PASS threshold: < 0.05")
    print(f"  Result: {'PASS' if passed else 'FAIL'}")

    return {
        "test": "random_timing",
        "real_sharpe": real_m["sharpe"],
        "random_mean": round(float(np.mean(random_sharpes)), 3),
        "random_std": round(float(np.std(random_sharpes)), 3),
        "random_p5": round(float(np.percentile(random_sharpes, 5)), 3),
        "random_p95": round(float(np.percentile(random_sharpes, 95)), 3),
        "p_value": round(p_value, 4),
        "threshold": 0.05,
        "passed": passed,
    }


# ── Test 3: Sub-Period Stability ───────────────────────────────────────────
def test_sub_period(data):
    print("\n=== Test 3: Sub-Period Stability ===")
    real_trades = backtest(data, generate_signals)

    if not real_trades:
        return {"test": "sub_period", "passed": False, "reason": "no trades"}

    all_dates = sorted([t["entry_date"] for t in real_trades])
    min_d = all_dates[0]
    max_d = all_dates[-1]
    total_days = (max_d - min_d).days
    period_len = total_days / 4

    sub_sharpes = []
    for i in range(4):
        p_start = min_d + timedelta(days=int(i * period_len))
        p_end = min_d + timedelta(days=int((i + 1) * period_len))
        sub = [t for t in real_trades if p_start <= t["entry_date"] < p_end]
        m = compute_metrics(sub)
        sub_sharpes.append(m["sharpe"])
        print(f"  Period {i+1} ({p_start.strftime('%Y-%m-%d')} to {p_end.strftime('%Y-%m-%d')}): "
              f"Sharpe={m['sharpe']}, trades={m['n_trades']}")

    passed = all(s > 0 for s in sub_sharpes)
    print(f"  All positive: {'PASS' if passed else 'FAIL'}")

    return {
        "test": "sub_period_stability",
        "sub_period_sharpes": sub_sharpes,
        "all_positive": passed,
        "passed": passed,
    }


# ── Test 4: Remove Top 3 Tickers ──────────────────────────────────────────
def test_remove_top3(data):
    print("\n=== Test 4: Remove Top 3 Tickers ===")
    real_trades = backtest(data, generate_signals)
    real_m = compute_metrics(real_trades)

    # Find top 3 by cumulative PnL
    ticker_pnl = {}
    for t in real_trades:
        ticker_pnl[t["ticker"]] = ticker_pnl.get(t["ticker"], 0) + t["pnl"]

    sorted_tickers = sorted(ticker_pnl.items(), key=lambda x: x[1], reverse=True)
    top3 = [t[0] for t in sorted_tickers[:3]]
    print(f"  Top 3 tickers by PnL: {top3}")
    for tk, pnl in sorted_tickers[:3]:
        print(f"    {tk}: ${pnl:.2f}")

    remaining = [t for t in TICKERS if t not in top3]
    reduced_trades = backtest(data, generate_signals, tickers=remaining)
    reduced_m = compute_metrics(reduced_trades)

    drop = 1 - (reduced_m["sharpe"] / real_m["sharpe"]) if real_m["sharpe"] != 0 else 1
    passed = drop < 0.50

    print(f"  Full Sharpe: {real_m['sharpe']}, Reduced Sharpe: {reduced_m['sharpe']}")
    print(f"  Drop: {drop:.1%}, PASS threshold: < 50%")
    print(f"  Result: {'PASS' if passed else 'FAIL'}")

    return {
        "test": "remove_top3_tickers",
        "full_sharpe": real_m["sharpe"],
        "reduced_sharpe": reduced_m["sharpe"],
        "removed_tickers": top3,
        "sharpe_drop": round(drop, 4),
        "threshold": 0.50,
        "passed": passed,
    }


# ── Test 5: Parameter Sensitivity ─────────────────────────────────────────
def test_param_sensitivity(data):
    print("\n=== Test 5: Parameter Sensitivity ===")
    lookbacks = [10, 15, 20, 30, 40]
    holds = [5, 7, 10, 15]
    vol_thresholds = [0.3, 0.5, 0.7, 1.0]

    results = []
    total = len(lookbacks) * len(holds) * len(vol_thresholds)
    count = 0

    for lb in lookbacks:
        for hd in holds:
            for vt in vol_thresholds:
                count += 1
                if count % 20 == 0:
                    print(f"  Progress: {count}/{total}")

                def sig_func(df, _lb=lb, _vt=vt):
                    return generate_signals(df, local_low_window=_lb,
                                           vol_threshold=_vt)

                trades = backtest(data, sig_func, hold_days=hd)
                m = compute_metrics(trades)
                results.append({
                    "lookback": lb, "hold": hd, "vol_threshold": vt,
                    "sharpe": m["sharpe"], "n_trades": m["n_trades"],
                })

    sharpes = [r["sharpe"] for r in results]
    above_threshold = sum(1 for s in sharpes if s > 0.3)
    pct = above_threshold / len(sharpes) if sharpes else 0
    passed = pct >= 0.30

    print(f"  Total combos: {len(results)}")
    print(f"  Sharpe > 0.3: {above_threshold} ({pct:.1%})")
    print(f"  PASS threshold: >= 30%")
    print(f"  Sharpe distribution: min={min(sharpes):.3f}, median={np.median(sharpes):.3f}, "
          f"max={max(sharpes):.3f}")
    print(f"  Result: {'PASS' if passed else 'FAIL'}")

    return {
        "test": "parameter_sensitivity",
        "total_combos": len(results),
        "combos_above_0.3": above_threshold,
        "pct_above_0.3": round(pct, 4),
        "threshold": 0.30,
        "sharpe_min": round(min(sharpes), 3),
        "sharpe_median": round(float(np.median(sharpes)), 3),
        "sharpe_max": round(max(sharpes), 3),
        "passed": passed,
        "best_params": sorted(results, key=lambda x: x["sharpe"], reverse=True)[:5],
    }


# ── Test 6: Cost Sensitivity ──────────────────────────────────────────────
def test_cost_sensitivity(data):
    print("\n=== Test 6: Cost Sensitivity ===")
    bps_levels = [5, 10, 20, 50]
    sharpe_by_bps = {}

    for bps in bps_levels:
        trades = backtest(data, generate_signals, slippage_bps=bps)
        m = compute_metrics(trades)
        sharpe_by_bps[bps] = m["sharpe"]
        print(f"  {bps} bps: Sharpe={m['sharpe']}, PF={m['profit_factor']}, trades={m['n_trades']}")

    # Find breakeven via interpolation
    bps_list = sorted(sharpe_by_bps.keys())
    breakeven_bps = 0
    for i in range(len(bps_list) - 1):
        s1 = sharpe_by_bps[bps_list[i]]
        s2 = sharpe_by_bps[bps_list[i + 1]]
        if s1 > 0 and s2 <= 0:
            # Linear interpolation
            frac = s1 / (s1 - s2) if (s1 - s2) != 0 else 0
            breakeven_bps = bps_list[i] + frac * (bps_list[i + 1] - bps_list[i])
            break
    else:
        if sharpe_by_bps[bps_list[-1]] > 0:
            breakeven_bps = bps_list[-1] + 10  # Beyond our test range (extrapolate)
        else:
            breakeven_bps = bps_list[0]

    passed = breakeven_bps >= 20

    print(f"  Estimated breakeven: {breakeven_bps:.1f} bps")
    print(f"  PASS threshold: >= 20 bps")
    print(f"  Result: {'PASS' if passed else 'FAIL'}")

    return {
        "test": "cost_sensitivity",
        "sharpe_by_bps": {str(k): v for k, v in sharpe_by_bps.items()},
        "breakeven_bps": round(breakeven_bps, 1),
        "threshold_bps": 20,
        "passed": passed,
    }


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("RSI Divergence + Volume Decline (Variant C) — Adversarial Validation")
    print("=" * 70)

    print("\nDownloading data...")
    data = download_data()
    print(f"  Downloaded {len(data)} tickers")

    if "SPY" not in data:
        print("ERROR: SPY data not available")
        sys.exit(1)

    # Baseline metrics
    print("\n--- Baseline Strategy ---")
    baseline_trades = backtest(data, generate_signals)
    baseline_m = compute_metrics(baseline_trades)
    bull_s, bear_s = compute_regime_sharpes(baseline_trades, data["SPY"])
    max_s = max(abs(bull_s), abs(bear_s))
    regime_gap = abs(bull_s - bear_s) / max_s if max_s > 0 else 0

    print(f"  Sharpe: {baseline_m['sharpe']}")
    print(f"  Win Rate: {baseline_m['win_rate']:.1%}")
    print(f"  Profit Factor: {baseline_m['profit_factor']}")
    print(f"  Max DD: {baseline_m['max_dd']:.2%}")
    print(f"  Trades: {baseline_m['n_trades']}")
    print(f"  Total PnL: ${baseline_m['total_pnl']:.2f}")
    print(f"  Bull Sharpe: {bull_s}, Bear Sharpe: {bear_s}, Regime Gap: {regime_gap:.3f}")

    # Run all 6 tests
    results = {}
    results["baseline"] = {
        **baseline_m,
        "bull_sharpe": bull_s,
        "bear_sharpe": bear_s,
        "regime_gap": round(regime_gap, 3),
    }

    t1 = test_inverse(data)
    results["test1_inverse"] = t1

    t2 = test_random_timing(data)
    results["test2_random_timing"] = t2

    t3 = test_sub_period(data)
    results["test3_sub_period"] = t3

    t4 = test_remove_top3(data)
    results["test4_remove_top3"] = t4

    t5 = test_param_sensitivity(data)
    results["test5_param_sensitivity"] = t5

    t6 = test_cost_sensitivity(data)
    results["test6_cost_sensitivity"] = t6

    # Summary
    tests = [t1, t2, t3, t4, t5, t6]
    passed_count = sum(1 for t in tests if t["passed"])

    print("\n" + "=" * 70)
    print(f"SUMMARY: {passed_count}/6 tests PASSED")
    print("=" * 70)
    for i, t in enumerate(tests, 1):
        status = "PASS ✓" if t["passed"] else "FAIL ✗"
        print(f"  Test {i} ({t['test']}): {status}")

    results["summary"] = {
        "tests_passed": passed_count,
        "tests_total": 6,
        "pass_rate": round(passed_count / 6, 2),
        "overall_pass": passed_count >= 4,
        "timestamp": datetime.now().isoformat(),
        "strategy": "RSI Divergence + Volume Decline (Variant C)",
    }

    # Save results
    out_path = Path("/home/jupiter/Lvl3Quant/data/rsi_divergence_c_adversarial.json")

    # Convert timestamps to strings for JSON
    def convert(obj):
        if isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Not serializable: {type(obj)}")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=convert)

    print(f"\nResults saved to {out_path}")
    return results


if __name__ == "__main__":
    main()
