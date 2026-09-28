#!/usr/bin/env python3
"""
Adversarial Validation: Cross-Asset Bond Yield Signal (Variant B)

Strategy: When 10Y Treasury yield drops >0.1% in 5 trading days,
buy quality stocks >5% below 20-day SMA. Hold 10 days.
Max $200/trade, max 3 concurrent, 2bps slippage baseline.

6 adversarial tests to validate the signal is real, not luck.
"""

import json
import warnings
import sys
from datetime import datetime, timedelta
from itertools import product

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────────
UNIVERSE = [
    "AAPL", "MSFT", "AVGO", "JPM", "JNJ", "PG", "KO", "PEP", "HD", "COST",
    "UNH", "LLY", "V", "MA", "ABBV", "MRK", "WMT", "AMZN", "GOOGL", "META",
]
START = "2021-06-01"  # buffer for lookbacks
END = "2026-07-31"
EVAL_START = "2022-01-01"
MAX_PER_TRADE = 200.0
MAX_CONCURRENT = 3
BASELINE_SLIPPAGE_BPS = 2

# Baseline params
YIELD_DROP_THRESH = 0.10   # percentage points
YIELD_LOOKBACK = 5         # trading days
DIP_THRESH = 0.05          # 5% below 20-SMA
HOLD_DAYS = 10
SMA_PERIOD = 20

OUTPUT_PATH = "/home/jupiter/Lvl3Quant/data/bond_yield_signal_adversarial.json"


# ── Data Download ───────────────────────────────────────────────────────
def download_data():
    """Download yield and stock data."""
    print("Downloading ^TNX yield data...")
    tnx = yf.download("^TNX", start=START, end=END, progress=False)
    if isinstance(tnx.columns, pd.MultiIndex):
        tnx.columns = tnx.columns.get_level_values(0)
    tnx_close = tnx["Close"].dropna()
    tnx_close.index = pd.to_datetime(tnx_close.index).tz_localize(None)

    print(f"Downloading {len(UNIVERSE)} stock tickers...")
    stock_data = yf.download(UNIVERSE, start=START, end=END, progress=False)
    if isinstance(stock_data.columns, pd.MultiIndex):
        closes = stock_data["Close"]
    else:
        closes = stock_data[["Close"]]
        closes.columns = UNIVERSE
    closes.index = pd.to_datetime(closes.index).tz_localize(None)

    return tnx_close, closes


# ── Strategy Engine ─────────────────────────────────────────────────────
def run_strategy(
    tnx_close,
    closes,
    yield_drop_thresh=YIELD_DROP_THRESH,
    yield_lookback=YIELD_LOOKBACK,
    dip_thresh=DIP_THRESH,
    hold_days=HOLD_DAYS,
    slippage_bps=BASELINE_SLIPPAGE_BPS,
    inverse_yield=False,
    inverse_dip=False,
    signal_dates_override=None,
    exclude_tickers=None,
    eval_start=EVAL_START,
):
    """
    Run the bond yield signal strategy and return trade-level results.

    inverse_yield: if True, buy when yield RISES instead of drops
    inverse_dip: if True, buy stocks ABOVE SMA instead of below
    signal_dates_override: if provided, use these dates as signal fire dates
    exclude_tickers: list of tickers to exclude
    """
    universe = [t for t in UNIVERSE if t not in (exclude_tickers or [])]
    eval_start_dt = pd.Timestamp(eval_start)

    # Compute yield change
    yield_change = tnx_close.diff(yield_lookback)

    # Determine signal fire dates
    if signal_dates_override is not None:
        signal_dates = sorted(signal_dates_override)
    else:
        if inverse_yield:
            # Yield RISES > threshold
            signal_mask = yield_change > yield_drop_thresh
        else:
            # Yield DROPS > threshold (change is negative)
            signal_mask = yield_change < -yield_drop_thresh
        signal_dates = signal_mask[signal_mask].index.tolist()

    signal_dates = [d for d in signal_dates if d >= eval_start_dt]

    # Compute SMA for each stock
    sma = closes[universe].rolling(SMA_PERIOD).mean()

    trades = []
    open_positions = []  # list of (exit_date, ticker)

    for sig_date in signal_dates:
        if sig_date not in closes.index:
            continue

        # Count open positions on this date
        open_positions = [(ed, tk) for ed, tk in open_positions if ed > sig_date]
        if len(open_positions) >= MAX_CONCURRENT:
            continue

        # Find eligible stocks
        eligible = []
        for ticker in universe:
            if ticker not in closes.columns or ticker not in sma.columns:
                continue
            price = closes.loc[sig_date, ticker] if sig_date in closes.index else np.nan
            sma_val = sma.loc[sig_date, ticker] if sig_date in sma.index else np.nan
            if pd.isna(price) or pd.isna(sma_val) or sma_val == 0:
                continue

            pct_below = (price - sma_val) / sma_val

            if inverse_dip:
                # Buy stocks ABOVE SMA
                if pct_below > dip_thresh:
                    eligible.append((ticker, price, abs(pct_below)))
            else:
                # Buy stocks BELOW SMA by threshold
                if pct_below < -dip_thresh:
                    eligible.append((ticker, price, abs(pct_below)))

        if not eligible:
            continue

        # Sort by magnitude of dip (or rise for inverse), take top candidate
        eligible.sort(key=lambda x: x[2], reverse=True)

        slots = MAX_CONCURRENT - len(open_positions)
        for ticker, entry_price, _ in eligible[:slots]:
            # Calculate exit date
            future_dates = closes.index[closes.index > sig_date]
            if len(future_dates) < hold_days:
                continue
            exit_date = future_dates[hold_days - 1]

            exit_price = closes.loc[exit_date, ticker]
            if pd.isna(exit_price):
                continue

            # Position sizing
            shares = max(1, int(MAX_PER_TRADE / entry_price))
            notional = shares * entry_price

            # Slippage cost (applied to both entry and exit)
            slip_cost = notional * slippage_bps / 10000 * 2  # round trip

            pnl = shares * (exit_price - entry_price) - slip_cost
            ret = pnl / notional

            trades.append({
                "entry_date": str(sig_date.date()),
                "exit_date": str(exit_date.date()),
                "ticker": ticker,
                "entry_price": float(entry_price),
                "exit_price": float(exit_price),
                "shares": shares,
                "pnl": float(pnl),
                "return": float(ret),
            })

            open_positions.append((exit_date, ticker))

    return trades


def compute_metrics(trades):
    """Compute Sharpe, Sortino, WR, PF, MDD, total return from trade list."""
    if not trades:
        return {
            "sharpe": 0.0, "sortino": 0.0, "win_rate": 0.0,
            "profit_factor": 0.0, "mdd_pct": 0.0, "total_return_pct": 0.0,
            "n_trades": 0, "total_pnl": 0.0,
        }

    returns = [t["return"] for t in trades]
    pnls = [t["pnl"] for t in trades]

    avg_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1) if len(returns) > 1 else 1e-9
    downside = np.std([r for r in returns if r < 0], ddof=1) if any(r < 0 for r in returns) else 1e-9

    sharpe = avg_ret / std_ret * np.sqrt(252 / HOLD_DAYS) if std_ret > 1e-9 else 0.0
    sortino = avg_ret / downside * np.sqrt(252 / HOLD_DAYS) if downside > 1e-9 else 0.0

    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    wr = len(wins) / len(pnls) if pnls else 0.0
    pf = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else float("inf")

    # MDD on cumulative PnL
    cum_pnl = np.cumsum(pnls)
    capital = MAX_PER_TRADE * MAX_CONCURRENT
    cum_ret = cum_pnl / capital
    peak = np.maximum.accumulate(cum_ret)
    dd = cum_ret - peak
    mdd = float(np.min(dd)) * 100 if len(dd) > 0 else 0.0

    total_ret = float(cum_ret[-1]) * 100 if len(cum_ret) > 0 else 0.0

    return {
        "sharpe": round(float(sharpe), 4),
        "sortino": round(float(sortino), 4),
        "win_rate": round(float(wr), 4),
        "profit_factor": round(float(min(pf, 999.0)), 4),
        "mdd_pct": round(float(mdd), 2),
        "total_return_pct": round(float(total_ret), 2),
        "n_trades": len(trades),
        "total_pnl": round(float(sum(pnls)), 2),
    }


# ── Test 1: Inverse Signal ─────────────────────────────────────────────
def test_inverse_signal(tnx_close, closes, real_sharpe):
    """Run opposite strategy. PASS if inverse Sharpe < 0.5x real."""
    print("\n[Test 1] Inverse Signal Test...")

    # Test A: inverse yield direction
    trades_inv_yield = run_strategy(tnx_close, closes, inverse_yield=True)
    m_inv_yield = compute_metrics(trades_inv_yield)

    # Test B: inverse dip direction
    trades_inv_dip = run_strategy(tnx_close, closes, inverse_dip=True)
    m_inv_dip = compute_metrics(trades_inv_dip)

    # Worst case for us: whichever inverse has higher Sharpe
    worst_inv_sharpe = max(m_inv_yield["sharpe"], m_inv_dip["sharpe"])
    threshold = 0.5 * real_sharpe
    passed = worst_inv_sharpe < threshold

    print(f"  Real Sharpe: {real_sharpe:.4f}")
    print(f"  Inverse yield Sharpe: {m_inv_yield['sharpe']:.4f} ({m_inv_yield['n_trades']} trades)")
    print(f"  Inverse dip Sharpe: {m_inv_dip['sharpe']:.4f} ({m_inv_dip['n_trades']} trades)")
    print(f"  Threshold (0.5x real): {threshold:.4f}")
    print(f"  PASS: {passed}")

    return {
        "test": "inverse_signal",
        "passed": passed,
        "real_sharpe": real_sharpe,
        "inverse_yield_sharpe": m_inv_yield["sharpe"],
        "inverse_yield_trades": m_inv_yield["n_trades"],
        "inverse_dip_sharpe": m_inv_dip["sharpe"],
        "inverse_dip_trades": m_inv_dip["n_trades"],
        "worst_inverse_sharpe": worst_inv_sharpe,
        "threshold": round(threshold, 4),
    }


# ── Test 2: Random Timing ──────────────────────────────────────────────
def test_random_timing(tnx_close, closes, real_sharpe):
    """Shuffle signal dates 1000 times. PASS if real Sharpe p < 0.05."""
    print("\n[Test 2] Random Timing Test (1000 permutations)...")

    # Get real signal dates
    yield_change = tnx_close.diff(YIELD_LOOKBACK)
    signal_mask = yield_change < -YIELD_DROP_THRESH
    real_signal_dates = signal_mask[signal_mask].index.tolist()
    eval_start_dt = pd.Timestamp(EVAL_START)
    real_signal_dates = [d for d in real_signal_dates if d >= eval_start_dt]
    n_signals = len(real_signal_dates)

    # All valid trading dates in eval period
    all_dates = closes.index[closes.index >= eval_start_dt].tolist()

    rng = np.random.RandomState(42)
    random_sharpes = []

    for i in range(1000):
        random_dates = sorted(rng.choice(all_dates, size=min(n_signals, len(all_dates)), replace=False))
        random_dates = [pd.Timestamp(d) for d in random_dates]
        trades = run_strategy(tnx_close, closes, signal_dates_override=random_dates)
        m = compute_metrics(trades)
        random_sharpes.append(m["sharpe"])

        if (i + 1) % 200 == 0:
            print(f"  {i+1}/1000 permutations done...")

    random_sharpes = np.array(random_sharpes)
    percentile = float(np.mean(random_sharpes >= real_sharpe))
    p_value = percentile
    passed = p_value < 0.05

    print(f"  Real Sharpe: {real_sharpe:.4f}")
    print(f"  Random Sharpe mean: {np.mean(random_sharpes):.4f} +/- {np.std(random_sharpes):.4f}")
    print(f"  p-value (rank): {p_value:.4f}")
    print(f"  PASS: {passed}")

    return {
        "test": "random_timing",
        "passed": passed,
        "real_sharpe": real_sharpe,
        "random_sharpe_mean": round(float(np.mean(random_sharpes)), 4),
        "random_sharpe_std": round(float(np.std(random_sharpes)), 4),
        "random_sharpe_median": round(float(np.median(random_sharpes)), 4),
        "p_value": round(p_value, 4),
        "percentile_rank": round((1 - p_value) * 100, 1),
        "n_permutations": 1000,
        "n_signal_dates": n_signals,
    }


# ── Test 3: Sub-Period Stability ────────────────────────────────────────
def test_subperiod_stability(tnx_close, closes):
    """Split into 4 sub-periods. PASS if ALL have positive Sharpe."""
    print("\n[Test 3] Sub-Period Stability...")

    eval_dates = closes.index[closes.index >= pd.Timestamp(EVAL_START)]
    n = len(eval_dates)
    quarter = n // 4

    periods = []
    for i in range(4):
        start_idx = i * quarter
        end_idx = (i + 1) * quarter if i < 3 else n
        p_start = eval_dates[start_idx]
        p_end = eval_dates[end_idx - 1]
        periods.append((p_start, p_end))

    results = []
    all_positive = True
    for i, (p_start, p_end) in enumerate(periods):
        trades = run_strategy(
            tnx_close, closes,
            eval_start=str(p_start.date()),
        )
        # Filter trades to this sub-period
        trades = [t for t in trades
                  if p_start <= pd.Timestamp(t["entry_date"]) <= p_end]
        m = compute_metrics(trades)

        if m["sharpe"] <= 0:
            all_positive = False

        period_label = f"{p_start.date()} to {p_end.date()}"
        print(f"  Period {i+1} ({period_label}): Sharpe={m['sharpe']:.4f}, "
              f"WR={m['win_rate']:.2%}, {m['n_trades']} trades")

        results.append({
            "period": period_label,
            "sharpe": m["sharpe"],
            "sortino": m["sortino"],
            "win_rate": m["win_rate"],
            "profit_factor": m["profit_factor"],
            "n_trades": m["n_trades"],
            "total_pnl": m["total_pnl"],
        })

    print(f"  All positive Sharpe: {all_positive}")
    print(f"  PASS: {all_positive}")

    return {
        "test": "subperiod_stability",
        "passed": all_positive,
        "sub_periods": results,
    }


# ── Test 4: Remove Top-3 Tickers ───────────────────────────────────────
def test_remove_top3(tnx_close, closes, real_sharpe):
    """Remove 3 highest-PnL tickers. PASS if Sharpe drops < 50%."""
    print("\n[Test 4] Remove Top-3 Tickers...")

    # Run baseline to find per-ticker PnL
    trades = run_strategy(tnx_close, closes)
    ticker_pnl = {}
    for t in trades:
        ticker_pnl[t["ticker"]] = ticker_pnl.get(t["ticker"], 0) + t["pnl"]

    sorted_tickers = sorted(ticker_pnl.items(), key=lambda x: x[1], reverse=True)
    top3 = [t[0] for t in sorted_tickers[:3]]
    top3_pnl = {t[0]: round(t[1], 2) for t in sorted_tickers[:3]}

    print(f"  Top 3 tickers by PnL: {top3_pnl}")

    # Re-run excluding top 3
    trades_ex = run_strategy(tnx_close, closes, exclude_tickers=top3)
    m_ex = compute_metrics(trades_ex)

    if real_sharpe > 0:
        drop_pct = (real_sharpe - m_ex["sharpe"]) / real_sharpe * 100
    else:
        drop_pct = 100.0

    passed = drop_pct < 50.0

    print(f"  Real Sharpe: {real_sharpe:.4f}")
    print(f"  Ex-top3 Sharpe: {m_ex['sharpe']:.4f} ({m_ex['n_trades']} trades)")
    print(f"  Sharpe drop: {drop_pct:.1f}%")
    print(f"  PASS: {passed}")

    return {
        "test": "remove_top3_tickers",
        "passed": passed,
        "top3_tickers": top3,
        "top3_pnl": top3_pnl,
        "real_sharpe": real_sharpe,
        "ex_top3_sharpe": m_ex["sharpe"],
        "ex_top3_trades": m_ex["n_trades"],
        "sharpe_drop_pct": round(drop_pct, 1),
    }


# ── Test 5: Parameter Sensitivity Grid ─────────────────────────────────
def test_param_sensitivity(tnx_close, closes):
    """Test parameter grid. PASS if >30% of combos have Sharpe > 0.3."""
    print("\n[Test 5] Parameter Sensitivity Grid...")

    yield_thresholds = [0.05, 0.08, 0.10, 0.15, 0.20]
    yield_lookbacks = [3, 5, 7, 10]
    dip_thresholds = [0.03, 0.05, 0.07, 0.10]
    hold_periods = [5, 7, 10, 15]

    total_combos = len(yield_thresholds) * len(yield_lookbacks) * len(dip_thresholds) * len(hold_periods)
    print(f"  Testing {total_combos} parameter combinations...")

    results = []
    count_above = 0
    done = 0

    for yt, yl, dt, hp in product(yield_thresholds, yield_lookbacks, dip_thresholds, hold_periods):
        trades = run_strategy(
            tnx_close, closes,
            yield_drop_thresh=yt,
            yield_lookback=yl,
            dip_thresh=dt,
            hold_days=hp,
        )
        m = compute_metrics(trades)

        if m["sharpe"] > 0.3:
            count_above += 1

        results.append({
            "yield_thresh": yt,
            "yield_lookback": yl,
            "dip_thresh": dt,
            "hold_days": hp,
            "sharpe": m["sharpe"],
            "n_trades": m["n_trades"],
        })

        done += 1
        if done % 80 == 0:
            print(f"  {done}/{total_combos} combinations done...")

    pct_above = count_above / total_combos * 100
    passed = pct_above > 30.0

    # Find best and worst
    results_sorted = sorted(results, key=lambda x: x["sharpe"], reverse=True)
    sharpes = [r["sharpe"] for r in results]

    print(f"  {count_above}/{total_combos} combos with Sharpe > 0.3 ({pct_above:.1f}%)")
    print(f"  Sharpe range: [{min(sharpes):.4f}, {max(sharpes):.4f}]")
    print(f"  Best combo: {results_sorted[0]}")
    print(f"  PASS: {passed}")

    return {
        "test": "parameter_sensitivity",
        "passed": passed,
        "total_combinations": total_combos,
        "count_sharpe_above_0_3": count_above,
        "pct_above_threshold": round(pct_above, 1),
        "sharpe_min": round(float(min(sharpes)), 4),
        "sharpe_max": round(float(max(sharpes)), 4),
        "sharpe_mean": round(float(np.mean(sharpes)), 4),
        "sharpe_median": round(float(np.median(sharpes)), 4),
        "best_params": results_sorted[0],
        "worst_params": results_sorted[-1],
    }


# ── Test 6: Cost Sensitivity ───────────────────────────────────────────
def test_cost_sensitivity(tnx_close, closes):
    """Test at various slippage levels. PASS if breakeven > 20 bps."""
    print("\n[Test 6] Cost Sensitivity...")

    cost_levels = [5, 10, 20, 50]
    results = []

    for bps in cost_levels:
        trades = run_strategy(tnx_close, closes, slippage_bps=bps)
        m = compute_metrics(trades)
        print(f"  {bps} bps: Sharpe={m['sharpe']:.4f}, PnL=${m['total_pnl']:.2f}, {m['n_trades']} trades")
        results.append({
            "slippage_bps": bps,
            "sharpe": m["sharpe"],
            "total_pnl": m["total_pnl"],
            "win_rate": m["win_rate"],
            "n_trades": m["n_trades"],
        })

    # Estimate breakeven by interpolation
    # Find where Sharpe crosses 0
    sharpes = [(r["slippage_bps"], r["sharpe"]) for r in results]

    # Also run at finer granularity near zero-crossing
    breakeven_bps = None
    for bps in range(1, 200):
        trades = run_strategy(tnx_close, closes, slippage_bps=bps)
        m = compute_metrics(trades)
        if m["sharpe"] <= 0 or m["total_pnl"] <= 0:
            breakeven_bps = bps - 1
            break
    else:
        breakeven_bps = 200  # still profitable at 200 bps

    passed = breakeven_bps > 20

    print(f"  Breakeven: ~{breakeven_bps} bps")
    print(f"  PASS: {passed}")

    return {
        "test": "cost_sensitivity",
        "passed": passed,
        "cost_levels": results,
        "breakeven_bps": breakeven_bps,
    }


# ── Main ────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("ADVERSARIAL VALIDATION: Bond Yield Signal (Variant B)")
    print("=" * 60)

    tnx_close, closes = download_data()
    print(f"Yield data: {tnx_close.index[0].date()} to {tnx_close.index[-1].date()}, {len(tnx_close)} points")
    print(f"Stock data: {closes.index[0].date()} to {closes.index[-1].date()}, {len(closes)} days")

    # Run baseline
    print("\nRunning baseline strategy...")
    baseline_trades = run_strategy(tnx_close, closes)
    baseline = compute_metrics(baseline_trades)
    real_sharpe = baseline["sharpe"]
    print(f"Baseline: Sharpe={baseline['sharpe']:.4f}, WR={baseline['win_rate']:.2%}, "
          f"PF={baseline['profit_factor']:.3f}, {baseline['n_trades']} trades, "
          f"PnL=${baseline['total_pnl']:.2f}")

    # Run all 6 tests
    test_results = []

    r1 = test_inverse_signal(tnx_close, closes, real_sharpe)
    test_results.append(r1)

    r2 = test_random_timing(tnx_close, closes, real_sharpe)
    test_results.append(r2)

    r3 = test_subperiod_stability(tnx_close, closes)
    test_results.append(r3)

    r4 = test_remove_top3(tnx_close, closes, real_sharpe)
    test_results.append(r4)

    r5 = test_param_sensitivity(tnx_close, closes)
    test_results.append(r5)

    r6 = test_cost_sensitivity(tnx_close, closes)
    test_results.append(r6)

    # Summary
    tests_passed = sum(1 for r in test_results if r["passed"])
    tests_total = len(test_results)
    overall_pass = tests_passed >= 5  # require 5/6

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in test_results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  [{status}] {r['test']}")
    print(f"\n  Tests passed: {tests_passed}/{tests_total}")
    print(f"  Overall: {'PASS' if overall_pass else 'FAIL'}")

    output = {
        "strategy": "Cross-Asset Bond Yield Signal (Variant B)",
        "validation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "eval_period": f"{EVAL_START} to {END}",
        "baseline_metrics": baseline,
        "tests": test_results,
        "summary": {
            "tests_passed": tests_passed,
            "tests_total": tests_total,
            "overall_pass": overall_pass,
            "pass_threshold": "5/6 tests required",
        },
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {OUTPUT_PATH}")

    return output


if __name__ == "__main__":
    main()
