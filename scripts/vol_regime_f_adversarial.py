#!/usr/bin/env python3
"""
Adversarial validation for Vol Regime Variant F (IV-RV Gap Entry).

Strategy: When VIX > 20d realized vol of SPY by 5+ pts, AND a quality stock
is >5% below its 20d high with RSI(14)<40, buy. Hold 10 days. Max $200/trade,
max 3 concurrent, 2bps slippage.

6 adversarial tests:
1. Inverse Signal
2. Random Timing (1000 permutations)
3. Sub-Period Stability (4 periods)
4. Remove Top-3 Tickers
5. Parameter Sensitivity Grid
6. Cost Sensitivity / Breakeven
"""

import json
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ─── Constants ───────────────────────────────────────────────────────────────
UNIVERSE = [
    "AAPL", "MSFT", "AVGO", "JPM", "JNJ", "PG", "KO", "PEP", "HD", "COST",
    "UNH", "LLY", "V", "MA", "ABBV", "MRK", "WMT", "AMZN", "GOOGL", "META",
]
START = "2022-01-01"
END = "2026-07-31"
MAX_PER_TRADE = 200.0
MAX_CONCURRENT = 3
DEFAULT_SLIP_BPS = 2
DEFAULT_GAP = 5
DEFAULT_DIP = 5  # percent below 20d high
DEFAULT_RSI = 40
DEFAULT_HOLD = 10  # trading days
PERM_ITERATIONS = 1000
RNG = np.random.RandomState(42)


# ─── Data Download ───────────────────────────────────────────────────────────
def download_data():
    """Download VIX + universe price data."""
    print("Downloading VIX...")
    vix = yf.download("^VIX", start=START, end=END, progress=False)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
    vix = vix[["Close"]].rename(columns={"Close": "VIX"})
    vix.index = pd.to_datetime(vix.index).tz_localize(None)

    print("Downloading SPY for realized vol...")
    spy = yf.download("SPY", start=START, end=END, progress=False)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    spy = spy[["Close"]].rename(columns={"Close": "SPY"})
    spy.index = pd.to_datetime(spy.index).tz_localize(None)

    print(f"Downloading {len(UNIVERSE)} tickers...")
    prices = yf.download(UNIVERSE, start=START, end=END, progress=False)
    if isinstance(prices.columns, pd.MultiIndex):
        close = prices["Close"]
    else:
        close = prices[["Close"]]
        close.columns = UNIVERSE[:1]
    close.index = pd.to_datetime(close.index).tz_localize(None)

    return vix, spy, close


def compute_realized_vol(spy, window=20):
    """20-day realized vol = std(daily log returns) * sqrt(252) * 100."""
    log_ret = np.log(spy["SPY"] / spy["SPY"].shift(1))
    rv = log_ret.rolling(window).std() * np.sqrt(252) * 100
    return rv.rename("RV20")


def compute_rsi(series, period=14):
    """Standard RSI."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


# ─── Strategy Engine ─────────────────────────────────────────────────────────
def run_strategy(
    vix, rv, close,
    gap_thresh=DEFAULT_GAP,
    dip_thresh=DEFAULT_DIP,
    rsi_thresh=DEFAULT_RSI,
    hold_days=DEFAULT_HOLD,
    slip_bps=DEFAULT_SLIP_BPS,
    signal_dates_override=None,
    inverse=False,
    exclude_tickers=None,
):
    """
    Run the Vol Regime F strategy and return list of trade dicts.

    If inverse=True, invert the signal logic.
    If signal_dates_override is provided, use those dates for signal generation
    (for permutation tests).
    If exclude_tickers is set, skip those tickers.
    """
    tickers = [t for t in UNIVERSE if t not in (exclude_tickers or [])]

    # Align dates
    common_idx = vix.index.intersection(rv.index).intersection(close.index)
    common_idx = common_idx.sort_values()

    vix_s = vix.loc[common_idx, "VIX"]
    rv_s = rv.loc[common_idx]

    # IV-RV gap
    iv_rv_gap = vix_s - rv_s

    # Determine signal dates (IV-RV gap condition met)
    if signal_dates_override is not None:
        signal_dates = set(pd.to_datetime(signal_dates_override))
    else:
        if inverse:
            # Inverse: VIX BELOW realized vol (gap < 0)
            signal_dates = set(common_idx[iv_rv_gap < 0])
        else:
            signal_dates = set(common_idx[iv_rv_gap >= gap_thresh])

    # Pre-compute RSI and 20d high for each ticker
    ticker_rsi = {}
    ticker_high20 = {}
    for t in tickers:
        if t not in close.columns:
            continue
        s = close[t].dropna()
        ticker_rsi[t] = compute_rsi(s)
        ticker_high20[t] = s.rolling(20).max()

    # Generate trades
    trades = []
    open_positions = []  # list of (exit_date_idx, ticker)

    date_list = sorted(common_idx)
    date_to_idx = {d: i for i, d in enumerate(date_list)}

    for d in date_list:
        # Close expired positions (they resolved at entry time, just tracking concurrency)
        open_positions = [(ed, t) for ed, t in open_positions if ed > d]

        if d not in signal_dates:
            continue

        current_open = len(open_positions)
        if current_open >= MAX_CONCURRENT:
            continue

        # Scan tickers for entry conditions
        for t in tickers:
            if current_open >= MAX_CONCURRENT:
                break
            if t not in close.columns or t not in ticker_rsi or t not in ticker_high20:
                continue

            price = close.at[d, t] if d in close.index else np.nan
            if pd.isna(price) or price <= 0:
                continue

            rsi_val = ticker_rsi[t].get(d, np.nan) if d in ticker_rsi[t].index else np.nan
            high20 = ticker_high20[t].get(d, np.nan) if d in ticker_high20[t].index else np.nan

            if pd.isna(rsi_val) or pd.isna(high20):
                continue

            pct_below_high = (high20 - price) / high20 * 100

            if inverse:
                # Inverse: buy stocks ABOVE 20d high or RSI > (100 - thresh)
                stock_cond = (pct_below_high < 0) or (rsi_val > (100 - rsi_thresh))
            else:
                stock_cond = (pct_below_high >= dip_thresh) and (rsi_val < rsi_thresh)

            if not stock_cond:
                continue

            # Already have open position in this ticker?
            if any(tk == t for _, tk in open_positions):
                continue

            # Calculate exit
            d_idx = date_to_idx[d]
            exit_idx = min(d_idx + hold_days, len(date_list) - 1)
            exit_date = date_list[exit_idx]

            exit_price = close.at[exit_date, t] if exit_date in close.index else np.nan
            if pd.isna(exit_price):
                continue

            # Position sizing
            shares = max(1, int(MAX_PER_TRADE / price))

            # Slippage
            slip_frac = slip_bps / 10000.0
            entry_cost = price * (1 + slip_frac)
            exit_proceeds = exit_price * (1 - slip_frac)

            pnl = (exit_proceeds - entry_cost) * shares
            ret = (exit_proceeds - entry_cost) / entry_cost

            trades.append({
                "ticker": t,
                "entry_date": str(d.date()),
                "exit_date": str(exit_date.date()),
                "entry_price": round(entry_cost, 4),
                "exit_price": round(exit_proceeds, 4),
                "shares": shares,
                "pnl": round(pnl, 2),
                "return": round(ret, 6),
            })

            open_positions.append((exit_date, t))
            current_open += 1

    return trades


def compute_metrics(trades):
    """Compute strategy metrics from trade list."""
    if not trades:
        return {
            "sharpe": 0.0, "sortino": 0.0, "win_rate": 0.0,
            "profit_factor": 0.0, "max_dd_pct": 0.0,
            "n_trades": 0, "total_pnl": 0.0,
        }

    returns = np.array([t["return"] for t in trades])
    pnls = np.array([t["pnl"] for t in trades])

    n = len(returns)
    mean_r = np.mean(returns)
    std_r = np.std(returns, ddof=1) if n > 1 else 1e-9

    # Annualize: ~25 trades/year assumption based on hold period
    trades_per_year = 252 / DEFAULT_HOLD
    sharpe = (mean_r / max(std_r, 1e-9)) * np.sqrt(trades_per_year)

    downside = returns[returns < 0]
    down_std = np.std(downside, ddof=1) if len(downside) > 1 else 1e-9
    sortino = (mean_r / max(down_std, 1e-9)) * np.sqrt(trades_per_year)

    winners = pnls[pnls > 0]
    losers = pnls[pnls < 0]
    win_rate = len(winners) / n * 100 if n > 0 else 0
    pf = abs(winners.sum() / losers.sum()) if len(losers) > 0 and losers.sum() != 0 else 999.0

    # Max drawdown on cumulative PnL
    cum_pnl = np.cumsum(pnls)
    peak = np.maximum.accumulate(cum_pnl)
    # Use total capital deployed as denominator
    total_capital = sum(t["entry_price"] * t["shares"] for t in trades)
    avg_capital = total_capital / n if n > 0 else 1
    dd = (cum_pnl - peak) / max(avg_capital, 1) * 100
    max_dd = dd.min() if len(dd) > 0 else 0

    return {
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "win_rate": round(win_rate, 1),
        "profit_factor": round(min(pf, 999), 2),
        "max_dd_pct": round(max_dd, 2),
        "n_trades": n,
        "total_pnl": round(pnls.sum(), 2),
    }


def compute_pnl_by_ticker(trades):
    """Return dict of ticker -> total PnL."""
    by_ticker = {}
    for t in trades:
        by_ticker[t["ticker"]] = by_ticker.get(t["ticker"], 0) + t["pnl"]
    return by_ticker


# ─── Adversarial Tests ───────────────────────────────────────────────────────

def test_inverse_signal(vix, rv, close, baseline_sharpe):
    """Test 1: Inverse signal should produce Sharpe < 0.5× baseline."""
    print("\n[1/6] Inverse Signal Test...")
    trades = run_strategy(vix, rv, close, inverse=True)
    metrics = compute_metrics(trades)
    inv_sharpe = metrics["sharpe"]
    threshold = 0.5 * baseline_sharpe
    passed = inv_sharpe < threshold
    print(f"  Inverse Sharpe: {inv_sharpe:.3f}, Threshold: <{threshold:.3f}, PASS: {passed}")
    return {
        "test": "inverse_signal",
        "inverse_sharpe": inv_sharpe,
        "baseline_sharpe": round(baseline_sharpe, 3),
        "threshold": round(threshold, 3),
        "n_trades": metrics["n_trades"],
        "total_pnl": metrics["total_pnl"],
        "passed": passed,
    }


def test_random_timing(vix, rv, close, baseline_sharpe):
    """Test 2: Shuffle signal dates 1000 times. Real Sharpe should beat >95%."""
    print("\n[2/6] Random Timing (Permutation) Test...")

    # Get the actual signal dates
    common_idx = vix.index.intersection(rv.index).intersection(close.index).sort_values()
    vix_s = vix.loc[common_idx, "VIX"]
    rv_s = rv.loc[common_idx]
    iv_rv_gap = vix_s - rv_s
    real_signal_dates = common_idx[iv_rv_gap >= DEFAULT_GAP].tolist()
    n_signal = len(real_signal_dates)
    all_dates = common_idx.tolist()

    perm_sharpes = []
    for i in range(PERM_ITERATIONS):
        if (i + 1) % 200 == 0:
            print(f"  Permutation {i+1}/{PERM_ITERATIONS}...")
        shuffled = list(RNG.choice(all_dates, size=n_signal, replace=False))
        trades = run_strategy(vix, rv, close, signal_dates_override=shuffled)
        m = compute_metrics(trades)
        perm_sharpes.append(m["sharpe"])

    perm_sharpes = np.array(perm_sharpes)
    p_value = np.mean(perm_sharpes >= baseline_sharpe)
    passed = p_value < 0.05

    print(f"  Permutation p-value: {p_value:.4f}, Mean perm Sharpe: {np.mean(perm_sharpes):.3f}")
    print(f"  PASS: {passed}")

    return {
        "test": "random_timing",
        "p_value": round(float(p_value), 4),
        "baseline_sharpe": round(baseline_sharpe, 3),
        "perm_mean_sharpe": round(float(np.mean(perm_sharpes)), 3),
        "perm_median_sharpe": round(float(np.median(perm_sharpes)), 3),
        "perm_std_sharpe": round(float(np.std(perm_sharpes)), 3),
        "perm_95th": round(float(np.percentile(perm_sharpes, 95)), 3),
        "n_permutations": PERM_ITERATIONS,
        "passed": passed,
    }


def test_sub_period_stability(vix, rv, close):
    """Test 3: 4 equal sub-periods, all must have positive Sharpe."""
    print("\n[3/6] Sub-Period Stability Test...")

    common_idx = vix.index.intersection(close.index).sort_values()
    total_days = len(common_idx)
    period_size = total_days // 4

    period_results = []
    for i in range(4):
        start_idx = i * period_size
        end_idx = (i + 1) * period_size if i < 3 else total_days
        period_dates = common_idx[start_idx:end_idx]

        p_start = period_dates[0]
        p_end = period_dates[-1]

        # Filter data to this period
        mask = (close.index >= p_start) & (close.index <= p_end)
        vix_mask = (vix.index >= p_start) & (vix.index <= p_end)
        rv_mask = (rv.index >= p_start) & (rv.index <= p_end)

        trades = run_strategy(
            vix[vix_mask], rv[rv_mask], close[mask],
        )
        metrics = compute_metrics(trades)

        period_results.append({
            "period": f"P{i+1}",
            "start": str(p_start.date()),
            "end": str(p_end.date()),
            "sharpe": metrics["sharpe"],
            "n_trades": metrics["n_trades"],
            "total_pnl": metrics["total_pnl"],
            "win_rate": metrics["win_rate"],
        })
        print(f"  P{i+1} ({p_start.date()} to {p_end.date()}): "
              f"Sharpe={metrics['sharpe']:.3f}, N={metrics['n_trades']}, PnL=${metrics['total_pnl']:.0f}")

    all_positive = all(p["sharpe"] > 0 for p in period_results)
    print(f"  All periods positive Sharpe: {all_positive}")

    return {
        "test": "sub_period_stability",
        "periods": period_results,
        "all_positive_sharpe": all_positive,
        "passed": all_positive,
    }


def test_remove_top3(vix, rv, close, baseline_sharpe, baseline_trades):
    """Test 4: Remove 3 highest-PnL tickers. Sharpe should drop < 50%."""
    print("\n[4/6] Remove Top-3 Tickers Test...")

    pnl_by_ticker = compute_pnl_by_ticker(baseline_trades)
    sorted_tickers = sorted(pnl_by_ticker.items(), key=lambda x: x[1], reverse=True)
    top3 = [t[0] for t in sorted_tickers[:3]]
    top3_pnl = [round(t[1], 2) for t in sorted_tickers[:3]]

    print(f"  Top 3 tickers by PnL: {list(zip(top3, top3_pnl))}")

    trades = run_strategy(vix, rv, close, exclude_tickers=top3)
    metrics = compute_metrics(trades)

    reduced_sharpe = metrics["sharpe"]
    drop_pct = (1 - reduced_sharpe / baseline_sharpe) * 100 if baseline_sharpe != 0 else 100
    passed = drop_pct < 50

    print(f"  Reduced Sharpe: {reduced_sharpe:.3f} (drop: {drop_pct:.1f}%)")
    print(f"  PASS (drop < 50%): {passed}")

    return {
        "test": "remove_top3_tickers",
        "removed_tickers": top3,
        "removed_pnl": top3_pnl,
        "baseline_sharpe": round(baseline_sharpe, 3),
        "reduced_sharpe": reduced_sharpe,
        "sharpe_drop_pct": round(drop_pct, 1),
        "n_trades": metrics["n_trades"],
        "total_pnl": metrics["total_pnl"],
        "passed": passed,
    }


def test_parameter_sensitivity(vix, rv, close):
    """Test 5: Grid over 4 params. >30% of combos should have Sharpe > 0.3."""
    print("\n[5/6] Parameter Sensitivity Grid...")

    gap_vals = [3, 5, 7, 10]
    dip_vals = [3, 5, 7, 10]
    rsi_vals = [35, 40, 45, 50]
    hold_vals = [5, 7, 10, 15]

    total_combos = len(gap_vals) * len(dip_vals) * len(rsi_vals) * len(hold_vals)
    print(f"  Total combinations: {total_combos}")

    results = []
    count_above = 0
    done = 0

    for gap in gap_vals:
        for dip in dip_vals:
            for rsi in rsi_vals:
                for hold in hold_vals:
                    trades = run_strategy(
                        vix, rv, close,
                        gap_thresh=gap, dip_thresh=dip,
                        rsi_thresh=rsi, hold_days=hold,
                    )
                    m = compute_metrics(trades)
                    above = m["sharpe"] > 0.3
                    if above:
                        count_above += 1
                    results.append({
                        "gap": gap, "dip": dip, "rsi": rsi, "hold": hold,
                        "sharpe": m["sharpe"], "n_trades": m["n_trades"],
                        "pnl": m["total_pnl"],
                    })
                    done += 1
                    if done % 64 == 0:
                        print(f"  Progress: {done}/{total_combos} ({count_above} above threshold so far)")

    pct_above = count_above / total_combos * 100
    passed = pct_above > 30

    # Best and worst combos
    results_sorted = sorted(results, key=lambda x: x["sharpe"], reverse=True)
    sharpes = [r["sharpe"] for r in results]

    print(f"  Combos with Sharpe > 0.3: {count_above}/{total_combos} ({pct_above:.1f}%)")
    print(f"  Best: gap={results_sorted[0]['gap']}, dip={results_sorted[0]['dip']}, "
          f"rsi={results_sorted[0]['rsi']}, hold={results_sorted[0]['hold']} → "
          f"Sharpe={results_sorted[0]['sharpe']:.3f}")
    print(f"  PASS (>30%): {passed}")

    return {
        "test": "parameter_sensitivity",
        "total_combos": total_combos,
        "combos_above_0_3": count_above,
        "pct_above_0_3": round(pct_above, 1),
        "mean_sharpe": round(float(np.mean(sharpes)), 3),
        "median_sharpe": round(float(np.median(sharpes)), 3),
        "best_combo": results_sorted[0],
        "worst_combo": results_sorted[-1],
        "top5": results_sorted[:5],
        "passed": passed,
    }


def test_cost_sensitivity(vix, rv, close):
    """Test 6: Test at 5, 10, 20, 50 bps. Find breakeven. PASS if > 20bps."""
    print("\n[6/6] Cost Sensitivity Test...")

    bps_levels = [0, 2, 5, 10, 20, 50]
    cost_results = []

    for bps in bps_levels:
        trades = run_strategy(vix, rv, close, slip_bps=bps)
        m = compute_metrics(trades)
        cost_results.append({
            "slip_bps": bps,
            "sharpe": m["sharpe"],
            "total_pnl": m["total_pnl"],
            "n_trades": m["n_trades"],
            "win_rate": m["win_rate"],
        })
        print(f"  {bps:3d} bps: Sharpe={m['sharpe']:.3f}, PnL=${m['total_pnl']:.0f}, WR={m['win_rate']:.1f}%")

    # Estimate breakeven bps (where PnL crosses zero) via interpolation
    pnls = [(r["slip_bps"], r["total_pnl"]) for r in cost_results]
    breakeven_bps = None
    for i in range(len(pnls) - 1):
        bps1, pnl1 = pnls[i]
        bps2, pnl2 = pnls[i + 1]
        if pnl1 >= 0 and pnl2 < 0:
            # Linear interpolation
            breakeven_bps = bps1 + (bps2 - bps1) * pnl1 / (pnl1 - pnl2)
            break

    if breakeven_bps is None:
        if all(r["total_pnl"] > 0 for r in cost_results):
            breakeven_bps = 999  # Profitable even at 50bps
        else:
            breakeven_bps = 0  # Never profitable

    passed = breakeven_bps > 20
    print(f"  Breakeven: ~{breakeven_bps:.0f} bps, PASS (>20bps): {passed}")

    return {
        "test": "cost_sensitivity",
        "levels": cost_results,
        "breakeven_bps": round(float(breakeven_bps), 1),
        "passed": passed,
    }


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("ADVERSARIAL VALIDATION: Vol Regime Variant F (IV-RV Gap Entry)")
    print("=" * 70)

    # Download data
    vix, spy, close = download_data()
    rv = compute_realized_vol(spy)

    print(f"\nData range: {close.index[0].date()} to {close.index[-1].date()}")
    print(f"Trading days: {len(close)}")
    print(f"Tickers: {len([c for c in close.columns if c in UNIVERSE])}")

    # Run baseline strategy
    print("\n--- Baseline Strategy ---")
    baseline_trades = run_strategy(vix, rv, close)
    baseline_metrics = compute_metrics(baseline_trades)
    baseline_sharpe = baseline_metrics["sharpe"]

    print(f"  Sharpe: {baseline_sharpe:.3f}")
    print(f"  Sortino: {baseline_metrics['sortino']:.3f}")
    print(f"  WR: {baseline_metrics['win_rate']:.1f}%")
    print(f"  PF: {baseline_metrics['profit_factor']:.2f}")
    print(f"  N trades: {baseline_metrics['n_trades']}")
    print(f"  Total PnL: ${baseline_metrics['total_pnl']:.2f}")
    print(f"  Max DD: {baseline_metrics['max_dd_pct']:.2f}%")

    # Run all 6 adversarial tests
    results = {"strategy": "vol_regime_variant_f", "timestamp": datetime.now().isoformat()}
    results["baseline"] = baseline_metrics

    tests = []

    # Test 1: Inverse Signal
    tests.append(test_inverse_signal(vix, rv, close, baseline_sharpe))

    # Test 2: Random Timing
    tests.append(test_random_timing(vix, rv, close, baseline_sharpe))

    # Test 3: Sub-Period Stability
    tests.append(test_sub_period_stability(vix, rv, close))

    # Test 4: Remove Top-3 Tickers
    tests.append(test_remove_top3(vix, rv, close, baseline_sharpe, baseline_trades))

    # Test 5: Parameter Sensitivity
    tests.append(test_parameter_sensitivity(vix, rv, close))

    # Test 6: Cost Sensitivity
    tests.append(test_cost_sensitivity(vix, rv, close))

    results["tests"] = tests

    # Summary
    n_passed = sum(1 for t in tests if t["passed"])
    n_total = len(tests)
    results["summary"] = {
        "tests_passed": n_passed,
        "tests_total": n_total,
        "pass_rate": round(n_passed / n_total * 100, 1),
        "overall_verdict": "PASS" if n_passed == n_total else (
            "MARGINAL" if n_passed >= 4 else "FAIL"
        ),
    }

    print("\n" + "=" * 70)
    print("ADVERSARIAL VALIDATION SUMMARY")
    print("=" * 70)
    for t in tests:
        status = "PASS" if t["passed"] else "FAIL"
        print(f"  [{status}] {t['test']}")
    print(f"\n  Overall: {n_passed}/{n_total} passed → {results['summary']['overall_verdict']}")
    print("=" * 70)

    # Save results
    out_path = Path("/home/jupiter/Lvl3Quant/data/vol_regime_f_adversarial.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    return results


if __name__ == "__main__":
    main()
