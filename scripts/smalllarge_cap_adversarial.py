#!/usr/bin/env python3
"""
Adversarial Validation: Small-Large Cap Spread Strategy
========================================================
Track IWM/SPY ratio on 20d momentum basis.
  - IWM outperforming SPY (ratio 20d momentum > 0) -> long IWM
  - SPY outperforming (ratio declining) -> long GLD
  - Rebalance every 10 days

6 adversarial checks:
  1. INVERSE DIRECTION
  2. RANDOM TIMING (1000 iterations)
  3. LOOK-AHEAD BIAS (1-day lag)
  4. COST SENSITIVITY (0.05% - 0.20%)
  5. SUB-PERIOD STABILITY (4 equal periods)
  6. PARAMETER SENSITIVITY (lookback x rebalance x threshold grid)
"""

import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────────
CAPITAL = 645.0
SLIPPAGE_PCT = 0.0002  # 0.02%
COMMISSION = 0.0
OOT_START = "2022-01-01"
OOT_END = "2026-07-29"
SEED = 42
N_RANDOM = 1000

TICKERS = ["SPY", "QQQ", "IWM", "GLD"]
VIX_TICKER = "^VIX"

LOOKBACK = 20
REBALANCE_DAYS = 10
MOMENTUM_THRESHOLD = 0.0

OUT_PATH = Path("/home/jupiter/Lvl3Quant/data/smalllarge_cap_adversarial_results.json")


def download_data():
    """Download SPY, QQQ, IWM, GLD, VIX with buffer for lookback."""
    all_tickers = TICKERS + [VIX_TICKER]
    start = (pd.Timestamp(OOT_START) - pd.DateOffset(days=120)).strftime("%Y-%m-%d")
    print(f"Downloading {all_tickers} from {start} to {OOT_END}...")
    data = yf.download(all_tickers, start=start, end=OOT_END, auto_adjust=True, progress=False)
    close = data["Close"].copy()
    close = close.dropna(subset=["SPY", "IWM", "GLD"])
    print(f"Data: {close.index[0].date()} to {close.index[-1].date()}, {len(close)} days")
    return close


def compute_signal(close, lookback=LOOKBACK, threshold=MOMENTUM_THRESHOLD, lag=0):
    """
    Compute IWM/SPY ratio momentum signal.
    Returns Series: 1 = long IWM, 0 = long GLD.
    """
    ratio = close["IWM"] / close["SPY"]
    mom = ratio.pct_change(lookback)
    if lag > 0:
        mom = mom.shift(lag)
    signal = (mom > threshold).astype(float)
    return signal


def run_strategy(close, signal, rebalance_days=REBALANCE_DAYS, slippage_pct=SLIPPAGE_PCT):
    """
    Run the small-large cap spread strategy.
    signal=1 -> long IWM, signal=0 -> long GLD.
    Rebalance every rebalance_days.
    Returns daily returns Series.
    """
    oot_mask = close.index >= OOT_START
    close_oot = close[oot_mask].copy()
    signal_oot = signal[oot_mask].copy()

    iwm_ret = close_oot["IWM"].pct_change()
    gld_ret = close_oot["GLD"].pct_change()

    # Rebalance signal: only change position every rebalance_days
    rebal_signal = signal_oot.copy()
    last_rebal = 0
    current_pos = np.nan
    for i in range(len(rebal_signal)):
        if np.isnan(signal_oot.iloc[i]):
            rebal_signal.iloc[i] = np.nan
            continue
        if np.isnan(current_pos) or (i - last_rebal >= rebalance_days):
            current_pos = signal_oot.iloc[i]
            last_rebal = i
        rebal_signal.iloc[i] = current_pos

    # Daily returns based on position
    daily_ret = pd.Series(0.0, index=close_oot.index)
    for i in range(1, len(close_oot)):
        pos = rebal_signal.iloc[i - 1]  # position from prior day
        if np.isnan(pos):
            continue
        if pos == 1.0:
            daily_ret.iloc[i] = iwm_ret.iloc[i]
        else:
            daily_ret.iloc[i] = gld_ret.iloc[i]

    # Apply slippage on rebalance days (position changes)
    pos_changes = rebal_signal.diff().fillna(0) != 0
    daily_ret[pos_changes] -= slippage_pct

    return daily_ret.dropna()


def calc_sharpe(daily_returns):
    """Annualized Sharpe from daily returns."""
    dr = daily_returns.dropna()
    if len(dr) < 10 or dr.std() == 0:
        return 0.0
    return float(dr.mean() / dr.std() * np.sqrt(252))


def calc_sortino(daily_returns):
    """Annualized Sortino."""
    dr = daily_returns.dropna()
    if len(dr) < 10:
        return 0.0
    downside = dr[dr < 0].std()
    if downside == 0:
        return 0.0
    return float(dr.mean() / downside * np.sqrt(252))


def calc_max_dd(daily_returns):
    """Max drawdown from daily returns."""
    cum = (1 + daily_returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return float(dd.min())


def calc_total_return(daily_returns):
    """Total return."""
    return float((1 + daily_returns).prod() - 1)


# ── Test 1: Inverse Direction ──────────────────────────────────────────
def test_inverse_direction(close, baseline_sharpe):
    print("\n[1/6] INVERSE DIRECTION TEST")
    signal = compute_signal(close)
    inv_signal = 1.0 - signal  # flip: long GLD when IWM outperforms, long IWM when SPY outperforms
    inv_ret = run_strategy(close, inv_signal)
    inv_sharpe = calc_sharpe(inv_ret)

    passed = (inv_sharpe < 0) and (baseline_sharpe > 2 * abs(inv_sharpe) if inv_sharpe != 0 else True)
    verdict = "PASS" if passed else "FAIL"
    print(f"  Baseline Sharpe: {baseline_sharpe:.3f}, Inverse Sharpe: {inv_sharpe:.3f} -> {verdict}")

    return {
        "test": "inverse_direction",
        "description": "Long GLD when IWM outperforming, long IWM when SPY outperforming. PASS if inverse Sharpe < 0 AND baseline > 2x inverse.",
        "baseline_sharpe": round(baseline_sharpe, 3),
        "inverse_sharpe": round(inv_sharpe, 3),
        "passed": passed,
        "verdict": verdict,
    }


# ── Test 2: Random Timing ──────────────────────────────────────────────
def test_random_timing(close, baseline_sharpe):
    print("\n[2/6] RANDOM TIMING TEST (1000 iterations)")
    rng = np.random.RandomState(SEED)

    # Get actual signal to measure trade frequency
    real_signal = compute_signal(close)
    oot_mask = close.index >= OOT_START
    real_oot = real_signal[oot_mask].dropna()
    freq = real_oot.mean()  # fraction of days in IWM

    random_sharpes = []
    for i in range(N_RANDOM):
        rand_signal = pd.Series(
            (rng.random(len(real_signal)) < freq).astype(float),
            index=real_signal.index,
        )
        rand_ret = run_strategy(close, rand_signal)
        random_sharpes.append(calc_sharpe(rand_ret))

    random_sharpes = np.array(random_sharpes)
    pct_rank = float((random_sharpes >= baseline_sharpe).mean())
    p_value = pct_rank
    passed = pct_rank <= 0.10  # 90th percentile
    verdict = "PASS" if passed else "FAIL"

    print(f"  Baseline Sharpe: {baseline_sharpe:.3f}")
    print(f"  Random: mean={random_sharpes.mean():.3f}, std={random_sharpes.std():.3f}, max={random_sharpes.max():.3f}")
    print(f"  Percentile rank: {(1 - pct_rank)*100:.1f}th -> {verdict}")

    return {
        "test": "random_timing",
        "description": f"{N_RANDOM} random IWM/GLD strategies at same frequency. PASS if baseline >= 90th percentile.",
        "actual_sharpe": round(baseline_sharpe, 3),
        "mean_random_sharpe": round(float(random_sharpes.mean()), 3),
        "std_random_sharpe": round(float(random_sharpes.std()), 3),
        "max_random_sharpe": round(float(random_sharpes.max()), 3),
        "min_random_sharpe": round(float(random_sharpes.min()), 3),
        "percentile_rank": round((1 - pct_rank) * 100, 1),
        "p_value": round(p_value, 4),
        "passed": passed,
        "verdict": verdict,
    }


# ── Test 3: Look-Ahead Bias ────────────────────────────────────────────
def test_lookahead_bias(close, baseline_sharpe):
    print("\n[3/6] LOOK-AHEAD BIAS TEST")
    lagged_signal = compute_signal(close, lag=1)
    lagged_ret = run_strategy(close, lagged_signal)
    lagged_sharpe = calc_sharpe(lagged_ret)

    within_30pct = abs(lagged_sharpe - baseline_sharpe) / max(abs(baseline_sharpe), 1e-9) <= 0.30
    passed = (lagged_sharpe > 0.5) and within_30pct
    verdict = "PASS" if passed else "FAIL"

    print(f"  Baseline Sharpe: {baseline_sharpe:.3f}, Lagged Sharpe: {lagged_sharpe:.3f}")
    print(f"  Within 30%: {within_30pct}, Lagged > 0.5: {lagged_sharpe > 0.5} -> {verdict}")

    return {
        "test": "lookahead_bias",
        "description": "1-day lag on all signals. PASS if lagged Sharpe > 0.5 AND within 30% of baseline.",
        "baseline_sharpe": round(baseline_sharpe, 3),
        "lagged_sharpe": round(lagged_sharpe, 3),
        "sharpe_degradation_pct": round(abs(lagged_sharpe - baseline_sharpe) / max(abs(baseline_sharpe), 1e-9) * 100, 1),
        "passed": passed,
        "verdict": verdict,
    }


# ── Test 4: Cost Sensitivity ───────────────────────────────────────────
def test_cost_sensitivity(close):
    print("\n[4/6] COST SENSITIVITY TEST")
    signal = compute_signal(close)
    cost_levels = [0.0005, 0.0010, 0.0015, 0.0020]
    results_list = []

    for cost in cost_levels:
        ret = run_strategy(close, signal, slippage_pct=cost)
        s = calc_sharpe(ret)
        results_list.append({
            "slippage_pct": cost * 100,
            "sharpe": round(s, 3),
            "positive": s > 0,
        })
        print(f"  Slippage {cost*100:.2f}%: Sharpe={s:.3f}")

    # Must survive at 0.10% (second level)
    survive_010 = results_list[1]["sharpe"] > 0
    verdict = "PASS" if survive_010 else "FAIL"
    print(f"  Survive at 0.10%: {survive_010} -> {verdict}")

    return {
        "test": "cost_sensitivity",
        "description": "Test at 0.05%, 0.10%, 0.15%, 0.20% slippage. PASS if survives at 0.10%.",
        "cost_levels": results_list,
        "survives_at_010pct": survive_010,
        "passed": survive_010,
        "verdict": verdict,
    }


# ── Test 5: Sub-Period Stability ────────────────────────────────────────
def test_sub_period_stability(close):
    print("\n[5/6] SUB-PERIOD STABILITY TEST")
    signal = compute_signal(close)
    ret = run_strategy(close, signal)

    n = len(ret)
    chunk = n // 4
    periods = []

    for i in range(4):
        start_idx = i * chunk
        end_idx = (i + 1) * chunk if i < 3 else n
        sub_ret = ret.iloc[start_idx:end_idx]
        s = calc_sharpe(sub_ret)
        periods.append({
            "period": f"Period {i+1}",
            "start": str(sub_ret.index[0].date()),
            "end": str(sub_ret.index[-1].date()),
            "days": len(sub_ret),
            "sharpe": round(s, 3),
            "positive": s > 0,
        })
        print(f"  Period {i+1}: {sub_ret.index[0].date()} to {sub_ret.index[-1].date()}, Sharpe={s:.3f}")

    n_positive = sum(p["positive"] for p in periods)
    passed = n_positive >= 3
    verdict = "PASS" if passed else "FAIL"
    print(f"  {n_positive}/4 periods positive -> {verdict}")

    return {
        "test": "sub_period_stability",
        "description": "Split OOT into 4 equal sub-periods. PASS if >= 3 have Sharpe > 0.",
        "sub_periods": periods,
        "n_positive": n_positive,
        "passed": passed,
        "verdict": verdict,
    }


# ── Test 6: Parameter Sensitivity ──────────────────────────────────────
def test_parameter_sensitivity(close):
    print("\n[6/6] PARAMETER SENSITIVITY TEST")
    lookbacks = [10, 15, 20, 30, 40]
    rebalance_periods = [5, 10, 15, 20]
    thresholds = [0.0, 0.005, 0.01, 0.015]

    total = len(lookbacks) * len(rebalance_periods) * len(thresholds)
    print(f"  Grid: {len(lookbacks)} lookbacks x {len(rebalance_periods)} rebalance x {len(thresholds)} thresholds = {total} combos")

    results_grid = []
    sharpes = []

    for lb in lookbacks:
        for rb in rebalance_periods:
            for th in thresholds:
                sig = compute_signal(close, lookback=lb, threshold=th)
                ret = run_strategy(close, sig, rebalance_days=rb)
                s = calc_sharpe(ret)
                sharpes.append(s)
                results_grid.append({
                    "lookback": lb,
                    "rebalance": rb,
                    "threshold": th,
                    "sharpe": round(s, 3),
                })

    sharpes = np.array(sharpes)
    n_above_03 = int((sharpes > 0.3).sum())
    pct_above_03 = n_above_03 / total
    passed = pct_above_03 >= 0.30
    verdict = "PASS" if passed else "FAIL"

    print(f"  {n_above_03}/{total} ({pct_above_03*100:.1f}%) have Sharpe > 0.3 -> {verdict}")
    print(f"  Sharpe range: [{sharpes.min():.3f}, {sharpes.max():.3f}], median={np.median(sharpes):.3f}")

    # Find best and worst
    best_idx = int(sharpes.argmax())
    worst_idx = int(sharpes.argmin())

    return {
        "test": "parameter_sensitivity",
        "description": f"Grid of {total} param combos. PASS if >= 30% have Sharpe > 0.3.",
        "total_combos": total,
        "n_sharpe_above_03": n_above_03,
        "pct_sharpe_above_03": round(pct_above_03 * 100, 1),
        "sharpe_median": round(float(np.median(sharpes)), 3),
        "sharpe_mean": round(float(sharpes.mean()), 3),
        "sharpe_min": round(float(sharpes.min()), 3),
        "sharpe_max": round(float(sharpes.max()), 3),
        "best_params": results_grid[best_idx],
        "worst_params": results_grid[worst_idx],
        "passed": passed,
        "verdict": verdict,
    }


# ── Main ────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("ADVERSARIAL VALIDATION: Small-Large Cap Spread Strategy")
    print("=" * 70)

    close = download_data()

    # Baseline strategy
    print("\n--- BASELINE STRATEGY ---")
    signal = compute_signal(close)
    baseline_ret = run_strategy(close, signal)
    baseline_sharpe = calc_sharpe(baseline_ret)
    baseline_sortino = calc_sortino(baseline_ret)
    baseline_mdd = calc_max_dd(baseline_ret)
    baseline_total = calc_total_return(baseline_ret)

    print(f"  Sharpe: {baseline_sharpe:.3f}")
    print(f"  Sortino: {baseline_sortino:.3f}")
    print(f"  Max DD: {baseline_mdd:.1%}")
    print(f"  Total Return: {baseline_total:.1%}")

    # Run all 6 tests
    t1 = test_inverse_direction(close, baseline_sharpe)
    t2 = test_random_timing(close, baseline_sharpe)
    t3 = test_lookahead_bias(close, baseline_sharpe)
    t4 = test_cost_sensitivity(close)
    t5 = test_sub_period_stability(close)
    t6 = test_parameter_sensitivity(close)

    tests = {
        "inverse_direction": t1,
        "random_timing": t2,
        "lookahead_bias": t3,
        "cost_sensitivity": t4,
        "sub_period_stability": t5,
        "parameter_sensitivity": t6,
    }

    n_passed = sum(1 for t in tests.values() if t["passed"])
    n_total = len(tests)

    result = {
        "metadata": {
            "script": "smalllarge_cap_adversarial.py",
            "strategy": "Small-Large Cap Spread (IWM/SPY ratio momentum -> IWM or GLD)",
            "run_date": datetime.now().isoformat(),
            "oot_start": OOT_START,
            "oot_end": OOT_END,
            "capital": CAPITAL,
            "slippage_pct": SLIPPAGE_PCT,
            "commission": COMMISSION,
            "lookback": LOOKBACK,
            "rebalance_days": REBALANCE_DAYS,
            "momentum_threshold": MOMENTUM_THRESHOLD,
            "n_random_strategies": N_RANDOM,
        },
        "baseline": {
            "sharpe": round(baseline_sharpe, 3),
            "sortino": round(baseline_sortino, 3),
            "max_drawdown": round(baseline_mdd, 4),
            "total_return": round(baseline_total, 4),
        },
        "tests": tests,
        "summary": {
            "tests_passed": n_passed,
            "tests_total": n_total,
            "pass_rate": f"{n_passed}/{n_total}",
            "overall_verdict": "PASS" if n_passed >= 5 else "FAIL",
            "failed_tests": [name for name, t in tests.items() if not t["passed"]],
        },
    }

    print("\n" + "=" * 70)
    print(f"OVERALL: {n_passed}/{n_total} tests passed -> {result['summary']['overall_verdict']}")
    if result["summary"]["failed_tests"]:
        print(f"Failed: {', '.join(result['summary']['failed_tests'])}")
    print("=" * 70)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
