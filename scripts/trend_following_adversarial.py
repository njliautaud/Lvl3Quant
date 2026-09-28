#!/usr/bin/env python3
"""
Adversarial Validation: Trend Following Variant C (score >= 4)
==============================================================
5 adversarial tests to determine if this strategy has genuine edge
or is just "be long QQQ in bull markets."

Tests:
  1. INVERSE DIRECTION — short QQQ when score >= 4
  2. RANDOM TIMING — 100 random long/cash strategies at same frequency
  3. SUB-PERIOD STABILITY — Sharpe across 3 equal OOT sub-periods
  4. TOP-TRADE REMOVAL — remove best 5% of daily returns
  5. PARAMETER SENSITIVITY — score >= 3, 4, 5 thresholds

Uses identical data pipeline and cost model as trend_following_backtest.py.
"""

import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ── Config (same as original) ─────────────────────────────────────────
CAPITAL = 645.0
SLIPPAGE_PCT = 0.0002  # 0.02%
OOT_START = "2022-01-01"
OOT_END = "2026-07-29"
SEED = 42
N_RANDOM = 100

TICKERS = ["QQQ", "SPY"]
VIX_TICKER = "^VIX"


def download_data():
    """Download QQQ, SPY, VIX with buffer for 200-SMA."""
    all_tickers = TICKERS + [VIX_TICKER]
    start = (pd.Timestamp(OOT_START) - pd.DateOffset(days=300)).strftime("%Y-%m-%d")
    print(f"Downloading data for {all_tickers} from {start}...")
    data = yf.download(all_tickers, start=start, end=OOT_END, auto_adjust=True, progress=False)
    close = data["Close"].copy()
    close = close.dropna(subset=["SPY"])
    print(f"Data: {close.index[0].date()} to {close.index[-1].date()}, {len(close)} days")
    return close


def compute_signals(close):
    """Compute all 6 trend signals daily."""
    signals = pd.DataFrame(index=close.index, dtype=float)
    signals["qqq_200sma"] = (close["QQQ"] > close["QQQ"].rolling(200).mean()).astype(float)
    signals["qqq_50sma"] = (close["QQQ"] > close["QQQ"].rolling(50).mean()).astype(float)
    signals["qqq_20sma"] = (close["QQQ"] > close["QQQ"].rolling(20).mean()).astype(float)
    signals["qqq_5d_mom"] = (close["QQQ"].pct_change(5) > 0).astype(float)
    signals["spy_200sma"] = (close["SPY"] > close["SPY"].rolling(200).mean()).astype(float)
    vix = close[VIX_TICKER] if VIX_TICKER in close.columns else close.get("^VIX")
    signals["vix_calm"] = (vix < 25).astype(float) if vix is not None else 1.0
    signal_cols = ["qqq_200sma", "qqq_50sma", "qqq_20sma", "qqq_5d_mom", "spy_200sma", "vix_calm"]
    signals["score"] = signals[signal_cols].sum(axis=1)
    return signals


def calc_sharpe(daily_returns):
    """Annualized Sharpe from daily returns series."""
    dr = daily_returns.dropna()
    if len(dr) < 10 or dr.std() == 0:
        return 0.0
    return float(dr.mean() / dr.std() * np.sqrt(252))


def apply_slippage(returns, trades_mask):
    """Apply slippage cost on trade entry days."""
    adj = returns.copy()
    adj[trades_mask] -= SLIPPAGE_PCT
    return adj


def run_variant_c(signals, qqq_ret, threshold=4):
    """Run variant C (or any threshold). Returns daily returns series."""
    oot = signals.loc[OOT_START:]
    ret = qqq_ret.loc[oot.index]
    invested = oot["score"] >= threshold
    trades_mask = invested != invested.shift(1)
    daily_ret = pd.Series(0.0, index=oot.index)
    daily_ret[invested] = ret[invested]
    daily_ret = apply_slippage(daily_ret, trades_mask & invested)
    invested_pct = invested.mean()
    return daily_ret, invested_pct, invested


# ── TEST 1: INVERSE DIRECTION ─────────────────────────────────────────
def test_inverse_direction(signals, qqq_ret):
    """Short QQQ when score >= 4. If positive Sharpe, signal is decorative."""
    print("\n" + "=" * 60)
    print("TEST 1: INVERSE DIRECTION (short QQQ when score >= 4)")
    print("=" * 60)

    oot = signals.loc[OOT_START:]
    ret = qqq_ret.loc[oot.index]

    invested = oot["score"] >= 4
    trades_mask = invested != invested.shift(1)

    # SHORT: negate returns when invested
    daily_ret = pd.Series(0.0, index=oot.index)
    daily_ret[invested] = -ret[invested]  # SHORT
    daily_ret = apply_slippage(daily_ret, trades_mask & invested)

    inverse_sharpe = calc_sharpe(daily_ret)
    # Also get long sharpe for reference
    long_ret, _, _ = run_variant_c(signals, qqq_ret, threshold=4)
    long_sharpe = calc_sharpe(long_ret)

    # PASS if inverse Sharpe is NEGATIVE (meaning long is the right direction)
    passed = inverse_sharpe < 0
    verdict = "PASS" if passed else "FAIL"

    print(f"  Long Sharpe:    {long_sharpe:.3f}")
    print(f"  Inverse Sharpe: {inverse_sharpe:.3f}")
    print(f"  Verdict: {verdict}")
    if not passed:
        print("  >>> BOTH directions profitable = signal is just market timing noise")

    return {
        "test": "inverse_direction",
        "description": "Short QQQ when score >= 4. FAIL if inverse also positive.",
        "long_sharpe": round(long_sharpe, 3),
        "inverse_sharpe": round(inverse_sharpe, 3),
        "passed": passed,
        "verdict": verdict,
    }


# ── TEST 2: RANDOM TIMING ─────────────────────────────────────────────
def test_random_timing(signals, qqq_ret):
    """100 random strategies with same invested frequency. Compare Sharpe distribution."""
    print("\n" + "=" * 60)
    print("TEST 2: RANDOM TIMING (100 random at same 68.1% frequency)")
    print("=" * 60)

    long_ret, invested_pct, invested_mask = run_variant_c(signals, qqq_ret, threshold=4)
    actual_sharpe = calc_sharpe(long_ret)

    oot_idx = signals.loc[OOT_START:].index
    ret = qqq_ret.loc[oot_idx]
    n_days = len(oot_idx)
    n_invested = int(invested_pct * n_days)

    rng = np.random.RandomState(SEED)
    random_sharpes = []

    for _ in range(N_RANDOM):
        # Random selection of days to be invested (same count)
        mask = np.zeros(n_days, dtype=bool)
        chosen = rng.choice(n_days, size=n_invested, replace=False)
        mask[chosen] = True

        rand_ret = pd.Series(0.0, index=oot_idx)
        rand_ret[mask] = ret.values[mask]

        # Apply slippage on "entry" days (transitions from not-invested to invested)
        rand_invested = pd.Series(mask, index=oot_idx)
        rand_trades = rand_invested != rand_invested.shift(1)
        rand_ret = apply_slippage(rand_ret, rand_trades & rand_invested)

        random_sharpes.append(calc_sharpe(rand_ret))

    random_sharpes = np.array(random_sharpes)
    mean_random = float(np.mean(random_sharpes))
    std_random = float(np.std(random_sharpes))
    p_value = float(np.mean(random_sharpes >= actual_sharpe))

    # PASS if actual Sharpe significantly beats random (p < 0.05)
    passed = p_value < 0.05
    verdict = "PASS" if passed else "FAIL"

    print(f"  Actual Sharpe:       {actual_sharpe:.3f}")
    print(f"  Mean random Sharpe:  {mean_random:.3f} +/- {std_random:.3f}")
    print(f"  Max random Sharpe:   {max(random_sharpes):.3f}")
    print(f"  p-value:             {p_value:.4f}")
    print(f"  Verdict: {verdict}")
    if not passed:
        print("  >>> Random timing produces similar Sharpe = strategy is just long equity beta")

    return {
        "test": "random_timing",
        "description": "100 random long/cash strategies at same frequency. FAIL if p >= 0.05.",
        "actual_sharpe": round(actual_sharpe, 3),
        "mean_random_sharpe": round(mean_random, 3),
        "std_random_sharpe": round(std_random, 3),
        "max_random_sharpe": round(float(max(random_sharpes)), 3),
        "min_random_sharpe": round(float(min(random_sharpes)), 3),
        "p_value": round(p_value, 4),
        "passed": passed,
        "verdict": verdict,
    }


# ── TEST 3: SUB-PERIOD STABILITY ──────────────────────────────────────
def test_sub_period_stability(signals, qqq_ret):
    """Split OOT into 3 equal sub-periods. All must have positive Sharpe."""
    print("\n" + "=" * 60)
    print("TEST 3: SUB-PERIOD STABILITY (3 equal OOT sub-periods)")
    print("=" * 60)

    long_ret, _, _ = run_variant_c(signals, qqq_ret, threshold=4)
    oot_idx = long_ret.index

    # Split into 3 equal parts
    n = len(oot_idx)
    split1 = n // 3
    split2 = 2 * n // 3

    periods = [
        ("Period 1", oot_idx[:split1]),
        ("Period 2", oot_idx[split1:split2]),
        ("Period 3", oot_idx[split2:]),
    ]

    sub_results = []
    all_positive = True

    for name, idx in periods:
        sub_ret = long_ret.loc[idx]
        sharpe = calc_sharpe(sub_ret)
        start_date = str(idx[0].date())
        end_date = str(idx[-1].date())
        is_positive = sharpe > 0

        if not is_positive:
            all_positive = False

        sub_results.append({
            "period": name,
            "start": start_date,
            "end": end_date,
            "days": len(idx),
            "sharpe": round(sharpe, 3),
            "positive": is_positive,
        })
        print(f"  {name} ({start_date} to {end_date}, {len(idx)}d): Sharpe = {sharpe:.3f} {'OK' if is_positive else 'NEGATIVE'}")

    passed = all_positive
    verdict = "PASS" if passed else "FAIL"
    print(f"  Verdict: {verdict}")
    if not passed:
        print("  >>> Strategy is unstable across sub-periods")

    return {
        "test": "sub_period_stability",
        "description": "Split OOT into 3 equal sub-periods. FAIL if any has negative Sharpe.",
        "sub_periods": sub_results,
        "all_positive": all_positive,
        "passed": passed,
        "verdict": verdict,
    }


# ── TEST 4: TOP-TRADE REMOVAL ─────────────────────────────────────────
def test_top_trade_removal(signals, qqq_ret):
    """Remove best 5% of daily returns. Sharpe must stay above 0.5."""
    print("\n" + "=" * 60)
    print("TEST 4: TOP-TRADE REMOVAL (remove best 5% of daily returns)")
    print("=" * 60)

    long_ret, _, invested_mask = run_variant_c(signals, qqq_ret, threshold=4)

    original_sharpe = calc_sharpe(long_ret)

    # Get only the invested days' returns
    invested_returns = long_ret[long_ret != 0].copy()
    n_invested = len(invested_returns)
    n_remove = max(1, int(n_invested * 0.05))

    # Find top 5% of returns
    top_indices = invested_returns.nlargest(n_remove).index

    # Zero out the top 5%
    trimmed_ret = long_ret.copy()
    trimmed_ret.loc[top_indices] = 0.0

    trimmed_sharpe = calc_sharpe(trimmed_ret)

    passed = trimmed_sharpe > 0.5
    verdict = "PASS" if passed else "FAIL"

    print(f"  Original Sharpe:        {original_sharpe:.3f}")
    print(f"  After removing top 5%:  {trimmed_sharpe:.3f}")
    print(f"  Invested days:          {n_invested}")
    print(f"  Days removed:           {n_remove}")
    print(f"  Threshold:              0.5")
    print(f"  Verdict: {verdict}")
    if not passed:
        print("  >>> Strategy is top-trade dependent — a few lucky days drive all returns")

    return {
        "test": "top_trade_removal",
        "description": "Remove best 5% of daily returns. FAIL if Sharpe drops below 0.5.",
        "original_sharpe": round(original_sharpe, 3),
        "trimmed_sharpe": round(trimmed_sharpe, 3),
        "invested_days": n_invested,
        "days_removed": n_remove,
        "passed": passed,
        "verdict": verdict,
    }


# ── TEST 5: PARAMETER SENSITIVITY ─────────────────────────────────────
def test_parameter_sensitivity(signals, qqq_ret):
    """Test thresholds 3, 4, 5. At least 2/3 must have Sharpe > 0.5."""
    print("\n" + "=" * 60)
    print("TEST 5: PARAMETER SENSITIVITY (score >= 3, 4, 5)")
    print("=" * 60)

    thresholds = [3, 4, 5]
    threshold_results = []
    sharpe_above_05 = 0

    for thresh in thresholds:
        ret, invested_pct, _ = run_variant_c(signals, qqq_ret, threshold=thresh)
        sharpe = calc_sharpe(ret)
        above = sharpe > 0.5

        if above:
            sharpe_above_05 += 1

        threshold_results.append({
            "threshold": thresh,
            "sharpe": round(sharpe, 3),
            "time_invested_pct": round(invested_pct * 100, 1),
            "above_0.5": above,
        })
        print(f"  Score >= {thresh}: Sharpe = {sharpe:.3f}, Invested = {invested_pct * 100:.1f}% {'OK' if above else 'BELOW 0.5'}")

    passed = sharpe_above_05 >= 2
    verdict = "PASS" if passed else "FAIL"
    print(f"  {sharpe_above_05}/3 thresholds above 0.5")
    print(f"  Verdict: {verdict}")
    if not passed:
        print("  >>> Only one threshold works — likely overfitted to that parameter")

    return {
        "test": "parameter_sensitivity",
        "description": "Test score >= 3, 4, 5. FAIL if fewer than 2/3 have Sharpe > 0.5.",
        "thresholds": threshold_results,
        "count_above_05": sharpe_above_05,
        "passed": passed,
        "verdict": verdict,
    }


# ── MAIN ───────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("ADVERSARIAL VALIDATION: Trend Following Variant C (score >= 4)")
    print("=" * 70)

    close = download_data()
    signals = compute_signals(close)
    qqq_ret = close["QQQ"].pct_change()

    # Run all 5 tests
    results = []
    results.append(test_inverse_direction(signals, qqq_ret))
    results.append(test_random_timing(signals, qqq_ret))
    results.append(test_sub_period_stability(signals, qqq_ret))
    results.append(test_top_trade_removal(signals, qqq_ret))
    results.append(test_parameter_sensitivity(signals, qqq_ret))

    # ── Summary ────────────────────────────────────────────────────────
    n_passed = sum(1 for r in results if r["passed"])
    n_total = len(results)

    print("\n" + "=" * 70)
    print("ADVERSARIAL VALIDATION SUMMARY")
    print("=" * 70)
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  {r['test']:<25} {status}")
    print(f"\n  OVERALL: {n_passed}/{n_total}")

    # Overall assessment
    if n_passed <= 2:
        overall = "KILL — strategy likely has no genuine edge"
    elif n_passed <= 3:
        overall = "WEAK — strategy has marginal edge, probably not worth trading"
    elif n_passed == 4:
        overall = "MODERATE — strategy shows some genuine edge but has weaknesses"
    else:
        overall = "STRONG — strategy passes all adversarial tests"

    print(f"  ASSESSMENT: {overall}")

    # Key insight about the nature of the strategy
    inverse_result = results[0]
    random_result = results[1]
    if not inverse_result["passed"] or not random_result["passed"]:
        print("\n  CRITICAL FINDING: This strategy is likely just 'be long QQQ in bull markets'.")
        print("  The signals are ALL trend-following indicators on QQQ/SPY.")
        print("  They filter OUT bear markets, leaving you long during uptrends = beta exposure.")

    # Save results
    output = {
        "metadata": {
            "script": "trend_following_adversarial.py",
            "strategy": "Trend Following Variant C (score >= 4)",
            "run_date": datetime.now().isoformat(),
            "oot_start": OOT_START,
            "oot_end": OOT_END,
            "capital": CAPITAL,
            "slippage_pct": SLIPPAGE_PCT,
            "n_random_strategies": N_RANDOM,
        },
        "tests": {r["test"]: r for r in results},
        "summary": {
            "tests_passed": n_passed,
            "tests_total": n_total,
            "score": f"{n_passed}/{n_total}",
            "assessment": overall,
        },
    }

    output_path = Path("/home/jupiter/Lvl3Quant/data/trend_following_adversarial_results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    return output


if __name__ == "__main__":
    main()
