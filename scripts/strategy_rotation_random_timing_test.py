#!/usr/bin/env python3
"""
RANDOM TIMING TEST for Strategy Rotation Variant A

Null hypothesis: the regime-switching logic doesn't add value;
any random entry timing with the same instruments produces similar results.

Method:
- Keep same instruments (QQQ, SPY, VIX-related) and same sub-strategy mechanics
- Instead of using actual regime signals to decide WHEN to be in each strategy,
  assign random strategy labels per week (same block structure as original)
- Run 100 iterations
- Compare mean random Sharpe to original Sharpe (2.107)

PASS: mean random Sharpe < 0.5 * original Sharpe (< 1.054)
"""

import json, sys, warnings, datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

OOT_START = "2022-01-01"
OOT_END = "2026-07-25"
STARTING_CAPITAL = 645.0
N_RANDOM_ITERS = 100
ORIGINAL_SHARPE = 2.107
PASS_THRESHOLD = 0.5  # mean random must be < 0.5 * original

RESULTS_PATH = Path("/home/jupiter/Lvl3Quant/data/strategy_rotation_random_timing_test.json")


def fetch_data():
    tickers = ["SPY", "QQQ", "^VIX"]
    raw = yf.download(tickers, start="2021-01-01", end=OOT_END,
                      auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:
        close = raw[["Close"]].copy()
        close.columns = tickers

    if isinstance(close.columns, pd.MultiIndex):
        close.columns = close.columns.get_level_values(-1)

    df = pd.DataFrame(index=close.index)
    df["SPY"] = close["SPY"]
    df["QQQ"] = close["QQQ"]
    df["VIX"] = close["^VIX"] if "^VIX" in close.columns else close.get("^GSPC", np.nan)
    df = df.dropna()
    return df


def compute_regime_signals(df):
    df = df.copy()
    df["SMA200"] = df["SPY"].rolling(200).mean()
    df["SPY_ret5"] = df["SPY"].pct_change(5)
    df["VIX_chg5"] = df["VIX"].pct_change(5)
    df["bull"] = (df["SPY"] > df["SMA200"]).astype(int)
    df = df.dropna()
    return df


def _daily_returns(prices):
    return prices.pct_change().fillna(0)


def variant_A_labels(df):
    """Original Variant A: VIX>25->vix_fade, bull->earnings_momentum, else->contrarian"""
    labels = []
    for i in range(len(df)):
        if df["VIX"].iloc[i] > 25:
            labels.append("vix_fade")
        elif df["bull"].iloc[i]:
            labels.append("earnings_momentum")
        else:
            labels.append("contrarian")
    return pd.Series(labels, index=df.index)


def execute_rotation(df, rotation_labels):
    """Execute the rotation strategy given labels per day."""
    spy_ret = _daily_returns(df["SPY"])
    qqq_ret = _daily_returns(df["QQQ"])

    daily_ret = pd.Series(0.0, index=df.index)
    hold_remaining = 0
    prev_label = None

    for i in range(len(df)):
        label = rotation_labels.iloc[i]
        if label != prev_label:
            hold_remaining = 0
        prev_label = label

        if label == "earnings_momentum":
            daily_ret.iloc[i] = qqq_ret.iloc[i]
        elif label == "contrarian":
            if hold_remaining > 0:
                daily_ret.iloc[i] = spy_ret.iloc[i]
                hold_remaining -= 1
            elif df["SPY_ret5"].iloc[i] <= -0.03:
                daily_ret.iloc[i] = spy_ret.iloc[i]
                hold_remaining = 4
        elif label == "vix_fade":
            if df["VIX"].iloc[i] > 25 and df["VIX_chg5"].iloc[i] < 0:
                daily_ret.iloc[i] = spy_ret.iloc[i]
        elif label == "cash":
            pass

    return daily_ret


def compute_sharpe(daily_ret):
    """Annualized Sharpe from daily returns."""
    if daily_ret.std() < 1e-12:
        return 0.0
    total_ret = (1 + daily_ret).prod() - 1
    ann_ret = (1 + total_ret) ** (252 / len(daily_ret)) - 1
    vol = daily_ret.std() * np.sqrt(252)
    return ann_ret / vol if vol > 0 else 0.0


def build_week_boundaries(dates):
    """Get week start indices for block-shuffling."""
    week_starts = [0]
    for i in range(1, len(dates)):
        if (dates[i].isocalendar()[1] != dates[i-1].isocalendar()[1] or
                dates[i].year != dates[i-1].year):
            week_starts.append(i)
    week_starts.append(len(dates))
    return week_starts


def random_timing_labels(df, week_starts, rng):
    """Generate random strategy labels: assign a random strategy per week.
    Uses the same 3 strategies as Variant A."""
    strategies = ["earnings_momentum", "contrarian", "vix_fade"]
    n_weeks = len(week_starts) - 1
    labels = [""] * len(df)

    for w in range(n_weeks):
        strat = rng.choice(strategies)
        for i in range(week_starts[w], week_starts[w + 1]):
            labels[i] = strat

    return pd.Series(labels, index=df.index)


def main():
    print("=" * 70)
    print("RANDOM TIMING TEST — Strategy Rotation Variant A")
    print(f"Original Sharpe: {ORIGINAL_SHARPE}")
    print(f"Pass threshold: mean random Sharpe < {PASS_THRESHOLD * ORIGINAL_SHARPE:.3f}")
    print(f"Iterations: {N_RANDOM_ITERS}")
    print("=" * 70)

    # Fetch and prepare data
    print("\n[1/4] Fetching data...")
    df = fetch_data()
    df = compute_regime_signals(df)
    df_oot = df.loc[OOT_START:OOT_END].copy()
    print(f"  OOT: {len(df_oot)} days ({df_oot.index[0].date()} -> {df_oot.index[-1].date()})")

    # Verify original Sharpe
    print("\n[2/4] Verifying original strategy...")
    orig_labels = variant_A_labels(df_oot)
    orig_ret = execute_rotation(df_oot, orig_labels)
    orig_sharpe = compute_sharpe(orig_ret)
    print(f"  Reproduced original Sharpe: {orig_sharpe:.3f} (reference: {ORIGINAL_SHARPE})")

    # Build week boundaries
    week_starts = build_week_boundaries(df_oot.index)
    n_weeks = len(week_starts) - 1
    print(f"  {n_weeks} weekly blocks for randomization")

    # Allocation breakdown of original
    orig_alloc = orig_labels.value_counts(normalize=True)
    print(f"  Original allocation: {dict(orig_alloc.round(3))}")

    # Run random timing iterations
    print(f"\n[3/4] Running {N_RANDOM_ITERS} random timing iterations...")
    random_sharpes = []
    random_allocations = []

    for it in range(N_RANDOM_ITERS):
        rng = np.random.RandomState(it * 7 + 42)
        rand_labels = random_timing_labels(df_oot, week_starts, rng)
        rand_ret = execute_rotation(df_oot, rand_labels)
        rand_sharpe = compute_sharpe(rand_ret)
        random_sharpes.append(rand_sharpe)

        alloc = rand_labels.value_counts(normalize=True).to_dict()
        random_allocations.append(alloc)

        if (it + 1) % 25 == 0:
            print(f"  ... {it+1}/{N_RANDOM_ITERS} done (running mean Sharpe: {np.mean(random_sharpes):.3f})")

    random_sharpes = np.array(random_sharpes)

    # Results
    mean_random = float(np.mean(random_sharpes))
    std_random = float(np.std(random_sharpes))
    median_random = float(np.median(random_sharpes))
    p_value = float(np.mean(random_sharpes >= orig_sharpe))
    beats_half = float(np.mean(random_sharpes >= orig_sharpe * 0.5))
    threshold = PASS_THRESHOLD * ORIGINAL_SHARPE
    passed = mean_random < threshold

    print(f"\n[4/4] Results")
    print("=" * 70)
    print(f"  Original Sharpe:       {orig_sharpe:.3f}")
    print(f"  Mean random Sharpe:    {mean_random:.3f}")
    print(f"  Std random Sharpe:     {std_random:.3f}")
    print(f"  Median random Sharpe:  {median_random:.3f}")
    print(f"  Min random Sharpe:     {float(np.min(random_sharpes)):.3f}")
    print(f"  Max random Sharpe:     {float(np.max(random_sharpes)):.3f}")
    print(f"  p-value (random >= original): {p_value:.4f}")
    print(f"  Fraction random >= 50% of original: {beats_half:.2%}")
    print(f"  Pass threshold:        mean < {threshold:.3f}")
    print(f"  RESULT:                {'PASS' if passed else 'FAIL'}")
    print("=" * 70)

    # Percentile analysis
    percentiles = [5, 25, 50, 75, 95]
    pct_values = {f"p{p}": round(float(np.percentile(random_sharpes, p)), 3)
                  for p in percentiles}
    print(f"  Percentiles: {pct_values}")

    # Mean allocation across random runs
    mean_alloc = {}
    for key in ["earnings_momentum", "contrarian", "vix_fade"]:
        vals = [a.get(key, 0) for a in random_allocations]
        mean_alloc[key] = round(float(np.mean(vals)), 3)
    print(f"  Mean random allocation: {mean_alloc}")
    print(f"  (should be ~33%/33%/33% if truly random)")

    # Save results
    results = {
        "test": "random_timing",
        "test_description": "Keep same instruments and sub-strategy mechanics, randomize WHEN each strategy is active (random strategy per week). Tests whether regime-switching timing adds value vs random timing.",
        "run_timestamp": dt.datetime.now().isoformat(),
        "oot_period": f"{OOT_START} to {OOT_END}",
        "starting_capital": STARTING_CAPITAL,
        "n_iterations": N_RANDOM_ITERS,
        "n_weekly_blocks": n_weeks,
        "original_sharpe": round(orig_sharpe, 3),
        "original_sharpe_reference": ORIGINAL_SHARPE,
        "original_allocation": {k: round(v, 3) for k, v in orig_alloc.items()},
        "random_timing_results": {
            "mean_sharpe": round(mean_random, 3),
            "std_sharpe": round(std_random, 3),
            "median_sharpe": round(median_random, 3),
            "min_sharpe": round(float(np.min(random_sharpes)), 3),
            "max_sharpe": round(float(np.max(random_sharpes)), 3),
            "percentiles": pct_values,
            "mean_allocation": mean_alloc,
        },
        "statistical_tests": {
            "p_value_random_ge_original": round(p_value, 4),
            "fraction_random_ge_half_original": round(beats_half, 4),
        },
        "pass_criterion": f"mean random Sharpe < {PASS_THRESHOLD} * original Sharpe = {threshold:.3f}",
        "pass": passed,
        "verdict": "PASS - Regime timing adds significant value over random timing" if passed
                   else "FAIL - Random timing produces similar Sharpe, regime logic may not add value",
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
