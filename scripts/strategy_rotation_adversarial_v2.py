#!/usr/bin/env python3
"""
Strategy Rotation A — Adversarial Validation v2
================================================
5 rigorous tests using the EXACT original implementation.
Tests: Inverse Direction, Random Timing, Sub-Period Stability,
       Top-Trade Removal, Regime-Shuffle.
"""

import json, sys, warnings, datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ── CONFIG (identical to original) ────────────────────────────────────
OOT_START = "2022-01-01"
OOT_END   = "2026-07-25"
STARTING_CAPITAL = 645.0
RESULTS_PATH = Path("/home/jupiter/Lvl3Quant/data/strategy_rotation_adversarial_v2_results.json")


# ══════════════════════════════════════════════════════════════════════
# VERBATIM COPY from strategy_rotation_backtest.py
# ══════════════════════════════════════════════════════════════════════

def fetch_data():
    """Download SPY, QQQ, ^VIX with progress."""
    print("[DATA] Downloading market data via yfinance ...")
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
    print(f"    Got {len(df)} trading days ({df.index[0].date()} -> {df.index[-1].date()})")
    return df


def _rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    return 100 - 100 / (1 + rs)


def compute_regime_signals(df):
    """Add regime indicator columns."""
    df = df.copy()
    df["SMA200"]  = df["SPY"].rolling(200).mean()
    df["RSI14"]   = _rsi(df["SPY"], 14)
    df["SPY_ret20"] = df["SPY"].pct_change(20)
    df["SPY_ret60"] = df["SPY"].pct_change(60)
    df["SPY_ret5"]  = df["SPY"].pct_change(5)
    df["VIX_chg5"]  = df["VIX"].pct_change(5)
    df["bull"] = (df["SPY"] > df["SMA200"]).astype(int)
    df = df.dropna()
    return df


def _daily_returns(prices):
    return prices.pct_change().fillna(0)


def variant_A(df):
    """Simple Regime Switch: bull->QQQ, bear->Contrarian, VIX>25->VIX Fade."""
    n = len(df)
    strat_labels = []
    for i in range(n):
        if df["VIX"].iloc[i] > 25:
            strat_labels.append("vix_fade")
        elif df["bull"].iloc[i]:
            strat_labels.append("earnings_momentum")
        else:
            strat_labels.append("contrarian")
    return pd.Series(strat_labels, index=df.index)


def execute_rotation(df, rotation_labels):
    """Given strategy labels per day, produce equity curve."""
    spy_ret = _daily_returns(df["SPY"])
    qqq_ret = _daily_returns(df["QQQ"])

    daily_ret = pd.Series(0.0, index=df.index)
    n_trades = 0
    prev_label = None
    hold_remaining = 0

    for i in range(len(df)):
        label = rotation_labels.iloc[i]
        if label != prev_label:
            n_trades += 1
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
        elif label == "spy_hold":
            daily_ret.iloc[i] = spy_ret.iloc[i]
        elif label == "cash":
            pass

    equity = STARTING_CAPITAL * (1 + daily_ret).cumprod()
    return daily_ret, equity, n_trades


def compute_metrics(daily_ret, equity, n_trades, df, rotation_labels):
    trading_days = daily_ret[daily_ret != 0]
    total_ret = (equity.iloc[-1] / equity.iloc[0]) - 1
    ann_ret = (1 + total_ret) ** (252 / len(daily_ret)) - 1
    vol = daily_ret.std() * np.sqrt(252) if daily_ret.std() > 0 else 1e-9
    sharpe = ann_ret / vol if vol > 0 else 0
    downside = daily_ret[daily_ret < 0].std() * np.sqrt(252) if (daily_ret < 0).sum() > 0 else 1e-9
    sortino = ann_ret / downside
    wins = (trading_days > 0).sum()
    losses = (trading_days < 0).sum()
    wr = wins / (wins + losses) if (wins + losses) > 0 else 0
    avg_win = trading_days[trading_days > 0].mean() if wins > 0 else 0
    avg_loss = abs(trading_days[trading_days < 0].mean()) if losses > 0 else 1e-9
    pf = (avg_win * wins) / (avg_loss * losses) if (avg_loss * losses) > 0 else 999
    peak = equity.cummax()
    dd = (equity - peak) / peak
    mdd = dd.min()
    bull_mask = df["bull"] == 1
    bear_mask = df["bull"] == 0

    def _sharpe_subset(rets):
        if len(rets) < 5 or rets.std() == 0:
            return 0.0
        return (rets.mean() / rets.std()) * np.sqrt(252)

    sharpe_bull = _sharpe_subset(daily_ret[bull_mask])
    sharpe_bear = _sharpe_subset(daily_ret[bear_mask])
    regime_gap = abs(sharpe_bull - sharpe_bear) / max(abs(sharpe_bull), abs(sharpe_bear), 0.01)

    return {
        "total_return": round(total_ret, 4),
        "annual_return": round(ann_ret, 4),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "profit_factor": round(pf, 3),
        "win_rate": round(wr, 3),
        "max_drawdown": round(mdd, 4),
        "n_trades": n_trades,
        "n_active_days": int((trading_days != 0).sum()) if len(trading_days) > 0 else 0,
        "sharpe_bull": round(sharpe_bull, 3),
        "sharpe_bear": round(sharpe_bear, 3),
        "regime_gap": round(regime_gap, 3),
        "final_equity": round(equity.iloc[-1], 2),
    }


# ══════════════════════════════════════════════════════════════════════
# ADVERSARIAL TESTS
# ══════════════════════════════════════════════════════════════════════

def test_1_inverse_direction(df_oot):
    """Take OPPOSITE positions using same rotation timing."""
    print("\n" + "=" * 70)
    print("TEST 1: INVERSE DIRECTION")
    print("=" * 70)

    # Run original to get baseline
    rotation_labels = variant_A(df_oot)
    daily_ret_orig, equity_orig, n_trades_orig = execute_rotation(df_oot, rotation_labels)
    metrics_orig = compute_metrics(daily_ret_orig, equity_orig, n_trades_orig, df_oot, rotation_labels)

    # Inverse: negate daily returns (same timing, opposite direction)
    daily_ret_inv = -daily_ret_orig
    equity_inv = STARTING_CAPITAL * (1 + daily_ret_inv).cumprod()

    # Compute inverse metrics manually
    total_ret = (equity_inv.iloc[-1] / equity_inv.iloc[0]) - 1
    ann_ret = (1 + total_ret) ** (252 / len(daily_ret_inv)) - 1 if total_ret > -1 else -1.0
    vol = daily_ret_inv.std() * np.sqrt(252) if daily_ret_inv.std() > 0 else 1e-9
    sharpe_inv = ann_ret / vol if vol > 0 else 0

    passed = sharpe_inv < 0
    print(f"  Original Sharpe:  {metrics_orig['sharpe']}")
    print(f"  Inverse Sharpe:   {round(sharpe_inv, 3)}")
    print(f"  Inverse Final $:  ${equity_inv.iloc[-1]:.2f}")
    print(f"  PASS condition:   Inverse Sharpe < 0")
    print(f"  Result:           {'PASS' if passed else 'FAIL'}")

    return {
        "test": "inverse_direction",
        "original_sharpe": metrics_orig["sharpe"],
        "inverse_sharpe": round(sharpe_inv, 3),
        "inverse_final_equity": round(equity_inv.iloc[-1], 2),
        "passed": passed,
    }


def test_2_random_timing(df_oot):
    """Random rotation schedules vs original."""
    print("\n" + "=" * 70)
    print("TEST 2: RANDOM TIMING (100 random rotation schedules)")
    print("=" * 70)

    # Original baseline
    rotation_labels = variant_A(df_oot)
    daily_ret_orig, equity_orig, n_trades_orig = execute_rotation(df_oot, rotation_labels)
    metrics_orig = compute_metrics(daily_ret_orig, equity_orig, n_trades_orig, df_oot, rotation_labels)
    orig_sharpe = metrics_orig["sharpe"]

    # Generate 100 random rotation schedules
    rng = np.random.RandomState(42)
    strategies = ["earnings_momentum", "contrarian", "vix_fade"]
    random_sharpes = []

    # Build week boundaries
    dates = df_oot.index
    week_ids = []
    current_week = 0
    for i in range(len(dates)):
        if i > 0 and (dates[i].isocalendar()[1] != dates[i-1].isocalendar()[1] or
                       dates[i].year != dates[i-1].year):
            current_week += 1
        week_ids.append(current_week)
    n_weeks = current_week + 1

    for trial in range(100):
        # Random strategy per week
        week_strats = rng.choice(strategies, size=n_weeks)
        random_labels = pd.Series(
            [week_strats[week_ids[i]] for i in range(len(df_oot))],
            index=df_oot.index
        )
        daily_ret_r, equity_r, n_trades_r = execute_rotation(df_oot, random_labels)
        # Compute Sharpe
        total_ret = (equity_r.iloc[-1] / equity_r.iloc[0]) - 1
        ann_ret = (1 + total_ret) ** (252 / len(daily_ret_r)) - 1 if total_ret > -1 else -1.0
        vol = daily_ret_r.std() * np.sqrt(252) if daily_ret_r.std() > 0 else 1e-9
        sharpe_r = ann_ret / vol if vol > 0 else 0
        random_sharpes.append(sharpe_r)

    mean_random = np.mean(random_sharpes)
    std_random = np.std(random_sharpes)
    max_random = np.max(random_sharpes)
    min_random = np.min(random_sharpes)
    ratio = orig_sharpe / mean_random if mean_random > 0 else float('inf')

    passed = mean_random < 0.3 and (orig_sharpe > 3 * mean_random if mean_random > 0 else True)

    print(f"  Original Sharpe:     {orig_sharpe}")
    print(f"  Mean random Sharpe:  {mean_random:.3f} (std={std_random:.3f})")
    print(f"  Random range:        [{min_random:.3f}, {max_random:.3f}]")
    print(f"  Original / Mean:     {ratio:.1f}x")
    print(f"  PASS conditions:     mean_random < 0.3 AND original > 3x mean_random")
    print(f"  Result:              {'PASS' if passed else 'FAIL'}")

    return {
        "test": "random_timing",
        "original_sharpe": orig_sharpe,
        "mean_random_sharpe": round(mean_random, 3),
        "std_random_sharpe": round(std_random, 3),
        "max_random_sharpe": round(max_random, 3),
        "min_random_sharpe": round(min_random, 3),
        "ratio_orig_to_mean": round(ratio, 2),
        "passed": passed,
    }


def test_3_subperiod_stability(df_oot):
    """Split into 3 sub-periods and check stability."""
    print("\n" + "=" * 70)
    print("TEST 3: SUB-PERIOD STABILITY")
    print("=" * 70)

    # Define sub-periods
    periods = {
        "2022_H1": ("2022-01-01", "2022-06-30"),
        "2022H2_2023": ("2022-07-01", "2023-12-31"),
        "2024_2026": ("2024-01-01", "2026-07-25"),
    }

    # Full baseline
    rotation_labels_full = variant_A(df_oot)
    daily_ret_full, equity_full, _ = execute_rotation(df_oot, rotation_labels_full)
    total_return_full = (equity_full.iloc[-1] / equity_full.iloc[0]) - 1

    sub_results = {}
    all_positive = True
    max_contribution = 0.0

    for pname, (pstart, pend) in periods.items():
        mask = (df_oot.index >= pstart) & (df_oot.index <= pend)
        df_sub = df_oot.loc[mask].copy()

        if len(df_sub) < 10:
            print(f"  {pname}: Too few days ({len(df_sub)}), SKIP")
            sub_results[pname] = {"sharpe": 0, "days": len(df_sub), "skipped": True}
            continue

        rotation_labels_sub = variant_A(df_sub)
        daily_ret_sub, equity_sub, n_trades_sub = execute_rotation(df_sub, rotation_labels_sub)
        metrics_sub = compute_metrics(daily_ret_sub, equity_sub, n_trades_sub, df_sub, rotation_labels_sub)

        # Contribution to total return
        sub_total_ret = metrics_sub["total_return"]
        contribution = sub_total_ret / total_return_full if total_return_full != 0 else 0

        if metrics_sub["sharpe"] < 0:
            all_positive = False

        max_contribution = max(max_contribution, abs(contribution))

        sub_results[pname] = {
            "sharpe": metrics_sub["sharpe"],
            "sortino": metrics_sub["sortino"],
            "total_return": metrics_sub["total_return"],
            "return_contribution": round(contribution, 3),
            "n_trades": metrics_sub["n_trades"],
            "days": len(df_sub),
            "final_equity": metrics_sub["final_equity"],
        }

        print(f"  {pname} ({len(df_sub)} days): Sharpe={metrics_sub['sharpe']}, "
              f"Return={metrics_sub['total_return']:.1%}, "
              f"Contribution={contribution:.1%}")

    no_dominant = max_contribution < 0.70
    passed = all_positive and no_dominant

    print(f"\n  All sub-periods Sharpe >= 0: {all_positive}")
    print(f"  No sub-period > 70% of return: {no_dominant} (max={max_contribution:.1%})")
    print(f"  Result: {'PASS' if passed else 'FAIL'}")

    return {
        "test": "subperiod_stability",
        "sub_periods": sub_results,
        "all_sharpe_positive": all_positive,
        "max_contribution": round(max_contribution, 3),
        "no_dominant_period": no_dominant,
        "passed": passed,
    }


def test_4_top_trade_removal(df_oot):
    """Remove top 3 and top 5 best daily returns."""
    print("\n" + "=" * 70)
    print("TEST 4: TOP-TRADE REMOVAL")
    print("=" * 70)

    rotation_labels = variant_A(df_oot)
    daily_ret_orig, equity_orig, n_trades = execute_rotation(df_oot, rotation_labels)
    metrics_orig = compute_metrics(daily_ret_orig, equity_orig, n_trades, df_oot, rotation_labels)

    results_removal = {}
    for n_remove in [3, 5]:
        # Find top N daily returns and zero them out
        daily_ret_mod = daily_ret_orig.copy()
        top_indices = daily_ret_mod.nlargest(n_remove).index
        daily_ret_mod.loc[top_indices] = 0.0

        equity_mod = STARTING_CAPITAL * (1 + daily_ret_mod).cumprod()

        total_ret = (equity_mod.iloc[-1] / equity_mod.iloc[0]) - 1
        ann_ret = (1 + total_ret) ** (252 / len(daily_ret_mod)) - 1 if total_ret > -1 else -1.0
        vol = daily_ret_mod.std() * np.sqrt(252) if daily_ret_mod.std() > 0 else 1e-9
        sharpe_mod = ann_ret / vol if vol > 0 else 0

        results_removal[f"remove_top_{n_remove}"] = {
            "sharpe": round(sharpe_mod, 3),
            "final_equity": round(equity_mod.iloc[-1], 2),
            "total_return": round(total_ret, 4),
            "removed_dates": [str(d.date()) for d in top_indices],
            "removed_returns": [round(daily_ret_orig.loc[d], 4) for d in top_indices],
        }

        print(f"  Remove top {n_remove}: Sharpe={sharpe_mod:.3f}, "
              f"Final=${equity_mod.iloc[-1]:.2f}")
        for d in top_indices:
            print(f"    Removed {d.date()}: +{daily_ret_orig.loc[d]:.2%}")

    sharpe_after_5 = results_removal["remove_top_5"]["sharpe"]
    passed = sharpe_after_5 > 0.5

    print(f"\n  Original Sharpe:         {metrics_orig['sharpe']}")
    print(f"  Sharpe after removing 5: {sharpe_after_5}")
    print(f"  PASS condition:          Sharpe > 0.5 after removing top 5")
    print(f"  Result:                  {'PASS' if passed else 'FAIL'}")

    return {
        "test": "top_trade_removal",
        "original_sharpe": metrics_orig["sharpe"],
        "removal_results": results_removal,
        "sharpe_after_top5_removal": sharpe_after_5,
        "passed": passed,
    }


def test_5_regime_shuffle(df_oot):
    """Shuffle bull/bear regime labels and re-run rotation 100 times."""
    print("\n" + "=" * 70)
    print("TEST 5: REGIME-SHUFFLE (100 iterations)")
    print("=" * 70)

    # Original
    rotation_labels = variant_A(df_oot)
    daily_ret_orig, equity_orig, n_trades = execute_rotation(df_oot, rotation_labels)
    metrics_orig = compute_metrics(daily_ret_orig, equity_orig, n_trades, df_oot, rotation_labels)
    orig_sharpe = metrics_orig["sharpe"]

    rng = np.random.RandomState(99)
    shuffled_sharpes = []

    # Get the original bull/bear proportion
    bull_days = (df_oot["bull"] == 1).sum()
    bear_days = (df_oot["bull"] == 0).sum()
    total_days = len(df_oot)
    bull_frac = bull_days / total_days

    print(f"  Original bull/bear split: {bull_days}/{bear_days} ({bull_frac:.1%} bull)")

    for trial in range(100):
        # Create shuffled bull/bear column (same proportion)
        df_shuffled = df_oot.copy()
        shuffled_bull = np.zeros(total_days, dtype=int)
        bull_indices = rng.choice(total_days, size=bull_days, replace=False)
        shuffled_bull[bull_indices] = 1
        df_shuffled["bull"] = shuffled_bull

        # Also need to recompute VIX_chg5 and SPY_ret5 — NO, keep those the same.
        # Only shuffle the bull/bear label. VIX stays real.
        # Re-run variant_A with shuffled bull labels
        # variant_A uses: VIX > 25, then bull flag — so we need to re-run with shuffled bull
        rotation_labels_s = variant_A(df_shuffled)
        daily_ret_s, equity_s, _ = execute_rotation(df_shuffled, rotation_labels_s)

        total_ret = (equity_s.iloc[-1] / equity_s.iloc[0]) - 1
        ann_ret = (1 + total_ret) ** (252 / len(daily_ret_s)) - 1 if total_ret > -1 else -1.0
        vol = daily_ret_s.std() * np.sqrt(252) if daily_ret_s.std() > 0 else 1e-9
        sharpe_s = ann_ret / vol if vol > 0 else 0
        shuffled_sharpes.append(sharpe_s)

    mean_shuffled = np.mean(shuffled_sharpes)
    std_shuffled = np.std(shuffled_sharpes)
    max_shuffled = np.max(shuffled_sharpes)

    passed = mean_shuffled < 0.5

    print(f"  Original Sharpe:         {orig_sharpe}")
    print(f"  Mean shuffled Sharpe:    {mean_shuffled:.3f} (std={std_shuffled:.3f})")
    print(f"  Shuffled range:          [{np.min(shuffled_sharpes):.3f}, {max_shuffled:.3f}]")
    print(f"  PASS condition:          mean shuffled Sharpe < 0.5")
    print(f"  Result:                  {'PASS' if passed else 'FAIL'}")

    return {
        "test": "regime_shuffle",
        "original_sharpe": orig_sharpe,
        "mean_shuffled_sharpe": round(mean_shuffled, 3),
        "std_shuffled_sharpe": round(std_shuffled, 3),
        "max_shuffled_sharpe": round(max_shuffled, 3),
        "min_shuffled_sharpe": round(np.min(shuffled_sharpes), 3),
        "bull_bear_split": f"{bull_days}/{bear_days}",
        "passed": passed,
    }


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("STRATEGY ROTATION A — ADVERSARIAL VALIDATION v2")
    print(f"OOT: {OOT_START} -> {OOT_END}  |  Capital: ${STARTING_CAPITAL}")
    print("=" * 70)

    # Fetch and prepare data
    df = fetch_data()
    df = compute_regime_signals(df)
    df_oot = df.loc[OOT_START:OOT_END].copy()
    print(f"OOT period: {len(df_oot)} days ({df_oot.index[0].date()} -> {df_oot.index[-1].date()})")

    # Baseline reproduction check
    print("\n--- BASELINE REPRODUCTION CHECK ---")
    rotation_labels = variant_A(df_oot)
    daily_ret, equity, n_trades = execute_rotation(df_oot, rotation_labels)
    metrics = compute_metrics(daily_ret, equity, n_trades, df_oot, rotation_labels)
    print(f"  Sharpe: {metrics['sharpe']} (expected ~2.13)")
    print(f"  Sortino: {metrics['sortino']} (expected ~2.99)")
    print(f"  WR: {metrics['win_rate']:.1%} (expected ~57.1%)")
    print(f"  Trades: {metrics['n_trades']} (expected ~69)")
    print(f"  MDD: {metrics['max_drawdown']:.1%} (expected ~-10.8%)")
    print(f"  Final equity: ${metrics['final_equity']} (expected ~$2,721)")

    # Run all 5 tests
    all_results = {
        "meta": {
            "run_timestamp": dt.datetime.now().isoformat(),
            "oot_start": OOT_START,
            "oot_end": OOT_END,
            "starting_capital": STARTING_CAPITAL,
        },
        "baseline_reproduction": metrics,
        "tests": {},
    }

    test_funcs = [
        test_1_inverse_direction,
        test_2_random_timing,
        test_3_subperiod_stability,
        test_4_top_trade_removal,
        test_5_regime_shuffle,
    ]

    for test_func in test_funcs:
        result = test_func(df_oot)
        all_results["tests"][result["test"]] = result

    # Summary
    print("\n" + "=" * 70)
    print("ADVERSARIAL VALIDATION v2 — SUMMARY")
    print("=" * 70)
    n_passed = 0
    n_total = len(all_results["tests"])
    for tname, tresult in all_results["tests"].items():
        status = "PASS" if tresult["passed"] else "FAIL"
        n_passed += 1 if tresult["passed"] else 0
        print(f"  {tname:30s} {status}")

    print(f"\n  OVERALL: {n_passed}/{n_total} tests passed")
    if n_passed == n_total:
        print("  VERDICT: Strategy Rotation A SURVIVES adversarial validation")
    elif n_passed >= 3:
        print("  VERDICT: Strategy Rotation A PARTIALLY survives (review failed tests)")
    else:
        print("  VERDICT: Strategy Rotation A FAILS adversarial validation")

    all_results["summary"] = {
        "tests_passed": n_passed,
        "tests_total": n_total,
        "verdict": "SURVIVES" if n_passed == n_total else
                   "PARTIAL" if n_passed >= 3 else "FAILS",
    }

    # Save results
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
