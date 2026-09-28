#!/usr/bin/env python3
"""
bps_ga_adversarial.py — Adversarial Validation of BPS GA Strategy
===================================================================
The BPS GA (Bull Put Spread, Genetic Algorithm universe) reports:
  - Sharpe ~2.59, R1 regime gap 0.40

Gap 0.40 is near the R1 boundary of 0.50. This script determines
whether the R1 pass is robust or fragile via three tests:

1. PERMUTATION TEST (200 iters): Shuffle daily returns, recompute
   Sharpe each time. p-value = fraction of shuffled Sharpe >= actual.
   PASS if p < 0.05.

2. BOOTSTRAP CI (1000 iters): Resample daily returns with replacement,
   compute Sharpe and R1 gap each time. Report: median Sharpe,
   5th/95th CI, % of bootstraps passing R1 (gap <= 0.50).

3. PARAMETER STABILITY: Perturb the GA universe by dropping 1-3 tickers
   and/or swapping in random replacements. Check if nearby universes
   still pass R1 and maintain Sharpe > 1.0.

Usage:
    python3 /home/jupiter/Lvl3Quant/research/bps_ga_adversarial.py
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path("/home/jupiter/Lvl3Quant")
OUTPUT = ROOT / "output" / "bps_ga_adversarial"
OUTPUT.mkdir(parents=True, exist_ok=True)

TRADING_DAYS = 252
RF_DAILY = 0.04 / TRADING_DAYS
STARTING_CAPITAL = 100_000

# GA-evolved 20-ticker universe
GA_UNIVERSE = [
    "NFLX", "NVDA", "PLTR", "WMT", "GM", "PFE", "LLY", "MCD",
    "CL", "T", "SMCI", "OXY", "HOOD", "JNJ", "F", "TGT",
    "VZ", "TSLA", "PANW", "TMUS",
]

# Full 87-ticker universe (for parameter stability — swap candidates)
ALL_TICKERS_87 = None  # loaded dynamically


def load_data():
    """Load BPS trades (full stack, 5% BA cost) and SPY for regime."""
    global ALL_TICKERS_87

    trades = pd.read_parquet(
        ROOT / "output" / "bps_full_stack_v2" / "trades_BA_05pct.parquet"
    )
    trades["close_date"] = pd.to_datetime(trades["close_date"])
    trades["open_date"] = pd.to_datetime(trades["open_date"])
    ALL_TICKERS_87 = sorted(trades["ticker"].unique().tolist())

    spy = pd.read_parquet(
        ROOT / "wheel_strategy_v1" / "data" / "cache" / "spy_prices.parquet"
    )
    spy["date"] = pd.to_datetime(spy["date"])
    spy = spy.sort_values("date")
    spy["spy_ret"] = spy["close"].pct_change()

    return trades, spy


def build_daily_pnl(trades_df, ticker_list=None):
    """Build daily PnL series from trade-level data, optionally filtered."""
    t = trades_df.copy()
    if ticker_list is not None:
        t = t[t["ticker"].isin(ticker_list)]
    daily = t.groupby("close_date")["realized_pnl"].sum()
    all_dates = pd.date_range(daily.index.min(), daily.index.max(), freq="B")
    daily = daily.reindex(all_dates, fill_value=0.0)
    return daily


def compute_metrics(daily_pnl, label=""):
    """Compute Sharpe, Sortino, PF, WR from daily PnL."""
    if len(daily_pnl) < 30:
        return {"label": label, "error": "insufficient data"}

    equity = STARTING_CAPITAL + daily_pnl.cumsum()
    rets = equity.pct_change().dropna()

    if rets.std() == 0:
        return {"label": label, "error": "zero variance"}

    excess = rets - RF_DAILY
    sharpe = float(excess.mean() / excess.std() * np.sqrt(TRADING_DAYS))

    down = excess[excess < 0]
    sortino = (
        float(excess.mean() / down.std() * np.sqrt(TRADING_DAYS))
        if len(down) > 5 and down.std() > 0
        else 0.0
    )

    gross_win = daily_pnl[daily_pnl > 0].sum()
    gross_loss = abs(daily_pnl[daily_pnl < 0].sum())
    pf = float(gross_win / gross_loss) if gross_loss > 0 else float("inf")

    wr = float((daily_pnl > 0).sum() / len(daily_pnl))

    peak = equity.cummax()
    max_dd = float((equity / peak - 1).min())

    total_pnl = float(daily_pnl.sum())
    cagr = float(
        (equity.iloc[-1] / STARTING_CAPITAL) ** (TRADING_DAYS / len(daily_pnl)) - 1
    )

    return {
        "label": label,
        "sharpe": sharpe,
        "sortino": sortino,
        "profit_factor": pf,
        "win_rate": wr,
        "max_dd": max_dd,
        "total_pnl": total_pnl,
        "cagr": cagr,
        "n_days": len(daily_pnl),
    }


def compute_r1_gap(daily_pnl, spy_df):
    """Compute R1 regime gap: |Sharpe_green - Sharpe_red| / max(|Sharpe_green|, |Sharpe_red|)."""
    equity = STARTING_CAPITAL + daily_pnl.cumsum()
    rets = equity.pct_change().dropna()
    excess = rets - RF_DAILY

    # Align SPY returns to our dates
    spy_ret = spy_df.set_index("date")["spy_ret"].reindex(excess.index).fillna(0)

    # Classify days
    green_mask = spy_ret > 0.002
    red_mask = spy_ret < -0.002

    green_excess = excess[green_mask]
    red_excess = excess[red_mask]

    if len(green_excess) < 10 or len(red_excess) < 10:
        return float("nan"), float("nan"), float("nan")

    green_sharpe = (
        float(green_excess.mean() / green_excess.std() * np.sqrt(TRADING_DAYS))
        if green_excess.std() > 0
        else 0.0
    )
    red_sharpe = (
        float(red_excess.mean() / red_excess.std() * np.sqrt(TRADING_DAYS))
        if red_excess.std() > 0
        else 0.0
    )

    denom = max(abs(green_sharpe), abs(red_sharpe))
    gap = abs(green_sharpe - red_sharpe) / denom if denom > 0 else float("nan")

    return gap, green_sharpe, red_sharpe


# ═══════════════════════════════════════════════════════════════
# TEST 1: PERMUTATION TEST
# ═══════════════════════════════════════════════════════════════


def run_permutation_test(daily_pnl, spy_df, n_perms=200):
    """Shuffle daily returns, recompute Sharpe. p-value = frac(shuffled >= actual)."""
    print(f"\n{'='*70}")
    print(f"  TEST 1: PERMUTATION TEST ({n_perms} shuffles)")
    print(f"{'='*70}")

    real_metrics = compute_metrics(daily_pnl, "Real")
    real_sharpe = real_metrics["sharpe"]
    real_gap, real_gs, real_rs = compute_r1_gap(daily_pnl, spy_df)

    print(f"  Real: Sharpe={real_sharpe:.3f}, Gap={real_gap:.3f}")
    print(f"    Green Sharpe={real_gs:.3f}, Red Sharpe={real_rs:.3f}")

    pnl_values = daily_pnl.values.copy()
    dates = daily_pnl.index

    perm_sharpes = []
    perm_gaps = []
    perm_r1_pass = 0

    np.random.seed(42)
    for i in range(n_perms):
        shuffled = np.random.permutation(pnl_values)
        shuf_series = pd.Series(shuffled, index=dates)

        m = compute_metrics(shuf_series, f"Perm_{i}")
        if "error" in m:
            continue
        perm_sharpes.append(m["sharpe"])

        gap, _, _ = compute_r1_gap(shuf_series, spy_df)
        if not np.isnan(gap):
            perm_gaps.append(gap)
            if gap <= 0.50:
                perm_r1_pass += 1

        if (i + 1) % 50 == 0:
            print(f"    ... {i+1}/{n_perms} done")

    perm_sharpes = np.array(perm_sharpes)
    perm_gaps = np.array(perm_gaps)

    p_sharpe = np.mean(perm_sharpes >= real_sharpe)
    p_gap = np.mean(perm_gaps <= real_gap) if len(perm_gaps) > 0 else 1.0

    print(f"\n  Permutation results:")
    print(
        f"    Sharpe: real={real_sharpe:.3f}, perm mean={perm_sharpes.mean():.3f} +/- {perm_sharpes.std():.3f}"
    )
    print(f"    Sharpe p-value: {p_sharpe:.4f} (fraction >= real)")
    print(
        f"    Gap: real={real_gap:.3f}, perm mean={perm_gaps.mean():.3f} +/- {perm_gaps.std():.3f}"
    )
    print(f"    Perm R1 pass rate: {perm_r1_pass}/{n_perms} ({100*perm_r1_pass/n_perms:.1f}%)")

    verdict = "PASS" if p_sharpe < 0.05 else "FAIL"
    print(
        f"\n  VERDICT: {verdict} — {'Sharpe is NOT random' if verdict == 'PASS' else 'Sharpe could be random!'}"
    )

    return {
        "test": "permutation",
        "n_perms": n_perms,
        "real_sharpe": real_sharpe,
        "real_gap": real_gap,
        "real_green_sharpe": real_gs,
        "real_red_sharpe": real_rs,
        "perm_sharpe_mean": float(perm_sharpes.mean()),
        "perm_sharpe_std": float(perm_sharpes.std()),
        "perm_sharpe_p5": float(np.percentile(perm_sharpes, 5)),
        "perm_sharpe_p95": float(np.percentile(perm_sharpes, 95)),
        "p_sharpe": float(p_sharpe),
        "perm_gap_mean": float(perm_gaps.mean()) if len(perm_gaps) > 0 else None,
        "perm_gap_std": float(perm_gaps.std()) if len(perm_gaps) > 0 else None,
        "p_gap": float(p_gap),
        "perm_r1_pass_rate": perm_r1_pass / n_perms,
        "verdict": verdict,
    }


# ═══════════════════════════════════════════════════════════════
# TEST 2: BOOTSTRAP CONFIDENCE INTERVAL
# ═══════════════════════════════════════════════════════════════


def run_bootstrap(daily_pnl, spy_df, n_boots=1000):
    """Resample daily returns with replacement, compute Sharpe + R1 gap."""
    print(f"\n{'='*70}")
    print(f"  TEST 2: BOOTSTRAP CONFIDENCE INTERVAL ({n_boots} resamples)")
    print(f"{'='*70}")

    real_metrics = compute_metrics(daily_pnl, "Real")
    real_sharpe = real_metrics["sharpe"]
    real_gap, _, _ = compute_r1_gap(daily_pnl, spy_df)

    pnl_values = daily_pnl.values.copy()
    dates = daily_pnl.index
    n = len(pnl_values)

    # For regime classification, we need aligned SPY returns
    spy_ret = spy_df.set_index("date")["spy_ret"].reindex(dates).fillna(0).values

    boot_sharpes = []
    boot_gaps = []
    boot_r1_pass = 0

    np.random.seed(123)
    for i in range(n_boots):
        # Resample WITH replacement (paired: same dates for PnL and SPY)
        idx = np.random.choice(n, size=n, replace=True)
        boot_pnl = pnl_values[idx]
        boot_spy = spy_ret[idx]

        # Compute Sharpe
        equity = STARTING_CAPITAL + np.cumsum(boot_pnl)
        rets = np.diff(equity) / equity[:-1]
        excess = rets - RF_DAILY

        if np.std(excess) > 0:
            sharpe = float(np.mean(excess) / np.std(excess) * np.sqrt(TRADING_DAYS))
        else:
            continue

        # Compute R1 gap from bootstrapped data
        green_mask = boot_spy > 0.002
        red_mask = boot_spy < -0.002

        green_exc = excess[green_mask[1:]]  # align with rets (1 shorter)
        red_exc = excess[red_mask[1:]]

        gap = float("nan")
        if len(green_exc) >= 10 and len(red_exc) >= 10:
            gs_std = np.std(green_exc)
            rs_std = np.std(red_exc)
            if gs_std > 0 and rs_std > 0:
                gs = float(np.mean(green_exc) / gs_std * np.sqrt(TRADING_DAYS))
                rs = float(np.mean(red_exc) / rs_std * np.sqrt(TRADING_DAYS))
                denom = max(abs(gs), abs(rs))
                if denom > 0:
                    gap = abs(gs - rs) / denom

        boot_sharpes.append(sharpe)
        if not np.isnan(gap):
            boot_gaps.append(gap)
            if gap <= 0.50:
                boot_r1_pass += 1

        if (i + 1) % 200 == 0:
            print(f"    ... {i+1}/{n_boots} done")

    boot_sharpes = np.array(boot_sharpes)
    boot_gaps = np.array(boot_gaps)

    r1_pass_rate = boot_r1_pass / len(boot_gaps) if len(boot_gaps) > 0 else 0

    print(f"\n  Bootstrap results:")
    print(f"    Sharpe: median={np.median(boot_sharpes):.3f}")
    print(
        f"    Sharpe 90% CI: [{np.percentile(boot_sharpes, 5):.3f}, {np.percentile(boot_sharpes, 95):.3f}]"
    )
    print(
        f"    Sharpe > 1.0: {100 * np.mean(boot_sharpes > 1.0):.1f}%"
    )
    print(
        f"    Sharpe > 2.0: {100 * np.mean(boot_sharpes > 2.0):.1f}%"
    )
    print(f"    Gap: median={np.median(boot_gaps):.3f}")
    print(
        f"    Gap 90% CI: [{np.percentile(boot_gaps, 5):.3f}, {np.percentile(boot_gaps, 95):.3f}]"
    )
    print(f"    R1 pass rate (gap <= 0.50): {100*r1_pass_rate:.1f}%")

    verdict = "ROBUST" if r1_pass_rate >= 0.70 else ("MARGINAL" if r1_pass_rate >= 0.50 else "FRAGILE")
    print(f"\n  VERDICT: {verdict} (R1 pass rate = {100*r1_pass_rate:.1f}%)")

    return {
        "test": "bootstrap",
        "n_boots": n_boots,
        "real_sharpe": real_sharpe,
        "real_gap": real_gap,
        "boot_sharpe_median": float(np.median(boot_sharpes)),
        "boot_sharpe_p5": float(np.percentile(boot_sharpes, 5)),
        "boot_sharpe_p95": float(np.percentile(boot_sharpes, 95)),
        "boot_sharpe_gt_1_pct": float(np.mean(boot_sharpes > 1.0)),
        "boot_sharpe_gt_2_pct": float(np.mean(boot_sharpes > 2.0)),
        "boot_gap_median": float(np.median(boot_gaps)),
        "boot_gap_p5": float(np.percentile(boot_gaps, 5)),
        "boot_gap_p95": float(np.percentile(boot_gaps, 95)),
        "boot_r1_pass_rate": r1_pass_rate,
        "verdict": verdict,
    }


# ═══════════════════════════════════════════════════════════════
# TEST 3: PARAMETER (UNIVERSE) STABILITY
# ═══════════════════════════════════════════════════════════════


def run_universe_stability(trades_df, spy_df, n_perturb=100):
    """
    Perturb the GA universe:
      A. Drop 1 ticker (20 combos)
      B. Drop 2 tickers (190 combos — sample 50)
      C. Swap 1-3 tickers with random non-GA tickers (50 random swaps)
    Check if Sharpe > 1.0 and gap <= 0.50 still hold.
    """
    print(f"\n{'='*70}")
    print(f"  TEST 3: UNIVERSE STABILITY (perturb GA 20-ticker selection)")
    print(f"{'='*70}")

    non_ga = [t for t in ALL_TICKERS_87 if t not in GA_UNIVERSE]
    print(f"  GA tickers: {len(GA_UNIVERSE)}, non-GA candidates: {len(non_ga)}")

    results = []

    # A. Leave-one-out (20 tests)
    print("\n  A. Leave-one-out (20 tests)...")
    for drop_ticker in GA_UNIVERSE:
        subset = [t for t in GA_UNIVERSE if t != drop_ticker]
        daily = build_daily_pnl(trades_df, subset)
        m = compute_metrics(daily, f"drop_{drop_ticker}")
        gap, gs, rs = compute_r1_gap(daily, spy_df)
        results.append({
            "method": "drop_1",
            "dropped": drop_ticker,
            "n_tickers": len(subset),
            "sharpe": m.get("sharpe", float("nan")),
            "gap": gap,
            "green_sharpe": gs,
            "red_sharpe": rs,
            "r1_pass": gap <= 0.50 if not np.isnan(gap) else False,
            "sharpe_gt_1": m.get("sharpe", 0) > 1.0,
        })

    loo_pass = sum(1 for r in results if r["r1_pass"])
    loo_sharpe_pass = sum(1 for r in results if r["sharpe_gt_1"])
    print(f"    R1 pass: {loo_pass}/20 ({100*loo_pass/20:.0f}%)")
    print(f"    Sharpe > 1: {loo_sharpe_pass}/20 ({100*loo_sharpe_pass/20:.0f}%)")

    # Find most fragile ticker (dropping it causes biggest gap increase)
    real_daily = build_daily_pnl(trades_df, GA_UNIVERSE)
    real_gap, _, _ = compute_r1_gap(real_daily, spy_df)
    loo_results = [r for r in results if r["method"] == "drop_1"]
    loo_results_sorted = sorted(loo_results, key=lambda x: x["gap"] if not np.isnan(x["gap"]) else 99, reverse=True)
    if loo_results_sorted:
        worst = loo_results_sorted[0]
        print(f"    Most fragile ticker: {worst['dropped']} (gap={worst['gap']:.3f} without it)")
        best = loo_results_sorted[-1]
        print(f"    Strongest drop: {best['dropped']} (gap={best['gap']:.3f} without it)")

    # B. Leave-two-out (sample 50 of C(20,2)=190)
    print("\n  B. Leave-two-out (50 random pairs)...")
    from itertools import combinations
    all_pairs = list(combinations(GA_UNIVERSE, 2))
    np.random.seed(77)
    sampled_pairs = [all_pairs[i] for i in np.random.choice(len(all_pairs), min(50, len(all_pairs)), replace=False)]

    l2o_results = []
    for pair in sampled_pairs:
        subset = [t for t in GA_UNIVERSE if t not in pair]
        daily = build_daily_pnl(trades_df, subset)
        m = compute_metrics(daily, f"drop_{pair}")
        gap, gs, rs = compute_r1_gap(daily, spy_df)
        l2o_results.append({
            "method": "drop_2",
            "dropped": list(pair),
            "n_tickers": len(subset),
            "sharpe": m.get("sharpe", float("nan")),
            "gap": gap,
            "r1_pass": gap <= 0.50 if not np.isnan(gap) else False,
            "sharpe_gt_1": m.get("sharpe", 0) > 1.0,
        })

    l2o_pass = sum(1 for r in l2o_results if r["r1_pass"])
    l2o_sharpe = sum(1 for r in l2o_results if r["sharpe_gt_1"])
    print(f"    R1 pass: {l2o_pass}/{len(l2o_results)} ({100*l2o_pass/len(l2o_results):.0f}%)")
    print(f"    Sharpe > 1: {l2o_sharpe}/{len(l2o_results)} ({100*l2o_sharpe/len(l2o_results):.0f}%)")
    results.extend(l2o_results)

    # C. Swap 1-3 tickers with non-GA tickers (50 random swaps)
    print("\n  C. Swap 1-3 tickers with non-GA alternatives (50 tests)...")
    swap_results = []
    np.random.seed(99)
    for _ in range(50):
        n_swap = np.random.choice([1, 2, 3])
        drop = list(np.random.choice(GA_UNIVERSE, n_swap, replace=False))
        add = list(np.random.choice(non_ga, n_swap, replace=False))
        subset = [t for t in GA_UNIVERSE if t not in drop] + add
        daily = build_daily_pnl(trades_df, subset)
        m = compute_metrics(daily, f"swap_{n_swap}")
        gap, gs, rs = compute_r1_gap(daily, spy_df)
        swap_results.append({
            "method": f"swap_{n_swap}",
            "dropped": drop,
            "added": add,
            "n_tickers": len(subset),
            "sharpe": m.get("sharpe", float("nan")),
            "gap": gap,
            "r1_pass": gap <= 0.50 if not np.isnan(gap) else False,
            "sharpe_gt_1": m.get("sharpe", 0) > 1.0,
        })

    swap_pass = sum(1 for r in swap_results if r["r1_pass"])
    swap_sharpe = sum(1 for r in swap_results if r["sharpe_gt_1"])
    print(f"    R1 pass: {swap_pass}/{len(swap_results)} ({100*swap_pass/len(swap_results):.0f}%)")
    print(f"    Sharpe > 1: {swap_sharpe}/{len(swap_results)} ({100*swap_sharpe/len(swap_results):.0f}%)")
    results.extend(swap_results)

    # D. Random 20-ticker baskets from full 87 (baseline comparison)
    print("\n  D. Random 20-ticker baskets from full universe (30 tests)...")
    rand_results = []
    np.random.seed(55)
    for _ in range(30):
        rand_tickers = list(np.random.choice(ALL_TICKERS_87, 20, replace=False))
        daily = build_daily_pnl(trades_df, rand_tickers)
        m = compute_metrics(daily, "random_20")
        gap, gs, rs = compute_r1_gap(daily, spy_df)
        rand_results.append({
            "method": "random_20",
            "tickers": rand_tickers,
            "sharpe": m.get("sharpe", float("nan")),
            "gap": gap,
            "r1_pass": gap <= 0.50 if not np.isnan(gap) else False,
            "sharpe_gt_1": m.get("sharpe", 0) > 1.0,
        })

    rand_pass = sum(1 for r in rand_results if r["r1_pass"])
    rand_sharpe_values = [r["sharpe"] for r in rand_results if not np.isnan(r["sharpe"])]
    real_metrics_full = compute_metrics(real_daily, "GA_20")
    rand_sharpe_gt_ga = sum(1 for s in rand_sharpe_values if s >= real_metrics_full.get("sharpe", 0))
    print(f"    R1 pass: {rand_pass}/{len(rand_results)} ({100*rand_pass/len(rand_results):.0f}%)")
    print(f"    Sharpe >= GA ({real_metrics_full.get('sharpe', 0):.2f}): {rand_sharpe_gt_ga}/{len(rand_results)}")
    print(f"    Random Sharpe range: [{min(rand_sharpe_values):.2f}, {max(rand_sharpe_values):.2f}]")
    print(f"    Random Sharpe mean: {np.mean(rand_sharpe_values):.2f}")

    # Summary
    total_perturb = len(loo_results) + len(l2o_results) + len(swap_results)
    total_pass = loo_pass + l2o_pass + swap_pass
    total_sharpe_pass = loo_sharpe_pass + l2o_sharpe + swap_sharpe

    real_metrics_full = compute_metrics(real_daily, "GA_20")

    verdict = "ROBUST" if total_pass / total_perturb >= 0.70 else (
        "MARGINAL" if total_pass / total_perturb >= 0.50 else "FRAGILE"
    )

    print(f"\n  OVERALL STABILITY:")
    print(f"    R1 pass rate across perturbations: {total_pass}/{total_perturb} ({100*total_pass/total_perturb:.1f}%)")
    print(f"    Sharpe > 1.0 rate: {total_sharpe_pass}/{total_perturb} ({100*total_sharpe_pass/total_perturb:.1f}%)")
    print(f"    Random baskets Sharpe percentile rank: "
          f"{100 * np.mean([s < real_metrics_full.get('sharpe', 0) for s in rand_sharpe_values]):.0f}%")
    print(f"  VERDICT: {verdict}")

    return {
        "test": "universe_stability",
        "loo_r1_pass_rate": loo_pass / 20,
        "loo_sharpe_gt1_rate": loo_sharpe_pass / 20,
        "l2o_r1_pass_rate": l2o_pass / len(l2o_results) if l2o_results else 0,
        "swap_r1_pass_rate": swap_pass / len(swap_results) if swap_results else 0,
        "total_r1_pass_rate": total_pass / total_perturb,
        "total_sharpe_gt1_rate": total_sharpe_pass / total_perturb,
        "random_20_r1_pass_rate": rand_pass / len(rand_results),
        "random_20_sharpe_mean": float(np.mean(rand_sharpe_values)),
        "random_20_sharpe_range": [float(min(rand_sharpe_values)), float(max(rand_sharpe_values))],
        "ga_sharpe_percentile_vs_random": float(
            np.mean([s < real_metrics_full.get("sharpe", 0) for s in rand_sharpe_values])
        ),
        "loo_details": loo_results,
        "verdict": verdict,
    }


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════


def main():
    t0 = time.time()

    print("=" * 70)
    print("BPS GA ADVERSARIAL VALIDATION")
    print("=" * 70)

    print("\nLoading data...")
    trades, spy = load_data()
    print(f"  Trades: {len(trades):,}")
    print(f"  SPY days: {len(spy):,}")

    # Build GA daily PnL
    daily_pnl = build_daily_pnl(trades, GA_UNIVERSE)
    print(f"  GA daily PnL days: {len(daily_pnl):,}")

    # Baseline metrics
    real = compute_metrics(daily_pnl, "BPS_GA_20")
    real_gap, real_gs, real_rs = compute_r1_gap(daily_pnl, spy)
    print(f"\n  BASELINE: Sharpe={real['sharpe']:.3f}, Sortino={real['sortino']:.3f}, "
          f"PF={real['profit_factor']:.2f}, WR={100*real['win_rate']:.1f}%")
    print(f"  R1 gap={real_gap:.3f}, Green Sharpe={real_gs:.3f}, Red Sharpe={real_rs:.3f}")
    print(f"  R1 PASS: {'YES' if real_gap <= 0.50 else 'NO'}")

    results = {"baseline": {**real, "gap": real_gap, "green_sharpe": real_gs, "red_sharpe": real_rs}}

    # Test 1: Permutation
    results["permutation"] = run_permutation_test(daily_pnl, spy, n_perms=200)

    # Test 2: Bootstrap
    results["bootstrap"] = run_bootstrap(daily_pnl, spy, n_boots=1000)

    # Test 3: Universe stability
    results["universe_stability"] = run_universe_stability(trades, spy)

    # ─── Final Summary ───
    print("\n" + "=" * 70)
    print("FINAL ADVERSARIAL SUMMARY — BPS GA")
    print("=" * 70)

    perm = results["permutation"]
    boot = results["bootstrap"]
    stab = results["universe_stability"]

    print(f"\n  1. Permutation test (n={perm['n_perms']}):")
    print(f"     Sharpe p-value: {perm['p_sharpe']:.4f} → {'PASS' if perm['p_sharpe'] < 0.05 else 'FAIL'}")
    print(f"     (real={perm['real_sharpe']:.3f} vs perm mean={perm['perm_sharpe_mean']:.3f})")

    print(f"\n  2. Bootstrap CI (n={boot['n_boots']}):")
    print(f"     Sharpe 90% CI: [{boot['boot_sharpe_p5']:.3f}, {boot['boot_sharpe_p95']:.3f}]")
    print(f"     R1 pass rate: {100*boot['boot_r1_pass_rate']:.1f}%")
    print(f"     Gap 90% CI: [{boot['boot_gap_p5']:.3f}, {boot['boot_gap_p95']:.3f}]")

    print(f"\n  3. Universe stability:")
    print(f"     Leave-1-out R1 pass: {100*stab['loo_r1_pass_rate']:.0f}%")
    print(f"     Leave-2-out R1 pass: {100*stab['l2o_r1_pass_rate']:.0f}%")
    print(f"     Swap 1-3 R1 pass: {100*stab['swap_r1_pass_rate']:.0f}%")
    print(f"     GA Sharpe percentile vs random 20: {100*stab['ga_sharpe_percentile_vs_random']:.0f}%")

    # Overall verdict
    perm_ok = perm["p_sharpe"] < 0.05
    boot_ok = boot["boot_r1_pass_rate"] >= 0.50
    stab_ok = stab["total_r1_pass_rate"] >= 0.50

    if perm_ok and boot_ok and stab_ok:
        overall = "ROBUST PASS — Keep in portfolio"
    elif perm_ok and (boot_ok or stab_ok):
        overall = "MARGINAL — Keep but monitor closely"
    elif perm_ok:
        overall = "FRAGILE R1 — Tighten parameters or add hedge"
    else:
        overall = "FAIL — Consider dropping"

    results["overall_verdict"] = overall
    print(f"\n  OVERALL: {overall}")

    # Save results
    def sanitize(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [sanitize(v) for v in obj]
        return obj

    with open(OUTPUT / "adversarial_results.json", "w") as f:
        json.dump(sanitize(results), f, indent=2, default=str)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s. Results saved.")
    print("=" * 70)


if __name__ == "__main__":
    main()
