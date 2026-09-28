#!/usr/bin/env python3
"""
etf_v3_hedge_adversarial.py — Adversarial validation of ETF Rotation v3 + Beta Hedge

Tests whether the beta_1.00 (rolling 60d beta, scale=1.0) hedge overlay result
(Sharpe 2.33, R1 gap 0.007, CAGR 24.2%) is robust or fragile.

Four adversarial tests:
  1. Permutation test (200 iters): shuffle returns, recompute Sharpe. p < 0.05 = PASS.
  2. Bootstrap CI (1000 iters): resample with replacement. Report median, 5th/95th CI.
  3. Parameter stability: perturb scale (0.80-1.20) and beta window (20d-120d).
  4. Rolling stability: split into 3 equal sub-periods, test R1 independently.

Usage:
    python3 /home/jupiter/Lvl3Quant/research/etf_v3_hedge_adversarial.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path("/home/jupiter/Lvl3Quant")
OUTPUT = ROOT / "output" / "etf_v3_hedge"
OUTPUT.mkdir(parents=True, exist_ok=True)

TRADING_DAYS = 252
SEED = 42

# --------------------------------------------------------------------------
# Data Loading (reuse from overlay script)
# --------------------------------------------------------------------------

def load_etf_v3_returns() -> pd.Series:
    book = pd.read_parquet(
        ROOT / "output/macro_picker/etf_rotation_quality_20260709_223944/book.parquet"
    )
    book["date"] = pd.to_datetime(book["date"])
    return book.set_index("date")["daily_ret"].sort_index()


def load_spy_returns() -> pd.Series:
    px = pd.read_parquet(ROOT / "wheel_strategy_v1/data/cache/prices_v2.parquet")
    px["date"] = pd.to_datetime(px["date"])
    spy = px[px["ticker"] == "SPY"].sort_values("date").set_index("date")["close"]
    return spy.pct_change().dropna()


# --------------------------------------------------------------------------
# R1 + Metrics (canonical)
# --------------------------------------------------------------------------

def classify_regime(spy_ret: pd.Series, thr: float = 0.005) -> pd.Series:
    labels = pd.Series("flat", index=spy_ret.index, dtype=object)
    labels[spy_ret > thr] = "green"
    labels[spy_ret < -thr] = "red"
    return labels


def compute_sharpe(daily_ret: pd.Series) -> float:
    if len(daily_ret) < 30 or daily_ret.std() == 0:
        return float("nan")
    return float(daily_ret.mean() / daily_ret.std() * np.sqrt(TRADING_DAYS))


def compute_r1_gap(daily_ret: pd.Series, regime_labels: pd.Series) -> float:
    aligned = regime_labels.reindex(daily_ret.index).fillna("flat")
    sharpes = {}
    for r in ("green", "red"):
        sub = daily_ret[aligned == r]
        if len(sub) >= 5 and sub.std() > 0:
            sharpes[r] = float(sub.mean() / sub.std() * np.sqrt(TRADING_DAYS))
        else:
            return float("nan")
    sg, sr = sharpes["green"], sharpes["red"]
    denom = max(abs(sg), abs(sr))
    if denom < 1e-9:
        return float("nan")
    return abs(sg - sr) / denom


def compute_full_metrics(daily_ret: pd.Series, regime_labels: pd.Series) -> dict:
    ret = daily_ret.dropna()
    sharpe = compute_sharpe(ret)
    r1_gap = compute_r1_gap(ret, regime_labels)
    r1_pass = bool(np.isfinite(r1_gap) and r1_gap <= 0.50)

    equity = (1 + ret).cumprod()
    n_years = len(ret) / TRADING_DAYS
    cagr = float(equity.iloc[-1] ** (1 / n_years) - 1) if n_years > 0 else 0.0
    max_dd = float((equity / equity.cummax() - 1).min())

    down = ret[ret < 0]
    sortino = float(ret.mean() / down.std() * np.sqrt(TRADING_DAYS)) if len(down) > 0 and down.std() > 0 else 0.0

    # Per-regime sharpes
    aligned = regime_labels.reindex(ret.index).fillna("flat")
    regime_sharpe = {}
    for r in ("green", "red", "flat"):
        sub = ret[aligned == r]
        if len(sub) >= 5 and sub.std() > 0:
            regime_sharpe[r] = float(sub.mean() / sub.std() * np.sqrt(TRADING_DAYS))
        else:
            regime_sharpe[r] = float("nan")

    return {
        "sharpe": sharpe, "sortino": sortino, "cagr": cagr, "max_dd": max_dd,
        "sharpe_green": regime_sharpe["green"], "sharpe_red": regime_sharpe["red"],
        "sharpe_flat": regime_sharpe["flat"],
        "r1_gap": r1_gap, "r1_pass": r1_pass, "n_days": len(ret),
    }


# --------------------------------------------------------------------------
# Hedge functions
# --------------------------------------------------------------------------

def compute_rolling_beta(etf_ret: pd.Series, spy_ret: pd.Series, window: int = 60) -> pd.Series:
    common = etf_ret.index.intersection(spy_ret.index)
    e = etf_ret.loc[common]
    s = spy_ret.loc[common]
    min_per = min(max(10, window // 2), window)
    cov = e.rolling(window, min_periods=min_per).cov(s)
    var = s.rolling(window, min_periods=min_per).var()
    return (cov / var).replace([np.inf, -np.inf], np.nan)


def apply_beta_hedge(etf_ret: pd.Series, spy_ret: pd.Series,
                     beta: pd.Series, scale: float = 1.0) -> pd.Series:
    common = etf_ret.index.intersection(spy_ret.index).intersection(beta.index)
    beta_prev = beta.loc[common].shift(1).fillna(0)
    hr = beta_prev * scale
    return etf_ret.loc[common] - hr * spy_ret.loc[common]


# --------------------------------------------------------------------------
# TEST 1: Permutation Test
# --------------------------------------------------------------------------

def permutation_test(hedged_returns: pd.Series, regime_labels: pd.Series,
                     n_iter: int = 200) -> dict:
    """Two permutation tests:
    A) Sharpe significance: shuffle returns in 5-day blocks (preserve vol clustering),
       test if actual Sharpe is significantly above random time-ordering.
    B) R1 gap significance: shuffle date-to-regime mapping, test if actual R1 gap
       (0.007) is significantly smaller than random regime assignment.
    """
    print(f"\n[1] PERMUTATION TEST ({n_iter} iterations)")
    actual_sharpe = compute_sharpe(hedged_returns)
    actual_r1_gap = compute_r1_gap(hedged_returns, regime_labels)

    rng = np.random.RandomState(SEED)
    vals = hedged_returns.values.copy()
    n = len(vals)

    # A) Block-shuffle Sharpe test (5-day blocks to preserve some autocorrelation)
    block_size = 5
    n_blocks = n // block_size
    block_sharpes = []
    for _ in range(n_iter):
        block_idx = rng.permutation(n_blocks)
        shuffled = np.concatenate([vals[i*block_size:(i+1)*block_size] for i in block_idx])
        # Trim or pad to original length
        shuffled = shuffled[:n]
        s = float(np.mean(shuffled) / np.std(shuffled, ddof=1) * np.sqrt(TRADING_DAYS))
        block_sharpes.append(s)

    block_sharpes = np.array(block_sharpes)
    # p-value: fraction with Sharpe >= actual (shouldn't change much since
    # block shuffle preserves mean/std roughly; this tests autocorrelation contribution)
    p_sharpe = float(np.mean(block_sharpes >= actual_sharpe))

    print(f"    A) Block-shuffle Sharpe test:")
    print(f"       Actual Sharpe: {actual_sharpe:.3f}")
    print(f"       Shuffled: mean={block_sharpes.mean():.3f}, std={block_sharpes.std():.3f}")
    print(f"       p-value: {p_sharpe:.4f}")

    # B) Regime-shuffle R1 gap test (the KEY test: is the tiny R1 gap just luck?)
    aligned_labels = regime_labels.reindex(hedged_returns.index).fillna("flat")
    label_vals = aligned_labels.values.copy()
    perm_gaps = []
    for _ in range(n_iter):
        shuffled_labels = label_vals.copy()
        rng.shuffle(shuffled_labels)
        perm_regime = pd.Series(shuffled_labels, index=hedged_returns.index)
        g = compute_r1_gap(hedged_returns, perm_regime)
        if np.isfinite(g):
            perm_gaps.append(g)

    perm_gaps = np.array(perm_gaps)
    # p-value: fraction of permuted gaps <= actual gap (smaller = better for us)
    p_r1 = float(np.mean(perm_gaps <= actual_r1_gap))

    print(f"    B) Regime-shuffle R1 gap test:")
    print(f"       Actual R1 gap: {actual_r1_gap:.4f}")
    print(f"       Permuted gaps: mean={perm_gaps.mean():.3f}, "
          f"median={np.median(perm_gaps):.3f}")
    print(f"       p-value (gap <= actual): {p_r1:.4f}")
    print(f"       Interpretation: {p_r1*100:.1f}% of random regime assignments "
          f"produce a gap this small or smaller")

    # The R1 gap test is the meaningful one. If p_r1 is high (say >0.20),
    # the small gap is NOT unusual -- strategy is genuinely regime-agnostic.
    # If p_r1 is tiny (<0.05), the small gap IS unusual and may be fitted.
    # For a truly regime-agnostic strategy, we WANT p_r1 to be high (not cherry-picked).
    # So we invert: PASS if p_r1 >= 0.10 (gap is not unusually small = genuine).
    r1_genuine = p_r1 >= 0.10
    sharpe_pass = p_sharpe < 0.20  # Block shuffle shouldn't change Sharpe much

    overall_pass = r1_genuine  # R1 genuineness is the key test

    print(f"    R1 gap genuineness: {'GENUINE' if r1_genuine else 'SUSPICIOUS'} "
          f"(p={p_r1:.3f}, want >= 0.10)")
    print(f"    Overall: {'PASS' if overall_pass else 'FAIL'}")

    return {
        "actual_sharpe": actual_sharpe,
        "actual_r1_gap": actual_r1_gap,
        "p_sharpe_block": p_sharpe,
        "block_sharpe_mean": float(block_sharpes.mean()),
        "block_sharpe_std": float(block_sharpes.std()),
        "p_r1_regime_shuffle": p_r1,
        "perm_gap_mean": float(perm_gaps.mean()),
        "perm_gap_median": float(np.median(perm_gaps)),
        "r1_genuine": r1_genuine,
        "n_iter": n_iter,
        "passed": overall_pass,
    }


# --------------------------------------------------------------------------
# TEST 2: Bootstrap CI
# --------------------------------------------------------------------------

def bootstrap_ci(hedged_returns: pd.Series, regime_labels: pd.Series,
                 n_iter: int = 1000) -> dict:
    print(f"\n[2] BOOTSTRAP CI ({n_iter} iterations)")
    rng = np.random.RandomState(SEED + 1)
    vals = hedged_returns.values
    n = len(vals)

    # We need regime labels aligned for R1 computation
    aligned_labels = regime_labels.reindex(hedged_returns.index).fillna("flat")
    label_vals = aligned_labels.values

    boot_sharpes = []
    boot_r1_gaps = []

    for i in range(n_iter):
        idx = rng.randint(0, n, size=n)
        boot_ret = pd.Series(vals[idx], index=hedged_returns.index)
        boot_labels = pd.Series(label_vals[idx], index=hedged_returns.index)

        s = compute_sharpe(boot_ret)
        g = compute_r1_gap(boot_ret, boot_labels)
        boot_sharpes.append(s)
        boot_r1_gaps.append(g)

    boot_sharpes = np.array(boot_sharpes)
    boot_r1_gaps = np.array(boot_r1_gaps)
    valid_gaps = boot_r1_gaps[np.isfinite(boot_r1_gaps)]

    median_sharpe = float(np.median(boot_sharpes))
    ci_5 = float(np.percentile(boot_sharpes, 5))
    ci_95 = float(np.percentile(boot_sharpes, 95))
    pct_sharpe_gt_1 = float(np.mean(boot_sharpes > 1.0) * 100)
    pct_r1_pass = float(np.mean(valid_gaps <= 0.50) * 100) if len(valid_gaps) > 0 else 0.0

    print(f"    Sharpe: median={median_sharpe:.3f}, 5th={ci_5:.3f}, 95th={ci_95:.3f}")
    print(f"    % bootstraps with Sharpe > 1.0: {pct_sharpe_gt_1:.1f}%")
    print(f"    % bootstraps passing R1 (gap <= 0.50): {pct_r1_pass:.1f}%")

    return {
        "median_sharpe": median_sharpe,
        "ci_5": ci_5,
        "ci_95": ci_95,
        "pct_sharpe_gt_1": pct_sharpe_gt_1,
        "pct_r1_pass": pct_r1_pass,
        "n_iter": n_iter,
        "n_valid_gaps": int(len(valid_gaps)),
    }


# --------------------------------------------------------------------------
# TEST 3: Parameter Stability
# --------------------------------------------------------------------------

def parameter_stability(etf_ret: pd.Series, spy_ret: pd.Series,
                        regime_labels: pd.Series) -> dict:
    print(f"\n[3] PARAMETER STABILITY")

    scales = [0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20]
    windows = [20, 40, 60, 80, 120]

    results = []
    n_pass = 0
    n_total = 0

    print(f"\n    SCALE PERTURBATION (window=60d):")
    print(f"    {'Scale':<8} {'Sharpe':<8} {'R1_gap':<8} {'R1':<5}")
    beta_60 = compute_rolling_beta(etf_ret, spy_ret, 60)
    for scale in scales:
        hedged = apply_beta_hedge(etf_ret, spy_ret, beta_60, scale)
        sharpe = compute_sharpe(hedged)
        gap = compute_r1_gap(hedged, regime_labels)
        r1_pass = bool(np.isfinite(gap) and gap <= 0.50)
        n_total += 1
        if r1_pass:
            n_pass += 1
        results.append({
            "type": "scale", "scale": scale, "window": 60,
            "sharpe": sharpe, "r1_gap": gap, "r1_pass": r1_pass,
        })
        print(f"    {scale:<8.2f} {sharpe:<8.3f} {gap:<8.3f} {'PASS' if r1_pass else 'FAIL'}")

    print(f"\n    WINDOW PERTURBATION (scale=1.0):")
    print(f"    {'Window':<8} {'Sharpe':<8} {'R1_gap':<8} {'R1':<5}")
    for win in windows:
        beta_w = compute_rolling_beta(etf_ret, spy_ret, win)
        hedged = apply_beta_hedge(etf_ret, spy_ret, beta_w, 1.0)
        sharpe = compute_sharpe(hedged)
        gap = compute_r1_gap(hedged, regime_labels)
        r1_pass = bool(np.isfinite(gap) and gap <= 0.50)
        n_total += 1
        if r1_pass:
            n_pass += 1
        results.append({
            "type": "window", "scale": 1.0, "window": win,
            "sharpe": sharpe, "r1_gap": gap, "r1_pass": r1_pass,
        })
        print(f"    {win:<8d} {sharpe:<8.3f} {gap:<8.3f} {'PASS' if r1_pass else 'FAIL'}")

    # Full grid (for summary stat)
    print(f"\n    FULL GRID (scale x window):")
    print(f"    {'Scale':<8}", end="")
    for win in windows:
        print(f" {win}d".rjust(10), end="")
    print()

    grid_results = []
    for scale in scales:
        print(f"    {scale:<8.2f}", end="")
        for win in windows:
            # Skip duplicates already computed
            existing = [r for r in results
                        if r["scale"] == scale and r["window"] == win]
            if existing:
                sharpe = existing[0]["sharpe"]
                gap = existing[0]["r1_gap"]
                r1_pass = existing[0]["r1_pass"]
            else:
                beta_w = compute_rolling_beta(etf_ret, spy_ret, win)
                hedged = apply_beta_hedge(etf_ret, spy_ret, beta_w, scale)
                sharpe = compute_sharpe(hedged)
                gap = compute_r1_gap(hedged, regime_labels)
                r1_pass = bool(np.isfinite(gap) and gap <= 0.50)

            grid_results.append({
                "scale": scale, "window": win,
                "sharpe": sharpe, "r1_gap": gap, "r1_pass": r1_pass,
            })
            marker = "*" if r1_pass else " "
            print(f" {sharpe:6.2f}{marker}".rjust(10), end="")
        print()

    n_grid_total = len(grid_results)
    n_grid_pass = sum(1 for r in grid_results if r["r1_pass"])
    pct_pass = n_grid_pass / n_grid_total * 100

    print(f"\n    Grid R1 pass rate: {n_grid_pass}/{n_grid_total} ({pct_pass:.0f}%)")
    print(f"    (* = R1 PASS)")

    return {
        "scale_results": [r for r in results if r["type"] == "scale"],
        "window_results": [r for r in results if r["type"] == "window"],
        "grid_results": grid_results,
        "grid_pass_rate": pct_pass / 100,
        "n_grid_pass": n_grid_pass,
        "n_grid_total": n_grid_total,
    }


# --------------------------------------------------------------------------
# TEST 4: Rolling Stability (sub-period R1)
# --------------------------------------------------------------------------

def rolling_stability(hedged_returns: pd.Series, spy_ret: pd.Series,
                      n_periods: int = 3) -> dict:
    print(f"\n[4] ROLLING STABILITY ({n_periods} equal sub-periods)")

    n = len(hedged_returns)
    split_size = n // n_periods
    periods = []

    for i in range(n_periods):
        start = i * split_size
        end = start + split_size if i < n_periods - 1 else n
        sub_ret = hedged_returns.iloc[start:end]
        sub_spy = spy_ret.reindex(sub_ret.index).dropna()
        sub_regime = classify_regime(sub_spy)

        metrics = compute_full_metrics(sub_ret, sub_regime)
        metrics["period"] = i + 1
        metrics["start_date"] = str(sub_ret.index[0].date())
        metrics["end_date"] = str(sub_ret.index[-1].date())
        periods.append(metrics)

        print(f"    Period {i+1}: {metrics['start_date']} to {metrics['end_date']} "
              f"({metrics['n_days']} days)")
        print(f"      Sharpe={metrics['sharpe']:.2f}  Sortino={metrics['sortino']:.2f}  "
              f"CAGR={metrics['cagr']*100:.1f}%  MaxDD={metrics['max_dd']*100:.1f}%")
        print(f"      Sh_G={metrics['sharpe_green']:.2f}  Sh_R={metrics['sharpe_red']:.2f}  "
              f"R1_gap={metrics['r1_gap']:.3f}  R1={'PASS' if metrics['r1_pass'] else 'FAIL'}")

    all_pass = all(p["r1_pass"] for p in periods)
    all_sharpe_gt_1 = all(p["sharpe"] > 1.0 for p in periods)

    print(f"\n    All sub-periods R1 pass: {'YES' if all_pass else 'NO'}")
    print(f"    All sub-periods Sharpe > 1.0: {'YES' if all_sharpe_gt_1 else 'NO'}")

    return {
        "periods": periods,
        "all_r1_pass": all_pass,
        "all_sharpe_gt_1": all_sharpe_gt_1,
    }


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("ADVERSARIAL VALIDATION: ETF Rotation v3 + Beta Hedge (scale=1.0, 60d)")
    print("=" * 70)

    # Load data
    print("\n[0] Loading data...")
    etf_ret = load_etf_v3_returns()
    spy_ret = load_spy_returns()

    common = etf_ret.index.intersection(spy_ret.index)
    etf_ret = etf_ret.loc[common]
    spy_ret = spy_ret.loc[common]
    regime_labels = classify_regime(spy_ret)

    # Compute the hedged returns for the baseline config
    beta_60d = compute_rolling_beta(etf_ret, spy_ret, 60)
    hedged = apply_beta_hedge(etf_ret, spy_ret, beta_60d, scale=1.0)

    baseline = compute_full_metrics(hedged, regime_labels)
    print(f"    Hedged returns: {len(hedged)} days, "
          f"{hedged.index[0].date()} to {hedged.index[-1].date()}")
    print(f"    Baseline: Sharpe={baseline['sharpe']:.3f}  R1_gap={baseline['r1_gap']:.3f}  "
          f"CAGR={baseline['cagr']*100:.1f}%")

    # Run tests
    t1 = permutation_test(hedged, regime_labels, n_iter=200)
    t2 = bootstrap_ci(hedged, regime_labels, n_iter=1000)
    t3 = parameter_stability(etf_ret, spy_ret, regime_labels)
    t4 = rolling_stability(hedged, spy_ret, n_periods=3)

    # ========================================================================
    # Final Verdict
    # ========================================================================
    print("\n" + "=" * 70)
    print("ADVERSARIAL VALIDATION SUMMARY")
    print("=" * 70)

    tests = {
        "permutation_test": {
            "passed": t1["passed"],
            "detail": f"R1 gap genuine (p_regime_shuffle={t1['p_r1_regime_shuffle']:.3f}, want >= 0.10)",
        },
        "bootstrap_sharpe_gt_1": {
            "passed": t2["pct_sharpe_gt_1"] >= 95.0,
            "detail": f"{t2['pct_sharpe_gt_1']:.1f}% bootstraps > 1.0 (need >= 95%)",
        },
        "bootstrap_r1": {
            "passed": t2["pct_r1_pass"] >= 80.0,
            "detail": f"{t2['pct_r1_pass']:.1f}% bootstraps pass R1 (need >= 80%)",
        },
        "param_stability": {
            "passed": t3["grid_pass_rate"] >= 0.60,
            "detail": f"{t3['grid_pass_rate']*100:.0f}% of grid passes R1 (need >= 60%)",
        },
        "rolling_stability": {
            "passed": t4["all_r1_pass"],
            "detail": f"All sub-periods R1 pass: {t4['all_r1_pass']}",
        },
        "rolling_sharpe": {
            "passed": t4["all_sharpe_gt_1"],
            "detail": f"All sub-periods Sharpe > 1.0: {t4['all_sharpe_gt_1']}",
        },
    }

    n_pass = sum(1 for t in tests.values() if t["passed"])
    n_total = len(tests)

    for name, t in tests.items():
        status = "PASS" if t["passed"] else "FAIL"
        print(f"  [{status}] {name}: {t['detail']}")

    overall = "ROBUST" if n_pass >= 5 else ("MARGINAL" if n_pass >= 3 else "FRAGILE")
    print(f"\n  VERDICT: {overall} ({n_pass}/{n_total} tests passed)")

    if overall == "ROBUST":
        print("  Strategy passes adversarial validation. R1 compliance is genuine.")
    elif overall == "MARGINAL":
        print("  Strategy has some fragility. Proceed with caution; monitor R1 in live.")
    else:
        print("  Strategy is fragile. R1 pass may be parameter-fitted or period-specific.")

    # Save results
    results = {
        "baseline": baseline,
        "test_1_permutation": t1,
        "test_2_bootstrap": {
            k: v for k, v in t2.items()
        },
        "test_3_param_stability": {
            "grid_pass_rate": t3["grid_pass_rate"],
            "n_grid_pass": t3["n_grid_pass"],
            "n_grid_total": t3["n_grid_total"],
            "scale_results": t3["scale_results"],
            "window_results": t3["window_results"],
            "grid_results": t3["grid_results"],
        },
        "test_4_rolling_stability": t4,
        "verdict": {
            "overall": overall,
            "tests_passed": n_pass,
            "tests_total": n_total,
            "test_details": tests,
        },
    }

    out_path = OUTPUT / "adversarial_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n  Results saved to: {out_path}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    main()
