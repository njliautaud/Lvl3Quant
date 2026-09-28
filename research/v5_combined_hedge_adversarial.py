#!/usr/bin/env python3
"""
v5_combined_hedge_adversarial.py — Adversarial validation of V5 combined hedge

Tests whether the R1-passing result (Sharpe 2.27, gap 0.46) is genuine edge
or data-mining artifact. Three tests:

1. PERMUTATION TEST: Randomly shuffle daily CSP returns across dates (destroy
   temporal signal). If shuffled versions also pass R1, the result is spurious.

2. DATE SENSITIVITY: Bootstrap subsets of OOS dates (80% random sample, 100x).
   Report % of subsamples where R1 passes. Robust result = >80%.

3. PARAMETER STABILITY: How sensitive is R1 gap to small perturbations of
   (base_hedge ± 0.05, stressed_hedge ± 0.10, vix_stress ± 2)? Fragile if
   gap crosses 0.50 with tiny changes.

Usage:
    python3 /home/jupiter/Lvl3Quant/research/v5_combined_hedge_adversarial.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path("/home/jupiter/Lvl3Quant")
WS_ROOT = ROOT / "wheel_strategy_v1"
OUTPUT = ROOT / "output" / "v5_combined_hedge_adversarial"
OUTPUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(WS_ROOT))

TRADING_DAYS = 252
RF_DAILY = 0.04 / TRADING_DAYS

# Import from the main analysis
sys.path.insert(0, str(ROOT / "research"))
from v5_combined_hedge import (
    vix_ts_sizing, load_v5_equity, load_spy_vix,
    apply_ts_sizing, apply_dynamic_beta_hedge, compute_metrics,
)

# Best config from sweep
BEST_BASE_HR = 0.20
BEST_STRESS_HR = 0.40
BEST_VIX_STRESS = 20.0


def run_permutation_test(base_eq, spy_close, vix, vix3m, n_perms=200):
    """
    Shuffle daily CSP returns, re-apply TS sizing + hedge.
    If random returns also produce R1-passing results, the hedge
    is just mechanically compressing returns, not capturing real edge.
    """
    print(f"\n{'='*70}")
    print(f"  TEST 1: PERMUTATION TEST ({n_perms} shuffles)")
    print(f"{'='*70}")

    # Get the real result
    ts_eq, _ = apply_ts_sizing(base_eq, vix, vix3m)
    real_combined = apply_dynamic_beta_hedge(ts_eq, spy_close, vix,
                                             BEST_BASE_HR, BEST_STRESS_HR, BEST_VIX_STRESS)
    real_m = compute_metrics(real_combined, spy_close, "Real")
    real_sharpe = real_m["sharpe"]
    real_gap = real_m.get("regime_gap", float("nan"))

    print(f"  Real: Sharpe={real_sharpe:.3f}, Gap={real_gap:.3f}")

    # Shuffle the BASE equity curve daily returns (before TS sizing)
    eq = base_eq.set_index("date")["equity"].sort_index().astype(float)
    daily_ret = eq.pct_change().fillna(0.0).values[1:]  # skip first NaN
    dates = eq.index

    perm_sharpes = []
    perm_gaps = []
    perm_r1_pass = 0

    np.random.seed(42)
    for i in range(n_perms):
        # Shuffle returns
        shuffled_ret = np.random.permutation(daily_ret)
        # Reconstruct equity curve
        shuffled_eq = np.zeros(len(dates))
        shuffled_eq[0] = float(eq.iloc[0])
        for j in range(len(shuffled_ret)):
            shuffled_eq[j + 1] = shuffled_eq[j] * (1.0 + shuffled_ret[j])

        shuf_df = pd.DataFrame({"date": dates, "equity": shuffled_eq})

        # Apply same overlay
        ts_shuf, _ = apply_ts_sizing(shuf_df, vix, vix3m)
        combined_shuf = apply_dynamic_beta_hedge(ts_shuf, spy_close, vix,
                                                  BEST_BASE_HR, BEST_STRESS_HR, BEST_VIX_STRESS)
        m = compute_metrics(combined_shuf, spy_close, f"Perm_{i}")
        perm_sharpes.append(m["sharpe"])
        gap = m.get("regime_gap", float("nan"))
        perm_gaps.append(gap)
        if not np.isnan(gap) and gap <= 0.50:
            perm_r1_pass += 1

        if (i + 1) % 50 == 0:
            print(f"    ... {i+1}/{n_perms} done")

    perm_sharpes = np.array(perm_sharpes)
    perm_gaps = np.array([g for g in perm_gaps if not np.isnan(g)])

    # p-value: fraction of permutations with Sharpe >= real
    p_sharpe = np.mean(perm_sharpes >= real_sharpe)
    # p-value for gap: fraction with gap <= real
    p_gap = np.mean(perm_gaps <= real_gap) if len(perm_gaps) > 0 else 1.0

    print(f"\n  Permutation results:")
    print(f"    Sharpe: real={real_sharpe:.3f}, perm mean={perm_sharpes.mean():.3f} ± {perm_sharpes.std():.3f}")
    print(f"    Sharpe p-value: {p_sharpe:.4f} (fraction >= real)")
    print(f"    Gap: real={real_gap:.3f}, perm mean={perm_gaps.mean():.3f} ± {perm_gaps.std():.3f}")
    print(f"    Gap p-value: {p_gap:.4f} (fraction <= real)")
    print(f"    Permutations passing R1: {perm_r1_pass}/{n_perms} ({100*perm_r1_pass/n_perms:.1f}%)")

    verdict = "PASS" if p_sharpe < 0.05 else "FAIL"
    print(f"\n  VERDICT: {verdict} — {'Sharpe is NOT random' if verdict == 'PASS' else 'Sharpe could be random!'}")

    return {
        "real_sharpe": real_sharpe, "real_gap": real_gap,
        "perm_sharpe_mean": float(perm_sharpes.mean()),
        "perm_sharpe_std": float(perm_sharpes.std()),
        "p_sharpe": float(p_sharpe),
        "p_gap": float(p_gap),
        "perm_r1_pass_rate": perm_r1_pass / n_perms,
        "verdict": verdict,
    }


def run_date_bootstrap(base_eq, spy_close, vix, vix3m, n_boots=200, sample_frac=0.80):
    """
    Bootstrap 80% of trading dates, 200 times. Report R1 pass rate.
    """
    print(f"\n{'='*70}")
    print(f"  TEST 2: DATE BOOTSTRAP ({n_boots} x {sample_frac:.0%} of dates)")
    print(f"{'='*70}")

    eq = base_eq.set_index("date")["equity"].sort_index().astype(float)
    dates = eq.index
    n_sample = int(len(dates) * sample_frac)

    boot_sharpes = []
    boot_gaps = []
    boot_r1_pass = 0

    np.random.seed(123)
    for i in range(n_boots):
        # Sample dates without replacement
        idx = np.sort(np.random.choice(len(dates), n_sample, replace=False))
        sampled_dates = dates[idx]

        # Subset equity curve to sampled dates
        sub_eq = pd.DataFrame({
            "date": sampled_dates,
            "equity": eq.iloc[idx].values,
        })

        # Apply overlays
        ts_sub, _ = apply_ts_sizing(sub_eq, vix, vix3m)
        combined_sub = apply_dynamic_beta_hedge(ts_sub, spy_close, vix,
                                                 BEST_BASE_HR, BEST_STRESS_HR, BEST_VIX_STRESS)
        m = compute_metrics(combined_sub, spy_close, f"Boot_{i}")
        boot_sharpes.append(m["sharpe"])
        gap = m.get("regime_gap", float("nan"))
        boot_gaps.append(gap)
        if not np.isnan(gap) and gap <= 0.50:
            boot_r1_pass += 1

        if (i + 1) % 50 == 0:
            print(f"    ... {i+1}/{n_boots} done")

    boot_sharpes = np.array(boot_sharpes)
    boot_gaps = np.array([g for g in boot_gaps if not np.isnan(g)])

    pass_rate = boot_r1_pass / n_boots
    print(f"\n  Bootstrap results:")
    print(f"    Sharpe: mean={boot_sharpes.mean():.3f} ± {boot_sharpes.std():.3f}, "
          f"[5th, 95th] = [{np.percentile(boot_sharpes, 5):.3f}, {np.percentile(boot_sharpes, 95):.3f}]")
    print(f"    Gap: mean={boot_gaps.mean():.3f} ± {boot_gaps.std():.3f}")
    print(f"    R1 pass rate: {boot_r1_pass}/{n_boots} ({100*pass_rate:.1f}%)")

    verdict = "ROBUST" if pass_rate >= 0.80 else "FRAGILE"
    print(f"\n  VERDICT: {verdict} — R1 pass rate {'≥80%' if verdict == 'ROBUST' else '<80%'}")

    return {
        "sharpe_mean": float(boot_sharpes.mean()),
        "sharpe_std": float(boot_sharpes.std()),
        "sharpe_5th": float(np.percentile(boot_sharpes, 5)),
        "sharpe_95th": float(np.percentile(boot_sharpes, 95)),
        "gap_mean": float(boot_gaps.mean()),
        "gap_std": float(boot_gaps.std()),
        "r1_pass_rate": pass_rate,
        "verdict": verdict,
    }


def run_param_stability(base_eq, spy_close, vix, vix3m):
    """
    Perturb best config parameters slightly. Check if R1 still passes.
    """
    print(f"\n{'='*70}")
    print(f"  TEST 3: PARAMETER STABILITY (perturbation around best config)")
    print(f"{'='*70}")

    perturbations = []
    for db in [-0.05, -0.025, 0, 0.025, 0.05]:
        for ds in [-0.10, -0.05, 0, 0.05, 0.10]:
            for dv in [-3, -1, 0, 1, 3]:
                b = BEST_BASE_HR + db
                s = BEST_STRESS_HR + ds
                v = BEST_VIX_STRESS + dv
                if b < 0 or s < 0 or s <= b or v < 10:
                    continue

                ts_eq, _ = apply_ts_sizing(base_eq, vix, vix3m)
                combined = apply_dynamic_beta_hedge(ts_eq, spy_close, vix, b, s, v)
                m = compute_metrics(combined, spy_close, f"b{b:.3f}/s{s:.2f}/v{v}")
                gap = m.get("regime_gap", float("nan"))
                perturbations.append({
                    "base_hr": b, "stress_hr": s, "vix_thresh": v,
                    "sharpe": m["sharpe"], "regime_gap": gap,
                    "r1_pass": not np.isnan(gap) and gap <= 0.50,
                    "db": db, "ds": ds, "dv": dv,
                })

    n_pass = sum(1 for p in perturbations if p["r1_pass"])
    n_total = len(perturbations)
    pass_rate = n_pass / n_total if n_total > 0 else 0

    print(f"  Tested {n_total} perturbations around best config (b={BEST_BASE_HR}, s={BEST_STRESS_HR}, v={BEST_VIX_STRESS})")
    print(f"  R1 pass rate: {n_pass}/{n_total} ({100*pass_rate:.1f}%)")

    # Show gap sensitivity
    gaps = [p["regime_gap"] for p in perturbations if not np.isnan(p["regime_gap"])]
    if gaps:
        print(f"  Gap range: [{min(gaps):.3f}, {max(gaps):.3f}]")
        print(f"  Gap mean: {np.mean(gaps):.3f} ± {np.std(gaps):.3f}")

    # Show which dimensions matter most
    for dim, delta_key in [("base_hr", "db"),
                            ("stress_hr", "ds"),
                            ("vix_thresh", "dv")]:
        center = [p for p in perturbations if all(
            p[k] == 0 for k in ["db", "ds", "dv"] if k != delta_key
        )]
        if center:
            center.sort(key=lambda x: x[dim])
            print(f"\n  {dim} sensitivity (others fixed):")
            for p in center:
                mark = "✓" if p["r1_pass"] else "✗"
                print(f"    {mark} {dim}={p[dim]:.3f}: Sharpe={p['sharpe']:.3f}, Gap={p['regime_gap']:.3f}")

    verdict = "STABLE" if pass_rate >= 0.60 else "FRAGILE"
    print(f"\n  VERDICT: {verdict}")

    return {
        "n_perturbations": n_total,
        "n_pass": n_pass,
        "pass_rate": pass_rate,
        "gap_range": [min(gaps), max(gaps)] if gaps else None,
        "gap_mean": float(np.mean(gaps)) if gaps else None,
        "verdict": verdict,
    }


def main():
    print("=" * 70)
    print("  V5 Combined Hedge — Adversarial Validation")
    print("=" * 70)

    base_eq = load_v5_equity()
    spy_close, vix, vix3m = load_spy_vix()
    print(f"  Data: {len(base_eq)} days, {base_eq['date'].min().date()} to {base_eq['date'].max().date()}")

    # Run all three tests
    perm = run_permutation_test(base_eq, spy_close, vix, vix3m, n_perms=200)
    boot = run_date_bootstrap(base_eq, spy_close, vix, vix3m, n_boots=200)
    stab = run_param_stability(base_eq, spy_close, vix, vix3m)

    # Summary
    print(f"\n{'='*70}")
    print(f"  ADVERSARIAL VALIDATION SUMMARY")
    print(f"{'='*70}")
    print(f"  1. Permutation test:    {perm['verdict']} (p={perm['p_sharpe']:.4f})")
    print(f"  2. Date bootstrap:      {boot['verdict']} (R1 pass rate={boot['r1_pass_rate']:.1%})")
    print(f"  3. Parameter stability: {stab['verdict']} (perturbation pass rate={stab['pass_rate']:.1%})")

    all_pass = (perm["verdict"] == "PASS" and boot["verdict"] == "ROBUST" and stab["verdict"] == "STABLE")
    overall = "VALIDATED" if all_pass else "CONCERNS"
    print(f"\n  OVERALL: {overall}")
    if all_pass:
        print(f"  ✓ V5 Combined Hedge is a robust, non-random, parameter-stable R1-passing strategy")
    else:
        issues = []
        if perm["verdict"] != "PASS":
            issues.append("Sharpe may be random (permutation)")
        if boot["verdict"] != "ROBUST":
            issues.append("R1 fragile across date subsets")
        if stab["verdict"] != "STABLE":
            issues.append("R1 sensitive to parameter perturbation")
        print(f"  ⚠ Issues: {'; '.join(issues)}")

    # Save
    results = {"permutation": perm, "bootstrap": boot, "stability": stab, "overall": overall}
    with open(OUTPUT / "adversarial_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved to {OUTPUT}")


if __name__ == "__main__":
    main()
