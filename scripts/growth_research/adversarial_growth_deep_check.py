#!/usr/bin/env python3
"""
DEEP ADVERSARIAL VALIDATION — Walk-Forward Growth Strategies
============================================================
Comprehensive checks for data leakage, unrealistic returns, parameter
snooping, outlier dependence, survivorship bias, and regime robustness.

Checks performed:
  1. Data Integrity (return realism, date coverage, cross-check vs yfinance)
  2. Parameter Leakage (train/test overlap, parameter stability)
  3. Permutation Test (proper ETF-shuffle, 1000 iterations)
  4. Sub-Period Consistency (first/last 14 folds)
  5. Outlier Analysis (remove top 5%, top 10 days)
  6. Survivorship Bias (TQQQ/SOXL launch dates, crash behavior)
  7. Regime Stratification (R1: green/red/flat SPY days)
  8. Drawdown Analysis (5 worst drawdowns)
"""

import numpy as np
import pandas as pd
import json, time, sys, warnings
from pathlib import Path
from collections import Counter

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
except ImportError:
    print("Installing yfinance...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance", "-q"])
    import yfinance as yf

# ── Paths ─────────────────────────────────────────────────────────────
BASE_DIR = Path("/home/jupiter/Lvl3Quant/output/growth_research/walkforward_validation")
RESULTS_PATH = BASE_DIR / "walkforward_results.json"
DM_RETURNS_PATH = BASE_DIR / "dm_oot_returns.csv"
BO_RETURNS_PATH = BASE_DIR / "bo_oot_returns.csv"
OUT_PATH = Path("/home/jupiter/Lvl3Quant/output/growth_research/adversarial_deep_check_results.json")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────

def load_all():
    """Load results JSON and OOT return CSVs."""
    with open(RESULTS_PATH) as f:
        results = json.load(f)

    dm_rets = pd.read_csv(DM_RETURNS_PATH, index_col=0, parse_dates=True).squeeze()
    bo_rets = pd.read_csv(BO_RETURNS_PATH, index_col=0, parse_dates=True).squeeze()

    # Rename if needed
    if hasattr(dm_rets, 'name'):
        dm_rets.name = "DM"
    if hasattr(bo_rets, 'name'):
        bo_rets.name = "BO"

    return results, dm_rets, bo_rets


def sharpe(dr):
    """Annualized Sharpe from daily returns."""
    dr = dr.dropna()
    if len(dr) < 20 or dr.std() == 0:
        return 0.0
    return dr.mean() / dr.std() * np.sqrt(252)


# ══════════════════════════════════════════════════════════════════════
# CHECK 1: DATA INTEGRITY
# ══════════════════════════════════════════════════════════════════════

def check_data_integrity(results, dm_rets, bo_rets):
    """Verify returns are realistic, dates are continuous, and match actual market data."""
    print("\n" + "=" * 72)
    print("CHECK 1: DATA INTEGRITY")
    print("=" * 72)

    issues = []
    details = {}

    # 1a. Return realism — no daily return > 20%
    dm_max = dm_rets.abs().max()
    bo_max = bo_rets.abs().max()
    dm_extreme = (dm_rets.abs() > 0.20).sum()
    bo_extreme = (bo_rets.abs() > 0.20).sum()

    details["dm_max_abs_return"] = round(float(dm_max), 4)
    details["bo_max_abs_return"] = round(float(bo_max), 4)
    details["dm_returns_above_20pct"] = int(dm_extreme)
    details["bo_returns_above_20pct"] = int(bo_extreme)

    if dm_extreme > 0:
        issues.append(f"DM has {dm_extreme} daily returns > 20% — possible bug")
    if bo_extreme > 0:
        issues.append(f"BO has {bo_extreme} daily returns > 20% — possible bug")

    print(f"  DM max |daily return|: {dm_max:.4f} ({dm_max*100:.2f}%)")
    print(f"  BO max |daily return|: {bo_max:.4f} ({bo_max*100:.2f}%)")
    print(f"  Returns > 20%: DM={dm_extreme}, BO={bo_extreme}")

    # 1b. Date coverage and gaps
    dm_dates = dm_rets.index
    bo_dates = bo_rets.index
    dm_gaps = pd.Series(dm_dates).diff().dt.days
    bo_gaps = pd.Series(bo_dates).diff().dt.days
    # Normal gaps: weekends (2-3 days), holidays (up to ~5 days)
    dm_big_gaps = (dm_gaps > 5).sum()
    bo_big_gaps = (bo_gaps > 5).sum()

    details["dm_date_range"] = f"{dm_dates[0].date()} to {dm_dates[-1].date()}"
    details["bo_date_range"] = f"{bo_dates[0].date()} to {bo_dates[-1].date()}"
    details["dm_n_days"] = len(dm_rets)
    details["bo_n_days"] = len(bo_rets)
    details["dm_gaps_over_5_days"] = int(dm_big_gaps)
    details["bo_gaps_over_5_days"] = int(bo_big_gaps)

    print(f"  DM: {len(dm_rets)} days, {dm_dates[0].date()} to {dm_dates[-1].date()}")
    print(f"  BO: {len(bo_rets)} days, {bo_dates[0].date()} to {bo_dates[-1].date()}")
    print(f"  Gaps > 5 calendar days: DM={dm_big_gaps}, BO={bo_big_gaps}")

    if dm_big_gaps > 5:
        issues.append(f"DM has {dm_big_gaps} suspicious gaps (>5 calendar days)")

    # 1c. Fold coverage — verify 28 folds exist with proper periods
    dm_folds = results["per_fold"]["dual_momentum"]
    bo_folds = results["per_fold"]["breakout"]
    n_valid_dm = sum(1 for f in dm_folds if f.get("valid", False))
    n_valid_bo = sum(1 for f in bo_folds if f.get("valid", False))

    details["dm_valid_folds"] = n_valid_dm
    details["bo_valid_folds"] = n_valid_bo
    details["total_folds"] = len(dm_folds)

    print(f"  Valid folds: DM={n_valid_dm}/{len(dm_folds)}, BO={n_valid_bo}/{len(bo_folds)}")

    # 1d. Cross-check against actual ETF prices for known dates
    print("  Cross-checking against yfinance data...")
    try:
        spy_data = yf.download("SPY", start="2013-01-01", end="2013-02-01",
                               auto_adjust=True, progress=False)
        if len(spy_data) > 5:
            spy_rets = spy_data["Close"].pct_change().dropna()
            # Check a few dates
            cross_check_ok = True
            mismatch_count = 0
            for date in spy_rets.index[:5]:
                if date in dm_rets.index:
                    # DM returns won't exactly match SPY since it may hold different ETFs,
                    # but they should be in the same ballpark (within 10x)
                    dm_r = abs(dm_rets.loc[date])
                    spy_r = abs(spy_rets.loc[date])
                    if dm_r > 0.001 and spy_r > 0.001:
                        ratio = dm_r / spy_r
                        if ratio > 20:  # If DM return is 20x SPY, something is very wrong
                            mismatch_count += 1

            details["cross_check_mismatches"] = mismatch_count
            if mismatch_count > 0:
                issues.append(f"{mismatch_count} cross-check mismatches (OOT returns wildly different from SPY)")
            print(f"    Cross-check mismatches (>20x SPY): {mismatch_count}")
        else:
            print("    WARNING: Could not download SPY data for cross-check")
            details["cross_check_mismatches"] = -1
    except Exception as e:
        print(f"    WARNING: Cross-check failed: {e}")
        details["cross_check_mismatches"] = -1

    # 1e. Check for identical returns across strategies (would indicate bug)
    common_dates = dm_rets.index.intersection(bo_rets.index)
    identical = (dm_rets.loc[common_dates] == bo_rets.loc[common_dates]).sum()
    identical_pct = identical / len(common_dates) * 100
    details["identical_returns_pct"] = round(float(identical_pct), 1)
    print(f"  Identical returns (DM==BO): {identical}/{len(common_dates)} ({identical_pct:.1f}%)")
    if identical_pct > 80:
        issues.append(f"DM and BO have {identical_pct:.1f}% identical returns — strategies may not be independent")

    passed = len(issues) == 0
    status = "PASS" if passed else "FAIL"
    print(f"\n  >>> CHECK 1 RESULT: {status}")
    for iss in issues:
        print(f"      ISSUE: {iss}")

    return {"status": status, "passed": passed, "issues": issues, "details": details}


# ══════════════════════════════════════════════════════════════════════
# CHECK 2: PARAMETER LEAKAGE
# ══════════════════════════════════════════════════════════════════════

def check_parameter_leakage(results):
    """Verify parameters are optimized only on training data, check for suspicious uniformity."""
    print("\n" + "=" * 72)
    print("CHECK 2: PARAMETER LEAKAGE")
    print("=" * 72)

    issues = []
    details = {}

    # 2a. Check train/test window overlap
    for strat_name, strat_key in [("Dual Momentum", "dual_momentum"), ("Breakout", "breakout")]:
        folds = results["per_fold"][strat_key]
        overlap_count = 0
        for i, fold in enumerate(folds):
            train_str = fold["train"]
            test_str = fold["test"]
            train_start, train_end = train_str.split("-", 1)[0], "-".join(train_str.split("-")[3:]) if train_str.count("-") >= 5 else train_str.split("-")[-1]

            # Parse dates more carefully
            parts = fold["train"].split("-")
            # Format: YYYY-MM-DD-YYYY-MM-DD
            train_s = pd.Timestamp(f"{parts[0]}-{parts[1]}-{parts[2]}")
            train_e = pd.Timestamp(f"{parts[3]}-{parts[4]}-{parts[5]}")

            parts2 = fold["test"].split("-")
            test_s = pd.Timestamp(f"{parts2[0]}-{parts2[1]}-{parts2[2]}")
            test_e = pd.Timestamp(f"{parts2[3]}-{parts2[4]}-{parts2[5]}")

            if train_e > test_s:
                overlap_count += 1
                if i < 3:  # Print first few
                    issues.append(f"{strat_name} fold {i}: train ends {train_e.date()} > test starts {test_s.date()}")

            # Check sliding: train_end should == test_start
            # Actually in the code, train_end = train_start + 36mo, test_start = train_end
            # So they should be equal (no gap, no overlap)

        details[f"{strat_key}_overlapping_folds"] = overlap_count
        print(f"  {strat_name}: {overlap_count} folds with train/test overlap")

    # 2b. Check parameter stability across folds
    for strat_name, strat_key in [("Dual Momentum", "dual_momentum"), ("Breakout", "breakout")]:
        folds = results["per_fold"][strat_key]
        params_list = [f["params"] for f in folds if f.get("valid", False)]
        param_counts = Counter(params_list)
        most_common = param_counts.most_common(1)[0] if param_counts else ("N/A", 0)
        unique_params = len(param_counts)

        details[f"{strat_key}_unique_params"] = unique_params
        details[f"{strat_key}_most_common_param"] = most_common[0]
        details[f"{strat_key}_most_common_count"] = most_common[1]
        details[f"{strat_key}_param_distribution"] = dict(param_counts)

        print(f"\n  {strat_name} parameter distribution:")
        for p, c in param_counts.most_common():
            print(f"    {p}: {c}/{len(params_list)} folds ({c/len(params_list)*100:.0f}%)")

        # Flag if ONE parameter set dominates > 80% of folds
        if most_common[1] / len(params_list) > 0.80:
            issues.append(f"{strat_name}: parameter '{most_common[0]}' used in {most_common[1]}/{len(params_list)} folds "
                         f"({most_common[1]/len(params_list)*100:.0f}%) — suspiciously uniform, possible info leakage")

    # 2c. Check train vs OOT Sharpe correlation
    # If OOT Sharpe is too correlated with train Sharpe, parameters may be overfit
    for strat_name, strat_key in [("Dual Momentum", "dual_momentum"), ("Breakout", "breakout")]:
        folds = results["per_fold"][strat_key]
        valid = [f for f in folds if f.get("valid", False) and "sharpe" in f]
        train_sharpes = [f["train_sharpe"] for f in valid]
        oot_sharpes = [f["sharpe"] for f in valid]

        if len(train_sharpes) > 5:
            corr = np.corrcoef(train_sharpes, oot_sharpes)[0, 1]
            details[f"{strat_key}_train_oot_sharpe_corr"] = round(float(corr), 3)
            print(f"\n  {strat_name} train-vs-OOT Sharpe correlation: {corr:.3f}")

            # Very high correlation could mean parameters aren't actually varying
            # Or the market regime is just consistent
            if corr > 0.9:
                issues.append(f"{strat_name}: train-OOT Sharpe correlation = {corr:.3f} — suspiciously high")

    # 2d. Check if ALL folds are positive (27/27 claimed) — what's the probability under null?
    for strat_name, strat_key in [("Dual Momentum", "dual_momentum"), ("Breakout", "breakout")]:
        folds = results["per_fold"][strat_key]
        valid = [f for f in folds if f.get("valid", False) and "sharpe" in f]
        n_positive = sum(1 for f in valid if f["sharpe"] > 0)
        n_total = len(valid)

        # Under null (no edge), P(positive Sharpe) ~ 0.5
        # P(all positive) = 0.5^n
        null_prob = 0.5 ** n_total
        details[f"{strat_key}_positive_folds"] = f"{n_positive}/{n_total}"
        details[f"{strat_key}_null_probability_all_positive"] = f"{null_prob:.2e}"

        print(f"\n  {strat_name}: {n_positive}/{n_total} positive Sharpe folds")
        print(f"    Probability under null (coin flip): {null_prob:.2e}")

        # Even more suspicious: all folds have Sharpe > 1.0
        n_above1 = sum(1 for f in valid if f["sharpe"] > 1.0)
        details[f"{strat_key}_folds_sharpe_above_1"] = f"{n_above1}/{n_total}"
        print(f"    Folds with Sharpe > 1.0: {n_above1}/{n_total}")

        if n_positive == n_total and n_total > 20:
            # This is not necessarily leakage — could be a genuinely strong strategy
            # But worth flagging for scrutiny
            print(f"    NOTE: Perfect score across {n_total} folds is extremely rare")

    passed = len(issues) == 0
    status = "PASS" if passed else "WARNING"
    print(f"\n  >>> CHECK 2 RESULT: {status}")
    for iss in issues:
        print(f"      ISSUE: {iss}")

    return {"status": status, "passed": passed, "issues": issues, "details": details}


# ══════════════════════════════════════════════════════════════════════
# CHECK 3: PERMUTATION TEST (proper ETF-shuffle)
# ══════════════════════════════════════════════════════════════════════

def check_permutation_test(dm_rets, bo_rets, n_perms=1000):
    """Proper permutation test: randomly assign ETF holdings per day."""
    print("\n" + "=" * 72)
    print("CHECK 3: PERMUTATION TEST (1000 shuffles)")
    print("=" * 72)

    details = {}

    # Download actual ETF returns for the permutation
    print("  Downloading ETF data for permutation baseline...")
    tickers = ["TQQQ", "QQQ", "SPY", "SOXL", "SHV"]
    try:
        etf_data = yf.download(tickers, start="2012-01-01", end="2026-08-01",
                               auto_adjust=True, progress=False)
        if isinstance(etf_data.columns, pd.MultiIndex):
            etf_prices = etf_data["Close"]
        else:
            etf_prices = etf_data
        etf_rets = etf_prices.pct_change().dropna(how="all")
    except Exception as e:
        print(f"  ERROR downloading ETF data: {e}")
        return {"status": "ERROR", "passed": False, "issues": [str(e)], "details": {}}

    for strat_name, oot_rets in [("Dual Momentum", dm_rets), ("Breakout", bo_rets)]:
        print(f"\n  --- {strat_name} ---")
        actual_sharpe = sharpe(oot_rets)

        # For permutation: on each day, randomly pick which ETF to hold
        common_dates = oot_rets.index.intersection(etf_rets.index)
        available_tickers = [t for t in tickers if t in etf_rets.columns]

        if len(common_dates) < 100:
            print(f"    Too few common dates ({len(common_dates)}), skipping")
            details[strat_name.lower().replace(" ", "_") + "_pvalue"] = None
            continue

        print(f"    Actual Sharpe: {actual_sharpe:.3f}")
        print(f"    Running {n_perms} permutations (random ETF assignment each day)...")

        null_sharpes = []
        rng = np.random.default_rng(42)

        # Pre-compute ETF returns matrix for speed
        etf_ret_matrix = etf_rets.loc[common_dates, available_tickers].values  # (n_days, n_tickers)
        n_days = len(common_dates)
        n_tickers = len(available_tickers)

        for _ in range(n_perms):
            # Random ETF assignment each day
            choices = rng.integers(0, n_tickers, size=n_days)
            perm_rets = etf_ret_matrix[np.arange(n_days), choices]
            perm_rets = perm_rets[~np.isnan(perm_rets)]
            if len(perm_rets) > 20 and np.std(perm_rets) > 0:
                null_sharpes.append(np.mean(perm_rets) / np.std(perm_rets) * np.sqrt(252))
            else:
                null_sharpes.append(0.0)

        null_sharpes = np.array(null_sharpes)
        p_value = (np.sum(null_sharpes >= actual_sharpe) + 1) / (n_perms + 1)
        null_mean = np.mean(null_sharpes)
        null_std = np.std(null_sharpes)
        null_p95 = np.percentile(null_sharpes, 95)
        null_p99 = np.percentile(null_sharpes, 99)

        key = strat_name.lower().replace(" ", "_")
        details[f"{key}_actual_sharpe"] = round(float(actual_sharpe), 3)
        details[f"{key}_pvalue"] = round(float(p_value), 4)
        details[f"{key}_null_mean"] = round(float(null_mean), 3)
        details[f"{key}_null_std"] = round(float(null_std), 3)
        details[f"{key}_null_p95"] = round(float(null_p95), 3)
        details[f"{key}_null_p99"] = round(float(null_p99), 3)

        print(f"    p-value: {p_value:.4f}")
        print(f"    Null distribution: mean={null_mean:.3f}, std={null_std:.3f}")
        print(f"    Null 95th pctile: {null_p95:.3f}, 99th: {null_p99:.3f}")

        status = "PASS" if p_value < 0.05 else "FAIL"
        print(f"    Result: {status} (p < 0.05 required)")

    # Also run a sign-flip block bootstrap (the original test used only 100, we'll do 1000)
    print("\n  --- Additional: Block sign-flip test (1000 iterations) ---")
    for strat_name, oot_rets in [("Dual Momentum", dm_rets), ("Breakout", bo_rets)]:
        dr = oot_rets.dropna().values
        actual_s = np.mean(dr) / np.std(dr) * np.sqrt(252) if np.std(dr) > 0 else 0
        block_size = 5
        n_blocks = len(dr) // block_size
        null_sharpes_sf = []
        rng2 = np.random.default_rng(123)

        for _ in range(1000):
            perm = dr.copy()
            flips = rng2.random(n_blocks) < 0.5
            for b in range(n_blocks):
                if flips[b]:
                    s = b * block_size
                    e = min(s + block_size, len(perm))
                    perm[s:e] = -perm[s:e]
            std = perm.std()
            null_sharpes_sf.append(perm.mean() / std * np.sqrt(252) if std > 0 else 0)

        null_sharpes_sf = np.array(null_sharpes_sf)
        p_sf = (np.sum(null_sharpes_sf >= actual_s) + 1) / 1001
        key = strat_name.lower().replace(" ", "_")
        details[f"{key}_signflip_pvalue"] = round(float(p_sf), 4)
        print(f"  {strat_name} sign-flip p-value: {p_sf:.4f} ({'PASS' if p_sf < 0.05 else 'FAIL'})")

    issues = []
    for key_check in ["dual_momentum_pvalue", "breakout_pvalue"]:
        pv = details.get(key_check)
        if pv is not None and pv >= 0.05:
            issues.append(f"{key_check}: p={pv:.4f} >= 0.05")
    for key_check in ["dual_momentum_signflip_pvalue", "breakout_signflip_pvalue"]:
        pv = details.get(key_check)
        if pv is not None and pv >= 0.05:
            issues.append(f"{key_check}: p={pv:.4f} >= 0.05")

    passed = len(issues) == 0
    status = "PASS" if passed else "FAIL"
    print(f"\n  >>> CHECK 3 RESULT: {status}")
    for iss in issues:
        print(f"      ISSUE: {iss}")

    return {"status": status, "passed": passed, "issues": issues, "details": details}


# ══════════════════════════════════════════════════════════════════════
# CHECK 4: SUB-PERIOD CONSISTENCY
# ══════════════════════════════════════════════════════════════════════

def check_subperiod_consistency(results, dm_rets, bo_rets):
    """Split into first 14 and last 14 folds, require Sharpe > 1.0 in both."""
    print("\n" + "=" * 72)
    print("CHECK 4: SUB-PERIOD CONSISTENCY")
    print("=" * 72)

    issues = []
    details = {}

    for strat_name, strat_key, oot_rets in [
        ("Dual Momentum", "dual_momentum", dm_rets),
        ("Breakout", "breakout", bo_rets)
    ]:
        folds = results["per_fold"][strat_key]
        valid_folds = [f for f in folds if f.get("valid", False)]

        mid = len(valid_folds) // 2
        first_half = valid_folds[:mid]
        second_half = valid_folds[mid:]

        # Get date ranges for each half
        first_end = first_half[-1]["test"].split("-")
        first_end_date = pd.Timestamp(f"{first_end[3]}-{first_end[4]}-{first_end[5]}")
        second_start = second_half[0]["test"].split("-")
        second_start_date = pd.Timestamp(f"{second_start[0]}-{second_start[1]}-{second_start[2]}")

        # Split OOT returns at the boundary
        h1_rets = oot_rets[oot_rets.index < second_start_date]
        h2_rets = oot_rets[oot_rets.index >= second_start_date]

        s1 = sharpe(h1_rets)
        s2 = sharpe(h2_rets)

        # Also compute from fold-level
        first_sharpes = [f["sharpe"] for f in first_half if "sharpe" in f]
        second_sharpes = [f["sharpe"] for f in second_half if "sharpe" in f]

        key = strat_name.lower().replace(" ", "_")
        details[f"{key}_first_half_sharpe"] = round(float(s1), 3)
        details[f"{key}_second_half_sharpe"] = round(float(s2), 3)
        details[f"{key}_first_half_days"] = len(h1_rets)
        details[f"{key}_second_half_days"] = len(h2_rets)
        details[f"{key}_first_half_fold_sharpes"] = [round(s, 3) for s in first_sharpes]
        details[f"{key}_second_half_fold_sharpes"] = [round(s, 3) for s in second_sharpes]
        details[f"{key}_first_half_mean_fold_sharpe"] = round(float(np.mean(first_sharpes)), 3)
        details[f"{key}_second_half_mean_fold_sharpe"] = round(float(np.mean(second_sharpes)), 3)

        print(f"\n  {strat_name}:")
        print(f"    First half ({len(first_half)} folds, {len(h1_rets)} days): Sharpe = {s1:.3f}")
        print(f"    Second half ({len(second_half)} folds, {len(h2_rets)} days): Sharpe = {s2:.3f}")
        print(f"    Mean fold Sharpe: H1={np.mean(first_sharpes):.3f}, H2={np.mean(second_sharpes):.3f}")

        h1_pass = s1 > 1.0
        h2_pass = s2 > 1.0
        both_pass = h1_pass and h2_pass

        status = "PASS" if both_pass else "FAIL"
        print(f"    Both > 1.0: {status}")

        if not h1_pass:
            issues.append(f"{strat_name} first half Sharpe = {s1:.3f} <= 1.0")
        if not h2_pass:
            issues.append(f"{strat_name} second half Sharpe = {s2:.3f} <= 1.0")

    passed = len(issues) == 0
    status = "PASS" if passed else "FAIL"
    print(f"\n  >>> CHECK 4 RESULT: {status}")
    for iss in issues:
        print(f"      ISSUE: {iss}")

    return {"status": status, "passed": passed, "issues": issues, "details": details}


# ══════════════════════════════════════════════════════════════════════
# CHECK 5: OUTLIER ANALYSIS
# ══════════════════════════════════════════════════════════════════════

def check_outlier_analysis(dm_rets, bo_rets):
    """Remove top 5% returns and top 10 days, check if Sharpe drops > 50%."""
    print("\n" + "=" * 72)
    print("CHECK 5: OUTLIER ANALYSIS")
    print("=" * 72)

    issues = []
    details = {}

    for strat_name, oot_rets in [("Dual Momentum", dm_rets), ("Breakout", bo_rets)]:
        key = strat_name.lower().replace(" ", "_")
        dr = oot_rets.dropna()
        original_sharpe = sharpe(dr)

        print(f"\n  {strat_name} (original Sharpe = {original_sharpe:.3f}):")

        # 5a. Remove top 5% of daily returns (by absolute value)
        threshold_5pct = np.percentile(dr.abs(), 95)
        trimmed_5pct = dr[dr.abs() <= threshold_5pct]
        sharpe_5pct = sharpe(trimmed_5pct)
        drop_5pct = (original_sharpe - sharpe_5pct) / original_sharpe * 100 if original_sharpe > 0 else 0

        details[f"{key}_sharpe_original"] = round(float(original_sharpe), 3)
        details[f"{key}_sharpe_no_top5pct"] = round(float(sharpe_5pct), 3)
        details[f"{key}_drop_top5pct"] = round(float(drop_5pct), 1)
        details[f"{key}_days_removed_5pct"] = len(dr) - len(trimmed_5pct)

        print(f"    Remove top 5% |returns| (>{threshold_5pct:.4f}):")
        print(f"      Days removed: {len(dr) - len(trimmed_5pct)}")
        print(f"      Sharpe: {sharpe_5pct:.3f} (drop: {drop_5pct:.1f}%)")

        if drop_5pct > 50:
            issues.append(f"{strat_name}: removing top 5% drops Sharpe by {drop_5pct:.1f}% — outlier-driven")

        # 5b. Remove top 10 individual best days
        top10_idx = dr.nlargest(10).index
        trimmed_10 = dr.drop(top10_idx)
        sharpe_10 = sharpe(trimmed_10)
        drop_10 = (original_sharpe - sharpe_10) / original_sharpe * 100 if original_sharpe > 0 else 0

        details[f"{key}_sharpe_no_top10days"] = round(float(sharpe_10), 3)
        details[f"{key}_drop_top10days"] = round(float(drop_10), 1)

        print(f"    Remove top 10 best days:")
        print(f"      Sharpe: {sharpe_10:.3f} (drop: {drop_10:.1f}%)")

        if drop_10 > 50:
            issues.append(f"{strat_name}: removing top 10 days drops Sharpe by {drop_10:.1f}% — outlier-driven")

        # 5c. Remove top AND bottom 5% (symmetric trim)
        lo = np.percentile(dr, 5)
        hi = np.percentile(dr, 95)
        sym_trimmed = dr[(dr >= lo) & (dr <= hi)]
        sharpe_sym = sharpe(sym_trimmed)
        drop_sym = (original_sharpe - sharpe_sym) / original_sharpe * 100 if original_sharpe > 0 else 0

        details[f"{key}_sharpe_symmetric_trim_5pct"] = round(float(sharpe_sym), 3)
        details[f"{key}_drop_symmetric_trim"] = round(float(drop_sym), 1)

        print(f"    Symmetric trim (5th-95th pctile):")
        print(f"      Sharpe: {sharpe_sym:.3f} (drop: {drop_sym:.1f}%)")

        # 5d. Contribution of top 10 days to total return
        total_return = dr.sum()
        top10_return = dr.nlargest(10).sum()
        top10_contribution = top10_return / total_return * 100 if total_return != 0 else 0

        details[f"{key}_top10_days_return_contribution"] = round(float(top10_contribution), 1)
        print(f"    Top 10 days contribution to total return: {top10_contribution:.1f}%")

    passed = len(issues) == 0
    status = "PASS" if passed else "FAIL"
    print(f"\n  >>> CHECK 5 RESULT: {status}")
    for iss in issues:
        print(f"      ISSUE: {iss}")

    return {"status": status, "passed": passed, "issues": issues, "details": details}


# ══════════════════════════════════════════════════════════════════════
# CHECK 6: SURVIVORSHIP BIAS
# ══════════════════════════════════════════════════════════════════════

def check_survivorship_bias(results, dm_rets, bo_rets):
    """Verify no returns before TQQQ/SOXL launch dates, check crash behavior."""
    print("\n" + "=" * 72)
    print("CHECK 6: SURVIVORSHIP BIAS")
    print("=" * 72)

    issues = []
    details = {}

    # TQQQ launched Feb 9, 2010; SOXL launched Mar 11, 2010
    tqqq_launch = pd.Timestamp("2010-02-09")
    soxl_launch = pd.Timestamp("2010-03-11")

    for strat_name, oot_rets in [("Dual Momentum", dm_rets), ("Breakout", bo_rets)]:
        key = strat_name.lower().replace(" ", "_")
        first_date = oot_rets.index[0]

        details[f"{key}_first_date"] = str(first_date.date())
        details[f"{key}_before_tqqq_launch"] = first_date < tqqq_launch
        details[f"{key}_before_soxl_launch"] = first_date < soxl_launch

        print(f"\n  {strat_name}:")
        print(f"    First OOT date: {first_date.date()}")
        print(f"    Before TQQQ launch (2010-02-09): {'YES - ISSUE' if first_date < tqqq_launch else 'No'}")
        print(f"    Before SOXL launch (2010-03-11): {'YES - ISSUE' if first_date < soxl_launch else 'No'}")

        if first_date < tqqq_launch:
            issues.append(f"{strat_name}: OOT returns start before TQQQ launch date")

    # Check behavior during known crashes
    crash_periods = [
        ("COVID crash", "2020-02-19", "2020-03-23"),
        ("2022 bear market", "2022-01-03", "2022-10-12"),
        ("2018 Q4 selloff", "2018-09-20", "2018-12-24"),
        ("2015 Aug flash crash", "2015-08-17", "2015-08-25"),
    ]

    print("\n  Behavior during known market crashes:")
    crash_details = {}

    for crash_name, start, end in crash_periods:
        s = pd.Timestamp(start)
        e = pd.Timestamp(end)

        for strat_name, oot_rets in [("Dual Momentum", dm_rets), ("Breakout", bo_rets)]:
            key = strat_name.lower().replace(" ", "_")
            mask = (oot_rets.index >= s) & (oot_rets.index <= e)
            crash_rets = oot_rets[mask]

            if len(crash_rets) > 0:
                crash_return = (1 + crash_rets).prod() - 1
                crash_dd = ((1 + crash_rets).cumprod().cummax() - (1 + crash_rets).cumprod()).max()
                crash_details[f"{key}_{crash_name}"] = {
                    "total_return": round(float(crash_return * 100), 2),
                    "max_drawdown": round(float(crash_dd * 100), 2),
                    "days": len(crash_rets),
                }
                print(f"    {crash_name} ({strat_name}): return={crash_return*100:.2f}%, "
                      f"max_dd={crash_dd*100:.2f}%, days={len(crash_rets)}")
            else:
                print(f"    {crash_name} ({strat_name}): no OOT data in this period")

    details["crash_behavior"] = crash_details

    passed = len(issues) == 0
    status = "PASS" if passed else "FAIL"
    print(f"\n  >>> CHECK 6 RESULT: {status}")
    for iss in issues:
        print(f"      ISSUE: {iss}")

    return {"status": status, "passed": passed, "issues": issues, "details": details}


# ══════════════════════════════════════════════════════════════════════
# CHECK 7: REGIME STRATIFICATION (R1)
# ══════════════════════════════════════════════════════════════════════

def check_regime_stratification(dm_rets, bo_rets):
    """R1 test: stratify by SPY green/red/flat days, report gap metric and ETF allocation."""
    print("\n" + "=" * 72)
    print("CHECK 7: REGIME STRATIFICATION (R1)")
    print("=" * 72)

    issues = []
    details = {}

    print("  Downloading SPY data for regime classification...")
    try:
        spy_data = yf.download("SPY", start="2012-01-01", end="2026-08-01",
                               auto_adjust=True, progress=False)
        spy_close = spy_data["Close"].squeeze()
        spy_ret = spy_close.pct_change()
    except Exception as e:
        print(f"  ERROR: {e}")
        return {"status": "ERROR", "passed": False, "issues": [str(e)], "details": {}}

    for strat_name, oot_rets in [("Dual Momentum", dm_rets), ("Breakout", bo_rets)]:
        key = strat_name.lower().replace(" ", "_")
        common = oot_rets.index.intersection(spy_ret.index)

        if len(common) < 100:
            print(f"  {strat_name}: insufficient common dates ({len(common)})")
            continue

        dr = oot_rets.loc[common]
        sr = spy_ret.loc[common]

        # Classify regime
        green = dr[sr > 0.001]
        red = dr[sr < -0.001]
        flat = dr[(sr >= -0.001) & (sr <= 0.001)]

        s_green = sharpe(green) if len(green) > 20 else 0.0
        s_red = sharpe(red) if len(red) > 20 else 0.0
        s_flat = sharpe(flat) if len(flat) > 20 else 0.0

        denom = max(abs(s_green), abs(s_red), 0.001)
        regime_gap = abs(s_green - s_red) / denom

        details[f"{key}_sharpe_green"] = round(float(s_green), 3)
        details[f"{key}_sharpe_red"] = round(float(s_red), 3)
        details[f"{key}_sharpe_flat"] = round(float(s_flat), 3)
        details[f"{key}_regime_gap"] = round(float(regime_gap), 3)
        details[f"{key}_n_green"] = len(green)
        details[f"{key}_n_red"] = len(red)
        details[f"{key}_n_flat"] = len(flat)
        details[f"{key}_pct_green"] = round(len(green) / len(common) * 100, 1)
        details[f"{key}_pct_red"] = round(len(red) / len(common) * 100, 1)
        details[f"{key}_pct_flat"] = round(len(flat) / len(common) * 100, 1)

        # Mean daily return by regime
        details[f"{key}_mean_return_green"] = round(float(green.mean() * 100), 4)
        details[f"{key}_mean_return_red"] = round(float(red.mean() * 100), 4)
        details[f"{key}_mean_return_flat"] = round(float(flat.mean() * 100), 4)

        gap_pass = regime_gap <= 0.50
        status = "PASS" if gap_pass else "FAIL"

        print(f"\n  {strat_name}:")
        print(f"    Green days (SPY > +0.1%): n={len(green)} ({len(green)/len(common)*100:.1f}%), "
              f"Sharpe={s_green:.3f}, mean_ret={green.mean()*100:.4f}%")
        print(f"    Red days (SPY < -0.1%):   n={len(red)} ({len(red)/len(common)*100:.1f}%), "
              f"Sharpe={s_red:.3f}, mean_ret={red.mean()*100:.4f}%")
        print(f"    Flat days:                n={len(flat)} ({len(flat)/len(common)*100:.1f}%), "
              f"Sharpe={s_flat:.3f}, mean_ret={flat.mean()*100:.4f}%")
        print(f"    Regime gap: {regime_gap:.3f} (threshold <= 0.50) [{status}]")

        if not gap_pass:
            issues.append(f"{strat_name}: regime gap = {regime_gap:.3f} > 0.50 — regime-dependent, not edge")

        # Check if strategy makes money on BOTH green and red days
        if s_green > 0 and s_red > 0:
            print(f"    GOOD: Positive Sharpe on both green and red days")
        elif s_green > 0 and s_red <= 0:
            print(f"    NOTE: Only profitable on green days — may be leveraged beta")
        elif s_green <= 0 and s_red > 0:
            print(f"    NOTE: Only profitable on red days — unusual, investigate")

    passed = len(issues) == 0
    status = "PASS" if passed else "FAIL"
    print(f"\n  >>> CHECK 7 RESULT: {status}")
    for iss in issues:
        print(f"      ISSUE: {iss}")

    return {"status": status, "passed": passed, "issues": issues, "details": details}


# ══════════════════════════════════════════════════════════════════════
# CHECK 8: DRAWDOWN ANALYSIS
# ══════════════════════════════════════════════════════════════════════

def check_drawdown_analysis(dm_rets, bo_rets):
    """Find 5 worst drawdowns, their dates, magnitude, and recovery time."""
    print("\n" + "=" * 72)
    print("CHECK 8: DRAWDOWN ANALYSIS")
    print("=" * 72)

    details = {}

    # Download SPY for context
    try:
        spy_data = yf.download("SPY", start="2012-01-01", end="2026-08-01",
                               auto_adjust=True, progress=False)
        spy_close = spy_data["Close"].squeeze()
        spy_ret = spy_close.pct_change()
    except:
        spy_ret = None

    for strat_name, oot_rets in [("Dual Momentum", dm_rets), ("Breakout", bo_rets)]:
        key = strat_name.lower().replace(" ", "_")
        dr = oot_rets.dropna()
        cum = (1 + dr).cumprod()
        running_max = cum.cummax()
        drawdown = (cum - running_max) / running_max

        # Find drawdown periods
        in_dd = drawdown < 0
        dd_groups = []
        current_dd_start = None
        current_dd_trough = 0
        current_dd_trough_date = None

        for i, (date, dd_val) in enumerate(drawdown.items()):
            if dd_val < 0:
                if current_dd_start is None:
                    current_dd_start = date
                if dd_val < current_dd_trough:
                    current_dd_trough = dd_val
                    current_dd_trough_date = date
            else:
                if current_dd_start is not None:
                    dd_groups.append({
                        "start": current_dd_start,
                        "trough_date": current_dd_trough_date,
                        "end": date,
                        "magnitude": current_dd_trough,
                        "duration_days": (date - current_dd_start).days,
                    })
                    current_dd_start = None
                    current_dd_trough = 0
                    current_dd_trough_date = None

        # If still in drawdown at end
        if current_dd_start is not None:
            dd_groups.append({
                "start": current_dd_start,
                "trough_date": current_dd_trough_date,
                "end": dr.index[-1],
                "magnitude": current_dd_trough,
                "duration_days": (dr.index[-1] - current_dd_start).days,
                "still_in_dd": True,
            })

        # Sort by magnitude (worst first)
        dd_groups.sort(key=lambda x: x["magnitude"])
        top5 = dd_groups[:5]

        print(f"\n  {strat_name} — Top 5 Drawdowns:")
        dd_list = []
        for j, dd in enumerate(top5):
            spy_context = ""
            if spy_ret is not None:
                spy_mask = (spy_ret.index >= dd["start"]) & (spy_ret.index <= dd.get("end", dd["trough_date"]))
                spy_period_ret = (1 + spy_ret[spy_mask]).prod() - 1
                spy_context = f", SPY={spy_period_ret*100:+.2f}%"

            recovery = "ongoing" if dd.get("still_in_dd") else f"{dd['duration_days']}d"
            print(f"    #{j+1}: {dd['magnitude']*100:.2f}% | "
                  f"{dd['start'].date()} to {dd.get('end', dd['trough_date']).date()} | "
                  f"Recovery: {recovery}{spy_context}")

            dd_list.append({
                "rank": j + 1,
                "magnitude_pct": round(float(dd["magnitude"] * 100), 2),
                "start": str(dd["start"].date()),
                "trough": str(dd["trough_date"].date()),
                "end": str(dd.get("end", dd["trough_date"]).date()),
                "duration_calendar_days": dd["duration_days"],
                "still_in_drawdown": dd.get("still_in_dd", False),
            })

        details[f"{key}_top5_drawdowns"] = dd_list
        details[f"{key}_total_drawdown_periods"] = len(dd_groups)

    print(f"\n  >>> CHECK 8: INFORMATIONAL (no pass/fail)")

    return {"status": "INFO", "passed": True, "issues": [], "details": details}


# ══════════════════════════════════════════════════════════════════════
# OVERALL VERDICT
# ══════════════════════════════════════════════════════════════════════

def overall_verdict(check_results):
    """Compute final verdict from all checks."""
    print("\n" + "=" * 72)
    print("OVERALL VERDICT")
    print("=" * 72)

    n_checks = len(check_results)
    n_pass = sum(1 for c in check_results.values() if c["passed"])
    n_fail = sum(1 for c in check_results.values() if not c["passed"] and c["status"] != "INFO")
    n_warn = sum(1 for c in check_results.values() if c["status"] == "WARNING")

    all_issues = []
    for name, result in check_results.items():
        for iss in result.get("issues", []):
            all_issues.append(f"[{name}] {iss}")

    print(f"\n  Checks passed: {n_pass}/{n_checks}")
    print(f"  Checks failed: {n_fail}")
    print(f"  Warnings: {n_warn}")
    print(f"  Total issues: {len(all_issues)}")

    if all_issues:
        print("\n  All issues:")
        for iss in all_issues:
            print(f"    - {iss}")

    # Determine verdict
    critical_fails = sum(1 for c in check_results.values()
                        if not c["passed"] and c["status"] == "FAIL")

    if critical_fails == 0 and n_warn <= 1:
        verdict = "VERIFIED"
        explanation = "All critical checks pass. Strategy edge appears real within walk-forward framework."
    elif critical_fails <= 1:
        verdict = "SUSPICIOUS"
        explanation = (f"{critical_fails} critical check(s) failed. Edge may be partially real but has "
                      "vulnerabilities that need investigation.")
    else:
        verdict = "REJECTED"
        explanation = f"{critical_fails} critical checks failed. Edge is likely not real or is unreliable."

    print(f"\n  {'='*50}")
    print(f"  VERDICT: {verdict}")
    print(f"  {explanation}")
    print(f"  {'='*50}")

    return verdict, explanation, all_issues


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 72)
    print("DEEP ADVERSARIAL VALIDATION")
    print("Walk-Forward Growth Strategies")
    print(f"Started: {pd.Timestamp.now()}")
    print("=" * 72)

    # Load data
    results, dm_rets, bo_rets = load_all()
    print(f"Loaded: {len(dm_rets)} DM days, {len(bo_rets)} BO days, {results['config']['n_folds']} folds")

    # Run all checks
    check_results = {}

    check_results["1_data_integrity"] = check_data_integrity(results, dm_rets, bo_rets)
    check_results["2_parameter_leakage"] = check_parameter_leakage(results)
    check_results["3_permutation_test"] = check_permutation_test(dm_rets, bo_rets, n_perms=1000)
    check_results["4_subperiod_consistency"] = check_subperiod_consistency(results, dm_rets, bo_rets)
    check_results["5_outlier_analysis"] = check_outlier_analysis(dm_rets, bo_rets)
    check_results["6_survivorship_bias"] = check_survivorship_bias(results, dm_rets, bo_rets)
    check_results["7_regime_stratification"] = check_regime_stratification(dm_rets, bo_rets)
    check_results["8_drawdown_analysis"] = check_drawdown_analysis(dm_rets, bo_rets)

    # Overall verdict
    verdict, explanation, all_issues = overall_verdict(check_results)

    # Save results
    elapsed = time.time() - t0
    output = {
        "generated": pd.Timestamp.now().isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "verdict": verdict,
        "explanation": explanation,
        "all_issues": all_issues,
        "checks": {name: {k: v for k, v in result.items()} for name, result in check_results.items()},
    }

    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to {OUT_PATH}")
    print(f"Elapsed: {elapsed:.0f}s ({elapsed/60:.1f}m)")


if __name__ == "__main__":
    main()
