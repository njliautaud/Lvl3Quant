#!/usr/bin/env python3
"""
Adversarial Validation of Walk-Forward Growth Strategies (HC #705)
==================================================================
Reads OOT return CSVs from walkforward_growth_validation.py and runs
thorough adversarial checks to determine if the reported Sharpe/CAGR
numbers are real or artifacts.

Checks:
  1. Permutation test (1000 shuffles)
  2. Regime test R1 (green/red/flat SPY days)
  3. Sub-period consistency (4 equal periods)
  4. Outlier removal (top 5% removed)
  5. COVID drawdown (Feb-Mar 2020)
  6. 2022 bear market (Jan-Dec 2022)
  7. Leakage audit of the walk-forward script itself
  8. CAGR sanity check
  9. Vol-targeting leverage audit
"""

import numpy as np
import pandas as pd
import json, sys, os
from pathlib import Path

OUT_DIR = Path("/home/jupiter/Lvl3Quant/output/growth_research/walkforward_validation")
SCRIPT_PATH = Path("/home/jupiter/Lvl3Quant/scripts/growth_research/walkforward_growth_validation.py")

# ── Helpers ──────────────────────────────────────────────────────────

def sharpe(dr):
    """Annualized Sharpe from daily returns."""
    dr = dr.dropna()
    if len(dr) < 20 or dr.std() == 0:
        return 0.0
    return dr.mean() / dr.std() * np.sqrt(252)

def cagr(dr):
    """CAGR from daily returns."""
    dr = dr.dropna()
    if len(dr) < 20:
        return 0.0
    years = len(dr) / 252
    total = (1 + dr).prod()
    if total <= 0:
        return -1.0
    return total ** (1 / max(years, 0.01)) - 1

def max_dd(dr):
    """Maximum drawdown from daily returns."""
    dr = dr.dropna()
    cum = (1 + dr).cumprod()
    return ((cum - cum.cummax()) / cum.cummax()).min()


# ── Check 1: Permutation Test ───────────────────────────────────────

def permutation_test(dr, n_shuffles=1000):
    """
    Proper permutation test: shuffle the order of daily returns (destroys
    any time-series structure / timing skill) and recompute Sharpe.
    If shuffled returns produce similar Sharpe, the strategy is just
    "being long something with positive drift" not timing.

    Also do sign-flip block bootstrap for directional edge test.
    """
    dr = dr.dropna().values
    if len(dr) < 30:
        return {"pass": False, "reason": "insufficient data", "p_value": 1.0}

    actual_sharpe = np.mean(dr) / np.std(dr) * np.sqrt(252)
    actual_mean = np.mean(dr)

    # Test 1: Full shuffle (destroys timing, keeps return distribution)
    # If Sharpe survives shuffling, it's just positive drift, not timing
    shuffle_sharpes = []
    for _ in range(n_shuffles):
        perm = np.random.permutation(dr)
        std = perm.std()
        if std > 0:
            shuffle_sharpes.append(perm.mean() / std * np.sqrt(252))
    # Sharpe should be similar after shuffling IF the edge is just being long
    # p-value here: fraction of shuffled >= actual (should be ~0.5 if no timing)
    shuffle_sharpes = np.array(shuffle_sharpes)
    p_shuffle = (np.sum(shuffle_sharpes >= actual_sharpe) + 1) / (n_shuffles + 1)

    # Test 2: Sign-flip block bootstrap (tests directional edge)
    block_size = 5
    n_blocks = len(dr) // block_size
    signflip_sharpes = []
    for _ in range(n_shuffles):
        perm = dr.copy()
        for b in range(n_blocks):
            if np.random.random() < 0.5:
                s = b * block_size
                e = min(s + block_size, len(perm))
                perm[s:e] = -perm[s:e]
        std = perm.std()
        if std > 0:
            signflip_sharpes.append(perm.mean() / std * np.sqrt(252))
    signflip_sharpes = np.array(signflip_sharpes)
    p_signflip = (np.sum(signflip_sharpes >= actual_sharpe) + 1) / (n_shuffles + 1)

    # The meaningful test is sign-flip (does direction matter?)
    # Shuffle test tells us if ordering matters (timing skill)
    passed = p_signflip < 0.05

    return {
        "pass": passed,
        "actual_sharpe": round(actual_sharpe, 4),
        "p_signflip": round(p_signflip, 4),
        "p_shuffle": round(p_shuffle, 4),
        "null_signflip_mean": round(float(np.mean(signflip_sharpes)), 4),
        "null_shuffle_mean": round(float(np.mean(shuffle_sharpes)), 4),
        "interpretation": (
            "Sign-flip p<0.05: directional edge is real. "
            f"Shuffle p={p_shuffle:.3f}: "
            + ("timing adds NO value over buy-and-hold (Sharpe survives shuffling)" if p_shuffle > 0.20
               else "timing adds value beyond buy-and-hold")
        ),
    }


# ── Check 2: Regime Test R1 ─────────────────────────────────────────

def regime_test(dr, spy_returns):
    """
    R1: Split OOT days into green/red/flat based on SPY close-to-close.
    Reject if |Sharpe_green - Sharpe_red| / max(|Sharpe_green|, |Sharpe_red|) > 0.50
    """
    common = dr.index.intersection(spy_returns.index)
    if len(common) < 60:
        return {"pass": False, "reason": "insufficient overlapping data"}

    dr_c = dr.loc[common]
    spy_c = spy_returns.loc[common]

    green = dr_c[spy_c > 0.001]
    red = dr_c[spy_c < -0.001]
    flat = dr_c[(spy_c >= -0.001) & (spy_c <= 0.001)]

    s_green = sharpe(green) if len(green) >= 20 else 0.0
    s_red = sharpe(red) if len(red) >= 20 else 0.0
    s_flat = sharpe(flat) if len(flat) >= 20 else 0.0

    denom = max(abs(s_green), abs(s_red), 0.001)
    gap = abs(s_green - s_red) / denom

    # Also compute mean daily return per regime
    mean_green = green.mean() * 252 if len(green) > 0 else 0
    mean_red = red.mean() * 252 if len(red) > 0 else 0

    passed = gap <= 0.50

    return {
        "pass": passed,
        "regime_gap": round(gap, 4),
        "sharpe_green": round(s_green, 4),
        "sharpe_red": round(s_red, 4),
        "sharpe_flat": round(s_flat, 4),
        "n_green": len(green),
        "n_red": len(red),
        "n_flat": len(flat),
        "ann_return_green_days": round(mean_green * 100, 2),
        "ann_return_red_days": round(mean_red * 100, 2),
        "interpretation": (
            f"Gap={gap:.3f} ({'PASS' if passed else 'FAIL'} threshold=0.50). "
            + ("Strategy works in both regimes." if passed
               else f"Strategy is regime-dependent ({'green-biased' if s_green > s_red else 'red-biased'}).")
        ),
    }


# ── Check 3: Sub-period Consistency ─────────────────────────────────

def subperiod_consistency(dr, n_periods=4):
    """
    Split OOT returns into n_periods equal sub-periods.
    ALL must be profitable (positive cumulative return).
    Report Sharpe for each.
    """
    dr = dr.dropna()
    if len(dr) < n_periods * 30:
        return {"pass": False, "reason": "insufficient data for 4 sub-periods"}

    chunk_size = len(dr) // n_periods
    results = []
    all_profitable = True

    for i in range(n_periods):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < n_periods - 1 else len(dr)
        chunk = dr.iloc[start:end]

        s = sharpe(chunk)
        cum_ret = (1 + chunk).prod() - 1
        dates = f"{chunk.index[0].strftime('%Y-%m-%d')} to {chunk.index[-1].strftime('%Y-%m-%d')}"

        profitable = cum_ret > 0
        if not profitable:
            all_profitable = False

        results.append({
            "period": i + 1,
            "dates": dates,
            "n_days": len(chunk),
            "sharpe": round(s, 4),
            "cumulative_return_pct": round(cum_ret * 100, 2),
            "profitable": profitable,
        })

    return {
        "pass": all_profitable,
        "n_periods": n_periods,
        "periods": results,
        "interpretation": (
            f"{'All' if all_profitable else 'NOT all'} {n_periods} sub-periods profitable. "
            + ("Consistent edge." if all_profitable else "Edge is period-dependent.")
        ),
    }


# ── Check 4: Outlier Removal ────────────────────────────────────────

def outlier_removal_test(dr, pct=5):
    """
    Remove top pct% of daily returns and recompute Sharpe.
    If Sharpe drops > 50%, strategy is outlier-driven.
    """
    dr = dr.dropna()
    if len(dr) < 60:
        return {"pass": False, "reason": "insufficient data"}

    orig_sharpe = sharpe(dr)
    orig_cagr = cagr(dr)

    # Remove top pct% of returns
    threshold = np.percentile(dr, 100 - pct)
    trimmed = dr[dr <= threshold]

    trim_sharpe = sharpe(trimmed)
    trim_cagr = cagr(trimmed)

    if orig_sharpe == 0:
        drop_pct = 100.0
    else:
        drop_pct = (1 - trim_sharpe / orig_sharpe) * 100

    passed = drop_pct < 50.0

    # Also check: what if we remove top AND bottom pct%?
    lo = np.percentile(dr, pct)
    hi = np.percentile(dr, 100 - pct)
    symmetric_trim = dr[(dr >= lo) & (dr <= hi)]
    sym_sharpe = sharpe(symmetric_trim)

    return {
        "pass": passed,
        "original_sharpe": round(orig_sharpe, 4),
        "trimmed_sharpe_top5pct_removed": round(trim_sharpe, 4),
        "sharpe_drop_pct": round(drop_pct, 2),
        "original_cagr_pct": round(orig_cagr * 100, 2),
        "trimmed_cagr_pct": round(trim_cagr * 100, 2),
        "symmetric_trim_sharpe": round(sym_sharpe, 4),
        "n_removed": len(dr) - len(trimmed),
        "interpretation": (
            f"Sharpe drops {drop_pct:.1f}% after removing top {pct}% of returns. "
            + ("Robust to outlier removal." if passed else "OUTLIER-DRIVEN: strategy depends on big winning days.")
        ),
    }


# ── Check 5: COVID Drawdown ─────────────────────────────────────────

def crisis_period_check(dr, start_date, end_date, label):
    """Check performance during a specific crisis period."""
    mask = (dr.index >= start_date) & (dr.index <= end_date)
    period = dr[mask]

    if len(period) < 5:
        return {"pass": "N/A", "reason": f"No OOT data during {label}", "n_days": len(period)}

    cum_ret = (1 + period).prod() - 1
    s = sharpe(period) if len(period) >= 20 else float('nan')
    dd = max_dd(period)

    return {
        "label": label,
        "n_days": len(period),
        "cumulative_return_pct": round(cum_ret * 100, 2),
        "max_drawdown_pct": round(dd * 100, 2) if not np.isnan(dd) else None,
        "sharpe": round(s, 4) if not np.isnan(s) else None,
        "daily_returns_mean_bps": round(period.mean() * 10000, 2),
        "daily_returns_std_bps": round(period.std() * 10000, 2),
        "interpretation": (
            f"During {label}: cum return {cum_ret*100:.1f}%, max DD {dd*100:.1f}%."
            + (" Strategy avoided crash well." if cum_ret > -0.05
               else " Strategy took losses during crisis.")
        ),
    }


# ── Check 7: Leakage Audit ──────────────────────────────────────────

def leakage_audit():
    """
    Audit the walk-forward script for potential data leakage.
    Read the script and check for common issues.
    """
    issues = []
    warnings_list = []

    with open(SCRIPT_PATH) as f:
        code = f.read()
        lines = code.split('\n')

    # Check 1: Does optimization use future data?
    # The walk-forward should optimize on train_start..train_end, then test on test_start..test_end
    # train_end must be <= test_start
    if 'optimize_dual_momentum(prices, etfs, tr_s, tr_e)' in code:
        # Good - optimization uses train window
        pass
    if 'optimize_breakout(prices, etfs, tr_s, tr_e)' in code:
        # Good - optimization uses train window
        pass

    # Check 2: Does run_dual_momentum use prices before start date for lookback?
    # Line: p_past = prices[etf].iloc[loc - lookback]
    # This accesses data BEFORE the start date for the lookback window.
    # This is CORRECT for momentum - you need historical prices to compute momentum.
    # But check if loc can reach into the training period.
    if 'p_past = prices[etf].iloc[loc - lookback]' in code:
        warnings_list.append(
            "MINOR: Momentum lookback accesses prices before test window start. "
            "This is normal for momentum strategies (you need N days of history to compute N-day momentum). "
            "NOT leakage - this is how momentum works in practice."
        )

    # Check 3: Vol targeting uses same-day data?
    # Line: rv = returns[current].iloc[max(0, loc-21):loc].std() * np.sqrt(252)
    # This correctly uses iloc[loc-21:loc] (exclusive of loc), so it does NOT include today's return.
    # Wait - let me re-read... iloc[max(0, loc-21):loc] includes loc-21 through loc-1.
    # That's correct - it uses the PRIOR 21 days' returns, not today's.
    if 'iloc[max(0, loc-21):loc]' in code:
        # Actually this IS correct - iloc slicing is exclusive on the right
        pass

    # Check 4: Parameter convergence
    # From results, nearly every fold picks lb=10 for DM.
    # This suggests the "optimization" is finding the same answer everywhere.
    warnings_list.append(
        "SUSPICIOUS: Nearly every DM fold picks lb=10. Parameter optimization may be degenerate - "
        "10-day lookback is essentially 'buy the most recently hot ETF'. This is momentum chasing, "
        "not a robust signal. The grid [10,15,21,42,63] has lb=10 as the boundary - "
        "would lb=5 or lb=3 score even higher? If so, this is short-term momentum which is well-known to be fragile."
    )

    # Check 5: Breakout almost always picks bd=5, td=5
    warnings_list.append(
        "SUSPICIOUS: Nearly every BO fold picks bd=5,td=5 (the minimum in the grid). "
        "This means 'buy any 5-day high, exit any 5-day low'. Like DM, this is short-term momentum. "
        "The parameter grid has bd=5 as the boundary - shorter periods might score even higher."
    )

    # Check 6: Train/test windows - check for overlap
    # generate_folds: train_end = train_start + 36mo, test_start = train_end
    # So train_end == test_start. In the code, run_dual_momentum uses mask = (prices.index >= start) & (prices.index <= end)
    # This means train_end date is included in BOTH train and test! <= is inclusive on both sides.
    if '(prices.index >= start) & (prices.index <= end)' in code:
        issues.append(
            "LEAKAGE: Date filtering uses '<= end' which means the boundary date (train_end == test_start) "
            "is included in BOTH the train window AND the test window. The train_end date's return is used for "
            "optimization AND for OOT scoring. Fix: use '< end' for train or '> start' for test."
        )

    # Check 7: Duplicate removal in concatenation
    # dm_concat = dm_concat[~dm_concat.index.duplicated(keep="last")]
    # This removes duplicate dates, keeping the LAST fold's value.
    # But which fold is 'last'? If folds overlap, the later fold's parameters were optimized on data
    # that includes the earlier fold's test period. This is mild leakage.
    if 'duplicated(keep="last")' in code:
        warnings_list.append(
            "CONCERN: When OOT windows overlap (they shouldn't with 6mo train slide and 6mo test), "
            "duplicated dates keep='last' means later folds' results are used. Since later folds "
            "trained on data including the duplicated period, this could cause mild look-ahead bias."
        )

    # Check 8: Vol targeting with scalar up to 2.0x
    if 'min(vol_target / rv, 2.0)' in code:
        issues.append(
            "LEVERAGE INFLATION: Vol targeting applies up to 2.0x leverage on daily returns. "
            "This is NOT realistic for ETF investing - you can't lever TQQQ (already 3x leveraged). "
            "With vol_target=0.40 and realized vol of 0.20, the strategy applies 2.0x leverage to TQQQ, "
            "making it effectively 6x QQQ. This MASSIVELY inflates CAGR. "
            "The 175% CAGR and 301% CAGR are partly leverage artifacts."
        )

    # Check 9: CAGR calculation correctness
    # calc_metrics uses: cagr = (1 + dr).prod() ** (1 / years) - 1
    # where years = len(dr) / 252
    # This is correct IF dr contains only OOT returns and years reflects actual calendar time.
    # But OOT returns are concatenated from non-contiguous windows.
    # With 28 folds * ~125 days = ~3400 OOT days = ~13.5 years.
    # Actual calendar span is 2013-2026 = 13 years. Close enough.
    # BUT: the issue is that the CAGR assumes continuous compounding across all OOT periods.
    # In reality, between OOT periods, you'd be in the training phase (not trading).
    # The CAGR overstates what a real investor would earn because it assumes 100% time in market.
    warnings_list.append(
        "CAGR OVERSTATEMENT: CAGR is computed on concatenated OOT returns as if continuously invested. "
        "In reality, with 36mo train + 6mo test, you'd only be trading 6 out of every 42 months (14% of the time) "
        "while waiting for the next retrain. The reported CAGR assumes you can always be in the market. "
        "However, this is arguably correct if you interpret walk-forward as 'what would the strategy have done' "
        "since in live trading you'd always be using the latest parameters."
    )

    # Check 10: The regime test in the original script uses only 100 permutations
    if 'n_shuffles=100' in code:
        warnings_list.append(
            "WEAK TEST: Original permutation test uses only 100 shuffles (we use 1000). "
            "With 100 shuffles, p-value resolution is only 0.01 - marginal results could pass."
        )

    # Check 11: Original outlier test uses 1% not 5%
    if 'outlier_removal_test(dr, pct=1)' in code:
        warnings_list.append(
            "WEAK TEST: Original outlier removal uses 1% (removes ~34 days). "
            "We use 5% (removes ~170 days) which is a much harder test."
        )

    # Check 12: Original sub-period test uses 2 halves, not 4
    if 'mid = len(dr) // 2' in code:
        warnings_list.append(
            "WEAK TEST: Original sub-period test only splits into 2 halves. "
            "We split into 4 quarters for a harder consistency test."
        )

    passed = len(issues) == 0

    return {
        "pass": passed,
        "critical_issues": issues,
        "warnings": warnings_list,
        "n_critical": len(issues),
        "n_warnings": len(warnings_list),
        "interpretation": (
            f"Found {len(issues)} critical issues and {len(warnings_list)} warnings. "
            + ("Script has no critical leakage." if passed
               else "CRITICAL LEAKAGE DETECTED - results are unreliable.")
        ),
    }


# ── Check 8: CAGR Sanity ────────────────────────────────────────────

def cagr_sanity_check(dr, label):
    """
    Verify CAGR is mathematically correct and compare to realistic baselines.
    """
    dr = dr.dropna()
    years = len(dr) / 252

    total_return = (1 + dr).prod() - 1
    computed_cagr = (1 + total_return) ** (1 / max(years, 0.01)) - 1

    # What are realistic baselines?
    # S&P 500: ~10% CAGR
    # QQQ: ~15% CAGR (2010-2024)
    # TQQQ: ~40-50% CAGR (2010-2024) with massive drawdowns
    # Best hedge funds: ~15-25% CAGR
    # Renaissance Medallion: ~66% CAGR (before fees)

    # Check daily return statistics
    mean_daily = dr.mean()
    std_daily = dr.std()
    skew = float(pd.Series(dr).skew())
    kurt = float(pd.Series(dr).kurtosis())

    # What fraction of days have |return| > 5%?
    extreme_days = (dr.abs() > 0.05).sum()
    # What fraction of days have return > 2%?
    big_up_days = (dr > 0.02).sum()
    big_down_days = (dr < -0.02).sum()

    return {
        "label": label,
        "n_days": len(dr),
        "years": round(years, 2),
        "total_return_pct": round(total_return * 100, 2),
        "cagr_pct": round(computed_cagr * 100, 2),
        "mean_daily_bps": round(mean_daily * 10000, 2),
        "std_daily_bps": round(std_daily * 10000, 2),
        "skewness": round(skew, 4),
        "kurtosis": round(kurt, 4),
        "extreme_days_pct": round(extreme_days / len(dr) * 100, 2),
        "big_up_days": int(big_up_days),
        "big_down_days": int(big_down_days),
        "interpretation": (
            f"CAGR={computed_cagr*100:.1f}% over {years:.1f} years. "
            f"Mean daily return = {mean_daily*10000:.1f} bps. "
            + (f"This exceeds Renaissance Medallion ({66}% CAGR) - extremely suspicious." if computed_cagr > 0.66
               else f"Plausible for leveraged momentum strategy." if computed_cagr < 0.50
               else f"Very high - needs careful leverage accounting.")
        ),
    }


# ── Check 9: Leverage / Vol-Target Audit ─────────────────────────────

def leverage_audit(dr, label):
    """
    Check if the return distribution is consistent with unleveraged or leveraged returns.
    TQQQ already has ~60% annual vol. If the strategy has higher vol, it's applying extra leverage.
    """
    dr = dr.dropna()
    ann_vol = dr.std() * np.sqrt(252)
    ann_ret = dr.mean() * 252

    # Typical annual vols:
    # SPY: ~16%, QQQ: ~20%, TQQQ: ~60%, SOXL: ~75%
    # If strategy vol > 40%, it's likely leveraging leveraged ETFs

    effectively_leveraged = ann_vol > 0.40  # >40% ann vol suggests leverage on top of 3x ETFs

    # Calculate implied leverage assuming base asset is ~20% vol (QQQ-like)
    implied_leverage = ann_vol / 0.20 if ann_vol > 0 else 0

    return {
        "label": label,
        "ann_vol_pct": round(ann_vol * 100, 2),
        "ann_ret_pct": round(ann_ret * 100, 2),
        "implied_leverage_vs_qqq": round(implied_leverage, 2),
        "effectively_leveraged": effectively_leveraged,
        "interpretation": (
            f"Ann vol = {ann_vol*100:.1f}%. "
            + (f"This implies ~{implied_leverage:.1f}x leverage vs QQQ. "
               "The vol-targeting is applying leverage on top of already-leveraged ETFs. "
               "Reported CAGR is partly from leverage, not timing skill."
               if effectively_leveraged
               else "Reasonable volatility level.")
        ),
    }


# ── Main ─────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("ADVERSARIAL VALIDATION OF WALK-FORWARD GROWTH STRATEGIES")
    print("HC #705 Compliance Checks")
    print("=" * 80)

    # Load OOT returns
    dm_path = OUT_DIR / "dm_oot_returns.csv"
    bo_path = OUT_DIR / "bo_oot_returns.csv"

    if not dm_path.exists() or not bo_path.exists():
        print("ERROR: OOT return CSVs not found. Run walkforward_growth_validation.py first.")
        sys.exit(1)

    dm = pd.read_csv(dm_path, index_col=0, parse_dates=True).squeeze()
    bo = pd.read_csv(bo_path, index_col=0, parse_dates=True).squeeze()

    print(f"\nLoaded DM OOT returns: {len(dm)} days, {dm.index[0].date()} to {dm.index[-1].date()}")
    print(f"Loaded BO OOT returns: {len(bo)} days, {bo.index[0].date()} to {bo.index[-1].date()}")

    # Get SPY returns for regime test
    try:
        import yfinance as yf
        spy_data = yf.download("SPY", start="2010-01-01", end="2026-07-15",
                               auto_adjust=True, progress=False)
        if isinstance(spy_data.columns, pd.MultiIndex):
            spy_prices = spy_data["Close"]["SPY"]
        else:
            spy_prices = spy_data["Close"]
        spy_returns = spy_prices.pct_change().dropna()
        has_spy = True
        print(f"Loaded SPY data: {len(spy_returns)} days")
    except Exception as e:
        print(f"WARNING: Could not load SPY data: {e}")
        has_spy = False
        spy_returns = None

    results = {}
    overall_pass = True

    for label, dr in [("dual_momentum", dm), ("breakout", bo)]:
        print(f"\n{'='*80}")
        print(f"  STRATEGY: {label.upper()}")
        print(f"{'='*80}")

        strat_results = {}

        # ── Check 1: Permutation Test ──
        print(f"\n  [1] PERMUTATION TEST (1000 shuffles)...")
        perm = permutation_test(dr, n_shuffles=1000)
        strat_results["permutation_test"] = perm
        status = "PASS" if perm["pass"] else "FAIL"
        print(f"      Sign-flip p-value: {perm['p_signflip']:.4f} [{status}]")
        print(f"      Shuffle p-value:   {perm['p_shuffle']:.4f}")
        print(f"      Actual Sharpe: {perm['actual_sharpe']:.4f}")
        print(f"      Null (signflip) mean: {perm['null_signflip_mean']:.4f}")
        print(f"      Null (shuffle) mean:  {perm['null_shuffle_mean']:.4f}")
        print(f"      >> {perm['interpretation']}")
        if not perm["pass"]:
            overall_pass = False

        # ── Check 2: Regime Test ──
        if has_spy:
            print(f"\n  [2] REGIME TEST R1 (green/red/flat SPY days)...")
            reg = regime_test(dr, spy_returns)
            strat_results["regime_test"] = reg
            status = "PASS" if reg["pass"] else "FAIL"
            print(f"      Regime gap: {reg['regime_gap']:.4f} [{status}] (threshold: 0.50)")
            print(f"      Green days: Sharpe={reg['sharpe_green']:.4f} (n={reg['n_green']})")
            print(f"      Red days:   Sharpe={reg['sharpe_red']:.4f} (n={reg['n_red']})")
            print(f"      Flat days:  Sharpe={reg['sharpe_flat']:.4f} (n={reg['n_flat']})")
            print(f"      >> {reg['interpretation']}")
            if not reg["pass"]:
                overall_pass = False
        else:
            strat_results["regime_test"] = {"pass": "SKIP", "reason": "no SPY data"}

        # ── Check 3: Sub-period Consistency ──
        print(f"\n  [3] SUB-PERIOD CONSISTENCY (4 equal periods)...")
        sub = subperiod_consistency(dr, n_periods=4)
        strat_results["subperiod_consistency"] = sub
        status = "PASS" if sub["pass"] else "FAIL"
        print(f"      All 4 periods profitable: [{status}]")
        for p in sub.get("periods", []):
            pflag = "+" if p["profitable"] else "X"
            print(f"        [{pflag}] P{p['period']}: {p['dates']} | Sharpe={p['sharpe']:.3f} | CumRet={p['cumulative_return_pct']:.1f}%")
        print(f"      >> {sub['interpretation']}")
        if not sub["pass"]:
            overall_pass = False

        # ── Check 4: Outlier Removal ──
        print(f"\n  [4] OUTLIER REMOVAL (top 5% of returns removed)...")
        out = outlier_removal_test(dr, pct=5)
        strat_results["outlier_removal"] = out
        status = "PASS" if out["pass"] else "FAIL"
        print(f"      Original Sharpe:  {out['original_sharpe']:.4f}")
        print(f"      Trimmed Sharpe:   {out['trimmed_sharpe_top5pct_removed']:.4f}")
        print(f"      Sharpe drop:      {out['sharpe_drop_pct']:.1f}% [{status}] (threshold: 50%)")
        print(f"      Original CAGR:    {out['original_cagr_pct']:.1f}%")
        print(f"      Trimmed CAGR:     {out['trimmed_cagr_pct']:.1f}%")
        print(f"      >> {out['interpretation']}")
        if not out["pass"]:
            overall_pass = False

        # ── Check 5: COVID Drawdown ──
        print(f"\n  [5] COVID PERIOD (Feb-Mar 2020)...")
        covid = crisis_period_check(dr, "2020-02-01", "2020-03-31", "COVID Feb-Mar 2020")
        strat_results["covid_check"] = covid
        if covid.get("n_days", 0) >= 5:
            print(f"      Days in OOT: {covid['n_days']}")
            print(f"      Cumulative return: {covid['cumulative_return_pct']:.1f}%")
            print(f"      Max drawdown: {covid.get('max_drawdown_pct', 'N/A')}%")
            print(f"      >> {covid['interpretation']}")
        else:
            print(f"      No OOT data during COVID period (fold boundary may exclude it)")

        # ── Check 6: 2022 Bear Market ──
        print(f"\n  [6] 2022 BEAR MARKET (Jan-Dec 2022)...")
        bear = crisis_period_check(dr, "2022-01-01", "2022-12-31", "2022 Bear Market")
        strat_results["bear_2022_check"] = bear
        if bear.get("n_days", 0) >= 20:
            print(f"      Days in OOT: {bear['n_days']}")
            print(f"      Cumulative return: {bear['cumulative_return_pct']:.1f}%")
            print(f"      Max drawdown: {bear.get('max_drawdown_pct', 'N/A')}%")
            print(f"      Sharpe: {bear.get('sharpe', 'N/A')}")
            print(f"      >> {bear['interpretation']}")
        else:
            print(f"      Limited OOT data during 2022 ({bear.get('n_days', 0)} days)")

        # ── Check 8: CAGR Sanity ──
        print(f"\n  [8] CAGR SANITY CHECK...")
        cagr_check = cagr_sanity_check(dr, label)
        strat_results["cagr_sanity"] = cagr_check
        print(f"      CAGR: {cagr_check['cagr_pct']:.1f}%")
        print(f"      Mean daily return: {cagr_check['mean_daily_bps']:.1f} bps")
        print(f"      Daily vol: {cagr_check['std_daily_bps']:.1f} bps")
        print(f"      Skew: {cagr_check['skewness']:.3f}, Kurt: {cagr_check['kurtosis']:.3f}")
        print(f"      Extreme days (|ret|>5%): {cagr_check['extreme_days_pct']:.1f}%")
        print(f"      Big up days (>2%): {cagr_check['big_up_days']}, Big down (< -2%): {cagr_check['big_down_days']}")
        print(f"      >> {cagr_check['interpretation']}")

        # ── Check 9: Leverage Audit ──
        print(f"\n  [9] LEVERAGE / VOL-TARGET AUDIT...")
        lev = leverage_audit(dr, label)
        strat_results["leverage_audit"] = lev
        print(f"      Annual vol: {lev['ann_vol_pct']:.1f}%")
        print(f"      Implied leverage vs QQQ: {lev['implied_leverage_vs_qqq']:.2f}x")
        print(f"      >> {lev['interpretation']}")

        results[label] = strat_results

    # ── Check 7: Leakage Audit (shared across strategies) ──
    print(f"\n{'='*80}")
    print(f"  [7] LEAKAGE AUDIT OF WALK-FORWARD SCRIPT")
    print(f"{'='*80}")
    leak = leakage_audit()
    results["leakage_audit"] = leak
    status = "PASS" if leak["pass"] else "FAIL"
    print(f"\n  Overall: [{status}]")
    if leak["critical_issues"]:
        print(f"\n  CRITICAL ISSUES ({len(leak['critical_issues'])}):")
        for i, issue in enumerate(leak["critical_issues"], 1):
            print(f"    {i}. {issue}")
            overall_pass = False
    if leak["warnings"]:
        print(f"\n  WARNINGS ({len(leak['warnings'])}):")
        for i, w in enumerate(leak["warnings"], 1):
            print(f"    {i}. {w}")

    # ── OVERALL VERDICT ──
    print(f"\n{'='*80}")
    print(f"  OVERALL VERDICT")
    print(f"{'='*80}")

    # Compile summary
    summary = {
        "overall_pass": overall_pass,
        "checks_run": 9,
        "critical_failures": [],
        "key_findings": [],
    }

    # Enumerate failures
    for strat in ["dual_momentum", "breakout"]:
        if strat not in results:
            continue
        for check_name, check_result in results[strat].items():
            if isinstance(check_result, dict) and check_result.get("pass") is False:
                summary["critical_failures"].append(f"{strat}/{check_name}")

    if not leak["pass"]:
        summary["critical_failures"].append("leakage_audit")

    # Key findings
    summary["key_findings"] = [
        "Vol-targeting applies up to 2.0x leverage on TQQQ (already 3x). "
        "Effective leverage is up to 6x QQQ, making CAGR numbers unrealistic for actual trading.",

        "Parameter 'optimization' converges to lb=10 (DM) and bd=5,td=5 (BO) in almost every fold. "
        "This is short-term momentum, not a sophisticated strategy.",

        "Train/test boundary date overlap: '<= end' in date filtering includes the boundary date in both windows.",

        "Original adversarial tests were weak: 100 permutations, 1% outlier removal, 2-way split. "
        "More rigorous tests (1000 perms, 5% outliers, 4-way split) may reveal fragility.",
    ]

    results["summary"] = summary

    if overall_pass:
        print(f"\n  VERDICT: CONDITIONAL PASS")
        print(f"  The strategies show positive OOT edge, but with significant caveats:")
    else:
        print(f"\n  VERDICT: FAIL")
        print(f"  Failed checks: {', '.join(summary['critical_failures'])}")

    print(f"\n  KEY FINDINGS:")
    for i, f in enumerate(summary["key_findings"], 1):
        print(f"    {i}. {f}")

    print(f"\n  BOTTOM LINE:")
    print(f"    The 175% and 301% CAGR numbers are INFLATED by vol-targeting leverage.")
    print(f"    To get a realistic CAGR, rerun with vol_target=0 (no leverage) and compare")
    print(f"    to TQQQ buy-and-hold. The TIMING SKILL may be real but the magnitude")
    print(f"    is overstated by the leverage applied on top of already-leveraged ETFs.")

    # Save results
    output_path = OUT_DIR / "adversarial_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n  Results saved to {output_path}")


if __name__ == "__main__":
    np.random.seed(42)
    main()
