"""
Deflated Sharpe Ratio (DSR) Analysis
Bailey & Lopez de Prado (2014): "The Deflated Sharpe Ratio: Correcting for
Selection Bias, Backtest Overfitting, and Non-Normality"

DSR answers: given N strategies tested, what is the probability that the best
observed Sharpe ratio is real (not due to multiple-testing luck)?

Reference: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551
"""

import json
import numpy as np
from scipy import stats
import math


# ─── DSR Formula ────────────────────────────────────────────────────────────

def expected_max_sharpe(N: int, T: int) -> float:
    """
    E[max SR] under the null (all strategies have SR=0) given N independent
    strategies and T observations each.

    Bailey & Lopez de Prado Eq. 4:
        E[max SR] ≈ (1 - γ) * Φ⁻¹(1 - 1/N) + γ * Φ⁻¹(1 - 1/(N·e))
    where γ = Euler-Mascheroni constant ≈ 0.5772156649

    This is the expected maximum of N standard normal draws, scaled by 1/√T
    for annualized SR. Since daily Sharpe ratios already embed T, we compute
    the expected max of a standard normal and then interpret relative to SE.

    For daily Sharpes with T days:
        SR_expected_max (annualized) = E[Z_max] * sqrt(252)
    where E[Z_max] is the expected max of N unit-normal draws.

    We return E[Z_max] so it can be compared on the same scale as the
    annualized Sharpe ratios from the sweep.
    """
    gamma = 0.5772156649  # Euler-Mascheroni constant

    # Expected max of N iid standard normals (Gumbel approximation)
    # Bailey & LdP Eq. 4
    e_max = ((1 - gamma) * stats.norm.ppf(1 - 1 / N)
             + gamma * stats.norm.ppf(1 - 1 / (N * math.e)))

    # Scale to annualized Sharpe (daily Sharpe * sqrt(252))
    # The sharpe_daily in our data is already annualized (daily mean/daily std * sqrt(252))
    # e_max above is in units of std devs of the SR estimator distribution
    # SR std error = sqrt((1 + 0.5*SR²) / T) ≈ 1/sqrt(T) for SR near 0
    # So E[max SR_annualized] = e_max / sqrt(T) * sqrt(252)  [annualized]
    # But our SR is already daily * sqrt(252), so:
    e_max_annualized = e_max * math.sqrt(252 / T)

    return e_max, e_max_annualized


def dsr(sr_best: float, sr_mean_null: float, T: int, V: float = 0.0,
        skew: float = 0.0, kurt: float = 3.0) -> dict:
    """
    Compute the Deflated Sharpe Ratio.

    Args:
        sr_best:     Best observed annualized Sharpe ratio
        sr_mean_null: Expected max Sharpe under null (E[max SR])
        T:           Number of observations (trading days)
        V:           Variance of the SR estimator (0 = use analytical SE)
        skew:        Return skewness (for non-normality correction)
        kurt:        Return excess kurtosis (for non-normality correction)

    Returns dict with DSR p-value and related stats.

    Bailey & LdP Eq. 8:
        DSR = Φ( (SR_hat - E[max SR*]) / sqrt(V[SR_hat]) )

    where V[SR_hat] = (1/T) * (1 + 0.5*SR_hat² - skew*SR_hat + (kurt-1)/4*SR_hat²)
    simplified for normal returns to: V = (1 + 0.5*SR²) / T
    """
    # Annualized SE of SR estimate
    # Full non-normality correction (Bailey & LdP Eq. 7):
    # V[SR] = (1/T) * (1 + 0.5*SR² - skew*SR + (kurt-1)/4 * SR²)
    sr_daily = sr_best / math.sqrt(252)  # convert back to per-period
    var_sr = (1 / T) * (1 - skew * sr_daily + (kurt - 1) / 4 * sr_daily**2
                        + 0.5 * sr_daily**2)
    se_sr = math.sqrt(var_sr) * math.sqrt(252)  # annualized SE

    # DSR = Φ( (SR_best - E[max SR*]) / SE[SR] )
    z = (sr_best - sr_mean_null) / se_sr
    dsr_pval = 1 - stats.norm.cdf(z)  # p-value: P(SR ≤ SR_best | H0)
    # Equivalently, DSR = Φ(z) is the probability that the true SR > E[max SR_null]
    dsr_stat = stats.norm.cdf(z)

    return {
        "sr_best": sr_best,
        "sr_expected_max_null": sr_mean_null,
        "se_sr": se_sr,
        "z_score": z,
        "dsr_stat": dsr_stat,       # Prob(SR* > null max) — higher = more significant
        "p_value": dsr_pval,        # H0: best SR is null luck — lower = more significant
        "passes_5pct": dsr_pval < 0.05,
        "passes_1pct": dsr_pval < 0.01,
    }


def annualize_sharpe(sharpe_daily_stat: float) -> float:
    """
    The 'sharpe_daily' field in our JSON is computed as:
        mean(daily_pnl) / std(daily_pnl) * sqrt(252)
    This is already annualized. Return as-is.
    """
    return sharpe_daily_stat


# ─── Analysis Functions ──────────────────────────────────────────────────────

def analyze_sweep(label: str, sharpes: list, T: int, N_search_space: int,
                  skew: float = 0.0, kurt: float = 3.0):
    """Run full DSR analysis for a sweep."""

    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")

    valid = [s for s in sharpes if s is not None and not math.isnan(s)]
    print(f"\n  Configs evaluated : {len(valid)}")
    print(f"  Search space (N)  : {N_search_space:,}  (total configs in IS sweep)")
    print(f"  T (trading days)  : {T}")
    print(f"  Sharpe range      : [{min(valid):.3f}, {max(valid):.3f}]")
    print(f"  Sharpe mean       : {np.mean(valid):.3f}")
    print(f"  Sharpe median     : {np.median(valid):.3f}")
    print(f"  Configs > 0       : {sum(s > 0 for s in valid)}/{len(valid)} ({100*sum(s>0 for s in valid)/len(valid):.1f}%)")
    print(f"  Configs > 1       : {sum(s > 1 for s in valid)}/{len(valid)}")
    print(f"  Configs > 2       : {sum(s > 2 for s in valid)}/{len(valid)}")

    sr_best = max(valid)
    print(f"\n  Best Sharpe       : {sr_best:.4f}")

    # Expected max under null for N_search_space independent strategies
    e_max_z, e_max_sr = expected_max_sharpe(N_search_space, T)
    print(f"\n  ─── Expected Max Sharpe Under H₀ (N={N_search_space:,}) ───")
    print(f"  E[max Z] (std normals)  : {e_max_z:.4f}")
    print(f"  E[max SR] annualized    : {e_max_sr:.4f}")
    print(f"  Interpretation: with {N_search_space:,} random strategies and {T} days,")
    print(f"  you'd expect the best Sharpe ≈ {e_max_sr:.2f} by pure luck.")

    # DSR
    result = dsr(sr_best, e_max_sr, T, skew=skew, kurt=kurt)

    print(f"\n  ─── Deflated Sharpe Ratio ───")
    print(f"  SR_best              : {result['sr_best']:.4f}")
    print(f"  SR_expected_max_null : {result['sr_expected_max_null']:.4f}")
    print(f"  SE(SR)               : {result['se_sr']:.4f}")
    print(f"  Z-score              : {result['z_score']:.4f}")
    print(f"  DSR statistic        : {result['dsr_stat']:.4f}  (Φ(z))")
    print(f"  p-value              : {result['p_value']:.6f}")
    print(f"\n  ✓ PASSES 5% significance : {result['passes_5pct']}")
    print(f"  ✓ PASSES 1% significance : {result['passes_1pct']}")

    # Also compute for ACTUAL N_tested (just the configs in this file)
    N_tested = len(valid)
    if N_tested != N_search_space:
        e_max_z2, e_max_sr2 = expected_max_sharpe(N_tested, T)
        result2 = dsr(sr_best, e_max_sr2, T, skew=skew, kurt=kurt)
        print(f"\n  ─── Sensitivity: Using N={N_tested} (only configs in this file) ───")
        print(f"  E[max SR] null       : {e_max_sr2:.4f}")
        print(f"  Z-score              : {result2['z_score']:.4f}")
        print(f"  p-value              : {result2['p_value']:.6f}")
        print(f"  PASSES 5%            : {result2['passes_5pct']}")

    # Minimum SR to pass DSR at various levels
    print(f"\n  ─── Minimum SR to Beat Null (for N={N_search_space:,}) ───")
    for alpha in [0.10, 0.05, 0.01]:
        z_thresh = stats.norm.ppf(1 - alpha)
        sr_min = e_max_sr + z_thresh * result['se_sr']
        print(f"  α={alpha:.2f} → need SR ≥ {sr_min:.4f}  (current: {sr_best:.4f}, "
              f"{'PASS' if sr_best >= sr_min else 'FAIL'})")

    return result


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*70)
    print("  DEFLATED SHARPE RATIO ANALYSIS")
    print("  Bailey & Lopez de Prado (2014)")
    print("  Multiple-testing correction for strategy selection bias")
    print("="*70)

    # ── 1. OOT CNN Sweep (cnn_oot_sim_results_20260312_083936.json) ──────────
    oot_path = ("C:/Users/Footb/Documents/Github/Lvl3Quant/alpha_discovery/"
                "results/cnn_oot_sim_results_20260312_083936.json")

    with open(oot_path) as f:
        oot_data = json.load(f)

    # Data is a list of configs
    oot_sharpes = [r["sharpe_daily"] for r in oot_data if "sharpe_daily" in r]
    oot_T = oot_data[0]["n_days"]  # 13 OOT days
    oot_configs = oot_data

    # Print top configs for reference
    print(f"\n  OOT file: {len(oot_sharpes)} configs over {oot_T} days")
    sorted_oot = sorted(oot_configs, key=lambda x: x["sharpe_daily"], reverse=True)
    print(f"\n  Top 5 OOT configs:")
    for r in sorted_oot[:5]:
        print(f"    {r['config']}: SR={r['sharpe_daily']:.3f}, "
              f"PnL=${r['total_pnl']:.0f}, trades={r['n_trades']}, "
              f"fill={r['fill_rate']:.1%}")

    # CRITICAL NOTE: The OOT file has 60 configs, but these were selected from
    # the IS sweep of 3,996 configs. The DSR must use N=3,996 as the search
    # space — that's how many strategies "competed" before OOT validation.
    # Using N=60 would be wrong (the selection already happened).

    N_IS_SWEEP = 3996  # confirmed from MEMORY.md "3,996 jobs finished"

    analyze_sweep(
        label="OOT CNN Sweep (cnn_oot_sim_results_20260312_083936.json)",
        sharpes=oot_sharpes,
        T=oot_T,
        N_search_space=N_IS_SWEEP,
        skew=0.0,    # assume roughly normal daily PnL
        kurt=3.0,    # normal kurtosis
    )

    # ── 2. Walk-Forward Fill Sim Results ─────────────────────────────────────
    wf_path = ("C:/Users/Footb/Documents/Github/Lvl3Quant/alpha_discovery/"
               "deep_models/results/wf_fill_sim_results.json")

    with open(wf_path) as f:
        wf_raw = json.load(f)

    wf_configs = wf_raw["configs"]
    wf_sharpes = [r["sharpe_daily"] for r in wf_configs if "sharpe_daily" in r]
    wf_T = wf_configs[0]["n_days"]  # 22 WF days

    print(f"\n\n  WF file: {len(wf_sharpes)} configs over {wf_T} days")
    sorted_wf = sorted(wf_configs, key=lambda x: x["sharpe_daily"], reverse=True)
    print(f"\n  Top 5 WF configs:")
    for r in sorted_wf[:5]:
        print(f"    {r['config']}: SR={r['sharpe_daily']:.3f}, "
              f"PnL=${r['total_pnl']:.0f}, trades={r['n_trades']}")

    # For the WF results, the search space is smaller:
    # These 21 configs are a subset of the OOT/IS sweep configs.
    # However, the selection bias that matters is the N from which we're
    # choosing — if we're reporting the BEST of 21 WF configs, that's N=21.
    # But if these 21 were pre-selected from 3,996, then N=3,996.
    # We'll show BOTH.

    analyze_sweep(
        label="Walk-Forward Fill Sim (wf_fill_sim_results.json)",
        sharpes=wf_sharpes,
        T=wf_T,
        N_search_space=N_IS_SWEEP,  # conservative: selected from full IS sweep
        skew=0.0,
        kurt=3.0,
    )

    # ── 3. IS sweep context: debiased_full_sweep (128 configs, 74 IS days) ──
    debiased_path = ("C:/Users/Footb/Documents/Github/Lvl3Quant/alpha_discovery/"
                     "results/debiased_full_sweep_20260308_041100.json")
    with open(debiased_path) as f:
        db_raw = json.load(f)

    db_results = db_raw["all_results"]
    db_sharpes = [r["sharpe"] for r in db_results if "sharpe" in r]
    db_T = db_raw["oos_days"]

    sorted_db = sorted(db_results, key=lambda x: x["sharpe"], reverse=True)
    print(f"\n\n  Debiased IS sweep: {len(db_sharpes)} configs over {db_T} IS days")
    print(f"\n  Top 5 IS debiased configs:")
    for r in sorted_db[:5]:
        print(f"    {r['config']}: SR={r['sharpe']:.3f}, "
              f"PnL_ticks={r['pnl_ticks']:.0f}, trades={r['trades']}")

    analyze_sweep(
        label="IS Debiased Full Sweep (128 configs, 74 IS days)",
        sharpes=db_sharpes,
        T=db_T,
        N_search_space=N_IS_SWEEP,  # 128 are subset of 3996
        skew=0.0,
        kurt=3.0,
    )

    # ── 4. Summary Table ─────────────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"\n  The DSR corrects for selection bias across N strategies.")
    print(f"  Key insight: with N=3,996 strategies, random chance produces")
    _, e_max_sr_3996 = expected_max_sharpe(3996, 13)
    _, e_max_sr_3996_22 = expected_max_sharpe(3996, 22)
    _, e_max_sr_3996_74 = expected_max_sharpe(3996, 74)
    print(f"  E[max SR] ≈ {e_max_sr_3996:.2f} (13 days), "
          f"{e_max_sr_3996_22:.2f} (22 days), "
          f"{e_max_sr_3996_74:.2f} (74 days) by pure luck.")
    print(f"\n  IMPORTANT CAVEAT: The DSR assumes independent strategies.")
    print(f"  Our configs share the SAME underlying model (BookSpatialCNN)")
    print(f"  and differ only in execution parameters (vol threshold, conv,")
    print(f"  hold time, latency). They are HIGHLY correlated.")
    print(f"  The effective N is much smaller than 3,996.")
    print(f"  A conservative effective N estimate:")
    print(f"    ~4 vol levels × ~3 conv levels × ~3 hold times = ~36 unique strategies")
    _, e_max_effective = expected_max_sharpe(36, 13)
    print(f"  E[max SR] under effective N=36, T=13: {e_max_effective:.4f}")
    print()


if __name__ == "__main__":
    main()
