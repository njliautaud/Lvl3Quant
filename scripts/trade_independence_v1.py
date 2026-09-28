"""
Trade Independence / Autocorrelation Analysis — Meta Production v1 (Shorts)
============================================================================
Investigates whether consecutive trade outcomes are independent.
Critical for position sizing: correlated trades mean effective N < raw N.

Applies meta top-50% filter (top 50% by prediction score, shorts = most negative).
Actuals are ALREADY net of 0.376 ticks passive commission.
"""

import numpy as np
import json
import os
from pathlib import Path
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from collections import defaultdict

OUT_DIR = Path("/home/jupiter/Lvl3Quant/output/trade_independence_v1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────
data = np.load("/home/jupiter/Lvl3Quant/output/meta_production_v1/concat_predictions.npz")
predictions = data["predictions"]
actuals = data["actuals"]

with open("/home/jupiter/Lvl3Quant/output/meta_production_v1/results.json") as f:
    results = json.load(f)

per_fold = results["per_fold"]
n_total = len(predictions)
print(f"Total predictions: {n_total}")
print(f"Folds: {len(per_fold)}")

# ── Apply top-50% filter (shorts = most negative predictions) ──────────────
# For shorts, the strongest signals are the most negative predictions
# Top 50% = bottom 50% of prediction scores
cutoff = np.percentile(predictions, 50)  # median
mask = predictions <= cutoff
preds_filt = predictions[mask]
acts_filt = actuals[mask]
n_filt = len(acts_filt)
print(f"\nTop-50% shorts filter: {n_filt} trades (cutoff <= {cutoff:.4f})")
print(f"Filtered mean P&L: {acts_filt.mean():.4f} ticks")
print(f"Filtered WR: {(acts_filt > 0).mean()*100:.1f}%")

# ── Build per-fold filtered arrays ─────────────────────────────────────────
fold_arrays = []
idx = 0
for fold_info in per_fold:
    n_test = fold_info["n_test"]
    fold_preds = predictions[idx:idx + n_test]
    fold_acts = actuals[idx:idx + n_test]
    fold_cutoff = np.percentile(fold_preds, 50)
    fold_mask = fold_preds <= fold_cutoff
    fold_arrays.append({
        "date": fold_info["date"],
        "preds": fold_preds[fold_mask],
        "acts": fold_acts[fold_mask],
        "all_preds": fold_preds,
        "all_acts": fold_acts,
        "n_raw": n_test,
    })
    idx += n_test

# ── 1. Autocorrelation of trade P&L at lags 1-20 ──────────────────────────
print("\n" + "="*70)
print("1. AUTOCORRELATION OF TRADE P&L (lags 1-20)")
print("="*70)

max_lag = 20
n = len(acts_filt)
mean_pnl = acts_filt.mean()
var_pnl = np.var(acts_filt)

autocorrs = []
for lag in range(1, max_lag + 1):
    if lag >= n:
        autocorrs.append(0.0)
        continue
    cov = np.mean((acts_filt[:-lag] - mean_pnl) * (acts_filt[lag:] - mean_pnl))
    ac = cov / var_pnl if var_pnl > 0 else 0.0
    autocorrs.append(ac)

# 95% confidence band for white noise
ci_95 = 1.96 / np.sqrt(n)
print(f"\n95% CI for white noise: +/- {ci_95:.4f}")
print(f"\nLag | Autocorr  | Significant?")
print("----|-----------|-------------")
sig_lags = []
for lag, ac in enumerate(autocorrs, 1):
    sig = "*" if abs(ac) > ci_95 else ""
    if sig:
        sig_lags.append(lag)
    print(f"  {lag:2d} | {ac:+.5f}  | {sig}")

print(f"\nSignificant lags (outside 95% CI): {sig_lags if sig_lags else 'NONE'}")

# ── 2. Runs Test for Randomness ───────────────────────────────────────────
print("\n" + "="*70)
print("2. RUNS TEST FOR RANDOMNESS (win/loss streaks)")
print("="*70)

wins = (acts_filt > 0).astype(int)
n_wins = wins.sum()
n_losses = n - n_wins

# Count runs
runs = 1
for i in range(1, n):
    if wins[i] != wins[i - 1]:
        runs += 1

# Expected runs and std under independence
expected_runs = 1 + (2 * n_wins * n_losses) / n
var_runs = (2 * n_wins * n_losses * (2 * n_wins * n_losses - n)) / (n**2 * (n - 1))
std_runs = np.sqrt(max(var_runs, 1e-10))
z_runs = (runs - expected_runs) / std_runs
p_runs = 2 * (1 - stats.norm.cdf(abs(z_runs)))

print(f"Total trades: {n}")
print(f"Wins: {n_wins} ({n_wins/n*100:.1f}%), Losses: {n_losses} ({n_losses/n*100:.1f}%)")
print(f"Observed runs: {runs}")
print(f"Expected runs (under independence): {expected_runs:.1f}")
print(f"Z-statistic: {z_runs:.3f}")
print(f"P-value: {p_runs:.4f}")
if z_runs < -1.96:
    print("RESULT: Significantly FEWER runs than expected -> STREAKY (trades cluster)")
elif z_runs > 1.96:
    print("RESULT: Significantly MORE runs than expected -> ALTERNATING")
else:
    print("RESULT: Runs consistent with independence (p > 0.05)")

# ── 3. Ljung-Box Test ─────────────────────────────────────────────────────
print("\n" + "="*70)
print("3. LJUNG-BOX TEST FOR SERIAL CORRELATION")
print("="*70)

lb_results = acorr_ljungbox(acts_filt, lags=20, return_df=True)
print(f"\nLag | LB Stat   | P-value  | Significant?")
print("----|-----------|----------|-------------")
lb_sig_lags = []
for lag_idx in lb_results.index:
    lb_stat = lb_results.loc[lag_idx, "lb_stat"]
    lb_pval = lb_results.loc[lag_idx, "lb_pvalue"]
    sig = "*" if lb_pval < 0.05 else ""
    if sig:
        lb_sig_lags.append(int(lag_idx))
    print(f"  {int(lag_idx):2d} | {lb_stat:9.3f} | {lb_pval:.5f} | {sig}")

print(f"\nSignificant lags (p < 0.05): {lb_sig_lags if lb_sig_lags else 'NONE'}")
lb_overall = lb_results.iloc[-1]
print(f"Overall (lag 20): LB={lb_overall['lb_stat']:.3f}, p={lb_overall['lb_pvalue']:.4f}")

# ── 4. Conditional Win Rate (Streakiness Test) ────────────────────────────
print("\n" + "="*70)
print("4. CONDITIONAL WIN RATE — STREAKINESS TEST")
print("="*70)

# P(win | prev win) vs P(win | prev loss)
prev_win_next_win = 0
prev_win_count = 0
prev_loss_next_win = 0
prev_loss_count = 0

for i in range(1, n):
    if wins[i - 1] == 1:
        prev_win_count += 1
        if wins[i] == 1:
            prev_win_next_win += 1
    else:
        prev_loss_count += 1
        if wins[i] == 1:
            prev_loss_next_win += 1

p_win_given_win = prev_win_next_win / prev_win_count if prev_win_count > 0 else 0
p_win_given_loss = prev_loss_next_win / prev_loss_count if prev_loss_count > 0 else 0
base_wr = n_wins / n

print(f"Base win rate: {base_wr*100:.2f}%")
print(f"P(win | prev win):  {p_win_given_win*100:.2f}%  (n={prev_win_count})")
print(f"P(win | prev loss): {p_win_given_loss*100:.2f}%  (n={prev_loss_count})")
print(f"Difference: {(p_win_given_win - p_win_given_loss)*100:+.2f} pp")

# Chi-square test for independence
observed = np.array([
    [prev_win_next_win, prev_win_count - prev_win_next_win],
    [prev_loss_next_win, prev_loss_count - prev_loss_next_win]
])
chi2, p_chi2, dof, expected = stats.chi2_contingency(observed)
print(f"\nChi-square test: chi2={chi2:.3f}, p={p_chi2:.4f}")
if p_chi2 < 0.05:
    print("RESULT: Consecutive outcomes are NOT independent (significant streakiness)")
else:
    print("RESULT: No significant streakiness detected")

# Extended: look at streak lengths
print("\n--- Streak Length Distribution ---")
streak_lens_w = []
streak_lens_l = []
current_streak = 1
for i in range(1, n):
    if wins[i] == wins[i - 1]:
        current_streak += 1
    else:
        if wins[i - 1] == 1:
            streak_lens_w.append(current_streak)
        else:
            streak_lens_l.append(current_streak)
        current_streak = 1
# last streak
if wins[-1] == 1:
    streak_lens_w.append(current_streak)
else:
    streak_lens_l.append(current_streak)

for label, streaks in [("Win", streak_lens_w), ("Loss", streak_lens_l)]:
    if streaks:
        arr = np.array(streaks)
        print(f"{label} streaks: mean={arr.mean():.2f}, max={arr.max()}, "
              f"p50={np.median(arr):.0f}, p90={np.percentile(arr, 90):.0f}, "
              f"count={len(arr)}")

# Expected mean streak length under independence
exp_win_streak = 1 / (1 - base_wr) if base_wr < 1 else float('inf')
exp_loss_streak = 1 / base_wr if base_wr > 0 else float('inf')
print(f"\nExpected mean win streak (under independence):  {exp_win_streak:.2f}")
print(f"Expected mean loss streak (under independence): {exp_loss_streak:.2f}")

# ── 5. Intra-day Clustering (inter-signal gap distribution) ───────────────
print("\n" + "="*70)
print("5. INTRA-DAY CLUSTERING — INTER-SIGNAL GAP DISTRIBUTION")
print("="*70)

# Reconstruct per-fold indices to measure gaps
# Since we don't have timestamps, use sequential index within each fold as proxy
# Each fold = 1 day of trading. Signals within a fold are sequential in time.
all_gaps = []
fold_signal_counts = []

for fa in fold_arrays:
    fold_n = len(fa["acts"])
    fold_signal_counts.append(fold_n)
    # Within the filtered set, compute gaps between selected signals
    # Use indices in the original fold as proxy for time
    fold_mask = fa["all_preds"] <= np.percentile(fa["all_preds"], 50)
    selected_indices = np.where(fold_mask)[0]
    if len(selected_indices) > 1:
        gaps = np.diff(selected_indices)
        all_gaps.extend(gaps.tolist())

all_gaps = np.array(all_gaps)
print(f"Total inter-signal gaps: {len(all_gaps)}")
if len(all_gaps) > 0:
    print(f"Gap distribution (in prediction indices):")
    print(f"  Mean: {all_gaps.mean():.2f}")
    print(f"  Median: {np.median(all_gaps):.1f}")
    print(f"  Std: {all_gaps.std():.2f}")
    print(f"  Min: {all_gaps.min()}, Max: {all_gaps.max()}")
    print(f"  p10: {np.percentile(all_gaps, 10):.0f}, p90: {np.percentile(all_gaps, 90):.0f}")

    # Test if gaps follow exponential (Poisson process = uniform/random spacing)
    # For a Poisson process, inter-arrival times are exponential
    # Coefficient of variation = 1 for exponential
    cv = all_gaps.std() / all_gaps.mean() if all_gaps.mean() > 0 else 0
    print(f"\n  Coefficient of Variation: {cv:.3f}")
    print(f"  (CV=1.0 for random/Poisson, CV<1 for regular, CV>1 for clustered)")
    if cv < 0.8:
        print("  RESULT: Signals are MORE REGULAR than random")
    elif cv > 1.2:
        print("  RESULT: Signals are CLUSTERED (bursty)")
    else:
        print("  RESULT: Signal spacing is approximately random")

    # Distribution of gap sizes
    print(f"\n  Gap size distribution:")
    for g in [1, 2, 3, 4, 5, 10, 20]:
        pct = (all_gaps <= g).mean() * 100
        print(f"    Gap <= {g:2d}: {pct:.1f}%")

print(f"\nSignals per fold (day):")
fc = np.array(fold_signal_counts)
print(f"  Mean: {fc.mean():.1f}, Std: {fc.std():.1f}, Min: {fc.min()}, Max: {fc.max()}")

# ── 6. Per-Fold Autocorrelation ───────────────────────────────────────────
print("\n" + "="*70)
print("6. PER-FOLD AUTOCORRELATION (is clustering fold-specific or systemic?)")
print("="*70)

fold_ac1 = []
fold_runs_z = []
print(f"\n{'Date':<12} {'N':>5} {'AC(1)':>8} {'AC(1) sig?':>10} {'Runs-Z':>8} {'Runs sig?':>10}")
print("-" * 60)

for fa in fold_arrays:
    acts_f = fa["acts"]
    nf = len(acts_f)
    date = fa["date"]

    # Lag-1 autocorrelation
    if nf > 5:
        mean_f = acts_f.mean()
        var_f = np.var(acts_f)
        if var_f > 0:
            cov_f = np.mean((acts_f[:-1] - mean_f) * (acts_f[1:] - mean_f))
            ac1 = cov_f / var_f
        else:
            ac1 = 0.0
        ci_f = 1.96 / np.sqrt(nf)
        ac1_sig = "*" if abs(ac1) > ci_f else ""
        fold_ac1.append(ac1)

        # Runs test per fold
        wins_f = (acts_f > 0).astype(int)
        nw_f = wins_f.sum()
        nl_f = nf - nw_f
        runs_f = 1
        for i in range(1, nf):
            if wins_f[i] != wins_f[i - 1]:
                runs_f += 1
        if nw_f > 0 and nl_f > 0:
            exp_r = 1 + (2 * nw_f * nl_f) / nf
            var_r = (2 * nw_f * nl_f * (2 * nw_f * nl_f - nf)) / (nf**2 * (nf - 1))
            z_r = (runs_f - exp_r) / np.sqrt(max(var_r, 1e-10))
            runs_sig = "*" if abs(z_r) > 1.96 else ""
            fold_runs_z.append(z_r)
        else:
            z_r = 0.0
            runs_sig = "N/A"
            fold_runs_z.append(0.0)

        print(f"{date:<12} {nf:5d} {ac1:+8.4f} {ac1_sig:>10} {z_r:+8.3f} {runs_sig:>10}")

fold_ac1 = np.array(fold_ac1)
fold_runs_z = np.array(fold_runs_z)
print(f"\nMean AC(1) across folds: {fold_ac1.mean():.4f} (std: {fold_ac1.std():.4f})")
print(f"Mean Runs-Z across folds: {fold_runs_z.mean():.3f} (std: {fold_runs_z.std():.3f})")
print(f"Folds with significant AC(1): {(np.abs(fold_ac1) > 1.96/np.sqrt(fc.mean())).sum()}/{len(fold_ac1)}")

# ── 7. Effective N (Practical Impact) ─────────────────────────────────────
print("\n" + "="*70)
print("7. EFFECTIVE N — PRACTICAL IMPACT OF AUTOCORRELATION")
print("="*70)

# Effective sample size: N_eff = N * (1 - rho1) / (1 + rho1)
# where rho1 = lag-1 autocorrelation
# More robust: use sum of autocorrelations
# N_eff = N / (1 + 2 * sum(rho_k for k=1..K))

rho1 = autocorrs[0]
n_eff_simple = n * (1 - rho1) / (1 + rho1) if (1 + rho1) > 0 else n

# Use first K significant autocorrelations for more robust estimate
K = min(10, max_lag)
sum_rho = sum(autocorrs[:K])
denom = 1 + 2 * sum_rho
n_eff_robust = n / denom if denom > 0 else n

print(f"Raw N: {n}")
print(f"Lag-1 autocorrelation: {rho1:.5f}")
print(f"Effective N (simple, using rho1 only): {n_eff_simple:.0f} ({n_eff_simple/n*100:.1f}% of raw)")
print(f"Sum of AC(1..{K}): {sum_rho:.5f}")
print(f"Effective N (robust, using lags 1-{K}): {n_eff_robust:.0f} ({n_eff_robust/n*100:.1f}% of raw)")

# Impact on standard error of mean P&L
se_raw = np.std(acts_filt) / np.sqrt(n)
se_eff = np.std(acts_filt) / np.sqrt(max(n_eff_robust, 1))
mean_pnl_val = acts_filt.mean()
t_raw = mean_pnl_val / se_raw if se_raw > 0 else 0
t_eff = mean_pnl_val / se_eff if se_eff > 0 else 0

print(f"\nMean P&L: {mean_pnl_val:.4f} ticks")
print(f"Std P&L: {np.std(acts_filt):.4f} ticks")
print(f"SE (raw N):       {se_raw:.5f} -> t-stat = {t_raw:.2f}")
print(f"SE (effective N): {se_eff:.5f} -> t-stat = {t_eff:.2f}")
print(f"P-value (raw):      {2*(1-stats.t.cdf(abs(t_raw), n-1)):.6f}")
print(f"P-value (effective): {2*(1-stats.t.cdf(abs(t_eff), max(n_eff_robust,2)-1)):.6f}")

# Sharpe ratio impact
daily_pnl_per_fold = [fa["acts"].sum() for fa in fold_arrays]
daily_pnl = np.array(daily_pnl_per_fold)
sharpe_daily = daily_pnl.mean() / daily_pnl.std() * np.sqrt(252) if daily_pnl.std() > 0 else 0
print(f"\nDaily P&L: mean={daily_pnl.mean():.2f}, std={daily_pnl.std():.2f}")
print(f"Annualized Sharpe (daily): {sharpe_daily:.2f}")

# ── Summary ───────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

summary = {
    "filter": "top_50pct_shorts",
    "n_total": int(n_total),
    "n_filtered": int(n_filt),
    "mean_pnl_ticks": float(mean_pnl_val),
    "win_rate": float(base_wr),
    "autocorrelation": {
        "lag_1": float(autocorrs[0]),
        "lag_2": float(autocorrs[1]),
        "lag_5": float(autocorrs[4]),
        "lag_10": float(autocorrs[9]),
        "significant_lags": sig_lags,
        "ci_95": float(ci_95),
    },
    "runs_test": {
        "observed_runs": int(runs),
        "expected_runs": float(expected_runs),
        "z_statistic": float(z_runs),
        "p_value": float(p_runs),
        "conclusion": "streaky" if z_runs < -1.96 else ("alternating" if z_runs > 1.96 else "independent"),
    },
    "ljung_box": {
        "lag20_stat": float(lb_overall["lb_stat"]),
        "lag20_pvalue": float(lb_overall["lb_pvalue"]),
        "significant_lags": lb_sig_lags,
        "conclusion": "serial_correlation" if lb_overall["lb_pvalue"] < 0.05 else "no_serial_correlation",
    },
    "conditional_win_rate": {
        "p_win_given_win": float(p_win_given_win),
        "p_win_given_loss": float(p_win_given_loss),
        "difference_pp": float((p_win_given_win - p_win_given_loss) * 100),
        "chi2": float(chi2),
        "chi2_pvalue": float(p_chi2),
        "conclusion": "streaky" if p_chi2 < 0.05 else "independent",
    },
    "effective_n": {
        "raw_n": int(n),
        "effective_n_simple": float(n_eff_simple),
        "effective_n_robust": float(n_eff_robust),
        "ratio": float(n_eff_robust / n),
        "t_stat_raw": float(t_raw),
        "t_stat_effective": float(t_eff),
    },
    "per_fold_ac1": {
        "mean": float(fold_ac1.mean()),
        "std": float(fold_ac1.std()),
        "values": fold_ac1.tolist(),
    },
    "daily_sharpe_annualized": float(sharpe_daily),
    "all_autocorrelations": [float(x) for x in autocorrs],
}

# Overall conclusion
independent = True
reasons = []
if sig_lags:
    reasons.append(f"AC significant at lags {sig_lags}")
if p_runs < 0.05:
    independent = False
    reasons.append(f"Runs test shows {'streakiness' if z_runs < 0 else 'alternation'} (p={p_runs:.4f})")
if lb_overall["lb_pvalue"] < 0.05:
    independent = False
    reasons.append(f"Ljung-Box rejects independence (p={lb_overall['lb_pvalue']:.4f})")
if p_chi2 < 0.05:
    independent = False
    reasons.append(f"Conditional WR differs (p={p_chi2:.4f})")

if independent and not sig_lags:
    conclusion = "TRADES APPEAR INDEPENDENT — safe to use raw N for position sizing"
elif independent and sig_lags:
    conclusion = f"MOSTLY INDEPENDENT — minor autocorrelation at lags {sig_lags}, effective N ~ {n_eff_robust/n*100:.0f}% of raw"
else:
    conclusion = f"TRADES SHOW DEPENDENCE — effective N = {n_eff_robust:.0f} ({n_eff_robust/n*100:.0f}% of raw). Reasons: {'; '.join(reasons)}"

summary["conclusion"] = conclusion
print(f"\n{conclusion}")

with open(OUT_DIR / "trade_independence_results.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nResults saved to {OUT_DIR / 'trade_independence_results.json'}")
