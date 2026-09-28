"""
Daily rvol filter analysis for WF fill_sim results.
RESEARCH ONLY - reads data, writes nothing to production.

Asks: does filtering out high-rvol days improve Sharpe for the best config?
"""

import json
import os
import numpy as np
from pathlib import Path

# ── paths ────────────────────────────────────────────────────────────────────
HMM_FILE     = r"C:\Users\Footb\Documents\Github\Lvl3Quant\alpha_discovery\deep_models\results\hmm_regime_states.json"
SIM_DIR      = r"C:\Users\Footb\Documents\Github\Lvl3Quant\data\processed\cnn_wf_sim_results"
LOG_IC_MAP   = {  # from the fill_sim log (22 WF-OOT days)
    "2025-12-01": 0.0688, "2025-12-02": 0.0629, "2025-12-03": 0.0820,
    "2025-12-04": 0.0662, "2025-12-05": 0.0784, "2025-12-08": 0.0856,
    "2025-12-09": 0.1138, "2025-12-10": 0.0657, "2025-12-11": 0.0778,
    "2025-12-12": 0.0401, "2025-12-15": 0.0548, "2025-12-16": 0.0303,
    "2025-12-17": -0.0069, "2025-12-18": 0.0088, "2025-12-19": 0.0829,
    "2025-12-22": 0.1155, "2025-12-23": 0.1551, "2025-12-24": 0.1504,
    "2025-12-26": 0.1423, "2025-12-30": 0.1295, "2025-12-31": 0.1099,
    "2026-01-02": 0.0120,
}

# Best config name (from wf_fill_sim_results.json #1 by Sharpe)
BEST_CONFIG = "vol80_morning_afternoon_chase_1t_3r_conv25_30min"
# The lat10 variant had slightly higher P&L but same underlying; use the base config
# (both lat0 and base are identical in results; lat10 has 1 less trade)

# ── load HMM rvol data ───────────────────────────────────────────────────────
with open(HMM_FILE) as f:
    hmm = json.load(f)

rvol_ticks = {}
for date, feat in hmm["oot_features"].items():
    rvol_ticks[date] = feat["daily_rvol_ticks"]

hmm_state = hmm["oot_states"]  # 1=active/low-rvol, 0=dead/high-rvol

# ── load per-day P&L for best config ────────────────────────────────────────
sim_dir = Path(SIM_DIR)

days_data = {}
for date in LOG_IC_MAP.keys():
    fname = sim_dir / f"{BEST_CONFIG}_{date}.json"
    if not fname.exists():
        print(f"  MISSING: {fname.name}")
        continue
    with open(fname) as f:
        res = json.load(f)
    rvol = rvol_ticks.get(date, None)
    state = hmm_state.get(date, None)
    days_data[date] = {
        "pnl":    res["total_pnl_dollars"],
        "trades": res["total_trades"],
        "rvol":   rvol,
        "state":  state,
        "ic":     LOG_IC_MAP[date],
    }

dates_sorted = sorted(days_data.keys())
print(f"\nLoaded {len(days_data)} days for config: {BEST_CONFIG}")

# ── helper: compute stats for a subset of days ───────────────────────────────
def stats(subset_dates, label, days_data, total_days):
    pnls = [days_data[d]["pnl"] for d in subset_dates]
    n = len(pnls)
    if n == 0:
        print(f"\n{label}: NO DAYS")
        return {}
    total_pnl  = sum(pnls)
    mean_pnl   = np.mean(pnls)
    std_pnl    = np.std(pnls, ddof=1) if n > 1 else 0.0
    sharpe     = (mean_pnl / std_pnl * np.sqrt(252)) if std_pnl > 0 else 0.0
    n_pos      = sum(1 for p in pnls if p > 0)
    n_neg      = sum(1 for p in pnls if p < 0)
    n_zero     = sum(1 for p in pnls if p == 0)
    pct_traded = n / total_days * 100
    trades     = sum(days_data[d]["trades"] for d in subset_dates)
    # annualise assuming 252 trading days, scale by fraction of days we'd trade
    # (if we skip 30% of days, annualised is based on 70% * 252)
    ann_pnl    = mean_pnl * 252 * (n / total_days)   # expected per year at this trade rate
    ann_pnl_full = mean_pnl * 252                     # if we could fill every day at this rate

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  Days traded: {n}/{total_days} ({pct_traded:.0f}%)")
    print(f"  Trades executed: {trades}")
    print(f"  Total P&L:  ${total_pnl:,.2f}")
    print(f"  Mean daily: ${mean_pnl:,.2f}  |  Std: ${std_pnl:,.2f}")
    print(f"  Sharpe (annualised, 252d): {sharpe:.3f}")
    print(f"  Win/Loss/Zero days: {n_pos}/{n_neg}/{n_zero}")
    print(f"  Ann P&L (at this trade rate, 252d): ${ann_pnl:,.0f}")
    return {"sharpe": sharpe, "n": n, "total_pnl": total_pnl, "mean_pnl": mean_pnl}


# ── per-day detail table ──────────────────────────────────────────────────────
print("\n" + "="*80)
print(f"  PER-DAY DETAIL: {BEST_CONFIG}")
print("="*80)
print(f"{'Date':<12} {'rvol_ticks':>10} {'HMM':>5} {'IC':>8} {'P&L':>10} {'Trades':>7}")
print("-"*60)
for d in dates_sorted:
    row = days_data[d]
    print(f"{d:<12} {row['rvol']:>10.4f} {str(row['state']):>5}  {row['ic']:>7.4f}  ${row['pnl']:>8,.2f}  {row['trades']:>5}")

total_days = len(dates_sorted)
all_dates  = dates_sorted

# ── 1. ALL 22 days ───────────────────────────────────────────────────────────
r_all = stats(all_dates, "1. ALL 22 DAYS (baseline)", days_data, total_days)

# ── 2. HMM filter: active (low-rvol) state only ──────────────────────────────
hmm_active_dates = [d for d in all_dates if days_data[d]["state"] == 1]
hmm_dead_dates   = [d for d in all_dates if days_data[d]["state"] == 0]
r_hmm = stats(hmm_active_dates, "2. HMM ACTIVE (state=1, low-rvol)", days_data, total_days)

# ── 3. Hard rvol threshold ≤ 0.35 ticks ──────────────────────────────────────
low35_dates  = [d for d in all_dates if days_data[d]["rvol"] <= 0.35]
r_low35 = stats(low35_dates, "3. rvol <= 0.35 ticks", days_data, total_days)

# ── 4. Hard rvol threshold <= 0.25 ticks ──────────────────────────────────────
low25_dates  = [d for d in all_dates if days_data[d]["rvol"] <= 0.25]
r_low25 = stats(low25_dates, "4. rvol <= 0.25 ticks", days_data, total_days)

# ── 5. Exclude worst rvol days: rvol > 0.45 ──────────────────────────────────
ex45_dates = [d for d in all_dates if days_data[d]["rvol"] <= 0.45]
r_ex45 = stats(ex45_dates, "5. Exclude rvol > 0.45 ticks", days_data, total_days)

# ── 6. Filtered days detail (what we're skipping) ────────────────────────────
print("\n" + "="*60)
print("  DAYS FILTERED BY HMM (state=0, high-rvol):")
print("-"*60)
print(f"{'Date':<12} {'rvol_ticks':>10} {'IC':>8} {'P&L':>10}")
for d in hmm_dead_dates:
    row = days_data[d]
    print(f"  {d:<12} {row['rvol']:>10.4f}  {row['ic']:>7.4f}  ${row['pnl']:>8,.2f}")

print("\n" + "="*60)
print("  DAYS FILTERED BY rvol > 0.35:")
filtered_35 = [d for d in all_dates if days_data[d]["rvol"] > 0.35]
for d in sorted(filtered_35):
    row = days_data[d]
    print(f"  {d:<12} {row['rvol']:>10.4f}  {row['ic']:>7.4f}  ${row['pnl']:>8,.2f}")

print("\n" + "="*60)
print("  DAYS FILTERED BY rvol > 0.45:")
filtered_45 = [d for d in all_dates if days_data[d]["rvol"] > 0.45]
for d in sorted(filtered_45):
    row = days_data[d]
    print(f"  {d:<12} {row['rvol']:>10.4f}  {row['ic']:>7.4f}  ${row['pnl']:>8,.2f}")

# ── 7. Sharpe improvement summary ────────────────────────────────────────────
print("\n" + "="*60)
print("  SHARPE IMPROVEMENT SUMMARY")
print("="*60)
baseline_sharpe = r_all.get("sharpe", 0)
for label, result in [
    ("HMM active (state=1)", r_hmm),
    ("rvol <= 0.35",          r_low35),
    ("rvol <= 0.25",          r_low25),
    ("rvol <= 0.45 (excl >0.45)", r_ex45),
]:
    s = result.get("sharpe", 0)
    delta = s - baseline_sharpe
    sign = "+" if delta >= 0 else ""
    n = result.get("n", 0)
    print(f"  {label:<30}  Sharpe={s:.3f}  ({sign}{delta:.3f} vs baseline)  n={n}/{total_days}")

# ── 8. Correlation: rvol vs P&L ──────────────────────────────────────────────
rvols = [days_data[d]["rvol"] for d in all_dates]
pnls  = [days_data[d]["pnl"]  for d in all_dates]
ics   = [days_data[d]["ic"]   for d in all_dates]

corr_rvol_pnl = np.corrcoef(rvols, pnls)[0, 1]
corr_rvol_ic  = np.corrcoef(rvols, ics)[0, 1]
corr_ic_pnl   = np.corrcoef(ics,   pnls)[0, 1]

print("\n" + "="*60)
print("  CORRELATIONS (22 WF days)")
print("="*60)
print(f"  rvol vs P&L:  r = {corr_rvol_pnl:.4f}")
print(f"  rvol vs IC:   r = {corr_rvol_ic:.4f}")
print(f"  IC   vs P&L:  r = {corr_ic_pnl:.4f}")

# ── 9. Quintile analysis: rvol quintiles vs mean P&L ─────────────────────────
rvol_arr = np.array(rvols)
pnl_arr  = np.array(pnls)
q_edges  = np.quantile(rvol_arr, [0.0, 0.20, 0.40, 0.60, 0.80, 1.0])

print("\n" + "="*60)
print("  RVOL QUINTILE ANALYSIS (mean daily P&L per quintile)")
print("="*60)
print(f"  {'Quintile':<12} {'rvol range':>22} {'Mean P&L':>12} {'N':>4}")
for i in range(5):
    lo, hi = q_edges[i], q_edges[i+1]
    mask = (rvol_arr >= lo) & (rvol_arr <= hi) if i == 4 else (rvol_arr >= lo) & (rvol_arr < hi)
    subset_pnl = pnl_arr[mask]
    if len(subset_pnl) == 0:
        continue
    print(f"  Q{i+1} (bottom 20%+)" if i == 0 else f"  Q{i+1}",
          end="")
    print(f"  rvol [{lo:.3f}, {hi:.3f}]  mean=${np.mean(subset_pnl):>8,.1f}  n={mask.sum()}")

print("\nAnalysis complete.")
