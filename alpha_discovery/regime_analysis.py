"""
Regime Analysis: Why did cancel_asym_chain signal stop working in Oct-Nov 2025?

Loads all 100 days of MBO features, computes per-day statistics, and performs
statistical comparison between the two periods:
  - Period 1 (in-sample):  Jul 14 - Sep 19, 2025 (first 50 days)
  - Period 2 (holdout):    Sep 22 - Nov 28, 2025 (last  50 days)

Key metrics:
  - Market structure: mid price, spread, volatility, bars per day
  - Signal behavior: cancel_asym_5 mean, ofi_5 mean
  - Predictive power: correlation of each signal with forward 100-bar return
  - Signal interaction: correlation between cancel_asym_5 and ofi_5
"""

import os
import sys
import numpy as np
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ─── paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR    = PROJECT_ROOT / "data" / "processed" / "mbo_features_cache"

sys.path.insert(0, str(PROJECT_ROOT))

# ─── feature index lookup ─────────────────────────────────────────────────────
from alpha_discovery.mbo_features import get_feature_names

FEATURE_NAMES = get_feature_names()
assert len(FEATURE_NAMES) == 340, f"Expected 340 features, got {len(FEATURE_NAMES)}"

# The NPZ array columns:
#   col 0 = mid   (first item in get_feature_names)
#   col 1 = spread
#   col N = FEATURE_NAMES[N]  (0-indexed, direct mapping)
MID_IDX    = 0   # mid
SPREAD_IDX = 1   # spread

def _find_idx(name: str) -> int:
    idx = FEATURE_NAMES.index(name)
    return idx

OFI5_IDX          = _find_idx("ofi_5")          # 96
CANCEL_ASYM5_IDX  = _find_idx("cancel_asym_5")  # 158

print(f"Feature indices: mid={MID_IDX}, spread={SPREAD_IDX}, "
      f"ofi_5={OFI5_IDX}, cancel_asym_5={CANCEL_ASYM5_IDX}")


# ─── per-day computation ───────────────────────────────────────────────────────
FORWARD_HORIZON = 100   # bars (~100 seconds of raw book snapshots)

def compute_day_stats(date_str: str, npz_path: Path) -> dict | None:
    """
    Load one NPZ file and return a dict of scalar statistics for that day.
    Returns None if the file cannot be loaded or has too few bars.
    """
    try:
        data = np.load(str(npz_path))["mbo_features"]   # shape (N, 340)
    except Exception as e:
        print(f"  ERROR loading {npz_path.name}: {e}")
        return None

    N = data.shape[0]
    if N < FORWARD_HORIZON + 50:
        print(f"  SKIP {date_str}: only {N} bars (need >{FORWARD_HORIZON+50})")
        return None

    # ── raw columns ──────────────────────────────────────────────────────────
    mid          = data[:, MID_IDX].astype(np.float64)
    spread_raw   = data[:, SPREAD_IDX].astype(np.float64)   # raw price units
    cancel_asym5 = data[:, CANCEL_ASYM5_IDX].astype(np.float64)
    ofi5         = data[:, OFI5_IDX].astype(np.float64)

    # ── spread in ticks (ES tick = 0.25) ─────────────────────────────────────
    ES_TICK = 0.25
    spread_ticks = spread_raw / ES_TICK

    # ── 1-bar return (mid-price change) ──────────────────────────────────────
    returns_1bar = np.diff(mid)           # length N-1

    # ── daily volatility = std of 1-bar returns ───────────────────────────────
    daily_vol = np.std(returns_1bar)

    # ── forward 100-bar return ───────────────────────────────────────────────
    # fwd[i] = mid[i+100] - mid[i]   for i in [0, N-100)
    fwd_return = mid[FORWARD_HORIZON:] - mid[:-FORWARD_HORIZON]   # length N-100

    # Align signals to same index range (drop last 100 bars)
    n_valid   = len(fwd_return)
    ca5_valid = cancel_asym5[:n_valid]
    ofi_valid = ofi5[:n_valid]

    # Remove NaN / inf rows from all three arrays together
    mask = (
        np.isfinite(fwd_return) &
        np.isfinite(ca5_valid)  &
        np.isfinite(ofi_valid)
    )
    fwd_r  = fwd_return[mask]
    ca5    = ca5_valid[mask]
    ofi    = ofi_valid[mask]

    if len(fwd_r) < 50:
        print(f"  SKIP {date_str}: only {len(fwd_r)} valid bars after NaN filter")
        return None

    # ── correlations ─────────────────────────────────────────────────────────
    def safe_corr(a, b):
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            return 0.0
        r, _ = stats.pearsonr(a, b)
        return float(r)

    corr_ca5_fwd   = safe_corr(ca5, fwd_r)
    corr_ofi5_fwd  = safe_corr(ofi, fwd_r)
    corr_ca5_ofi5  = safe_corr(ca5, ofi)

    return {
        "date":            date_str,
        "n_bars":          N,
        "avg_mid":         float(np.mean(mid)),
        "avg_spread_ticks": float(np.mean(spread_ticks)),
        "daily_vol":       float(daily_vol),
        "mean_cancel_asym5": float(np.mean(cancel_asym5)),
        "mean_ofi5":       float(np.mean(ofi5)),
        "corr_ca5_fwd":    corr_ca5_fwd,
        "corr_ofi5_fwd":   corr_ofi5_fwd,
        "corr_ca5_ofi5":   corr_ca5_ofi5,
    }


# ─── main ─────────────────────────────────────────────────────────────────────
def main():
    # Gather all NPZ files, sorted by date
    files = sorted(CACHE_DIR.glob("*_mbo_features.npz"))
    assert len(files) == 100, f"Expected 100 files, found {len(files)}"

    print(f"\nLoading {len(files)} daily NPZ files from:\n  {CACHE_DIR}\n")

    # ── compute per-day statistics ────────────────────────────────────────────
    rows = []
    for npz in files:
        date_str = npz.name.replace("_mbo_features.npz", "")
        stats_row = compute_day_stats(date_str, npz)
        if stats_row is not None:
            rows.append(stats_row)
            print(f"  {date_str}: corr_ca5={stats_row['corr_ca5_fwd']:+.4f}  "
                  f"corr_ofi5={stats_row['corr_ofi5_fwd']:+.4f}  "
                  f"spread={stats_row['avg_spread_ticks']:.3f}tk  "
                  f"vol={stats_row['daily_vol']:.4f}  "
                  f"bars={stats_row['n_bars']}")

    print(f"\nLoaded {len(rows)} days successfully.\n")
    assert len(rows) == 100, f"Expected 100 valid days, got {len(rows)}"

    # ── split by period ───────────────────────────────────────────────────────
    dates = [r["date"] for r in rows]
    IS_CUTOFF = "2025-09-19"   # last IS day
    period1 = [r for r in rows if r["date"] <= IS_CUTOFF]
    period2 = [r for r in rows if r["date"] >  IS_CUTOFF]

    print(f"Period 1 (IS, Jul 14 – Sep 19):   {len(period1)} days  "
          f"[{period1[0]['date']} to {period1[-1]['date']}]")
    print(f"Period 2 (OOS, Sep 22 – Nov 28):  {len(period2)} days  "
          f"[{period2[0]['date']} to {period2[-1]['date']}]")
    print()

    # ── metric comparison ─────────────────────────────────────────────────────
    METRICS = [
        ("avg_mid",             "Average mid price"),
        ("avg_spread_ticks",    "Average spread (ticks)"),
        ("daily_vol",           "Daily volatility (std 1-bar returns)"),
        ("mean_cancel_asym5",   "Mean cancel_asym_5"),
        ("mean_ofi5",           "Mean ofi_5"),
        ("corr_ca5_fwd",        "Corr(cancel_asym_5, fwd_100bar)"),
        ("corr_ofi5_fwd",       "Corr(ofi_5, fwd_100bar)"),
        ("corr_ca5_ofi5",       "Corr(cancel_asym_5, ofi_5)"),
        ("n_bars",              "Bars per day"),
    ]

    # Build numpy arrays per period
    def arr(rows, key):
        return np.array([r[key] for r in rows], dtype=float)

    print("=" * 90)
    print(f"{'Metric':<40} {'Period1 Mean':>12} {'P1 Std':>8} {'Period2 Mean':>12} {'P2 Std':>8} {'t-stat':>8} {'p-val':>8} {'sig':>4}")
    print("=" * 90)

    results = {}
    for key, label in METRICS:
        a = arr(period1, key)
        b = arr(period2, key)
        t, p = stats.ttest_ind(a, b, equal_var=False)   # Welch's t-test
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
        print(f"{label:<40} {np.mean(a):>12.5f} {np.std(a):>8.5f} "
              f"{np.mean(b):>12.5f} {np.std(b):>8.5f} "
              f"{t:>8.3f} {p:>8.4f} {sig:>4}")
        results[key] = dict(
            p1_mean=float(np.mean(a)), p1_std=float(np.std(a)),
            p2_mean=float(np.mean(b)), p2_std=float(np.std(b)),
            t_stat=float(t), p_val=float(p),
        )

    print("=" * 90)
    print("Significance: * p<0.05  ** p<0.01  *** p<0.001\n")

    # ── monthly drill-down for cancel_asym_5 predictive power ─────────────────
    print("\nMonthly drill-down: Corr(cancel_asym_5, fwd_100bar)")
    print("-" * 50)
    from collections import defaultdict
    monthly = defaultdict(list)
    for r in rows:
        month = r["date"][:7]
        monthly[month].append(r["corr_ca5_fwd"])
    for month in sorted(monthly):
        vals = monthly[month]
        print(f"  {month}: mean={np.mean(vals):+.4f}  std={np.std(vals):.4f}  "
              f"positive_days={sum(v>0 for v in vals)}/{len(vals)}")

    # ── monthly drill-down for ofi_5 predictive power ─────────────────────────
    print("\nMonthly drill-down: Corr(ofi_5, fwd_100bar)")
    print("-" * 50)
    monthly_ofi = defaultdict(list)
    for r in rows:
        month = r["date"][:7]
        monthly_ofi[month].append(r["corr_ofi5_fwd"])
    for month in sorted(monthly_ofi):
        vals = monthly_ofi[month]
        print(f"  {month}: mean={np.mean(vals):+.4f}  std={np.std(vals):.4f}  "
              f"positive_days={sum(v>0 for v in vals)}/{len(vals)}")

    # ── regime change detection: rolling 10-day average of key metrics ─────────
    print("\nRolling 10-day averages (key metrics over time)")
    print("-" * 70)
    print(f"{'Date':<12} {'corr_ca5':>10} {'corr_ofi5':>10} {'spread_tk':>10} {'vol':>10}")
    print("-" * 70)
    for i in range(9, len(rows)):
        window = rows[i-9:i+1]
        date   = rows[i]["date"]
        m_ca5  = np.mean([r["corr_ca5_fwd"]      for r in window])
        m_ofi  = np.mean([r["corr_ofi5_fwd"]      for r in window])
        m_spd  = np.mean([r["avg_spread_ticks"]    for r in window])
        m_vol  = np.mean([r["daily_vol"]           for r in window])
        print(f"  {date}  {m_ca5:>+10.4f} {m_ofi:>+10.4f} {m_spd:>10.4f} {m_vol:>10.5f}")

    # ── summary interpretation ────────────────────────────────────────────────
    ca5 = results["corr_ca5_fwd"]
    ofi = results["corr_ofi5_fwd"]
    vol = results["daily_vol"]
    spd = results["avg_spread_ticks"]
    ca5_ofi = results["corr_ca5_ofi5"]

    print("\n" + "=" * 90)
    print("INTERPRETATION SUMMARY")
    print("=" * 90)

    print(f"\n1. cancel_asym_5 predictive power:")
    print(f"   IS mean corr  = {ca5['p1_mean']:+.4f} | OOS mean corr = {ca5['p2_mean']:+.4f}")
    print(f"   Change = {ca5['p2_mean']-ca5['p1_mean']:+.4f} | p={ca5['p_val']:.4f}")

    print(f"\n2. ofi_5 predictive power:")
    print(f"   IS mean corr  = {ofi['p1_mean']:+.4f} | OOS mean corr = {ofi['p2_mean']:+.4f}")
    print(f"   Change = {ofi['p2_mean']-ofi['p1_mean']:+.4f} | p={ofi['p_val']:.4f}")

    print(f"\n3. Market structure changes:")
    print(f"   Spread:    IS={spd['p1_mean']:.4f} ticks | OOS={spd['p2_mean']:.4f} ticks | p={spd['p_val']:.4f}")
    print(f"   Volatility: IS={vol['p1_mean']:.5f}  | OOS={vol['p2_mean']:.5f}  | p={vol['p_val']:.4f}")

    print(f"\n4. cancel_asym_5 vs ofi_5 correlation (interaction):")
    print(f"   IS={ca5_ofi['p1_mean']:+.4f} | OOS={ca5_ofi['p2_mean']:+.4f} | p={ca5_ofi['p_val']:.4f}")

    # Classify causes
    causes = []
    if ca5['p_val'] < 0.05:
        direction = "improved" if ca5['p2_mean'] > ca5['p1_mean'] else "deteriorated"
        causes.append(f"cancel_asym_5 predictive power significantly {direction} (p={ca5['p_val']:.4f})")
    if ofi['p_val'] < 0.05:
        direction = "improved" if ofi['p2_mean'] > ofi['p1_mean'] else "deteriorated"
        causes.append(f"ofi_5 predictive power significantly {direction} (p={ofi['p_val']:.4f})")
    if vol['p_val'] < 0.05:
        direction = "higher" if vol['p2_mean'] > vol['p1_mean'] else "lower"
        causes.append(f"Volatility regime shifted to {direction} (p={vol['p_val']:.4f})")
    if spd['p_val'] < 0.05:
        direction = "wider" if spd['p2_mean'] > spd['p1_mean'] else "tighter"
        causes.append(f"Spread regime shifted to {direction} (p={spd['p_val']:.4f})")
    if ca5_ofi['p_val'] < 0.05:
        causes.append(f"cancel_asym/ofi correlation structure changed (p={ca5_ofi['p_val']:.4f})")

    print(f"\nSIGNIFICANT REGIME CHANGES DETECTED ({len(causes)}):")
    for c in causes:
        print(f"  - {c}")
    if not causes:
        print("  No statistically significant changes detected (all p >= 0.05)")

    print("\nAnalysis complete.")
    return rows, results


if __name__ == "__main__":
    rows, results = main()

    # ── optional: send to Discord ──────────────────────────────────────────────
    try:
        import subprocess, json

        # Build a compact Discord summary
        ca5 = results["corr_ca5_fwd"]
        ofi = results["corr_ofi5_fwd"]
        vol = results["daily_vol"]
        spd = results["avg_spread_ticks"]
        ca5_ofi = results["corr_ca5_ofi5"]

        def sig(p):
            return "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))

        msg = (
            "**REGIME ANALYSIS COMPLETE — cancel_asym_chain failure investigation**\n"
            "```\n"
            f"{'Metric':<38} {'IS mean':>9} {'OOS mean':>9} {'sig':>5}\n"
            f"{'-'*65}\n"
            f"{'corr(cancel_asym_5, fwd_100bar)':<38} {ca5['p1_mean']:>+9.4f} {ca5['p2_mean']:>+9.4f} {sig(ca5['p_val']):>5}\n"
            f"{'corr(ofi_5, fwd_100bar)':<38} {ofi['p1_mean']:>+9.4f} {ofi['p2_mean']:>+9.4f} {sig(ofi['p_val']):>5}\n"
            f"{'corr(cancel_asym_5, ofi_5)':<38} {ca5_ofi['p1_mean']:>+9.4f} {ca5_ofi['p2_mean']:>+9.4f} {sig(ca5_ofi['p_val']):>5}\n"
            f"{'Daily volatility':<38} {vol['p1_mean']:>+9.5f} {vol['p2_mean']:>+9.5f} {sig(vol['p_val']):>5}\n"
            f"{'Spread (ticks)':<38} {spd['p1_mean']:>+9.4f} {spd['p2_mean']:>+9.4f} {sig(spd['p_val']):>5}\n"
            "```\n"
            "Significance: * p<0.05  ** p<0.01  *** p<0.001  ns=not significant\n"
            "_Full output in terminal. Script: alpha_discovery/regime_analysis.py_"
        )

        # Write msg to temp file and call Node to send
        tmp = Path(__file__).parent.parent / "_regime_discord_msg.tmp"
        tmp.write_text(msg, encoding="utf-8")

        # Send via the discord lib used in this project
        send_script = Path(__file__).parent.parent / "_send_regime_msg.js"
        send_script.write_text(
            f"const d = require('./lib/discord');\n"
            f"const fs = require('fs');\n"
            f"const msg = fs.readFileSync('{tmp.as_posix()}', 'utf8');\n"
            f"d.sendToUser(msg).then(() => {{ fs.unlinkSync('{tmp.as_posix()}'); process.exit(0); }});\n",
            encoding="utf-8"
        )
        result = subprocess.run(
            ["node", str(send_script)],
            cwd=str(Path(__file__).parent.parent.parent / "teleclaude-main"),
            capture_output=True, text=True, timeout=15
        )
        if result.returncode == 0:
            print("Discord summary sent.")
        else:
            print(f"Discord send note: {result.stderr[:200]}")
        # Clean up
        try:
            send_script.unlink()
        except Exception:
            pass

    except Exception as e:
        print(f"(Discord send skipped: {e})")
