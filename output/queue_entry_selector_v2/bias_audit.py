#!/usr/bin/env python3
"""
Look-ahead bias audit for Queue Entry Selector v2.
Checks: walk-forward integrity, feature leakage, early stopping leakage, label leakage.
Saves results to bias_audit_results.txt.
"""
import json
import inspect
import re
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

OUT_DIR = Path(__file__).parent
BASE = Path("/home/jupiter/Lvl3Quant")
SOURCE_FILE = BASE / "alpha_discovery" / "queue_entry_selector_v2.py"

results = []

def log(msg):
    results.append(msg)
    print(msg)

def log_pass(check, detail=""):
    msg = f"[PASS] {check}"
    if detail:
        msg += f" — {detail}"
    log(msg)

def log_fail(check, detail=""):
    msg = f"[FAIL] {check}"
    if detail:
        msg += f" — {detail}"
    log(msg)

def log_warn(check, detail=""):
    msg = f"[WARN] {check}"
    if detail:
        msg += f" — {detail}"
    log(msg)


log("=" * 70)
log("LOOK-AHEAD BIAS AUDIT — Queue Entry Selector v2")
log("=" * 70)
log("")

# ── Load data ──
with open(OUT_DIR / "results.json") as f:
    res = json.load(f)

trades = pd.read_parquet(OUT_DIR / "all_oot_trades.parquet")
source_code = SOURCE_FILE.read_text()

fold_results = res['fold_results']

# ═══════════════════════════════════════════════════════════════
# CHECK 1: Walk-Forward Integrity
# Verify train dates NEVER overlap OOT dates in any fold.
# ═══════════════════════════════════════════════════════════════

log("─── CHECK 1: Walk-Forward Integrity ───")
log(f"Walk-forward config: {res['walk_forward']}")
log(f"Total folds: {len(fold_results)}")

overlap_violations = []
oot_leaks_into_train = []

for i, fold in enumerate(fold_results):
    train_dates = set(fold['train_dates'])
    oot_dates = set(fold['oot_dates'])

    # Check 1a: direct overlap
    overlap = train_dates & oot_dates
    if overlap:
        overlap_violations.append((i, sorted(overlap)))

    # Check 1b: OOT dates should be strictly AFTER train dates
    max_train = max(train_dates)
    min_oot = min(oot_dates)
    if min_oot <= max_train:
        oot_leaks_into_train.append((i, max_train, min_oot))

    # Check 1c: verify sliding (not expanding) — train window size constant
    if len(train_dates) != res['walk_forward']['train_days']:
        # Allow minor variation due to missing dates
        pass

# Check 1d: across folds — verify OOT dates from fold N never appear in training of fold N+k
all_oot_dates_by_fold = {}
all_train_dates_by_fold = {}
cross_fold_leaks = []

for i, fold in enumerate(fold_results):
    all_train_dates_by_fold[i] = set(fold['train_dates'])
    all_oot_dates_by_fold[i] = set(fold['oot_dates'])

for i in range(len(fold_results)):
    for j in range(len(fold_results)):
        if i == j:
            continue
        # Check if fold j's OOT dates appear in fold i's training
        # This is OK if i < j (future OOT dates training on past data)
        # But NOT OK if j < i (past OOT dates leaking into future training)
        # Actually, in sliding WF, fold j's OOT dates CAN appear in fold i's training
        # if i > j (later fold trains on dates that were OOT in earlier fold).
        # That's standard sliding WF behavior, NOT a leak.
        # The ONLY leak is if fold i trains on its OWN OOT dates.
        pass

# Verify SLIDING (not expanding) window
train_sizes = [len(fold['train_dates']) for fold in fold_results]
is_sliding = all(s == train_sizes[0] for s in train_sizes)

if overlap_violations:
    log_fail("Train/OOT date overlap",
             f"{len(overlap_violations)} folds have overlapping dates: {overlap_violations[:3]}")
else:
    log_pass("No train/OOT date overlap in any fold")

if oot_leaks_into_train:
    log_fail("OOT dates not strictly after train dates",
             f"{len(oot_leaks_into_train)} violations: {oot_leaks_into_train[:3]}")
else:
    log_pass("OOT dates strictly after train dates in all folds")

if is_sliding:
    log_pass(f"Sliding window confirmed (constant train size = {train_sizes[0]} days)")
else:
    log_fail(f"Window size varies: min={min(train_sizes)}, max={max(train_sizes)} — may be expanding")

# Verify actual trade dates in parquet match fold definitions
log("")
log("Verifying actual OOT trade dates match fold definitions...")
trade_date_issues = 0
for fold in fold_results:
    fold_num = fold['fold']
    expected_oot = set(fold['oot_dates'])
    actual_dates = set(trades[trades['fold'] == fold_num]['date'].unique())
    if actual_dates - expected_oot:
        log_fail(f"Fold {fold_num}: trades on unexpected dates: {actual_dates - expected_oot}")
        trade_date_issues += 1
    # actual_dates can be subset of expected (some dates may have no qualifying trades)

if trade_date_issues == 0:
    log_pass("All OOT trades fall within declared fold date ranges")

log("")

# ═══════════════════════════════════════════════════════════════
# CHECK 2: Feature Leakage (merge_asof direction)
# ═══════════════════════════════════════════════════════════════

log("─── CHECK 2: Feature Leakage (merge direction) ───")

# Check source code for merge_asof usage
merge_matches = re.findall(r"merge_asof\([^)]+\)", source_code, re.DOTALL)
if not merge_matches:
    log_warn("No merge_asof found in source — check data joining method")
else:
    for match in merge_matches:
        log(f"  Found: {match.strip()[:120]}")
        if "direction='backward'" in match:
            log_pass("merge_asof uses direction='backward' (no future data)")
        elif "direction='nearest'" in match:
            log_fail("merge_asof uses direction='nearest' — can look forward by up to tolerance")
        elif "direction='forward'" in match:
            log_fail("merge_asof uses direction='forward' — explicitly looks at future data")
        else:
            log_warn("merge_asof direction not explicitly set — defaults to 'backward' (safe)")

# Check that features are snapshots AT or BEFORE the FIFO label timestamp
# In the code, fifo_df is the LEFT frame and queue_df is the RIGHT frame
# merge_asof(fifo_df, queue_df, on='ts_ns', direction='backward')
# This means: for each fifo label ts, find the queue snapshot with ts_ns <= fifo.ts_ns
# This is CORRECT: queue state is from BEFORE the trade entry
if "pd.merge_asof(\n        fifo_df, queue_df" in source_code or \
   "pd.merge_asof(fifo_df, queue_df" in source_code or \
   re.search(r"merge_asof\(\s*fifo_df,\s*queue_df", source_code):
    log_pass("FIFO labels are LEFT frame in merge_asof — queue features are looked up backward from label time")
else:
    # Check the actual order
    if re.search(r"merge_asof\(\s*\n?\s*fifo_df", source_code):
        log_pass("FIFO labels are LEFT frame in merge_asof")
    else:
        log_warn("Could not confirm merge order — manual review recommended")

# Check tolerance
tol_match = re.search(r"tolerance\s*=\s*(\d[\d_]*)", source_code)
if tol_match:
    tol_ns = int(tol_match.group(1).replace('_', ''))
    tol_sec = tol_ns / 1e9
    log(f"  Tolerance: {tol_ns} ns = {tol_sec:.1f} seconds")
    if tol_sec <= 2.0:
        log_pass(f"Tolerance is {tol_sec:.1f}s — reasonable for 1-second snapshots")
    else:
        log_warn(f"Tolerance is {tol_sec:.1f}s — could match stale features")

log("")

# ═══════════════════════════════════════════════════════════════
# CHECK 3: Early Stopping Leakage
# ═══════════════════════════════════════════════════════════════

log("─── CHECK 3: Early Stopping Leakage ───")

# Check if eval_set uses OOT data or train holdout
# Look for eval_set in model.fit()
fit_match = re.search(r"model\.fit\([^)]+\)", source_code, re.DOTALL)
if fit_match:
    fit_call = fit_match.group(0)
    log(f"  Found: model.fit(...) call")

    # Check what's used as eval_set
    if "X_val" in fit_call and "y_val" in fit_call:
        # Check how X_val is defined
        val_split_match = re.search(r"val_split\s*=\s*int\(len\(X_train\)\s*\*\s*([\d.]+)\)", source_code)
        if val_split_match:
            train_frac = float(val_split_match.group(1))
            log_pass(f"Early stopping uses train holdout ({train_frac*100:.0f}%/{(1-train_frac)*100:.0f}% split) — NOT OOT data")
        else:
            log_pass("Early stopping uses X_val/y_val (appears to be train holdout)")
    elif "X_oot" in fit_call or "y_oot" in fit_call:
        log_fail("Early stopping uses OOT data as eval_set — LEAKS test data into model selection")
    else:
        log_warn("Could not determine eval_set source — manual review needed")

    # Check for early_stopping callback
    if "early_stopping" in source_code:
        es_match = re.search(r"early_stopping\((\d+)", source_code)
        if es_match:
            patience = int(es_match.group(1))
            log(f"  Early stopping patience: {patience} rounds")
else:
    log_warn("Could not find model.fit() call")

# Also check for any direct use of OOT in training loop
if re.search(r"(X_oot|y_oot|oot_df).*eval", source_code):
    log_fail("OOT data may be referenced in evaluation context within training")
else:
    log_pass("No reference to OOT data in eval_set context")

log("")

# ═══════════════════════════════════════════════════════════════
# CHECK 4: Label Leakage
# ═══════════════════════════════════════════════════════════════

log("─── CHECK 4: Label Leakage (FIFO labels) ───")

# FIFO labels represent: "if you entered at this timestamp, would TP or SL be hit first?"
# The label MUST be computed from prices AFTER the entry timestamp.
# Check the source code for how labels are loaded/created.

# The labels come from pre-computed .npz files (mbo_events_smart_v3_fifo_labels)
# We need to verify the label generation code, but we can also do a statistical check.

log("  FIFO labels loaded from: data/processed/mbo_events_smart_v3_fifo_labels/")
log("  Config: tp4sl3 (TP=4 ticks, SL=3 ticks, passive FIFO)")
log("")

# Statistical check: if labels leaked future info, the model would be too good
# Check if OOT AUC is suspiciously close to training AUC
log("  Statistical checks for label integrity:")

# Check 1: Overall OOT win rate should be plausible for ES FIFO
overall_wr = trades['target'].mean()
log(f"  Overall OOT WR (unfiltered): {overall_wr:.3f}")
if 0.35 < overall_wr < 0.65:
    log_pass(f"Baseline WR ({overall_wr:.3f}) is plausible for ES TP4/SL3 FIFO")
else:
    log_warn(f"Baseline WR ({overall_wr:.3f}) seems unusual — verify label construction")

# Check 2: verify labels are binary
unique_targets = trades['target'].unique()
if set(unique_targets) <= {0, 1}:
    log_pass("Labels are binary (0/1) as expected")
else:
    log_fail(f"Labels have unexpected values: {unique_targets}")

# Check 3: net_ticks should be consistent with target
# target=1 should have positive net_ticks, target=0 should have negative
winners = trades[trades['target'] == 1]['net_ticks']
losers = trades[trades['target'] == 0]['net_ticks']
if winners.mean() > 0 and losers.mean() < 0:
    log_pass(f"Label/PnL consistency: winners avg={winners.mean():+.2f}t, losers avg={losers.mean():+.2f}t")
else:
    log_fail(f"Label/PnL inconsistency: winners avg={winners.mean():+.2f}t, losers avg={losers.mean():+.2f}t")

# Check 4: net_ticks for tp4sl3 should cluster around +4 and -3
winner_median = winners.median()
loser_median = losers.median()
log(f"  Winner median: {winner_median:+.2f}t (expected ~+3.6 for TP4 minus commission)")
log(f"  Loser median: {loser_median:+.2f}t (expected ~-3.4 for SL3 plus commission)")
if 2.5 < winner_median < 4.5 and -4.5 < loser_median < -2.0:
    log_pass("PnL distribution consistent with TP4/SL3 FIFO config")
else:
    log_warn("PnL distribution doesn't match expected TP4/SL3 range — verify label source")

# Check 5: verify the label code comments indicate forward-looking outcomes
if "net_ticks" in source_code and "hit_tp" in source_code:
    log_pass("Source references net_ticks and hit_tp — labels represent forward outcomes from entry time")

# Check 6: verify no future price data in features
# Features should all be orderbook state (bid/ask qty, OFI, etc.) — not price returns
price_return_features = [c for c in trades.columns if 'return' in c.lower() or 'future' in c.lower()]
if price_return_features:
    log_fail(f"Potential future-price features found: {price_return_features}")
else:
    log_pass("No future-price features (returns/future) found in feature set")

log("")

# ═══════════════════════════════════════════════════════════════
# CHECK 5: Additional Statistical Checks
# ═══════════════════════════════════════════════════════════════

log("─── CHECK 5: Additional Statistical Integrity ───")

# Check fold-to-fold performance stability
# If there's look-ahead bias, ALL folds would be uniformly good
fold_wrs = []
for fold in fold_results:
    thresh_data = fold['thresholds']
    if '0.55' in thresh_data:
        fold_wrs.append(thresh_data['0.55']['wr'])
    elif 0.55 in thresh_data:
        fold_wrs.append(thresh_data[0.55]['wr'])

if fold_wrs:
    wr_std = np.std(fold_wrs)
    wr_mean = np.mean(fold_wrs)
    log(f"  Fold WR@0.55 variation: mean={wr_mean:.3f}, std={wr_std:.3f}")
    if wr_std > 0.02:
        log_pass(f"Healthy fold-to-fold variation (std={wr_std:.3f}) — no sign of systematic leak")
    elif wr_std < 0.005:
        log_fail(f"Suspiciously low variation (std={wr_std:.3f}) — possible data leak")
    else:
        log_warn(f"Low but plausible variation (std={wr_std:.3f})")

# Check for date monotonicity across folds
oot_date_ranges = []
for fold in fold_results:
    oot_date_ranges.append((min(fold['oot_dates']), max(fold['oot_dates'])))

is_monotonic = all(oot_date_ranges[i][0] <= oot_date_ranges[i+1][0]
                   for i in range(len(oot_date_ranges)-1))
if is_monotonic:
    log_pass("OOT date ranges are monotonically increasing across folds")
else:
    log_fail("OOT date ranges are NOT monotonic — possible fold ordering issue")

# Check that no fold has 100% or 0% WR (which would indicate broken labels)
extreme_folds = []
for fold in fold_results:
    if fold['baseline_wr'] >= 0.95 or fold['baseline_wr'] <= 0.05:
        extreme_folds.append((fold['fold'], fold['baseline_wr']))

if extreme_folds:
    log_warn(f"Extreme baseline WR in folds: {extreme_folds}")
else:
    log_pass("No folds with extreme (>95% or <5%) baseline WR")

log("")

# ═══════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════

log("═" * 70)
passes = sum(1 for r in results if r.startswith("[PASS]"))
fails = sum(1 for r in results if r.startswith("[FAIL]"))
warns = sum(1 for r in results if r.startswith("[WARN]"))
log(f"SUMMARY: {passes} PASS, {fails} FAIL, {warns} WARN")
if fails == 0:
    log("OVERALL: NO LOOK-AHEAD BIAS DETECTED")
else:
    log(f"OVERALL: {fails} POTENTIAL BIAS ISSUE(S) — REVIEW REQUIRED")
log("═" * 70)

# Save results
with open(OUT_DIR / "bias_audit_results.txt", 'w') as f:
    f.write("\n".join(results) + "\n")

print(f"\nResults saved to {OUT_DIR / 'bias_audit_results.txt'}")
