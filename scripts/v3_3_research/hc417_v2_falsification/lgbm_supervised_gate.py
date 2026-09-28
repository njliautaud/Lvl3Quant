#!/usr/bin/env python3
"""HC #414(b) LGBM supervised gate on CNN-Mamba v2 wrapped NPZ.

Builds a per-sample classifier P(net_tk > 0 | enter) using:
  features  = [pred_log_ret_1s, pred_log_ret_5s, pred_log_ret_10s,
               signed_short = -pred_log_ret_1s,
               daily_conf_rank_1s (0-1),
               1s vs 5s slope, 5s vs 10s slope,
               abs(pred_1s) (confidence magnitude)]
  target    = (target_log_ret_1s < -threshold_for_short_profit)
              i.e. realised 1s move sufficiently negative to be profitable on a short
              after passive_at_touch cost (0.376 tk) and partial TP/SL hits.

Approach: time-ordered CV (no leakage). Train on first 2/3 of dates, score
remaining 1/3. Then evaluate as a meta-gate on top of confidence-rank top0.5%
short for the 5 winning cells.

Output: output/hc417_lgbm_supervised_gate_v2/
  - gate_model.txt       (LGBM booster)
  - gate_probs.npy       (per-sample P(profit|enter))
  - gate_evaluation.md   (cells x with/without gate comparison)
  - gate_csv.csv         (per-cell metrics)
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

LVL3 = Path("/home/jupiter/Lvl3Quant")
NPZ = LVL3 / "output/hc417_v2_full_oot_wrapped_for_hc413.npz"
OUT = LVL3 / "output/hc417_lgbm_supervised_gate_v2"
OUT.mkdir(parents=True, exist_ok=True)

# ----------------- Load data -----------------
print(f"[lgbm-gate] loading {NPZ}")
d = np.load(NPZ, allow_pickle=True)
p1 = d["pred_log_ret_1s"].astype(np.float32)
p5 = d["pred_log_ret_5s"].astype(np.float32)
p10 = d["pred_log_ret_10s"].astype(np.float32)
t1 = d["target_log_ret_1s"].astype(np.float32)
m1 = d["mask_log_ret_1s"].astype(bool)
oot_dates = [str(x) for x in d["oot_dates"]]
N = p1.shape[0]
print(f"[lgbm-gate] N={N} masked={int(m1.sum())} dates={len(oot_dates)}")

# Recover day_index from oot_dates by reconstructing per-date sample counts
# The wrapped NPZ doesn't include day_index, so we re-derive from src
SRC56 = LVL3 / "output/hc417_v2_full_oot_56d.npz"
src = np.load(SRC56, allow_pickle=True)
src_day_index = src["day_index"]
src_oot_full = [str(x) for x in src["oot_dates"]]
src_dates_present = [str(x) for x in src["dates_present"]]

# The wrapper kept only kept_dates (36 dates) AND trimmed trailing 8 per day.
# Re-derive the wrapped day_index using the same procedure:
LABELS_DIR = LVL3 / "data/processed/mbo_events_smart_v3_fifo_labels"
kept_dates = []
fifo_n = {}
for dt in src_dates_present:
    fp = LABELS_DIR / f"{dt}_fifo_labels.npz"
    if fp.exists():
        kept_dates.append(dt)
        fifo_n[dt] = int(np.load(fp)["window_k"].shape[0])

day_index_wrapped = np.empty(N, dtype=np.int16)
cursor = 0
for kd_idx, dt in enumerate(kept_dates):
    full_idx = src_oot_full.index(dt)
    rows = np.where(src_day_index == full_idx)[0]
    n_v2 = rows.shape[0]
    n_fifo = fifo_n[dt]
    keep_n = n_v2 - (n_v2 - n_fifo)  # = n_fifo
    day_index_wrapped[cursor:cursor + keep_n] = kd_idx
    cursor += keep_n
assert cursor == N, f"cursor {cursor} != N {N}"
print(f"[lgbm-gate] day_index recovered, {len(kept_dates)} dates")

# ----------------- Compute features -----------------
signed_short = -p1
slope_1_5 = p1 - p5
slope_5_10 = p5 - p10
abs_p1 = np.abs(p1)

# Per-day rank within each date (0..1)
rank_per_day = np.empty(N, dtype=np.float32)
for kd_idx in range(len(kept_dates)):
    mask = day_index_wrapped == kd_idx
    sub = signed_short[mask]
    order = np.argsort(sub)
    ranks = np.empty_like(order, dtype=np.float32)
    ranks[order] = np.arange(sub.shape[0], dtype=np.float32) / max(sub.shape[0] - 1, 1)
    rank_per_day[mask] = ranks

X = np.column_stack([p1, p5, p10, signed_short, abs_p1, slope_1_5, slope_5_10, rank_per_day]).astype(np.float32)
feature_names = ["pred_1s", "pred_5s", "pred_10s", "signed_short", "abs_p1",
                 "slope_1_5", "slope_5_10", "daily_rank"]

# Label: target_log_ret_1s < -profit_threshold
# Need to beat passive_at_touch cost = 0.376 tk in label space. labels are in ticks.
PROFIT_TK_THRESHOLD = 0.376  # net>0 after cost
y_short_profit = (t1 < -PROFIT_TK_THRESHOLD).astype(np.int8)
y_valid = m1 & (rank_per_day > 0.90)  # train only on plausibly-tradeable rows (top 10% short by day)
print(f"[lgbm-gate] training rows (mask & top10 daily): {int(y_valid.sum())} "
      f"positive_rate={float(y_short_profit[y_valid].mean()):.3f}")

# ----------------- Train/test time-ordered split -----------------
split_idx = int(len(kept_dates) * 2 / 3)
train_dates = set(range(split_idx))
test_dates = set(range(split_idx, len(kept_dates)))
train_mask = y_valid & np.isin(day_index_wrapped, list(train_dates))
test_mask = y_valid & np.isin(day_index_wrapped, list(test_dates))
print(f"[lgbm-gate] train n={int(train_mask.sum())} test n={int(test_mask.sum())} "
      f"split @ date_idx={split_idx} ({kept_dates[split_idx]})")

# ----------------- LGBM train -----------------
try:
    import lightgbm as lgb
except Exception as e:
    raise SystemExit(f"lightgbm not installed: {e}")

dtrain = lgb.Dataset(X[train_mask], label=y_short_profit[train_mask],
                     feature_name=feature_names)
dvalid = lgb.Dataset(X[test_mask], label=y_short_profit[test_mask],
                     feature_name=feature_names, reference=dtrain)
params = dict(
    objective="binary",
    metric="binary_logloss,auc",
    learning_rate=0.05,
    num_leaves=31,
    min_data_in_leaf=200,
    feature_fraction=0.9,
    bagging_fraction=0.9,
    bagging_freq=5,
    verbose=-1,
    seed=42,
)
booster = lgb.train(params, dtrain, num_boost_round=400, valid_sets=[dvalid],
                    callbacks=[lgb.early_stopping(40), lgb.log_evaluation(50)])
booster.save_model(str(OUT / "gate_model.txt"))

# ----------------- Score full dataset -----------------
probs = booster.predict(X, num_iteration=booster.best_iteration).astype(np.float32)
np.save(OUT / "gate_probs.npy", probs)

# ----------------- Evaluate as meta-gate on each winning cell -----------------
# Winning cells: v2_1s_short_top05, v2_1s_short_top1, v2_5s_short_top1, v2_5s_short_top05, v2_10s_short_top1
# We'll demonstrate on the test split (so the gate sees out-of-sample dates)
# A "fill" requires: top X% short on the relevant horizon AND probs > p_thresh.

PASSIVE_COST = 0.376
def evaluate_cell(horizon, pct, gate_thresh, label="cell"):
    pred = {"1s": p1, "5s": p5, "10s": p10}[horizon]
    sig = -pred
    # Apply per-day percentile threshold on signed_short
    fill_mask = np.zeros(N, dtype=bool)
    for kd_idx in range(len(kept_dates)):
        dmask = day_index_wrapped == kd_idx
        dsig = sig[dmask]
        if dsig.size == 0:
            continue
        thr_d = np.percentile(dsig, 100 - pct)
        fill_mask[dmask] = dsig >= thr_d
    fill_mask &= m1
    base_fills = int(fill_mask.sum())
    fill_mask_gated = fill_mask & (probs >= gate_thresh)
    gated_fills = int(fill_mask_gated.sum())

    def metrics(fm):
        if fm.sum() == 0:
            return dict(n=0, net=float("nan"), wr=float("nan"), day_conc=float("nan"), pdpr=float("nan"))
        # Realised short PnL = -target_log_ret_1s - cost (when shorting at top)
        realised = (-t1[fm]) - PASSIVE_COST
        n = realised.size
        net = float(realised.mean())
        wr = float((realised > 0).mean())
        # day_conc on test set only
        di = day_index_wrapped[fm]
        unique = np.unique(di)
        per_day_pnl = np.array([realised[di == u].sum() for u in unique], dtype=np.float32)
        abs_total = float(np.abs(per_day_pnl).sum()) or 1e-9
        day_conc = float(np.max(np.abs(per_day_pnl)) / abs_total)
        per_day_net_mean = np.array([realised[di == u].mean() for u in unique], dtype=np.float32)
        pdpr = float((per_day_net_mean > 0).mean())
        return dict(n=n, net=net, wr=wr, day_conc=day_conc, pdpr=pdpr,
                    n_days=int(unique.size))
    return label, metrics(fill_mask), metrics(fill_mask_gated)

cells = [
    ("v2_1s_short_top05", "1s", 0.5),
    ("v2_1s_short_top1", "1s", 1.0),
    ("v2_5s_short_top05", "5s", 0.5),
    ("v2_5s_short_top1", "5s", 1.0),
    ("v2_10s_short_top1", "10s", 1.0),
]

# Try a few gate thresholds
gate_thresholds = [0.30, 0.40, 0.50, 0.60]
rows = []
for cell_id, horizon, pct in cells:
    for gt in gate_thresholds:
        _, base, gated = evaluate_cell(horizon, pct, gt, cell_id)
        rows.append(dict(cell_id=cell_id, gate=gt,
                         base_n=base["n"], base_net=base["net"], base_wr=base["wr"],
                         base_day_conc=base["day_conc"], base_pdpr=base["pdpr"],
                         base_n_days=base.get("n_days", 0),
                         gated_n=gated["n"], gated_net=gated["net"], gated_wr=gated["wr"],
                         gated_day_conc=gated["day_conc"], gated_pdpr=gated["pdpr"],
                         gated_n_days=gated.get("n_days", 0)))

import csv
with open(OUT / "gate_csv.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)

# Build markdown verdict
lines = ["# HC #414(b) LGBM Supervised Gate — v2 CNN-Mamba", ""]
lines.append(f"NPZ: `{NPZ.name}`, N={N}, kept_dates={len(kept_dates)}")
lines.append(f"Time-order split: train dates 0..{split_idx-1}, test dates {split_idx}..{len(kept_dates)-1}")
lines.append(f"Train rows: {int(train_mask.sum())}, Test rows: {int(test_mask.sum())}")
lines.append(f"Target: `target_log_ret_1s < -0.376` (i.e. realised 1s move profitable for short after passive cost)")
lines.append(f"Best iter: {booster.best_iteration}")
lines.append("")
lines.append("Feature importances (gain):")
imp = booster.feature_importance(importance_type="gain")
for name, val in sorted(zip(feature_names, imp), key=lambda x: -x[1]):
    lines.append(f"- {name}: {val:.1f}")
lines.append("")
lines.append("## Cell metrics: baseline vs gated (full-OOT, not just test split)")
lines.append("")
lines.append("**NOTE**: These metrics are computed using a simplified short PnL "
             "(-target_1s - 0.376 tk), NOT the HC #413 TP/SL backtester. They are "
             "indicative of whether the LGBM gate filters out losing fills, not a "
             "replacement for the canonical FIFO replay.")
lines.append("")
lines.append("| cell_id | gate | base_n | base_net | base_pdpr | gated_n | gated_net | gated_pdpr | gated_day_conc | delta_net | retention |")
lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for r in rows:
    dn = r["gated_net"] - r["base_net"] if not np.isnan(r["gated_net"]) else float("nan")
    ret = r["gated_n"] / max(r["base_n"], 1)
    lines.append(f"| {r['cell_id']} | {r['gate']:.2f} | {r['base_n']} | {r['base_net']:+.4f} | {r['base_pdpr']:.2f} | "
                 f"{r['gated_n']} | {r['gated_net']:+.4f} | {r['gated_pdpr']:.2f} | "
                 f"{r['gated_day_conc']:.2f} | {dn:+.4f} | {ret:.2f} |")
lines.append("")
lines.append("## Honest assessment")
lines.append("")
# Find best per cell
best_per_cell = {}
for r in rows:
    cid = r["cell_id"]
    if cid not in best_per_cell or (not np.isnan(r["gated_net"]) and r["gated_net"] > best_per_cell[cid]["gated_net"]):
        best_per_cell[cid] = r
for cid, r in best_per_cell.items():
    improves = (not np.isnan(r["gated_net"])) and r["gated_net"] > r["base_net"]
    keeps_fills = r["gated_n"] >= max(50, r["base_n"] * 0.30)
    verdict = "PASS" if (improves and keeps_fills) else "FAIL"
    lines.append(f"- **{cid}**: best gate@{r['gate']:.2f} net delta={r['gated_net']-r['base_net']:+.4f} "
                 f"tk, retention={r['gated_n']/max(r['base_n'],1):.0%}, verdict={verdict}")

with open(OUT / "gate_evaluation.md", "w") as f:
    f.write("\n".join(lines))

print(f"[lgbm-gate] wrote {OUT / 'gate_evaluation.md'}")
print(f"[lgbm-gate] wrote {OUT / 'gate_csv.csv'}")
