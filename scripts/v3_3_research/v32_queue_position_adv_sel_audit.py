#!/usr/bin/env python3
"""
HC #336 / #342 / #344 — Queue-position + Adverse-Selection audit on v3.2 fold 0 OOT.

Closes 2 of the 5 live-readiness gates the user asked about:
  Gate 2 (queue-position): proxy via FIFO fill_rate per confidence band (true L2 queue deferred to v2).
  Gate 3 (adverse-selection): realized PRICE MOVE in position direction at +1s/+5s/+10s/+30s post-fill.

Inputs (read-only):
  /home/jupiter/Lvl3Quant/output/v3_2_deep_sim_20260512/fold_00_oot_predictions.npz  (241,351 preds, 5 OOT days)
  /home/jupiter/Lvl3Quant/data/processed/mbo_events_smart_v3_fifo_labels/{OOT}_fifo_labels.npz

Output:
  /home/jupiter/Lvl3Quant/output/v3_2_queue_adv_sel_audit_20260514/v3_2_queue_adv_sel_audit.json
  + per-band markdown table for Discord

Author: autonomous dispatch per HC #342 (Jupiter never idle) + HC #344 (live-readiness audit).
NOT trainer code. NEW analysis script under scripts/v3_3_research/ per HC #307D permit.
"""
import json
import numpy as np
from pathlib import Path
from datetime import datetime

OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/v3_2_queue_adv_sel_audit_20260514")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG = OUTPUT_DIR / "audit.log"

def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")

log("=== HC #336 QUEUE-POSITION + ADVERSE-SELECTION AUDIT ===")
log("Loading v3.2 fold 0 OOT predictions...")
PREDS = np.load("/home/jupiter/Lvl3Quant/output/v3_2_deep_sim_20260512/fold_00_oot_predictions.npz")
OOT_DATES = list(PREDS["oot_dates"])
log(f"  OOT dates: {OOT_DATES}")
log(f"  n_samples: {int(PREDS['n_samples'])}")

# Load FIFO labels for all OOT dates, concat
log("Loading FIFO labels...")
fifo_root = Path("/home/jupiter/Lvl3Quant/data/processed/mbo_events_smart_v3_fifo_labels")
fifo_parts = {}
n_per_day = []
for d in OOT_DATES:
    fp = fifo_root / f"{d}_fifo_labels.npz"
    if not fp.exists():
        log(f"  MISSING: {fp}")
        continue
    z = np.load(fp, allow_pickle=False)
    n_per_day.append(len(z["window_k"]))
    for k in z.keys():
        fifo_parts.setdefault(k, []).append(z[k])
fifo = {k: np.concatenate(v) for k, v in fifo_parts.items()}
log(f"  FIFO rows: {sum(n_per_day)} ({n_per_day})")

n_preds = int(PREDS["pred_log_ret_1s"].shape[0])
n_fifo = sum(n_per_day)
n = min(n_preds, n_fifo)
log(f"  Using min(preds={n_preds}, fifo={n_fifo}) = {n}")

# === ADVERSE-SELECTION: realized price move in position direction post-fill ===
# In LOG-RET space. Convert to ticks at end.
# Long fill: in_position_move = +realized_log_ret (positive = favorable, negative = adverse)
# Short fill: in_position_move = -realized_log_ret

# Tick conversion: 1 tick on ES = 0.25 pts. At ES ~5800 mid (Feb 2026), 1 tick ≈ 4.31e-5 log-ret
ES_PX_REF = 5800.0
LOG_RET_PER_TICK = np.log((ES_PX_REF + 0.25) / ES_PX_REF)  # ≈ 4.31e-5
TICKS_PER_LOG_RET = 1.0 / LOG_RET_PER_TICK  # ≈ 23,200
log(f"  Tick conversion: 1 log-ret unit = {TICKS_PER_LOG_RET:.0f} ticks (ES @ ~{ES_PX_REF})")

rlr = {h: PREDS[f"target_log_ret_{h}"][:n].astype(np.float64) for h in ("1s", "5s", "10s", "30s")}
rlr_mask = {h: (PREDS[f"mask_log_ret_{h}"][:n].astype(bool) & np.isfinite(rlr[h])) for h in rlr}
pred_1s = PREDS["pred_log_ret_1s"][:n].astype(np.float64)
mask_1s = PREDS["mask_log_ret_1s"][:n].astype(bool) & np.isfinite(pred_1s)

# Confidence bands per side
def bands_long(p):
    return {
        "Top0.1%": np.quantile(p, 0.999),
        "Top0.5%": np.quantile(p, 0.995),
        "Top1%":   np.quantile(p, 0.99),
        "Top5%":   np.quantile(p, 0.95),
        "Top10%":  np.quantile(p, 0.90),
    }

def bands_short(p):
    return {
        "Top0.1%": np.quantile(p, 0.001),
        "Top0.5%": np.quantile(p, 0.005),
        "Top1%":   np.quantile(p, 0.01),
        "Top5%":   np.quantile(p, 0.05),
        "Top10%":  np.quantile(p, 0.10),
    }

results = {"long": {}, "short": {}}

for side in ("long", "short"):
    threshes = bands_long(pred_1s) if side == "long" else bands_short(pred_1s)
    side_sign = 1.0 if side == "long" else -1.0
    for band, t in threshes.items():
        if side == "long":
            sel = (pred_1s >= t) & mask_1s
        else:
            sel = (pred_1s <= t) & mask_1s
        if sel.sum() < 5:
            continue
        moves = {}
        n_eff = {}
        for h in ("1s", "5s", "10s", "30s"):
            sel_h = sel & rlr_mask[h]
            n_eff[h] = int(sel_h.sum())
            moves[h] = side_sign * rlr[h][sel_h] * TICKS_PER_LOG_RET if sel_h.sum() > 0 else np.array([np.nan])
        if n_eff["30s"] < 5:
            continue
        results[side][band] = {
            "n_fills": int(sel.sum()),
            "n_eff_per_horizon": n_eff,
            "mean_move_ticks": {h: float(np.mean(moves[h])) for h in moves},
            "median_move_ticks": {h: float(np.median(moves[h])) for h in moves},
            "p25_move_ticks": {h: float(np.quantile(moves[h], 0.25)) for h in moves},
            "p75_move_ticks": {h: float(np.quantile(moves[h], 0.75)) for h in moves},
            "frac_adverse_30s_pct": float(np.mean(moves["30s"] < 0) * 100),
            "adverse_cost_30s_mean_ticks": float(np.mean(np.minimum(moves["30s"], 0))),
            "mfe_30s_mean_ticks": float(np.mean(np.maximum(moves["30s"], 0))),
            "mae_30s_mean_ticks": float(np.mean(np.minimum(moves["30s"], 0))),
        }
        log(f"  {side.upper()} {band}: n_fills={int(sel.sum())}, n_eff_30s={n_eff['30s']}, "
            f"mean_30s={np.mean(moves['30s']):+.3f}t, adv_frac={np.mean(moves['30s']<0)*100:.1f}%, "
            f"adv_cost={np.mean(np.minimum(moves['30s'],0)):+.3f}t")

# === QUEUE-POSITION PROXY via FIFO fill_rate per band ===
# True queue model needs MBO L2 reconstruction (deferred to v2 of this work, HC #336 v2).
# This proxy: if fill_rate at top-X% confidence is HIGH, our orders are getting served — favorable queue.
# If fill_rate is LOW, we're getting picked off by faster participants — unfavorable queue.
queue_proxy = {"long": {}, "short": {}}
for side in ("long", "short"):
    threshes = bands_long(pred_1s) if side == "long" else bands_short(pred_1s)
    for band, t in threshes.items():
        if side == "long":
            sel = (pred_1s >= t) & mask_1s
        else:
            sel = (pred_1s <= t) & mask_1s
        if sel.sum() < 5:
            continue
        band_q = {"n_signals": int(sel.sum())}
        for bracket in ("tp4sl3", "tp8sl5"):
            col = f"{bracket}_{side}_filled"
            if col in fifo:
                f_arr = fifo[col][:n]
                n_fills = int((sel & f_arr).sum())
                band_q[bracket] = {
                    "n_fills": n_fills,
                    "fill_rate_pct": (n_fills / sel.sum() * 100) if sel.sum() > 0 else 0,
                }
        queue_proxy[side][band] = band_q
        log(f"  QUEUE {side.upper()} {band}: signals={band_q['n_signals']}, "
            f"tp4sl3_fill={band_q.get('tp4sl3',{}).get('fill_rate_pct',0):.1f}%, "
            f"tp8sl5_fill={band_q.get('tp8sl5',{}).get('fill_rate_pct',0):.1f}%")

# === SAVE JSON ===
out = {
    "metadata": {
        "spec": "HC #336 queue-position + adverse-selection audit on v3.2 fold 0 OOT",
        "hc_refs": ["#336", "#342", "#344", "#341"],
        "oot_dates": list(map(str, OOT_DATES)),
        "n_preds_used": n,
        "pred_basis": "pred_log_ret_1s for confidence bands per HC #341",
        "tick_basis": "ES futures (0.25 pt = 1 tick), ref price 5800.0",
        "adverse_def": "realized PRICE MOVE in position direction post-fill, in ticks. Negative = adverse.",
        "queue_model": "PROXY via FIFO fill_rate (true L2 queue model = HC #336 v2 deferred)",
        "generated_at": datetime.now().isoformat(),
    },
    "adverse_selection": results,
    "queue_position_proxy": queue_proxy,
}
json_path = OUTPUT_DIR / "v3_2_queue_adv_sel_audit.json"
with open(json_path, "w") as f:
    json.dump(out, f, indent=2)
log(f"Saved JSON: {json_path}")

# === MARKDOWN SUMMARY TABLE ===
md = ["# HC #336 Queue+AdverseSel Audit — v3.2 Fold 0 OOT\n"]
md.append(f"OOT dates: {', '.join(map(str, OOT_DATES))} | n={n} preds\n")
md.append("## ADVERSE-SELECTION (mean realized move in position direction, ticks)\n")
md.append("| Side | Band | n_fills | +1s | +5s | +10s | +30s | adv_frac_30s | adv_cost_30s |")
md.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
for side in ("long", "short"):
    for band, r in results[side].items():
        m = r["mean_move_ticks"]
        md.append(f"| {side} | {band} | {r['n_fills']} | {m['1s']:+.3f} | {m['5s']:+.3f} | "
                  f"{m['10s']:+.3f} | {m['30s']:+.3f} | {r['frac_adverse_30s_pct']:.1f}% | "
                  f"{r['adverse_cost_30s_mean_ticks']:+.3f} |")

md.append("\n## QUEUE-POSITION PROXY (FIFO fill rate per band, %)\n")
md.append("| Side | Band | n_signals | tp4sl3 fill% | tp8sl5 fill% |")
md.append("|---|---|---:|---:|---:|")
for side in ("long", "short"):
    for band, q in queue_proxy[side].items():
        md.append(f"| {side} | {band} | {q['n_signals']} | "
                  f"{q.get('tp4sl3',{}).get('fill_rate_pct',0):.1f}% | "
                  f"{q.get('tp8sl5',{}).get('fill_rate_pct',0):.1f}% |")
md.append("\n_True L2 queue model = HC #336 v2 deferred. Proxy = FIFO fill rate from real MBO bid/ask labels._\n")

md_path = OUTPUT_DIR / "v3_2_queue_adv_sel_audit.md"
with open(md_path, "w") as f:
    f.write("\n".join(md))
log(f"Saved Markdown: {md_path}")
log("=== DONE ===")
