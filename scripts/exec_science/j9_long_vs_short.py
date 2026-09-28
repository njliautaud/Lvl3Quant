#!/usr/bin/env python3
"""
J9 — LONG vs SHORT side asymmetry on fifo_tp8sl5_net at conf bands.

DIRECTIVES.md notes: "Short side has significantly better edge than long side
at all confidence levels." Verify on v3.3 5-day OOT predictions.

For each conf band, split into LONG (pred>0) and SHORT (pred<0) buckets:
- n, gross/passive/market mean+total+WR
- Decide: do we trade BOTH sides or SHORT-only?
"""
import numpy as np
from pathlib import Path

OUT = Path("/home/jupiter/Lvl3Quant/output/exec_science_v3_3_overnight")
PASSIVE = 0.376
MARKET = 1.376

d = np.load("/home/jupiter/Lvl3Quant/output/cnn_mamba_v3_3_uncertainty_weighted/fold_00_predictions.npz", allow_pickle=True)
BASE = "fifo_tp8sl5_net"
pred = np.asarray(d[f"pred_{BASE}"], dtype=np.float64)
tgt = np.asarray(d[f"target_{BASE}"], dtype=np.float64)
mask = np.asarray(d[f"mask_{BASE}"], dtype=np.float64) if f"mask_{BASE}" in d.files else np.ones_like(pred)

v = (mask > 0) & np.isfinite(pred) & np.isfinite(tgt)
p = pred[v]; t = tgt[v]
conf = np.abs(p)

rows = []
for band in [25.0, 10.0, 5.0, 1.0, 0.5, 0.1]:
    cutoff = np.percentile(conf, 100.0 - band)
    sel = conf >= cutoff
    pp = p[sel]; tt = t[sel]
    pnl = np.sign(pp) * tt
    for side_name, side_mask in [("ALL", np.ones_like(pp, dtype=bool)),
                                  ("LONG", pp > 0),
                                  ("SHORT", pp < 0)]:
        if side_mask.sum() < 5:
            continue
        pnl_s = pnl[side_mask]
        p_pass = pnl_s - PASSIVE
        p_mkt = pnl_s - MARKET
        rows.append({
            "band_pct": band, "side": side_name,
            "n": int(side_mask.sum()),
            "gross_mean": float(pnl_s.mean()),
            "gross_total": float(pnl_s.sum()),
            "gross_wr": float(np.mean(pnl_s > 0)),
            "passive_mean": float(p_pass.mean()),
            "passive_total": float(p_pass.sum()),
            "passive_wr": float(np.mean(p_pass > 0)),
            "market_mean": float(p_mkt.mean()),
            "market_total": float(p_mkt.sum()),
            "market_wr": float(np.mean(p_mkt > 0)),
        })

cols = ["band_pct","side","n","gross_total","gross_mean","gross_wr",
        "passive_total","passive_mean","passive_wr",
        "market_total","market_mean","market_wr"]
with open(OUT/"j9_long_vs_short.csv","w") as f:
    f.write(",".join(cols)+"\n")
    for r in rows:
        f.write(",".join(f"{r[c]}" for c in cols)+"\n")

lines = []
lines.append("="*110)
lines.append("J9 — LONG vs SHORT on fifo_tp8sl5_net (5-day OOT, after costs in ticks/RT)")
lines.append("="*110)
for band in [25.0, 10.0, 5.0, 1.0, 0.5, 0.1]:
    bandrows = [r for r in rows if abs(r['band_pct']-band)<1e-6]
    if not bandrows: continue
    lines.append(f"\n--- top-{band}% conf ---")
    lines.append(f"{'side':>6s} {'n':>6s} {'gMean':>8s} {'gTotal':>9s} {'gWR':>6s} {'pMean':>8s} {'pTotal':>9s} {'pWR':>6s} {'mMean':>8s} {'mTotal':>9s} {'mWR':>6s}")
    for r in bandrows:
        lines.append(f"{r['side']:>6s} {r['n']:>6d} {r['gross_mean']:>8.3f} {r['gross_total']:>9.1f} {r['gross_wr']:>6.3f} {r['passive_mean']:>8.3f} {r['passive_total']:>9.1f} {r['passive_wr']:>6.3f} {r['market_mean']:>8.3f} {r['market_total']:>9.1f} {r['market_wr']:>6.3f}")
txt = "\n".join(lines)
(OUT/"j9_summary.txt").write_text(txt)
print(txt)
