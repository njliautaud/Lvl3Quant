#!/usr/bin/env python3
"""
J8 — PER-DAY (approx) STABILITY on fifo_tp8sl5_net.

Asserts the headline 78% WR over 5 days isn't a single-day fluke.
Splits the 241351 sample array into 5 approximately equal chunks (samples
are emitted by dataset in date order). Reports WR/mean_ticks/total/PF for
each chunk at each conf band on fifo_tp8sl5_net.

Caveat: chunk-to-date mapping is APPROXIMATE — true date split needs
inference-script enhancement (TODO). Useful as sanity check.
"""
import numpy as np
from pathlib import Path

OUT = Path("/home/jupiter/Lvl3Quant/output/exec_science_v3_3_overnight")
BASE = "fifo_tp8sl5_net"
PASSIVE = 0.376
MARKET = 1.376

d = np.load("/home/jupiter/Lvl3Quant/output/cnn_mamba_v3_3_uncertainty_weighted/fold_00_predictions.npz", allow_pickle=True)
pred = np.asarray(d[f"pred_{BASE}"], dtype=np.float64)
tgt = np.asarray(d[f"target_{BASE}"], dtype=np.float64)
mask = np.asarray(d[f"mask_{BASE}"], dtype=np.float64) if f"mask_{BASE}" in d.files else np.ones_like(pred)
oot_dates = [str(x) for x in d["oot_dates"]]
n = len(pred)
print(f"n_total={n}, oot_dates={oot_dates}", flush=True)

# Split into 5 equal chunks
chunks = np.array_split(np.arange(n), 5)

rows = []
for chunk_i, idx in enumerate(chunks):
    date_label = oot_dates[chunk_i] if chunk_i < len(oot_dates) else f"chunk{chunk_i}"
    p_all = pred[idx]; t_all = tgt[idx]; m_all = mask[idx]
    v = (m_all > 0) & np.isfinite(p_all) & np.isfinite(t_all)
    if v.sum() < 100:
        continue
    p = p_all[v]; t = t_all[v]
    conf = np.abs(p)
    for band in [10.0, 5.0, 1.0, 0.5]:
        cutoff = np.percentile(conf, 100.0 - band)
        sel = conf >= cutoff
        if sel.sum() < 10:
            continue
        pp = p[sel]; tt = t[sel]
        pnl = np.sign(pp) * tt
        passive = pnl - PASSIVE
        market = pnl - MARKET
        rows.append({
            "chunk": chunk_i, "date_approx": date_label, "band_pct": band,
            "n": int(sel.sum()),
            "gross_total": float(pnl.sum()),
            "gross_mean": float(pnl.mean()),
            "gross_wr": float(np.mean(pnl > 0)),
            "passive_total": float(passive.sum()),
            "passive_mean": float(passive.mean()),
            "passive_wr": float(np.mean(passive > 0)),
            "market_total": float(market.sum()),
            "market_mean": float(market.mean()),
            "market_wr": float(np.mean(market > 0)),
        })

# CSV
cols = ["chunk","date_approx","band_pct","n","gross_total","gross_mean","gross_wr",
        "passive_total","passive_mean","passive_wr","market_total","market_mean","market_wr"]
with open(OUT/"j8_per_day_fifo_tp8sl5_net.csv","w") as f:
    f.write(",".join(cols)+"\n")
    for r in rows:
        f.write(",".join(f"{r[c]}" for c in cols)+"\n")

# summary
lines = []
lines.append("="*110)
lines.append("J8 — PER-DAY APPROX STABILITY (fifo_tp8sl5_net, 5 chunks≈5 OOT days)")
lines.append("="*110)
for band in [10.0, 5.0, 1.0, 0.5]:
    lines.append(f"\n--- band top-{band}% ---")
    lines.append(f"{'chunk':>5s} {'date':>10s} {'n':>5s} {'gross_total':>11s} {'g_mean':>7s} {'g_wr':>6s} {'p_total':>8s} {'p_wr':>6s} {'m_total':>8s} {'m_wr':>6s}")
    for r in [r for r in rows if abs(r['band_pct']-band)<1e-6]:
        lines.append(f"{r['chunk']:>5d} {r['date_approx']:>10s} {r['n']:>5d} {r['gross_total']:>11.1f} {r['gross_mean']:>7.3f} {r['gross_wr']:>6.3f} {r['passive_total']:>8.1f} {r['passive_wr']:>6.3f} {r['market_total']:>8.1f} {r['market_wr']:>6.3f}")
txt = "\n".join(lines)
(OUT/"j8_summary.txt").write_text(txt)
print(txt)
