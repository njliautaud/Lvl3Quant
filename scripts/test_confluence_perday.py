"""
Per-day regime stratification for v4 confluence configs (HC #428 R1).

OOT covers 5 consecutive trading days. The npz lacks per-event date stamps,
but the dispatcher samples at stride=250 with a window_t1=1500 warmup pad.
Reconstruct per-day sample boundaries from the source MBO event counts.

For each day:
  - log_ret_30s realized P&L net of 0.376 ticks
  - Sharpe per-day
  - WR per-day
  - Day classification: green (ES close > open) / red / flat — proxy via mean(target_log_ret_30s)
    sign on the day (positive = trending up day)

Reports two flagship configs:
  - BEST_HOLD30:        top5 + STACK_fifo_or
  - BEST_FIFO_SHARPE:   top20 + STACK_fifo_or
"""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path

NPZ_DIR = Path("/home/jupiter/Lvl3Quant/output/cnn_mamba_v3_4_2_hc454phase2_smoke_run2")
OOT_DATES = ["20260223","20260224","20260225","20260226","20260227"]
STRIDE = 250
WIN_T1 = 1500
RT = 0.376
TICK_USD = 12.50

def per_day_indices(n_total):
    """Compute per-day start/end indices in the OOT npz from raw event counts."""
    bounds = []
    cur = 0
    for dt in OOT_DATES:
        p = Path(f"/home/jupiter/Lvl3Quant/data/processed/mbo_events_smart_v3/{dt}_mbo_events.npz")
        n_raw = len(np.load(p)["events"])
        # samples per day = (n_raw - WIN_T1) // STRIDE + 1, approximately
        n_samples = max(0, (n_raw - WIN_T1) // STRIDE + 1)
        bounds.append((dt, cur, cur + n_samples))
        cur += n_samples
    # Final-bucket overflow correction
    if cur != n_total:
        # scale uniformly
        scale = n_total / cur
        new_bounds = []
        running = 0
        for dt, lo, hi in bounds:
            new_lo = int(round(lo * scale))
            new_hi = int(round(hi * scale))
            new_bounds.append((dt, new_lo, new_hi))
            running = new_hi
        bounds = new_bounds
        bounds[-1] = (bounds[-1][0], bounds[-1][1], n_total)
    return bounds

def stat(x, name):
    x = x[np.isfinite(x)]
    if len(x) < 30: return {"n":int(len(x)), f"{name}_mean":np.nan, f"{name}_WR":np.nan, f"{name}_Sh":np.nan, f"{name}_USD":np.nan}
    return {"n":int(len(x)),
            f"{name}_mean": float(np.mean(x)),
            f"{name}_WR":   float((x>0).mean()*100),
            f"{name}_Sh":   float(np.mean(x)/(np.std(x)+1e-9)),
            f"{name}_USD":  float(np.sum(x)*TICK_USD)}

def main(ep="ep1"):
    npz = NPZ_DIR / f"fold_00_{ep}_oot.npz"
    if not npz.exists():
        print(f"missing: {npz}"); return
    d = np.load(npz)
    n = len(d["pred_log_ret_1s"])
    bounds = per_day_indices(n)
    print(f"== {ep} OOT n={n:,}")
    for dt, lo, hi in bounds:
        print(f"  {dt}: rows [{lo:,}, {hi:,})   = {hi-lo:,} samples")
    p1 = d["pred_log_ret_1s"]
    p_fifo43 = d["pred_fifo_tp4sl3_net"]
    p_fifo85 = d["pred_fifo_tp8sl5_net"]
    t30 = d["target_log_ret_30s"].astype(np.float64)
    sgn = np.where(p1 > 0, 1.0, -1.0)
    R30 = sgn * t30 - RT
    conf = np.abs(p1)
    q95 = np.quantile(conf, 0.95)
    q80 = np.quantile(conf, 0.80)
    fifo_or = ((p_fifo43 * sgn) > 0) | ((p_fifo85 * sgn) > 0)

    configs = {
        "top20+FIFO_or": (conf >= q80) & fifo_or,
        "top5+FIFO_or":  (conf >= q95) & fifo_or,
        "BASE_top20":    (conf >= q80),
        "BASE_top5":     (conf >= q95),
    }

    rows = []
    for cname, mask in configs.items():
        # day classifier
        day_mean_t30 = []
        for dt, lo, hi in bounds:
            sub = t30[lo:hi]; sub = sub[np.isfinite(sub)]
            mu = float(np.mean(sub)) if len(sub) else np.nan
            day_mean_t30.append((dt, mu))
        # per-day perf
        for dt, lo, hi in bounds:
            day_mask = np.zeros(n, bool); day_mask[lo:hi] = True
            sel = mask & day_mask
            sub = R30[sel]; sub = sub[np.isfinite(sub)]
            day_mu_t = dict(day_mean_t30)[dt]
            regime = "green" if day_mu_t > 0 else ("red" if day_mu_t < 0 else "flat")
            s = stat(sub, "h30")
            rows.append({"config":cname, "date":dt, "regime":regime, "day_drift_t30":day_mu_t, **s})

    df = pd.DataFrame(rows)
    print()
    print(df.to_string(index=False, float_format=lambda x: f"{x:+.4f}" if isinstance(x,float) else str(x)))

    # cross-day Sharpe stability
    print()
    print("=== CROSS-DAY SHARPE STABILITY ===")
    for cname in configs:
        sub = df[df.config == cname]
        shr = sub["h30_Sh"].dropna()
        if len(shr) < 3:
            print(f"  {cname}: insufficient days"); continue
        print(f"  {cname}: per-day Sh = {[f'{x:+.3f}' for x in shr.values]}  min={shr.min():+.3f}  max={shr.max():+.3f}  spread={shr.max()-shr.min():+.3f}")

    out = NPZ_DIR / f"fold_00_{ep}_perday_confluence.csv"
    df.to_csv(out, index=False, float_format="%.5f")
    print(f"\nsaved: {out}")

if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv)>1 else "ep1")
