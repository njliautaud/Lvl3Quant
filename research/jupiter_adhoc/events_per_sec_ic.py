import json
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr

ROOT = Path("/home/jupiter/Lvl3Quant")
CNN_PRED_FILE = ROOT / "data/processed/wider_cnn_preds/oos_predictions_wider_cnn_20260318_080909.npz"
MBO_EVENTS_DIR = ROOT / "data/processed/mbo_events"
BARS_PER_SESSION = 234000
BAR_WARMUP = 120
WIN = 5.0

def load_mbo(date_str):
    ymd = date_str.replace("-", "")
    path = MBO_EVENTS_DIR / f"{ymd}_mbo_events.npz"
    if not path.exists():
        return None
    return np.load(path, allow_pickle=True)["timestamps"]

def eps(ts_ns, query_ns, win=5.0):
    wns = int(win * 1e9)
    ilo = np.searchsorted(ts_ns, query_ns - wns, side="left")
    ihi = np.searchsorted(ts_ns, query_ns + wns, side="right")
    return (ihi - ilo) / (2.0 * win)

def ic(p, r, mask):
    if mask.sum() < 50:
        return None
    return float(pearsonr(p[mask], r[mask])[0])

cnn = np.load(CNN_PRED_FILE, allow_pickle=True)
dates = sorted([k.replace("_preds","") for k in cnn.files if k.endswith("_preds")])
print(f"Dates: {len(dates)}")

AP, AR, AE, AT = [], [], [], []
skip = 0
for date in dates:
    P = cnn[f"{date}_preds"]
    R = cnn[f"{date}_targets"]
    ts = load_mbo(date)
    if ts is None or len(ts) < 100:
        skip += 1
        continue
    t0, t1 = int(ts[0]), int(ts[-1])
    dur = t1 - t0
    if dur <= 0:
        skip += 1
        continue
    bd = dur / BARS_PER_SESSION
    bi = np.arange(BAR_WARMUP, BAR_WARMUP + len(P))
    bt = (t0 + bi * bd).astype(np.int64)
    ok = (bt >= t0 + int(WIN * 1e9)) & (bt <= t1 - int(WIN * 1e9))
    if ok.sum() < 50:
        skip += 1
        continue
    ts_s = np.sort(ts.astype(np.int64))
    e = eps(ts_s, bt[ok])
    AP.append(P[ok])
    AR.append(R[ok])
    AE.append(e)
    AT.append((bt[ok] - t0) / dur)
    print(f"  {date}: n={ok.sum()} evts/s p50={np.median(e):.0f}")

print(f"Done: {len(AP)} dates skip={skip}")
P = np.concatenate(AP)
R = np.concatenate(AR)
E = np.concatenate(AE)
T = np.concatenate(AT)
print(f"N={len(P):,} evts/s p5={np.percentile(E,5):.0f} p25={np.percentile(E,25):.0f} p50={np.percentile(E,50):.0f} p75={np.percentile(E,75):.0f} p95={np.percentile(E,95):.0f}")
ic_all = float(pearsonr(P, R)[0])
print(f"Overall IC: {ic_all:.4f}")

q25, q50, q75 = np.percentile(E, [25, 50, 75])
print(f"Q thresholds: {q25:.0f} {q50:.0f} {q75:.0f}")
qr = {}
for nm, mask in [("Q1", E<=q25), ("Q2", (E>q25)&(E<=q50)), ("Q3", (E>q50)&(E<=q75)), ("Q4", E>q75)]:
    v = ic(P, R, mask)
    me = float(E[mask].mean()) if mask.sum() > 0 else 0
    qr[nm] = {"n": int(mask.sum()), "ic": v, "evts_mean": me}
    print(f"  {nm}: n={mask.sum():,} e/s={me:.0f} IC={v}")

ar = {}
for nm, mask in [("lt100", E<100), ("100_300", (E>=100)&(E<300)), ("300_600", (E>=300)&(E<600)), ("600_1000", (E>=600)&(E<1000)), ("gt1000", E>=1000)]:
    v = ic(P, R, mask)
    me = float(E[mask].mean()) if mask.sum() > 0 else 0
    ar[nm] = {"n": int(mask.sum()), "ic": v, "evts_mean": me}
    print(f"  abs {nm}: n={mask.sum():,} e/s={me:.0f} IC={v}")

tr = {}
for nm, mask in [("open", T<0.2), ("midlow", (T>=0.2)&(T<0.4)), ("midday", (T>=0.4)&(T<0.6)), ("midhigh", (T>=0.6)&(T<0.8)), ("close", T>=0.8)]:
    v = ic(P, R, mask)
    me = float(E[mask].mean()) if mask.sum() > 0 else 0
    tr[nm] = {"n": int(mask.sum()), "ic": v, "evts_mean": me}
    print(f"  tod {nm}: n={mask.sum():,} e/s={me:.0f} IC={v}")

res = {
    "overall_ic": ic_all,
    "n": int(len(P)),
    "dates": int(len(AP)),
    "evts_pcts": {"p5": float(np.percentile(E,5)), "p25": float(np.percentile(E,25)), "p50": float(np.percentile(E,50)), "p75": float(np.percentile(E,75)), "p95": float(np.percentile(E,95))},
    "q_thresholds": {"q25": float(q25), "q50": float(q50), "q75": float(q75)},
    "quartiles": qr,
    "abs_bins": ar,
    "tod": tr
}
open("/home/jupiter/events_per_sec_ic.json","w").write(json.dumps(res, indent=2))
raise RuntimeError(f"DONE ic={ic_all:.4f} Q1_ic={qr['Q1']['ic']} Q4_ic={qr['Q4']['ic']} n={len(P):,}")
