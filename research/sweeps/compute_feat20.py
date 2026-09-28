import numpy as np, glob, os, time

IN_DIR  = "/home/jupiter/Lvl3Quant/data/processed/mbo_events_feat18/"
OUT_DIR = "/home/jupiter/Lvl3Quant/data/processed/mbo_events_feat20/"
LOG     = "/home/jupiter/Lvl3Quant/compute_feat20.log"
os.makedirs(OUT_DIR, exist_ok=True)

FEAT_NAMES = [
    "time_delta_log","event_type_id","side_id","price_rel_ticks",
    "qty_log","spread_ticks","cancel_side_asym_50","rolling_ofi_500",
    "event_density_20","price_mom_10","qty_price_mom_50",
    "price_sign_mom_200","event_type_entropy_200","fill_add_restore_100",
    "spread_velocity_50","price_sign_sq","fade_side_price","restore_x_price",
    "rolling_rvol_500","rolling_vwap_dist_200"
]

def rolling_std_fast(x, W):
    x = x.astype(np.float64)
    n = len(x)
    out = np.empty(n, dtype=np.float32)
    cs  = np.concatenate([[0.0], np.cumsum(x)])
    cs2 = np.concatenate([[0.0], np.cumsum(x*x)])
    sums  = cs[W:] - cs[:n-W+1]
    sums2 = cs2[W:] - cs2[:n-W+1]
    mean  = sums / W
    var   = sums2/W - mean**2
    np.clip(var, 0, None, out=var)
    out[W-1:] = np.sqrt(var).astype(np.float32)
    for i in range(W-1):
        out[i] = float(x[:i+1].std()) if i > 0 else 0.0
    return out

def rolling_vwap_dist(price, qty_log, W):
    price = price.astype(np.float64)
    qty = np.exp(np.clip(qty_log.astype(np.float64), -5, 10))
    pq = price * qty
    n = len(price)
    cs_pq = np.concatenate([[0.0], np.cumsum(pq)])
    cs_q  = np.concatenate([[0.0], np.cumsum(qty)])
    out = np.zeros(n, dtype=np.float32)
    sum_pq = cs_pq[W:] - cs_pq[:n-W+1]
    sum_q  = cs_q[W:]  - cs_q[:n-W+1]
    vwap = sum_pq / np.maximum(sum_q, 1e-8)
    out[W-1:] = (price[W-1:] - vwap).astype(np.float32)
    cum_pq, cum_q = 0.0, 0.0
    for i in range(min(W-1, n)):
        cum_pq += pq[i]; cum_q += qty[i]
        out[i] = float(price[i] - cum_pq/cum_q) if cum_q > 1e-8 else 0.0
    return out

def log(msg):
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(msg + "\n")

files = sorted(glob.glob(IN_DIR + "*.npz"))
log(f"feat18->feat20 | {len(files)} files | +rolling_rvol_500 +rolling_vwap_dist_200")
t0 = time.time()

for fi, fpath in enumerate(files):
    fname = os.path.basename(fpath)
    out_path = os.path.join(OUT_DIR, fname)
    if os.path.exists(out_path):
        continue
    try:
        d = np.load(fpath, allow_pickle=True)
        ev = d["events"].astype(np.float32)
        N = ev.shape[0]
        if ev.shape[1] != 18:
            log(f"  SKIP {fname}: expected 18 cols, got {ev.shape[1]}")
            continue
        price = ev[:, 3]
        qty_l = ev[:, 4]
        rvol  = rolling_std_fast(price, 500)
        vdist = rolling_vwap_dist(price, qty_l, 200)
        ev20  = np.concatenate([ev, rvol[:,None], vdist[:,None]], axis=1)
        save_dict = {"events": ev20, "feature_names": np.array(FEAT_NAMES)}
        for k in d.files:
            if k not in ("events", "feature_names"):
                save_dict[k] = d[k]
        np.savez_compressed(out_path, **save_dict)
        if (fi+1) % 20 == 0:
            log(f"  [{fi+1}/{len(files)}] {time.time()-t0:.0f}s")
    except Exception as e:
        log(f"  ERROR {fname}: {e}")

log(f"Done in {time.time()-t0:.0f}s. Out: {OUT_DIR}")
