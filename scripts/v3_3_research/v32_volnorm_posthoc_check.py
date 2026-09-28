"""
Post-hoc vol-normalization sanity check.

Question: If we scale v3.2's existing predictions by realized vol at the prediction
step, does IC recover at RTH-open? This tests whether v3.3.1 Option C (vol-normalized
targets) is worth the rebuild — if post-hoc scaling improves IC, then training with
vol-norm targets should be even better.

Method:
  - Compute realized vol per prediction step from the MBO event window (std of
    last K=300 events' price changes, ~30s rolling vol proxy)
  - Scale predictions: pred_norm = pred * realized_vol
  - Compute IC of (pred_norm, target) per horizon × per session bucket
  - Compare vs raw IC

Output: output/v3_2_deep_sim_20260512/volnorm_posthoc.json
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

MBO_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/mbo_events_smart_v3")
PREDS_PATH = Path("/home/jupiter/Lvl3Quant/output/v3_2_deep_sim_20260512/fold_00_oot_predictions.npz")
OUT_JSON = Path("/home/jupiter/Lvl3Quant/output/v3_2_deep_sim_20260512/volnorm_posthoc.json")

STRIDE = 250
WINDOW = 1500
VOL_LOOKBACK = 300  # ~30s of MBO events for realized vol proxy
OOT_DATES = ["20260223", "20260224", "20260225", "20260226", "20260227"]
HORIZONS = ["log_ret_1s", "log_ret_5s", "log_ret_10s", "log_ret_30s"]


def ic(pred, target, mask):
    m = mask.astype(bool) & np.isfinite(pred) & np.isfinite(target)
    if m.sum() < 50:
        return float("nan"), int(m.sum())
    try:
        r, _ = spearmanr(pred[m], target[m])
    except Exception:
        r = float("nan")
    return float(r), int(m.sum())


def main():
    pred = np.load(PREDS_PATH, allow_pickle=True)

    # Compute per-step realized vol from MBO event price changes
    # The smart_v3 'events' array has 25 features; we need price. Per the trainer
    # spec, feature index 0 = bid_price (smart normalized). We'll use abs diff
    # of bid_price as a proxy for realized vol over the lookback window.
    all_ts = []
    all_realized_vol = []
    for date in OOT_DATES:
        d = np.load(MBO_DIR / f"{date}_mbo_events.npz", allow_pickle=True)
        n_events = d["timestamps"].shape[0]
        events = d["events"]  # (N, 25)
        # Price proxy: use feature 0 (already normalized). Compute |Δ| over lookback.
        price_proxy = events[:, 0]
        n_steps = max(0, (n_events - WINDOW) // STRIDE + 1)
        sei = WINDOW - 1 + np.arange(n_steps) * STRIDE
        sei = sei[sei < n_events]
        # For each step, std of price_proxy[sei-VOL_LOOKBACK:sei]
        rv = np.zeros(len(sei), dtype=np.float32)
        for i, idx in enumerate(sei):
            lo = max(0, idx - VOL_LOOKBACK)
            seg = price_proxy[lo:idx]
            rv[i] = float(np.std(seg)) if len(seg) > 5 else 1.0
        all_ts.append(d["timestamps"][sei])
        all_realized_vol.append(rv)
        print(f"  {date}: {len(sei):,} steps  rv_mean={float(np.mean(rv)):.4f} rv_std={float(np.std(rv)):.4f}")

    all_ts = np.concatenate(all_ts)
    all_realized_vol = np.concatenate(all_realized_vol)

    n = min(int(pred["n_samples"]), len(all_ts))
    all_ts = all_ts[:n]
    all_realized_vol = all_realized_vol[:n]

    # Session bucketing (ET)
    ts_dt64 = all_ts.astype("datetime64[ns]")
    minutes_utc = (ts_dt64.astype("datetime64[m]") - ts_dt64.astype("datetime64[D]").astype("datetime64[m]")).astype(int)
    minutes_et = (minutes_utc - 5 * 60) % (24 * 60)
    rth_open = (minutes_et >= 9 * 60 + 30) & (minutes_et < 10 * 60 + 30)
    rth_mid = (minutes_et >= 10 * 60 + 30) & (minutes_et < 15 * 60)
    rth_close = (minutes_et >= 15 * 60) & (minutes_et < 16 * 60)
    non_rth = ~(rth_open | rth_mid | rth_close)

    # Normalize realized vol to mean=1 for scaling
    rv_norm = all_realized_vol / max(float(np.mean(all_realized_vol)), 1e-6)
    print(f"\nRealized vol stats: mean(raw)={float(np.mean(all_realized_vol)):.4f}  rv_norm mean={float(np.mean(rv_norm)):.4f}")
    print(f"  open rv_norm mean: {float(np.mean(rv_norm[rth_open])):.3f}")
    print(f"  mid  rv_norm mean: {float(np.mean(rv_norm[rth_mid])):.3f}")

    findings = {
        "rv_norm_by_session": {
            "open_mean": float(np.mean(rv_norm[rth_open])),
            "mid_mean": float(np.mean(rv_norm[rth_mid])),
            "close_mean": float(np.mean(rv_norm[rth_close])) if rth_close.sum() > 0 else None,
            "non_rth_mean": float(np.mean(rv_norm[non_rth])) if non_rth.sum() > 0 else None,
        },
        "ic_raw_vs_volnorm": {},
    }

    buckets = {"open_30m": rth_open, "mid": rth_mid, "close_60m": rth_close, "non_rth": non_rth, "all": np.ones(n, dtype=bool)}

    for bname, bmask in buckets.items():
        if bmask.sum() < 100:
            continue
        findings["ic_raw_vs_volnorm"][bname] = {}
        for h in HORIZONS:
            p_raw = pred[f"pred_{h}"][:n]
            t = pred[f"target_{h}"][:n]
            m = pred[f"mask_{h}"][:n].astype(bool) & bmask
            ic_raw, n_raw = ic(p_raw, t, m)
            # Vol-scaled prediction = pred * rv_norm
            p_scaled = p_raw * rv_norm
            ic_scaled, _ = ic(p_scaled, t, m)
            findings["ic_raw_vs_volnorm"][bname][h] = {
                "n": n_raw,
                "IC_raw": ic_raw,
                "IC_volnorm": ic_scaled,
                "Δ": (ic_scaled - ic_raw) if (np.isfinite(ic_raw) and np.isfinite(ic_scaled)) else None,
            }

    with open(OUT_JSON, "w") as f:
        json.dump(findings, f, indent=2, default=lambda x: None if (isinstance(x, float) and not np.isfinite(x)) else x)

    print(f"\nWrote {OUT_JSON}\n")
    print(json.dumps(findings, indent=2, default=lambda x: None if (isinstance(x, float) and not np.isfinite(x)) else x))


if __name__ == "__main__":
    main()
