"""
v3.2 event-type + clock-hour stratification using actual MBO event timestamps.

Loads each OOT day's MBO event file, downsamples to STRIDE positions to align
with predictions.npz, then stratifies the 1s/5s/10s/30s edge by:
  1. Clock hour (RTH 09:30-16:00 ET, premarket, postmarket)
  2. MBO event type at the prediction step (add/cancel/trade/modify/etc)
  3. Combination: clock × event-type

Output: output/v3_2_deep_sim_20260512/event_type_clock_audit.csv + .json

Why this matters: if 1s edge is concentrated on specific event types (e.g. sweeps,
spread-crossing trades) and specific hours (e.g. opening 30 min), v3.3 feature
engineering can sharpen those signals. If edge is diffuse → architecture-only fix.
"""
from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

MBO_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/mbo_events_smart_v3")
PREDS_PATH = Path("/home/jupiter/Lvl3Quant/output/v3_2_deep_sim_20260512/fold_00_oot_predictions.npz")
OUT_CSV = Path("/home/jupiter/Lvl3Quant/output/v3_2_deep_sim_20260512/event_type_clock_audit.csv")
OUT_JSON = Path("/home/jupiter/Lvl3Quant/output/v3_2_deep_sim_20260512/event_type_clock_audit.json")

STRIDE = 250
WINDOW = 1500  # window_size_t1
OOT_DATES = ["20260223", "20260224", "20260225", "20260226", "20260227"]
HORIZONS = ["log_ret_1s", "log_ret_5s", "log_ret_10s", "log_ret_30s"]

# Databento MBO event type mapping (per smart_v3 preprocessing)
EVENT_TYPE_NAMES = {
    1: "add", 2: "cancel", 3: "modify", 4: "clear",
    5: "trade", 6: "fill", 10: "snapshot", -1: "unknown",
}


def ic_da(pred, target, mask):
    m = mask.astype(bool) & np.isfinite(pred) & np.isfinite(target)
    if m.sum() < 50:
        return float("nan"), float("nan"), int(m.sum())
    p, t = pred[m], target[m]
    try:
        ic, _ = spearmanr(p, t)
    except Exception:
        ic = float("nan")
    da = float(((p > 0) == (t > 0)).mean())
    return float(ic), float(da), int(m.sum())


def main():
    pred = np.load(PREDS_PATH, allow_pickle=True)

    # Build per-prediction-step metadata: timestamp + event_type at the END of window
    all_ts = []
    all_etype = []
    day_offsets = [0]
    for date in OOT_DATES:
        f = MBO_DIR / f"{date}_mbo_events.npz"
        d = np.load(f, allow_pickle=True)
        n_events = d["timestamps"].shape[0]
        # Prediction step k corresponds to event index WINDOW-1 + k*STRIDE
        n_steps = max(0, (n_events - WINDOW) // STRIDE + 1)
        step_event_idx = WINDOW - 1 + np.arange(n_steps) * STRIDE
        step_event_idx = step_event_idx[step_event_idx < n_events]
        ts = d["timestamps"][step_event_idx]
        et = d["event_type_raw"][step_event_idx]
        all_ts.append(ts)
        all_etype.append(et)
        day_offsets.append(day_offsets[-1] + len(ts))
        print(f"  {date}: {n_events:,} events → {len(ts):,} pred-steps  (cum {day_offsets[-1]:,})")

    all_ts = np.concatenate(all_ts)
    all_etype = np.concatenate(all_etype)

    n_pred = int(pred["n_samples"])
    print(f"Predictions: {n_pred:,}   metadata: {len(all_ts):,}")

    n = min(n_pred, len(all_ts))
    all_ts = all_ts[:n]
    all_etype = all_etype[:n]

    # Convert ts (nanoseconds since epoch UTC) → ET clock hour
    # ET = UTC-5 (or UTC-4 in DST). Feb dates → EST = UTC-5
    ts_dt64 = all_ts.astype("datetime64[ns]")
    hours_utc = (ts_dt64.astype("datetime64[h]") - ts_dt64.astype("datetime64[D]").astype("datetime64[h]")).astype(int)
    hours_et = (hours_utc - 5) % 24

    # === Axis A: CLOCK HOUR (ET) ===
    rows = []
    for hr in range(24):
        m_hr = hours_et == hr
        if m_hr.sum() < 100:
            continue
        for h in HORIZONS:
            p = pred[f"pred_{h}"][:n][m_hr]
            t = pred[f"target_{h}"][:n][m_hr]
            mk = pred[f"mask_{h}"][:n][m_hr].astype(bool)
            ic, da, nn = ic_da(p, t, mk)
            rows.append({"axis": "clock_hour_ET", "bucket": f"{hr:02d}:00", "horizon": h, "n": nn, "IC": ic, "DA": da})

    # === Axis B: EVENT TYPE at prediction step ===
    unique_et = sorted(np.unique(all_etype).tolist())
    for et_id in unique_et:
        m_et = all_etype == et_id
        if m_et.sum() < 100:
            continue
        et_name = EVENT_TYPE_NAMES.get(int(et_id), f"type_{int(et_id)}")
        for h in HORIZONS:
            p = pred[f"pred_{h}"][:n][m_et]
            t = pred[f"target_{h}"][:n][m_et]
            mk = pred[f"mask_{h}"][:n][m_et].astype(bool)
            ic, da, nn = ic_da(p, t, mk)
            rows.append({"axis": "event_type", "bucket": f"{int(et_id)}_{et_name}", "horizon": h, "n": nn, "IC": ic, "DA": da})

    # === Axis C: RTH BUCKETS (09:30-10:30 open, 10:30-15:00 mid, 15:00-16:00 close, other = non-RTH) ===
    minutes_et = (ts_dt64.astype("datetime64[m]") - ts_dt64.astype("datetime64[D]").astype("datetime64[m]")).astype(int)
    minutes_et = (minutes_et - 5 * 60) % (24 * 60)  # ET minutes-of-day
    rth_open = (minutes_et >= 9 * 60 + 30) & (minutes_et < 10 * 60 + 30)
    rth_mid = (minutes_et >= 10 * 60 + 30) & (minutes_et < 15 * 60)
    rth_close = (minutes_et >= 15 * 60) & (minutes_et < 16 * 60)
    rth_full = (minutes_et >= 9 * 60 + 30) & (minutes_et < 16 * 60)
    non_rth = ~rth_full

    for name, mask_b in [("RTH_open_30m", rth_open), ("RTH_mid", rth_mid), ("RTH_close_60m", rth_close), ("non_RTH", non_rth)]:
        if mask_b.sum() < 100:
            continue
        for h in HORIZONS:
            p = pred[f"pred_{h}"][:n][mask_b]
            t = pred[f"target_{h}"][:n][mask_b]
            mk = pred[f"mask_{h}"][:n][mask_b].astype(bool)
            ic, da, nn = ic_da(p, t, mk)
            rows.append({"axis": "session_bucket", "bucket": name, "horizon": h, "n": nn, "IC": ic, "DA": da})

    # Write
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["axis", "bucket", "horizon", "n", "IC", "DA"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Summary
    summary = {
        "n_pred_aligned": n,
        "clock_hour_IC_1s_top5": sorted(
            [(r["bucket"], r["IC"], r["n"]) for r in rows if r["axis"] == "clock_hour_ET" and r["horizon"] == "log_ret_1s" and r["n"] >= 1000],
            key=lambda x: -x[1] if x[1] == x[1] else 0,
        )[:8],
        "event_type_IC_1s": {
            r["bucket"]: {"IC": r["IC"], "DA": r["DA"], "n": r["n"]}
            for r in rows if r["axis"] == "event_type" and r["horizon"] == "log_ret_1s"
        },
        "session_bucket_IC": {
            r["bucket"]: {h: next((x["IC"] for x in rows if x["axis"] == "session_bucket" and x["bucket"] == r["bucket"] and x["horizon"] == h), None) for h in HORIZONS}
            for r in rows if r["axis"] == "session_bucket" and r["horizon"] == "log_ret_1s"
        },
    }
    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2, default=lambda x: None if (isinstance(x, float) and not np.isfinite(x)) else x)

    print(f"\nWrote {len(rows)} rows to {OUT_CSV}")
    print(f"Wrote summary to {OUT_JSON}")
    print("\n--- SUMMARY ---")
    print(json.dumps(summary, indent=2, default=lambda x: None if (isinstance(x, float) and not np.isfinite(x)) else x))


if __name__ == "__main__":
    main()
