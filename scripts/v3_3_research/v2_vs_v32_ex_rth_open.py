"""
HC #317 — v2 vs v3.2 head-to-head, EXCLUDING first 30 min of RTH.

Hypothesis: RTH-open destroys v3.2 longer-horizon signal (regime-blind preds vs
1.66× wider open vol). When we strip those 30 min, does v3.2 actually beat v2?

Both models score the same 5 OOT days (20260223-20260227).

v2 config: STRIDE=500, WINDOW=1000, horizons=[1s, 5s, 10s], shape (N,3) per fold
v3.2 config: STRIDE=250, WINDOW=1500, horizons=[1s,5s,10s,30s,60s,5min]+aux heads

Output: output/v3_2_deep_sim_20260512/v2_vs_v32_ex_rth_open.json
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

MBO_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/mbo_events_smart_v3")
V2_DIR = Path("/home/jupiter/Lvl3Quant/output/cnn_mamba_v2_smart_v3_mar")
V32_PREDS = Path("/home/jupiter/Lvl3Quant/output/v3_2_deep_sim_20260512/fold_00_oot_predictions.npz")
OUT = Path("/home/jupiter/Lvl3Quant/output/v3_2_deep_sim_20260512/v2_vs_v32_ex_rth_open.json")

V2_STRIDE, V2_WINDOW = 500, 1000
V32_STRIDE, V32_WINDOW = 250, 1500
OOT_DATES = ["20260223", "20260224", "20260225", "20260226", "20260227"]
HORIZONS = {"1s": 0, "5s": 1, "10s": 2}


def safe_ic(p, t):
    m = np.isfinite(p) & np.isfinite(t)
    if m.sum() < 50:
        return float("nan"), int(m.sum())
    try:
        r, _ = spearmanr(p[m], t[m])
    except Exception:
        r = float("nan")
    return float(r), int(m.sum())


def safe_da(p, t):
    m = np.isfinite(p) & np.isfinite(t)
    if m.sum() < 50:
        return float("nan"), int(m.sum())
    return float((np.sign(p[m]) == np.sign(t[m])).mean() * 100.0), int(m.sum())


def build_timestamps(date: str, stride: int, window: int, n_pred: int) -> np.ndarray:
    mbo = np.load(MBO_DIR / f"{date}_mbo_events.npz", allow_pickle=True)
    ts = mbo["timestamps"]
    n_ev = len(ts)
    out = np.zeros(n_pred, dtype=np.int64)
    for i in range(n_pred):
        ev_idx = min(i * stride + window - 1, n_ev - 1)
        out[i] = ts[ev_idx]
    return out


def session_masks(ts_ns: np.ndarray) -> dict:
    dt64 = ts_ns.astype("datetime64[ns]")
    minutes_utc = (dt64.astype("datetime64[m]") - dt64.astype("datetime64[D]").astype("datetime64[m]")).astype(int)
    minutes_et = (minutes_utc - 5 * 60) % (24 * 60)
    rth_open  = (minutes_et >= 9 * 60 + 30) & (minutes_et < 10 * 60 + 30)
    rth_mid   = (minutes_et >= 10 * 60 + 30) & (minutes_et < 15 * 60)
    rth_close = (minutes_et >= 15 * 60) & (minutes_et < 16 * 60)
    rth_any   = (minutes_et >= 9 * 60 + 30) & (minutes_et < 16 * 60)
    non_rth   = ~rth_any
    ex_open   = ~rth_open
    return {
        "all": np.ones(len(ts_ns), dtype=bool),
        "ex_first_30m_rth": ex_open,
        "rth_open_30m": rth_open,
        "rth_mid": rth_mid,
        "rth_close_60m": rth_close,
        "non_rth": non_rth,
    }


def conf_band(pred, target, mask, frac):
    """IC + DA restricted to top-frac of |pred| within the mask."""
    m = mask & np.isfinite(pred) & np.isfinite(target)
    if m.sum() < 200:
        return {"n": int(m.sum()), "IC": float("nan"), "DA": float("nan")}
    abs_pred = np.abs(pred)
    thr = float(np.quantile(abs_pred[m], 1.0 - frac))
    sel = m & (abs_pred >= thr)
    if sel.sum() < 20:
        return {"n": int(sel.sum()), "IC": float("nan"), "DA": float("nan")}
    ic, _ = safe_ic(pred[sel], target[sel])
    da, _ = safe_da(pred[sel], target[sel])
    return {"n": int(sel.sum()), "IC": ic, "DA_pct": da}


def main():
    # ============== Load v3.2 (single concat npz, all 5 OOT days) ==============
    v32 = np.load(V32_PREDS, allow_pickle=True)
    n32 = int(v32["n_samples"])
    # Build v3.2 timestamps by stitching all 5 OOT days
    v32_ts_parts = []
    for date in OOT_DATES:
        mbo = np.load(MBO_DIR / f"{date}_mbo_events.npz", allow_pickle=True)
        ts = mbo["timestamps"]; n_ev = len(ts)
        n_steps = max(0, (n_ev - V32_WINDOW) // V32_STRIDE + 1)
        sei = V32_WINDOW - 1 + np.arange(n_steps) * V32_STRIDE
        sei = sei[sei < n_ev]
        v32_ts_parts.append(ts[sei])
    v32_ts = np.concatenate(v32_ts_parts)[:n32]
    v32_masks = session_masks(v32_ts)

    # ============== Load v2 (per-fold, concatenate) ==============
    v2_preds = {h: [] for h in HORIZONS}
    v2_targs = {h: [] for h in HORIZONS}
    v2_ts_parts = []
    for fold, date in enumerate(OOT_DATES):
        f = V2_DIR / f"fold_{fold:02d}_oot_predictions.npz"
        d = np.load(f, allow_pickle=True)
        p = d["predictions"]; lab = d["labels"]
        n_pred = p.shape[0]
        ts = build_timestamps(date, V2_STRIDE, V2_WINDOW, n_pred)
        v2_ts_parts.append(ts)
        for h, hi in HORIZONS.items():
            v2_preds[h].append(p[:, hi])
            v2_targs[h].append(lab[:, hi])
    v2_ts = np.concatenate(v2_ts_parts)
    v2_preds = {h: np.concatenate(v2_preds[h]) for h in HORIZONS}
    v2_targs = {h: np.concatenate(v2_targs[h]) for h in HORIZONS}
    v2_masks = session_masks(v2_ts)

    print(f"v2 n_total: {len(v2_ts):,}")
    print(f"v3.2 n_total: {n32:,}")
    print(f"v2 ex_open: {v2_masks['ex_first_30m_rth'].sum():,}")
    print(f"v3.2 ex_open: {v32_masks['ex_first_30m_rth'].sum():,}")

    findings = {
        "setup": {
            "v2_n": len(v2_ts),
            "v32_n": n32,
            "v2_stride_window": [V2_STRIDE, V2_WINDOW],
            "v32_stride_window": [V32_STRIDE, V32_WINDOW],
            "oot_dates": OOT_DATES,
        },
        "comparisons": {},
    }

    BUCKETS = ["all", "ex_first_30m_rth", "rth_open_30m", "rth_mid"]
    for bucket in BUCKETS:
        block = {}
        for h in HORIZONS:
            v2_p = v2_preds[h]; v2_t = v2_targs[h]; m2 = v2_masks[bucket]
            v32_p = v32[f"pred_log_ret_{h}"][:n32]
            v32_t = v32[f"target_log_ret_{h}"][:n32]
            v32_m = v32[f"mask_log_ret_{h}"][:n32].astype(bool) & v32_masks[bucket]

            # Aggregate IC + DA
            v2_ic, v2_n = safe_ic(v2_p[m2], v2_t[m2])
            v2_da, _ = safe_da(v2_p[m2], v2_t[m2])
            v32_ic, v32_n = safe_ic(v32_p[v32_m], v32_t[v32_m])
            v32_da, _ = safe_da(v32_p[v32_m], v32_t[v32_m])

            # Confidence bands (top 1%/5%/10%)
            v2_bands = {f"top{int(b*100)}pct": conf_band(v2_p, v2_t, m2, b) for b in (0.01, 0.05, 0.10)}
            v32_bands = {f"top{int(b*100)}pct": conf_band(v32_p, v32_t, v32_m, b) for b in (0.01, 0.05, 0.10)}

            block[h] = {
                "v2_agg":  {"n": v2_n, "IC": v2_ic, "DA_pct": v2_da},
                "v32_agg": {"n": v32_n, "IC": v32_ic, "DA_pct": v32_da},
                "Δ_IC_agg": (v32_ic - v2_ic) if (np.isfinite(v2_ic) and np.isfinite(v32_ic)) else None,
                "v2_bands": v2_bands,
                "v32_bands": v32_bands,
            }
        findings["comparisons"][bucket] = block

    # ============== Verdict ==============
    verdict_lines = []
    for h in HORIZONS:
        e = findings["comparisons"]["ex_first_30m_rth"][h]
        a = findings["comparisons"]["all"][h]
        verdict_lines.append(
            f"{h}: agg IC v2→v32 ALL: {a['v2_agg']['IC']:.4f}→{a['v32_agg']['IC']:.4f} (Δ={a['Δ_IC_agg']:+.4f}) | "
            f"EX-OPEN: {e['v2_agg']['IC']:.4f}→{e['v32_agg']['IC']:.4f} (Δ={e['Δ_IC_agg']:+.4f})"
        )
    findings["verdict_lines"] = verdict_lines

    with open(OUT, "w") as f:
        json.dump(findings, f, indent=2,
                  default=lambda x: None if (isinstance(x, float) and not np.isfinite(x)) else x)

    print("\n=== AGG (ALL session) ===")
    for h in HORIZONS:
        v2_agg = findings["comparisons"]["all"][h]["v2_agg"]
        v32_agg = findings["comparisons"]["all"][h]["v32_agg"]
        print(f"{h}: v2 IC={v2_agg['IC']:.4f} DA={v2_agg['DA_pct']:.2f}%  v3.2 IC={v32_agg['IC']:.4f} DA={v32_agg['DA_pct']:.2f}%  Δ_IC={v32_agg['IC']-v2_agg['IC']:+.4f}")

    print("\n=== EX_FIRST_30M_RTH ===")
    for h in HORIZONS:
        v2_agg = findings["comparisons"]["ex_first_30m_rth"][h]["v2_agg"]
        v32_agg = findings["comparisons"]["ex_first_30m_rth"][h]["v32_agg"]
        print(f"{h}: v2 IC={v2_agg['IC']:.4f} DA={v2_agg['DA_pct']:.2f}%  v3.2 IC={v32_agg['IC']:.4f} DA={v32_agg['DA_pct']:.2f}%  Δ_IC={v32_agg['IC']-v2_agg['IC']:+.4f}")

    print("\n=== RTH_OPEN_30M ===")
    for h in HORIZONS:
        v2_agg = findings["comparisons"]["rth_open_30m"][h]["v2_agg"]
        v32_agg = findings["comparisons"]["rth_open_30m"][h]["v32_agg"]
        print(f"{h}: v2 IC={v2_agg['IC']:.4f} DA={v2_agg['DA_pct']:.2f}%  v3.2 IC={v32_agg['IC']:.4f} DA={v32_agg['DA_pct']:.2f}%  Δ_IC={v32_agg['IC']-v2_agg['IC']:+.4f}")

    print("\n=== TOP-1% CONFIDENCE BAND (EX-OPEN) ===")
    for h in HORIZONS:
        v2_b = findings["comparisons"]["ex_first_30m_rth"][h]["v2_bands"]["top1pct"]
        v32_b = findings["comparisons"]["ex_first_30m_rth"][h]["v32_bands"]["top1pct"]
        print(f"{h}: v2 IC={v2_b['IC']:.4f} DA={v2_b['DA_pct']:.2f}% (n={v2_b['n']})  v3.2 IC={v32_b['IC']:.4f} DA={v32_b['DA_pct']:.2f}% (n={v32_b['n']})")

    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
