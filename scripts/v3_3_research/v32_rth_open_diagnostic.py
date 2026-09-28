"""
RTH-open diagnostic — WHY does the open-30m destroy v3.2's longer-horizon signal?

Hypotheses tested:
  H1: Vol distribution at open is wider → harder regression → worse IC
  H2: Event-type mix at open differs (more sweeps/trades, fewer adds)
  H3: Realized log_ret distribution at open has fatter tails → fewer mid-band hits
  H4: Predictions are systematically biased at open (skewed pred distribution)
  H5: target_log_ret_30s distribution at open is meaningfully different

Compares RTH-open-30m vs RTH-mid for each hypothesis.

Output: output/v3_2_deep_sim_20260512/rth_open_diagnostic.json
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy import stats

MBO_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/mbo_events_smart_v3")
PREDS_PATH = Path("/home/jupiter/Lvl3Quant/output/v3_2_deep_sim_20260512/fold_00_oot_predictions.npz")
OUT_JSON = Path("/home/jupiter/Lvl3Quant/output/v3_2_deep_sim_20260512/rth_open_diagnostic.json")

STRIDE = 250
WINDOW = 1500
OOT_DATES = ["20260223", "20260224", "20260225", "20260226", "20260227"]
HORIZONS = ["log_ret_1s", "log_ret_5s", "log_ret_10s", "log_ret_30s"]


def main():
    pred = np.load(PREDS_PATH, allow_pickle=True)

    # Rebuild per-pred metadata (timestamps + event types)
    all_ts, all_etype = [], []
    for date in OOT_DATES:
        d = np.load(MBO_DIR / f"{date}_mbo_events.npz", allow_pickle=True)
        n_events = d["timestamps"].shape[0]
        n_steps = max(0, (n_events - WINDOW) // STRIDE + 1)
        sei = WINDOW - 1 + np.arange(n_steps) * STRIDE
        sei = sei[sei < n_events]
        all_ts.append(d["timestamps"][sei])
        all_etype.append(d["event_type_raw"][sei])
    all_ts = np.concatenate(all_ts)
    all_etype = np.concatenate(all_etype)

    n = min(int(pred["n_samples"]), len(all_ts))
    all_ts = all_ts[:n]
    all_etype = all_etype[:n]

    ts_dt64 = all_ts.astype("datetime64[ns]")
    minutes_utc = (ts_dt64.astype("datetime64[m]") - ts_dt64.astype("datetime64[D]").astype("datetime64[m]")).astype(int)
    minutes_et = (minutes_utc - 5 * 60) % (24 * 60)

    rth_open = (minutes_et >= 9 * 60 + 30) & (minutes_et < 10 * 60 + 30)
    rth_mid = (minutes_et >= 10 * 60 + 30) & (minutes_et < 15 * 60)

    print(f"RTH open samples: {rth_open.sum():,}")
    print(f"RTH mid samples:  {rth_mid.sum():,}")

    findings = {"sample_counts": {"open_30m": int(rth_open.sum()), "mid": int(rth_mid.sum())}}

    # H1: Vol distribution (|target_5s|)
    abs_5s = np.abs(pred["target_log_ret_5s"][:n])
    findings["H1_vol_distribution"] = {
        "open_mean_abs_5s": float(np.nanmean(abs_5s[rth_open])),
        "mid_mean_abs_5s": float(np.nanmean(abs_5s[rth_mid])),
        "open_std_abs_5s": float(np.nanstd(abs_5s[rth_open])),
        "mid_std_abs_5s": float(np.nanstd(abs_5s[rth_mid])),
        "open_p99_abs_5s": float(np.nanquantile(abs_5s[rth_open], 0.99)),
        "mid_p99_abs_5s": float(np.nanquantile(abs_5s[rth_mid], 0.99)),
    }

    # H2: Event-type mix
    et_open, cnt_open = np.unique(all_etype[rth_open], return_counts=True)
    et_mid, cnt_mid = np.unique(all_etype[rth_mid], return_counts=True)
    et_names = {0: "type_0", 1: "add", 2: "cancel", 3: "modify", 4: "clear", 5: "trade"}
    findings["H2_event_type_mix"] = {
        "open_pct": {et_names.get(int(e), f"type_{int(e)}"): float(c / cnt_open.sum() * 100) for e, c in zip(et_open, cnt_open)},
        "mid_pct": {et_names.get(int(e), f"type_{int(e)}"): float(c / cnt_mid.sum() * 100) for e, c in zip(et_mid, cnt_mid)},
    }

    # H3 & H5: Realized return distribution per horizon
    findings["H3_target_distribution"] = {}
    for h in HORIZONS:
        t = pred[f"target_{h}"][:n]
        m = pred[f"mask_{h}"][:n].astype(bool)
        t_open = t[rth_open & m]
        t_mid = t[rth_mid & m]
        if len(t_open) > 100 and len(t_mid) > 100:
            findings["H3_target_distribution"][h] = {
                "open_mean": float(np.mean(t_open)),
                "mid_mean": float(np.mean(t_mid)),
                "open_std": float(np.std(t_open)),
                "mid_std": float(np.std(t_mid)),
                "open_skew": float(stats.skew(t_open)),
                "mid_skew": float(stats.skew(t_mid)),
                "open_kurt": float(stats.kurtosis(t_open)),
                "mid_kurt": float(stats.kurtosis(t_mid)),
                "open_p1": float(np.quantile(t_open, 0.01)),
                "open_p99": float(np.quantile(t_open, 0.99)),
                "mid_p1": float(np.quantile(t_mid, 0.01)),
                "mid_p99": float(np.quantile(t_mid, 0.99)),
            }

    # H4: Prediction distribution
    findings["H4_pred_distribution"] = {}
    for h in HORIZONS:
        p = pred[f"pred_{h}"][:n]
        m = pred[f"mask_{h}"][:n].astype(bool) & np.isfinite(p)
        p_open = p[rth_open & m]
        p_mid = p[rth_mid & m]
        if len(p_open) > 100 and len(p_mid) > 100:
            findings["H4_pred_distribution"][h] = {
                "open_mean": float(np.mean(p_open)),
                "mid_mean": float(np.mean(p_mid)),
                "open_std": float(np.std(p_open)),
                "mid_std": float(np.std(p_mid)),
                "open_abs_mean": float(np.mean(np.abs(p_open))),
                "mid_abs_mean": float(np.mean(np.abs(p_mid))),
            }

    # Summary verdict
    h1 = findings["H1_vol_distribution"]
    vol_ratio = h1["open_std_abs_5s"] / h1["mid_std_abs_5s"]
    findings["VERDICT"] = {
        "vol_inflation_at_open": f"{vol_ratio:.2f}x vs mid (std of |target_5s|)",
        "p99_inflation_at_open": f"{h1['open_p99_abs_5s']/h1['mid_p99_abs_5s']:.2f}x vs mid (p99 of |target_5s|)",
    }
    # Pred-magnitude inflation
    for h in HORIZONS:
        if h in findings["H4_pred_distribution"]:
            ratio = findings["H4_pred_distribution"][h]["open_abs_mean"] / findings["H4_pred_distribution"][h]["mid_abs_mean"]
            findings["VERDICT"][f"pred_abs_inflation_at_open_{h}"] = f"{ratio:.2f}x"

    with open(OUT_JSON, "w") as f:
        json.dump(findings, f, indent=2)
    print(json.dumps(findings, indent=2))


if __name__ == "__main__":
    main()
