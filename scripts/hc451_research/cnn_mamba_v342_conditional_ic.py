"""
HC #451 follow-up — is CNN-Mamba v3.4.2 already capturing salience?

For each OOT day with both predictions and salience tags:
- Reconstruct ts_ns per prediction from MBO events + (stride=250, window_t1=1500).
- For each prediction, flag recent_sweep / recent_large_print (any tag True in last 1.0s).
- Partition Spearman IC of pred_log_ret_{1s,5s,10s} vs target_log_ret_{1s,5s,10s}.
- Conditional MFE = mean(|target|) per subset.

Per-day stats appended to:
  /home/jupiter/Lvl3Quant/output/hc451_salience_tags/conditional_ic.csv
"""
from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

PROJECT_ROOT = Path("/home/jupiter/Lvl3Quant")
PRED_DIR = PROJECT_ROOT / "output/cnn_mamba_v3_4_2_fixedmtl/oot_47day_perdate"
SAL_DIR = PROJECT_ROOT / "output/hc451_salience_tags/per_day"
EVENTS_DIR = PROJECT_ROOT / "data/processed/mbo_events_smart_v3"
OUT_CSV = PROJECT_ROOT / "output/hc451_salience_tags/conditional_ic.csv"

WINDOW_T1 = 1500
STRIDE = 250
LOOKBACK_NS = 1_000_000_000  # 1.0 second
HORIZONS = ("1s", "5s", "10s")


def list_common_dates() -> list[str]:
    pred_dates = {f.replace("oot_", "").replace(".npz", "")
                  for f in os.listdir(PRED_DIR) if f.endswith(".npz")}
    sal_dates = {f.replace("_salience.parquet", "")
                 for f in os.listdir(SAL_DIR) if f.endswith("_salience.parquet")}
    return sorted(pred_dates & sal_dates)


def reconstruct_ts(date_str: str, n_preds: int) -> np.ndarray:
    """Return ts_ns aligned to each row of the prediction NPZ."""
    ev_path = EVENTS_DIR / f"{date_str}_mbo_events.npz"
    ev = np.load(ev_path, allow_pickle=True)
    ts = ev["timestamps"]
    labels = ev["labels_1s"]
    n_events = len(ts)
    out = []
    for start in range(0, n_events - WINDOW_T1 + 1, STRIDE):
        li = start + WINDOW_T1 - 1
        if not np.isnan(labels[li]):
            out.append(ts[li])
    arr = np.asarray(out, dtype=np.int64)
    if len(arr) != n_preds:
        raise RuntimeError(f"{date_str}: reconstructed {len(arr)} samples vs npz {n_preds}")
    return arr


def has_recent_tag(pred_ts: np.ndarray, tag_ts: np.ndarray) -> np.ndarray:
    """For each pred_ts, True if any tag fired in (pred_ts - LOOKBACK_NS, pred_ts]."""
    if len(tag_ts) == 0:
        return np.zeros(len(pred_ts), dtype=bool)
    # right boundary <= pred_ts
    right = np.searchsorted(tag_ts, pred_ts, side="right")
    # left boundary > pred_ts - LOOKBACK_NS  =>  side='right' on (pred_ts - lookback)
    left = np.searchsorted(tag_ts, pred_ts - LOOKBACK_NS, side="right")
    return (right - left) > 0


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 30:
        return float("nan")
    # Drop NaNs from either
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 30:
        return float("nan")
    rho, _ = spearmanr(x[m], y[m])
    return float(rho)


def main() -> int:
    dates = list_common_dates()
    print(f"Common dates: {len(dates)}", flush=True)

    rows = []
    for date_str in dates:
        try:
            npz = np.load(PRED_DIR / f"oot_{date_str}.npz", allow_pickle=True)
            n = len(npz["pred_log_ret_1s"])
            ts = reconstruct_ts(date_str, n)

            sal = pd.read_parquet(SAL_DIR / f"{date_str}_salience.parquet",
                                  columns=["ts_ns", "sweep_tag", "large_print_tag"])
            sweep_ts = sal.loc[sal["sweep_tag"], "ts_ns"].values.astype(np.int64)
            large_ts = sal.loc[sal["large_print_tag"], "ts_ns"].values.astype(np.int64)
            sweep_ts.sort()
            large_ts.sort()

            recent_sweep = has_recent_tag(ts, sweep_ts)
            recent_large = has_recent_tag(ts, large_ts)

            n_sweep = int(recent_sweep.sum())
            n_large = int(recent_large.sum())
            print(f"{date_str}: n={n}, recent_sweep={n_sweep} ({n_sweep/n:.1%}), "
                  f"recent_large={n_large} ({n_large/n:.1%})", flush=True)

            for h in HORIZONS:
                pred = npz[f"pred_log_ret_{h}"].astype(np.float64)
                tgt = npz[f"target_log_ret_{h}"].astype(np.float64)
                mask = npz[f"mask_log_ret_{h}"].astype(bool)
                # apply mask -> NaN
                pred_m = np.where(mask, pred, np.nan)
                tgt_m = np.where(mask, tgt, np.nan)
                abs_tgt = np.abs(tgt_m)

                for cond_name, cond in (
                    ("recent_sweep_true", recent_sweep),
                    ("recent_sweep_false", ~recent_sweep),
                    ("recent_large_print_true", recent_large),
                    ("recent_large_print_false", ~recent_large),
                ):
                    sel = cond & mask
                    n_sel = int(sel.sum())
                    if n_sel < 30:
                        ic = float("nan")
                        mfe = float("nan")
                    else:
                        ic = safe_spearman(pred_m[sel], tgt_m[sel])
                        mfe = float(np.nanmean(abs_tgt[sel]))
                    rows.append({
                        "date": date_str,
                        "horizon": h,
                        "subset": cond_name,
                        "n": n_sel,
                        "ic": ic,
                        "mean_abs_target": mfe,
                    })
        except Exception as e:
            print(f"{date_str}: ERROR {e}", flush=True)
            continue

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {len(df)} rows -> {OUT_CSV}", flush=True)

    # Aggregate: weighted IC by n across days
    print("\n=== AGGREGATE (weighted by sample count) ===", flush=True)
    agg_rows = []
    for h in HORIZONS:
        for subset in ("recent_sweep_true", "recent_sweep_false",
                       "recent_large_print_true", "recent_large_print_false"):
            sub = df[(df["horizon"] == h) & (df["subset"] == subset) &
                     df["ic"].notna()]
            if len(sub) == 0:
                continue
            w = sub["n"].values.astype(float)
            ic_w = float(np.sum(sub["ic"].values * w) / w.sum())
            mfe_w = float(np.sum(sub["mean_abs_target"].values * w) / w.sum())
            n_total = int(w.sum())
            print(f"  {h:>3s}  {subset:28s}  n={n_total:>9,}  IC={ic_w:+.4f}  meanAbsTgt={mfe_w:.4f}",
                  flush=True)
            agg_rows.append({"horizon": h, "subset": subset, "n_total": n_total,
                             "ic_weighted": ic_w, "mean_abs_target_weighted": mfe_w})
    agg_df = pd.DataFrame(agg_rows)
    agg_path = OUT_CSV.parent / "conditional_ic_aggregate.csv"
    agg_df.to_csv(agg_path, index=False)
    print(f"\nWrote aggregate -> {agg_path}", flush=True)

    # Deltas
    print("\n=== DELTAS (true - false) ===", flush=True)
    for h in HORIZONS:
        for tag in ("recent_sweep", "recent_large_print"):
            row_t = agg_df[(agg_df["horizon"] == h) & (agg_df["subset"] == f"{tag}_true")]
            row_f = agg_df[(agg_df["horizon"] == h) & (agg_df["subset"] == f"{tag}_false")]
            if len(row_t) == 0 or len(row_f) == 0:
                continue
            d_ic = float(row_t["ic_weighted"].values[0] - row_f["ic_weighted"].values[0])
            d_mfe = float(row_t["mean_abs_target_weighted"].values[0] - row_f["mean_abs_target_weighted"].values[0])
            print(f"  {h:>3s}  {tag:>20s}  dIC={d_ic:+.4f}  dMeanAbsTgt={d_mfe:+.4f}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
