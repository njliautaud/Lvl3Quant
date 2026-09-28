"""
HC #417 falsification — Phase 1: build v2-native MFE-at-confidence matrix.

Reuses the EXACT aggregation logic from
scripts/v3_3_research/hc411_regime_agnostic.py (compute_cell + aggregate_full_oot
+ the wide-format builder). Does NOT modify hc411_regime_agnostic.py.

Input:  /home/jupiter/Lvl3Quant/output/hc417_v2_full_oot_56d.npz   (CNN-Mamba v2 full OOT)
Output: /home/jupiter/Lvl3Quant/output/hc417_v2_native_mfe_matrix.csv

Schema matches hc411 mfe_at_confidence_matrix.csv exactly:
  model, horizon, side,
  mfe_top05, mae_top05, net_top05, wr_top05, n_top05,
  mfe_top1,  mae_top1,  net_top1,  wr_top1,  n_top1,
  mfe_top5,  mae_top5,  net_top5,  wr_top5,  n_top5,
  mfe_top10, mae_top10, net_top10, wr_top10, n_top10

model column = 'v2'.

NPZ note:
  - day_index indexes into oot_dates (56-len), but only `dates_present` (36) have
    predictions. The mfe aggregation is over the full 56d sample-space; the
    confidence ranking happens over all valid samples.
  - HC #411 does the same global ranking (no per-date conf threshold), so this
    is consistent.

MAE proxy (1s/5s/10s — no intra-horizon MAE available in v2 NPZ):
  Same proxy as HC #411 for 1s/5s/10s: mean of |negative-only signed realized move|.
  v2 NPZ has no `target_pred_mae_30s_ticks`, so 30s is skipped (no preds exist
  for 30s anyway — pred heads only at 1s/5s/10s).
"""
from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(PROJ / "scripts" / "v3_3_research"))

# Re-import the canonical aggregation functions verbatim (no modification).
from hc411_regime_agnostic import (
    HORIZONS as HC411_HORIZONS,
    SIDES,
    CONF_TIERS,
    COMMISSION,
    compute_cell,
)

V2_NPZ = PROJ / "output" / "hc417_v2_full_oot_56d.npz"
OUT_CSV = PROJ / "output" / "hc417_v2_native_mfe_matrix.csv"

# v2 model has pred heads only at 1s/5s/10s (no 30s head).
V2_HORIZONS = ["1s", "5s", "10s"]


def _log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def load_v2(npz_path: Path):
    _log(f"loading {npz_path}")
    d = np.load(npz_path, allow_pickle=False)
    bundle = {}
    for h in V2_HORIZONS:
        pk = f"pred_log_ret_{h}"
        tk = f"target_log_ret_{h}"
        mk = f"mask_log_ret_{h}"
        pred = d[pk].astype(np.float32).copy()
        tgt = d[tk].astype(np.float32).copy()
        mask = d[mk].astype(bool)
        # NaN-out where mask is false so isfinite gating in compute_cell removes them.
        pred[~mask] = np.nan
        tgt[~mask] = np.nan
        bundle[h] = (pred, tgt)
    n = bundle["1s"][0].shape[0]
    day_index = d["day_index"].astype(np.int32)
    oot_dates = [str(x) for x in d["oot_dates"]]
    n_dates = len(oot_dates)
    _log(f"v2: n={n}, n_dates={n_dates} (oot_dates), unique_present_in_day_index={len(np.unique(day_index))}")
    return bundle, day_index, n_dates, oot_dates


def aggregate_full_oot(bundle, date_idx, n_dates, model_label: str = "v2"):
    rows = []
    for h in V2_HORIZONS:
        pred, target = bundle[h]
        for side in SIDES:
            sign = 1.0 if side == "long" else -1.0
            for tier_name, tier_frac in CONF_TIERS:
                m = compute_cell(
                    pred, target, None, sign, tier_frac,
                    date_idx, n_dates, 0, n_dates, h,
                )
                if m is None:
                    continue
                rows.append(dict(
                    model=model_label,
                    horizon=h,
                    side=side,
                    conf_tier=tier_name,
                    aggregate_n=m["n_fills"],
                    aggregate_mfe=m["mfe_mean_tk"],
                    aggregate_mae=m["mae_mean_tk"],
                    aggregate_net=m["net_tk_per_fill"],
                    aggregate_wr=m["wr_pct"],
                ))
    return rows


def to_wide(agg_rows):
    df = pd.DataFrame(agg_rows)
    wide = []
    for (m, h, side), grp in df.groupby(["model", "horizon", "side"]):
        row = dict(model=m, horizon=h, side=side)
        for tier_name, _ in CONF_TIERS:
            sub = grp[grp["conf_tier"] == tier_name]
            if len(sub) == 0:
                continue
            r0 = sub.iloc[0]
            short = tier_name.lower().replace(".", "")
            row[f"mfe_{short}"] = r0["aggregate_mfe"]
            row[f"mae_{short}"] = r0["aggregate_mae"]
            row[f"net_{short}"] = r0["aggregate_net"]
            row[f"wr_{short}"] = r0["aggregate_wr"]
            row[f"n_{short}"] = r0["aggregate_n"]
        wide.append(row)
    cols = ["model", "horizon", "side"]
    for tier_name, _ in CONF_TIERS:
        short = tier_name.lower().replace(".", "")
        for prefix in ["mfe", "mae", "net", "wr", "n"]:
            cols.append(f"{prefix}_{short}")
    df_w = pd.DataFrame(wide)
    df_w = df_w[[c for c in cols if c in df_w.columns]]
    return df_w


def main():
    t0 = time.time()
    bundle, date_idx, n_dates, oot_dates = load_v2(V2_NPZ)
    agg = aggregate_full_oot(bundle, date_idx, n_dates, model_label="v2")
    df_wide = to_wide(agg)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_wide.to_csv(OUT_CSV, index=False)
    _log(f"wrote {OUT_CSV} ({len(df_wide)} rows)  elapsed={time.time()-t0:.1f}s")
    print(df_wide.to_string(index=False))


if __name__ == "__main__":
    main()
