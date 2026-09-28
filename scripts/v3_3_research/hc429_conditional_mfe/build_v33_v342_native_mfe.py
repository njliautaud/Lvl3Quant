"""
HC #429 — Build confidence-conditional MFE/MAE matrices for v3.3 and v3.4.2.

Mirrors EXACTLY the schema and methodology of
  output/hc417_v2_native_mfe_matrix.csv
  (produced by scripts/v3_3_research/hc417_v2_falsification/build_v2_native_mfe.py)

Re-uses compute_cell + CONF_TIERS + COMMISSION from hc411_regime_agnostic.py
unchanged. Generates v3.3, v3.4.2, and a combined (v2 + v3.3 + v3.4.2) stacked CSV.

Per the HC #429 directive:
  - For each (model, horizon ∈ {1s, 5s, 10s, 30s}, side ∈ {long, short},
    conf_band ∈ {top05, top1, top5, top10}):
      rank by signed prediction, take top-k by |pred| where sign matches side,
      report mean MFE, mean MAE, net = MFE − MAE − 0.376, WR%, n.
  - Realized MFE/MAE come from the same NPZ keys used by hc411:
      target_log_ret_<h>  → favorable-direction realized
      proxy MAE for 1s/5s/10s (negative-only portion of signed realized)
      true MAE for 30s    → target_pred_mae_30s_ticks
  - This is identical to what hc411 + the v2-native builder use, so the new
    matrices are apples-to-apples comparable with hc417_v2_native_mfe_matrix.csv.

Inputs:
  - v3.3:   output/cnn_mamba_v3_3_uncertainty_weighted/fold_00_predictions.npz
            (5-day OOT, referenced by canonical_avg_move_v3_3.json)
  - v3.4.2: output/cnn_mamba_v3_4_2_fixedmtl/fold_00_ep1_oot.npz
            (5-day OOT, 241351 samples)

Outputs:
  - output/hc429_v33_native_mfe_matrix.csv
  - output/hc429_v342_native_mfe_matrix.csv
  - output/hc429_native_mfe_matrix_combined.csv     (v2 + v3.3 + v3.4.2 stacked)
  - output/hc429_gaps.md                            (any horizons skipped + why)

CPU-only. Read-only on NPZs. Does NOT modify hc411 or hc417 scripts.
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

from hc411_regime_agnostic import (
    SIDES,
    CONF_TIERS,
    COMMISSION,
    compute_cell,
)

V33_NPZ = PROJ / "output" / "cnn_mamba_v3_3_uncertainty_weighted" / "fold_00_predictions.npz"
V342_NPZ = PROJ / "output" / "cnn_mamba_v3_4_2_fixedmtl" / "fold_00_ep1_oot.npz"
V2_EXISTING_CSV = PROJ / "output" / "hc417_v2_native_mfe_matrix.csv"

OUT_V33 = PROJ / "output" / "hc429_v33_native_mfe_matrix.csv"
OUT_V342 = PROJ / "output" / "hc429_v342_native_mfe_matrix.csv"
OUT_COMBINED = PROJ / "output" / "hc429_native_mfe_matrix_combined.csv"
OUT_GAPS = PROJ / "output" / "hc429_gaps.md"

HORIZONS_ALL = ["1s", "5s", "10s", "30s"]

# Track gaps for the gaps.md report
GAPS = []


def _log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def load_npz_bundle(npz_path: Path, model_label: str):
    """
    Returns:
      bundle:    {h: (pred, target)} with masked entries NaN'd
      mae_30s:   target_pred_mae_30s_ticks (np.float32) or None
      date_idx:  np.int32 of date indices per sample
      n_dates:   int
      horizons_present: list[str]
    """
    _log(f"{model_label}: loading {npz_path}")
    d = np.load(npz_path, allow_pickle=False)
    files = set(d.files)

    horizons_present = []
    bundle = {}
    for h in HORIZONS_ALL:
        pk = f"pred_log_ret_{h}"
        tk = f"target_log_ret_{h}"
        mk = f"mask_log_ret_{h}"
        if pk not in files or tk not in files:
            GAPS.append(f"- **{model_label} / {h}**: missing `{pk}` or `{tk}` — horizon SKIPPED.")
            continue
        pred = d[pk].astype(np.float32).copy()
        tgt = d[tk].astype(np.float32).copy()
        if mk in files:
            mask = d[mk].astype(bool)
            pred[~mask] = np.nan
            tgt[~mask] = np.nan
        else:
            GAPS.append(f"- **{model_label} / {h}**: no mask key `{mk}` — kept all samples, isfinite gating only.")
        bundle[h] = (pred, tgt)
        horizons_present.append(h)

    # 30s true MAE
    mae_30s = None
    if "target_pred_mae_30s_ticks" in files:
        mae_30s = d["target_pred_mae_30s_ticks"].astype(np.float32).copy()
        if "mask_pred_mae_30s_ticks" in files:
            mmask = d["mask_pred_mae_30s_ticks"].astype(bool)
            mae_30s = np.where(mmask, mae_30s, np.nan).astype(np.float32)
    else:
        GAPS.append(f"- **{model_label} / 30s MAE**: no `target_pred_mae_30s_ticks` — 30s MAE will use the negative-only proxy (lower bound on true intra-window adverse).")

    # date_idx: prefer oot_dates partition; otherwise treat as single day
    n_samples = bundle[horizons_present[0]][0].shape[0]
    if "oot_dates" in files:
        oot_dates = [str(x) for x in d["oot_dates"]]
        n_dates = len(oot_dates)
        if n_dates > 0:
            samples_per_day = n_samples // n_dates
            if samples_per_day == 0:
                date_idx = np.zeros(n_samples, dtype=np.int32)
                n_dates = 1
                GAPS.append(f"- **{model_label}**: oot_dates={n_dates} > n_samples={n_samples}; collapsed to single-day index.")
            else:
                date_idx = np.minimum(np.arange(n_samples) // samples_per_day, n_dates - 1).astype(np.int32)
        else:
            date_idx = np.zeros(n_samples, dtype=np.int32)
            n_dates = 1
    else:
        # No oot_dates → fall back to a single-day index. compute_cell will still
        # produce correct aggregate metrics; only day_conc is affected and we
        # don't gate on day_conc here (this is the matrix builder, not the
        # promotion gate).
        date_idx = np.zeros(n_samples, dtype=np.int32)
        n_dates = 1
        GAPS.append(f"- **{model_label}**: NPZ has no `oot_dates` field — date_idx collapsed to single-day. day_conc in this matrix is non-meaningful; aggregate MFE/MAE/net/WR/n are unaffected.")

    _log(f"{model_label}: n={n_samples}, n_dates={n_dates}, horizons_present={horizons_present}")
    return bundle, mae_30s, date_idx, n_dates, horizons_present


def aggregate_full_oot(bundle, mae_30s_arr, date_idx, n_dates, horizons_present,
                        model_label: str):
    """Build long-form (model, horizon, side, conf_tier) rows."""
    rows = []
    for h in horizons_present:
        pred, target = bundle[h]
        for side in SIDES:
            sign = 1.0 if side == "long" else -1.0
            for tier_name, tier_frac in CONF_TIERS:
                m = compute_cell(
                    pred, target, mae_30s_arr, sign, tier_frac,
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
    """Pivot long → wide schema matching hc417_v2_native_mfe_matrix.csv."""
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

    # ---- v3.3 ----
    v33_bundle, v33_mae, v33_di, v33_nd, v33_hp = load_npz_bundle(V33_NPZ, "v3.3")
    v33_agg = aggregate_full_oot(v33_bundle, v33_mae, v33_di, v33_nd, v33_hp, "v3.3")
    v33_wide = to_wide(v33_agg)
    OUT_V33.parent.mkdir(parents=True, exist_ok=True)
    v33_wide.to_csv(OUT_V33, index=False)
    _log(f"v3.3: wrote {OUT_V33} ({len(v33_wide)} rows)")

    # ---- v3.4.2 ----
    v342_bundle, v342_mae, v342_di, v342_nd, v342_hp = load_npz_bundle(V342_NPZ, "v3.4.2")
    v342_agg = aggregate_full_oot(v342_bundle, v342_mae, v342_di, v342_nd, v342_hp, "v3.4.2")
    v342_wide = to_wide(v342_agg)
    v342_wide.to_csv(OUT_V342, index=False)
    _log(f"v3.4.2: wrote {OUT_V342} ({len(v342_wide)} rows)")

    # ---- combined (v2 + v3.3 + v3.4.2 stacked) ----
    if V2_EXISTING_CSV.exists():
        v2_wide = pd.read_csv(V2_EXISTING_CSV)
    else:
        GAPS.append(f"- **combined**: v2 baseline CSV `{V2_EXISTING_CSV}` not found — combined matrix excludes v2.")
        v2_wide = pd.DataFrame()

    combined = pd.concat(
        [df for df in [v2_wide, v33_wide, v342_wide] if len(df) > 0],
        ignore_index=True, sort=False,
    )
    # Reorder columns: model, horizon, side, then tier blocks in canonical order
    cols = ["model", "horizon", "side"]
    for tier_name, _ in CONF_TIERS:
        short = tier_name.lower().replace(".", "")
        for prefix in ["mfe", "mae", "net", "wr", "n"]:
            c = f"{prefix}_{short}"
            if c in combined.columns:
                cols.append(c)
    combined = combined[[c for c in cols if c in combined.columns]]
    combined.to_csv(OUT_COMBINED, index=False)
    _log(f"combined: wrote {OUT_COMBINED} ({len(combined)} rows)")

    # ---- gaps.md ----
    lines = []
    lines.append("# HC #429 — conditional MFE matrix gaps\n")
    lines.append(f"Generated: {datetime.now().isoformat()}")
    lines.append("")
    lines.append(f"- v2 baseline: `{V2_EXISTING_CSV}`")
    lines.append(f"- v3.3 NPZ: `{V33_NPZ}`")
    lines.append(f"- v3.4.2 NPZ: `{V342_NPZ}`")
    lines.append(f"- Cost (passive at touch + commission): {COMMISSION} tk")
    lines.append("")
    lines.append("## Gaps / caveats")
    lines.append("")
    if not GAPS:
        lines.append("None — all 4 horizons populated for v3.3 and v3.4.2, true 30s MAE used, proxy used for 1s/5s/10s (same as v2 native baseline).")
    else:
        for g in GAPS:
            lines.append(g)
    lines.append("")
    lines.append("## MAE methodology")
    lines.append("")
    lines.append("- **30s MAE**: true intra-horizon adverse from `target_pred_mae_30s_ticks` (where available).")
    lines.append("- **1s/5s/10s MAE**: PROXY = mean magnitude of negative-only signed realized horizon-end moves. This is a LOWER BOUND on true intra-window adverse (true MAE is at least this big). Identical methodology to `output/hc417_v2_native_mfe_matrix.csv` so the matrices are directly comparable.")
    OUT_GAPS.write_text("\n".join(lines))
    _log(f"wrote {OUT_GAPS}")

    # ---- print headline cells for stdout report ----
    _log("=== headline cells: top0.5%, mfe / mae / net (ticks) ===")
    for src_name, w in [("v3.3", v33_wide), ("v3.4.2", v342_wide)]:
        for _, r in w.iterrows():
            print(f"  {src_name:8s} {r['horizon']:>3s} {r['side']:5s}  "
                  f"mfe_top05={r.get('mfe_top05', float('nan')):>7.3f}  "
                  f"mae_top05={r.get('mae_top05', float('nan')):>7.3f}  "
                  f"net_top05={r.get('net_top05', float('nan')):>7.3f}  "
                  f"wr_top05={r.get('wr_top05', float('nan')):>5.1f}  "
                  f"n_top05={int(r.get('n_top05', 0))}",
                  flush=True)

    _log(f"DONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
