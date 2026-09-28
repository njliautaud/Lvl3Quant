"""
HC #411 — Regime-agnostic signal research.

Stratifies v3.3 and v3.4.2 OOT predictions across:
  model x horizon x side x conf_tier x sub_window
Then aggregates regime stability across the 4 sub-windows (contiguous date chunks).

A cell is "regime_stable" iff it passes the HC #408 honesty gate
  (n_fills >= 50, day_conc <= 0.20, ci_low_95(net) > 0)
in ALL 4 sub-windows.

A (model, horizon, conf_tier) is "bidirectional_regime_stable" iff BOTH long
AND short sides are regime_stable.

MAE caveat: NPZs only store target_pred_mae_30s_ticks (30s adverse-move).
For h != 30s we use a proxy: mean of clip(target_log_ret_h * side_sign, max=0)
i.e. the negative-only portion of the favorable signed realized move. Documented
in verdict.md.

CPU-only Jupiter. Read-only on source NPZs. Does NOT modify HC #408/410 scripts.
"""
from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path("/home/jupiter/Lvl3Quant")
V33_NPZ = PROJ / "output" / "v3_3_extended_oot_20260514" / "extended_oot_predictions.npz"
V342_5D = PROJ / "data" / "v342_oot_npz" / "v342_5d.npz"
V342_11D = PROJ / "data" / "v342_oot_npz" / "v342_11d_ext.npz"

HORIZONS = ["1s", "5s", "10s", "30s"]
SIDES = ["long", "short"]
CONF_TIERS = [
    ("Top0.5", 0.005),
    ("Top1",   0.01),
    ("Top5",   0.05),
    ("Top10",  0.10),
]
N_WINDOWS = 4

COMMISSION = 0.376
HONESTY_N_FILLS_MIN = 50
HONESTY_DAY_CONC_MAX = 0.20

TS = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = PROJ / "output" / f"hc411_regime_agnostic_{TS}"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def mean_ci_lower(arr, z=1.96):
    n = len(arr)
    if n < 2:
        return float("nan")
    se = arr.std(ddof=1) / np.sqrt(n)
    return float(arr.mean() - z * se)


def split_dates_into_windows(n_dates: int, n_windows: int = N_WINDOWS):
    """Split n_dates into n_windows contiguous chunks. Returns list of (date_lo, date_hi_exclusive)."""
    base = n_dates // n_windows
    rem = n_dates % n_windows
    sizes = [base + (1 if i < rem else 0) for i in range(n_windows)]
    bounds = []
    cursor = 0
    for s in sizes:
        bounds.append((cursor, cursor + s))
        cursor += s
    return bounds


def load_v33():
    _log(f"v3.3: loading {V33_NPZ}")
    d = np.load(V33_NPZ, allow_pickle=False)
    n = d["pred_log_ret_1s"].shape[0]
    oot_dates = list(d["oot_dates"])
    n_dates = len(oot_dates)
    samples_per_day = n // n_dates
    date_idx = np.minimum(np.arange(n) // samples_per_day, n_dates - 1).astype(np.int32)
    bundle = {}
    for h in HORIZONS:
        bundle[h] = (d[f"pred_log_ret_{h}"].astype(np.float32),
                     d[f"target_log_ret_{h}"].astype(np.float32))
    mae_30s = d["target_pred_mae_30s_ticks"].astype(np.float32) if "target_pred_mae_30s_ticks" in d.files else None
    _log(f"v3.3: n={n}, n_dates={n_dates}")
    return bundle, mae_30s, date_idx, n_dates, oot_dates


def load_v342():
    _log("v3.4.2: loading 5d + 11d_ext")
    d1 = np.load(V342_5D, allow_pickle=False)
    d2 = np.load(V342_11D, allow_pickle=False)
    bundle = {}
    for h in HORIZONS:
        p = np.concatenate([d1[f"pred_log_ret_{h}"], d2[f"pred_log_ret_{h}"]]).astype(np.float32)
        t = np.concatenate([d1[f"target_log_ret_{h}"], d2[f"target_log_ret_{h}"]]).astype(np.float32)
        bundle[h] = (p, t)
    mae_30s = None
    if "target_pred_mae_30s_ticks" in d1.files and "target_pred_mae_30s_ticks" in d2.files:
        mae_30s = np.concatenate([d1["target_pred_mae_30s_ticks"],
                                  d2["target_pred_mae_30s_ticks"]]).astype(np.float32)
    n1 = d1["pred_log_ret_1s"].shape[0]
    n2 = d2["pred_log_ret_1s"].shape[0]
    dates_5 = ["20260223", "20260224", "20260225", "20260226", "20260227"]
    dates_11 = ["20260301", "20260302", "20260303", "20260304", "20260305",
                "20260308", "20260309", "20260310", "20260311", "20260312", "20260315"]
    oot_dates = dates_5 + dates_11
    n_dates = len(oot_dates)
    sub1 = n1 // 5
    sub2 = n2 // 11
    di1 = np.minimum(np.arange(n1) // sub1, 4).astype(np.int32)
    di2 = np.minimum(np.arange(n2) // sub2, 10).astype(np.int32) + 5
    date_idx = np.concatenate([di1, di2])
    _log(f"v3.4.2: n={n1 + n2} (5d={n1}, 11d={n2}), n_dates={n_dates}")
    return bundle, mae_30s, date_idx, n_dates, oot_dates


def compute_cell(pred, target, mae_30s_arr, side_sign, tier_frac,
                  date_idx, n_dates, window_date_lo, window_date_hi, horizon):
    """
    Compute metrics for one (side, tier, sub_window) cell.
    Confidence ranking is done WITHIN the sub-window (top-k of in-window samples).
    """
    in_win = (date_idx >= window_date_lo) & (date_idx < window_date_hi)
    ok = in_win & np.isfinite(pred) & np.isfinite(target)
    if not ok.any():
        return None
    ok_idx = np.where(ok)[0]
    signed_pred = side_sign * pred[ok_idx]
    n_ok = len(ok_idx)
    k = max(1, int(n_ok * tier_frac))
    if k >= n_ok:
        sel = ok_idx
    else:
        part = np.argpartition(-signed_pred, k - 1)[:k]
        sel = ok_idx[part]
    n_cell = len(sel)
    if n_cell == 0:
        return None

    realized = side_sign * target[sel]
    net = realized - COMMISSION
    mean_mfe = float(realized.mean())
    mean_net = float(net.mean())
    wr = float((realized > 0).mean())
    ci_low = mean_ci_lower(net)

    # day_conc within sub-window: use date indices relative to window
    rel_dates = date_idx[sel] - window_date_lo
    win_n_dates = max(1, window_date_hi - window_date_lo)
    counts = np.bincount(rel_dates, minlength=win_n_dates)
    day_conc = float(counts.max() / n_cell) if n_cell else 1.0

    # MAE
    if horizon == "30s" and mae_30s_arr is not None:
        mae_vals = mae_30s_arr[sel]
        mae_vals = mae_vals[np.isfinite(mae_vals)]
        mae_mean = float(mae_vals.mean()) if mae_vals.size > 0 else float("nan")
    else:
        # Proxy: mean of negative-only signed realized move
        adverse = np.clip(realized, a_min=None, a_max=0.0)
        # Report as positive magnitude
        mae_mean = float(-adverse.mean())

    promote = bool(
        n_cell >= HONESTY_N_FILLS_MIN
        and day_conc <= HONESTY_DAY_CONC_MAX
        and ci_low > 0
    )

    return dict(
        n_fills=n_cell,
        mfe_mean_tk=round(mean_mfe, 4),
        mae_mean_tk=round(mae_mean, 4),
        net_tk_per_fill=round(mean_net, 4),
        wr_pct=round(100 * wr, 2),
        day_conc=round(day_conc, 4),
        ci_low_95_net=round(ci_low, 4),
        promote_in_window=promote,
    )


def evaluate_model(model_name, bundle, mae_30s_arr, date_idx, n_dates):
    """Returns sub-window detail rows."""
    bounds = split_dates_into_windows(n_dates, N_WINDOWS)
    _log(f"{model_name}: sub-window date bounds (date_lo, date_hi_excl): {bounds}")
    rows = []
    for h in HORIZONS:
        pred, target = bundle[h]
        for side in SIDES:
            sign = 1.0 if side == "long" else -1.0
            for tier_name, tier_frac in CONF_TIERS:
                for w_idx, (lo, hi) in enumerate(bounds):
                    m = compute_cell(pred, target, mae_30s_arr, sign, tier_frac,
                                     date_idx, n_dates, lo, hi, h)
                    if m is None:
                        continue
                    rows.append(dict(
                        model=model_name,
                        horizon=h,
                        side=side,
                        conf_tier=tier_name,
                        sub_window=w_idx,
                        window_date_lo=lo,
                        window_date_hi_excl=hi,
                        **m,
                    ))
    return rows, bounds


def aggregate_full_oot(model_name, bundle, mae_30s_arr, date_idx, n_dates):
    """Compute the full-OOT aggregate metric per (model, horizon, side, conf_tier) for sanity vs HC #410."""
    rows = []
    for h in HORIZONS:
        pred, target = bundle[h]
        for side in SIDES:
            sign = 1.0 if side == "long" else -1.0
            for tier_name, tier_frac in CONF_TIERS:
                m = compute_cell(pred, target, mae_30s_arr, sign, tier_frac,
                                 date_idx, n_dates, 0, n_dates, h)
                if m is None:
                    continue
                rows.append(dict(
                    model=model_name,
                    horizon=h,
                    side=side,
                    conf_tier=tier_name,
                    aggregate_n=m["n_fills"],
                    aggregate_mfe=m["mfe_mean_tk"],
                    aggregate_mae=m["mae_mean_tk"],
                    aggregate_net=m["net_tk_per_fill"],
                    aggregate_wr=m["wr_pct"],
                    aggregate_promote=m["promote_in_window"],
                ))
    return rows


def main():
    t0 = time.time()
    sub_window_rows = []
    aggregate_rows = []
    bounds_by_model = {}

    v33_bundle, v33_mae, v33_di, v33_nd, v33_dates = load_v33()
    r, b = evaluate_model("v3.3", v33_bundle, v33_mae, v33_di, v33_nd)
    sub_window_rows += r
    bounds_by_model["v3.3"] = (b, v33_dates)
    aggregate_rows += aggregate_full_oot("v3.3", v33_bundle, v33_mae, v33_di, v33_nd)

    v342_bundle, v342_mae, v342_di, v342_nd, v342_dates = load_v342()
    r, b = evaluate_model("v3.4.2", v342_bundle, v342_mae, v342_di, v342_nd)
    sub_window_rows += r
    bounds_by_model["v3.4.2"] = (b, v342_dates)
    aggregate_rows += aggregate_full_oot("v3.4.2", v342_bundle, v342_mae, v342_di, v342_nd)

    df_sub = pd.DataFrame(sub_window_rows)
    df_agg = pd.DataFrame(aggregate_rows)

    # Save raw sub-window detail (for reference / debugging)
    sub_path = OUT_DIR / "sub_window_detail.csv"
    df_sub.to_csv(sub_path, index=False)
    _log(f"sub-window detail: {sub_path} ({len(df_sub)} rows)")

    # ====================================================================
    # OUTPUT 1: mfe_at_confidence_matrix.csv — wide table
    #   rows: (model, horizon, side)
    #   columns: mfe/mae/net/wr/n at each of 4 conf tiers (from full-OOT aggregate)
    # ====================================================================
    wide_rows = []
    for (m, h, side), grp in df_agg.groupby(["model", "horizon", "side"]):
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
        wide_rows.append(row)
    df_wide = pd.DataFrame(wide_rows)
    wide_cols = ["model", "horizon", "side"]
    for tier_name, _ in CONF_TIERS:
        short = tier_name.lower().replace(".", "")
        for prefix in ["mfe", "mae", "net", "wr", "n"]:
            c = f"{prefix}_{short}"
            if c in df_wide.columns:
                wide_cols.append(c)
    df_wide = df_wide[[c for c in wide_cols if c in df_wide.columns]]
    wide_path = OUT_DIR / "mfe_at_confidence_matrix.csv"
    df_wide.to_csv(wide_path, index=False)
    _log(f"mfe_at_confidence_matrix: {wide_path} ({len(df_wide)} rows)")

    # ====================================================================
    # OUTPUT 2: regime_stability_matrix.csv
    #   rows: (model, horizon, side, conf_tier)
    #   cols: window_0_promote ... window_3_promote, n_windows_promoted,
    #         regime_stable, min_net, max_net, std_net, aggregate_net
    # ====================================================================
    stability_rows = []
    keys = df_sub.groupby(["model", "horizon", "side", "conf_tier"]).groups.keys()
    for key in keys:
        m, h, side, tier = key
        sub = df_sub[(df_sub["model"] == m) & (df_sub["horizon"] == h)
                     & (df_sub["side"] == side) & (df_sub["conf_tier"] == tier)]
        win_promote = {w: False for w in range(N_WINDOWS)}
        win_net = {w: float("nan") for w in range(N_WINDOWS)}
        for _, r in sub.iterrows():
            w = int(r["sub_window"])
            win_promote[w] = bool(r["promote_in_window"])
            win_net[w] = float(r["net_tk_per_fill"])
        n_promoted = sum(1 for w in range(N_WINDOWS) if win_promote[w])
        regime_stable = (n_promoted == N_WINDOWS)
        nets = np.array([win_net[w] for w in range(N_WINDOWS)], dtype=np.float64)
        nets_finite = nets[np.isfinite(nets)]
        min_net = float(nets_finite.min()) if nets_finite.size else float("nan")
        max_net = float(nets_finite.max()) if nets_finite.size else float("nan")
        std_net = float(nets_finite.std(ddof=1)) if nets_finite.size >= 2 else float("nan")
        # Aggregate-OOT net for cross-check
        agg = df_agg[(df_agg["model"] == m) & (df_agg["horizon"] == h)
                     & (df_agg["side"] == side) & (df_agg["conf_tier"] == tier)]
        aggregate_net = float(agg.iloc[0]["aggregate_net"]) if len(agg) else float("nan")
        aggregate_mfe = float(agg.iloc[0]["aggregate_mfe"]) if len(agg) else float("nan")
        aggregate_mae = float(agg.iloc[0]["aggregate_mae"]) if len(agg) else float("nan")
        stability_rows.append(dict(
            model=m,
            horizon=h,
            side=side,
            conf_tier=tier,
            window_0_promote=win_promote[0],
            window_1_promote=win_promote[1],
            window_2_promote=win_promote[2],
            window_3_promote=win_promote[3],
            n_windows_promoted=n_promoted,
            regime_stable=regime_stable,
            window_0_net=round(win_net[0], 4) if np.isfinite(win_net[0]) else float("nan"),
            window_1_net=round(win_net[1], 4) if np.isfinite(win_net[1]) else float("nan"),
            window_2_net=round(win_net[2], 4) if np.isfinite(win_net[2]) else float("nan"),
            window_3_net=round(win_net[3], 4) if np.isfinite(win_net[3]) else float("nan"),
            min_net=round(min_net, 4),
            max_net=round(max_net, 4),
            std_net=round(std_net, 4) if np.isfinite(std_net) else float("nan"),
            aggregate_net=round(aggregate_net, 4) if np.isfinite(aggregate_net) else float("nan"),
            aggregate_mfe=round(aggregate_mfe, 4) if np.isfinite(aggregate_mfe) else float("nan"),
            aggregate_mae=round(aggregate_mae, 4) if np.isfinite(aggregate_mae) else float("nan"),
        ))
    df_stab = pd.DataFrame(stability_rows)
    stab_path = OUT_DIR / "regime_stability_matrix.csv"
    df_stab.to_csv(stab_path, index=False)
    _log(f"regime_stability_matrix: {stab_path} ({len(df_stab)} rows)")

    # ====================================================================
    # OUTPUT 3: bidirectional_cells.csv
    # ====================================================================
    bidir_rows = []
    for (m, h, tier), grp in df_stab.groupby(["model", "horizon", "conf_tier"]):
        long_row = grp[grp["side"] == "long"]
        short_row = grp[grp["side"] == "short"]
        if len(long_row) == 0 or len(short_row) == 0:
            continue
        L = long_row.iloc[0]
        S = short_row.iloc[0]
        long_stable = bool(L["regime_stable"])
        short_stable = bool(S["regime_stable"])
        bidir_stable = long_stable and short_stable
        agg_long = df_agg[(df_agg["model"] == m) & (df_agg["horizon"] == h)
                          & (df_agg["side"] == "long") & (df_agg["conf_tier"] == tier)]
        agg_short = df_agg[(df_agg["model"] == m) & (df_agg["horizon"] == h)
                           & (df_agg["side"] == "short") & (df_agg["conf_tier"] == tier)]
        long_mfe = float(agg_long.iloc[0]["aggregate_mfe"]) if len(agg_long) else float("nan")
        short_mfe = float(agg_short.iloc[0]["aggregate_mfe"]) if len(agg_short) else float("nan")
        long_net = float(L["aggregate_net"])
        short_net = float(S["aggregate_net"])
        weaker_net = min(long_net, short_net) if np.isfinite(long_net) and np.isfinite(short_net) else float("nan")
        bidir_rows.append(dict(
            model=m,
            horizon=h,
            conf_tier=tier,
            long_regime_stable=long_stable,
            short_regime_stable=short_stable,
            bidirectional_regime_stable=bidir_stable,
            bidirectional_mfe_long_mean=round(long_mfe, 4),
            bidirectional_mfe_short_mean=round(short_mfe, 4),
            long_aggregate_net=round(long_net, 4),
            short_aggregate_net=round(short_net, 4),
            weaker_side_net=round(weaker_net, 4) if np.isfinite(weaker_net) else float("nan"),
            long_min_net=L["min_net"],
            short_min_net=S["min_net"],
        ))
    df_bidir = pd.DataFrame(bidir_rows)
    # Sort: bidirectional first, then by weaker_side_net desc
    df_bidir["__sort_key1"] = (~df_bidir["bidirectional_regime_stable"]).astype(int)
    df_bidir["__sort_key2"] = -df_bidir["weaker_side_net"].fillna(-1e9)
    df_bidir = df_bidir.sort_values(["__sort_key1", "__sort_key2"]).drop(columns=["__sort_key1", "__sort_key2"])
    bidir_path = OUT_DIR / "bidirectional_cells.csv"
    df_bidir.to_csv(bidir_path, index=False)
    _log(f"bidirectional_cells: {bidir_path} ({len(df_bidir)} rows)")

    # ====================================================================
    # OUTPUT 4: verdict.md
    # ====================================================================
    n_bidir_per_model = df_bidir[df_bidir["bidirectional_regime_stable"]].groupby("model").size().to_dict()
    n_unidir_per_model = df_stab[df_stab["regime_stable"]].groupby(["model", "side"]).size().to_dict()

    # Top regime-agnostic winners: prefer bidirectional cells, fallback to unidirectional
    bidir_winners = df_bidir[df_bidir["bidirectional_regime_stable"]].copy()
    unidir_winners = df_stab[df_stab["regime_stable"]].copy()

    lines = []
    lines.append("# HC #411 — Regime-agnostic signal verdict\n")
    lines.append(f"Generated: {datetime.now().isoformat()}")
    lines.append(f"Output dir: `{OUT_DIR}`\n")
    lines.append(f"Models analyzed: v3.3 (15d OOT), v3.4.2 (16d OOT)")
    lines.append(f"Sub-windows per model: {N_WINDOWS} contiguous date chunks")
    lines.append(f"Honesty gate per window: n_fills>=50 AND day_conc<=0.20 AND CI_low_95(net)>0")
    lines.append(f"Cost: net = realized - {COMMISSION} ticks (HC #405 commission-only)\n")

    # Sub-window bounds documentation
    lines.append("## Sub-window date bounds")
    for m_name, (bnds, dates) in bounds_by_model.items():
        lines.append(f"### {m_name}")
        for wi, (lo, hi) in enumerate(bnds):
            d_lo = dates[lo] if lo < len(dates) else "?"
            d_hi = dates[hi - 1] if (hi - 1) < len(dates) else "?"
            lines.append(f"- window {wi}: dates [{d_lo} ... {d_hi}] (indices {lo}..{hi-1}, n_dates={hi-lo})")
        lines.append("")

    # MFE/MAE table at confidence
    lines.append("## MFE / MAE table at confidence (units: ticks, full-OOT aggregate)")
    lines.append("")
    lines.append("MFE = mean favorable-direction realized horizon-end move (signed by side).")
    lines.append("MAE = at 30s: mean of `target_pred_mae_30s_ticks` (true realized adverse).")
    lines.append("       at 1s/5s/10s: proxy = mean magnitude of negative-only realized moves (no intra-horizon MAE available in NPZ).")
    lines.append("")
    for m_name in ["v3.3", "v3.4.2"]:
        lines.append(f"### {m_name}")
        lines.append("")
        lines.append("| horizon | side | Top0.5 MFE | MAE | net | Top1 MFE | MAE | net | Top5 MFE | MAE | net | Top10 MFE | MAE | net |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for h in HORIZONS:
            for side in SIDES:
                sub = df_wide[(df_wide["model"] == m_name) & (df_wide["horizon"] == h) & (df_wide["side"] == side)]
                if len(sub) == 0:
                    continue
                r = sub.iloc[0]
                def cell(p):
                    v = r.get(p, float("nan"))
                    if pd.isna(v):
                        return "n/a"
                    return f"{v:.3f}"
                row = (f"| {h} | {side} | "
                       f"{cell('mfe_top05')} | {cell('mae_top05')} | {cell('net_top05')} | "
                       f"{cell('mfe_top1')} | {cell('mae_top1')} | {cell('net_top1')} | "
                       f"{cell('mfe_top5')} | {cell('mae_top5')} | {cell('net_top5')} | "
                       f"{cell('mfe_top10')} | {cell('mae_top10')} | {cell('net_top10')} |")
                lines.append(row)
        lines.append("")

    # Regime-agnostic winners
    lines.append("## Regime-agnostic winners")
    lines.append("")
    lines.append(f"Bidirectional regime-stable cells (BOTH long AND short pass all 4 sub-windows):")
    for m_name in ["v3.3", "v3.4.2"]:
        c = int(n_bidir_per_model.get(m_name, 0))
        lines.append(f"- **{m_name}: {c} bidirectional regime-stable cells**")
    lines.append("")
    lines.append(f"Unidirectional regime-stable cells (one side passes all 4 sub-windows):")
    for m_name in ["v3.3", "v3.4.2"]:
        for side in SIDES:
            c = int(n_unidir_per_model.get((m_name, side), 0))
            lines.append(f"- {m_name} / {side}: {c} cells")
    lines.append("")

    if len(bidir_winners) > 0:
        lines.append("### Bidirectional regime-stable cells (the gold standard)")
        lines.append("")
        lines.append("| model | horizon | conf_tier | long_net | short_net | long_min_net | short_min_net | long_MFE | short_MFE |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for _, r in bidir_winners.iterrows():
            lines.append(f"| {r['model']} | {r['horizon']} | {r['conf_tier']} | "
                         f"{r['long_aggregate_net']:.3f} | {r['short_aggregate_net']:.3f} | "
                         f"{r['long_min_net']:.3f} | {r['short_min_net']:.3f} | "
                         f"{r['bidirectional_mfe_long_mean']:.3f} | {r['bidirectional_mfe_short_mean']:.3f} |")
        lines.append("")
    else:
        lines.append("### Bidirectional regime-stable cells")
        lines.append("")
        lines.append("**NO bidirectional regime-stable cell exists in either model.**")
        lines.append("")
        lines.append("No (model x horizon x conf_tier) cell has BOTH long AND short sides passing the HC #408 honesty gate in ALL 4 sub-windows simultaneously.")
        lines.append("")

    if len(unidir_winners) > 0:
        lines.append("### Unidirectional regime-stable cells (fallback — may still be deployable but regime-fragile)")
        lines.append("")
        unidir_winners = unidir_winners.copy()
        unidir_winners["__sort"] = -unidir_winners["aggregate_net"].fillna(-1e9)
        unidir_winners = unidir_winners.sort_values(["__sort"]).drop(columns=["__sort"])
        lines.append("| model | horizon | side | conf_tier | aggregate_net | min_win_net | max_win_net | std_net | aggregate_MFE | aggregate_MAE |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for _, r in unidir_winners.head(15).iterrows():
            lines.append(f"| {r['model']} | {r['horizon']} | {r['side']} | {r['conf_tier']} | "
                         f"{r['aggregate_net']:.3f} | {r['min_net']:.3f} | {r['max_net']:.3f} | "
                         f"{r['std_net']:.3f} | {r['aggregate_mfe']:.3f} | {r['aggregate_mae']:.3f} |")
        lines.append("")
    else:
        lines.append("### Unidirectional regime-stable cells")
        lines.append("")
        lines.append("**NO unidirectional regime-stable cell exists in either model.**")
        lines.append("")
        lines.append("No single (model x horizon x side x conf_tier) cell passes the HC #408 honesty gate in ALL 4 sub-windows.")
        lines.append("")

    # TP/SL recommendations
    lines.append("## TP / SL recommendations")
    lines.append("")
    lines.append("Convention: TP = mean MFE (favorable); SL = -1.5 x mean MAE (1.5x buffer over historical adverse).")
    lines.append("TP/SL units = ticks. Net cost (commission) = 0.376 tk per round trip.")
    lines.append("")
    if len(bidir_winners) > 0:
        lines.append("### Bidirectional winners — TP/SL (deploy these as long+short pairs)")
        lines.append("")
        lines.append("| model | horizon | conf_tier | long_TP | long_SL | short_TP | short_SL |")
        lines.append("|---|---|---|---|---|---|---|")
        for _, r in bidir_winners.iterrows():
            # Need MAE per side from df_agg
            m, h, tier = r["model"], r["horizon"], r["conf_tier"]
            agg_long = df_agg[(df_agg["model"] == m) & (df_agg["horizon"] == h)
                              & (df_agg["side"] == "long") & (df_agg["conf_tier"] == tier)].iloc[0]
            agg_short = df_agg[(df_agg["model"] == m) & (df_agg["horizon"] == h)
                               & (df_agg["side"] == "short") & (df_agg["conf_tier"] == tier)].iloc[0]
            long_tp = float(agg_long["aggregate_mfe"])
            long_sl = -1.5 * float(agg_long["aggregate_mae"])
            short_tp = float(agg_short["aggregate_mfe"])
            short_sl = -1.5 * float(agg_short["aggregate_mae"])
            lines.append(f"| {m} | {h} | {tier} | {long_tp:.3f} | {long_sl:.3f} | {short_tp:.3f} | {short_sl:.3f} |")
        lines.append("")
    if len(unidir_winners) > 0:
        lines.append("### Unidirectional winners — TP/SL (top 10 by aggregate_net)")
        lines.append("")
        lines.append("| model | horizon | side | conf_tier | TP (=MFE) | SL (=-1.5xMAE) | aggregate_net |")
        lines.append("|---|---|---|---|---|---|---|")
        for _, r in unidir_winners.head(10).iterrows():
            tp = float(r["aggregate_mfe"])
            sl = -1.5 * float(r["aggregate_mae"])
            lines.append(f"| {r['model']} | {r['horizon']} | {r['side']} | {r['conf_tier']} | "
                         f"{tp:.3f} | {sl:.3f} | {r['aggregate_net']:.3f} |")
        lines.append("")

    # Honest assessment
    lines.append("## Honest assessment")
    lines.append("")
    if len(bidir_winners) > 0:
        lines.append(f"**{len(bidir_winners)} bidirectional regime-stable cells exist.** These are the regime-agnostic signals the user demanded.")
        lines.append("Recommendation: wire the top bidirectional winner(s) into Razer paper trader with the TP/SL above. Run live paper for 2 weeks before any capital allocation.")
    elif len(unidir_winners) > 0:
        lines.append("**NO truly bidirectional regime-agnostic cell was found.** Some unidirectional cells survive all 4 sub-windows, but they are inherently regime-dependent (e.g. a long-only winner will die in a sustained bearish regime).")
        lines.append("")
        lines.append("This matches user's concern about HC #410's '30s LONG Top0.5 winner' — long-only signals are fragile across regimes by construction.")
        lines.append("")
        lines.append("Next research directions (in priority order):")
        lines.append("1. **Regime classifier ensemble** — explicit volatility/trend regime detector that picks long-only vs short-only model per regime. Risk: regime classifier itself becomes the fragile component.")
        lines.append("2. **Symmetric loss retraining** — current models may be biased toward one side by training-data imbalance. Retrain v3.3/v3.4.2 with class-balanced or returns-quantile-balanced loss.")
        lines.append("3. **Longer OOT horizon** — 15-16 days is short; 4 sub-windows of ~4 days each are noisy. Extend OOT to 60+ days to get larger sub-windows with tighter CIs.")
        lines.append("4. **Cross-model confluence** — require both v3.3 AND v3.4.2 to agree on direction (intersection of top-tier preds). May yield fewer but more stable bidirectional fills.")
    else:
        lines.append("**NO regime-stable cell of ANY kind was found** — neither bidirectional nor unidirectional cells pass the honesty gate in all 4 sub-windows.")
        lines.append("")
        lines.append("This means: every promoted aggregate-OOT result in HC #410 was carried by 1-2 favorable sub-windows and would have failed in the other(s). DO NOT deploy these models to paper trader on signal-only basis.")
        lines.append("")
        lines.append("Next research directions:")
        lines.append("1. **Investigate sub-window failure modes** — read `regime_stability_matrix.csv` and identify which windows broke each candidate. Is it always the same date range?")
        lines.append("2. **Extend OOT** — 4 contiguous chunks of ~4 days each have wide CIs; with 60+ days of OOT the gate may become passable.")
        lines.append("3. **Add execution layer** — perhaps the raw horizon-end signal isn't tradeable but a smart-execution agent (Neptune RL/MLP) can lift expectancy by adaptive exit.")
        lines.append("4. **Re-examine commission assumption** — if a market maker rebate path exists (passive fills only, with negative commission), the bar drops.")
    lines.append("")

    lines.append("## Limitations & caveats")
    lines.append("")
    lines.append(f"- **Sub-window count = {N_WINDOWS}**. With 15-16 OOT days that's ~4 days per window — wide CIs. Promote-gate may be too strict at this granularity.")
    lines.append("- **MAE at 1s/5s/10s is a proxy** (mean of negative-only realized horizon-end moves) — NOT a true intra-horizon adverse-excursion. The 30s MAE is the true `target_pred_mae_30s_ticks`. Treat non-30s MAE numbers as a lower bound on true intra-window adverse.")
    lines.append("- **Confidence ranking is in-window**: within each sub-window the top-k is taken from that window's samples (NOT a global top-k restricted to the window). This is the correct semantic for 'would I have traded this in real time inside this regime'.")
    lines.append("- **No v2 in this analysis** — v2 has discrete labels {0, 0.5, 1.0} and cannot produce tick-level MFE/MAE/net.")
    lines.append("- **Cost model**: net = realized - 0.376 (commission only, passive entry assumed). Market-cross adds +1.0 tick spread.")
    lines.append("- **OOT periods are short and unique per model** — v3.3 = 15 days in March 2026, v3.4.2 = 16 days Feb 23-Mar 15. The regimes covered are NOT exhaustive — even a bidirectional regime-stable cell here may fail on out-of-sample regimes not represented (e.g. extreme vol events).")
    lines.append("- **Independence assumption in CI**: 95% CI uses iid normal approximation per sub-window. Intra-day autocorrelation likely inflates CI tightness (real CIs are wider).")

    verdict_path = OUT_DIR / "verdict.md"
    verdict_path.write_text("\n".join(lines))
    _log(f"verdict: {verdict_path}")

    elapsed = time.time() - t0
    _log(f"DONE in {elapsed:.1f}s")
    _log(f"Bidirectional regime-stable cell counts: {n_bidir_per_model}")


if __name__ == "__main__":
    main()
