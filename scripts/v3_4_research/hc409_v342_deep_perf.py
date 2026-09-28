"""
HC #409 D2 — Deep performance analysis: v3.4.2 ep1 OOT (per-sample predictions).

Companion to scripts/v3_3_research/hc409_deep_perf_compare.py (v3.3 side).

Inputs (READ-ONLY):
  output/v342_fold_00_ep1_oot_inference.npz   (241,351 samples, 5 OOT days
                                                2026-02-23..2026-02-27)
  output/v3_3_extended_oot_20260514/extended_oot_predictions.npz
                                              (673,184 samples, 15 OOT days)

Outputs (own dir only):
  output/hc409_v342_deep_perf_<ts>/
    v342_full_perf_matrix.csv         per-(horizon,side,conf_tier) full table
    v342_head_correlation.csv         cross-head Spearman matrix
    v342_calibration_buckets.csv      decile calibration per horizon
    v342_mfe_mae_by_tier.csv          MFE/MAE per tier per horizon (30s, 60s)
    v342_fifo_target_by_tier.csv      realized FIFO TP4SL3 / TP8SL5 net per tier
    V342_vs_V33_SUMMARY.csv           side-by-side IC / tier-IC / WR / cost-hit
    SUMMARY.json                      run metadata + headline deltas

Honors:
  - HC #405 cost framing: report pct_above_0.376 (passive), pct_above_0.876
    (hybrid), pct_above_1.376 (market) separately. Do NOT conflate.
  - HC #408 conf tiers: Top10/5/1/0.5/0.1 (matches v3.3 script).
  - HC #344 day_conc: max-day-share of selected cells.
  - HC #409: no trade-system promotion here; this is COMPARISON ONLY.

Caveats:
  - v3.4.2 OOT window (5 days 02-23..02-27) is a SUBSET of v3.3 OOT (15 days
    2026-01-02..2026-02-27 or similar). Per-sample IC/MFE comparable;
    day_conc / 15-day stability NOT directly comparable.
  - v3.4.2 NPZ does NOT contain `oot_dates` array; we partition samples into
    5 equal day-chunks under the assumption inference iterated days in order.

CPU-only. <2 min. Read-only on NPZs. Writes only to OUT dir.
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scstats

PROJ = Path("/home/jupiter/Lvl3Quant")
NPZ_V342 = PROJ / "output" / "v342_fold_00_ep1_oot_inference.npz"
NPZ_V33 = PROJ / "output" / "v3_3_extended_oot_20260514" / "extended_oot_predictions.npz"

HORIZONS = ["1s", "5s", "10s", "30s"]  # core scalar IC heads (match v3.3 script)
EXT_HORIZONS = ["1s", "5s", "10s", "30s", "60s", "5min"]  # v3.4.2 also has these
SIDES = ["long", "short"]
TIERS = [
    ("Top10", 0.10),
    ("Top5", 0.05),
    ("Top1", 0.01),
    ("Top0.5", 0.005),
    ("Top0.1", 0.001),
]

# HC #405 cost framing (commission only; spread crossing only on market orders)
COMMISSION_TK = 0.376
HALF_SPREAD_TK = 0.5
MARKET_CROSS_TK = 1.0

COST_THRESHOLDS = {
    "passive_0.376": COMMISSION_TK,                       # passive limit
    "hybrid_0.876": COMMISSION_TK + HALF_SPREAD_TK,        # half-spread fill
    "market_1.376": COMMISSION_TK + MARKET_CROSS_TK,       # full cross
}

# v3.4.2 NPZ assumed day partition (no `oot_dates` key present)
V342_OOT_DAYS = ["2026-02-23", "2026-02-24", "2026-02-25",
                 "2026-02-26", "2026-02-27"]
N_DAYS_V342 = len(V342_OOT_DAYS)

OUT = PROJ / "output" / f"hc409_v342_deep_perf_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
OUT.mkdir(parents=True, exist_ok=True)


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _load_npz_keys(npz_path: Path) -> dict:
    """Load NPZ and return a {key: ndarray} dict. Print schema."""
    d = np.load(npz_path, allow_pickle=True)
    _log(f"loaded {npz_path.name}: {len(d.files)} keys")
    return {k: d[k] for k in d.files}


def _safe_corr(a: np.ndarray, b: np.ndarray, kind: str = "pearson") -> float:
    if len(a) < 3:
        return float("nan")
    try:
        if kind == "pearson":
            r, _ = scstats.pearsonr(a, b)
        else:
            r, _ = scstats.spearmanr(a, b)
        return float(r)
    except Exception:
        return float("nan")


def _select_topk_signed(signed_pred: np.ndarray, k: int) -> np.ndarray:
    """Return indices of top-k by signed_pred (largest)."""
    n = len(signed_pred)
    if k >= n:
        return np.arange(n)
    return np.argpartition(-signed_pred, k - 1)[:k]


def compute_perf_matrix(arrs: dict, model_tag: str, n_days: int) -> pd.DataFrame:
    """Build the per-(horizon,side,conf_tier) performance matrix.

    Mirrors hc409_deep_perf_compare.py columns + adds the 3 HC #405 cost
    thresholds explicitly (passive/hybrid/market).
    """
    n_samples = int(arrs[f"pred_log_ret_1s"].shape[0])
    samples_per_day = n_samples // n_days
    date_idx_full = np.minimum(np.arange(n_samples) // samples_per_day, n_days - 1)

    rows = []
    for h in HORIZONS:
        pred = arrs[f"pred_log_ret_{h}"]
        target = arrs[f"target_log_ret_{h}"]
        mask_key = f"mask_log_ret_{h}"
        if mask_key in arrs:
            mask = arrs[mask_key].astype(bool)
        else:
            mask = np.ones_like(pred, dtype=bool)
        ok = mask & np.isfinite(pred) & np.isfinite(target)
        n_ok = int(ok.sum())
        if n_ok < 100:
            _log(f"  [{model_tag}] horizon={h} skipped (n_ok={n_ok})")
            continue

        pred_ok = pred[ok]
        target_ok = target[ok]
        date_ok = date_idx_full[ok]

        ic_p = _safe_corr(pred_ok, target_ok, "pearson")
        ic_s = _safe_corr(pred_ok, target_ok, "spearman")
        _log(f"  [{model_tag}] h={h} n_ok={n_ok} ic_p={ic_p:.4f} ic_s={ic_s:.4f}")

        for side in SIDES:
            sign = 1.0 if side == "long" else -1.0
            signed_pred = sign * pred_ok

            for tier_name, tier_frac in TIERS:
                k = max(1, int(n_ok * tier_frac))
                sel = _select_topk_signed(signed_pred, k)
                n_cell = len(sel)
                if n_cell < 20:
                    continue
                pred_c = pred_ok[sel]
                target_c = target_ok[sel]
                realized = sign * target_c  # in log-ret units; treat as ticks-ish

                # Convert log-ret to ticks: ES tick = 0.25 pts; px ~ 5000 => 1 tick ~ 5e-5.
                # Targets are stored as log-ret; v3.3 script treats them as "tk" raw.
                # We preserve the same units / framing here for direct comparability.
                ic_p_tier = _safe_corr(pred_c * sign, realized, "pearson") if n_cell >= 30 else float("nan")
                ic_s_tier = _safe_corr(pred_c * sign, realized, "spearman") if n_cell >= 30 else float("nan")

                gross_mean = float(realized.mean())
                gross_median = float(np.median(realized))
                wr = float((realized > 0).mean())
                pct_above_passive = float((realized > COST_THRESHOLDS["passive_0.376"]).mean())
                pct_above_hybrid = float((realized > COST_THRESHOLDS["hybrid_0.876"]).mean())
                pct_above_market = float((realized > COST_THRESHOLDS["market_1.376"]).mean())

                net_passive = realized - COST_THRESHOLDS["passive_0.376"]
                mean_net_passive = float(net_passive.mean())
                std_net_passive = float(net_passive.std(ddof=1)) if n_cell > 1 else float("nan")
                sharpe_passive = (mean_net_passive / std_net_passive) if std_net_passive and std_net_passive > 0 else float("nan")

                counts = np.bincount(date_ok[sel], minlength=n_days)
                day_conc = float(counts.max() / n_cell) if n_cell > 0 else float("nan")

                rows.append(dict(
                    model=model_tag,
                    horizon=h, side=side, conf_tier=tier_name,
                    n_cell=n_cell,
                    overall_ic_pearson=round(ic_p, 4),
                    overall_ic_spearman=round(ic_s, 4),
                    tier_ic_pearson=round(ic_p_tier, 4),
                    tier_ic_spearman=round(ic_s_tier, 4),
                    gross_mean=round(gross_mean, 6),
                    gross_median=round(gross_median, 6),
                    wr=round(wr, 4),
                    pct_above_passive_0p376=round(pct_above_passive, 4),
                    pct_above_hybrid_0p876=round(pct_above_hybrid, 4),
                    pct_above_market_1p376=round(pct_above_market, 4),
                    mean_net_passive=round(mean_net_passive, 6),
                    sharpe_passive=round(sharpe_passive, 4),
                    day_conc=round(day_conc, 4),
                ))
    return pd.DataFrame(rows)


def compute_head_correlation(arrs: dict, horizons: list[str]) -> pd.DataFrame:
    rows = []
    for i, h1 in enumerate(horizons):
        for h2 in horizons[i + 1:]:
            a = arrs[f"pred_log_ret_{h1}"]
            b = arrs[f"pred_log_ret_{h2}"]
            n_min = min(len(a), len(b))
            rho = _safe_corr(a[:n_min], b[:n_min], "spearman")
            rows.append(dict(head_a=h1, head_b=h2, spearman=round(rho, 4)))
    return pd.DataFrame(rows)


def compute_calibration(arrs: dict, horizons: list[str]) -> pd.DataFrame:
    rows = []
    for h in horizons:
        pred = arrs[f"pred_log_ret_{h}"]
        target = arrs[f"target_log_ret_{h}"]
        mask_key = f"mask_log_ret_{h}"
        if mask_key in arrs:
            mask = arrs[mask_key].astype(bool)
        else:
            mask = np.ones_like(pred, dtype=bool)
        ok = mask & np.isfinite(pred) & np.isfinite(target)
        p = pred[ok]
        t = target[ok]
        if len(p) < 100:
            continue
        edges = np.percentile(p, np.linspace(0, 100, 11))
        bucket = np.digitize(p, edges[1:-1])
        for b in range(10):
            m = bucket == b
            if m.sum() < 50:
                continue
            rows.append(dict(
                horizon=h,
                pred_decile=b + 1,
                n=int(m.sum()),
                pred_mean=round(float(p[m].mean()), 6),
                realized_mean=round(float(t[m].mean()), 6),
                realized_pct_pos=round(float((t[m] > 0).mean()), 4),
            ))
    return pd.DataFrame(rows)


def compute_mfe_mae_by_tier(arrs: dict) -> pd.DataFrame:
    """v3.4.2-only: realized MFE/MAE per (signal-horizon, side, conf_tier).

    Uses pred_log_ret_<h> as the SELECTION signal, and target_pred_mfe_30s_ticks /
    target_pred_mae_30s_ticks (and 60s) as the REALIZED outcomes.

    target_pred_mfe_30s_ticks = realized max-favorable-excursion in TICKS within 30s.
    target_pred_mae_30s_ticks = realized max-adverse-excursion in TICKS within 30s.

    Sign convention assumption (validated by columns existing in pairs):
      The MFE/MAE targets are stored as absolute (long-equivalent) ticks where
      positive = favorable for a LONG.  For a SHORT, MFE = -MAE_long, MAE = -MFE_long.
      We report SIDE-AWARE realized MFE and MAE separately.
    """
    rows = []
    if "target_pred_mfe_30s_ticks" not in arrs:
        return pd.DataFrame()

    mfe30 = arrs["target_pred_mfe_30s_ticks"]
    mae30 = arrs["target_pred_mae_30s_ticks"]
    mfe60 = arrs["target_pred_mfe_60s_ticks"]
    mae60 = arrs["target_pred_mae_60s_ticks"]
    mask30 = arrs["mask_pred_mfe_30s_ticks"].astype(bool) & \
             arrs["mask_pred_mae_30s_ticks"].astype(bool)
    mask60 = arrs["mask_pred_mfe_60s_ticks"].astype(bool) & \
             arrs["mask_pred_mae_60s_ticks"].astype(bool)

    for h in HORIZONS:
        pred = arrs[f"pred_log_ret_{h}"]
        for side in SIDES:
            sign = 1.0 if side == "long" else -1.0
            signed_pred = sign * pred

            for tier_name, tier_frac in TIERS:
                for window_label, mfe, mae, mwin in [
                    ("30s", mfe30, mae30, mask30),
                    ("60s", mfe60, mae60, mask60),
                ]:
                    ok = mwin & np.isfinite(signed_pred) & np.isfinite(mfe) & np.isfinite(mae)
                    n_ok = int(ok.sum())
                    if n_ok < 100:
                        continue
                    sp = signed_pred[ok]
                    mfe_ok = mfe[ok]
                    mae_ok = mae[ok]
                    k = max(1, int(n_ok * tier_frac))
                    sel = _select_topk_signed(sp, k)
                    if len(sel) < 20:
                        continue
                    # side-aware: long takes mfe/mae as-is; short flips sign
                    if side == "long":
                        realized_mfe = mfe_ok[sel]
                        realized_mae = mae_ok[sel]
                    else:
                        realized_mfe = -mae_ok[sel]  # short's favorable = -long_mae
                        realized_mae = -mfe_ok[sel]  # short's adverse   = -long_mfe
                    rows.append(dict(
                        signal_horizon=h,
                        side=side,
                        conf_tier=tier_name,
                        window=window_label,
                        n=len(sel),
                        mean_mfe_tk=round(float(realized_mfe.mean()), 3),
                        median_mfe_tk=round(float(np.median(realized_mfe)), 3),
                        mean_mae_tk=round(float(realized_mae.mean()), 3),
                        median_mae_tk=round(float(np.median(realized_mae)), 3),
                        mfe_minus_mae=round(float(realized_mfe.mean() - realized_mae.mean()), 3),
                        pct_mfe_above_passive=round(float((realized_mfe > COST_THRESHOLDS["passive_0.376"]).mean()), 4),
                        pct_mfe_above_market=round(float((realized_mfe > COST_THRESHOLDS["market_1.376"]).mean()), 4),
                    ))
    return pd.DataFrame(rows)


def compute_fifo_by_tier(arrs: dict) -> pd.DataFrame:
    """v3.4.2-only: realized FIFO TP/SL net per (signal-horizon, side, conf_tier).

    Uses target_fifo_tp4sl3_net / target_fifo_tp8sl5_net columns (in TICKS,
    already net of TP/SL outcomes from the labeling pipeline). This is the
    closest thing to "real FIFO order queue with adverse selection" available
    from the inference NPZ alone.
    """
    rows = []
    if "target_fifo_tp4sl3_net" not in arrs:
        return pd.DataFrame()

    targets = {
        "tp4sl3": (arrs["target_fifo_tp4sl3_net"], arrs["mask_fifo_tp4sl3_net"].astype(bool),
                   arrs.get("target_fifo_tp4sl3_hit_tp")),
        "tp8sl5": (arrs["target_fifo_tp8sl5_net"], arrs["mask_fifo_tp8sl5_net"].astype(bool),
                   arrs.get("target_fifo_tp8sl5_hit_tp")),
    }

    for h in HORIZONS:
        pred = arrs[f"pred_log_ret_{h}"]
        for side in SIDES:
            sign = 1.0 if side == "long" else -1.0
            signed_pred = sign * pred

            for tier_name, tier_frac in TIERS:
                for fifo_name, (net, m, hit_tp) in targets.items():
                    ok = m & np.isfinite(signed_pred) & np.isfinite(net)
                    n_ok = int(ok.sum())
                    if n_ok < 100:
                        continue
                    sp = signed_pred[ok]
                    net_ok = net[ok]
                    hit_ok = hit_tp[ok] if hit_tp is not None else None
                    k = max(1, int(n_ok * tier_frac))
                    sel = _select_topk_signed(sp, k)
                    if len(sel) < 20:
                        continue
                    # FIFO targets in v3.4 are stored long-side; flip for short.
                    realized_net = sign * net_ok[sel]
                    rows.append(dict(
                        signal_horizon=h,
                        side=side,
                        conf_tier=tier_name,
                        fifo_setup=fifo_name,
                        n=len(sel),
                        mean_net_tk=round(float(realized_net.mean()), 3),
                        median_net_tk=round(float(np.median(realized_net)), 3),
                        wr_net_pos=round(float((realized_net > 0).mean()), 4),
                        pct_net_above_passive=round(float((realized_net > COST_THRESHOLDS["passive_0.376"]).mean()), 4),
                        pct_net_above_market=round(float((realized_net > COST_THRESHOLDS["market_1.376"]).mean()), 4),
                        mean_hit_tp=round(float(hit_ok[sel].mean()), 4) if hit_ok is not None else float("nan"),
                    ))
    return pd.DataFrame(rows)


def build_v342_vs_v33(v342_perf: pd.DataFrame, v33_perf: pd.DataFrame) -> pd.DataFrame:
    """Side-by-side: per (horizon,side,conf_tier), overall+tier IC, WR,
    pct-above-cost thresholds. Includes deltas."""
    keys = ["horizon", "side", "conf_tier"]
    metrics = [
        "n_cell", "overall_ic_pearson", "tier_ic_pearson", "wr",
        "pct_above_passive_0p376", "pct_above_hybrid_0p876",
        "pct_above_market_1p376", "mean_net_passive", "sharpe_passive",
        "day_conc",
    ]
    left = v342_perf[keys + metrics].rename(columns={m: f"v342_{m}" for m in metrics})
    right = v33_perf[keys + metrics].rename(columns={m: f"v33_{m}" for m in metrics})
    side_by_side = left.merge(right, on=keys, how="outer")
    for m in metrics:
        if m == "n_cell":
            continue
        side_by_side[f"delta_{m}"] = (
            side_by_side[f"v342_{m}"] - side_by_side[f"v33_{m}"]
        ).round(4)
    return side_by_side


def main() -> None:
    t0 = time.time()
    _log(f"OUT={OUT}")
    _log(f"loading v3.4.2 NPZ: {NPZ_V342}")
    arrs_v342 = _load_npz_keys(NPZ_V342)
    n_samples_v342 = int(arrs_v342["pred_log_ret_1s"].shape[0])
    _log(f"v3.4.2 n_samples={n_samples_v342} (assumed {N_DAYS_V342} OOT days)")

    # Print schema discovery for the log
    _log("v3.4.2 NPZ schema discovery:")
    print(f"  total_keys={len(arrs_v342)}")
    for k in sorted(arrs_v342.keys()):
        a = arrs_v342[k]
        print(f"    {k}: shape={a.shape} dtype={a.dtype}")

    _log(f"loading v3.3 NPZ: {NPZ_V33}")
    arrs_v33 = _load_npz_keys(NPZ_V33)
    n_samples_v33 = int(arrs_v33["pred_log_ret_1s"].shape[0])
    n_days_v33 = int(len(arrs_v33["oot_dates"]))
    _log(f"v3.3 n_samples={n_samples_v33} n_days={n_days_v33}")

    # ---- v3.4.2 perf matrix ----
    _log("computing v3.4.2 perf matrix...")
    v342_perf = compute_perf_matrix(arrs_v342, "v3.4.2", N_DAYS_V342)
    v342_perf.to_csv(OUT / "v342_full_perf_matrix.csv", index=False)
    _log(f"  -> {OUT/'v342_full_perf_matrix.csv'} ({len(v342_perf)} rows)")

    # ---- v3.3 perf matrix (recomputed with same cost framing for direct compare) ----
    _log("computing v3.3 perf matrix (re-run with 3-threshold cost framing)...")
    v33_perf = compute_perf_matrix(arrs_v33, "v3.3", n_days_v33)
    v33_perf.to_csv(OUT / "v33_full_perf_matrix_recomputed.csv", index=False)
    _log(f"  -> {OUT/'v33_full_perf_matrix_recomputed.csv'} ({len(v33_perf)} rows)")

    # ---- head correlation ----
    _log("computing v3.4.2 head correlation matrix...")
    hc = compute_head_correlation(arrs_v342, EXT_HORIZONS)
    hc.to_csv(OUT / "v342_head_correlation.csv", index=False)
    _log(f"  -> {OUT/'v342_head_correlation.csv'} ({len(hc)} pairs)")

    # ---- calibration ----
    _log("computing v3.4.2 calibration buckets...")
    calib = compute_calibration(arrs_v342, EXT_HORIZONS)
    calib.to_csv(OUT / "v342_calibration_buckets.csv", index=False)
    _log(f"  -> {OUT/'v342_calibration_buckets.csv'} ({len(calib)} rows)")

    # ---- MFE/MAE per tier (v3.4.2 ONLY — v3.3 lacks these targets in scope) ----
    _log("computing v3.4.2 MFE/MAE per tier (30s/60s windows)...")
    mfemae = compute_mfe_mae_by_tier(arrs_v342)
    mfemae.to_csv(OUT / "v342_mfe_mae_by_tier.csv", index=False)
    _log(f"  -> {OUT/'v342_mfe_mae_by_tier.csv'} ({len(mfemae)} rows)")
    has_mfemae = len(mfemae) > 0

    # ---- FIFO TP/SL per tier (v3.4.2 ONLY) ----
    _log("computing v3.4.2 FIFO TP/SL per tier...")
    fifo = compute_fifo_by_tier(arrs_v342)
    fifo.to_csv(OUT / "v342_fifo_target_by_tier.csv", index=False)
    _log(f"  -> {OUT/'v342_fifo_target_by_tier.csv'} ({len(fifo)} rows)")
    has_fifo = len(fifo) > 0

    # ---- v3.4.2 vs v3.3 side-by-side ----
    _log("building V342_vs_V33_SUMMARY...")
    side_by_side = build_v342_vs_v33(v342_perf, v33_perf)
    side_by_side.to_csv(OUT / "V342_vs_V33_SUMMARY.csv", index=False)
    _log(f"  -> {OUT/'V342_vs_V33_SUMMARY.csv'} ({len(side_by_side)} rows)")

    # ---- Headline deltas for stdout report ----
    v342_overall_ic = {}
    v33_overall_ic = {}
    for h in HORIZONS:
        v342_overall_ic[h] = float(arrs_v342[f"metric_ic_log_ret_{h}"]) \
            if f"metric_ic_log_ret_{h}" in arrs_v342 else float("nan")
    # v3.3 overall pearson on the fly:
    for h in HORIZONS:
        p = arrs_v33[f"pred_log_ret_{h}"]
        t = arrs_v33[f"target_log_ret_{h}"]
        m = arrs_v33[f"mask_log_ret_{h}"].astype(bool)
        ok = m & np.isfinite(p) & np.isfinite(t)
        v33_overall_ic[h] = _safe_corr(p[ok], t[ok], "pearson")

    delta_ic = {h: round(v342_overall_ic[h] - v33_overall_ic[h], 4) for h in HORIZONS}

    # Find biggest-gain conf-tier (by overall_ic_pearson is constant per horizon;
    # use tier_ic_pearson delta instead)
    best_tier_gain = None
    if "delta_tier_ic_pearson" in side_by_side.columns:
        cand = side_by_side.dropna(subset=["delta_tier_ic_pearson"])
        if len(cand):
            row = cand.loc[cand["delta_tier_ic_pearson"].idxmax()]
            best_tier_gain = dict(
                horizon=row["horizon"], side=row["side"], conf_tier=row["conf_tier"],
                v342=float(row["v342_tier_ic_pearson"]),
                v33=float(row["v33_tier_ic_pearson"]),
                delta=float(row["delta_tier_ic_pearson"]),
            )

    # 1s-head Top1 MFE compare (v3.4.2 gross_mean for Top1 long vs short)
    top1_1s_v342 = v342_perf[(v342_perf.horizon == "1s") & (v342_perf.conf_tier == "Top1")]
    top1_1s_v33 = v33_perf[(v33_perf.horizon == "1s") & (v33_perf.conf_tier == "Top1")]

    summary = dict(
        generated_at=datetime.now().isoformat(),
        npz_v342=str(NPZ_V342),
        npz_v33=str(NPZ_V33),
        v342_n_samples=n_samples_v342,
        v342_n_days_assumed=N_DAYS_V342,
        v342_oot_dates_assumed=V342_OOT_DAYS,
        v33_n_samples=n_samples_v33,
        v33_n_days=n_days_v33,
        cost_framing="HC #405: passive=0.376, hybrid=0.876, market=1.376 (TICKS)",
        v342_overall_ic_from_npz_metric=v342_overall_ic,
        v33_overall_ic_pearson=v33_overall_ic,
        delta_ic_v342_minus_v33=delta_ic,
        best_tier_ic_gain=best_tier_gain,
        v342_top1_1s_perf=top1_1s_v342.to_dict("records") if len(top1_1s_v342) else None,
        v33_top1_1s_perf=top1_1s_v33.to_dict("records") if len(top1_1s_v33) else None,
        mfe_mae_computable_from_npz=has_mfemae,
        fifo_targets_computable_from_npz=has_fifo,
        caveats=[
            "v3.4.2 OOT window (5 days 2026-02-23..27) is SUBSET of v3.3 OOT (15 days). "
            "Per-sample IC + MFE/WR comparable; 15-day stability NOT.",
            "v3.4.2 NPZ has no `oot_dates` array; partitioned into 5 equal chunks. "
            "day_conc for v3.4.2 is an approximation.",
            "MFE/MAE here uses LABEL targets baked into NPZ (target_pred_mfe_*_ticks). "
            "For tick-by-tick FIFO order queue with adverse selection on price-path "
            "data, run on Neptune against MBO replay.",
            "No trade-system promotion — comparison only (HC #408 gate is separate).",
        ],
        elapsed_sec=round(time.time() - t0, 2),
    )
    (OUT / "SUMMARY.json").write_text(json.dumps(summary, indent=2, default=str))
    _log(f"SUMMARY.json -> {OUT/'SUMMARY.json'}")

    # ---- Concise stdout report ----
    print("\n" + "=" * 72)
    print("V342 vs V33 HEADLINE")
    print("=" * 72)
    print(f"{'horizon':<8}{'v3.3_IC':<12}{'v3.4.2_IC':<12}{'delta':<10}")
    for h in HORIZONS:
        d = delta_ic[h]
        sign = "+" if d >= 0 else ""
        print(f"{h:<8}{v33_overall_ic[h]:<12.4f}{v342_overall_ic[h]:<12.4f}{sign}{d:<10.4f}")

    if best_tier_gain:
        print(f"\nBiggest tier-IC gain: "
              f"h={best_tier_gain['horizon']} side={best_tier_gain['side']} "
              f"tier={best_tier_gain['conf_tier']} "
              f"v3.3={best_tier_gain['v33']:.4f} -> v3.4.2={best_tier_gain['v342']:.4f} "
              f"(delta {best_tier_gain['delta']:+.4f})")

    print("\nTop1 / 1s-head gross_mean (positive = favorable for the side):")
    for side in SIDES:
        v342_row = top1_1s_v342[top1_1s_v342.side == side]
        v33_row = top1_1s_v33[top1_1s_v33.side == side]
        if len(v342_row) and len(v33_row):
            v342_gm = float(v342_row["gross_mean"].iloc[0])
            v33_gm = float(v33_row["gross_mean"].iloc[0])
            d = v342_gm - v33_gm
            sign = "+" if d >= 0 else ""
            print(f"  {side:<6} v3.3={v33_gm:+.6f}  v3.4.2={v342_gm:+.6f}  delta={sign}{d:.6f}")

    if has_mfemae:
        print("\n1s-head Top1 realized MFE (ticks) — 30s window:")
        mm = mfemae[(mfemae.signal_horizon == "1s") &
                    (mfemae.conf_tier == "Top1") &
                    (mfemae.window == "30s")]
        for _, r in mm.iterrows():
            print(f"  {r['side']:<6} mean_mfe={r['mean_mfe_tk']:+.3f}tk  "
                  f"mean_mae={r['mean_mae_tk']:+.3f}tk  "
                  f"mfe-mae={r['mfe_minus_mae']:+.3f}tk  n={int(r['n'])}")

    if has_fifo:
        print("\n1s-head Top1 realized FIFO TP4SL3 net (ticks):")
        ff = fifo[(fifo.signal_horizon == "1s") &
                  (fifo.conf_tier == "Top1") &
                  (fifo.fifo_setup == "tp4sl3")]
        for _, r in ff.iterrows():
            print(f"  {r['side']:<6} mean_net={r['mean_net_tk']:+.3f}tk  "
                  f"wr={r['wr_net_pos']:.3f}  hit_tp={r['mean_hit_tp']:.3f}  n={int(r['n'])}")

    print(f"\nElapsed: {time.time()-t0:.1f}s")
    print(f"OUT: {OUT}")


if __name__ == "__main__":
    main()
