#!/usr/bin/env python3
"""
HC #428 R1 — Per-day audit of HC #429 Optuna sweep top-20 configs.

Purpose: detect whether the sweep top configs (best_configs.json) are
regime-monoculture artifacts. The sweep evaluated on ONLY 5 OOT dates
(20260223-0227); HC #428 R1 forbids deployment of any config developed
on a single-regime window. This script:

  1. Replays each top-20 config from best_configs.json against the
     v3.4.2 5-day prediction NPZ.
  2. Emits per-date metrics (n_fills, mean_net, Sharpe, PF, WR).
  3. Cross-joins with regime labels (when output/regime_labels/oot_dates_regime.parquet
     covers those dates) to flag regime monoculture (all-up / all-down / all-flat).
  4. Computes day-conc per config (HC #344 cap = 0.70).
  5. Writes verdict.md, per_config_per_day.csv, summary.json.

Read-only on source NPZs/parquets. Output: output/hc428_r1_sweep_per_day_<ts>/
"""
from __future__ import annotations

import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path("/home/jupiter/Lvl3Quant")
SWEEP_DIR = PROJ / "output" / "v342_execution_optuna_20260519"
BEST_CONFIGS = SWEEP_DIR / "best_configs.json"
LEADERBOARD = SWEEP_DIR / "leaderboard.csv"
V342_PRED = PROJ / "output" / "cnn_mamba_v3_4_2_fixedmtl" / "fold_00_predictions.npz"
REGIME_LABELS = PROJ / "output" / "regime_labels" / "oot_dates_regime.parquet"
SWEEP_DATES = ["20260223", "20260224", "20260225", "20260226", "20260227"]

TS = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = PROJ / "output" / f"hc428_r1_sweep_per_day_{TS}"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def load_predictions():
    z = np.load(V342_PRED, allow_pickle=True)
    keys = list(z.files)
    _log(f"loaded NPZ: {len(keys)} keys, n_samples={z['pred_log_ret_1s'].shape[0]:,}")
    if "oot_dates" in keys:
        _log(f"oot_dates: {list(z['oot_dates'])}")
    return z


def derive_sample_dates(z, sweep_dates=SWEEP_DATES):
    """The NPZ is concatenated across the 5 dates in order. Without a per-sample
    date column, we approximate by equal-split (each date contributes roughly
    n_total / 5 samples, but in practice samples are non-uniform). Try to find
    a date column, else fall back to per-fold equal split with a warning.
    """
    files = list(z.files)
    for cand in ["sample_date", "date", "dates", "ts_date"]:
        if cand in files:
            arr = z[cand]
            if arr.shape[0] == z["pred_log_ret_1s"].shape[0]:
                _log(f"Using sample-date column: {cand}")
                return np.asarray(arr).astype(str)

    # Approximate fallback: equal split (NOT correct but gives ballpark)
    n_total = z["pred_log_ret_1s"].shape[0]
    n_per = n_total // len(sweep_dates)
    _log(f"WARNING: no per-sample date column. Using equal-split fallback: {n_per:,}/date")
    arr = np.empty(n_total, dtype="<U8")
    for i, d in enumerate(sweep_dates):
        lo = i * n_per
        hi = (i + 1) * n_per if i < len(sweep_dates) - 1 else n_total
        arr[lo:hi] = d
    return arr


def simulate_config(z, sample_dates, cfg, commission_ticks_default=0.376):
    """Lightweight per-date FIFO-style simulator that mirrors the sweep's
    acceptance scoring. We use the same prediction heads + confluence as
    Optuna trial. Costs: commission_ticks (per cfg), spread is via order_type.

    NOTE: This is an audit replay (per-day decomposition), NOT a re-validation
    of the sweep's deploy_eligible filter. Aim is to expose day attribution.
    """
    p = cfg["params"]
    horizon = p["head_horizon"]
    side = p["side"]
    conf_thr = float(p["conf_thr"])
    cancel_window = int(p["cancel_window"])
    hold_seconds = float(p["hold_seconds"])
    tp = float(p["tp_ticks"])
    sl = float(p["sl_ticks"])
    spread = float(p["spread_ticks"])
    tod_start = int(p["tod_start_hour"])
    tod_end = int(p["tod_end_hour"])
    pred_strength_min = float(p["pred_strength_min"])
    sigma_halt_mult = float(p["sigma_halt_mult"])
    commission = float(p.get("commission_ticks", commission_ticks_default))
    use_fifo = bool(p.get("use_fifo_confluence", False))
    fifo_head = p.get("fifo_confluence_head", "pred_fifo_tp8sl5_net")
    fifo_thr = float(p.get("fifo_confluence_thr_ticks", 0.0))
    use_hconf = bool(p.get("use_horizon_confluence", False))
    hconf_h = p.get("confluence_horizon", "1s")

    # Predictions for this horizon
    pred_key = f"pred_log_ret_{horizon}"
    if pred_key not in z.files:
        return None
    pred = z[pred_key]
    target_h_key = f"target_log_ret_{horizon}"
    if target_h_key not in z.files:
        return None
    target = z[target_h_key]
    mask_key = f"mask_log_ret_{horizon}"
    mask = (z[mask_key].astype(bool) if mask_key in z.files
            else np.ones_like(pred, dtype=bool))

    # Side filter: convert to signed predicted move (ticks). log_ret approximated
    # as fractional move; we'll use raw pred values and direction.
    if side == "long":
        signal = pred > conf_thr
    elif side == "short":
        signal = pred < -conf_thr
    else:
        return None

    # Pred strength filter (absolute prediction magnitude in ticks, approx)
    pred_ticks = pred * 4.0 / 1e-3  # rough scaling — not exact
    signal &= np.abs(pred_ticks) >= pred_strength_min

    # FIFO confluence
    if use_fifo and fifo_head in z.files:
        fifo_pred = z[fifo_head]
        if side == "long":
            signal &= fifo_pred >= fifo_thr
        else:
            signal &= fifo_pred <= -fifo_thr

    # Horizon confluence
    if use_hconf:
        hc_pred_key = f"pred_log_ret_{hconf_h}"
        if hc_pred_key in z.files:
            hc_pred = z[hc_pred_key]
            if side == "long":
                signal &= hc_pred > 0
            else:
                signal &= hc_pred < 0

    signal &= mask

    # Per-fill net ticks (target * side_sign) − costs
    side_sign = 1 if side == "long" else -1
    # Approximate target in ticks: log_ret * ES_price / tick_size (ES ~5000 / 0.25 = 20000)
    target_ticks = target * 20000.0
    pnl_ticks = target_ticks * side_sign

    # TP/SL clipping (intra-trade): if realized move exceeds tp → tp; if breaches sl → -sl
    realized = np.where(pnl_ticks > tp, tp, pnl_ticks)
    realized = np.where(pnl_ticks < -sl, -sl, realized)

    # Net of costs (commission + spread crossing if non-passive)
    cost = commission + (0.0 if "passive" in p.get("order_type", "") else 1.0)
    net = realized - cost

    # Per-date breakdown
    rows = []
    for d in SWEEP_DATES:
        m = (sample_dates == d) & signal
        n = int(m.sum())
        if n == 0:
            rows.append({"date": d, "n_fills": 0, "mean_net": np.nan, "sharpe": np.nan,
                          "pf": np.nan, "wr": np.nan, "n_neg": 0, "n_pos": 0})
            continue
        x = net[m]
        mean_net = float(x.mean())
        std = float(x.std(ddof=1)) if n > 1 else 1.0
        sharpe = (mean_net / std) * math.sqrt(n) if std > 0 else 0.0
        pos_sum = float(x[x > 0].sum())
        neg_sum = float(-x[x < 0].sum())
        pf = (pos_sum / neg_sum) if neg_sum > 0 else 999.0
        n_pos = int((x > 0).sum())
        n_neg = int((x < 0).sum())
        wr = (n_pos / n) * 100.0
        rows.append({"date": d, "n_fills": n, "mean_net": mean_net, "sharpe": sharpe,
                      "pf": pf, "wr": wr, "n_neg": n_neg, "n_pos": n_pos})
    return rows


def main():
    _log("HC #428 R1 sweep per-day audit starting")
    _log(f"OUT_DIR: {OUT_DIR}")
    z = load_predictions()
    sample_dates = derive_sample_dates(z)
    best = json.loads(BEST_CONFIGS.read_text())[:20]
    _log(f"loaded {len(best)} top configs from best_configs.json")

    all_rows = []
    for cfg in best:
        rows = simulate_config(z, sample_dates, cfg)
        if rows is None:
            continue
        trial = cfg["trial"]
        sweep_value = cfg["value"]
        for r in rows:
            r["trial"] = trial
            r["sweep_sharpe"] = sweep_value
            r["horizon"] = cfg["params"]["head_horizon"]
            r["side"] = cfg["params"]["side"]
            all_rows.append(r)

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_DIR / "per_config_per_day.csv", index=False)
    _log(f"wrote per_config_per_day.csv ({len(df)} rows)")

    # Day-conc check per config
    summary_rows = []
    for trial, g in df.groupby("trial"):
        total = g["n_fills"].sum()
        if total == 0:
            day_conc = 1.0
            sharpe_min, sharpe_max = float("nan"), float("nan")
        else:
            day_conc = float(g["n_fills"].max() / total)
            sh = g[g["n_fills"] > 0]["sharpe"]
            sharpe_min = float(sh.min()) if len(sh) > 0 else float("nan")
            sharpe_max = float(sh.max()) if len(sh) > 0 else float("nan")
        summary_rows.append({
            "trial": int(trial),
            "horizon": g.iloc[0]["horizon"],
            "side": g.iloc[0]["side"],
            "sweep_sharpe": float(g.iloc[0]["sweep_sharpe"]),
            "total_fills": int(total),
            "day_conc": day_conc,
            "n_days_with_fills": int((g["n_fills"] > 0).sum()),
            "sharpe_min_day": sharpe_min,
            "sharpe_max_day": sharpe_max,
            "hc344_day_conc_pass": day_conc <= 0.70,
        })
    sdf = pd.DataFrame(summary_rows).sort_values("sweep_sharpe", ascending=False)
    sdf.to_csv(OUT_DIR / "summary_by_trial.csv", index=False)

    # Regime check (if labels exist)
    regime_info = "NOT AVAILABLE (sweep dates not in regime labels file)"
    try:
        rdf = pd.read_parquet(REGIME_LABELS)
        rdf["date_s"] = rdf["date"].astype(str)
        hit = rdf[rdf["date_s"].isin(SWEEP_DATES)]
        if not hit.empty:
            regime_info = hit[["date_s", "close_minus_open_ticks", "trend_label"]].to_dict("records")
    except Exception as e:
        regime_info = f"ERR reading regime parquet: {e}"

    summary_json = {
        "ts": TS,
        "sweep_dir": str(SWEEP_DIR),
        "n_configs_audited": len(best),
        "sweep_dates": SWEEP_DATES,
        "regime_classification": regime_info,
        "n_configs_passing_hc344_day_conc": int(sdf["hc344_day_conc_pass"].sum()),
        "n_configs_with_fills_on_all_5_days": int((sdf["n_days_with_fills"] == 5).sum()),
        "n_configs_with_fills_on_ge_3_days": int((sdf["n_days_with_fills"] >= 3).sum()),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary_json, indent=2, default=str))

    # Verdict.md
    md = ["# HC #428 R1 Sweep Per-Day Audit",
          f"**Timestamp**: {TS}",
          f"**Sweep**: {SWEEP_DIR.name}",
          f"**Predictions**: {V342_PRED.name} (5 OOT dates: {SWEEP_DATES})",
          "",
          "## Caveat",
          "Per-sample date column was unavailable in v3.4.2 NPZ; this audit uses",
          "equal-split fallback (each date ≈ n_total/5 samples). RESULTS ARE",
          "APPROXIMATE; treat as monoculture detector, not as exact metrics replay.",
          "",
          "## Regime Classification of Sweep Window",
          str(regime_info),
          "",
          "## Top-20 Configs Summary",
          sdf.to_string(index=False),
          "",
          "## Gate Results",
          f"- HC #344 day-conc ≤ 0.70 passing: {summary_json['n_configs_passing_hc344_day_conc']}/{len(best)}",
          f"- Configs with fills on all 5 days: {summary_json['n_configs_with_fills_on_all_5_days']}/{len(best)}",
          f"- Configs with fills on ≥3 days: {summary_json['n_configs_with_fills_on_ge_3_days']}/{len(best)}",
          "",
          "## Verdict",
          "HC #428 R1 PROPER VALIDATION requires v3.4.2 predictions on the FULL 40+ day",
          "OOT range. The current 5-day NPZ is insufficient. This audit only exposes",
          "intra-sweep-window day-attribution; the deeper monoculture test (sweep window",
          "= all-green or all-red days) requires regime labels of 20260223-0227 (now",
          "running in parallel, PID look in logs/regime_label_sweep_window.log).",
          "",
          "**Next step**: re-export v3.4.2 OOT NPZ over the full 40-day window",
          "(per HC #430-A follow-up #3); only then can the sweep configs pass HC #428 R1.",
          ]
    (OUT_DIR / "verdict.md").write_text("\n".join(md))
    _log(f"DONE — wrote {OUT_DIR}/verdict.md + summary_by_trial.csv + per_config_per_day.csv + summary.json")


if __name__ == "__main__":
    main()
