#!/usr/bin/env python3
"""
external_pressure_stream_v1.py — HC #486 follow-up.

Tests stream-coherence on EXTERNAL market-pressure signals (OFI, signed trade
flow, book-OFI, composite) at the 250ms-stride event grid, joining v3.4.2 OOT
predictions to per-event OFI features and v4 alpha labels.

Output: /home/jupiter/Lvl3Quant/output/external_pressure_stream_v1/
  - summary.csv
  - coverage_vs_edge.csv
  - winning_cells.txt
  - .regen_complete.json
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/jupiter/Lvl3Quant")
OFI_DIR = ROOT / "data/processed/mbo_events_smart_v3_ofi_features"
LAB_DIR = ROOT / "data/processed/mbo_events_smart_v3_alpha_labels_v4"
PRED_DIR = ROOT / "output/cnn_mamba_v3_4_2_fixedmtl/oot_47day_perdate"
OUT_DIR = ROOT / "output/external_pressure_stream_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COMMISSION_TICKS_RT = 0.376
STRIDE = 250
OFFSET = 1499  # v4_idx = OFFSET + k * STRIDE

# Pressure signals (per-event in OFI file; we'll subsample at 250ms stride)
# Note: queue_imbalance_current is not in the OFI feature file. Substitute
# ofi_book_5s (5s book-update OFI signed flow) as P3 -- closest proxy to
# microstructure book pressure available without re-running feature gen.
PRESSURE_SIGNALS = [
    ("ofi_aggressive_5s",   "P1_OFI_5s"),
    ("trade_signed_flow_5s","P2_signed_trade_flow_5s"),
    ("ofi_book_5s",         "P3_book_ofi_5s_proxy"),  # substitute for queue_imb
    # P4 composite computed inline below
]
K_VALUES = [4, 20, 40, 80]
HORIZONS = ["1s", "5s", "10s", "30s"]
SIDES = ["long", "short"]
POLICIES = ["forward", "mirror"]

# Deploy gates (HC #428 R1)
GATE_NET = 0.10
GATE_WR = 0.52
GATE_REGIME_IMB = 0.50
GATE_DAYCONC = 0.70


def load_day(date_str: str):
    """Return aligned per-pred-row dict, or None if missing."""
    ofi_path = OFI_DIR / f"{date_str}_ofi.npz"
    lab_path = LAB_DIR / f"{date_str}_alpha_labels.npz"
    pred_path = PRED_DIR / f"oot_{date_str}.npz"
    if not (ofi_path.exists() and pred_path.exists()):
        return None
    ofi = np.load(ofi_path)
    pred = np.load(pred_path)
    n_pred = pred["pred_log_ret_1s"].shape[0]
    v4_idx = OFFSET + np.arange(n_pred) * STRIDE
    n_ofi = ofi["ofi_aggressive_5s"].shape[0]
    if v4_idx[-1] >= n_ofi:
        # truncate
        cap = (n_ofi - OFFSET) // STRIDE
        v4_idx = v4_idx[:cap]
        n_pred = cap

    # Subsample pressure signals
    out = {}
    out["date"] = date_str
    out["n"] = n_pred
    out["P1_OFI_5s"]            = ofi["ofi_aggressive_5s"][v4_idx]
    out["P2_signed_trade_flow_5s"] = ofi["trade_signed_flow_5s"][v4_idx]
    out["P3_book_ofi_5s_proxy"] = ofi["ofi_book_5s"][v4_idx]
    spread = ofi["spread_ticks_now"][v4_idx]
    # P4 composite = sign(OFI_5s)*|OFI_5s|/(1+spread) — strong flow at TIGHT
    # spreads. Equivalent direction to OFI_5s but down-weighted when spread wide.
    p4 = ofi["ofi_aggressive_5s"][v4_idx] / (1.0 + np.maximum(spread, 0.0))
    out["P4_spread_weighted_OFI_5s"] = p4

    # Realized returns in TICKS, hold-to-h policy
    masks = {}
    rets = {}
    for h in HORIZONS:
        rk = f"target_log_ret_{h}"
        mk = f"mask_log_ret_{h}"
        r = pred[rk][:n_pred].astype(np.float64)  # already in ticks (per inspection)
        m = (pred[mk][:n_pred] > 0.5) & np.isfinite(r)
        rets[h] = r
        masks[h] = m
    out["rets"] = rets
    out["masks"] = masks

    # Day regime classification — green / red / flat by 5min log_ret sum
    # Use target_log_ret_5min if available, else cumulative log_ret_30s
    if "target_log_ret_5min" in pred.files:
        lr = pred["target_log_ret_5min"][:n_pred]
        m5 = pred["mask_log_ret_5min"][:n_pred] > 0.5
        valid = m5 & np.isfinite(lr)
        # Sum over day gives total close-to-close return; but log_ret_5min
        # is at every event so sum over-counts. Use mean*duration_factor.
        # Simpler: take first/last valid and use cum log return.
        # Actually best: use the last valid 5min log_ret summed across non-overlapping windows.
        # Even simpler: just check sign of mean(target_log_ret_30s) of day.
        pass
    # Use mean of target_log_ret_30s as day-sign proxy
    r30 = pred["target_log_ret_30s"][:n_pred]
    m30 = (pred["mask_log_ret_30s"][:n_pred] > 0.5) & np.isfinite(r30)
    day_drift = float(np.mean(r30[m30])) if m30.sum() > 0 else 0.0
    # threshold: |drift| < 0.05 ticks/event ~ flat
    if day_drift > 0.05:
        out["regime"] = "green"
    elif day_drift < -0.05:
        out["regime"] = "red"
    else:
        out["regime"] = "flat"
    out["day_drift"] = day_drift
    return out


def stream_coherence(p: np.ndarray, K: int):
    """Return (sign_consistency, mean_abs) for forward window of length K.
    For each index i: looks at p[i+1 .. i+K]. Last K indices: NaN-padded.
    Fully vectorized via cumsum on positive/negative/abs indicators.
    """
    n = len(p)
    sign_p = np.sign(p)
    abs_p = np.abs(p)
    consistency = np.full(n, np.nan, dtype=np.float64)
    mean_abs    = np.full(n, np.nan, dtype=np.float64)

    pos = (sign_p > 0).astype(np.float64)
    neg = (sign_p < 0).astype(np.float64)
    cs_pos = np.concatenate(([0.0], np.cumsum(pos)))
    cs_neg = np.concatenate(([0.0], np.cumsum(neg)))
    cs_abs = np.concatenate(([0.0], np.cumsum(abs_p)))

    last_i = n - K  # process i in [0, n-K)
    if last_i <= 0:
        return consistency, mean_abs
    i_arr = np.arange(last_i)
    a = i_arr + 1
    b = i_arr + K + 1
    pos_match = cs_pos[b] - cs_pos[a]
    neg_match = cs_neg[b] - cs_neg[a]
    abs_sum   = cs_abs[b] - cs_abs[a]
    s = sign_p[:last_i]
    match = np.where(s > 0, pos_match, np.where(s < 0, neg_match, 0.0))
    consistency[:last_i] = match / K
    mean_abs[:last_i]    = abs_sum / K
    return consistency, mean_abs


def evaluate_cell(rows: list, pressure_name: str, K: int, h: str, side: str,
                  policy: str):
    """
    rows: list of per-day dicts each containing:
      'pressure': np.ndarray (n_day,) — values of pressure signal at stride
      'strong_long': np.ndarray (n_day,) bool — pressure-strong AND sign>0
      'strong_short': np.ndarray (n_day,) bool — pressure-strong AND sign<0
      'ret_h_ticks': np.ndarray (n_day,) — target log_ret_h in ticks
      'mask_h': np.ndarray (n_day,) bool
      'date': str
      'regime': str
    """
    daily_means = []
    daily_n = []
    daily_dates = []
    daily_regimes = []

    all_net = []  # for global aggregates

    for d in rows:
        r = d["ret_h_ticks"]
        m = d["mask_h"]
        if side == "long":
            strong = d["strong_long"]
            trade_sign = +1
        else:
            strong = d["strong_short"]
            trade_sign = -1
        if policy == "mirror":
            trade_sign = -trade_sign
        cell = strong & m
        if cell.sum() < 1:
            continue
        net = trade_sign * r[cell] - COMMISSION_TICKS_RT
        daily_means.append(float(np.mean(net)))
        daily_n.append(int(cell.sum()))
        daily_dates.append(d["date"])
        daily_regimes.append(d["regime"])
        all_net.append(net)

    if not all_net:
        return None
    all_net_arr = np.concatenate(all_net)
    n = int(all_net_arr.size)
    if n < 10:
        return None
    net_mean = float(np.mean(all_net_arr))
    wr = float(np.mean(all_net_arr > 0))
    daily = np.array(daily_means)
    daily_counts = np.array(daily_n)
    n_days = len(daily)
    prof_days = int((daily > 0).sum())
    # sharpe per-day
    if daily.std() > 1e-9:
        sharpe = float(daily.mean() / daily.std() * np.sqrt(252))
    else:
        sharpe = 0.0
    # regime stratification
    green_daily = daily[np.array([rg == "green" for rg in daily_regimes])]
    red_daily   = daily[np.array([rg == "red"   for rg in daily_regimes])]
    flat_daily  = daily[np.array([rg == "flat"  for rg in daily_regimes])]
    def safe_sharpe(arr):
        if len(arr) < 2 or arr.std() < 1e-9:
            return 0.0
        return float(arr.mean() / arr.std() * np.sqrt(252))
    sh_green = safe_sharpe(green_daily)
    sh_red   = safe_sharpe(red_daily)
    if max(abs(sh_green), abs(sh_red)) > 1e-9:
        regime_imb = abs(sh_green - sh_red) / max(abs(sh_green), abs(sh_red))
    else:
        regime_imb = 0.0
    day_conc = float(daily_counts.max() / daily_counts.sum()) if daily_counts.sum() > 0 else 1.0

    gate_net = net_mean > GATE_NET
    gate_wr = wr >= GATE_WR
    target_pdays = int(0.65 * n_days)
    gate_pdays = prof_days >= target_pdays
    gate_regime = regime_imb <= GATE_REGIME_IMB
    gate_dayconc = day_conc <= GATE_DAYCONC
    pass_all = bool(gate_net and gate_wr and gate_pdays and gate_regime and gate_dayconc)

    return {
        "pressure_signal": pressure_name,
        "K": K,
        "horizon": h,
        "side": side,
        "policy": policy,
        "n_events": n,
        "n_days_eval": n_days,
        "net_ticks_per_event": net_mean,
        "win_rate": wr,
        "prof_days": prof_days,
        "sharpe_all_days": sharpe,
        "sharpe_green": sh_green,
        "sharpe_red": sh_red,
        "regime_imbalance": regime_imb,
        "day_concentration": day_conc,
        "gate_net": gate_net,
        "gate_wr": gate_wr,
        "gate_pdays": gate_pdays,
        "gate_regime": gate_regime,
        "gate_dayconc": gate_dayconc,
        "pass_all_gates": pass_all,
    }


def main():
    t0 = time.time()
    # List OOT dates
    dates = sorted([p.stem.replace("_ofi", "") for p in OFI_DIR.glob("*_ofi.npz")])
    # Filter to those with predictions
    dates = [d for d in dates if (PRED_DIR / f"oot_{d}.npz").exists()]
    print(f"[info] Evaluating {len(dates)} dates", flush=True)

    # Load all days once
    day_data = []
    for d in dates:
        try:
            dd = load_day(d)
            if dd is not None:
                day_data.append(dd)
        except Exception as e:
            print(f"[skip] {d}: {e}", flush=True)
    print(f"[info] Loaded {len(day_data)} days", flush=True)

    # Iterate cells
    PRESSURE_LIST = [
        ("P1_OFI_5s",                "P1_OFI_5s"),
        ("P2_signed_trade_flow_5s",  "P2_signed_trade_flow_5s"),
        ("P3_book_ofi_5s_proxy",     "P3_book_ofi_5s_proxy"),
        ("P4_spread_weighted_OFI_5s","P4_spread_weighted_OFI_5s"),
    ]

    rows_out = []
    for key, pretty in PRESSURE_LIST:
        for K in K_VALUES:
            # Precompute pressure-strong masks once per (sig, K)
            base = []
            for dd in day_data:
                p = dd[key]
                n_day = len(p)
                if n_day <= K + 1:
                    base.append(None)
                    continue
                consistency, mean_abs = stream_coherence(p, K)
                valid_window = np.isfinite(mean_abs)
                if valid_window.sum() < 100:
                    base.append(None)
                    continue
                thr = np.percentile(mean_abs[valid_window], 75)
                sign_anchor = np.sign(p)
                strong = valid_window & (consistency >= 0.85) & (mean_abs >= thr)
                strong_long  = strong & (sign_anchor > 0)
                strong_short = strong & (sign_anchor < 0)
                base.append({
                    "date": dd["date"],
                    "regime": dd["regime"],
                    "strong_long": strong_long,
                    "strong_short": strong_short,
                })
            for h in HORIZONS:
                rh = []
                for dd, b in zip(day_data, base):
                    if b is None:
                        continue
                    rh.append({
                        "date": b["date"],
                        "regime": b["regime"],
                        "strong_long": b["strong_long"],
                        "strong_short": b["strong_short"],
                        "ret_h_ticks": dd["rets"][h],
                        "mask_h": dd["masks"][h],
                    })
                for side in SIDES:
                    for policy in POLICIES:
                        res = evaluate_cell(rh, pretty, K, h, side, policy)
                        if res is None:
                            continue
                        rows_out.append(res)
                        print(f"  [{pretty} K={K} h={h} {side}/{policy}] "
                              f"n={res['n_events']} net={res['net_ticks_per_event']:.3f} "
                              f"wr={res['win_rate']:.3f} pass={res['pass_all_gates']}",
                              flush=True)

    df = pd.DataFrame(rows_out)
    df.to_csv(OUT_DIR / "summary.csv", index=False)

    # Winning cells
    wins = df[df["pass_all_gates"]].copy()
    with open(OUT_DIR / "winning_cells.txt", "w") as f:
        if len(wins) == 0:
            f.write("NO CELLS PASS ALL GATES.\n")
        else:
            for _, r in wins.sort_values("net_ticks_per_event", ascending=False).iterrows():
                f.write(
                    f"{r['pressure_signal']} K={r['K']} h={r['horizon']} "
                    f"{r['side']}/{r['policy']}  net={r['net_ticks_per_event']:.3f} "
                    f"wr={r['win_rate']:.3f}  prof_days={r['prof_days']}/{r['n_days_eval']} "
                    f"sharpe={r['sharpe_all_days']:.2f} regimeIMB={r['regime_imbalance']:.2f} "
                    f"dayConc={r['day_concentration']:.2f} n={r['n_events']}\n"
                )

    # Coverage vs edge — per (signal, K) pick best (h, side, policy) on net_ticks
    cov_rows = []
    for (sig, K), g in df.groupby(["pressure_signal", "K"]):
        # Mean events per day across all (h,side,policy) for same (sig,K) is roughly
        # same since pressure-strong selection doesn't depend on h. Take forward+long
        # at h=10s as representative.
        rep = g[(g["side"] == "long") & (g["policy"] == "forward") &
                (g["horizon"] == "10s")]
        if len(rep) == 0:
            continue
        rep = rep.iloc[0]
        # Total denominator = sum of n across all days for that day's pressure stream
        cov_rows.append({
            "pressure_signal": sig,
            "K": K,
            "n_events_rep": int(rep["n_events"]),
            "best_net": float(g["net_ticks_per_event"].max()),
            "best_cell": (g.sort_values("net_ticks_per_event", ascending=False)
                          .iloc[0][["horizon", "side", "policy"]].to_dict()),
        })
    cov_df = pd.DataFrame(cov_rows)
    cov_df.to_csv(OUT_DIR / "coverage_vs_edge.csv", index=False)

    # Anchor rows appended to summary
    anchors = []
    anchors.append({
        "pressure_signal": "ANCHOR_morning_OFI_standalone_best",
        "K": None, "horizon": "1s", "side": "long", "policy": "tp2sl2_top20",
        "n_events": 315560, "net_ticks_per_event": -0.518,
        "win_rate": 0.414, "prof_days": 0, "n_days_eval": 32,
        "pass_all_gates": False,
        "notes": "ofi_aggressive_10s, 1s long, top20pct, tp2/sl2",
    })
    anchors.append({
        "pressure_signal": "ANCHOR_snapshot_baseline_5s_short_top1pct",
        "K": None, "horizon": "5s", "side": "short", "policy": "snapshot_top1",
        "net_ticks_per_event": 0.008,
        "pass_all_gates": False,
        "notes": "morning snapshot baseline closest-miss",
    })
    anchors.append({
        "pressure_signal": "ANCHOR_model_stream_K4_10s_short",
        "K": 4, "horizon": "10s", "side": "short", "policy": "forward",
        "net_ticks_per_event": -0.169,
        "pass_all_gates": False,
        "notes": "Step 1 closest-miss on model-prediction stream coherence",
    })
    with open(OUT_DIR / "anchors.json", "w") as f:
        json.dump(anchors, f, indent=2, default=str)

    # Done marker
    elapsed = time.time() - t0
    with open(OUT_DIR / ".regen_complete.json", "w") as f:
        json.dump({
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_days_evaluated": len(day_data),
            "n_cells": len(rows_out),
            "n_winning_cells": int(wins.shape[0]),
            "elapsed_sec": round(elapsed, 1),
            "script": "scripts/external_pressure_stream_v1.py",
        }, f, indent=2)
    print(f"[done] {elapsed:.1f}s, days={len(day_data)} cells={len(rows_out)} wins={len(wins)}",
          flush=True)


if __name__ == "__main__":
    main()
