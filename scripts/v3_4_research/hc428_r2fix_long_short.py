#!/usr/bin/env python3
"""HC #428 R2-fix + LONG-side exploration.

Sub-objectives:
A. Build R2-compliant variants of t1422 (5s) and t2831 (10s) by clipping
   cancel_window <= horizon (in 250 ms eval ticks: 5s=20, 10s=40) and
   hold_seconds <= 1.5 * horizon. Re-validate per-day Sharpe + regime ratio.
B. Diagnose t1554 R1 over-attribution: % P&L from 2 RED days, Sharpe ex-2.
C. Find LONG-side counterparts at 5s/10s, R2-clip them up front, identify
   any LONG+SHORT 50/50 ensemble that meets both R1<=0.50 and R2 gates.

Reuses canonical full_market_replay primitives (read-only).
Writes 3 markdown reports + summary JSON. Read-only on data dirs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(ROOT))

from scripts.v3_3_research.full_market_replay import (  # noqa: E402
    _load_fifo_labels,
    _queue_position_model,
    _adverse_selection,
    _entry_price_edge_ticks,
    _mfe_mae_per_fill,
    _pick_exit_horizon,
    ES_RT_COMMISSION_TICKS_DEFAULT,
    EVAL_STRIDE_SEC,
    PRICE_UNIT_TO_TICKS,
)

OUT = ROOT / "output"
LABELS_DIR = ROOT / "data/processed/mbo_events_smart_v3_fifo_labels"

# Date sets per the existing NPZs
V342_5D = ["20260223", "20260224", "20260225", "20260226", "20260227"]
V342_12D = ["20260301", "20260302", "20260303", "20260304", "20260305",
            "20260306", "20260309", "20260310", "20260311", "20260312",
            "20260313", "20260315"]
V342_17D = V342_5D + V342_12D
V33_5D = V342_5D

# Regime classification
REGIME_CSV = OUT / "hc428_regime_classification_oot.csv"
_reg_df = pd.read_csv(REGIME_CSV, dtype={"date": str})
REGIME = dict(zip(_reg_df["date"], _reg_df["regime"]))

# Canonical config dicts ------------------------------------------------------
T1422 = {
    "trial": 1422, "head_horizon": "5s", "side": "short",
    "conf_thr": 0.0829977785764972,
    "order_type": "passive_at_touch_plus_2",
    "cancel_window": 80, "hold_seconds": 1.0952866568486876,
    "spread_ticks": 0.693391384134378,
    "tod_start_hour": 13, "tod_end_hour": 15,
    "pred_strength_min": 0.8035401709017903,
    "sigma_halt_mult": 9.392375633554254,
    "commission_ticks": 0.3374439484547556,
    "use_fifo_confluence": False,
    "fifo_confluence_head": "pred_fifo_tp8sl5_net",
    "fifo_confluence_thr_ticks": 1.6129959066597321,
    "use_horizon_confluence": True, "confluence_horizon": "5s",
    "source": "v3.4.2",
}
T1554 = {
    "trial": 1554, "head_horizon": "30s", "side": "short",
    "conf_thr": 0.08574633671758808,
    "order_type": "passive_at_touch_plus_2",
    "cancel_window": 38, "hold_seconds": 2.1436026730594473,
    "spread_ticks": 0.799290577985101,
    "tod_start_hour": 14, "tod_end_hour": 15,
    "pred_strength_min": 0.09517319127280299,
    "sigma_halt_mult": 6.218254410325431,
    "commission_ticks": 0.34712034949404064,
    "use_fifo_confluence": False,
    "fifo_confluence_head": "pred_fifo_tp8sl5_net",
    "fifo_confluence_thr_ticks": 1.1979357578266216,
    "use_horizon_confluence": True, "confluence_horizon": "10s",
    "source": "v3.4.2",
}
T2831 = {
    "trial": 2831, "head_horizon": "10s", "side": "short",
    "conf_thr": 0.06209578541158193,
    "order_type": "passive_at_touch_plus_2",
    "cancel_window": 57, "hold_seconds": 2.3681128869797683,
    "spread_ticks": 1.5494016518812743,
    "tod_start_hour": 13, "tod_end_hour": 15,
    "pred_strength_min": 0.6355315666345525,
    "sigma_halt_mult": 6.592435934064023,
    "commission_ticks": 0.4030989394691603,
    "use_fifo_confluence": True,
    "fifo_confluence_head": "pred_fifo_tp8sl5_net",
    "fifo_confluence_thr_ticks": -0.8831534684536466,
    "use_horizon_confluence": True, "confluence_horizon": "10s",
    "source": "v3.3",
}

# --- Loaders --------------------------------------------------------------
def load_v342_17d():
    """Load 17 days of v3.4.2 predictions by concatenating 5d + 12d."""
    keep_keys = None
    parts = {}
    n_per_day = []
    date_arr = []
    sources = [
        (OUT / "cnn_mamba_v3_4_2_fixedmtl/fold_00_predictions.npz", V342_5D),
        (OUT / "v342_fold_00_ep1_oot_inference_extended_wrapped.npz", V342_12D),
    ]
    # First pass: load and split per date using fifo labels n_per_day
    for npz_path, dates in sources:
        z = np.load(npz_path, allow_pickle=True)
        # Get per-date n from fifo labels (predictions NPZ is per-sample, aligned
        # to fifo labels)
        per_date_n = []
        for dt in dates:
            lz = np.load(LABELS_DIR / f"{dt}_fifo_labels.npz", allow_pickle=False)
            per_date_n.append(int(lz["window_k"].shape[0]))
        n_npz = z["pred_log_ret_5s"].shape[0]
        # Predictions NPZ might NOT match exactly; truncate to min
        n_total_dates = sum(per_date_n)
        # If pred_npz has fewer than expected, we keep min and adjust
        if n_npz < n_total_dates:
            # scale per_date proportionally is wrong; instead, accept truncation
            # at the end (last day clipped)
            remaining = n_npz
            new_per = []
            for nd in per_date_n:
                take = min(nd, remaining)
                new_per.append(take)
                remaining -= take
                if remaining <= 0:
                    break
            # any unfilled days get 0
            while len(new_per) < len(per_date_n):
                new_per.append(0)
            per_date_n = new_per
        n_per_day.extend(per_date_n)
        for i, dt in enumerate(dates):
            date_arr.extend([dt] * per_date_n[i])
        # Collect needed arrays
        if keep_keys is None:
            keep_keys = [k for k in z.keys() if k.startswith("pred_log_ret_")
                         or k.startswith("mask_log_ret_")
                         or k.startswith("target_log_ret_")
                         or k.startswith("mask_p_") or k.startswith("pred_p_")]
        for k in keep_keys:
            if k in z.keys():
                arr = z[k][:sum(per_date_n)]
                parts.setdefault(k, []).append(arr)
    out = {k: np.concatenate(v) for k, v in parts.items()}
    out["_dates"] = np.array(date_arr)
    out["_n_per_day"] = n_per_day
    return out


def load_v33_5d():
    z = np.load(OUT / "cnn_mamba_v3_3_uncertainty_weighted/fold_00_predictions.npz",
                allow_pickle=True)
    date_arr = []
    for dt in V33_5D:
        lz = np.load(LABELS_DIR / f"{dt}_fifo_labels.npz", allow_pickle=False)
        date_arr.extend([dt] * int(lz["window_k"].shape[0]))
    out = {k: z[k] for k in z.keys()
           if k.startswith("pred_log_ret_") or k.startswith("mask_log_ret_")
           or k.startswith("target_log_ret_")}
    n = z["pred_log_ret_5s"].shape[0]
    out["_dates"] = np.array(date_arr[:n])
    return out


# --- Runner: replicate full_market_replay logic but per-day -----------------
def run_replay_per_day(preds_dict, dates_list, cfg):
    """Run replay using only the dates_list (subset). Returns per-trade DF."""
    horizon = cfg["head_horizon"]
    side = cfg["side"]
    side_sign = 1.0 if side == "long" else -1.0
    pred_key = f"pred_log_ret_{horizon}"
    mask_key = f"mask_log_ret_{horizon}"
    pred = preds_dict[pred_key]
    mask = preds_dict[mask_key].astype(bool) & np.isfinite(pred)
    sample_dates = preds_dict["_dates"]
    n = pred.shape[0]
    # Restrict to dates_list
    date_keep = np.isin(sample_dates[:n], dates_list)

    # Load fifo labels for these dates (concat in same order as sample_dates)
    # We can build them from per-date load
    fifo_parts = {}
    fifo_date_arr = []
    for dt in dates_list:
        lz = np.load(LABELS_DIR / f"{dt}_fifo_labels.npz", allow_pickle=False)
        for k in lz.keys():
            fifo_parts.setdefault(k, []).append(lz[k])
        fifo_date_arr.extend([dt] * int(lz["window_k"].shape[0]))
    fifo = {k: np.concatenate(v) for k, v in fifo_parts.items()}
    fifo_dates = np.array(fifo_date_arr)
    # Align: take first min(n, fifo_n) and ensure dates match by trimming the longer
    # Actually our preds and fifo loaders both run sample_dates per fifo n_per_day,
    # so they should match. Confirm:
    n_fifo = len(fifo_date_arr)
    # Trim pred to same dates as fifo by matching consecutively per date
    use_n = min(n, n_fifo)
    pred = pred[:use_n]
    mask = mask[:use_n]
    sample_dates_used = sample_dates[:use_n]
    for k in fifo_parts:
        fifo[k] = fifo[k][:use_n]
    # Now both arrays are aligned IF the order of dates matches. Our preds_dict
    # for v3.4.2 17d uses 5d then 12d; our fifo here uses dates_list as passed.
    # To keep alignment, dates_list MUST equal sample_dates ordering. We accept
    # dates_list==V342_17D (or V33_5D) usage from caller.

    # Apply date restriction (only keep samples whose date is in dates_list)
    keep = np.isin(sample_dates_used, dates_list) & mask
    # Use only valid preds among the kept set for percentile threshold
    p_valid = pred[keep]
    if p_valid.size == 0:
        return None
    conf = cfg["conf_thr"]
    if side == "long":
        thr = float(np.quantile(p_valid, 1.0 - conf))
        sel = keep & (pred >= thr)
    else:
        thr = float(np.quantile(p_valid, conf))
        sel = keep & (pred <= thr)
    sel_idx = np.where(sel)[0]
    n_signals = int(sel.sum())
    if n_signals == 0:
        return None

    side_key = side
    filled_lbl = fifo[f"tp4sl3_{side_key}_filled"][sel_idx]
    exit_reason_lbl = fifo[f"tp4sl3_{side_key}_exit_reason"][sel_idx]
    hold_time_lbl = fifo[f"tp4sl3_{side_key}_hold_time_ns"][sel_idx]

    filled_mask, q_arrival, _ = _queue_position_model(
        cfg["order_type"], cfg["cancel_window"],
        filled_lbl, exit_reason_lbl, hold_time_lbl,
    )
    filled_idx_in_sel = np.where(filled_mask)[0]
    filled_global_idx = sel_idx[filled_idx_in_sel]
    if filled_global_idx.size == 0:
        return None

    # Build tgt_lr dict for _pick_exit_horizon path
    horizon_choice = _pick_exit_horizon(cfg["hold_seconds"])
    tgt_key = f"target_log_ret_{horizon_choice}"
    tgt_mask_key = f"mask_log_ret_{horizon_choice}"
    if tgt_key not in preds_dict:
        # fall back to horizon
        tgt_key = f"target_log_ret_{horizon}"
        tgt_mask_key = f"mask_log_ret_{horizon}"
    lr_exit = preds_dict[tgt_key][:use_n][filled_global_idx]
    lr_mask_exit = preds_dict[tgt_mask_key][:use_n][filled_global_idx].astype(bool) & np.isfinite(lr_exit)

    edge_offset = _entry_price_edge_ticks(cfg["order_type"], 1.0)
    net = side_sign * lr_exit * PRICE_UNIT_TO_TICKS + edge_offset - ES_RT_COMMISSION_TICKS_DEFAULT
    net = np.where(lr_mask_exit, net, 0.0)

    fill_dates = sample_dates_used[filled_global_idx]
    df = pd.DataFrame({"date": fill_dates, "net_ticks": net})
    return df


def metrics_from_df(df):
    """Return dict with overall + per-regime sharpe + day_conc."""
    if df is None or len(df) == 0:
        return {"n_trades": 0, "sharpe_overall": 0.0, "sharpe_green": 0.0,
                "sharpe_red": 0.0, "sharpe_flat": 0.0,
                "day_conc": 1.0, "total_net": 0.0,
                "r1_ratio": 1.0, "per_day": {}}
    # Per-day aggregate first then sharpe of daily means
    df = df.copy()
    df["regime"] = df["date"].map(REGIME).fillna("UNK")
    per_day = df.groupby("date")["net_ticks"].agg(["count", "sum", "mean", "std"])
    total = float(df["net_ticks"].sum())
    day_conc = float(per_day["sum"].abs().max() / max(1e-9, per_day["sum"].abs().sum()))
    # HC #428 convention (matches hc428_summary.json):
    #   overall = daily-aggregated Sharpe * sqrt(252)
    #   per-regime = trade-level Sharpe * sqrt(252) within regime trades
    def daily_sharpe(sub):
        if len(sub) == 0: return 0.0
        daily = sub.groupby("date")["net_ticks"].sum().values
        if daily.size < 2: return 0.0
        mu = daily.mean(); sd = daily.std(ddof=1)
        if sd <= 1e-9: return 0.0
        return float(mu / sd * np.sqrt(252))
    def trade_sharpe_regime(arr):
        if arr.size < 5: return 0.0
        mu = arr.mean(); sd = arr.std(ddof=1)
        if sd <= 1e-9: return 0.0
        return float(mu / sd * np.sqrt(252))
    overall = daily_sharpe(df)
    s_green = trade_sharpe_regime(df.loc[df.regime == "GREEN", "net_ticks"].values)
    s_red = trade_sharpe_regime(df.loc[df.regime == "RED", "net_ticks"].values)
    s_flat = trade_sharpe_regime(df.loc[df.regime == "FLAT", "net_ticks"].values)
    mx = max(abs(s_green), abs(s_red), abs(s_flat), 1e-9)
    r1 = abs(s_green - s_red) / max(abs(s_green), abs(s_red), 1e-9) if s_green and s_red else 1.0
    return {
        "n_trades": int(len(df)),
        "total_net": total,
        "sharpe_overall": overall,
        "sharpe_green": s_green, "sharpe_red": s_red, "sharpe_flat": s_flat,
        "day_conc": day_conc,
        "r1_ratio": r1,
        "per_day": per_day.to_dict(),
    }


# --- R2 helpers --------------------------------------------------------------
def r2_clip(cfg):
    """Apply HC #428 R2 clips: cancel_eval (250 ms) <= horizon, hold <= 1.5*horizon."""
    horizon_sec = float(cfg["head_horizon"].rstrip("s"))
    max_cancel_evals = int(horizon_sec / EVAL_STRIDE_SEC)  # e.g. 5s -> 20
    new = dict(cfg)
    if cfg["cancel_window"] > max_cancel_evals:
        new["cancel_window"] = max_cancel_evals
    max_hold = 1.5 * horizon_sec
    if cfg["hold_seconds"] > max_hold:
        new["hold_seconds"] = max_hold
    return new


def r2_pass(cfg, mfe_p90_sec=None):
    horizon_sec = float(cfg["head_horizon"].rstrip("s"))
    cancel_sec = cfg["cancel_window"] * EVAL_STRIDE_SEC
    limit = mfe_p90_sec if mfe_p90_sec is not None else horizon_sec
    return cancel_sec <= limit + 1e-6 and cfg["hold_seconds"] <= 1.5 * horizon_sec + 1e-6


# ============================================================================
# SUB-OBJECTIVE A: R2-fix t1422 + t2831
# ============================================================================
def run_subobj_a(preds_v342, preds_v33):
    print("\n=== SUB-OBJECTIVE A: R2-fix t1422 + t2831 ===")
    out = {}

    # Per HC #428 problem statement: t1422 cancel clip = 5s (p90 MFE@5s = 5s, horizon 5s);
    # t2831 cancel clip = 6s (p90 MFE@10s = 6s).
    # That's stricter than the generic horizon clip — apply the user's specified clip.
    t1422_fix = r2_clip(T1422)
    # User-specified: cancel_window in evals for 5s = 20
    t1422_fix["cancel_window"] = 20
    t1422_fix["hold_seconds"] = min(T1422["hold_seconds"], 7.5)

    t2831_fix = r2_clip(T2831)
    # p90 MFE @ 10s = 6s → cancel_window = 24 evals
    t2831_fix["cancel_window"] = 24
    t2831_fix["hold_seconds"] = min(T2831["hold_seconds"], 15.0)

    for label, cfg, preds, dates in [
        ("t1422_R2fix", t1422_fix, preds_v342, V342_17D),
        ("t2831_R2fix", t2831_fix, preds_v33, V33_5D),
    ]:
        df = run_replay_per_day(preds, dates, cfg)
        m = metrics_from_df(df)
        m["config"] = cfg
        m["cancel_sec"] = cfg["cancel_window"] * EVAL_STRIDE_SEC
        m["r2_pass"] = r2_pass(cfg)
        m["r1_pass"] = (m["r1_ratio"] <= 0.50)
        # Day breakdown
        if df is not None and len(df):
            per_day = df.groupby("date")["net_ticks"].agg(["count", "sum", "mean"])
            per_day = per_day.reset_index()
            per_day["regime"] = per_day["date"].map(REGIME)
            m["per_day_df"] = per_day.to_dict(orient="records")
        else:
            m["per_day_df"] = []
        out[label] = m
        print(f"  {label}: n={m['n_trades']} sharpe_ovr={m['sharpe_overall']:.2f} "
              f"sharpe_g={m['sharpe_green']:.2f} sharpe_r={m['sharpe_red']:.2f} "
              f"r1_ratio={m['r1_ratio']:.3f} R1={'PASS' if m['r1_pass'] else 'FAIL'} "
              f"R2={'PASS' if m['r2_pass'] else 'FAIL'}")

    # Write markdown
    md = ["# HC #428 R2-FIX Re-Validation", "",
          f"_Generated: 2026-05-19_", "",
          "## Summary",
          "",
          "| Config | n_trades | Sharpe_ovr | Sharpe_green | Sharpe_red | Sharpe_flat | day_conc | R1 ratio | R1 | R2 | cancel(s) | hold(s) |",
          "|--------|---------:|-----------:|-------------:|-----------:|------------:|---------:|---------:|:--:|:--:|----------:|--------:|",
          ]
    for label, m in out.items():
        cfg = m["config"]
        md.append(
            f"| {label} | {m['n_trades']} | {m['sharpe_overall']:.2f} | "
            f"{m['sharpe_green']:.2f} | {m['sharpe_red']:.2f} | "
            f"{m['sharpe_flat']:.2f} | {m['day_conc']:.3f} | {m['r1_ratio']:.3f} | "
            f"{'PASS' if m['r1_pass'] else 'FAIL'} | "
            f"{'PASS' if m['r2_pass'] else 'FAIL'} | "
            f"{m['cancel_sec']:.1f} | {cfg['hold_seconds']:.2f} |"
        )
    md.append("")
    md.append("## R2-Fix Parameter Changes")
    md.append("")
    md.append("| Config | cancel_window evals (orig→fix) | cancel_sec (orig→fix) | hold_seconds (orig→fix) |")
    md.append("|--------|-------------------------------:|----------------------:|-------------------------:|")
    md.append(f"| t1422 | 80 → {t1422_fix['cancel_window']} | 20.0 → "
              f"{t1422_fix['cancel_window']*EVAL_STRIDE_SEC:.1f} | "
              f"{T1422['hold_seconds']:.2f} → {t1422_fix['hold_seconds']:.2f} |")
    md.append(f"| t2831 | 57 → {t2831_fix['cancel_window']} | 14.25 → "
              f"{t2831_fix['cancel_window']*EVAL_STRIDE_SEC:.1f} | "
              f"{T2831['hold_seconds']:.2f} → {t2831_fix['hold_seconds']:.2f} |")
    md.append("")
    for label, m in out.items():
        md.append(f"## {label}")
        md.append("")
        md.append(f"- cancel_window: {m['config']['cancel_window']} evals "
                  f"({m['cancel_sec']:.1f}s)")
        md.append(f"- hold_seconds: {m['config']['hold_seconds']:.2f}")
        md.append(f"- n_trades: {m['n_trades']}, total_net_ticks: {m['total_net']:.1f}")
        md.append(f"- Sharpe overall: {m['sharpe_overall']:.2f}")
        md.append(f"- Per regime: GREEN {m['sharpe_green']:.2f} | RED "
                  f"{m['sharpe_red']:.2f} | FLAT {m['sharpe_flat']:.2f}")
        md.append(f"- day_conc: {m['day_conc']:.3f}")
        md.append(f"- R1 ratio: {m['r1_ratio']:.3f} → {'PASS' if m['r1_pass'] else 'FAIL'}")
        md.append(f"- R2: {'PASS' if m['r2_pass'] else 'FAIL'}")
        if m["per_day_df"]:
            md.append("")
            md.append("Per-day breakdown:")
            md.append("")
            md.append("| date | regime | n | net_ticks | mean |")
            md.append("|------|:------:|--:|----------:|-----:|")
            for r in m["per_day_df"]:
                md.append(f"| {r['date']} | {r.get('regime','?')} | "
                          f"{r['count']} | {r['sum']:.1f} | {r['mean']:.3f} |")
        md.append("")

    (OUT / "hc428_r2fix_revalidate.md").write_text("\n".join(md))
    print(f"  → wrote {OUT / 'hc428_r2fix_revalidate.md'}")
    return out


# ============================================================================
# SUB-OBJECTIVE B: t1554 attribution
# ============================================================================
def run_subobj_b(preds_v342):
    print("\n=== SUB-OBJECTIVE B: t1554 attribution ===")
    df = run_replay_per_day(preds_v342, V342_17D, T1554)
    if df is None or len(df) == 0:
        print("  no trades — abort")
        return None
    per_day = df.groupby("date")["net_ticks"].agg(["count", "sum", "mean"]).reset_index()
    per_day["regime"] = per_day["date"].map(REGIME)
    total = float(per_day["sum"].sum())
    red2 = ["20260223", "20260226"]
    red2_pnl = float(per_day[per_day["date"].isin(red2)]["sum"].sum())
    pct_red2 = red2_pnl / max(1e-9, total) * 100.0
    excl = df[~df["date"].isin(red2)].copy()
    def daily_sharpe(sub):
        if len(sub) == 0: return 0.0
        daily = sub.groupby("date")["net_ticks"].sum().values
        if daily.size < 2: return 0.0
        mu = daily.mean(); sd = daily.std(ddof=1)
        if sd <= 1e-9: return 0.0
        return float(mu/sd * np.sqrt(252))
    s_full = daily_sharpe(df)
    s_excl = daily_sharpe(excl)
    verdict = ("REJECT — regime-tailored (excluding 2 RED days drops Sharpe <1.0)"
               if s_excl < 1.0 else
               ("KEEP & FLAG (Sharpe ex-RED still >2.0)" if s_excl >= 2.0
                else "MARGINAL — re-validate after gap-fill"))
    md = ["# HC #428 t1554 R1 Over-Attribution Diagnosis", "",
          f"_Generated: 2026-05-19_", "",
          f"**Total net ticks (17d)**: {total:.1f}",
          f"**P&L from 2 RED days (0223+0226)**: {red2_pnl:.1f} "
          f"({pct_red2:.1f}% of total)",
          f"**Trade Sharpe full 17d**: {s_full:.2f}",
          f"**Trade Sharpe excluding 0223+0226**: {s_excl:.2f}",
          "",
          f"**VERDICT: {verdict}**",
          "",
          "## Per-day P&L",
          "",
          "| date | regime | n | net_ticks | mean |",
          "|------|:------:|--:|----------:|-----:|"]
    for _, r in per_day.iterrows():
        md.append(f"| {r['date']} | {r['regime']} | {int(r['count'])} | "
                  f"{r['sum']:.1f} | {r['mean']:.3f} |")
    (OUT / "hc428_t1554_attribution.md").write_text("\n".join(md))
    print(f"  → wrote {OUT / 'hc428_t1554_attribution.md'}")
    print(f"  Total: {total:.1f}, RED2: {red2_pnl:.1f} ({pct_red2:.1f}%), "
          f"Sharpe full={s_full:.2f}, ex={s_excl:.2f} → {verdict}")
    return {
        "total": total, "red2_pnl": red2_pnl, "pct_red2": pct_red2,
        "sharpe_full": s_full, "sharpe_excl": s_excl, "verdict": verdict,
    }


# ============================================================================
# SUB-OBJECTIVE C: LONG-side counterparts + LONG+SHORT ensemble
# ============================================================================
def load_leaderboard():
    lb = pd.read_csv(OUT / "v342_execution_optuna_20260518/leaderboard.csv")
    lb = lb[lb["state"] == "COMPLETE"].copy()
    return lb


def cfg_from_lb_row(row):
    return {
        "trial": int(row["number"]),
        "head_horizon": row["params_head_horizon"],
        "side": row["params_side"],
        "conf_thr": float(row["params_conf_thr"]),
        "order_type": row["params_order_type"],
        "cancel_window": int(row["params_cancel_window"]),
        "hold_seconds": float(row["params_hold_seconds"]),
        "spread_ticks": float(row["params_spread_ticks"]),
        "tod_start_hour": int(row["params_tod_start_hour"]),
        "tod_end_hour": int(row["params_tod_end_hour"]),
        "pred_strength_min": float(row["params_pred_strength_min"]),
        "sigma_halt_mult": float(row["params_sigma_halt_mult"]),
        "commission_ticks": float(row["params_commission_ticks"]),
        "use_fifo_confluence": bool(row["params_use_fifo_confluence"]),
        "fifo_confluence_head": row["params_fifo_confluence_head"],
        "fifo_confluence_thr_ticks": float(row["params_fifo_confluence_thr_ticks"]),
        "use_horizon_confluence": bool(row["params_use_horizon_confluence"]),
        "confluence_horizon": row["params_confluence_horizon"],
        "source": "v3.4.2",
    }


def run_subobj_c(preds_v342, short_results):
    print("\n=== SUB-OBJECTIVE C: LONG-side exploration ===")
    lb = load_leaderboard()
    # Filter to long side, horizons 5s or 10s, sharpe>5, n_fills>=20
    cand = lb[(lb["params_side"] == "long")
              & (lb["params_head_horizon"].isin(["5s", "10s"]))
              & (lb["user_attrs_final_sharpe"] > 5.0)
              & (lb["user_attrs_final_n_fills"] >= 20)
              ].sort_values("user_attrs_final_sharpe", ascending=False)
    print(f"  candidates after filter: {len(cand)}")
    # Take top ~30 then R2-clip and re-validate
    top = cand.head(30)
    results = []
    short_fix_df_by_label = {}
    # Pre-fetch short R2-fix per-trade df for ensemble
    for label, cfg in [("t1422_R2fix", short_results["t1422_R2fix"]["config"])]:
        short_fix_df_by_label[label] = run_replay_per_day(preds_v342, V342_17D, cfg)

    for _, row in top.iterrows():
        cfg = cfg_from_lb_row(row)
        cfg_clipped = r2_clip(cfg)
        df = run_replay_per_day(preds_v342, V342_17D, cfg_clipped)
        m = metrics_from_df(df)
        m["trial"] = cfg["trial"]
        m["horizon"] = cfg["head_horizon"]
        m["original_sharpe"] = float(row["user_attrs_final_sharpe"])
        m["r1_pass"] = m["r1_ratio"] <= 0.50
        m["r2_pass"] = r2_pass(cfg_clipped)
        m["config"] = cfg_clipped
        m["df"] = df
        results.append(m)

    # Rank LONG by combined (r1_pass + sharpe_overall)
    results.sort(key=lambda r: (r["r1_pass"] and r["r2_pass"], r["sharpe_overall"]),
                 reverse=True)
    top_long = results[:10]
    print(f"  top long results computed: {len(top_long)}")

    # Try ensembles: for each LONG 5s candidate, pair with t1422_R2fix
    ensemble_results = []
    short_df = short_fix_df_by_label["t1422_R2fix"]
    for r in top_long:
        if r["horizon"] != "5s":
            continue
        if r["df"] is None or short_df is None:
            continue
        # 50/50 capital split = concat trades (each trade = 0.5 unit)
        long_df = r["df"].copy()
        long_df["net_ticks"] = long_df["net_ticks"] * 0.5
        s_df = short_df.copy()
        s_df["net_ticks"] = s_df["net_ticks"] * 0.5
        combined = pd.concat([long_df, s_df], ignore_index=True)
        cm = metrics_from_df(combined)
        cm["long_trial"] = r["trial"]
        cm["short"] = "t1422_R2fix"
        cm["r1_pass"] = cm["r1_ratio"] <= 0.50
        cm["r2_pass"] = True  # both legs R2-clipped
        ensemble_results.append(cm)
    ensemble_results.sort(key=lambda c: (c["r1_pass"], c["sharpe_overall"]), reverse=True)

    md = ["# HC #428 LONG-Side Counterparts + LONG+SHORT Ensemble Candidates", "",
          f"_Generated: 2026-05-19_", "",
          "## Top LONG-side candidates (R2-clipped up front)",
          "",
          "| trial | horizon | n_trades | Sharpe_ovr | Sharpe_g | Sharpe_r | day_conc | R1 ratio | R1 | R2 | original_optuna_sharpe |",
          "|------:|--------:|---------:|-----------:|---------:|---------:|---------:|---------:|:--:|:--:|-----------------------:|"]
    for r in top_long:
        md.append(f"| {r['trial']} | {r['horizon']} | {r['n_trades']} | "
                  f"{r['sharpe_overall']:.2f} | {r['sharpe_green']:.2f} | "
                  f"{r['sharpe_red']:.2f} | {r['day_conc']:.3f} | "
                  f"{r['r1_ratio']:.3f} | "
                  f"{'PASS' if r['r1_pass'] else 'FAIL'} | "
                  f"{'PASS' if r['r2_pass'] else 'FAIL'} | "
                  f"{r['original_sharpe']:.2f} |")
    md.append("")
    md.append("## LONG+SHORT 50/50 Ensemble Candidates (long X + t1422_R2fix)")
    md.append("")
    md.append("| long_trial | n_trades | Sharpe_ovr | Sharpe_g | Sharpe_r | day_conc | R1 ratio | R1 | R2 |")
    md.append("|-----------:|---------:|-----------:|---------:|---------:|---------:|---------:|:--:|:--:|")
    for c in ensemble_results[:10]:
        md.append(f"| {c['long_trial']} | {c['n_trades']} | {c['sharpe_overall']:.2f} | "
                  f"{c['sharpe_green']:.2f} | {c['sharpe_red']:.2f} | "
                  f"{c['day_conc']:.3f} | {c['r1_ratio']:.3f} | "
                  f"{'PASS' if c['r1_pass'] else 'FAIL'} | "
                  f"{'PASS' if c['r2_pass'] else 'FAIL'} |")
    md.append("")
    md.append("## Configs (top 3 long)")
    md.append("")
    for r in top_long[:3]:
        md.append(f"### Trial {r['trial']} ({r['horizon']} long)")
        md.append("```json")
        md.append(json.dumps({k: v for k, v in r["config"].items() if not k.startswith("_")},
                             indent=2, default=str))
        md.append("```")
        md.append("")

    (OUT / "hc428_long_short_ensemble_candidates.md").write_text("\n".join(md))
    print(f"  → wrote {OUT / 'hc428_long_short_ensemble_candidates.md'}")

    return {"top_long": [{k: v for k, v in r.items() if k != "df"}
                          for r in top_long],
            "ensemble": ensemble_results[:10]}


# ============================================================================
def main():
    print("Loading predictions NPZs...")
    preds_v342 = load_v342_17d()
    preds_v33 = load_v33_5d()
    print(f"  v3.4.2 17d: n={preds_v342['pred_log_ret_5s'].shape[0]}")
    print(f"  v3.3   5d : n={preds_v33['pred_log_ret_5s'].shape[0]}")

    a = run_subobj_a(preds_v342, preds_v33)
    b = run_subobj_b(preds_v342)
    c = run_subobj_c(preds_v342, a)

    summary = {
        "subobj_a": {k: {kk: vv for kk, vv in v.items()
                          if kk not in ("per_day_df", "per_day", "config")}
                      for k, v in a.items()},
        "subobj_b": b,
        "subobj_c": {"top_long_summary": [
            {"trial": r["trial"], "horizon": r["horizon"],
             "sharpe_overall": r["sharpe_overall"],
             "r1_pass": r["r1_pass"], "r2_pass": r["r2_pass"],
             "r1_ratio": r["r1_ratio"]}
            for r in c["top_long"][:5]],
            "ensemble_top3": [
                {"long_trial": e["long_trial"],
                 "sharpe_overall": e["sharpe_overall"],
                 "r1_pass": e["r1_pass"], "r2_pass": e["r2_pass"],
                 "r1_ratio": e["r1_ratio"]}
                for e in c["ensemble"][:3]],
        },
    }
    (OUT / "hc428_r2fix_long_short_summary.json").write_text(
        json.dumps(summary, indent=2, default=str))
    print(f"\n→ wrote {OUT / 'hc428_r2fix_long_short_summary.json'}")
    print("\nDONE")


if __name__ == "__main__":
    main()
