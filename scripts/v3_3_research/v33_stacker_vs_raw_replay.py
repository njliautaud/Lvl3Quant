#!/usr/bin/env python3
"""
HC #396 weekend lane: Stacker-vs-raw market replay validation.

Compares two signals through the canonical HC #377 5-component market replay:
  A) Raw pred_fifo_tp8sl5_net head (from v3.3 60d champion NPZ).
  B) Meta-MLP stacker output (predicts fifo_tp8sl5_net from the OTHER 31 heads
     + 3 book-context proxies).

The stacker was trained on the first 80% of fold_00_predictions.npz rows
(after dropping invalid-target rows) and validated on the last 20%. We honor
that split: BOTH signals are evaluated on the val rows ONLY, so the
percentile thresholds and replay metrics are computed on a clean held-out
slab.

Replay engine: re-uses scripts.v3_3_research.v33_production_readiness_full_sweep
.evaluate_cell with regime_mask = val_slab. Operational config is the best
raw-head config from output/v3_3_production_readiness_20260516/sweep_results_full.csv
filtered to head=fifo_tp8sl5_net, regime=all, n_filled>=30, sorted by Sharpe.

NOT MALWARE. Pure analysis driver — reuses existing replay library as-is.
Writes only under output/meta_mlp_v3_3/.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

LVL3 = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(LVL3))

from scripts.v3_3_research.v33_production_readiness_full_sweep import (  # noqa: E402
    evaluate_cell,
    select_signals,
)
from scripts.v3_3_research.full_market_replay import (  # noqa: E402
    _load_fifo_labels,
)

PREDS = LVL3 / "output/cnn_mamba_v3_3_uncertainty_weighted/fold_00_predictions.npz"
LABELS_DIR = LVL3 / "data/processed/mbo_events_smart_v3_fifo_labels"
STACKER_CKPT = LVL3 / "output/meta_mlp_v3_3/stacker_final.pt"
SWEEP_CSV = LVL3 / "output/v3_3_production_readiness_20260516/sweep_results_full.csv"
OUT_DIR = LVL3 / "output/meta_mlp_v3_3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_HEAD = "fifo_tp8sl5_net"
INPUT_HEADS = [
    "log_ret_1s", "log_ret_5s", "log_ret_10s",
    "log_ret_30s", "log_ret_60s", "log_ret_5min",
    "p_up_5s", "p_up_10s", "p_up_30s", "p_up_60s",
    "log_ret_10s_q10", "log_ret_10s_q50", "log_ret_10s_q90",
    "log_ret_30s_q10", "log_ret_30s_q50", "log_ret_30s_q90",
    "log_ret_60s_q10", "log_ret_60s_q50", "log_ret_60s_q90",
    "pred_mfe_30s_ticks", "pred_mae_30s_ticks",
    "pred_mfe_60s_ticks", "pred_mae_60s_ticks",
    "pred_time_to_mfe_secs",
    "p_reversal_15s", "p_reversal_30s", "p_reversal_60s",
    "pred_realized_vol_30s_ticks",
    "fifo_tp4sl3_net",
    "fifo_tp4sl3_hit_tp", "fifo_tp8sl5_hit_tp",
]


class StackerMLP(nn.Module):
    def __init__(self, in_dim: int = 34):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def build_features(d, n_total):
    cols = []
    for h in INPUT_HEADS:
        k = f"pred_{h}"
        if k not in d.files:
            raise KeyError(f"missing {k}")
        cols.append(np.nan_to_num(d[k][:n_total], nan=0.0).astype(np.float32))
    vol30 = np.nan_to_num(d["pred_pred_realized_vol_30s_ticks"][:n_total], nan=1.0).astype(np.float32)
    qimb = (np.nan_to_num(d["pred_p_up_30s"][:n_total], nan=0.5) - 0.5).astype(np.float32)
    spread = np.ones_like(vol30, dtype=np.float32)
    cols += [vol30, qimb, spread]
    return np.stack(cols, axis=1)


def find_best_raw_config():
    """Best raw-head config: head=fifo_tp8sl5_net, regime=all, n_filled>=30,
    require day_conc<=1.0 (we just want the highest sharpe — gate violations
    are noted in output but don't block comparison).
    """
    df = pd.read_csv(SWEEP_CSV)
    f = df[(df["head"] == TARGET_HEAD)
           & (df["regime"] == "all")
           & (df["n_filled"] >= 30)].copy()
    f = f.dropna(subset=["sharpe"])
    f = f.sort_values("sharpe", ascending=False)
    if len(f) == 0:
        raise RuntimeError("No raw-head config found with n_filled>=30 for fifo_tp8sl5_net.")
    return f.iloc[0].to_dict()


def main():
    t0 = time.time()
    print(f"[init] preds: {PREDS}")
    print(f"[init] stacker: {STACKER_CKPT}")
    print(f"[init] out: {OUT_DIR}")

    # ---------------- Load predictions NPZ + FIFO labels ----------------
    d = np.load(PREDS, allow_pickle=True)
    keys = set(d.files)
    n_samples = int(d["n_samples"])
    oot_dates = [str(x) for x in d["oot_dates"]]
    print(f"[data] n_samples={n_samples} dates={oot_dates}")

    fifo = _load_fifo_labels(LABELS_DIR, oot_dates)
    n_total = min(n_samples, sum(fifo["_n_per_day"]))
    print(f"[data] n_total = min(preds={n_samples}, fifo={sum(fifo['_n_per_day'])}) = {n_total}")

    # tgt_lr arrays at TICK encoding (per library convention)
    preds_all = {"tgt_lr": {}, "tgt_lr_mask": {}}
    for h in ("1s", "5s", "10s", "30s"):
        preds_all["tgt_lr"][h] = d[f"target_log_ret_{h}"][:n_total].astype(np.float64)
        preds_all["tgt_lr_mask"][h] = d[f"mask_log_ret_{h}"][:n_total].astype(bool) \
                                       & np.isfinite(preds_all["tgt_lr"][h])

    # ---------------- Reproduce the stacker's train/val split ----------------
    # The trainer drops rows where mask_<target> is False or target is non-finite,
    # then chronologically splits 80/20. We mirror that to identify the val slab
    # IN THE ORIGINAL NPZ INDEX SPACE so the replay (which uses original ts_ns
    # from fifo labels) lines up.
    y_raw = np.nan_to_num(d[f"target_{TARGET_HEAD}"][:n_total], nan=0.0).astype(np.float32)
    mask_t = np.asarray(d.get(f"mask_{TARGET_HEAD}", np.ones_like(y_raw, dtype=bool))[:n_total],
                        dtype=bool)
    keep = mask_t & np.isfinite(y_raw)
    kept_idx = np.where(keep)[0]
    N_keep = kept_idx.size
    n_train_keep = int(N_keep * 0.8)
    val_kept_idx = kept_idx[n_train_keep:]      # original-NPZ indices of val rows
    train_kept_idx = kept_idx[:n_train_keep]
    print(f"[split] N_keep={N_keep}  n_train={n_train_keep}  n_val={val_kept_idx.size}")
    print(f"[split] val range in orig idx: [{val_kept_idx.min()}, {val_kept_idx.max()}]")

    # val regime mask aligned to n_total
    val_mask = np.zeros(n_total, dtype=bool)
    val_mask[val_kept_idx] = True

    # ---------------- Compute stacker predictions on FULL n_total ----------------
    # We need stacker preds at every NPZ row (so we can run the replay loop
    # uniformly with select_signals filtering val_mask). Rows outside val
    # are masked out anyway.
    X = build_features(d, n_total)        # (n_total, 34)
    print(f"[feat] X shape={X.shape}")

    # Load stacker
    ckpt = torch.load(STACKER_CKPT, map_location="cpu", weights_only=False)
    feat_mu = np.asarray(ckpt["feat_mu"], dtype=np.float32)
    feat_sd = np.asarray(ckpt["feat_sd"], dtype=np.float32)
    in_dim = int(ckpt.get("in_dim", X.shape[1]))
    print(f"[stacker] in_dim={in_dim}  feat_mu[0:3]={feat_mu[:3]}  feat_sd[0:3]={feat_sd[:3]}")

    model = StackerMLP(in_dim=in_dim)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    Xn = (X - feat_mu) / feat_sd
    with torch.no_grad():
        stacker_pred_full = model(torch.from_numpy(Xn)).numpy().astype(np.float64)
    print(f"[stacker] pred stats: mean={stacker_pred_full.mean():+.4f}  std={stacker_pred_full.std():.4f}")

    # ---------------- Identify best raw-head config ----------------
    best = find_best_raw_config()
    print("[config] best raw config from sweep:")
    for k in ("band", "band_frac", "side", "order_type", "cancel_window", "hold_s",
              "exit_horizon", "regime", "n_filled", "fill_rate", "pnl_ticks_per_fill",
              "sharpe", "sortino", "profit_factor", "win_rate", "day_conc"):
        print(f"  {k:22s} = {best.get(k)}")

    cfg_band_frac = float(best["band_frac"])
    cfg_side = str(best["side"])
    cfg_otype = str(best["order_type"])
    cfg_cw = int(best["cancel_window"])
    cfg_hold = float(best["hold_s"])
    cfg_exit_h = str(best["exit_horizon"])

    # ---------------- Run replay on val slab for raw + stacker ----------------
    raw_pred = np.nan_to_num(d[f"pred_{TARGET_HEAD}"][:n_total], nan=0.0).astype(np.float64)
    raw_mask = (np.asarray(d.get(f"mask_{TARGET_HEAD}", np.ones(n_total, dtype=bool))[:n_total],
                           dtype=bool)
                & np.isfinite(raw_pred))
    stk_mask = np.isfinite(stacker_pred_full)

    # Mask both preds to VAL slab BEFORE evaluate_cell, so the percentile gate
    # (computed inside select_signals on pred[mask]) operates on val-only
    # populations. This is the fair comparison.
    raw_mask_val = raw_mask & val_mask
    stk_mask_val = stk_mask & val_mask
    print(f"[replay] Running raw replay (val slab only, val-internal percentile)...")
    res_raw = evaluate_cell(
        pred=raw_pred, mask=raw_mask_val, bullish_high=True,
        side=cfg_side, band_frac=cfg_band_frac, order_type=cfg_otype,
        cancel_window=cfg_cw, hold_seconds=cfg_hold, exit_horizon=cfg_exit_h,
        fifo=fifo, preds_all=preds_all, regime_mask=None, n_total=n_total,
    )

    print(f"[replay] Running stacker replay (val slab only, val-internal percentile)...")
    res_stk = evaluate_cell(
        pred=stacker_pred_full, mask=stk_mask_val, bullish_high=True,
        side=cfg_side, band_frac=cfg_band_frac, order_type=cfg_otype,
        cancel_window=cfg_cw, hold_seconds=cfg_hold, exit_horizon=cfg_exit_h,
        fifo=fifo, preds_all=preds_all, regime_mask=None, n_total=n_total,
    )

    # Also: run BOTH on full slab (no val mask) for context
    print(f"[replay] Running raw replay (full slab) for context...")
    res_raw_full = evaluate_cell(
        pred=raw_pred, mask=raw_mask, bullish_high=True,
        side=cfg_side, band_frac=cfg_band_frac, order_type=cfg_otype,
        cancel_window=cfg_cw, hold_seconds=cfg_hold, exit_horizon=cfg_exit_h,
        fifo=fifo, preds_all=preds_all, regime_mask=None, n_total=n_total,
    )
    print(f"[replay] Running stacker replay (full slab) — INCLUDES TRAIN ROWS, biased...")
    res_stk_full = evaluate_cell(
        pred=stacker_pred_full, mask=stk_mask, bullish_high=True,
        side=cfg_side, band_frac=cfg_band_frac, order_type=cfg_otype,
        cancel_window=cfg_cw, hold_seconds=cfg_hold, exit_horizon=cfg_exit_h,
        fifo=fifo, preds_all=preds_all, regime_mask=None, n_total=n_total,
    )

    METRIC_KEYS = [
        "n_signals", "n_filled", "fill_rate",
        "pnl_ticks_per_fill", "pnl_ticks_total",
        "sharpe", "sortino", "profit_factor", "win_rate",
        "avg_mfe_ticks", "avg_mae_ticks", "max_dc_ticks",
        "adv_sel_30s_avg", "avg_queue_pos", "commission_total",
        "day_conc", "pass_hc344", "edge_offset_ticks",
    ]

    out_rows = []
    for label, res in [
        ("raw_val", res_raw),
        ("stacker_val", res_stk),
        ("raw_full_slab", res_raw_full),
        ("stacker_full_slab_biased", res_stk_full),
    ]:
        row = {"signal": label}
        for k in METRIC_KEYS:
            v = res.get(k)
            try:
                row[k] = float(v) if not isinstance(v, bool) else bool(v)
            except Exception:
                row[k] = v
        # carry config too for record
        row["band_frac"] = cfg_band_frac
        row["side"] = cfg_side
        row["order_type"] = cfg_otype
        row["cancel_window"] = cfg_cw
        row["hold_s"] = cfg_hold
        row["exit_horizon"] = cfg_exit_h
        out_rows.append(row)

    df_out = pd.DataFrame(out_rows)
    csv_path = OUT_DIR / "stacker_vs_raw_replay.csv"
    df_out.to_csv(csv_path, index=False)
    print(f"[done] wrote {csv_path}")
    print(df_out[["signal", "n_signals", "n_filled", "fill_rate",
                  "pnl_ticks_per_fill", "sharpe", "sortino",
                  "profit_factor", "win_rate", "day_conc"]].to_string(index=False))

    # Brief markdown summary
    raw_v = res_raw
    stk_v = res_stk
    md = [
        "# Stacker vs Raw Head — Full Market Replay (HC #396)",
        "",
        f"- Predictions NPZ: `{PREDS}`",
        f"- Stacker ckpt: `{STACKER_CKPT}`",
        f"- Best raw-head config from sweep: band={best['band']} band_frac={cfg_band_frac} "
        f"side={cfg_side} order_type={cfg_otype} cancel_window={cfg_cw} hold_s={cfg_hold} "
        f"exit_horizon={cfg_exit_h}",
        f"- Eval slab: VAL ONLY (last 20% chronological — matches stacker's held-out split)",
        f"- Val slab spans original idx [{val_kept_idx.min()}, {val_kept_idx.max()}] / n_total={n_total}",
        "",
        "## Headline (VAL SLAB, fair comparison)",
        "",
        "| Signal | n_signals | n_filled | fill_rate | ticks/fill | Sharpe | Sortino | PF | WR (%) | day_conc |",
        "|---|---|---|---|---|---|---|---|---|---|",
        f"| RAW head     | {raw_v['n_signals']} | {raw_v['n_filled']} | {raw_v['fill_rate']:.3f} | "
        f"{raw_v['pnl_ticks_per_fill']:+.4f} | {raw_v['sharpe']:.1f} | {raw_v['sortino']:.1f} | "
        f"{raw_v['profit_factor']:.3f} | {raw_v['win_rate']:.2f} | {raw_v['day_conc']:.3f} |",
        f"| META stacker | {stk_v['n_signals']} | {stk_v['n_filled']} | {stk_v['fill_rate']:.3f} | "
        f"{stk_v['pnl_ticks_per_fill']:+.4f} | {stk_v['sharpe']:.1f} | {stk_v['sortino']:.1f} | "
        f"{stk_v['profit_factor']:.3f} | {stk_v['win_rate']:.2f} | {stk_v['day_conc']:.3f} |",
        "",
        "## Verdict",
        "",
    ]
    if np.isfinite(stk_v["sharpe"]) and np.isfinite(raw_v["sharpe"]):
        delta = stk_v["sharpe"] - raw_v["sharpe"]
        winner = "stacker" if delta > 0 else "raw"
        md.append(f"- Sharpe Δ (stacker − raw) = **{delta:+.1f}** → **{winner}** wins on Sharpe.")
    md.append(f"- ticks/fill Δ = **{stk_v['pnl_ticks_per_fill'] - raw_v['pnl_ticks_per_fill']:+.4f}** (stacker − raw)")
    md.append(f"- HC #344 (day_conc<=0.20 & n_filled>=30): raw={'PASS' if raw_v['pass_hc344'] else 'FAIL'}  stacker={'PASS' if stk_v['pass_hc344'] else 'FAIL'}")
    md.append("")
    md.append("## Caveat")
    md.append("- HC #396 prior agent reported val IC 0.230 (4.8x raw 0.048). This script tests "
             "whether that IC boost translates to PnL after queue-position + adverse-selection + "
             "commission realism.")
    md.append("- Full-slab numbers are appended for context but are BIASED for the stacker "
             "(includes its train rows).")

    md_path = OUT_DIR / "stacker_vs_raw_replay.md"
    md_path.write_text("\n".join(md))
    print(f"[done] wrote {md_path}")
    print(f"[done] elapsed {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
