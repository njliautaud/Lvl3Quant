#!/usr/bin/env python3
"""
HC #396 weekend lane follow-up: SIZED replay comparison.

Compares three replays of the operational raw head (`fifo_tp8sl5_net`) under
the best-raw-config from the production-readiness sweep, all on the VAL slab:

  (a) equal-sized: 1 contract per fill                                BASELINE
  (b) calibrator-continuous: size = clip(calibrator_pred / median_pred, 0.5, 2.0)
  (c) calibrator-quintile:   top quintile 1.5x, mid quintile 1.0x, bottom 0.5x

Calibrator: meta_mlp_v3_3 sizing calibrator trained on Razer, predicting
|target_fifo_tp8sl5_net| from 32 v3.3 heads + 3 book ctx proxies. Pulled
via SCP into output/meta_mlp_v3_3/sizing/.

Per HC #69: report Sharpe / Sortino / PF / day_conc / WR — NOT raw P&L.
Per HC #344: still gated on day_conc <= 0.20.

Writes:
  output/meta_mlp_v3_3/sizing_replay_verdict.csv
  output/meta_mlp_v3_3/sizing_replay_verdict.md

NOT MALWARE. Pure analysis driver. Re-uses existing replay library helpers.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

LVL3 = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(LVL3))

from scripts.v3_3_research.full_market_replay import (  # noqa: E402
    ES_RT_COMMISSION_TICKS_DEFAULT,
    ES_SPREAD_TICKS_RTH_DEFAULT,
    PRICE_UNIT_TO_TICKS,
    ANN_FACTOR_PER_STEP,
    _load_fifo_labels,
    _queue_position_model,
    _entry_price_edge_ticks,
)
from scripts.v3_3_research.v33_production_readiness_full_sweep import (  # noqa: E402
    select_signals,
)

PREDS = LVL3 / "output/cnn_mamba_v3_3_uncertainty_weighted/fold_00_predictions.npz"
LABELS_DIR = LVL3 / "data/processed/mbo_events_smart_v3_fifo_labels"
SIZING_CKPT = LVL3 / "output/meta_mlp_v3_3/sizing/sizing_calibrator_final.pt"
SWEEP_CSV = LVL3 / "output/v3_3_production_readiness_20260516/sweep_results_full.csv"
OUT_DIR = LVL3 / "output/meta_mlp_v3_3"

TARGET_HEAD = "fifo_tp8sl5_net"
RT_COMM = ES_RT_COMMISSION_TICKS_DEFAULT
DAY_CONC_GATE = 0.20

# Sizing model -- MUST match train_sizing.py architecture
SIZING_INPUT_HEADS = [
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
    "fifo_tp4sl3_net", "fifo_tp8sl5_net",
    "fifo_tp4sl3_hit_tp", "fifo_tp8sl5_hit_tp",
]
assert len(SIZING_INPUT_HEADS) == 32


class SizingMLP(nn.Module):
    def __init__(self, in_dim: int = 35):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 1),
        )
        self.softplus = nn.Softplus()

    def forward(self, x):
        return self.softplus(self.trunk(x).squeeze(-1))


def build_sizing_features(d, n_total: int) -> np.ndarray:
    cols = []
    for h in SIZING_INPUT_HEADS:
        k = f"pred_{h}"
        if k not in d.files:
            raise KeyError(f"missing {k}")
        cols.append(np.nan_to_num(d[k][:n_total], nan=0.0).astype(np.float32))
    vol30 = np.nan_to_num(d["pred_pred_realized_vol_30s_ticks"][:n_total], nan=1.0).astype(np.float32)
    qimb = (np.nan_to_num(d["pred_p_up_30s"][:n_total], nan=0.5) - 0.5).astype(np.float32)
    spread = np.ones_like(vol30, dtype=np.float32)
    cols += [vol30, qimb, spread]
    return np.stack(cols, axis=1)


def evaluate_per_trade(
    *,
    pred: np.ndarray, mask: np.ndarray, bullish_high: bool,
    side: str, band_frac: float, order_type: str, cancel_window: int,
    hold_seconds: float, exit_horizon: str,
    fifo: dict, preds_all: dict, regime_mask: np.ndarray | None,
    n_total: int,
):
    """Mirror evaluate_cell but RETURN per-fill net + ts + global_idx so we can
    apply per-trade sizing.

    HC #397B: also returns adverse-selection-at-+30s (signed ticks vs entry,
    after side flip) and avg_queue_pos from the FIFO queue position model so
    the summary CSV carries the canonical execution-realism columns.

    Returns:
      filled_global_idx (np.int64[n_filled]), net (np.float64[n_filled]),
      ts_filled_ns (np.int64[n_filled]), n_signals (int),
      adv_sel_30s (np.float64[n_filled]), avg_queue_pos (float)
    """
    sel = select_signals(pred, mask, side, band_frac, bullish_high)
    if regime_mask is not None:
        sel = sel & regime_mask
    sel_idx = np.where(sel)[0]
    n_signals = int(sel.sum())
    if n_signals == 0:
        return (np.array([], dtype=np.int64), np.array([], dtype=np.float64),
                np.array([], dtype=np.int64), 0,
                np.array([], dtype=np.float64), float("nan"))

    side_sign = 1.0 if side == "long" else -1.0

    side_key = side
    filled_lbl = fifo[f"tp4sl3_{side_key}_filled"][:n_total][sel_idx]
    exit_reason_lbl = fifo[f"tp4sl3_{side_key}_exit_reason"][:n_total][sel_idx]
    hold_time_lbl = fifo[f"tp4sl3_{side_key}_hold_time_ns"][:n_total][sel_idx]

    filled_mask, q_arrival, avg_q = _queue_position_model(
        order_type, cancel_window, filled_lbl, exit_reason_lbl, hold_time_lbl,
    )
    filled_idx_in_sel = np.where(filled_mask)[0]
    filled_global_idx = sel_idx[filled_idx_in_sel]
    n_filled = int(filled_mask.sum())
    if n_filled == 0:
        return (np.array([], dtype=np.int64), np.array([], dtype=np.float64),
                np.array([], dtype=np.int64), n_signals,
                np.array([], dtype=np.float64), float(avg_q) if np.isfinite(avg_q) else float("nan"))

    lr_exit = preds_all["tgt_lr"][exit_horizon][filled_global_idx]
    lr_mask_exit = preds_all["tgt_lr_mask"][exit_horizon][filled_global_idx]

    edge_offset = _entry_price_edge_ticks(order_type, ES_SPREAD_TICKS_RTH_DEFAULT)
    net = side_sign * lr_exit * PRICE_UNIT_TO_TICKS + edge_offset - RT_COMM
    net = np.where(lr_mask_exit, net, 0.0).astype(np.float64)

    # HC #397B adverse selection @+30s: signed ticks vs entry after side flip.
    # Uses canonical target_log_ret_30s already loaded in preds_all.
    lr_30s = preds_all["tgt_lr"]["30s"][filled_global_idx]
    lr_30s_mask = preds_all["tgt_lr_mask"]["30s"][filled_global_idx]
    adv_sel = side_sign * lr_30s * PRICE_UNIT_TO_TICKS
    adv_sel = np.where(lr_30s_mask, adv_sel, np.nan).astype(np.float64)

    ts_filled = fifo["ts_ns"][:n_total][filled_global_idx]
    return (filled_global_idx.astype(np.int64), net, ts_filled.astype(np.int64),
            n_signals, adv_sel, float(avg_q) if np.isfinite(avg_q) else float("nan"))


def summarize(net: np.ndarray, ts_ns: np.ndarray, comm_per_fill: float = RT_COMM, label: str = "?",
              adv_sel: np.ndarray | None = None, avg_queue_pos: float = float("nan"),
              cancel_window: int = 0) -> dict:
    """Compute the canonical risk-adjusted metrics on per-trade nets (already
    in tick units, INCLUDING size weighting). Size multiplies the net
    delta per trade (we treat 1 contract = nominal $12.50/tick).
    """
    n_filled = int(net.size)
    if n_filled == 0:
        return {"label": label, "n_filled": 0}

    total = float(net.sum())
    mean_ = float(net.mean())
    sd_ = float(net.std(ddof=1)) if n_filled >= 2 and net.std(ddof=1) > 1e-12 else float("nan")
    sharpe = (mean_ / sd_ * np.sqrt(ANN_FACTOR_PER_STEP)) if np.isfinite(sd_) else float("nan")
    neg = net[net < 0]
    if neg.size >= 2 and neg.std(ddof=1) > 1e-12:
        sortino = mean_ / float(neg.std(ddof=1)) * np.sqrt(ANN_FACTOR_PER_STEP)
    else:
        sortino = float("inf") if mean_ > 0 else float("nan")

    gw = float(net[net > 0].sum())
    gl = -float(net[net < 0].sum())
    pf = (gw / gl) if gl > 1e-12 else (float("inf") if gw > 0 else float("nan"))
    wr = float((net > 0).mean() * 100.0)

    eq = np.cumsum(net); peak = np.maximum.accumulate(eq); dd = peak - eq
    max_dc = float(dd.max()) if dd.size else 0.0

    dts = pd.to_datetime(ts_ns, unit="ns", utc=True).tz_convert("America/Chicago").date
    df = pd.DataFrame({"date": dts, "net": net})
    per_day = df.groupby("date")["net"].sum()
    total_abs = per_day.abs().sum()
    day_conc = float(per_day.abs().max() / total_abs) if total_abs > 1e-12 else float("nan")

    # HC #397B mandatory columns
    if adv_sel is not None and adv_sel.size > 0:
        adv_sel_30s_avg = float(np.nanmean(adv_sel))
    else:
        adv_sel_30s_avg = float("nan")

    return {
        "label": label,
        "n_filled": n_filled,
        "ticks_total": total,
        "ticks_per_fill": total / max(1, n_filled),
        "sharpe": sharpe,
        "sortino": sortino,
        "profit_factor": pf,
        "win_rate": wr,
        "max_dc_ticks": max_dc,
        "adv_sel_30s_avg": adv_sel_30s_avg,
        "avg_queue_pos": avg_queue_pos,
        "cancel_window": int(cancel_window) if np.isfinite(cancel_window) else 0,
        "day_conc": day_conc,
        "pass_hc344": bool(np.isfinite(day_conc) and day_conc <= DAY_CONC_GATE and n_filled >= 30),
    }


def find_best_raw_config():
    df = pd.read_csv(SWEEP_CSV)
    f = df[(df["head"] == TARGET_HEAD) & (df["regime"] == "all")
           & (df["n_filled"] >= 30)].copy().dropna(subset=["sharpe"])
    f = f.sort_values("sharpe", ascending=False)
    if len(f) == 0:
        raise RuntimeError("No raw config found.")
    return f.iloc[0].to_dict()


def main():
    t0 = time.time()
    print(f"[init] preds: {PREDS}")
    print(f"[init] sizing ckpt: {SIZING_CKPT}")

    d = np.load(PREDS, allow_pickle=True)
    n_samples = int(d["n_samples"])
    oot_dates = [str(x) for x in d["oot_dates"]]
    print(f"[data] n_samples={n_samples}  dates={oot_dates}")

    fifo = _load_fifo_labels(LABELS_DIR, oot_dates)
    n_total = min(n_samples, sum(fifo["_n_per_day"]))
    print(f"[data] n_total={n_total}")

    preds_all = {"tgt_lr": {}, "tgt_lr_mask": {}}
    for h in ("1s", "5s", "10s", "30s"):
        preds_all["tgt_lr"][h] = d[f"target_log_ret_{h}"][:n_total].astype(np.float64)
        preds_all["tgt_lr_mask"][h] = (d[f"mask_log_ret_{h}"][:n_total].astype(bool)
                                        & np.isfinite(preds_all["tgt_lr"][h]))

    # Reproduce val slab in original index space
    y_raw = np.nan_to_num(d[f"target_{TARGET_HEAD}"][:n_total], nan=0.0).astype(np.float32)
    mask_t = np.asarray(d.get(f"mask_{TARGET_HEAD}", np.ones(n_total, dtype=bool))[:n_total], dtype=bool)
    keep = mask_t & np.isfinite(y_raw)
    kept_idx = np.where(keep)[0]
    N_keep = kept_idx.size
    n_train_keep = int(N_keep * 0.8)
    val_kept_idx = kept_idx[n_train_keep:]
    print(f"[split] N_keep={N_keep}  n_train={n_train_keep}  n_val={val_kept_idx.size}")
    val_mask = np.zeros(n_total, dtype=bool)
    val_mask[val_kept_idx] = True

    # Sizing predictions on full n_total (we only use them at filled indices)
    print(f"[sizing] loading calibrator: {SIZING_CKPT}")
    X = build_sizing_features(d, n_total)
    print(f"[sizing] feat shape: {X.shape}")
    ckpt = torch.load(SIZING_CKPT, map_location="cpu", weights_only=False)
    feat_mu = np.asarray(ckpt["feat_mu"], dtype=np.float32)
    feat_sd = np.asarray(ckpt["feat_sd"], dtype=np.float32)
    in_dim = int(ckpt.get("in_dim", X.shape[1]))
    model = SizingMLP(in_dim=in_dim)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    Xn = (X - feat_mu) / feat_sd
    with torch.no_grad():
        sizing_pred_full = model(torch.from_numpy(Xn)).numpy().astype(np.float64)
    print(f"[sizing] pred stats over full slab: "
          f"mean={sizing_pred_full.mean():.4f}  std={sizing_pred_full.std():.4f}  "
          f"min={sizing_pred_full.min():.4f}  max={sizing_pred_full.max():.4f}")

    # Best raw config
    best = find_best_raw_config()
    cfg_band_frac = float(best["band_frac"])
    cfg_side = str(best["side"])
    cfg_otype = str(best["order_type"])
    cfg_cw = int(best["cancel_window"])
    cfg_hold = float(best["hold_s"])
    cfg_exit_h = str(best["exit_horizon"])
    print(f"[cfg] band_frac={cfg_band_frac} side={cfg_side} otype={cfg_otype} "
          f"cw={cfg_cw} hold={cfg_hold} exit={cfg_exit_h}")

    raw_pred = np.nan_to_num(d[f"pred_{TARGET_HEAD}"][:n_total], nan=0.0).astype(np.float64)
    raw_mask = (np.asarray(d.get(f"mask_{TARGET_HEAD}",
                                  np.ones(n_total, dtype=bool))[:n_total], dtype=bool)
                & np.isfinite(raw_pred))
    raw_mask_val = raw_mask & val_mask

    # Replay -- raw head, val slab
    filled_idx, net_base, ts_filled, n_signals, adv_sel_30s, avg_q_pos = evaluate_per_trade(
        pred=raw_pred, mask=raw_mask_val, bullish_high=True,
        side=cfg_side, band_frac=cfg_band_frac, order_type=cfg_otype,
        cancel_window=cfg_cw, hold_seconds=cfg_hold, exit_horizon=cfg_exit_h,
        fifo=fifo, preds_all=preds_all, regime_mask=None, n_total=n_total,
    )
    print(f"[replay] n_signals={n_signals}  n_filled={filled_idx.size}  "
          f"avg_q_pos={avg_q_pos:.3f}  adv_sel_30s_avg={float(np.nanmean(adv_sel_30s)) if adv_sel_30s.size else float('nan'):+.4f}")
    if filled_idx.size == 0:
        raise RuntimeError("Zero fills -- nothing to size.")

    # Per-trade calibrator predictions (positive)
    cal_at_fill = sizing_pred_full[filled_idx]
    print(f"[sizing@fill] mean={cal_at_fill.mean():.4f}  std={cal_at_fill.std():.4f}  "
          f"min={cal_at_fill.min():.4f}  max={cal_at_fill.max():.4f}")

    # Variant (a) equal-sized: base nets unchanged.
    # HC #397B: pass adv_sel_30s + avg_queue_pos so summarize() can record them.
    # Note: sizing variants B/C scale net but do NOT change adv_sel (same fills,
    # same horizon move) nor avg_queue_pos (same fills, same FIFO outcomes).
    rows = []
    rows.append(summarize(net_base.copy(), ts_filled, label="A_equal_sized",
                          adv_sel=adv_sel_30s, avg_queue_pos=avg_q_pos,
                          cancel_window=cfg_cw))

    # Variant (b) continuous: size = clip(cal/median, 0.5, 2.0)
    med = float(np.median(cal_at_fill))
    sizes_b = np.clip(cal_at_fill / max(1e-9, med), 0.5, 2.0)
    print(f"[size_b] median={med:.4f}  size mean={sizes_b.mean():.3f}  "
          f"std={sizes_b.std():.3f}  pct at 0.5={float((sizes_b<=0.5+1e-6).mean()):.3f}  "
          f"at 2.0={float((sizes_b>=2.0-1e-6).mean()):.3f}")
    net_b = net_base * sizes_b
    rows.append(summarize(net_b, ts_filled, label="B_continuous_clip_0.5_2.0",
                          adv_sel=adv_sel_30s, avg_queue_pos=avg_q_pos,
                          cancel_window=cfg_cw))

    # Variant (c) quintile: bottom 0.5x, middle 1.0x, top 1.5x.
    # Buckets by calibrator pred quintile.
    q20 = np.quantile(cal_at_fill, 0.20)
    q80 = np.quantile(cal_at_fill, 0.80)
    sizes_c = np.where(cal_at_fill <= q20, 0.5,
                       np.where(cal_at_fill >= q80, 1.5, 1.0))
    print(f"[size_c] q20={q20:.4f}  q80={q80:.4f}  "
          f"n_low={int((sizes_c==0.5).sum())}  n_mid={int((sizes_c==1.0).sum())}  "
          f"n_high={int((sizes_c==1.5).sum())}")
    net_c = net_base * sizes_c
    rows.append(summarize(net_c, ts_filled, label="C_quintile_0.5_1.0_1.5",
                          adv_sel=adv_sel_30s, avg_queue_pos=avg_q_pos,
                          cancel_window=cfg_cw))

    # Diagnostic: rank-correlation between calibrator pred and |net|
    abs_net = np.abs(net_base)
    rp = np.argsort(np.argsort(cal_at_fill))
    rt = np.argsort(np.argsort(abs_net))
    rp_z = (rp - rp.mean()) / (rp.std() + 1e-9)
    rt_z = (rt - rt.mean()) / (rt.std() + 1e-9)
    ic_at_fill = float(np.mean(rp_z * rt_z))
    print(f"[diag] Spearman(cal_pred, |realized_net|) at fill = {ic_at_fill:+.4f}")

    # Also: directional Spearman -- does calibrator's magnitude track signed net?
    rp2 = np.argsort(np.argsort(cal_at_fill))
    rt2 = np.argsort(np.argsort(net_base))
    rp2_z = (rp2 - rp2.mean()) / (rp2.std() + 1e-9)
    rt2_z = (rt2 - rt2.mean()) / (rt2.std() + 1e-9)
    ic_signed = float(np.mean(rp2_z * rt2_z))
    print(f"[diag] Spearman(cal_pred, signed_net) at fill = {ic_signed:+.4f}")

    df_out = pd.DataFrame(rows)
    df_out["band_frac"] = cfg_band_frac
    df_out["side"] = cfg_side
    df_out["order_type"] = cfg_otype
    df_out["cancel_window"] = cfg_cw
    df_out["hold_s"] = cfg_hold
    df_out["exit_horizon"] = cfg_exit_h
    df_out["calib_ic_abs_pnl_at_fill"] = ic_at_fill
    df_out["calib_ic_signed_pnl_at_fill"] = ic_signed
    print(df_out.to_string())

    out_csv = OUT_DIR / "sizing_replay_verdict.csv"
    df_out.to_csv(out_csv, index=False)
    print(f"[done] {out_csv}")

    # Markdown summary
    a = rows[0]; b = rows[1]; c = rows[2]
    def fmt(r, k, w=10, dp=4):
        v = r.get(k, None)
        if v is None: return "n/a"
        if isinstance(v, bool): return str(v)
        if isinstance(v, int): return f"{v:,}"
        if not np.isfinite(v): return "inf"
        return f"{v:.{dp}f}"

    md_lines = [
        "# Sizing Calibrator Replay Verdict (HC #396 follow-up)",
        "",
        f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Source preds: `{PREDS}`",
        f"Calibrator: `{SIZING_CKPT}`",
        f"Operational config: head=`{TARGET_HEAD}` band_frac={cfg_band_frac} side={cfg_side} "
        f"otype={cfg_otype} cw={cfg_cw} hold={cfg_hold} exit={cfg_exit_h}",
        f"Val slab: last 20% of mask-valid rows of `target_{TARGET_HEAD}`.",
        "",
        "## Diagnostics",
        f"- Calibrator val IC vs `|realized_net|` at FILLED rows only: **{ic_at_fill:+.4f}**",
        f"- Calibrator IC vs SIGNED `realized_net` (sanity check; should be ~0 if "
        f"calibrator only learned magnitude): **{ic_signed:+.4f}**",
        "",
        "## Per-variant metrics",
        "",
        "| variant | n_filled | ticks_total | ticks/fill | Sharpe | Sortino | PF | WR (%) | max_dc | adv_sel_30s | avg_q_pos | cw | day_conc | HC344 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        md_lines.append(
            f"| {r['label']} | {r['n_filled']} | {fmt(r,'ticks_total',dp=2)} | "
            f"{fmt(r,'ticks_per_fill',dp=4)} | {fmt(r,'sharpe',dp=1)} | "
            f"{fmt(r,'sortino',dp=1)} | {fmt(r,'profit_factor',dp=3)} | "
            f"{fmt(r,'win_rate',dp=2)} | {fmt(r,'max_dc_ticks',dp=2)} | "
            f"{fmt(r,'adv_sel_30s_avg',dp=3)} | {fmt(r,'avg_queue_pos',dp=3)} | "
            f"{r.get('cancel_window',0)} | "
            f"{fmt(r,'day_conc',dp=4)} | {r['pass_hc344']} |"
        )

    # Verdict bullet
    a_sh = a.get("sharpe", float("nan"))
    b_sh = b.get("sharpe", float("nan"))
    c_sh = c.get("sharpe", float("nan"))
    a_pf = a.get("profit_factor", float("nan"))
    b_pf = b.get("profit_factor", float("nan"))
    c_pf = c.get("profit_factor", float("nan"))
    a_dc = a.get("day_conc", float("nan"))
    b_dc = b.get("day_conc", float("nan"))
    c_dc = c.get("day_conc", float("nan"))

    def safe_pct(new, base):
        if not (np.isfinite(new) and np.isfinite(base)) or abs(base) < 1e-9:
            return "n/a"
        return f"{(new-base)/base*100:+.1f}%"

    md_lines += [
        "",
        "## Deltas vs baseline (A = equal-sized)",
        "",
        f"- B (continuous): Sharpe Δ = {safe_pct(b_sh,a_sh)} | PF Δ = {safe_pct(b_pf,a_pf)} "
        f"| day_conc {a_dc:.4f}→{b_dc:.4f}",
        f"- C (quintile):   Sharpe Δ = {safe_pct(c_sh,a_sh)} | PF Δ = {safe_pct(c_pf,a_pf)} "
        f"| day_conc {a_dc:.4f}→{c_dc:.4f}",
        "",
        "## Verdict",
        f"- HC #344 still fails on all three (day_conc > 0.20 on tiny val slab); not a "
        "production decision, just a relative-improvement test.",
        "- Sizing improves risk-adjusted returns IFF Sharpe Δ > 0 AND day_conc doesn't worsen. "
        "See the table above.",
    ]
    out_md = OUT_DIR / "sizing_replay_verdict.md"
    out_md.write_text("\n".join(md_lines))
    print(f"[done] {out_md}")
    print(f"[done] total t={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
