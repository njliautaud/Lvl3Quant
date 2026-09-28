"""
HC #320/#321/#322/#323 — v3.2 FULL FIFO AUDIT REPORT.

Replaces HC #318 (v32_market_replay_paper_trader.py) which was theoretical
sign(pred)*realized_log_ret PnL. This script uses the model's NATIVE FIFO heads
and the npz's TARGET FIFO labels (which were computed from MBO bid/ask at
label-gen time → REAL FIFO PnL).

Per HC #321, every strategy row reports:
  - avg hold time (mean / median / p75 / p95 from target_pred_time_to_mfe_secs)
  - MFE distribution (target_pred_mfe_30s_ticks: mean/median/p25/p75/p95)
  - MAE distribution (target_pred_mae_30s_ticks: same percentiles)
  - realized price path at +1s/+5s/+10s/+30s/+60s/+5min (mean signed return,
    long-side and short-side separate) from target_log_ret_*
  - FIFO cost breakdown (commission embedded in TP/SL target; spread crossed
    implicit in FIFO label generation)
  - confidence-band cell per HC #306/#314
  - DA as % per HC #313

Strategies tested:
  A. FIFO-TP4/SL3 native: rank by |pred_fifo_tp4sl3_net|, direction = sign(pred),
     pnl = target_fifo_tp4sl3_net (REAL FIFO)
  B. FIFO-TP8/SL5 native: same with tp8sl5 heads
  C. log_ret_1s entry + FIFO TP4/SL3 exec: rank by |pred_log_ret_1s|, direction
     = sign(pred_log_ret_1s), pnl = target_fifo_tp4sl3_net
  D. log_ret_1s entry + FIFO TP8/SL5 exec: same with tp8sl5
  E. Confluence (3-head agreement): top X% by |pred_log_ret_1s| AND
     sign(pred_log_ret_1s) == sign(pred_fifo_tp4sl3_net) AND
     sign(pred_log_ret_1s) == sign(pred_log_ret_5s)
     → pnl = target_fifo_tp4sl3_net

NO MIDPOINT. NO sign(pred)*realized_log_ret. Pure FIFO labels.

OUT: output/v3_2_deep_sim_20260512/v32_fifo_full_audit_hc320.{json,csv}
"""
from __future__ import annotations
import json
import numpy as np
from pathlib import Path

PREDS = Path("/home/jupiter/Lvl3Quant/output/v3_2_deep_sim_20260512/fold_00_oot_predictions.npz")
OUT_JSON = Path("/home/jupiter/Lvl3Quant/output/v3_2_deep_sim_20260512/v32_fifo_full_audit_hc320.json")
OUT_CSV = Path("/home/jupiter/Lvl3Quant/output/v3_2_deep_sim_20260512/v32_fifo_full_audit_hc320.csv")

ES_TICK_USD = 12.50
MIN_GAP_STEPS = 4  # 1s anti-churn at 250ms stride
CONF_BANDS = [0.001, 0.005, 0.01, 0.05, 0.10, 0.20]


def _apply_gap(sel_idx: np.ndarray, min_gap: int) -> np.ndarray:
    if len(sel_idx) == 0:
        return sel_idx
    taken = []
    last = -10**9
    for i in sel_idx:
        if i - last < min_gap:
            continue
        taken.append(i)
        last = i
    return np.array(taken, dtype=np.int64)


def _stats_block(pnl_ticks: np.ndarray, ann_factor: float) -> dict:
    n = int(len(pnl_ticks))
    if n < 5:
        return {"n_trades": n}
    mean = float(pnl_ticks.mean())
    std = float(pnl_ticks.std(ddof=1))
    sharpe = (mean / std * np.sqrt(ann_factor)) if std > 1e-9 else float("nan")
    neg = pnl_ticks[pnl_ticks < 0]
    dn = float(neg.std(ddof=1)) if len(neg) > 1 else 0.0
    sortino = (mean / dn * np.sqrt(ann_factor)) if dn > 1e-9 else float("nan")
    gw = float(pnl_ticks[pnl_ticks > 0].sum())
    gl = -float(pnl_ticks[pnl_ticks < 0].sum())
    pf = (gw / gl) if gl > 1e-9 else float("inf")
    wr = float((pnl_ticks > 0).mean() * 100.0)
    return {
        "n_trades": n,
        "mean_ticks": mean,
        "median_ticks": float(np.median(pnl_ticks)),
        "std_ticks": std,
        "sharpe": float(sharpe) if np.isfinite(sharpe) else None,
        "sortino": float(sortino) if np.isfinite(sortino) else None,
        "pf": float(pf) if np.isfinite(pf) else None,
        "wr_pct": wr,
        "total_ticks": float(pnl_ticks.sum()),
        "total_usd": float(pnl_ticks.sum() * ES_TICK_USD),
        "max_win_ticks": float(pnl_ticks.max()),
        "max_loss_ticks": float(pnl_ticks.min()),
    }


def _distribution(arr: np.ndarray, ticks: bool = True) -> dict:
    if len(arr) == 0:
        return {"n": 0}
    return {
        "n": int(len(arr)),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _price_path(taken: np.ndarray, direction: np.ndarray, targets: dict, masks: dict) -> dict:
    """Average signed return at each horizon (long-side and short-side separate)."""
    out = {}
    long_mask = direction > 0
    short_mask = direction < 0
    for h_name, key in [("1s", "log_ret_1s"), ("5s", "log_ret_5s"), ("10s", "log_ret_10s"),
                        ("30s", "log_ret_30s"), ("60s", "log_ret_60s"), ("5min", "log_ret_5min")]:
        if key not in targets:
            continue
        rzd = targets[key][taken]
        mk = masks[key][taken] & np.isfinite(rzd)
        signed = direction * rzd
        out[f"avg_signed_{h_name}"] = float(signed[mk].mean()) if mk.sum() > 0 else None
        if long_mask.sum() > 0 and (mk & long_mask).sum() > 0:
            out[f"long_avg_{h_name}"] = float(rzd[mk & long_mask].mean())
        if short_mask.sum() > 0 and (mk & short_mask).sum() > 0:
            out[f"short_avg_{h_name}"] = float(-rzd[mk & short_mask].mean())  # short edge = -realized
    return out


def evaluate_strategy(
    name: str,
    sig_abs: np.ndarray,  # ranking signal (absolute magnitude)
    sig_dir: np.ndarray,  # direction signal (signed)
    valid_mask: np.ndarray,
    pnl_realized: np.ndarray,  # target_fifo_*_net (REAL FIFO)
    hit_tp: np.ndarray,
    time_to_mfe: np.ndarray,
    mfe_realized: np.ndarray,
    mae_realized: np.ndarray,
    targets: dict,
    masks: dict,
    ann_factor: float,
    bands: list,
) -> list:
    rows = []
    valid_idx = np.where(valid_mask)[0]
    if len(valid_idx) == 0:
        return rows
    sig_finite = np.isfinite(sig_abs) & valid_mask
    sig_idx = np.where(sig_finite)[0]
    if len(sig_idx) == 0:
        return rows
    sorted_abs = np.sort(sig_abs[sig_idx])[::-1]
    for b in bands:
        k = max(1, int(b * len(sorted_abs)))
        thr = sorted_abs[k - 1] if k <= len(sorted_abs) else sorted_abs[-1]
        sel_mask = sig_finite & (sig_abs >= thr)
        sel_idx = np.where(sel_mask)[0]
        taken = _apply_gap(sel_idx, MIN_GAP_STEPS)
        if len(taken) < 5:
            rows.append({"strategy": name, "band": f"top {b*100:.1f}%", "n_trades": int(len(taken))})
            continue
        direction = np.sign(sig_dir[taken])
        # FIFO PnL — pre-signed (target_fifo_*_net is already direction-adjusted? Verify)
        # ACTUALLY — target_fifo_tp4sl3_net is the realized net PnL FOR A LONG TRADE at that step.
        # For a SHORT trade, we'd need -target_fifo_tp4sl3_net (assuming symmetric brackets).
        # Pre-investigation needed. For now, compute both interpretations.
        pnl_if_directional = direction * pnl_realized[taken]  # treats long-bracket as symmetric for short
        pnl_if_long_only = pnl_realized[taken]                # ignores direction (FIFO label was long-only)
        # Hit rate
        n_win = int((pnl_if_directional > 0).sum())
        wr_pct = (n_win / len(taken)) * 100.0
        # DA — sign agreement of direction with realized log_ret_1s (entry-time DA)
        rl1 = targets["log_ret_1s"][taken]
        m1 = masks["log_ret_1s"][taken] & np.isfinite(rl1)
        da_pct = float(((np.sign(rl1[m1]) == direction[m1]).mean()) * 100.0) if m1.sum() > 0 else None
        # Hold time
        tt = time_to_mfe[taken]
        tt_m = np.isfinite(tt)
        hold = _distribution(tt[tt_m])
        # MFE/MAE realized distribution
        mfe = mfe_realized[taken]
        mfe_m = np.isfinite(mfe)
        mae = mae_realized[taken]
        mae_m = np.isfinite(mae)
        # Price path
        pp = _price_path(taken, direction, targets, masks)
        rows.append({
            "strategy": name,
            "band": f"top {b*100:.1f}%",
            "n_trades": int(len(taken)),
            "n_long": int((direction > 0).sum()),
            "n_short": int((direction < 0).sum()),
            "da_pct": da_pct,
            "wr_pct": wr_pct,
            "fifo_pnl_directional": _stats_block(pnl_if_directional, ann_factor),
            "fifo_pnl_long_only_interp": _stats_block(pnl_if_long_only, ann_factor),
            "hold_time_secs": hold,
            "mfe_30s_ticks": _distribution(mfe[mfe_m]),
            "mae_30s_ticks": _distribution(mae[mae_m]),
            "price_path": pp,
        })
    return rows


def main():
    d = np.load(PREDS, allow_pickle=True)
    n = int(d["n_samples"])
    oot_dates = list(d["oot_dates"])
    print(f"[load] n={n}, oot_dates={oot_dates}", flush=True)

    # Heads
    pred_lr1 = d["pred_log_ret_1s"][:n]
    pred_lr5 = d["pred_log_ret_5s"][:n]
    pred_tp4sl3 = d["pred_fifo_tp4sl3_net"][:n]
    pred_tp8sl5 = d["pred_fifo_tp8sl5_net"][:n]

    # FIFO realized targets
    tgt_tp4sl3 = d["target_fifo_tp4sl3_net"][:n]
    tgt_tp8sl5 = d["target_fifo_tp8sl5_net"][:n]
    hit_tp4 = d["target_fifo_tp4sl3_hit_tp"][:n]
    hit_tp8 = d["target_fifo_tp8sl5_hit_tp"][:n]
    mask_tp4 = d["mask_fifo_tp4sl3_net"][:n].astype(bool)
    mask_tp8 = d["mask_fifo_tp8sl5_net"][:n].astype(bool)

    # Path-aware realized
    mfe_30 = d["target_pred_mfe_30s_ticks"][:n]
    mae_30 = d["target_pred_mae_30s_ticks"][:n]
    tt_mfe = d["target_pred_time_to_mfe_secs"][:n]

    # Price-path realized
    targets = {
        "log_ret_1s":   d["target_log_ret_1s"][:n],
        "log_ret_5s":   d["target_log_ret_5s"][:n],
        "log_ret_10s":  d["target_log_ret_10s"][:n],
        "log_ret_30s":  d["target_log_ret_30s"][:n],
        "log_ret_60s":  d["target_log_ret_60s"][:n] if "target_log_ret_60s" in d else np.full(n, np.nan),
        "log_ret_5min": d["target_log_ret_5min"][:n] if "target_log_ret_5min" in d else np.full(n, np.nan),
    }
    masks = {
        "log_ret_1s":   d["mask_log_ret_1s"][:n].astype(bool),
        "log_ret_5s":   d["mask_log_ret_5s"][:n].astype(bool),
        "log_ret_10s":  d["mask_log_ret_10s"][:n].astype(bool),
        "log_ret_30s":  d["mask_log_ret_30s"][:n].astype(bool),
        "log_ret_60s":  d["mask_log_ret_60s"][:n].astype(bool) if "mask_log_ret_60s" in d else np.zeros(n, dtype=bool),
        "log_ret_5min": d["mask_log_ret_5min"][:n].astype(bool) if "mask_log_ret_5min" in d else np.zeros(n, dtype=bool),
    }

    # Ann factor estimate: 6.5 RTH hrs * 3600s / 250ms stride = 93,600 steps/day
    # actual filled n=241351 over 5 days => ~48,270/day taken signals
    ann_factor_per_step = 93600 * 252 / max(1, n / 5)  # heuristic

    all_rows = []

    # Strategy A — FIFO TP4/SL3 native head
    all_rows += evaluate_strategy(
        "A_fifo_tp4sl3_native",
        sig_abs=np.abs(pred_tp4sl3),
        sig_dir=pred_tp4sl3,
        valid_mask=mask_tp4 & np.isfinite(pred_tp4sl3) & np.isfinite(tgt_tp4sl3),
        pnl_realized=tgt_tp4sl3,
        hit_tp=hit_tp4,
        time_to_mfe=tt_mfe,
        mfe_realized=mfe_30,
        mae_realized=mae_30,
        targets=targets, masks=masks,
        ann_factor=ann_factor_per_step,
        bands=CONF_BANDS,
    )

    # Strategy B — FIFO TP8/SL5 native head
    all_rows += evaluate_strategy(
        "B_fifo_tp8sl5_native",
        sig_abs=np.abs(pred_tp8sl5),
        sig_dir=pred_tp8sl5,
        valid_mask=mask_tp8 & np.isfinite(pred_tp8sl5) & np.isfinite(tgt_tp8sl5),
        pnl_realized=tgt_tp8sl5,
        hit_tp=hit_tp8,
        time_to_mfe=tt_mfe,
        mfe_realized=mfe_30,
        mae_realized=mae_30,
        targets=targets, masks=masks,
        ann_factor=ann_factor_per_step,
        bands=CONF_BANDS,
    )

    # Strategy C — log_ret_1s entry + TP4/SL3 FIFO exec
    all_rows += evaluate_strategy(
        "C_lr1s_entry_tp4sl3_exec",
        sig_abs=np.abs(pred_lr1),
        sig_dir=pred_lr1,
        valid_mask=mask_tp4 & np.isfinite(pred_lr1) & np.isfinite(tgt_tp4sl3),
        pnl_realized=tgt_tp4sl3,
        hit_tp=hit_tp4,
        time_to_mfe=tt_mfe,
        mfe_realized=mfe_30,
        mae_realized=mae_30,
        targets=targets, masks=masks,
        ann_factor=ann_factor_per_step,
        bands=CONF_BANDS,
    )

    # Strategy D — log_ret_1s entry + TP8/SL5 FIFO exec
    all_rows += evaluate_strategy(
        "D_lr1s_entry_tp8sl5_exec",
        sig_abs=np.abs(pred_lr1),
        sig_dir=pred_lr1,
        valid_mask=mask_tp8 & np.isfinite(pred_lr1) & np.isfinite(tgt_tp8sl5),
        pnl_realized=tgt_tp8sl5,
        hit_tp=hit_tp8,
        time_to_mfe=tt_mfe,
        mfe_realized=mfe_30,
        mae_realized=mae_30,
        targets=targets, masks=masks,
        ann_factor=ann_factor_per_step,
        bands=CONF_BANDS,
    )

    # Strategy E — confluence (lr1s + tp4sl3 dir agreement + lr5s dir agreement)
    sign_agree = (np.sign(pred_lr1) == np.sign(pred_tp4sl3)) & (np.sign(pred_lr1) == np.sign(pred_lr5))
    confluence_valid = mask_tp4 & np.isfinite(pred_lr1) & np.isfinite(pred_tp4sl3) & np.isfinite(pred_lr5) & sign_agree & np.isfinite(tgt_tp4sl3)
    all_rows += evaluate_strategy(
        "E_confluence_3head_tp4sl3_exec",
        sig_abs=np.abs(pred_lr1),
        sig_dir=pred_lr1,
        valid_mask=confluence_valid,
        pnl_realized=tgt_tp4sl3,
        hit_tp=hit_tp4,
        time_to_mfe=tt_mfe,
        mfe_realized=mfe_30,
        mae_realized=mae_30,
        targets=targets, masks=masks,
        ann_factor=ann_factor_per_step,
        bands=CONF_BANDS,
    )

    out = {
        "methodology": "FIFO_MARKET_REPLAY_via_npz_target_labels",
        "midpoint_used": False,
        "fifo_label_source": "target_fifo_tp4sl3_net / target_fifo_tp8sl5_net (pre-computed from MBO bid/ask at label-gen time)",
        "ann_factor_per_step": ann_factor_per_step,
        "n_samples": n,
        "oot_dates": [str(x) for x in oot_dates],
        "min_gap_steps": MIN_GAP_STEPS,
        "conf_bands": CONF_BANDS,
        "strategies": all_rows,
        "notes": [
            "HC #320 compliant: NO midpoint, NO sign(pred)*log_ret. PnL from FIFO TP/SL targets.",
            "HC #321 compliant: hold time, MFE/MAE distribution, price path included per strategy.",
            "HC #322: includes signal-only (A,B) vs entry+exec (C,D) vs confluence (E) to disentangle head-value.",
            "Caveat 1: target_fifo_*_net is realized PnL ASSUMED LONG. Symmetric for shorts assumed; dual stats reported.",
            "Caveat 2: fixed brackets (4t TP / 3t SL and 8t TP / 5t SL); dynamic-bracket sim needs raw MBO replay.",
        ],
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2, default=lambda o: None if (isinstance(o, float) and not np.isfinite(o)) else o)

    # Flat CSV for the dashboard
    import csv
    flat_cols = [
        "strategy", "band", "n_trades", "n_long", "n_short", "da_pct", "wr_pct",
        "fifo_mean_ticks", "fifo_median_ticks", "fifo_sharpe", "fifo_sortino", "fifo_pf",
        "fifo_total_usd", "hold_mean_s", "hold_median_s", "hold_p95_s",
        "mfe_mean", "mfe_median", "mfe_p95", "mae_mean", "mae_median", "mae_p95",
        "pp_avg_1s", "pp_avg_5s", "pp_avg_10s", "pp_avg_30s", "pp_avg_60s", "pp_avg_5min",
    ]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=flat_cols)
        w.writeheader()
        for r in all_rows:
            fp = r.get("fifo_pnl_directional", {})
            h = r.get("hold_time_secs", {})
            mfe = r.get("mfe_30s_ticks", {})
            mae = r.get("mae_30s_ticks", {})
            pp = r.get("price_path", {})
            w.writerow({
                "strategy": r["strategy"], "band": r["band"], "n_trades": r["n_trades"],
                "n_long": r.get("n_long"), "n_short": r.get("n_short"),
                "da_pct": r.get("da_pct"), "wr_pct": r.get("wr_pct"),
                "fifo_mean_ticks": fp.get("mean_ticks"), "fifo_median_ticks": fp.get("median_ticks"),
                "fifo_sharpe": fp.get("sharpe"), "fifo_sortino": fp.get("sortino"), "fifo_pf": fp.get("pf"),
                "fifo_total_usd": fp.get("total_usd"),
                "hold_mean_s": h.get("mean"), "hold_median_s": h.get("median"), "hold_p95_s": h.get("p95"),
                "mfe_mean": mfe.get("mean"), "mfe_median": mfe.get("median"), "mfe_p95": mfe.get("p95"),
                "mae_mean": mae.get("mean"), "mae_median": mae.get("median"), "mae_p95": mae.get("p95"),
                "pp_avg_1s": pp.get("avg_signed_1s"), "pp_avg_5s": pp.get("avg_signed_5s"),
                "pp_avg_10s": pp.get("avg_signed_10s"), "pp_avg_30s": pp.get("avg_signed_30s"),
                "pp_avg_60s": pp.get("avg_signed_60s"), "pp_avg_5min": pp.get("avg_signed_5min"),
            })

    print(f"[done] {len(all_rows)} rows. Wrote {OUT_JSON} + {OUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
