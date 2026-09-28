#!/usr/bin/env python3
"""
Branch A — 10s Candidate 4-Branch Validation (HC #491 R5 durable, HC #448 R2 closest-to-profit)

Validates the CNN-Mamba v3.4.2 top-1% SHORT @ 10s horizon candidate against:
  (B') bucket-confluence — top-1% short by pred_log_ret_10s
  (C') canonical FIFO replay — uses precomputed target_fifo_tp4sl3_net / tp8sl5_net (HC #74)
  (D') time-of-day — stratified by sample_idx-within-day proxy
  (E') per-day Sharpe + green/red regime split (HC #428 R1)
  Plus MFE-within-horizon check (HC #428 R2) using target_pred_time_to_mfe_secs ≤ 10s gate.

Inputs: output/cnn_mamba_v3_4_2_fixedmtl/oot_47day_perdate/oot_*.npz (existing, 34 dates)
Output: output/branch_A_10s_validation_v1/{summary.json, headline.txt, perday.csv}

Idempotent. Atomic write (tmp + rename). Single file. No external deps beyond numpy.

Cost constant: 0.376 ticks passive commission (HC ES_RT_COMMISSION_TICKS).
"""
import json
import os
import sys
from glob import glob

import numpy as np

ROOT = "/home/jupiter/Lvl3Quant"
IN_DIR = f"{ROOT}/output/cnn_mamba_v3_4_2_fixedmtl/oot_47day_perdate"
OUT_DIR = f"{ROOT}/output/branch_A_10s_validation_v1"
COST_PASSIVE_TICKS = 0.376
TICKS_PER_LOG_RET = 20000.0  # ES ~5000 * (1tick/0.25pt) — approximate; FIFO field is authoritative
TOP_PCT = 0.01
HORIZON_SECS = 10


def load_all():
    files = sorted(glob(f"{IN_DIR}/oot_*.npz"))
    if not files:
        sys.exit(f"No oot_*.npz files found in {IN_DIR}")
    arrs = {}
    for f in files:
        d = np.load(f, allow_pickle=True)
        for k in d.keys():
            if k in ("metrics_loss",) or k.startswith("metric_"):
                continue
            arrs.setdefault(k, []).append(d[k])
    out = {}
    for k, v in arrs.items():
        try:
            out[k] = np.concatenate(v)
        except Exception:
            out[k] = np.array(v)
    return out, files


def short_top1_filter(pred_10s, mask_10s, sample_dates=None, per_day=False):
    """Select samples in bottom 1% of pred_log_ret_10s (most-negative = strongest short signal).
    If per_day=True, threshold is computed per OOT date (matches Branch A horizon analysis)."""
    valid = mask_10s.astype(bool)
    if valid.sum() == 0:
        return np.zeros_like(pred_10s, dtype=bool)
    if not per_day:
        valid_preds = pred_10s[valid]
        cutoff = np.quantile(valid_preds, TOP_PCT)
        return valid & (pred_10s <= cutoff)
    # per-day threshold
    sel = np.zeros_like(pred_10s, dtype=bool)
    for d in np.unique(sample_dates):
        m = (sample_dates == d) & valid
        if m.sum() == 0:
            continue
        thr = np.quantile(pred_10s[m], TOP_PCT)
        sel |= m & (pred_10s <= thr)
    return sel


def realized_short_ticks(target_log_ret, sel):
    """Realized SHORT P&L in ticks: short profits when ret < 0 → P&L = -ret * tick_factor."""
    rets = target_log_ret[sel]
    rets = rets[~np.isnan(rets)]
    if len(rets) == 0:
        return np.array([])
    return -rets * TICKS_PER_LOG_RET


def sharpe(ticks_arr):
    if len(ticks_arr) < 2:
        return 0.0
    s = ticks_arr.std(ddof=1)
    return float(ticks_arr.mean() / s * np.sqrt(252)) if s > 1e-9 else 0.0


def regime_classify_es_perday(sample_dates):
    """Classify each unique date as green/red using fallback heuristic.
    Without external ES close data here we use the within-day target_log_ret_10s mean sign as proxy.
    Returns dict {date: 'green'|'red'|'flat'}."""
    return None  # delegated below — use per-day net P&L sign of the FILTER as proxy with caveat


def per_day_breakdown(sample_dates, ticks_net, fifo_net):
    """Aggregate net ticks per OOT date for the SELECTED short trades."""
    rows = []
    for date in np.unique(sample_dates):
        m = sample_dates == date
        if m.sum() == 0:
            continue
        d_raw = ticks_net[m] if len(ticks_net) == len(sample_dates) else None
        d_fifo = fifo_net[m] if (fifo_net is not None and len(fifo_net) == len(sample_dates)) else None
        n_trades_day = int(np.isfinite(d_raw).sum()) if d_raw is not None else 0
        rows.append({
            "date": str(date),
            "n_trades": n_trades_day,
            "raw_net_ticks_mean": float(np.nanmean(d_raw)) if d_raw is not None and np.isfinite(d_raw).any() else 0.0,
            "raw_net_ticks_sum": float(np.nansum(d_raw)) if d_raw is not None and np.isfinite(d_raw).any() else 0.0,
            "fifo_net_ticks_mean": float(np.nanmean(d_fifo)) if d_fifo is not None and len(d_fifo) and np.isfinite(d_fifo).any() else None,
            "fifo_net_ticks_sum": float(np.nansum(d_fifo)) if d_fifo is not None and len(d_fifo) and np.isfinite(d_fifo).any() else 0.0,
            "fifo_n_filled": int(np.isfinite(d_fifo).sum()) if d_fifo is not None else 0,
        })
    return rows


def time_of_day_stratify(sample_dates, ticks_net):
    """Bucket by within-day sample index (proxy for time-of-day in absence of explicit timestamps).
    Splits each day into 6 equal-time buckets."""
    out = {b: [] for b in range(6)}
    for date in np.unique(sample_dates):
        m = sample_dates == date
        idxs = np.where(m)[0]
        if len(idxs) == 0:
            continue
        n = len(idxs)
        # bucket assignment
        bucket_of_sample = np.minimum(5, (np.arange(n) * 6 // n))
        ticks_in_day = ticks_net[idxs]  # already filtered length = total n_samples, but we passed sel-aligned
        for i, b in enumerate(bucket_of_sample):
            if i < len(ticks_in_day) and np.isfinite(ticks_in_day[i]):
                out[int(b)].append(float(ticks_in_day[i]))
    return {f"bucket_{k}": {"n": len(v), "mean_ticks": float(np.mean(v)) if v else 0.0,
                              "wr_pct": float(100 * np.mean(np.array(v) > 0)) if v else 0.0}
            for k, v in out.items()}


def mfe_within_horizon_check(time_to_mfe, mfe_30s, mask_mfe, mask_t2m, sel):
    """HC #428 R2: MFE realized within HORIZON_SECS only. Returns p50/p90 of MFE among those samples."""
    valid = sel & mask_mfe.astype(bool) & mask_t2m.astype(bool)
    if not valid.any():
        return None
    within = valid & (time_to_mfe <= HORIZON_SECS)
    mfes = mfe_30s[within]
    mfes = mfes[np.isfinite(mfes)]
    if len(mfes) == 0:
        return None
    return {
        "n_samples_with_mfe_within_horizon": int(len(mfes)),
        "n_total_selected": int(sel.sum()),
        "frac_mfe_within_horizon": float(len(mfes) / max(1, sel.sum())),
        "mfe_p50_ticks": float(np.quantile(mfes, 0.50)),
        "mfe_p90_ticks": float(np.quantile(mfes, 0.90)),
        "tp_recommended_cap": float(np.quantile(mfes, 0.90)),  # HC #428 R2 TP cap
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"[load] reading {IN_DIR}/oot_*.npz ...", flush=True)
    arrs, files = load_all()
    print(f"[load] {len(files)} files, total samples: {len(arrs['pred_log_ret_10s'])}", flush=True)

    pred_10s = arrs["pred_log_ret_10s"]
    target_10s = arrs["target_log_ret_10s"]
    mask_10s = arrs["mask_log_ret_10s"]
    fifo_net_tp4sl3 = arrs["target_fifo_tp4sl3_net"]
    mask_fifo_tp4 = arrs["mask_fifo_tp4sl3_net"]
    fifo_net_tp8sl5 = arrs["target_fifo_tp8sl5_net"]
    mask_fifo_tp8 = arrs["mask_fifo_tp8sl5_net"]
    sample_dates = arrs["sample_dates"]
    time_to_mfe = arrs["target_pred_time_to_mfe_secs"]
    mask_t2m = arrs["mask_pred_time_to_mfe_secs"]
    mfe_30s = arrs["target_pred_mfe_30s_ticks"]
    mask_mfe30 = arrs["mask_pred_mfe_30s_ticks"]

    # === Two filters: GLOBAL top-1% and PER-DAY top-1% (Branch A method) ===
    sel_global = short_top1_filter(pred_10s, mask_10s, per_day=False)
    sel_perday = short_top1_filter(pred_10s, mask_10s, sample_dates=sample_dates, per_day=True)
    print(f"[filter] global top-1% short: {int(sel_global.sum())} | per-day top-1% short: {int(sel_perday.sum())}", flush=True)
    sel = sel_perday  # primary analysis uses per-day to match Branch A
    n_sel = int(sel.sum())

    # === Raw realized return (in ticks, log-ret * 20000) ===
    raw_ticks_per_trade = realized_short_ticks(target_10s, sel)
    raw_net_mean = float(raw_ticks_per_trade.mean() - COST_PASSIVE_TICKS) if len(raw_ticks_per_trade) else 0.0
    raw_wr = float(100 * (raw_ticks_per_trade > COST_PASSIVE_TICKS).mean()) if len(raw_ticks_per_trade) else 0.0

    # === FIFO canonical replay (HC #74) ===
    fifo_sel = sel & mask_fifo_tp4.astype(bool)
    fifo_pnl_tp4 = fifo_net_tp4sl3[fifo_sel]
    fifo_pnl_tp4 = fifo_pnl_tp4[np.isfinite(fifo_pnl_tp4)]
    fifo_net_tp4_mean = float(fifo_pnl_tp4.mean() - COST_PASSIVE_TICKS) if len(fifo_pnl_tp4) else None

    fifo_sel8 = sel & mask_fifo_tp8.astype(bool)
    fifo_pnl_tp8 = fifo_net_tp8sl5[fifo_sel8]
    fifo_pnl_tp8 = fifo_pnl_tp8[np.isfinite(fifo_pnl_tp8)]
    fifo_net_tp8_mean = float(fifo_pnl_tp8.mean() - COST_PASSIVE_TICKS) if len(fifo_pnl_tp8) else None

    # === Per-day breakdown ===
    # Align per-day arrays: ticks for ALL samples (so per-day can index by sample_dates)
    raw_ticks_all = -target_10s * TICKS_PER_LOG_RET  # short P&L
    raw_ticks_all = np.where(sel, raw_ticks_all - COST_PASSIVE_TICKS, np.nan)
    fifo_net_all = np.where(sel & mask_fifo_tp4.astype(bool),
                            fifo_net_tp4sl3 - COST_PASSIVE_TICKS, np.nan)
    perday = per_day_breakdown(sample_dates, raw_ticks_all, fifo_net_all)

    # Green/red regime: use per-day FIFO net sign (canonical, unit-correct)
    # Fallback to raw if no FIFO fills on that day
    def day_sign(r):
        v = r["fifo_net_ticks_sum"] if r["fifo_n_filled"] > 0 else r["raw_net_ticks_sum"]
        return v
    greens = [r for r in perday if day_sign(r) > 0 and r["n_trades"] > 0]
    reds = [r for r in perday if day_sign(r) < 0 and r["n_trades"] > 0]
    flats = [r for r in perday if day_sign(r) == 0 and r["n_trades"] > 0]
    g_mean = float(np.mean([r["raw_net_ticks_mean"] for r in greens])) if greens else 0.0
    r_mean = float(np.mean([r["raw_net_ticks_mean"] for r in reds])) if reds else 0.0
    regime_imbalance = float(abs(g_mean - r_mean) / max(abs(g_mean), abs(r_mean), 1e-9)) if (greens or reds) else 0.0

    # Day-concentration check (HC #344): max single day's contribution to total
    day_sums = np.array([r["raw_net_ticks_sum"] for r in perday])
    total_abs = float(np.sum(np.abs(day_sums)))
    day_conc = float(np.max(np.abs(day_sums)) / total_abs) if total_abs > 0 else 0.0

    # Per-day Sharpe (annualized, using daily means)
    daily_means = np.array([r["raw_net_ticks_mean"] for r in perday if r["n_trades"] > 0])
    perday_sharpe = sharpe(daily_means) if len(daily_means) > 1 else 0.0

    # === Time of day stratification ===
    tod = time_of_day_stratify(sample_dates, raw_ticks_all)

    # === MFE within horizon ===
    mfe_check = mfe_within_horizon_check(time_to_mfe, mfe_30s, mask_mfe30, mask_t2m, sel)

    # === Verdict ===
    gates = {
        "HC_428_R1_min_40d": len([r for r in perday if r["n_trades"] > 0]) >= 40,
        "HC_428_R1_regime_balanced": regime_imbalance <= 0.50,
        "HC_344_day_conc_cap": day_conc <= 0.70,
        "HC_74_FIFO_positive": (fifo_net_tp4_mean is not None and fifo_net_tp4_mean > 0),
        "raw_edge_positive": raw_net_mean > 0,
    }
    verdict = "ACCEPT" if all(gates.values()) else "REJECT"
    rejection_reasons = [k for k, v in gates.items() if not v]

    # Also compute FIFO for global filter as sanity comparator
    fifo_global_sel4 = sel_global & mask_fifo_tp4.astype(bool)
    fg4 = fifo_net_tp4sl3[fifo_global_sel4]
    fg4 = fg4[np.isfinite(fg4)]
    fifo_global_tp4_mean = float(fg4.mean() - COST_PASSIVE_TICKS) if len(fg4) else None

    summary = {
        "candidate": "CNN-Mamba v3.4.2 top-1% SHORT @ 10s horizon (per-day filter, matches Branch A)",
        "n_files_loaded": len(files),
        "n_total_samples": int(len(pred_10s)),
        "n_trades_selected_perday": n_sel,
        "n_trades_selected_global": int(sel_global.sum()),
        "n_days_with_trades": len([r for r in perday if r["n_trades"] > 0]),
        "n_days_total": len(perday),
        "fifo_global_tp4sl3_net_sanity": fifo_global_tp4_mean,
        "cost_passive_ticks": COST_PASSIVE_TICKS,
        "raw_pnl": {
            "net_ticks_per_trade": raw_net_mean,
            "win_rate_pct": raw_wr,
            "n_evaluated": int(len(raw_ticks_per_trade)),
        },
        "fifo_canonical_replay_HC74": {
            "tp4sl3": {
                "net_ticks_per_trade": fifo_net_tp4_mean,
                "n_filled_trades": int(len(fifo_pnl_tp4)),
                "fill_rate_pct": float(100 * len(fifo_pnl_tp4) / max(1, n_sel)),
            },
            "tp8sl5": {
                "net_ticks_per_trade": fifo_net_tp8_mean,
                "n_filled_trades": int(len(fifo_pnl_tp8)),
                "fill_rate_pct": float(100 * len(fifo_pnl_tp8) / max(1, n_sel)),
            },
        },
        "regime_HC428_R1": {
            "n_green_days": len(greens),
            "n_red_days": len(reds),
            "n_flat_days": len(flats),
            "green_mean_ticks": g_mean,
            "red_mean_ticks": r_mean,
            "regime_imbalance": regime_imbalance,
            "day_concentration": day_conc,
            "per_day_sharpe_ann": perday_sharpe,
        },
        "time_of_day_buckets": tod,
        "mfe_within_horizon_HC428_R2": mfe_check,
        "gates": gates,
        "rejection_reasons": rejection_reasons,
        "verdict": verdict,
    }

    # === Atomic write ===
    tmp_json = f"{OUT_DIR}/summary.json.tmp"
    tmp_head = f"{OUT_DIR}/headline.txt.tmp"
    tmp_csv = f"{OUT_DIR}/perday.csv.tmp"

    with open(tmp_json, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    with open(tmp_head, "w") as f:
        f.write(f"Branch A 10s Validation — {summary['candidate']}\n")
        f.write(f"Trades: {n_sel} | Days w/trades: {summary['n_days_with_trades']}\n\n")
        f.write(f"RAW net ticks/trade: {raw_net_mean:+.3f} | WR: {raw_wr:.1f}%\n")
        if fifo_net_tp4_mean is not None:
            f.write(f"FIFO tp4sl3 net ticks/trade: {fifo_net_tp4_mean:+.3f} | filled: {len(fifo_pnl_tp4)} ({100*len(fifo_pnl_tp4)/max(1,n_sel):.0f}%)\n")
        if fifo_net_tp8_mean is not None:
            f.write(f"FIFO tp8sl5 net ticks/trade: {fifo_net_tp8_mean:+.3f} | filled: {len(fifo_pnl_tp8)} ({100*len(fifo_pnl_tp8)/max(1,n_sel):.0f}%)\n")
        f.write(f"\nGreen days: {len(greens)} @ {g_mean:+.3f} | Red days: {len(reds)} @ {r_mean:+.3f}\n")
        f.write(f"Regime imbalance: {regime_imbalance:.2f} (cap 0.50) | Day conc: {day_conc:.2f} (cap 0.70)\n")
        f.write(f"Per-day Sharpe (ann): {perday_sharpe:.2f}\n")
        if mfe_check:
            f.write(f"\nMFE within 10s horizon: p50={mfe_check['mfe_p50_ticks']:.2f}t p90={mfe_check['mfe_p90_ticks']:.2f}t ({100*mfe_check['frac_mfe_within_horizon']:.0f}% of trades)\n")
        f.write(f"\nVERDICT: {verdict}\n")
        if rejection_reasons:
            f.write(f"Rejection reasons: {', '.join(rejection_reasons)}\n")
    with open(tmp_csv, "w") as f:
        f.write("date,n_trades,raw_net_mean,raw_net_sum,fifo_net_mean,fifo_n_filled\n")
        for r in perday:
            f.write(f"{r['date']},{r['n_trades']},{r['raw_net_ticks_mean']:.4f},{r['raw_net_ticks_sum']:.4f},"
                    f"{r['fifo_net_ticks_mean'] if r['fifo_net_ticks_mean'] is not None else ''},{r['fifo_n_filled']}\n")

    os.replace(tmp_json, f"{OUT_DIR}/summary.json")
    os.replace(tmp_head, f"{OUT_DIR}/headline.txt")
    os.replace(tmp_csv, f"{OUT_DIR}/perday.csv")
    print(f"[write] {OUT_DIR}/{{summary.json,headline.txt,perday.csv}}", flush=True)
    print(f"[verdict] {verdict}")
    if rejection_reasons:
        print(f"[reject] {rejection_reasons}")


if __name__ == "__main__":
    main()
