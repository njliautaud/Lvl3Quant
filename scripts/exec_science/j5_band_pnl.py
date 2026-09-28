#!/usr/bin/env python3
"""
J5 — BAND-PnL SIMULATION on v3.3 5-day OOT predictions.

HC #388: "performance is deeper than raw IC there's tons of outputs now that
are very crucial. And at confidence bands only"

For each SIGNED head (log_ret_*, fifo_*_net) at each confidence band:
- Take signed predictions; bin by top-X% confidence (|pred| as confidence)
- For each sample: virtual trade direction = sign(pred), realized = target (already in target units)
- Compute trade-stats:
    - n_long, n_short, total trades, % of all eligible samples
    - WR long/short, mean PnL long/short (in target units)
    - net PnL gross, net PnL after passive-limit cost (0.376 ticks RT), net PnL after market cost (1.376 ticks RT)
    - Sharpe-style: mean / std of PnL series
    - Sortino: mean / std of negative PnL
    - Profit factor: sum(positive) / |sum(negative)|
    - Max drawdown of cumulative PnL

NOTE: log_ret targets are in LOG RETURN units (not ticks). To convert log_ret to ticks
for ES at ~5000 level: 1 tick = 0.25/5000 = 5e-5 in log_ret terms. So log_ret*20000 ≈ ticks.
We multiply log_ret targets by SCALE (configurable, default 20000) so cost can be subtracted in ticks.

For fifo_*_net heads: target is ALREADY in net ticks (HC #75 5-component FIFO sim). No scaling.

Output: j5_band_pnl_ladder.csv + j5_summary.txt
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np


CONF_BANDS = [50.0, 25.0, 10.0, 5.0, 1.0, 0.5, 0.1]

PASSIVE_COST_TICKS = 0.376   # commission only (limit-at-bid/ask)
MARKET_COST_TICKS = 1.376    # commission + 1 tick spread crossing

# Heads whose target is in TICKS (no scaling)
HEADS_TICKS = {"fifo_tp4sl3_net", "fifo_tp8sl5_net"}

# Heads whose target is LOG_RETURN (multiply by LOG2TICK_SCALE)
HEADS_LOG = {"log_ret_1s","log_ret_5s","log_ret_10s","log_ret_30s","log_ret_60s","log_ret_5min",
             "log_ret_10s_q50","log_ret_30s_q50","log_ret_60s_q50"}

# Quantile heads — q10/q90 — directional predict but we evaluate sign vs sign of underlying log_ret
# We skip those for direct PnL since target is the q-truth (median or quantile)


def trade_stats(p, t, scale_to_ticks):
    """p = pred (signed), t = target. Returns dict of stats with cost variants."""
    n = len(p)
    if n < 50:
        return None
    pos = p > 0
    neg = p < 0
    # in target units → convert to ticks
    pnl_ticks = np.sign(p) * t * scale_to_ticks  # if pred>0 we long, profit = target*scale
    res = {
        "n_total": int(n),
        "n_long": int(pos.sum()),
        "n_short": int(neg.sum()),
        "gross_total_ticks": float(pnl_ticks.sum()),
        "gross_mean_ticks": float(pnl_ticks.mean()),
        "gross_std_ticks": float(pnl_ticks.std() + 1e-12),
        "gross_wr": float(np.mean(pnl_ticks > 0)),
    }
    # passive cost
    pnl_passive = pnl_ticks - PASSIVE_COST_TICKS
    res["passive_total_ticks"] = float(pnl_passive.sum())
    res["passive_mean_ticks"] = float(pnl_passive.mean())
    res["passive_wr"] = float(np.mean(pnl_passive > 0))
    res["passive_sharpe"] = float(pnl_passive.mean() / (pnl_passive.std() + 1e-12)) * np.sqrt(n)
    neg_pass = pnl_passive[pnl_passive < 0]
    res["passive_sortino"] = float(pnl_passive.mean() / (np.std(neg_pass) + 1e-12)) * np.sqrt(n) if neg_pass.size > 5 else float("nan")
    pos_p = pnl_passive[pnl_passive > 0].sum()
    neg_p = -pnl_passive[pnl_passive < 0].sum()
    res["passive_pf"] = float(pos_p / (neg_p + 1e-12))
    # cumulative drawdown (passive)
    cum = np.cumsum(pnl_passive)
    peak = np.maximum.accumulate(cum)
    res["passive_max_dd_ticks"] = float((peak - cum).max())
    # market cost
    pnl_market = pnl_ticks - MARKET_COST_TICKS
    res["market_total_ticks"] = float(pnl_market.sum())
    res["market_mean_ticks"] = float(pnl_market.mean())
    res["market_wr"] = float(np.mean(pnl_market > 0))
    res["market_sharpe"] = float(pnl_market.mean() / (pnl_market.std() + 1e-12)) * np.sqrt(n)
    neg_m = pnl_market[pnl_market < 0]
    res["market_sortino"] = float(pnl_market.mean() / (np.std(neg_m) + 1e-12)) * np.sqrt(n) if neg_m.size > 5 else float("nan")
    pos_m = pnl_market[pnl_market > 0].sum()
    neg_m_sum = -pnl_market[pnl_market < 0].sum()
    res["market_pf"] = float(pos_m / (neg_m_sum + 1e-12))
    # by side
    if pos.sum() >= 5:
        res["long_mean_ticks"] = float(pnl_ticks[pos].mean())
        res["long_wr"] = float(np.mean(pnl_ticks[pos] > 0))
    else:
        res["long_mean_ticks"] = float("nan")
        res["long_wr"] = float("nan")
    if neg.sum() >= 5:
        res["short_mean_ticks"] = float(pnl_ticks[neg].mean())
        res["short_wr"] = float(np.mean(pnl_ticks[neg] > 0))
    else:
        res["short_mean_ticks"] = float("nan")
        res["short_wr"] = float("nan")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="/home/jupiter/Lvl3Quant/output/cnn_mamba_v3_3_uncertainty_weighted/fold_00_predictions.npz")
    ap.add_argument("--outdir", default="/home/jupiter/Lvl3Quant/output/exec_science_v3_3_overnight")
    ap.add_argument("--log2tick-scale", type=float, default=20000.0,
                    help="Multiplier from log_return to ticks (ES ~5000 → 1 tick=5e-5 in log_ret).")
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    d = np.load(args.npz, allow_pickle=True)
    keys = set(d.keys())

    heads = sorted(HEADS_TICKS | HEADS_LOG)
    heads = [h for h in heads if f"pred_{h}" in keys and f"target_{h}" in keys]

    rows = []
    for h in heads:
        pred = np.asarray(d[f"pred_{h}"], dtype=np.float64)
        target = np.asarray(d[f"target_{h}"], dtype=np.float64)
        mask_key = f"mask_{h}"
        mask = np.asarray(d[mask_key], dtype=np.float64) if mask_key in keys else np.ones_like(pred)
        valid = (mask > 0) & np.isfinite(pred) & np.isfinite(target)
        if valid.sum() < 100:
            continue
        p = pred[valid]
        t = target[valid]
        conf = np.abs(p)  # confidence proxy = magnitude
        scale_to_ticks = 1.0 if h in HEADS_TICKS else args.log2tick_scale
        # 100% baseline + each band
        for band_pct in [100.0] + CONF_BANDS:
            if band_pct < 100.0:
                cutoff = np.percentile(conf, 100.0 - band_pct)
                sel = conf >= cutoff
                pp = p[sel]; tt = t[sel]
            else:
                pp = p; tt = t
            stats = trade_stats(pp, tt, scale_to_ticks)
            if stats is None:
                continue
            row = {"head": h, "band_pct": band_pct, "scale_to_ticks": scale_to_ticks}
            row.update(stats)
            rows.append(row)
        print(f"[J5] {h:32s}  done", flush=True)

    # CSV
    cols = ["head","band_pct","scale_to_ticks","n_total","n_long","n_short",
            "gross_total_ticks","gross_mean_ticks","gross_wr",
            "passive_total_ticks","passive_mean_ticks","passive_wr","passive_sharpe","passive_sortino","passive_pf","passive_max_dd_ticks",
            "market_total_ticks","market_mean_ticks","market_wr","market_sharpe","market_sortino","market_pf",
            "long_mean_ticks","long_wr","short_mean_ticks","short_wr"]
    with open(out / "j5_band_pnl_ladder.csv","w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(f"{r.get(c, '')}" for c in cols) + "\n")

    # summary
    lines = []
    lines.append("="*100)
    lines.append("J5 — BAND-PnL SIMULATION (v3.3 5-day OOT, 241,351 samples)")
    lines.append(f"     log_ret→tick scale = {args.log2tick_scale}, passive cost = 0.376t, market cost = 1.376t")
    lines.append("="*100)
    lines.append("")
    # Best heads by passive Sharpe at each band
    for band in [10.0, 5.0, 1.0, 0.5, 0.1]:
        lines.append(f"\n--- TOP HEADS @ TOP-{band}% CONF (PASSIVE COST) ---")
        bandrows = [r for r in rows if abs(r["band_pct"] - band) < 1e-6]
        bandrows.sort(key=lambda r: r["passive_total_ticks"], reverse=True)
        lines.append(f"{'head':28s} {'n':>6s} {'passive_total':>14s} {'mean_ticks':>11s} {'sharpe':>8s} {'sortino':>8s} {'pf':>6s} {'wr':>6s} {'maxDD':>8s}")
        for r in bandrows[:8]:
            lines.append(f"{r['head']:28s} {r['n_total']:>6d} {r['passive_total_ticks']:>14.1f} {r['passive_mean_ticks']:>11.4f} {r['passive_sharpe']:>8.2f} {r['passive_sortino']:>8.2f} {r['passive_pf']:>6.2f} {r['passive_wr']:>6.3f} {r['passive_max_dd_ticks']:>8.1f}")
    # Short-only stats for fifo_tp8sl5_net (best signal)
    lines.append("")
    lines.append("="*100)
    lines.append("FOCUS: fifo_tp8sl5_net — band ladder (passive cost)")
    lines.append("="*100)
    lines.append(f"{'band':>6s} {'n':>7s} {'pTotal':>10s} {'pMean':>8s} {'pSharpe':>8s} {'pSortino':>9s} {'pPF':>6s} {'pWR':>6s} {'shortWR':>8s} {'longWR':>8s}")
    target_h = "fifo_tp8sl5_net"
    for r in [r for r in rows if r["head"]==target_h]:
        lines.append(f"{r['band_pct']:>6.1f} {r['n_total']:>7d} {r['passive_total_ticks']:>10.1f} {r['passive_mean_ticks']:>8.3f} {r['passive_sharpe']:>8.2f} {r['passive_sortino']:>9.2f} {r['passive_pf']:>6.2f} {r['passive_wr']:>6.3f} {r.get('short_wr',float('nan')):>8.3f} {r.get('long_wr',float('nan')):>8.3f}")

    txt = "\n".join(lines)
    (out / "j5_summary.txt").write_text(txt)
    print(txt, flush=True)


if __name__ == "__main__":
    main()
