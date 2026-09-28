#!/usr/bin/env python3
"""
J6 — FIFO + signal CONFLUENCE strategy at confidence bands.

Premise: fifo_tp8sl5_net is the operational head (target = net ticks of an actual
TP/SL trade in FIFO market replay). At top-0.5% conf alone, WR=80.4%, mean +2.49t.

Question: Can we INCREASE PnL/WR by requiring AGREEMENT with other heads?
- fifo_tp8sl5_net.sign == fifo_tp4sl3_net.sign  (both FIFO heads agree)
- fifo_tp8sl5_net.sign == log_ret_5s.sign        (short signal agrees)
- fifo_tp8sl5_net.sign == log_ret_1s.sign        (faster signal agrees)
- 3-way: tp8sl5 ∧ log_ret_1s ∧ log_ret_5s
- 4-way: + log_ret_10s

For each combo at each conf band (1%, 0.5%, 0.1% of fifo_tp8sl5_net), report:
n, WR, mean_ticks, total_ticks, after passive cost, after market cost, Sharpe, Sortino.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

CONF_BANDS = [10.0, 5.0, 1.0, 0.5, 0.1]
PASSIVE = 0.376
MARKET = 1.376


def make_signs(d, head):
    pred = np.asarray(d[f"pred_{head}"], dtype=np.float64)
    target = np.asarray(d[f"target_{head}"], dtype=np.float64)
    mask = np.asarray(d[f"mask_{head}"], dtype=np.float64) if f"mask_{head}" in d.files else np.ones_like(pred)
    return pred, target, mask


def stats(pnl_ticks):
    if len(pnl_ticks) < 5:
        return None
    pnl_p = pnl_ticks - PASSIVE
    pnl_m = pnl_ticks - MARKET
    neg_p = pnl_p[pnl_p < 0]; pos_p = pnl_p[pnl_p > 0]
    neg_m = pnl_m[pnl_m < 0]; pos_m = pnl_m[pnl_m > 0]
    r = {
        "n": int(len(pnl_ticks)),
        "gross_total": float(pnl_ticks.sum()),
        "gross_mean": float(pnl_ticks.mean()),
        "gross_wr": float(np.mean(pnl_ticks > 0)),
        "passive_total": float(pnl_p.sum()),
        "passive_mean": float(pnl_p.mean()),
        "passive_wr": float(np.mean(pnl_p > 0)),
        "passive_sharpe": float(pnl_p.mean() / (pnl_p.std() + 1e-12)) * np.sqrt(len(pnl_p)),
        "passive_sortino": float(pnl_p.mean() / (np.std(neg_p) + 1e-12)) * np.sqrt(len(pnl_p)) if neg_p.size >= 3 else float("nan"),
        "passive_pf": float(pos_p.sum() / (-neg_p.sum() + 1e-12)) if neg_p.size > 0 else float("inf"),
        "market_total": float(pnl_m.sum()),
        "market_mean": float(pnl_m.mean()),
        "market_wr": float(np.mean(pnl_m > 0)),
        "market_sharpe": float(pnl_m.mean() / (pnl_m.std() + 1e-12)) * np.sqrt(len(pnl_m)),
        "market_sortino": float(pnl_m.mean() / (np.std(neg_m) + 1e-12)) * np.sqrt(len(pnl_m)) if neg_m.size >= 3 else float("nan"),
        "market_pf": float(pos_m.sum() / (-neg_m.sum() + 1e-12)) if neg_m.size > 0 else float("inf"),
    }
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="/home/jupiter/Lvl3Quant/output/cnn_mamba_v3_3_uncertainty_weighted/fold_00_predictions.npz")
    ap.add_argument("--outdir", default="/home/jupiter/Lvl3Quant/output/exec_science_v3_3_overnight")
    args = ap.parse_args()
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)

    d = np.load(args.npz, allow_pickle=True)
    keys = set(d.files)

    BASE = "fifo_tp8sl5_net"
    pred8, tgt8, mask8 = make_signs(d, BASE)
    base_valid = (mask8 > 0) & np.isfinite(pred8) & np.isfinite(tgt8)

    confluence_heads = ["fifo_tp4sl3_net", "log_ret_1s", "log_ret_5s", "log_ret_10s", "log_ret_30s"]
    other_signs = {}
    for h in confluence_heads:
        if f"pred_{h}" not in keys:
            continue
        p, _, m = make_signs(d, h)
        v = (m > 0) & np.isfinite(p)
        s = np.where(v, np.sign(p), 0).astype(np.int8)
        other_signs[h] = s

    # define combos
    combos = [
        ("base_only", []),
        ("+tp4sl3", ["fifo_tp4sl3_net"]),
        ("+log_ret_1s", ["log_ret_1s"]),
        ("+log_ret_5s", ["log_ret_5s"]),
        ("+log_ret_10s", ["log_ret_10s"]),
        ("+tp4sl3+lr1s", ["fifo_tp4sl3_net","log_ret_1s"]),
        ("+lr1s+lr5s", ["log_ret_1s","log_ret_5s"]),
        ("+lr1s+lr5s+lr10s", ["log_ret_1s","log_ret_5s","log_ret_10s"]),
        ("+tp4sl3+lr1s+lr5s", ["fifo_tp4sl3_net","log_ret_1s","log_ret_5s"]),
        ("ALL5", confluence_heads),
    ]

    conf = np.abs(pred8)
    rows = []
    for band_pct in CONF_BANDS:
        cutoff = np.percentile(conf[base_valid], 100.0 - band_pct)
        base_band = base_valid & (conf >= cutoff)
        sgn_base = np.sign(pred8)

        for combo_name, extras in combos:
            sel = base_band.copy()
            for h in extras:
                if h not in other_signs:
                    continue
                # require sign agreement (or extra head is in-band signed)
                sel = sel & (other_signs[h] == sgn_base) & (other_signs[h] != 0)
            n = int(sel.sum())
            if n < 5:
                continue
            pnl = np.sign(pred8[sel]) * tgt8[sel]  # fifo target is signed net ticks per trade
            st = stats(pnl)
            row = {"band_pct": band_pct, "combo": combo_name}
            row.update(st)
            rows.append(row)

    cols = ["band_pct","combo","n","gross_total","gross_mean","gross_wr",
            "passive_total","passive_mean","passive_wr","passive_sharpe","passive_sortino","passive_pf",
            "market_total","market_mean","market_wr","market_sharpe","market_sortino","market_pf"]
    with open(out/"j6_fifo_confluence.csv","w") as f:
        f.write(",".join(cols)+"\n")
        for r in rows:
            f.write(",".join(f"{r.get(c,'')}" for c in cols)+"\n")

    # summary
    lines = []
    lines.append("="*120)
    lines.append("J6 — FIFO + SIGNAL CONFLUENCE (base=fifo_tp8sl5_net, costs in ticks/RT)")
    lines.append("="*120)
    for band in CONF_BANDS:
        lines.append("")
        lines.append(f"--- BAND: top-{band}% of fifo_tp8sl5_net confidence ---")
        lines.append(f"{'combo':22s} {'n':>6s} {'gross':>8s} {'pTot':>8s} {'pMean':>7s} {'pWR':>6s} {'pSh':>6s} {'pSor':>7s} {'pPF':>6s} {'mTot':>8s} {'mMean':>7s} {'mWR':>6s} {'mSh':>6s} {'mPF':>6s}")
        bandrows = [r for r in rows if abs(r["band_pct"]-band)<1e-6]
        bandrows.sort(key=lambda r: r["passive_total"], reverse=True)
        for r in bandrows:
            lines.append(f"{r['combo']:22s} {r['n']:>6d} {r['gross_total']:>8.1f} {r['passive_total']:>8.1f} {r['passive_mean']:>7.3f} {r['passive_wr']:>6.3f} {r['passive_sharpe']:>6.2f} {r['passive_sortino']:>7.2f} {r['passive_pf']:>6.2f} {r['market_total']:>8.1f} {r['market_mean']:>7.3f} {r['market_wr']:>6.3f} {r['market_sharpe']:>6.2f} {r['market_pf']:>6.2f}")

    lines.append("")
    lines.append("="*120)
    lines.append("KEY READS:")
    lines.append("  pTot=passive total ticks (commission only, 0.376/RT)")
    lines.append("  mTot=market total ticks (commission + 1t spread, 1.376/RT)")
    lines.append("  Sharpe/Sortino use sqrt(N) — NOT tradeable per-period, just signal strength")
    lines.append("="*120)
    txt = "\n".join(lines)
    (out/"j6_summary.txt").write_text(txt)
    print(txt)

if __name__ == "__main__":
    main()
