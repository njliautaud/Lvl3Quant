#!/usr/bin/env python3
"""
HC #346 PASS 3 — Realistic exit timing + use the GROUND-TRUTH FIFO labels

Pass 2 finding: Long IOC with realistic TP/SL loses -2 to -3 ticks. The signal
direction is right (60-70% DA) but TP/SL exit timing isn't capturing it.

Hypotheses to test:
  H1. TIME-BASED exit beats PRICE-BASED (exit at fixed 1s/2s/5s, no TP/SL)
  H2. The fifo_tp4sl3_net / fifo_tp8sl5_net labels (which incorporate real
      MBO fill mechanics) tell the truth — slice by ALL filter variants
  H3. Multi-head AGREEMENT entry filter helps (require log_ret_1s, log_ret_5s,
      log_ret_10s same sign + p_up>0.55 + reversal<0.4)
  H4. Asymmetric reward: take entries where pred_mfe / pred_mae > 1.5
  H5. RL for EXIT decision (state=time-elapsed + current pnl + heads, action=hold/exit)

Outputs: output/v3_2_allnight_research_20260514/pass3_realistic/
"""
from __future__ import annotations

import json
import math
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path("/home/jupiter/Lvl3Quant")
PRED_NPZ = ROOT / "output/v3_2_deep_sim_20260512/fold_00_oot_predictions.npz"
OUT = ROOT / "output/v3_2_allnight_research_20260514/pass3_realistic"
OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "pass3.log"

MARKET_COST_TICKS = 1.376
PASSIVE_COST_TICKS = 0.376


def log(msg):
    ts = datetime.utcnow().isoformat(timespec="seconds")
    line = f"[{ts}Z] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


# ES log-return ≈ tick at price 5050 → 4.95e-5 per tick
TICK_LOG = 4.95e-5

def logret_to_ticks(x):
    return x / TICK_LOG


def main():
    log("PASS 3 — Realistic exits + ground-truth FIFO labels")
    d = np.load(PRED_NPZ, allow_pickle=True)
    n = int(d["n_samples"])

    p1 = d["pred_log_ret_1s"]
    p5 = d["pred_log_ret_5s"]
    p10 = d["pred_log_ret_10s"]
    pup5 = d["pred_p_up_5s"]
    pup10 = d["pred_p_up_10s"]
    rev15 = d["pred_p_reversal_15s"]
    rev30 = d["pred_p_reversal_30s"]
    pred_mfe = d["pred_pred_mfe_30s_ticks"]
    pred_mae = d["pred_pred_mae_30s_ticks"]
    pred_vol = d["pred_pred_realized_vol_30s_ticks"]

    msk = (d["mask_log_ret_1s"].astype(bool) & d["mask_log_ret_5s"].astype(bool)
           & d["mask_log_ret_10s"].astype(bool) & np.isfinite(p1) & np.isfinite(p5))

    # Ground-truth FIFO labels in ticks
    fnet43 = d["target_fifo_tp4sl3_net"]
    fmsk43 = d["mask_fifo_tp4sl3_net"].astype(bool)
    fnet85 = d["target_fifo_tp8sl5_net"]
    fmsk85 = d["mask_fifo_tp8sl5_net"].astype(bool)

    out_summary = {}

    # ─────── H1: TIME-BASED exit ───────
    log("H1. Time-based exits using log_ret targets")
    h1_rows = []
    long_mask = msk & (p1 > 0)
    long_idx = np.where(long_mask)[0]
    abs_p = np.abs(p1[long_idx])
    for bname, bfrac in [("Top0.5%", 0.005), ("Top1%", 0.01), ("Top5%", 0.05)]:
        n_take = max(5, int(long_idx.size * bfrac))
        cell = long_idx[np.argsort(-abs_p)[:n_take]]
        for hkey, hcol in [("1s","log_ret_1s"), ("5s","log_ret_5s"),
                           ("10s","log_ret_10s"), ("30s","log_ret_30s"),
                           ("60s","log_ret_60s")]:
            t = d[f"target_{hcol}"][cell]
            m = d[f"mask_{hcol}"].astype(bool)[cell]
            k = m & np.isfinite(t)
            if k.sum() < 5: continue
            # Realized move at this horizon, signed for long (pred>0). Convert z-score → ticks
            # using sqrt(seconds) anchor at 30s sd_mfe~5 ticks
            hsec = {"1s":1,"5s":5,"10s":10,"30s":30,"60s":60}[hkey]
            sd_anchor_30s = 5.0  # ticks (1-sigma 30s realized move ≈ 5t)
            tick_per_z = sd_anchor_30s * math.sqrt(hsec/30.0)
            ticks = t[k] * tick_per_z
            net_ioc = ticks - MARKET_COST_TICKS
            net_passive = ticks - PASSIVE_COST_TICKS  # if hypothetically filled
            h1_rows.append({
                "band": bname, "exit_horizon": hkey, "n": int(k.sum()),
                "mean_ticks_realized": float(np.mean(ticks)),
                "wr_ioc_pct": float(np.mean(net_ioc > 0)*100),
                "mean_pnl_ioc_t": float(np.mean(net_ioc)),
                "sharpe_toy_ioc": float(np.mean(net_ioc)/np.std(net_ioc)*math.sqrt(k.sum())) if np.std(net_ioc)>0 else None,
                "mean_pnl_passive_t": float(np.mean(net_passive)),
            })
    out_summary["H1_time_exits"] = h1_rows

    # ─────── H2: Slice fifo labels by every filter variant ───────
    log("H2. Slice fifo_tp4sl3_net by 16 filter combos")
    h2_rows = []
    # Filter components for LONG entries
    base_long = msk & (p1 > 0)
    filters = {
        "agree_15": (np.sign(p1) == np.sign(p5)) & (np.sign(p1) == np.sign(p10)),
        "p_up_strong": pup5 > 0.6,
        "no_reversal": (rev15 < 0.4) & (rev30 < 0.4),
        "vol_mid": (pred_vol > np.nanpercentile(pred_vol, 33)) & (pred_vol < np.nanpercentile(pred_vol, 67)),
        "mfe_mae_ratio_15": pred_mfe > 1.5 * np.abs(pred_mae),
    }
    fnames = list(filters.keys())
    for cfg in range(2 ** len(fnames)):
        active = [fnames[i] for i in range(len(fnames)) if (cfg >> i) & 1]
        cmask = base_long.copy()
        for fn in active:
            cmask &= filters[fn]
        if cmask.sum() < 100: continue
        # Top1% entries among filtered
        idx_c = np.where(cmask)[0]
        n_take = max(5, int(idx_c.size * 0.01))
        cell = idx_c[np.argsort(-np.abs(p1[idx_c]))[:n_take]]
        # FIFO ground truth
        fm = fmsk43[cell]
        if fm.sum() < 5: continue
        pnl = fnet43[cell][fm]
        h2_rows.append({
            "filters": active, "cfg": cfg, "n_signals": int(cell.size),
            "n_filled": int(fm.sum()), "fill_pct": float(np.mean(fm)*100),
            "mean_pnl_t_filled": float(np.mean(pnl)),
            "wr_pct": float(np.mean(pnl > 0)*100),
            "sharpe_toy": float(np.mean(pnl)/np.std(pnl)*math.sqrt(fm.sum())) if np.std(pnl)>0 else None,
        })
    h2_sorted = sorted([r for r in h2_rows if r["sharpe_toy"] is not None],
                      key=lambda r: -r["sharpe_toy"])
    out_summary["H2_fifo_filtered_top20"] = h2_sorted[:20]
    log(f"  H2: {len(h2_rows)} configs, best PnL={h2_sorted[0]['mean_pnl_t_filled']:.3f}t")

    # ─────── H3: Same for tp8sl5 (wider TP, less stop-out) ───────
    log("H3. Same filter sweep on fifo_tp8sl5_net")
    h3_rows = []
    for cfg in range(2 ** len(fnames)):
        active = [fnames[i] for i in range(len(fnames)) if (cfg >> i) & 1]
        cmask = base_long.copy()
        for fn in active:
            cmask &= filters[fn]
        if cmask.sum() < 100: continue
        idx_c = np.where(cmask)[0]
        n_take = max(5, int(idx_c.size * 0.01))
        cell = idx_c[np.argsort(-np.abs(p1[idx_c]))[:n_take]]
        fm = fmsk85[cell]
        if fm.sum() < 5: continue
        pnl = fnet85[cell][fm]
        h3_rows.append({
            "filters": active, "n_signals": int(cell.size), "n_filled": int(fm.sum()),
            "fill_pct": float(np.mean(fm)*100),
            "mean_pnl_t_filled": float(np.mean(pnl)),
            "wr_pct": float(np.mean(pnl > 0)*100),
            "sharpe_toy": float(np.mean(pnl)/np.std(pnl)*math.sqrt(fm.sum())) if np.std(pnl)>0 else None,
        })
    h3_sorted = sorted([r for r in h3_rows if r["sharpe_toy"] is not None],
                      key=lambda r: -r["sharpe_toy"])
    out_summary["H3_fifo85_filtered_top20"] = h3_sorted[:20]
    log(f"  H3: {len(h3_rows)} configs, best PnL={h3_sorted[0]['mean_pnl_t_filled']:.3f}t")

    # ─────── H4: SHORT side — same exhaustive sweep ───────
    log("H4. Mirror sweep for SHORT side")
    h4_rows = []
    base_short = msk & (p1 < 0)
    short_filters = {
        "agree_15": (np.sign(p1) == np.sign(p5)) & (np.sign(p1) == np.sign(p10)),
        "p_up_weak": pup5 < 0.4,
        "no_reversal": (rev15 < 0.4) & (rev30 < 0.4),
        "vol_mid": (pred_vol > np.nanpercentile(pred_vol, 33)) & (pred_vol < np.nanpercentile(pred_vol, 67)),
        "mae_dominates": np.abs(pred_mae) > 1.5 * pred_mfe,  # short signal: expected adverse > favorable in LONG terms
    }
    sfnames = list(short_filters.keys())
    for cfg in range(2 ** len(sfnames)):
        active = [sfnames[i] for i in range(len(sfnames)) if (cfg >> i) & 1]
        cmask = base_short.copy()
        for fn in active:
            cmask &= short_filters[fn]
        if cmask.sum() < 100: continue
        idx_c = np.where(cmask)[0]
        n_take = max(5, int(idx_c.size * 0.01))
        cell = idx_c[np.argsort(-np.abs(p1[idx_c]))[:n_take]]
        fm = fmsk43[cell]
        if fm.sum() < 5: continue
        # Sign-flip for short approximation (HC #320 conservative)
        pnl = -fnet43[cell][fm]
        h4_rows.append({
            "filters": active, "n_signals": int(cell.size), "n_filled": int(fm.sum()),
            "fill_pct": float(np.mean(fm)*100),
            "mean_pnl_t_short_signflip": float(np.mean(pnl)),
            "wr_pct": float(np.mean(pnl > 0)*100),
            "sharpe_toy": float(np.mean(pnl)/np.std(pnl)*math.sqrt(fm.sum())) if np.std(pnl)>0 else None,
        })
    h4_sorted = sorted([r for r in h4_rows if r["sharpe_toy"] is not None],
                      key=lambda r: -r["sharpe_toy"])
    out_summary["H4_short_filtered_top20"] = h4_sorted[:20]
    log(f"  H4 (short signflip): {len(h4_rows)} configs, best PnL={h4_sorted[0]['mean_pnl_t_short_signflip']:.3f}t")

    # ─────── H5: Tighter Top0.1% with confluence + ground-truth FIFO ───────
    log("H5. Top0.1% × confluence × FIFO PnL")
    h5_rows = []
    for side, smask in [("long", p1 > 0), ("short", p1 < 0)]:
        for cfg_n, gates in enumerate([
            [],  # no gates
            ["agree_15"],
            ["agree_15", "no_reversal"],
            ["agree_15", "no_reversal", "vol_mid"],
            ["agree_15", "no_reversal", "p_up_strong" if side=="long" else "p_up_weak"],
            ["agree_15", "no_reversal", "vol_mid", "p_up_strong" if side=="long" else "p_up_weak"],
        ]):
            base = msk & smask
            for g in gates:
                if g == "agree_15":
                    base &= (np.sign(p1) == np.sign(p5)) & (np.sign(p1) == np.sign(p10))
                elif g == "no_reversal":
                    base &= (rev15 < 0.4) & (rev30 < 0.4)
                elif g == "vol_mid":
                    base &= ((pred_vol > np.nanpercentile(pred_vol, 33))
                             & (pred_vol < np.nanpercentile(pred_vol, 67)))
                elif g == "p_up_strong":
                    base &= pup5 > 0.6
                elif g == "p_up_weak":
                    base &= pup5 < 0.4
            if base.sum() < 100: continue
            idx_c = np.where(base)[0]
            n_take = max(3, int(idx_c.size * 0.001))
            cell = idx_c[np.argsort(-np.abs(p1[idx_c]))[:n_take]]
            fm = fmsk43[cell]
            if fm.sum() < 3: continue
            pnl = (fnet43[cell][fm] if side=="long" else -fnet43[cell][fm])
            h5_rows.append({
                "side": side, "gates": gates, "n_signals": int(cell.size),
                "n_filled": int(fm.sum()),
                "fill_pct": float(np.mean(fm)*100),
                "mean_pnl_t": float(np.mean(pnl)),
                "wr_pct": float(np.mean(pnl > 0)*100),
                "sharpe_toy": float(np.mean(pnl)/np.std(pnl)*math.sqrt(fm.sum())) if np.std(pnl)>0 else None,
            })
    out_summary["H5_top01pct_confluence"] = h5_rows

    # ─────── H6: ALL FIFO PnL distribution by side+confluence ───────
    log("H6. PnL distribution percentiles for headline configs")
    h6_rows = []
    headline_configs = [
        ("long", "Top1% no-gate", base_long, 0.01),
        ("long", "Top1% agree_15+no_rev",
         base_long & ((np.sign(p1)==np.sign(p5))&(np.sign(p1)==np.sign(p10)))
                   & (rev15 < 0.4) & (rev30 < 0.4), 0.01),
        ("long", "Top0.5% agree_15+no_rev+vol_mid",
         base_long & ((np.sign(p1)==np.sign(p5))&(np.sign(p1)==np.sign(p10)))
                   & (rev15 < 0.4) & (rev30 < 0.4)
                   & ((pred_vol > np.nanpercentile(pred_vol, 33))
                      & (pred_vol < np.nanpercentile(pred_vol, 67))), 0.005),
        ("short", "Top1% no-gate", base_short, 0.01),
        ("short", "Top1% agree_15+no_rev",
         base_short & ((np.sign(p1)==np.sign(p5))&(np.sign(p1)==np.sign(p10)))
                    & (rev15 < 0.4) & (rev30 < 0.4), 0.01),
    ]
    for side, lbl, mm, frac in headline_configs:
        idx_c = np.where(mm)[0]
        if idx_c.size < 30: continue
        n_take = max(5, int(idx_c.size * frac))
        cell = idx_c[np.argsort(-np.abs(p1[idx_c]))[:n_take]]
        fm = fmsk43[cell]
        if fm.sum() < 5: continue
        pnl = (fnet43[cell][fm] if side=="long" else -fnet43[cell][fm])
        h6_rows.append({
            "side": side, "label": lbl,
            "n_signals": int(cell.size), "n_filled": int(fm.sum()),
            "fill_pct": float(np.mean(fm)*100),
            "mean_pnl_t": float(np.mean(pnl)),
            "median_pnl_t": float(np.median(pnl)),
            "p25_pnl_t": float(np.percentile(pnl, 25)),
            "p75_pnl_t": float(np.percentile(pnl, 75)),
            "p95_pnl_t": float(np.percentile(pnl, 95)),
            "max_pnl_t": float(np.max(pnl)),
            "min_pnl_t": float(np.min(pnl)),
            "wr_pct": float(np.mean(pnl > 0)*100),
            "sharpe_toy": float(np.mean(pnl)/np.std(pnl)*math.sqrt(fm.sum())) if np.std(pnl)>0 else None,
        })
    out_summary["H6_pnl_distributions"] = h6_rows

    # Write all
    with open(OUT / "pass3_results.json", "w") as f:
        json.dump(out_summary, f, indent=2,
                  default=lambda o: None if isinstance(o,float) and not math.isfinite(o) else o)

    md = ["# PASS 3 — Realistic Exits + Ground-Truth FIFO Slicing\n\n",
          "## H1. Time-Based Exits (long, IOC market, exit at fixed horizon)\n",
          "| Band | Exit | n | Realized (t) | WR_IOC% | Net IOC PnL (t) | Sharpe-toy |\n|---|---|---:|---:|---:|---:|---:|\n"]
    for r in h1_rows:
        md.append(f"| {r['band']} | {r['exit_horizon']} | {r['n']} | "
                  f"{r['mean_ticks_realized']:.3f} | {r['wr_ioc_pct']:.1f} | "
                  f"{r['mean_pnl_ioc_t']:.3f} | {r['sharpe_toy_ioc']} |\n")

    md.append("\n## H2. TOP 15 LONG configs (FIFO tp4sl3, ground truth)\n")
    md.append("| Filters | n_sig | n_fill | Fill% | PnL/fill (t) | WR% | Sharpe-toy |\n|---|---:|---:|---:|---:|---:|---:|\n")
    for r in h2_sorted[:15]:
        md.append(f"| {','.join(r['filters']) or 'NONE'} | {r['n_signals']} | "
                  f"{r['n_filled']} | {r['fill_pct']:.1f} | "
                  f"{r['mean_pnl_t_filled']:.3f} | {r['wr_pct']:.1f} | "
                  f"{r['sharpe_toy']:.2f} |\n")

    md.append("\n## H3. TOP 15 LONG configs (FIFO tp8sl5, wider stops)\n")
    md.append("| Filters | n_sig | n_fill | Fill% | PnL/fill (t) | WR% | Sharpe-toy |\n|---|---:|---:|---:|---:|---:|---:|\n")
    for r in h3_sorted[:15]:
        md.append(f"| {','.join(r['filters']) or 'NONE'} | {r['n_signals']} | "
                  f"{r['n_filled']} | {r['fill_pct']:.1f} | "
                  f"{r['mean_pnl_t_filled']:.3f} | {r['wr_pct']:.1f} | "
                  f"{r['sharpe_toy']:.2f} |\n")

    md.append("\n## H4. TOP 15 SHORT configs (FIFO tp4sl3 sign-flip)\n")
    md.append("| Filters | n_sig | n_fill | Fill% | PnL/fill (t) | WR% | Sharpe-toy |\n|---|---:|---:|---:|---:|---:|---:|\n")
    for r in h4_sorted[:15]:
        md.append(f"| {','.join(r['filters']) or 'NONE'} | {r['n_signals']} | "
                  f"{r['n_filled']} | {r['fill_pct']:.1f} | "
                  f"{r['mean_pnl_t_short_signflip']:.3f} | {r['wr_pct']:.1f} | "
                  f"{r['sharpe_toy']:.2f} |\n")

    md.append("\n## H5. Top0.1% × Confluence Gate Sweep × FIFO Ground Truth\n")
    md.append("| Side | Gates | n_sig | n_fill | Fill% | PnL/fill (t) | WR% | Sharpe-toy |\n|---|---|---:|---:|---:|---:|---:|---:|\n")
    for r in sorted(h5_rows, key=lambda r: -(r["sharpe_toy"] or -1e9)):
        md.append(f"| {r['side']} | {','.join(r['gates']) or 'NONE'} | {r['n_signals']} | "
                  f"{r['n_filled']} | {r['fill_pct']:.1f} | "
                  f"{r['mean_pnl_t']:.3f} | {r['wr_pct']:.1f} | "
                  f"{r['sharpe_toy']} |\n")

    md.append("\n## H6. PnL Distributions for Headline Configs\n")
    md.append("| Side | Config | n_fill | Fill% | Mean | Med | p25 | p75 | p95 | Max | Min | WR% | Sharpe |\n|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for r in h6_rows:
        md.append(f"| {r['side']} | {r['label']} | {r['n_filled']} | {r['fill_pct']:.1f} | "
                  f"{r['mean_pnl_t']:.3f} | {r['median_pnl_t']:.3f} | "
                  f"{r['p25_pnl_t']:.3f} | {r['p75_pnl_t']:.3f} | "
                  f"{r['p95_pnl_t']:.3f} | {r['max_pnl_t']:.2f} | {r['min_pnl_t']:.2f} | "
                  f"{r['wr_pct']:.1f} | {r['sharpe_toy']} |\n")

    (OUT / "PASS3.md").write_text("".join(md))
    log("PASS 3 complete")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"FATAL: {e}\n{traceback.format_exc()}")
        sys.exit(1)
