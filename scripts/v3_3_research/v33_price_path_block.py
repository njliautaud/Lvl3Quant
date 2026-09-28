#!/usr/bin/env python3
"""
HC #363 deliverable 6 — v3.3 price-path block per HC #361.

For each top-ranked SHORT signal cell (head × side × band), characterizes the
intra-trade price path using realized labels from `fold_00_predictions.npz`:

  - Realized log-return at 1s / 5s / 10s / 30s after signal
  - MFE 30s (best favorable excursion, ticks)
  - MAE 30s (worst adverse excursion, ticks)
  - time-to-MFE (seconds, peak-favorable timing)
  - realized vol 30s (ticks)
  - Time-exit P&L analysis: if we exited at 1s / 5s / 10s / 30s instead of FIFO
    tp/sl, what would the average tick move be? (passive SHORT P&L = -ret)

The goal: tell the user concretely WHEN to take profit / cut loss for each
deploy candidate. Particularly important for the HC #357 survivors
(p_reversal_60s SHORT Top 0.1%, fifo_tp8sl5_net SHORT Top 0.1%,
log_ret_60s_q50 SHORT Top 1%).

Outputs:
  output/v3_3_full_execution_analysis_20260514/price_path/
    price_path_summary.md
    price_path_per_cell.json

Per HC #307D — NEW analysis script, no trainer mods.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/jupiter/Lvl3Quant")
PRED_NPZ = ROOT / "output/cnn_mamba_v3_3_uncertainty_weighted/fold_00_predictions.npz"
OUT_DIR = ROOT / "output/v3_3_full_execution_analysis_20260514/price_path"
COMMISSION_TICKS = 0.376
SHORT_HIGH_HEADS = {"p_reversal_15s", "p_reversal_30s", "p_reversal_60s"}

# Cells to analyze — top HC #357 survivors + top FIFO Sharpe + top JOINT score
CANDIDATE_CELLS = [
    # (head, side, band_frac, label)
    ("p_reversal_60s",   "SHORT", 0.001, "Top0.1% (HC357 #2, JOINT #1)"),
    ("fifo_tp8sl5_net",  "SHORT", 0.001, "Top0.1% (HC357 #1, FIFO #2)"),
    ("log_ret_60s_q50",  "SHORT", 0.01,  "Top1% (HC357 #3, low-σ)"),
    ("log_ret_10s_q50",  "SHORT", 0.001, "Top0.1% (FIFO #3)"),
    ("fifo_tp4sl3_net",  "SHORT", 0.005, "Top0.5% (largest n_fills survivor)"),
]

# NOTE: empirically `target_log_ret_*` in this NPZ are already in TICK units
# (verified: target_log_ret_1s range -100..+24, std 2.0). No scaling needed.
# Some 5s/10s/30s entries are NaN — filter when computing.



def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not PRED_NPZ.exists():
        print(f"ERR: {PRED_NPZ} missing", file=sys.stderr)
        return 1

    print(f"Loading {PRED_NPZ}")
    npz = np.load(PRED_NPZ, allow_pickle=True)
    n_total = int(npz["n_samples"])
    print(f"Samples={n_total:,}")

    # Pull realized labels (already in ticks, with NaN entries)
    horizons = ["1s", "5s", "10s", "30s"]
    realized_ticks = {h: npz[f"target_log_ret_{h}"].astype(np.float64) for h in horizons}

    mfe_30 = npz["target_pred_mfe_30s_ticks"].astype(np.float64)
    mae_30 = npz["target_pred_mae_30s_ticks"].astype(np.float64)
    ttm_30 = npz["target_pred_time_to_mfe_secs"].astype(np.float64)
    rv_30 = npz["target_pred_realized_vol_30s_ticks"].astype(np.float64)
    mfe_mask = npz.get("mask_pred_mfe_30s_ticks")
    mfe_mask = mfe_mask.astype(bool) if mfe_mask is not None else ~np.isnan(mfe_30)

    fifo_target = npz["target_fifo_tp4sl3_net"].astype(np.float64)
    fifo_mask_raw = npz.get("mask_fifo_tp4sl3_net")
    fifo_mask = fifo_mask_raw.astype(bool) if fifo_mask_raw is not None else ~np.isnan(fifo_target)

    results = {"cells": []}
    md_lines = []
    md_lines.append("# v3.3 PRICE-PATH BLOCK — HC #363 deliverable 6 (per HC #361)\n")
    md_lines.append(f"Source: `{PRED_NPZ.name}` (5 OOT days, {n_total:,} events).")
    md_lines.append(f"Note: `target_log_ret_*` empirically already in tick units; no log-return conversion applied.")
    md_lines.append(f"All metrics computed on FIFO-fillable samples (mask_fifo_tp4sl3_net=1). NaN entries filtered.\n")

    for head, side, band_frac, label in CANDIDATE_CELLS:
        key = f"pred_{head}"
        if key not in npz.files:
            print(f"SKIP {head}: no pred field")
            continue
        pred = npz[key].astype(np.float64)
        # SHORT band selection: low pred for normal heads, high pred for reversal heads
        if head in SHORT_HIGH_HEADS:
            thr = np.nanquantile(np.where(fifo_mask, pred, np.nan), 1 - band_frac)
            sel = fifo_mask & (pred >= thr)
        else:
            thr = np.nanquantile(np.where(fifo_mask, pred, np.nan), band_frac)
            sel = fifo_mask & (pred <= thr)
        n_sel = int(sel.sum())
        if n_sel < 10:
            print(f"SKIP {head} {side} {label}: only {n_sel} cells")
            continue

        print(f"\n=== {head} {side} {label}: n={n_sel} ===")

        # Realized SHORT P&L at each exit horizon (ticks). SHORT P&L = -ret (passive, ignoring commission for now)
        exit_pnl = {}
        for h in horizons:
            rt = -realized_ticks[h][sel]  # negate for SHORT
            net = rt - COMMISSION_TICKS  # apply commission for round trip
            mu = float(np.nanmean(net))
            sd = float(np.nanstd(net, ddof=1)) if net.size > 1 else 0.0
            sh = mu / sd if sd > 1e-9 else 0.0
            exit_pnl[h] = {
                "n": int(np.sum(np.isfinite(net))),
                "mean_t": mu,
                "std_t": sd,
                "sharpe": sh,
                "median_t": float(np.nanmedian(net)),
                "p25": float(np.nanpercentile(net, 25)),
                "p75": float(np.nanpercentile(net, 75)),
                "win_rate": float(np.mean(net > 0)),
            }

        # FIFO baseline (the tp4sl3 sim)
        fifo_short = -fifo_target[sel] - COMMISSION_TICKS
        fifo_mu = float(np.nanmean(fifo_short))
        fifo_sd = float(np.nanstd(fifo_short, ddof=1)) if fifo_short.size > 1 else 0.0
        fifo_sh = fifo_mu / fifo_sd if fifo_sd > 1e-9 else 0.0

        # MFE / MAE / TTM (in their natural orientation; for SHORT a "favorable" excursion = price going DOWN = NEGATIVE log_ret → POSITIVE excursion in SHORT pnl)
        # Note: target_pred_mfe_30s_ticks is the long-side MFE (price going up). For SHORT we
        # take |MAE| as the favorable excursion proxy (worst adverse for a long = best for a short).
        mfe_sel = sel & mfe_mask
        if mfe_sel.sum() >= 10:
            mfe_long = mfe_30[mfe_sel]   # long-side MFE = max up move; for SHORT this is adverse
            mae_long = mae_30[mfe_sel]   # long-side MAE = max down move; for SHORT this is FAVORABLE
            ttm_long = ttm_30[mfe_sel]
            rv_sel = rv_30[mfe_sel]
            mfe_short_favorable = -mae_long  # flip: drawdown for long = profit for short (in ticks, positive)
            mae_short_adverse = mfe_long      # flip: max-up for long = max adverse for short
            mfe_summary = {
                "n": int(mfe_sel.sum()),
                "short_favorable_mean_t": float(np.mean(mfe_short_favorable)),
                "short_favorable_median_t": float(np.median(mfe_short_favorable)),
                "short_favorable_p75_t": float(np.percentile(mfe_short_favorable, 75)),
                "short_adverse_mean_t": float(np.mean(mae_short_adverse)),
                "short_adverse_median_t": float(np.median(mae_short_adverse)),
                "short_adverse_p75_t": float(np.percentile(mae_short_adverse, 75)),
                "time_to_long_mfe_mean_s": float(np.mean(ttm_long)),
                "time_to_long_mfe_median_s": float(np.median(ttm_long)),
                "realized_vol_30s_mean_t": float(np.mean(rv_sel)),
                "fav_to_adv_ratio_mean": float(np.mean(mfe_short_favorable) / max(np.mean(mae_short_adverse), 1e-6)),
            }
        else:
            mfe_summary = None

        cell_result = {
            "head": head, "side": side, "band_label": label,
            "n_selected": n_sel,
            "exit_horizons": exit_pnl,
            "fifo_baseline": {"n": int(np.sum(np.isfinite(fifo_short))), "mean_t": fifo_mu, "sharpe": fifo_sh},
            "price_path": mfe_summary,
        }
        results["cells"].append(cell_result)

        md_lines.append(f"## {head} {side} {label} (n={n_sel})\n")
        md_lines.append(f"### Time-exit P&L (SHORT, after {COMMISSION_TICKS:.3f}t commission)\n")
        md_lines.append("| Exit | n | Mean ticks | Median | Sharpe | Win-rate | P25 / P75 |")
        md_lines.append("|---|---|---|---|---|---|---|")
        for h in horizons:
            ep = exit_pnl[h]
            md_lines.append(f"| {h:>4} | {ep['n']} | {ep['mean_t']:+.3f} | {ep['median_t']:+.3f} | {ep['sharpe']:+.3f} | {ep['win_rate']*100:.1f}% | {ep['p25']:+.2f} / {ep['p75']:+.2f} |")
        md_lines.append(f"| **tp4sl3 FIFO** | {int(np.sum(np.isfinite(fifo_short)))} | {fifo_mu:+.3f} | n/a | {fifo_sh:+.3f} | n/a | n/a |")
        md_lines.append("")
        if mfe_summary:
            md_lines.append(f"### Price-path characteristics (30s window, on FIFO-fillable+MFE-labeled)\n")
            md_lines.append("| Metric | Value |")
            md_lines.append("|---|---|")
            md_lines.append(f"| Short-side favorable excursion (mean / median / p75) | {mfe_summary['short_favorable_mean_t']:.2f} / {mfe_summary['short_favorable_median_t']:.2f} / {mfe_summary['short_favorable_p75_t']:.2f} ticks |")
            md_lines.append(f"| Short-side adverse excursion (mean / median / p75) | {mfe_summary['short_adverse_mean_t']:.2f} / {mfe_summary['short_adverse_median_t']:.2f} / {mfe_summary['short_adverse_p75_t']:.2f} ticks |")
            md_lines.append(f"| Favorable / Adverse ratio (mean) | {mfe_summary['fav_to_adv_ratio_mean']:.2f} |")
            md_lines.append(f"| Time to long-MFE peak (mean / median, sec) | {mfe_summary['time_to_long_mfe_mean_s']:.1f} / {mfe_summary['time_to_long_mfe_median_s']:.1f} |")
            md_lines.append(f"| Realized vol 30s (mean) | {mfe_summary['realized_vol_30s_mean_t']:.2f} ticks |")
        # Optimal exit recommendation
        best_h = max(horizons, key=lambda h: exit_pnl[h]["sharpe"])
        md_lines.append(f"\n**Optimal time-exit (best Sharpe): {best_h} → Sharpe {exit_pnl[best_h]['sharpe']:+.3f}, mean {exit_pnl[best_h]['mean_t']:+.3f} ticks, win-rate {exit_pnl[best_h]['win_rate']*100:.1f}%.**\n")

    # Cross-cell comparison: optimal exit horizon
    md_lines.append("## Cross-cell summary — best time-exit horizon per cell\n")
    md_lines.append("| Cell | Best exit | Sharpe | Mean t | Win-rate | n |")
    md_lines.append("|---|---|---|---|---|---|")
    for c in results["cells"]:
        best_h = max(horizons, key=lambda h: c["exit_horizons"][h]["sharpe"])
        ep = c["exit_horizons"][best_h]
        md_lines.append(f"| {c['head']} {c['band_label']} | {best_h} | {ep['sharpe']:+.3f} | {ep['mean_t']:+.3f} | {ep['win_rate']*100:.1f}% | {ep['n']} |")

    md_lines.append("\n## Interpretation guide\n")
    md_lines.append("- **Win-rate > 55% AND mean > 0 AND Sharpe > 0.15** → cell is robustly tradeable at that horizon.")
    md_lines.append("- **Favorable/Adverse ratio > 1.2** → the price-path is asymmetric in our favor; supports use of trailing stops or wider TP.")
    md_lines.append("- **Time-to-MFE < 10s** → exit fast; ride the edge to peak then bail before mean reversion.")
    md_lines.append("- If FIFO Sharpe ≫ best time-exit Sharpe → the tp4sl3 levels are picking off the FAVORABLE slice; FIFO is over-optimistic about path realizability.")

    md_path = OUT_DIR / "price_path_summary.md"
    md_path.write_text("\n".join(md_lines) + "\n")
    print(f"Wrote {md_path}")

    json_path = OUT_DIR / "price_path_per_cell.json"
    json_path.write_text(json.dumps(results, indent=2, default=float))
    print(f"Wrote {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
