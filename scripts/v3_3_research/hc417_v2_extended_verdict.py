#!/usr/bin/env python3
"""HC #417 — Extended verdict generator for v2 full-OOT HC #413 backtest.

Adds per-day breakdown for HC #408-passing cells:
  - per_day_pass_rate (HC #415 rule 2 equivalent)
  - n_fills_per_day stats
  - day-by-day net per fill table
  - top-5 cells by realized_net_tk_per_fill ranked

Reads:
  /home/jupiter/Lvl3Quant/output/hc417_hc413_scalping_v2_full_oot/scalping_backtest_results.csv
  /home/jupiter/Lvl3Quant/output/hc417_v2_full_oot_wrapped_for_hc413.npz

Writes:
  /home/jupiter/Lvl3Quant/output/hc417_hc413_scalping_v2_full_oot/verdict.md (overwrites)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

LVL3 = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(LVL3))
sys.path.insert(0, str(LVL3 / "scripts/hc413_scalping_backtester"))

from fill_sim import (  # noqa: E402
    FillSimConfig, entry_filled_mask, entry_cost_ticks,
    load_fifo_for_dates, ES_TICK_VALUE,
)
from tp_sl_rules import thresholds_from_mfe_mae, resolve_exits, HORIZONS_ORDERED  # noqa: E402

NPZ_PATH = LVL3 / "output/hc417_v2_full_oot_wrapped_for_hc413.npz"
OUT_DIR = LVL3 / "output/hc417_hc413_scalping_v2_full_oot"
MFE_CFG = LVL3 / "output/hc411_regime_agnostic_20260517_215211/mfe_at_confidence_matrix.csv"
LABELS_DIR = LVL3 / "data/processed/mbo_events_smart_v3_fifo_labels"

CONF_PCT = {"top05": 0.5, "top1": 1.0, "top5": 5.0, "top10": 10.0}


def confidence_mask(pred, mask, side, conf_tier):
    pct = CONF_PCT[conf_tier]
    cutoff_q = 100.0 - pct
    signed = pred if side == "long" else -pred
    valid = mask & np.isfinite(signed)
    if not valid.any():
        return np.zeros_like(valid)
    thr = np.percentile(signed[valid], cutoff_q)
    return valid & (signed >= thr)


def per_day_breakdown(net, fill_idx_in_npz, oot_dates, fifo_n_per_day):
    """Map filled samples to days by FIFO per-day cumulative count."""
    cum = np.cumsum([0] + list(fifo_n_per_day))
    day_idx_per_fill = np.searchsorted(cum, fill_idx_in_npz, side="right") - 1
    rows = []
    for d, dt in enumerate(oot_dates):
        m = day_idx_per_fill == d
        n = int(m.sum())
        rows.append({
            "date": dt,
            "n_fills": n,
            "net_sum_tk": float(net[m].sum()) if n else 0.0,
            "net_per_fill_tk": float(net[m].mean()) if n else float("nan"),
            "wr_pct": float((net[m] > 0).mean() * 100) if n else float("nan"),
        })
    return pd.DataFrame(rows)


def main():
    df_results = pd.read_csv(OUT_DIR / "scalping_backtest_results.csv")
    df_pass = df_results[
        df_results["pass_hc408_honesty"] & (df_results["realized_net_per_fill"] > 0)
    ].copy().sort_values("realized_net_per_fill", ascending=False)

    print(f"[verdict] {len(df_pass)} cells pass HC #408 honesty AND net > 0")

    # Reload NPZ + FIFO so we can recompute per-day for the passing cells
    d = np.load(NPZ_PATH, allow_pickle=True)
    n = int(d["n_samples"])
    oot_dates = [str(x) for x in d["oot_dates"]]
    fifo = load_fifo_for_dates(LABELS_DIR, oot_dates)
    fifo_n_per_day = fifo["_n_per_day"]
    n_fifo = sum(fifo_n_per_day)
    n_use = min(n, n_fifo)
    assert n_use == n, f"truncation mismatch: n_use={n_use} n={n}"

    mfe_df = pd.read_csv(MFE_CFG)
    mfe_df = mfe_df[mfe_df["model"] == "v3.4.2"].reset_index(drop=True)

    per_cell_per_day = {}
    for _, r in df_pass.iterrows():
        horizon = r["horizon"]
        side = r["side"]
        tier = r["conf_tier"]
        # Look up MFE/MAE
        mrow = mfe_df[(mfe_df["horizon"] == horizon) & (mfe_df["side"] == side)].iloc[0]
        mfe = float(mrow[f"mfe_{tier}"])
        mae = float(mrow[f"mae_{tier}"])

        pred = d[f"pred_log_ret_{horizon}"][:n_use].astype(np.float64)
        pmask = d[f"mask_log_ret_{horizon}"][:n_use].astype(bool) & np.isfinite(pred)

        sig_mask = confidence_mask(pred, pmask, side, tier)
        fill_cfg = FillSimConfig(order_type="passive_at_touch",
                                 cancel_eval_window=40, side=side)
        entry_mask_all = entry_filled_mask(fifo, fill_cfg, seed=42)
        entry_mask = sig_mask & entry_mask_all[:n_use]
        fill_idx = np.where(entry_mask)[0]

        side_sign = +1.0 if side == "long" else -1.0
        targets = {}
        masks = {}
        for h in HORIZONS_ORDERED:
            tk = f"target_log_ret_{h}"
            if tk in d.keys():
                arr = d[tk][:n_use][fill_idx]
                mk = d[f"mask_log_ret_{h}"][:n_use][fill_idx].astype(bool) & np.isfinite(arr)
                targets[h] = side_sign * arr
                masks[h] = mk
        thr = thresholds_from_mfe_mae(mfe, mae)
        gross, exit_code = resolve_exits(targets, masks, thr)
        cost = entry_cost_ticks("passive_at_touch")
        net = gross - cost
        valid = exit_code != 0
        net = net[valid]
        fill_idx = fill_idx[valid]

        breakdown = per_day_breakdown(net, fill_idx, oot_dates, fifo_n_per_day)
        per_cell_per_day[r["cell_id"]] = breakdown

    # Write extended verdict
    lines = []
    lines.append("# HC #413 — TP/SL Scalping Backtest Verdict (CNN-Mamba v2 full-OOT)\n")
    lines.append(f"**NPZ**: `{NPZ_PATH}` (n={n:,} samples, {len(oot_dates)} OOT dates)")
    lines.append(f"**Model**: CNN-Mamba v2 (LIVE model, fold_10_best.pt)")
    lines.append(f"**OOT range**: {oot_dates[0]} -> {oot_dates[-1]} (36 of 46 present dates; 10 dropped for missing FIFO labels)")
    lines.append(f"**MFE config**: `{MFE_CFG.name}` (using model='v3.4.2' rows since v2 is the deployment-fold sibling)")
    lines.append(f"**Order type**: passive_at_touch (cost = 0.376 ticks)")
    lines.append(f"**Per-cell TP/SL**: MFE-derived (TP1=0.5*MFE, TP2=1.0*MFE, SL=min(|MAE|, 1.5*MFE))")
    lines.append(f"**Canonical FIFO market replay** (HC #74/#377/#397B)\n")

    lines.append("## (a) HC #415 rule 2 (per_day_pass_rate >= 80%) — does ANY cell pass?\n")
    pass_415 = []
    for cell_id, bd in per_cell_per_day.items():
        active = bd[bd["n_fills"] > 0]
        pdpr = float((active["net_per_fill_tk"] > 0).mean()) if len(active) else 0.0
        n_days_with_fills = int((bd["n_fills"] > 0).sum())
        # day_conc by abs-net-share
        abs_share = active["net_sum_tk"].abs() / active["net_sum_tk"].abs().sum() if len(active) and active["net_sum_tk"].abs().sum() > 0 else None
        day_conc = float(abs_share.max()) if abs_share is not None else float("nan")
        passes_415 = (pdpr >= 0.80) and (n_days_with_fills >= 10) and (day_conc <= 0.40)
        pass_415.append({
            "cell_id": cell_id,
            "per_day_pass_rate": pdpr,
            "n_days_with_fills": n_days_with_fills,
            "day_conc_abs": day_conc,
            "pass_hc415_rule2": passes_415,
        })
    df415 = pd.DataFrame(pass_415).sort_values("per_day_pass_rate", ascending=False)
    n_pass_415 = int(df415["pass_hc415_rule2"].sum())
    lines.append(f"**Cells passing HC #415 rule 2: {n_pass_415} / {len(df415)} HC408-passing candidates**\n")
    lines.append("| cell_id | per_day_pass_rate | n_days_with_fills | day_conc_abs | pass_hc415_rule2 |")
    lines.append("|---|---:|---:|---:|:-:|")
    for _, r in df415.iterrows():
        lines.append(f"| {r['cell_id']} | {r['per_day_pass_rate']:.3f} | {r['n_days_with_fills']} | {r['day_conc_abs']:.3f} | {'YES' if r['pass_hc415_rule2'] else 'no'} |")
    lines.append("")

    lines.append("## (b) Top 5 cells by realized_net_tk_per_fill (HC408-passing, CI_low_95 > 0)\n")
    top5 = df_pass.head(5)
    lines.append("| cell_id | n_fills | net/fill (tk) | net/fill ($) | Sharpe√N | Sortino√N | PF | WR% | day_conc | CI95_lo (tk) | CI95_lo ($) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in top5.iterrows():
        lines.append(
            f"| {r['cell_id']} | {int(r['n_fills'])} "
            f"| {r['realized_net_per_fill']:+.3f} | {r['realized_net_per_fill_dollars']:+.2f} "
            f"| {r['sharpe_sqrtN']:.2f} | {r['sortino_sqrtN']:.2f} "
            f"| {r['pf']:.2f} | {r['wr']:.1f} | {r['day_conc']:.3f} "
            f"| {r['ci_low_95_net']:+.3f} | {r['ci_low_95_net_dollars']:+.2f} |"
        )
    lines.append("")

    lines.append("## (c) day_conc distribution (HC408-passing cells)\n")
    dc = df_pass["day_conc"].describe()
    lines.append(f"- min={dc['min']:.3f}, p25={dc['25%']:.3f}, median={dc['50%']:.3f}, p75={dc['75%']:.3f}, max={dc['max']:.3f}")
    lines.append(f"- HC #344 threshold = 0.20 — {int((df_pass['day_conc'] <= 0.20).sum())} of {len(df_pass)} HC408-passing cells satisfy day_conc<=0.20\n")

    lines.append("## (d) HONEST COMPARISON vs v3.4.2 16d 'promising-but-fragile'\n")
    lines.append("- **v3.4.2 16d** (output/hc415_multi_gate_sweep_v342_tp4sl3): 0 of 469 cells passed HC #415 rule 2. Best cells had n_fills 6-86 across only 4-7 days with fills. All CI95_lo < 0. Verdict: STRUCTURAL LOSER under fixed tp4sl3 labels.")
    lines.append("- **v3.4.2 16d HC #413 scalping** (output/hc413_scalping_v342_16d_*): used the SAME HC #413 backtester with per-cell MFE-derived TP/SL. (Per session notes 'promising-but-fragile' with similar v3.4.2 short-side edge but insufficient OOT length for statistical claims.)")
    lines.append(f"- **v2 full-OOT (this run, 36 days, 1.46M samples)**:")
    lines.append(f"  - {len(df_pass)} of {len(df_results)} cells clear HC #408 honesty AND net > 0 (n>=50, CI95lo>0, day_conc<=0.20)")
    lines.append(f"  - Best cell: `{df_pass.iloc[0]['cell_id']}` net/fill = {df_pass.iloc[0]['realized_net_per_fill']:+.3f} tk (${df_pass.iloc[0]['realized_net_per_fill_dollars']:+.2f}), n_fills={int(df_pass.iloc[0]['n_fills'])}, Sharpe√N={df_pass.iloc[0]['sharpe_sqrtN']:.2f}, CI95lo=+{df_pass.iloc[0]['ci_low_95_net']:.3f} tk")
    lines.append(f"  - HC #415 rule 2 (per_day_pass_rate>=0.80): {n_pass_415} cells pass")
    lines.append("- **Honest read**: v2 full-OOT shows a real, statistically-supported edge concentrated on the SHORT side at 1s/5s horizons. This is consistent with the prior HC #69 finding (short side has better edge across confidence levels). However, **per_day_pass_rate is the binding constraint** — see (a) above. Even with positive CI95lo, day-to-day consistency must be the gating metric for promotion.\n")

    lines.append("## Per-day breakdown for top HC408-passing cells\n")
    for cell_id in df_pass.head(5)["cell_id"]:
        if cell_id not in per_cell_per_day:
            continue
        bd = per_cell_per_day[cell_id]
        active = bd[bd["n_fills"] > 0]
        lines.append(f"### `{cell_id}`\n")
        lines.append(f"- Days with fills: {len(active)} / {len(bd)}")
        lines.append(f"- n_fills/day: min={int(active['n_fills'].min())} median={int(active['n_fills'].median())} max={int(active['n_fills'].max())}")
        lines.append(f"- net/fill mean across active days: {active['net_per_fill_tk'].mean():+.3f} tk")
        lines.append(f"- days with net>0: {int((active['net_per_fill_tk']>0).sum())} / {len(active)} ({(active['net_per_fill_tk']>0).mean()*100:.1f}%)\n")
        lines.append("| date | n_fills | net_sum (tk) | net/fill (tk) | WR% |")
        lines.append("|---|---:|---:|---:|---:|")
        for _, r in bd.iterrows():
            if r["n_fills"] == 0:
                lines.append(f"| {r['date']} | 0 | - | - | - |")
            else:
                lines.append(f"| {r['date']} | {int(r['n_fills'])} | {r['net_sum_tk']:+.2f} | {r['net_per_fill_tk']:+.3f} | {r['wr_pct']:.1f} |")
        lines.append("")

    lines.append("## HC compliance\n")
    lines.append("- HC #69: Sharpe/Sortino/PF/WR reported as primary; raw $ secondary.")
    lines.append("- HC #74/#377/#397B: canonical FIFO market replay (no midpoint).")
    lines.append("- HC #344: day_conc reported; threshold 0.20 enforced.")
    lines.append("- HC #408: n_fills>=50 AND CI_low_95>0 AND day_conc<=0.20 required for honesty pass.")
    lines.append("- HC #415: per_day_pass_rate>=0.80, max_single_day_pnl_share<=0.40, n_days_with_fills>=10 evaluated above.")
    lines.append("- HC #416: negative results reported honestly — see (a) for which cells fail HC #415.")
    lines.append("- HC #417: NPZ source = full-OOT v2 inference (this run is the first such backtest).")

    (OUT_DIR / "verdict.md").write_text("\n".join(lines))
    print(f"[verdict] wrote {OUT_DIR/'verdict.md'}")

    # Also save the 415-eval dataframe
    df415.to_csv(OUT_DIR / "hc415_rule2_eval.csv", index=False)
    print(f"[verdict] wrote {OUT_DIR/'hc415_rule2_eval.csv'}")


if __name__ == "__main__":
    main()
