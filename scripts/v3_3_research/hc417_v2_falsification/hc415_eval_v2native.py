"""
HC #417 falsification — Phase 2 finalizer:
  Generate HC #415 rule-2 evaluation CSV + verdict for the v2-native MFE backtest.

Reuses logic from scripts/v3_3_research/hc417_v2_extended_verdict.py but:
  - reads scalping_backtest_results.csv from output/hc417_hc413_v2native_mfe/
  - uses output/hc417_v2_native_mfe_matrix.csv as the MFE config (model='v2')

Writes:
  output/hc417_hc413_v2native_mfe/hc415_rule2_eval.csv
  output/hc417_hc413_v2native_mfe/verdict.md   (overwrites)
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
OUT_DIR = LVL3 / "output/hc417_hc413_v2native_mfe"
MFE_CFG = LVL3 / "output/hc417_v2_native_mfe_matrix.csv"
LABELS_DIR = LVL3 / "data/processed/mbo_events_smart_v3_fifo_labels"
MODEL_TAG = "v2"

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

    d = np.load(NPZ_PATH, allow_pickle=True)
    n = int(d["n_samples"])
    oot_dates = [str(x) for x in d["oot_dates"]]
    fifo = load_fifo_for_dates(LABELS_DIR, oot_dates)
    fifo_n_per_day = fifo["_n_per_day"]
    n_fifo = sum(fifo_n_per_day)
    n_use = min(n, n_fifo)

    mfe_df = pd.read_csv(MFE_CFG)
    mfe_df = mfe_df[mfe_df["model"] == MODEL_TAG].reset_index(drop=True)

    per_cell_per_day = {}
    for _, r in df_pass.iterrows():
        horizon = r["horizon"]; side = r["side"]; tier = r["conf_tier"]
        mrow_q = mfe_df[(mfe_df["horizon"] == horizon) & (mfe_df["side"] == side)]
        if len(mrow_q) == 0:
            continue
        mrow = mrow_q.iloc[0]
        mfe = float(mrow[f"mfe_{tier}"]); mae = float(mrow[f"mae_{tier}"])
        pred = d[f"pred_log_ret_{horizon}"][:n_use].astype(np.float64)
        pmask = d[f"mask_log_ret_{horizon}"][:n_use].astype(bool) & np.isfinite(pred)
        sig_mask = confidence_mask(pred, pmask, side, tier)
        fill_cfg = FillSimConfig(order_type="passive_at_touch",
                                 cancel_eval_window=40, side=side)
        entry_mask_all = entry_filled_mask(fifo, fill_cfg, seed=42)
        entry_mask = sig_mask & entry_mask_all[:n_use]
        fill_idx = np.where(entry_mask)[0]
        side_sign = +1.0 if side == "long" else -1.0
        targets = {}; masks = {}
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
        net = net[valid]; fill_idx = fill_idx[valid]
        breakdown = per_day_breakdown(net, fill_idx, oot_dates, fifo_n_per_day)
        per_cell_per_day[r["cell_id"]] = breakdown

    # HC #415 rule 2 eval
    pass_415 = []
    for cell_id, bd in per_cell_per_day.items():
        active = bd[bd["n_fills"] > 0]
        pdpr = float((active["net_per_fill_tk"] > 0).mean()) if len(active) else 0.0
        n_days_with_fills = int((bd["n_fills"] > 0).sum())
        # day_conc_abs = max share of |net_sum_tk| concentrated in one day
        abs_sum = bd["net_sum_tk"].abs().sum()
        day_conc = float(bd["net_sum_tk"].abs().max() / abs_sum) if abs_sum > 0 else float("nan")
        passes_415 = bool(pdpr >= 0.80 and n_days_with_fills >= 10)
        pass_415.append({
            "cell_id": cell_id,
            "per_day_pass_rate": pdpr,
            "n_days_with_fills": n_days_with_fills,
            "day_conc_abs": day_conc,
            "pass_hc415_rule2": passes_415,
        })
    df415 = pd.DataFrame(pass_415).sort_values("per_day_pass_rate", ascending=False)
    df415.to_csv(OUT_DIR / "hc415_rule2_eval.csv", index=False)
    print(f"[verdict] wrote {OUT_DIR/'hc415_rule2_eval.csv'}")

    # Compact verdict.md (overwrites prior)
    lines = []
    lines.append("# HC #413 — TP/SL Scalping Backtest Verdict (CNN-Mamba v2 full-OOT, v2-NATIVE MFE)\n")
    lines.append(f"**NPZ**: `{NPZ_PATH}` (n={n:,} samples, {len(oot_dates)} OOT dates)")
    lines.append(f"**Model tag**: v2 (CNN-Mamba v2, fold_10_best.pt)")
    lines.append(f"**MFE config**: `{MFE_CFG.name}` — V2-NATIVE (computed from v2 predictions, replaces v3.4.2 borrowed config)")
    lines.append(f"**Order type**: passive_at_touch (cost = 0.376 ticks)")
    lines.append(f"**Canonical FIFO market replay** (HC #74/#377/#397B)\n")
    lines.append("## HC #408-passing cells (n_fills>=50, day_conc<=0.20, CI95lo>0, net>0)\n")
    lines.append("| cell_id | n_fills | net/fill (tk) | Sharpe√N | Sortino√N | PF | WR% | day_conc | CI95_lo (tk) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in df_pass.iterrows():
        lines.append(f"| {r['cell_id']} | {int(r['n_fills'])} | {r['realized_net_per_fill']:+.3f} | "
                     f"{r['sharpe_sqrtN']:.2f} | {r.get('sortino_sqrtN', float('nan')):.2f} | "
                     f"{r['pf']:.2f} | {r['wr']:.1f} | {r['day_conc']:.3f} | {r['ci_low_95_net']:+.3f} |")
    lines.append("")
    n_pass_415 = int(df415["pass_hc415_rule2"].sum())
    lines.append(f"## HC #415 rule 2 (per_day_pass_rate>=0.80, n_days_with_fills>=10)\n")
    lines.append(f"**Cells passing HC #415 rule 2: {n_pass_415} / {len(df415)} HC408+net>0 candidates**\n")
    lines.append("| cell_id | per_day_pass_rate | n_days_with_fills | day_conc_abs | pass_hc415_rule2 |")
    lines.append("|---|---:|---:|---:|:-:|")
    for _, r in df415.iterrows():
        lines.append(f"| {r['cell_id']} | {r['per_day_pass_rate']:.3f} | {r['n_days_with_fills']} | "
                     f"{r['day_conc_abs']:.3f} | {'YES' if r['pass_hc415_rule2'] else 'no'} |")
    lines.append("")
    lines.append("## Bottom line\n")
    lines.append(f"- HC #408 passing AND net>0: **{len(df_pass)}** cells")
    lines.append(f"- HC #415 rule 2 passing: **{n_pass_415}** cells")
    if len(df_pass) > 0:
        b = df_pass.iloc[0]
        lines.append(f"- Best: `{b['cell_id']}` net/fill={b['realized_net_per_fill']:+.3f} tk, n={int(b['n_fills'])}, Sharpe√N={b['sharpe_sqrtN']:.2f}, CI95lo={b['ci_low_95_net']:+.3f}")
    lines.append("\n## Caveats\n")
    lines.append("- Uses v2-NATIVE MFE matrix (Caveat A of HC #417 falsification resolved).")
    lines.append("- Still no v3.4.2-style 30s head in v2 NPZ; only 1s/5s/10s horizons are evaluated.")
    lines.append("- MAE proxy at 1s/5s/10s = mean magnitude of negative-only signed realized move (same proxy as HC #411).")
    (OUT_DIR / "verdict.md").write_text("\n".join(lines))
    print(f"[verdict] wrote {OUT_DIR/'verdict.md'}")


if __name__ == "__main__":
    main()
