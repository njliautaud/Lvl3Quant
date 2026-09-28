#!/usr/bin/env python3
"""
post_process_toxicity_composition_v1.py

Three-head (adverse / MFE / toxicity) composite gating + fill-sim test.

Tests whether HIGH-toxicity gating rescues the maker paradigm from the
adverse-selection problem that killed it earlier (post-fill PF 0.70).

Hypothesis: gating by HIGH pred_toxicity selects events where opposite-side
aggression is ABOUT to happen -- meaning fills come when market wants to
trade at our price, not when market is running away from us.

Data (Razer / SCP'd to Jupiter):
  toxicity/fold_NN_oot_YYYYMMDD.parquet   columns: event_id, ts_ns, side, y_true_toxicity, y_pred_toxicity
  mfe/fold_NN_oot_YYYYMMDD.parquet        columns: event_id, ts_ns, side, y_true_mfe,    y_pred_mfe
  mfe/fold_NN_oot_YYYYMMDD_adverse.parquet                ..., y_true_adverse, y_pred_adverse

K=2 framework:
  win = (y_true_mfe >= 2.0) AND (y_true_adverse < 2.0)
  loss = otherwise
  payoff = +2 ticks on win, -2 ticks on loss, minus cost (maker 0.376 / taker 1.376)

Maker fill proxy: filled iff y_true_adverse >= 0.5 ticks (opposite side
printed at our limit). This is the SAME proxy that killed the earlier
maker finding -- we are testing whether the toxicity filter rescues it.

Output:
  output/toxicity_h10s_v1/composition_fillsim_report.json
"""
import json
import math
import os
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
DATA_ROOT = Path("/home/jupiter/teleclaude-main/tmp_tox_data")
OUT_JSON_LOCAL = Path("/home/jupiter/Lvl3Quant/output/toxicity_h10s_v1/composition_fillsim_report.json")
OUT_JSON_LOCAL.parent.mkdir(parents=True, exist_ok=True)

# K=2 first-passage framework
TP_TICKS = 2.0
SL_TICKS = 2.0
COST_MAKER_TICKS = 0.376
COST_TAKER_TICKS = 1.376

# Maker fill proxy: realized opposite-side travel >= 0.5 ticks
FILL_THRESHOLD_TICKS = 0.5

Q_LIST = [1, 2, 4, 6, 10]  # percent
GATES = ["adverse_only", "toxicity_only", "composite"]
COSTS = {"maker": COST_MAKER_TICKS, "taker": COST_TAKER_TICKS}


# ----------------------------------------------------------------------------
# Load & join
# ----------------------------------------------------------------------------
def load_all_folds():
    """Inner-join the three heads per fold; return one big DataFrame with fold_id, oot_date."""
    tox_files = sorted(glob(str(DATA_ROOT / "toxicity" / "fold_*.parquet")))
    rows = []
    diag = []
    for tp in tox_files:
        name = os.path.basename(tp)
        # fold_NN_oot_YYYYMMDD.parquet
        parts = name.replace(".parquet", "").split("_")
        fold_id = int(parts[1])
        oot_date = parts[3]

        mfe_p = DATA_ROOT / "mfe" / f"fold_{fold_id:02d}_oot_{oot_date}.parquet"
        adv_p = DATA_ROOT / "mfe" / f"fold_{fold_id:02d}_oot_{oot_date}_adverse.parquet"
        if not mfe_p.exists() or not adv_p.exists():
            continue

        df_tox = pd.read_parquet(tp)
        df_mfe = pd.read_parquet(mfe_p)
        df_adv = pd.read_parquet(adv_p)

        # mfe & adverse share (event_id, ts_ns, side)
        df_ma = df_mfe.merge(
            df_adv[["event_id", "side", "y_true_adverse", "y_pred_adverse"]],
            on=["event_id", "side"], how="inner",
        )
        df = df_ma.merge(
            df_tox[["event_id", "side", "y_true_toxicity", "y_pred_toxicity"]],
            on=["event_id", "side"], how="inner",
        )
        df["fold_id"] = fold_id
        df["oot_date"] = oot_date
        rows.append(df)

        diag.append({
            "fold_id": fold_id,
            "oot_date": oot_date,
            "n_tox": len(df_tox),
            "n_mfe": len(df_mfe),
            "n_adv": len(df_adv),
            "n_joined": len(df),
        })

    big = pd.concat(rows, ignore_index=True)
    return big, diag


# ----------------------------------------------------------------------------
# Payoff & metrics
# ----------------------------------------------------------------------------
def compute_payoffs(df, cost_ticks):
    """K=2: win if MFE>=2 AND adverse<2; payoff = +2 win, -2 loss; net = payoff - cost."""
    win = (df["y_true_mfe"].values >= TP_TICKS) & (df["y_true_adverse"].values < SL_TICKS)
    payoff = np.where(win, TP_TICKS, -SL_TICKS)
    net = payoff - cost_ticks
    return win, net


def metrics_block(net, n_total, n_filled):
    if len(net) == 0 or n_filled == 0:
        return {
            "n_total": int(n_total), "n_filled": int(n_filled),
            "fill_rate": 0.0, "WR": None, "mean_ticks": None,
            "PF": None, "Sharpe": None, "sum_ticks": 0.0,
        }
    wins = net[net > 0]
    losses = net[net < 0]
    sum_w = float(wins.sum()) if len(wins) else 0.0
    sum_l = float(-losses.sum()) if len(losses) else 0.0
    pf = (sum_w / sum_l) if sum_l > 0 else (float("inf") if sum_w > 0 else None)
    mean = float(net.mean())
    std = float(net.std(ddof=1)) if len(net) > 1 else 0.0
    sharpe = (mean / std * math.sqrt(len(net))) if std > 0 else None
    wr = float((net > 0).mean())  # WR after costs
    return {
        "n_total": int(n_total),
        "n_filled": int(n_filled),
        "fill_rate": float(n_filled / n_total) if n_total else 0.0,
        "WR": wr,
        "mean_ticks": mean,
        "PF": pf,
        "Sharpe": sharpe,
        "sum_ticks": float(net.sum()),
    }


# ----------------------------------------------------------------------------
# Gating
# ----------------------------------------------------------------------------
def gate_mask(df, gate, q_pct):
    """Return boolean mask selecting top q_pct% per the gate definition."""
    n = len(df)
    k = max(1, int(round(n * q_pct / 100.0)))
    if gate == "adverse_only":
        # LOWEST predicted adverse cost
        idx = np.argsort(df["y_pred_adverse"].values)[:k]
    elif gate == "toxicity_only":
        # HIGHEST predicted toxicity
        idx = np.argsort(-df["y_pred_toxicity"].values)[:k]
    elif gate == "composite":
        # rank-based: tox_rank (high=better) MINUS adv_rank (low=better)
        tox_rank = pd.Series(df["y_pred_toxicity"].values).rank(method="average").values
        adv_rank = pd.Series(df["y_pred_adverse"].values).rank(method="average").values
        score = tox_rank - adv_rank  # high tox + low adv -> high score
        idx = np.argsort(-score)[:k]
    else:
        raise ValueError(gate)
    mask = np.zeros(n, dtype=bool)
    mask[idx] = True
    return mask


# ----------------------------------------------------------------------------
# Main sweep
# ----------------------------------------------------------------------------
def main():
    print("Loading & joining all folds...")
    df_all, diag = load_all_folds()
    print(f"Total joined rows: {len(df_all):,} across {df_all['fold_id'].nunique()} folds")
    for d in diag:
        print(f"  fold {d['fold_id']:02d} {d['oot_date']}: tox={d['n_tox']:>6} mfe={d['n_mfe']:>6} adv={d['n_adv']:>6} joined={d['n_joined']:>6}")

    # Sanity: side coverage, true-adverse distribution
    print(f"\nside counts: {df_all['side'].value_counts().to_dict()}")
    print(f"y_true_adverse: mean={df_all['y_true_adverse'].mean():.3f}  >=0.5 frac={(df_all['y_true_adverse']>=FILL_THRESHOLD_TICKS).mean():.3f}")
    print(f"y_true_mfe:     mean={df_all['y_true_mfe'].mean():.3f}     >=2.0 frac={(df_all['y_true_mfe']>=TP_TICKS).mean():.3f}")

    # ------------------------------------------------------------------
    # Sweep
    # ------------------------------------------------------------------
    results = {}
    for gate in GATES:
        # gate is applied PER FOLD (so quantiles are per-day, fair)
        for q in Q_LIST:
            # mask across all folds, computed within-fold
            masks = []
            for fid, gdf in df_all.groupby("fold_id", sort=True):
                m = gate_mask(gdf, gate, q)
                ser = pd.Series(m, index=gdf.index)
                masks.append(ser)
            mask = pd.concat(masks).sort_index().values
            gated = df_all[mask].copy()
            for cost_name, cost_t in COSTS.items():
                wins, net = compute_payoffs(gated, cost_t)
                # PRE-FILL (all gated trades count, taker assumption -- always filled)
                pre_metrics = metrics_block(net, len(gated), len(gated))

                # POST-FILL (maker proxy: opposite-side aggression >= 0.5t)
                fill_mask = gated["y_true_adverse"].values >= FILL_THRESHOLD_TICKS
                net_filled = net[fill_mask]
                post_metrics = metrics_block(net_filled, len(gated), int(fill_mask.sum()))

                key = f"{gate}__Q{q}__{cost_name}"
                results[key] = {
                    "gate": gate, "Q_pct": q, "cost": cost_name, "cost_ticks": cost_t,
                    "pre_fill": pre_metrics,
                    "post_fill": post_metrics,
                }

    # ------------------------------------------------------------------
    # Find best gate at maker post-fill for each gate type
    # ------------------------------------------------------------------
    def best_post_fill_maker(gate_name):
        cands = [(k, v) for k, v in results.items()
                 if v["gate"] == gate_name and v["cost"] == "maker"
                 and v["post_fill"]["n_filled"] >= 30
                 and v["post_fill"]["mean_ticks"] is not None]
        if not cands:
            return None
        cands.sort(key=lambda kv: (kv[1]["post_fill"]["Sharpe"] or -1e9), reverse=True)
        return cands[0]

    best_adv = best_post_fill_maker("adverse_only")
    best_tox = best_post_fill_maker("toxicity_only")
    best_comp = best_post_fill_maker("composite")

    # ------------------------------------------------------------------
    # Regime stratification on best composite (or fall back to adverse if comp absent)
    # ------------------------------------------------------------------
    regime_block = None
    if best_comp is not None:
        gate, q, _ = best_comp[1]["gate"], best_comp[1]["Q_pct"], None
        masks = []
        for fid, gdf in df_all.groupby("fold_id", sort=True):
            m = gate_mask(gdf, gate, q)
            ser = pd.Series(m, index=gdf.index)
            masks.append(ser)
        mask = pd.concat(masks).sort_index().values
        gated = df_all[mask].copy()
        _, net = compute_payoffs(gated, COST_MAKER_TICKS)
        fm = gated["y_true_adverse"].values >= FILL_THRESHOLD_TICKS
        filled = gated[fm].copy()
        filled["net_ticks"] = net[fm]

        # per-day metrics
        per_day = []
        for d, g in filled.groupby("oot_date"):
            x = g["net_ticks"].values
            mean = float(x.mean()) if len(x) else 0.0
            wr = float((x > 0).mean()) if len(x) else 0.0
            std = float(x.std(ddof=1)) if len(x) > 1 else 0.0
            sh = (mean / std * math.sqrt(len(x))) if std > 0 else None
            per_day.append({"oot_date": d, "n": int(len(x)), "mean_ticks": mean, "WR": wr, "Sharpe": sh, "sum_ticks": float(x.sum())})

        # Day classification: green if mean_ticks > 0 (fallback per spec), red if mean_ticks < 0
        # (HC #428 R1 prefers ES close-to-close; we don't have an ES daily file mounted, so we
        # use realized post-fill PnL per day as the regime proxy.)
        greens = [d for d in per_day if d["mean_ticks"] > 0]
        reds = [d for d in per_day if d["mean_ticks"] < 0]
        def agg_sharpe(rows):
            if not rows:
                return None, 0
            xs = []
            for r in rows:
                # use sum_ticks weighted by n; or just average Sharpe across days
                if r["Sharpe"] is not None:
                    xs.append(r["Sharpe"])
            if not xs:
                return None, len(rows)
            return float(np.mean(xs)), len(rows)
        sh_g, n_g = agg_sharpe(greens)
        sh_r, n_r = agg_sharpe(reds)
        if sh_g is not None and sh_r is not None and max(abs(sh_g), abs(sh_r)) > 0:
            spread = abs(sh_g - sh_r) / max(abs(sh_g), abs(sh_r))
        else:
            spread = None

        # day concentration: max day_sum / total_sum (only if total positive)
        sums = [d["sum_ticks"] for d in per_day]
        total = float(sum(sums))
        if total > 0:
            day_conc = max(sums) / total
        else:
            day_conc = None

        regime_block = {
            "gate": gate, "Q_pct": q, "cost": "maker",
            "per_day": per_day,
            "n_green_days": n_g, "n_red_days": n_r,
            "Sharpe_green_avg": sh_g, "Sharpe_red_avg": sh_r,
            "regime_spread": spread,
            "regime_pass_R1": (spread is not None and spread <= 0.50),
            "day_conc": day_conc,
            "day_conc_pass": (day_conc is not None and day_conc <= 0.70),
            "regime_classification_method": "fallback: per-day realized PnL sign (no ES daily file mounted)",
        }

    # ------------------------------------------------------------------
    # Compose verdict
    # ------------------------------------------------------------------
    def fmt(kv):
        if kv is None:
            return None
        k, v = kv
        return {"key": k, **v}

    verdict_blob = {
        "best_adverse_only_maker_postfill": fmt(best_adv),
        "best_toxicity_only_maker_postfill": fmt(best_tox),
        "best_composite_maker_postfill": fmt(best_comp),
    }

    # Final verdict: composite must be (a) net-profitable (mean_ticks > 0)
    # AND (b) better Sharpe than adverse-only baseline (c) pass regime R1
    accept = False
    weak = False
    rationale = []
    if best_comp is None or best_comp[1]["post_fill"]["mean_ticks"] is None:
        rationale.append("no valid composite post-fill")
    else:
        comp_mean = best_comp[1]["post_fill"]["mean_ticks"]
        comp_pf = best_comp[1]["post_fill"]["PF"]
        comp_sh = best_comp[1]["post_fill"]["Sharpe"]
        adv_mean = best_adv[1]["post_fill"]["mean_ticks"] if best_adv else None
        adv_pf = best_adv[1]["post_fill"]["PF"] if best_adv else None

        net_profitable = comp_mean is not None and comp_mean > 0
        beats_adv = (adv_mean is None) or (comp_mean > adv_mean)
        pf_ok = comp_pf is not None and comp_pf > 1.20
        regime_ok = regime_block is not None and regime_block["regime_pass_R1"]
        dayconc_ok = regime_block is not None and regime_block["day_conc_pass"]

        rationale.append(f"composite post-fill mean_ticks={comp_mean:.3f} (>0? {net_profitable})")
        rationale.append(f"PF={comp_pf} (>1.20? {pf_ok})")
        rationale.append(f"Sharpe={comp_sh}")
        if adv_mean is not None:
            rationale.append(f"adverse-only baseline mean={adv_mean:.3f}, PF={adv_pf}; composite improves? {beats_adv}")
        if regime_block is not None:
            rationale.append(f"regime spread={regime_block['regime_spread']}, pass={regime_ok}; day_conc={regime_block['day_conc']}, pass={dayconc_ok}")

        if net_profitable and pf_ok and beats_adv and regime_ok and dayconc_ok:
            accept = True
        elif net_profitable and beats_adv:
            weak = True

    verdict = "ACCEPT" if accept else ("WEAK ACCEPT" if weak else "REJECT")

    report = {
        "meta": {
            "framework": "K=2 first-passage (TP=2, SL=2)",
            "fill_proxy": "maker filled iff y_true_adverse >= 0.5 ticks",
            "cost_maker_ticks": COST_MAKER_TICKS,
            "cost_taker_ticks": COST_TAKER_TICKS,
            "Q_pct_grid": Q_LIST,
            "gates": GATES,
            "n_folds": int(df_all["fold_id"].nunique()),
            "n_rows_joined_total": int(len(df_all)),
            "fold_diagnostics": diag,
        },
        "all_results": results,
        "best": verdict_blob,
        "regime": regime_block,
        "verdict": verdict,
        "rationale": rationale,
    }

    with open(OUT_JSON_LOCAL, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nWrote {OUT_JSON_LOCAL}")

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------
    def print_best(name, kv):
        if kv is None:
            print(f"  {name}: NONE")
            return
        k, v = kv
        pf = v["post_fill"]
        print(f"  {name}: Q={v['Q_pct']}% | n_total={pf['n_total']} n_filled={pf['n_filled']} fill_rate={pf['fill_rate']:.3f}")
        print(f"      WR={pf['WR']:.3f}  mean_ticks={pf['mean_ticks']:.3f}  PF={pf['PF']}  Sharpe={pf['Sharpe']}")

    print("\n=== BEST POST-FILL @ MAKER COST (per gate) ===")
    print_best("adverse_only ", best_adv)
    print_best("toxicity_only", best_tox)
    print_best("composite    ", best_comp)

    print(f"\n=== VERDICT: {verdict} ===")
    for r in rationale:
        print(f"  - {r}")


if __name__ == "__main__":
    main()
