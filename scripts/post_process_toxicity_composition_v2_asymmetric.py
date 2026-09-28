#!/usr/bin/env python3
"""
post_process_toxicity_composition_v2_asymmetric.py

Three-head (adverse / MFE / toxicity) composite gating + fill-sim with
TP/SL frameworks bounded by realized MFE distribution at h=10s.

HC #428 R2 binding: TP <= p90(MFE within horizon). p90 measured first;
any non-compliant framework is skipped and documented.

Frameworks tested:
  K=1 symmetric:        TP=+1.0, SL=-1.0  -- win = MFE>=1.0 AND ADV<1.0
  K=1/SL=2 asymmetric:  TP=+1.0, SL=-2.0  -- win = MFE>=1.0 AND ADV<2.0
  K=1.5 symmetric:      TP=+1.5, SL=-1.5  -- win = MFE>=1.5 AND ADV<1.5

For each framework x {maker 0.376t, taker 1.376t} x {adverse_only,
toxicity_only, composite} x Q in {1,2,4,6,10}% report n_total, n_filled,
fill_rate (maker proxy: ADV>=0.5), WR, mean_ticks net, PF, Sharpe.

Best cell -> regime stratification (per-day Sharpe), R1 spread <= 0.50,
day-conc <= 0.70.
"""
import json
import math
import os
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
DATA_ROOT = Path("/home/jupiter/teleclaude-main/tmp_tox_data")
OUT_JSON_LOCAL = Path("/home/jupiter/Lvl3Quant/output/toxicity_h10s_v1/composition_v2_asymmetric_report.json")
OUT_JSON_LOCAL.parent.mkdir(parents=True, exist_ok=True)

COST_MAKER_TICKS = 0.376
COST_TAKER_TICKS = 1.376
FILL_THRESHOLD_TICKS = 0.5

Q_LIST = [1, 2, 4, 6, 10]
GATES = ["adverse_only", "toxicity_only", "composite"]
COSTS = {"maker": COST_MAKER_TICKS, "taker": COST_TAKER_TICKS}

FRAMEWORKS = [
    {"name": "K1_sym",    "TP": 1.0, "SL": 1.0},
    {"name": "K1_SL2",    "TP": 1.0, "SL": 2.0},
    {"name": "K1p5_sym",  "TP": 1.5, "SL": 1.5},
]

# HC #428 R2 binding ceiling
P90_CEILING_NOTE = "TP must be <= p90(realized MFE within horizon h=10s)"


# ----------------------------------------------------------------------------
def load_all_folds():
    tox_files = sorted(glob(str(DATA_ROOT / "toxicity" / "fold_*.parquet")))
    rows, diag = [], []
    for tp in tox_files:
        name = os.path.basename(tp)
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
        diag.append({"fold_id": fold_id, "oot_date": oot_date,
                     "n_tox": len(df_tox), "n_mfe": len(df_mfe),
                     "n_adv": len(df_adv), "n_joined": len(df)})
    big = pd.concat(rows, ignore_index=True)
    return big, diag


# ----------------------------------------------------------------------------
def compute_payoffs(df, tp_ticks, sl_ticks, cost_ticks):
    """Win: MFE>=TP AND ADV<SL  -> +TP. Loss: otherwise -> -SL.  Net = payoff - cost."""
    win = (df["y_true_mfe"].values >= tp_ticks) & (df["y_true_adverse"].values < sl_ticks)
    payoff = np.where(win, tp_ticks, -sl_ticks)
    net = payoff - cost_ticks
    return win, net


def metrics_block(net, n_total, n_filled):
    if len(net) == 0 or n_filled == 0:
        return {"n_total": int(n_total), "n_filled": int(n_filled), "fill_rate": 0.0,
                "WR": None, "mean_ticks": None, "PF": None, "Sharpe": None, "sum_ticks": 0.0}
    wins = net[net > 0]
    losses = net[net < 0]
    sum_w = float(wins.sum()) if len(wins) else 0.0
    sum_l = float(-losses.sum()) if len(losses) else 0.0
    pf = (sum_w / sum_l) if sum_l > 0 else (float("inf") if sum_w > 0 else None)
    mean = float(net.mean())
    std = float(net.std(ddof=1)) if len(net) > 1 else 0.0
    sharpe = (mean / std * math.sqrt(len(net))) if std > 0 else None
    wr = float((net > 0).mean())
    return {"n_total": int(n_total), "n_filled": int(n_filled),
            "fill_rate": float(n_filled / n_total) if n_total else 0.0,
            "WR": wr, "mean_ticks": mean, "PF": pf, "Sharpe": sharpe,
            "sum_ticks": float(net.sum())}


def gate_mask(df, gate, q_pct):
    n = len(df)
    k = max(1, int(round(n * q_pct / 100.0)))
    if gate == "adverse_only":
        idx = np.argsort(df["y_pred_adverse"].values)[:k]
    elif gate == "toxicity_only":
        idx = np.argsort(-df["y_pred_toxicity"].values)[:k]
    elif gate == "composite":
        tox_rank = pd.Series(df["y_pred_toxicity"].values).rank(method="average").values
        adv_rank = pd.Series(df["y_pred_adverse"].values).rank(method="average").values
        score = tox_rank - adv_rank
        idx = np.argsort(-score)[:k]
    else:
        raise ValueError(gate)
    mask = np.zeros(n, dtype=bool)
    mask[idx] = True
    return mask


def per_fold_gate_mask(df_all, gate, q_pct):
    masks = []
    for fid, gdf in df_all.groupby("fold_id", sort=True):
        m = gate_mask(gdf, gate, q_pct)
        ser = pd.Series(m, index=gdf.index)
        masks.append(ser)
    return pd.concat(masks).sort_index().values


# ----------------------------------------------------------------------------
def main():
    print("Loading & joining all folds...")
    df_all, diag = load_all_folds()
    print(f"Total joined rows: {len(df_all):,} across {df_all['fold_id'].nunique()} folds")

    # --- p90 ceiling check ---
    mfe = df_all["y_true_mfe"].values
    p90 = float(np.percentile(mfe, 90))
    p_block = {
        "p50": float(np.percentile(mfe, 50)),
        "p75": float(np.percentile(mfe, 75)),
        "p85": float(np.percentile(mfe, 85)),
        "p90": p90,
        "p95": float(np.percentile(mfe, 95)),
        "mean": float(mfe.mean()),
        "n": int(len(mfe)),
    }
    print(f"MFE p90 = {p90:.3f} ticks  (binding TP ceiling per HC #428 R2)")

    # Filter frameworks by p90 compliance
    fw_status = []
    valid_fws = []
    for fw in FRAMEWORKS:
        compliant = fw["TP"] <= p90
        fw_status.append({"name": fw["name"], "TP": fw["TP"], "SL": fw["SL"],
                          "p90_compliant": compliant})
        if compliant:
            valid_fws.append(fw)
    print(f"Frameworks compliant with HC #428 R2: {[f['name'] for f in valid_fws]}")
    if not valid_fws:
        report = {"meta": {"halt_reason": f"All frameworks violate TP<=p90={p90}"},
                  "p_dist": p_block, "framework_compliance": fw_status}
        with open(OUT_JSON_LOCAL, "w") as f:
            json.dump(report, f, indent=2, default=str)
        return

    # --- Big sweep ---
    results = {}
    for fw in valid_fws:
        for gate in GATES:
            for q in Q_LIST:
                mask = per_fold_gate_mask(df_all, gate, q)
                gated = df_all[mask].copy()
                for cost_name, cost_t in COSTS.items():
                    _, net = compute_payoffs(gated, fw["TP"], fw["SL"], cost_t)
                    pre_metrics = metrics_block(net, len(gated), len(gated))
                    fill_mask = gated["y_true_adverse"].values >= FILL_THRESHOLD_TICKS
                    net_filled = net[fill_mask]
                    post_metrics = metrics_block(net_filled, len(gated), int(fill_mask.sum()))
                    key = f"{fw['name']}__{gate}__Q{q}__{cost_name}"
                    results[key] = {
                        "framework": fw["name"], "TP": fw["TP"], "SL": fw["SL"],
                        "gate": gate, "Q_pct": q, "cost": cost_name, "cost_ticks": cost_t,
                        "pre_fill": pre_metrics, "post_fill": post_metrics,
                    }

    # --- Pick best per framework + global best ---
    def best_cell_filter(filter_fn):
        cands = [(k, v) for k, v in results.items() if filter_fn(v)
                 and v["post_fill"]["n_filled"] >= 30
                 and v["post_fill"]["Sharpe"] is not None]
        if not cands:
            return None
        cands.sort(key=lambda kv: kv[1]["post_fill"]["Sharpe"], reverse=True)
        return cands[0]

    per_fw_best = {}
    for fw in valid_fws:
        for cost_name in ["maker", "taker"]:
            kv = best_cell_filter(lambda v, fw=fw, c=cost_name:
                                  v["framework"] == fw["name"] and v["cost"] == c)
            per_fw_best[f"{fw['name']}_{cost_name}"] = (
                {"key": kv[0], **kv[1]} if kv else None
            )

    # Specifically: K=1 maker composite Q=1%
    k1_maker_comp_q1 = results.get("K1_sym__composite__Q1__maker")
    k1sl2_maker_comp_q1 = results.get("K1_SL2__composite__Q1__maker")

    # Global best by Sharpe (post-fill, n_filled >= 30)
    global_best_kv = best_cell_filter(lambda v: True)
    global_best = {"key": global_best_kv[0], **global_best_kv[1]} if global_best_kv else None

    # --- Regime stratification on global best ---
    regime_block = None
    if global_best is not None:
        fw_name = global_best["framework"]
        fw_obj = next(f for f in valid_fws if f["name"] == fw_name)
        gate, q, cost_name = global_best["gate"], global_best["Q_pct"], global_best["cost"]
        cost_t = COSTS[cost_name]
        mask = per_fold_gate_mask(df_all, gate, q)
        gated = df_all[mask].copy()
        _, net = compute_payoffs(gated, fw_obj["TP"], fw_obj["SL"], cost_t)
        fm = gated["y_true_adverse"].values >= FILL_THRESHOLD_TICKS
        filled = gated[fm].copy()
        filled["net_ticks"] = net[fm]

        per_day = []
        for d, g in filled.groupby("oot_date"):
            x = g["net_ticks"].values
            mean = float(x.mean()) if len(x) else 0.0
            wr = float((x > 0).mean()) if len(x) else 0.0
            std = float(x.std(ddof=1)) if len(x) > 1 else 0.0
            sh = (mean / std * math.sqrt(len(x))) if std > 0 else None
            per_day.append({"oot_date": d, "n": int(len(x)), "mean_ticks": mean,
                            "WR": wr, "Sharpe": sh, "sum_ticks": float(x.sum())})

        greens = [d for d in per_day if d["mean_ticks"] > 0]
        reds = [d for d in per_day if d["mean_ticks"] < 0]
        def agg_sh(rows):
            xs = [r["Sharpe"] for r in rows if r["Sharpe"] is not None]
            return (float(np.mean(xs)) if xs else None), len(rows)
        sh_g, n_g = agg_sh(greens)
        sh_r, n_r = agg_sh(reds)
        if sh_g is not None and sh_r is not None and max(abs(sh_g), abs(sh_r)) > 0:
            spread = abs(sh_g - sh_r) / max(abs(sh_g), abs(sh_r))
        else:
            spread = None
        sums = [d["sum_ticks"] for d in per_day]
        total = float(sum(sums))
        day_conc = (max(sums) / total) if total > 0 else None

        regime_block = {
            "winner_key": global_best["key"],
            "framework": fw_name, "gate": gate, "Q_pct": q, "cost": cost_name,
            "per_day": per_day,
            "n_green_days": n_g, "n_red_days": n_r,
            "Sharpe_green_avg": sh_g, "Sharpe_red_avg": sh_r,
            "regime_spread": spread,
            "regime_pass_R1": (spread is not None and spread <= 0.50),
            "day_conc": day_conc,
            "day_conc_pass": (day_conc is not None and day_conc <= 0.70),
            "regime_classification_method": "fallback: per-day realized PnL sign (no ES daily file mounted)",
        }

    # --- Verdict ---
    accept = False; weak = False; rationale = []
    if global_best is None:
        rationale.append("no valid global best")
    else:
        pf = global_best["post_fill"]
        net_profitable = pf["mean_ticks"] is not None and pf["mean_ticks"] > 0
        pf_ok = pf["PF"] is not None and pf["PF"] > 1.20
        regime_ok = regime_block is not None and regime_block["regime_pass_R1"]
        dayconc_ok = regime_block is not None and regime_block["day_conc_pass"]
        rationale.append(f"winner={global_best['key']}  mean_ticks={pf['mean_ticks']}  PF={pf['PF']}  Sharpe={pf['Sharpe']}  WR={pf['WR']}")
        rationale.append(f"net_profitable={net_profitable}  PF>1.20={pf_ok}  regime_pass_R1={regime_ok}  day_conc_pass={dayconc_ok}")
        if net_profitable and pf_ok and regime_ok and dayconc_ok:
            accept = True
        elif net_profitable:
            weak = True
    verdict = "ACCEPT" if accept else ("WEAK ACCEPT" if weak else "REJECT")

    report = {
        "meta": {
            "frameworks": FRAMEWORKS,
            "p90_ceiling_note": P90_CEILING_NOTE,
            "p90_value_ticks": p90,
            "framework_compliance": fw_status,
            "fill_proxy": "maker filled iff y_true_adverse >= 0.5 ticks",
            "cost_maker_ticks": COST_MAKER_TICKS,
            "cost_taker_ticks": COST_TAKER_TICKS,
            "Q_pct_grid": Q_LIST,
            "gates": GATES,
            "n_folds": int(df_all["fold_id"].nunique()),
            "n_rows_joined_total": int(len(df_all)),
            "fold_diagnostics": diag,
        },
        "mfe_distribution": p_block,
        "all_results": results,
        "per_framework_best": per_fw_best,
        "k1_maker_composite_Q1": k1_maker_comp_q1,
        "k1sl2_maker_composite_Q1": k1sl2_maker_comp_q1,
        "global_best": global_best,
        "regime": regime_block,
        "verdict": verdict,
        "rationale": rationale,
    }

    with open(OUT_JSON_LOCAL, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Wrote {OUT_JSON_LOCAL}")

    # --- Console summary ---
    print(f"\nMFE p90 = {p90:.3f}  -- frameworks compliant: {[f['name'] for f in valid_fws]}")
    print("\n=== PER-FRAMEWORK BEST (post-fill, by Sharpe) ===")
    for k, v in per_fw_best.items():
        if v is None:
            print(f"  {k}: NONE")
            continue
        pf = v["post_fill"]
        print(f"  {k}: {v['gate']} Q={v['Q_pct']}% | n_filled={pf['n_filled']}  fill_rate={pf['fill_rate']:.3f}  WR={pf['WR']:.3f}  mean={pf['mean_ticks']:.4f}  PF={pf['PF']}  Sharpe={pf['Sharpe']:.3f}")
    print("\n=== K=1 maker composite Q=1% ===")
    if k1_maker_comp_q1:
        pf = k1_maker_comp_q1["post_fill"]
        print(f"  n_total={pf['n_total']} n_filled={pf['n_filled']} fill_rate={pf['fill_rate']:.3f} WR={pf['WR']} mean={pf['mean_ticks']} PF={pf['PF']} Sharpe={pf['Sharpe']}")
    print("\n=== K=1/SL=2 maker composite Q=1% ===")
    if k1sl2_maker_comp_q1:
        pf = k1sl2_maker_comp_q1["post_fill"]
        print(f"  n_total={pf['n_total']} n_filled={pf['n_filled']} fill_rate={pf['fill_rate']:.3f} WR={pf['WR']} mean={pf['mean_ticks']} PF={pf['PF']} Sharpe={pf['Sharpe']}")
    print(f"\n=== GLOBAL BEST ===")
    if global_best:
        print(f"  key={global_best['key']}")
        pf = global_best["post_fill"]
        print(f"  n_filled={pf['n_filled']} WR={pf['WR']} mean={pf['mean_ticks']} PF={pf['PF']} Sharpe={pf['Sharpe']}")
    if regime_block:
        print(f"\n=== REGIME (winner) ===")
        print(f"  n_green={regime_block['n_green_days']}  n_red={regime_block['n_red_days']}")
        print(f"  Sh_g={regime_block['Sharpe_green_avg']}  Sh_r={regime_block['Sharpe_red_avg']}  spread={regime_block['regime_spread']}  pass={regime_block['regime_pass_R1']}")
        print(f"  day_conc={regime_block['day_conc']}  pass={regime_block['day_conc_pass']}")
    print(f"\n=== VERDICT: {verdict} ===")
    for r in rationale:
        print(f"  - {r}")


if __name__ == "__main__":
    main()
