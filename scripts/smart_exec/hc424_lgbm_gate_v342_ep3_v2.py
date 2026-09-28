#!/usr/bin/env python3
"""HC #424 §3 v2 — Percentile-based gate diagnostic.

v1 used absolute predicted-net > 0 (after cost) and got 0 fills because the
realized net mean in this OOT window is negative (long -0.11, short -0.13)
which is below the 0.376 commission. The LGBM learned that signal correctly:
"on average, all trades lose money".

This v2 evaluates whether the LGBM has ANY useful ranking by taking the
TOP-K percentile of predicted net and reporting verdict at each percentile.
If multi-head ranking is informative, top-decile should be profitable even
if average isn't.

HC #420 authorized.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import mlflow

sys.path.insert(0, "/home/jupiter/Lvl3Quant/scripts/smart_exec")
from hc424_lgbm_gate_v342_ep3 import (  # noqa: E402
    load_aligned_data, time_split_per_day, train_lgbm,
    canonical_verdict, MULTI_HEAD_FEATURES, COMMISSION_RT_TICKS,
    OUT_DIR, NPZ_PATH, OOT_DATES, DAY_CONC_GATE,
)


def percentile_gate_verdict(name, side, pred_score, net_realized, filled_mask, date_idx, top_pct):
    """Take top `top_pct` percentile by pred_score among filled signals."""
    if top_pct <= 0 or top_pct > 100:
        raise ValueError("top_pct must be in (0, 100]")
    # Among filled only
    fi = np.where(filled_mask)[0]
    if len(fi) == 0:
        return None
    scores = pred_score[fi]
    cutoff = np.percentile(scores, 100 - top_pct)
    take_local = scores >= cutoff
    take = np.zeros_like(filled_mask)
    take[fi[take_local]] = True

    realized = net_realized[take]
    if len(realized) == 0:
        return None
    n = len(realized)
    mu = float(realized.mean()); sd = float(realized.std(ddof=1)) if n > 1 else 0
    sharpe = mu / sd * np.sqrt(n) if sd > 0 else 0
    neg = realized[realized < 0]; nsd = float(neg.std(ddof=1)) if len(neg) > 1 else 0
    sortino = mu / nsd * np.sqrt(n) if nsd > 0 else 0
    gwin = float(realized[realized > 0].sum()); gloss = float(-realized[realized < 0].sum())
    pf = gwin / gloss if gloss > 0 else float("inf")
    wr = (realized > 0).sum() / max(1, ((realized > 0).sum() + (realized < 0).sum()))
    days = date_idx[take]
    cnt = np.bincount(days) if len(days) > 0 else np.array([1])
    dc = cnt.max() / cnt.sum()
    return {
        "name": name, "side": side, "top_pct": top_pct,
        "n_fills": n, "ticks_per_fill": mu, "total_ticks": mu * n,
        "sharpe": float(sharpe), "sortino": float(sortino),
        "pf": float(pf), "wr": float(wr),
        "adv_sel_30s_avg": mu, "day_conc": float(dc),
        "pass_hc344": bool(dc < DAY_CONC_GATE),
        "queue_pos": "N/A (FIFO label realized)",
        "cancel_window": "N/A (FIFO label realized)",
    }


def main():
    mlflow.set_tracking_uri("http://jupiter:5000")
    mlflow.set_experiment("hc424_jupiter_exec_research_v342")

    t0 = time.time()
    data = load_aligned_data()
    X = data["X"]
    train_mask, hold_mask = time_split_per_day(data["date_idx"])
    n_train = int(train_mask.sum()); n_hold = int(hold_mask.sum())
    print(f"[split] train={n_train} hold={n_hold}")

    with mlflow.start_run(run_name=f"lgbm_gate_v342_ep3_pctl_{int(time.time())}") as run:
        run_id = run.info.run_id
        print(f"[mlflow] run_id={run_id}")
        mlflow.log_params({
            "source_npz": str(NPZ_PATH),
            "npz_sha256": "bf513ec9eef5ffad47e841d30893222934f3537396fc75c59062e351e43b9838",
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "n_train": n_train, "n_holdout": n_hold,
            "feature_set": "multi_head_30",
            "method": "top_percentile_gate (v2 diagnostic)",
            "commission_ticks": COMMISSION_RT_TICKS,
            "oot_dates": ",".join(OOT_DATES),
        })

        all_verdicts = []
        feature_imp_dict = {}

        for side, y_full, filled_full in [
            ("long", data["long_net"], data["long_filled"]),
            ("short", data["short_net"], data["short_filled"]),
        ]:
            print(f"\n[train {side}] multi-head LGBM...")
            X_tr = X[train_mask]; X_ho = X[hold_mask]
            y_tr = y_full[train_mask]; y_ho = y_full[hold_mask]
            filled_ho = filled_full[hold_mask]; date_ho = data["date_idx"][hold_mask]
            net_realized = y_ho - COMMISSION_RT_TICKS

            model = train_lgbm(X_tr, y_tr, X_ho)
            pred_mh = model.predict(X_ho)

            # Single-head model
            single_idx = data["feature_names"].index("pred_log_ret_1s")
            model_s = train_lgbm(X_tr[:, single_idx:single_idx+1], y_tr, X_ho[:, single_idx:single_idx+1])
            pred_sh = model_s.predict(X_ho[:, single_idx:single_idx+1])

            # Raw 1s prediction as ranking baseline
            raw_1s = X_ho[:, single_idx]
            if side == "short":
                raw_1s = -raw_1s

            for top_pct in [50, 20, 10, 5, 2, 1]:
                vmh = percentile_gate_verdict(
                    f"multihead_lgbm_{side}", side, pred_mh,
                    net_realized, filled_ho, date_ho, top_pct)
                if vmh: all_verdicts.append(vmh)
                vsh = percentile_gate_verdict(
                    f"singlehead_lgbm_{side}", side, pred_sh,
                    net_realized, filled_ho, date_ho, top_pct)
                if vsh: all_verdicts.append(vsh)
                vraw = percentile_gate_verdict(
                    f"raw_signal_{side}", side, raw_1s,
                    net_realized, filled_ho, date_ho, top_pct)
                if vraw: all_verdicts.append(vraw)

                if vmh and vsh and vraw:
                    print(f"  top {top_pct}%: "
                          f"MH n={vmh['n_fills']:>5d} tpf={vmh['ticks_per_fill']:+.3f} sh={vmh['sharpe']:.2f} pf={vmh['pf']:.2f} wr={vmh['wr']:.1%} | "
                          f"SH n={vsh['n_fills']:>5d} tpf={vsh['ticks_per_fill']:+.3f} pf={vsh['pf']:.2f} | "
                          f"RAW n={vraw['n_fills']:>5d} tpf={vraw['ticks_per_fill']:+.3f} pf={vraw['pf']:.2f}")

            imp = model.feature_importances_
            ranked = sorted(zip(data["feature_names"], imp), key=lambda x: -x[1])
            feature_imp_dict[side] = [(n, int(v)) for n, v in ranked]
            print(f"  top-10 features ({side}): {ranked[:10]}")

        # Save verdicts
        df = pd.DataFrame(all_verdicts)
        csv = OUT_DIR / "lgbm_gate_v342_ep3_pctl_verdicts.csv"
        df.to_csv(csv, index=False)
        mlflow.log_artifact(str(csv))

        imp_path = OUT_DIR / "lgbm_gate_v342_ep3_pctl_feature_importance.json"
        with open(imp_path, "w") as f:
            json.dump(feature_imp_dict, f, indent=2, default=str)
        mlflow.log_artifact(str(imp_path))

        # Log key metrics (best Sharpe per side)
        for side in ("long", "short"):
            sub = [v for v in all_verdicts if v["name"] == f"multihead_lgbm_{side}"]
            if sub:
                best = max(sub, key=lambda v: v["sharpe"])
                for k in ("n_fills", "ticks_per_fill", "sharpe", "sortino", "pf", "wr", "day_conc", "top_pct"):
                    try:
                        mlflow.log_metric(f"best_mh_{side}_{k}", float(best[k]))
                    except Exception:
                        pass

        mlflow.log_metric("wall_time_seconds", time.time() - t0)

        # Print summary
        print("\n" + "=" * 110)
        print(f"{'name':<28} {'side':<5} {'top%':>5} {'n_fills':>8} {'tpf':>8} {'sharpe':>8} {'sortino':>8} {'pf':>6} {'wr':>7} {'day_conc':>9} {'hc344':>6}")
        print("=" * 110)
        for v in all_verdicts:
            pf_s = f"{v['pf']:.2f}" if v['pf'] != float('inf') else "inf"
            print(f"{v['name']:<28} {v['side']:<5} {v['top_pct']:>5d} "
                  f"{v['n_fills']:>8d} {v['ticks_per_fill']:>+8.3f} "
                  f"{v['sharpe']:>8.2f} {v['sortino']:>8.2f} {pf_s:>6} "
                  f"{v['wr']:>6.2%} {v['day_conc']:>9.3f} {str(v['pass_hc344']):>6}")
        print("=" * 110)
        print(f"\nMLflow run: http://jupiter:5000/#/experiments/906502745046598612/runs/{run_id}")
        return all_verdicts, run_id


if __name__ == "__main__":
    main()
