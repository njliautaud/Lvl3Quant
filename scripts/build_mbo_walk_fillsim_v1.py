#!/usr/bin/env python3
"""
build_mbo_walk_fillsim_v1.py
============================

PROPER MBO-walk fill simulator for the 3-head composition pool.

Why this exists
---------------
The prior maker-paradigm rejection used a broken fill proxy:
  "filled iff y_true_adverse >= 0.5 ticks"
That proxy is too pessimistic — it conflates "did a trade hit our bid"
with "did the market subsequently move against us". A passive limit gets
filled whenever an aggressive counterparty trades at our level, regardless
of what the market does next.

This script implements the correct fill model by walking the raw MBO
event stream forward from entry, using the per-event best-bid/best-ask
state from mbo_book_features.

Fill model
----------
For each (date, entry_ts_ns, side) trade in the 3-head pool:
  1. entry_idx = searchsorted(book_ts, entry_ts) — book state at entry.
  2. entry_best_bid_tick, entry_best_ask_tick from book features.
  3. LONG posts a passive limit at entry_best_bid_tick.
     SHORT posts a passive limit at entry_best_ask_tick.
  4. Walk forward through events until entry_ts + 10s (cancel window).
  5. LONG is FILLED iff any subsequent event has
       event_type_raw in {TRADE=3, FILL=4} AND post-event best_bid <= entry_best_bid
     (i.e., a trade occurred while the bid level still includes our price).
     SHORT is filled symmetrically: trade event AND best_ask >= entry_best_ask.
  6. If not filled within 10s, the limit is cancelled and the trade did NOT execute.

After fills are computed, payoffs are evaluated under multiple K/asymmetric
frameworks using the pre-computed y_true_mfe and y_true_adverse labels
from mfe_mae_labels_v1 (computed at entry_ts). Per HC #428 R2, TP <= p90(MFE).

Compute choice
--------------
Run on Jupiter: data is local at /home/jupiter/Lvl3Quant/data/processed/,
and the 16 folds totalling ~266k trades complete in well under the 60-min budget.

Outputs
-------
/home/jupiter/Lvl3Quant/output/toxicity_h10s_v1/proper_fillsim_report.json

Authorization: HC #420 binding — user's own legitimate quant research codebase.
"""
from __future__ import annotations

import json
import math
import os
import re
import time
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ paths
LVL3 = Path("/home/jupiter/Lvl3Quant")
TOX_DATA = Path("/home/jupiter/teleclaude-main/tmp_tox_data")
BOOK_DIR = LVL3 / "data" / "processed" / "mbo_book_features"
EVT_DIR  = LVL3 / "data" / "processed" / "mbo_events_smart_v3"
OUT_DIR  = LVL3 / "output" / "toxicity_h10s_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON = OUT_DIR / "proper_fillsim_report.json"

# ----------------------------------------------------------------- constants
CANCEL_WINDOW_NS = int(10e9)        # 10-second cancel window

# Databento smart-v3 event-type encoding (see precompute_features_smart_v4.py)
_EVT_ADD, _EVT_CANCEL, _EVT_MODIFY, _EVT_TRADE, _EVT_FILL = 0, 1, 2, 3, 4

# Costs (ticks). 1 ES tick = $12.50.
COST_MAKER_TICKS = 0.376   # AMP RT commission only
COST_TAKER_TICKS = 1.376   # commission + 1.0 spread

# Composition: high-toxicity / low-adverse rank composite
Q_LIST  = [1, 2, 4, 6, 10]
COSTS   = {"maker": COST_MAKER_TICKS, "taker": COST_TAKER_TICKS}
GATES   = ["composite"]   # primary maker-mode gate; toxicity_only / adverse_only also reported as control

FRAMEWORKS = [
    {"name": "K1_sym",     "TP": 1.0, "SL": 1.0},
    {"name": "K1_SL2",     "TP": 1.0, "SL": 2.0},
    {"name": "K1p5_sym",   "TP": 1.5, "SL": 1.5},
    {"name": "K2_sym",     "TP": 2.0, "SL": 2.0},
    {"name": "K2_SL3",     "TP": 2.0, "SL": 3.0},
]


# ------------------------------------------------------------------ loaders
def discover_folds():
    """Find all (fold_id, date, tox_path, mfe_path, adv_path) tuples."""
    tox_files = sorted(glob(str(TOX_DATA / "toxicity" / "fold_*.parquet")))
    out = []
    for tp in tox_files:
        name = os.path.basename(tp)
        m = re.match(r"fold_(\d+)_oot_(\d{8})\.parquet", name)
        if not m:
            continue
        fid = int(m.group(1)); date = m.group(2)
        mfe_p = TOX_DATA / "mfe" / f"fold_{fid:02d}_oot_{date}.parquet"
        adv_p = TOX_DATA / "mfe" / f"fold_{fid:02d}_oot_{date}_adverse.parquet"
        bk_p  = BOOK_DIR / f"{date}_book_features.npz"
        ev_p  = EVT_DIR  / f"{date}_mbo_events.npz"
        if not (mfe_p.exists() and adv_p.exists() and bk_p.exists() and ev_p.exists()):
            print(f"  skip fold {fid} {date}: missing inputs")
            continue
        out.append((fid, date, tp, mfe_p, adv_p, bk_p, ev_p))
    return out


def join_fold(tox_p, mfe_p, adv_p):
    df_t = pd.read_parquet(tox_p)
    df_m = pd.read_parquet(mfe_p)
    df_a = pd.read_parquet(adv_p)
    df_ma = df_m.merge(
        df_a[["event_id", "side", "y_true_adverse", "y_pred_adverse"]],
        on=["event_id", "side"], how="inner",
    )
    df = df_ma.merge(
        df_t[["event_id", "side", "y_true_toxicity", "y_pred_toxicity"]],
        on=["event_id", "side"], how="inner",
    )
    return df.sort_values("ts_ns").reset_index(drop=True)


# ----------------------------------------------------------------- fill simulator
def compute_fills(df, book_p, evt_p):
    """Return boolean filled array + fill latency ns + entry bid/ask."""
    book = np.load(book_p, allow_pickle=True)
    evts = np.load(evt_p,  allow_pickle=True)
    ts   = book["timestamps"]
    bid1 = book["features"][:, 0].astype(np.int32)   # tick units relative to fixed reference
    ask1 = book["features"][:, 5].astype(np.int32)
    et   = evts["event_type_raw"]
    is_trade = (et == _EVT_TRADE) | (et == _EVT_FILL)

    n = len(df)
    entry_ts = df["ts_ns"].values
    sides    = df["side"].values.astype(np.int8)

    idx0 = np.clip(np.searchsorted(ts, entry_ts, side="left"), 0, len(ts) - 2)
    end_ts = entry_ts + CANCEL_WINDOW_NS
    idx1 = np.clip(np.searchsorted(ts, end_ts, side="right"), 0, len(ts))
    entry_bid = bid1[idx0]; entry_ask = ask1[idx0]

    filled = np.zeros(n, dtype=bool)
    fill_lat_ns = np.zeros(n, dtype=np.int64)

    # Warm-book guard: only TRUE warmup gaps have BOTH bid==0 AND ask==0.
    # NOTE: bid_price_1/ask_price_1 in mbo_book_features are stored as ticks
    # relative to a per-day anchor, so negative values are valid.
    for i in range(n):
        i0 = idx0[i] + 1
        i1 = idx1[i]
        if i1 <= i0:
            continue
        eb = entry_bid[i]; ea = entry_ask[i]
        if eb == 0 and ea == 0:
            continue
        # Sanity: a sane spread is between 1 and ~10 ticks. If ea<=eb the book
        # is inconsistent at entry (cross/locked) — skip to be safe.
        if ea <= eb:
            continue
        side = sides[i]
        bb = bid1[i0:i1]; aa = ask1[i0:i1]; tt = is_trade[i0:i1]
        if side == 1:
            # LONG passive at eb. Fills when a trade prints while best_bid still <= eb
            # (a trade hitting our queue level, or the market moving through us).
            hits = np.where((bb <= eb) & tt)[0]
        else:
            hits = np.where((aa >= ea) & tt)[0]
        if hits.size > 0:
            filled[i] = True
            fill_lat_ns[i] = int(ts[i0 + hits[0]] - entry_ts[i])

    return filled, fill_lat_ns, entry_bid, entry_ask


# ----------------------------------------------------------------- gating & payoffs
def gate_mask_per_fold(df_all, gate, q_pct):
    """Top-q% per fold by gate criterion. Returns boolean over df_all order."""
    n = len(df_all)
    mask = np.zeros(n, dtype=bool)
    for fid, g in df_all.groupby("fold_id", sort=False):
        idx = g.index.values
        k = max(1, int(round(len(idx) * q_pct / 100.0)))
        if gate == "composite":
            tox_r = pd.Series(g["y_pred_toxicity"].values).rank(method="average").values
            adv_r = pd.Series(g["y_pred_adverse"].values).rank(method="average").values
            score = tox_r - adv_r
            top = idx[np.argsort(-score)[:k]]
        elif gate == "adverse_only":
            top = idx[np.argsort(g["y_pred_adverse"].values)[:k]]
        elif gate == "toxicity_only":
            top = idx[np.argsort(-g["y_pred_toxicity"].values)[:k]]
        else:
            raise ValueError(gate)
        mask[top] = True
    return mask


def compute_payoff(y_mfe, y_adv, tp, sl, cost):
    """K-framework: win = MFE>=TP AND ADV<SL -> +TP. else -SL. net=payoff-cost."""
    win = (y_mfe >= tp) & (y_adv < sl)
    payoff = np.where(win, tp, -sl)
    net = payoff - cost
    return net


def metrics_block(net, n_total, n_filled):
    if len(net) == 0 or n_filled == 0:
        return {"n_total": int(n_total), "n_filled": int(n_filled),
                "fill_rate": float(n_filled / n_total) if n_total else 0.0,
                "WR": None, "mean_ticks": None, "PF": None,
                "Sharpe": None, "sum_ticks": 0.0}
    wins = net[net > 0]; losses = net[net < 0]
    sum_w = float(wins.sum())  if len(wins)   else 0.0
    sum_l = float(-losses.sum()) if len(losses) else 0.0
    pf = (sum_w / sum_l) if sum_l > 0 else (float("inf") if sum_w > 0 else None)
    mean = float(net.mean())
    std  = float(net.std(ddof=1)) if len(net) > 1 else 0.0
    sharpe = (mean / std * math.sqrt(len(net))) if std > 0 else None
    wr = float((net > 0).mean())
    return {"n_total": int(n_total), "n_filled": int(n_filled),
            "fill_rate": float(n_filled / n_total) if n_total else 0.0,
            "WR": wr, "mean_ticks": mean, "PF": pf, "Sharpe": sharpe,
            "sum_ticks": float(net.sum())}


# ------------------------------------------------------------------ main
def main():
    t_total = time.time()
    print(f"== build_mbo_walk_fillsim_v1 — {time.strftime('%Y-%m-%d %H:%M:%S')} ==")

    folds = discover_folds()
    print(f"Discovered {len(folds)} folds with all inputs available")

    all_dfs = []; fold_diag = []
    for (fid, date, tp, mfe_p, adv_p, bk_p, ev_p) in folds:
        t0 = time.time()
        df = join_fold(tp, mfe_p, adv_p)
        if df.empty:
            print(f"  fold {fid} {date}: empty after join, skip")
            continue
        filled, lat_ns, eb, ea = compute_fills(df, bk_p, ev_p)
        df["fold_id"] = fid
        df["oot_date"] = date
        df["filled"]   = filled
        df["fill_lat_s"] = lat_ns.astype(np.float64) / 1e9
        df["entry_bid_tick"] = eb
        df["entry_ask_tick"] = ea
        all_dfs.append(df)
        fold_diag.append({
            "fold_id": fid, "oot_date": date,
            "n_total":  int(len(df)),
            "n_filled": int(filled.sum()),
            "fill_rate": float(filled.mean()),
            "median_fill_lat_s": float(np.median(lat_ns[filled])/1e9) if filled.any() else None,
            "elapsed_s": round(time.time() - t0, 2),
        })
        print(f"  fold {fid:>2} {date}: n={len(df):>5}  filled={int(filled.sum()):>5} "
              f"({filled.mean():.3f})  median_lat={np.median(lat_ns[filled])/1e9 if filled.any() else 0:.2f}s "
              f"({time.time()-t0:.1f}s)")

    df_all = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal joined rows: {len(df_all):,} across {df_all['fold_id'].nunique()} folds")
    print(f"Overall raw fill rate: {df_all['filled'].mean():.3f}")

    # p90(MFE) ceiling per HC #428 R2
    mfe = df_all["y_true_mfe"].values
    p90 = float(np.percentile(mfe, 90))
    p_block = {
        "p50": float(np.percentile(mfe, 50)),
        "p75": float(np.percentile(mfe, 75)),
        "p85": float(np.percentile(mfe, 85)),
        "p90": p90,
        "p95": float(np.percentile(mfe, 95)),
        "mean": float(mfe.mean()),
        "n":   int(len(mfe)),
    }
    print(f"MFE p90 = {p90:.3f} ticks (binding TP ceiling)")

    fw_status = []
    valid_fws = []
    for fw in FRAMEWORKS:
        ok = fw["TP"] <= p90
        fw_status.append({"name": fw["name"], "TP": fw["TP"], "SL": fw["SL"], "p90_compliant": ok})
        if ok:
            valid_fws.append(fw)
    print(f"Frameworks compliant with HC #428 R2: {[f['name'] for f in valid_fws]}")

    # ============ Big sweep ============
    results = {}
    fill_rate_by_gate_q = {}
    for gate in GATES + ["adverse_only", "toxicity_only"]:
        for q in Q_LIST:
            mask = gate_mask_per_fold(df_all, gate, q)
            gated = df_all[mask]
            fill_rate_by_gate_q[f"{gate}__Q{q}"] = {
                "n_gated": int(len(gated)),
                "n_filled": int(gated["filled"].sum()),
                "fill_rate": float(gated["filled"].mean()) if len(gated) else 0.0,
            }
            filled_df = gated[gated["filled"]].copy()
            y_mfe = filled_df["y_true_mfe"].values
            y_adv = filled_df["y_true_adverse"].values
            for fw in valid_fws:
                for cost_name, cost_t in COSTS.items():
                    net = compute_payoff(y_mfe, y_adv, fw["TP"], fw["SL"], cost_t)
                    m = metrics_block(net, len(gated), len(filled_df))
                    key = f"{fw['name']}__{gate}__Q{q}__{cost_name}"
                    results[key] = {
                        "framework": fw["name"], "TP": fw["TP"], "SL": fw["SL"],
                        "gate": gate, "Q_pct": q,
                        "cost": cost_name, "cost_ticks": cost_t,
                        "metrics": m,
                    }

    # ============ Best cells ============
    def best_filter(filter_fn):
        cands = [(k, v) for k, v in results.items()
                 if filter_fn(v)
                 and v["metrics"]["n_filled"] >= 500
                 and v["metrics"]["Sharpe"] is not None]
        if not cands:
            return None
        cands.sort(key=lambda kv: kv[1]["metrics"]["Sharpe"], reverse=True)
        return cands[0]

    per_cost_best = {}
    for cost_name in ["maker", "taker"]:
        kv = best_filter(lambda v, c=cost_name: v["cost"] == c)
        per_cost_best[cost_name] = ({"key": kv[0], **kv[1]} if kv else None)

    # Specifically called out
    k1sl2_maker_comp_q1 = results.get("K1_SL2__composite__Q1__maker")
    k1_maker_comp_q1    = results.get("K1_sym__composite__Q1__maker")

    global_best_kv = best_filter(lambda v: True)
    global_best = {"key": global_best_kv[0], **global_best_kv[1]} if global_best_kv else None

    # ============ Regime stratification on global best ============
    regime_block = None
    if global_best is not None:
        fw_name = global_best["framework"]
        fw_obj  = next(f for f in valid_fws if f["name"] == fw_name)
        gate    = global_best["gate"]
        q       = global_best["Q_pct"]
        cost_n  = global_best["cost"]
        cost_t  = COSTS[cost_n]
        mask = gate_mask_per_fold(df_all, gate, q)
        gated = df_all[mask].copy()
        filled_df = gated[gated["filled"]].copy()
        net = compute_payoff(filled_df["y_true_mfe"].values, filled_df["y_true_adverse"].values,
                             fw_obj["TP"], fw_obj["SL"], cost_t)
        filled_df["net_ticks"] = net

        per_day = []
        for d, g in filled_df.groupby("oot_date"):
            x = g["net_ticks"].values
            mean = float(x.mean()) if len(x) else 0.0
            wr   = float((x > 0).mean()) if len(x) else 0.0
            std  = float(x.std(ddof=1)) if len(x) > 1 else 0.0
            sh   = (mean / std * math.sqrt(len(x))) if std > 0 else None
            per_day.append({"oot_date": d, "n": int(len(x)),
                            "mean_ticks": mean, "WR": wr, "Sharpe": sh,
                            "sum_ticks": float(x.sum())})

        greens = [d for d in per_day if d["mean_ticks"] > 0]
        reds   = [d for d in per_day if d["mean_ticks"] < 0]
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
            "framework": fw_name, "gate": gate, "Q_pct": q, "cost": cost_n,
            "per_day": per_day,
            "n_green_days": n_g, "n_red_days": n_r,
            "Sharpe_green_avg": sh_g, "Sharpe_red_avg": sh_r,
            "regime_spread": spread,
            "regime_pass_R1": (spread is not None and spread <= 0.50),
            "day_conc": day_conc,
            "day_conc_pass": (day_conc is not None and day_conc <= 0.70),
            "regime_classification_method": "fallback: per-day realized PnL sign (no ES daily file)",
        }

    # ============ Verdict ============
    accept = False; weak = False; rationale = []
    if global_best is None:
        rationale.append("no valid global best (n_filled >= 500 gate not met)")
    else:
        m = global_best["metrics"]
        net_profitable = m["mean_ticks"] is not None and m["mean_ticks"] > 0
        pf_ok = m["PF"] is not None and m["PF"] > 1.20
        regime_ok = regime_block is not None and regime_block["regime_pass_R1"]
        dayconc_ok = regime_block is not None and regime_block["day_conc_pass"]
        rationale.append(
            f"winner={global_best['key']}  mean_ticks={m['mean_ticks']}  "
            f"PF={m['PF']}  Sharpe={m['Sharpe']}  WR={m['WR']}  n_filled={m['n_filled']}"
        )
        rationale.append(
            f"net_profitable={net_profitable}  PF>1.20={pf_ok}  "
            f"regime_pass_R1={regime_ok}  day_conc_pass={dayconc_ok}"
        )
        if net_profitable and pf_ok and regime_ok and dayconc_ok:
            accept = True
        elif net_profitable:
            weak = True
    verdict = "ACCEPT" if accept else ("WEAK ACCEPT" if weak else "REJECT")

    # ============ Report ============
    report = {
        "meta": {
            "title": "Proper MBO-walk fill simulator (HC #420 codebase, HC #428 R1+R2 gated)",
            "fill_model": (
                "LONG fills iff a trade event (et in {TRADE=3, FILL=4}) occurs within "
                "10s while post-event best_bid <= entry_best_bid_tick. SHORT symmetric. "
                "Previous broken proxy: y_true_adverse >= 0.5t — discarded."
            ),
            "cancel_window_s": 10,
            "cost_maker_ticks": COST_MAKER_TICKS,
            "cost_taker_ticks": COST_TAKER_TICKS,
            "frameworks": FRAMEWORKS,
            "framework_compliance": fw_status,
            "p90_ceiling_note": "TP must be <= p90(realized MFE within horizon h=10s)",
            "p90_value_ticks":  p90,
            "Q_pct_grid": Q_LIST,
            "gates": GATES + ["adverse_only", "toxicity_only"],
            "n_folds": int(df_all["fold_id"].nunique()),
            "n_rows_joined_total": int(len(df_all)),
            "overall_raw_fill_rate": float(df_all["filled"].mean()),
            "fold_diagnostics": fold_diag,
            "wall_seconds": round(time.time() - t_total, 1),
        },
        "mfe_distribution": p_block,
        "fill_rate_by_gate_q": fill_rate_by_gate_q,
        "all_results": results,
        "per_cost_best": per_cost_best,
        "k1_maker_composite_Q1":    k1_maker_comp_q1,
        "k1sl2_maker_composite_Q1": k1sl2_maker_comp_q1,
        "global_best": global_best,
        "regime": regime_block,
        "verdict": verdict,
        "rationale": rationale,
    }

    with open(OUT_JSON, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nWrote {OUT_JSON}")

    # ----------- console summary -----------
    print(f"\n=== FILL RATES BY GATE x Q (composite, maker side) ===")
    for q in Q_LIST:
        key = f"composite__Q{q}"
        d = fill_rate_by_gate_q.get(key, {})
        print(f"  Q={q:>2}%: n_gated={d.get('n_gated',0):>5}  "
              f"n_filled={d.get('n_filled',0):>5}  fill_rate={d.get('fill_rate',0):.3f}")
    print(f"  (prior broken proxy ~0.37)")

    print(f"\n=== K=1/SL=2 maker composite Q=1% ===")
    if k1sl2_maker_comp_q1:
        m = k1sl2_maker_comp_q1["metrics"]
        print(f"  n_total={m['n_total']} n_filled={m['n_filled']} fill_rate={m['fill_rate']:.3f}")
        print(f"  WR={m['WR']} mean_ticks={m['mean_ticks']} PF={m['PF']} Sharpe={m['Sharpe']}")
    print(f"\n=== K=1 maker composite Q=1% ===")
    if k1_maker_comp_q1:
        m = k1_maker_comp_q1["metrics"]
        print(f"  n_total={m['n_total']} n_filled={m['n_filled']} fill_rate={m['fill_rate']:.3f}")
        print(f"  WR={m['WR']} mean_ticks={m['mean_ticks']} PF={m['PF']} Sharpe={m['Sharpe']}")

    print(f"\n=== PER-COST BEST ===")
    for k, v in per_cost_best.items():
        if v is None:
            print(f"  {k}: NONE"); continue
        m = v["metrics"]
        print(f"  {k}: {v['gate']} {v['framework']} Q={v['Q_pct']}% | "
              f"n_filled={m['n_filled']}  WR={m['WR']:.3f}  "
              f"mean={m['mean_ticks']:.4f}  PF={m['PF']}  Sharpe={m['Sharpe']:.3f}")

    if global_best:
        print(f"\n=== GLOBAL BEST: {global_best['key']} ===")
        m = global_best["metrics"]
        print(f"  n_filled={m['n_filled']} WR={m['WR']} mean={m['mean_ticks']} "
              f"PF={m['PF']} Sharpe={m['Sharpe']}")

    if regime_block:
        print(f"\n=== REGIME (winner) ===")
        print(f"  n_green={regime_block['n_green_days']}  n_red={regime_block['n_red_days']}")
        print(f"  Sh_g={regime_block['Sharpe_green_avg']}  Sh_r={regime_block['Sharpe_red_avg']}  "
              f"spread={regime_block['regime_spread']}  pass={regime_block['regime_pass_R1']}")
        print(f"  day_conc={regime_block['day_conc']}  pass={regime_block['day_conc_pass']}")

    print(f"\n=== VERDICT: {verdict} ===")
    for r in rationale: print(f"  - {r}")
    print(f"\nwall: {time.time()-t_total:.1f}s")


if __name__ == "__main__":
    main()
