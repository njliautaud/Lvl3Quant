#!/usr/bin/env python3
"""
Grade ONE fold of the h5s low-LR multifold training.
Called by h5s_multifold_auto_grader.sh once per newly-landed fold.

Per HC #420: user's own quant codebase, full authorization.

For the fold's OOT prediction file, runs:
  (A) Label-FIFO 12-cell sweep (1s/5s × top-1/2/5/10/20/50%) — fast screening
  (B) Canonical FIFOReplayEngine grade on surviving cells (1s_top10%, 1s_top5%, 5s_top10%)
  (C) Fill-prob gate at q>=0.50 (best gate from h5s_fold0_fillprob_gated_REPORT.md)

Writes:
  output/h5s_multifold_fold{NN}_fifo_REPORT.md           (full per-fold report)
  output/h5s_multifold_fold{NN}_fifo/summary.csv         (label-FIFO 12 cells)
  output/h5s_multifold_fold{NN}_canonical/summary.csv    (canonical + gate)
  Appends one-line headline to output/h5s_multifold_AGGREGATE.md

Usage:
  python3 h5s_multifold_grade_one_fold.py <fold_id> <local_pred_path>
"""
from __future__ import annotations

import sys
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

LVL3_ROOT = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(LVL3_ROOT))

MBO_EVENT_DIR = LVL3_ROOT / "data" / "processed" / "mbo_events_smart_v3"
FILL_PROB_MODEL = LVL3_ROOT / "models" / "fill_prob_head_v1.lgb"
AGGREGATE_PATH = LVL3_ROOT / "output" / "h5s_multifold_AGGREGATE.md"

WINDOW_SIZE = 1000
STRIDE      = 500
COMMISSION  = 0.376
HORIZONS    = ["1s", "5s"]
TOP_PCTS    = [1, 2, 5, 10, 20, 50]
SURVIVING_CELLS = [("1s", 10), ("1s", 5), ("5s", 10)]
GATE_QUANTILE   = 0.50
PROXY_HOLD_S    = 0.415

TP_TICKS = 2.0
SL_TICKS = 1.0
H_SEC_BY_HORIZON = {"1s": 1.0, "5s": 5.0}
HOLD_FACTOR  = 1.5
CANCEL_FACTOR = 1.0

ANNUAL_SQRT = np.sqrt(252 * 6.5 * 3600)


def label_fifo_sweep(preds_2d: np.ndarray, labels_by_h: Dict[str, np.ndarray]) -> List[dict]:
    """The 12-cell label-FIFO sweep. Same protocol as h5s_lowlr_fifo_sweep.py."""
    n_windows = preds_2d.shape[0]
    event_idx = np.arange(n_windows) * STRIDE + (WINDOW_SIZE - 1)
    results = []
    for h_i, h in enumerate(HORIZONS):
        p = preds_2d[:, h_i]
        lab = labels_by_h[h][event_idx]
        valid = ~np.isnan(lab)
        p_v, lab_v = p[valid], lab[valid]
        abs_p = np.abs(p_v)
        sign_p = np.sign(p_v)
        for pct in TOP_PCTS:
            k = max(1, int(np.ceil(len(p_v) * pct / 100.0)))
            idx_sorted = np.argsort(-abs_p)[:k]
            sig = sign_p[idx_sorted]
            l = lab_v[idx_sorted]
            per_trade = sig * l - COMMISSION
            net = float(per_trade.sum())
            mean = float(per_trade.mean())
            std = float(per_trade.std(ddof=1)) if len(per_trade) > 1 else 0.0
            sharpe = (mean / std * ANNUAL_SQRT) if std > 0 else 0.0
            results.append({
                "horizon": h, "top_pct": pct, "net_ticks": net,
                "net_per_trade": mean, "count": int(len(per_trade)),
                "sharpe": sharpe, "positive": int(net > 0),
            })
    return results


def select_topk_by_abs(preds: np.ndarray, pct: float):
    abs_p = np.abs(preds)
    n = preds.shape[0]
    k = max(1, int(np.ceil(n * pct / 100.0)))
    idx = np.argsort(-abs_p)[:k]
    sign = np.sign(preds[idx])
    return idx, sign, abs_p[idx]


def map_idx_to_ts_ns(date_str: str, idx_in_window_space: np.ndarray) -> np.ndarray:
    mbo_path = MBO_EVENT_DIR / f"{date_str}_mbo_events.npz"
    mbo = np.load(mbo_path, allow_pickle=False)
    ts_events = mbo["timestamps"].astype(np.int64)
    n_events = len(ts_events)
    event_idx = np.minimum(idx_in_window_space * STRIDE + WINDOW_SIZE - 1, n_events - 1)
    return ts_events[event_idx]


def run_fifo_replay(date_str: str, signals: List[dict], h_sec: float) -> pd.DataFrame:
    from alpha_discovery.deep_models.fifo_market_replay import FIFOReplayEngine
    engine = FIFOReplayEngine(
        date=date_str,
        cancel_after_ns=int(CANCEL_FACTOR * h_sec * 1_000_000_000),
        max_hold_ns=int(HOLD_FACTOR * h_sec * 1_000_000_000),
    )
    trades = engine.simulate(signals=signals, tp_ticks=TP_TICKS, sl_ticks=SL_TICKS, order_type="limit")
    rows = []
    for t in trades:
        hold_s = (t.exit_ts_ns - t.entry_ts_ns) / 1e9 if (t.entry_ts_ns and t.exit_ts_ns) else 0.0
        rows.append({
            "date": date_str, "direction": t.direction, "hold_s": float(hold_s),
            "fill_type": t.exit_reason, "net_ticks": float(t.pnl_ticks_net),
            "queue_ahead": int(t.queue_ahead), "queue_wait_ns": int(t.queue_wait_ns),
            "slippage_ticks": float(t.slippage_ticks),
            "pred_strength": float(t.pred_strength),
            "entry_ts_ns": int(t.entry_ts_ns or 0),
        })
    return pd.DataFrame(rows)


def compute_fill_features(df: pd.DataFrame) -> pd.DataFrame:
    feat = pd.DataFrame(index=df.index)
    feat["queue_ahead"] = df["queue_ahead"].astype(float)
    feat["queue_ahead_log"] = np.log1p(feat["queue_ahead"])
    feat["pred_strength"] = df["pred_strength"].astype(float)
    feat["pred_strength_squared"] = feat["pred_strength"] ** 2
    feat["direction_binary"] = (df["direction"] == "short").astype(int)
    ts_ns = df["entry_ts_ns"].astype(np.int64).values
    sec_in_day = (ts_ns / 1e9) % 86400.0
    feat["time_of_day_hour"] = sec_in_day / 3600.0
    days_since_epoch = (ts_ns // (86400 * 1_000_000_000)).astype(int)
    feat["day_of_week"] = (days_since_epoch + 3) % 7
    feat["hold_s"] = PROXY_HOLD_S
    feat["hold_s_log"] = np.log1p(PROXY_HOLD_S)
    feat["queue_wait_ns"] = df["queue_wait_ns"].astype(float)
    return feat[["queue_ahead", "queue_ahead_log", "pred_strength", "pred_strength_squared",
                 "direction_binary", "time_of_day_hour", "day_of_week",
                 "hold_s", "hold_s_log", "queue_wait_ns"]]


def canonical_fifo_with_gate(preds_2d: np.ndarray, date_str: str, fill_model) -> List[dict]:
    """Run canonical FIFO + fill-prob gate on surviving cells."""
    horizon_idx = {"1s": 0, "5s": 1}
    out = []
    for h, pct in SURVIVING_CELLS:
        h_sec = H_SEC_BY_HORIZON[h]
        preds = preds_2d[:, horizon_idx[h]]
        idx, sign, strength = select_topk_by_abs(preds, pct)
        ts_ns = map_idx_to_ts_ns(date_str, idx)
        signals = [{"ts_ns": int(ts_ns[i]),
                    "direction": "long" if sign[i] > 0 else "short",
                    "strength": float(strength[i])} for i in range(len(idx))]
        if not signals:
            out.append({"horizon": h, "top_pct": pct, "ungated": None, "gated": None})
            continue
        fills = run_fifo_replay(date_str, signals, h_sec)
        if fills.empty:
            out.append({"horizon": h, "top_pct": pct, "ungated": None, "gated": None})
            continue
        feats = compute_fill_features(fills)
        p_fill = fill_model.predict(feats.values)
        fills["p_fill"] = p_fill
        # Ungated stats
        n_u = len(fills)
        net_u = float(fills["net_ticks"].sum())
        per_u = float(fills["net_ticks"].mean())
        wr_u = float((fills["net_ticks"] > 0).mean())
        gp = float(fills.loc[fills["net_ticks"] > 0, "net_ticks"].sum())
        gl = float(-fills.loc[fills["net_ticks"] < 0, "net_ticks"].sum())
        pf_u = (gp / gl) if gl > 0 else float("inf")
        # Quantile gate
        thr_q = float(np.quantile(fills["p_fill"].values, GATE_QUANTILE))
        keep = fills["p_fill"] >= thr_q
        n_g = int(keep.sum())
        if n_g > 0:
            sub = fills[keep]
            net_g = float(sub["net_ticks"].sum())
            per_g = float(sub["net_ticks"].mean())
            wr_g = float((sub["net_ticks"] > 0).mean())
            gpg = float(sub.loc[sub["net_ticks"] > 0, "net_ticks"].sum())
            glg = float(-sub.loc[sub["net_ticks"] < 0, "net_ticks"].sum())
            pf_g = (gpg / glg) if glg > 0 else float("inf")
        else:
            net_g, per_g, wr_g, pf_g = 0.0, 0.0, 0.0, 0.0
        out.append({
            "horizon": h, "top_pct": pct,
            "ungated": {"n": n_u, "net_ticks": net_u, "net_per_trade": per_u, "wr": wr_u, "pf": pf_u},
            "gated":   {"n": n_g, "net_ticks": net_g, "net_per_trade": per_g, "wr": wr_g, "pf": pf_g, "threshold": thr_q},
        })
    return out


def append_aggregate_row(fold_id: int, date_str: str, label_results: List[dict],
                         canonical_results: List[dict]):
    """Append one headline row per fold to the aggregate file."""
    AGGREGATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    first_write = not AGGREGATE_PATH.exists()
    # Pull headline cells
    lf_1s_top10 = next((r for r in label_results if r["horizon"] == "1s" and r["top_pct"] == 10), None)
    cf_1s_top10 = next((r for r in canonical_results if r["horizon"] == "1s" and r["top_pct"] == 10), None)
    label_str = f"{lf_1s_top10['net_per_trade']:+.3f}" if lf_1s_top10 else "n/a"
    label_n = str(lf_1s_top10["count"]) if lf_1s_top10 else "n/a"
    if cf_1s_top10 and cf_1s_top10["ungated"]:
        canon_str = f"{cf_1s_top10['ungated']['net_per_trade']:+.3f}"
        canon_n = str(cf_1s_top10["ungated"]["n"])
        gated_str = f"{cf_1s_top10['gated']['net_per_trade']:+.3f}" if cf_1s_top10["gated"]["n"] > 0 else "n/a"
        gated_n = str(cf_1s_top10["gated"]["n"]) if cf_1s_top10["gated"]["n"] > 0 else "n/a"
    else:
        canon_str = canon_n = gated_str = gated_n = "n/a"

    with open(AGGREGATE_PATH, "a") as f:
        if first_write:
            f.write("# h5s low-LR multifold — auto-graded aggregate\n\n")
            f.write("Headline cell: 1s_top10% (best fold-0 surviving). "
                    "Auto-graded as folds land. Updated by `h5s_multifold_auto_grader.sh` (15-min cron).\n\n")
            f.write("| Fold | OOT Date | Label-FIFO net/trade | Label N | Canonical net/trade | Canon N | Gated (q>=0.5) net/trade | Gated N |\n")
            f.write("|------|----------|---------------------|---------|---------------------|---------|--------------------------|--------|\n")
        f.write(f"| {fold_id:02d} | {date_str} | {label_str} | {label_n} | {canon_str} | {canon_n} | {gated_str} | {gated_n} |\n")


def write_fold_report(fold_id: int, date_str: str, label_results, canonical_results):
    out_dir_label = LVL3_ROOT / "output" / f"h5s_multifold_fold{fold_id:02d}_fifo"
    out_dir_label.mkdir(parents=True, exist_ok=True)
    out_dir_canon = LVL3_ROOT / "output" / f"h5s_multifold_fold{fold_id:02d}_canonical"
    out_dir_canon.mkdir(parents=True, exist_ok=True)

    # Label-FIFO CSV
    with open(out_dir_label / "summary.csv", "w") as f:
        f.write("horizon,top_pct,net_ticks,net_per_trade,count,sharpe,positive\n")
        for r in label_results:
            f.write(f"{r['horizon']},{r['top_pct']},{r['net_ticks']:.2f},{r['net_per_trade']:.4f},"
                    f"{r['count']},{r['sharpe']:.2f},{r['positive']}\n")

    # Canonical+gate CSV
    with open(out_dir_canon / "summary.csv", "w") as f:
        f.write("horizon,top_pct,gate,n,net_ticks,net_per_trade,wr,pf,threshold\n")
        for r in canonical_results:
            for gate_kind in ["ungated", "gated"]:
                g = r[gate_kind]
                if g is None:
                    f.write(f"{r['horizon']},{r['top_pct']},{gate_kind},0,0,0,0,0,\n")
                    continue
                thr = g.get("threshold", "")
                f.write(f"{r['horizon']},{r['top_pct']},{gate_kind},{g['n']},{g['net_ticks']:.2f},"
                        f"{g['net_per_trade']:.4f},{g['wr']:.3f},{g['pf']:.2f},{thr}\n")

    # Markdown report
    report = [
        f"# h5s multifold fold-{fold_id:02d} FIFO Grade",
        "",
        f"**OOT date**: {date_str}",
        f"**Gate**: q≥{GATE_QUANTILE} fill-prob (from fill_prob_head_v1.lgb, proxy hold {PROXY_HOLD_S} s)",
        f"**FIFO**: canonical (queue-aware passive limit, ES 0.376 commission, TP 2 / SL 1)",
        "",
        "## Headline cells (canonical + gate)",
        "",
        "| Cell | Ungated n | Ungated /trade | Gated n | Gated /trade | WR (gated) | PF (gated) |",
        "|------|-----------|----------------|---------|--------------|------------|------------|",
    ]
    for r in canonical_results:
        cell = f"{r['horizon']}_top{r['top_pct']}%"
        u, g = r["ungated"], r["gated"]
        if u is None:
            report.append(f"| {cell} | n/a | n/a | n/a | n/a | n/a | n/a |")
            continue
        gn = g["n"] if g else 0
        gp = f"{g['net_per_trade']:+.3f}" if (g and g["n"] > 0) else "n/a"
        gwr = f"{g['wr']:.3f}" if (g and g["n"] > 0) else "n/a"
        gpf = f"{g['pf']:.2f}" if (g and g["n"] > 0) else "n/a"
        report.append(f"| {cell} | {u['n']} | {u['net_per_trade']:+.3f} | {gn} | {gp} | {gwr} | {gpf} |")
    report += [
        "",
        "## Label-FIFO 12-cell screen",
        "",
        "| Horizon | Top% | Trades | Net Ticks | /Trade | Sharpe | Pos? |",
        "|---------|------|--------|-----------|--------|--------|------|",
    ]
    for r in label_results:
        report.append(f"| {r['horizon']} | {r['top_pct']}% | {r['count']} | {r['net_ticks']:.2f} | "
                      f"{r['net_per_trade']:+.4f} | {r['sharpe']:.2f} | {'+' if r['positive'] else '-'} |")
    report += [
        "",
        "## Files",
        "",
        f"- Label-FIFO CSV: `output/h5s_multifold_fold{fold_id:02d}_fifo/summary.csv`",
        f"- Canonical+gate CSV: `output/h5s_multifold_fold{fold_id:02d}_canonical/summary.csv`",
        f"- Aggregate row appended to `output/h5s_multifold_AGGREGATE.md`",
        "",
        f"Auto-graded by `scripts/h5s_multifold_auto_grader.sh`.",
    ]
    report_path = LVL3_ROOT / "output" / f"h5s_multifold_fold{fold_id:02d}_fifo_REPORT.md"
    report_path.write_text("\n".join(report))


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 h5s_multifold_grade_one_fold.py <fold_id> <local_pred_path>")
        sys.exit(2)
    fold_id = int(sys.argv[1])
    pred_path = Path(sys.argv[2])

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger(f"grade_fold_{fold_id}")
    log.info(f"Grading fold {fold_id} from {pred_path}")

    if not pred_path.exists():
        log.error(f"Prediction file not found: {pred_path}")
        sys.exit(2)

    d = np.load(pred_path, allow_pickle=True)
    preds = d["predictions"].astype(np.float32)
    oot_files = list(d["oot_files"])
    if not oot_files:
        log.error("No oot_files in predictions npz")
        sys.exit(2)
    date_str = Path(str(oot_files[0])).stem.replace("_mbo_events", "")

    # Label-FIFO 12 cells
    mbo_path = MBO_EVENT_DIR / f"{date_str}_mbo_events.npz"
    if not mbo_path.exists():
        log.error(f"MBO file missing for {date_str}: {mbo_path}")
        sys.exit(3)
    mbo = np.load(mbo_path, allow_pickle=True)
    labels_by_h = {"1s": mbo["labels_1s"], "5s": mbo["labels_5s"]}
    log.info(f"Running label-FIFO 12-cell sweep for {date_str}")
    label_results = label_fifo_sweep(preds, labels_by_h)

    # Canonical + gate
    import lightgbm as lgb
    fill_model = lgb.Booster(model_file=str(FILL_PROB_MODEL))
    log.info("Running canonical FIFO + fill-prob gate for surviving cells")
    canonical_results = canonical_fifo_with_gate(preds, date_str, fill_model)

    # Persist
    write_fold_report(fold_id, date_str, label_results, canonical_results)
    append_aggregate_row(fold_id, date_str, label_results, canonical_results)

    log.info(f"Fold {fold_id} ({date_str}) graded. Reports written.")
    print(f"GRADED_FOLD={fold_id:02d}")


if __name__ == "__main__":
    main()
