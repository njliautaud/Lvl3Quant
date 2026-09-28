#!/usr/bin/env python3
"""
HC #495.1 — Fill-prob-gated FIFO re-grade of h5s multifold fold-0 surviving cells.

Per HC #420 authorization (user's own quant codebase).

For each surviving cell from the fold-0 grade (1s_top10%, 1s_top5%, 5s_top10%):
  1. Run canonical FIFO replay (FIFOReplayEngine, passive limit, ES costs, queue-aware).
  2. Score each fill ex-ante through fill_prob_head_v1.lgb.
  3. Gate at thresholds [0.4, 0.5, 0.6, 0.7] -- only accept trades where p(fill) >= threshold.
  4. Compute FIFO net t/trade, n trades, WR, PF at each gate level vs ungated.

Question: does the fill-prob gate IMPROVE net (filter adverse-selection) or just shrink n?

Outputs:
  output/h5s_fold0_fillprob_gated/summary.csv
  output/h5s_fold0_fillprob_gated_REPORT.md
"""
from __future__ import annotations

import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb

LVL3_ROOT = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(LVL3_ROOT))

# Inputs
PRED_NPZ = LVL3_ROOT / "output" / "cnn_mamba_v3_h5s_lowlr_multifold" / "fold_00_oot_predictions.npz"
MBO_EVENT_DIR = LVL3_ROOT / "data" / "processed" / "mbo_events_smart_v3"
FILL_PROB_MODEL = LVL3_ROOT / "models" / "fill_prob_head_v1.lgb"
OUT_DIR = LVL3_ROOT / "output" / "h5s_fold0_fillprob_gated"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Window mapping for h5s low-LR multifold
WINDOW_SIZE = 1000
STRIDE      = 500
COMMISSION  = 0.376

# Replay params (passive limit, h=1s/5s bounds)
TP_TICKS = 2.0
SL_TICKS = 1.0
H_SEC_BY_HORIZON = {"1s": 1.0, "5s": 5.0}
HOLD_FACTOR = 1.5    # max_hold_s = HOLD_FACTOR * h
CANCEL_FACTOR = 1.0  # cancel_after_s = CANCEL_FACTOR * h

# Surviving cells from fold-0 grade
SURVIVING_CELLS = [
    ("1s", 10),
    ("1s", 5),
    ("5s", 10),
]

# Gate thresholds — original spec [0.4, 0.5, 0.6, 0.7] PLUS calibrated thresholds
# at the actual p_fill distribution (the proxy-hold model outputs a narrow [0.30, 0.36] band).
# We report both. Calibrated = top quartile/median/p25 of observed p_fill.
GATE_THRESHOLDS = [0.4, 0.5, 0.6, 0.7]
CALIBRATED_QUANTILES = [0.25, 0.50, 0.75]  # keep top 75/50/25% by p_fill

# Median hold from training data (used as ex-ante proxy)
PROXY_HOLD_S = 0.415  # median observed hold in fifo_v7_grade/fills.parquet

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(OUT_DIR / "run.log"), logging.StreamHandler()],
)
log = logging.getLogger("h5s_fillprob_gated")


def select_topk_by_abs(preds: np.ndarray, pct: float):
    """Select top-pct by |pred|. Returns (idx_in_window_space, sign, strength)."""
    abs_p = np.abs(preds)
    n = preds.shape[0]
    k = max(1, int(np.ceil(n * pct / 100.0)))
    idx = np.argsort(-abs_p)[:k]
    sign = np.sign(preds[idx])
    return idx, sign, abs_p[idx]


def map_idx_to_ts_ns(date_str: str, idx_in_window_space: np.ndarray) -> np.ndarray:
    """Map window indices to MBO event timestamps."""
    mbo_path = MBO_EVENT_DIR / f"{date_str}_mbo_events.npz"
    mbo = np.load(mbo_path, allow_pickle=False)
    ts_events = mbo["timestamps"].astype(np.int64)
    n_events = len(ts_events)
    event_idx = np.minimum(idx_in_window_space * STRIDE + WINDOW_SIZE - 1, n_events - 1)
    return ts_events[event_idx]


def run_fifo_replay(date_str: str, signals: List[dict], h_sec: float) -> pd.DataFrame:
    """Run canonical FIFO replay and return per-trade DataFrame."""
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
            "date": date_str,
            "direction": t.direction,
            "hold_s": float(hold_s),
            "fill_type": t.exit_reason,
            "net_ticks": float(t.pnl_ticks_net),
            "net_dollars": float(t.pnl_dollars),
            "queue_ahead": int(t.queue_ahead),
            "queue_wait_ns": int(t.queue_wait_ns),
            "slippage_ticks": float(t.slippage_ticks),
            "pred_strength": float(t.pred_strength),
            "entry_ts_ns": int(t.entry_ts_ns or 0),
        })
    return pd.DataFrame(rows)


def compute_fill_features(df: pd.DataFrame, use_proxy_hold: bool = True) -> pd.DataFrame:
    """Compute the 10 features the fill_prob_head_v1 model expects.

    use_proxy_hold=True: replace hold_s with the median observed hold (ex-ante / live-realistic).
    use_proxy_hold=False: use realized hold_s (post-hoc / cheating).
    """
    feat = pd.DataFrame(index=df.index)
    feat["queue_ahead"] = df["queue_ahead"].astype(float)
    feat["queue_ahead_log"] = np.log1p(feat["queue_ahead"])
    feat["pred_strength"] = df["pred_strength"].astype(float)
    feat["pred_strength_squared"] = feat["pred_strength"] ** 2
    # direction_binary: 1 if short, 0 if long
    feat["direction_binary"] = (df["direction"] == "short").astype(int)
    # time_of_day_hour: derive from entry_ts_ns
    ts_ns = df["entry_ts_ns"].astype(np.int64).values
    ts_sec_utc = ts_ns / 1e9
    # ES futures RTH starts 9:30 ET (13:30 UTC during EDT). Use UTC hour decomp.
    # Use modulo seconds-per-day, then convert to hour+minute/60
    sec_in_day = ts_sec_utc % 86400.0
    feat["time_of_day_hour"] = sec_in_day / 3600.0
    # day_of_week: derive from epoch (days since epoch, % 7)
    days_since_epoch = (ts_ns // (86400 * 1_000_000_000)).astype(int)
    # Thu Jan 1 1970 = day 0 -> dow 3 (Thu). Adjust to Mon=0.
    feat["day_of_week"] = (days_since_epoch + 3) % 7
    # hold_s features
    if use_proxy_hold:
        feat["hold_s"] = PROXY_HOLD_S
        feat["hold_s_log"] = np.log1p(PROXY_HOLD_S)
    else:
        feat["hold_s"] = df["hold_s"].astype(float)
        feat["hold_s_log"] = np.log1p(feat["hold_s"])
    feat["queue_wait_ns"] = df["queue_wait_ns"].astype(float)
    # Order columns to match training
    return feat[["queue_ahead", "queue_ahead_log", "pred_strength", "pred_strength_squared",
                 "direction_binary", "time_of_day_hour", "day_of_week",
                 "hold_s", "hold_s_log", "queue_wait_ns"]]


def grade_one_cell(preds_2d: np.ndarray, date_str: str, horizon: str, pct: int, h_idx: int,
                   fill_model: lgb.Booster) -> dict:
    """Run FIFO + fill-prob gate for one (horizon, pct) cell."""
    h_sec = H_SEC_BY_HORIZON[horizon]
    preds = preds_2d[:, h_idx]

    idx, sign, strength = select_topk_by_abs(preds, pct)
    ts_ns = map_idx_to_ts_ns(date_str, idx)

    signals = []
    for i in range(len(idx)):
        direction = "long" if sign[i] > 0 else "short"
        signals.append({"ts_ns": int(ts_ns[i]), "direction": direction, "strength": float(strength[i])})

    if not signals:
        return None

    log.info(f"  [{horizon}_top{pct}%] simulating {len(signals)} signals on {date_str}")
    fills = run_fifo_replay(date_str, signals, h_sec)

    if fills.empty:
        log.warning(f"  [{horizon}_top{pct}%] no fills produced")
        return None

    # Compute ex-ante fill-prob features (proxy hold)
    feats = compute_fill_features(fills, use_proxy_hold=True)
    p_fill = fill_model.predict(feats.values)
    fills["p_fill"] = p_fill

    # Compute per-cell stats: ungated + per threshold
    n = len(fills)
    net_total = fills["net_ticks"].sum()
    net_per = fills["net_ticks"].mean()
    wins = (fills["net_ticks"] > 0).sum()
    wr = wins / n if n else 0.0
    gross_profit = fills.loc[fills["net_ticks"] > 0, "net_ticks"].sum()
    gross_loss = -fills.loc[fills["net_ticks"] < 0, "net_ticks"].sum()
    pf = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    result = {
        "horizon": horizon,
        "top_pct": pct,
        "ungated": {
            "n": int(n),
            "net_ticks": float(net_total),
            "net_per_trade": float(net_per),
            "wr": float(wr),
            "pf": float(pf),
        },
        "gated": {},
        "p_fill_stats": {
            "min": float(p_fill.min()), "max": float(p_fill.max()),
            "mean": float(p_fill.mean()), "median": float(np.median(p_fill)),
            "p25": float(np.percentile(p_fill, 25)), "p75": float(np.percentile(p_fill, 75)),
        },
    }

    def _stats_for_keep(keep_mask, label):
        n_k = int(keep_mask.sum())
        if n_k == 0:
            return {"n": 0, "net_ticks": 0.0, "net_per_trade": 0.0, "wr": 0.0, "pf": 0.0, "uplift_per_trade": 0.0}
        sub = fills[keep_mask]
        net_t = float(sub["net_ticks"].sum())
        net_p = float(sub["net_ticks"].mean())
        wins_k = int((sub["net_ticks"] > 0).sum())
        wr_k = wins_k / n_k
        gp = float(sub.loc[sub["net_ticks"] > 0, "net_ticks"].sum())
        gl = float(-sub.loc[sub["net_ticks"] < 0, "net_ticks"].sum())
        pf_k = (gp / gl) if gl > 0 else float("inf")
        return {
            "n": n_k, "net_ticks": net_t, "net_per_trade": net_p,
            "wr": wr_k, "pf": pf_k, "uplift_per_trade": net_p - net_per,
        }

    for thr in GATE_THRESHOLDS:
        result["gated"][thr] = _stats_for_keep(fills["p_fill"] >= thr, f"p>={thr}")

    # Calibrated quantile gates (top X% by p_fill)
    result["gated_cal"] = {}
    for q in CALIBRATED_QUANTILES:
        # Keep observations with p_fill >= the q-th quantile
        thr_q = float(np.quantile(fills["p_fill"].values, q))
        keep = fills["p_fill"] >= thr_q
        result["gated_cal"][q] = {
            "threshold_value": thr_q,
            **_stats_for_keep(keep, f"q>={q}"),
        }

    # Save fills with p_fill annotation
    fills.to_parquet(OUT_DIR / f"fills_{horizon}_top{pct}.parquet")

    return result


def main():
    t0 = time.time()
    log.info("=" * 80)
    log.info("Loading fold-0 predictions")
    d = np.load(PRED_NPZ, allow_pickle=True)
    preds = d["predictions"].astype(np.float32)
    horizons = list(d["horizons"])
    oot_files = list(d["oot_files"])
    log.info(f"preds shape: {preds.shape}, horizons: {horizons}")
    log.info(f"oot_files: {oot_files}")
    date_str = Path(str(oot_files[0])).stem.replace("_mbo_events", "")
    log.info(f"Grading date: {date_str}")

    log.info(f"Loading fill_prob model from {FILL_PROB_MODEL}")
    fill_model = lgb.Booster(model_file=str(FILL_PROB_MODEL))
    log.info(f"Model features: {fill_model.feature_name()}")

    all_results = []
    for horizon, pct in SURVIVING_CELLS:
        h_idx = horizons.index(horizon) if horizon in horizons else (0 if horizon == "1s" else 1)
        result = grade_one_cell(preds, date_str, horizon, pct, h_idx, fill_model)
        if result is not None:
            all_results.append(result)

    # CSV summary
    rows = []
    for r in all_results:
        u = r["ungated"]
        rows.append({
            "horizon": r["horizon"], "top_pct": r["top_pct"], "gate": "ungated",
            "n": u["n"], "net_ticks": u["net_ticks"], "net_per_trade": u["net_per_trade"],
            "wr": u["wr"], "pf": u["pf"], "uplift_per_trade": 0.0,
        })
        for thr in GATE_THRESHOLDS:
            g = r["gated"][thr]
            rows.append({
                "horizon": r["horizon"], "top_pct": r["top_pct"], "gate": f"p>={thr}",
                "n": g["n"], "net_ticks": g["net_ticks"], "net_per_trade": g["net_per_trade"],
                "wr": g["wr"], "pf": g["pf"], "uplift_per_trade": g["uplift_per_trade"],
            })
        for q in CALIBRATED_QUANTILES:
            gc = r["gated_cal"][q]
            rows.append({
                "horizon": r["horizon"], "top_pct": r["top_pct"], "gate": f"q>={q} (thr={gc['threshold_value']:.4f})",
                "n": gc["n"], "net_ticks": gc["net_ticks"], "net_per_trade": gc["net_per_trade"],
                "wr": gc["wr"], "pf": gc["pf"], "uplift_per_trade": gc["uplift_per_trade"],
            })
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "summary.csv", index=False)

    # Determine best gate by uplift across surviving cells
    best_thr = None
    best_uplift = -1e9
    best_kind = None
    for thr in GATE_THRESHOLDS:
        uplifts = [r["gated"][thr]["uplift_per_trade"] for r in all_results if r["gated"][thr]["n"] > 0]
        if not uplifts:
            log.info(f"Threshold p>={thr}: NO TRADES (gate rejects all -- p_fill range is narrow)")
            continue
        avg_uplift = float(np.mean(uplifts))
        log.info(f"Threshold p>={thr}: avg uplift = {avg_uplift:+.4f} t/trade, cells active={len(uplifts)}")
        if avg_uplift > best_uplift:
            best_uplift = avg_uplift
            best_thr = thr
            best_kind = "absolute"

    for q in CALIBRATED_QUANTILES:
        uplifts = [r["gated_cal"][q]["uplift_per_trade"] for r in all_results if r["gated_cal"][q]["n"] > 0]
        if not uplifts:
            continue
        avg_uplift = float(np.mean(uplifts))
        log.info(f"Quantile q>={q}: avg uplift = {avg_uplift:+.4f} t/trade")
        if avg_uplift > best_uplift:
            best_uplift = avg_uplift
            best_thr = q
            best_kind = "quantile"

    log.info(f"\nBEST GATE: {best_kind}={best_thr} (avg uplift {best_uplift:+.4f} t/trade)")

    # Save analysis as JSON
    import json
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump({"results": all_results, "best_threshold": best_thr,
                   "best_uplift": best_uplift, "best_kind": best_kind}, f, indent=2, default=str)

    # Print summary table
    print("\n" + "=" * 100)
    print("FILL-PROB GATED FIFO RESULTS (fold-0, 20260412)")
    print("=" * 100)
    print(f"{'Cell':<15}{'Gate':<10}{'N':>6}{'NetT':>10}{'/Trade':>10}{'WR':>8}{'PF':>8}{'Uplift':>10}")
    for r in all_results:
        cell = f"{r['horizon']}_top{r['top_pct']}%"
        u = r["ungated"]
        print(f"{cell:<15}{'ungated':<10}{u['n']:>6}{u['net_ticks']:>10.2f}{u['net_per_trade']:>10.4f}{u['wr']:>8.3f}{u['pf']:>8.2f}{'-':>10}")
        for thr in GATE_THRESHOLDS:
            g = r["gated"][thr]
            print(f"{'':<15}{'p>='+str(thr):<10}{g['n']:>6}{g['net_ticks']:>10.2f}{g['net_per_trade']:>10.4f}{g['wr']:>8.3f}{g['pf']:>8.2f}{g['uplift_per_trade']:>+10.4f}")
        for q in CALIBRATED_QUANTILES:
            gc = r["gated_cal"][q]
            print(f"{'':<15}{'q>='+str(q):<10}{gc['n']:>6}{gc['net_ticks']:>10.2f}{gc['net_per_trade']:>10.4f}{gc['wr']:>8.3f}{gc['pf']:>8.2f}{gc['uplift_per_trade']:>+10.4f}")
        print()

    print(f"\nBest gate: {best_kind}={best_thr} (avg uplift {best_uplift:+.4f} t/trade across surviving cells)")
    print(f"Outputs: {OUT_DIR}")
    log.info(f"Done in {(time.time()-t0):.0f}s")


if __name__ == "__main__":
    main()
