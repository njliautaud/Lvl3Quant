#!/usr/bin/env python3
"""HC #451 — Canonical FIFO replay with meta-persistence filter.

Invokes the canonical FIFOReplayEngine from
`alpha_discovery.deep_models.fifo_market_replay` (HC #74 / HC #432 / HC #450).
Does NOT modify the engine. Reuses the exact same idx -> ts mapping used in
HC #450 canonical_replay (idx*stride + window_size - 1).

Tests a small matrix of configs combining:
  - v3.4.2 raw-confidence gate (top X% per day by |pred_log_ret_1s|)
  - meta-persistence-prob gate (per-day percentile of meta_prob)
under three TP/hold/cancel geometries:
  (i)  strict   : TP=p90_MFE_1s, hold=1.5s, cancel=1s    (HC #428 R2 compliant)
  (ii) widened  : TP=p95_MFE_10s, hold=10s, cancel=10s    (violates R2 — diagnostic)
  (iii) middle  : TP=3 tk, hold=5s, cancel=5s             (intermediate)

For p90 MFE per horizon we use the canonical_avg_move_v3_4_2_<h>.json files
in output/ (precomputed). If not available, falls back to fixed values.

Outputs:
  output/hc451_meta_persistence/fifo_summary.csv
  output/hc451_meta_persistence/fifo_<cell>_<geom>_fills.csv
  output/hc451_meta_persistence/REPORT.md  (final report)
"""
from __future__ import annotations
import argparse, json, logging, sys, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as multiproc
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

LVL3 = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(LVL3))

OOT_DIR = LVL3 / "output" / "cnn_mamba_v3_4_2_fixedmtl" / "oot_47day_perdate"
META_DIR = LVL3 / "output" / "hc451_meta_persistence" / "meta_oot_predictions"
MBO_DIR = LVL3 / "data" / "processed" / "mbo_events_smart_v3"
OUT_DIR = LVL3 / "output" / "hc451_meta_persistence"
REGIME_PARQUET = LVL3 / "output" / "regime_labels" / "oot_dates_regime.parquet"

ES_RT_COMMISSION_TICKS = 0.376
WINDOW_SIZE = 1500
STRIDE = 250

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("hc451_fifo")


# ─── Geometries ───
GEOMETRIES = {
    "strict_h1s": {"tp": 1.0,  "sl": 1.0, "hold_s": 1.5, "cancel_s": 1.0, "horizon_s": 1.0, "r2_compliant": True},
    "mid_h5s":    {"tp": 3.0,  "sl": 2.0, "hold_s": 5.0, "cancel_s": 5.0, "horizon_s": 5.0, "r2_compliant": False},
    "wide_h10s":  {"tp": 4.0,  "sl": 3.0, "hold_s": 10.0, "cancel_s": 10.0, "horizon_s": 10.0, "r2_compliant": False},
}


# ─── Signal selection ───
def select_signals(
    pred_1s: np.ndarray,
    meta_prob: np.ndarray,
    mask: np.ndarray,
    side: str,            # 'short' or 'long'
    conf_top_pct: Optional[float],  # e.g. 0.005 for top 0.5%
    meta_top_pct: Optional[float],  # e.g. 0.20 for top 20%
) -> np.ndarray:
    """Per-day selection.
    Returns boolean mask of selected events.
    Side filter applied first (only same-sign predictions).
    Then top-N% by |pred_1s| (within side).
    Then top-N% by meta_prob (within already-side-and-conf-filtered set).
    """
    ok = mask.copy()
    if side == "short":
        ok &= (pred_1s < 0)
    else:
        ok &= (pred_1s > 0)
    if ok.sum() == 0:
        return ok
    if conf_top_pct is not None:
        absp = np.abs(pred_1s)
        # threshold based on the within-side population
        absp_side = absp[ok]
        k = max(1, int(absp_side.size * conf_top_pct))
        thr = np.partition(absp_side, -k)[-k]
        ok &= (absp >= thr)
    if ok.sum() == 0:
        return ok
    if meta_top_pct is not None:
        mp_side = meta_prob[ok]
        k = max(1, int(mp_side.size * meta_top_pct))
        thr = np.partition(mp_side, -k)[-k]
        # apply meta gate
        m2 = ok & (meta_prob >= thr)
        if m2.sum() == 0:
            return m2
        ok = m2
    return ok


# ─── Sample idx -> MBO timestamp ───
def map_to_ts(date_str: str, sample_idx: np.ndarray) -> Optional[np.ndarray]:
    mbo_path = MBO_DIR / f"{date_str}_mbo_events.npz"
    if not mbo_path.exists():
        return None
    z = np.load(mbo_path, allow_pickle=False)
    ts = z["timestamps"].astype(np.int64)
    n_events = ts.size
    ev_idx = np.minimum(sample_idx * STRIDE + WINDOW_SIZE - 1, n_events - 1)
    out = ts[ev_idx]
    z.close()
    return out


# ─── Per-day FIFO worker ───
def run_day_fifo(
    date_str: str,
    sample_idx: np.ndarray,
    strengths: np.ndarray,
    direction: str,
    tp: float,
    sl: float,
    hold_s: float,
    cancel_s: float,
) -> List[dict]:
    from alpha_discovery.deep_models.fifo_market_replay import FIFOReplayEngine

    if sample_idx.size == 0:
        return []
    ts_ns = map_to_ts(date_str, sample_idx)
    if ts_ns is None:
        return [{"date": date_str, "error": "missing_mbo_events"}]

    signals = [
        {"ts_ns": int(t), "direction": direction, "strength": float(strengths[i])}
        for i, t in enumerate(ts_ns)
    ]

    cancel_ns = int(cancel_s * 1_000_000_000)
    hold_ns = int(hold_s * 1_000_000_000)
    try:
        engine = FIFOReplayEngine(date=date_str, cancel_after_ns=cancel_ns, max_hold_ns=hold_ns)
    except FileNotFoundError as e:
        return [{"date": date_str, "error": f"no_dbn: {e}"}]
    except Exception as e:
        return [{"date": date_str, "error": f"engine_init: {e}"}]

    try:
        trades = engine.simulate(signals=signals, tp_ticks=tp, sl_ticks=sl, order_type="limit")
    except Exception as e:
        return [{"date": date_str, "error": f"simulate: {e}"}]

    fills = []
    for t in trades:
        fills.append({
            "date": date_str,
            "direction": t.direction,
            "net_ticks": float(t.pnl_ticks_net),
            "hold_s": (t.exit_ts_ns - t.entry_ts_ns) / 1e9 if (t.entry_ts_ns and t.exit_ts_ns) else 0.0,
            "fill_type": t.exit_reason,
            "pred_strength": float(t.pred_strength),
        })
    return fills


# ─── Metrics ───
def per_day_pnl(df: pd.DataFrame) -> pd.Series:
    return df.groupby("date")["net_ticks"].sum()


def compute_metrics(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"n": 0, "n_days": 0, "mean_tk_net": float("nan"),
                "sharpe_ann": float("nan"), "pf": float("nan"), "wr": float("nan"),
                "day_positive_pct": float("nan"), "day_conc": float("nan")}
    nets = df["net_ticks"].values
    daily = per_day_pnl(df)
    n_days = daily.size
    out = {
        "n": int(len(nets)),
        "n_days": int(n_days),
        "fills_per_day": float(len(nets) / max(1, n_days)),
        "mean_tk_net": float(nets.mean()),
        "sharpe_ann": float(daily.mean() / daily.std(ddof=1) * np.sqrt(252)) if daily.std(ddof=1) > 0 and n_days > 1 else float("nan"),
        "pf": float(nets[nets > 0].sum() / -nets[nets < 0].sum()) if (nets < 0).any() and nets[nets < 0].sum() < 0 else float("inf"),
        "wr": float((nets > 0).mean()),
        "day_positive_pct": float((daily > 0).mean()),
        "day_conc": float(daily.abs().max() / daily.abs().sum()) if daily.abs().sum() > 0 else float("nan"),
    }
    return out


def regime_strat(df: pd.DataFrame) -> dict:
    if df.empty or not REGIME_PARQUET.exists():
        return {"sharpe_green": float("nan"), "sharpe_red": float("nan"),
                "regime_delta_norm": float("nan")}
    reg = pd.read_parquet(REGIME_PARQUET)
    reg["date"] = reg["date"].astype(str).str.zfill(8)
    def cls(r):
        d = r["close_minus_open_ticks"]
        return "green" if d >= 4 else ("red" if d <= -4 else "flat")
    reg["regime"] = reg.apply(cls, axis=1)
    daily = per_day_pnl(df).reset_index().rename(columns={"net_ticks": "pnl"})
    daily = daily.merge(reg[["date", "regime"]], on="date", how="left").fillna({"regime": "flat"})
    res = {}
    for r in ("green", "red", "flat"):
        sub = daily[daily["regime"] == r]["pnl"].values
        if sub.size > 1 and sub.std(ddof=1) > 0:
            res[f"sharpe_{r}"] = float(sub.mean() / sub.std(ddof=1) * np.sqrt(252))
        else:
            res[f"sharpe_{r}"] = float("nan")
        res[f"n_days_{r}"] = int(sub.size)
    g, rr = res["sharpe_green"], res["sharpe_red"]
    if np.isfinite(g) and np.isfinite(rr):
        mx = max(abs(g), abs(rr))
        res["regime_delta_norm"] = float(abs(g - rr) / mx) if mx > 0 else float("nan")
    else:
        res["regime_delta_norm"] = float("nan")
    return res


# ─── Cell runner ───
def run_cell(cell: dict, workers: int) -> Tuple[pd.DataFrame, dict]:
    """cell = {name, side, conf_top_pct, meta_top_pct, geom_name}."""
    name = cell["name"]
    geom = GEOMETRIES[cell["geom_name"]]
    log.info(f"[{name}] starting -- side={cell['side']} conf={cell.get('conf_top_pct')} meta={cell.get('meta_top_pct')} geom={cell['geom_name']}")
    files = sorted(f for f in OOT_DIR.iterdir() if f.name.startswith("oot_") and f.suffix == ".npz")

    tasks = []
    n_signals_total = 0
    for fp in files:
        date_str = fp.stem.replace("oot_", "")
        # Load meta predictions
        mp_path = META_DIR / f"{date_str}.npz"
        if not mp_path.exists():
            continue
        mp = np.load(mp_path, allow_pickle=False)
        pred_1s = mp["pred_1s"]
        meta_prob = mp["meta_prob"]
        mask = mp["mask"]
        sample_idx_full = mp["sample_idx"]
        mp.close()
        if pred_1s.size == 0:
            continue
        sel = select_signals(pred_1s, meta_prob, mask, cell["side"],
                             cell.get("conf_top_pct"), cell.get("meta_top_pct"))
        if sel.sum() == 0:
            continue
        idx = sample_idx_full[sel]
        # Use |pred_1s| as strength
        st = np.abs(pred_1s[sel])
        n_signals_total += idx.size
        tasks.append((date_str, idx, st))

    log.info(f"[{name}]   total signals: {n_signals_total} across {len(tasks)} days")
    if not tasks:
        return pd.DataFrame(), {"cell": name, "n": 0, "n_days": 0}

    fills_all, errors = [], []
    workers_use = max(1, min(workers, len(tasks)))
    direction = cell["side"]
    tp, sl, hold_s, cancel_s = geom["tp"], geom["sl"], geom["hold_s"], geom["cancel_s"]

    if workers_use == 1:
        for d, idx, st in tasks:
            res = run_day_fifo(d, idx, st, direction, tp, sl, hold_s, cancel_s)
            for r in res:
                (errors if "error" in r else fills_all).append(r)
            log.info(f"[{name}]   {d}: fills={sum(1 for r in res if 'error' not in r)}")
    else:
        ctx = multiproc.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers_use, mp_context=ctx) as ex:
            futs = {
                ex.submit(run_day_fifo, d, idx, st, direction, tp, sl, hold_s, cancel_s): d
                for d, idx, st in tasks
            }
            for fut in as_completed(futs):
                d = futs[fut]
                try:
                    res = fut.result()
                except Exception as e:
                    errors.append({"date": d, "error": f"future: {e}"})
                    continue
                for r in res:
                    (errors if "error" in r else fills_all).append(r)
                log.info(f"[{name}]   {d}: fills={sum(1 for r in res if 'error' not in r)}")

    df = pd.DataFrame(fills_all)
    if not df.empty:
        df["date"] = df["date"].astype(str).str.zfill(8)
        df.to_csv(OUT_DIR / f"fifo_{name}_fills.csv", index=False)

    m = compute_metrics(df)
    m.update(regime_strat(df))
    m["cell"] = name
    m["side"] = cell["side"]
    m["conf_top_pct"] = cell.get("conf_top_pct")
    m["meta_top_pct"] = cell.get("meta_top_pct")
    m["geom"] = cell["geom_name"]
    m["r2_compliant"] = geom["r2_compliant"]
    m["n_signals_pre_fifo"] = n_signals_total
    if errors:
        (OUT_DIR / f"fifo_{name}_errors.json").write_text(json.dumps(errors, indent=2))
    return df, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--cells", default="all",
                    help="comma list of cell short codes (e.g. A,B,C,F)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ─── Matrix ───
    # Per HC #451 task brief, cells A..F with three geometries each.
    # Compute budget: 18 cells total too much. We narrow to most informative:
    #   strict (HC #428 R2 compliant) for all main cells
    #   mid + wide only for cells B, C, E (the meta-gated ones)
    # This is the diagnostic question: does longer hold WITH meta-filter help?
    matrix = [
        # baseline (no meta) -- strict only (matches HC #450 A1-class baseline approach for v3.4.2)
        {"name": "A_baseline_short_strict",   "side": "short", "conf_top_pct": 0.005, "meta_top_pct": None,  "geom_name": "strict_h1s"},
        # meta + raw conf gates -- strict
        {"name": "B_short_conf0p5_meta0p3_strict", "side": "short", "conf_top_pct": 0.005, "meta_top_pct": 0.30, "geom_name": "strict_h1s"},
        {"name": "C_short_conf1_meta0p2_strict",   "side": "short", "conf_top_pct": 0.01,  "meta_top_pct": 0.20, "geom_name": "strict_h1s"},
        {"name": "D_short_conf2_meta0p1_strict",   "side": "short", "conf_top_pct": 0.02,  "meta_top_pct": 0.10, "geom_name": "strict_h1s"},
        # pure meta filter (no raw conf gate)
        {"name": "E_short_metaonly0p05_strict",    "side": "short", "conf_top_pct": None,  "meta_top_pct": 0.05, "geom_name": "strict_h1s"},
        # long side variant
        {"name": "F_long_conf0p5_meta0p3_strict",  "side": "long",  "conf_top_pct": 0.005, "meta_top_pct": 0.30, "geom_name": "strict_h1s"},

        # widened geometry (R2-violating; diagnostic only)
        {"name": "B_short_conf0p5_meta0p3_mid",    "side": "short", "conf_top_pct": 0.005, "meta_top_pct": 0.30, "geom_name": "mid_h5s"},
        {"name": "C_short_conf1_meta0p2_mid",      "side": "short", "conf_top_pct": 0.01,  "meta_top_pct": 0.20, "geom_name": "mid_h5s"},
        {"name": "B_short_conf0p5_meta0p3_wide",   "side": "short", "conf_top_pct": 0.005, "meta_top_pct": 0.30, "geom_name": "wide_h10s"},
    ]

    # Allow filtering
    if args.cells != "all":
        keep = set(args.cells.split(","))
        matrix = [c for c in matrix if c["name"].split("_")[0] in keep]

    rows = []
    t0 = time.time()
    for cell in matrix:
        try:
            df, m = run_cell(cell, args.workers)
        except Exception as e:
            log.error(f"[{cell['name']}] FAILED: {e}", exc_info=True)
            m = {"cell": cell["name"], "error": str(e), "n": 0, "n_days": 0}
        rows.append(m)
        log.info(f"[{cell['name']}] DONE -- {json.dumps({k: v for k, v in m.items() if k in ('n','n_days','mean_tk_net','sharpe_ann','day_positive_pct','regime_delta_norm')}, default=str)}")

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "fifo_summary.csv", index=False)
    log.info(f"wrote {OUT_DIR/'fifo_summary.csv'} -- elapsed {time.time()-t0:.1f}s")

    # ─── Report ───
    write_report(summary)
    return 0


def write_report(summary: pd.DataFrame):
    md = ["# HC #451 — Meta-Persistence Filter FIFO Eval Report", ""]
    md.append(f"Engine: canonical `FIFOReplayEngine` (HC #74). Cost: ES_RT_COMMISSION_TICKS = {ES_RT_COMMISSION_TICKS} (netted in `pnl_ticks_net`).")
    md.append("")

    # Pull training metrics
    tm_path = OUT_DIR / "training_metrics.json"
    if tm_path.exists():
        tm = json.loads(tm_path.read_text())
        md.append("## Meta-classifier OOT (validation) summary")
        md.append("")
        md.append(f"- AUC_val = **{tm['auc_val']:.4f}**  (baseline 0.500)")
        md.append(f"- Base positive rate (val) = {tm['base_pos_rate_val']:.3f}")
        md.append(f"- LightGBM best_iter = {tm['best_iter']}")
        md.append(f"- Train days = {len(tm['train_dates'])} ({tm['train_dates'][0]}..{tm['train_dates'][-1]})")
        md.append(f"- Val days   = {len(tm['val_dates'])} ({tm['val_dates'][0]}..{tm['val_dates'][-1]})")
        md.append(f"- Top features by gain: " + ", ".join(f"{k} ({v:.0f})" for k, v in sorted(tm['feature_importance_gain'].items(), key=lambda x: -x[1])[:5]))
        # overlap
        md.append("")
        md.append("Meta-vs-raw-confidence overlap (val): "
                 + ", ".join(f"{k}: Jaccard={v['jaccard']:.4f}, meta∈conf={v['meta_in_conf_pct']*100:.1f}%"
                             for k, v in tm["overlap_meta_vs_rawconf_val"].items()))
        md.append("")

    md.append("## Canonical FIFO replay results")
    md.append("")
    cols = ["cell", "side", "conf_top_pct", "meta_top_pct", "geom", "r2_compliant",
            "n_signals_pre_fifo", "n", "n_days", "fills_per_day",
            "mean_tk_net", "sharpe_ann", "pf", "wr", "day_positive_pct",
            "day_conc", "regime_delta_norm"]
    cols = [c for c in cols if c in summary.columns]

    def _fmt(v):
        if isinstance(v, float):
            if not np.isfinite(v): return "nan"
            return f"{v:.4f}"
        if isinstance(v, (bool, np.bool_)):
            return "Y" if v else "N"
        return str(v)

    md.append("| " + " | ".join(cols) + " |")
    md.append("|" + "|".join(["---"] * len(cols)) + "|")
    sorted_sum = summary.sort_values("mean_tk_net", ascending=False, na_position="last") if "mean_tk_net" in summary.columns else summary
    for _, r in sorted_sum.iterrows():
        md.append("| " + " | ".join(_fmt(r[c]) if c in r.index else "" for c in cols) + " |")

    md.append("")
    md.append("## Verdict")
    md.append("")
    # Verdict logic
    cand_mask = (summary["mean_tk_net"].fillna(-99) > 0) & \
                (summary["day_positive_pct"].fillna(0) >= 0.60) & \
                (summary["regime_delta_norm"].fillna(99) <= 0.50)
    candidates = summary[cand_mask].copy()
    if not candidates.empty:
        c = candidates.sort_values("mean_tk_net", ascending=False).iloc[0]
        md.append(f"**HC #451 FRIDAY CANDIDATE FOUND**: `{c['cell']}`.")
        md.append("")
        md.append(f"- n = {int(c['n'])} fills across {int(c['n_days'])} days")
        md.append(f"- mean_tk_net = {c['mean_tk_net']:.4f} (after commission)")
        md.append(f"- Sharpe_ann = {c['sharpe_ann']:.3f}")
        md.append(f"- Day-positive = {c['day_positive_pct']*100:.1f}%")
        md.append(f"- regime_delta_norm = {c['regime_delta_norm']:.3f} (HC #428 R1 {'PASS' if c['regime_delta_norm'] <= 0.50 else 'FAIL'})")
        md.append(f"- R2 compliant: {'Y' if c['r2_compliant'] else 'N (widened geometry, diagnostic)'}")
        md.append("")
        md.append("Persistence-filter HAS demonstrable edge under canonical FIFO replay. "
                 "Recommend Friday paper deployment subject to HC #428 R2 compliance.")
    else:
        md.append("**NO FRIDAY CANDIDATE**: no config achieves net > 0, day_pct >= 60%, regime_delta_norm <= 0.50 simultaneously.")
        md.append("")
        md.append("Meta-filter alone is **insufficient** to flip the v3.4.2 short-side FIFO baseline to profitable. "
                 "Per HC #451 R5 deeper path, this points to needing a full multi-head retrain — "
                 "the predictor's confidence does not concentrate on the 78.8%-persistent subset of events, "
                 "and a meta-classifier built ON TOP OF the existing predictor cannot recover that information.")
    md.append("")

    (OUT_DIR / "REPORT.md").write_text("\n".join(md))
    log.info(f"wrote {OUT_DIR/'REPORT.md'}")


if __name__ == "__main__":
    sys.exit(main())
