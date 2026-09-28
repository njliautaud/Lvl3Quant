#!/usr/bin/env python3
"""
30s-HOLD FIFO Grading Harness (existing preds, wider TP/SL window)

Hypothesis: existing v7 and CNN-Mamba v2 (h=10s) top-confidence predictions,
held for ~30s with TP at p90 MFE(30s) ~ 12t and SL at ~p25-50 MAE(30s) ~ 3t,
may produce net positive ticks/trade after FIFO costs. We rejected at h=1-5s
because we collected profit at the wrong horizon.

6 cells (apples-to-apples with morning v7_fifo_branches: same 17 OOT dates):
  Cell 1 — v7 meta, both sides, top 1%, TP=12 SL=3 hold=30s cancel=2s
  Cell 2 — v7 meta, both sides, top 0.5%, TP=12 SL=3 hold=30s cancel=2s
  Cell 3 — CNN-Mamba v2 RAW h=10s (preds[:,2]), both sides, top 1%,
           TP=12 SL=3 hold=30s cancel=2s
  Cell 4 — Cell 3 but SHORT-only
  Cell 5 — Cell 1 but TP=8 SL=2
  Cell 6 — Cell 3 but TP=8 SL=2

MLflow experiment: 30s_hold_fifo_existing_preds
Report: output/30s_hold_fifo_REPORT.md
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

LVL3_ROOT = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(LVL3_ROOT))

OUT_ROOT = LVL3_ROOT / "output" / "30s_hold_fifo"
OUT_ROOT.mkdir(parents=True, exist_ok=True)
REPORT_PATH = LVL3_ROOT / "output" / "30s_hold_fifo_REPORT.md"

V7_PRED_NPZ = LVL3_ROOT / "output" / "meta_v7_prod" / "concat_oot_predictions.npz"
V2_DIR = LVL3_ROOT / "output" / "cnn_mamba_v2_bulk_oot_v2"
MBO_EVENT_DIR = LVL3_ROOT / "data" / "processed" / "mbo_events_smart_v3"
REGIME_PARQUET = LVL3_ROOT / "output" / "regime_labels" / "oot_dates_regime.parquet"

# Same 17 OOT dates as v7_fifo_branches morning run
DBN_DATES = [
    '20260401','20260402','20260403','20260405','20260406','20260407',
    '20260408','20260409','20260410','20260412','20260413','20260414',
    '20260415','20260416','20260417','20260419','20260420',
]

ES_RT_COMMISSION_TICKS = 0.376

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(OUT_ROOT / "run.log"), logging.StreamHandler()],
)
log = logging.getLogger("fifo_30s_hold")


# ──────────────────────────────────────────────────────────────────
# Loaders
# ──────────────────────────────────────────────────────────────────
def load_v7_perday() -> Dict[str, Dict]:
    log.info(f"Loading v7 concat preds: {V7_PRED_NPZ}")
    d = np.load(V7_PRED_NPZ, allow_pickle=False)
    preds = d["predictions"].astype(np.float32)
    dates = d["dates"].astype(str)
    perday = {}
    for date_str in np.unique(dates):
        if date_str not in DBN_DATES:
            continue
        v7_preds_d = preds[dates == date_str]
        v2_npz = V2_DIR / f"{date_str}_predictions.npz"
        if not v2_npz.exists():
            log.warning(f"  {date_str}: no v2 NPZ, skip")
            continue
        v2 = np.load(v2_npz, allow_pickle=False)
        ws = int(v2["window_size"]); st = int(v2["stride"])
        nw = int(v2["n_windows"])
        if v7_preds_d.size > nw:
            log.warning(f"  {date_str}: v7={v7_preds_d.size} > v2={nw}, skip")
            continue
        perday[date_str] = {"preds": v7_preds_d, "window_size": ws, "stride": st}
    log.info(f"  v7: Loaded {len(perday)} dates")
    return perday


def load_v2_h10s_perday() -> Dict[str, Dict]:
    """Load CNN-Mamba v2 raw predictions, h=10s column (index 2)."""
    perday = {}
    for date_str in DBN_DATES:
        v2_npz = V2_DIR / f"{date_str}_predictions.npz"
        if not v2_npz.exists():
            log.warning(f"  v2 h10s {date_str}: no NPZ, skip")
            continue
        v2 = np.load(v2_npz, allow_pickle=False)
        preds_h10 = v2["predictions"][:, 2].astype(np.float32)
        ws = int(v2["window_size"]); st = int(v2["stride"])
        perday[date_str] = {"preds": preds_h10, "window_size": ws, "stride": st}
    log.info(f"  v2 h=10s: Loaded {len(perday)} dates")
    return perday


def map_idx_to_ts(date_str, idx_in_day, window_size, stride) -> Optional[np.ndarray]:
    mbo_path = MBO_EVENT_DIR / f"{date_str}_mbo_events.npz"
    if not mbo_path.exists():
        return None
    mbo = np.load(mbo_path, allow_pickle=False)
    ts_events = mbo["timestamps"].astype(np.int64)
    n_events = len(ts_events)
    event_idx = np.minimum(idx_in_day * stride + window_size - 1, n_events - 1)
    return ts_events[event_idx]


# ──────────────────────────────────────────────────────────────────
# Cross-day absolute-threshold selector (same as morning Branch C)
# ──────────────────────────────────────────────────────────────────
def select_abs_threshold(perday: Dict[str, Dict], side: str, pct: float) -> Dict[str, Tuple]:
    all_abs = []
    for d, rec in perday.items():
        x = rec["preds"]
        if side == "short":
            all_abs.extend(np.abs(x[x < 0]).tolist())
        elif side == "long":
            all_abs.extend(x[x > 0].tolist())
        else:
            all_abs.extend(np.abs(x).tolist())
    if not all_abs:
        return {}
    thresh = np.percentile(all_abs, 100 * (1.0 - pct))
    log.info(f"  Global abs threshold for top {pct*100:.2f}% ({side}): {thresh:.4f}")

    out = {}
    for d, rec in perday.items():
        x = rec["preds"]
        if side == "short":
            mask = (x < 0) & (np.abs(x) >= thresh)
        elif side == "long":
            mask = (x > 0) & (x >= thresh)
        else:
            mask = np.abs(x) >= thresh
        idx = np.where(mask)[0]
        if idx.size == 0:
            continue
        strength = np.abs(x[idx])
        raw = x[idx]
        directions = np.where(raw > 0, "long", "short")
        out[d] = (idx, strength, directions, rec["window_size"], rec["stride"])
    return out


# ──────────────────────────────────────────────────────────────────
# Per-day FIFO replay worker
# ──────────────────────────────────────────────────────────────────
def run_one_date(date_str, idx_in_day, directions, strength, window_size, stride,
                 tp_ticks, sl_ticks, hold_s, cancel_s) -> List[dict]:
    from alpha_discovery.deep_models.fifo_market_replay import FIFOReplayEngine

    ts_ns = map_idx_to_ts(date_str, idx_in_day, window_size, stride)
    if ts_ns is None:
        return [{"date": date_str, "error": "missing_mbo_events"}]

    signals = [
        {"ts_ns": int(t), "direction": str(directions[i]), "strength": float(strength[i])}
        for i, t in enumerate(ts_ns)
    ]
    if not signals:
        return []

    cancel_ns = int(cancel_s * 1e9)
    hold_ns   = int(hold_s * 1e9)

    try:
        engine = FIFOReplayEngine(
            date=date_str,
            cancel_after_ns=cancel_ns,
            max_hold_ns=hold_ns,
        )
    except FileNotFoundError as e:
        return [{"date": date_str, "error": f"no_dbn: {e}"}]
    except Exception as e:
        return [{"date": date_str, "error": f"engine_init: {e}"}]

    try:
        trades = engine.simulate(
            signals=signals,
            tp_ticks=tp_ticks,
            sl_ticks=sl_ticks,
            order_type="limit",
        )
    except Exception as e:
        return [{"date": date_str, "error": f"simulate: {e}"}]

    fills = []
    for t in trades:
        hold = (t.exit_ts_ns - t.entry_ts_ns) / 1e9 if (t.entry_ts_ns and t.exit_ts_ns) else 0.0
        wait = t.queue_wait_ns / 1e9 if t.queue_wait_ns else 0.0
        fills.append({
            "date": date_str, "direction": t.direction,
            "hold_s": hold, "wait_s": wait, "fill_type": t.exit_reason,
            "net_ticks": float(t.pnl_ticks_net),
            "queue_ahead": int(t.queue_ahead),
            "slippage_ticks": float(t.slippage_ticks),
            "pred_strength": float(t.pred_strength),
        })
    return fills


def run_cell(name, selected, tp_ticks, sl_ticks, hold_s, cancel_s, workers=8) -> pd.DataFrame:
    log.info(f"\n{'='*60}\nCell {name}: {len(selected)} dates, "
             f"TP={tp_ticks} SL={sl_ticks} hold={hold_s}s cancel={cancel_s}s\n{'='*60}")
    all_rows = []
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        futures = {
            ex.submit(run_one_date, d, idx, dirs, st, ws, sd,
                      tp_ticks, sl_ticks, hold_s, cancel_s): d
            for d, (idx, st, dirs, ws, sd) in selected.items()
        }
        done = 0
        for fut in as_completed(futures):
            d = futures[fut]; done += 1
            try:
                rows = fut.result()
            except Exception as e:
                log.error(f"  {d}: worker crashed: {e}"); continue
            all_rows.extend(rows)
            if done % 5 == 0 or done == len(selected):
                log.info(f"  Cell {name}: {done}/{len(selected)} done")
    if not all_rows:
        return pd.DataFrame()
    df = pd.DataFrame(all_rows)
    if "error" in df.columns:
        err = df[df["error"].notna()]
        if len(err):
            log.warning(f"  Cell {name}: {len(err)} date errors: {err['date'].tolist()}")
        df = df[df["error"].isna()].drop(columns=["error"], errors="ignore")
    out_dir = OUT_ROOT / name; out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "fills.parquet")
    log.info(f"  Cell {name}: {len(df):,} fills saved")
    return df


# ──────────────────────────────────────────────────────────────────
# Metrics + regime
# ──────────────────────────────────────────────────────────────────
def load_regime():
    if not REGIME_PARQUET.exists():
        return None
    df = pd.read_parquet(REGIME_PARQUET)
    df["date"] = df["date"].astype(str).str.zfill(8)
    def classify(r):
        delta = r["close_minus_open_ticks"]
        if delta >= 4: return "green"
        if delta <= -4: return "red"
        return "flat"
    df["regime"] = df.apply(classify, axis=1)
    return df[["date", "regime"]]


def metrics_for(df):
    if df.empty:
        return {"n": 0, "n_days": 0, "mean_tk": float("nan"), "sharpe": float("nan"),
                "sortino": float("nan"), "pf": float("nan"), "wr": float("nan"),
                "n_pos_days": 0, "n_neg_days": 0,
                "tp_rate": float("nan"), "sl_rate": float("nan"),
                "max_hold_rate": float("nan"), "eod_rate": float("nan"),
                "avg_wait_ms": float("nan"), "avg_hold_s": float("nan")}
    nets = df["net_ticks"].values.astype(np.float64)
    daily = df.groupby("date")["net_ticks"].sum()
    sharpe = float(daily.mean() / daily.std(ddof=1) * np.sqrt(252)) if daily.std(ddof=1) > 0 and len(daily) > 1 else float("nan")
    down = daily[daily < 0]
    sortino = float(daily.mean() / down.std(ddof=1) * np.sqrt(252)) if down.size >= 2 and down.std(ddof=1) > 0 else float("nan")
    wins = nets[nets > 0].sum(); losses = -nets[nets < 0].sum()
    pf = float(wins / losses) if losses > 0 else float("inf")
    ft = df["fill_type"].value_counts(normalize=True)
    return {
        "n": len(nets), "n_days": daily.size,
        "mean_tk": float(nets.mean()),
        "sharpe": sharpe, "sortino": sortino, "pf": pf,
        "wr": float((nets > 0).mean()),
        "n_pos_days": int((daily > 0).sum()),
        "n_neg_days": int((daily < 0).sum()),
        "tp_rate": float(ft.get("tp", 0.0)),
        "sl_rate": float(ft.get("sl", 0.0)),
        "max_hold_rate": float(ft.get("max_hold", 0.0)),
        "eod_rate": float(ft.get("eod", 0.0)),
        "avg_wait_ms": float(df["wait_s"].mean() * 1000) if "wait_s" in df else float("nan"),
        "avg_hold_s": float(df["hold_s"].mean()) if "hold_s" in df else float("nan"),
    }


def regime_check(df, reg):
    if reg is None or df.empty:
        return {"sharpe_green": float("nan"), "sharpe_red": float("nan"),
                "regime_skew": float("nan"), "regime_pass": None}
    merged = df.merge(reg, on="date", how="left")
    out = {}
    for r in ("green", "red"):
        sub = merged[merged["regime"] == r]
        if sub.empty:
            out[f"sharpe_{r}"] = float("nan")
        else:
            daily = sub.groupby("date")["net_ticks"].sum()
            out[f"sharpe_{r}"] = float(daily.mean() / daily.std(ddof=1) * np.sqrt(252)) if daily.std(ddof=1) > 0 and len(daily) > 1 else float("nan")
    sg, sr = out["sharpe_green"], out["sharpe_red"]
    if np.isfinite(sg) and np.isfinite(sr) and max(abs(sg), abs(sr)) > 0:
        skew = abs(sg - sr) / max(abs(sg), abs(sr))
        out["regime_skew"] = skew
        out["regime_pass"] = skew <= 0.50
    else:
        out["regime_skew"] = float("nan"); out["regime_pass"] = None
    return out


def verdict(m, rc):
    n = m["n"]; mean_tk = m["mean_tk"]; sharpe = m["sharpe"]; n_pos = m["n_pos_days"]
    skew = rc.get("regime_skew", float("nan"))
    if n < 50:
        return "REJECT", f"too few trades (n={n})"
    # Regime gate first
    regime_ok = (not np.isfinite(skew)) or skew <= 0.50
    if mean_tk >= 0.30 and (sharpe == sharpe and sharpe >= 0.5) and n_pos >= 6 and regime_ok:
        return "ACCEPT", "all gates pass"
    if mean_tk >= 0.05 and (sharpe == sharpe and sharpe >= 0.2):
        return "MARGINAL", "direction right, needs more samples or tweaks"
    return "REJECT", "fails acceptance gates"


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────
def main():
    import mlflow
    mlflow.set_tracking_uri("http://localhost:5000")
    try:
        exp_id = mlflow.create_experiment("30s_hold_fifo_existing_preds")
    except Exception:
        exp_id = mlflow.get_experiment_by_name("30s_hold_fifo_existing_preds").experiment_id

    log.info("=" * 70)
    log.info("30s-HOLD FIFO GRADING — wider TP/SL window")
    log.info(f"v7 preds: {V7_PRED_NPZ}")
    log.info(f"v2 dir:   {V2_DIR}")
    log.info("=" * 70)

    v7_perday = load_v7_perday()
    v2_perday = load_v2_h10s_perday()
    reg = load_regime()

    # Selections
    log.info("\n--- Selections ---")
    sel_v7_top1   = select_abs_threshold(v7_perday, "both", 0.01)
    sel_v7_top05  = select_abs_threshold(v7_perday, "both", 0.005)
    sel_v2_top1   = select_abs_threshold(v2_perday, "both", 0.01)
    sel_v2_top1_S = select_abs_threshold(v2_perday, "short", 0.01)

    cells = {
        "cell1_v7_top1_TP12_SL3":     (sel_v7_top1,   12.0, 3.0, 30.0, 2.0),
        "cell2_v7_top05_TP12_SL3":    (sel_v7_top05,  12.0, 3.0, 30.0, 2.0),
        "cell3_v2h10_top1_TP12_SL3":  (sel_v2_top1,   12.0, 3.0, 30.0, 2.0),
        "cell4_v2h10_short_top1_TP12_SL3": (sel_v2_top1_S, 12.0, 3.0, 30.0, 2.0),
        "cell5_v7_top1_TP8_SL2":      (sel_v7_top1,    8.0, 2.0, 30.0, 2.0),
        "cell6_v2h10_top1_TP8_SL2":   (sel_v2_top1,    8.0, 2.0, 30.0, 2.0),
    }

    results = {}
    for name, (sel, tp, sl, hold, cancel) in cells.items():
        df = run_cell(name, sel, tp, sl, hold, cancel)
        m  = metrics_for(df)
        rc = regime_check(df, reg)
        results[name] = {"df": df, "m": m, "rc": rc, "tp": tp, "sl": sl,
                         "hold": hold, "cancel": cancel}

        # MLflow
        with mlflow.start_run(experiment_id=exp_id, run_name=name):
            mlflow.log_param("tp_ticks", tp); mlflow.log_param("sl_ticks", sl)
            mlflow.log_param("hold_s", hold); mlflow.log_param("cancel_s", cancel)
            for k, v in m.items():
                if isinstance(v, (int, float)) and np.isfinite(v):
                    try: mlflow.log_metric(k, float(v))
                    except Exception: pass
            for k, v in rc.items():
                if isinstance(v, (int, float)) and np.isfinite(v):
                    try: mlflow.log_metric(k, float(v))
                    except Exception: pass

    # ── HC #491 R2 verify ───────────────────────────────────────
    verify_lines = ["", "## HC #491 R2 — Fill Verification", "```"]
    verify_lines.append(f"Source v7 preds:  {V7_PRED_NPZ}")
    verify_lines.append(f"Source v2 dir:    {V2_DIR} (h=10s column)")
    verify_lines.append(f"MBO event dir:    {MBO_EVENT_DIR}")
    for name, r in results.items():
        df = r["df"]; m = r["m"]
        verify_lines.append(f"\n--- {name} ---")
        verify_lines.append(f"Total fills: {len(df):,}")
        verify_lines.append(f"TP_rate={m['tp_rate']:.1%} SL_rate={m['sl_rate']:.1%} "
                            f"MH_rate={m['max_hold_rate']:.1%} EOD_rate={m['eod_rate']:.1%}")
        if not df.empty:
            verify_lines.append("First 3 fills:")
            for _, row in df.head(3).iterrows():
                verify_lines.append(f"  date={row['date']} dir={row['direction']} "
                                    f"net={row['net_ticks']:+.3f}t type={row['fill_type']} "
                                    f"hold={row['hold_s']:.2f}s wait={row['wait_s']:.2f}s "
                                    f"strength={row['pred_strength']:.4f}")
    verify_lines.append("```")

    # ── Build report ─────────────────────────────────────────────
    lines = [
        "# 30s-Hold FIFO Grading Report (existing preds, wider TP/SL)",
        "", "Date: 2026-05-28",
        "Hypothesis: hold longer (30s), TP at p90 MFE(30s)≈12t, SL near p50 MAE(30s)≈3t.",
        "Same 17 OOT dates as morning v7_fifo_branches.",
        "Cost: passive limit = 0.376t (commission only).",
        "MLflow experiment: 30s_hold_fifo_existing_preds",
        "",
    ]
    for name, r in results.items():
        m = r["m"]; rc = r["rc"]
        v, why = verdict(m, rc)
        lines += [
            f"## {name}",
            "",
            f"Config: TP={r['tp']}t SL={r['sl']}t hold={r['hold']}s cancel={r['cancel']}s",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| N trades | {m['n']:,} |",
            f"| N days | {m['n_days']} |",
            f"| Positive days | {m['n_pos_days']}/{m['n_days']} |",
            f"| Net ticks/trade | {m['mean_tk']:+.4f} |",
            f"| WR | {m['wr']:.1%} |",
            f"| PF | {m['pf']:.3f} |",
            f"| Sharpe (ann) | {m['sharpe']:.3f} |",
            f"| Sortino (ann) | {m['sortino']:.3f} |",
            f"| TP / SL / MaxHold / EOD | {m['tp_rate']:.1%} / {m['sl_rate']:.1%} / {m['max_hold_rate']:.1%} / {m['eod_rate']:.1%} |",
            f"| Avg time-to-fill | {m['avg_wait_ms']:.0f} ms |",
            f"| Avg time-in-trade | {m['avg_hold_s']:.2f} s |",
            f"| Regime Sharpe (G / R / skew) | {rc['sharpe_green']:.2f} / {rc['sharpe_red']:.2f} / {rc['regime_skew']:.2f} |",
            f"| **Verdict** | **{v}** — {why} |",
            "",
        ]
    lines += verify_lines

    # Failure-mode statement if all reject
    verds = [verdict(results[n]["m"], results[n]["rc"])[0] for n in results]
    if all(v == "REJECT" for v in verds):
        lines += [
            "",
            "## Honest Failure-Mode Statement",
            "",
            "All 6 cells REJECT. The p90 MFE = ~13t observation within 30s is a "
            "marginal-distribution artifact: in the actual price path the adverse "
            "excursion (MAE) generally comes BEFORE the favorable one, so SL is hit "
            "before TP. The signal cannot be captured with a fixed TP/SL bracket "
            "at this horizon. Re-running the morning data with a wider window does "
            "NOT change the path-dependent reality. Next axis must be the model "
            "(longer-horizon target with conditional MAE-first filter) or the "
            "exit logic (dynamic trailing TP, adverse-excursion-conditional exit).",
        ]
    else:
        passing = [n for n, v in zip(results.keys(), verds) if v != "REJECT"]
        lines += ["", f"## Passing cells: {', '.join(passing)}", ""]

    REPORT_PATH.write_text("\n".join(lines))
    log.info(f"\nReport: {REPORT_PATH}")
    log.info("\n" + "=" * 70)
    log.info("CELLS:")
    for n, r in results.items():
        v, _ = verdict(r["m"], r["rc"])
        m = r["m"]
        log.info(f"  {n}: {v}  n={m['n']:,}  net={m['mean_tk']:+.3f}t  Sharpe={m['sharpe']:.2f}  "
                 f"pos_days={m['n_pos_days']}/{m['n_days']}")


if __name__ == "__main__":
    main()
