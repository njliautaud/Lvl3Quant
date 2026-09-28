#!/usr/bin/env python3
"""
HC #491 R1 — v7 FIFO branch evaluation (Branches A, B, C, D)

Branches:
  A — SHORT-ONLY: same as base (top 5%, cancel=1s) but only short signals
  B — TIGHT CANCEL 0.25s: both sides top 5%, cancel=0.25s
  C1 — TOP 1% CONFIDENCE: short-only, top 1% of abs(pred), cancel=1s
  C2 — TOP 2% CONFIDENCE: short-only, top 2% of abs(pred), cancel=1s
  D — COMBINED: short-only + tight cancel 0.25s + top 2% confidence

Base harness cancel_after_ns default: 30s (FIFOReplayEngine init default).
v7 grade script override: 1.0s (CANCEL_S = 1.0).
Branch B override: 0.25s.

MLflow experiment: v7_fifo_branches
Output: /home/jupiter/Lvl3Quant/output/v7_fifo_branches/
Report: /home/jupiter/Lvl3Quant/output/v7_fifo_branches_REPORT.md
"""
from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
import sys
import time as time_mod
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

LVL3_ROOT = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(LVL3_ROOT))

OUT_ROOT = LVL3_ROOT / "output" / "v7_fifo_branches"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

REPORT_PATH = LVL3_ROOT / "output" / "v7_fifo_branches_REPORT.md"

V7_PRED_NPZ = LVL3_ROOT / "output" / "meta_v7_prod" / "concat_oot_predictions.npz"
V2_DIR = LVL3_ROOT / "output" / "cnn_mamba_v2_bulk_oot_v2"
MBO_EVENT_DIR = LVL3_ROOT / "data" / "processed" / "mbo_events_smart_v3"
REGIME_PARQUET = LVL3_ROOT / "output" / "regime_labels" / "oot_dates_regime.parquet"

# HC #428 R2 bounds for h=1s
TP_TICKS = 2.0   # p82 of realized MFE within 1s (p90=2.5, safe)
SL_TICKS = 1.0
HOLD_S   = 1.5   # ≤ 1.5 * h=1s
# Base cancel (v7 grade) = 1.0s, Branch B override = 0.25s

ES_TICK_VALUE = 12.50
ES_RT_COMMISSION_TICKS = 0.376

LOG_PATH = OUT_ROOT / "run.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()],
)
log = logging.getLogger("fifo_v7_branches")

# 17 dates with local DBN tape (from base run)
DBN_DATES = [
    '20260401','20260402','20260403','20260405','20260406','20260407',
    '20260408','20260409','20260410','20260412','20260413','20260414',
    '20260415','20260416','20260417','20260419','20260420',
]


# ──────────────────────────────────────────────────────────────────────
# Load v7 per-day (same logic as fifo_v7_grade.py)
# ──────────────────────────────────────────────────────────────────────
def load_v7_perday() -> Dict[str, Dict]:
    log.info(f"Loading v7 concat preds: {V7_PRED_NPZ}")
    d = np.load(V7_PRED_NPZ, allow_pickle=False)
    preds = d["predictions"].astype(np.float32)
    dates = d["dates"].astype(str)
    log.info(f"  Total preds: {preds.size:,}, unique dates: {len(np.unique(dates))}")

    perday = {}
    for date_str in np.unique(dates):
        if date_str not in DBN_DATES:
            continue
        mask = dates == date_str
        v7_preds_d = preds[mask]
        v2_npz = V2_DIR / f"{date_str}_predictions.npz"
        if not v2_npz.exists():
            log.warning(f"  {date_str}: no v2 NPZ, skip")
            continue
        v2 = np.load(v2_npz, allow_pickle=False)
        ws = int(v2["window_size"])
        st = int(v2["stride"])
        nw = int(v2["n_windows"])
        if v7_preds_d.size > nw:
            log.warning(f"  {date_str}: v7={v7_preds_d.size} > v2 n_windows={nw}, skip")
            continue
        perday[date_str] = {"preds": v7_preds_d, "window_size": ws, "stride": st}
    log.info(f"  Loaded {len(perday)} dates with DBN coverage")
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


# ──────────────────────────────────────────────────────────────────────
# Signal selectors
# ──────────────────────────────────────────────────────────────────────
def select_topk(perday: Dict[str, Dict], side: str, pct: float) -> Dict[str, Tuple]:
    """Select top pct of abs(pred) per day, filtered by side."""
    out = {}
    for d, rec in perday.items():
        x = rec["preds"]
        if side == "long":
            mask = x > 0
            strength = x
        elif side == "short":
            mask = x < 0
            strength = -x
        else:  # both
            mask = np.ones(len(x), dtype=bool)
            strength = np.abs(x)

        side_s = strength[mask]
        if side_s.size == 0:
            continue
        k = max(1, int(side_s.size * pct))
        thresh = np.partition(side_s, -k)[-k]
        selected = mask & (strength >= thresh)
        idx = np.where(selected)[0]
        if idx.size == 0:
            continue
        # Determine direction per signal
        raw = rec["preds"][idx]
        directions = np.where(raw > 0, "long", "short")
        out[d] = (idx, strength[idx], directions, rec["window_size"], rec["stride"])
    return out


def select_abs_threshold(perday: Dict[str, Dict], side: str, pct: float) -> Dict[str, Tuple]:
    """Cross-day absolute threshold: pool all abs(preds) from all 17 days, find top pct percentile."""
    # Compute global threshold across all days
    all_abs = []
    for d, rec in perday.items():
        x = rec["preds"]
        if side == "short":
            all_abs.extend(np.abs(x[x < 0]).tolist())
        elif side == "long":
            all_abs.extend(x[x > 0].tolist())
        else:
            all_abs.extend(np.abs(x).tolist())
    thresh = np.percentile(all_abs, 100 * (1.0 - pct))
    log.info(f"  Global abs threshold for top {pct*100:.1f}%: {thresh:.4f}")

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


# ──────────────────────────────────────────────────────────────────────
# Per-day FIFO replay worker
# ──────────────────────────────────────────────────────────────────────
def run_one_date(
    date_str: str,
    idx_in_day: np.ndarray,
    directions: np.ndarray,
    strength: np.ndarray,
    window_size: int,
    stride: int,
    cancel_s: float,
) -> List[dict]:
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

    cancel_ns = int(cancel_s * 1_000_000_000)
    hold_ns   = int(HOLD_S * 1_000_000_000)

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
            tp_ticks=TP_TICKS,
            sl_ticks=SL_TICKS,
            order_type="limit",
        )
    except Exception as e:
        return [{"date": date_str, "error": f"simulate: {e}"}]

    fills = []
    for t in trades:
        hold_s = (t.exit_ts_ns - t.entry_ts_ns) / 1e9 if (t.entry_ts_ns and t.exit_ts_ns) else 0.0
        fills.append({
            "date": date_str,
            "direction": t.direction,
            "hold_s": hold_s,
            "fill_type": t.exit_reason,
            "net_ticks": float(t.pnl_ticks_net),
            "net_dollars": float(t.pnl_dollars),
            "queue_ahead": int(t.queue_ahead),
            "queue_wait_ns": int(t.queue_wait_ns),
            "slippage_ticks": float(t.slippage_ticks),
            "pred_strength": float(t.pred_strength),
        })
    return fills


def run_branch(
    name: str,
    selected: Dict[str, Tuple],
    cancel_s: float,
    workers: int = 8,
) -> pd.DataFrame:
    log.info(f"\n{'='*60}")
    log.info(f"Running Branch {name}: {len(selected)} dates, cancel={cancel_s}s")
    log.info(f"{'='*60}")

    all_rows = []
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        futures = {
            ex.submit(
                run_one_date,
                d, idx, dirs, strength, ws, st, cancel_s
            ): d
            for d, (idx, strength, dirs, ws, st) in selected.items()
        }
        done = 0
        for fut in as_completed(futures):
            d = futures[fut]
            done += 1
            try:
                rows = fut.result()
            except Exception as e:
                log.error(f"  {d}: worker crashed: {e}")
                continue
            all_rows.extend(rows)
            if done % 5 == 0 or done == len(selected):
                log.info(f"  Branch {name}: {done}/{len(selected)} dates done")

    if not all_rows:
        log.error(f"Branch {name}: NO fills produced!")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    if "error" in df.columns:
        err = df[df["error"].notna()]
        if len(err):
            log.warning(f"  Branch {name}: {len(err)} date errors: {err['date'].tolist()}")
        df = df[df["error"].isna()].drop(columns=["error"], errors="ignore")

    out_dir = OUT_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "fills.parquet")
    log.info(f"  Branch {name}: {len(df):,} fills saved")
    return df


# ──────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────
def load_regime_labels() -> Optional[pd.DataFrame]:
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


def metrics_for(df: pd.DataFrame) -> dict:
    if df.empty or len(df) == 0:
        return {"n": 0, "n_days": 0, "mean_tk_net": float("nan"),
                "sharpe_ann": float("nan"), "sortino_ann": float("nan"),
                "pf": float("nan"), "wr": float("nan"), "day_pos_pct": float("nan"),
                "n_pos_days": 0, "n_neg_days": 0}
    nets = df["net_ticks"].values.astype(np.float64)
    n = len(nets)
    daily = df.groupby("date")["net_ticks"].sum()
    n_days = daily.size
    mean_tk = float(nets.mean())
    sharpe = float(daily.mean() / daily.std(ddof=1) * np.sqrt(252)) if daily.std(ddof=1) > 0 and n_days > 1 else float("nan")
    down = daily[daily < 0]
    sortino = float(daily.mean() / down.std(ddof=1) * np.sqrt(252)) if down.size >= 2 and down.std(ddof=1) > 0 else float("nan")
    wins = nets[nets > 0].sum()
    losses = -nets[nets < 0].sum()
    pf = float(wins / losses) if losses > 0 else float("inf")
    wr = float((nets > 0).mean())
    day_pos = float((daily > 0).mean())
    n_pos_days = int((daily > 0).sum())
    n_neg_days = int((daily < 0).sum())
    return {
        "n": n, "n_days": n_days, "mean_tk_net": mean_tk,
        "sharpe_ann": sharpe, "sortino_ann": sortino,
        "pf": pf, "wr": wr, "day_pos_pct": day_pos,
        "n_pos_days": n_pos_days, "n_neg_days": n_neg_days,
    }


def regime_check(df: pd.DataFrame, reg: Optional[pd.DataFrame]) -> dict:
    """Check |Sharpe_green - Sharpe_red| / max(|Sg|, |Sr|) <= 0.50."""
    if reg is None or df.empty:
        return {"sharpe_green": float("nan"), "sharpe_red": float("nan"), "regime_skew": float("nan"), "regime_pass": None}
    merged = df.merge(reg, on="date", how="left")
    results = {}
    for regime in ["green", "red", "flat"]:
        sub = merged[merged["regime"] == regime]
        if sub.empty:
            results[f"sharpe_{regime}"] = float("nan")
        else:
            daily = sub.groupby("date")["net_ticks"].sum()
            results[f"sharpe_{regime}"] = float(daily.mean() / daily.std(ddof=1) * np.sqrt(252)) if daily.std(ddof=1) > 0 and len(daily) > 1 else float("nan")
    sg = results.get("sharpe_green", float("nan"))
    sr = results.get("sharpe_red", float("nan"))
    if np.isfinite(sg) and np.isfinite(sr) and max(abs(sg), abs(sr)) > 0:
        skew = abs(sg - sr) / max(abs(sg), abs(sr))
        regime_pass = skew <= 0.50
    else:
        skew = float("nan")
        regime_pass = None
    results["regime_skew"] = skew
    results["regime_pass"] = regime_pass
    return results


def daily_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    d = df.groupby("date")["net_ticks"].agg(["sum", "count", lambda x: (x > 0).mean()])
    d.columns = ["pnl_ticks", "n_trades", "wr"]
    d["positive"] = d["pnl_ticks"] > 0
    return d.reset_index()


def verify_fills(df: pd.DataFrame, name: str) -> str:
    """HC #491 R2: print first 3 fill rows + non-zero fill count."""
    lines = [f"\n--- Verify Branch {name} ---"]
    if df.empty:
        lines.append("WARNING: ZERO FILLS — branch broken or no matching signals")
        return "\n".join(lines)
    lines.append(f"Total fills: {len(df):,} (non-zero: {(df['net_ticks'] != 0).sum():,})")
    lines.append("First 3 fills:")
    for _, row in df.head(3).iterrows():
        lines.append(f"  date={row['date']} dir={row['direction']} net={row['net_ticks']:+.3f}t "
                     f"type={row['fill_type']} strength={row['pred_strength']:.4f}")
    return "\n".join(lines)


def verdict(m: dict, name: str, extra: str = "") -> str:
    n = m["n"]
    pf = m["pf"]
    wr = m["wr"]
    mean_tk = m["mean_tk_net"]
    n_pos = m["n_pos_days"]
    n_days = m["n_days"]
    sharpe = m["sharpe_ann"]

    if n < 100:
        return f"REJECT — trade count too thin ({n} < 100 minimum)"
    if n_pos >= 8 and pf > 1.0 and mean_tk > 0:
        label = "PASS"
    elif n_pos >= 6 and pf > 0.80 and mean_tk > -0.2:
        label = "WEAK PASS"
    else:
        label = "REJECT"

    reason = (f"n={n}, {n_pos}/{n_days} positive days, "
              f"net={mean_tk:+.3f}t/trade, WR={wr:.1%}, PF={pf:.3f}, Sharpe={sharpe:.2f}")
    if extra:
        reason += f" | {extra}"
    return f"{label} — {reason}"


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main():
    import mlflow

    mlflow.set_tracking_uri("http://localhost:5000")
    try:
        exp_id = mlflow.create_experiment("v7_fifo_branches")
    except Exception:
        exp_id = mlflow.get_experiment_by_name("v7_fifo_branches").experiment_id

    log.info("=" * 70)
    log.info("HC #491 R1 — v7 FIFO Branch Evaluation")
    log.info(f"TP={TP_TICKS}t SL={SL_TICKS}t hold≤{HOLD_S}s")
    log.info(f"Base cancel (harness default): 30s | v7 grade override: 1.0s")
    log.info(f"Branch B override: 0.25s")
    log.info("=" * 70)

    perday = load_v7_perday()
    if not perday:
        log.error("No aligned dates — abort.")
        sys.exit(1)

    reg = load_regime_labels()

    # ── Branch A: SHORT-ONLY, top 5%, cancel=1.0s ────────────────────
    log.info("\n[Branch A] SHORT-ONLY, top 5% per day, cancel=1.0s")
    selA = select_topk(perday, "short", 0.05)
    dfA = run_branch("branch_A_short_only", selA, cancel_s=1.0)

    # ── Branch B: BOTH sides, top 5%, cancel=0.25s ───────────────────
    log.info("\n[Branch B] BOTH sides, top 5%, cancel=0.25s (tight)")
    selB_short = select_topk(perday, "short", 0.05)
    selB_long  = select_topk(perday, "long", 0.05)
    # Run separately then merge
    dfB_short = run_branch("branch_B_short", selB_short, cancel_s=0.25)
    dfB_long  = run_branch("branch_B_long",  selB_long,  cancel_s=0.25)
    dfB = pd.concat([dfB_short, dfB_long], ignore_index=True) if not dfB_short.empty or not dfB_long.empty else pd.DataFrame()
    if not dfB.empty:
        (OUT_ROOT / "branch_B_tight_cancel").mkdir(parents=True, exist_ok=True)
        dfB.to_parquet(OUT_ROOT / "branch_B_tight_cancel" / "fills.parquet")

    # ── Branch C1: SHORT-ONLY, top 1%, cancel=1.0s ───────────────────
    log.info("\n[Branch C1] SHORT-ONLY, top 1% confidence, cancel=1.0s")
    selC1 = select_abs_threshold(perday, "short", 0.01)
    dfC1 = run_branch("branch_C1_top1pct", selC1, cancel_s=1.0)

    # ── Branch C2: SHORT-ONLY, top 2%, cancel=1.0s ───────────────────
    log.info("\n[Branch C2] SHORT-ONLY, top 2% confidence, cancel=1.0s")
    selC2 = select_abs_threshold(perday, "short", 0.02)
    dfC2 = run_branch("branch_C2_top2pct", selC2, cancel_s=1.0)

    # ── Branch D: SHORT-ONLY + tight cancel 0.25s + top 2% ───────────
    log.info("\n[Branch D] SHORT-ONLY + top 2% + tight cancel=0.25s")
    # Reuse selC2 (short top 2%) but with 0.25s cancel
    dfD = run_branch("branch_D_combined", selC2, cancel_s=0.25)

    # ── Compute metrics for all branches ─────────────────────────────
    branches = {
        "A (short-only, 5%, 1s cancel)": dfA,
        "B (both, 5%, 0.25s cancel)":    dfB,
        "C1 (short, top1%, 1s cancel)":  dfC1,
        "C2 (short, top2%, 1s cancel)":  dfC2,
        "D (short, top2%, 0.25s cancel)": dfD,
    }

    results = {}
    for bname, df in branches.items():
        m = metrics_for(df)
        rc = regime_check(df, reg)
        db = daily_breakdown(df)
        results[bname] = {"metrics": m, "regime": rc, "daily": db, "df": df}

    # ── HC #491 R2: verify fills ──────────────────────────────────────
    verify_lines = []
    for bname, df in branches.items():
        verify_lines.append(verify_fills(df, bname))

    # ── MLflow logging ────────────────────────────────────────────────
    for bname, res in results.items():
        m = res["metrics"]
        rc = res["regime"]
        safe_name = bname.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "").replace("/", "_")
        with mlflow.start_run(experiment_id=exp_id, run_name=safe_name):
            mlflow.log_param("branch", bname)
            mlflow.log_param("tp_ticks", TP_TICKS)
            mlflow.log_param("sl_ticks", SL_TICKS)
            mlflow.log_param("hold_s", HOLD_S)
            for k, v in m.items():
                if v is not None and v != float("inf") and str(v) != "nan":
                    try:
                        mlflow.log_metric(k, float(v))
                    except Exception:
                        pass
            for k, v in rc.items():
                if v is not None and isinstance(v, (int, float)) and str(v) != "nan":
                    try:
                        mlflow.log_metric(k, float(v))
                    except Exception:
                        pass

    # ── Build report ──────────────────────────────────────────────────
    report_lines = [
        "# HC #491 R1 — v7 FIFO Branch Evaluation Report",
        "",
        f"Date: 2026-05-28",
        f"Predictions: v7 prod concat ({V7_PRED_NPZ.name})",
        f"17 OOT dates with local DBN tape",
        f"Config: TP={TP_TICKS}t SL={SL_TICKS}t hold≤{HOLD_S}s",
        f"Base harness cancel default: 30s | v7 grade override: 1.0s | Branch B override: 0.25s",
        f"Cost: passive limit = 0.376t (commission only), no crossing cost",
        "",
        "## Baseline (v7 base, both sides, top 5%, cancel=1.0s)",
        "| Metric | Value |",
        "|--------|-------|",
        "| Net ticks/trade | -0.621 |",
        "| WR | 29.5% |",
        "| PF | 0.334 |",
        "| Positive days | 0/17 |",
        "| Sharpe (ann) | -25.3 |",
        "| Verdict | HARD REJECT |",
        "",
    ]

    for bname, res in results.items():
        m = res["metrics"]
        rc = res["regime"]
        db = res["daily"]
        df = res["df"]

        sg = rc.get("sharpe_green", float("nan"))
        sr = rc.get("sharpe_red", float("nan"))
        skew = rc.get("regime_skew", float("nan"))
        rpass = rc.get("regime_pass", None)
        regime_str = (f"Sharpe_green={sg:.2f}, Sharpe_red={sr:.2f}, "
                      f"skew={skew:.2f} → {'PASS' if rpass else 'FAIL' if rpass is False else 'N/A'}")

        verd = verdict(m, bname, regime_str)

        report_lines += [
            f"## Branch {bname}",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| N trades | {m['n']:,} |",
            f"| N days | {m['n_days']} |",
            f"| Positive days | {m['n_pos_days']}/{m['n_days']} |",
            f"| Net ticks/trade | {m['mean_tk_net']:+.4f} |",
            f"| WR | {m['wr']:.1%} |",
            f"| PF | {m['pf']:.4f} |",
            f"| Sharpe (ann) | {m['sharpe_ann']:.4f} |",
            f"| Sortino (ann) | {m['sortino_ann']:.4f} |",
            f"| Regime check | {regime_str} |",
            "",
        ]

        # Per-day breakdown
        if not db.empty:
            report_lines.append("### Per-day breakdown")
            report_lines.append("| Date | N trades | PnL ticks | WR | Positive |")
            report_lines.append("|------|----------|-----------|-----|---------|")
            for _, row in db.iterrows():
                report_lines.append(
                    f"| {row['date']} | {int(row['n_trades'])} | {row['pnl_ticks']:+.2f} "
                    f"| {row['wr']:.1%} | {'YES' if row['positive'] else 'NO'} |"
                )
            report_lines.append("")

        report_lines += [
            f"**Verdict: {verd}**",
            "",
        ]

    # HC #491 R2: verify section
    report_lines += ["", "## HC #491 R2 — Fill Verification", "```"]
    report_lines.extend(verify_lines)
    report_lines.append("```")

    # Recommendation if all reject
    all_verdicts = [verdict(results[b]["metrics"], b) for b in results]
    all_reject = all(v.startswith("REJECT") for v in all_verdicts)

    report_lines += [
        "",
        "## Summary & Recommendation",
        "",
    ]

    for bname in branches:
        verd = verdict(results[bname]["metrics"], bname)
        report_lines.append(f"- **{bname}**: {verd}")

    if all_reject:
        report_lines += [
            "",
            "### All branches REJECTED.",
            "",
            "Root cause: adverse selection under FIFO persists across all variants. "
            "The 1s prediction horizon is too short for limit order queue-wait dynamics. "
            "Even short-only, tight-cancel, and top-1% confidence fail to overcome the "
            "systematic adverse fill bias (-0.55 slippage ticks average in base).",
            "",
            "**Next axis per HC #488 R2 (axis rotation):**",
            "",
            "**MODEL AXIS** — retrain on longer horizon (5s or 10s target), which gives "
            "the limit order time to sit in queue WITHOUT being adversely selected. "
            "The 5s CNN-Mamba v2 predictions (preds[:,1]) show IC=0.141 and MFE "
            "extends to 30s — far better alignment with passive limit fill mechanics. "
            "Test: re-run the same harness using preds[:,1] (5s horizon) with "
            "hold≤7.5s, cancel≤5s, TP≤p90 MFE(5s). This is the highest-probability "
            "next move given: (a) signal still exists at 5s, (b) queue wait ~370ms avg "
            "is a small fraction of 5s, (c) short edge is strongest at 5s per decay analysis.",
        ]
    else:
        passing = [b for b in branches if not verdict(results[b]["metrics"], b).startswith("REJECT")]
        report_lines += [
            "",
            f"**Passing branches: {', '.join(passing)}**",
            "Recommend: run regime-stratified validation per HC #428 R1 on passing branch(es).",
        ]

    report_text = "\n".join(report_lines)
    REPORT_PATH.write_text(report_text)
    log.info(f"\nReport saved: {REPORT_PATH}")
    log.info("\n" + "=" * 70)
    log.info("BRANCH EVALUATION COMPLETE")
    log.info("=" * 70)
    for bname in branches:
        verd = verdict(results[bname]["metrics"], bname)
        log.info(f"  {bname}: {verd}")


if __name__ == "__main__":
    main()
