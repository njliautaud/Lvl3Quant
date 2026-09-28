#!/usr/bin/env python3
"""
HC #450 R5 — Canonical FIFO replay for the two profitable cells from the
HC #450 R3+R4 diagnostic, plus EMA-smoothed-signal variants.

Per HC #74: FIFO market replay only (NEVER midpoint).
Per HC #428 R1: regime-agnostic OOT (>=40 days) — stratify Sharpe by ES regime.
Per HC #428 R2: hold_seconds <= 1.5*h, cancel_window <= h. For h=1s -> hold<=1.5s, cancel<=1.0s.

Reuses the canonical FIFO replay engine from
  alpha_discovery.deep_models.fifo_market_replay.FIFOReplayEngine
exactly as the HC #432 / HC #443 / HC #444 chain does. Does not reinvent.

Cells (10):
  A1..A5 : CNN-Mamba v2 short top-1% horizon-1s, EMA N in {1(raw), 4, 10, 20, 40}
  B1..B5 : PatchTST       long  top-1% horizon-1s, EMA N in {1(raw), 4, 10, 20, 40}

EMA is CAUSAL — applied per-day on the 1s prediction stream BEFORE
confidence banding so that smoothing changes which timestamps are selected.

Outputs:
  output/hc450_canonical_replay/summary.csv
  output/hc450_canonical_replay/summary.md
  output/hc450_canonical_replay/{cell}_fifo_fills.csv
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

LVL3_ROOT = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(LVL3_ROOT))

OUT_DIR = LVL3_ROOT / "output" / "hc450_canonical_replay"
OUT_DIR.mkdir(parents=True, exist_ok=True)

V2_DIR = LVL3_ROOT / "output" / "cnn_mamba_v2_bulk_oot_v2"
PATCHTST_DIR = LVL3_ROOT / "output" / "patchtst_bulk_oot"
MBO_EVENT_DIR = LVL3_ROOT / "data" / "processed" / "mbo_events_smart_v3"
REGIME_PARQUET = LVL3_ROOT / "output" / "regime_labels" / "oot_dates_regime.parquet"

ES_TICK_VALUE = 12.50
ES_RT_COMMISSION_TICKS = 0.376  # CANONICAL — HC #74 / CLAUDE.md

# HC #428 R2 bounds for h=1s
H_SEC = 1.0
HOLD_S = 1.5            # <= 1.5*h = 1.5s
CANCEL_S = 1.0          # <= h = 1.0s

# entry params (passive limit at touch) — passive scalp
TP_TICKS = 1.0
SL_TICKS = 1.0
ORDER_TYPE = "passive_at_touch"  # mapped to 'limit' in engine

# horizon column index — both NPZs are ['1s', '5s', '10s']
H1S_IDX = 0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("hc450_replay")


# ──────────────────────────────────────────────────────────────────────
# Signal generation
# ──────────────────────────────────────────────────────────────────────
def causal_ema(x: np.ndarray, n: int) -> np.ndarray:
    """Causal exponential moving average with span n.
    n=1 returns the input unchanged. alpha = 2/(n+1) (standard pandas EMA).
    Uses only past+current samples (no leakage).
    """
    if n <= 1:
        return x.astype(np.float64)
    alpha = 2.0 / (n + 1)
    y = np.empty_like(x, dtype=np.float64)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1 - alpha) * y[i - 1]
    return y


def load_perday_preds(model_dir: Path) -> Dict[str, dict]:
    """Load all per-date NPZs from model_dir.
    Returns {date_str: {'preds_1s': np.ndarray, 'window_size': int, 'stride': int}}.
    Only files matching <YYYYMMDD>_predictions.npz are loaded.
    """
    out = {}
    files = sorted(model_dir.glob("*_predictions.npz"))
    for fp in files:
        date_str = fp.stem.replace("_predictions", "")
        if not (len(date_str) == 8 and date_str.isdigit()):
            continue
        d = np.load(fp, allow_pickle=False)
        preds = d["predictions"]
        if preds.ndim != 2 or preds.shape[1] < 1:
            d.close(); continue
        out[date_str] = {
            "preds_1s": preds[:, H1S_IDX].astype(np.float64),
            "window_size": int(d["window_size"]),
            "stride": int(d["stride"]),
        }
        d.close()
    return out


def select_top_conf_per_day(
    perday: Dict[str, dict],
    side: str,
    pct: float,
    ema_n: int,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, int, int]]:
    """For each day, apply causal EMA(n) then select the top `pct` fraction
    by |signed strength| on the requested side. Returns
      {date: (idx_in_day, strength, window_size, stride)}
    Per-day selection (not global) matches HC #450 R3 diagnostic semantics
    where the top-1% cell was identified per-day.
    """
    out = {}
    for d, rec in perday.items():
        x_raw = rec["preds_1s"]
        if x_raw.size == 0:
            continue
        x = causal_ema(x_raw, ema_n)
        if side == "long":
            mask_side = x > 0
            strength = x
        else:  # short
            mask_side = x < 0
            strength = -x  # positive strength magnitude
        side_strengths = strength[mask_side]
        if side_strengths.size == 0:
            continue
        k = max(1, int(side_strengths.size * pct))
        thresh = np.partition(side_strengths, -k)[-k]
        selected = mask_side & (strength >= thresh)
        idx = np.where(selected)[0]
        if idx.size == 0:
            continue
        out[d] = (idx, strength[idx], rec["window_size"], rec["stride"])
    return out


# ──────────────────────────────────────────────────────────────────────
# Map sample idx → MBO ts_ns (per-day)
# ──────────────────────────────────────────────────────────────────────
def map_idx_to_ts(date_str: str, idx_in_day: np.ndarray, window_size: int, stride: int) -> Optional[np.ndarray]:
    mbo_path = MBO_EVENT_DIR / f"{date_str}_mbo_events.npz"
    if not mbo_path.exists():
        return None
    mbo = np.load(mbo_path, allow_pickle=False)
    ts_events = mbo["timestamps"].astype(np.int64)
    n_events = len(ts_events)
    event_idx = np.minimum(idx_in_day * stride + window_size - 1, n_events - 1)
    return ts_events[event_idx]


# ──────────────────────────────────────────────────────────────────────
# Per-day FIFO replay (called in worker processes)
# ──────────────────────────────────────────────────────────────────────
def run_one_date(
    date_str: str,
    idx_in_day: np.ndarray,
    direction: str,
    strength: np.ndarray,
    window_size: int,
    stride: int,
) -> List[dict]:
    from alpha_discovery.deep_models.fifo_market_replay import FIFOReplayEngine

    ts_ns = map_idx_to_ts(date_str, idx_in_day, window_size, stride)
    if ts_ns is None:
        return [{"date": date_str, "error": "missing_mbo_events"}]

    signals = [
        {"ts_ns": int(t), "direction": direction, "strength": float(strength[i])}
        for i, t in enumerate(ts_ns)
    ]
    if not signals:
        return []

    cancel_ns = int(CANCEL_S * 1_000_000_000)
    hold_ns = int(HOLD_S * 1_000_000_000)

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
            order_type="limit",  # passive_at_touch maps to limit
        )
    except Exception as e:
        return [{"date": date_str, "error": f"simulate: {e}"}]

    fills: List[dict] = []
    for t in trades:
        hold_seconds = (t.exit_ts_ns - t.entry_ts_ns) / 1e9 if (t.entry_ts_ns and t.exit_ts_ns) else 0.0
        fills.append({
            "date": date_str,
            "ts_signal_ns": int(t.signal_ts_ns),
            "ts_entry_ns":  int(t.entry_ts_ns) if t.entry_ts_ns else 0,
            "ts_exit_ns":   int(t.exit_ts_ns)  if t.exit_ts_ns else 0,
            "direction":    t.direction,
            "order_type":   t.order_type,
            "entry_raw":    int(t.entry_price_raw) if t.entry_price_raw else 0,
            "exit_raw":     int(t.exit_price_raw)  if t.exit_price_raw  else 0,
            "hold_s":       hold_seconds,
            "fill_type":    t.exit_reason,
            "net_ticks":    float(t.pnl_ticks_net),
            "net_dollars":  float(t.pnl_dollars),
            "queue_ahead":  int(t.queue_ahead),
            "queue_wait_ns": int(t.queue_wait_ns),
            "slippage_ticks": float(t.slippage_ticks),
            "pred_strength":  float(t.pred_strength),
        })
    return fills


# ──────────────────────────────────────────────────────────────────────
# Regime classification (HC #428 R1)
# ──────────────────────────────────────────────────────────────────────
def load_regime_labels() -> pd.DataFrame:
    df = pd.read_parquet(REGIME_PARQUET)
    df["date"] = df["date"].astype(str).str.zfill(8)
    def classify(d):
        delta = d["close_minus_open_ticks"]
        if delta >= 4:
            return "green"
        if delta <= -4:
            return "red"
        return "flat"
    df["regime"] = df.apply(classify, axis=1)
    return df[["date", "regime", "close_minus_open_ticks"]]


# ──────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────
def per_day_pnl(fills_df: pd.DataFrame) -> pd.Series:
    return fills_df.groupby("date")["net_ticks"].sum()


def metrics_for(fills_df: pd.DataFrame) -> dict:
    if fills_df.empty:
        return {k: np.nan for k in [
            "n", "n_days", "fills_per_day_mean", "mean_tk_net",
            "sharpe_ann", "sortino_ann", "pf", "wr",
            "day_positive_pct", "day_conc",
        ]}
    nets = fills_df["net_ticks"].values.astype(np.float64)
    n = len(nets)
    daily = per_day_pnl(fills_df)
    n_days = daily.size
    mean_tk_net = float(nets.mean())  # already net of commission via engine pnl_ticks_net

    # Daily Sharpe annualized (252 trading days)
    if daily.std(ddof=1) > 0 and n_days > 1:
        sharpe_ann = float(daily.mean() / daily.std(ddof=1) * np.sqrt(252))
    else:
        sharpe_ann = float("nan")
    # Sortino (only downside std)
    down = daily[daily < 0]
    if down.size >= 1 and down.std(ddof=1) > 0:
        sortino_ann = float(daily.mean() / down.std(ddof=1) * np.sqrt(252))
    else:
        sortino_ann = float("nan") if daily.mean() <= 0 else float("inf")
    # PF / WR
    wins = nets[nets > 0]
    losses = nets[nets < 0]
    pf = float(wins.sum() / -losses.sum()) if losses.size > 0 and losses.sum() < 0 else float("inf")
    wr = float((nets > 0).mean())
    day_positive_pct = float((daily > 0).mean())
    # day concentration: top 1 day share of total positive pnl
    total = daily.sum()
    if abs(total) > 0:
        day_conc = float(daily.abs().max() / daily.abs().sum())
    else:
        day_conc = float("nan")
    return {
        "n": n,
        "n_days": int(n_days),
        "fills_per_day_mean": float(n / max(1, n_days)),
        "mean_tk_net": mean_tk_net,
        "sharpe_ann": sharpe_ann,
        "sortino_ann": sortino_ann,
        "pf": pf,
        "wr": wr,
        "day_positive_pct": day_positive_pct,
        "day_conc": day_conc,
    }


def regime_sharpe(daily_pnl: pd.Series, regimes: pd.DataFrame) -> dict:
    """Sharpe stratified per regime (green/red/flat). Returns dict + |delta|/max."""
    df = daily_pnl.reset_index().rename(columns={"net_ticks": "pnl"})
    df = df.merge(regimes, on="date", how="left")
    df["regime"] = df["regime"].fillna("flat")
    out = {}
    for r in ("green", "red", "flat"):
        sub = df[df["regime"] == r]["pnl"].values.astype(np.float64)
        if sub.size > 1 and sub.std(ddof=1) > 0:
            sh = float(sub.mean() / sub.std(ddof=1) * np.sqrt(252))
        else:
            sh = float("nan")
        out[f"sharpe_{r}"] = sh
        out[f"n_days_{r}"] = int(sub.size)
        out[f"mean_pnl_{r}"] = float(sub.mean()) if sub.size > 0 else float("nan")
    g = out["sharpe_green"]; r = out["sharpe_red"]
    if np.isfinite(g) and np.isfinite(r):
        mx = max(abs(g), abs(r))
        out["regime_delta"] = abs(g - r)
        out["regime_delta_ratio"] = float(out["regime_delta"] / mx) if mx > 0 else float("nan")
    else:
        out["regime_delta"] = float("nan")
        out["regime_delta_ratio"] = float("nan")
    return out


# ──────────────────────────────────────────────────────────────────────
# Cell runner
# ──────────────────────────────────────────────────────────────────────
def run_cell(
    cell_name: str,
    model_dir: Path,
    side: str,
    pct: float,
    ema_n: int,
    workers: int,
) -> Tuple[pd.DataFrame, dict]:
    log.info(f"[{cell_name}] loading model preds from {model_dir}")
    perday = load_perday_preds(model_dir)
    log.info(f"[{cell_name}]   {len(perday)} OOT days loaded")

    sigs = select_top_conf_per_day(perday, side, pct, ema_n)
    log.info(f"[{cell_name}]   {len(sigs)} signal days "
             f"(side={side}, top {pct*100:.2f}%, ema_n={ema_n})")

    tasks = [(d, idx, side, st, ws, sd) for d, (idx, st, ws, sd) in sigs.items()]

    fills_all: List[dict] = []
    errors: List[dict] = []
    workers = max(1, min(workers, len(tasks)))
    if workers == 1 or len(tasks) <= 1:
        for d, idx, direction, st, ws, sd in tasks:
            fills = run_one_date(d, idx, direction, st, ws, sd)
            for f in fills:
                (errors if "error" in f else fills_all).append(f)
            log.info(f"[{cell_name}]   {d}: {sum(1 for f in fills if 'error' not in f)} fills")
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            futs = {
                ex.submit(run_one_date, d, idx, dir_, st, ws, sd): d
                for d, idx, dir_, st, ws, sd in tasks
            }
            for fut in as_completed(futs):
                d = futs[fut]
                try:
                    fills = fut.result()
                except Exception as e:
                    errors.append({"date": d, "error": f"future: {e}"})
                    continue
                for f in fills:
                    (errors if "error" in f else fills_all).append(f)
                log.info(f"[{cell_name}]   {d}: {sum(1 for f in fills if 'error' not in f)} fills")

    df = pd.DataFrame(fills_all)
    if not df.empty:
        df["date"] = df["date"].astype(str).str.zfill(8)
        df.to_csv(OUT_DIR / f"{cell_name}_fifo_fills.csv", index=False)

    if errors:
        (OUT_DIR / f"{cell_name}_errors.json").write_text(json.dumps(errors, indent=2))
        log.warning(f"[{cell_name}] {len(errors)} day errors recorded")

    if df.empty:
        return df, {"n": 0, "n_days": 0}

    m = metrics_for(df)
    # regime stratification
    regimes = load_regime_labels()
    daily = df.groupby("date")["net_ticks"].sum().rename("net_ticks")
    rs = regime_sharpe(daily, regimes)
    m.update(rs)

    # HC #428 R1: regime-balance gate
    rd = m.get("regime_delta_ratio", float("nan"))
    r1_pass = bool(np.isfinite(rd) and rd <= 0.50)
    # HC #344: day_conc <= 0.70
    dc_pass = bool(np.isfinite(m["day_conc"]) and m["day_conc"] <= 0.70)
    # HC #428 R2: hold/cancel/horizon bounds — encoded in constants
    r2_pass = (HOLD_S <= 1.5 * H_SEC) and (CANCEL_S <= H_SEC)
    # net profitability after commission (engine pnl_ticks_net already nets commission)
    profitable = bool(m["mean_tk_net"] > 0)
    m["hc428_r1_pass"] = r1_pass
    m["hc428_r2_pass"] = r2_pass
    m["hc344_dayconc_pass"] = dc_pass
    m["profitable_after_costs"] = profitable
    m["overall_pass"] = r1_pass and r2_pass and dc_pass and profitable and m["n_days"] >= 40
    m["cell"] = cell_name
    m["ema_n"] = ema_n
    m["side"] = side
    m["pct"] = pct
    return df, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8,
                    help="parallel workers (per-day, spawn). Each opens a DBN file.")
    ap.add_argument("--cells", default="all",
                    help="comma list of cell names to run (e.g. A1,A2). 'all' = all 10.")
    args = ap.parse_args()

    matrix = [
        # (cell_name, model_dir, side, top_pct, ema_n)
        ("A1_v2_short_raw",   V2_DIR,       "short", 0.01, 1),
        ("A2_v2_short_ema4",  V2_DIR,       "short", 0.01, 4),
        ("A3_v2_short_ema10", V2_DIR,       "short", 0.01, 10),
        ("A4_v2_short_ema20", V2_DIR,       "short", 0.01, 20),
        ("A5_v2_short_ema40", V2_DIR,       "short", 0.01, 40),
        ("B1_patchtst_long_raw",   PATCHTST_DIR, "long",  0.01, 1),
        ("B2_patchtst_long_ema4",  PATCHTST_DIR, "long",  0.01, 4),
        ("B3_patchtst_long_ema10", PATCHTST_DIR, "long",  0.01, 10),
        ("B4_patchtst_long_ema20", PATCHTST_DIR, "long",  0.01, 20),
        ("B5_patchtst_long_ema40", PATCHTST_DIR, "long",  0.01, 40),
    ]

    if args.cells != "all":
        keep = set(args.cells.split(","))
        matrix = [r for r in matrix if r[0].split("_")[0] in keep or r[0] in keep]

    rows = []
    for name, mdir, side, pct, ema_n in matrix:
        log.info(f"=== RUNNING CELL {name} ===")
        try:
            df, m = run_cell(name, mdir, side, pct, ema_n, args.workers)
            rows.append(m)
        except Exception as e:
            log.error(f"[{name}] FAILED: {e}", exc_info=True)
            rows.append({"cell": name, "ema_n": ema_n, "side": side, "pct": pct,
                         "error": str(e), "n": 0, "n_days": 0})

    summary = pd.DataFrame(rows)
    # column order
    front = ["cell", "side", "ema_n", "n", "n_days", "fills_per_day_mean",
             "mean_tk_net", "sharpe_ann", "sortino_ann", "pf", "wr",
             "day_positive_pct", "day_conc",
             "sharpe_green", "sharpe_red", "sharpe_flat",
             "n_days_green", "n_days_red", "n_days_flat",
             "regime_delta", "regime_delta_ratio",
             "hc428_r1_pass", "hc428_r2_pass", "hc344_dayconc_pass",
             "profitable_after_costs", "overall_pass"]
    front = [c for c in front if c in summary.columns]
    other = [c for c in summary.columns if c not in front]
    summary = summary[front + other]
    summary.to_csv(OUT_DIR / "summary.csv", index=False)
    log.info(f"wrote {OUT_DIR/'summary.csv'}")

    # ─── markdown ───
    md = ["# HC #450 R5 — Canonical FIFO Replay (definitive go/no-go)", "",
          f"Engine: `alpha_discovery.deep_models.fifo_market_replay.FIFOReplayEngine` (HC #74).",
          f"Cost: ES_RT_COMMISSION_TICKS = {ES_RT_COMMISSION_TICKS} (already netted in engine `pnl_ticks_net`).",
          f"Order type: passive limit at touch. TP = {TP_TICKS} tk. SL = {SL_TICKS} tk.",
          f"HC #428 R2 bounds: horizon = {H_SEC}s, hold_s = {HOLD_S}, cancel_s = {CANCEL_S}.",
          f"Regime labels: ES close-minus-open ticks; |Δ|<4 = flat, ≥+4 = green, ≤-4 = red "
          f"(canonical per `output/regime_labels/oot_dates_regime.parquet`).",
          "",
          "## Results (one row per cell, sorted by mean_tk_net desc)",
          ""]
    cols = ["cell", "ema_n", "n", "n_days", "mean_tk_net", "sharpe_ann",
            "sortino_ann", "pf", "wr", "day_positive_pct", "day_conc",
            "sharpe_green", "sharpe_red", "sharpe_flat", "regime_delta_ratio",
            "hc428_r1_pass", "overall_pass"]
    cols = [c for c in cols if c in summary.columns]
    sorted_sum = summary.sort_values("mean_tk_net", ascending=False, na_position="last") \
        if "mean_tk_net" in summary.columns else summary
    # header
    md.append("| " + " | ".join(cols) + " |")
    md.append("|" + "|".join(["---"] * len(cols)) + "|")
    def _fmt(v):
        if isinstance(v, float):
            if not np.isfinite(v):
                return "nan"
            return f"{v:.4f}"
        if isinstance(v, (bool, np.bool_)):
            return "Y" if v else "N"
        return str(v)
    for _, r in sorted_sum.iterrows():
        md.append("| " + " | ".join(_fmt(r[c]) if c in r.index else "" for c in cols) + " |")

    # verdict
    md += ["", "## Verdict", ""]
    winners = sorted_sum[sorted_sum.get("overall_pass", False) == True] if "overall_pass" in sorted_sum.columns else sorted_sum.iloc[:0]
    if not winners.empty:
        w = winners.iloc[0]
        md.append(
            f"**WINNER: cell {w['cell']}.** mean_tk_net = {w['mean_tk_net']:.4f} tk after "
            f"commission. Sharpe_ann = {w['sharpe_ann']:.3f} (regime-balanced: "
            f"|delta_ratio| = {w['regime_delta_ratio']:.3f}). "
            f"HC #428 R1 {'PASS' if w['hc428_r1_pass'] else 'FAIL'}. "
            f"HC #428 R2 {'PASS' if w['hc428_r2_pass'] else 'FAIL'}. "
            f"Day-positive = {w['day_positive_pct']*100:.0f}%."
        )
    else:
        md.append("**NO CELL PASSES HC #428.** Closest-to-pass and the reason it failed:")
        # rank by mean_tk_net first
        for _, r in sorted_sum.head(5).iterrows():
            reasons = []
            if not r.get("profitable_after_costs", False): reasons.append("unprofitable after commission")
            if not r.get("hc428_r1_pass", False):
                rd = r.get("regime_delta_ratio", float("nan"))
                reasons.append(f"regime-imbalanced (|delta_ratio|={rd:.2f})" if np.isfinite(rd) else "regime data missing")
            if not r.get("hc344_dayconc_pass", False):
                dc = r.get("day_conc", float("nan"))
                reasons.append(f"day_conc={dc:.2f} > 0.70")
            if not r.get("hc428_r2_pass", False): reasons.append("R2 bound violation")
            if r.get("n_days", 0) < 40: reasons.append(f"only {int(r.get('n_days', 0))} OOT days (<40)")
            md.append(f"- `{r['cell']}`: mean_tk_net={r['mean_tk_net']:.4f}, sharpe_ann={r.get('sharpe_ann', float('nan')):.2f}, "
                      f"day%={r.get('day_positive_pct', 0)*100:.0f}%. FAILED: {', '.join(reasons) or 'unknown'}")

    # smoothing wins/loses
    md += ["", "## Smoothing impact (EMA-N vs raw within each model)", ""]
    for prefix in ("A", "B"):
        block = summary[summary["cell"].str.startswith(prefix)].copy()
        if block.empty:
            continue
        raw = block[block["ema_n"] == 1]
        if raw.empty:
            continue
        raw_net = float(raw["mean_tk_net"].iloc[0])
        md.append(f"**{'CNN-Mamba v2 short' if prefix == 'A' else 'PatchTST long'} (raw mean_tk_net = {raw_net:.4f}):**")
        for _, r in block.iterrows():
            if r["ema_n"] == 1:
                continue
            delta = float(r["mean_tk_net"]) - raw_net
            beats = delta > 0
            md.append(f"- EMA-{int(r['ema_n'])}: mean_tk_net = {r['mean_tk_net']:.4f} (delta = {delta:+.4f}) — "
                      f"smoothed beats raw? {'YES' if beats else 'NO'}")

    (OUT_DIR / "summary.md").write_text("\n".join(md))
    log.info(f"wrote {OUT_DIR/'summary.md'}")
    log.info("\n" + "\n".join(md[:30]))


if __name__ == "__main__":
    main()
