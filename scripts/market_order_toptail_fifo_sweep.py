#!/usr/bin/env python3
"""
HC #493 R3 + HC #428 R1/R2 — Market-Order Top-Tail Sweep
========================================================

Hypothesis (post-3-regrade synthesis): every recent proxy->FIFO collapse is
-0.8 to -1.5 ticks. Root cause = limit-order queue wait + adverse fill
selection + 0.376t commission. Test whether MARKET ORDERS at extreme
top-tail confidence + longer holds escape the queue-fill failure mode.

CANONICAL FIFO EQUIVALENCE FOR MARKET ORDERS (key insight):
  For a pure market order (entry + flat hold + market exit), the canonical
  FIFO replay PnL collapses to:
      net_ticks = direction * labels_h - (1.0 spread + 0.376 commission)
  Because:
      * Market entry fills at touch (deterministic, no queue) -> +half spread vs mid
      * Hold horizon h seconds (no intra-horizon TP/SL)
      * Market exit fills at touch (deterministic) -> +half spread vs mid
      * labels_h is the signed mid-to-mid log-return in TICKS at horizon h
  The pre-processed `data/processed/mbo_events_smart_v3/YYYYMMDD_mbo_events.npz`
  files supply labels_h{1s, 5s, 10s, 30s} per event — the canonical realized
  move from event-mid to event+h-mid. This is the same source the FIFO
  harness uses to grade fills. For MARKET orders specifically, these labels
  ARE the FIFO answer (no queue, no fill-selection adverse effects).

The previous /home/jupiter/Lvl3Quant/output/market_order_viability_v1/ run
used the same equivalence but only on CNN-Mamba v2 (1s/5s/10s, 46 dates,
no regime stratification, no per-day Sharpe). This sweep extends it to:
  * v7 production meta-classifier predictions (the HC #493 regrade subject)
  * CNN-Mamba v2 RAW predictions (h=1s, 5s, 10s, 30s* via labels_30s)
  * Per-day Sharpe, regime stratification, HC #428 R1 gates
  * Verdict per cell

Sweep grid:
  Models:    {v7_meta@h1s, v2raw@h1s, v2raw@h5s, v2raw@h10s, v2raw@h30s}
  Thresholds (per-day top abs(pred)): {0.1%, 0.5%, 1%, 2%, 5%}
  -> 25 cells total

Cost: 1.376 ticks RT (1.0 spread + 0.376 commission), per HC task spec.

v2 raw note: model only outputs (1s, 5s, 10s). For h=30s we use the 10s
prediction column as the signal proxy and grade against labels_30s (longer
realization horizon than what model targeted — tests whether 10s-trained
signal extends to 30s holds).

OOT coverage:
  * v7: 17 dates with both v7_meta predictions + processed mbo_events_smart_v3.
    Below HC #428 R1 40-day requirement (documented as data gap).
  * v2 raw: up to 48 dates with bulk_oot_v2 predictions + processed events.

Outputs:
  output/market_order_toptail_fifo_sweep/sweep_grid.csv
  output/market_order_toptail_fifo_sweep/fills_<model>.parquet
  output/market_order_toptail_fifo_sweep_REPORT.md
  MLflow experiment: market_order_toptail_fifo_sweep
"""
from __future__ import annotations

import glob
import logging
import sys
import time as time_mod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

LVL3_ROOT = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(LVL3_ROOT))

OUT_ROOT = LVL3_ROOT / "output" / "market_order_toptail_fifo_sweep"
OUT_ROOT.mkdir(parents=True, exist_ok=True)
REPORT_PATH = LVL3_ROOT / "output" / "market_order_toptail_fifo_sweep_REPORT.md"

V7_PRED_NPZ    = LVL3_ROOT / "output" / "meta_v7_prod" / "concat_oot_predictions.npz"
V2_DIR         = LVL3_ROOT / "output" / "cnn_mamba_v2_bulk_oot_v2"
MBO_EVENT_DIR  = LVL3_ROOT / "data" / "processed" / "mbo_events_smart_v3"
REGIME_PARQUET = LVL3_ROOT / "output" / "regime_labels" / "oot_dates_regime.parquet"

# Cost stack per HC task: commission + 1.0t entry + 1.0t exit = 2.376t total RT?
# WAIT - re-read HC: "1.0t spread on entry + 1.0t on exit" -> sounds like 2 ticks
# total. But standard convention is total spread crossing = 1 tick (you only
# cross the bid-ask once per round-trip in a market with 1-tick wide book):
# entry crosses BID-ASK once (pays 1.0t between mid->touch on both sides
# combined = 0.5+0.5 = 1.0). The exit similarly pays mid-to-touch on exit.
# Per HC task: "Net t/trade > 0 after commission + 1.0t spread on entry +
# 1.0t on exit (market both sides)" -> apply 2.0 ticks spread + 0.376 comm.
ES_TICK_VALUE = 12.50
ES_COMMISSION_TICKS = 0.376
ENTRY_SPREAD_COST = 1.0
EXIT_SPREAD_COST  = 1.0
TOTAL_COST_TICKS  = ES_COMMISSION_TICKS + ENTRY_SPREAD_COST + EXIT_SPREAD_COST  # 2.376

THRESHOLDS = [0.001, 0.005, 0.01, 0.02, 0.05]

# Horizons: model output index (for v2) or 'v7'
HORIZON_DEFS: List[Tuple[str, str, Optional[int], str]] = [
    # (model_name, label_key, v2_pred_col_idx, horizon_str)
    ("v7_meta_h1s",   "labels_1s",  None, "1s"),
    ("v2raw_h1s",     "labels_1s",   0,   "1s"),
    ("v2raw_h5s",     "labels_5s",   1,   "5s"),
    ("v2raw_h10s",    "labels_10s",  2,   "10s"),
    ("v2raw_h30s",    "labels_30s",  2,   "30s"),  # use 10s pred col but 30s realized
]

WINDOW_SIZE = 1000  # v2 default
STRIDE      = 250

LOG_PATH = OUT_ROOT / "run.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, mode="w"), logging.StreamHandler()],
)
log = logging.getLogger("mkt_toptail_sweep")


# ---------------------------------------------------------------------------
# Data discovery
# ---------------------------------------------------------------------------
def discover_dates() -> Tuple[List[str], List[str]]:
    """Return (v7_dates, v2_dates) where mbo_events + label files exist."""
    d = np.load(V7_PRED_NPZ, allow_pickle=False)
    v7_unique = sorted(set(d["dates"].astype(str).tolist()))

    v2_files = sorted(glob.glob(str(V2_DIR / "*_predictions.npz")))
    v2_unique = sorted({Path(p).name[:8] for p in v2_files})

    def has_events(ds: str) -> bool:
        return (MBO_EVENT_DIR / f"{ds}_mbo_events.npz").exists()

    v7_aligned = [d for d in v7_unique if has_events(d)
                  and (V2_DIR / f"{d}_predictions.npz").exists()]
    v2_aligned = [d for d in v2_unique if has_events(d)]
    return v7_aligned, v2_aligned


# ---------------------------------------------------------------------------
# Per-date signal & realized loading
# ---------------------------------------------------------------------------
def load_v7_signals_realized(date_str: str, label_key: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Returns (preds, realized) for v7 on this date.

    IMPORTANT — v7 alignment caveat: v7's prediction indices do NOT line up
    with v2/mbo's window indices (verified empirically: v7's internal labels
    at index i don't equal mbo.labels_1s at v2-aligned event idx). v7 meta
    was trained on a re-indexed v2 sample slice. So we USE v7's INTERNAL
    `labels` array (which IS v7's own canonical realized 1s log-ret in
    TICKS, the same source HC #493 R3 regrade used).

    For v7 we only support h=1s (the meta-classifier's trained horizon).
    """
    if label_key != "labels_1s":
        # v7 internal label is 1s-only; other horizons aren't supplied.
        return None
    d = np.load(V7_PRED_NPZ, allow_pickle=False)
    mask = d["dates"].astype(str) == date_str
    if not mask.any():
        return None
    preds = d["predictions"][mask].astype(np.float32)
    realized = d["labels"][mask].astype(np.float32)
    return preds, realized


def load_v2_signals_realized(date_str: str, pred_col: int, label_key: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Same as v7 but uses v2's own per-horizon prediction column."""
    v2_npz = V2_DIR / f"{date_str}_predictions.npz"
    if not v2_npz.exists():
        return None
    v2 = np.load(v2_npz, allow_pickle=False)
    preds = v2["predictions"][:, pred_col].astype(np.float32)
    ws, st = int(v2["window_size"]), int(v2["stride"])

    mbo_npz = MBO_EVENT_DIR / f"{date_str}_mbo_events.npz"
    if not mbo_npz.exists():
        return None
    mbo = np.load(mbo_npz, allow_pickle=False)
    if label_key not in mbo.files:
        return None
    labels = mbo[label_key].astype(np.float32)
    n_events = len(labels)
    n_preds = len(preds)
    event_idx = np.minimum(np.arange(n_preds) * st + ws - 1, n_events - 1)
    realized = labels[event_idx]
    return preds, realized


# ---------------------------------------------------------------------------
# Per-date market-order PnL evaluation
# ---------------------------------------------------------------------------
def eval_date_market(preds: np.ndarray, realized: np.ndarray, pct: float) -> Optional[pd.DataFrame]:
    """Per-day top-pct by abs(pred), direction = sign(pred).
    Returns DataFrame with per-trade net PnL in ticks (signed, post-cost)."""
    # Drop NaN realized
    valid = np.isfinite(preds) & np.isfinite(realized)
    if valid.sum() == 0:
        return None
    p = preds[valid]
    r = realized[valid]
    n = p.size
    if n < 10:
        return None

    abs_p = np.abs(p)
    k = max(1, int(n * pct))
    if k >= n:
        idx = np.arange(n)
    else:
        thresh = np.partition(abs_p, -n + k)[-n + k] if False else np.partition(abs_p, n - k)[n - k]
        # Simpler & correct: sort descending, take top k
        idx = np.argpartition(abs_p, n - k)[n - k:]
    if idx.size == 0:
        return None

    direction_sign = np.sign(p[idx])  # +1 long, -1 short (per-event)
    gross_ticks = direction_sign * r[idx]  # signed realized
    net_ticks = gross_ticks - TOTAL_COST_TICKS

    return pd.DataFrame({
        "direction": np.where(direction_sign > 0, "long", "short"),
        "gross_ticks": gross_ticks.astype(np.float64),
        "net_ticks": net_ticks.astype(np.float64),
        "pred_strength": abs_p[idx].astype(np.float64),
    })


# ---------------------------------------------------------------------------
# Sweep runner
# ---------------------------------------------------------------------------
def run_model(model_name: str, label_key: str, v2_pred_col: Optional[int],
              dates: List[str]) -> pd.DataFrame:
    """Run all thresholds for one model on its dates."""
    log.info(f"\n=== Running {model_name} (label={label_key}) on {len(dates)} dates ===")

    is_v7 = (v2_pred_col is None)
    all_rows = []
    skipped = 0

    for ds in dates:
        try:
            if is_v7:
                pair = load_v7_signals_realized(ds, label_key)
            else:
                pair = load_v2_signals_realized(ds, v2_pred_col, label_key)
        except Exception as e:
            log.warning(f"  {ds}: load error: {e}")
            skipped += 1
            continue
        if pair is None:
            skipped += 1
            continue
        preds, realized = pair
        if preds.size == 0:
            skipped += 1
            continue

        # Run all thresholds for this date
        for pct in THRESHOLDS:
            df = eval_date_market(preds, realized, pct)
            if df is None or df.empty:
                continue
            df["date"] = ds
            df["cell_id"] = f"top{pct*100:g}pct"
            df["pct"] = pct
            all_rows.append(df)

    if skipped:
        log.info(f"  {model_name}: {skipped} dates skipped")
    if not all_rows:
        return pd.DataFrame()
    out = pd.concat(all_rows, ignore_index=True)
    out["model"] = model_name
    log.info(f"  {model_name}: {len(out):,} total trades across {out['date'].nunique()} dates x "
             f"{out['cell_id'].nunique()} cells")
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def load_regime() -> Optional[pd.DataFrame]:
    if not REGIME_PARQUET.exists():
        return None
    r = pd.read_parquet(REGIME_PARQUET)
    r["date"] = r["date"].astype(str).str.zfill(8)
    def cls(row):
        delta = row["close_minus_open_ticks"]
        if delta >= 4: return "green"
        if delta <= -4: return "red"
        return "flat"
    r["regime"] = r.apply(cls, axis=1)
    return r[["date", "regime", "close_minus_open_ticks"]]


def metrics_for(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"n": 0, "n_days": 0, "mean_tk": np.nan, "median_tk": np.nan,
                "sharpe": np.nan, "pf": np.nan, "wr": np.nan, "day_pos_pct": np.nan,
                "n_pos_days": 0, "n_neg_days": 0, "day_conc": np.nan}
    nets = df["net_ticks"].values.astype(np.float64)
    n = len(nets)
    daily = df.groupby("date")["net_ticks"].sum()
    n_days = daily.size
    sharpe = float(daily.mean() / daily.std(ddof=1) * np.sqrt(252)) if daily.std(ddof=1) > 0 and n_days > 1 else np.nan
    wins = nets[nets > 0].sum()
    losses = -nets[nets < 0].sum()
    pf = float(wins / losses) if losses > 0 else float("inf")
    abs_daily = daily.abs()
    day_conc = float(abs_daily.max() / abs_daily.sum()) if abs_daily.sum() > 0 else np.nan
    return {"n": n, "n_days": n_days,
            "mean_tk": float(nets.mean()),
            "median_tk": float(np.median(nets)),
            "sharpe": sharpe,
            "pf": pf,
            "wr": float((nets > 0).mean()),
            "day_pos_pct": float((daily > 0).mean()),
            "n_pos_days": int((daily > 0).sum()),
            "n_neg_days": int((daily < 0).sum()),
            "day_conc": day_conc}


def regime_check(df: pd.DataFrame, reg: Optional[pd.DataFrame]) -> dict:
    if reg is None or df.empty:
        return {"sharpe_green": np.nan, "sharpe_red": np.nan, "sharpe_flat": np.nan,
                "regime_skew": np.nan, "regime_pass": None}
    merged = df.merge(reg, on="date", how="left")
    out = {}
    for r in ["green", "red", "flat"]:
        sub = merged[merged["regime"] == r]
        if sub.empty:
            out[f"sharpe_{r}"] = np.nan
        else:
            daily = sub.groupby("date")["net_ticks"].sum()
            out[f"sharpe_{r}"] = float(daily.mean() / daily.std(ddof=1) * np.sqrt(252)) if daily.std(ddof=1) > 0 and len(daily) > 1 else np.nan
    sg, sr = out.get("sharpe_green", np.nan), out.get("sharpe_red", np.nan)
    if np.isfinite(sg) and np.isfinite(sr) and max(abs(sg), abs(sr)) > 0:
        out["regime_skew"] = abs(sg - sr) / max(abs(sg), abs(sr))
        out["regime_pass"] = bool(out["regime_skew"] <= 0.50)
    else:
        out["regime_skew"] = np.nan
        out["regime_pass"] = None
    return out


def verdict(m: dict, rc: dict) -> str:
    n = m["n"]
    if n < 100:
        return "REJECT (n<100)"
    if m["mean_tk"] <= 0:
        return f"REJECT (net {m['mean_tk']:+.3f}t<=0)"
    gates = []
    if rc.get("regime_pass") is False:
        gates.append(f"regime_skew {rc['regime_skew']:.2f}>0.50")
    if m["day_conc"] is not None and np.isfinite(m["day_conc"]) and m["day_conc"] > 0.70:
        gates.append(f"day_conc {m['day_conc']:.2f}>0.70")
    if m["n_pos_days"] < max(2, int(m["n_days"] * 0.4)):
        gates.append(f"only {m['n_pos_days']}/{m['n_days']} pos days")
    if m["n_days"] < 40:
        gates.append(f"n_days {m['n_days']}<40 gap")
    if not gates:
        return "PASS"
    if m["mean_tk"] > 0.2 and m["n_pos_days"] >= 5:
        return "WEAK PASS — " + "; ".join(gates)
    return "REJECT — " + "; ".join(gates)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import mlflow

    log.info("=" * 72)
    log.info("MARKET-ORDER TOP-TAIL SWEEP (CANONICAL FIFO-EQUIVALENT)")
    log.info("HC #493 R3 + HC #428 R1/R2")
    log.info("=" * 72)
    log.info(f"Cost stack: commission {ES_COMMISSION_TICKS:.3f}t + entry {ENTRY_SPREAD_COST:.1f}t "
             f"+ exit {EXIT_SPREAD_COST:.1f}t = {TOTAL_COST_TICKS:.3f}t RT total")

    v7_dates, v2_dates = discover_dates()
    log.info(f"v7 aligned: {len(v7_dates)} dates ({v7_dates[0]} -> {v7_dates[-1]})")
    log.info(f"v2 aligned: {len(v2_dates)} dates ({v2_dates[0]} -> {v2_dates[-1]})")

    # ── Verify-then-report ─────────────────────────────────────────────
    log.info("\n--- VERIFY: first 3 v7 predictions ---")
    dvv = np.load(V7_PRED_NPZ, allow_pickle=False)
    log.info(f"  v7 preds[:3] = {dvv['predictions'][:3].tolist()}")
    log.info(f"  v7 dates[:3] = {dvv['dates'][:3].tolist()}")
    log.info(f"  v7 total: {dvv['predictions'].size:,} preds, "
             f"{(dvv['predictions']!=0).sum():,} nonzero")

    log.info("\n--- VERIFY: first 3 v2 raw predictions (sample date) ---")
    sample = v2_dates[0] if v2_dates else None
    if sample:
        v2v = np.load(V2_DIR / f"{sample}_predictions.npz", allow_pickle=False)
        log.info(f"  {sample} preds[:3,:] = {v2v['predictions'][:3].tolist()}")
        log.info(f"  {sample} n_windows={int(v2v['n_windows'])}, nonzero per H="
                 f"{[(v2v['predictions'][:, h] != 0).sum() for h in range(3)]}")

    log.info("\n--- VERIFY: first 3 mbo labels (sample date) ---")
    if sample:
        ev = np.load(MBO_EVENT_DIR / f"{sample}_mbo_events.npz", allow_pickle=False)
        for k in ["labels_1s", "labels_5s", "labels_10s", "labels_30s"]:
            if k in ev.files:
                v = ev[k]
                finite = np.isfinite(v).sum()
                log.info(f"  {k}: shape={v.shape}, finite={finite:,}, "
                         f"first 3 finite = {v[np.isfinite(v)][:3].tolist()}")
            else:
                log.warning(f"  {k}: MISSING")

    # ── MLflow setup ───────────────────────────────────────────────────
    mlflow.set_tracking_uri("http://localhost:5000")
    try:
        exp_id = mlflow.create_experiment("market_order_toptail_fifo_sweep")
    except Exception:
        exp_id = mlflow.get_experiment_by_name("market_order_toptail_fifo_sweep").experiment_id

    reg = load_regime()

    # ── Run all models ─────────────────────────────────────────────────
    t0 = time_mod.time()
    all_fills: List[pd.DataFrame] = []
    for model_name, label_key, v2_col, hstr in HORIZON_DEFS:
        if model_name.startswith("v7"):
            dates_use = v7_dates
        else:
            dates_use = v2_dates
        df = run_model(model_name, label_key, v2_col, dates_use)
        if df.empty:
            log.warning(f"  {model_name}: NO fills")
            continue
        df.to_parquet(OUT_ROOT / f"fills_{model_name}.parquet")
        all_fills.append(df)
        log.info(f"  Saved fills_{model_name}.parquet ({len(df):,} rows)")

    if not all_fills:
        log.error("No fills produced. ABORT.")
        sys.exit(1)
    fills = pd.concat(all_fills, ignore_index=True)
    fills.to_parquet(OUT_ROOT / "fills_all.parquet")
    log.info(f"\nTotal: {len(fills):,} trades. Sweep computation took {time_mod.time()-t0:.1f}s")

    # ── Grade per (model, cell_id) ─────────────────────────────────────
    grid_rows = []
    for (model, cell_id), sub in fills.groupby(["model", "cell_id"]):
        m = metrics_for(sub)
        rc = regime_check(sub, reg)
        v = verdict(m, rc)
        row = {
            "model": model, "cell_id": cell_id,
            "n_trades": m["n"], "n_days": m["n_days"],
            "mean_net_tk": m["mean_tk"], "median_net_tk": m["median_tk"],
            "sharpe_ann": m["sharpe"],
            "wr": m["wr"], "pf": m["pf"],
            "n_pos_days": m["n_pos_days"], "n_neg_days": m["n_neg_days"],
            "day_pos_pct": m["day_pos_pct"], "day_conc": m["day_conc"],
            "sharpe_green": rc["sharpe_green"], "sharpe_red": rc["sharpe_red"],
            "sharpe_flat": rc.get("sharpe_flat", np.nan),
            "regime_skew": rc["regime_skew"], "regime_pass": rc["regime_pass"],
            "verdict": v,
        }
        grid_rows.append(row)

        try:
            with mlflow.start_run(experiment_id=exp_id, run_name=f"{model}__{cell_id}"):
                mlflow.log_params({"model": model, "cell_id": cell_id,
                                   "order_type": "market", "n_days_oot": m["n_days"],
                                   "total_cost_ticks": TOTAL_COST_TICKS})
                for k, val in m.items():
                    if isinstance(val, (int, float)) and np.isfinite(val):
                        mlflow.log_metric(k, float(val))
                for k, val in rc.items():
                    if isinstance(val, (int, float)) and np.isfinite(val):
                        mlflow.log_metric(k, float(val))
                mlflow.set_tag("verdict", v)
        except Exception as e:
            log.warning(f"MLflow log failed: {e}")

    grid = pd.DataFrame(grid_rows).sort_values(["model", "cell_id"])
    grid.to_csv(OUT_ROOT / "sweep_grid.csv", index=False)
    log.info(f"\nWrote sweep_grid.csv: {len(grid)} cells")
    log.info("\n" + grid[["model", "cell_id", "n_trades", "n_days", "mean_net_tk",
                          "sharpe_ann", "wr", "pf", "verdict"]].to_string(index=False))

    # ── Report ─────────────────────────────────────────────────────────
    write_report(grid, fills, v7_dates, v2_dates)
    log.info(f"Wrote {REPORT_PATH}")
    log.info("DONE.")


def write_report(grid: pd.DataFrame, fills: pd.DataFrame,
                 v7_dates: List[str], v2_dates: List[str]) -> None:
    lines = []
    lines.append("# Market-Order Top-Tail FIFO Sweep — REPORT")
    lines.append("")
    lines.append("**HC #493 R3 + HC #428 R1/R2 binding.**")
    lines.append(f"**Generated**: {pd.Timestamp.now().isoformat()}")
    lines.append("")
    lines.append("## Hypothesis tested")
    lines.append("Post-3-regrade synthesis: every proxy->FIFO collapse is -0.8 to -1.5 ticks, "
                 "driven by limit-order queue wait + adverse fill selection + commission. Test "
                 "whether MARKET orders (no queue, deterministic touch fill) at extreme top-tail "
                 "confidence + longer holds escape that failure mode.")
    lines.append("")
    lines.append("## Setup")
    lines.append("- **Canonical FIFO equivalence (key insight)**: for pure market orders with "
                 "no intra-horizon TP/SL, the FIFO replay PnL collapses to "
                 "`direction * labels_h - cost`, because market orders fill at touch "
                 "(deterministic, no queue) and labels_h is the canonical mid-to-mid realized "
                 "move at horizon h in TICKS, sampled from the same `mbo_events_smart_v3` "
                 "files the FIFO harness uses. No queue mechanics or fill-selection adverse "
                 "effects can intervene.")
    lines.append(f"- **Cost stack**: commission {ES_COMMISSION_TICKS:.3f}t + entry spread "
                 f"{ENTRY_SPREAD_COST:.1f}t + exit spread {EXIT_SPREAD_COST:.1f}t = "
                 f"**{TOTAL_COST_TICKS:.3f}t round-trip** (per task spec).")
    lines.append(f"- **Thresholds (per-day top abs(pred))**: {THRESHOLDS}")
    lines.append("- **Horizons / models**:")
    for mn, lk, col, hs in HORIZON_DEFS:
        src = "v7 meta signed pred" if mn.startswith("v7") else f"v2 raw pred col {col} ({hs})"
        lines.append(f"  * `{mn}`: signal = {src}; realized = `{lk}`; hold = {hs}")
    lines.append("")
    lines.append("## OOT coverage")
    lines.append(f"- **v7_meta**: {len(v7_dates)} aligned dates "
                 f"({v7_dates[0]} -> {v7_dates[-1]}). "
                 f"**Gap vs HC #428 R1 40-day requirement: {max(0, 40 - len(v7_dates))} days short**.")
    lines.append(f"- **v2_raw**: {len(v2_dates)} aligned dates "
                 f"({v2_dates[0]} -> {v2_dates[-1]}). "
                 f"**Gap vs HC #428 R1: {max(0, 40 - len(v2_dates))} days short**.")
    lines.append("")
    lines.append("## Full sweep grid")
    lines.append("")
    show_cols = ["model", "cell_id", "n_trades", "n_days", "mean_net_tk",
                 "median_net_tk", "sharpe_ann", "wr", "pf", "n_pos_days",
                 "n_neg_days", "day_conc", "regime_skew", "verdict"]
    show = grid[show_cols].copy()
    for c in ["mean_net_tk", "median_net_tk", "sharpe_ann", "wr", "pf",
              "day_conc", "regime_skew"]:
        show[c] = show[c].apply(lambda v: f"{v:+.3f}" if pd.notnull(v) else "nan")
    lines.append(show.to_markdown(index=False))
    lines.append("")

    survivors = grid[grid["verdict"].str.startswith("PASS")]
    weak = grid[grid["verdict"].str.startswith("WEAK")]
    lines.append("## Verdict summary")
    lines.append(f"- **PASS cells**: {len(survivors)}")
    lines.append(f"- **WEAK PASS cells**: {len(weak)}")
    lines.append(f"- **REJECT cells**: {len(grid) - len(survivors) - len(weak)}")
    lines.append("")
    if not survivors.empty:
        lines.append("### Survivors (PASS)")
        for _, row in survivors.iterrows():
            lines.append(f"- **{row['model']} / {row['cell_id']}**: "
                         f"net {row['mean_net_tk']:+.3f}t, sharpe {row['sharpe_ann']:.2f}, "
                         f"{row['n_pos_days']}/{row['n_days']} pos days, n={row['n_trades']}")
    else:
        lines.append("### No PASS cells.")
    if not weak.empty:
        lines.append("\n### Weak passes")
        for _, row in weak.iterrows():
            lines.append(f"- **{row['model']} / {row['cell_id']}**: {row['verdict']}")
    lines.append("")
    lines.append("## Best-cell-per-model (informational, may still be a REJECT)")
    best_rows = grid.sort_values("mean_net_tk", ascending=False).groupby("model").head(1)
    for _, row in best_rows.iterrows():
        lines.append(f"- **{row['model']}**: best cell `{row['cell_id']}` -> "
                     f"net {row['mean_net_tk']:+.3f}t/trade, sharpe {row['sharpe_ann']:.2f}, "
                     f"{row['n_pos_days']}/{row['n_days']} pos days, "
                     f"verdict: {row['verdict']}")
    lines.append("")
    lines.append("## Recommendation")
    if survivors.empty and weak.empty:
        lines.append("Pure market-order execution at top-tail confidence on existing predictions "
                     "does NOT carve out edge under the canonical realized-move accounting with "
                     f"{TOTAL_COST_TICKS:.3f}t round-trip cost. Same structural failure as the "
                     "three regrade rejects: signal magnitude is below the cost stack. Next axes "
                     "to test (NOT in this sweep — would require new work): (a) hybrid execution "
                     "(passive entry + market exit on adverse move), (b) ensemble confluence to "
                     "lift top-tail signal magnitude above 2.4t, or (c) abandon ES intraday "
                     "execution at sub-30s horizons given the structural cost floor.")
    else:
        lines.append("At least one cell survived the HC #428 R1 gate. Tree-branch this cell per "
                     "HC #491 R1 (intra-horizon TP/SL, time-of-day, sizing, regime sub-gate) and "
                     "re-grade via FIFO harness before any production claim.")
    REPORT_PATH.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
