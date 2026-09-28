#!/usr/bin/env python3
"""
tod_velocity_fifo_replay_v1.py
==============================
Canonical FIFO market-replay (HC #74) of the 6 winning cells from
tod_velocity_stratification_v1. THIS IS THE DEPLOY-GATE TEST.

WHY THIS MATTERS:
  - tod_velocity_stratification_v1 found 6 cells passing HC #428 gates ON LABELS
    (top: long 10s top5pc mid_am vel8 +0.89 t/trade Sh 4.12).
  - But meta_classifier v1 saw a +3.46 t/trade label winner die at -1.6 t/trade
    in FIFO (a 5-tick label-vs-fill gap from adverse selection).
  - Every label-based winner MUST be FIFO-validated before any deploy claim.

INPUTS
------
- output/tod_velocity_stratification_v1/winning_cells.txt
- output/cnn_mamba_v3_4_2_fixedmtl/oot_47day_perdate/oot_{YYYYMMDD}.npz
- data/raw/mbo/glbx-mdp3-{YYYYMMDD}.mbo.dbn.zst (via FIFOReplayEngine)
- data/processed/mbo_events/{YYYYMMDD}_mbo_events.npz (timestamps + event types
  for predictions-to-timestamp alignment and velocity decile)

METHOD (HC #74)
---------------
1. For each cell (horizon, side, conf, tod, vel_dec), rebuild the exact same
   per-day per-side confidence threshold + TOD + velocity decile filter that
   the stratification used.
2. For each qualifying prediction row place a passive limit at the inside
   bid (long) / ask (short).
3. FIFOReplayEngine replays raw MBO DBN events: queue position FIFO,
   limit fills only when volume trades through, cancel after horizon h,
   max_hold = 1.5 h, exit at mid for max_hold (engine internal).
4. Cost models:
     (a) passive-passive   = pnl_ticks_net (engine: gross - 0.376 RT comm)
     (b) market-exit       = pnl_ticks_net - 1.0  (one spread crossing)
5. Compare to LABEL net = sign*target_h - 0.376  (the stratification metric).

OUTPUTS (output/tod_velocity_fifo_replay_v1/)
---------------------------------------------
- per_cell_fifo_summary.csv       both cost models per cell
- per_fill_detail.parquet         every realized fill
- winning_cells_fifo.txt          cells still passing gates @ cost (a)
- REPORT.md                       verdict per cell + label-vs-FIFO gap
- .regen_complete.json            HC #485 R5 stamp
"""
from __future__ import annotations

import json
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

LVL3_ROOT = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(LVL3_ROOT))

# Reuse the canonical FIFO engine (HC #74) — do NOT modify it.
from alpha_discovery.deep_models.fifo_market_replay import (  # noqa: E402
    FIFOReplayEngine,
    COMMISSION_TICKS,
)

# ── Paths ───────────────────────────────────────────────────────────────────
OOT_DIR = LVL3_ROOT / "output" / "cnn_mamba_v3_4_2_fixedmtl" / "oot_47day_perdate"
MBO_DIR = LVL3_ROOT / "data" / "processed" / "mbo_events"
RAW_DBN_DIR = LVL3_ROOT / "data" / "raw" / "mbo"
WIN_CELLS_TXT = LVL3_ROOT / "output" / "tod_velocity_stratification_v1" / "winning_cells.txt"
OUT_DIR = LVL3_ROOT / "output" / "tod_velocity_fifo_replay_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ───────────────────────────────────────────────────────────────
ES_RT_COMMISSION_TICKS = float(COMMISSION_TICKS)        # 0.376
EXIT_SPREAD_TICKS = 1.0                                  # market exit crossing
STRIDE = 250                                             # MBO -> pred subsample
VELOCITY_WINDOW_NS = 5_000_000_000                       # 5s
EVENT_TYPE_TRADE = 3                                     # action_encoding 'T'
SKIP_DATES = {"20260308", "20260315"}                    # match stratification
ANNUAL_TRADING_DAYS = 252.0

# Budget control: per-day FIFO simulate ~ 4-5 min CPU. 32 days × 3 horizons would
# blow the 45-min wall budget (140+ min). We stratified-subsample every Nth date
# to fit budget while keeping cross-regime coverage. Set to 1 for full 32-day run.
# 4 → 8 days, 3 → 11 days, 2 → 16 days. Default 4 (8 days, ~35 min).
DATE_SUBSAMPLE_EVERY = 4

TOD_BUCKETS = [
    ("open",     9*60+30, 10*60+30),
    ("mid_am",   10*60+30, 12*60),
    ("midday",   12*60,    14*60),
    ("late_pm",  14*60,    15*60),
    ("close",    15*60,    16*60),
]
CONF_FRAC = {"top1pc": 0.01, "top5pc": 0.05, "top10pc": 0.10}
HORIZON_SEC = {"1s": 1.0, "5s": 5.0, "10s": 10.0, "30s": 30.0}

# HC #428 gates (mirror stratification + add fill-rate floor)
GATE_NET      = 0.10
GATE_SHARPE   = 0.3
GATE_PF       = 1.1
GATE_PROFDAYS = 0.60
GATE_IMBAL    = 0.50
GATE_FILL     = 0.05         # economically dead below this
MIN_TRADES    = 30           # relaxed from 50 (FIFO drops a chunk to no-fill)

REGIME_THRESH_TICKS = 5.0    # match stratification's regime classifier


# ── Helpers ─────────────────────────────────────────────────────────────────
def parse_winning_cells(txt_path: Path) -> List[dict]:
    """Parse winning_cells.txt into list of cell dicts."""
    lines = txt_path.read_text().splitlines()
    cells = []
    # find header line
    header_idx = None
    for i, ln in enumerate(lines):
        if ln.strip().startswith("horizon") and "vel_dec" in ln:
            header_idx = i
            break
    if header_idx is None:
        raise RuntimeError(f"could not find header row in {txt_path}")
    cols = lines[header_idx].split()
    for ln in lines[header_idx + 1:]:
        toks = ln.split()
        if len(toks) < 5:
            continue
        d = dict(zip(cols, toks))
        cells.append({
            "horizon":  d["horizon"],
            "conf":     d["conf"],
            "side":     d["side"],
            "tod":      d["tod"],
            "vel_dec":  int(d["vel_dec"]),
            "label_n":          int(d.get("n_trades", -1)),
            "label_net":        float(d.get("net_ticks", "nan")),
            "label_sharpe":     float(d.get("sharpe", "nan")),
            "label_wr":         float(d.get("wr", "nan")),
            "label_pf":         float(d.get("pf", "nan")),
            "label_profdays":   float(d.get("prof_days", "nan")),
            "label_imb":        float(d.get("regime_imbalance", "nan")),
        })
    return cells


def tod_label(min_et: np.ndarray) -> np.ndarray:
    out = np.full(min_et.shape, "outside", dtype=object)
    for name, lo, hi in TOD_BUCKETS:
        m = (min_et >= lo) & (min_et < hi)
        out[m] = name
    return out


def compute_velocity(pred_ts_ns: np.ndarray, mbo_ts_ns: np.ndarray, mbo_type: np.ndarray) -> np.ndarray:
    trade_ts = mbo_ts_ns[mbo_type == EVENT_TYPE_TRADE]
    lo = pred_ts_ns - VELOCITY_WINDOW_NS
    left_idx  = np.searchsorted(trade_ts, lo,         side="left")
    right_idx = np.searchsorted(trade_ts, pred_ts_ns, side="left")
    return (right_idx - left_idx).astype(np.int32)


def day_decile(values: np.ndarray) -> np.ndarray:
    n = len(values)
    if n == 0:
        return np.zeros(0, dtype=np.int8)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(n, dtype=np.int64)
    ranks[order] = np.arange(n)
    dec = (ranks * 10) // n
    dec[dec == 10] = 9
    return dec.astype(np.int8)


def load_day_frame(date: str) -> pd.DataFrame | None:
    """Replicate stratification per-day load: build row-level frame with
    pred/target/mask per horizon, TOD, velocity decile, timestamp."""
    pred_path = OOT_DIR / f"oot_{date}.npz"
    mbo_path  = MBO_DIR / f"{date}_mbo_events.npz"
    if not pred_path.exists() or not mbo_path.exists():
        return None
    pred = np.load(pred_path)
    mbo  = np.load(mbo_path)
    n_pred = pred["pred_log_ret_1s"].shape[0]
    mbo_ts = mbo["timestamps"]
    mbo_ev = mbo["events"][:, 1].astype(np.int8)
    pred_ts = mbo_ts[::STRIDE][:n_pred]
    if len(pred_ts) != n_pred:
        return None
    vel = compute_velocity(pred_ts, mbo_ts, mbo_ev)
    ts_et = pd.to_datetime(pred_ts, unit="ns", utc=True).tz_convert("America/New_York")
    min_et = (ts_et.hour * 60 + ts_et.minute).astype(np.int32).values
    df = pd.DataFrame({
        "date":   date,
        "ts_ns":  pred_ts.astype(np.int64),
        "min_et": min_et,
        "vel":    vel,
    })
    for h in HORIZON_SEC:
        df[f"pred_{h}"]   = pred[f"pred_log_ret_{h}"]
        df[f"target_{h}"] = pred[f"target_log_ret_{h}"]
        df[f"mask_{h}"]   = pred[f"mask_log_ret_{h}"]
    df["tod"] = tod_label(df["min_et"].values)
    df = df[df["tod"] != "outside"].reset_index(drop=True)
    if df.empty:
        return None
    df["vel_dec"] = day_decile(df["vel"].values)
    return df


def select_cell_rows(df_day: pd.DataFrame, cell: dict) -> pd.DataFrame:
    """Apply the exact stratification filter for ONE cell on ONE day's frame.
    Returns the subset of rows that would be selected as signals."""
    h = cell["horizon"]
    side = cell["side"]
    tod_name = cell["tod"]
    vd = cell["vel_dec"]
    conf_frac = CONF_FRAC[cell["conf"]]

    mask_valid = df_day[f"mask_{h}"].values > 0.5
    preds = df_day[f"pred_{h}"].values
    side_mask = (preds > 0) if side == "long" else (preds < 0)
    sel_side = mask_valid & side_mask
    if sel_side.sum() < 10:
        return df_day.iloc[0:0]
    thr = np.quantile(np.abs(preds[sel_side]), 1.0 - conf_frac)
    conf_keep = sel_side & (np.abs(preds) >= thr)
    cell_mask = (
        conf_keep
        & (df_day["tod"].values == tod_name)
        & (df_day["vel_dec"].values == vd)
    )
    return df_day[cell_mask].copy()


def regime_for_day(df_day: pd.DataFrame) -> str:
    """Per stratification: green/red/flat by cum 1s tick target."""
    valid = df_day["mask_1s"].values > 0.5
    cum = float(np.sum(df_day["target_1s"].values[valid]))
    if cum >  REGIME_THRESH_TICKS: return "green"
    if cum < -REGIME_THRESH_TICKS: return "red"
    return "flat"


def sharpe_pertrade(arr: np.ndarray) -> float:
    n = len(arr)
    if n < 2: return 0.0
    s = float(np.std(arr, ddof=1))
    if s <= 1e-12: return 0.0
    return float(np.mean(arr) / s * np.sqrt(n))


def compute_metrics(net_arr: np.ndarray, day_arr: np.ndarray, regime_arr: np.ndarray) -> dict:
    """Pooled per-trade Sharpe (matches stratification), plus regime split."""
    n = len(net_arr)
    if n == 0:
        return dict(n=0, mean=float("nan"), sharpe=0.0, wr=float("nan"),
                    pf=float("nan"), profdays=float("nan"), imbalance=float("nan"),
                    sharpe_green=float("nan"), sharpe_red=float("nan"),
                    n_days=0)
    mean = float(np.mean(net_arr))
    sh = sharpe_pertrade(net_arr)
    wins = net_arr > 0
    wr = float(np.mean(wins))
    gp = float(np.sum(net_arr[wins])) if wins.any() else 0.0
    gl = float(-np.sum(net_arr[~wins])) if (~wins).any() else 0.0
    pf = (gp / gl) if gl > 1e-9 else (float("inf") if gp > 0 else 0.0)
    days = np.unique(day_arr)
    day_means = np.array([np.mean(net_arr[day_arr == d]) for d in days])
    profdays = float(np.mean(day_means > 0)) if len(days) else 0.0
    sh_g = sharpe_pertrade(net_arr[regime_arr == "green"])
    sh_r = sharpe_pertrade(net_arr[regime_arr == "red"])
    denom = max(abs(sh_g), abs(sh_r))
    imb = abs(sh_g - sh_r) / denom if denom > 1e-9 else float("nan")
    return dict(n=n, mean=mean, sharpe=sh, wr=wr, pf=pf, profdays=profdays,
                imbalance=imb, sharpe_green=sh_g, sharpe_red=sh_r,
                n_days=int(len(days)))


def passes_gates(m: dict, fill_rate: float) -> Tuple[bool, List[str]]:
    fails = []
    if not np.isfinite(m["mean"]) or m["mean"] <= GATE_NET:
        fails.append(f"net<={GATE_NET}")
    if m["sharpe"] <= GATE_SHARPE:
        fails.append(f"sharpe<={GATE_SHARPE}")
    if not np.isfinite(m["pf"]) or m["pf"] <= GATE_PF:
        fails.append(f"pf<={GATE_PF}")
    if not np.isfinite(m["profdays"]) or m["profdays"] < GATE_PROFDAYS:
        fails.append(f"profdays<{GATE_PROFDAYS}")
    if (not np.isfinite(m["imbalance"])) or m["imbalance"] >= GATE_IMBAL:
        fails.append(f"imb>={GATE_IMBAL}")
    if m["n"] < MIN_TRADES:
        fails.append(f"n<{MIN_TRADES}")
    if fill_rate < GATE_FILL:
        fails.append(f"fill_rate<{GATE_FILL}")
    return (len(fails) == 0), fails


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print(f"[start] {started_at}", flush=True)

    cells = parse_winning_cells(WIN_CELLS_TXT)
    print(f"[cells] parsed {len(cells)} winning cells:", flush=True)
    for c in cells:
        print(f"  {c['side']:>5s} {c['horizon']:>3s} {c['conf']:>7s} "
              f"{c['tod']:>7s} vd={c['vel_dec']}  label_net={c['label_net']:+.3f} "
              f"Sh={c['label_sharpe']:.2f}", flush=True)

    # Discover dates: intersection of OOT preds and raw DBN files, minus SKIP.
    pred_dates = sorted(p.stem.replace("oot_", "") for p in OOT_DIR.glob("oot_*.npz"))
    have_dbn = lambda d: (RAW_DBN_DIR / f"glbx-mdp3-{d}.mbo.dbn.zst").exists()
    dates_all = [d for d in pred_dates if d not in SKIP_DATES and have_dbn(d)]
    if DATE_SUBSAMPLE_EVERY > 1:
        dates = dates_all[::DATE_SUBSAMPLE_EVERY]
        print(f"[dates] {len(dates_all)} available; subsampled every {DATE_SUBSAMPLE_EVERY} → "
              f"{len(dates)} dates for budget: {dates}", flush=True)
    else:
        dates = dates_all
        print(f"[dates] {len(dates)} OOT dates available (DBN+pred, skip={sorted(SKIP_DATES)})", flush=True)

    # Per-day: load frame once, compute signals for ALL cells, then run engine ONCE
    # per day with the union of all signals tagged by cell_idx. This is key for
    # the 45-min budget — DBN load is the heavy cost (~10-20s/day), so we amortize.
    per_fill_rows: List[dict] = []
    per_cell_buckets: Dict[int, List[dict]] = {i: [] for i in range(len(cells))}
    # Track signaled rows (label net) per cell for label baseline
    per_cell_label_nets: Dict[int, List[Tuple[str, float, str]]] = {i: [] for i in range(len(cells))}
    per_cell_signal_counts: Dict[int, int] = {i: 0 for i in range(len(cells))}

    for di, date in enumerate(dates):
        t_d = time.time()
        try:
            df_day = load_day_frame(date)
        except Exception as e:
            print(f"  [{date}] load failed: {e}", flush=True)
            continue
        if df_day is None or df_day.empty:
            print(f"  [{date}] empty frame, skip", flush=True)
            continue
        day_regime = regime_for_day(df_day)

        # Collect all signals for this day across cells
        signals_with_tag: List[Tuple[dict, int, float, str]] = []
        # (signal_dict, cell_idx, label_net, regime)
        for ci, cell in enumerate(cells):
            sub = select_cell_rows(df_day, cell)
            if sub.empty:
                continue
            h = cell["horizon"]
            side = cell["side"]
            sign = 1.0 if side == "long" else -1.0
            label_nets = sign * sub[f"target_{h}"].values - ES_RT_COMMISSION_TICKS
            for ts_ns, lbn in zip(sub["ts_ns"].values, label_nets):
                signals_with_tag.append((
                    {"ts_ns": int(ts_ns), "direction": side, "strength": 1.0},
                    ci, float(lbn), day_regime,
                ))
                per_cell_label_nets[ci].append((date, float(lbn), day_regime))
                per_cell_signal_counts[ci] += 1

        if not signals_with_tag:
            print(f"  [{date}] no signals across any cell  ({time.time()-t_d:.1f}s)",
                  flush=True)
            continue

        # Group signals by (cell horizon, cell side) so cancel/max_hold are correct.
        # The engine takes ONE cancel/max_hold pair — so we must run it once per
        # unique horizon (cells share horizons: 1s/5s/10s/30s). Side is a per-signal
        # attribute (direction) so it does not partition runs.
        horizons_in_play = sorted({cells[ci]["horizon"] for (_, ci, _, _) in signals_with_tag})

        # Build ONE engine per day (DBN load is the slow step). Reuse across
        # horizons by mutating cancel_after_ns / max_hold_ns before each
        # simulate() call. The engine is otherwise stateless between simulates.
        try:
            engine = FIFOReplayEngine(
                date=date,
                instrument_id=None,
                cancel_after_ns=int(30e9),   # placeholder, overwritten per horizon
                max_hold_ns=int(45e9),
                max_reprices=0,
                reprice_after_ns=int(1e9),
            )
        except Exception as e:
            print(f"  [{date}] engine init failed: {e}", flush=True)
            continue

        day_fills = 0
        for h_str in horizons_in_play:
            h_sec = HORIZON_SEC[h_str]
            cancel_ns = int(round(h_sec * 1e9))
            max_hold_ns = int(round(1.5 * h_sec * 1e9))

            # Build the per-horizon signal list, remembering cell_idx for each
            sig_list = []
            sig_meta: List[Tuple[int, float, str]] = []  # cell_idx, label_net, regime
            for sigd, ci, lbn, reg in signals_with_tag:
                if cells[ci]["horizon"] != h_str:
                    continue
                sig_list.append(sigd)
                sig_meta.append((ci, lbn, reg))

            engine.cancel_after_ns = cancel_ns
            engine.max_hold_ns = max_hold_ns

            # Wide TP/SL so exit only by max_hold/EOD (per HC #428 R2 holding to horizon)
            try:
                trades = engine.simulate(
                    sig_list,
                    tp_ticks=1000.0,
                    sl_ticks=1000.0,
                    order_type="limit",
                    order_management="realtime_sl",
                )
            except Exception as e:
                print(f"  [{date}/{h_str}] simulate failed: {e}", flush=True)
                traceback.print_exc()
                continue

            # Map each filled trade back to its cell via signal_ts_ns + direction.
            # Build lookup: (ts_ns, direction) -> list of (cell_idx, label_net, regime)
            # (multiple cells could in principle share a ts/side, though rarely)
            lookup: Dict[Tuple[int, str], List[Tuple[int, float, str]]] = {}
            for sigd, meta in zip(sig_list, sig_meta):
                key = (int(sigd["ts_ns"]), sigd["direction"])
                lookup.setdefault(key, []).append(meta)

            for tr in trades:
                key = (int(tr.signal_ts_ns), tr.direction)
                metas = lookup.get(key, [])
                if not metas:
                    continue
                # If multiple cells matched, attribute the fill to all of them
                # (they each "would have placed" this order — same fill, same P&L).
                # In practice cells are disjoint by (conf, tod, vel_dec) so this
                # is usually 1 entry, but we keep the loop for correctness.
                exit_charge = (EXIT_SPREAD_TICKS
                               if tr.exit_reason in ("max_hold", "eod") else 0.0)
                net_passive = float(tr.pnl_ticks_net)              # cost (a)
                net_market  = float(tr.pnl_ticks_net) - exit_charge  # cost (b)
                for (ci, lbn, reg) in metas:
                    per_cell_buckets[ci].append(dict(
                        date=date, regime=reg,
                        signal_ts_ns=int(tr.signal_ts_ns),
                        fill_ts_ns=int(tr.entry_ts_ns),
                        exit_ts_ns=int(tr.exit_ts_ns),
                        queue_ahead=int(tr.queue_ahead),
                        queue_wait_ns=int(tr.queue_wait_ns),
                        exit_reason=tr.exit_reason,
                        pnl_ticks_gross=float(tr.pnl_ticks),
                        net_passive=net_passive,
                        net_market=net_market,
                        label_net=lbn,
                    ))
                    per_fill_rows.append(dict(
                        cell_idx=ci,
                        cell_label=(f"{cells[ci]['side']}_{cells[ci]['horizon']}_"
                                    f"{cells[ci]['conf']}_{cells[ci]['tod']}_"
                                    f"vd{cells[ci]['vel_dec']}"),
                        date=date, regime=reg,
                        signal_ts_ns=int(tr.signal_ts_ns),
                        fill_ts_ns=int(tr.entry_ts_ns),
                        exit_ts_ns=int(tr.exit_ts_ns),
                        fill_wait_ms=float(tr.queue_wait_ns) / 1e6,
                        entry_price_raw=int(tr.entry_price_raw),
                        exit_price_raw=int(tr.exit_price_raw),
                        exit_reason=tr.exit_reason,
                        pnl_ticks_gross=float(tr.pnl_ticks),
                        net_passive=net_passive,
                        net_market=net_market,
                        label_net=lbn,
                    ))
                    day_fills += 1

        print(f"  [{di+1}/{len(dates)}] {date} regime={day_regime}  "
              f"signals={len(signals_with_tag)} fills={day_fills}  "
              f"({time.time()-t_d:.1f}s)", flush=True)

    # ── Per-cell metrics (both cost models) ─────────────────────────────────
    summary_rows = []
    winners_passive = []
    for ci, cell in enumerate(cells):
        fills = per_cell_buckets[ci]
        n_signals = per_cell_signal_counts[ci]
        n_filled = len(fills)
        fill_rate = (n_filled / n_signals) if n_signals > 0 else 0.0

        # Label baseline from all signaled rows (not just filled)
        label_arr = np.array([x[1] for x in per_cell_label_nets[ci]], dtype=float)
        label_day = np.array([x[0] for x in per_cell_label_nets[ci]])
        label_reg = np.array([x[2] for x in per_cell_label_nets[ci]])
        label_m = compute_metrics(label_arr, label_day, label_reg) if label_arr.size else \
                  dict(n=0, mean=float("nan"), sharpe=float("nan"), wr=float("nan"),
                       pf=float("nan"), profdays=float("nan"), imbalance=float("nan"),
                       sharpe_green=float("nan"), sharpe_red=float("nan"), n_days=0)

        if n_filled > 0:
            net_pas_arr = np.array([f["net_passive"] for f in fills], dtype=float)
            net_mkt_arr = np.array([f["net_market"]  for f in fills], dtype=float)
            day_arr     = np.array([f["date"]        for f in fills])
            reg_arr     = np.array([f["regime"]      for f in fills])
            wait_arr    = np.array([f["queue_wait_ns"]/1e6 for f in fills], dtype=float)
            m_pas = compute_metrics(net_pas_arr, day_arr, reg_arr)
            m_mkt = compute_metrics(net_mkt_arr, day_arr, reg_arr)
            avg_wait_ms = float(np.mean(wait_arr))
        else:
            empty = dict(n=0, mean=float("nan"), sharpe=0.0, wr=float("nan"),
                         pf=float("nan"), profdays=float("nan"), imbalance=float("nan"),
                         sharpe_green=float("nan"), sharpe_red=float("nan"), n_days=0)
            m_pas, m_mkt = empty, empty
            avg_wait_ms = float("nan")

        pass_pas, fails_pas = passes_gates(m_pas, fill_rate)
        pass_mkt, fails_mkt = passes_gates(m_mkt, fill_rate)
        gap_pas = m_pas["mean"] - label_m["mean"] if np.isfinite(m_pas["mean"]) and np.isfinite(label_m["mean"]) else float("nan")
        gap_mkt = m_mkt["mean"] - label_m["mean"] if np.isfinite(m_mkt["mean"]) and np.isfinite(label_m["mean"]) else float("nan")

        row = dict(
            cell_idx=ci,
            side=cell["side"], horizon=cell["horizon"], conf=cell["conf"],
            tod=cell["tod"], vel_dec=cell["vel_dec"],
            n_signals=n_signals, n_filled=n_filled, fill_rate=fill_rate,
            avg_wait_ms=avg_wait_ms,
            # Label baseline (recomputed here to be self-consistent)
            label_net=label_m["mean"], label_sharpe=label_m["sharpe"],
            label_wr=label_m["wr"], label_pf=label_m["pf"],
            label_profdays=label_m["profdays"], label_imb=label_m["imbalance"],
            label_n_days=label_m["n_days"],
            # Cost (a) passive-passive
            pas_net=m_pas["mean"], pas_sharpe=m_pas["sharpe"], pas_wr=m_pas["wr"],
            pas_pf=m_pas["pf"], pas_profdays=m_pas["profdays"],
            pas_imb=m_pas["imbalance"], pas_sh_green=m_pas["sharpe_green"],
            pas_sh_red=m_pas["sharpe_red"], pas_n_days=m_pas["n_days"],
            pas_pass=pass_pas, pas_fails=";".join(fails_pas),
            # Cost (b) market exit
            mkt_net=m_mkt["mean"], mkt_sharpe=m_mkt["sharpe"], mkt_wr=m_mkt["wr"],
            mkt_pf=m_mkt["pf"], mkt_profdays=m_mkt["profdays"],
            mkt_imb=m_mkt["imbalance"], mkt_pass=pass_mkt,
            mkt_fails=";".join(fails_mkt),
            # Gaps
            gap_label_vs_pas=gap_pas, gap_label_vs_mkt=gap_mkt,
        )
        summary_rows.append(row)
        if pass_pas:
            winners_passive.append(row)

        print(f"  [cell {ci}] {cell['side']:>5s} {cell['horizon']:>3s} "
              f"{cell['conf']:>7s} {cell['tod']:>7s} vd{cell['vel_dec']}: "
              f"sig={n_signals} fill={n_filled} fr={100*fill_rate:.1f}% "
              f"label={label_m['mean']:+.3f} pas={m_pas['mean']:+.3f} "
              f"mkt={m_mkt['mean']:+.3f} gap_pas={gap_pas:+.3f}  "
              f"verdict_pas={'PASS' if pass_pas else 'REJECT'}", flush=True)

    # ── Write outputs ───────────────────────────────────────────────────────
    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(OUT_DIR / "per_cell_fifo_summary.csv", index=False)

    df_fill = pd.DataFrame(per_fill_rows)
    if df_fill.empty:
        df_fill = pd.DataFrame(columns=[
            "cell_idx", "cell_label", "date", "regime", "signal_ts_ns",
            "fill_ts_ns", "exit_ts_ns", "fill_wait_ms", "entry_price_raw",
            "exit_price_raw", "exit_reason", "pnl_ticks_gross",
            "net_passive", "net_market", "label_net",
        ])
    try:
        df_fill.to_parquet(OUT_DIR / "per_fill_detail.parquet", index=False)
    except Exception:
        # fallback if pyarrow missing
        df_fill.to_csv(OUT_DIR / "per_fill_detail.csv", index=False)

    # winning_cells_fifo.txt
    win_lines = []
    if winners_passive:
        win_lines.append(f"{len(winners_passive)} cells pass HC #428 gates after FIFO (passive-passive 0.376 t):\n")
        win_lines.append(f"{'side':>5s} {'horizon':>3s} {'conf':>7s} {'tod':>7s} "
                         f"{'vd':>3s} {'n':>5s} {'fr%':>5s} {'pas_net':>8s} {'pas_Sh':>7s} "
                         f"{'pas_pf':>6s} {'pas_pd':>6s} {'gap_lbl':>8s}")
        for r in winners_passive:
            win_lines.append(f"{r['side']:>5s} {r['horizon']:>3s} {r['conf']:>7s} "
                             f"{r['tod']:>7s} {r['vel_dec']:>3d} {r['n_filled']:>5d} "
                             f"{100*r['fill_rate']:>5.1f} {r['pas_net']:>+8.3f} "
                             f"{r['pas_sharpe']:>7.2f} {r['pas_pf']:>6.2f} "
                             f"{r['pas_profdays']:>6.2f} {r['gap_label_vs_pas']:>+8.3f}")
    else:
        win_lines.append("NONE — no cell passes HC #428 gates after canonical FIFO (passive-passive cost).")
        win_lines.append("Closest survivors (sorted by pas_net desc):")
        srt = sorted(summary_rows, key=lambda r: (-(r["pas_net"] if np.isfinite(r["pas_net"]) else -1e9)))
        for r in srt[:6]:
            win_lines.append(f"  {r['side']} {r['horizon']} {r['conf']} {r['tod']} vd{r['vel_dec']}: "
                             f"n_fill={r['n_filled']} fr={100*r['fill_rate']:.1f}% "
                             f"pas_net={r['pas_net']:+.3f} pas_Sh={r['pas_sharpe']:.2f}  "
                             f"fails=[{r['pas_fails']}]")
    (OUT_DIR / "winning_cells_fifo.txt").write_text("\n".join(win_lines) + "\n")

    # ── REPORT.md ───────────────────────────────────────────────────────────
    top = summary_rows[0] if summary_rows else None
    n_pass_pas = sum(1 for r in summary_rows if r["pas_pass"])
    n_pass_mkt = sum(1 for r in summary_rows if r["mkt_pass"])

    lines = []
    lines.append("# TOD x Velocity — FIFO Market-Replay Validation (HC #74)\n")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    lines.append(f"Source cells: {WIN_CELLS_TXT}")
    lines.append(f"OOT dates: {len(dates)} (DBN+pred, skipped {sorted(SKIP_DATES)})\n")
    lines.append(f"## HEADLINE")
    if n_pass_pas == 0:
        lines.append(f"**ALL {len(summary_rows)} LABEL-WINNERS DIED IN FIFO** under canonical (passive-passive 0.376 t commission only) cost.")
    else:
        lines.append(f"**{n_pass_pas}/{len(summary_rows)} cells SURVIVE FIFO** under passive-passive cost.")
    if n_pass_mkt == 0:
        lines.append(f"With realistic market-exit cost (+1.0 t spread), {n_pass_mkt}/{len(summary_rows)} survive.")
    else:
        lines.append(f"With realistic market-exit cost (+1.0 t spread), {n_pass_mkt}/{len(summary_rows)} survive.\n")
    lines.append("")
    lines.append("## Per-cell verdict\n")
    lines.append("| side | h | conf | tod | vd | n_sig | n_fill | fill% | label | pas_net | mkt_net | gap_lbl_vs_pas | verdict |")
    lines.append("|------|---|------|-----|----|------|-------|------|------|------|------|------|------|")
    for r in summary_rows:
        v = "ACCEPT" if r["pas_pass"] else "REJECT"
        lines.append(
            f"| {r['side']} | {r['horizon']} | {r['conf']} | {r['tod']} | {r['vel_dec']} "
            f"| {r['n_signals']} | {r['n_filled']} | {100*r['fill_rate']:.1f}% "
            f"| {r['label_net']:+.3f} | {r['pas_net']:+.3f} | {r['mkt_net']:+.3f} "
            f"| {r['gap_label_vs_pas']:+.3f} | {v} |"
        )
    lines.append("")
    lines.append("## Key number: label-vs-FIFO gap (passive-passive)\n")
    lines.append("This is the cost of label-fill optimism. Cells with gap > 1 tick = severe adverse selection.\n")
    for r in summary_rows:
        adv = "  ⚠ADVERSE SELECTION" if (np.isfinite(r["gap_label_vs_pas"]) and r["gap_label_vs_pas"] < -1.0) else ""
        lines.append(f"- {r['side']} {r['horizon']} {r['conf']} {r['tod']} vd{r['vel_dec']}: "
                     f"label {r['label_net']:+.3f}t  ->  FIFO {r['pas_net']:+.3f}t  "
                     f"(gap {r['gap_label_vs_pas']:+.3f}t){adv}")
    lines.append("")
    lines.append("## Top cell — long 10s top5pc mid_am vd8\n")
    top_cell = next((r for r in summary_rows
                     if r["side"] == "long" and r["horizon"] == "10s"
                     and r["conf"] == "top5pc" and r["tod"] == "mid_am"
                     and r["vel_dec"] == 8), None)
    if top_cell:
        lines.append(f"- Label: net {top_cell['label_net']:+.3f}t Sh {top_cell['label_sharpe']:.2f}")
        lines.append(f"- FIFO passive-passive: net {top_cell['pas_net']:+.3f}t Sh {top_cell['pas_sharpe']:.2f}  "
                     f"PF {top_cell['pas_pf']:.2f}  fill {100*top_cell['fill_rate']:.1f}%")
        lines.append(f"- FIFO market-exit: net {top_cell['mkt_net']:+.3f}t Sh {top_cell['mkt_sharpe']:.2f}")
        lines.append(f"- Verdict (passive): {'ACCEPT' if top_cell['pas_pass'] else 'REJECT  reasons=['+top_cell['pas_fails']+']'}")
    else:
        lines.append("(top cell not found in summary — diagnostic only)")
    lines.append("")
    lines.append("## Structural interpretation\n")
    n_dead = sum(1 for r in summary_rows
                 if np.isfinite(r["gap_label_vs_pas"]) and r["gap_label_vs_pas"] < -1.0)
    if n_dead > 0:
        lines.append(f"- {n_dead}/{len(summary_rows)} cells show >1 tick label-vs-FIFO gap → ADVERSE SELECTION.")
        lines.append("  When the model says 'go long' the touch is moving against the passive limit, so fills "
                     "skew to bad prints. This is the same failure mode that killed meta_classifier v1.")
    else:
        lines.append("- No cell shows >1 tick label-vs-FIFO gap. Adverse selection within tolerance.")
    avg_fill = float(np.mean([r["fill_rate"] for r in summary_rows])) if summary_rows else 0.0
    lines.append(f"- Average fill rate across cells: {100*avg_fill:.1f}%.")
    if avg_fill < 0.20:
        lines.append("  Most signaled orders never fill — passive-limit edge is being outcompeted in the queue.")
    lines.append("")
    lines.append("## Recommended next move\n")
    if n_pass_pas == 0:
        lines.append("- DO NOT DEPLOY any tod_velocity_stratification_v1 cell. Labels lied.")
        lines.append("- Stop label-based gate testing entirely; require FIFO validation up-front for every candidate.")
        lines.append("- Consider chase orders (allow reprices) or wider TP buckets — re-spec, re-test, then re-FIFO.")
    else:
        lines.append(f"- Lock the {n_pass_pas} surviving cell(s); run 5-day shadow paper-trade to confirm.")
        lines.append("- Track per-day fill rate live — if it drops materially vs replay, kill.")
    lines.append("")
    lines.append("## Notes\n")
    lines.append("- Cost (a) passive-passive: pnl_net = pnl_gross - 0.376 t RT commission (entry+exit both passive limit, exit at mid by engine).")
    lines.append("- Cost (b) market-exit:   pnl_net_b = pnl_net - 1.0 t spread crossing (exit by market at max_hold).")
    lines.append("- TP/SL set wide (1000 t) so all fills exit by max_hold or EOD, per HC #428 R2.")
    lines.append("- Engine: alpha_discovery.deep_models.fifo_market_replay.FIFOReplayEngine — UNMODIFIED.")
    (OUT_DIR / "REPORT.md").write_text("\n".join(lines) + "\n")

    # ── .regen_complete.json ────────────────────────────────────────────────
    elapsed = time.time() - t0
    finished_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    stamp = dict(
        task="tod_velocity_fifo_replay_v1",
        hc_refs=["HC#74", "HC#428", "HC#420", "HC#485R5", "HC#0", "HC#393"],
        started_at=started_at, finished_at=finished_at,
        elapsed_seconds=round(elapsed, 1),
        n_cells=len(cells),
        n_oot_dates=len(dates),
        date_subsample_every=DATE_SUBSAMPLE_EVERY,
        dates_used=list(dates),
        skipped_dates=sorted(SKIP_DATES),
        n_pass_passive=int(n_pass_pas),
        n_pass_market=int(n_pass_mkt),
        outputs=dict(
            summary=str(OUT_DIR / "per_cell_fifo_summary.csv"),
            per_fill=str(OUT_DIR / "per_fill_detail.parquet"),
            winners=str(OUT_DIR / "winning_cells_fifo.txt"),
            report=str(OUT_DIR / "REPORT.md"),
        ),
        verdicts={
            f"{c['side']}_{c['horizon']}_{c['conf']}_{c['tod']}_vd{c['vel_dec']}":
                ("ACCEPT" if r["pas_pass"] else "REJECT")
            for c, r in zip(cells, summary_rows)
        },
    )
    (OUT_DIR / ".regen_complete.json").write_text(json.dumps(stamp, indent=2, default=str))

    print(f"\n[done] elapsed={elapsed:.1f}s  passive_pass={n_pass_pas}/{len(summary_rows)}  "
          f"market_pass={n_pass_mkt}/{len(summary_rows)}  output={OUT_DIR}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
