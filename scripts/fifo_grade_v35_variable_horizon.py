#!/usr/bin/env python3
"""
fifo_grade_v35_variable_horizon.py — HC #497 v3.5 variable-horizon FIFO grader.

Replays per-event v3.5 predictions through the canonical FIFO market replay
engine, using exits that respect EACH HEAD's predicted horizon (HC #428 R2):

  Trade ENTRY:
      head-A prob crosses entry threshold (default 0.6 for long, < 1-0.6 for
      short — i.e. directional conviction) AND
      head-B predicted persistence > min_hold_s (default 5.0)

  Trade EXIT (whichever first):
      (a) head-A direction flips below the entry threshold
      (b) head-B predicted persistence has elapsed since entry
      (c) head-C cumulative-K-tick first-passage triggers (using K matched
          to head-A horizon — default K=4)
      (d) max-hold safety cap (default 300s)

  Limit ORDER CANCEL window:
      <= head-B predicted persistence (HC #428 R2 compliance)

Inputs:
  --preds-glob  pattern of per-day v3.5 prediction NPZ files. Expected keys:
                prob_A (N,4), pred_B (N,), prob_C (N,3,3), prob_D (N,3),
                ts_ns (N,)  [event timestamps]
  --mbo-dir     dir of MBO event NPZ files (matches v3.5 stream timestamps)

Outputs (per --out-dir):
  per_day.csv        stratified per-day Sharpe/PF/WR/trade-count, regime
                     classification (green/red/flat by ES close-to-close)
  summary.json       headline summary
  REPORT.md          plain-English summary

Regime classification (HC #428 R1):
  We compute per-day green/red/flat tags from the daily mid-price drift
  observable in the MBO event stream. Reject report if
      |Sharpe_green - Sharpe_red| / max(|S_g|, |S_r|) > 0.50.

Smoke usage (synthetic preds, no real data):
  python fifo_grade_v35_variable_horizon.py --smoke --out-dir /tmp/v35_grade_smoke
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("fifo_v35_grade")

# ── Cost constants (HC #52) ──────────────────────────────────────────────────
TICK_USD = 12.50
RT_COMMISSION = 4.70
RT_COMMISSION_TICKS = RT_COMMISSION / TICK_USD  # 0.376

# ── Defaults (Monday-tunable) ────────────────────────────────────────────────
DEFAULT_ENTRY_PROB_LONG = 0.60         # head-A prob > this for long
DEFAULT_ENTRY_PROB_SHORT = 0.40        # head-A prob < this for short
DEFAULT_HEAD_A_K_IDX = 1                # 0=5s, 1=10s, 2=30s, 3=60s
DEFAULT_HEAD_C_K_IDX = 1                # 0=K2, 1=K4, 2=K8
DEFAULT_MIN_PERSISTENCE_S = 5.0
DEFAULT_MAX_HOLD_S = 300.0
DEFAULT_REGIME_FLAT_BAND_TICKS = 4.0    # |close-open| <= 4 ticks → flat day


# ── Trade record ─────────────────────────────────────────────────────────────
@dataclass
class V35Trade:
    date: str
    entry_idx: int
    direction: str          # "long" or "short"
    entry_ts_ns: int
    exit_ts_ns: int
    entry_price: float      # ticks
    exit_price: float       # ticks
    pnl_ticks_gross: float
    pnl_ticks_net: float    # gross - commission
    hold_s: float
    exit_reason: str        # 'a_flip', 'b_persistence', 'c_first_passage', 'd_cap'


# ── Synthetic prediction generator for smoke ─────────────────────────────────
def synth_preds_for_day(n_events: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """Uniform-random plausible predictions for one day. Used for smoke run."""
    return {
        # prob_A: 4 horizons (5,10,30,60s)
        "prob_A": rng.uniform(0.3, 0.7, size=(n_events, 4)).astype(np.float32),
        # pred_B: persistence in seconds (uniform 0.5–30s)
        "pred_B": rng.uniform(0.5, 30.0, size=n_events).astype(np.float32),
        # prob_C: 3 K bands, 3-class softmax. Synthesize by drawing dirichlet.
        "prob_C": rng.dirichlet([1.0, 1.0, 1.0], size=(n_events, 3)).astype(np.float32),
        # prob_D: regime softmax
        "prob_D": rng.dirichlet([1.0, 1.0, 1.0], size=n_events).astype(np.float32),
    }


# ── Per-day FIFO grader (simplified — uses labels_1s as price proxy) ─────────
def grade_one_day(
    preds: Dict[str, np.ndarray],
    ts_ns: np.ndarray,
    labels_1s: np.ndarray,
    date: str,
    entry_prob_long: float = DEFAULT_ENTRY_PROB_LONG,
    entry_prob_short: float = DEFAULT_ENTRY_PROB_SHORT,
    head_a_k_idx: int = DEFAULT_HEAD_A_K_IDX,
    head_c_k_idx: int = DEFAULT_HEAD_C_K_IDX,
    min_persistence_s: float = DEFAULT_MIN_PERSISTENCE_S,
    max_hold_s: float = DEFAULT_MAX_HOLD_S,
) -> List[V35Trade]:
    """Walk through events for one day, generate trades with variable-horizon exits.

    SCAFFOLD NOTE: This grader uses labels_1s as a per-event price-step proxy
    (its diff approximates one-tick price increments). For the full canonical
    FIFO realism (queue position, partial fills, spread crossings), wire
    `FIFOReplayEngine.simulate()` from alpha_discovery/deep_models/fifo_market_replay.py
    here. Doing that requires raw DBN files which are NOT needed for the
    HC #497 smoke deliverable.
    """
    n = len(ts_ns)
    trades: List[V35Trade] = []
    if n < 2:
        return trades

    # Reconstruct cumulative price path from labels_1s diffs (ticks)
    incr = np.diff(labels_1s, prepend=labels_1s[0])
    incr = np.where(np.isnan(incr), 0.0, incr).astype(np.float64)
    price = np.cumsum(incr).astype(np.float64)  # ticks since SOD

    prob_A = preds["prob_A"]  # (N, 4)
    pred_B = preds["pred_B"]  # (N,)
    K_C_value = [2.0, 4.0, 8.0][head_c_k_idx]

    i = 0
    while i < n:
        pA = prob_A[i, head_a_k_idx]
        pers = pred_B[i]
        # Entry gate
        direction = None
        if pA > entry_prob_long and pers > min_persistence_s:
            direction = "long"
        elif pA < entry_prob_short and pers > min_persistence_s:
            direction = "short"
        if direction is None:
            i += 1
            continue

        # Open trade — entry at current price (passive limit @ touch assumed)
        entry_price = price[i]
        entry_ts = int(ts_ns[i])
        exit_horizon_ns = int(min(pers, max_hold_s) * 1e9)
        exit_deadline = entry_ts + exit_horizon_ns
        max_deadline = entry_ts + int(max_hold_s * 1e9)
        # Walk forward looking for exit triggers
        exit_idx = i + 1
        exit_reason = "d_cap"
        while exit_idx < n:
            t = ts_ns[exit_idx]
            if t >= max_deadline:
                exit_reason = "d_cap"
                break
            # (b) persistence elapsed
            if t >= exit_deadline:
                exit_reason = "b_persistence"
                break
            # (a) head-A flip relative to direction
            pA_now = prob_A[exit_idx, head_a_k_idx]
            if direction == "long" and pA_now < (1.0 - entry_prob_long):
                exit_reason = "a_flip"
                break
            if direction == "short" and pA_now > entry_prob_short + 0.2:
                # short flip = directional prob now strongly long
                exit_reason = "a_flip"
                break
            # (c) cumulative K-tick first-passage — using realized price-path move
            move = price[exit_idx] - entry_price
            if direction == "long" and move >= K_C_value:
                exit_reason = "c_first_passage"
                break
            if direction == "short" and move <= -K_C_value:
                exit_reason = "c_first_passage"
                break
            if direction == "long" and move <= -K_C_value:
                # adverse hit on what we expected to be long-favorable
                exit_reason = "c_first_passage"
                break
            if direction == "short" and move >= K_C_value:
                exit_reason = "c_first_passage"
                break
            exit_idx += 1
        if exit_idx >= n:
            exit_idx = n - 1
            exit_reason = "d_cap"

        exit_price = price[exit_idx]
        exit_ts = int(ts_ns[exit_idx])
        gross_ticks = (exit_price - entry_price) if direction == "long" else (entry_price - exit_price)
        net_ticks = gross_ticks - RT_COMMISSION_TICKS
        hold_s = (exit_ts - entry_ts) / 1e9
        trades.append(V35Trade(
            date=date,
            entry_idx=i,
            direction=direction,
            entry_ts_ns=entry_ts,
            exit_ts_ns=exit_ts,
            entry_price=float(entry_price),
            exit_price=float(exit_price),
            pnl_ticks_gross=float(gross_ticks),
            pnl_ticks_net=float(net_ticks),
            hold_s=float(hold_s),
            exit_reason=exit_reason,
        ))
        # Skip past this trade to avoid pyramiding
        i = exit_idx + 1
    return trades


# ── Day-level metrics ────────────────────────────────────────────────────────
def day_metrics(trades: List[V35Trade]) -> Dict:
    if not trades:
        return {"n_trades": 0, "net_ticks": 0.0, "sharpe": 0.0, "wr": 0.0, "pf": 0.0}
    pnl = np.array([t.pnl_ticks_net for t in trades])
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    sharpe = float(pnl.mean() / (pnl.std() + 1e-9))
    pf = float(wins.sum() / abs(losses.sum())) if losses.sum() < 0 else float("inf")
    return {
        "n_trades": len(trades),
        "net_ticks_total": float(pnl.sum()),
        "net_ticks_per_trade": float(pnl.mean()),
        "sharpe_per_trade": sharpe,
        "wr": float((pnl > 0).mean()),
        "pf": pf,
        "n_long": int(sum(t.direction == "long" for t in trades)),
        "n_short": int(sum(t.direction == "short" for t in trades)),
        "exit_reason_dist": {
            r: int(sum(t.exit_reason == r for t in trades))
            for r in ("a_flip", "b_persistence", "c_first_passage", "d_cap")
        },
    }


# ── Regime classification (green/red/flat per close-to-close) ───────────────
def classify_regime(labels_1s: np.ndarray, flat_band_ticks: float) -> str:
    """Use accumulated price path from labels_1s diffs to classify day."""
    incr = np.diff(labels_1s, prepend=labels_1s[0])
    incr = np.where(np.isnan(incr), 0.0, incr)
    move = float(np.sum(incr))
    if abs(move) <= flat_band_ticks:
        return "flat"
    return "green" if move > 0 else "red"


def regime_skew_check(per_day: List[Dict]) -> Dict:
    by_regime = {"green": [], "red": [], "flat": []}
    for d in per_day:
        by_regime[d["regime"]].append(d["sharpe_per_trade"])
    out = {}
    for k, v in by_regime.items():
        out[k] = {
            "n_days": len(v),
            "mean_sharpe": float(np.mean(v)) if v else 0.0,
        }
    sg = out["green"]["mean_sharpe"]
    sr = out["red"]["mean_sharpe"]
    denom = max(abs(sg), abs(sr), 1e-9)
    skew = abs(sg - sr) / denom
    out["green_red_skew"] = float(skew)
    out["passes_hc428_r1"] = bool(skew <= 0.50)
    return out


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="Synthetic predictions on first day of MBO data")
    ap.add_argument("--preds-glob", type=str, default="",
                    help="Glob of per-day v3.5 prediction NPZ files")
    ap.add_argument("--mbo-dir", type=str,
                    default="/home/nick/Lvl3Quant/data/processed/mbo_events_smart_v3",
                    help="Dir of MBO event NPZ files (for ts + labels_1s)")
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit-events-per-day", type=int, default=0,
                    help="If >0, only process first N events per day (smoke fast)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        log.info("=== SMOKE MODE: synthetic preds on 1 MBO day ===")
        mbo_files = sorted(Path(args.mbo_dir).glob("*_mbo_events.npz"))
        if not mbo_files:
            log.warning(f"No MBO files in {args.mbo_dir} — falling back to "
                        "fully synthetic timestamps")
            n = 50000
            rng = np.random.default_rng(args.seed)
            ts_ns = np.cumsum(rng.integers(1_000_000, 50_000_000, size=n)).astype(np.int64)
            labels_1s = rng.normal(0, 1, size=n).astype(np.float32) * 0.2
            preds = synth_preds_for_day(n, rng)
            day_files = [("SYN", ts_ns, labels_1s, preds)]
        else:
            mbo_files = mbo_files[:1]
            day_files = []
            for f in mbo_files:
                with np.load(f, allow_pickle=False) as z:
                    ts_ns = z["timestamps"].astype(np.int64)
                    labels_1s = z["labels_1s"].astype(np.float32)
                if args.limit_events_per_day > 0:
                    ts_ns = ts_ns[: args.limit_events_per_day]
                    labels_1s = labels_1s[: args.limit_events_per_day]
                rng = np.random.default_rng(args.seed)
                preds = synth_preds_for_day(len(ts_ns), rng)
                date = f.stem.split("_")[0]
                day_files.append((date, ts_ns, labels_1s, preds))
    else:
        if not args.preds_glob:
            log.error("--preds-glob required for non-smoke run")
            sys.exit(2)
        pred_paths = sorted(glob.glob(args.preds_glob))
        if not pred_paths:
            log.error(f"No prediction files match {args.preds_glob}")
            sys.exit(2)
        day_files = []
        for p in pred_paths:
            # date = filename token starting with 8 digits
            date = ""
            for token in Path(p).stem.split("_"):
                if len(token) == 8 and token.isdigit():
                    date = token
                    break
            if not date:
                log.warning(f"Could not parse date from {p}, skipping")
                continue
            mbo_path = Path(args.mbo_dir) / f"{date}_mbo_events.npz"
            if not mbo_path.exists():
                log.warning(f"Missing MBO file {mbo_path}, skipping {date}")
                continue
            with np.load(p, allow_pickle=False) as zp:
                preds = {k: zp[k] for k in zp.files}
            with np.load(mbo_path, allow_pickle=False) as zm:
                ts_ns = zm["timestamps"].astype(np.int64)
                labels_1s = zm["labels_1s"].astype(np.float32)
            day_files.append((date, ts_ns, labels_1s, preds))

    per_day_records = []
    all_trades = []
    for date, ts_ns, labels_1s, preds in day_files:
        log.info(f"[{date}] grading {len(ts_ns):,} events")
        trades = grade_one_day(preds, ts_ns, labels_1s, date)
        m = day_metrics(trades)
        regime = classify_regime(labels_1s, DEFAULT_REGIME_FLAT_BAND_TICKS)
        m["date"] = date
        m["regime"] = regime
        per_day_records.append(m)
        all_trades.extend(trades)
        log.info(f"[{date}] n_trades={m['n_trades']} net={m.get('net_ticks_total',0):.2f} "
                 f"Sharpe={m.get('sharpe_per_trade',0):.2f} WR={m.get('wr',0):.2f} "
                 f"regime={regime}")

    # Aggregate metrics
    skew = regime_skew_check(per_day_records)
    summary = {
        "n_days": len(per_day_records),
        "n_trades_total": sum(d["n_trades"] for d in per_day_records),
        "net_ticks_total": sum(d.get("net_ticks_total", 0.0) for d in per_day_records),
        "per_day": per_day_records,
        "regime_check": skew,
        "config": {
            "entry_prob_long": DEFAULT_ENTRY_PROB_LONG,
            "entry_prob_short": DEFAULT_ENTRY_PROB_SHORT,
            "head_a_k_idx": DEFAULT_HEAD_A_K_IDX,
            "head_c_k_idx": DEFAULT_HEAD_C_K_IDX,
            "min_persistence_s": DEFAULT_MIN_PERSISTENCE_S,
            "max_hold_s": DEFAULT_MAX_HOLD_S,
            "commission_ticks": RT_COMMISSION_TICKS,
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Per-day CSV
    import csv
    with open(out_dir / "per_day.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "regime", "n_trades", "n_long", "n_short",
                    "net_ticks_total", "net_ticks_per_trade",
                    "sharpe_per_trade", "wr", "pf"])
        for d in per_day_records:
            w.writerow([d["date"], d["regime"], d["n_trades"],
                        d.get("n_long", 0), d.get("n_short", 0),
                        f"{d.get('net_ticks_total', 0):.3f}",
                        f"{d.get('net_ticks_per_trade', 0):.4f}",
                        f"{d.get('sharpe_per_trade', 0):.4f}",
                        f"{d.get('wr', 0):.3f}",
                        f"{d.get('pf', 0):.3f}"])

    # Plain-English report
    pf_skew = "PASS" if skew["passes_hc428_r1"] else "FAIL"
    md = []
    md.append("# v3.5 Variable-Horizon FIFO Grader Report")
    md.append("")
    md.append(f"- Days graded: {summary['n_days']}")
    md.append(f"- Total trades: {summary['n_trades_total']}")
    md.append(f"- Total net ticks: {summary['net_ticks_total']:.2f}")
    md.append(f"- HC #428 R1 regime-skew check: **{pf_skew}** (skew = {skew['green_red_skew']:.3f})")
    md.append("")
    md.append("## Per-regime breakdown")
    for r in ("green", "red", "flat"):
        info = skew[r]
        md.append(f"- {r}: {info['n_days']} days, mean Sharpe = {info['mean_sharpe']:.3f}")
    md.append("")
    md.append("See per_day.csv and summary.json for per-day detail.")
    (out_dir / "REPORT.md").write_text("\n".join(md))
    log.info(f"Wrote summary.json, per_day.csv, REPORT.md to {out_dir}")
    log.info(f"DONE — {summary['n_trades_total']} trades over {summary['n_days']} days, "
             f"net {summary['net_ticks_total']:.2f} ticks, skew {pf_skew}")


if __name__ == "__main__":
    main()
