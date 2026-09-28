#!/usr/bin/env python3
"""
headc_fifo_gate_standalone.py — Parameterized standalone FIFO exploitability
gate for the head-C first-passage family (K=1, K=2, ...).

Generalizes headc_k1_fifo_gate_standalone.py. Same canonical engine, identical
methodology: TAU gate -> FIFOReplayEngine, limit + market passes.
Per HC #428 R2: TP=SL=K_TICKS (the predicted amplitude), cancel=h, hold=1.5h.
FIFO market replay ONLY (HC #74).

Example (K=2 base run):
  python3 scripts/headc_fifo_gate_standalone.py \
    --out-dir output/p_alpha_headc_v1 --k-ticks 2 \
    --experiment p_alpha_headc_firstpassage_v1 \
    --parent-run-id 8d709c1f6eb14066b6ff3c109ffc65c1

CPU only. One day at a time to keep RAM modest.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

LVL3_ROOT = Path("/home/nick/Lvl3Quant")

p = argparse.ArgumentParser()
p.add_argument("--out-dir", required=True, help="Run output dir with fold_*_oot_*.npz")
p.add_argument("--k-ticks", type=float, required=True, help="TP=SL=K ticks")
p.add_argument("--horizon-s", type=float, default=5.0)
p.add_argument("--tau-long", type=float, default=0.70)
p.add_argument("--tau-short", type=float, default=0.30)
p.add_argument("--experiment", default=None, help="MLflow experiment name")
p.add_argument("--parent-run-id", default=None)
p.add_argument("--suffix", default="", help="Suffix for output filenames (tau sweeps)")
p.add_argument("--no-mlflow", action="store_true")
args = p.parse_args()

OUT_DIR = Path(args.out_dir)
if not OUT_DIR.is_absolute():
    OUT_DIR = LVL3_ROOT / OUT_DIR
HORIZON_S = args.horizon_s
K_TICKS = args.k_ticks
TP_TICKS = K_TICKS
SL_TICKS = K_TICKS
CANCEL_NS = int(HORIZON_S * 1e9)
HOLD_NS = int(1.5 * HORIZON_S * 1e9)
TAU_LONG = args.tau_long
TAU_SHORT = args.tau_short
ORDER_TYPES = ["limit", "market"]
SUF = args.suffix

LOG_FILE = LVL3_ROOT / "logs" / f"headc_fifo_gate_{OUT_DIR.name}{SUF}_{time.strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("headc_gate")
log.info(f"Gate params: out_dir={OUT_DIR} K={K_TICKS} h={HORIZON_S}s "
         f"tau=({TAU_LONG},{TAU_SHORT}) TP=SL={TP_TICKS} cancel={CANCEL_NS/1e9}s hold={HOLD_NS/1e9}s")

# Load canonical FIFO engine
fifo_path = LVL3_ROOT / "alpha_discovery" / "deep_models" / "fifo_market_replay.py"
spec = importlib.util.spec_from_file_location("fifo_market_replay", str(fifo_path))
fmod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fmod)
FIFOReplayEngine = fmod.FIFOReplayEngine


def trade_net_ticks(t) -> float:
    # TradeResult dataclass uses pnl_ticks_net (after commission)
    if hasattr(t, "pnl_ticks_net"):
        return float(t.pnl_ticks_net)
    if isinstance(t, dict):
        return float(t.get("pnl_ticks_net", t.get("net_ticks", np.nan)))
    return float("nan")


def main():
    npz_files = sorted(OUT_DIR.glob("fold_*_oot_*.npz"))
    log.info(f"Found {len(npz_files)} fold NPZs in {OUT_DIR}")

    per_day_rows = []
    for npz in npz_files:
        date = npz.stem.split("_oot_")[1]
        fold = int(npz.stem.split("_")[1])
        with np.load(npz, allow_pickle=False) as z:
            preds = z["predictions"]
            ts_ns = z["ts_ns"]

        long_mask = preds >= TAU_LONG
        short_mask = preds <= TAU_SHORT
        sig_idx = np.where(long_mask | short_mask)[0]
        log.info(f"Fold {fold} {date}: n_preds={len(preds)} "
                 f"n_long={int(long_mask.sum())} n_short={int(short_mask.sum())}")

        if len(sig_idx) == 0:
            for ot in ORDER_TYPES:
                per_day_rows.append({"fold": fold, "date": date, "order_type": ot,
                                     "n_signals": 0, "n_trades": 0,
                                     "net_ticks_per_trade": float("nan"),
                                     "sum_net_ticks": 0.0, "win_rate": float("nan"),
                                     "sharpe": float("nan"), "pf": float("nan")})
            continue

        signals = [{"ts_ns": int(ts_ns[i]),
                    "direction": "long" if preds[i] >= TAU_LONG else "short",
                    "strength": float(abs(preds[i] - 0.5) * 2.0)}
                   for i in sig_idx]

        try:
            eng = FIFOReplayEngine(date=date, cancel_after_ns=CANCEL_NS,
                                   max_hold_ns=HOLD_NS)
        except Exception as e:
            log.warning(f"  FIFO {date}: engine load error: {e}")
            for ot in ORDER_TYPES:
                per_day_rows.append({"fold": fold, "date": date, "order_type": ot,
                                     "n_signals": len(signals), "n_trades": 0,
                                     "error": str(e)[:80]})
            continue

        for ot in ORDER_TYPES:
            try:
                trades = eng.simulate(signals=signals, tp_ticks=TP_TICKS,
                                      sl_ticks=SL_TICKS, order_type=ot)
            except Exception as e:
                log.warning(f"  FIFO {date} {ot}: simulate error: {e}")
                per_day_rows.append({"fold": fold, "date": date, "order_type": ot,
                                     "n_signals": len(signals), "n_trades": 0,
                                     "error": str(e)[:80]})
                continue

            net = np.array([trade_net_ticks(t) for t in trades], dtype=np.float64)
            net = net[~np.isnan(net)]
            if len(net) == 0:
                per_day_rows.append({"fold": fold, "date": date, "order_type": ot,
                                     "n_signals": len(signals), "n_trades": 0,
                                     "net_ticks_per_trade": float("nan"),
                                     "sum_net_ticks": 0.0, "win_rate": float("nan"),
                                     "sharpe": float("nan"), "pf": float("nan")})
                log.info(f"  FIFO {date} {ot}: 0 trades from {len(signals)} signals")
                continue

            wins = net[net > 0]
            losses = net[net < 0]
            wr = float((net > 0).mean())
            pf = float(wins.sum() / abs(losses.sum())) if losses.sum() != 0 else float("inf")
            std = float(net.std(ddof=1)) if len(net) > 1 else float("nan")
            sh = float(net.mean() / std) if std and std > 0 else float("nan")
            per_day_rows.append({
                "fold": fold, "date": date, "order_type": ot,
                "n_signals": len(signals), "n_trades": len(net),
                "net_ticks_per_trade": float(net.mean()),
                "sum_net_ticks": float(net.sum()),
                "win_rate": wr, "pf": pf, "sharpe": sh,
            })
            log.info(f"  FIFO {date} {ot}: n_sig={len(signals)} n_trades={len(net)} "
                     f"ntpt={net.mean():+.3f} WR={wr:.3f} PF={pf:.2f} SH={sh:+.2f}")

        del eng  # free MBO records before next day

    csv_path = OUT_DIR / f"fifo_gate_results{SUF}.csv"
    keys = sorted({k for r in per_day_rows for k in r.keys()})
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in per_day_rows:
            w.writerow(r)
    log.info(f"Wrote {csv_path}")

    summary = {"params": {"k_ticks": K_TICKS, "horizon_s": HORIZON_S,
                          "tau_long": TAU_LONG, "tau_short": TAU_SHORT,
                          "tp_ticks": TP_TICKS, "sl_ticks": SL_TICKS,
                          "cancel_s": CANCEL_NS / 1e9, "hold_s": HOLD_NS / 1e9}}
    for ot in ORDER_TYPES:
        rows = [r for r in per_day_rows if r.get("order_type") == ot]
        valid = [r for r in rows if r.get("n_trades", 0) > 0
                 and not (isinstance(r.get("net_ticks_per_trade"), float)
                          and np.isnan(r["net_ticks_per_trade"]))]
        if not valid:
            summary[ot] = {"status": "no_trades", "n_days": len(rows)}
            continue
        ntpts = np.array([r["net_ticks_per_trade"] for r in valid])
        n_trades = np.array([r["n_trades"] for r in valid])
        all_net_sum = float(sum(r["sum_net_ticks"] for r in valid))
        pos_days = int((ntpts > 0).sum())
        summary[ot] = {
            "n_days_with_trades": len(valid),
            "n_days_total": len(rows),
            "n_days_positive": pos_days,
            "pct_days_positive": pos_days / len(valid),
            "mean_ntpt": float(ntpts.mean()),
            "median_ntpt": float(np.median(ntpts)),
            "total_net_ticks": all_net_sum,
            "total_trades": int(n_trades.sum()),
            "mean_trades_per_day": float(n_trades.mean()),
            "day_sharpe": float(ntpts.mean() / ntpts.std(ddof=1)) if len(ntpts) > 1 and ntpts.std(ddof=1) > 0 else None,
            "commission_floor_ticks": 0.376,
            "viable_per_hc494_r1": bool(ntpts.mean() > 0.376 and n_trades.mean() >= 5),
            "viable_directional": bool(ntpts.mean() > 0
                                       and pos_days >= 0.6 * len(valid)
                                       and n_trades.mean() >= 5),
        }

    summary_path = OUT_DIR / f"fifo_gate_summary{SUF}.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    log.info(f"Wrote {summary_path}")
    log.info(json.dumps(summary, indent=2))

    if not args.no_mlflow and args.experiment:
        try:
            import mlflow
            mlflow.set_tracking_uri("http://localhost:5000")
            mlflow.set_experiment(args.experiment)
            with mlflow.start_run(run_name=f"fifo_gate_standalone{SUF}"):
                if args.parent_run_id:
                    mlflow.set_tag("parent_run_id", args.parent_run_id)
                mlflow.log_params(summary["params"])
                for ot in ORDER_TYPES:
                    s = summary.get(ot, {})
                    for k in ("mean_ntpt", "median_ntpt", "total_net_ticks",
                              "mean_trades_per_day", "pct_days_positive", "day_sharpe"):
                        v = s.get(k)
                        if v is not None and isinstance(v, (int, float)) and np.isfinite(v):
                            mlflow.log_metric(f"{ot}_{k}", float(v))
                mlflow.log_artifact(str(csv_path))
                mlflow.log_artifact(str(summary_path))
            log.info("Logged to MLflow")
        except Exception as e:
            log.warning(f"MLflow logging failed (non-fatal): {e}")

    log.info("DONE")


if __name__ == "__main__":
    main()
