#!/usr/bin/env python3
"""
HC #270 — Phase 1: Per-signal FIFO outcome labeler.

For each OOT date with CNN-Mamba predictions:
  - Generate signals at gate_threshold=0.0 (so every prediction with min_interval=500ms
    becomes a candidate signal — keeps both sides; meta-LGBM learns side selection).
  - Run FIFOReplayEngine.simulate(...) on those signals using the CURRENT BEST entry rules
    (passive limit, TP=8 SL=5 cancel=2s hold=30s — matching tonight's HC #268-validated winner).
  - For each signal, emit one row to parquet with:
      (date, signal_ts_ns, direction, pred_1s, pred_5s, pred_10s,
       abs_pred_1s, abs_pred_5s, abs_pred_10s, signal_strength,
       is_filled, pnl_ticks_gross, pnl_ticks_net, exit_reason, hold_time_ns,
       queue_wait_ns, mid_at_signal, spread_at_signal,
       label_winner, label_net_ticks)
  - Output: <out_dir>/<date>_signals_labeled.parquet

This is the LABEL source for the meta-LGBM gate. Phase 2 (separate script) will
add PatchTST agreement, microstructure features, vol regime, ToD regime.
"""

from pathlib import Path
import sys
import argparse
import logging
import numpy as np

LVL3_ROOT = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(LVL3_ROOT))

CNN_PRED_DIR = LVL3_ROOT / "output" / "cnn_mamba_v2_all_oot"
MBO_EVENT_DIR = LVL3_ROOT / "data" / "processed" / "mbo_events_smart_v3"
DEFAULT_OUT = LVL3_ROOT / "output" / "meta_lgbm_labels"

COMMISSION_TICKS = 0.376

from alpha_discovery.deep_models.fifo_market_replay import (  # noqa: E402
    FIFOReplayEngine, generate_signals,
)

DEFAULT_STRIDE = 250
DEFAULT_WINDOW = 3000

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("meta_lgbm_label")


def build_data_for_date(date_str: str):
    cnn_path = CNN_PRED_DIR / f"{date_str}_predictions.npz"
    mbo_path = MBO_EVENT_DIR / f"{date_str}_mbo_events.npz"
    if not cnn_path.exists():
        return None, f"missing CNN pred {cnn_path.name}"
    if not mbo_path.exists():
        return None, f"missing MBO event {mbo_path.name}"
    d = np.load(cnn_path, allow_pickle=True)
    mbo = np.load(mbo_path, allow_pickle=True)
    preds = d["predictions"]
    labels = d["labels"]
    ts_events = mbo["timestamps"]
    n_events = len(ts_events)
    stride = int(d["stride"]) if "stride" in d else DEFAULT_STRIDE
    window = int(d["window_size"]) if "window_size" in d else DEFAULT_WINDOW
    n_preds = len(preds)
    pred_ts = np.empty(n_preds, dtype=np.int64)
    for i in range(n_preds):
        event_idx = min(i * stride + window - 1, n_events - 1)
        pred_ts[i] = ts_events[event_idx]
    return {
        "date": date_str,
        "predictions": preds,
        "labels": labels,
        "timestamps_ns": pred_ts,
        "n_preds": n_preds,
        "n_events": n_events,
        "stride": stride,
        "window": window,
    }, None


def label_one_date(date_str: str, out_dir: Path,
                   tp_ticks: float = 8.0, sl_ticks: float = 5.0,
                   cancel_ms: float = 2000, max_hold_ms: float = 30000,
                   min_interval_ms: int = 500,
                   gate_threshold: float = 0.0):
    out_path = out_dir / f"{date_str}_signals_labeled.parquet"
    if out_path.exists():
        log.info(f"[{date_str}] already labeled → {out_path.name}, skipping")
        return out_path
    data, err = build_data_for_date(date_str)
    if err:
        log.warning(f"[{date_str}] {err}")
        return None
    log.info(f"[{date_str}] {data['n_preds']:,} CNN-Mamba predictions over {data['n_events']:,} events")
    signals = generate_signals(
        data,
        gate_threshold=gate_threshold,
        horizon=0,  # 1s direction
        min_interval_ns=int(min_interval_ms * 1_000_000),
    )
    log.info(f"[{date_str}] generated {len(signals):,} signals (gate≥{gate_threshold}, ≥{min_interval_ms}ms apart)")
    if not signals:
        log.warning(f"[{date_str}] zero signals, skipping")
        return None
    try:
        engine = FIFOReplayEngine(
            date=date_str,
            instrument_id=None,
            cancel_after_ns=int(cancel_ms * 1e6),
            max_hold_ns=int(max_hold_ms * 1e6),
            max_reprices=3,
            reprice_after_ns=int(1000 * 1e6),
        )
    except FileNotFoundError as e:
        log.error(f"[{date_str}] DBN missing: {e}")
        return None
    trades = engine.simulate(signals, tp_ticks=tp_ticks, sl_ticks=sl_ticks,
                             order_type="limit")
    trade_by_ts = {int(t.signal_ts_ns): t for t in (trades or [])}
    log.info(f"[{date_str}] simulated → {len(trades):,} trades ({100*len(trades)/max(1,len(signals)):.1f}% fill rate)")

    # Build per-signal records
    pred_1s = data["predictions"][:, 0]
    pred_5s = data["predictions"][:, 1]
    pred_10s = data["predictions"][:, 2]
    pred_ts = data["timestamps_ns"]
    # Build a ts_ns → row index lookup (vectorized)
    ts2idx = {int(ts): i for i, ts in enumerate(pred_ts)}

    rows = []
    for s in signals:
        ts = int(s["ts_ns"])
        i = ts2idx.get(ts, -1)
        if i < 0:
            continue
        t = trade_by_ts.get(ts)
        is_filled = t is not None
        rows.append({
            "date": date_str,
            "signal_ts_ns": ts,
            "pred_idx": i,
            "direction": s["direction"],
            "signal_strength": float(s["strength"]),
            "pred_1s": float(pred_1s[i]),
            "pred_5s": float(pred_5s[i]),
            "pred_10s": float(pred_10s[i]),
            "abs_pred_1s": float(abs(pred_1s[i])),
            "abs_pred_5s": float(abs(pred_5s[i])),
            "abs_pred_10s": float(abs(pred_10s[i])),
            "is_filled": is_filled,
            "pnl_ticks_gross": float(t.pnl_ticks) if is_filled else float("nan"),
            "pnl_ticks_net": float(t.pnl_ticks_net) if is_filled else float("nan"),
            "exit_reason": str(t.exit_reason) if is_filled else "unfilled",
            "hold_time_ns": int(t.hold_time_ns) if is_filled else -1,
            "queue_wait_ns": int(t.queue_wait_ns) if is_filled else -1,
            "mid_at_signal": float(t.mid_at_signal) if is_filled else float("nan"),
            "spread_at_signal": (float(t.spread_at_signal) if is_filled and t.spread_at_signal is not None
                                 else float("nan")),
            # Labels for LGBM:
            "label_winner": (1 if (is_filled and t.pnl_ticks_net > 0) else 0),
            "label_net_ticks": (float(t.pnl_ticks_net) if is_filled else float("nan")),
        })

    import pandas as pd
    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    log.info(f"[{date_str}] wrote {len(df):,} rows → {out_path}")
    log.info(f"[{date_str}]   filled: {df['is_filled'].sum():,} ({100*df['is_filled'].mean():.1f}%)")
    if df["is_filled"].sum() > 0:
        f = df[df["is_filled"]]
        log.info(f"[{date_str}]   sum NET ticks (all filled): {f['pnl_ticks_net'].sum():+.2f}")
        log.info(f"[{date_str}]   winners: {f['label_winner'].sum():,} ({100*f['label_winner'].mean():.1f}%)")
    return out_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--date", type=str, default=None,
                   help="Single date YYYYMMDD; if omitted, runs all dates in CNN_PRED_DIR")
    p.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT))
    p.add_argument("--tp", type=float, default=8.0)
    p.add_argument("--sl", type=float, default=5.0)
    p.add_argument("--cancel-ms", type=float, default=2000)
    p.add_argument("--hold-ms", type=float, default=30000)
    p.add_argument("--min-interval-ms", type=int, default=500)
    p.add_argument("--gate-threshold", type=float, default=0.0)
    args = p.parse_args()
    out_dir = Path(args.out_dir)

    if args.date:
        dates = [args.date]
    else:
        dates = sorted([f.name[:8] for f in CNN_PRED_DIR.glob("*_predictions.npz")])

    log.info(f"Labeling {len(dates)} dates → {out_dir}")
    log.info(f"Entry rules: TP={args.tp} SL={args.sl} cancel={args.cancel_ms}ms hold={args.hold_ms}ms")
    ok = 0
    for d in dates:
        try:
            r = label_one_date(d, out_dir,
                               tp_ticks=args.tp, sl_ticks=args.sl,
                               cancel_ms=args.cancel_ms, max_hold_ms=args.hold_ms,
                               min_interval_ms=args.min_interval_ms,
                               gate_threshold=args.gate_threshold)
            if r is not None:
                ok += 1
        except Exception as e:
            log.error(f"[{d}] FAILED: {e}", exc_info=True)
    log.info(f"DONE: {ok}/{len(dates)} dates labeled")


if __name__ == "__main__":
    main()
