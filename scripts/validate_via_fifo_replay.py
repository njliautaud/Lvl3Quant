#!/usr/bin/env python3
"""
HC #265 — FIFO market replay validator for supervised_exec_v3 h15 top-1% gate.

Runs the realized-fill FIFO simulator (alpha_discovery.deep_models.fifo_market_replay)
on the trade entries produced by the supervised exec MLP's top-1% gate. Reports
realized fills + realized exits + net-of-commission P&L.

Workflow:
  1. For each fold in supervised_exec_v3_h15_cpu_v2/fold_NN_predictions.npz:
       a. Read predicted_mfe + meta col 1 (pred_idx into CNN-Mamba stream)
       b. Compute top-1% mask
       c. Map mask back to CNN-Mamba stream length (one entry per CNN-Mamba pred)
  2. Load corresponding CNN-Mamba predictions from cnn_mamba_v2_bulk_oot/<date>_predictions.npz
  3. Load MBO event timestamps from data/processed/mbo_events_smart_v3/<date>_mbo_events.npz
  4. Call generate_signals() with exec_mlp gate, then FIFOReplayEngine.simulate()
  5. Aggregate realized-fill P&L per fold, then across folds

This script DOES NOT modify fifo_market_replay.py — it imports its public API.
"""

from pathlib import Path
import sys
import re
import numpy as np

LVL3_ROOT = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(LVL3_ROOT))

V3_OUT = LVL3_ROOT / "output" / "supervised_exec_v3_h15_cpu_v2"
CNN_PRED_DIR = LVL3_ROOT / "output" / "cnn_mamba_v2_bulk_oot"
MBO_EVENT_DIR = LVL3_ROOT / "data" / "processed" / "mbo_events_smart_v3"
COMMISSION_TICKS = 0.376
TICK_USD = 12.50

# Import FIFO replay public API
from alpha_discovery.deep_models.fifo_market_replay import (  # noqa: E402
    FIFOReplayEngine, generate_signals, find_dbn_path,
)

# CNN-Mamba prediction stride/window — required to map prediction idx→event idx
# (read from CNN-Mamba file's metadata; defaults below match v2)
DEFAULT_STRIDE = 250
DEFAULT_WINDOW = 3000


def build_data_for_date(date_str: str):
    """Build the `data` dict that generate_signals expects from CNN-Mamba pred file."""
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
    }, None


RANKER_COL = {"mfe": 0, "mae": 1, "pnl": 2, "prof": 3}


def build_exec_mlp_gate(fold_npz: Path, n_cnn_preds: int, top_pct: float,
                        ranker: str = "mfe", side_filter: str = "all"):
    """Convert v3 top-X% indices into an Exec MLP gate vector of length n_cnn_preds.

    ranker:      'mfe'|'mae'|'pnl'|'prof' — which v3 prediction column to rank by
    side_filter: 'all'|'long'|'short'      — restrict to one signal_dir
    """
    fd = np.load(fold_npz, allow_pickle=True)
    preds = fd["predictions"]    # (n_v3, 4)
    meta = fd["meta"]            # (n_v3, 3): ts_s, pred_idx, signal_dir
    col = RANKER_COL[ranker]
    score = preds[:, col]
    pred_idx = meta[:, 1].astype(int)
    signal_dir = meta[:, 2]
    n_v3 = len(score)
    eligible = np.ones(n_v3, dtype=bool)
    if side_filter == "long":
        eligible &= signal_dir > 0
    elif side_filter == "short":
        eligible &= signal_dir < 0
    elig_idx = np.where(eligible)[0]
    if len(elig_idx) == 0:
        return {"gate": np.zeros(n_cnn_preds, dtype=np.float32),
                "confidence": np.zeros(n_cnn_preds, dtype=np.float32)}, 0
    k = max(1, int(len(elig_idx) * top_pct))
    sub_score = score[elig_idx]
    order = np.argsort(-sub_score)[:k]
    sort_idx = elig_idx[order]
    selected_pred_idx = pred_idx[sort_idx]
    gate = np.zeros(n_cnn_preds, dtype=np.float32)
    valid = (selected_pred_idx >= 0) & (selected_pred_idx < n_cnn_preds)
    gate[selected_pred_idx[valid]] = 1.0
    confidence = np.zeros(n_cnn_preds, dtype=np.float32)
    for i, idx in enumerate(selected_pred_idx):
        if 0 <= idx < n_cnn_preds:
            confidence[idx] = float(score[sort_idx[i]])
    return {"gate": gate, "confidence": confidence}, k


def run_fold(fold_npz: Path, top_pct: float = 0.01,
             tp_ticks: float = 6.0, sl_ticks: float = 4.0,
             cancel_ms: float = 2000, max_hold_ms: float = 30000,
             order_type: str = "limit",
             ranker: str = "mfe", side_filter: str = "all"):
    fd = np.load(fold_npz, allow_pickle=True)
    oot_dates = fd["oot_dates"]
    if len(oot_dates) == 0:
        return None
    date_str = str(oot_dates[0])
    data, err = build_data_for_date(date_str)
    if err:
        return {"fold": fold_npz.name, "date": date_str, "error": err}
    exec_mlp, k_target = build_exec_mlp_gate(
        fold_npz, data["n_preds"], top_pct, ranker=ranker, side_filter=side_filter)
    signals = generate_signals(
        data,
        gate_threshold=0.0,           # let exec_mlp do all gating
        horizon=0,                     # 1s horizon for direction
        min_interval_ns=int(500 * 1e6),
        exec_mlp=exec_mlp,
        exec_mlp_gate_threshold=0.5,
    )
    if not signals:
        return {"fold": fold_npz.name, "date": date_str, "n_signals": 0,
                "n_trades": 0, "k_target": k_target}
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
        return {"fold": fold_npz.name, "date": date_str, "error": f"DBN missing: {e}"}
    trades = engine.simulate(
        signals, tp_ticks=tp_ticks, sl_ticks=sl_ticks, order_type=order_type
    )
    if not trades:
        return {"fold": fold_npz.name, "date": date_str, "n_signals": len(signals),
                "n_trades": 0, "k_target": k_target}
    pnls_ticks = np.array([t.pnl_ticks for t in trades])
    net_ticks = pnls_ticks - COMMISSION_TICKS
    return {
        "fold": fold_npz.name,
        "date": date_str,
        "n_signals": len(signals),
        "n_trades": len(trades),
        "k_target": k_target,
        "fill_rate": len(trades) / max(1, len(signals)),
        "avg_pnl_gross_t": float(np.mean(pnls_ticks)),
        "avg_pnl_net_t": float(np.mean(net_ticks)),
        "sum_pnl_gross_t": float(np.sum(pnls_ticks)),
        "sum_pnl_net_t": float(np.sum(net_ticks)),
        "wr": float(np.mean(pnls_ticks > 0)),
        "med_pnl_net_t": float(np.median(net_ticks)),
    }


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--ranker", default="mfe", choices=list(RANKER_COL.keys()))
    p.add_argument("--side", default="all", choices=["all", "long", "short"])
    p.add_argument("--order-type", default="limit", choices=["limit", "market", "chase"])
    p.add_argument("--top-pct", type=float, default=0.01)
    p.add_argument("--tp", type=float, default=6.0)
    p.add_argument("--sl", type=float, default=4.0)
    p.add_argument("--cancel-ms", type=float, default=2000)
    p.add_argument("--hold-ms", type=float, default=30000)
    args = p.parse_args()
    folds = sorted(V3_OUT.glob("fold_*_predictions.npz"))
    if not folds:
        print("FAIL: no v3 fold predictions in", V3_OUT)
        sys.exit(2)
    print(f"Running FIFO market replay on {len(folds)} folds | "
          f"ranker={args.ranker} side={args.side} order={args.order_type} "
          f"top={args.top_pct*100:.1f}% TP={args.tp}t SL={args.sl}t "
          f"cancel={args.cancel_ms:.0f}ms hold={args.hold_ms:.0f}ms\n")
    print(f"{'fold':>4} {'date':>10} {'sig':>5} {'trd':>5} {'fill%':>6} "
          f"{'WR':>5} {'g/t':>7} {'n/t':>7} {'med':>7}")
    print("-" * 76)
    rows = []
    for f in folds:
        r = run_fold(
            f, top_pct=args.top_pct, tp_ticks=args.tp, sl_ticks=args.sl,
            cancel_ms=args.cancel_ms, max_hold_ms=args.hold_ms,
            order_type=args.order_type, ranker=args.ranker,
            side_filter=args.side,
        )
        if r is None:
            continue
        fid = int(re.search(r"fold_(\d+)_", f.name).group(1))
        if "error" in r:
            print(f"{fid:>4} {r['date']:>10}  ERROR: {r['error']}")
            continue
        if r.get("n_trades", 0) == 0:
            print(f"{fid:>4} {r['date']:>10} {r.get('n_signals',0):>5} {0:>5} "
                  f"{'-':>6} {'-':>5} {'-':>7} {'-':>7} {'-':>7}")
            continue
        rows.append((fid, r))
        print(f"{fid:>4} {r['date']:>10} {r['n_signals']:>5} {r['n_trades']:>5} "
              f"{100*r['fill_rate']:>5.1f}% {r['wr']:.3f} "
              f"{r['avg_pnl_gross_t']:+7.3f} {r['avg_pnl_net_t']:+7.3f} "
              f"{r['med_pnl_net_t']:+7.3f}")

    if not rows:
        print("\nFAIL: no folds produced trades")
        sys.exit(1)
    print()
    n_folds = len(rows)
    sum_g = sum(r["sum_pnl_gross_t"] for _, r in rows)
    sum_n = sum(r["sum_pnl_net_t"] for _, r in rows)
    n_trd = sum(r["n_trades"] for _, r in rows)
    avg_g = sum_g / max(1, n_trd)
    avg_n = sum_n / max(1, n_trd)
    pos_n_folds = sum(1 for _, r in rows if r["sum_pnl_net_t"] > 0)
    nets = np.array([r["avg_pnl_net_t"] for _, r in rows])
    print(f"Folds with realized trades:        {n_folds}")
    print(f"Total realized trades:             {n_trd}")
    print(f"Sum gross ticks:                   {sum_g:+.2f}")
    print(f"Sum NET   ticks:                   {sum_n:+.2f}")
    print(f"Avg ticks/trade gross:             {avg_g:+.3f}")
    print(f"Avg ticks/trade NET:               {avg_n:+.3f}")
    print(f"Median fold avg-NET ticks/trade:   {float(np.median(nets)):+.3f}")
    print(f"% folds with positive sum_net:     {100*pos_n_folds/n_folds:.1f}%")
    avg_trades_per_day = n_trd / n_folds
    print(f"Avg trades/day (FIFO realized):    {avg_trades_per_day:.1f}")
    print(f"Implied $/day NET (1 contract):    ${avg_n * avg_trades_per_day * TICK_USD:+,.0f}")
    print()
    print("VERDICT (HC #265 honest readout):")
    if sum_n > 0 and pos_n_folds / n_folds >= 0.6:
        print("  PASS — FIFO-realized P&L is positive AND ≥60% of folds positive")
    elif sum_n > 0:
        print("  WEAK — FIFO sum positive but inconsistent across folds")
    else:
        print("  FAIL — FIFO realized P&L is non-positive after commission")


if __name__ == "__main__":
    main()
