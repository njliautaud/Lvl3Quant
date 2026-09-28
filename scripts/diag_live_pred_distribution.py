"""
diag_live_pred_distribution.py — READ-ONLY diagnostic.

Goal: confirm whether the live CNN-Mamba v2 prediction distribution matches
the backtest distribution (training NPZ pred_log_ret_1s). Does NOT modify
the live trader; replays recent events through a fresh engine instance
on CPU to avoid GPU contention with the running shadow trader.

Usage (Razer):
    python3 diag_live_pred_distribution.py \\
        --events-file "C:\\Users\\claude\\Lvl3Quant\\live_trading\\logs\\live_events.jsonl" \\
        --tail-lines 80000 \\
        --device cpu \\
        --out-json "C:\\Users\\claude\\Lvl3Quant\\live_trading_linux\\logs\\diag_pred_dist.json"

Outputs:
    - stdout summary table (percentiles, gate pass-rates)
    - JSON file with full distribution stats
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import deque
from pathlib import Path

import numpy as np

# --- replicate exact same encode logic as paper_trading_v2_1s_short_top05.py ---

TICK_SIZE = 0.25
WINDOW_SIZE = 1000
STRIDE = 250
DEFAULT_WEIGHTS = Path(__file__).resolve().parent.parent / "output" / "cnn_mamba_v2_smart_v3_mar" / "fold_10_best.pt"
DEFAULT_STATS = Path(__file__).resolve().parent.parent / "output" / "cnn_mamba_v2_smart_v3_mar" / "fold_09_feature_stats.npz"


def encode_event(ev: dict, prev_ts_ns: int, best_bid: float, best_ask: float) -> tuple:
    """Returns (feat6, ts_ns, new_prev_ts_ns, best_bid, best_ask)."""
    ts_ns = int(ev.get("timestamp_ns", 0))
    bbo = ev.get("bbo") or {}
    if bbo:
        b = float(bbo.get("bid_price", 0) or 0)
        a = float(bbo.get("ask_price", 0) or 0)
        if b > 0:
            best_bid = b
        if a > 0:
            best_ask = a

    action = int(ev.get("action", 0) or 0)
    side_raw = int(ev.get("side", 0) or 0)
    price_ticks = float(ev.get("price_ticks", 0) or 0)
    size = int(ev.get("size", 1) or 1)

    etype = 3 if action >= 2 else 0
    price_rel_ticks = float(price_ticks)
    spread_ticks = ((best_ask - best_bid) / TICK_SIZE
                    if best_bid > 0 and best_ask > 0 else 0.0)
    delta_us = max(0, (ts_ns - prev_ts_ns) / 1000) if prev_ts_ns else 0
    time_delta_log = math.log1p(delta_us) if delta_us > 0 else 0.0
    qty_log = math.log(max(1, size))
    return (time_delta_log, etype, side_raw, price_rel_ticks, qty_log, spread_ticks), ts_ns, best_bid, best_ask


def tail_lines(path: Path, n: int) -> list:
    """Return last n non-empty lines of file. Naive but adequate for ~hundreds of MB."""
    with open(path, "rb") as fh:
        fh.seek(0, 2)
        end = fh.tell()
        chunk = 1 << 20
        data = b""
        while end > 0 and data.count(b"\n") <= n + 5:
            read_size = min(chunk, end)
            end -= read_size
            fh.seek(end)
            data = fh.read(read_size) + data
    lines = [ln for ln in data.splitlines() if ln.strip()]
    return [ln.decode("utf-8", errors="ignore") for ln in lines[-n:]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events-file", required=True)
    ap.add_argument("--tail-lines", type=int, default=80000)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--stats", default=str(DEFAULT_STATS))
    ap.add_argument("--out-json", default="diag_pred_dist.json")
    args = ap.parse_args()

    # imports are local so torch only loads when actually invoked
    # Insert root so `from live_trading.xxx import ...` works (shim in live_trading_linux re-exports).
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "live_trading"))
    from cnn_mamba_v2_inference import CNNMambaV2Inference
    from streaming_features_smart_v3 import StreamingFeaturesSmartV3, N_FEATURES

    print(f"[diag] loading engine on {args.device} ...", flush=True)
    engine = CNNMambaV2Inference(
        weights_path=args.weights,
        stats_path=args.stats,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        device=args.device,
    )
    streamer = StreamingFeaturesSmartV3()

    print(f"[diag] tailing {args.tail_lines} events from {args.events_file} ...", flush=True)
    raw_lines = tail_lines(Path(args.events_file), args.tail_lines)
    print(f"[diag] got {len(raw_lines)} raw lines", flush=True)

    pred_1s_list = []
    pred_5s_list = []
    pred_10s_list = []
    prev_ts_ns = 0
    best_bid = 0.0
    best_ask = 0.0
    n_events = 0
    n_decode_fail = 0
    sample_features = []

    for line in raw_lines:
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            n_decode_fail += 1
            continue
        (feat6, ts_ns, best_bid, best_ask) = encode_event(ev, prev_ts_ns, best_bid, best_ask)
        prev_ts_ns = ts_ns if ts_ns else prev_ts_ns
        feat_vec = streamer.update(*feat6)
        if len(sample_features) < 5:
            sample_features.append(list(map(float, feat_vec)))
        n_events += 1
        pred = engine.add_event(np.asarray(feat_vec, dtype=np.float32))
        if pred is None:
            continue
        pred_1s_list.append(float(pred["pred_1s"]))
        pred_5s_list.append(float(pred["pred_5s"]))
        pred_10s_list.append(float(pred["pred_10s"]))

    p1 = np.asarray(pred_1s_list, dtype=np.float64)
    p5 = np.asarray(pred_5s_list, dtype=np.float64)
    p10 = np.asarray(pred_10s_list, dtype=np.float64)

    def summarize(arr, name):
        if len(arr) == 0:
            return {"name": name, "n": 0}
        d = {
            "name": name,
            "n": int(len(arr)),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }
        for q in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99, 0.995, 0.999]:
            d[f"q{q:.4f}"] = float(np.quantile(arr, q))
        return d

    out = {
        "events_file": args.events_file,
        "tail_lines_requested": args.tail_lines,
        "raw_lines_read": len(raw_lines),
        "n_decode_fail": n_decode_fail,
        "n_events_processed": n_events,
        "n_predictions": len(p1),
        "device": args.device,
        "weights": args.weights,
        "stats": args.stats,
        "n_features": N_FEATURES,
        "window_size": WINDOW_SIZE,
        "stride": STRIDE,
        "skip_normalize": engine._skip_normalize,
        "sample_features_first5": sample_features,
        "pred_1s": summarize(p1, "pred_log_ret_1s"),
        "pred_5s": summarize(p5, "pred_log_ret_5s"),
        "pred_10s": summarize(p10, "pred_log_ret_10s"),
        "gate_check": {
            "global_floor_-0.6926": float((p1 <= -0.6926).mean()) if len(p1) else None,
            "pct_le_-0.10": float((p1 <= -0.10).mean()) if len(p1) else None,
            "pct_le_-0.01": float((p1 <= -0.01).mean()) if len(p1) else None,
            "pct_le_-0.001": float((p1 <= -0.001).mean()) if len(p1) else None,
            "pct_ge_+0.001": float((p1 >= 0.001).mean()) if len(p1) else None,
        },
    }
    Path(args.out_json).write_text(json.dumps(out, indent=2))

    print("\n=== LIVE PRED DISTRIBUTION ===")
    print(f"n_predictions={out['n_predictions']} from {out['n_events_processed']} events "
          f"(decode_fail={out['n_decode_fail']})")
    print(f"skip_normalize={out['skip_normalize']}  n_features={out['n_features']}")
    print()
    for tag, d in [("pred_1s", out["pred_1s"]), ("pred_5s", out["pred_5s"]),
                   ("pred_10s", out["pred_10s"])]:
        if d.get("n", 0) == 0:
            print(f"{tag}: NO PREDS"); continue
        print(f"{tag}: n={d['n']} mean={d['mean']:+.6f} std={d['std']:+.6f} "
              f"min={d['min']:+.4f} max={d['max']:+.4f}")
        print(f"  q .1%={d['q0.0010']:+.4f} q .5%={d['q0.0050']:+.4f} q1%={d['q0.0100']:+.4f} "
              f"q5%={d['q0.0500']:+.4f} q50%={d['q0.5000']:+.4f} q95%={d['q0.9500']:+.4f} "
              f"q99%={d['q0.9900']:+.4f} q99.5%={d['q0.9950']:+.4f} q99.9%={d['q0.9990']:+.4f}")
    print()
    g = out["gate_check"]
    print("GATE PASS RATES (negative-pred side):")
    print(f"  pred_1s ≤ -0.6926 (live gate floor): {100*g['global_floor_-0.6926']:.4f}%  "
          f"(backtest reference: 0.4651%)")
    print(f"  pred_1s ≤ -0.10:                     {100*g['pct_le_-0.10']:.4f}%  "
          f"(backtest reference: 33.33%)")
    print(f"  pred_1s ≤ -0.01:                     {100*g['pct_le_-0.01']:.4f}%  "
          f"(backtest reference: 46.05%)")
    print(f"  pred_1s ≤ -0.001:                    {100*g['pct_le_-0.001']:.4f}%  "
          f"(backtest reference: 47.93%)")
    print(f"  pred_1s ≥ +0.001:                    {100*g['pct_ge_+0.001']:.4f}%  "
          f"(backtest reference: 51.79%)")
    print()
    print(f"[diag] wrote {args.out_json}")


if __name__ == "__main__":
    main()
