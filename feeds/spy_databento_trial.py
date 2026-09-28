#!/usr/bin/env python3
"""spy_databento_trial.py — SPY MBO recorder using Databento (trial-tier).

Mirrors `live_trading_linux/mbo_recorder.py` for ES/Rithmic, but pulls from
Databento US Equities (XNAS.ITCH or DBEQ.BASIC). Produces the SAME 6-column
NPZ + JSONL fan-out that downstream training and paper-trading code expects.

Modes:
  --mode live          Subscribe to live MBO via databento.Live (needs key)
  --mode historical    Replay a historical window (uses trial credits)
  --mode fixture       Generate a synthetic 5-min sample (NO key needed)
                       — for smoke-testing the pipeline end-to-end.

Env vars:
  DATABENTO_API_KEY    Trial or paid key. If absent, only --mode fixture works.

Outputs:
  data/processed/spy_mbo_events/<YYYYMMDD>_mbo_events.npz
  live_trading_linux/logs/live_events_spy.jsonl

HC compliance:
  - 6-column schema identical to ES recorder (HC throughout CLAUDE.md).
  - NO training launches.
  - MLflow tagging optional via MLFLOW_TRACKING_URI.

Usage:
  python feeds/spy_databento_trial.py --mode fixture --minutes 5
  python feeds/spy_databento_trial.py --mode historical \
         --start 2026-06-03T13:30:00 --end 2026-06-03T14:30:00
  python feeds/spy_databento_trial.py --mode live --symbol SPY
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Make feeds/ importable
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from schema_adapter import SchemaAdapter, ACTION_ENCODING, SIDE_ENCODING

# --------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------
LVL3 = Path("/home/jupiter/Lvl3Quant")
OUT_DIR = LVL3 / "data" / "processed" / "spy_mbo_events"
LOG_DIR = LVL3 / "live_trading_linux" / "logs"
FANOUT_PATH = LOG_DIR / "live_events_spy.jsonl"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------
log = logging.getLogger("spy_feed")
log.setLevel(logging.INFO)
for _h in (logging.FileHandler(LOG_DIR / "spy_databento.log"),
           logging.StreamHandler(sys.stdout)):
    _h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(_h)

# --------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------
SPY_TICK_SIZE = 0.01           # USD per tick for SPY (NMS minimum)
DEFAULT_SYMBOL = "SPY"
DEFAULT_DATASET = "DBEQ.BASIC"  # Databento consolidated US equities. Free-tier
                                # candidates include this; XNAS.ITCH is Nasdaq
                                # native and requires more entitlements.


# --------------------------------------------------------------------
# Output writer — shared by all modes
# --------------------------------------------------------------------
class FeedWriter:
    """Buffers 6-col events, fan-outs JSONL, flushes NPZ per UTC date."""

    def __init__(self, out_dir: Path, fanout_path: Path, symbol: str):
        self.out_dir = out_dir
        self.fanout_path = fanout_path
        self.symbol = symbol
        self.events_buf: list[list[float]] = []
        self.ts_buf: list[int] = []
        self._fh = open(fanout_path, "a", buffering=1)
        self._fan_date = datetime.now(timezone.utc).strftime("%Y%m%d")

    def fanout(self, ts_ns: int, row: list, best_bid: float, best_ask: float,
               best_bid_size: int, best_ask_size: int):
        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        if today != self._fan_date:
            self._fh.close()
            self._fh = open(self.fanout_path, "w", buffering=1)
            self._fan_date = today
        # Mirror schema of live_trading_linux/logs/live_events.jsonl
        line = json.dumps({
            "timestamp_ns": ts_ns,
            "side": int(row[2]),
            "action": int(row[1]),
            "price_ticks": float(row[3]),  # relative to mid in ticks
            "size": int(round(math.expm1(row[4]))),
            "order_id": 0,
            "bbo": {
                "bid_price": best_bid, "ask_price": best_ask,
                "bid_size": best_bid_size, "ask_size": best_ask_size,
            },
            "symbol": self.symbol,
        }, separators=(",", ":"))
        self._fh.write(line + "\n")

    def append(self, row: list, ts_ns: int):
        self.events_buf.append(row)
        self.ts_buf.append(ts_ns)

    def flush(self) -> Path | None:
        if not self.events_buf:
            return None
        date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        out_path = self.out_dir / f"{date_str}_mbo_events.npz"
        events_arr = np.array(self.events_buf, dtype=np.float32)
        ts_arr = np.array(self.ts_buf, dtype=np.int64)
        if out_path.exists():
            ex = np.load(out_path, allow_pickle=True)
            events_arr = np.concatenate([ex["events"], events_arr])
            ts_arr = np.concatenate([ex["timestamps"], ts_arr])
        n = events_arr.shape[0]
        meta = json.dumps({
            "date": date_str,
            "instrument_id": 0,
            "symbol": self.symbol,
            "tick_size": SPY_TICK_SIZE,
            "n_events": n,
            "n_total_records": n,
            "feature_names": ["time_delta_log", "event_type_id", "side_id",
                              "price_rel_ticks", "qty_log", "spread_ticks"],
            "action_encoding": ACTION_ENCODING,
            "side_encoding": SIDE_ENCODING,
            "label_horizons": ["1s", "5s", "10s", "30s"],
            "source": "databento_trial",
            "vendor": "databento",
        })
        nan_labels = np.full(n, np.nan, dtype=np.float32)
        np.savez(out_path,
                 events=events_arr, timestamps=ts_arr,
                 labels_1s=nan_labels, labels_5s=nan_labels,
                 labels_10s=nan_labels, labels_30s=nan_labels,
                 metadata=np.array([meta]))
        log.info("Flushed %d events -> %s (total=%d)",
                 len(self.events_buf), out_path, n)
        self.events_buf.clear(); self.ts_buf.clear()
        return out_path

    def close(self):
        try:
            self._fh.close()
        except Exception:
            pass


# --------------------------------------------------------------------
# Fixture mode — generate a synthetic 5-minute SPY MBO sample
# (used when no DATABENTO_API_KEY is present; proves pipeline end-to-end)
# --------------------------------------------------------------------
def run_fixture(symbol: str, minutes: float, writer: FeedWriter,
                adapter: SchemaAdapter) -> dict:
    log.info("FIXTURE mode: synthesizing %.1f min of SPY MBO (no API key)",
             minutes)
    rng = random.Random(42)
    now_ns = int(time.time() * 1e9)
    bid = 580.10
    ask = 580.11
    last_ts = now_ns
    n_events = int(minutes * 60 * 50)  # ~50 events/sec ≈ moderate liquidity
    for i in range(n_events):
        # Time delta in microseconds, log-normal-ish
        dt_us = max(1, int(rng.expovariate(1 / 20_000)))
        last_ts += dt_us * 1000
        # 50% updates, 30% trades, 20% cancels
        roll = rng.random()
        if roll < 0.5:
            side = rng.choice(["B", "A"])
            action = "A"
            px = bid if side == "B" else ask
            sz = rng.randint(50, 500)
        elif roll < 0.8:
            side = rng.choice(["B", "A"])
            action = "T"
            px = bid if side == "A" else ask  # opposite side hit
            sz = rng.randint(10, 200)
        else:
            side = rng.choice(["B", "A"])
            action = "C"
            px = bid if side == "B" else ask
            sz = rng.randint(50, 300)

        # Drift the BBO every ~1000 events
        if i and i % 1000 == 0:
            drift = rng.choice([-0.01, 0.0, 0.0, 0.01])
            bid = round(bid + drift, 2)
            ask = round(bid + 0.01, 2)

        rec = {"ts_event": last_ts, "action": action, "side": side,
               "price": px, "size": sz}
        out = adapter.from_databento_mbo(rec)
        if out is None:
            continue
        writer.append(out.row, out.ts_ns)
        writer.fanout(out.ts_ns, out.row,
                      adapter.best_bid, adapter.best_ask,
                      adapter.best_bid_size, adapter.best_ask_size)

    out_path = writer.flush()
    return {"mode": "fixture", "n_events": n_events,
            "out_path": str(out_path) if out_path else None,
            "fanout_path": str(writer.fanout_path)}


# --------------------------------------------------------------------
# Historical mode — Databento Historical API (uses trial credits)
# --------------------------------------------------------------------
def run_historical(symbol: str, dataset: str, start: str, end: str,
                   writer: FeedWriter, adapter: SchemaAdapter) -> dict:
    api_key = os.environ.get("DATABENTO_API_KEY")
    if not api_key:
        raise RuntimeError("DATABENTO_API_KEY not set — cannot run historical mode")
    import databento as db
    log.info("HISTORICAL mode: %s %s [%s, %s]", dataset, symbol, start, end)
    client = db.Historical(api_key)
    # Try MBO first; if entitlement insufficient, caller can rerun with mbp_1
    data = client.timeseries.get_range(
        dataset=dataset, symbols=[symbol], schema="mbo",
        start=start, end=end, stype_in="raw_symbol",
    )
    n = 0
    for rec in data:
        out = adapter.from_databento_mbo(rec)
        if out is None:
            continue
        writer.append(out.row, out.ts_ns)
        writer.fanout(out.ts_ns, out.row,
                      adapter.best_bid, adapter.best_ask,
                      adapter.best_bid_size, adapter.best_ask_size)
        n += 1
    out_path = writer.flush()
    return {"mode": "historical", "n_events": n,
            "out_path": str(out_path) if out_path else None}


# --------------------------------------------------------------------
# Live mode — Databento Live API
# --------------------------------------------------------------------
def run_live(symbol: str, dataset: str, writer: FeedWriter,
             adapter: SchemaAdapter, flush_minutes: float) -> dict:
    api_key = os.environ.get("DATABENTO_API_KEY")
    if not api_key:
        raise RuntimeError("DATABENTO_API_KEY not set — cannot run live mode")
    import databento as db
    log.info("LIVE mode: %s %s, flush every %.1f min", dataset, symbol, flush_minutes)
    client = db.Live(key=api_key)
    client.subscribe(dataset=dataset, schema="mbo", stype_in="raw_symbol",
                     symbols=[symbol])
    stop = {"v": False}
    def _on_sigint(*_):
        log.info("SIGINT received; stopping live capture"); stop["v"] = True
    signal.signal(signal.SIGINT, _on_sigint)
    last_flush = time.time()
    n = 0
    for rec in client:
        if stop["v"]:
            break
        out = adapter.from_databento_mbo(rec)
        if out is None:
            continue
        writer.append(out.row, out.ts_ns)
        writer.fanout(out.ts_ns, out.row,
                      adapter.best_bid, adapter.best_ask,
                      adapter.best_bid_size, adapter.best_ask_size)
        n += 1
        if time.time() - last_flush >= flush_minutes * 60:
            writer.flush()
            last_flush = time.time()
    writer.flush()
    return {"mode": "live", "n_events": n}


# --------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="SPY Databento trial recorder")
    ap.add_argument("--mode", choices=["fixture", "historical", "live"],
                    default="fixture")
    ap.add_argument("--symbol", default=DEFAULT_SYMBOL)
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--start", help="ISO-8601 start time (historical mode)")
    ap.add_argument("--end", help="ISO-8601 end time (historical mode)")
    ap.add_argument("--minutes", type=float, default=5.0,
                    help="Fixture: minutes to synthesize")
    ap.add_argument("--flush-minutes", type=float, default=5.0,
                    help="Live: NPZ flush cadence")
    args = ap.parse_args()

    writer = FeedWriter(OUT_DIR, FANOUT_PATH, args.symbol)
    adapter = SchemaAdapter(tick_size=SPY_TICK_SIZE, instrument=args.symbol)

    try:
        if args.mode == "fixture":
            summary = run_fixture(args.symbol, args.minutes, writer, adapter)
        elif args.mode == "historical":
            if not (args.start and args.end):
                ap.error("--start and --end required for historical mode")
            summary = run_historical(args.symbol, args.dataset,
                                     args.start, args.end, writer, adapter)
        else:
            summary = run_live(args.symbol, args.dataset, writer, adapter,
                               args.flush_minutes)
    finally:
        writer.close()

    log.info("SUMMARY: %s", json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
