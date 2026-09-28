#!/usr/bin/env python3
"""spy_recorder.py - Subsecond SPY market-data recorder for the SPY pivot.

Produces:
  1. Daily NPZ files at data/processed/spy_mbo_events/YYYYMMDD_spy_events.npz
     in the SAME 6-col float32 schema as the ES mbo_recorder.py:
       [time_delta_log, event_type_id, side_id, price_rel_ticks, qty_log, spread_ticks]
     (tick size = $0.01 for SPY shares, NOT $0.25 like ES.)
  2. A live JSONL fan-out at logs/spy_live_events.jsonl that downstream paper
     traders subscribe to. One line per event (BBO update or trade).

Data sources (auto-detect, in priority order):
  * Alpaca free real-time IEX feed (when ALPACA_KEY_ID + ALPACA_SECRET_KEY are set
    in env or live_trading_linux/.env). Sub-millisecond timestamps, real-time.
  * yfinance polling fallback (~1s, no key needed). Coarser but works zero-config
    so the pipeline can START running today before user has Alpaca creds.

The SPY pivot premise (HC #518 R2): same MBO microstructure features that
produced ES alpha translate to SPY's order book. Cost wall is much lower
on equities ($0.005/share commission vs ES's 0.376-tick wall), so smaller
gross edges can clear cost.

Usage:
    python3 spy_recorder.py --source auto --flush-minutes 5
    python3 spy_recorder.py --source yfinance      # force polling fallback
    python3 spy_recorder.py --source alpaca        # force Alpaca real-time
"""
from __future__ import annotations
import argparse
import asyncio
import json
import logging
import math
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from dotenv import load_dotenv
    _ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
    if _ENV_PATH.exists():
        load_dotenv(_ENV_PATH)
except ImportError:
    pass

# ---- Paths ----
_ROOT = Path("/home/jupiter/Lvl3Quant")
OUT_DIR = _ROOT / "data" / "processed" / "spy_mbo_events"
LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)
FANOUT_PATH = LOG_DIR / "spy_live_events.jsonl"

# ---- Constants for SPY equities ----
TICK_SIZE = 0.01          # 1 cent
LOT_SIZE = 100            # round lot
SYMBOL = "SPY"

# ---- Logging ----
log = logging.getLogger("spy_recorder")
log.setLevel(logging.INFO)
if not log.handlers:
    for h in [logging.FileHandler(LOG_DIR / "spy_recorder.log"),
              logging.StreamHandler()]:
        h.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s [spy_recorder] %(message)s"))
        log.addHandler(h)


class SPYRecorder:
    """Buffers live SPY events and flushes to daily NPZ + live JSONL fan-out.

    Schema matches ES mbo_recorder.py exactly so downstream code that
    consumes the JSONL line shape doesn't need to change.
    """

    def __init__(self, flush_minutes: float = 5.0):
        self.flush_minutes = flush_minutes
        self.best_bid = self.best_ask = self.mid_price = 0.0
        self.best_bid_size = self.best_ask_size = 0
        self.prev_ts_ns = 0
        self.events_buf: list[list[float]] = []
        self.ts_buf: list[int] = []
        self._stop = asyncio.Event()
        self._fanout_path = FANOUT_PATH
        self._fanout_path.parent.mkdir(parents=True, exist_ok=True)
        self._fanout_fh = open(self._fanout_path, "a", buffering=1)
        self._fanout_date = datetime.now(timezone.utc).strftime("%Y%m%d")
        self._n_events_today = 0
        self._n_trades_today = 0
        log.info("Fan-out JSONL: %s", self._fanout_path)
        log.info("NPZ output: %s", OUT_DIR)

    # -----------------------------------------------------------------
    # Fan-out + encoding
    # -----------------------------------------------------------------
    def _fanout_rotate_if_needed(self):
        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        if today != self._fanout_date:
            self._fanout_fh.close()
            self._fanout_fh = open(self._fanout_path, "w", buffering=1)
            self._fanout_date = today
            self._n_events_today = 0
            self._n_trades_today = 0
            log.info("Fan-out JSONL rotated for %s", today)

    def _fanout_write(self, ts_ns: int, side: int, action: int,
                      price_ticks: float, size: int, order_id: int = 0):
        self._fanout_rotate_if_needed()
        line = json.dumps({
            "timestamp_ns": ts_ns,
            "symbol": SYMBOL,
            "side": side,
            "action": action,
            "price_ticks": price_ticks,
            "size": size,
            "order_id": order_id,
            "bbo": {
                "bid_price": self.best_bid,
                "ask_price": self.best_ask,
                "bid_size": self.best_bid_size,
                "ask_size": self.best_ask_size,
            },
            "tick_size": TICK_SIZE,
        }, separators=(",", ":"))
        self._fanout_fh.write(line + "\n")
        self._fanout_fh.flush()

    def _spread_ticks(self) -> float:
        if self.best_bid > 0 and self.best_ask > 0:
            return (self.best_ask - self.best_bid) / TICK_SIZE
        return 0.0

    def _encode(self, ts_ns: int, etype: float, side: float,
                price: float, qty: int):
        delta_us = max(0, (ts_ns - self.prev_ts_ns) / 1000) if self.prev_ts_ns else 0
        self.prev_ts_ns = ts_ns
        price_rel = (price - self.mid_price) / TICK_SIZE if (
            self.mid_price > 0 and price > 0) else 0.0
        self.events_buf.append([
            math.log1p(delta_us) if delta_us > 0 else 0.0,
            etype, side, price_rel, math.log(max(1, qty)), self._spread_ticks(),
        ])
        self.ts_buf.append(ts_ns)
        self._n_events_today += 1

    # -----------------------------------------------------------------
    # Event ingest (called by source adapter)
    # -----------------------------------------------------------------
    def on_quote(self, ts_ns: int, bid: float, bid_sz: int,
                 ask: float, ask_sz: int):
        """Process a quote (BBO update). Emits up to two events: bid + ask."""
        bid_changed = ask_changed = False
        if bid and bid > 0:
            if bid != self.best_bid or bid_sz != self.best_bid_size:
                self.best_bid = bid
                self.best_bid_size = bid_sz
                bid_changed = True
        if ask and ask > 0:
            if ask != self.best_ask or ask_sz != self.best_ask_size:
                self.best_ask = ask
                self.best_ask_size = ask_sz
                ask_changed = True
        if self.best_bid > 0 and self.best_ask > 0:
            self.mid_price = (self.best_bid + self.best_ask) / 2.0
        if bid_changed:
            self._encode(ts_ns, 0.0, 0.0, self.best_bid, self.best_bid_size)
            self._fanout_write(ts_ns, 0, 0,
                               self.best_bid / TICK_SIZE, self.best_bid_size)
        if ask_changed:
            self._encode(ts_ns, 0.0, 1.0, self.best_ask, self.best_ask_size)
            self._fanout_write(ts_ns, 1, 0,
                               self.best_ask / TICK_SIZE, self.best_ask_size)

    def on_trade(self, ts_ns: int, price: float, size: int,
                 aggressor_side: Optional[int] = None):
        """Process a trade. aggressor_side: 0=buyer-initiated, 1=seller-initiated,
        None=unknown (infer from price vs mid)."""
        if aggressor_side is None:
            if self.mid_price > 0:
                aggressor_side = 0 if price >= self.mid_price else 1
            else:
                aggressor_side = 2
        # Match ES schema: side 1.0 = buyer aggressor, 0.0 = seller aggressor
        side_f = 1.0 if aggressor_side == 0 else (
            0.0 if aggressor_side == 1 else 2.0)
        self._encode(ts_ns, 3.0, side_f, price, size)
        self._fanout_write(ts_ns, int(aggressor_side), 3,
                           price / TICK_SIZE, size)
        self._n_trades_today += 1

    # -----------------------------------------------------------------
    # Flush + lifecycle
    # -----------------------------------------------------------------
    def flush(self):
        if not self.events_buf:
            return
        date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        out_path = OUT_DIR / f"{date_str}_spy_events.npz"
        events_arr = np.array(self.events_buf, dtype=np.float32)
        ts_arr = np.array(self.ts_buf, dtype=np.int64)
        if out_path.exists():
            try:
                ex = np.load(out_path, allow_pickle=True)
                events_arr = np.concatenate([ex["events"], events_arr])
                ts_arr = np.concatenate([ex["timestamps"], ts_arr])
            except Exception as e:
                log.warning("Could not append to %s (%s) — overwriting.",
                            out_path, e)
        n = events_arr.shape[0]
        meta = json.dumps({
            "date": date_str, "symbol": SYMBOL, "tick_size": TICK_SIZE,
            "lot_size": LOT_SIZE, "venue": "iex_free_or_yfinance",
            "n_events": n, "n_total_records": n,
            "feature_names": [
                "time_delta_log", "event_type_id", "side_id",
                "price_rel_ticks", "qty_log", "spread_ticks"],
            "action_encoding": {"A": 0, "C": 1, "M": 2, "T": 3, "F": 4},
            "side_encoding": {"B": 0, "A": 1, "N": 2},
            "label_horizons": ["1s", "5s", "10s", "30s"],
            "source": "spy_recorder_live",
        })
        nan_labels = np.full(n, np.nan, dtype=np.float32)
        np.savez(out_path, events=events_arr, timestamps=ts_arr,
                 labels_1s=nan_labels, labels_5s=nan_labels,
                 labels_10s=nan_labels, labels_30s=nan_labels,
                 metadata=np.array([meta]))
        log.info("Flushed %d events -> %s (total=%d, trades_today=%d)",
                 len(self.events_buf), out_path, n, self._n_trades_today)
        self.events_buf.clear()
        self.ts_buf.clear()

    async def _flush_loop(self):
        while not self._stop.is_set():
            try:
                await asyncio.wait_for(
                    self._stop.wait(),
                    timeout=self.flush_minutes * 60)
                break
            except asyncio.TimeoutError:
                self.flush()

    def request_stop(self):
        self._stop.set()
        try:
            self.flush()
            self._fanout_fh.close()
        except Exception:
            pass


# =====================================================================
# Source adapter: Alpaca free real-time IEX
# =====================================================================
async def run_alpaca(rec: SPYRecorder):
    """Connect to Alpaca's free IEX real-time feed via alpaca-py SDK.

    Requires ALPACA_KEY_ID and ALPACA_SECRET_KEY in env or .env.
    Alpaca's free tier ('iex' feed) gives real-time IEX-only quotes and
    trades — that's only ~2-3% of total SPY volume but it IS subsecond
    real-time, which is what the pipeline needs to start ingesting today.
    Upgrade path: switch feed='sip' (paid) for full consolidated NBBO.
    """
    try:
        from alpaca.data.live import StockDataStream
    except ImportError:
        log.error("alpaca-py not installed. pip install alpaca-py")
        return False

    key = os.environ.get("ALPACA_KEY_ID") or os.environ.get("APCA_API_KEY_ID")
    sec = os.environ.get("ALPACA_SECRET_KEY") or os.environ.get("APCA_API_SECRET_KEY")
    if not key or not sec:
        log.warning("ALPACA_KEY_ID / ALPACA_SECRET_KEY not set — cannot use Alpaca source.")
        return False

    log.info("Connecting to Alpaca free IEX real-time stream for SPY...")
    stream = StockDataStream(key, sec, feed="iex")

    async def on_quote(q):
        # alpaca-py timestamp is datetime in UTC
        ts_ns = int(q.timestamp.timestamp() * 1_000_000_000)
        rec.on_quote(ts_ns,
                     bid=float(q.bid_price),
                     bid_sz=int(q.bid_size),
                     ask=float(q.ask_price),
                     ask_sz=int(q.ask_size))

    async def on_trade(t):
        ts_ns = int(t.timestamp.timestamp() * 1_000_000_000)
        rec.on_trade(ts_ns, price=float(t.price), size=int(t.size),
                     aggressor_side=None)

    stream.subscribe_quotes(on_quote, SYMBOL)
    stream.subscribe_trades(on_trade, SYMBOL)

    flush_task = asyncio.create_task(rec._flush_loop())
    try:
        # alpaca-py StockDataStream._run_forever is a coroutine
        await stream._run_forever()
    except asyncio.CancelledError:
        pass
    except Exception as e:
        log.error("Alpaca stream error: %s", e)
    finally:
        flush_task.cancel()
        try:
            await stream.close()
        except Exception:
            pass
    return True


# =====================================================================
# Source adapter: yfinance polling fallback (~1s)
# =====================================================================
async def run_yfinance(rec: SPYRecorder, poll_seconds: float = 1.0):
    """yfinance-based polling fallback. ~1s granularity. No API key.

    Pulls SPY's last-trade price via yfinance.Ticker.fast_info / .info every
    poll_seconds. Quotes are NOT available at sub-second from yfinance, so
    bid/ask are inferred (last price ± 1 cent assumed spread). This is a
    LOW-FIDELITY bootstrap source — purpose is to keep the pipeline running
    while user signs up for Alpaca / Databento.
    """
    try:
        import yfinance as yf
    except ImportError:
        log.error("yfinance not installed. pip install yfinance")
        return False

    log.info("yfinance polling source @ %.2fs (low-fidelity bootstrap).",
             poll_seconds)
    log.info("For real subsecond data set ALPACA_KEY_ID / ALPACA_SECRET_KEY "
             "and rerun with --source alpaca.")
    ticker = yf.Ticker(SYMBOL)
    flush_task = asyncio.create_task(rec._flush_loop())
    last_price = None
    last_volume = None
    try:
        while not rec._stop.is_set():
            try:
                fi = ticker.fast_info
                price = float(fi.get("last_price") or fi.get("lastPrice") or 0)
                day_vol = int(fi.get("last_volume") or fi.get("dayVolume") or 0)
            except Exception as e:
                log.debug("yfinance poll error: %s", e)
                await asyncio.sleep(poll_seconds)
                continue

            ts_ns = int(time.time() * 1_000_000_000)
            if price > 0:
                # Synthesize NBBO ±1 cent (SPY usually 1-cent wide)
                bid = round(price - 0.005, 2)
                ask = round(price + 0.005, 2)
                rec.on_quote(ts_ns, bid=bid, bid_sz=100, ask=ask, ask_sz=100)
                # If volume bumped since last poll, emit a synthetic trade
                if last_volume is not None and day_vol > last_volume:
                    vol_inc = min(day_vol - last_volume, 100_000)
                    rec.on_trade(ts_ns, price=price, size=vol_inc,
                                 aggressor_side=None)
                last_price = price
                last_volume = day_vol
            await asyncio.sleep(poll_seconds)
    finally:
        flush_task.cancel()
    return True


# =====================================================================
# Entry point
# =====================================================================
async def main_async(args):
    rec = SPYRecorder(flush_minutes=args.flush_minutes)

    try:
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, rec.request_stop)
    except NotImplementedError:
        pass

    src = args.source
    if src == "auto":
        if (os.environ.get("ALPACA_KEY_ID") or
                os.environ.get("APCA_API_KEY_ID")):
            src = "alpaca"
        else:
            src = "yfinance"
        log.info("Auto-detected source: %s", src)

    if src == "alpaca":
        ok = await run_alpaca(rec)
        if not ok:
            log.warning("Alpaca path failed — falling back to yfinance.")
            await run_yfinance(rec, args.poll_seconds)
    elif src == "yfinance":
        await run_yfinance(rec, args.poll_seconds)
    else:
        log.error("Unknown source: %s", src)
        return

    rec.flush()
    log.info("Recorder stopped.")


def main():
    p = argparse.ArgumentParser(description="SPY subsecond market-data recorder")
    p.add_argument("--source", default="auto",
                   choices=["auto", "alpaca", "yfinance"],
                   help="Data source (auto: alpaca if creds set, else yfinance)")
    p.add_argument("--flush-minutes", type=float, default=5.0,
                   help="NPZ flush interval (default 5min)")
    p.add_argument("--poll-seconds", type=float, default=1.0,
                   help="yfinance polling interval (default 1s)")
    args = p.parse_args()
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
