#!/usr/bin/env python3
"""mbo_recorder.py — Live MBO event recorder via Rithmic.
Encodes BBO+Trade events into the same 6-col float32 NPZ format as training data.
Usage: python3 mbo_recorder.py --symbol ESM6 --exchange CME --flush-minutes 5
"""
from __future__ import annotations
import argparse, asyncio, json, logging, math, os, platform, signal, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rithmic_client import RithmicClient, BBOEvent, TradeEvent

ENV_PATH = Path(__file__).resolve().parent / ".env"
if platform.system() == "Windows":
    OUT_DIR = Path(r"C:\Users\claude\Lvl3Quant\data\processed\mbo_events")
else:
    OUT_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/mbo_events")
LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
TICK_SIZE = 0.25

# Fan-out JSONL path — platform-aware
if platform.system() == "Windows":
    FANOUT_PATH = Path(r"C:\Users\claude\Lvl3Quant\live_trading\logs\live_events.jsonl")
else:
    FANOUT_PATH = Path(__file__).resolve().parent / "logs" / "live_events.jsonl"

log = logging.getLogger("mbo_recorder")
log.setLevel(logging.INFO)
for h in [logging.FileHandler(LOG_DIR / "mbo_recorder.log"), logging.StreamHandler()]:
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(h)


class MBORecorder:
    """Buffers live Rithmic events and flushes to daily NPZ files."""

    def __init__(self, symbol: str, exchange: str, flush_minutes: float = 5.0):
        self.symbol, self.exchange, self.flush_minutes = symbol, exchange, flush_minutes
        self.best_bid = self.best_ask = self.mid_price = 0.0
        self.best_bid_size = self.best_ask_size = 0
        self.prev_ts_ns: int = 0
        self.events_buf: list[list[float]] = []
        self.ts_buf: list[int] = []
        self._client: RithmicClient | None = None
        self._stop = asyncio.Event()
        # Fan-out JSONL broadcaster
        self._fanout_path = FANOUT_PATH
        self._fanout_path.parent.mkdir(parents=True, exist_ok=True)
        self._fanout_fh = open(self._fanout_path, "a", buffering=1)
        self._fanout_date = datetime.now(timezone.utc).strftime("%Y%m%d")
        log.info("Fan-out JSONL: %s", self._fanout_path)

    def _fanout_rotate_if_needed(self):
        """Truncate/rotate fan-out JSONL at midnight UTC."""
        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        if today != self._fanout_date:
            self._fanout_fh.close()
            # Truncate — open in write mode then switch back to append
            self._fanout_fh = open(self._fanout_path, "w", buffering=1)
            self._fanout_date = today
            log.info("Fan-out JSONL rotated for %s", today)

    def _fanout_write(self, ts_ns: int, side: int, action: int,
                      price_ticks: float, size: int, order_id: int = 0):
        """Write one event as JSON line to the fan-out file (< 1ms)."""
        self._fanout_rotate_if_needed()
        line = json.dumps({
            "timestamp_ns": ts_ns,
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
        }, separators=(",", ":"))
        self._fanout_fh.write(line + "\n")
        self._fanout_fh.flush()

    def _spread_ticks(self) -> float:
        if self.best_bid > 0 and self.best_ask > 0:
            return (self.best_ask - self.best_bid) / TICK_SIZE
        return 0.0

    def _encode(self, ts_ns: int, etype: float, side: float, price: float, qty: int):
        delta_us = max(0, (ts_ns - self.prev_ts_ns) / 1000) if self.prev_ts_ns else 0
        self.prev_ts_ns = ts_ns
        price_rel = (price - self.mid_price) / TICK_SIZE if self.mid_price > 0 and price > 0 else 0.0
        self.events_buf.append([
            math.log1p(delta_us) if delta_us > 0 else 0.0,
            etype, side, price_rel, math.log(max(1, qty)), self._spread_ticks(),
        ])
        self.ts_buf.append(ts_ns)

    async def on_md(self, ev: BBOEvent | TradeEvent) -> None:
        ts_ns = int(ev.ssboe * 1_000_000_000 + ev.usecs * 1000)
        if isinstance(ev, BBOEvent):
            if ev.has_bid and ev.bid_price > 0:
                self.best_bid = ev.bid_price
                self.best_bid_size = ev.bid_size
            if ev.has_ask and ev.ask_price > 0:
                self.best_ask = ev.ask_price
                self.best_ask_size = ev.ask_size
            if self.best_bid > 0 and self.best_ask > 0:
                self.mid_price = (self.best_bid + self.best_ask) / 2.0
            if ev.has_bid:
                self._encode(ts_ns, 0.0, 0.0, ev.bid_price, ev.bid_size)
                self._fanout_write(ts_ns, 0, 0,
                                   ev.bid_price / TICK_SIZE, ev.bid_size)
            if ev.has_ask:
                self._encode(ts_ns, 0.0, 1.0, ev.ask_price, ev.ask_size)
                self._fanout_write(ts_ns, 1, 0,
                                   ev.ask_price / TICK_SIZE, ev.ask_size)
        elif isinstance(ev, TradeEvent):
            side = {1: 1.0, 2: 0.0}.get(ev.aggressor, 2.0)
            self._encode(ts_ns, 3.0, side, ev.trade_price, ev.trade_size)
            self._fanout_write(ts_ns, int(side), 3,
                               ev.trade_price / TICK_SIZE, ev.trade_size)

    def flush(self) -> None:
        if not self.events_buf:
            return
        date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        out_path = OUT_DIR / f"{date_str}_mbo_events.npz"
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        events_arr = np.array(self.events_buf, dtype=np.float32)
        ts_arr = np.array(self.ts_buf, dtype=np.int64)
        if out_path.exists():
            ex = np.load(out_path, allow_pickle=True)
            events_arr = np.concatenate([ex["events"], events_arr])
            ts_arr = np.concatenate([ex["timestamps"], ts_arr])
        n = events_arr.shape[0]
        meta = json.dumps({
            "date": date_str, "instrument_id": 0, "tick_size": TICK_SIZE,
            "n_events": n, "n_total_records": n,
            "feature_names": ["time_delta_log", "event_type_id", "side_id",
                              "price_rel_ticks", "qty_log", "spread_ticks"],
            "action_encoding": {"A": 0, "C": 1, "M": 2, "T": 3, "F": 4},
            "side_encoding": {"B": 0, "A": 1, "N": 2},
            "label_horizons": ["1s", "5s", "10s", "30s"],
            "source": "mbo_recorder_live", "symbol": self.symbol,
        })
        nan_labels = np.full(n, np.nan, dtype=np.float32)
        np.savez(out_path, events=events_arr, timestamps=ts_arr,
                 labels_1s=nan_labels, labels_5s=nan_labels,
                 labels_10s=nan_labels, labels_30s=nan_labels,
                 metadata=np.array([meta]))
        log.info("Flushed %d events -> %s (total=%d)", len(self.events_buf), out_path, n)
        self.events_buf.clear()
        self.ts_buf.clear()

    async def _flush_loop(self):
        while not self._stop.is_set():
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.flush_minutes * 60)
                break
            except asyncio.TimeoutError:
                self.flush()

    async def run(self):
        load_dotenv(ENV_PATH)
        # ---- Singleton lock: prevent duplicate instances ----
        lock_path = LOG_DIR / "mbo_recorder.lock"
        if lock_path.exists():
            try:
                old_pid = int(lock_path.read_text().strip())
                # Check if process is still alive
                import signal as _signal_mod
                os.kill(old_pid, 0)  # Raises OSError if dead
                log.error("ANOTHER MBO RECORDER IS ALREADY RUNNING (PID %d). "
                          "Only ONE Rithmic MD connection allowed. Exiting.", old_pid)
                sys.exit(1)
            except (OSError, ValueError):
                log.info("Stale lock file (PID gone). Taking over.")
        lock_path.write_text(str(os.getpid()))
        log.info("Singleton lock acquired (PID %d) at %s", os.getpid(), lock_path)
        try:
            await self._run_inner()
        finally:
            try:
                lock_path.unlink()
            except Exception:
                pass

    async def _run_inner(self):
        log.info("MBO recorder: %s @ %s, flush every %.1fmin", self.symbol, self.exchange, self.flush_minutes)
        reconnect_delay = 1.0
        while not self._stop.is_set():
            try:
                self._client = RithmicClient()
                await self._client.connect(md_only=True)
                self._client.set_md_callback(self.on_md)
                await self._client.subscribe_md(self.symbol, self.exchange)
                log.info("Connected and subscribed. Recording...")
                connect_time = asyncio.get_event_loop().time()
                flush_task = asyncio.create_task(self._flush_loop())
                while not self._stop.is_set():
                    await asyncio.sleep(1)
                    if self._client._ws_md is None or not RithmicClient._is_open(self._client._ws_md):
                        log.warning("MD socket closed, reconnecting...")
                        break
                flush_task.cancel()
                try:
                    await flush_task
                except asyncio.CancelledError:
                    pass
                # If connection lasted < 30s, treat as failed — apply backoff
                uptime = asyncio.get_event_loop().time() - connect_time
                if uptime < 30:
                    log.warning("Connection only lasted %.1fs — applying backoff (%.0fs)", uptime, reconnect_delay)
                    await asyncio.sleep(reconnect_delay)
                    reconnect_delay = min(reconnect_delay * 2, 120)
                else:
                    reconnect_delay = 1.0  # Reset backoff only on stable connections
            except Exception as e:
                log.error("Connection error: %s — retry in %.0fs", e, reconnect_delay)
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, 120)
            finally:
                self.flush()
                if self._client:
                    try:
                        await self._client.disconnect()
                    except Exception:
                        pass
        log.info("MBO recorder stopped.")

    def request_stop(self):
        self._stop.set()
        try:
            self._fanout_fh.close()
        except Exception:
            pass


def main():
    p = argparse.ArgumentParser(description="Live MBO event recorder")
    p.add_argument("--symbol", default="ESM6", help="Rithmic symbol (default: ESM6)")
    p.add_argument("--exchange", default="CME", help="Exchange (default: CME)")
    p.add_argument("--flush-minutes", type=float, default=5.0, help="Flush interval in minutes")
    a = p.parse_args()
    rec = MBORecorder(a.symbol, a.exchange, a.flush_minutes)
    loop = asyncio.new_event_loop()
    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, rec.request_stop)
    except NotImplementedError:
        # Windows doesn't support add_signal_handler
        pass
    try:
        loop.run_until_complete(rec.run())
    except KeyboardInterrupt:
        rec.request_stop()
    finally:
        loop.close()


if __name__ == "__main__":
    main()
