#!/usr/bin/env python3
"""
latency_tracker.py — End-to-end latency instrumentation for live inference (HC #148).

Tracks per-event timing through the full pipeline:
  T0: MBO event received (wall clock at processing start)
  T1: Feature computation complete
  T2: CNN-Mamba v2 inference complete (None if no prediction this step)
  T3: PatchTST inference complete (None if no prediction / not in confluence mode)
  T4: Execution decision complete (entry/exit/hold decision made)

Reports rolling p50/p95/p99 latencies every N events.

Usage:
    tracker = LatencyTracker(log_interval=10000)

    # In hot loop:
    tracker.event_start()           # T0
    ... features ...
    tracker.features_done()         # T1
    ... cnn_mamba inference ...
    tracker.cnn_mamba_done()        # T2
    ... patchtst inference ...
    tracker.patchtst_done()         # T3
    ... execution decision ...
    tracker.decision_done()         # T4
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger("latency")


@dataclass
class LatencySnapshot:
    """Single event's timing breakdown (microseconds)."""
    features_us: float = 0.0
    cnn_mamba_us: float = 0.0    # 0 if no prediction this step
    patchtst_us: float = 0.0     # 0 if no prediction or not confluence
    decision_us: float = 0.0     # 0 if no prediction this step
    total_us: float = 0.0        # T4 - T0 (or T1 - T0 if no prediction)
    had_prediction: bool = False


class LatencyTracker:
    """
    End-to-end latency tracker for the live inference pipeline.

    Measures wall-clock time through each stage using time.perf_counter_ns()
    for nanosecond precision.
    """

    def __init__(
        self,
        log_interval: int = 10_000,     # Log summary every N events
        history_size: int = 100_000,     # Rolling window for percentiles
        log_dir: Optional[Path] = None,
    ):
        self.log_interval = log_interval
        self.history_size = history_size

        # Rolling buffers (microseconds)
        self._feat_latencies = deque(maxlen=history_size)
        self._pred_latencies = deque(maxlen=history_size)  # only when prediction made
        self._cnn_latencies = deque(maxlen=history_size)
        self._ptst_latencies = deque(maxlen=history_size)
        self._decision_latencies = deque(maxlen=history_size)
        self._total_latencies = deque(maxlen=history_size)  # all events
        self._total_pred_latencies = deque(maxlen=history_size)  # only prediction events

        # Per-event timestamps (perf_counter_ns)
        self._t0: int = 0
        self._t1: int = 0
        self._t2: int = 0
        self._t3: int = 0
        self._t4: int = 0
        self._has_pred: bool = False

        # Counters
        self.n_events: int = 0
        self.n_predictions: int = 0
        self._last_report_events: int = 0

        # CSV log
        self._csv_path = None
        if log_dir:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            from datetime import datetime
            ts = datetime.now().strftime("%Y%m%d_%H%M")
            self._csv_path = log_dir / f"latency_{ts}.csv"
            with open(self._csv_path, "w") as f:
                f.write("event_num,features_us,cnn_mamba_us,patchtst_us,decision_us,total_us,had_pred\n")

        log.info("LatencyTracker initialized: interval=%d events, history=%d",
                 log_interval, history_size)

    def event_start(self):
        """Mark T0: MBO event processing begins."""
        self._t0 = time.perf_counter_ns()
        self._t1 = 0
        self._t2 = 0
        self._t3 = 0
        self._t4 = 0
        self._has_pred = False

    def features_done(self):
        """Mark T1: Feature computation complete."""
        self._t1 = time.perf_counter_ns()

    def cnn_mamba_done(self, had_prediction: bool = False):
        """Mark T2: CNN-Mamba inference complete."""
        self._t2 = time.perf_counter_ns()
        if had_prediction:
            self._has_pred = True

    def patchtst_done(self):
        """Mark T3: PatchTST inference complete."""
        self._t3 = time.perf_counter_ns()

    def decision_done(self):
        """Mark T4: Execution decision complete. Finalizes this event's timing."""
        self._t4 = time.perf_counter_ns()
        self._record()

    def event_done_no_prediction(self):
        """Shortcut: event processed but no model prediction (stride not reached)."""
        self._t4 = time.perf_counter_ns()
        self._has_pred = False
        self._record()

    def _record(self):
        """Record latencies and check if summary is due."""
        self.n_events += 1

        # Convert ns → us
        feat_us = (self._t1 - self._t0) / 1000.0 if self._t1 > 0 else 0.0

        self._feat_latencies.append(feat_us)

        if self._has_pred:
            self.n_predictions += 1
            cnn_us = (self._t2 - self._t1) / 1000.0 if self._t2 > 0 else 0.0
            ptst_us = (self._t3 - self._t2) / 1000.0 if self._t3 > 0 else 0.0
            decision_us = (self._t4 - (self._t3 if self._t3 > 0 else self._t2)) / 1000.0
            total_us = (self._t4 - self._t0) / 1000.0

            self._cnn_latencies.append(cnn_us)
            if self._t3 > 0:
                self._ptst_latencies.append(ptst_us)
            self._decision_latencies.append(decision_us)
            self._total_pred_latencies.append(total_us)

        total_us = (self._t4 - self._t0) / 1000.0 if self._t4 > 0 else feat_us
        self._total_latencies.append(total_us)

        # Write to CSV (sample every 100th event to avoid I/O overhead)
        if self._csv_path and self.n_events % 100 == 0:
            cnn_us = (self._t2 - self._t1) / 1000.0 if self._t2 > self._t1 else 0.0
            ptst_us = (self._t3 - self._t2) / 1000.0 if self._t3 > self._t2 else 0.0
            dec_us = (self._t4 - max(self._t3, self._t2, self._t1)) / 1000.0 if self._t4 > 0 else 0.0
            with open(self._csv_path, "a") as f:
                f.write(f"{self.n_events},{feat_us:.1f},{cnn_us:.1f},{ptst_us:.1f},"
                        f"{dec_us:.1f},{total_us:.1f},{int(self._has_pred)}\n")

        # Periodic summary
        if self.n_events - self._last_report_events >= self.log_interval:
            self._report()
            self._last_report_events = self.n_events

    def _percentiles(self, buf: deque, label: str) -> str:
        """Compute p50/p95/p99 from buffer."""
        if len(buf) == 0:
            return f"{label}: no data"
        arr = np.array(buf, dtype=np.float64)
        p50, p95, p99 = np.percentile(arr, [50, 95, 99])
        mean = arr.mean()
        return f"{label}: mean={mean:.0f}µs  p50={p50:.0f}µs  p95={p95:.0f}µs  p99={p99:.0f}µs"

    def _report(self):
        """Log latency summary."""
        log.info("=" * 70)
        log.info("LATENCY REPORT — %d events, %d predictions (%.1f%% have pred)",
                 self.n_events, self.n_predictions,
                 100 * self.n_predictions / max(1, self.n_events))
        log.info(self._percentiles(self._feat_latencies, "  Features   "))
        log.info(self._percentiles(self._cnn_latencies, "  CNN-Mamba  "))
        if self._ptst_latencies:
            log.info(self._percentiles(self._ptst_latencies, "  PatchTST   "))
        log.info(self._percentiles(self._decision_latencies, "  Decision   "))
        log.info(self._percentiles(self._total_latencies, "  Total/event"))
        log.info(self._percentiles(self._total_pred_latencies, "  Total/pred "))
        log.info("=" * 70)

    def get_summary(self) -> dict:
        """Return current latency stats as a dictionary."""
        def _stats(buf):
            if len(buf) == 0:
                return {"mean": 0, "p50": 0, "p95": 0, "p99": 0}
            arr = np.array(buf, dtype=np.float64)
            p50, p95, p99 = np.percentile(arr, [50, 95, 99])
            return {"mean": float(arr.mean()), "p50": float(p50),
                    "p95": float(p95), "p99": float(p99)}

        return {
            "n_events": self.n_events,
            "n_predictions": self.n_predictions,
            "features_us": _stats(self._feat_latencies),
            "cnn_mamba_us": _stats(self._cnn_latencies),
            "patchtst_us": _stats(self._ptst_latencies),
            "decision_us": _stats(self._decision_latencies),
            "total_per_event_us": _stats(self._total_latencies),
            "total_per_prediction_us": _stats(self._total_pred_latencies),
        }

    def final_report(self):
        """Force a final report."""
        self._report()
        summary = self.get_summary()
        log.info("FINAL LATENCY SUMMARY (JSON): %s",
                 {k: v for k, v in summary.items() if not k.startswith("n_")})
        return summary
