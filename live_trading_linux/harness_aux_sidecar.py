#!/usr/bin/env python3
"""
harness_aux_sidecar.py — multi-model prediction recorder (v3.3 + v3.4.2)
========================================================================
Runs as a SEPARATE process from paper_trading_v2_1s_short_top05.py. Tails the
same live_events.jsonl produced by the MBO recorder, applies the SAME HC #423
encoder fixes the v2 paper trader applies (lines 1114-1158), feeds a rolling
1500-event window through V33Inference and V342Inference, and appends one JSONL
row per prediction tick to logs/harness_aux_predictions_<SESSION_TAG>.jsonl.

This sidecar is READ-ONLY against live_events.jsonl. It NEVER writes there,
NEVER submits orders, NEVER posts to Discord. The v2 paper trader remains the
canonical live-trading critical-path file.

Joins downstream with the v2 paper trader's own JSONL by ts_ns to build the
multi-model corpus per HARNESS_EXTENSION_PLAN.md Step 5 + HC #443 R2.

Friday harness extension — HC #443 R1 / HC #448 R2.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import signal
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Deque, Dict, Optional

import numpy as np

# ── Env defaults must match training pipeline before importing adapters ───────
os.environ.setdefault("MAMBA_FEATURE_SET", "smart_v3")
os.environ.setdefault("SKIP_NORMALIZE", "1")
os.environ.setdefault("EVENT_WINDOW_SIZE", "1500")
os.environ.setdefault("EVENT_STRIDE", "250")

# Make script dir + repo root importable (mirrors paper_trading_v2 lines 61-63)
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS (mirrored from paper_trading_v2_1s_short_top05.py — verified 2026-05-21)
# ═══════════════════════════════════════════════════════════════════════════════

TICK_SIZE: float = 0.25          # paper_trading_v2 line 71
WINDOW_SIZE: int = 1500          # v3.3 / v3.4.2 expect WINDOW_SIZE_T1=1500
STRIDE: int = 250                # paper_trading_v2 line 101

# StreamingFeaturesSmartV3.update returns 25 features (N_EVENT_FEATURES).
# v3.3/v3.4.2 adapters expect events_t1 with N_T1_FEATURES = 39
# = 25 event features + 4 PT placeholder + 10 book-history placeholder.
# The model has a Linear(39 -> 25) t1_adapter that handles this projection,
# initialized to pass the 25 event features straight through. For live use
# (no PT meta-model, no book-history derived in this sidecar) we zero-pad.
N_EVENT_FEATURES: int = 25
N_T1_FEATURES: int = 39
N_PAD: int = N_T1_FEATURES - N_EVENT_FEATURES  # 14 zero pads

# Default paths — Razer (Windows) layout. Override via CLI for Jupiter testing.
LVL3 = Path(os.environ.get("LVL3_ROOT", r"C:\Users\claude\Lvl3Quant"))

DEFAULT_V33_WEIGHTS = LVL3 / "output" / "cnn_mamba_v3_3_uncertainty_weighted" / "fold_00_intra_ckpt.pt"
DEFAULT_V33_STATS = LVL3 / "output" / "cnn_mamba_v3_3_uncertainty_weighted" / "fold_00_feature_stats.npz"
DEFAULT_V342_WEIGHTS = LVL3 / "output" / "cnn_mamba_v3_4_2_fixedmtl" / "fold_00_intra_ckpt.pt"
DEFAULT_V342_STATS = LVL3 / "output" / "cnn_mamba_v3_4_2_fixedmtl" / "fold_00_feature_stats.npz"

DEFAULT_EVENTS_JSONL = LVL3 / "live_trading" / "logs" / "live_events.jsonl"
DEFAULT_OUT_DIR = LVL3 / "live_trading" / "logs"


# ═══════════════════════════════════════════════════════════════════════════════
# Logging setup
# ═══════════════════════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s sidecar %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("harness_aux_sidecar")


# ═══════════════════════════════════════════════════════════════════════════════
# Sidecar
# ═══════════════════════════════════════════════════════════════════════════════
class HarnessAuxSidecar:
    """Tail live_events.jsonl, run v3.3 + v3.4.2 in parallel, append JSONL."""

    def __init__(
        self,
        v33_weights: Path,
        v33_stats: Path,
        v342_weights: Path,
        v342_stats: Path,
        events_jsonl: Path,
        out_dir: Path,
        device: str = "cuda",
        session_tag: Optional[str] = None,
    ):
        self.v33_weights = Path(v33_weights)
        self.v33_stats = Path(v33_stats)
        self.v342_weights = Path(v342_weights)
        self.v342_stats = Path(v342_stats)
        self.events_jsonl = Path(events_jsonl)
        self.out_dir = Path(out_dir)
        self.device = device
        if session_tag is None:
            session_tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.session_tag = session_tag
        self.out_path = self.out_dir / f"harness_aux_predictions_{session_tag}.jsonl"

        # Inference adapters (lazy load)
        self._v33 = None
        self._v342 = None

        # Streaming features engine + rolling window
        self._streamer = None
        self._window: Deque[np.ndarray] = deque(maxlen=WINDOW_SIZE)

        # State
        self._stop = False
        self._out_fh = None
        self.n_events = 0
        self.n_preds_emitted = 0
        self.n_v33_ok = 0
        self.n_v342_ok = 0
        self.n_v342_book_miss = 0
        self._prev_ts_ns: int = 0
        self._best_bid: float = 0.0
        self._best_ask: float = 0.0
        self._mid_price: float = 0.0
        self._events_since_pred: int = 0

    # ── lazy load ──────────────────────────────────────────────────────────────
    def load(self) -> None:
        # Import locally to avoid torch load at module-import time (lets --smoke
        # pass without GPU drivers).
        from streaming_features_smart_v3 import StreamingFeaturesSmartV3, N_FEATURES
        if N_FEATURES != N_EVENT_FEATURES:
            log.warning(
                "streaming_features_smart_v3.N_FEATURES=%d but sidecar expected %d. "
                "Continuing with the runtime value; verify pad math.",
                N_FEATURES, N_EVENT_FEATURES,
            )
        self._streamer = StreamingFeaturesSmartV3()

        from v3_3_inference import V33Inference
        from v3_4_2_inference import V342Inference
        log.info("Loading v3.3 adapter weights=%s", self.v33_weights)
        self._v33 = V33Inference(
            weights_path=str(self.v33_weights),
            stats_path=str(self.v33_stats),
            device=self.device,
        )
        log.info("Loading v3.4.2 adapter weights=%s", self.v342_weights)
        self._v342 = V342Inference(
            weights_path=str(self.v342_weights),
            stats_path=str(self.v342_stats),
            device=self.device,
        )
        log.info(
            "adapters loaded. window=%d stride=%d device=%s out=%s",
            WINDOW_SIZE, STRIDE, self.device, self.out_path,
        )

    # ── signal handlers ────────────────────────────────────────────────────────
    def install_signal_handlers(self) -> None:
        def _handler(signum, _frame):
            log.info("signal %d received — stopping cleanly", signum)
            self._stop = True
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, _handler)
            except (ValueError, OSError):
                # Non-main thread or platform restriction — ignore.
                pass

    # ── event encoding (HC #423 — mirror of paper_trading_v2 1114-1158) ────────
    def _encode_event(self, ev: Dict) -> Optional[np.ndarray]:
        ts_ns = int(ev.get("timestamp_ns", 0) or 0)

        # Book update from bbo
        bbo = ev.get("bbo") or {}
        if bbo:
            b = float(bbo.get("bid_price", 0) or 0)
            a = float(bbo.get("ask_price", 0) or 0)
            if b > 0:
                self._best_bid = b
            if a > 0:
                self._best_ask = a
            if self._best_bid > 0 and self._best_ask > 0:
                self._mid_price = 0.5 * (self._best_bid + self._best_ask)

        # Encoder fields (HC #423 §3)
        action = int(ev.get("action", 0) or 0)
        side_raw = int(ev.get("side", 0) or 0)
        price_ticks = float(ev.get("price_ticks", 0) or 0)
        size = int(ev.get("size", 1) or 1)

        # FIX (b): use raw action as event_type_id (not collapsed {0,3})
        etype = action

        # FIX (a): price_rel_ticks = absolute_ticks - mid_ticks, clipped [-50, +50]
        if self._best_bid > 0 and self._best_ask > 0:
            mid_ticks = (self._best_bid + self._best_ask) / (2.0 * TICK_SIZE)
            price_rel_ticks = price_ticks - mid_ticks
            if price_rel_ticks > 50.0:
                price_rel_ticks = 50.0
            elif price_rel_ticks < -50.0:
                price_rel_ticks = -50.0
        else:
            price_rel_ticks = 0.0

        spread_ticks = (
            (self._best_ask - self._best_bid) / TICK_SIZE
            if (self._best_bid > 0 and self._best_ask > 0)
            else 0.0
        )

        # FIX (c): time_delta_log = log1p(seconds) — NOT microseconds
        delta_s = (
            max(0.0, (ts_ns - self._prev_ts_ns) / 1e9)
            if self._prev_ts_ns
            else 0.0
        )
        self._prev_ts_ns = ts_ns
        time_delta_log = math.log1p(delta_s) if delta_s > 0 else 0.0
        qty_log = math.log(max(1, size))

        # Streaming feature update returns 25-wide vec
        feat_25 = self._streamer.update(
            time_delta_log, etype, side_raw,
            price_rel_ticks, qty_log, spread_ticks,
        )
        feat_25 = np.asarray(feat_25, dtype=np.float32).reshape(-1)
        if feat_25.shape[0] != N_EVENT_FEATURES:
            log.error("unexpected streamer output width %d (expected %d)",
                      feat_25.shape[0], N_EVENT_FEATURES)
            return None

        # Zero-pad to 39-wide N_T1_FEATURES. Pad slots are the 4 PT meta-model
        # outputs (unavailable in sidecar) + 10 book-history derived features
        # (also unavailable here). t1_adapter Linear(39->25) was initialized to
        # pass the first 25 features through with the pad slots at small weight.
        feat_39 = np.zeros(N_T1_FEATURES, dtype=np.float32)
        feat_39[:N_EVENT_FEATURES] = feat_25
        return feat_39

    # ── prediction tick ───────────────────────────────────────────────────────
    def _run_predictions(self, ts_ns: int, event_seq: int) -> None:
        # Build flat (1500, 39) ndarray from rolling window
        feat_window = np.stack(self._window, axis=0).astype(np.float32)

        # v3.3 — flat ndarray input is supported (zero-pads t2/t3 internally)
        v33_reason = "ok"
        try:
            v33_out = self._v33.predict(feat_window)
        except Exception as e:
            log.exception("v3.3 inference exception")
            v33_out = None
            v33_reason = f"v33_failed: {type(e).__name__}"

        # v3.4.2 — dict input with explicit None book_pyramid → adapter returns
        # all-NaN with reason="book_features_missing" (expected per plan Step 4).
        v342_reason = "ok"
        try:
            v342_out = self._v342.predict({
                "events_t1": feat_window,
                "book_pyramid": None,
            })
        except Exception as e:
            log.exception("v3.4.2 inference exception")
            v342_out = None
            v342_reason = f"v342_failed: {type(e).__name__}"

        # Track counters
        if v33_out is not None and v33_out.get("reason") == "ok":
            self.n_v33_ok += 1
        if v342_out is not None:
            r = v342_out.get("reason")
            if r == "ok":
                self.n_v342_ok += 1
            elif r == "book_features_missing":
                self.n_v342_book_miss += 1

        self.n_preds_emitted += 1
        row = self._build_row(ts_ns, event_seq, v33_out, v33_reason, v342_out, v342_reason)
        self._emit_row(row)

        if self.n_preds_emitted % 100 == 0:
            log.info(
                "tick %d events / %d preds | v33_ok=%d v342_ok=%d v342_book_miss=%d",
                self.n_events, self.n_preds_emitted,
                self.n_v33_ok, self.n_v342_ok, self.n_v342_book_miss,
            )

    def _build_row(
        self,
        ts_ns: int,
        event_seq: int,
        v33_out: Optional[Dict],
        v33_reason_override: str,
        v342_out: Optional[Dict],
        v342_reason_override: str,
    ) -> Dict:
        def _f(d: Optional[Dict], key: str) -> float:
            if d is None:
                return float("nan")
            v = d.get(key, float("nan"))
            try:
                return float(v)
            except (TypeError, ValueError):
                return float("nan")

        def _safe_num(x: float) -> Optional[float]:
            # JSON does not support NaN/Inf in strict mode; we use allow_nan=True
            # (default) so leave NaN/Inf in; post-process tooling reads as NaN.
            return x

        row: Dict = {
            "ts_ns": int(ts_ns),
            "event_seq": int(event_seq),
            "n_preds_emitted": int(self.n_preds_emitted),
            "best_bid": _safe_num(self._best_bid),
            "best_ask": _safe_num(self._best_ask),
            "mid_price": _safe_num(self._mid_price),
            # v3.3
            "v33_pred_log_ret_1s": _f(v33_out, "pred_log_ret_1s"),
            "v33_pred_log_ret_5s": _f(v33_out, "pred_log_ret_5s"),
            "v33_pred_log_ret_10s": _f(v33_out, "pred_log_ret_10s"),
            "v33_pred_log_ret_30s": _f(v33_out, "pred_log_ret_30s"),
            "v33_conf": _f(v33_out, "conf"),
            "v33_reason": (v33_out.get("reason") if v33_out else v33_reason_override),
            "v33_diag_log_ret_60s": _f(v33_out, "diag_pred_log_ret_60s"),
            "v33_diag_log_ret_5min": _f(v33_out, "diag_pred_log_ret_5min"),
            # v3.4.2
            "v342_pred_log_ret_1s": _f(v342_out, "pred_log_ret_1s"),
            "v342_pred_log_ret_5s": _f(v342_out, "pred_log_ret_5s"),
            "v342_pred_log_ret_10s": _f(v342_out, "pred_log_ret_10s"),
            "v342_pred_log_ret_30s": _f(v342_out, "pred_log_ret_30s"),
            "v342_conf": _f(v342_out, "conf"),
            "v342_reason": (v342_out.get("reason") if v342_out else v342_reason_override),
        }
        return row

    def _emit_row(self, row: Dict) -> None:
        if self._out_fh is None:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            # Open in append + line-buffered mode
            self._out_fh = open(self.out_path, "a", buffering=1)
            log.info("opened output JSONL: %s", self.out_path)
        line = json.dumps(row, default=str)
        self._out_fh.write(line)
        self._out_fh.write("\n")
        self._out_fh.flush()

    # ── main tail loop ─────────────────────────────────────────────────────────
    def run(self, max_seconds: Optional[float] = None) -> None:
        """Tail live_events.jsonl from EOF, run inference every STRIDE events.

        max_seconds: if set, exit after that many seconds (used by dry-tail test).
        """
        if self._streamer is None or self._v33 is None or self._v342 is None:
            self.load()
        self.install_signal_handlers()

        if not self.events_jsonl.exists():
            log.warning(
                "live_events.jsonl not found at %s — sleeping until it appears",
                self.events_jsonl,
            )

        start_ts = time.time()
        last_inode: Optional[int] = None
        fh = None
        event_seq = 0

        try:
            while not self._stop:
                if max_seconds is not None and (time.time() - start_ts) >= max_seconds:
                    log.info("max_seconds=%.1f reached — exiting", max_seconds)
                    break

                # Open or re-open if rotated
                if fh is None:
                    if not self.events_jsonl.exists():
                        time.sleep(0.5)
                        continue
                    try:
                        fh = open(self.events_jsonl, "r")
                        fh.seek(0, 2)  # tail from EOF
                        try:
                            last_inode = os.fstat(fh.fileno()).st_ino
                        except OSError:
                            last_inode = None
                        log.info("opened %s @ EOF", self.events_jsonl)
                    except OSError as e:
                        log.warning("open failed: %s — retrying", e)
                        time.sleep(0.5)
                        continue

                line = fh.readline()
                if not line:
                    # Check for rotation
                    try:
                        st = os.stat(self.events_jsonl)
                        if last_inode is not None and st.st_ino != last_inode:
                            log.info("live_events.jsonl rotated — reopening from EOF")
                            fh.close()
                            fh = None
                            continue
                    except OSError:
                        pass
                    time.sleep(0.05)
                    continue

                line = line.strip()
                if not line:
                    continue

                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue

                feat = self._encode_event(ev)
                if feat is None:
                    continue
                self._window.append(feat)
                self.n_events += 1
                event_seq += 1
                self._events_since_pred += 1

                # Warmup: need full window before first prediction.
                if len(self._window) < WINDOW_SIZE:
                    continue

                # Predict every STRIDE events.
                if self._events_since_pred >= STRIDE:
                    self._events_since_pred = 0
                    ts_ns = int(ev.get("timestamp_ns", 0) or 0)
                    self._run_predictions(ts_ns=ts_ns, event_seq=event_seq)

        finally:
            if fh is not None:
                try:
                    fh.close()
                except OSError:
                    pass
            if self._out_fh is not None:
                try:
                    self._out_fh.close()
                except OSError:
                    pass
            log.info(
                "SHUTDOWN — total events=%d preds=%d v33_ok=%d v342_ok=%d v342_book_miss=%d out=%s",
                self.n_events, self.n_preds_emitted,
                self.n_v33_ok, self.n_v342_ok, self.n_v342_book_miss,
                self.out_path,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multi-model prediction sidecar (v3.3 + v3.4.2) — Friday harness extension",
    )
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--v33-weights", default=str(DEFAULT_V33_WEIGHTS))
    p.add_argument("--v33-stats", default=str(DEFAULT_V33_STATS))
    p.add_argument("--v342-weights", default=str(DEFAULT_V342_WEIGHTS))
    p.add_argument("--v342-stats", default=str(DEFAULT_V342_STATS))
    p.add_argument("--events-jsonl", default=str(DEFAULT_EVENTS_JSONL))
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--session-tag", default=None,
                   help="Override session tag in output filename (default: UTC timestamp)")
    p.add_argument("--max-seconds", type=float, default=None,
                   help="Optional run-duration cap (for dry-tail testing)")
    p.add_argument("--smoke", action="store_true",
                   help="Import + config print only; no engine load, no tail")
    return p.parse_args()


def _smoke_test(args: argparse.Namespace) -> int:
    """Import sanity + config print. No weights load, no live tail."""
    print("[smoke] harness_aux_sidecar module imported OK")
    print(f"[smoke] WINDOW_SIZE = {WINDOW_SIZE}")
    print(f"[smoke] STRIDE = {STRIDE}")
    print(f"[smoke] TICK_SIZE = {TICK_SIZE}")
    print(f"[smoke] N_EVENT_FEATURES = {N_EVENT_FEATURES}")
    print(f"[smoke] N_T1_FEATURES = {N_T1_FEATURES}")
    print(f"[smoke] N_PAD = {N_PAD}")
    print(f"[smoke] device = {args.device}")
    print(f"[smoke] v33_weights = {args.v33_weights}")
    print(f"[smoke] v33_stats = {args.v33_stats}")
    print(f"[smoke] v342_weights = {args.v342_weights}")
    print(f"[smoke] v342_stats = {args.v342_stats}")
    print(f"[smoke] events_jsonl = {args.events_jsonl}")
    print(f"[smoke] out_dir = {args.out_dir}")
    # Try importing streaming_features_smart_v3 (no torch needed for that one).
    try:
        from streaming_features_smart_v3 import StreamingFeaturesSmartV3, N_FEATURES  # noqa: F401
        print(f"[smoke] streaming_features_smart_v3 imported, N_FEATURES={N_FEATURES}")
        if N_FEATURES != N_EVENT_FEATURES:
            print(f"[smoke] WARN: streamer N_FEATURES={N_FEATURES} != expected {N_EVENT_FEATURES}")
    except Exception as e:
        print(f"[smoke] WARN streaming_features import: {type(e).__name__}: {e}")
    print("[smoke] PASS")
    return 0


def main() -> int:
    args = _parse_args()
    if args.smoke:
        return _smoke_test(args)

    sidecar = HarnessAuxSidecar(
        v33_weights=Path(args.v33_weights),
        v33_stats=Path(args.v33_stats),
        v342_weights=Path(args.v342_weights),
        v342_stats=Path(args.v342_stats),
        events_jsonl=Path(args.events_jsonl),
        out_dir=Path(args.out_dir),
        device=args.device,
        session_tag=args.session_tag,
    )
    sidecar.run(max_seconds=args.max_seconds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
