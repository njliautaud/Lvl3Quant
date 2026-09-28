#!/usr/bin/env python3
"""
jupiter_v33_shadow_processor.py — Jupiter-side batched v3.3 shadow inference.

PURPOSE
-------
Friday harness deliverable (HC #448 R2), re-routed to Jupiter to avoid
risking the Razer live-trading stack. Pulls the latest MBO events from
Razer's live_events.jsonl over SSH, replays them through the same
StreamingFeaturesSmartV3 + HC #423 encoder pipeline used by the v2 paper
trader, runs v3.3 in BATCHED soft-degrade mode on Jupiter CPU, and writes
its own shadow predictions JSONL.

STRICTLY READ-ONLY against Razer. Does NOT submit orders, does NOT post
to Discord, does NOT feed any trading logic. Predictions are
miscalibrated (live emits only 25/39 T1 features and no T2/T3 stacks).
Use the JSONL for offline IC / signal-quality analysis only.

State persists across cron invocations via processor_state.pkl
(streamer + window deque + last ts_ns).

Run manually:
    /usr/bin/python3 /home/jupiter/Lvl3Quant/staging/harness_v3/jupiter_v33_shadow_processor.py

Cron-friendly: exits on its own after one pull/process pass.
"""
from __future__ import annotations

import json
import math
import os
import pickle
import subprocess
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

# Env defaults must match training pipeline before importing adapters.
os.environ.setdefault("MAMBA_FEATURE_SET", "smart_v3")
os.environ.setdefault("SKIP_NORMALIZE", "1")
os.environ.setdefault("EVENT_WINDOW_SIZE", "1500")
os.environ.setdefault("EVENT_STRIDE", "250")

# Bootstrap paths so we can import the Jupiter copy of the streamer +
# the staging copy of the v3.3 adapter. Insert order matters because
# sys.path.insert(0, ...) prepends — the LAST insert ends up first.
# We want staging/harness_v3 to win over live_trading_linux for
# v3_3_inference (the staging copy has the per-tier dt_rank fixup and
# env-var overrides for the 128/64/4/8 ckpt).
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_REPO_ROOT))                                # alpha_discovery
sys.path.insert(0, str(_REPO_ROOT / "live_trading_linux"))         # streaming_features_smart_v3
sys.path.insert(0, str(_HERE))                                     # staging/harness_v3 (FIRST)


# ─── Constants ─────────────────────────────────────────────────────────────────
TICK_SIZE: float = 0.25
WINDOW_SIZE: int = 1500
STRIDE: int = 250
N_EVENT_FEATURES: int = 25
N_T1_FEATURES: int = 39
BATCH_SIZE: int = 64
MAX_WALL_SECONDS: float = 240.0  # 4 min budget inside 5-min cron

# Default pull size — covers ~9-15 min of typical session-hours events.
DEFAULT_PULL_TAIL: int = 20000

RAZER_HOST = os.environ.get("RAZER_SSH_HOST", "claude@razer")
RAZER_LIVE_EVENTS = (
    r"C:\Users\claude\Lvl3Quant\live_trading\logs\live_events.jsonl"
)

V33_WEIGHTS = _REPO_ROOT / "output" / "cnn_mamba_v3_3_uncertainty_weighted" / "fold_00_intra_ckpt.pt"
V33_STATS = _REPO_ROOT / "output" / "cnn_mamba_v3_3_uncertainty_weighted" / "fold_00_feature_stats.npz"

STATE_DIR = _HERE / "state"
STATE_PATH = STATE_DIR / "processor_state.pkl"
STATE_JSON_PATH = STATE_DIR / "processor_state.json"  # human-readable summary
TMP_PULL_PATH = Path("/tmp/v33_shadow_pull.jsonl")

OUT_JSONL_PATH = _REPO_ROOT / "logs" / "harness_v33_predictions.jsonl"

MODEL_VERSION = "v3.3_uncertainty_weighted_fold0_intra_ckpt"
SOFTDEGRADE_REASON = "live_emits_25_feats_only_no_T2_T3"


# ─── Logging ───────────────────────────────────────────────────────────────────
def _log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}", flush=True)


# ─── State container ───────────────────────────────────────────────────────────
class ProcessorState:
    """Persists the streamer + rolling window + last-processed cursor."""

    def __init__(self) -> None:
        # Defer streamer import — only available when packages installed.
        from streaming_features_smart_v3 import StreamingFeaturesSmartV3
        self.streamer = StreamingFeaturesSmartV3()
        self.window: Deque[np.ndarray] = deque(maxlen=WINDOW_SIZE)
        self.last_processed_ts_ns: int = 0
        self.prev_ts_ns: int = 0
        self.best_bid: float = 0.0
        self.best_ask: float = 0.0
        self.events_since_pred: int = 0
        self.total_events_seen: int = 0
        self.total_preds_emitted: int = 0
        self.last_pull_time_iso: Optional[str] = None

    def save(self) -> None:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = STATE_PATH.with_suffix(".pkl.tmp")
        with open(tmp, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, STATE_PATH)

        summary = {
            "last_processed_event_ts_ns": int(self.last_processed_ts_ns),
            "last_pull_time": self.last_pull_time_iso,
            "total_events_seen": int(self.total_events_seen),
            "total_preds_emitted": int(self.total_preds_emitted),
            "window_len": len(self.window),
            "streamer_warm": bool(self.streamer.is_warm()),
            "streamer_n_events": int(self.streamer.n_events),
        }
        tmpj = STATE_JSON_PATH.with_suffix(".json.tmp")
        with open(tmpj, "w") as f:
            json.dump(summary, f, indent=2)
        os.replace(tmpj, STATE_JSON_PATH)

    @classmethod
    def load_or_new(cls) -> "ProcessorState":
        if STATE_PATH.exists():
            try:
                with open(STATE_PATH, "rb") as f:
                    st = pickle.load(f)
                _log(
                    f"state loaded: last_ts={st.last_processed_ts_ns} "
                    f"window={len(st.window)} streamer_n={st.streamer.n_events} "
                    f"warm={st.streamer.is_warm()}"
                )
                return st
            except Exception as e:
                _log(f"state load failed ({e}); starting fresh")
        return cls()


# ─── SSH pull ──────────────────────────────────────────────────────────────────
def pull_razer_tail(tail_lines: int) -> int:
    """Pull the last N lines of live_events.jsonl from Razer to TMP_PULL_PATH.

    Returns the byte size of the pulled file, or 0 on failure.
    """
    TMP_PULL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if TMP_PULL_PATH.exists():
        TMP_PULL_PATH.unlink()

    cmd = [
        "ssh",
        "-o", "ConnectTimeout=10",
        "-o", "ServerAliveInterval=15",
        RAZER_HOST,
        f'powershell -Command "Get-Content {RAZER_LIVE_EVENTS} -Tail {tail_lines}"',
    ]
    t0 = time.time()
    try:
        with open(TMP_PULL_PATH, "w") as out_fh:
            proc = subprocess.run(
                cmd, stdout=out_fh, stderr=subprocess.PIPE, timeout=90,
            )
    except subprocess.TimeoutExpired:
        _log("ssh pull TIMEOUT")
        return 0
    except Exception as e:
        _log(f"ssh pull EXCEPTION: {e}")
        return 0

    dt = time.time() - t0
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace")[:200]
        _log(f"ssh pull rc={proc.returncode}: {err}")
        return 0

    size = TMP_PULL_PATH.stat().st_size if TMP_PULL_PATH.exists() else 0
    _log(f"pulled {size} bytes in {dt:.1f}s ({tail_lines} tail lines)")
    return size


# ─── HC #423 encode → 25-feat → 39-feat ────────────────────────────────────────
def encode_event(state: ProcessorState, ev: Dict) -> Optional[Tuple[int, np.ndarray]]:
    """Replicates harness_aux_sidecar._encode_event but writes to ProcessorState.

    Returns (ts_ns, feat_39) on success, None on bad row.
    """
    try:
        ts_ns = int(ev.get("timestamp_ns", 0) or 0)
        if ts_ns <= 0:
            return None

        # Book update from BBO snapshot embedded in the event
        bbo = ev.get("bbo") or {}
        if bbo:
            b = float(bbo.get("bid_price", 0) or 0)
            a = float(bbo.get("ask_price", 0) or 0)
            if b > 0:
                state.best_bid = b
            if a > 0:
                state.best_ask = a

        action = int(ev.get("action", 0) or 0)
        side_raw = int(ev.get("side", 0) or 0)
        price_ticks = float(ev.get("price_ticks", 0) or 0)
        size = int(ev.get("size", 1) or 1)

        # HC #423 — use raw action as event_type_id (not collapsed)
        etype = action

        # HC #423 — price_rel_ticks = absolute_ticks - mid_ticks, clipped [-50, +50]
        if state.best_bid > 0 and state.best_ask > 0:
            mid_ticks = (state.best_bid + state.best_ask) / (2.0 * TICK_SIZE)
            price_rel_ticks = price_ticks - mid_ticks
            if price_rel_ticks > 50.0:
                price_rel_ticks = 50.0
            elif price_rel_ticks < -50.0:
                price_rel_ticks = -50.0
        else:
            price_rel_ticks = 0.0

        spread_ticks = (
            (state.best_ask - state.best_bid) / TICK_SIZE
            if (state.best_bid > 0 and state.best_ask > 0)
            else 0.0
        )

        # HC #423 — time_delta_log = log1p(seconds)
        if state.prev_ts_ns and ts_ns > state.prev_ts_ns:
            delta_s = (ts_ns - state.prev_ts_ns) / 1e9
        else:
            delta_s = 0.0
        state.prev_ts_ns = ts_ns
        time_delta_log = math.log1p(delta_s) if delta_s > 0 else 0.0
        qty_log = math.log(max(1, size))

        feat_25 = state.streamer.update(
            time_delta_log, etype, side_raw,
            price_rel_ticks, qty_log, spread_ticks,
        )
        feat_25 = np.asarray(feat_25, dtype=np.float32).reshape(-1)
        if feat_25.shape[0] != N_EVENT_FEATURES:
            return None

        feat_39 = np.zeros(N_T1_FEATURES, dtype=np.float32)
        feat_39[:N_EVENT_FEATURES] = feat_25
        return ts_ns, feat_39
    except Exception as e:
        _log(f"encode error: {type(e).__name__}: {e}")
        return None


# ─── Output writer ─────────────────────────────────────────────────────────────
def append_predictions(rows: List[Dict]) -> None:
    if not rows:
        return
    OUT_JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSONL_PATH, "a", buffering=1) as f:
        for r in rows:
            f.write(json.dumps(r, default=str))
            f.write("\n")


# ─── Main pass ─────────────────────────────────────────────────────────────────
def run_once() -> int:
    t_start = time.time()
    state = ProcessorState.load_or_new()
    state.last_pull_time_iso = datetime.now(timezone.utc).isoformat()

    pulled_bytes = pull_razer_tail(DEFAULT_PULL_TAIL)
    if pulled_bytes <= 0:
        _log("nothing pulled — exiting")
        state.save()
        return 1

    # Lazy load v3.3 adapter only if needed (skip on empty pull)
    from v3_3_inference import V33Inference
    _log(f"loading v3.3 adapter ({V33_WEIGHTS.name})")
    v33 = V33Inference(
        weights_path=str(V33_WEIGHTS),
        stats_path=str(V33_STATS),
        device="cpu",
    )
    _log("v3.3 adapter loaded")

    # ── Pass 1: stream events through encoder, build prediction-tick samples ──
    pending_batches: List[Tuple[int, int, np.ndarray]] = []  # (ts_ns, event_idx_in_pull, window_snapshot)
    n_skipped_old = 0
    n_processed = 0
    new_high_water_ts = state.last_processed_ts_ns

    with open(TMP_PULL_PATH, "r") as f:
        for line_idx, line in enumerate(f):
            if (time.time() - t_start) > MAX_WALL_SECONDS:
                _log("TRUNCATED — wall budget exhausted during encode pass")
                break
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue

            ts_ns_peek = int(ev.get("timestamp_ns", 0) or 0)
            if ts_ns_peek <= state.last_processed_ts_ns:
                n_skipped_old += 1
                continue

            out = encode_event(state, ev)
            if out is None:
                continue
            ts_ns, feat_39 = out
            state.window.append(feat_39)
            n_processed += 1
            state.events_since_pred += 1
            state.total_events_seen += 1
            if ts_ns > new_high_water_ts:
                new_high_water_ts = ts_ns

            # Only sample once warm + window full
            if (
                state.streamer.is_warm()
                and len(state.window) >= WINDOW_SIZE
                and state.events_since_pred >= STRIDE
            ):
                snapshot = np.stack(state.window, axis=0).astype(np.float32)
                pending_batches.append((ts_ns, line_idx, snapshot))
                state.events_since_pred = 0

    encode_dt = time.time() - t_start
    _log(
        f"encode pass: processed={n_processed} skipped_old={n_skipped_old} "
        f"pending_predict_samples={len(pending_batches)} dt={encode_dt:.1f}s "
        f"window={len(state.window)} streamer_n={state.streamer.n_events} "
        f"warm={state.streamer.is_warm()}"
    )

    # ── Pass 2: run BATCHED v3.3 inference ───────────────────────────────────
    rows_out: List[Dict] = []
    pull_iso = state.last_pull_time_iso
    for batch_start in range(0, len(pending_batches), BATCH_SIZE):
        if (time.time() - t_start) > MAX_WALL_SECONDS:
            _log(
                f"TRUNCATED — wall budget exhausted at batch {batch_start}/"
                f"{len(pending_batches)}; remaining samples dropped"
            )
            break
        chunk = pending_batches[batch_start:batch_start + BATCH_SIZE]
        windows = [w for (_, _, w) in chunk]
        t_b0 = time.time()
        try:
            preds = v33.predict_batch(windows)
        except Exception as e:
            _log(f"predict_batch failed: {type(e).__name__}: {e}")
            preds = [{
                "pred_log_ret_1s": float("nan"),
                "pred_log_ret_5s": float("nan"),
                "pred_log_ret_10s": float("nan"),
                "pred_log_ret_30s": float("nan"),
                "conf": float("nan"),
                "reason": f"batch_failed_{type(e).__name__}",
                "diag_pred_log_ret_60s": float("nan"),
                "diag_pred_log_ret_5min": float("nan"),
            } for _ in chunk]
        t_b1 = time.time()
        per_event_ms = (t_b1 - t_b0) * 1000.0 / max(1, len(chunk))
        _log(
            f"batch {batch_start//BATCH_SIZE + 1}: "
            f"B={len(chunk)} dt={(t_b1 - t_b0):.1f}s "
            f"per-event={per_event_ms:.0f}ms"
        )

        proc_dur_ms_each = (t_b1 - t_b0) * 1000.0 / max(1, len(chunk))
        for (ts_ns, event_idx, _), p in zip(chunk, preds):
            rows_out.append({
                "ts_ns": int(ts_ns),
                "event_idx_in_stream": int(event_idx),
                "v33_pred_log_ret_1s": float(p.get("pred_log_ret_1s", float("nan"))),
                "v33_pred_log_ret_5s": float(p.get("pred_log_ret_5s", float("nan"))),
                "v33_pred_log_ret_10s": float(p.get("pred_log_ret_10s", float("nan"))),
                "v33_pred_log_ret_30s": float(p.get("pred_log_ret_30s", float("nan"))),
                "v33_conf": float(p.get("conf", float("nan"))),
                "v33_reason": str(p.get("reason", "ok")),
                "v33_diag_log_ret_60s": float(p.get("diag_pred_log_ret_60s", float("nan"))),
                "v33_diag_log_ret_5min": float(p.get("diag_pred_log_ret_5min", float("nan"))),
                "mode": "soft_degrade",
                "softdegrade_reason": SOFTDEGRADE_REASON,
                "model_version": MODEL_VERSION,
                "pull_ts": pull_iso,
                "process_dur_ms": float(proc_dur_ms_each),
            })
            state.total_preds_emitted += 1

    # ── Flush + persist state atomically ─────────────────────────────────────
    append_predictions(rows_out)
    state.last_processed_ts_ns = new_high_water_ts
    state.save()

    wall = time.time() - t_start
    per_event_amort_ms = (wall * 1000.0 / n_processed) if n_processed else 0.0
    _log(
        f"DONE wall={wall:.1f}s events={n_processed} preds_written={len(rows_out)} "
        f"per_event_amort={per_event_amort_ms:.1f}ms "
        f"output={OUT_JSONL_PATH}"
    )

    return 0


if __name__ == "__main__":
    sys.exit(run_once())
