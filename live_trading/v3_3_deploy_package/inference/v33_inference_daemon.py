"""
v33_inference_daemon.py — HC #360 / #368 inference daemon for live Razer host.

WHAT THIS DAEMON DOES:
  1. Reads arch params (window_size, head_spec, head_sigmas, normalization scheme)
     directly from .pt ckpt blob per HC #360. Nothing is hardcoded.
  2. Spawns inference loop that consumes the live NPZ window-feed produced by
     mbo_recorder.py and writes predictions to JSONL for paper_trader.py.
  3. Writes a heartbeat file every prediction so the model_stale_seconds
     watchdog (SafetyNetEnsemble) can monitor it.
  4. Emits per-head σ envelope for σ-spike halt watchdog.

WHAT THIS DAEMON DOES NOT DO:
  - It does NOT make trade decisions. paper_trader.py owns entry/exit.
  - It does NOT modify the model. Pure inference.

Runtime contract (consumed by deploy_v33_to_razer.ps1 step 6):
  python v33_inference_daemon.py
      --config <framework_config.json>
      --output-jsonl <log_dir>\\v33_predictions_<ts>.jsonl
      [--feed-source npz|stdin]
      [--heartbeat-file <path>]

Output JSONL schema (one line per prediction):
  {
    "ts_iso": "2026-05-14T23:59:01.234",
    "ts_epoch_ms": 1747...,
    "window_start_event_idx": 12345,
    "predictions": { "<head_name>": <float>, ... },   # 32 heads
    "head_sigmas": { "<head_name>": <float>, ... },   # learned σ from ckpt
    "percentiles": { "<head_name>": <0-100>, ... },   # vs OOT distribution
    "inference_latency_ms": <float>,
    "model_id": "cnn_mamba_v3_3_uncertainty_weighted/fold_00/ckpt_intra",
    "ckpt_mtime_iso": "...",
    "schema_version": "1.0"
  }

Authorized: HC #368. Reads-only against ckpt (HC #307D compliant — does NOT modify trainer).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
except ImportError:
    print("[fatal] torch not importable", file=sys.stderr)
    sys.exit(2)


SCHEMA_VERSION = "1.0"
HEARTBEAT_INTERVAL_PREDICTIONS = 1  # write heartbeat every prediction


# ---------------------------------------------------------------------------
# Ckpt loading — HC #360 compliant
# ---------------------------------------------------------------------------
def load_ckpt_arch(ckpt_path: Path) -> Dict:
    """Extract arch metadata from .pt ckpt without instantiating the model.

    Required keys in ckpt blob (set by trainer):
      - arch.window_size
      - arch.normalization     ("zscore" | "robust")
      - arch.head_names        list[str]
      - arch.head_sigmas       dict[str, float]  (learned σ per head from MTL)
      - arch.feature_dim
      - arch.trunk_dim
      - oot_percentile_table   dict[head, np.ndarray sorted OOT distribution]
                               (used to map raw pred → percentile for entry gate)
    """
    if not ckpt_path.exists():
        raise FileNotFoundError(f"ckpt not found: {ckpt_path}")

    print(f"[ckpt] loading {ckpt_path} (CPU map)")
    blob = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    arch = blob.get("arch")
    if not isinstance(arch, dict):
        raise ValueError(
            f"ckpt missing 'arch' dict — HC #360 violation. Got keys: {list(blob.keys())[:20]}"
        )

    required = ["window_size", "normalization", "head_names", "head_sigmas",
                "feature_dim", "trunk_dim"]
    missing = [k for k in required if k not in arch]
    if missing:
        raise ValueError(f"ckpt arch missing required keys: {missing}")

    return {
        "blob": blob,
        "window_size": int(arch["window_size"]),
        "normalization": str(arch["normalization"]),
        "head_names": list(arch["head_names"]),
        "head_sigmas": dict(arch["head_sigmas"]),
        "feature_dim": int(arch["feature_dim"]),
        "trunk_dim": int(arch["trunk_dim"]),
        "oot_percentile_table": blob.get("oot_percentile_table", {}),
    }


def build_model(arch: Dict, blob: Dict, device: str) -> "torch.nn.Module":
    """Construct model and load weights. Imports the v3.3 model class lazily
    so this daemon doesn't pin a transitive on the alpha_discovery tree at
    import time (Razer may not have the full repo)."""
    try:
        # v3.3 family
        from alpha_discovery.deep_models.train_cnn_mamba_v3_3_uncertainty_weighted import (
            CNNMambaV33Uncertainty,
        )
    except ImportError as e:
        raise ImportError(
            f"could not import CNNMambaV33Uncertainty — Razer must have alpha_discovery package on path. {e}"
        )

    model = CNNMambaV33Uncertainty(
        feature_dim=arch["feature_dim"],
        head_names=arch["head_names"],
        trunk_dim=arch["trunk_dim"],
    )
    state = blob.get("model_state_dict") or blob.get("state_dict")
    if state is None:
        raise ValueError("ckpt missing model_state_dict / state_dict")
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model


# ---------------------------------------------------------------------------
# Feed adapter — consumes window vectors from MBO recorder
# ---------------------------------------------------------------------------
class NPZWindowFeed:
    """Polls a directory for the latest *_window.npz from mbo_recorder.

    Contract (mbo_recorder writes):
      <feed_dir>/window_<epoch_ms>.npz  with keys:
        - 'features'   (window_size, feature_dim) float32, already normalized
        - 'event_idx'  scalar int (index of last event in window)
        - 'ts_epoch_ms' scalar int
    """
    def __init__(self, feed_dir: Path, poll_interval_sec: float = 0.05):
        self.feed_dir = feed_dir
        self.poll = poll_interval_sec
        self._last_seen = None

    def next_window(self, timeout_sec: float = 30.0) -> Optional[Dict]:
        start = time.time()
        while time.time() - start < timeout_sec:
            candidates = sorted(self.feed_dir.glob("window_*.npz"))
            if candidates:
                latest = candidates[-1]
                if latest != self._last_seen:
                    self._last_seen = latest
                    try:
                        z = np.load(latest)
                        return {
                            "features": z["features"],
                            "event_idx": int(z["event_idx"]),
                            "ts_epoch_ms": int(z["ts_epoch_ms"]),
                            "src_path": str(latest),
                        }
                    except Exception as e:
                        print(f"[feed] failed to load {latest}: {e}", file=sys.stderr)
            time.sleep(self.poll)
        return None


# ---------------------------------------------------------------------------
# Percentile mapping (raw pred → 0-100 vs OOT distribution)
# ---------------------------------------------------------------------------
def compute_percentile(value: float, sorted_oot: np.ndarray) -> float:
    if sorted_oot is None or len(sorted_oot) == 0:
        return 50.0
    idx = np.searchsorted(sorted_oot, value, side="right")
    return float(100.0 * idx / len(sorted_oot))


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="framework_config.json path")
    p.add_argument("--output-jsonl", required=True, help="predictions output jsonl")
    p.add_argument("--heartbeat-file", default=None,
                   help="written every prediction so model_stale watchdog can see freshness")
    p.add_argument("--device", default=None, help="cuda|cpu (auto-detect default)")
    p.add_argument("--feed-dir", default=None,
                   help="directory MBO recorder writes window_*.npz into (overrides config)")
    args = p.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    sig = cfg.get("signal_layer", {})

    ckpt_path = Path(sig.get("ckpt_path"))
    arch_info = load_ckpt_arch(ckpt_path)
    blob = arch_info["blob"]
    head_names: List[str] = arch_info["head_names"]
    head_sigmas: Dict[str, float] = arch_info["head_sigmas"]
    oot_table: Dict[str, np.ndarray] = arch_info["oot_percentile_table"]
    window_size = arch_info["window_size"]
    feature_dim = arch_info["feature_dim"]

    print(f"[arch] window_size={window_size} feature_dim={feature_dim} "
          f"n_heads={len(head_names)} normalization={arch_info['normalization']}")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(arch_info, blob, device)
    print(f"[model] loaded on {device}, eval mode")

    feed_dir = Path(args.feed_dir or sig.get("feed_dir") or "C:/Users/claude/Lvl3Quant/live_data/window_feed")
    feed_dir.mkdir(parents=True, exist_ok=True)
    feed = NPZWindowFeed(feed_dir)

    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    hb_path = Path(args.heartbeat_file) if args.heartbeat_file else None

    ckpt_mtime_iso = datetime.fromtimestamp(ckpt_path.stat().st_mtime, tz=timezone.utc).isoformat()
    model_id = f"cnn_mamba_v3_3_uncertainty_weighted/fold_00/{ckpt_path.stem}"

    n_emitted = 0
    print(f"[loop] writing predictions → {out_path}")

    with open(out_path, "a", buffering=1) as fh:  # line-buffered
        while True:
            window = feed.next_window(timeout_sec=30.0)
            if window is None:
                # stale feed — emit a stall marker (paper trader's mbo_feed watchdog will catch)
                print("[stall] no new window in 30s", file=sys.stderr)
                continue

            features = window["features"]
            if features.shape != (window_size, feature_dim):
                print(f"[skip] bad shape {features.shape} expected ({window_size},{feature_dim})",
                      file=sys.stderr)
                continue

            t0 = time.perf_counter()
            with torch.no_grad():
                x = torch.from_numpy(features).float().unsqueeze(0).to(device)
                out = model(x)  # dict[head_name -> tensor(1,)]
            latency_ms = (time.perf_counter() - t0) * 1000.0

            preds = {h: float(out[h].cpu().item()) for h in head_names if h in out}
            pcts = {h: compute_percentile(v, oot_table.get(h)) for h, v in preds.items()}

            now = datetime.now(timezone.utc)
            record = {
                "ts_iso": now.isoformat(),
                "ts_epoch_ms": int(now.timestamp() * 1000),
                "window_start_event_idx": window["event_idx"] - window_size,
                "window_end_event_idx": window["event_idx"],
                "predictions": preds,
                "head_sigmas": head_sigmas,
                "percentiles": pcts,
                "inference_latency_ms": round(latency_ms, 3),
                "model_id": model_id,
                "ckpt_mtime_iso": ckpt_mtime_iso,
                "schema_version": SCHEMA_VERSION,
            }
            fh.write(json.dumps(record) + "\n")
            n_emitted += 1

            # heartbeat for model_stale watchdog
            if hb_path is not None and (n_emitted % HEARTBEAT_INTERVAL_PREDICTIONS == 0):
                try:
                    hb_path.write_text(json.dumps({
                        "last_pred_ts_iso": record["ts_iso"],
                        "last_pred_event_idx": window["event_idx"],
                        "n_emitted": n_emitted,
                        "latency_ms": record["inference_latency_ms"],
                    }))
                except Exception as e:
                    print(f"[hb] write failed: {e}", file=sys.stderr)

            if n_emitted % 100 == 0:
                print(f"[loop] {n_emitted} predictions, last latency {latency_ms:.1f}ms")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("[exit] keyboard interrupt")
    except Exception as e:
        print(f"[fatal] {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
