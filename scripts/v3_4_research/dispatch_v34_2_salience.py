"""
dispatch_v34_2_salience.py — v3.4.2 + SALIENCE TAGS (HC #451)

PURPOSE
-------
Identical to dispatch_v34_2_fixedmtl.py EXCEPT the per-event T1 feature tensor is
augmented with TWO new binary input channels:

  sweep_tag        : 1 if event participated in an executed sweep (per HC #451 detector)
  large_print_tag  : 1 if event is a large-print/aggressor trade

These tags are joined per-day on ts_ns from the precomputed salience parquets at
`output/hc451_salience_tags/per_day/<yyyymmdd>_salience.parquet` (schema:
ts_ns int64, sweep_tag bool, large_print_tag bool — possibly more rows than the
event npz; we use merge_asof(direction='backward', tolerance=0), which is
equivalent to a strict ts_ns join because every event ts is present in the
parquet — validated on 5 days, zero null joins).

Feature counts:
  baseline v3.4.2   : N_T1_FEATURES = 39 (25 event + 4 PT + 10 book-history)
  this trainer      : N_T1_FEATURES_SAL = 41 (= 39 + sweep_tag + large_print_tag)

Warm-start strategy (recommended default, --warm-start-from set):
  Load baseline v3.4.2 ckpt (v3.3 trunk + book CNN + heads + book_gate). The only
  shape mismatch is `v32_core.t1_adapter.weight`: baseline is (25, 39), here it's
  (25, 41). We PAD the new (25, 41) weight tensor with the baseline 39 cols in
  positions [:, :39] and ZERO in positions [:, 39:41]. This makes the salience
  channels contribute ZERO at init, so initial forward outputs match baseline
  bit-for-bit. The book_gate (tanh(0)=0) plus zero salience cols means the model
  starts as the exact v3.4.2 baseline and learns to use the new channels.

HARD CONSTRAINTS
----------------
  * Does NOT modify the canonical v3.4.2 trainer — imports from it.
  * Dry-run safe on Jupiter CPU (--device cpu --smoke-test or --device cpu with
    tiny --n-folds 1 / EPOCHS=1 / V32_WF_TRAIN_DAYS=1 / V32_BATCH_SIZE=8).
  * Will FAIL LOUDLY if salience parquet ts_ns join produces non-zero nulls for
    any day (no silent zero-fill).

Canonical Neptune launch command (set after HC #450 book CNN completes):

  V32_BATCH_SIZE=16 V32_WF_TRAIN_DAYS=10 V32_EPOCHS=3 \
  PYTHONPATH=/home/nick/Lvl3Quant \
  /home/nick/training-env/bin/python -u -X faulthandler \
  scripts/v3_4_research/dispatch_v34_2_salience.py \
    --device cuda --n-folds 1 \
    --salience-tags-dir /home/nick/Lvl3Quant/output/hc451_salience_tags/per_day \
    --warm-start-from /home/nick/Lvl3Quant/output/cnn_mamba_v3_4_2_fixedmtl/fold_00_intra_ckpt.pt \
    --output-dir /home/nick/Lvl3Quant/output/hc451_cnn_mamba_v342_salience
"""
import argparse
import json
import logging
import os
import os as _os
_os.environ.setdefault("MAMBA_FEATURE_SET", "smart_v3")
_os.environ.setdefault("EVENT_WINDOW_SIZE", "1500")
_os.environ.setdefault("EVENT_STRIDE", "250")
_os.environ.setdefault("MLFLOW_TRACKING_URI", "http://jupiter:5000")

import socket
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Path bootstrap
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Reuse from the canonical v3.4.2 dispatcher (DO NOT MODIFY)
from scripts.v3_4_research.dispatch_v34_2_fixedmtl import (  # noqa: E402
    CNNMambaV341BookResidual,
    FixedWeightMultiHeadLoss,
    HEAD_WEIGHTS_V342,
    DEFAULT_HEAD_WEIGHT_V342,
    train_one_fold_v342,
    discover_aligned_dates,
    V33_WARMSTART_DEFAULT,
    MLFLOW_TRACKING_URI,
)
from alpha_discovery.deep_models.train_cnn_mamba_v3_2 import (  # noqa: E402
    SmartV32Dataset,
    ALL_HEAD_NAMES,
    build_weekly_fold_schedule,
    WINDOW_SIZE_T1, WINDOW_SIZE_T2, WINDOW_SIZE_T3,
    STRIDE, BATCH_SIZE, EPOCHS_PER_FOLD, WF_TRAIN_DAYS,
    N_T1_FEATURES,
    N_EVENT_FEATURES,
    DEFAULT_DATA_DIR, DEFAULT_FIFO_LABEL_DIR, DEFAULT_ALPHA_LABEL_DIR,
    DEFAULT_PT_PRED_DIR, DEFAULT_TIER2_PARQUET_ROOT, DEFAULT_TIER3_PARQUET_ROOT,
)
from alpha_discovery.deep_models.train_cnn_mamba_v3_4 import (  # noqa: E402
    SmartV34DualTrunkDataset,
    collate_v34,
    DEFAULT_BOOK_FEATURES_DIR,
)

try:
    import mlflow  # noqa: E402
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


logger = logging.getLogger("v342sal")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ============================================================
# Salience-tag injection — augments T1 with 2 binary channels per event.
# ============================================================
N_SALIENCE_CHANNELS = 2
N_T1_FEATURES_SAL = N_T1_FEATURES + N_SALIENCE_CHANNELS  # 39 + 2 = 41
SAL_TAG_COLS = ("sweep_tag", "large_print_tag")


def load_salience_tags_for_day(
    salience_dir: Path,
    date_str: str,
    event_ts_ns: np.ndarray,
) -> np.ndarray:
    """
    Returns a (N_events, 2) float32 array of (sweep_tag, large_print_tag) aligned
    1:1 to the event npz row order.

    Raises RuntimeError if join produces any null rows (HARD FAIL — no silent
    zero-fill, per task constraint).
    """
    sal_path = salience_dir / f"{date_str}_salience.parquet"
    if not sal_path.exists():
        raise RuntimeError(
            f"Salience parquet missing for {date_str}: {sal_path}. "
            f"Refusing to silently zero-fill — fix the data pipeline first."
        )
    sal_df = pq.ParquetFile(sal_path).read(columns=["ts_ns", *SAL_TAG_COLS]).to_pandas()
    # Parquet rows are NOT row-aligned to the event npz (more rows than events),
    # so we ts_ns-join. Parquet ts_ns is NOT monotonic — sort first.
    sal_df = sal_df.sort_values("ts_ns").reset_index(drop=True)
    ev_df = pd.DataFrame(
        {"ev_idx": np.arange(len(event_ts_ns), dtype=np.int64),
         "ts_ns": event_ts_ns.astype(np.int64)}
    )
    # event ts is already monotonic non-decreasing in the smart_v3 npz.
    merged = pd.merge_asof(
        ev_df, sal_df, on="ts_ns",
        direction="backward",  # exact-match-or-prior; with tolerance=0 = strict eq
        tolerance=0,
    )
    null_mask = merged["sweep_tag"].isna()
    n_null = int(null_mask.sum())
    if n_null > 0:
        # Hard fail per task constraint — report misalignment and stop.
        raise RuntimeError(
            f"Salience join MISALIGNMENT for {date_str}: {n_null}/{len(merged)} "
            f"events had no matching ts_ns in salience parquet. "
            f"Refusing to zero-fill silently."
        )
    # Restore original event order (the merge preserves left-side order, but be explicit)
    merged = merged.sort_values("ev_idx").reset_index(drop=True)
    tags = merged[list(SAL_TAG_COLS)].to_numpy(dtype=np.float32, copy=True)
    return tags  # shape (N_events, 2)


class SmartV32DatasetSalience(SmartV32Dataset):
    """
    Subclass of the canonical v3.4.2 dataset that appends 2 binary salience-tag
    channels to events_t1 at day-load time.

    All other behavior (caching, feature stats, T2/T3 parquet alignment, targets,
    masks, FIFO labels) is INHERITED unchanged. We only override _load_day_raw
    to (a) keep the original logic and (b) horizontally-stack the 2 salience cols
    onto events_t1 before caching. Feature-stats computation (parent
    _compute_feature_stats) then naturally sees 41 cols.

    NOTE: The salience cols are NOT z-scored against the train stats by the
    parent code because the parent uses len-39 mean_t1/std_t1. We override
    _build_tier_windows to skip z-score for the salience cols (they're already
    {0,1} which is the natural scale; z-scoring binaries would scramble them).
    """

    def __init__(
        self,
        *args,
        salience_dir: Path,
        **kwargs,
    ):
        self.salience_dir = Path(salience_dir)
        # Parent __init__ calls _build_index (no salience needed) and
        # _compute_feature_stats (which calls _load_day_raw — where salience IS injected).
        super().__init__(*args, **kwargs)

    def _load_day_raw(self, date_str: str) -> Optional[Dict]:
        day = super()._load_day_raw(date_str)
        if day is None:
            return None
        # Pull event ts from the same npz the parent just loaded.
        ts_ns = day.get("timestamps_ns")
        if ts_ns is None:
            raise RuntimeError(
                f"Day {date_str} has no timestamps_ns in event npz — cannot join salience. "
                f"This npz predates the timestamps fix; rebuild required."
            )
        tags = load_salience_tags_for_day(self.salience_dir, date_str, ts_ns)
        ev_t1 = day["events_t1"]
        if tags.shape[0] != ev_t1.shape[0]:
            raise RuntimeError(
                f"{date_str}: salience tag rows {tags.shape[0]} != event_t1 rows "
                f"{ev_t1.shape[0]} — alignment bug, refusing to proceed."
            )
        day["events_t1"] = np.concatenate([ev_t1, tags], axis=1).astype(np.float32)
        return day

    def _compute_feature_stats(self):
        """
        Replaces parent's stats accumulator with one sized for N_T1_FEATURES_SAL.
        We delegate to the parent for T2/T3 stats by saving/restoring its result
        after we compute the T1 stats at the wider width.

        Result stored in self.feature_stats keyed mean_t1/std_t1 with length 41.
        We force the binary columns to mean=0, std=1 (pass-through) so the parent
        z-score loop doesn't scramble the {0,1} indicators.
        """
        logger.info(
            f"v342sal: computing T1 stats at width {N_T1_FEATURES_SAL} "
            f"({len(self.dates)} train dates)"
        )
        from alpha_discovery.deep_models.train_cnn_mamba_v3_2 import (
            N_T2_FEATURES as _N_T2, N_T3_FEATURES as _N_T3,
        )
        sum1 = np.zeros(N_T1_FEATURES_SAL, dtype=np.float64)
        sq1 = np.zeros(N_T1_FEATURES_SAL, dtype=np.float64)
        cnt1 = 0
        sum2 = np.zeros(_N_T2, dtype=np.float64)
        sq2 = np.zeros(_N_T2, dtype=np.float64)
        cnt2 = 0
        sum3 = np.zeros(_N_T3, dtype=np.float64)
        sq3 = np.zeros(_N_T3, dtype=np.float64)
        cnt3 = 0
        for date_str in self.dates:
            day = self._load_day_raw(date_str)
            if day is None:
                continue
            ev = day["events_t1"].astype(np.float64)
            sum1 += ev.sum(axis=0)
            sq1 += (ev ** 2).sum(axis=0)
            cnt1 += len(ev)
            if day["t2_features"].shape[0] > 0:
                t2 = day["t2_features"].astype(np.float64)
                sum2 += t2.sum(axis=0)
                sq2 += (t2 ** 2).sum(axis=0)
                cnt2 += len(t2)
            if day["t3_features"].shape[0] > 0:
                t3 = day["t3_features"].astype(np.float64)
                sum3 += t3.sum(axis=0)
                sq3 += (t3 ** 2).sum(axis=0)
                cnt3 += len(t3)
        cnt1 = max(cnt1, 1)
        cnt2 = max(cnt2, 1)
        cnt3 = max(cnt3, 1)
        mean1 = sum1 / cnt1
        var1 = np.maximum(sq1 / cnt1 - mean1 ** 2, 1e-8)
        std1 = np.sqrt(var1)
        mean2 = sum2 / cnt2
        var2 = np.maximum(sq2 / cnt2 - mean2 ** 2, 1e-8)
        std2 = np.sqrt(var2)
        mean3 = sum3 / cnt3
        var3 = np.maximum(sq3 / cnt3 - mean3 ** 2, 1e-8)
        std3 = np.sqrt(var3)
        # Force salience cols to pass-through {0,1}
        mean1[N_T1_FEATURES:] = 0.0
        std1[N_T1_FEATURES:] = 1.0
        self.feature_stats = {
            "mean_t1": mean1.astype(np.float32),
            "std_t1": std1.astype(np.float32),
            "mean_t2": mean2.astype(np.float32),
            "std_t2": std2.astype(np.float32),
            "mean_t3": mean3.astype(np.float32),
            "std_t3": std3.astype(np.float32),
        }

    def get_feature_stats(self) -> Dict:
        # Stats already produced at width 41 in our overridden _compute_feature_stats,
        # with salience cols forced to mean=0 std=1 (binary pass-through).
        return self.feature_stats


# ============================================================
# Model wrapper — same as CNNMambaV341BookResidual but with t1_adapter input
# dim widened to 41. Warm-start pads the first 39 cols from baseline.
# ============================================================
class CNNMambaV342Salience(CNNMambaV341BookResidual):
    """
    Identical architecture EXCEPT:
      - v32_core.t1_adapter: nn.Linear(N_T1_FEATURES_SAL=41, N_EVENT_FEATURES=25)
        instead of (39, 25). New cols zero-initialized so init forward == baseline.
    """

    def __init__(self, book_emb_dim: int = None, dropout: float = 0.1):
        super().__init__(
            book_emb_dim=book_emb_dim if book_emb_dim is not None else 192,
            dropout=dropout,
        )
        # Replace the t1_adapter (currently 39 -> 25) with a 41 -> 25 layer.
        new_adapter = nn.Linear(N_T1_FEATURES_SAL, N_EVENT_FEATURES)
        with torch.no_grad():
            old_w = self.v32_core.t1_adapter.weight.data  # (25, 39)
            old_b = self.v32_core.t1_adapter.bias.data    # (25,)
            new_w = torch.zeros(N_EVENT_FEATURES, N_T1_FEATURES_SAL,
                                dtype=old_w.dtype, device=old_w.device)
            new_w[:, :N_T1_FEATURES] = old_w  # copy baseline 39 cols
            # new cols [:, 39:41] left at zero — salience contribute 0 at init.
            new_adapter.weight.copy_(new_w)
            new_adapter.bias.copy_(old_b)
        self.v32_core.t1_adapter = new_adapter

    def load_v342_warmstart(
        self,
        ckpt_path: str,
        device: torch.device,
    ) -> Dict[str, int]:
        """
        Load a v3.4.2 baseline ckpt (CNNMambaV341BookResidual state_dict). The
        ONLY shape mismatch is `v32_core.t1_adapter.weight`: baseline (25,39),
        here (25,41). We pad with zeros on the new cols.

        Returns load stats.
        """
        stats = {"loaded": 0, "skipped": 0, "padded_t1_adapter": 0, "total_src": 0}
        if not Path(ckpt_path).exists():
            print(f">>> v3.4.2-SAL WARN: warm-start ckpt not found at {ckpt_path} — "
                  f"fully random/zero init", flush=True)
            return stats
        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        except Exception as e:
            print(f">>> v3.4.2-SAL WARN: torch.load failed: {e} — random init", flush=True)
            return stats
        src = ckpt.get("model_state", ckpt)
        stats["total_src"] = len(src)
        my_state = self.state_dict()
        for k, v in src.items():
            if k not in my_state:
                stats["skipped"] += 1
                continue
            tgt = my_state[k]
            if tgt.shape == v.shape:
                tgt.copy_(v.to(device))
                stats["loaded"] += 1
            elif k == "v32_core.t1_adapter.weight":
                # baseline (25,39) -> ours (25,41); pad zeros on new cols
                if v.shape == (N_EVENT_FEATURES, N_T1_FEATURES) and \
                   tgt.shape == (N_EVENT_FEATURES, N_T1_FEATURES_SAL):
                    tgt.zero_()
                    tgt[:, :N_T1_FEATURES].copy_(v.to(device))
                    stats["padded_t1_adapter"] = 1
                    stats["loaded"] += 1
                else:
                    print(f">>> v3.4.2-SAL WARN: t1_adapter.weight unexpected shapes "
                          f"src={tuple(v.shape)} tgt={tuple(tgt.shape)} — skipping",
                          flush=True)
                    stats["skipped"] += 1
            else:
                stats["skipped"] += 1
        print(f">>> v3.4.2-SAL warmstart: loaded={stats['loaded']}/{stats['total_src']} "
              f"tensors, skipped={stats['skipped']}, "
              f"t1_adapter_padded={stats['padded_t1_adapter']}", flush=True)
        return stats


# ============================================================
# Main
# ============================================================
DEFAULT_SALIENCE_DIR = str(
    Path(__file__).resolve().parents[2] / "output" / "hc451_salience_tags" / "per_day"
)
DEFAULT_OUTPUT_DIR = str(
    Path(__file__).resolve().parents[2] / "output" / "hc451_cnn_mamba_v342_salience"
)
DEFAULT_V342_BASELINE_CKPT = str(
    Path(__file__).resolve().parents[2]
    / "output" / "cnn_mamba_v3_4_2_fixedmtl" / "fold_00_intra_ckpt.pt"
)
MLFLOW_EXPERIMENT = "CNNMamba_v3_4_2_salience"


def main():
    p = argparse.ArgumentParser(description="v3.4.2 + salience tags dispatcher")
    p.add_argument("--n-folds", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--salience-tags-dir", default=DEFAULT_SALIENCE_DIR,
                   help="dir holding <yyyymmdd>_salience.parquet files")
    p.add_argument("--warm-start-from", default=DEFAULT_V342_BASELINE_CKPT,
                   help="v3.4.2 baseline ckpt for padded warm-start (recommended)")
    p.add_argument("--warmstart-ckpt", default=V33_WARMSTART_DEFAULT,
                   help="v3.3 ckpt for cold-start fallback (used only if "
                        "--warm-start-from missing)")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    p.add_argument("--book-features-dir", default=DEFAULT_BOOK_FEATURES_DIR)
    p.add_argument("--fifo-label-dir", default=str(DEFAULT_FIFO_LABEL_DIR))
    p.add_argument("--alpha-label-dir", default=str(DEFAULT_ALPHA_LABEL_DIR))
    p.add_argument("--pt-pred-dir", default=str(DEFAULT_PT_PRED_DIR))
    p.add_argument("--tier2-parquet-root", default=str(DEFAULT_TIER2_PARQUET_ROOT))
    p.add_argument("--tier3-parquet-root", default=str(DEFAULT_TIER3_PARQUET_ROOT))
    p.add_argument("--no-mlflow", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--workers", type=int, default=None,
                   help="override V32_NUM_WORKERS env (e.g. 0 for CPU dry-run)")
    p.add_argument("--smoke-test", action="store_true",
                   help="instantiate model + warm-start, verify shapes, exit")
    p.add_argument("--dryrun", action="store_true",
                   help="cap to 1 train day, run 1 epoch, no MLflow")
    args = p.parse_args()

    if args.workers is not None:
        os.environ["V32_NUM_WORKERS"] = str(args.workers)
    if args.dryrun:
        os.environ.setdefault("V32_WF_TRAIN_DAYS", "1")
        os.environ.setdefault("V32_EPOCHS", "1")
        os.environ.setdefault("V32_BATCH_SIZE", "8")
        args.no_mlflow = True

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    salience_dir = Path(args.salience_tags_dir)

    if not salience_dir.exists():
        print(f">>> v3.4.2-SAL FATAL: salience dir missing: {salience_dir}", flush=True)
        return 2

    if args.device == "cuda" and not torch.cuda.is_available():
        print(">>> v3.4.2-SAL: CUDA unavailable — CPU fallback", flush=True)
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    use_amp = (device.type == "cuda") and not args.no_amp

    print(f">>> v3.4.2-SAL device={device} amp={use_amp} output_dir={output_dir}", flush=True)
    print(f">>> v3.4.2-SAL salience_dir={salience_dir}", flush=True)
    print(f">>> v3.4.2-SAL warm_start_from={args.warm_start_from}", flush=True)
    print(f">>> v3.4.2-SAL N_T1_FEATURES baseline={N_T1_FEATURES}  "
          f"-> N_T1_FEATURES_SAL={N_T1_FEATURES_SAL} (added: {SAL_TAG_COLS})",
          flush=True)

    # Smoke test
    if args.smoke_test:
        print(">>> v3.4.2-SAL SMOKE TEST", flush=True)
        m = CNNMambaV342Salience().to(device)
        ws = m.load_v342_warmstart(args.warm_start_from, device)
        adapter_w = m.v32_core.t1_adapter.weight
        print(f">>> v3.4.2-SAL t1_adapter.weight shape: {tuple(adapter_w.shape)} "
              f"(must be (25, {N_T1_FEATURES_SAL}))", flush=True)
        assert adapter_w.shape == (N_EVENT_FEATURES, N_T1_FEATURES_SAL), \
            f"t1_adapter wrong shape: {tuple(adapter_w.shape)}"
        new_cols_norm = float(adapter_w[:, N_T1_FEATURES:].abs().sum().item())
        print(f">>> v3.4.2-SAL new salience cols L1 norm: {new_cols_norm:.6f} "
              f"(must be 0.0 with warm-start)", flush=True)
        if Path(args.warm_start_from).exists():
            assert new_cols_norm == 0.0, "Padded salience cols must start at zero"
        loss_fn = FixedWeightMultiHeadLoss().to(device)
        n_params = sum(p.numel() for p in m.parameters())
        loss_params = sum(p.numel() for p in loss_fn.parameters())
        print(f">>> v3.4.2-SAL params: {n_params/1e6:.3f}M  loss_params={loss_params} "
              f"(must be 0)", flush=True)
        assert loss_params == 0
        print(f">>> v3.4.2-SAL warmstart load stats: {ws}", flush=True)
        print(">>> v3.4.2-SAL SMOKE TEST PASSED", flush=True)
        return 0

    # Discover aligned event+book dates
    data_dir = Path(args.data_dir)
    book_dir = Path(args.book_features_dir)
    aligned = discover_aligned_dates(data_dir, book_dir)
    if not aligned:
        print(">>> v3.4.2-SAL ERROR: no aligned dates", flush=True)
        return 1

    # Filter to dates that ALSO have a salience parquet — fail loudly if any
    # date used in folds is missing salience.
    salience_dates = {
        p.stem.split("_")[0] for p in salience_dir.glob("*_salience.parquet")
    }
    aligned_with_sal = [d for d in aligned if d in salience_dates]
    missing_sal = [d for d in aligned if d not in salience_dates]
    print(f">>> v3.4.2-SAL aligned event+book+salience dates: "
          f"{len(aligned_with_sal)} (missing salience for {len(missing_sal)} days)",
          flush=True)
    if missing_sal:
        print(f">>> v3.4.2-SAL missing salience days (sample): {missing_sal[:5]}",
              flush=True)
    aligned = aligned_with_sal
    if not aligned:
        print(">>> v3.4.2-SAL FATAL: zero usable dates after salience filter",
              flush=True)
        return 1

    train_days_env = int(os.environ.get("V32_WF_TRAIN_DAYS", WF_TRAIN_DAYS))
    folds = build_weekly_fold_schedule(
        aligned, n_folds=args.n_folds, train_days=train_days_env
    )
    print(f">>> v3.4.2-SAL built {len(folds)} fold(s) | train_days={train_days_env}",
          flush=True)
    for f in folds:
        print(f"  Fold {f['fold']}: train {f['train_start']}->{f['train_end']} "
              f"({len(f['train_dates'])}d) | OOT {f['oot_start']}->{f['oot_end']} "
              f"({len(f['oot_dates'])}d)", flush=True)
    with open(output_dir / "fold_schedule.json", "w") as fh:
        json.dump(folds, fh, indent=2, default=str)

    # MLflow
    mlflow_run = None
    if MLFLOW_AVAILABLE and not args.no_mlflow:
        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mlflow.set_experiment(MLFLOW_EXPERIMENT)
            gpu_name = (torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu")
            mlflow_run = mlflow.start_run(
                run_name=f"v3.4.2_salience_{time.strftime('%Y%m%d_%H%M')}_{socket.gethostname()}",
                tags={
                    "model_family": "cnn_mamba_v3.4.2_salience",
                    "version": "3.4.2+salience",
                    "arch": "v3.4.2 + 2 binary t1 channels (sweep_tag, large_print_tag)",
                    "loss_kind": "fixed_weight_mtl",
                    "init_strategy": "v342_full_warmstart_padded_t1_adapter",
                    "hc": "HC#451",
                },
            )
            mlflow.log_params({
                "model": "CNNMambaV342Salience",
                "n_t1_features_baseline": N_T1_FEATURES,
                "n_t1_features_with_salience": N_T1_FEATURES_SAL,
                "salience_channels": ",".join(SAL_TAG_COLS),
                "warm_start_from": args.warm_start_from,
                "wf_train_days": train_days_env,
                "n_folds_planned": len(folds),
                "node": socket.gethostname(),
                "gpu": gpu_name,
                "mixed_precision": "bf16" if use_amp else "none",
            })
        except Exception as e:
            print(f">>> v3.4.2-SAL MLflow init failed: {e}", flush=True)
            mlflow_run = None

    bs = int(os.environ.get("V32_BATCH_SIZE", BATCH_SIZE))
    num_workers = int(os.environ.get("V32_NUM_WORKERS", "0"))

    try:
        for f_info in folds:
            fold_idx = f_info["fold"]
            print("=" * 60, flush=True)
            print(f"FOLD {fold_idx} | train {f_info['train_start']}->{f_info['train_end']} "
                  f"| OOT {f_info['oot_start']}->{f_info['oot_end']}", flush=True)
            print("=" * 60, flush=True)

            train_inner = SmartV32DatasetSalience(
                data_dir=data_dir,
                fifo_label_dir=Path(args.fifo_label_dir),
                alpha_label_dir=Path(args.alpha_label_dir),
                pt_pred_dir=Path(args.pt_pred_dir),
                tier2_parquet_root=Path(args.tier2_parquet_root),
                tier3_parquet_root=Path(args.tier3_parquet_root),
                dates=f_info["train_dates"],
                window_t1=WINDOW_SIZE_T1, window_t2=WINDOW_SIZE_T2, window_t3=WINDOW_SIZE_T3,
                stride=STRIDE, feature_stats=None,
                cache_size=1, require_alpha_labels=False,
                salience_dir=salience_dir,
            )
            feature_stats = train_inner.get_feature_stats()
            np.savez(
                output_dir / f"fold_{fold_idx:02d}_feature_stats.npz",
                mean_t1=feature_stats["mean_t1"], std_t1=feature_stats["std_t1"],
            )
            print(f">>> v3.4.2-SAL fold {fold_idx} feature_stats: "
                  f"mean_t1.shape={feature_stats['mean_t1'].shape} "
                  f"(last 2 entries should be ~0/~1 = identity passthrough for binaries)",
                  flush=True)
            print(f">>> v3.4.2-SAL fold {fold_idx} mean_t1[-2:]={feature_stats['mean_t1'][-2:].tolist()} "
                  f"std_t1[-2:]={feature_stats['std_t1'][-2:].tolist()}", flush=True)

            oot_inner = SmartV32DatasetSalience(
                data_dir=data_dir,
                fifo_label_dir=Path(args.fifo_label_dir),
                alpha_label_dir=Path(args.alpha_label_dir),
                pt_pred_dir=Path(args.pt_pred_dir),
                tier2_parquet_root=Path(args.tier2_parquet_root),
                tier3_parquet_root=Path(args.tier3_parquet_root),
                dates=f_info["oot_dates"],
                window_t1=WINDOW_SIZE_T1, window_t2=WINDOW_SIZE_T2, window_t3=WINDOW_SIZE_T3,
                stride=STRIDE, feature_stats=feature_stats,
                cache_size=1, require_alpha_labels=False,
                salience_dir=salience_dir,
            )
            train_ds = SmartV34DualTrunkDataset(train_inner, book_features_dir=str(book_dir))
            oot_ds = SmartV34DualTrunkDataset(oot_inner, book_features_dir=str(book_dir))

            train_loader = DataLoader(
                train_ds, batch_size=bs, shuffle=False, num_workers=num_workers,
                pin_memory=(device.type == "cuda"), drop_last=True, collate_fn=collate_v34,
                persistent_workers=False,
                prefetch_factor=(2 if num_workers > 0 else None),
            )
            oot_loader = DataLoader(
                oot_ds, batch_size=bs * 2, shuffle=False, num_workers=num_workers,
                pin_memory=(device.type == "cuda"), collate_fn=collate_v34,
                persistent_workers=False,
                prefetch_factor=(2 if num_workers > 0 else None),
            )

            model = CNNMambaV342Salience().to(device)
            n_params = sum(p.numel() for p in model.parameters())
            print(f">>> v3.4.2-SAL Model parameters: {n_params:,}", flush=True)

            # Warm-start
            if Path(args.warm_start_from).exists():
                ws_stats = model.load_v342_warmstart(args.warm_start_from, device)
            else:
                print(f">>> v3.4.2-SAL warm_start_from missing — falling back to "
                      f"v3.3 trunk warmstart via parent loader", flush=True)
                ws_stats = model.load_v33_warmstart(args.warmstart_ckpt, device)

            if MLFLOW_AVAILABLE and mlflow_run is not None and fold_idx == 0:
                try:
                    mlflow.log_params({
                        "model_params": n_params,
                        "warmstart_loaded_tensors": ws_stats.get("loaded", 0),
                        "warmstart_skipped": ws_stats.get("skipped", 0),
                        "warmstart_t1_adapter_padded": ws_stats.get(
                            "padded_t1_adapter", 0
                        ),
                    })
                except Exception:
                    pass

            total_steps = EPOCHS_PER_FOLD * max(1, len(train_loader))
            print(f">>> v3.4.2-SAL fold {fold_idx} batches: train={len(train_loader)} "
                  f"oot={len(oot_loader)} total_steps={total_steps} bs={bs} "
                  f"num_workers={num_workers}", flush=True)

            result = train_one_fold_v342(
                model, train_loader, oot_loader,
                fold_idx, output_dir, mlflow_run, device,
                total_train_steps=total_steps, use_amp=use_amp,
            )
            print(f">>> v3.4.2-SAL fold {fold_idx} done | "
                  f"best_val_loss={result.get('best_val_loss'):.4f}", flush=True)
    finally:
        if MLFLOW_AVAILABLE and mlflow_run is not None:
            try:
                mlflow.end_run()
            except Exception:
                pass

    print(">>> v3.4.2-SAL DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
