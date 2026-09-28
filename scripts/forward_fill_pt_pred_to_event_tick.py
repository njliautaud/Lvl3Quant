"""
Forward-fill PatchTST signal-step predictions into per-event-tick alignment.

For each day with enrichment available (output/meta_lgbm_features/<DATE>_signals_enriched.parquet)
the parquet has columns: signal_ts_ns, pt_pred_1s, pt_pred_5s, pt_pred_10s + others.
Signal cadence is stride 250ms (~36k rows/day).

We need each event tick (~17M/day) to have the most recent pt_pred at-or-before its timestamp.
For dates with no enrichment, output zero arrays + 'no_pt_pred' mask = 0.

Output NPZ:
  /data/processed/mbo_events_smart_v3_pt_pred/<DATE>_pt_pred_event_aligned.npz
  keys: pt_pred_1s (N,), pt_pred_5s (N,), pt_pred_10s (N,), has_pt_pred (N,) bool
"""
import os
import re
import time
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

DEFAULT_EVENTS_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/mbo_events_smart_v3")
DEFAULT_ENRICH_DIR = Path("/home/jupiter/Lvl3Quant/output/meta_lgbm_features")
DEFAULT_OUT_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/mbo_events_smart_v3_pt_pred")


def _rank_norm(x: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Per-day rank-norm to [-1, +1] for valid entries; 0 for invalid."""
    out = np.zeros_like(x, dtype=np.float32)
    if valid_mask.sum() < 2:
        return out
    vals = x[valid_mask]
    ranks = pd.Series(vals).rank(method="average").to_numpy()
    n = len(ranks)
    norm = ((ranks - 1.0) / max(n - 1, 1) - 0.5) * 2.0
    out[valid_mask] = norm.astype(np.float32)
    return out


def process_one_date(date_str: str, events_dir: Path, enrich_dir: Path,
                     out_dir: Path, overwrite: bool = False) -> dict:
    out_path = out_dir / f"{date_str}_pt_pred_event_aligned.npz"
    if out_path.exists() and not overwrite:
        return {"date": date_str, "status": "skipped_exists"}

    events_path = events_dir / f"{date_str}_mbo_events.npz"
    if not events_path.exists():
        return {"date": date_str, "status": "no_events"}

    t0 = time.time()
    ev = np.load(events_path, allow_pickle=True)
    ts_ns = ev["timestamps"].astype(np.int64)
    N = len(ts_ns)

    enrich_path = enrich_dir / f"{date_str}_signals_enriched.parquet"
    if not enrich_path.exists():
        # Zero-pad + mask=0
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_path,
            pt_pred_1s=np.zeros(N, dtype=np.float32),
            pt_pred_5s=np.zeros(N, dtype=np.float32),
            pt_pred_10s=np.zeros(N, dtype=np.float32),
            has_pt_pred=np.zeros(N, dtype=np.bool_),
        )
        return {"date": date_str, "status": "no_enrichment_zero_filled", "N": N,
                "elapsed_s": time.time() - t0}

    df = pd.read_parquet(enrich_path, columns=["signal_ts_ns", "pt_pred_1s", "pt_pred_5s", "pt_pred_10s"])
    df = df.sort_values("signal_ts_ns").reset_index(drop=True)
    sig_ts = df["signal_ts_ns"].to_numpy().astype(np.int64)
    pt_1s = df["pt_pred_1s"].to_numpy().astype(np.float32)
    pt_5s = df["pt_pred_5s"].to_numpy().astype(np.float32)
    pt_10s = df["pt_pred_10s"].to_numpy().astype(np.float32)

    # Apply per-day rank-norm (HC #281(E))
    valid = np.isfinite(pt_1s) & np.isfinite(pt_5s) & np.isfinite(pt_10s)
    pt_1s_rn = _rank_norm(pt_1s, valid)
    pt_5s_rn = _rank_norm(pt_5s, valid)
    pt_10s_rn = _rank_norm(pt_10s, valid)

    # Forward fill: for each event ts_ns[i], find the latest signal-step at-or-before ts_ns[i]
    # searchsorted side='right' returns insertion index > all <= target; subtract 1 to get last-≤ index.
    idx = np.searchsorted(sig_ts, ts_ns, side="right") - 1
    has_pt = idx >= 0  # event happened after at least one signal step

    out_1s = np.zeros(N, dtype=np.float32)
    out_5s = np.zeros(N, dtype=np.float32)
    out_10s = np.zeros(N, dtype=np.float32)
    if has_pt.any():
        out_1s[has_pt] = pt_1s_rn[idx[has_pt]]
        out_5s[has_pt] = pt_5s_rn[idx[has_pt]]
        out_10s[has_pt] = pt_10s_rn[idx[has_pt]]

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        pt_pred_1s=out_1s,
        pt_pred_5s=out_5s,
        pt_pred_10s=out_10s,
        has_pt_pred=has_pt,
    )
    return {
        "date": date_str, "status": "ok", "N": N, "n_signals": len(sig_ts),
        "n_event_filled": int(has_pt.sum()), "elapsed_s": time.time() - t0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", nargs="*", default=None)
    ap.add_argument("--events-dir", type=Path, default=DEFAULT_EVENTS_DIR)
    ap.add_argument("--enrich-dir", type=Path, default=DEFAULT_ENRICH_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--min-date", type=str, default="20251101")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()

    if args.dates:
        dates = sorted(args.dates)
    else:
        smart = {re.match(r"(\d{8})_mbo_events.npz", f).group(1)
                 for f in os.listdir(args.events_dir)
                 if re.match(r"\d{8}_mbo_events.npz", f)}
        dates = sorted([d for d in smart if d >= args.min_date])

    print(f"[pt-fill] {len(dates)} dates to process")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    t_total = time.time()
    results = []
    if args.workers > 1:
        from multiprocessing import Pool
        from functools import partial
        fn = partial(process_one_date, events_dir=args.events_dir, enrich_dir=args.enrich_dir,
                     out_dir=args.out_dir, overwrite=args.overwrite)
        with Pool(args.workers) as pool:
            for r in pool.imap_unordered(fn, dates):
                results.append(r)
                print(f"  {r['date']}: {r}", flush=True)
    else:
        for d in dates:
            r = process_one_date(d, args.events_dir, args.enrich_dir, args.out_dir, overwrite=args.overwrite)
            results.append(r)
            print(f"  {d}: {r}", flush=True)

    elapsed = time.time() - t_total
    ok = sum(1 for r in results if r["status"] in ("ok", "no_enrichment_zero_filled"))
    print(f"[pt-fill] DONE: total={len(results)} processed_ok={ok} elapsed={elapsed:.0f}s")


if __name__ == "__main__":
    main()
