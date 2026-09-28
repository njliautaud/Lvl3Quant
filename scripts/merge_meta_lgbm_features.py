#!/usr/bin/env python3
"""
HC #270 — Phase 2: Feature merger for meta-LGBM gate.

Reads each per-date Phase-1 labeled parquet (signal-level FIFO outcomes) and joins:
  (a) PatchTST confluence — predictions + sign-agreement + magnitude features
  (b) Microstructure features at signal timestamp from mbo_book_features/
  (c) Vol-regime features — rolling vol z-score over mid-price change
  (d) Time-of-day regime — minutes from RTH open + regime bin
  (e) Signal persistence — sign-agreement of CNN-Mamba over recent K predictions

Output: <out_dir>/<date>_signals_enriched.parquet  (feature columns prefixed)

Idempotent: skips dates whose enriched output already exists.
Phase-1 input dir: /home/jupiter/Lvl3Quant/output/meta_lgbm_labels/
Phase-2 output dir: /home/jupiter/Lvl3Quant/output/meta_lgbm_features/
"""

from pathlib import Path
import argparse
import logging
import sys
import numpy as np
import pandas as pd

LVL3_ROOT = Path("/home/jupiter/Lvl3Quant")
LABEL_DIR = LVL3_ROOT / "output" / "meta_lgbm_labels"
PATCHTST_DIR = LVL3_ROOT / "output" / "patchtst_bulk_oot"
BOOK_FEAT_DIR = LVL3_ROOT / "data" / "processed" / "mbo_book_features"
DEFAULT_OUT = LVL3_ROOT / "output" / "meta_lgbm_features"

# RTH open assumption: 09:30 ET (US equity / ES RTH session start).
# ES MBO timestamps are nanoseconds Unix-time (UTC).  We bin time-of-day from the
# *first signal of the day* relative to its hour of day in ET (ET = UTC-5 standard / -4 DST).
# Simpler & robust: derive minutes-since-day-start in ET using pandas tz_convert.
RTH_OPEN_HHMM_ET = (9, 30)
RTH_CLOSE_HHMM_ET = (16, 0)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("meta_lgbm_features")


def _load_patchtst(date_str: str):
    p = PATCHTST_DIR / f"{date_str}_predictions.npz"
    if not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    preds = d["predictions"]  # (n, 3) — 1s/5s/10s
    stride = int(d["stride"]) if "stride" in d else 250
    window = int(d["window_size"]) if "window_size" in d else 3000
    return {"preds": preds, "stride": stride, "window": window, "n": len(preds)}


def _load_book_features(date_str: str):
    p = BOOK_FEAT_DIR / f"{date_str}_book_features.npz"
    if not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    feats = d["features"]
    ts = d["timestamps"]
    names = list(d["feature_names"])
    name_idx = {n: i for i, n in enumerate(names)}
    return {"features": feats, "timestamps": ts, "name_idx": name_idx}


def _patchtst_ts_for_idx(i: int, stride: int, window: int, ts_events: np.ndarray) -> int:
    n_events = len(ts_events)
    event_idx = min(i * stride + window - 1, n_events - 1)
    return int(ts_events[event_idx])


def _build_patchtst_ts_lookup(patch, ts_events_book: np.ndarray) -> dict:
    """Map each PatchTST prediction's ts -> row index. Uses book ts as the event clock."""
    out = {}
    n = patch["n"]
    stride = patch["stride"]
    window = patch["window"]
    for i in range(n):
        ts = _patchtst_ts_for_idx(i, stride, window, ts_events_book)
        out[ts] = i
    return out


def _binary_search_book_idx(book_ts: np.ndarray, target_ts: int) -> int:
    """Return the largest book-event index with ts <= target_ts (last-known-state)."""
    idx = int(np.searchsorted(book_ts, target_ts, side="right") - 1)
    if idx < 0:
        idx = 0
    return idx


def _et_minute_of_day(ts_ns: int) -> float:
    """Minutes-since-midnight in US/Eastern."""
    ts = pd.Timestamp(ts_ns, unit="ns", tz="UTC").tz_convert("US/Eastern")
    return ts.hour * 60 + ts.minute + ts.second / 60.0


def _tod_regime_bin(min_et: float) -> int:
    """0=premarket, 1=open(9:30-10:00), 2=morning(10:00-11:30),
       3=midday(11:30-14:00), 4=afternoon(14:00-15:30), 5=close(15:30-16:00), 6=after."""
    if min_et < 9 * 60 + 30: return 0
    if min_et < 10 * 60: return 1
    if min_et < 11 * 60 + 30: return 2
    if min_et < 14 * 60: return 3
    if min_et < 15 * 60 + 30: return 4
    if min_et < 16 * 60: return 5
    return 6


def _compute_vol_regime(features: np.ndarray, name_idx: dict, ts: np.ndarray, signal_ts: np.ndarray):
    """Rolling vol of mid_price_change_ticks over a 30s lookback at each signal ts.
       Returns z-score relative to the day's distribution (percentile rank)."""
    mpc_col = name_idx.get("mid_price_change_ticks")
    if mpc_col is None:
        return np.zeros(len(signal_ts), dtype=np.float32), np.zeros(len(signal_ts), dtype=np.float32)
    mpc = features[:, mpc_col]
    abs_mpc = np.abs(mpc).astype(np.float64)

    # Cumulative-sum-of-squares for variance window
    cs = np.concatenate([[0.0], np.cumsum(abs_mpc * abs_mpc)])

    LOOKBACK_NS = 30 * 1_000_000_000

    vols = np.empty(len(signal_ts), dtype=np.float32)
    for k, sts in enumerate(signal_ts):
        end_idx = _binary_search_book_idx(ts, sts) + 1
        start_ts = sts - LOOKBACK_NS
        start_idx = _binary_search_book_idx(ts, start_ts)
        window_n = max(1, end_idx - start_idx)
        ssq = cs[end_idx] - cs[start_idx]
        vols[k] = np.sqrt(ssq / window_n)
    # Percentile rank within the day
    order = np.argsort(vols)
    pct = np.empty_like(vols, dtype=np.float32)
    pct[order] = np.linspace(0, 1, len(vols), dtype=np.float32) if len(vols) > 1 else 0.5
    return vols, pct


def _compute_signal_persistence(pred_idx: np.ndarray, pred_signs: np.ndarray, k: int = 5):
    """Fraction of last K CNN-Mamba 1s pred signs that match the current signal's sign.
       pred_signs is full per-prediction sign array (+1/-1/0).
       For signal at row i in pred-frame, look at signs[max(0,i-k):i+1] vs current sign."""
    out = np.zeros(len(pred_idx), dtype=np.float32)
    for j, i in enumerate(pred_idx):
        s = pred_signs[i]
        if s == 0:
            out[j] = 0.5
            continue
        lo = max(0, i - k)
        win = pred_signs[lo: i + 1]
        if len(win) == 0:
            out[j] = 0.5
        else:
            out[j] = float(np.mean(win == s))
    return out


def merge_one_date(date_str: str, out_dir: Path, force: bool = False) -> Path | None:
    out_path = out_dir / f"{date_str}_signals_enriched.parquet"
    if out_path.exists() and not force:
        log.info(f"[{date_str}] enriched already exists, skipping")
        return out_path

    label_path = LABEL_DIR / f"{date_str}_signals_labeled.parquet"
    if not label_path.exists():
        log.warning(f"[{date_str}] no Phase-1 label parquet yet")
        return None

    df = pd.read_parquet(label_path)
    if df.empty:
        log.warning(f"[{date_str}] empty Phase-1 parquet")
        return None
    log.info(f"[{date_str}] loaded {len(df):,} labeled signals")

    # ----- Microstructure -----
    book = _load_book_features(date_str)
    if book is None:
        log.warning(f"[{date_str}] no book_features → skipping microstructure")
        ms_cols = {}
    else:
        sig_ts = df["signal_ts_ns"].to_numpy(dtype=np.int64)
        # Bulk binary search
        idxs = np.searchsorted(book["timestamps"], sig_ts, side="right") - 1
        idxs[idxs < 0] = 0
        F = book["features"]
        N = book["name_idx"]
        # 5-level book imbalance
        bs1 = F[idxs, N["bid_size_1"]]; as1 = F[idxs, N["ask_size_1"]]
        bs5 = sum(F[idxs, N[f"bid_size_{k}"]] for k in range(1, 6))
        as5 = sum(F[idxs, N[f"ask_size_{k}"]] for k in range(1, 6))
        l1_imb = (bs1 - as1) / np.maximum(bs1 + as1, 1.0)
        l5_imb = (bs5 - as5) / np.maximum(bs5 + as5, 1.0)
        # Multi-level depth ratios
        bs2 = F[idxs, N["bid_size_2"]]; bs3 = F[idxs, N["bid_size_3"]]
        as2 = F[idxs, N["ask_size_2"]]; as3 = F[idxs, N["ask_size_3"]]
        l2_imb = (bs2 - as2) / np.maximum(bs2 + as2, 1.0)
        l3_imb = (bs3 - as3) / np.maximum(bs3 + as3, 1.0)
        ms_cols = {
            "ms_spread_ticks": F[idxs, N["spread_ticks"]],
            "ms_l1_imb": l1_imb.astype(np.float32),
            "ms_l2_imb": l2_imb.astype(np.float32),
            "ms_l3_imb": l3_imb.astype(np.float32),
            "ms_l5_imb": l5_imb.astype(np.float32),
            "ms_depth_imb_5": F[idxs, N["depth_imbalance_5"]],
            "ms_rolling_imb_100": F[idxs, N["rolling_imbalance_100"]],
            "ms_trade_intensity_100": F[idxs, N["trade_intensity_100"]],
            "ms_cum_delta": F[idxs, N["cum_delta"]],
            "ms_net_order_flow": F[idxs, N["net_order_flow"]],
            "ms_bid_size_1": bs1, "ms_ask_size_1": as1,
            "ms_bid_size_5_total": bs5.astype(np.float32),
            "ms_ask_size_5_total": as5.astype(np.float32),
            "ms_mid_price_change_ticks": F[idxs, N["mid_price_change_ticks"]],
            "ms_spread_change_ticks": F[idxs, N["spread_change_ticks"]],
        }
        # Vol regime
        vol_30s, vol_pct = _compute_vol_regime(F, N, book["timestamps"], sig_ts)
        ms_cols["vol_30s_abs_mpc"] = vol_30s
        ms_cols["vol_30s_day_pct"] = vol_pct

    # ----- Time-of-day -----
    et_min = np.array([_et_minute_of_day(int(t)) for t in df["signal_ts_ns"]], dtype=np.float32)
    rth_open_min = RTH_OPEN_HHMM_ET[0] * 60 + RTH_OPEN_HHMM_ET[1]
    tod_cols = {
        "tod_minutes_from_open": et_min - rth_open_min,
        "tod_minute_of_day_et": et_min,
        "tod_regime_bin": np.array([_tod_regime_bin(m) for m in et_min], dtype=np.int8),
    }

    # ----- Signal persistence -----
    # Reconstruct full per-prediction sign array via labels file (need pred_1s for ALL preds)
    cnn_pred_path = LVL3_ROOT / "output" / "cnn_mamba_v2_all_oot" / f"{date_str}_predictions.npz"
    persist_cols = {}
    if cnn_pred_path.exists():
        cnn = np.load(cnn_pred_path, allow_pickle=True)
        cnn_p1s = cnn["predictions"][:, 0]
        cnn_signs = np.sign(cnn_p1s).astype(np.int8)
        pred_idx_arr = df["pred_idx"].to_numpy(dtype=np.int64)
        persist = _compute_signal_persistence(pred_idx_arr, cnn_signs, k=5)
        persist_cols["signal_persist_5"] = persist
        # Also longer
        persist20 = _compute_signal_persistence(pred_idx_arr, cnn_signs, k=20)
        persist_cols["signal_persist_20"] = persist20

    # ----- PatchTST confluence -----
    patch = _load_patchtst(date_str)
    pt_cols = {}
    if patch is not None and book is not None:
        # PatchTST timestamp lookup uses the same book event clock
        ts_book = book["timestamps"]
        pt_ts = np.empty(patch["n"], dtype=np.int64)
        for i in range(patch["n"]):
            ev = min(i * patch["stride"] + patch["window"] - 1, len(ts_book) - 1)
            pt_ts[i] = ts_book[ev]
        # Map signal ts → patchtst row via direct lookup (CNN-Mamba shares stride/window)
        pt_lookup = {int(t): i for i, t in enumerate(pt_ts)}
        sig_ts = df["signal_ts_ns"].to_numpy(dtype=np.int64)
        pt_idx = np.array([pt_lookup.get(int(t), -1) for t in sig_ts], dtype=np.int64)
        valid = pt_idx >= 0
        n_match = int(valid.sum())
        log.info(f"[{date_str}]   PatchTST timestamp match: {n_match}/{len(sig_ts)} ({100*n_match/max(1,len(sig_ts)):.1f}%)")
        # Initialize columns with NaN; fill where valid
        pt_p1s = np.full(len(sig_ts), np.nan, dtype=np.float32)
        pt_p5s = np.full(len(sig_ts), np.nan, dtype=np.float32)
        pt_p10s = np.full(len(sig_ts), np.nan, dtype=np.float32)
        if n_match > 0:
            pt_p1s[valid] = patch["preds"][pt_idx[valid], 0]
            pt_p5s[valid] = patch["preds"][pt_idx[valid], 1]
            pt_p10s[valid] = patch["preds"][pt_idx[valid], 2]
        pt_cols["pt_pred_1s"] = pt_p1s
        pt_cols["pt_pred_5s"] = pt_p5s
        pt_cols["pt_pred_10s"] = pt_p10s
        pt_cols["pt_abs_pred_1s"] = np.abs(pt_p1s)
        pt_cols["pt_abs_pred_5s"] = np.abs(pt_p5s)
        pt_cols["pt_abs_pred_10s"] = np.abs(pt_p10s)
        # Sign agreement vs CNN-Mamba
        cnn_p1s = df["pred_1s"].to_numpy(dtype=np.float32)
        cnn_p5s = df["pred_5s"].to_numpy(dtype=np.float32)
        cnn_p10s = df["pred_10s"].to_numpy(dtype=np.float32)
        pt_cols["pt_cnn_sign_agree_1s"] = (np.sign(cnn_p1s) == np.sign(pt_p1s)).astype(np.float32)
        pt_cols["pt_cnn_sign_agree_5s"] = (np.sign(cnn_p5s) == np.sign(pt_p5s)).astype(np.float32)
        pt_cols["pt_cnn_sign_agree_10s"] = (np.sign(cnn_p10s) == np.sign(pt_p10s)).astype(np.float32)
        # Where PatchTST is NaN, set agreement to 0.5 (unknown)
        for c in ["pt_cnn_sign_agree_1s", "pt_cnn_sign_agree_5s", "pt_cnn_sign_agree_10s"]:
            pt_cols[c] = np.where(np.isnan(pt_p1s), 0.5, pt_cols[c]).astype(np.float32)

    # ----- Stitch all into df -----
    out = df.copy()
    for col_dict in (ms_cols, tod_cols, persist_cols, pt_cols):
        for k, v in col_dict.items():
            out[k] = v

    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    log.info(f"[{date_str}] wrote {len(out):,} rows × {len(out.columns)} cols → {out_path.name}")
    return out_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--date", type=str, default=None)
    p.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT))
    p.add_argument("--force", action="store_true")
    p.add_argument("--watch", action="store_true",
                   help="Loop forever processing new Phase-1 parquets as they appear (5s poll)")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.date:
        merge_one_date(args.date, out_dir, force=args.force)
        return

    if args.watch:
        import time
        seen_done = set()
        log.info(f"WATCH MODE: polling {LABEL_DIR} every 5s for new parquets...")
        while True:
            label_paths = sorted(LABEL_DIR.glob("*_signals_labeled.parquet"))
            for lp in label_paths:
                date_str = lp.name[:8]
                if date_str in seen_done:
                    continue
                # Skip if already enriched
                ep = out_dir / f"{date_str}_signals_enriched.parquet"
                if ep.exists() and not args.force:
                    seen_done.add(date_str)
                    continue
                try:
                    merge_one_date(date_str, out_dir, force=args.force)
                    seen_done.add(date_str)
                except Exception as e:
                    log.error(f"[{date_str}] merge FAILED: {e}", exc_info=True)
            time.sleep(5)

    # Otherwise: process all Phase-1 parquets currently on disk, once.
    label_paths = sorted(LABEL_DIR.glob("*_signals_labeled.parquet"))
    log.info(f"Processing {len(label_paths)} Phase-1 parquets")
    ok = 0
    for lp in label_paths:
        date_str = lp.name[:8]
        try:
            r = merge_one_date(date_str, out_dir, force=args.force)
            if r is not None:
                ok += 1
        except Exception as e:
            log.error(f"[{date_str}] FAILED: {e}", exc_info=True)
    log.info(f"DONE: {ok}/{len(label_paths)} merged")


if __name__ == "__main__":
    main()
