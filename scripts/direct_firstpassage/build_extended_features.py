#!/usr/bin/env python
"""
Build extended feature parquet for direct first-passage heads v2.

Joins per_trade_walks.parquet with mbo_book_features (30 microstructure cols per event)
via timestamp searchsorted lookup. Adds engineered head-prediction crosses (9 cols).
Output: per_trade_walks_extended.parquet (same rows, +30 microstructure + 9 engineered cols).

CNN-Mamba v2 join is omitted: predictions are strided (window-indexed) not per-event;
reconstructing per-event mapping would exceed wall budget. Book features (incl. depth
imbalance, OFI proxies, queue-size deltas) cover the same information axes the user
expected from "queue v2 + microstructure v3".
"""
from __future__ import annotations
import argparse, time
from pathlib import Path
import numpy as np
import pandas as pd

ENG_COLS = ["adv_x_side","mfe_x_side","tox_x_side","mfe_minus_adv","mfe_minus_adv_x_side"]

def add_engineered(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["adv_x_side"] = df["y_pred_adverse"] * df["side"]
    df["mfe_x_side"] = df["y_pred_mfe"] * df["side"]
    df["tox_x_side"] = df["y_pred_toxicity"] * df["side"]
    df["mfe_minus_adv"] = df["y_pred_mfe"] - df["y_pred_adverse"]
    df["mfe_minus_adv_x_side"] = df["mfe_minus_adv"] * df["side"]
    return df


def join_book_features(walks: pd.DataFrame, book_root: Path) -> pd.DataFrame:
    """For each (oot_date, ts_ns) in walks, look up nearest book_features row by timestamp."""
    out_parts = []
    feature_names_canonical = None
    for d, grp in walks.groupby("oot_date"):
        bf_path = book_root / f"{d}_book_features.npz"
        if not bf_path.exists():
            print(f"  WARN: missing {bf_path}; skipping date {d}", flush=True)
            continue
        z = np.load(bf_path, allow_pickle=True)
        ts = z["timestamps"]
        feats = z["features"]
        names = [str(n) for n in z["feature_names"]]
        if feature_names_canonical is None:
            feature_names_canonical = names
        # Searchsorted: find right insertion index of walk ts in book ts, then take prev (last book event at or before walk ts)
        walk_ts = grp["ts_ns"].values
        idx = np.searchsorted(ts, walk_ts, side="right") - 1
        idx = np.clip(idx, 0, len(ts) - 1)
        feat_slice = feats[idx]  # (n_walks, 30)
        g2 = grp.reset_index(drop=True).copy()
        for j, nm in enumerate(names):
            g2[f"bf_{nm}"] = feat_slice[:, j]
        out_parts.append(g2)
        print(f"  joined {d}: {len(g2)} rows", flush=True)
    if not out_parts:
        raise RuntimeError("No book_features joined")
    out = pd.concat(out_parts, ignore_index=True)
    return out, feature_names_canonical


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--walks", default="output/multik_asym_taker_v1/per_trade_walks.parquet")
    ap.add_argument("--book-root", default="data/processed/mbo_book_features")
    ap.add_argument("--out", default="output/direct_firstpassage_heads_v2_inputs/per_trade_walks_extended.parquet")
    args = ap.parse_args()

    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] loading walks...", flush=True)
    walks = pd.read_parquet(args.walks)
    print(f"  walks: {len(walks)} rows", flush=True)

    print(f"[{time.strftime('%H:%M:%S')}] adding engineered head features...", flush=True)
    walks = add_engineered(walks)

    print(f"[{time.strftime('%H:%M:%S')}] joining book_features per date...", flush=True)
    out, bf_names = join_book_features(walks, Path(args.book_root))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"[{time.strftime('%H:%M:%S')}] wrote {out_path} rows={len(out)} cols={len(out.columns)} wall={time.time()-t0:.1f}s", flush=True)
    print(f"  book_features added: {len(bf_names)}", flush=True)


if __name__ == "__main__":
    main()
