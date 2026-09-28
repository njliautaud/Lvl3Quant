#!/usr/bin/env python3
"""
build_class_features.py — Build classification features + trigger labels for
the confluence-trigger classifier (Razer meta-classifier).

Reads the existing meta_features.parquet (32 pred_* heads + targets + date)
and adds 4 binary trigger flags + 2 labels.

Trigger flags (computed within each date — confidence ranks are intra-day):
  trig_1s_x_5s_pup      : top-5% |pred_log_ret_1s| AND top-5% |pred_p_up_5s - 0.5| AND sign-agreement
  trig_30s_q90_x_1s     : top-10% |pred_log_ret_30s_q90| AND top-10% |pred_log_ret_1s| AND sign-agreement
  trig_30s_x_1s         : top-20% |pred_log_ret_30s|     AND top-20% |pred_log_ret_1s| AND sign-agreement
  trig_mfe_x_1s         : top-10% |pred_pred_mfe_30s_ticks| AND top-10% |pred_log_ret_1s| AND sign-agreement

Labels:
  y_trigger              : OR over the 4 trigger flags
  y_profitable_trigger   : y_trigger AND realized 5s move in predicted direction > 0.5 ticks

Output: /home/jupiter/Lvl3Quant/output/razer_classifier/class_features.parquet

ES_TICK conversion: log return × notional ≈ ticks; but we already have target_log_ret_5s in
the same units as the prediction heads (model-emitted log return). The 0.5-tick profitability
floor is applied AS LOG-RETURN MAGNITUDE in the same units — i.e., we use the model's own
output scale. This matches what stream_continuation_backtest.py does (target_log_ret_5s
is treated as ticks directly per the existing harness).
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path("/home/jupiter/Lvl3Quant/output/razer_meta/meta_features.parquet")
OUT_DIR = Path("/home/jupiter/Lvl3Quant/output/razer_classifier")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = OUT_DIR / "class_features.parquet"

PROFIT_FLOOR_TICKS = 0.5  # passive-limit profitability floor (commission 0.376 + margin)

# Direction signal: head value → signed direction. For p_up, center at 0.5.
def directional_signal(name: str, vals: np.ndarray) -> np.ndarray:
    if "p_up" in name:
        return vals - 0.5
    if "p_reversal" in name:
        return -(vals - 0.5)  # higher reversal probability = signal flip
    return vals


def top_q_mask(vals: np.ndarray, q_frac: float) -> np.ndarray:
    """Mask of events whose |vals| rank in top q_frac (rank-based)."""
    n = len(vals)
    if n == 0:
        return np.zeros(0, dtype=bool)
    abs_v = np.abs(vals)
    # rank descending; top q_frac of the distribution
    k = max(1, int(np.ceil(n * q_frac)))
    threshold = np.partition(abs_v, n - k)[n - k]
    return abs_v >= threshold


def trigger_flag(df: pd.DataFrame, head_a: str, head_b: str, q_frac: float) -> np.ndarray:
    """Compute a single trigger flag across the whole dataset, per-date ranked."""
    out = np.zeros(len(df), dtype=bool)
    for date, idx in df.groupby("date").groups.items():
        idx = np.asarray(idx)
        a = df[head_a].values[idx]
        b = df[head_b].values[idx]
        sa = directional_signal(head_a, a)
        sb = directional_signal(head_b, b)
        mask_a = top_q_mask(sa, q_frac)
        mask_b = top_q_mask(sb, q_frac)
        sign_ok = np.sign(sa) == np.sign(sb)
        flag = mask_a & mask_b & sign_ok & (np.sign(sa) != 0)
        out[idx] = flag
    return out


def main():
    print(f"[build_class] loading {SRC}")
    df = pd.read_parquet(SRC)
    print(f"[build_class] rows: {len(df):,}  dates: {df['date'].nunique()}")
    print(f"[build_class] columns: {len(df.columns)}")

    # Sanity: confirm needed columns exist
    needed = [
        "pred_log_ret_1s", "pred_p_up_5s",
        "pred_log_ret_30s_q90", "pred_log_ret_30s",
        "pred_pred_mfe_30s_ticks",
        "target_log_ret_5s",
    ]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        print(f"[build_class] MISSING COLUMNS: {missing}", file=sys.stderr)
        # Try to recover with similar names
        for m in missing:
            cands = [c for c in df.columns if m.split("_")[-1] in c]
            print(f"  '{m}' -> similar columns: {cands[:6]}", file=sys.stderr)
        sys.exit(1)

    # Drop NaN target rows first (same filter the regression pipeline used)
    before = len(df)
    df = df.dropna(subset=["target_log_ret_5s"]).reset_index(drop=True)
    print(f"[build_class] rows after dropna(target_log_ret_5s): {before:,} -> {len(df):,}")

    # Compute trigger flags
    print("[build_class] computing trig_1s_x_5s_pup ...")
    df["trig_1s_x_5s_pup"] = trigger_flag(df, "pred_log_ret_1s", "pred_p_up_5s", 0.05)
    print(f"  fires: {df['trig_1s_x_5s_pup'].sum():,} ({df['trig_1s_x_5s_pup'].mean()*100:.3f}%)")

    print("[build_class] computing trig_30s_q90_x_1s ...")
    df["trig_30s_q90_x_1s"] = trigger_flag(df, "pred_log_ret_30s_q90", "pred_log_ret_1s", 0.10)
    print(f"  fires: {df['trig_30s_q90_x_1s'].sum():,} ({df['trig_30s_q90_x_1s'].mean()*100:.3f}%)")

    print("[build_class] computing trig_30s_x_1s ...")
    df["trig_30s_x_1s"] = trigger_flag(df, "pred_log_ret_30s", "pred_log_ret_1s", 0.20)
    print(f"  fires: {df['trig_30s_x_1s'].sum():,} ({df['trig_30s_x_1s'].mean()*100:.3f}%)")

    print("[build_class] computing trig_mfe_x_1s ...")
    df["trig_mfe_x_1s"] = trigger_flag(df, "pred_pred_mfe_30s_ticks", "pred_log_ret_1s", 0.10)
    print(f"  fires: {df['trig_mfe_x_1s'].sum():,} ({df['trig_mfe_x_1s'].mean()*100:.3f}%)")

    # OR label
    df["y_trigger"] = (
        df["trig_1s_x_5s_pup"] |
        df["trig_30s_q90_x_1s"] |
        df["trig_30s_x_1s"] |
        df["trig_mfe_x_1s"]
    ).astype(np.int8)
    print(f"[build_class] y_trigger: {df['y_trigger'].sum():,} ({df['y_trigger'].mean()*100:.3f}%)")

    # Profitable trigger: y_trigger AND realized 5s move in predicted direction > floor
    # Direction from pred_log_ret_1s sign
    pred_dir = np.sign(df["pred_log_ret_1s"].values)
    realized_dir_move = pred_dir * df["target_log_ret_5s"].values  # signed
    df["y_profitable_trigger"] = (
        (df["y_trigger"] == 1) & (realized_dir_move > PROFIT_FLOOR_TICKS)
    ).astype(np.int8)
    n_prof = int(df["y_profitable_trigger"].sum())
    print(f"[build_class] y_profitable_trigger: {n_prof:,} ({df['y_profitable_trigger'].mean()*100:.3f}%)")
    print(f"  conditional rate (given y_trigger=1): {n_prof / max(1, df['y_trigger'].sum())*100:.2f}%")

    # Save
    print(f"[build_class] writing {OUT}")
    df.to_parquet(OUT, compression="zstd", index=False)
    print(f"[build_class] done. file size: {OUT.stat().st_size / 1e6:.1f} MB")

    # Print label stats per-date for sanity
    print("\n[build_class] per-date label stats:")
    grp = df.groupby("date").agg(
        n_events=("y_trigger", "size"),
        n_trig=("y_trigger", "sum"),
        n_prof=("y_profitable_trigger", "sum"),
    )
    grp["trig_rate"] = grp["n_trig"] / grp["n_events"]
    grp["prof_rate"] = grp["n_prof"] / grp["n_events"]
    print(grp.to_string())


if __name__ == "__main__":
    main()
