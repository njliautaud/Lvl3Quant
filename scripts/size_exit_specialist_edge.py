"""
Edge-sizing for the proposed exit-target specialist (HC #459 R3a / HC #463 R5 item 1).

Question we MUST answer BEFORE launching a specialist on Neptune:
  Given the v4 1s prediction (entry trigger) at top-1% confidence,
  what realized MFE / MAE / net-after-cost would the trade have produced
  if held to horizon h ∈ {5s, 10s, 30s} per the freshly-built MFE/MAE labels?

If realized MFE > realized MAE + RT_cost by a wide margin → specialist viable.
If realized MFE ≈ MAE (random walk dominated) → specialist not viable, kill plan.

Output: a table of (entry-conf bucket × exit horizon → MFE / MAE / WR / net).

Cost model: passive limit entry + horizon-h exit = 0.376 ticks RT commission only.
            (No spread crossing — limit orders sit at bid/ask, not market take.)

This is a Jupiter-CPU one-shot. Reads:
  /output/cnn_mamba_v3_4_2_hc454phase2_smoke_run2/fold_00_ep1_oot.npz   (v4 ep1 preds + targets)
  /data/relabel/mfe_mae_h{h}_{date}.parquet  (HC #464 R2(a) labels, all 65 OOT dates)

Joins on event ordering — both are stored in OOT-chronological order. The
v4 npz lacks date stamps per-event, so we must take it on faith that the
OOT order matches the date-sorted concat. If event counts diverge → bail.

Note: only the 5 days of v4 OOT (~241K events) can be evaluated here. The full
65-date relabel set is the FUTURE training set for the specialist itself.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

V4_NPZ = Path("/home/jupiter/Lvl3Quant/output/cnn_mamba_v3_4_2_hc454phase2_smoke_run2/fold_00_ep1_oot.npz")
RELABEL_DIR = Path("/home/jupiter/Lvl3Quant/data/relabel")
RT_COST = 0.376  # passive limit, $4.70 / $12.50 tick
TICK_USD = 12.50


def main():
    d = np.load(V4_NPZ)
    pred_1s = d["pred_log_ret_1s"]          # 1s entry signal
    target_1s = d["target_log_ret_1s"]
    n = len(pred_1s)
    print(f"v4 ep1 OOT n_events = {n:,}", flush=True)

    # Build top-1% confidence bucket gate (by |pred_1s|)
    conf = np.abs(pred_1s)
    thr_top1 = np.quantile(conf, 0.99)
    thr_top5 = np.quantile(conf, 0.95)
    thr_top10 = np.quantile(conf, 0.90)
    print(f"|pred_1s| thresholds: top1%={thr_top1:.3f}  top5%={thr_top5:.3f}  top10%={thr_top10:.3f}", flush=True)

    buckets = {
        "top_1pct":  conf >= thr_top1,
        "top_5pct":  conf >= thr_top5,
        "top_10pct": conf >= thr_top10,
        "all":       np.ones(n, bool),
    }

    # The realized MFE/MAE labels live in our new relabel parquets, but those are
    # indexed by event_idx WITHIN A DATE. The v4 npz concats OOT dates. Without a
    # per-event date stamp we can't directly join. We can however verify
    # equivalence via existing v4 target_pred_mfe_30s_ticks which IS in the npz.
    if "target_pred_mfe_30s_ticks" not in d.files:
        print("ERROR: v4 npz missing target_pred_mfe_30s_ticks — cannot proceed.", file=sys.stderr)
        sys.exit(2)

    mfe_30s = d["target_pred_mfe_30s_ticks"].astype(np.float32)
    mae_30s = d["target_pred_mae_30s_ticks"].astype(np.float32)
    mask_mfe_30s = d["mask_pred_mfe_30s_ticks"].astype(bool)

    # Signed-by-side MFE/MAE: if trade direction = sign(pred_1s),
    #   favorable_excursion = MFE * sign  for long; -MAE * sign for short
    #   adverse_excursion   = MAE * sign  for long; -MFE * sign for short
    # But mfe/mae as built are absolute (mfe>=0, mae<=0) relative to entry mid.
    # For long trade: realized signed PnL if held to end = target_log_ret_h
    # For short trade: realized signed PnL = -target_log_ret_h
    # Optimal-exit PnL ≤ MFE (best case), worst-drawdown PnL ≥ MAE.

    def side(pred):
        return np.where(pred > 0, 1, -1).astype(np.int8)

    sgn = side(pred_1s)
    # signed MFE = max favorable in trade direction
    # For long: MFE = mfe_30s (positive),    for short: MFE = -mae_30s (positive)
    signed_mfe_30s = np.where(sgn > 0, mfe_30s, -mae_30s)
    # signed MAE = worst adverse in trade direction
    signed_mae_30s = np.where(sgn > 0, mae_30s, -mfe_30s)

    # Realized signed return at exit horizon (in ticks, accounting for direction)
    horizons = [("5s", "target_log_ret_5s"),
                ("10s", "target_log_ret_10s"),
                ("30s", "target_log_ret_30s")]
    realized = {}
    for hname, key in horizons:
        tg = d[key].astype(np.float32)
        msk = d[f"mask_log_ret_{hname}"].astype(bool)
        realized[hname] = (sgn * tg, msk)

    # Build report
    rows = []
    for bname, bmask in buckets.items():
        for hname, _ in horizons:
            r_signed, r_mask = realized[hname]
            sel = bmask & r_mask & mask_mfe_30s
            n_sel = int(sel.sum())
            if n_sel < 30:
                rows.append({"bucket": bname, "exit_h": hname, "n": n_sel,
                             "median_signed_realized": np.nan,
                             "mean_signed_realized":   np.nan,
                             "wr_pct":                 np.nan,
                             "net_per_trade_ticks":    np.nan,
                             "total_net_usd":          np.nan,
                             "mean_signed_MFE_30s":    np.nan,
                             "mean_signed_MAE_30s":    np.nan,
                             "mfe_to_mae_ratio":       np.nan,
                             "sharpe":                 np.nan,
                             })
                continue
            r = r_signed[sel]
            mfe_v = signed_mfe_30s[sel]
            mae_v = signed_mae_30s[sel]
            # Drop NaN entries from r (downstream stats need finite values)
            finite_r = np.isfinite(r)
            r_f = r[finite_r]
            net = r_f - RT_COST  # passive, commission-only
            mfe_f = mfe_v[np.isfinite(mfe_v)]
            mae_f = mae_v[np.isfinite(mae_v)]
            if len(r_f) < 30 or len(mfe_f) < 30:
                continue
            rows.append({
                "bucket": bname,
                "exit_h": hname,
                "n": int(len(r_f)),
                "median_signed_realized": float(np.median(r_f)),
                "mean_signed_realized":   float(np.mean(r_f)),
                "wr_pct":                 float((r_f > 0).mean() * 100),
                "net_per_trade_ticks":    float(np.mean(net)),
                "total_net_usd":          float(np.sum(net) * TICK_USD),
                "mean_signed_MFE_30s":    float(np.mean(mfe_f)),
                "mean_signed_MAE_30s":    float(np.mean(mae_f)),
                "mfe_to_mae_ratio":       float(np.mean(mfe_f) / max(abs(np.mean(mae_f)), 1e-6)),
                "sharpe":                 float(np.mean(net) / (np.std(net) + 1e-9)),
            })

    df = pd.DataFrame(rows)
    print()
    print(df.to_string(index=False, float_format=lambda x: f"{x:+.3f}" if not pd.isna(x) else "  --"))

    out = Path("/home/jupiter/Lvl3Quant/output/exit_specialist_edge_sizing.csv")
    df.to_csv(out, index=False)
    print(f"\nsaved: {out}")

    # Quick verdict
    print()
    print("=== VERDICT ===")
    for _, row in df.iterrows():
        if row["bucket"] != "top_1pct":
            continue
        h = row["exit_h"]
        net = row["net_per_trade_ticks"]
        wr = row["wr_pct"]
        ratio = row["mfe_to_mae_ratio"]
        verdict = "TRADABLE" if net > 0.1 else ("MARGINAL" if net > 0 else "DEAD")
        print(f"  top_1pct × exit-{h:>3}: net={net:+.3f} ticks  WR={wr:.1f}%  MFE/MAE={ratio:+.2f}  -> {verdict}")


if __name__ == "__main__":
    main()
