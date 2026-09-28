#!/usr/bin/env python3
"""
HC #495.1 + HC #491 R5 — Parameterized FIFO grader for h5s low-LR multifold.

Apples-to-apples replica of scripts/h5s_lowlr_fifo_sweep.py (label-FIFO):
  signed_pnl_per_trade = sign(pred)*label - 0.376  (passive limit, commission only)

Adds q>=0.50 fill-prob gate variant using precomputed exec_features_v1
(fill_prob_1s for h=1s, fill_prob_3s as proxy for h=5s).

Usage:
  python3 grade_h5s_fold_N.py --fold 2
  python3 grade_h5s_fold_N.py --fold 2 --gate 0.50
  python3 grade_h5s_fold_N.py --fold 2 --pred-dir <dir> --out-dir <dir>
"""
import argparse, json
from pathlib import Path
import numpy as np

ROOT = Path("/home/nick/Lvl3Quant") if Path("/home/nick/Lvl3Quant").exists() else Path("/home/jupiter/Lvl3Quant")

DEFAULT_PRED_DIR = ROOT / "output" / "cnn_mamba_v3_h5s_lowlr_multifold"
MBO_DIR          = ROOT / "data" / "processed" / "mbo_events_smart_v3"
EXEC_DIR         = ROOT / "output" / "exec_features_v1"

WINDOW_SIZE = 1000
STRIDE      = 500
COMMISSION  = 0.376
HORIZONS    = ["1s", "5s"]
TOP_PCTS    = [1, 2, 5, 10, 20, 50]
SHARPE_MIN, REGIME_SKEW_MAX = 1.0, 0.50
TRADES_PER_DAY_MIN = 5
ANNUAL_SQRT = np.sqrt(252 * 6.5 * 3600)


def grade_one_variant(preds, labels_by_h, gate_mask=None, variant_label="ungated"):
    """Run the 12-cell sweep with optional per-window gate_mask (bool array, True=keep)."""
    results = []
    n = preds.shape[0]
    for h_i, h in enumerate(HORIZONS):
        p = preds[:, h_i]
        lab = labels_by_h[h]
        valid = ~np.isnan(lab)
        if gate_mask is not None:
            valid = valid & gate_mask
        p_v, lab_v = p[valid], lab[valid]
        if len(p_v) < 5:
            for pct in TOP_PCTS:
                results.append(dict(horizon=h, top_pct=pct, variant=variant_label,
                                    net_ticks=0.0, net_ticks_per_trade=0.0, count=0,
                                    sharpe=0.0, days_total=1, days_positive=0,
                                    trades_per_day=0.0, regime_skew=float("nan"),
                                    verdict="INSUFFICIENT_SAMPLES"))
            continue
        abs_p, sign_p = np.abs(p_v), np.sign(p_v)
        for pct in TOP_PCTS:
            k = max(1, int(np.ceil(len(p_v) * pct / 100.0)))
            idx_sorted = np.argsort(-abs_p)[:k]
            sig, l = sign_p[idx_sorted], lab_v[idx_sorted]
            per_trade = sig * l - COMMISSION
            net = float(per_trade.sum())
            cnt = int(len(per_trade))
            mean_t = float(per_trade.mean())
            std_t = float(per_trade.std(ddof=1)) if cnt > 1 else 0.0
            sh = (mean_t / std_t * ANNUAL_SQRT) if std_t > 0 else 0.0
            days_pos = 1 if net > 0 else 0
            if net > 0 and sh >= SHARPE_MIN and cnt / 1.0 >= TRADES_PER_DAY_MIN:
                v = "MARGINAL_1DAY"
            elif net > 0:
                v = "MARGINAL"
            else:
                v = "FAIL"
            results.append(dict(horizon=h, top_pct=pct, variant=variant_label,
                                net_ticks=net, net_ticks_per_trade=mean_t, count=cnt,
                                sharpe=sh, days_total=1, days_positive=days_pos,
                                trades_per_day=cnt / 1.0, regime_skew=float("nan"),
                                verdict=v))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True, help="Fold index (e.g. 2)")
    ap.add_argument("--gate", type=float, default=0.50, help="Fill-prob gate threshold (default 0.50)")
    ap.add_argument("--pred-dir", type=str, default=str(DEFAULT_PRED_DIR))
    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--out-tag", type=str, default="h5s_multifold")
    args = ap.parse_args()

    pred_dir = Path(args.pred_dir)
    pred_file = pred_dir / f"fold_{args.fold:02d}_oot_predictions.npz"
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "output" / f"{args.out_tag}_fold{args.fold}_fifo"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== HC #495.1 FOLD-{args.fold} FIFO GRADE ===")
    print(f"Pred file: {pred_file}")
    if not pred_file.exists():
        raise SystemExit(f"FATAL: predictions file not found: {pred_file}")

    d = np.load(pred_file, allow_pickle=True)
    preds = d["predictions"]
    oot_files = list(d["oot_files"])
    horizons = list(d["horizons"])
    print(f"VERIFY shape={preds.shape}, horizons={horizons}, oot_files={oot_files}")
    print(f"VERIFY first 3 rows:\n{preds[:3]}")
    print(f"VERIFY nonzero={np.count_nonzero(preds)}, nan={int(np.isnan(preds).sum())}")
    print(f"VERIFY min/max/mean/std = {preds.min():.4f}/{preds.max():.4f}/{preds.mean():.4f}/{preds.std():.4f}")
    assert preds.shape[1] == 2 and [str(h) for h in horizons] == ["1s","5s"], "shape/horizons mismatch"
    assert np.isnan(preds).sum() == 0, "predictions contain NaN"

    if len(oot_files) != 1:
        print(f"WARN: expected 1 OOT file, got {len(oot_files)} — using first only")
    date_str = Path(str(oot_files[0])).stem.replace("_mbo_events", "")
    print(f"OOT date: {date_str}")

    mbo = np.load(MBO_DIR / f"{date_str}_mbo_events.npz", allow_pickle=True)
    labels_1s = mbo["labels_1s"]
    labels_5s = mbo["labels_5s"]

    n = preds.shape[0]
    event_idx = np.arange(n) * STRIDE + (WINDOW_SIZE - 1)
    print(f"VERIFY mapping: {n} windows, max event_idx={event_idx.max()}, mbo length={len(labels_1s)}")
    # clip any over-end (shouldn't happen on aligned data)
    event_idx = np.clip(event_idx, 0, len(labels_1s) - 1)

    labels_by_h = {"1s": labels_1s[event_idx], "5s": labels_5s[event_idx]}

    # === Variant A: ungated ===
    ungated = grade_one_variant(preds, labels_by_h, gate_mask=None, variant_label="ungated")

    # === Variant B: q>=gate fill-prob gate ===
    gated = []
    gate_info = {}
    exec_path = EXEC_DIR / f"{date_str}_exec_features.npz"
    if exec_path.exists():
        ef = np.load(exec_path, allow_pickle=True)
        ef_feat = ef["features"]              # (n_decision, 44)
        ef_names = [str(x) for x in ef["feature_names"]]
        ef_n_windows = int(ef["n_windows"])
        ef_n_events = int(ef["n_events"])
        decision_stride = int(ef["decision_stride"])
        col_1s = ef_names.index("fill_prob_1s")
        col_5s = ef_names.index("fill_prob_3s")  # 3s proxy for 5s (no 5s head trained)
        # Map each prediction window (event_idx) to the nearest decision-stride row
        # Decision events at: k * decision_stride, k=0..ef_n_windows-1
        ef_event_centers = np.arange(ef_n_windows) * decision_stride
        # For each pred event, find nearest decision row
        idx_map = np.searchsorted(ef_event_centers, event_idx, side="left")
        idx_map = np.clip(idx_map, 0, ef_n_windows - 1)
        # Pick the closer of idx_map vs idx_map-1
        left = np.clip(idx_map - 1, 0, ef_n_windows - 1)
        d_left = np.abs(event_idx - ef_event_centers[left])
        d_right = np.abs(event_idx - ef_event_centers[idx_map])
        choose_left = d_left < d_right
        nearest = np.where(choose_left, left, idx_map)

        fp_1s = ef_feat[nearest, col_1s]
        fp_5s = ef_feat[nearest, col_5s]
        # For the multi-horizon sweep, build per-h gate masks; reuse grade_one_variant
        # Slight twist: gate mask is per-horizon, so call twice and merge
        # Variant B per-h: gate by the matching fill_prob column
        gated_1s = grade_one_variant(preds[:, :1].repeat(2, axis=1) * 0, {"1s": labels_by_h["1s"], "5s": labels_by_h["5s"]},
                                     gate_mask=None, variant_label="placeholder")
        # Clean: do directly by horizon
        results_b = []
        for h_i, h in enumerate(HORIZONS):
            gate_q = fp_1s if h == "1s" else fp_5s
            mask = gate_q >= args.gate
            sub = grade_one_variant(preds, labels_by_h, gate_mask=mask, variant_label=f"gate_q>={args.gate}")
            # take only rows for this horizon
            results_b.extend([r for r in sub if r["horizon"] == h])
        gated = results_b
        gate_info = {
            "exec_features": str(exec_path),
            "gate_threshold": args.gate,
            "n_total": int(n),
            "n_pass_gate_1s": int((fp_1s >= args.gate).sum()),
            "n_pass_gate_5s_proxy3s": int((fp_5s >= args.gate).sum()),
            "fp_1s_min_max_mean": [float(fp_1s.min()), float(fp_1s.max()), float(fp_1s.mean())],
            "fp_5s_proxy_min_max_mean": [float(fp_5s.min()), float(fp_5s.max()), float(fp_5s.mean())],
            "note_5s_uses_3s_proxy": True,
        }
        print(f"Gate {args.gate}: 1s_pass={gate_info['n_pass_gate_1s']}/{n}, 5s_proxy3s_pass={gate_info['n_pass_gate_5s_proxy3s']}/{n}")
    else:
        print(f"WARN: exec_features not found at {exec_path} — skipping gated variant")
        for h in HORIZONS:
            for pct in TOP_PCTS:
                gated.append(dict(horizon=h, top_pct=pct, variant=f"gate_q>={args.gate}",
                                  net_ticks=0.0, net_ticks_per_trade=0.0, count=0,
                                  sharpe=0.0, days_total=1, days_positive=0,
                                  trades_per_day=0.0, regime_skew=float("nan"),
                                  verdict="EXEC_FEATURES_MISSING"))

    # === Write CSV ===
    csv_path = out_dir / "summary.csv"
    all_results = ungated + gated
    with open(csv_path, "w") as f:
        f.write("variant,horizon,top_pct,net_ticks,net_ticks_per_trade,count,days_total,days_positive,trades_per_day,sharpe,regime_skew,verdict\n")
        for r in all_results:
            f.write(f"{r['variant']},{r['horizon']},{r['top_pct']},{r['net_ticks']:.4f},"
                    f"{r['net_ticks_per_trade']:.4f},{r['count']},{r['days_total']},"
                    f"{r['days_positive']},{r['trades_per_day']:.2f},{r['sharpe']:.2f},"
                    f"{r['regime_skew']},{r['verdict']}\n")

    # === JSON for downstream multifold aggregator ===
    meta = dict(fold=args.fold, date=date_str, n_windows=int(n), pred_file=str(pred_file),
                results=all_results, gate_info=gate_info)
    with open(out_dir / "result.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    # === Headline print ===
    print("\n" + "="*100)
    print(f"FOLD {args.fold} SUMMARY (OOT {date_str})")
    print("="*100)
    print(f"{'Variant':<22}{'Horizon':<8}{'Top%':<6}{'NetTicks':>10}{'T/Trade':>10}{'Count':>8}{'Sharpe':>10}{'Verdict':>20}")
    for r in all_results:
        print(f"{r['variant']:<22}{r['horizon']:<8}{r['top_pct']:<6}{r['net_ticks']:>10.2f}"
              f"{r['net_ticks_per_trade']:>10.4f}{r['count']:>8}{r['sharpe']:>10.2f}{r['verdict']:>20}")

    headline = [r for r in all_results if r["horizon"] == "1s" and r["top_pct"] == 10]
    print("\nHEADLINE (1s_top10%):")
    for r in headline:
        print(f"  [{r['variant']}] net={r['net_ticks']:.2f} | t/trade={r['net_ticks_per_trade']:+.4f} | "
              f"N={r['count']} | Sharpe={r['sharpe']:.2f} | {r['verdict']}")

    print(f"\nOutput: {out_dir}/summary.csv  +  result.json")


if __name__ == "__main__":
    main()
