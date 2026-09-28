#!/usr/bin/env python3
"""
HC #264 — Validation test for supervised_exec_v3 h15 walk-forward results.

Independently recomputes top-X% metrics from fold_NN_predictions.npz and
compares against the trainer's printed log to confirm:
  1. The trainer's threshold_analysis numbers match what the saved npz says.
  2. Cross-fold consistency: % folds with positive top-1% PnL (HC #254 ≥60%).
  3. Single-fold concentration: max single-fold contribution / total (HC #254 ≤40%).
  4. Net-of-commission ticks (subtract 0.376 t RT).

Sort key per the trainer: predicted_mfe (col 0 of preds), descending.
PnL units in the saved npz are GROSS ticks (no commission subtracted).
"""

from pathlib import Path
import re
import sys
import numpy as np

OUT_DIR = Path("/home/jupiter/Lvl3Quant/output/supervised_exec_v3_h15_cpu_v2")
LOG_PATH = Path("/home/jupiter/Lvl3Quant/logs/supervised_exec_v3_h15_cpu_v2.log")
COMMISSION_TICKS = 0.376  # ES AMP RT
TIERS = [("top_1pct", 0.01), ("top_5pct", 0.05), ("top_10pct", 0.10),
         ("top_20pct", 0.20), ("top_50pct", 0.50), ("all", 1.00)]


def fold_metrics(npz_path: Path, tier_pct: float):
    d = np.load(npz_path, allow_pickle=True)
    preds = d["predictions"]
    tgts = d["targets"]
    oot_dates = d["oot_dates"]
    pred_mfe = preds[:, 0]
    actual_mfe = tgts[:, 0]
    actual_pnl = tgts[:, 2]
    actual_win = tgts[:, 3]
    n = len(preds)
    sort_idx = np.argsort(-pred_mfe)
    pnl_sorted = actual_pnl[sort_idx]
    mfe_sorted = actual_mfe[sort_idx]
    win_sorted = actual_win[sort_idx]
    k = max(1, int(n * tier_pct))
    sub_pnl = pnl_sorted[:k]
    sub_mfe = mfe_sorted[:k]
    sub_win = win_sorted[:k]
    wr = float(np.mean(sub_win))
    avg_pnl = float(np.mean(sub_pnl))
    sum_pnl = float(np.sum(sub_pnl))
    avg_mfe = float(np.mean(sub_mfe))
    wins = sub_pnl[sub_pnl > 0]
    losses = sub_pnl[sub_pnl < 0]
    pf = float(np.sum(wins) / max(1e-8, abs(np.sum(losses)))) if len(losses) else float("inf")
    if len(sub_pnl) > 1:
        downside = sub_pnl[sub_pnl < 0]
        if len(downside) > 0:
            dd = float(np.sqrt(np.mean(downside ** 2)))
            sortino = float(avg_pnl / (dd + 1e-8))
        else:
            sortino = avg_pnl * 10.0
    else:
        sortino = 0.0
    net_per_trade = avg_pnl - COMMISSION_TICKS
    return {
        "n": k,
        "wr": wr,
        "avg_pnl_gross": avg_pnl,
        "avg_pnl_net": net_per_trade,
        "sum_pnl_gross": sum_pnl,
        "sum_pnl_net": sum_pnl - k * COMMISSION_TICKS,
        "avg_mfe": avg_mfe,
        "pf": pf,
        "sortino": sortino,
        "oot_date": str(oot_dates[0]) if len(oot_dates) else "?",
    }


def parse_log_top1pct():
    """Pull the trainer's printed top_1pct PnL per fold for cross-check."""
    if not LOG_PATH.exists():
        return {}
    txt = LOG_PATH.read_text()
    out = {}
    cur_fold = None
    for line in txt.splitlines():
        m = re.search(r"FOLD (\d+) OOT Results:", line)
        if m:
            cur_fold = int(m.group(1))
            continue
        if cur_fold is not None:
            m = re.search(r"top_1pct: n=\s*(\d+) WR=([\d.]+) PnL=([+-][\d.]+) MFE=([\d.]+) PF=([\d.]+) Sortino=([+-]?[\d.]+)", line)
            if m:
                out[cur_fold] = {
                    "n": int(m.group(1)),
                    "wr": float(m.group(2)),
                    "pnl": float(m.group(3)),
                    "mfe": float(m.group(4)),
                    "pf": float(m.group(5)),
                    "sortino": float(m.group(6)),
                }
                cur_fold = None
    return out


def main():
    fold_paths = sorted(OUT_DIR.glob("fold_*_predictions.npz"))
    if not fold_paths:
        print("FAIL: no fold prediction files found in", OUT_DIR)
        sys.exit(2)
    log_top1 = parse_log_top1pct()
    rows = []
    print(f"{'fold':>4} {'date':>10} {'n':>4} {'WR':>5} {'PnL_g':>7} {'PnL_n':>7} "
          f"{'MFE':>6} {'PF':>6} {'Sort':>6}  log_match")
    print("-" * 84)
    mismatches = 0
    for p in fold_paths:
        fid = int(re.search(r"fold_(\d+)_", p.name).group(1))
        m = fold_metrics(p, 0.01)
        rows.append({"fold": fid, **m})
        log_m = log_top1.get(fid)
        if log_m is None:
            match = "no_log"
        elif (abs(log_m["pnl"] - m["avg_pnl_gross"]) < 0.01 and
              abs(log_m["pf"] - m["pf"]) < 0.05 and
              log_m["n"] == m["n"]):
            match = "OK"
        else:
            match = (f"MISMATCH log_pnl={log_m['pnl']:+.3f} "
                     f"vs npz_pnl={m['avg_pnl_gross']:+.3f}")
            mismatches += 1
        print(f"{fid:>4} {m['oot_date']:>10} {m['n']:>4} {m['wr']:.3f} "
              f"{m['avg_pnl_gross']:+7.3f} {m['avg_pnl_net']:+7.3f} "
              f"{m['avg_mfe']:6.3f} {m['pf']:6.2f} {m['sortino']:+6.3f}  {match}")
    n_folds = len(rows)
    pos_gross = sum(1 for r in rows if r["avg_pnl_gross"] > 0)
    pos_net = sum(1 for r in rows if r["avg_pnl_net"] > 0)
    total_gross = sum(r["sum_pnl_gross"] for r in rows)
    total_net = sum(r["sum_pnl_net"] for r in rows)
    if total_gross != 0:
        max_share_gross = max(abs(r["sum_pnl_gross"] / total_gross)
                              for r in rows) if total_gross else 0
    else:
        max_share_gross = 1.0
    print()
    print(f"Folds completed:                 {n_folds}")
    print(f"% folds with positive top-1% gross:  {100*pos_gross/n_folds:.1f}% "
          f"(HC #254 floor: 60%)")
    print(f"% folds with positive top-1% NET:    {100*pos_net/n_folds:.1f}%")
    print(f"Sum top-1% PnL gross (ticks):    {total_gross:+.2f}")
    print(f"Sum top-1% PnL net   (ticks):    {total_net:+.2f}")
    print(f"Max single-fold gross share:     {100*max_share_gross:.1f}% "
          f"(HC #254 ceiling: 40%)")
    print(f"Trainer-log mismatches:          {mismatches} / {n_folds}")
    if total_gross > 0:
        # extrapolate $ assuming 235 trades/day average top-1% (fold-0 baseline)
        avg_n = np.mean([r["n"] for r in rows])
        avg_pnl_net = total_net / sum(r["n"] for r in rows)
        usd_per_day = avg_pnl_net * avg_n * 12.50
        print(f"Avg trades/day top-1%:           {avg_n:.0f}")
        print(f"Avg net ticks/trade:             {avg_pnl_net:+.3f}")
        print(f"Implied $/day (1 contract):      ${usd_per_day:+,.0f}")
    pf_overall_gross = (
        sum(max(0, r["sum_pnl_gross"]) for r in rows)
        / max(1e-8, sum(max(0, -r["sum_pnl_gross"]) for r in rows))
    )
    print(f"Overall PF (sum positive / sum negative folds): {pf_overall_gross:.2f}")

    # ---- Robust (median) aggregator — outlier-resistant ----
    pnls_gross = np.array([r["avg_pnl_gross"] for r in rows])
    pnls_net = np.array([r["avg_pnl_net"] for r in rows])
    median_gross = float(np.median(pnls_gross))
    median_net = float(np.median(pnls_net))
    # Trimmed mean (drop top + bottom fold)
    if len(pnls_net) >= 3:
        sorted_net = np.sort(pnls_net)
        trimmed_net = float(np.mean(sorted_net[1:-1]))
    else:
        trimmed_net = float(np.mean(pnls_net))
    print()
    print("ROBUST AGGREGATORS (outlier-resistant):")
    print(f"  Median gross ticks/trade:      {median_gross:+.3f}")
    print(f"  Median NET ticks/trade:        {median_net:+.3f}")
    print(f"  Trimmed-mean NET ticks/trade:  {trimmed_net:+.3f}  "
          f"(drop best+worst fold)")
    avg_n = np.mean([r["n"] for r in rows])
    print(f"  Median-based $/day (1 ctr):    "
          f"${median_net * avg_n * 12.50:+,.0f}")
    print(f"  Trimmed-based $/day (1 ctr):   "
          f"${trimmed_net * avg_n * 12.50:+,.0f}")
    # Verdict
    print()
    pass_consistency = (pos_gross / n_folds) >= 0.60
    pass_concentration = max_share_gross <= 0.40
    pass_log_match = mismatches == 0
    pass_net_pos = total_net > 0
    print("HC #254 consistency (≥60% pos):  ", "PASS" if pass_consistency else "FAIL")
    print("HC #254 concentration (≤40%):    ", "PASS" if pass_concentration else "FAIL")
    print("Log↔NPZ trainer integrity:       ", "PASS" if pass_log_match else "FAIL")
    print("Net-of-commission profitable:    ", "PASS" if pass_net_pos else "FAIL")
    overall = pass_consistency and pass_concentration and pass_log_match and pass_net_pos
    print()
    print("OVERALL:", "PASS — top-1% gate is real" if overall else "FAIL — see above")
    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
