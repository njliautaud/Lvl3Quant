#!/usr/bin/env python3
"""HC #458 R1 — Tradability matrix per horizon.

Loads a multi-horizon predictions NPZ (v3.x OOT format) and reports, per
horizon: IC, predicted-move distribution in ticks, hit-rates above
market/passive break-even, and simulated FIFO net P&L at each threshold.

The point: an IC of 0.30 with predicted moves of 0.5 ticks is REJECTED;
an IC of 0.15 with predicted moves of 2 ticks is ACCEPTED.

Usage:
    python3 scripts/hc458_tradability_matrix.py <path_to_oot_npz>

Output:
    per-horizon table + JSON summary alongside the input file.
"""
import json
import sys
from pathlib import Path

import numpy as np

# ES contract / cost constants (CLAUDE.md canonical)
ES_TICK_VALUE = 12.50          # USD per tick
ES_RT_COMMISSION_TICKS = 0.376 # $4.70 / $12.50
MARKET_COST_TICKS = 1.376      # commission + 1.0 tick spread crossing
PASSIVE_COST_TICKS = 0.376     # commission only

# NOTE: Despite the field name "log_ret_*", v3.x targets and predictions
# are ALREADY IN TICKS (verified empirically: target_log_ret_30s ranges
# -50 to +50, abs_p90 ≈ 13 ticks). No conversion needed.
LOG_RET_PER_TICK = 1.0  # identity — pred/target ARE ticks

HORIZONS = ["1s", "5s", "10s", "30s", "60s", "5min"]

# Thresholds in ticks (predicted magnitude)
THRESHOLDS = [0.5, 1.0, 1.4, 2.0, 3.0]


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 10:
        return float("nan")
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    d = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def pct(arr: np.ndarray, q: float) -> float:
    return float(np.percentile(arr, q)) if arr.size else float("nan")


def analyze_horizon(d, h: str) -> dict:
    pred_key = f"pred_log_ret_{h}"
    tgt_key = f"target_log_ret_{h}"
    mask_key = f"mask_log_ret_{h}"
    if pred_key not in d:
        return {"horizon": h, "error": "missing pred key"}

    pred = d[pred_key]
    tgt = d[tgt_key]
    mask = d[mask_key].astype(bool) if mask_key in d else np.ones_like(pred, dtype=bool)
    valid = mask & np.isfinite(pred) & np.isfinite(tgt)
    pred = pred[valid]
    tgt = tgt[valid]
    n = pred.size
    if n < 100:
        return {"horizon": h, "n": int(n), "error": "too few samples"}

    # Convert to ticks
    pred_ticks = pred / LOG_RET_PER_TICK
    tgt_ticks = tgt / LOG_RET_PER_TICK

    # IC
    ic = spearman(pred, tgt)

    # Predicted-move distribution (absolute)
    abs_pred = np.abs(pred_ticks)
    dist = {
        "mean": float(abs_pred.mean()),
        "p50": pct(abs_pred, 50),
        "p75": pct(abs_pred, 75),
        "p90": pct(abs_pred, 90),
        "p99": pct(abs_pred, 99),
    }

    # Tradability at each threshold (market AND passive)
    threshold_rows = []
    for t in THRESHOLDS:
        # Market order leg
        sel_m = abs_pred >= t
        n_m = int(sel_m.sum())
        if n_m == 0:
            threshold_rows.append({"threshold_ticks": t, "n_market": 0,
                                    "n_passive": 0, "skipped": True})
            continue
        # P&L: sign(pred) * realized_ticks - cost
        signed_realized = np.sign(pred_ticks[sel_m]) * tgt_ticks[sel_m]
        net_pnl_market = signed_realized - MARKET_COST_TICKS
        # passive: only count signals where t ≥ 1.5 (else we'd just queue and never fill)
        # but show the metric anyway for transparency
        net_pnl_passive = signed_realized - PASSIVE_COST_TICKS

        threshold_rows.append({
            "threshold_ticks": t,
            "n_market": int(n_m),
            "frac_signals": float(n_m / n),
            "raw_hit_rate": float((signed_realized > 0).mean()),
            "raw_avg_realized_ticks": float(signed_realized.mean()),
            "net_avg_ticks_market": float(net_pnl_market.mean()),
            "net_total_ticks_market": float(net_pnl_market.sum()),
            "net_total_usd_market": float(net_pnl_market.sum() * ES_TICK_VALUE),
            "net_avg_ticks_passive": float(net_pnl_passive.mean()),
            "net_total_ticks_passive": float(net_pnl_passive.sum()),
            "net_total_usd_passive": float(net_pnl_passive.sum() * ES_TICK_VALUE),
            "sharpe_market": float(net_pnl_market.mean() / net_pnl_market.std()
                                    if net_pnl_market.std() > 0 else 0.0),
            "sharpe_passive": float(net_pnl_passive.mean() / net_pnl_passive.std()
                                     if net_pnl_passive.std() > 0 else 0.0),
        })

    # Verdict: best market-cost threshold by total USD (require ≥100 trades)
    best = None
    for row in threshold_rows:
        if row.get("skipped") or row["n_market"] < 100:
            continue
        if best is None or row["net_total_usd_market"] > best["net_total_usd_market"]:
            best = row

    return {
        "horizon": h,
        "n": int(n),
        "ic": ic,
        "pred_abs_ticks": dist,
        "thresholds": threshold_rows,
        "best_market_threshold": best,
    }


def print_summary(results):
    print("\n" + "=" * 100)
    print("HC #458 R1 — TRADABILITY MATRIX")
    print("=" * 100)
    print(f"{'horizon':<8} {'N':>8} {'IC':>7} {'meanT':>7} {'p90T':>7} "
          f"{'best_t':>8} {'n_trades':>9} {'net_usd':>11} {'sharpe':>8}  verdict")
    for r in results:
        if "error" in r:
            print(f"{r['horizon']:<8} ERROR: {r['error']}")
            continue
        best = r.get("best_market_threshold")
        h = r["horizon"]
        ic = r["ic"]
        dist = r["pred_abs_ticks"]
        if best is None:
            verdict = "UNTRADEABLE (no threshold yielded ≥100 trades)"
            print(f"{h:<8} {r['n']:>8d} {ic:>+7.3f} {dist['mean']:>7.2f} "
                  f"{dist['p90']:>7.2f} {'—':>8} {'—':>9} {'—':>11} {'—':>8}  {verdict}")
            continue
        usd = best["net_total_usd_market"]
        verdict = "TRADEABLE" if usd > 0 else "NEGATIVE-EV"
        if usd > 0 and best["sharpe_market"] > 0.05:
            verdict += " + POSITIVE-SHARPE"
        print(f"{h:<8} {r['n']:>8d} {ic:>+7.3f} {dist['mean']:>7.2f} "
              f"{dist['p90']:>7.2f} {best['threshold_ticks']:>8.1f} "
              f"{best['n_market']:>9d} ${usd:>9.0f} "
              f"{best['sharpe_market']:>+8.3f}  {verdict}")
    print("=" * 100)


def main():
    if len(sys.argv) < 2:
        print("Usage: hc458_tradability_matrix.py <oot_npz_path>")
        sys.exit(1)
    path = Path(sys.argv[1])
    if not path.exists():
        print(f"ERROR: {path} not found")
        sys.exit(1)
    d = np.load(path)
    results = [analyze_horizon(d, h) for h in HORIZONS]
    print_summary(results)
    out = path.with_suffix(".tradability.json")
    with open(out, "w") as fh:
        json.dump({"source": str(path), "results": results}, fh, indent=2,
                   default=lambda x: float(x) if isinstance(x, np.floating) else None)
    print(f"\nJSON: {out}")


if __name__ == "__main__":
    main()
