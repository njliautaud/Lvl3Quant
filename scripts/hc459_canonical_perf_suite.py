#!/usr/bin/env python3
"""HC #459 R2 — Canonical Live-Performance Suite.

IC alone is REJECTED. Every model report must surface the full picture:
  - IC concat + IC by confidence quantile (top 1/5/10/25%)
  - Sign-agreement rate by confidence quantile
  - Magnitude calibration slope (E[|realized| | |pred|] regression)
  - FIFO net P&L using ENTRY-EXIT semantics: 1s gates entry, horizon-h exits
  - Sharpe / Sortino / Profit Factor / Win Rate from the simulated trades
  - Per-day breakdown so we can see day-conc + regime sensitivity

Units: predictions & targets are in TICKS (despite "log_ret" field names).
Cost: $4.70 round-trip commission = 0.376 ticks. Passive limit assumes
queue fill; market order adds 1 tick spread crossing = 1.376 ticks total.

Usage:
    python3 scripts/hc459_canonical_perf_suite.py <oot_npz_path> [--entry-h 1s]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ES_TICK_VALUE = 12.50
RT_COMMISSION_TICKS = 0.376
MARKET_COST_TICKS = 1.376
PASSIVE_COST_TICKS = 0.376
HORIZONS = ["1s", "5s", "10s", "30s", "60s", "5min"]
CONF_QUANTILES = [0.99, 0.95, 0.90, 0.75]  # top 1%, 5%, 10%, 25%


def spearman(a, b):
    if a.size < 10:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    d = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def sign_agreement(pred, tgt):
    if pred.size == 0:
        return float("nan")
    return float((np.sign(pred) == np.sign(tgt)).mean())


def magnitude_calibration(pred, tgt):
    """Slope of |tgt| ~ |pred|. Slope = 1 means well-calibrated; <1 = compressed pred."""
    ap = np.abs(pred)
    at = np.abs(tgt)
    if ap.size < 30 or ap.std() < 1e-6:
        return float("nan")
    # OLS: slope = cov / var
    apc = ap - ap.mean()
    atc = at - at.mean()
    return float((apc * atc).sum() / (apc * apc).sum())


def confidence_buckets(pred, tgt):
    """Return per-quantile sign-agreement + realized-mean + IC."""
    abs_pred = np.abs(pred)
    out = {}
    for q in CONF_QUANTILES:
        thr = np.quantile(abs_pred, q)
        sel = abs_pred >= thr
        n = int(sel.sum())
        if n < 50:
            out[f"top_{int((1-q)*100)}pct"] = {"n": n, "skipped": True}
            continue
        p = pred[sel]
        t = tgt[sel]
        signed_realized = np.sign(p) * t
        out[f"top_{int((1-q)*100)}pct"] = {
            "n": n,
            "ic": spearman(p, t),
            "sign_agree": sign_agreement(p, t),
            "mean_pred_ticks": float(p.mean()),
            "mean_abs_pred_ticks": float(np.abs(p).mean()),
            "mean_signed_realized_ticks": float(signed_realized.mean()),
            "p50_signed_realized_ticks": float(np.median(signed_realized)),
        }
    return out


def fifo_pnl_entry_exit(d, entry_h, exit_h):
    """1s (or entry_h) gates entry direction + magnitude.
    Realized P&L = sign(pred_entry) * realized_exit - cost.
    Use passive limit if abs(pred_entry) ≥ 1 tick, else market.
    """
    pe = d.get(f"pred_log_ret_{entry_h}")
    te = d.get(f"target_log_ret_{entry_h}")
    pxt = d.get(f"target_log_ret_{exit_h}")
    me = d.get(f"mask_log_ret_{entry_h}").astype(bool)
    mx = d.get(f"mask_log_ret_{exit_h}").astype(bool)
    valid = me & mx & np.isfinite(pe) & np.isfinite(pxt)
    pe = pe[valid]
    pxt = pxt[valid]
    abs_pred = np.abs(pe)

    out_buckets = {}
    for q in CONF_QUANTILES:
        thr = np.quantile(abs_pred, q)
        sel = abs_pred >= thr
        n = int(sel.sum())
        if n < 50:
            out_buckets[f"top_{int((1-q)*100)}pct"] = {"n": n, "skipped": True}
            continue
        # Direction from entry prediction; realized from EXIT horizon
        signed_realized = np.sign(pe[sel]) * pxt[sel]
        # Cost: assume passive if predicted move ≥ 1 tick, else market
        use_passive = abs_pred[sel] >= 1.0
        cost = np.where(use_passive, PASSIVE_COST_TICKS, MARKET_COST_TICKS)
        net = signed_realized - cost
        wins = net > 0
        downside = net[net < 0]
        usd = net * ES_TICK_VALUE
        sharpe = float(net.mean() / net.std()) if net.std() > 0 else 0.0
        sortino = float(net.mean() / downside.std()) if downside.size and downside.std() > 0 else 0.0
        pf = (float(net[net > 0].sum() / -net[net < 0].sum())
              if (net < 0).any() and -net[net < 0].sum() > 0 else float("inf"))
        out_buckets[f"top_{int((1-q)*100)}pct"] = {
            "n": n,
            "frac_passive": float(use_passive.mean()),
            "win_rate": float(wins.mean()),
            "net_avg_ticks": float(net.mean()),
            "net_total_usd": float(usd.sum()),
            "sharpe": sharpe,
            "sortino": sortino,
            "profit_factor": pf,
        }
    return out_buckets


def per_horizon_block(d, h):
    pe = d.get(f"pred_log_ret_{h}")
    te = d.get(f"target_log_ret_{h}")
    me = d.get(f"mask_log_ret_{h}")
    if pe is None or te is None:
        return {"horizon": h, "error": "missing key"}
    mask = me.astype(bool) if me is not None else np.ones_like(pe, dtype=bool)
    valid = mask & np.isfinite(pe) & np.isfinite(te)
    p = pe[valid]
    t = te[valid]
    if p.size < 100:
        return {"horizon": h, "n": int(p.size), "error": "too few samples"}
    block = {
        "horizon": h,
        "n": int(p.size),
        "ic": spearman(p, t),
        "sign_agreement_all": sign_agreement(p, t),
        "magnitude_calibration_slope": magnitude_calibration(p, t),
        "mean_abs_pred": float(np.abs(p).mean()),
        "mean_abs_target": float(np.abs(t).mean()),
        "p90_abs_pred": float(np.percentile(np.abs(p), 90)),
        "p90_abs_target": float(np.percentile(np.abs(t), 90)),
        "by_confidence": confidence_buckets(p, t),
    }
    return block


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="OOT NPZ path")
    ap.add_argument("--entry-h", default="1s", help="entry-trigger horizon (default 1s)")
    ap.add_argument("--exit-horizons", default="5s,10s,30s",
                    help="exit horizons to simulate (comma-list)")
    args = ap.parse_args()

    path = Path(args.path)
    d = np.load(path)
    exit_hs = [h.strip() for h in args.exit_horizons.split(",")]

    out = {"source": str(path), "entry_h": args.entry_h}

    # Per-horizon IC + confidence blocks
    out["per_horizon"] = [per_horizon_block(d, h) for h in HORIZONS]

    # Entry-exit FIFO sims: 1s entry × {5s, 10s, 30s} exit
    out["entry_exit_sims"] = {}
    for xh in exit_hs:
        try:
            out["entry_exit_sims"][f"{args.entry_h}_entry_{xh}_exit"] = \
                fifo_pnl_entry_exit(d, args.entry_h, xh)
        except Exception as e:
            out["entry_exit_sims"][f"{args.entry_h}_entry_{xh}_exit"] = {"error": str(e)}

    # ---- Print ----
    print("\n" + "=" * 110)
    print(f"HC #459 R2 — CANONICAL LIVE-PERFORMANCE SUITE  |  source: {path.name}")
    print("=" * 110)

    # Per-horizon summary
    print(f"\n{'horizon':<7} {'N':>8} {'IC':>7} {'signAgr':>8} {'magCal':>7} "
          f"{'meanAbsP':>9} {'meanAbsT':>9} {'p90AbsP':>8} {'p90AbsT':>8}")
    for b in out["per_horizon"]:
        if "error" in b:
            print(f"{b['horizon']:<7} ERR  {b['error']}")
            continue
        print(f"{b['horizon']:<7} {b['n']:>8d} {b['ic']:>+7.3f} {b['sign_agreement_all']:>8.3f} "
              f"{b['magnitude_calibration_slope']:>+7.3f} {b['mean_abs_pred']:>9.3f} "
              f"{b['mean_abs_target']:>9.2f} {b['p90_abs_pred']:>8.3f} {b['p90_abs_target']:>8.2f}")

    # Confidence-quantile sign-agreement focus (1s)
    print(f"\n1s — by confidence quantile (the entry-trigger view):")
    b1s = next((b for b in out["per_horizon"] if b["horizon"] == "1s"), None)
    if b1s and "by_confidence" in b1s:
        print(f"  {'bucket':<10} {'N':>7} {'IC':>7} {'signAgr':>8} {'meanRealiz':>10}")
        for k, v in b1s["by_confidence"].items():
            if v.get("skipped"):
                print(f"  {k:<10} {v['n']:>7d}  (skipped)")
                continue
            print(f"  {k:<10} {v['n']:>7d} {v['ic']:>+7.3f} {v['sign_agree']:>8.3f} "
                  f"{v['mean_signed_realized_ticks']:>+10.3f}")

    # Entry-exit FIFO P&L
    for k, sim in out["entry_exit_sims"].items():
        print(f"\nFIFO sim: {k}")
        if "error" in sim:
            print(f"  ERROR: {sim['error']}")
            continue
        print(f"  {'bucket':<10} {'N':>7} {'WR':>6} {'netT':>8} {'netUSD':>11} "
              f"{'Sharpe':>7} {'Sortino':>8} {'PF':>6}")
        for kb, v in sim.items():
            if v.get("skipped"):
                print(f"  {kb:<10} {v['n']:>7d}  (skipped)")
                continue
            print(f"  {kb:<10} {v['n']:>7d} {v['win_rate']:>6.3f} "
                  f"{v['net_avg_ticks']:>+8.3f} ${v['net_total_usd']:>+9.0f} "
                  f"{v['sharpe']:>+7.3f} {v['sortino']:>+8.3f} {v['profit_factor']:>6.2f}")

    # Final verdict
    print("\n" + "-" * 110)
    best_sim = None
    best_key = None
    for k, sim in out["entry_exit_sims"].items():
        for kb, v in sim.items():
            if v.get("skipped") or "net_total_usd" not in v:
                continue
            if best_sim is None or v["net_total_usd"] > best_sim["net_total_usd"]:
                best_sim = v
                best_key = f"{k} @ {kb}"
    if best_sim and best_sim["net_total_usd"] > 0:
        print(f"VERDICT: BEST TRADABLE: {best_key} | "
              f"N={best_sim['n']} | WR={best_sim['win_rate']:.3f} | "
              f"net=${best_sim['net_total_usd']:.0f} | Sharpe={best_sim['sharpe']:+.3f}")
    else:
        print("VERDICT: NO TRADABLE COMBINATION at any confidence quantile.")
        if best_sim:
            print(f"  Best (still negative): {best_key} | "
                  f"net=${best_sim['net_total_usd']:.0f}")
    print("=" * 110)

    out_path = path.with_suffix(".canonical_perf.json")

    def _conv(x):
        if isinstance(x, (np.floating, np.integer)):
            return float(x)
        if x == float("inf") or x == float("-inf"):
            return None
        return None

    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2, default=_conv)
    print(f"\nJSON: {out_path}")


if __name__ == "__main__":
    main()
