#!/usr/bin/env python3
"""
Regime-stratified analysis of confluence-gated trades (HC #428 R1).

Takes top-N% CNN-Mamba x PatchTST agreement trades (short side, 10s horizon)
and breaks down performance by green/red/flat market regime.

PASS criteria:
  |Sharpe_green - Sharpe_red| / max(|Sharpe_green|, |Sharpe_red|) <= 0.50
  day_concentration <= 0.70
"""
import os, json, sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

ROOT = Path("/home/jupiter/Lvl3Quant")
CM_DIR = ROOT / "output" / "cnn_mamba_v2_bulk_oot_v2"
PT_DIR = ROOT / "output" / "patchtst_bulk_oot"
OUT_DIR = ROOT / "output" / "confluence_regime_stratify_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Cost constants
HYBRID_COST = 0.876   # passive entry + market exit
COMMISSION_ONLY = 0.376  # passive both sides

# Regime thresholds (ticks, ES close-to-close)
REGIME_THRESHOLD = 8  # ticks

HORIZON_IDX = 2  # 10s
TOP_PCT = 2  # top 2% both models

def load_regime_data():
    """Load ES daily close-to-close for regime classification."""
    regime_file = ROOT / "data" / "processed" / "es_daily_regimes.csv"
    if regime_file.exists():
        return pd.read_csv(regime_file, dtype={"date": str})

    # Build from MBO data: use first/last mid prices
    regimes = {}
    mbo_dir = ROOT / "data" / "processed" / "mbo_events_smart_v3"
    if not mbo_dir.exists():
        # Try raw MBO
        mbo_dir = ROOT / "data" / "processed" / "mbo_events"

    for f in sorted(mbo_dir.glob("*.npz")):
        date = f.stem.split("_")[0]
        try:
            d = np.load(f, allow_pickle=True)
            if "mid_prices" in d:
                mids = d["mid_prices"]
            elif "features" in d:
                # mid price usually first feature or compute from bid/ask
                feats = d["features"]
                if feats.ndim == 2 and feats.shape[1] >= 2:
                    mids = (feats[:, 0] + feats[:, 1]) / 2
                else:
                    continue
            else:
                continue

            if len(mids) > 100:
                open_px = float(mids[50])  # skip first few events
                close_px = float(mids[-50])  # skip last few
                if open_px > 0:
                    change_ticks = (close_px - open_px) / 0.25
                    regimes[date] = change_ticks
        except Exception:
            continue

    if not regimes:
        return None

    df = pd.DataFrame([{"date": k, "change_ticks": v} for k, v in regimes.items()])
    df["regime"] = "flat"
    df.loc[df["change_ticks"] >= REGIME_THRESHOLD, "regime"] = "green"
    df.loc[df["change_ticks"] <= -REGIME_THRESHOLD, "regime"] = "red"
    df.to_csv(regime_file, index=False)
    return df


def main():
    print("=" * 80)
    print("CONFLUENCE REGIME STRATIFICATION (HC #428 R1)")
    print(f"Filter: top {TOP_PCT}% CNN-Mamba x PatchTST agreement, SHORT side, 10s horizon")
    print("=" * 80)

    # Load regime data
    regime_df = load_regime_data()
    if regime_df is None:
        print("ERROR: Cannot build regime data. Falling back to equal treatment.")
        regime_lookup = {}
    else:
        regime_lookup = dict(zip(regime_df["date"], regime_df["regime"]))
        print(f"\nRegime data: {len(regime_lookup)} days")
        for r in ["green", "red", "flat"]:
            n = sum(1 for v in regime_lookup.values() if v == r)
            print(f"  {r}: {n} days")

    # Load predictions
    cm_files = {f.replace("_predictions.npz", ""): CM_DIR / f
                for f in os.listdir(CM_DIR) if f.endswith(".npz")}
    pt_files = {f.replace("_predictions.npz", ""): PT_DIR / f
                for f in os.listdir(PT_DIR) if f.endswith(".npz")}
    overlap = sorted(set(cm_files) & set(pt_files))
    print(f"\nOverlapping prediction dates: {len(overlap)}")

    # Process each day
    daily_results = []

    for date in overlap:
        try:
            cm = np.load(cm_files[date], allow_pickle=True)
            pt = np.load(pt_files[date], allow_pickle=True)

            cm_pred = cm["predictions"][:, HORIZON_IDX]
            pt_pred = pt["predictions"][:, HORIZON_IDX]

            # Get labels
            if "labels" in cm:
                labels = cm["labels"][:, HORIZON_IDX] if cm["labels"].ndim > 1 else cm["labels"]
            elif "targets" in cm:
                labels = cm["targets"][:, HORIZON_IDX] if cm["targets"].ndim > 1 else cm["targets"]
            else:
                continue

            n = min(len(cm_pred), len(pt_pred), len(labels))
            cm_pred, pt_pred, labels = cm_pred[:n], pt_pred[:n], labels[:n]

            # Short signals: negative predictions
            cm_short = cm_pred < 0
            pt_short = pt_pred < 0

            # Top N% by magnitude (most confident shorts)
            cm_thr = np.percentile(np.abs(cm_pred[cm_short]), 100 - TOP_PCT) if cm_short.sum() > 0 else 999
            pt_thr = np.percentile(np.abs(pt_pred[pt_short]), 100 - TOP_PCT) if pt_short.sum() > 0 else 999

            cm_top = cm_short & (np.abs(cm_pred) >= cm_thr)
            pt_top = pt_short & (np.abs(pt_pred) >= pt_thr)
            both_top = cm_top & pt_top

            if both_top.sum() == 0:
                continue

            # For shorts: profit = -label (we sold, price went down = profit)
            realized = -labels[both_top]

            regime = regime_lookup.get(date, "unknown")

            daily_results.append({
                "date": date,
                "regime": regime,
                "n_trades": int(both_top.sum()),
                "gross_ticks": float(realized.mean()),
                "net_hybrid": float(realized.mean() - HYBRID_COST),
                "net_passive": float(realized.mean() - COMMISSION_ONLY),
                "total_ticks": float(realized.sum()),
                "wr": float((realized > 0).mean()),
                "std": float(realized.std()),
            })
        except Exception as e:
            print(f"  Skip {date}: {e}")
            continue

    if not daily_results:
        print("ERROR: No daily results produced.")
        return

    df = pd.DataFrame(daily_results)
    print(f"\nDays with confluence trades: {len(df)}")
    print(f"Total trades: {df['n_trades'].sum()}")
    print(f"Trades/day: {df['n_trades'].mean():.1f}")

    # Per-regime analysis
    print("\n" + "=" * 80)
    print("PER-REGIME PERFORMANCE")
    print("=" * 80)

    results = {}
    for regime in ["green", "red", "flat", "ALL"]:
        if regime == "ALL":
            sub = df
        else:
            sub = df[df["regime"] == regime]

        if len(sub) == 0:
            print(f"\n--- {regime.upper()} --- NO DATA")
            continue

        daily_pnl = sub["net_hybrid"]
        sharpe = daily_pnl.mean() / daily_pnl.std() * np.sqrt(252) if daily_pnl.std() > 0 else 0
        sortino_denom = daily_pnl[daily_pnl < 0].std()
        sortino = daily_pnl.mean() / sortino_denom * np.sqrt(252) if sortino_denom > 0 else 0

        total_gross = sub["gross_ticks"].sum()
        total_net = sub["net_hybrid"].sum()

        # Win rate across trades
        profitable_days = (sub["net_hybrid"] > 0).sum()

        r = {
            "n_days": len(sub),
            "n_trades": int(sub["n_trades"].sum()),
            "sharpe": float(sharpe),
            "sortino": float(sortino),
            "mean_net_hybrid": float(sub["net_hybrid"].mean()),
            "total_net_hybrid": float(total_net),
            "daily_wr": float(profitable_days / len(sub)),
            "mean_wr": float(sub["wr"].mean()),
            "trades_per_day": float(sub["n_trades"].mean()),
        }
        results[regime] = r

        print(f"\n--- {regime.upper()} ({r['n_days']} days, {r['n_trades']} trades) ---")
        print(f"  Sharpe: {r['sharpe']:+.2f}")
        print(f"  Sortino: {r['sortino']:+.2f}")
        print(f"  Net hybrid/trade: {r['mean_net_hybrid']:+.3f} ticks")
        print(f"  Daily profitable: {r['daily_wr']*100:.1f}%")
        print(f"  Avg WR: {r['mean_wr']*100:.1f}%")
        print(f"  Trades/day: {r['trades_per_day']:.1f}")

    # HC #428 R1 gates
    print("\n" + "=" * 80)
    print("HC #428 R1 GATE CHECKS")
    print("=" * 80)

    if "green" in results and "red" in results:
        sg = results["green"]["sharpe"]
        sr = results["red"]["sharpe"]
        delta = abs(sg - sr) / max(abs(sg), abs(sr)) if max(abs(sg), abs(sr)) > 0 else 0
        passes_regime = delta <= 0.50
        print(f"\n  Sharpe_green: {sg:+.2f}")
        print(f"  Sharpe_red:   {sr:+.2f}")
        print(f"  |delta|/max:  {delta:.3f} (threshold: 0.50)")
        print(f"  REGIME GATE:  {'PASS ✓' if passes_regime else 'FAIL ✗'}")
    else:
        print("\n  Cannot compute regime gate — missing green or red data")
        passes_regime = None

    if "ALL" in results:
        total = results["ALL"]["n_trades"]
        max_day = df["n_trades"].max()
        day_conc = max_day / total if total > 0 else 0
        passes_conc = day_conc <= 0.70
        print(f"\n  Day concentration: {day_conc:.3f} (threshold: 0.70)")
        print(f"  CONCENTRATION GATE: {'PASS ✓' if passes_conc else 'FAIL ✗'}")

    # Per-day detail
    print("\n" + "=" * 80)
    print("PER-DAY DETAIL")
    print("=" * 80)
    for _, row in df.iterrows():
        flag = "✓" if row["net_hybrid"] > 0 else "✗"
        print(f"  {row['date']} [{row['regime']:5s}] n={row['n_trades']:4d} "
              f"gross={row['gross_ticks']:+.3f} net_hybrid={row['net_hybrid']:+.3f} "
              f"WR={row['wr']*100:.0f}% {flag}")

    # Save
    output = {
        "config": {
            "top_pct": TOP_PCT,
            "side": "short",
            "horizon": "10s",
            "hybrid_cost": HYBRID_COST,
            "regime_threshold_ticks": REGIME_THRESHOLD,
        },
        "per_regime": results,
        "passes_regime_gate": passes_regime,
        "per_day": daily_results,
    }

    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    df.to_csv(OUT_DIR / "daily_detail.csv", index=False)
    print(f"\nSaved to {OUT_DIR}")


if __name__ == "__main__":
    main()
