#!/usr/bin/env python3
"""
Confluence Stacking v1: CNN-Mamba v2 x PatchTST signal agreement analysis.

When BOTH models agree on direction + high confidence, do realized ticks improve
enough to clear execution costs?

Cost thresholds:
  - Hybrid (passive entry, market exit): 0.876 ticks
  - Market orders both sides: 1.376 ticks
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path("/home/jupiter/Lvl3Quant")
CM_DIR = ROOT / "output" / "cnn_mamba_v2_bulk_oot_v2"
PT_DIR = ROOT / "output" / "patchtst_bulk_oot"
OUT_DIR = ROOT / "output" / "confluence_stacking_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Cost constants (ES futures, AMP/Rithmic)
HYBRID_COST = 0.876   # passive + market
MARKET_COST = 1.376   # market both sides

# Horizons: index 0=1s, 1=5s, 2=10s
HORIZON_IDX = 2  # 10s — primary signal horizon
HORIZON_NAME = "10s"


def load_overlapping_dates():
    """Find dates with predictions from both models."""
    cm_files = {f.replace("_predictions.npz", ""): CM_DIR / f
                for f in os.listdir(CM_DIR) if f.endswith(".npz")}
    pt_files = {f.replace("_predictions.npz", ""): PT_DIR / f
                for f in os.listdir(PT_DIR) if f.endswith(".npz")}
    overlap = sorted(set(cm_files) & set(pt_files))
    return overlap, cm_files, pt_files


def load_day(date, cm_files, pt_files):
    """Load and align predictions for one day. Returns dict or None."""
    cm = np.load(cm_files[date], allow_pickle=True)
    pt = np.load(pt_files[date], allow_pickle=True)

    cm_pred = cm["predictions"][:, HORIZON_IDX]
    cm_label = cm["labels"][:, HORIZON_IDX]
    pt_pred = pt["predictions"][:, HORIZON_IDX]
    pt_label = pt["labels"][:, HORIZON_IDX]

    # Align by taking min length (stride/window should match)
    n = min(len(cm_pred), len(pt_pred))
    if n < 100:
        return None

    cm_pred = cm_pred[:n]
    pt_pred = pt_pred[:n]
    labels = cm_label[:n]  # labels should be identical
    pt_labels = pt_label[:n]

    # Replace NaN labels with 0 (no movement = no profit/loss)
    nan_count = np.isnan(labels).sum()
    if nan_count > 0:
        labels = np.nan_to_num(labels, nan=0.0)

    return {
        "date": date,
        "cm_pred": cm_pred,
        "pt_pred": pt_pred,
        "labels": labels,
        "n": n,
    }


def compute_percentile_thresholds(preds, direction="short"):
    """Compute percentile thresholds for short (low predictions) or long."""
    if direction == "short":
        # For shorts, lower prediction = higher confidence
        return {pct: np.percentile(preds, pct) for pct in [1, 2, 5, 10, 20]}
    else:
        return {pct: np.percentile(preds, 100 - pct) for pct in [1, 2, 5, 10, 20]}


def analyze_confluence(all_data):
    """Main analysis: sweep thresholds, compute realized ticks for confluence events."""
    # Stack all days
    all_cm = np.concatenate([d["cm_pred"] for d in all_data])
    all_pt = np.concatenate([d["pt_pred"] for d in all_data])
    all_labels = np.concatenate([d["labels"] for d in all_data])
    all_dates = np.concatenate([[d["date"]] * d["n"] for d in all_data])

    total_events = len(all_cm)
    print(f"\nTotal aligned events: {total_events:,}")
    print(f"Total days: {len(all_data)}")

    # Prediction correlation between models
    pred_corr = np.corrcoef(all_cm, all_pt)[0, 1]
    print(f"Prediction correlation (CM vs PT): {pred_corr:.4f}")

    # Labels are in ticks (realized movement at 10s horizon)
    # Negative label = price went down (short profitable)
    # For SHORT trades: realized_ticks = -labels (positive when price drops)

    results = []
    thresholds = [1, 2, 5, 10, 20]

    for pct in thresholds:
        cm_thresh = np.percentile(all_cm, pct)
        pt_thresh = np.percentile(all_pt, pct)

        # ── Single model baselines ──
        cm_short = all_cm <= cm_thresh
        pt_short = all_pt <= pt_thresh

        # ── Confluence: BOTH models in top N% short ──
        confluence = cm_short & pt_short

        # ── Realized ticks ──
        for label, mask, name in [
            ("CM_only", cm_short, f"CNN-Mamba top {pct}%"),
            ("PT_only", pt_short, f"PatchTST top {pct}%"),
            ("Confluence", confluence, f"Both top {pct}%"),
        ]:
            n_events = mask.sum()
            if n_events < 10:
                continue

            realized = -all_labels[mask]  # negative label = short profit
            avg_realized = realized.mean()
            median_realized = np.median(realized)
            std_realized = realized.std()

            # Net after costs
            net_hybrid = avg_realized - HYBRID_COST
            net_market = avg_realized - MARKET_COST

            # Win rate (realized > 0)
            wr = (realized > 0).mean()

            # Per-day breakdown
            unique_dates = np.unique(all_dates[mask])
            daily_avgs = []
            daily_profitable = 0
            for dt in unique_dates:
                day_mask = mask & (all_dates == dt)
                if day_mask.sum() > 0:
                    day_avg = -all_labels[day_mask].mean()
                    daily_avgs.append(day_avg)
                    if day_avg > HYBRID_COST:
                        daily_profitable += 1

            daily_sharpe = (np.mean(daily_avgs) / np.std(daily_avgs) * np.sqrt(252)
                           if len(daily_avgs) > 1 and np.std(daily_avgs) > 0 else 0)

            results.append({
                "filter": name,
                "label": label,
                "pct_threshold": pct,
                "n_events": n_events,
                "n_days": len(unique_dates),
                "avg_realized_ticks": avg_realized,
                "median_realized_ticks": median_realized,
                "std_realized_ticks": std_realized,
                "net_hybrid": net_hybrid,
                "net_market": net_market,
                "win_rate": wr,
                "daily_sharpe": daily_sharpe,
                "daily_profitable_frac": daily_profitable / len(unique_dates) if unique_dates.size else 0,
                "events_per_day": n_events / len(unique_dates) if unique_dates.size else 0,
            })

    # ── Confluence LIFT analysis ──
    print("\n" + "=" * 80)
    print("CONFLUENCE STACKING RESULTS (SHORT side, 10s horizon)")
    print("=" * 80)

    df = pd.DataFrame(results)

    for pct in thresholds:
        sub = df[df["pct_threshold"] == pct]
        if sub.empty:
            continue
        print(f"\n--- Top {pct}% threshold ---")
        for _, row in sub.iterrows():
            status = "PROFITABLE" if row["net_hybrid"] > 0 else "unprofitable"
            print(f"  {row['filter']:25s} | n={row['n_events']:6d} | "
                  f"avg={row['avg_realized_ticks']:+.3f} | "
                  f"net_hybrid={row['net_hybrid']:+.3f} | "
                  f"net_mkt={row['net_market']:+.3f} | "
                  f"WR={row['win_rate']:.1%} | "
                  f"Sharpe={row['daily_sharpe']:+.2f} | {status}")

        # Compute lift
        cm_row = sub[sub["label"] == "CM_only"]
        conf_row = sub[sub["label"] == "Confluence"]
        if not cm_row.empty and not conf_row.empty:
            lift = conf_row.iloc[0]["avg_realized_ticks"] - cm_row.iloc[0]["avg_realized_ticks"]
            print(f"  >> LIFT from confluence: {lift:+.3f} ticks")

    return df


def analyze_rolling_consistency(all_data):
    """Alternative: rolling prediction consistency within CNN-Mamba.
    If model keeps predicting short for N consecutive events, is conviction higher?
    Vectorized implementation to avoid memory blowup."""
    print("\n" + "=" * 80)
    print("ROLLING CONSISTENCY ANALYSIS (CNN-Mamba v2, 10s horizon)")
    print("=" * 80)

    windows = [3, 5, 7, 10]

    # Process per-window using vectorized rolling
    for w in windows:
        all_realized = []
        all_short_frac = []
        all_is_short = []
        all_magnitude = []

        for d in all_data:
            cm = d["cm_pred"]
            labels = d["labels"]
            n = len(cm)
            if n < w + 1:
                continue

            # Vectorized rolling short fraction using cumsum
            is_neg = (cm < 0).astype(np.float32)
            cumsum_neg = np.cumsum(is_neg)
            # short_frac[i] = fraction of cm[i-w:i] that are negative
            short_frac = np.empty(n - w)
            short_frac = (cumsum_neg[w:] - cumsum_neg[:n - w]) / w

            # Vectorized rolling magnitude using cumsum
            abs_cm = np.abs(cm).astype(np.float64)
            cumsum_abs = np.cumsum(abs_cm)
            avg_mag = (cumsum_abs[w:] - cumsum_abs[:n - w]) / w

            realized = -labels[w:]  # short P&L
            current_short = cm[w:] < 0

            all_realized.append(realized)
            all_short_frac.append(short_frac)
            all_is_short.append(current_short)
            all_magnitude.append(avg_mag)

        realized = np.concatenate(all_realized)
        short_frac = np.concatenate(all_short_frac)
        is_short = np.concatenate(all_is_short)
        magnitude = np.concatenate(all_magnitude)

        print(f"\n--- Window={w} ---")
        for name, mask in [
            ("All short events", is_short),
            (f">=50% agree (w={w})", (short_frac >= 0.5) & is_short),
            (f"100% agree (w={w})", (short_frac == 1.0) & is_short),
        ]:
            n_ev = mask.sum()
            if n_ev < 10:
                continue
            avg = realized[mask].mean()
            wr = (realized[mask] > 0).mean()

            # Top 10% magnitude
            high_avg_str = ""
            if n_ev > 100:
                mag_thresh = np.percentile(magnitude[mask], 90)
                high_mask = mask & (magnitude >= mag_thresh)
                high_n = high_mask.sum()
                if high_n > 0:
                    high_avg = realized[high_mask].mean()
                    high_avg_str = f" | top10%_mag: avg={high_avg:+.3f} n={high_n}"

            net_h = avg - HYBRID_COST
            status = "PROFITABLE" if net_h > 0 else "unprofitable"
            print(f"  {name:30s} | n={n_ev:6d} | avg={avg:+.3f} | "
                  f"net_hybrid={net_h:+.3f} | WR={wr:.1%} | {status}{high_avg_str}")


def analyze_combined_score(all_data):
    """Combined score: average of normalized CM and PT predictions.
    Test if combined score provides better filtering than either alone."""
    print("\n" + "=" * 80)
    print("COMBINED SCORE ANALYSIS (averaged z-scores)")
    print("=" * 80)

    all_cm = np.concatenate([d["cm_pred"] for d in all_data])
    all_pt = np.concatenate([d["pt_pred"] for d in all_data])
    all_labels = np.concatenate([d["labels"] for d in all_data])
    all_dates = np.concatenate([[d["date"]] * d["n"] for d in all_data])

    # Z-score normalize each model's predictions
    cm_z = (all_cm - all_cm.mean()) / all_cm.std()
    pt_z = (all_pt - all_pt.mean()) / all_pt.std()

    # Combined score: average z-score
    combined = (cm_z + pt_z) / 2.0

    # IC of combined vs individual
    ic_cm = np.corrcoef(all_cm, all_labels)[0, 1]
    ic_pt = np.corrcoef(all_pt, all_labels)[0, 1]
    ic_combined = np.corrcoef(combined, all_labels)[0, 1]
    print(f"\nIC (10s): CM={ic_cm:.4f}, PT={ic_pt:.4f}, Combined={ic_combined:.4f}")
    print(f"IC lift from combining: {ic_combined - ic_cm:+.4f} vs CM, {ic_combined - ic_pt:+.4f} vs PT")

    results = []
    for pct in [1, 2, 5, 10, 20]:
        # Combined score threshold (most negative = strongest short signal)
        thresh = np.percentile(combined, pct)
        mask = combined <= thresh

        realized = -all_labels[mask]
        n = mask.sum()
        if n < 10:
            continue

        avg = realized.mean()
        wr = (realized > 0).mean()
        net_h = avg - HYBRID_COST
        net_m = avg - MARKET_COST

        # Per-day
        unique_dates = np.unique(all_dates[mask])
        daily_avgs = []
        for dt in unique_dates:
            day_mask = mask & (all_dates == dt)
            if day_mask.sum() > 0:
                daily_avgs.append(-all_labels[day_mask].mean())
        daily_sharpe = (np.mean(daily_avgs) / np.std(daily_avgs) * np.sqrt(252)
                       if len(daily_avgs) > 1 and np.std(daily_avgs) > 0 else 0)

        status = "PROFITABLE" if net_h > 0 else "unprofitable"
        print(f"  Combined top {pct:2d}% | n={n:6d} | avg={avg:+.3f} | "
              f"net_hybrid={net_h:+.3f} | net_mkt={net_m:+.3f} | "
              f"WR={wr:.1%} | Sharpe={daily_sharpe:+.2f} | {status}")

        results.append({
            "filter": f"Combined top {pct}%",
            "pct_threshold": pct,
            "n_events": n,
            "avg_realized_ticks": avg,
            "net_hybrid": net_h,
            "net_market": net_m,
            "win_rate": wr,
            "daily_sharpe": daily_sharpe,
            "events_per_day": n / len(unique_dates) if unique_dates.size else 0,
        })

    return pd.DataFrame(results)


def write_report(confluence_df, combined_df, pred_corr, n_days, n_events):
    """Write summary report."""
    report = []
    report.append("# Confluence Stacking v1: CNN-Mamba v2 x PatchTST")
    report.append(f"\nAnalysis date: 2026-05-23")
    report.append(f"Overlapping OOT days: {n_days}")
    report.append(f"Total aligned events: {n_events:,}")
    report.append(f"Prediction correlation: {pred_corr:.4f}")
    report.append(f"Horizon: {HORIZON_NAME}")
    report.append(f"\nCost thresholds: hybrid={HYBRID_COST}, market={MARKET_COST}")

    report.append("\n## Confluence Results (SHORT side)")
    report.append("")

    # Best confluence result
    conf_rows = confluence_df[confluence_df["label"] == "Confluence"].copy()
    conf_rows = conf_rows.dropna(subset=["net_hybrid"])
    if not conf_rows.empty:
        best = conf_rows.loc[conf_rows["net_hybrid"].idxmax()]
        report.append(f"Best confluence filter: {best['filter']}")
        report.append(f"  Avg realized: {best['avg_realized_ticks']:+.3f} ticks")
        report.append(f"  Net hybrid: {best['net_hybrid']:+.3f} ticks")
        report.append(f"  Net market: {best['net_market']:+.3f} ticks")
        report.append(f"  Win rate: {best['win_rate']:.1%}")
        report.append(f"  Daily Sharpe: {best['daily_sharpe']:+.2f}")
        report.append(f"  Events: {best['n_events']:,} ({best['events_per_day']:.0f}/day)")

    report.append("\n## Full Results Table")
    report.append("")
    report.append(confluence_df.to_string(index=False))

    if not combined_df.empty:
        report.append("\n## Combined Z-Score Results")
        report.append("")
        report.append(combined_df.to_string(index=False))

    # Verdict
    report.append("\n## Verdict")
    profitable_rows = confluence_df[
        (confluence_df["label"] == "Confluence") & (confluence_df["net_hybrid"] > 0)
    ].dropna(subset=["net_hybrid"])
    if not profitable_rows.empty:
        report.append("\nConfluence stacking DOES push realized ticks above hybrid breakeven.")
        best = profitable_rows.loc[profitable_rows["net_hybrid"].idxmax()]
        report.append(f"Best config: {best['filter']} at net +{best['net_hybrid']:.3f} ticks/trade.")
    else:
        report.append("\nConfluence stacking does NOT clear hybrid breakeven on its own.")
        valid_conf = conf_rows.dropna(subset=["net_hybrid"])
        best = valid_conf.loc[valid_conf["net_hybrid"].idxmax()] if not valid_conf.empty else None
        if best is not None:
            gap = -best["net_hybrid"]
            report.append(f"Closest: {best['filter']} at net {best['net_hybrid']:+.3f} (gap: {gap:.3f} ticks).")

    return "\n".join(report)


def main():
    print("=" * 80)
    print("CONFLUENCE STACKING v1: CNN-Mamba v2 x PatchTST")
    print("=" * 80)

    # Load data
    overlap_dates, cm_files, pt_files = load_overlapping_dates()
    print(f"\nOverlapping dates: {len(overlap_dates)}")

    all_data = []
    for date in overlap_dates:
        d = load_day(date, cm_files, pt_files)
        if d is not None:
            all_data.append(d)

    print(f"Loaded {len(all_data)} days successfully")

    if not all_data:
        print("ERROR: No data loaded!")
        return

    total_events = sum(d["n"] for d in all_data)
    all_cm = np.concatenate([d["cm_pred"] for d in all_data])
    all_pt = np.concatenate([d["pt_pred"] for d in all_data])
    pred_corr = np.corrcoef(all_cm, all_pt)[0, 1]

    # ── Main confluence analysis ──
    confluence_df = analyze_confluence(all_data)

    # ── Combined z-score analysis ──
    combined_df = analyze_combined_score(all_data)

    # ── Rolling consistency (CNN-Mamba self-agreement) ──
    analyze_rolling_consistency(all_data)

    # ── Save results ──
    confluence_df.to_csv(OUT_DIR / "confluence_results.csv", index=False)
    if not combined_df.empty:
        combined_df.to_csv(OUT_DIR / "combined_score_results.csv", index=False)

    report = write_report(confluence_df, combined_df, pred_corr, len(all_data), total_events)
    (OUT_DIR / "REPORT.md").write_text(report)

    print(f"\n\nResults saved to {OUT_DIR}")
    print(f"  confluence_results.csv")
    print(f"  REPORT.md")


if __name__ == "__main__":
    main()
