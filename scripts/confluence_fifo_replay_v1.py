#!/usr/bin/env python3
"""
Confluence FIFO Replay v1: Validate confluence stacking via canonical FIFO labels.

The confluence stacking v1 found that CNN-Mamba v2 + PatchTST top 2% short
yields +1.19 avg realized ticks (proxy log_ret labels). Per HC #74, no setup
may be called profitable without canonical FIFO market replay validation.

This script aligns confluence-filtered events to pre-computed FIFO label files
(mbo_events_smart_v3_fifo_labels/) which contain per-window FIFO outcomes
from actual MBO order book replay.

FIFO configs tested: tp4sl3 (4-tick TP, 3-tick SL) and tp8sl5 (8-tick TP, 5-tick SL).
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path("/home/jupiter/Lvl3Quant")
CM_DIR = ROOT / "output" / "cnn_mamba_v2_bulk_oot_v2"
PT_DIR = ROOT / "output" / "patchtst_bulk_oot"
FIFO_DIR = ROOT / "data" / "processed" / "mbo_events_smart_v3_fifo_labels"
OUT_DIR = ROOT / "output" / "confluence_fifo_replay_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Cost constants
HYBRID_COST = 0.876   # passive entry + market exit
MARKET_COST = 1.376   # market both sides
COMMISSION_TICKS = 0.376  # RT commission only (already in FIFO net_ticks)

# Horizon index: 0=1s, 1=5s, 2=10s
HORIZON_IDX = 2

# Percentile thresholds to test
THRESHOLDS = [1, 2, 5, 10, 20]

# FIFO bracket configs available in the label files
FIFO_CONFIGS = ["tp4sl3", "tp8sl5"]


def find_triple_overlap():
    """Find dates with predictions from both models AND FIFO labels."""
    cm_dates = {f.replace("_predictions.npz", "") for f in os.listdir(CM_DIR) if f.endswith(".npz")}
    pt_dates = {f.replace("_predictions.npz", "") for f in os.listdir(PT_DIR) if f.endswith(".npz")}
    fifo_dates = {f.replace("_fifo_labels.npz", "") for f in os.listdir(FIFO_DIR) if f.endswith(".npz")}
    overlap = sorted(cm_dates & pt_dates & fifo_dates)
    return overlap


def load_day(date):
    """Load predictions + FIFO labels for one day. Returns dict or None."""
    cm = np.load(CM_DIR / f"{date}_predictions.npz", allow_pickle=True)
    pt = np.load(PT_DIR / f"{date}_predictions.npz", allow_pickle=True)
    fifo = np.load(FIFO_DIR / f"{date}_fifo_labels.npz", allow_pickle=True)

    cm_pred = cm["predictions"][:, HORIZON_IDX]
    pt_pred = pt["predictions"][:, HORIZON_IDX]

    # FIFO labels are indexed by window_k
    fifo_k = fifo["window_k"]

    # Align: take min length across all three sources
    n = min(len(cm_pred), len(pt_pred), int(fifo_k.max()) + 1 if len(fifo_k) > 0 else 0)
    if n < 100:
        return None

    cm_pred = cm_pred[:n]
    pt_pred = pt_pred[:n]

    # Build FIFO lookup arrays aligned to prediction indices
    fifo_data = {}
    for cfg in FIFO_CONFIGS:
        for side in ["short", "long"]:
            prefix = f"{cfg}_{side}"
            filled = np.zeros(n, dtype=bool)
            net_ticks = np.zeros(n, dtype=np.float32)
            gross_ticks = np.zeros(n, dtype=np.float32)
            hit_tp = np.zeros(n, dtype=bool)
            exit_reason = np.full(n, "no_fifo", dtype="U16")

            # Map FIFO data by window_k
            valid_mask = fifo_k < n
            valid_k = fifo_k[valid_mask]
            filled[valid_k] = fifo[f"{prefix}_filled"][valid_mask]
            net_ticks[valid_k] = fifo[f"{prefix}_net_ticks"][valid_mask]
            gross_ticks[valid_k] = fifo[f"{prefix}_gross_ticks"][valid_mask]
            hit_tp[valid_k] = fifo[f"{prefix}_hit_tp"][valid_mask]
            exit_reason[valid_k] = fifo[f"{prefix}_exit_reason"][valid_mask]

            fifo_data[f"{prefix}_filled"] = filled
            fifo_data[f"{prefix}_net_ticks"] = net_ticks
            fifo_data[f"{prefix}_gross_ticks"] = gross_ticks
            fifo_data[f"{prefix}_hit_tp"] = hit_tp
            fifo_data[f"{prefix}_exit_reason"] = exit_reason

    return {
        "date": date,
        "cm_pred": cm_pred,
        "pt_pred": pt_pred,
        "fifo": fifo_data,
        "n": n,
    }


def analyze_fifo(all_data):
    """Run FIFO analysis on confluence-filtered events."""
    # Stack all days
    all_cm = np.concatenate([d["cm_pred"] for d in all_data])
    all_pt = np.concatenate([d["pt_pred"] for d in all_data])
    all_dates = np.concatenate([[d["date"]] * d["n"] for d in all_data])

    # Stack FIFO data
    fifo_stacked = {}
    for key in all_data[0]["fifo"]:
        fifo_stacked[key] = np.concatenate([d["fifo"][key] for d in all_data])

    total = len(all_cm)
    print(f"Total aligned events: {total:,}")
    print(f"Total days: {len(all_data)}")

    results = []

    for cfg in FIFO_CONFIGS:
        print(f"\n{'='*80}")
        print(f"FIFO CONFIG: {cfg.upper()} (SHORT side)")
        print(f"{'='*80}")

        filled_all = fifo_stacked[f"{cfg}_short_filled"]
        net_all = fifo_stacked[f"{cfg}_short_net_ticks"]
        gross_all = fifo_stacked[f"{cfg}_short_gross_ticks"]
        hit_tp_all = fifo_stacked[f"{cfg}_short_hit_tp"]
        exit_reason_all = fifo_stacked[f"{cfg}_short_exit_reason"]

        # Baseline: all events
        baseline_filled = filled_all.sum()
        baseline_fill_rate = baseline_filled / total
        baseline_net = net_all[filled_all].mean() if baseline_filled > 0 else 0
        print(f"\nBaseline (all events): fill_rate={baseline_fill_rate:.1%}, "
              f"net_ticks={baseline_net:+.3f}, filled={baseline_filled:,}/{total:,}")

        for pct in THRESHOLDS:
            cm_thresh = np.percentile(all_cm, pct)
            pt_thresh = np.percentile(all_pt, pct)

            # Single model masks
            cm_short = all_cm <= cm_thresh
            pt_short = all_pt <= pt_thresh
            # Confluence mask
            confluence = cm_short & pt_short

            print(f"\n--- Top {pct}% threshold ---")

            for label, mask, name in [
                ("CM_only", cm_short, f"CNN-Mamba top {pct}%"),
                ("PT_only", pt_short, f"PatchTST top {pct}%"),
                ("Confluence", confluence, f"Both top {pct}%"),
            ]:
                n_events = mask.sum()
                if n_events < 5:
                    continue

                # FIFO outcomes for filtered events
                filtered_filled = filled_all[mask]
                filtered_net = net_all[mask]
                filtered_gross = gross_all[mask]
                filtered_hit_tp = hit_tp_all[mask]
                filtered_exit = exit_reason_all[mask]

                n_filled = filtered_filled.sum()
                fill_rate = n_filled / n_events if n_events > 0 else 0

                if n_filled > 0:
                    avg_net = filtered_net[filtered_filled].mean()
                    avg_gross = filtered_gross[filtered_filled].mean()
                    hit_tp_rate = filtered_hit_tp[filtered_filled].mean()

                    # Exit reason distribution
                    exit_counts = {}
                    for r in filtered_exit[filtered_filled]:
                        exit_counts[r] = exit_counts.get(r, 0) + 1
                else:
                    avg_net = 0
                    avg_gross = 0
                    hit_tp_rate = 0
                    exit_counts = {}

                # Per-day breakdown
                unique_dates = np.unique(all_dates[mask])
                daily_nets = []
                daily_fills = []
                daily_profitable = 0
                for dt in unique_dates:
                    day_mask = mask & (all_dates == dt)
                    day_filled = filled_all[day_mask]
                    day_net = net_all[day_mask]
                    n_day_filled = day_filled.sum()
                    daily_fills.append(n_day_filled)
                    if n_day_filled > 0:
                        day_avg_net = day_net[day_filled].mean()
                        daily_nets.append(day_avg_net)
                        if day_avg_net > 0:
                            daily_profitable += 1
                    else:
                        daily_nets.append(0.0)

                daily_nets_arr = np.array(daily_nets)
                if len(daily_nets_arr) > 1 and daily_nets_arr.std() > 0:
                    daily_sharpe = daily_nets_arr.mean() / daily_nets_arr.std() * np.sqrt(252)
                    # Sortino
                    downside = daily_nets_arr[daily_nets_arr < 0]
                    if len(downside) > 0:
                        downside_std = np.sqrt(np.mean(downside**2))
                        daily_sortino = daily_nets_arr.mean() / downside_std * np.sqrt(252) if downside_std > 0 else 0
                    else:
                        daily_sortino = 99.0  # no losing days
                else:
                    daily_sharpe = 0
                    daily_sortino = 0

                profitable_frac = daily_profitable / len(unique_dates) if len(unique_dates) > 0 else 0

                status = "PROFITABLE" if avg_net > 0 else "unprofitable"
                print(f"  {name:25s} | n={n_events:6d} | filled={n_filled:5d} ({fill_rate:.1%}) | "
                      f"net={avg_net:+.3f} | gross={avg_gross:+.3f} | "
                      f"TP_hit={hit_tp_rate:.1%} | Sharpe={daily_sharpe:+.2f} | {status}")

                results.append({
                    "fifo_config": cfg,
                    "filter": name,
                    "label": label,
                    "pct_threshold": pct,
                    "n_events": n_events,
                    "n_filled": n_filled,
                    "fill_rate": fill_rate,
                    "avg_net_ticks": avg_net,
                    "avg_gross_ticks": avg_gross,
                    "tp_hit_rate": hit_tp_rate,
                    "n_days": len(unique_dates),
                    "daily_sharpe": daily_sharpe,
                    "daily_sortino": daily_sortino,
                    "profitable_day_frac": profitable_frac,
                    "avg_fills_per_day": np.mean(daily_fills) if daily_fills else 0,
                    "exit_reasons": json.dumps(exit_counts),
                })

    return pd.DataFrame(results)


def write_report(df, n_days, n_events):
    """Write summary report."""
    lines = []
    lines.append("# Confluence FIFO Replay v1: CNN-Mamba v2 x PatchTST")
    lines.append(f"\nDays analyzed: {n_days}")
    lines.append(f"Total aligned events: {n_events:,}")
    lines.append(f"FIFO configs: {', '.join(FIFO_CONFIGS)}")
    lines.append(f"\nProxy baseline (from confluence_stacking_v1):")
    lines.append(f"  Both top 2% short: +1.19 avg realized ticks (log_ret proxy)")
    lines.append(f"  Hybrid breakeven: {HYBRID_COST} ticks")

    for cfg in FIFO_CONFIGS:
        lines.append(f"\n## {cfg.upper()} Results")
        sub = df[df["fifo_config"] == cfg]
        conf_rows = sub[sub["label"] == "Confluence"]

        if conf_rows.empty:
            lines.append("No confluence results.")
            continue

        for _, row in conf_rows.iterrows():
            status = "PROFITABLE" if row["avg_net_ticks"] > 0 else "UNPROFITABLE"
            lines.append(f"\n### {row['filter']} [{status}]")
            lines.append(f"  Events: {row['n_events']:,}, Filled: {row['n_filled']:,} ({row['fill_rate']:.1%})")
            lines.append(f"  Net ticks (filled): {row['avg_net_ticks']:+.3f}")
            lines.append(f"  Gross ticks: {row['avg_gross_ticks']:+.3f}")
            lines.append(f"  TP hit rate: {row['tp_hit_rate']:.1%}")
            lines.append(f"  Daily Sharpe: {row['daily_sharpe']:+.2f}")
            lines.append(f"  Daily Sortino: {row['daily_sortino']:+.2f}")
            lines.append(f"  Profitable days: {row['profitable_day_frac']:.1%}")
            lines.append(f"  Avg fills/day: {row['avg_fills_per_day']:.1f}")

    # Verdict
    lines.append("\n## Verdict")
    # Focus on the top 2% confluence (the proxy winner)
    for cfg in FIFO_CONFIGS:
        sub = df[(df["fifo_config"] == cfg) & (df["label"] == "Confluence") & (df["pct_threshold"] == 2)]
        if not sub.empty:
            row = sub.iloc[0]
            if row["avg_net_ticks"] > 0:
                lines.append(f"\n{cfg}: top 2% confluence SURVIVES FIFO at +{row['avg_net_ticks']:.3f} net ticks "
                             f"(fill rate {row['fill_rate']:.1%}, {row['n_filled']} fills)")
            else:
                lines.append(f"\n{cfg}: top 2% confluence FAILS FIFO at {row['avg_net_ticks']:+.3f} net ticks "
                             f"(fill rate {row['fill_rate']:.1%}, {row['n_filled']} fills)")

    return "\n".join(lines)


def main():
    print("=" * 80)
    print("CONFLUENCE FIFO REPLAY v1: CNN-Mamba v2 x PatchTST")
    print("=" * 80)

    dates = find_triple_overlap()
    print(f"\nTriple-overlap dates (CM + PT + FIFO): {len(dates)}")

    all_data = []
    for date in dates:
        d = load_day(date)
        if d is not None:
            all_data.append(d)
            print(f"  Loaded {date}: {d['n']:,} events")

    print(f"\nSuccessfully loaded {len(all_data)} days")

    if not all_data:
        print("ERROR: No data loaded!")
        return

    total_events = sum(d["n"] for d in all_data)

    # Run FIFO analysis
    results_df = analyze_fifo(all_data)

    # Save results
    results_df.to_csv(OUT_DIR / "confluence_fifo_results.csv", index=False)

    report = write_report(results_df, len(all_data), total_events)
    (OUT_DIR / "REPORT.md").write_text(report)

    print(f"\n\nResults saved to {OUT_DIR}")

    # Print key verdict
    print("\n" + "=" * 80)
    print("KEY QUESTION: Does +1.19 tick proxy survive FIFO replay?")
    print("=" * 80)
    for cfg in FIFO_CONFIGS:
        sub = results_df[(results_df["fifo_config"] == cfg) &
                         (results_df["label"] == "Confluence") &
                         (results_df["pct_threshold"] == 2)]
        if not sub.empty:
            row = sub.iloc[0]
            verdict = "YES" if row["avg_net_ticks"] > 0 else "NO"
            print(f"  {cfg}: {verdict} — net={row['avg_net_ticks']:+.3f} ticks, "
                  f"fill_rate={row['fill_rate']:.1%}, "
                  f"Sharpe={row['daily_sharpe']:+.2f}, "
                  f"profitable_days={row['profitable_day_frac']:.1%}")


if __name__ == "__main__":
    main()
