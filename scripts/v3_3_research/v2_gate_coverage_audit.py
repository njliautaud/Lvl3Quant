"""
HC #352 — v2 Gate Coverage Audit
=================================
Quantify how many v2 short-side signals the live Razer confluence gate is
rejecting that would have been part of the proven top-10% profitable band.

Per HC #349: FIFO-floor numbers herein are FLOOR-ONLY (commission + 1-tick-spread
crossing for market orders, commission-only for passive limits). Full
queue-position + adverse-selection comes in the next deliverable.

READ-ONLY: This script does NOT modify any model, trainer, or live config.
"""
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------------------------------------------------
# Constants from CLAUDE.md
# ----------------------------------------------------------------------------
TICK_VALUE_USD = 12.50
RT_COMM_TICKS = 0.376  # AMP commission round-trip in ticks
MARKET_COST_TICKS = 1.376  # commission + 1-tick spread crossing
PASSIVE_COST_TICKS = 0.376  # commission only

# Live framework_config.json gate settings (read 2026-05-14)
LIVE_GATE_PCTL_1S = 99.5
LIVE_GATE_PCTL_5S = 99.0
LIVE_GATE_PCTL_10S = 99.0

OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/v2_gate_coverage_audit_20260514")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

V2_FOLDS_DIR = Path("/home/jupiter/Lvl3Quant/output/cnn_mamba_v2_smart_v3_mar")
CONFLUENCE_NPZ = Path(
    "/home/jupiter/Lvl3Quant/output/confluence_features/all_confluence_features.npz"
)

# Roughly 6.5 hours RTH per session
RTH_HOURS = 6.5


def load_v2_smart_v3_mar():
    """Concatenate all per-day v2 OOT folds with their date stamps."""
    fold_files = sorted(
        [
            f
            for f in V2_FOLDS_DIR.glob("fold_*_oot_predictions.npz")
            if "concat" not in f.name
        ]
    )
    preds_list = []
    labels_list = []
    dates_list = []
    rows_per_day = {}
    for fp in fold_files:
        d = np.load(fp, allow_pickle=True)
        P = d["predictions"]
        L = d["labels"]
        oot_files = d["oot_files"]
        # Date from oot_files[0] basename
        oot = str(oot_files[0])
        date = oot.split("/")[-1].split("_")[0]
        # Mask out non-usable rows (NaN labels, etc.)
        mask = np.isfinite(L).all(axis=1) & np.isfinite(P).all(axis=1)
        P = P[mask]
        L = L[mask]
        # NOTE: fold_01 and fold_10 are duplicates (20260224) — keep both since they
        # ARE different walk-forward checkpoint OOTs; aggregate as "session-day-coverage"
        # equivalents. We will count unique-date-sessions below.
        preds_list.append(P)
        labels_list.append(L)
        dates_list.append(np.full(len(P), date))
        rows_per_day.setdefault(date, 0)
        rows_per_day[date] += len(P)
    P_all = np.concatenate(preds_list, axis=0)
    L_all = np.concatenate(labels_list, axis=0)
    D_all = np.concatenate(dates_list, axis=0)
    return P_all, L_all, D_all, rows_per_day


def gate_summary(preds_1s, preds_5s, preds_10s, label_1s, label_5s, label_10s, n_unique_days):
    """Compute per-gate-percentile short-side coverage + net edge metrics."""
    # Percentile rank for SHORT-SIDE = bottom percentile (most negative).
    # A gate at "P99 short" means: only accept preds in the bottom 1% (signal ≤ P1).
    # We parameterize as: for gate level X (e.g. 90, 95, 99, 99.5), accept short if
    # pred ≤ percentile (100-X) of that horizon's pred distribution.
    rows = []
    gates = [50, 70, 80, 90, 95, 97.5, 99, 99.5, 99.7, 99.9]
    for g in gates:
        thr_1s = np.percentile(preds_1s, 100 - g)
        thr_5s = np.percentile(preds_5s, 100 - g)
        thr_10s = np.percentile(preds_10s, 100 - g)
        # Cross-horizon AND gate (matches live config: 1s AND 5s AND 10s all pass)
        # NOTE: live config uses separate per-horizon percentiles. We mirror that.
        mask_1s = preds_1s <= thr_1s
        mask_5s = preds_5s <= thr_5s
        mask_10s = preds_10s <= thr_10s
        mask_all = mask_1s & mask_5s & mask_10s
        n_1s_only = mask_1s.sum()
        n_all = mask_all.sum()

        # Net-edge per trade (FIFO FLOOR — commission + spread for market, commission only passive)
        # Realized SHORT profit = -label (price went DOWN -> short profits)
        for horizon_name, label in [("1s", label_1s), ("5s", label_5s), ("10s", label_10s)]:
            sub_mask = locals()[f"mask_{horizon_name}"]
            n = sub_mask.sum()
            if n < 10:
                continue
            realized_ticks = -label[sub_mask]
            avg_move = realized_ticks.mean()
            wr = (realized_ticks > 0).mean()
            net_market = avg_move - MARKET_COST_TICKS
            net_passive = avg_move - PASSIVE_COST_TICKS
            rows.append(
                dict(
                    gate=g,
                    horizon=horizon_name,
                    use="single_horizon",
                    n_signals=int(n),
                    n_per_session=n / n_unique_days,
                    avg_realized_ticks=float(avg_move),
                    wr=float(wr),
                    net_ticks_market=float(net_market),
                    net_ticks_passive=float(net_passive),
                    total_net_ticks_market=float(net_market * n / n_unique_days),
                    total_net_ticks_passive=float(net_passive * n / n_unique_days),
                )
            )

        # ALL-HORIZON AND-gate (mirrors live confluence) — use 1s realized for edge
        n = mask_all.sum()
        if n >= 10:
            realized_ticks_1s = -label_1s[mask_all]
            realized_ticks_5s = -label_5s[mask_all]
            realized_ticks_10s = -label_10s[mask_all]
            for hn, rt in [("1s", realized_ticks_1s), ("5s", realized_ticks_5s), ("10s", realized_ticks_10s)]:
                rows.append(
                    dict(
                        gate=g,
                        horizon=hn,
                        use="all_horizon_AND",
                        n_signals=int(n),
                        n_per_session=n / n_unique_days,
                        avg_realized_ticks=float(rt.mean()),
                        wr=float((rt > 0).mean()),
                        net_ticks_market=float(rt.mean() - MARKET_COST_TICKS),
                        net_ticks_passive=float(rt.mean() - PASSIVE_COST_TICKS),
                        total_net_ticks_market=float((rt.mean() - MARKET_COST_TICKS) * n / n_unique_days),
                        total_net_ticks_passive=float((rt.mean() - PASSIVE_COST_TICKS) * n / n_unique_days),
                    )
                )
    return rows


def analyze_patchtst_veto():
    """Quantify PatchTST veto impact using aligned all_confluence_features.npz."""
    d = np.load(CONFLUENCE_NPZ, allow_pickle=True)
    # Filter rows where everything we need is finite
    cnn1 = d["cnn_pred_1s"]
    cnn5 = d["cnn_pred_5s"]
    cnn10 = d["cnn_pred_10s"]
    pt1 = d["ptst_pred_1s"]
    pt5 = d["ptst_pred_5s"]
    pt10 = d["ptst_pred_10s"]
    am1 = d["actual_move_1s"]
    am5 = d["actual_move_5s"]
    am10 = d["actual_move_10s"]
    dates = d["date"]

    mask = (
        np.isfinite(cnn1)
        & np.isfinite(cnn5)
        & np.isfinite(cnn10)
        & np.isfinite(pt1)
        & np.isfinite(pt5)
        & np.isfinite(pt10)
        & np.isfinite(am1)
        & np.isfinite(am5)
        & np.isfinite(am10)
    )
    cnn1, cnn5, cnn10 = cnn1[mask], cnn5[mask], cnn10[mask]
    pt1, pt5, pt10 = pt1[mask], pt5[mask], pt10[mask]
    am1, am5, am10 = am1[mask], am5[mask], am10[mask]
    dates_f = dates[mask]
    n_days = len(np.unique(dates_f))
    n_total = len(cnn1)

    # Define CNN top-10% short: cnn_pred_1s ≤ P10 of cnn1
    p10_1s = np.percentile(cnn1, 10)
    short_mask = cnn1 <= p10_1s

    # PatchTST "veto" per live config: veto if PatchTST DISAGREES at 5s and 10s.
    # PatchTST agrees on short if pt5 < 0 and pt10 < 0 (predicts down). We veto if
    # pt5 >= 0 OR pt10 >= 0 (PT predicts flat/up disagreeing with CNN short).
    veto_mask = (pt5 >= 0) | (pt10 >= 0)
    # Among CNN top-10% shorts: who survives the veto, who gets killed
    in_band = short_mask
    in_band_kept = in_band & ~veto_mask
    in_band_vetoed = in_band & veto_mask

    def stats(m, label_arr, horizon):
        if m.sum() == 0:
            return None
        rt = -label_arr[m]
        return dict(
            horizon=horizon,
            n=int(m.sum()),
            n_per_session=m.sum() / n_days,
            avg_realized_ticks=float(rt.mean()),
            wr=float((rt > 0).mean()),
            net_market=float(rt.mean() - MARKET_COST_TICKS),
            net_passive=float(rt.mean() - PASSIVE_COST_TICKS),
        )

    out = dict(
        n_total_rows=n_total,
        n_unique_days=int(n_days),
        cnn_top10_short_n=int(short_mask.sum()),
        cnn_top10_short_frac=float(short_mask.mean()),
        veto_reject_rate_in_top10_short=float(in_band_vetoed.sum() / max(short_mask.sum(), 1)),
        admitted_after_veto=stats(in_band_kept, am1, "1s"),
        rejected_by_veto=stats(in_band_vetoed, am1, "1s"),
        admitted_after_veto_5s=stats(in_band_kept, am5, "5s"),
        rejected_by_veto_5s=stats(in_band_vetoed, am5, "5s"),
        admitted_after_veto_10s=stats(in_band_kept, am10, "10s"),
        rejected_by_veto_10s=stats(in_band_vetoed, am10, "10s"),
    )
    return out


def plot_gate_curve(rows, n_unique_days, out_path):
    """Signal-count + net-tick-PnL/session vs percentile gate."""
    # Pick the 1s single-horizon view (most representative of live 1s gate)
    rows_1s = [r for r in rows if r["use"] == "single_horizon" and r["horizon"] == "1s"]
    rows_1s.sort(key=lambda r: r["gate"])
    gates = [r["gate"] for r in rows_1s]
    n_per_sess = [r["n_per_session"] for r in rows_1s]
    edge_passive = [r["net_ticks_passive"] for r in rows_1s]
    edge_market = [r["net_ticks_market"] for r in rows_1s]
    total_passive = [r["total_net_ticks_passive"] for r in rows_1s]
    total_market = [r["total_net_ticks_market"] for r in rows_1s]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.semilogy(gates, n_per_sess, "o-", color="C0", lw=2)
    ax.axvline(LIVE_GATE_PCTL_1S, color="r", ls="--", label=f"Live 1s gate (P{LIVE_GATE_PCTL_1S})")
    ax.axvline(90, color="g", ls=":", label="Proven top-10% (P90)")
    ax.set_xlabel("Short-side gate percentile")
    ax.set_ylabel("Signals per RTH session (log)")
    ax.set_title("Signal count vs gate setting (1s horizon, v2 short)")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(gates, edge_passive, "o-", color="C2", label="Passive limit (cost 0.376t)")
    ax.plot(gates, edge_market, "s-", color="C3", label="Market order (cost 1.376t)")
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(LIVE_GATE_PCTL_1S, color="r", ls="--", alpha=0.6)
    ax.axvline(90, color="g", ls=":", alpha=0.6)
    ax.set_xlabel("Short-side gate percentile")
    ax.set_ylabel("Net ticks per trade (FIFO FLOOR)")
    ax.set_title("Per-trade net edge vs gate setting")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(gates, total_passive, "o-", color="C2", label="Passive limit")
    ax.plot(gates, total_market, "s-", color="C3", label="Market order")
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(LIVE_GATE_PCTL_1S, color="r", ls="--", alpha=0.6)
    ax.axvline(90, color="g", ls=":", alpha=0.6)
    ax.set_xlabel("Short-side gate percentile")
    ax.set_ylabel("Total net ticks per session (FIFO FLOOR)")
    ax.set_title("Session-level PnL: edge × count")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    # Win rate
    wr = [r["wr"] * 100 for r in rows_1s]
    ax.plot(gates, wr, "o-", color="C4")
    ax.axhline(50, color="k", ls="--", lw=0.5)
    ax.axvline(LIVE_GATE_PCTL_1S, color="r", ls="--", alpha=0.6)
    ax.axvline(90, color="g", ls=":", alpha=0.6)
    ax.set_xlabel("Short-side gate percentile")
    ax.set_ylabel("Win rate %")
    ax.set_title("Win rate vs gate (1s)")
    ax.grid(alpha=0.3)

    plt.suptitle(
        f"v2 Short-Side Gate Coverage Audit (HC #352) — n_unique_days={n_unique_days}, "
        f"FIFO-FLOOR ONLY per HC #349",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"Saved plot -> {out_path}")


def main():
    print("Loading v2 smart_v3_mar OOT folds...")
    P, L, D, rows_per_day = load_v2_smart_v3_mar()
    n_unique_days = len(np.unique(D))
    print(f"  total rows: {len(P)}, unique dates: {n_unique_days}")
    print(f"  Note: fold_01 and fold_10 share date 20260224 (two checkpoints same day)")
    # Take 1s/5s/10s columns
    p1, p5, p10 = P[:, 0], P[:, 1], P[:, 2]
    l1, l5, l10 = L[:, 0], L[:, 1], L[:, 2]

    print("\nComputing gate sweep...")
    rows = gate_summary(p1, p5, p10, l1, l5, l10, n_unique_days)

    print("\nAnalyzing PatchTST veto on aligned confluence data...")
    veto = analyze_patchtst_veto()

    # Find optimal gate by max total net ticks per session, market-order basis
    opt_market = max(
        [r for r in rows if r["use"] == "single_horizon" and r["horizon"] == "1s"],
        key=lambda r: r["total_net_ticks_market"],
    )
    opt_passive = max(
        [r for r in rows if r["use"] == "single_horizon" and r["horizon"] == "1s"],
        key=lambda r: r["total_net_ticks_passive"],
    )

    # Save full table
    out_json = OUTPUT_DIR / "gate_sweep.json"
    with open(out_json, "w") as f:
        json.dump(
            dict(
                gate_sweep=rows,
                patchtst_veto=veto,
                optimal_market=opt_market,
                optimal_passive=opt_passive,
                n_unique_days=int(n_unique_days),
                rows_per_day={k: int(v) for k, v in rows_per_day.items()},
                live_gate=dict(
                    p1s=LIVE_GATE_PCTL_1S, p5s=LIVE_GATE_PCTL_5S, p10s=LIVE_GATE_PCTL_10S
                ),
            ),
            f,
            indent=2,
        )
    print(f"Saved -> {out_json}")

    # Plot
    plot_gate_curve(rows, n_unique_days, OUTPUT_DIR / "gate_coverage_histogram.png")

    # ------------- Build REPORT.md -------------
    rows_1s = [r for r in rows if r["use"] == "single_horizon" and r["horizon"] == "1s"]
    rows_1s.sort(key=lambda r: r["gate"])
    g_lookup = {r["gate"]: r for r in rows_1s}

    # P90 vs P99 coverage at 1s
    n_p90 = g_lookup[90]["n_signals"]
    n_p99 = g_lookup[99]["n_signals"]
    n_p995 = g_lookup[99.5]["n_signals"]
    frac_kept_p99 = n_p99 / n_p90
    frac_kept_p995 = n_p995 / n_p90

    md = []
    md.append("# v2 Short-Side Gate Coverage Audit (HC #352)")
    md.append("")
    md.append(f"**Date:** 2026-05-14")
    md.append(
        f"**Data source:** `output/cnn_mamba_v2_smart_v3_mar/fold_*_oot_predictions.npz` "
        f"(canonical v2 OOT: IC_1s/5s/10s ≈ 0.22/0.11/0.07, matches CLAUDE.md)"
    )
    md.append(f"**Confluence source:** `output/confluence_features/all_confluence_features.npz` "
              f"(aligned CNN-Mamba + PatchTST + realized moves)")
    md.append(f"**Unique OOT trading days analyzed:** {n_unique_days}")
    md.append(f"**Total OOT rows:** {len(P):,}")
    md.append("")
    md.append(
        "**HC #349 ANNOTATION — FIFO-FLOOR ONLY: Every PnL/edge number in this "
        "report uses commission ($4.70 RT = 0.376 ticks) + 1-tick spread crossing "
        "for market orders (1.376 ticks total) and commission-only for passive "
        "limits (0.376 ticks). NO queue-position model, NO adverse-selection cost. "
        "These numbers are a LOWER BOUND on tradeable edge and MUST NOT be used "
        "as a headline 'this works in live' claim. Full queue+adv-sel comes in the "
        "next deliverable.**"
    )
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 1. Coverage at P90 (top-10%) vs current P99.5 / P99")
    md.append("")
    md.append("**Live gate (from `live_trading/framework_config.json`):**")
    md.append(f"- `min_percentile_1s = {LIVE_GATE_PCTL_1S}` (top 0.5%)")
    md.append(f"- `min_percentile_5s = {LIVE_GATE_PCTL_5S}` (top 1%)")
    md.append(f"- `min_percentile_10s = {LIVE_GATE_PCTL_10S}` (top 1%)")
    md.append("- PatchTST veto: enabled on 5s and 10s")
    md.append("")
    md.append("**Per-session signal count, v2 short-side, single-horizon thresholds:**")
    md.append("")
    md.append("| Gate | n signals | n/session | avg move (t) | WR | net market (t) | net passive (t) | session net market | session net passive |")
    md.append("|-----:|----------:|----------:|-------------:|----:|---------------:|----------------:|-------------------:|--------------------:|")
    for r in rows_1s:
        md.append(
            f"| P{r['gate']:>5g} | {r['n_signals']:>8,} | {r['n_per_session']:>6.1f} | "
            f"{r['avg_realized_ticks']:>+6.3f} | {r['wr']*100:>5.1f}% | "
            f"{r['net_ticks_market']:>+6.3f} | {r['net_ticks_passive']:>+6.3f} | "
            f"{r['total_net_ticks_market']:>+7.2f} | {r['total_net_ticks_passive']:>+7.2f} |"
        )
    md.append("")
    md.append(f"**Coverage retention: live P99 gate keeps {frac_kept_p99*100:.1f}% of P90 short signals "
              f"({n_p99:,}/{n_p90:,}). P99.5 keeps {frac_kept_p995*100:.1f}% ({n_p995:,}/{n_p90:,}).**")
    md.append("")
    md.append(f"**Throwaway: P99 discards {(1-frac_kept_p99)*100:.1f}% of the proven top-10% short band. "
              f"P99.5 discards {(1-frac_kept_p995)*100:.1f}%.**")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 2. Net edge per trade per band — short side, FIFO FLOOR")
    md.append("")
    md.append("Per-horizon realized-move stats at the proven and live gate levels "
              "(short profit = price went DOWN, so realized = -label_ticks):")
    md.append("")
    md.append("| Gate | Horizon | n | n/sess | avg move (t) | WR | net market | net passive |")
    md.append("|-----:|:--------|--:|------:|------:|----:|------:|------:|")
    for g in [90, 95, 99, 99.5]:
        for h in ["1s", "5s", "10s"]:
            r = next(
                (rr for rr in rows if rr["use"] == "single_horizon" and rr["gate"] == g and rr["horizon"] == h),
                None,
            )
            if r is None:
                continue
            md.append(
                f"| P{g:g} | {h} | {r['n_signals']:>6,} | {r['n_per_session']:>5.1f} | "
                f"{r['avg_realized_ticks']:>+6.3f} | {r['wr']*100:>5.1f}% | "
                f"{r['net_ticks_market']:>+6.3f} | {r['net_ticks_passive']:>+6.3f} |"
            )
    md.append("")
    md.append("**Interpretation (1s horizon, short side):**")
    md.append(
        f"- At P90, avg short move = {g_lookup[90]['avg_realized_ticks']:+.3f} ticks "
        f"({g_lookup[90]['wr']*100:.1f}% WR). "
        f"Net market = {g_lookup[90]['net_ticks_market']:+.3f}t — "
        f"{'PROFITABLE' if g_lookup[90]['net_ticks_market']>0 else 'LOSING'} on market orders."
        f" Net passive = {g_lookup[90]['net_ticks_passive']:+.3f}t."
    )
    md.append(
        f"- At P99, avg short move = {g_lookup[99]['avg_realized_ticks']:+.3f} ticks "
        f"({g_lookup[99]['wr']*100:.1f}% WR). "
        f"Net market = {g_lookup[99]['net_ticks_market']:+.3f}t. "
        f"Net passive = {g_lookup[99]['net_ticks_passive']:+.3f}t."
    )
    md.append(
        f"- At P99.5, avg short move = {g_lookup[99.5]['avg_realized_ticks']:+.3f} ticks "
        f"({g_lookup[99.5]['wr']*100:.1f}% WR). "
        f"Net market = {g_lookup[99.5]['net_ticks_market']:+.3f}t. "
        f"Net passive = {g_lookup[99.5]['net_ticks_passive']:+.3f}t."
    )
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 3. PatchTST veto impact")
    md.append("")
    md.append(f"From `all_confluence_features.npz` ({veto['n_total_rows']:,} aligned rows, "
              f"{veto['n_unique_days']} unique days):")
    md.append("")
    md.append(f"- CNN-Mamba top-10% short band (cnn_pred_1s ≤ P10): n = {veto['cnn_top10_short_n']:,}")
    md.append(f"- PatchTST veto rule: reject if PT predicts ≥0 at 5s OR 10s (disagrees with CNN short).")
    md.append(f"- **Veto rejection rate within CNN top-10% shorts: "
              f"{veto['veto_reject_rate_in_top10_short']*100:.1f}%**")
    md.append("")
    md.append("**Edge of trades ADMITTED (PatchTST agrees) vs REJECTED (PatchTST vetoes), 1s realized:**")
    md.append("")
    md.append("| Group | n | n/sess | avg move (t) | WR | net market | net passive |")
    md.append("|:------|--:|------:|------:|----:|-------:|-------:|")
    a = veto["admitted_after_veto"]
    r = veto["rejected_by_veto"]
    md.append(
        f"| Admitted (PT agrees) | {a['n']:,} | {a['n_per_session']:.1f} | "
        f"{a['avg_realized_ticks']:+.3f} | {a['wr']*100:.1f}% | "
        f"{a['net_market']:+.3f} | {a['net_passive']:+.3f} |"
    )
    md.append(
        f"| Rejected (PT vetoes) | {r['n']:,} | {r['n_per_session']:.1f} | "
        f"{r['avg_realized_ticks']:+.3f} | {r['wr']*100:.1f}% | "
        f"{r['net_market']:+.3f} | {r['net_passive']:+.3f} |"
    )
    md.append("")
    helps = a["avg_realized_ticks"] > r["avg_realized_ticks"]
    md.append(
        f"**Verdict: PatchTST veto {'HELPS' if helps else 'HURTS'} — "
        f"admitted trades show {a['avg_realized_ticks']:+.3f}t vs rejected {r['avg_realized_ticks']:+.3f}t "
        f"(delta = {a['avg_realized_ticks']-r['avg_realized_ticks']:+.3f}t). "
        f"{'PT is correctly filtering noise.' if helps else 'PT is throwing away tradeable trades — veto should be loosened or dropped.'}**"
    )
    md.append("")
    md.append("**5s horizon:**")
    a5 = veto["admitted_after_veto_5s"]; r5 = veto["rejected_by_veto_5s"]
    md.append(f"- Admitted: avg {a5['avg_realized_ticks']:+.3f}t, WR {a5['wr']*100:.1f}%")
    md.append(f"- Rejected: avg {r5['avg_realized_ticks']:+.3f}t, WR {r5['wr']*100:.1f}%")
    md.append("**10s horizon:**")
    a10 = veto["admitted_after_veto_10s"]; r10 = veto["rejected_by_veto_10s"]
    md.append(f"- Admitted: avg {a10['avg_realized_ticks']:+.3f}t, WR {a10['wr']*100:.1f}%")
    md.append(f"- Rejected: avg {r10['avg_realized_ticks']:+.3f}t, WR {r10['wr']*100:.1f}%")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 4. Recommended gate setting")
    md.append("")
    md.append("Optimization criterion: max total net-tick PnL per session "
              "(edge × signal count), FIFO FLOOR basis. Single-horizon 1s gate, "
              "short side only.")
    md.append("")
    md.append("**Optimal under MARKET-ORDER cost (1.376t):**")
    md.append(f"- Gate: **P{opt_market['gate']:g}**")
    md.append(f"- n/session: {opt_market['n_per_session']:.1f}")
    md.append(f"- avg move: {opt_market['avg_realized_ticks']:+.3f}t (WR {opt_market['wr']*100:.1f}%)")
    md.append(f"- net per trade: {opt_market['net_ticks_market']:+.3f}t")
    md.append(f"- **session net ticks: {opt_market['total_net_ticks_market']:+.2f}t** "
              f"(= ${opt_market['total_net_ticks_market']*TICK_VALUE_USD:+,.0f}/session)")
    md.append("")
    md.append("**Optimal under PASSIVE-LIMIT cost (0.376t):**")
    md.append(f"- Gate: **P{opt_passive['gate']:g}**")
    md.append(f"- n/session: {opt_passive['n_per_session']:.1f}")
    md.append(f"- avg move: {opt_passive['avg_realized_ticks']:+.3f}t (WR {opt_passive['wr']*100:.1f}%)")
    md.append(f"- net per trade: {opt_passive['net_ticks_passive']:+.3f}t")
    md.append(f"- **session net ticks: {opt_passive['total_net_ticks_passive']:+.2f}t** "
              f"(= ${opt_passive['total_net_ticks_passive']*TICK_VALUE_USD:+,.0f}/session)")
    md.append("")
    md.append("**Live setting comparison (P99.5 1s gate, FIFO floor 1s realized):**")
    p995 = g_lookup[99.5]
    md.append(f"- n/session: {p995['n_per_session']:.1f}")
    md.append(f"- session net market: {p995['total_net_ticks_market']:+.2f}t "
              f"(${p995['total_net_ticks_market']*TICK_VALUE_USD:+,.0f}/sess)")
    md.append(f"- session net passive: {p995['total_net_ticks_passive']:+.2f}t "
              f"(${p995['total_net_ticks_passive']*TICK_VALUE_USD:+,.0f}/sess)")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## CAVEATS")
    md.append("")
    md.append("1. **FIFO FLOOR ONLY (HC #349):** All numbers are commission + 1-tick-spread "
              "lower bounds. Queue position and adverse selection are NOT modeled. "
              "Real live edge will be LOWER once queue model is layered on top.")
    md.append("2. **No fill probability:** Passive-limit numbers assume the limit fills "
              "at touch within the holding window. Real fill rates at top bands are "
              "well below 100% — separate fill-prob study needed (Razer paper-trader logs).")
    md.append("3. **Per-prediction trades, no stride dedup:** With 250-event stride, "
              "consecutive predictions can re-fire on the same alpha event. "
              "Real trade count after dedup will be lower.")
    md.append("4. **Sliding window walk-forward labels:** v2 smart_v3_mar uses 60d-train / "
              "1d-OOT sliding folds — proper out-of-sample, no leakage.")
    md.append("5. **STRONGEST MODEL FOR EXECUTION (per HC #350):** v2 short top-10% remains "
              "the only setup with demonstrated face-value edge in the relevant cost "
              "stack; this audit answers whether the live gate is denying us that edge.")
    md.append("")
    md.append("![Gate coverage histogram](gate_coverage_histogram.png)")
    md.append("")

    report_path = OUTPUT_DIR / "REPORT.md"
    report_path.write_text("\n".join(md))
    print(f"Saved report -> {report_path}")

    # Print 5-bullet summary to stdout
    print("\n" + "=" * 70)
    print("5-BULLET SUMMARY:")
    print("=" * 70)
    print(
        f"1. Live P99.5 1s gate admits {n_p995:,} of {n_p90:,} P90 short signals "
        f"= {frac_kept_p995*100:.1f}% retention (throws away {(1-frac_kept_p995)*100:.1f}% of the proven band)."
    )
    print(
        f"2. P90 short avg = {g_lookup[90]['avg_realized_ticks']:+.3f}t (WR {g_lookup[90]['wr']*100:.1f}%), "
        f"P99.5 avg = {g_lookup[99.5]['avg_realized_ticks']:+.3f}t "
        f"(WR {g_lookup[99.5]['wr']*100:.1f}%) — tighter gate selects bigger moves per trade."
    )
    print(
        f"3. Session net ticks: live P99.5 = {p995['total_net_ticks_market']:+.2f}t market / "
        f"{p995['total_net_ticks_passive']:+.2f}t passive; "
        f"optimal market = P{opt_market['gate']:g} ({opt_market['total_net_ticks_market']:+.2f}t); "
        f"optimal passive = P{opt_passive['gate']:g} ({opt_passive['total_net_ticks_passive']:+.2f}t)."
    )
    print(
        f"4. PatchTST veto rejects {veto['veto_reject_rate_in_top10_short']*100:.1f}% of CNN top-10% shorts; "
        f"admitted={veto['admitted_after_veto']['avg_realized_ticks']:+.3f}t vs "
        f"rejected={veto['rejected_by_veto']['avg_realized_ticks']:+.3f}t — "
        f"{'HELPS' if veto['admitted_after_veto']['avg_realized_ticks']>veto['rejected_by_veto']['avg_realized_ticks'] else 'HURTS'}."
    )
    print(
        f"5. RECOMMENDED: gate = P{opt_passive['gate']:g} (passive-limit basis) yielding "
        f"~{opt_passive['n_per_session']:.0f} short signals/session × "
        f"{opt_passive['net_ticks_passive']:+.3f}t net = "
        f"{opt_passive['total_net_ticks_passive']:+.2f}t FIFO-floor session PnL "
        f"(${opt_passive['total_net_ticks_passive']*TICK_VALUE_USD:+,.0f}/sess). "
        f"FIFO FLOOR ONLY — re-validate under queue+adv-sel before going live."
    )


if __name__ == "__main__":
    main()
