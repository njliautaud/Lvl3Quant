"""
adverse_selection_feature_study_v1.py
=====================================
Analyze which MBO microstructure features predict adverse selection
(filled-but-losing trades) vs profitable fills vs unfilled orders.

Uses FIFO labels (tp4sl3_short) and smart_v3 event features (25 features).

Output: output/adverse_selection_features_v1/
  - feature_importance.csv
  - REPORT.md
"""

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ============================================================
# Config
# ============================================================
BASE = Path("/home/jupiter/Lvl3Quant")
FIFO_DIR = BASE / "data/processed/mbo_events_smart_v3_fifo_labels"
EVENT_DIR = BASE / "data/processed/mbo_events_smart_v3"
OUT_DIR = BASE / "output/adverse_selection_features_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Use tp4sl3_short (short side has better signal per CLAUDE.md)
CONFIG = "tp4sl3_short"
WINDOW_SIZE = 3000  # CNN-Mamba window size
LOOKBACK = 10       # events before decision point to summarize

# 25 feature names from precompute_features_smart_v3.py
FEATURE_NAMES = [
    "time_delta_log",           # 0: inter-event time (log)
    "event_type_id",            # 1: add/cancel/modify/trade/fill
    "side_id",                  # 2: bid(-1) vs ask(+1)
    "price_rel_ticks",          # 3: price relative to mid (ticks)
    "qty_log",                  # 4: order quantity (log)
    "spread_ticks",             # 5: bid-ask spread
    "cancel_side_asym_50",      # 6: cancel asymmetry (50-event)
    "rolling_ofi_500",          # 7: order flow imbalance (500-event z-score)
    "event_density_20",         # 8: event arrival rate (20-event)
    "price_mom_10",             # 9: price momentum (10-event z-score)
    "qty_price_mom_50",         # 10: qty-weighted price momentum (50-event z-score)
    "price_sign_mom_200",       # 11: price sign momentum (200-event)
    "event_type_entropy_200",   # 12: event type entropy (200-event)
    "fill_add_restoration_100", # 13: fill-then-add restoration ratio
    "spread_velocity_50",       # 14: spread change rate (z-score)
    "queue_replenishment",      # 15: add/cancel rate ratio (z-score)
    "mom_divergence",           # 16: short-long momentum divergence
    "ofi_x_spread",             # 17: OFI * spread interaction (z-score)
    "vol_weighted_pmom",        # 18: volume-weighted price momentum (z-score)
    "buy_sell_intensity",       # 19: bid/ask EWMA ratio [-1, 1]
    "realized_volatility",      # 20: price change variance (z-score)
    "sweep_intensity",          # 21: consecutive same-side trades
    "ofi_short_100",            # 22: short-window OFI (z-score)
    "ofi_long_2000",            # 23: long-window OFI (z-score)
    "ofi_acceleration",         # 24: OFI rate of change (z-score)
]

# Select 15 representative dates (mix of recent)
DATES = [
    "20260310", "20260316", "20260320", "20260325", "20260330",
    "20260403", "20260407", "20260410", "20260414", "20260417",
    "20260421", "20260424", "20260427", "20260428", "20260429",
]


def load_day(date: str):
    """Load FIFO labels and event features for a date. Returns DataFrame or None."""
    fifo_path = FIFO_DIR / f"{date}_fifo_labels.npz"
    event_path = EVENT_DIR / f"{date}_mbo_events.npz"

    if not fifo_path.exists() or not event_path.exists():
        return None

    fifo = np.load(fifo_path)
    events_data = np.load(event_path)
    events = events_data["events"]  # (N, 25)
    n_events = len(events)

    window_k = fifo["window_k"]
    filled = fifo[f"{CONFIG}_filled"]
    net_ticks = fifo[f"{CONFIG}_net_ticks"]
    gross_ticks = fifo[f"{CONFIG}_gross_ticks"]
    exit_reason = fifo[f"{CONFIG}_exit_reason"]
    hold_time_ns = fifo[f"{CONFIG}_hold_time_ns"]

    # Compute decision event index for each window
    # decision_event_index = window_k * 250 + (WINDOW_SIZE - 1)
    decision_idx = window_k * 250 + (WINDOW_SIZE - 1)

    rows = []
    for i in range(len(window_k)):
        di = int(decision_idx[i])
        if di < LOOKBACK or di >= n_events:
            continue

        # Classify outcome
        is_filled = bool(filled[i])
        nt = float(net_ticks[i])
        gt = float(gross_ticks[i])

        if not is_filled:
            outcome = "UNFILLED"
        elif nt > 0:
            outcome = "PROFITABLE"
        elif nt < 0:
            outcome = "ADVERSE"
        else:
            outcome = "BREAKEVEN"

        # Extract features at decision point
        feat_at_decision = events[di]

        # Extract summary of last LOOKBACK events before decision
        window_events = events[di - LOOKBACK + 1: di + 1]  # (LOOKBACK, 25)

        row = {"date": date, "window_k": int(window_k[i]),
               "outcome": outcome, "net_ticks": nt, "gross_ticks": gt,
               "exit_reason": str(exit_reason[i]),
               "hold_time_s": float(hold_time_ns[i]) / 1e9}

        # Decision-point features
        for j, name in enumerate(FEATURE_NAMES):
            row[f"dp_{name}"] = float(feat_at_decision[j])

        # Window summary features (mean and std over last 10 events)
        for j, name in enumerate(FEATURE_NAMES):
            col_vals = window_events[:, j]
            row[f"win_mean_{name}"] = float(np.mean(col_vals))
            row[f"win_std_{name}"] = float(np.std(col_vals))

        # Trend: last value minus first value in window
        for j, name in enumerate(FEATURE_NAMES):
            row[f"win_trend_{name}"] = float(window_events[-1, j] - window_events[0, j])

        rows.append(row)

    if not rows:
        return None
    return pd.DataFrame(rows)


def compute_separation_metrics(df: pd.DataFrame, feature_cols: list):
    """Compute how well each feature separates PROFITABLE vs ADVERSE fills."""
    filled = df[df["outcome"].isin(["PROFITABLE", "ADVERSE"])].copy()
    profitable = filled[filled["outcome"] == "PROFITABLE"]
    adverse = filled[filled["outcome"] == "ADVERSE"]

    if len(profitable) < 20 or len(adverse) < 20:
        print(f"WARNING: Too few samples (profitable={len(profitable)}, adverse={len(adverse)})")
        return pd.DataFrame()

    results = []
    for col in feature_cols:
        p_vals = profitable[col].dropna().values
        a_vals = adverse[col].dropna().values

        if len(p_vals) < 10 or len(a_vals) < 10:
            continue

        # KS test
        ks_stat, ks_pval = stats.ks_2samp(p_vals, a_vals)

        # Mean difference (standardized by pooled std)
        pooled_std = np.sqrt((np.var(p_vals) + np.var(a_vals)) / 2)
        if pooled_std < 1e-10:
            cohens_d = 0.0
        else:
            cohens_d = (np.mean(p_vals) - np.mean(a_vals)) / pooled_std

        # Rank-biserial correlation
        # positive = feature higher for profitable
        combined = np.concatenate([p_vals, a_vals])
        labels = np.concatenate([np.ones(len(p_vals)), np.zeros(len(a_vals))])
        try:
            rbc, rbc_pval = stats.pointbiserialr(labels, combined)
        except:
            rbc, rbc_pval = 0.0, 1.0

        # AUC (feature as classifier)
        from sklearn.metrics import roc_auc_score
        try:
            auc = roc_auc_score(labels, combined)
        except:
            auc = 0.5

        results.append({
            "feature": col,
            "mean_profitable": np.mean(p_vals),
            "mean_adverse": np.mean(a_vals),
            "cohens_d": cohens_d,
            "ks_stat": ks_stat,
            "ks_pval": ks_pval,
            "rank_corr": rbc,
            "rank_pval": rbc_pval,
            "auc": auc,
            "n_profitable": len(p_vals),
            "n_adverse": len(a_vals),
        })

    return pd.DataFrame(results)


def compute_fill_vs_unfill_metrics(df: pd.DataFrame, feature_cols: list):
    """Compute how features differ between filled and unfilled orders."""
    filled = df[df["outcome"] != "UNFILLED"]
    unfilled = df[df["outcome"] == "UNFILLED"]

    if len(filled) < 20 or len(unfilled) < 20:
        return pd.DataFrame()

    results = []
    for col in feature_cols:
        f_vals = filled[col].dropna().values
        u_vals = unfilled[col].dropna().values

        if len(f_vals) < 10 or len(u_vals) < 10:
            continue

        ks_stat, ks_pval = stats.ks_2samp(f_vals, u_vals)
        pooled_std = np.sqrt((np.var(f_vals) + np.var(u_vals)) / 2)
        if pooled_std < 1e-10:
            cohens_d = 0.0
        else:
            cohens_d = (np.mean(f_vals) - np.mean(u_vals)) / pooled_std

        results.append({
            "feature": col,
            "mean_filled": np.mean(f_vals),
            "mean_unfilled": np.mean(u_vals),
            "cohens_d": cohens_d,
            "ks_stat": ks_stat,
            "ks_pval": ks_pval,
        })

    return pd.DataFrame(results)


def main():
    print("=" * 70)
    print("ADVERSE SELECTION FEATURE STUDY V1")
    print("=" * 70)

    # Load data
    all_dfs = []
    for date in DATES:
        df = load_day(date)
        if df is not None:
            all_dfs.append(df)
            n_filled = (df["outcome"] != "UNFILLED").sum()
            n_total = len(df)
            print(f"  {date}: {n_total} windows, {n_filled} filled ({100*n_filled/n_total:.1f}%)")
        else:
            print(f"  {date}: SKIPPED (missing data)")

    if not all_dfs:
        print("ERROR: No data loaded!")
        return

    df = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal: {len(df)} windows")
    print(f"Outcome distribution:")
    for outcome, count in df["outcome"].value_counts().items():
        print(f"  {outcome}: {count} ({100*count/len(df):.1f}%)")

    # Exit reason breakdown for filled trades
    filled_df = df[df["outcome"] != "UNFILLED"]
    print(f"\nExit reasons (filled trades):")
    for reason, count in filled_df["exit_reason"].value_counts().items():
        print(f"  {reason}: {count}")

    # Feature columns
    dp_cols = [f"dp_{name}" for name in FEATURE_NAMES]
    win_mean_cols = [f"win_mean_{name}" for name in FEATURE_NAMES]
    win_std_cols = [f"win_std_{name}" for name in FEATURE_NAMES]
    win_trend_cols = [f"win_trend_{name}" for name in FEATURE_NAMES]
    all_feature_cols = dp_cols + win_mean_cols + win_std_cols + win_trend_cols

    # ============================================================
    # Analysis 1: PROFITABLE vs ADVERSE separation
    # ============================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 1: PROFITABLE vs ADVERSE fill separation")
    print("=" * 70)

    sep_df = compute_separation_metrics(df, all_feature_cols)
    if sep_df.empty:
        print("Not enough data for separation analysis")
        return

    sep_df["abs_cohens_d"] = sep_df["cohens_d"].abs()
    sep_df = sep_df.sort_values("abs_cohens_d", ascending=False)

    # Save full results
    sep_df.to_csv(OUT_DIR / "feature_importance.csv", index=False)

    # Print top 25
    print("\nTop 25 features separating PROFITABLE vs ADVERSE fills:")
    print(f"{'Rank':>4} {'Feature':<45} {'Cohen_d':>8} {'AUC':>6} {'KS':>6} {'KS_p':>8}")
    print("-" * 85)
    for rank, (_, row) in enumerate(sep_df.head(25).iterrows(), 1):
        print(f"{rank:>4} {row['feature']:<45} {row['cohens_d']:>+8.4f} "
              f"{row['auc']:>6.3f} {row['ks_stat']:>6.3f} {row['ks_pval']:>8.2e}")

    # ============================================================
    # Analysis 2: FILLED vs UNFILLED separation
    # ============================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 2: FILLED vs UNFILLED separation")
    print("=" * 70)

    fill_sep = compute_fill_vs_unfill_metrics(df, dp_cols)
    if not fill_sep.empty:
        fill_sep["abs_cohens_d"] = fill_sep["cohens_d"].abs()
        fill_sep = fill_sep.sort_values("abs_cohens_d", ascending=False)
        fill_sep.to_csv(OUT_DIR / "fill_vs_unfill_importance.csv", index=False)

        print("\nTop 15 features separating FILLED vs UNFILLED:")
        print(f"{'Rank':>4} {'Feature':<40} {'Cohen_d':>8} {'KS':>6} {'Mean_Fill':>10} {'Mean_Unfill':>10}")
        print("-" * 85)
        for rank, (_, row) in enumerate(fill_sep.head(15).iterrows(), 1):
            print(f"{rank:>4} {row['feature']:<40} {row['cohens_d']:>+8.4f} "
                  f"{row['ks_stat']:>6.3f} {row['mean_filled']:>10.4f} {row['mean_unfilled']:>10.4f}")

    # ============================================================
    # Analysis 3: Key microstructure features deep dive
    # ============================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 3: Key microstructure features — per-outcome distributions")
    print("=" * 70)

    key_features = [
        "dp_spread_ticks",
        "dp_rolling_ofi_500",
        "dp_price_mom_10",
        "dp_buy_sell_intensity",
        "dp_realized_volatility",
        "dp_ofi_short_100",
        "dp_ofi_long_2000",
        "dp_ofi_acceleration",
        "dp_sweep_intensity",
        "dp_cancel_side_asym_50",
        "dp_mom_divergence",
        "dp_spread_velocity_50",
    ]

    for feat in key_features:
        if feat not in df.columns:
            continue
        print(f"\n{feat}:")
        for outcome in ["UNFILLED", "PROFITABLE", "ADVERSE"]:
            subset = df[df["outcome"] == outcome][feat].dropna()
            if len(subset) < 5:
                continue
            print(f"  {outcome:>12}: mean={subset.mean():>+8.4f}  std={subset.std():>7.4f}  "
                  f"med={subset.median():>+8.4f}  [p5={subset.quantile(0.05):>+7.3f}, "
                  f"p95={subset.quantile(0.95):>+7.3f}]  n={len(subset)}")

    # ============================================================
    # Analysis 4: Conditional adverse selection rate by feature quintile
    # ============================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 4: Adverse selection rate by feature quintile (filled only)")
    print("=" * 70)

    filled_only = df[df["outcome"].isin(["PROFITABLE", "ADVERSE"])].copy()
    filled_only["is_adverse"] = (filled_only["outcome"] == "ADVERSE").astype(int)

    key_dp_features = [
        "dp_spread_ticks", "dp_rolling_ofi_500", "dp_price_mom_10",
        "dp_buy_sell_intensity", "dp_realized_volatility", "dp_ofi_short_100",
        "dp_ofi_acceleration", "dp_sweep_intensity", "dp_cancel_side_asym_50",
        "dp_spread_velocity_50",
    ]

    quintile_results = []
    for feat in key_dp_features:
        if feat not in filled_only.columns:
            continue
        try:
            filled_only["_q"] = pd.qcut(filled_only[feat], 5, labels=False, duplicates="drop")
        except:
            continue

        print(f"\n{feat}:")
        print(f"  {'Q':>3} {'Range':>25} {'N':>6} {'Adv%':>7} {'AvgNet':>8}")
        for q in sorted(filled_only["_q"].dropna().unique()):
            qdf = filled_only[filled_only["_q"] == q]
            adv_rate = qdf["is_adverse"].mean()
            avg_net = qdf["net_ticks"].mean()
            feat_min = qdf[feat].min()
            feat_max = qdf[feat].max()
            n = len(qdf)
            print(f"  {int(q):>3} [{feat_min:>+9.3f}, {feat_max:>+9.3f}] {n:>6} {100*adv_rate:>6.1f}% {avg_net:>+8.3f}")
            quintile_results.append({
                "feature": feat, "quintile": int(q),
                "feat_min": feat_min, "feat_max": feat_max,
                "n": n, "adverse_rate": adv_rate, "avg_net_ticks": avg_net
            })

    if quintile_results:
        pd.DataFrame(quintile_results).to_csv(OUT_DIR / "quintile_adverse_rates.csv", index=False)

    # ============================================================
    # Analysis 5: Multi-feature adverse selection prediction (simple logistic)
    # ============================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 5: Logistic regression — predicting adverse fills")
    print("=" * 70)

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, classification_report

    filled_only2 = df[df["outcome"].isin(["PROFITABLE", "ADVERSE"])].copy()
    y = (filled_only2["outcome"] == "ADVERSE").astype(int).values

    # Use decision-point features only for simplicity
    X = filled_only2[dp_cols].values

    # Remove constant columns
    col_std = np.std(X, axis=0)
    valid_cols = col_std > 1e-8
    X = X[:, valid_cols]
    valid_dp_names = [dp_cols[i] for i in range(len(dp_cols)) if valid_cols[i]]

    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Fit logistic regression
    lr = LogisticRegression(max_iter=1000, C=0.1, penalty="l2")
    lr.fit(X, y)

    y_prob = lr.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, y_prob)
    print(f"\nLogistic Regression (in-sample, 25 dp features):")
    print(f"  AUC: {auc:.4f}")
    print(f"  Baseline adverse rate: {y.mean():.3f}")

    # Feature coefficients
    coef_df = pd.DataFrame({
        "feature": valid_dp_names,
        "coefficient": lr.coef_[0],
        "abs_coefficient": np.abs(lr.coef_[0])
    }).sort_values("abs_coefficient", ascending=False)

    print(f"\n  Top logistic regression coefficients (positive = more adverse):")
    for _, row in coef_df.head(15).iterrows():
        direction = "-> ADVERSE" if row["coefficient"] > 0 else "-> PROFITABLE"
        print(f"    {row['feature']:<40} {row['coefficient']:>+8.4f}  {direction}")

    coef_df.to_csv(OUT_DIR / "logistic_coefficients.csv", index=False)

    # ============================================================
    # Generate Report
    # ============================================================
    print("\n" + "=" * 70)
    print("Generating REPORT.md ...")

    # Get top 5 adverse predictors and top 5 profitable predictors
    top_adverse = sep_df[sep_df["cohens_d"] < 0].head(5)  # negative = higher for adverse
    top_profitable = sep_df[sep_df["cohens_d"] > 0].head(5)  # positive = higher for profitable

    n_profitable = (df["outcome"] == "PROFITABLE").sum()
    n_adverse = (df["outcome"] == "ADVERSE").sum()
    n_unfilled = (df["outcome"] == "UNFILLED").sum()
    n_total = len(df)
    fill_rate = (n_profitable + n_adverse) / n_total * 100
    adverse_rate_of_fills = n_adverse / max(n_adverse + n_profitable, 1) * 100

    report_lines = [
        "# Adverse Selection Feature Study V1",
        "",
        f"**Config**: {CONFIG} | **Dates**: {len(all_dfs)} days | **Windows**: {n_total:,}",
        "",
        "## Outcome Distribution",
        f"- Unfilled: {n_unfilled:,} ({100*n_unfilled/n_total:.1f}%)",
        f"- Profitable fills: {n_profitable:,} ({100*n_profitable/n_total:.1f}%)",
        f"- Adverse fills: {n_adverse:,} ({100*n_adverse/n_total:.1f}%)",
        f"- Fill rate: {fill_rate:.1f}% | Adverse rate (of fills): {adverse_rate_of_fills:.1f}%",
        "",
        "## Top Features Predicting ADVERSE Fills",
        "| Feature | Cohen's d | AUC | Interpretation |",
        "|---------|-----------|-----|----------------|",
    ]

    for _, row in sep_df.head(10).iterrows():
        feat = row["feature"]
        d = row["cohens_d"]
        auc_val = row["auc"]
        if d > 0:
            interp = "Higher values -> more profitable"
        else:
            interp = "Higher values -> more adverse"
        report_lines.append(f"| {feat} | {d:+.4f} | {auc_val:.3f} | {interp} |")

    report_lines += [
        "",
        f"## Logistic Regression AUC: {auc:.4f}",
        f"Baseline adverse rate: {y.mean():.1%}",
        "",
        "## Key Findings",
    ]

    # Summarize key findings
    best_feat = sep_df.iloc[0]
    report_lines.append(f"- Strongest separator: **{best_feat['feature']}** (|d|={best_feat['abs_cohens_d']:.3f}, AUC={best_feat['auc']:.3f})")
    report_lines.append(f"- Logistic regression with 25 decision-point features achieves AUC={auc:.3f}")

    report = "\n".join(report_lines) + "\n"
    (OUT_DIR / "REPORT.md").write_text(report)
    print("Done! Output in:", OUT_DIR)


if __name__ == "__main__":
    main()
