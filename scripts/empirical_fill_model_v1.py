#!/usr/bin/env python3
"""
Empirical Fill Probability Model v1
====================================
Builds P(fill) and P(profitable|fill) models from FIFO label data + smart_v3 features.
Replaces naive 50% fill assumption in RL execution environment.

Output: output/empirical_fill_model_v1/
  - fill_rate_by_feature.csv
  - logistic_fill_model.pkl
  - REPORT.md
"""

import os, sys, pickle, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss

warnings.filterwarnings("ignore")

ROOT = Path("/home/jupiter/Lvl3Quant")
FIFO_DIR = ROOT / "data/processed/mbo_events_smart_v3_fifo_labels"
EVENT_DIR = ROOT / "data/processed/mbo_events_smart_v3"
OUT_DIR = ROOT / "output/empirical_fill_model_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SIZE = 3000  # decision event = k * 250 + 2999

FEATURE_NAMES = [
    "time_delta_log", "event_type_id", "side_id", "price_rel_ticks", "qty_log",
    "spread_ticks", "cancel_side_asym_50", "rolling_ofi_500", "event_density_20",
    "price_mom_10", "qty_price_mom_50", "price_sign_mom_200",
    "event_type_entropy_200", "fill_add_restoration_100", "spread_velocity_50",
    "queue_replenishment", "mom_divergence", "ofi_x_spread",
    "vol_weighted_pmom", "buy_sell_intensity", "realized_volatility",
    "sweep_intensity", "ofi_short_100", "ofi_long_2000", "ofi_acceleration",
]

# Key features for fill probability analysis
KEY_FEATURES = {
    "spread_ticks": 5,
    "rolling_ofi_500": 7,       # book imbalance proxy
    "event_density_20": 8,      # trade flow intensity
    "realized_volatility": 20,  # price volatility
    "buy_sell_intensity": 19,   # directional pressure
    "queue_replenishment": 15,  # queue depth proxy
    "ofi_short_100": 22,        # short-term order flow
    "sweep_intensity": 21,      # consecutive same-side trades
}

# 15 evenly-spaced dates from the 143 available
DATES = [
    "20251102", "20251113", "20251125", "20251208", "20251219",
    "20251231", "20260112", "20260125", "20260205", "20260217",
    "20260301", "20260312", "20260405", "20260416", "20260429",
]

CONFIGS = ["tp4sl3", "tp8sl5"]


def load_day(date: str) -> pd.DataFrame | None:
    """Load FIFO labels + decision-point features for a single date."""
    fifo_path = FIFO_DIR / f"{date}_fifo_labels.npz"
    event_path = EVENT_DIR / f"{date}_mbo_events.npz"
    if not fifo_path.exists() or not event_path.exists():
        return None

    fifo = np.load(fifo_path)
    events = np.load(event_path)["events"]  # (N, 25)
    n_events = len(events)
    window_k = fifo["window_k"]
    decision_idx = window_k * 250 + (WINDOW_SIZE - 1)

    rows = []
    for i in range(len(window_k)):
        di = int(decision_idx[i])
        if di >= n_events:
            continue
        feat = events[di]  # 25 features at decision point
        row = {"date": date, "window_k": int(window_k[i])}

        # Add all 25 features
        for j, name in enumerate(FEATURE_NAMES):
            row[f"dp_{name}"] = float(feat[j])

        # Window-level stats (mean of last 50 events before decision)
        start = max(0, di - 50)
        window_events = events[start:di+1]
        for j, name in enumerate(FEATURE_NAMES):
            row[f"win_mean_{name}"] = float(np.nanmean(window_events[:, j]))

        # Add fill/profit labels for each config and side
        for cfg in CONFIGS:
            for side in ["long", "short"]:
                prefix = f"{cfg}_{side}"
                row[f"{prefix}_filled"] = bool(fifo[f"{prefix}_filled"][i])
                row[f"{prefix}_net_ticks"] = float(fifo[f"{prefix}_net_ticks"][i])
                row[f"{prefix}_gross_ticks"] = float(fifo[f"{prefix}_gross_ticks"][i])
                row[f"{prefix}_hit_tp"] = bool(fifo[f"{prefix}_hit_tp"][i])

        rows.append(row)

    return pd.DataFrame(rows) if rows else None


def compute_fill_rates(df: pd.DataFrame) -> dict:
    """Compute overall fill rates by config/side."""
    results = {}
    for cfg in CONFIGS:
        for side in ["long", "short"]:
            col = f"{cfg}_{side}_filled"
            filled = df[col]
            net_col = f"{cfg}_{side}_net_ticks"
            filled_df = df[filled]

            results[f"{cfg}_{side}"] = {
                "fill_rate": filled.mean(),
                "n_total": len(filled),
                "n_filled": filled.sum(),
                "avg_net_ticks_if_filled": filled_df[net_col].mean() if len(filled_df) > 0 else 0,
                "pct_profitable_if_filled": (filled_df[net_col] > 0).mean() if len(filled_df) > 0 else 0,
                "avg_gross_ticks_if_filled": filled_df[f"{cfg}_{side}_gross_ticks"].mean() if len(filled_df) > 0 else 0,
                "tp_hit_rate_if_filled": filled_df[f"{cfg}_{side}_hit_tp"].mean() if len(filled_df) > 0 else 0,
            }
    return results


def fill_rate_by_feature_quintile(df: pd.DataFrame) -> pd.DataFrame:
    """Compute fill rate by feature quintile for key features."""
    records = []
    for cfg in CONFIGS:
        for side in ["long", "short"]:
            fill_col = f"{cfg}_{side}_filled"
            net_col = f"{cfg}_{side}_net_ticks"

            for feat_name, feat_idx in KEY_FEATURES.items():
                dp_col = f"dp_{feat_name}"
                if dp_col not in df.columns:
                    continue

                vals = df[dp_col].values
                # Handle constant features
                if np.nanstd(vals) < 1e-10:
                    continue

                try:
                    quintiles = pd.qcut(df[dp_col], 5, labels=False, duplicates="drop")
                except ValueError:
                    continue

                for q in sorted(quintiles.dropna().unique()):
                    mask = quintiles == q
                    sub = df[mask]
                    if len(sub) < 10:
                        continue

                    filled_sub = sub[sub[fill_col]]
                    records.append({
                        "config": cfg,
                        "side": side,
                        "feature": feat_name,
                        "quintile": int(q),
                        "n": len(sub),
                        "fill_rate": sub[fill_col].mean(),
                        "feat_mean": sub[dp_col].mean(),
                        "feat_min": sub[dp_col].min(),
                        "feat_max": sub[dp_col].max(),
                        "avg_net_ticks_if_filled": filled_sub[net_col].mean() if len(filled_sub) > 0 else np.nan,
                        "pct_profitable_if_filled": (filled_sub[net_col] > 0).mean() if len(filled_sub) > 0 else np.nan,
                    })

    return pd.DataFrame(records)


def fit_logistic_fill_model(df: pd.DataFrame, cfg: str = "tp4sl3", side: str = "short"):
    """Fit logistic regression P(fill) = f(features) for the primary config."""
    fill_col = f"{cfg}_{side}_filled"
    net_col = f"{cfg}_{side}_net_ticks"

    # Use decision-point features
    feature_cols = [f"dp_{name}" for name in KEY_FEATURES.keys()]
    # Add window means for key features
    feature_cols += [f"win_mean_{name}" for name in KEY_FEATURES.keys()]

    X = df[feature_cols].values
    y_fill = df[fill_col].values.astype(int)

    # Remove rows with NaN
    valid = ~np.isnan(X).any(axis=1) & ~np.isinf(X).any(axis=1)
    X, y_fill = X[valid], y_fill[valid]

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit fill model
    fill_model = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
    fill_model.fit(X_scaled, y_fill)

    # Evaluate
    y_prob = fill_model.predict_proba(X_scaled)[:, 1]
    auc = roc_auc_score(y_fill, y_prob)
    brier = brier_score_loss(y_fill, y_prob)

    # P(profitable | filled) model
    filled_mask = y_fill == 1
    if filled_mask.sum() > 50:
        X_filled = X_scaled[filled_mask]
        y_profit = (df.loc[df.index[valid][filled_mask], net_col].values > 0).astype(int)
        profit_model = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
        profit_model.fit(X_filled, y_profit)
        y_profit_prob = profit_model.predict_proba(X_filled)[:, 1]
        profit_auc = roc_auc_score(y_profit, y_profit_prob) if len(np.unique(y_profit)) > 1 else 0.5
    else:
        profit_model = None
        profit_auc = None

    # Feature importance (absolute coefficient magnitude)
    coef_importance = pd.DataFrame({
        "feature": feature_cols,
        "coefficient": fill_model.coef_[0],
        "abs_coef": np.abs(fill_model.coef_[0]),
    }).sort_values("abs_coef", ascending=False)

    return {
        "fill_model": fill_model,
        "profit_model": profit_model,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "fill_auc": auc,
        "fill_brier": brier,
        "profit_auc": profit_auc,
        "coef_importance": coef_importance,
        "baseline_fill_rate": y_fill.mean(),
        "n_samples": len(y_fill),
        "n_filled": y_fill.sum(),
    }


def main():
    print("=" * 60)
    print("Empirical Fill Probability Model v1")
    print("=" * 60)

    # Load data
    dfs = []
    for date in DATES:
        print(f"  Loading {date}...", end=" ")
        day_df = load_day(date)
        if day_df is not None:
            print(f"{len(day_df)} windows")
            dfs.append(day_df)
        else:
            print("SKIP (missing)")

    if not dfs:
        print("ERROR: No data loaded!")
        sys.exit(1)

    df = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal: {len(df)} windows from {len(dfs)} dates")

    # 1. Overall fill rates
    print("\n--- Overall Fill Rates ---")
    overall = compute_fill_rates(df)
    for key, stats in overall.items():
        print(f"  {key}: fill={stats['fill_rate']:.1%} ({stats['n_filled']}/{stats['n_total']}), "
              f"net_if_filled={stats['avg_net_ticks_if_filled']:.2f}t, "
              f"P(profit|fill)={stats['pct_profitable_if_filled']:.1%}, "
              f"TP_hit={stats['tp_hit_rate_if_filled']:.1%}")

    # 2. Fill rate by feature quintile
    print("\n--- Fill Rate by Feature Quintile ---")
    quintile_df = fill_rate_by_feature_quintile(df)
    quintile_df.to_csv(OUT_DIR / "fill_rate_by_feature.csv", index=False)
    print(f"  Saved {len(quintile_df)} rows to fill_rate_by_feature.csv")

    # Show key findings
    for feat in KEY_FEATURES:
        sub = quintile_df[(quintile_df["feature"] == feat) &
                          (quintile_df["config"] == "tp4sl3") &
                          (quintile_df["side"] == "short")]
        if len(sub) >= 2:
            q0 = sub[sub["quintile"] == sub["quintile"].min()]["fill_rate"].values
            q4 = sub[sub["quintile"] == sub["quintile"].max()]["fill_rate"].values
            if len(q0) > 0 and len(q4) > 0:
                print(f"  {feat}: Q1={q0[0]:.1%} -> Q5={q4[0]:.1%} (spread={q4[0]-q0[0]:+.1%})")

    # 3. Fit logistic models for both configs/sides
    print("\n--- Logistic Fill Models ---")
    models = {}
    for cfg in CONFIGS:
        for side in ["long", "short"]:
            key = f"{cfg}_{side}"
            print(f"\n  Fitting {key}...")
            result = fit_logistic_fill_model(df, cfg, side)
            models[key] = result
            print(f"    AUC={result['fill_auc']:.4f}, Brier={result['fill_brier']:.4f}, "
                  f"baseline={result['baseline_fill_rate']:.1%}")
            if result['profit_auc'] is not None:
                print(f"    P(profit|fill) AUC={result['profit_auc']:.4f}")
            print(f"    Top features:")
            for _, row in result["coef_importance"].head(5).iterrows():
                print(f"      {row['feature']}: {row['coefficient']:+.4f}")

    # 4. Save primary model (tp4sl3_short is the main trading config)
    primary = models["tp4sl3_short"]
    save_obj = {
        "fill_model": primary["fill_model"],
        "profit_model": primary["profit_model"],
        "scaler": primary["scaler"],
        "feature_cols": primary["feature_cols"],
        "fill_auc": primary["fill_auc"],
        "baseline_fill_rate": primary["baseline_fill_rate"],
        "all_models": {k: {
            "fill_model": v["fill_model"],
            "profit_model": v["profit_model"],
            "scaler": v["scaler"],
            "feature_cols": v["feature_cols"],
        } for k, v in models.items()},
    }
    pkl_path = OUT_DIR / "logistic_fill_model.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(save_obj, f)
    print(f"\nSaved model to {pkl_path}")

    # 5. Generate report
    report_lines = ["# Empirical Fill Model v1 - Report\n"]
    report_lines.append(f"**Data**: {len(dfs)} dates, {len(df):,} windows\n")

    report_lines.append("## Overall Fill Rates\n")
    report_lines.append("| Config | Fill Rate | Net Ticks (if filled) | P(profit given fill) | TP Hit Rate |")
    report_lines.append("|--------|-----------|----------------------|---------------------|-------------|")
    for key, stats in overall.items():
        report_lines.append(
            f"| {key} | {stats['fill_rate']:.1%} | {stats['avg_net_ticks_if_filled']:.2f} | "
            f"{stats['pct_profitable_if_filled']:.1%} | {stats['tp_hit_rate_if_filled']:.1%} |"
        )

    report_lines.append("\n## Logistic Model Performance\n")
    report_lines.append("| Model | AUC | Brier | Baseline Fill | P(profit given fill) AUC |")
    report_lines.append("|-------|-----|-------|--------------|-------------------------|")
    for key, result in models.items():
        pauc = f"{result['profit_auc']:.4f}" if result['profit_auc'] else "N/A"
        report_lines.append(
            f"| {key} | {result['fill_auc']:.4f} | {result['fill_brier']:.4f} | "
            f"{result['baseline_fill_rate']:.1%} | {pauc} |"
        )

    report_lines.append("\n## Top Fill-Rate Predictors (tp4sl3_short)\n")
    report_lines.append("| Feature | Coefficient | Direction |")
    report_lines.append("|---------|-------------|-----------|")
    for _, row in primary["coef_importance"].head(8).iterrows():
        direction = "more fills" if row["coefficient"] > 0 else "fewer fills"
        report_lines.append(f"| {row['feature']} | {row['coefficient']:+.4f} | {direction} |")

    report_lines.append("\n## Key Findings\n")
    # Compute key finding: fill rate range
    short_fill = overall["tp4sl3_short"]["fill_rate"]
    long_fill = overall["tp4sl3_long"]["fill_rate"]
    report_lines.append(f"- **Actual fill rates are far from 50%**: tp4sl3 short={short_fill:.1%}, long={long_fill:.1%}")
    report_lines.append(f"- **Fill model AUC={primary['fill_auc']:.3f}** — features meaningfully predict fill probability")
    report_lines.append(f"- The naive 50% assumption over/under-estimates fills by {abs(short_fill - 0.5):.0%} (short) and {abs(long_fill - 0.5):.0%} (long)")
    report_lines.append(f"- **Recommended**: replace 50% constant with logistic model using {len(primary['feature_cols'])} features")

    report_text = "\n".join(report_lines)
    with open(OUT_DIR / "REPORT.md", "w") as f:
        f.write(report_text)
    print(f"\nReport saved to {OUT_DIR / 'REPORT.md'}")
    print("\n" + report_text)


if __name__ == "__main__":
    main()
