#!/usr/bin/env python3
"""
day_classifier_forward_walk_v1.py

Forward-walk / leave-one-out validation of the single-feature day-gating rule
identified by day_classifier_v1 (`trend_ticks_open_to_945` ascending, K=8 or K=12).

The original rule selected K and (feature, direction) by examining ALL 15 days.
This script enforces strict separation: for each held-out day d, the
(feature, direction, K) tuple is selected on the 14 TRAIN days only, then applied
to day d. Aggregating across all 15 LOO folds gives an honest forward-walk
estimate of the rule's robustness.

VERDICTS:
  ACCEPT-CONSERVATIVE: profit_days/days_sel >= 60%, pooled t/trade >= +1.0,
                       and selected feature stable (>=10/15 folds).
  ACCEPT-PARTIAL:      profit_days/days_sel >= 60% but pooled t/trade < +1.0.
  REJECT:              profit_days/days_sel < 60%.

NOTE on small samples: 15 days is tiny. Forward-walk reduces but does not
eliminate small-sample risk. 16+ more OOT days needed before live deployment.

Per HC #485 R5 we emit a `.regen_complete.json` marker.
"""

from __future__ import annotations

import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/home/jupiter/Lvl3Quant")
OUT = REPO / "output" / "day_classifier_forward_walk_v1"
OUT.mkdir(parents=True, exist_ok=True)

PER_DAY_FIFO = REPO / "output" / "meta_classifier_v1_fifo" / "per_day_fifo.csv"
DAY_FEATURES = REPO / "output" / "day_classifier_v1" / "day_features.parquet"

CANDIDATE = "short_10s_thr55"
K_GRID = [4, 6, 8, 10, 12]
DIRECTIONS = ["asc", "desc"]

# Features used by day_classifier_v1 (exclude meta cols)
EXCLUDE_COLS = {
    "date",
    "label_profitable",
    "day_mean_net_realized",
    "day_sum_net_realized",
    "n_filled",
}


def load_data():
    """Return per-day frame with one row per date for the target candidate."""
    per_day = pd.read_csv(PER_DAY_FIFO)
    per_day = per_day[per_day["candidate"] == CANDIDATE].copy()
    per_day["date"] = per_day["date"].astype(int)

    feats = pd.read_parquet(DAY_FEATURES)
    feats["date"] = feats["date"].astype(int)

    # Use per_day's authoritative P&L columns; keep only days present in both
    merged = feats.merge(
        per_day[["date", "n_filled", "day_mean_net_realized", "day_sum_net_realized"]],
        on="date",
        how="inner",
        suffixes=("_feat", ""),
    )
    # Drop the redundant feature-side copies if they came along
    for col in ("n_filled_feat", "day_mean_net_realized_feat", "day_sum_net_realized_feat"):
        if col in merged.columns:
            merged = merged.drop(columns=[col])

    # Sort by date so day index is reproducible
    merged = merged.sort_values("date").reset_index(drop=True)
    return merged


def pooled_t_per_trade(df: pd.DataFrame) -> float:
    """Pooled mean of net realized ticks per trade across selected days."""
    total_pnl = df["day_sum_net_realized"].sum()
    total_trades = df["n_filled"].sum()
    if total_trades <= 0:
        return float("nan")
    return float(total_pnl / total_trades)


def day_sharpe_ann(day_means: np.ndarray) -> float:
    """Annualised Sharpe of per-day mean t/trade (252 trading days)."""
    if len(day_means) < 2:
        return float("nan")
    mu = float(np.mean(day_means))
    sd = float(np.std(day_means, ddof=1))
    if sd == 0 or math.isnan(sd):
        return float("nan")
    return mu / sd * math.sqrt(252)


def select_topk(train_df: pd.DataFrame, feature: str, direction: str, k: int) -> pd.DataFrame:
    """Return top-K train days by feature in the chosen direction."""
    ascending = direction == "asc"
    return train_df.sort_values(feature, ascending=ascending).head(k)


def score_rule_on_train(train_df: pd.DataFrame, feature: str, direction: str, k: int):
    """Score a (feature, direction, K) rule on the training fold.

    Returns (profit_ratio, pooled_tt). Higher profit_ratio wins; ties broken by pooled_tt.
    """
    if k > len(train_df):
        return -1.0, -1e18
    sel = select_topk(train_df, feature, direction, k)
    n_profit = int((sel["day_mean_net_realized"] > 0).sum())
    profit_ratio = n_profit / k
    pooled = pooled_t_per_trade(sel)
    if math.isnan(pooled):
        pooled = -1e18
    return profit_ratio, pooled


def main():
    t0 = time.time()
    df = load_data()
    n_days = len(df)
    if n_days < 5:
        raise SystemExit(f"Too few days: {n_days}")

    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]

    fold_rows = []
    selection_counts: dict[tuple[str, str], int] = {}
    selection_counts_with_k: dict[tuple[str, str, int], int] = {}

    for d_idx in range(n_days):
        held_out = df.iloc[[d_idx]].copy()
        train = df.drop(index=d_idx).copy()
        held_out_date = int(held_out["date"].iloc[0])

        # Search over all (feature, direction, K) on train only
        best_key = None
        best_score = (-1.0, -1e18)  # (profit_ratio, pooled_tt)
        for feature in feature_cols:
            # Skip degenerate features (constant on train)
            if train[feature].nunique() < 2:
                continue
            for direction in DIRECTIONS:
                for k in K_GRID:
                    if k > len(train):
                        continue
                    pr, pooled = score_rule_on_train(train, feature, direction, k)
                    score = (pr, pooled)
                    if score > best_score:
                        best_score = score
                        best_key = (feature, direction, k)

        if best_key is None:
            # No usable rule (shouldn't happen)
            fold_rows.append(
                dict(
                    fold=d_idx,
                    held_out_date=held_out_date,
                    selected_feature=None,
                    selected_direction=None,
                    selected_K=None,
                    train_profit_ratio=None,
                    train_pooled_tt=None,
                    day_included=False,
                    held_out_day_pnl_mean_tt=float(held_out["day_mean_net_realized"].iloc[0]),
                    held_out_day_n_filled=float(held_out["n_filled"].iloc[0]),
                    held_out_day_sum_pnl=float(held_out["day_sum_net_realized"].iloc[0]),
                )
            )
            continue

        feature, direction, k = best_key
        selection_counts[(feature, direction)] = selection_counts.get((feature, direction), 0) + 1
        selection_counts_with_k[(feature, direction, k)] = (
            selection_counts_with_k.get((feature, direction, k), 0) + 1
        )

        # Apply rule: would held-out day be in top-K if we ranked ALL 15 days
        # by this feature in this direction? Use the train-derived threshold:
        # held-out day is "selected" if its feature value would place it in the
        # top-K of the full 15-day ranking under the same rule.
        full_sorted = df.sort_values(feature, ascending=(direction == "asc"))
        topk_full = set(full_sorted.head(k)["date"].astype(int).tolist())
        day_included = held_out_date in topk_full

        train_pr, train_pooled = score_rule_on_train(train, feature, direction, k)
        fold_rows.append(
            dict(
                fold=d_idx,
                held_out_date=held_out_date,
                selected_feature=feature,
                selected_direction=direction,
                selected_K=k,
                train_profit_ratio=round(train_pr, 4),
                train_pooled_tt=round(train_pooled, 4),
                day_included=bool(day_included),
                held_out_day_pnl_mean_tt=float(held_out["day_mean_net_realized"].iloc[0]),
                held_out_day_n_filled=float(held_out["n_filled"].iloc[0]),
                held_out_day_sum_pnl=float(held_out["day_sum_net_realized"].iloc[0]),
            )
        )

    folds_df = pd.DataFrame(fold_rows)
    folds_df.to_csv(OUT / "loo_folds.csv", index=False)

    # Out-of-sample aggregation: only the held-out days that were INCLUDED by
    # their train-selected rule contribute.
    included = folds_df[folds_df["day_included"] == True].copy()
    days_selected = len(included)
    profit_days = int((included["held_out_day_pnl_mean_tt"] > 0).sum())
    profit_ratio = profit_days / days_selected if days_selected > 0 else float("nan")

    included_for_pooled = included.rename(
        columns={
            "held_out_day_sum_pnl": "day_sum_net_realized",
            "held_out_day_n_filled": "n_filled",
        }
    )
    pooled_tt = pooled_t_per_trade(included_for_pooled) if days_selected > 0 else float("nan")
    sharpe = (
        day_sharpe_ann(included["held_out_day_pnl_mean_tt"].to_numpy())
        if days_selected >= 2
        else float("nan")
    )

    # Baseline (all 15 days, no gate)
    baseline_profit_days = int((df["day_mean_net_realized"] > 0).sum())
    baseline_pooled = pooled_t_per_trade(df)
    baseline_sharpe = day_sharpe_ann(df["day_mean_net_realized"].to_numpy())

    # Stability table
    stab_rows = []
    for (feat, dirn), cnt in sorted(selection_counts.items(), key=lambda kv: -kv[1]):
        stab_rows.append(
            dict(
                feature=feat,
                direction=dirn,
                selected_n_folds=cnt,
                selection_freq=round(cnt / n_days, 4),
            )
        )
    stab_df = pd.DataFrame(stab_rows)
    stab_df.to_csv(OUT / "rule_stability.csv", index=False)

    stab_with_k_rows = []
    for (feat, dirn, k), cnt in sorted(selection_counts_with_k.items(), key=lambda kv: -kv[1]):
        stab_with_k_rows.append(
            dict(
                feature=feat,
                direction=dirn,
                K=k,
                selected_n_folds=cnt,
                selection_freq=round(cnt / n_days, 4),
            )
        )
    pd.DataFrame(stab_with_k_rows).to_csv(OUT / "rule_stability_with_k.csv", index=False)

    # Verdict
    top_feature_freq = max(selection_counts.values()) if selection_counts else 0
    top_feature_key = (
        max(selection_counts.items(), key=lambda kv: kv[1])[0] if selection_counts else (None, None)
    )
    stable = top_feature_freq >= 10  # ≥10/15 folds same feature+direction

    if days_selected == 0 or math.isnan(profit_ratio):
        verdict = "REJECT"
        verdict_reason = "No held-out day was included by its train-selected rule."
    elif profit_ratio < 0.60:
        verdict = "REJECT"
        verdict_reason = (
            f"forward-walk profit_days/days_sel = {profit_days}/{days_selected} "
            f"= {profit_ratio:.1%} < 60%. K-selection drove the original +2.72 t/trade."
        )
    elif pooled_tt < 1.0:
        verdict = "ACCEPT-PARTIAL"
        verdict_reason = (
            f"profit_ratio {profit_ratio:.1%} ≥ 60% but pooled t/trade "
            f"{pooled_tt:+.3f} < +1.0 — gate works on hit-rate but not magnitude."
        )
    elif not stable:
        verdict = "ACCEPT-PARTIAL"
        verdict_reason = (
            f"profit_ratio {profit_ratio:.1%} and pooled t/trade {pooled_tt:+.3f} pass, "
            f"but top feature only stable {top_feature_freq}/{n_days} folds (need ≥10)."
        )
    else:
        verdict = "ACCEPT-CONSERVATIVE"
        verdict_reason = (
            f"profit_ratio {profit_ratio:.1%} ≥ 60%, pooled t/trade {pooled_tt:+.3f} ≥ +1.0, "
            f"and rule stable {top_feature_freq}/{n_days} folds on "
            f"({top_feature_key[0]}, {top_feature_key[1]})."
        )

    # REPORT.md
    runtime = time.time() - t0
    now = datetime.now(timezone.utc).isoformat()
    lines = []
    lines.append("# day_classifier_forward_walk_v1 — REPORT")
    lines.append("")
    lines.append(f"**Generated:** {now}")
    lines.append(f"**Runtime:** {runtime:.1f}s")
    lines.append(f"**Candidate:** `{CANDIDATE}`")
    lines.append(f"**Days available:** {n_days}")
    lines.append(f"**K grid:** {K_GRID}")
    lines.append(f"**Directions:** {DIRECTIONS}")
    lines.append(f"**Features searched per fold:** {len(feature_cols)}")
    lines.append("")
    lines.append(f"## Verdict: {verdict}")
    lines.append("")
    lines.append(verdict_reason)
    lines.append("")
    lines.append("## Forward-walk LOO out-of-sample metrics")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    lines.append(f"| days_selected (held-out days included by train rule) | {days_selected} |")
    lines.append(f"| profit_days | {profit_days} |")
    lines.append(
        f"| profit_ratio | {profit_ratio:.2%}" + (" |" if not math.isnan(profit_ratio) else " (n/a) |")
    )
    lines.append(f"| pooled t/trade | {pooled_tt:+.3f} |")
    lines.append(f"| day-Sharpe (annualised) | {sharpe:+.2f} |")
    lines.append("")
    lines.append("## Baseline (no gate, all 15 days)")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    lines.append(f"| profit_days / total | {baseline_profit_days}/{n_days} |")
    lines.append(f"| pooled t/trade | {baseline_pooled:+.3f} |")
    lines.append(f"| day-Sharpe (annualised) | {baseline_sharpe:+.2f} |")
    lines.append("")
    lines.append("## Rule stability (selection frequency across LOO folds)")
    lines.append("")
    lines.append("| feature | direction | folds_selected | freq |")
    lines.append("|---|---|---:|---:|")
    for r in stab_rows[:15]:
        lines.append(
            f"| `{r['feature']}` | {r['direction']} | {r['selected_n_folds']}/{n_days} | {r['selection_freq']:.2%} |"
        )
    lines.append("")
    lines.append("## Rule stability incl. K (top entries)")
    lines.append("")
    lines.append("| feature | direction | K | folds_selected | freq |")
    lines.append("|---|---|---:|---:|---:|")
    for r in stab_with_k_rows[:15]:
        lines.append(
            f"| `{r['feature']}` | {r['direction']} | {r['K']} | "
            f"{r['selected_n_folds']}/{n_days} | {r['selection_freq']:.2%} |"
        )
    lines.append("")
    lines.append("## Per-fold detail")
    lines.append("")
    lines.append("| fold | held_out_date | sel_feature | dir | K | train_pr | train_pooled | included | day_pnl_tt |")
    lines.append("|---:|---:|---|---|---:|---:|---:|---|---:|")
    for _, r in folds_df.iterrows():
        lines.append(
            f"| {r['fold']} | {r['held_out_date']} | `{r['selected_feature']}` | "
            f"{r['selected_direction']} | {r['selected_K']} | "
            f"{r['train_profit_ratio']} | {r['train_pooled_tt']} | "
            f"{'YES' if r['day_included'] else 'no'} | {r['held_out_day_pnl_mean_tt']:+.3f} |"
        )
    lines.append("")
    lines.append("## Honest small-sample discussion")
    lines.append("")
    lines.append(
        "**15 days is tiny.** Forward-walk LOO removes the K-selection-on-all-days "
        "bias of the original day_classifier_v1 result, but it does NOT eliminate "
        "small-sample risk. With 15 observations:"
    )
    lines.append("")
    lines.append("- One outlier day flipping inclusion can swing profit_ratio by ~7 percentage points.")
    lines.append("- The training fold has only 14 days; the (feature, direction, K) search has "
                 f"~{len(feature_cols)} features × 2 directions × {len(K_GRID)} Ks = "
                 f"~{len(feature_cols) * 2 * len(K_GRID)} candidate rules per fold, vastly more "
                 "candidates than training observations — so the per-fold rule choice itself "
                 "carries selection variance.")
    lines.append("- Stratified regime check (HC #428 R1) requires ≥40 OOT days. We have 15.")
    lines.append("")
    lines.append(
        "**Required before any live deployment:** validate on the NEXT 16+ OOT days "
        "(target: 40+ days, all regimes per HC #428 R1)."
    )
    lines.append("")
    lines.append("## Files produced")
    lines.append("- `REPORT.md` (this file)")
    lines.append("- `loo_folds.csv` — one row per held-out day")
    lines.append("- `rule_stability.csv` — (feature, direction) selection frequency")
    lines.append("- `rule_stability_with_k.csv` — (feature, direction, K) selection frequency")
    lines.append("- `.regen_complete.json` — per HC #485 R5")

    report_path = OUT / "REPORT.md"
    report_path.write_text("\n".join(lines))

    # .regen_complete.json (HC #485 R5)
    regen = {
        "script": "scripts/day_classifier_forward_walk_v1.py",
        "completed_at": now,
        "runtime_s": round(runtime, 2),
        "candidate": CANDIDATE,
        "n_days": n_days,
        "verdict": verdict,
        "fw_days_selected": days_selected,
        "fw_profit_days": profit_days,
        "fw_profit_ratio": None if math.isnan(profit_ratio) else round(profit_ratio, 4),
        "fw_pooled_tt": None if math.isnan(pooled_tt) else round(pooled_tt, 4),
        "fw_day_sharpe_ann": None if math.isnan(sharpe) else round(sharpe, 4),
        "baseline_profit_days": baseline_profit_days,
        "baseline_pooled_tt": round(baseline_pooled, 4),
        "top_feature": top_feature_key[0],
        "top_direction": top_feature_key[1],
        "top_feature_freq": top_feature_freq,
        "stable": bool(stable),
        "outputs": [
            "REPORT.md",
            "loo_folds.csv",
            "rule_stability.csv",
            "rule_stability_with_k.csv",
        ],
    }
    (OUT / ".regen_complete.json").write_text(json.dumps(regen, indent=2))

    # stdout summary
    print("=" * 72)
    print(f"day_classifier_forward_walk_v1 — VERDICT: {verdict}")
    print("=" * 72)
    print(verdict_reason)
    print()
    print(f"Forward-walk days_selected:    {days_selected}")
    print(f"Forward-walk profit_days:      {profit_days} / {days_selected}"
          + (f" = {profit_ratio:.1%}" if days_selected else ""))
    print(f"Forward-walk pooled t/trade:   {pooled_tt:+.3f}")
    print(f"Forward-walk day-Sharpe (ann): {sharpe:+.2f}")
    print(f"Baseline (no gate):            {baseline_profit_days}/{n_days} days, "
          f"pooled {baseline_pooled:+.3f} t/trade, Sharpe {baseline_sharpe:+.2f}")
    print()
    print("Top-3 (feature, direction) selection frequencies across LOO folds:")
    for r in stab_rows[:3]:
        print(f"  {r['feature']:<32} {r['direction']:<5} "
              f"{r['selected_n_folds']:>2}/{n_days}  ({r['selection_freq']:.0%})")
    print()
    print(f"Outputs written to: {OUT}")
    print(f"Runtime: {runtime:.1f}s")


if __name__ == "__main__":
    main()
