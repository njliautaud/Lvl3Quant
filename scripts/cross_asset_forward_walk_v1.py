#!/usr/bin/env python3
"""
cross_asset_forward_walk_v1.py

Forward-walk / leave-one-out validation of the cross-asset day-gating rule
identified by cross_asset_day_classifier_v1 (VIX_change_5d desc AUC=0.848,
SPX_5d_return asc strong runner-up).

The original cross-asset result selected the headline feature by examining
ALL 15 days. This script enforces strict separation: for each held-out day d,
the (feature, direction, K) tuple is selected on the 14 TRAIN days only, then
applied to day d. Aggregating across all 15 LOO folds gives an honest
forward-walk estimate of the AUGMENTED feature-set's robustness.

This is the cross-asset analogue of day_classifier_forward_walk_v1.py, run
against the combined feature set (ES intrinsic ~13 features + cross-asset 7
features + calendar 5 features = ~23 features per day).

VERDICTS:
  ACCEPT-CONSERVATIVE: profit_days/days_sel >= 60%, pooled t/trade >= +1.0,
                       and any (feature, direction) stable (>=10/15 folds).
  REJECT:              otherwise.

NOTE on small samples: 15 days is tiny. Bonferroni noise floor at ~23 features
× 2 directions is roughly AUC 0.78 on this sample size, so the original 0.848
is suggestive but not conclusive. Forward-walk LOO is the honest test.

Per HC #485 R5 we emit a `.regen_complete.json` marker.
"""

from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/home/jupiter/Lvl3Quant")
OUT = REPO / "output" / "cross_asset_forward_walk_v1"
OUT.mkdir(parents=True, exist_ok=True)

COMBINED_FEATURES = REPO / "output" / "cross_asset_day_classifier_v1" / "combined_features.parquet"
PER_DAY_FIFO = REPO / "output" / "meta_classifier_v1_fifo" / "per_day_fifo.csv"

CANDIDATE = "short_10s_thr55"
K_GRID = [4, 6, 8, 10, 12]
DIRECTIONS = ["asc", "desc"]

# Meta / target columns to exclude from the feature search
EXCLUDE_COLS = {
    "date",
    "label_profitable",
    "day_mean_net_realized",
    "day_sum_net_realized",
    "n_filled",
}


def load_data() -> pd.DataFrame:
    """Load combined cross-asset features merged with per-day FIFO P&L.

    The combined_features.parquet already contains day_mean_net_realized,
    day_sum_net_realized, and n_filled for the short_10s_thr55 candidate
    that the cross_asset_day_classifier_v1 was built on. We confirm via
    per_day_fifo.csv that targets align.
    """
    feats = pd.read_parquet(COMBINED_FEATURES)
    feats["date"] = feats["date"].astype(int)

    per_day = pd.read_csv(PER_DAY_FIFO)
    per_day = per_day[per_day["candidate"] == CANDIDATE].copy()
    per_day["date"] = per_day["date"].astype(int)

    # If combined_features already has PnL cols, just verify consistency on date set;
    # otherwise merge them in. We prefer the authoritative per_day_fifo values.
    pnl_cols = ["n_filled", "day_mean_net_realized", "day_sum_net_realized"]
    have_pnl_in_feats = all(c in feats.columns for c in pnl_cols)

    if have_pnl_in_feats:
        # Drop and re-merge from authoritative source to be safe
        feats = feats.drop(columns=pnl_cols)

    merged = feats.merge(
        per_day[["date"] + pnl_cols],
        on="date",
        how="inner",
    )

    merged = merged.sort_values("date").reset_index(drop=True)
    return merged


def pooled_t_per_trade(df: pd.DataFrame) -> float:
    total_pnl = df["day_sum_net_realized"].sum()
    total_trades = df["n_filled"].sum()
    if total_trades <= 0:
        return float("nan")
    return float(total_pnl / total_trades)


def day_sharpe_ann(day_means: np.ndarray) -> float:
    if len(day_means) < 2:
        return float("nan")
    mu = float(np.mean(day_means))
    sd = float(np.std(day_means, ddof=1))
    if sd == 0 or math.isnan(sd):
        return float("nan")
    return mu / sd * math.sqrt(252)


def select_topk(train_df: pd.DataFrame, feature: str, direction: str, k: int) -> pd.DataFrame:
    ascending = direction == "asc"
    return train_df.sort_values(feature, ascending=ascending).head(k)


def score_rule_on_train(train_df: pd.DataFrame, feature: str, direction: str, k: int):
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

    # Track key cross-asset features specifically requested in the task
    vix_change_5d_desc_count = 0
    spx_5d_return_asc_count = 0

    for d_idx in range(n_days):
        held_out = df.iloc[[d_idx]].copy()
        train = df.drop(index=d_idx).copy()
        held_out_date = int(held_out["date"].iloc[0])

        best_key = None
        best_score = (-1.0, -1e18)
        for feature in feature_cols:
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
        if feature == "VIX_change_5d" and direction == "desc":
            vix_change_5d_desc_count += 1
        if feature == "SPX_5d_return" and direction == "asc":
            spx_5d_return_asc_count += 1

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

    baseline_profit_days = int((df["day_mean_net_realized"] > 0).sum())
    baseline_pooled = pooled_t_per_trade(df)
    baseline_sharpe = day_sharpe_ann(df["day_mean_net_realized"].to_numpy())

    # Stability tables
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

    top_feature_freq = max(selection_counts.values()) if selection_counts else 0
    top_feature_key = (
        max(selection_counts.items(), key=lambda kv: kv[1])[0] if selection_counts else (None, None)
    )
    stable = top_feature_freq >= 10

    if days_selected == 0 or math.isnan(profit_ratio):
        verdict = "REJECT"
        verdict_reason = "No held-out day was included by its train-selected rule."
    elif profit_ratio < 0.60:
        verdict = "REJECT"
        verdict_reason = (
            f"forward-walk profit_days/days_sel = {profit_days}/{days_selected} "
            f"= {profit_ratio:.1%} < 60%. Cross-asset feature edge does not survive "
            f"strict train/test separation."
        )
    elif pooled_tt < 1.0:
        verdict = "REJECT"
        verdict_reason = (
            f"profit_ratio {profit_ratio:.1%} >= 60% but pooled t/trade "
            f"{pooled_tt:+.3f} < +1.0 — gate hit-rate ok but magnitude fails."
        )
    elif not stable:
        verdict = "REJECT"
        verdict_reason = (
            f"profit_ratio {profit_ratio:.1%} and pooled t/trade {pooled_tt:+.3f} pass, "
            f"but no (feature, direction) stable enough ({top_feature_freq}/{n_days} < 10)."
        )
    else:
        verdict = "ACCEPT-CONSERVATIVE"
        verdict_reason = (
            f"profit_ratio {profit_ratio:.1%} >= 60%, pooled t/trade {pooled_tt:+.3f} >= +1.0, "
            f"and rule stable {top_feature_freq}/{n_days} folds on "
            f"({top_feature_key[0]}, {top_feature_key[1]})."
        )

    runtime = time.time() - t0
    now = datetime.now(timezone.utc).isoformat()
    lines = []
    lines.append("# cross_asset_forward_walk_v1 — REPORT")
    lines.append("")
    lines.append(f"**Generated:** {now}")
    lines.append(f"**Runtime:** {runtime:.1f}s")
    lines.append(f"**Candidate:** `{CANDIDATE}`")
    lines.append(f"**Days available:** {n_days}")
    lines.append(f"**K grid:** {K_GRID}")
    lines.append(f"**Directions:** {DIRECTIONS}")
    lines.append(f"**Features searched per fold (augmented w/ cross-asset):** {len(feature_cols)}")
    lines.append("")
    lines.append(f"## Verdict: {verdict}")
    lines.append("")
    lines.append(verdict_reason)
    lines.append("")
    lines.append("## Cross-asset headline-feature stability")
    lines.append("")
    lines.append("| feature | direction | folds_selected | freq |")
    lines.append("|---|---|---:|---:|")
    lines.append(f"| `VIX_change_5d` | desc | {vix_change_5d_desc_count}/{n_days} | "
                 f"{vix_change_5d_desc_count / n_days:.2%} |")
    lines.append(f"| `SPX_5d_return` | asc | {spx_5d_return_asc_count}/{n_days} | "
                 f"{spx_5d_return_asc_count / n_days:.2%} |")
    lines.append("")
    lines.append("## Forward-walk LOO out-of-sample metrics")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    lines.append(f"| days_selected (held-out days included by train rule) | {days_selected} |")
    lines.append(f"| profit_days | {profit_days} |")
    lines.append(
        f"| profit_ratio | "
        + (f"{profit_ratio:.2%} |" if not math.isnan(profit_ratio) else "n/a |")
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
    for r in stab_rows[:20]:
        lines.append(
            f"| `{r['feature']}` | {r['direction']} | "
            f"{r['selected_n_folds']}/{n_days} | {r['selection_freq']:.2%} |"
        )
    lines.append("")
    lines.append("## Rule stability incl. K (top entries)")
    lines.append("")
    lines.append("| feature | direction | K | folds_selected | freq |")
    lines.append("|---|---|---:|---:|---:|")
    for r in stab_with_k_rows[:20]:
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
        "**15 days is tiny.** Adding 7 cross-asset features to ~13 ES intrinsic features "
        "and 5 calendar features puts the candidate-rule space at "
        f"~{len(feature_cols)} features x 2 directions x {len(K_GRID)} K-values = "
        f"~{len(feature_cols) * 2 * len(K_GRID)} candidate rules per train fold of 14 days. "
        "Per-fold selection variance dominates."
    )
    lines.append("")
    lines.append(
        "Bonferroni-style noise floor for AUC on this sample size with ~46 hypotheses is "
        "roughly 0.78. The original headline AUC of 0.848 for VIX_change_5d desc was "
        "suggestive but not conclusive on the post-hoc selection."
    )
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

    (OUT / "REPORT.md").write_text("\n".join(lines))

    regen = {
        "script": "scripts/cross_asset_forward_walk_v1.py",
        "completed_at": now,
        "runtime_s": round(runtime, 2),
        "candidate": CANDIDATE,
        "n_days": n_days,
        "n_features": len(feature_cols),
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
        "vix_change_5d_desc_freq": vix_change_5d_desc_count,
        "spx_5d_return_asc_freq": spx_5d_return_asc_count,
        "outputs": [
            "REPORT.md",
            "loo_folds.csv",
            "rule_stability.csv",
            "rule_stability_with_k.csv",
        ],
    }
    (OUT / ".regen_complete.json").write_text(json.dumps(regen, indent=2))

    print("=" * 72)
    print(f"cross_asset_forward_walk_v1 — VERDICT: {verdict}")
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
    print(f"VIX_change_5d desc freq: {vix_change_5d_desc_count}/{n_days} "
          f"({vix_change_5d_desc_count / n_days:.0%})")
    print(f"SPX_5d_return asc freq:  {spx_5d_return_asc_count}/{n_days} "
          f"({spx_5d_return_asc_count / n_days:.0%})")
    print()
    print("Top-5 (feature, direction) selection frequencies across LOO folds:")
    for r in stab_rows[:5]:
        print(f"  {r['feature']:<32} {r['direction']:<5} "
              f"{r['selected_n_folds']:>2}/{n_days}  ({r['selection_freq']:.0%})")
    print()
    print(f"Outputs written to: {OUT}")
    print(f"Runtime: {runtime:.1f}s")


if __name__ == "__main__":
    main()
