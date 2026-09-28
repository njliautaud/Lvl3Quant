#!/usr/bin/env python3
"""
eval_classifier_stream.py — Run each Razer-trained classifier through the
stream-continuation backtest framework.

For each model in {xgb, mlp}, builds a synthetic "head" by combining:
  signed_signal = prob * sign(pred_log_ret_1s)
where prob is the classifier's predicted P(profitable_trigger) and the
direction is taken from the v4 1s head (the classifier is direction-agnostic).

This way |signal| = probability (confidence cuts work as-is on quantile),
and sign(signal) = direction (compatible with simulate_stream_day).

Sweeps the same (conf_cut, exit_M, exit_floor) grid as the baseline.

Outputs:
  output/razer_classifier/classifier_tradability_matrix.parquet
  output/razer_classifier/REPORT.md
"""
from __future__ import annotations
import sys
import time
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/jupiter/Lvl3Quant/scripts")
from stream_continuation_backtest import (  # noqa: E402
    simulate_stream_day,
    classify_regimes,
    build_tradability_row,
    StreamConfig,
    CONF_QUANTILES,
    EXIT_RULES_M,
    EXIT_FLOOR_FRACS,
    cache_day,
    list_common_dates,
)

ROOT = Path("/home/jupiter/Lvl3Quant")
CLS_DIR = ROOT / "output/razer_classifier"
PERDAY_DIR = ROOT / "output/cnn_mamba_v3_4_2_fixedmtl/oot_47day_perdate"

MODELS = ["xgb", "mlp"]


def load_cls_preds(model: str) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for npz in sorted(CLS_DIR.glob(f"cls_preds_{model}_*.npz")):
        d = np.load(npz, allow_pickle=True)
        date_str = str(d["date"])
        out[date_str] = d["prob"].astype(np.float64)
    return out


def build_cls_day_df(date_str: str, cls_probs: np.ndarray) -> pd.DataFrame:
    """Per-day df with synthetic 'pred_cls' = prob * sign(pred_log_ret_1s)."""
    npz_path = PERDAY_DIR / f"oot_{date_str}.npz"
    if not npz_path.exists():
        return pd.DataFrame()
    d = np.load(npz_path, allow_pickle=True)
    if "pred_log_ret_1s" not in d.files or "target_log_ret_5s" not in d.files:
        return pd.DataFrame()
    n_full = d["pred_log_ret_1s"].shape[0]

    cols = {}
    for k in d.files:
        if k.startswith("pred_") or k.startswith("target_") or k.startswith("mask_"):
            cols[k] = d[k]
    cols["pred_k"] = np.arange(n_full, dtype=np.int64)
    cols["date"] = np.full(n_full, date_str, dtype=object)
    df_full = pd.DataFrame(cols)
    df_kept = df_full.dropna(subset=["target_log_ret_5s"]).reset_index(drop=True)
    if len(df_kept) != len(cls_probs):
        print(f"  [WARN] {date_str}: cls_probs={len(cls_probs)} vs filtered df={len(df_kept)} — skip")
        return pd.DataFrame()

    # Build synthetic signed signal: prob * sign(1s head)
    dir_sign = np.sign(df_kept["pred_log_ret_1s"].values)
    # Where 1s head is 0 (extremely rare), break ties with 5s p_up sign
    if "pred_p_up_5s" in df_kept.columns:
        fallback = np.sign(df_kept["pred_p_up_5s"].values - 0.5)
        dir_sign = np.where(dir_sign == 0, fallback, dir_sign)
    df_kept["pred_cls"] = (cls_probs * dir_sign).astype(np.float32)

    # Dummy MFE/MAE columns (NaN — matches load_day behavior in harness)
    for h in ("1s", "5s", "10s", "30s"):
        if f"mfe_h{h}" not in df_kept.columns:
            df_kept[f"mfe_h{h}"] = np.nan
            df_kept[f"mae_h{h}"] = np.nan
            df_kept[f"time_to_mfe_h{h}"] = np.nan
    if "mid_ticks" not in df_kept.columns:
        df_kept["mid_ticks"] = np.nan
    if "ts_ns" not in df_kept.columns:
        df_kept["ts_ns"] = 0
    return df_kept


def main():
    t0 = time.time()
    print("=" * 78)
    print("CLASSIFIER STREAM-CONTINUATION EVALUATION")
    print("=" * 78)

    common_dates = list_common_dates()
    print(f"[setup] common dates: {len(common_dates)}")
    day_caches = {}
    for d in common_dates:
        try:
            cp = cache_day(d)
        except Exception:
            cp = ""
        day_caches[d] = cp
    regimes = classify_regimes(day_caches)
    print(f"[setup] regimes: green={sum(1 for v in regimes.values() if v=='green')}, "
          f"red={sum(1 for v in regimes.values() if v=='red')}, "
          f"flat={sum(1 for v in regimes.values() if v=='flat')}")

    all_rows: List[Dict] = []

    for model in MODELS:
        print(f"\n--- CLASSIFIER: {model.upper()} ---")
        probs = load_cls_preds(model)
        print(f"  loaded {len(probs)} days of classifier probs")
        if not probs:
            continue

        # Quick label-quality diagnostics
        per_day_dfs = {}
        for date_str, p in probs.items():
            df_day = build_cls_day_df(date_str, p)
            if not df_day.empty:
                per_day_dfs[date_str] = df_day
        print(f"  built {len(per_day_dfs)} per-day dataframes")

        configs = [
            StreamConfig(head="pred_cls", conf_q=q, exit_M=m, exit_floor_frac=f)
            for q in CONF_QUANTILES
            for m in EXIT_RULES_M
            for f in EXIT_FLOOR_FRACS
        ]
        print(f"  {len(configs)} configs over {len(per_day_dfs)} days")

        all_trades_list = []
        for date_str, df_day in per_day_dfs.items():
            for cfg in configs:
                tdf = simulate_stream_day(df_day, cfg)
                if not tdf.empty:
                    tdf["model"] = model
                    all_trades_list.append(tdf)
        if not all_trades_list:
            continue
        all_trades_df = pd.concat(all_trades_list, ignore_index=True)
        print(f"  total trades for {model}: {len(all_trades_df)}")

        grouped = all_trades_df.groupby(["head", "conf_q", "exit_M", "exit_floor_frac"])
        for (h, q, m, f), sub in grouped:
            row = build_tradability_row(h, q, m, f, sub, regimes)
            if row:
                row["model"] = model
                all_rows.append(row)
        print(f"  added {len(grouped)} config rows for {model}")

    if not all_rows:
        print("[FATAL] no rows")
        sys.exit(1)
    matrix_df = pd.DataFrame(all_rows).sort_values("sharpe_per_trade", ascending=False)
    out_path = CLS_DIR / "classifier_tradability_matrix.parquet"
    matrix_df.to_parquet(out_path, index=False)
    print(f"\n[eval] wrote {out_path} ({len(matrix_df)} rows)")

    write_report(matrix_df, t0)


def write_report(matrix_df: pd.DataFrame, t0: float):
    passing = matrix_df[matrix_df["regime_pass"]].copy()
    top_pass = passing.head(5)
    overall_top = matrix_df.head(5)
    best_per_model = {}
    for m in MODELS:
        sub = matrix_df[matrix_df["model"] == m]
        if not sub.empty:
            best_per_model[m] = sub.iloc[0]

    fi_path = CLS_DIR / "cls_feature_importances.json"
    fi_data = json.loads(fi_path.read_text()) if fi_path.exists() else None

    lines = []
    lines.append("# Razer Classifier Stream-Continuation Report")
    lines.append("")
    lines.append("Compliance: HC #466 (full-output utilization), HC #467 (stream-continuation),")
    lines.append("HC #428 R1 (regime gate), HC #344 (day-conc cap), HC #468 (Razer GPU on alpha).")
    lines.append("")
    lines.append("## Setup")
    lines.append("- Target: y_profitable_trigger = (any of 4 confluence-pair triggers) AND realized 5s move > 0.5 ticks in predicted direction.")
    lines.append("- Class balance: 25,375 positives out of 1,578,006 events (1.61%).")
    lines.append("- 32 v4 prediction heads -> 2 classifiers (XGBoost-GPU, PyTorch-MLP) trained on Razer RTX 3070.")
    lines.append("- Walk-forward: 15-day-train / 1-day-test sliding, 17 OOT folds.")
    lines.append("- Direction: sign(pred_log_ret_1s). Confidence: classifier probability.")
    lines.append("- Stream-continuation sweep: 5 conf_cuts x 3 exit_M x 2 exit_floor = 30 configs per model.")
    lines.append("- Baselines: standalone-head Sharpe -0.349; meta-regression MLP Sharpe -0.084.")
    lines.append("")

    lines.append("## Headline — Top 5 classifier configs by Sharpe (regime-pass only)")
    lines.append("")
    if len(top_pass) > 0:
        lines.append("| model | conf cut | exit_M | floor | n_trades | mean hold (s) | WR | mean net ticks | Sharpe | regime_skew | day_conc |")
        lines.append("|-------|----------|--------|-------|----------|---------------|----|-----------------|--------|-------------|----------|")
        for _, r in top_pass.iterrows():
            lines.append(
                f"| {r['model']} | top {r['conf_quantile_top_pct']}% | {int(r['exit_M'])} | {r['exit_floor_frac']:.2f} | "
                f"{int(r['n_trades'])} | {r['mean_hold_s']:.1f} | {r['wr']:.1%} | {r['mean_net_ticks']:.3f} | "
                f"{r['sharpe_per_trade']:.3f} | {r['regime_skew']:.2f} | {r['day_concentration']:.2f} |"
            )
    else:
        lines.append("NO classifier config cleared the regime-skew (<=0.50) gate.")
    lines.append("")

    lines.append("## Top 5 overall by Sharpe (gates shown)")
    lines.append("")
    lines.append("| model | conf cut | exit_M | floor | n_trades | WR | mean net ticks | Sharpe | regime_pass | day_conc_pass |")
    lines.append("|-------|----------|--------|-------|----------|----|-----------------|--------|--------------|---------------|")
    for _, r in overall_top.iterrows():
        lines.append(
            f"| {r['model']} | top {r['conf_quantile_top_pct']}% | {int(r['exit_M'])} | {r['exit_floor_frac']:.2f} | "
            f"{int(r['n_trades'])} | {r['wr']:.1%} | {r['mean_net_ticks']:.3f} | {r['sharpe_per_trade']:.3f} | "
            f"{'yes' if r['regime_pass'] else 'no'} | {'yes' if r['day_conc_pass'] else 'no'} |"
        )
    lines.append("")

    lines.append("## Best config per classifier")
    lines.append("")
    for m, r in best_per_model.items():
        lines.append(
            f"- **{m.upper()}**: conf top {r['conf_quantile_top_pct']}%, exit_M={int(r['exit_M'])}, floor={r['exit_floor_frac']:.2f} "
            f"-> {int(r['n_trades'])} trades, mean_net={r['mean_net_ticks']:.3f}t, WR={r['wr']:.1%}, "
            f"Sharpe={r['sharpe_per_trade']:.3f}, regime_pass={'yes' if r['regime_pass'] else 'no'}"
        )
    lines.append("")

    lines.append("## Verdict vs baselines")
    lines.append("")
    best_overall = matrix_df.iloc[0]
    lines.append(f"- Standalone-head baseline Sharpe: -0.349")
    lines.append(f"- Meta-REGRESSION MLP best Sharpe: -0.084")
    lines.append(f"- Best classifier Sharpe: {best_overall['sharpe_per_trade']:.3f} ({best_overall['model']}, conf top {best_overall['conf_quantile_top_pct']}%)")
    lines.append(f"- **Classifier beats standalone-head baseline: {'YES' if best_overall['sharpe_per_trade'] > -0.349 else 'NO'}**")
    lines.append(f"- **Classifier beats meta-regression baseline: {'YES' if best_overall['sharpe_per_trade'] > -0.084 else 'NO'}**")
    lines.append(f"- **Classifier reaches positive Sharpe: {'YES' if best_overall['sharpe_per_trade'] > 0 else 'NO'}**")
    lines.append("")

    if fi_data and "xgb" in fi_data:
        items = sorted(fi_data["xgb"].items(), key=lambda kv: -kv[1])
        lines.append("## Feature importances — XGBoost Classifier (top 10 of 32 heads)")
        lines.append("")
        lines.append("| rank | head | importance |")
        lines.append("|------|------|------------|")
        for i, (f, w) in enumerate(items[:10], start=1):
            lines.append(f"| {i} | {f} | {w:.5f} |")
        lines.append("")

    lines.append("---")
    lines.append(f"Wall time: {time.time()-t0:.1f}s. Total config rows: {len(matrix_df)}.")
    (CLS_DIR / "REPORT.md").write_text("\n".join(lines))
    print(f"[eval] wrote {CLS_DIR / 'REPORT.md'}")


if __name__ == "__main__":
    main()
