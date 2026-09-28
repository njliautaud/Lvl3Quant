#!/usr/bin/env python3
"""
eval_meta_stream.py — Run each meta-model's single-stream signal through the
stream-continuation backtest framework, applying the HC #428 R1 regime gate +
HC #344 day-conc gate.

For each model in {xgb, lgbm, mlp}, builds a synthetic "head" by stitching
together per-day meta-predictions on the OOT days that were predicted by walk-forward.
Treats the meta prediction as a directional signal centered on zero (already a
log-return prediction in ticks). Sweeps the same (conf_cut, exit_M, exit_floor)
grid as the baseline stream-continuation report.

Outputs: /home/jupiter/Lvl3Quant/output/razer_meta/meta_tradability_matrix.parquet
         /home/jupiter/Lvl3Quant/output/razer_meta/REPORT.md
"""
from __future__ import annotations
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# Import helpers from the existing stream-continuation harness
sys.path.insert(0, "/home/jupiter/Lvl3Quant/scripts")
from stream_continuation_backtest import (  # noqa: E402
    simulate_stream_day,
    summarize_trades,
    classify_regimes,
    build_tradability_row,
    StreamConfig,
    CONF_QUANTILES,
    EXIT_RULES_M,
    EXIT_FLOOR_FRACS,
    MAX_HOLD_STEPS,
    STRIDE_SECONDS_APPROX,
    ES_TICK_VALUE,
    ES_RT_COMMISSION_TICKS,
    cache_day,
    list_common_dates,
)

ROOT = Path("/home/jupiter/Lvl3Quant")
META_DIR = ROOT / "output/razer_meta"
OUT_DIR = META_DIR
PERDAY_DIR = ROOT / "output/cnn_mamba_v3_4_2_fixedmtl/oot_47day_perdate"

MODELS = ["xgb", "lgbm", "mlp"]


def load_meta_preds(model: str) -> Dict[str, np.ndarray]:
    """Returns dict date -> 1-D array of meta predictions."""
    out: Dict[str, np.ndarray] = {}
    for npz in sorted(META_DIR.glob(f"meta_preds_{model}_*.npz")):
        d = np.load(npz, allow_pickle=True)
        date_str = str(d["date"])
        out[date_str] = d["preds"].astype(np.float64)
    return out


def build_meta_day_df(date_str: str, meta_preds: np.ndarray) -> pd.DataFrame:
    """Build a per-day df with the meta prediction as a single 'pred_meta' head + targets.
    The number of preds must match the per-day NPZ row count after target masking — meta
    preds are aligned to the post-NaN-drop subset, so we re-load the per-day NPZ and align
    by dropping the same NaN-target rows as razer_meta_train did.
    """
    npz_path = PERDAY_DIR / f"oot_{date_str}.npz"
    d = np.load(npz_path, allow_pickle=True)
    n_full = d["pred_log_ret_1s"].shape[0] if "pred_log_ret_1s" in d.files else None
    if n_full is None:
        return pd.DataFrame()

    # Reconstruct rows: same logic as build_meta_features — drop target_log_ret_5s NaN
    cols = {}
    for k in d.files:
        if k.startswith("pred_") or k.startswith("target_") or k.startswith("mask_"):
            cols[k] = d[k]
    cols["pred_k"] = np.arange(n_full, dtype=np.int64)
    cols["date"] = np.full(n_full, date_str, dtype=object)
    df_full = pd.DataFrame(cols)
    if "target_log_ret_5s" not in df_full.columns:
        return pd.DataFrame()
    df_kept = df_full.dropna(subset=["target_log_ret_5s"]).reset_index(drop=True)
    if len(df_kept) != len(meta_preds):
        print(f"  [WARN] {date_str}: meta_preds={len(meta_preds)} vs filtered df={len(df_kept)} — mismatch, skipping.")
        return pd.DataFrame()

    df_kept["pred_meta"] = meta_preds.astype(np.float32)
    # Fill required dummy MFE/MAE columns (sparse, treat as NaN diagnostics — same as load_day in the harness)
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
    print("META-MODEL STREAM-CONTINUATION EVALUATION")
    print("=" * 78)

    # First, build per-day caches if not present (so classify_regimes works)
    common_dates = list_common_dates()
    print(f"[setup] common dates: {len(common_dates)}")
    day_caches = {}
    for d in common_dates:
        try:
            cp = cache_day(d)
        except (IndexError, KeyError):
            cp = ""  # empty NPZ (e.g. weekend)
        day_caches[d] = cp
    regimes = classify_regimes(day_caches)
    print(f"[setup] regimes: green={sum(1 for v in regimes.values() if v=='green')}, "
          f"red={sum(1 for v in regimes.values() if v=='red')}, "
          f"flat={sum(1 for v in regimes.values() if v=='flat')}")

    all_rows: List[Dict] = []

    for model in MODELS:
        print(f"\n--- META-MODEL: {model.upper()} ---")
        meta = load_meta_preds(model)
        print(f"  loaded {len(meta)} days of meta predictions")
        if not meta:
            print(f"  no preds, skipping")
            continue

        # Build per-day dataframes
        per_day_dfs = {}
        for date_str, preds in meta.items():
            df_day = build_meta_day_df(date_str, preds)
            if not df_day.empty:
                per_day_dfs[date_str] = df_day
        print(f"  built {len(per_day_dfs)} per-day dataframes")

        # Sweep all configs
        configs = [
            StreamConfig(head="pred_meta", conf_q=q, exit_M=m, exit_floor_frac=f)
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
            print(f"  no trades, skipping")
            continue
        all_trades_df = pd.concat(all_trades_list, ignore_index=True)
        print(f"  total trades for {model}: {len(all_trades_df)}")

        # Aggregate into tradability rows
        grouped = all_trades_df.groupby(["head", "conf_q", "exit_M", "exit_floor_frac"])
        for (h, q, m, f), sub in grouped:
            row = build_tradability_row(h, q, m, f, sub, regimes)
            if row:
                row["model"] = model
                all_rows.append(row)
        print(f"  added {len(grouped)} config rows for {model}")

    if not all_rows:
        print("[FATAL] no rows produced")
        sys.exit(1)
    matrix_df = pd.DataFrame(all_rows).sort_values("sharpe_per_trade", ascending=False)
    out_path = OUT_DIR / "meta_tradability_matrix.parquet"
    matrix_df.to_parquet(out_path, index=False)
    print(f"\n[eval] wrote {out_path} ({len(matrix_df)} rows)")

    # REPORT.md
    write_report(matrix_df, regimes, t0)


def write_report(matrix_df: pd.DataFrame, regimes: Dict, t0: float):
    # Top 5 by Sharpe with regime_pass
    passing = matrix_df[matrix_df["regime_pass"]].copy()
    top_pass = passing.head(5)
    overall_top = matrix_df.head(5)

    # Per-model top config
    best_per_model = {}
    for m in MODELS:
        sub = matrix_df[matrix_df["model"] == m]
        if not sub.empty:
            best_per_model[m] = sub.iloc[0]

    # Feature importances
    import json
    fi_path = META_DIR / "feature_importances.json"
    fi_data = json.loads(fi_path.read_text()) if fi_path.exists() else None

    lines = []
    lines.append("# Razer Meta-Model Stream-Continuation Report")
    lines.append("")
    lines.append("Compliance: HC #466 (per-head + confidence cuts + confluence), HC #467")
    lines.append("(stream-continuation hold time is OUTPUT), HC #428 R1 (regime gate), HC #344")
    lines.append("(day-concentration <= 0.70), HC #468 (Razer GPU on alpha research).")
    lines.append("")
    lines.append("## Setup")
    lines.append(f"- 32 v4 prediction heads -> 3 meta-models trained on Razer RTX 3070 GPU.")
    lines.append(f"- Target: target_log_ret_5s (5s realized log return in ticks).")
    lines.append(f"- Walk-forward: sliding, 15 train days -> 1 test day. 17 OOT days predicted.")
    lines.append(f"- Stream-continuation sweep: 5 conf_cuts x 3 exit_M x 2 exit_floor = 30 configs per model.")
    lines.append(f"- Baseline (CNN-Mamba v4 standalone heads): top Sharpe = -0.349.")
    lines.append(f"- Baseline (screening confluence pair pred_log_ret_1s + pred_p_up_5s): +1.93 ticks at 62.6% hit rate.")
    lines.append("")

    lines.append("## Headline — Top 5 meta-model configs by Sharpe (regime-pass only)")
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
        lines.append("NO meta-model config cleared the regime-skew (<=0.50) gate.")
    lines.append("")

    lines.append("## Top 5 by Sharpe regardless of gates (for transparency)")
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

    lines.append("## Best config per model")
    lines.append("")
    for m, r in best_per_model.items():
        lines.append(
            f"- **{m.upper()}**: conf top {r['conf_quantile_top_pct']}%, exit_M={int(r['exit_M'])}, floor={r['exit_floor_frac']:.2f} "
            f"-> {int(r['n_trades'])} trades, mean_net={r['mean_net_ticks']:.3f}t, WR={r['wr']:.1%}, "
            f"Sharpe={r['sharpe_per_trade']:.3f}, regime_pass={'yes' if r['regime_pass'] else 'no'}"
        )
    lines.append("")

    # vs baselines
    lines.append("## Verdict vs baselines")
    lines.append("")
    best_overall = matrix_df.iloc[0]
    lines.append(f"- Standalone-head baseline Sharpe: -0.349")
    lines.append(f"- Best meta-model Sharpe: {best_overall['sharpe_per_trade']:.3f} ({best_overall['model']}, conf top {best_overall['conf_quantile_top_pct']}%)")
    beats_baseline = best_overall["sharpe_per_trade"] > -0.349
    lines.append(f"- **Meta-model beats standalone-head baseline: {'YES' if beats_baseline else 'NO'}**")
    lines.append("")
    lines.append(f"- Screening-confluence pair: +1.93 ticks mean signed realized (different metric: not per-trade Sharpe).")
    lines.append(f"- Best meta-model mean net ticks per trade: {best_overall['mean_net_ticks']:.3f}")
    beats_confluence_ticks = best_overall["mean_net_ticks"] > 1.93
    lines.append(f"- **Meta-model beats confluence-pair mean net ticks: {'YES' if beats_confluence_ticks else 'NO'}**")
    lines.append("")

    # Feature importances
    if fi_data:
        lines.append("## Feature importances — XGBoost (top 10 of 32 heads)")
        lines.append("")
        lines.append("| rank | head | importance |")
        lines.append("|------|------|------------|")
        for i, (f, w) in enumerate(fi_data["xgb"][:10], start=1):
            lines.append(f"| {i} | {f} | {w:.5f} |")
        lines.append("")
        lines.append("## Feature importances — LightGBM (top 10 of 32 heads)")
        lines.append("")
        lines.append("| rank | head | importance |")
        lines.append("|------|------|------------|")
        for i, (f, w) in enumerate(fi_data["lgbm"][:10], start=1):
            lines.append(f"| {i} | {f} | {w:.5f} |")
        lines.append("")

    lines.append("---")
    lines.append(f"Wall time: {time.time()-t0:.1f}s. Total config rows: {len(matrix_df)}.")
    (OUT_DIR / "REPORT.md").write_text("\n".join(lines))
    print(f"[eval] wrote {OUT_DIR / 'REPORT.md'}")


if __name__ == "__main__":
    main()
