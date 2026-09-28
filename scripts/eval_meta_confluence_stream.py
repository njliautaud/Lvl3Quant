#!/usr/bin/env python3
"""
eval_meta_confluence_stream.py — Run the just-trained meta_confl classifiers
(razer_meta_confluence_train.py outputs) through the stream-continuation
backtest framework. Thin wrapper around eval_classifier_stream.

Mirrors eval_classifier_stream.main() exactly but overrides:
  - CLS_DIR -> output/razer_meta_confluence
  - npz glob -> meta_confl_preds_<model>_*.npz
  - output -> meta_confluence_tradability_matrix.parquet + REPORT.md
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
import eval_classifier_stream as ecs
from stream_continuation_backtest import (  # noqa: E402
    simulate_stream_day,
    StreamConfig,
    CONF_QUANTILES,
    EXIT_RULES_M,
    EXIT_FLOOR_FRACS,
    build_tradability_row,
)

ROOT = Path("/home/jupiter/Lvl3Quant")
META_CONFL_DIR = ROOT / "output/razer_meta_confluence"
MODELS = ["xgb", "mlp"]


def load_meta_confl_preds(model: str) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for npz in sorted(META_CONFL_DIR.glob(f"meta_confl_preds_{model}_*.npz")):
        d = np.load(npz, allow_pickle=True)
        date_str = str(d["date"])
        out[date_str] = d["prob"].astype(np.float64)
    return out


def main():
    t0 = time.time()
    print("=" * 78)
    print("META-CONFLUENCE STREAM-CONTINUATION EVALUATION")
    print("=" * 78)

    common_dates = ecs.list_common_dates()
    print(f"[setup] common dates: {len(common_dates)}")
    day_caches = {}
    for d in common_dates:
        try:
            cp = ecs.cache_day(d)
        except Exception:
            cp = ""
        day_caches[d] = cp
    regimes = ecs.classify_regimes(day_caches)
    print(
        f"[setup] regimes: green={sum(1 for v in regimes.values() if v=='green')}, "
        f"red={sum(1 for v in regimes.values() if v=='red')}, "
        f"flat={sum(1 for v in regimes.values() if v=='flat')}"
    )

    all_rows: List[Dict] = []

    for model in MODELS:
        print(f"\n--- META-CONFLUENCE: {model.upper()} ---")
        probs = load_meta_confl_preds(model)
        print(f"  loaded {len(probs)} days of meta_confl probs")
        if not probs:
            continue

        per_day_dfs = {}
        for date_str, p in probs.items():
            df_day = ecs.build_cls_day_df(date_str, p)
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
    out_path = META_CONFL_DIR / "meta_confluence_tradability_matrix.parquet"
    matrix_df.to_parquet(out_path, index=False)
    print(f"\n[eval] wrote {out_path} ({len(matrix_df)} rows)")

    # Compact REPORT.md
    passing = matrix_df[matrix_df["regime_pass"]].copy() if "regime_pass" in matrix_df.columns else matrix_df
    if "day_concentration" in passing.columns:
        passing = passing[passing["day_concentration"] <= 0.70]
    top_pass = passing.head(10)
    rpt = META_CONFL_DIR / "REPORT.md"
    with open(rpt, "w") as f:
        f.write("# Meta-Confluence Stream-Continuation Report\n\n")
        f.write("Compliance: HC #466 + #467 + #428 R1 + #344 + #468 + #469 R2 (5-OOT smoke until 40-day rerun).\n\n")
        f.write("## Setup\n")
        f.write("- Source: razer_meta_confluence_train.py — 32 v4 heads + 88 pair-confluence indicators (binary + signed for top-50 robust pairs).\n")
        f.write(f"- Models: {MODELS}.  Walk-forward: 15-day-train / 1-day-test sliding, 17 OOT folds.\n")
        f.write(f"- Eval rows total: {len(matrix_df)}.  Rows passing regime + day_conc≤0.70: {len(passing)}.\n\n")
        f.write("## Top 10 by Sharpe-per-trade (regime-pass, day_conc≤0.70)\n\n")
        if len(top_pass) > 0:
            f.write("| model | conf cut | exit_M | floor | n_trades | mean hold (s) | WR | mean net ticks | Sharpe | regime_skew | day_conc |\n")
            f.write("|-------|----------|--------|-------|----------|---------------|-----|-----------------|--------|-------------|----------|\n")
            for _, r in top_pass.iterrows():
                f.write(
                    f"| {r.get('model','?')} | top {r.get('conf_quantile_top_pct','?')}% | {int(r.get('exit_M',0))} | "
                    f"{r.get('exit_floor_frac',0):.2f} | {int(r.get('n_trades',0))} | "
                    f"{r.get('mean_hold_s',float('nan')):.1f} | {r.get('wr',float('nan')):.1%} | "
                    f"{r.get('mean_net_ticks',float('nan')):.3f} | {r.get('sharpe_per_trade',float('nan')):.3f} | "
                    f"{r.get('regime_skew',float('nan')):.2f} | {r.get('day_concentration',float('nan')):.2f} |\n"
                )
        else:
            f.write("(none survived gates)\n")
    print(f"[eval] report -> {rpt}")
    (META_CONFL_DIR / "meta_confluence_eval.DONE").touch()


if __name__ == "__main__":
    main()
