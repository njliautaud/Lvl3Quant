"""
ga_formula_pit.py — Re-run of ga_formula with the PIT (point-in-time)
fundamentals + sector-flow leadership layers merged into the feature panel.

Goal: directly test whether the new PIT layers unblock the 7 dead sectors that
collapsed in GA v2 OOT (Financials, Real Estate, ConsCyc, CommSvc, ConsDef,
Healthcare, Energy).

This module REUSES ga_formula's GA machinery (search loop, evaluator, fitness,
artifact writer). Only load_data is overridden to merge the new files, and a
wider DEFAULT_FEATURES_PIT is supplied.

Run (one sector):
    cd /home/jupiter/Lvl3Quant
    python3 -m wheel_strategy_v1.strategy.macro_picker.ga_formula_pit \
        --sector Healthcare --pop 80 --generations 40 \
        --fstore-version v2 \
        --universe-file universe_v2.parquet \
        --prices-file   prices_v2.parquet \
        --out-tag v2_pit

Run (all 7 dead sectors with the new shelf):
    python3 -m wheel_strategy_v1.strategy.macro_picker.ga_formula_pit \
        --sector dead_sectors --pop 80 --generations 40 \
        --fstore-version v2 \
        --universe-file universe_v2.parquet \
        --prices-file   prices_v2.parquet \
        --out-tag v2_pit
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

# Reuse the v1 GA machinery — only load_data + feature list change.
from wheel_strategy_v1.strategy.macro_picker.ga_formula import (  # noqa: E402
    HARD_CALMAR_FLOOR,
    build_feature_matrix,
    evaluate_formula,
    format_expression,
    ga_search,
    OUT_DIR,
    START,
    END,
    OOT_START,
)

ROOT = Path("/home/jupiter/Lvl3Quant")
WHEEL = ROOT / "wheel_strategy_v1"
CACHE = WHEEL / "data" / "cache"

# Sectors that failed GA v2 OOT (Sharpe < 0.5 on single-slice OOT).
DEAD_SECTORS = [
    "Energy", "Healthcare", "Consumer Defensive", "Communication Services",
    "Consumer Cyclical", "Real Estate", "Financial Services",
]

# Old GA v2 baseline features.
BASELINE_FEATURES = [
    "flow_sectorRet_r20", "flow_sectorRet_r60", "flow_sectorAUM_z20",
    "flow_sectorAUM_z60", "flow_sectorRel_r20", "flow_sectorRel_r60",
    "factor_momentum_load", "factor_quality_load", "factor_value_load",
    "factor_lowvol_load", "factor_momentum_rank", "factor_quality_rank",
    "fund_debtEquity_z", "fund_earningsYield_z", "fund_evEbitda_z",
    "fund_fcfMargin_z", "fund_fcfYield_z", "fund_gross_margin_z",
    "fund_net_margin_z", "fund_revScale_log_z", "fund_roicProxy_z",
]

# New PIT-fundamentals features (from data/feature_store/v2/fund_pit_features.parquet)
PIT_FUND_FEATURES = [
    "fund_pit_gross_margin_z", "fund_pit_net_margin_z", "fund_pit_ebitda_margin_z",
    "fund_pit_roe_z", "fund_pit_de_z", "fund_pit_fcf_yield_z",
    "fund_pit_current_ratio_z",
    "fund_pit_rev_growth_z", "fund_pit_eps_growth_z",
    "fund_pit_fcf_growth_z", "fund_pit_ni_growth_z",
    "fund_pit_margin_trend_z", "fund_pit_beat_rate_z",
]

# New sector-flow features (from data/feature_store/v2/sector_flow_features.parquet)
PIT_FLOW_FEATURES = [
    "flow_pit_dv_z_z", "flow_pit_dv_chg_60d_z",
    "flow_pit_rs_60d_z", "flow_pit_rs_252d_z",
    "flow_pit_px_200dma_z", "flow_pit_vol_252d_z",
    "flow_pit_corr_tlt_z", "flow_pit_corr_hyg_z",
    "flow_pit_corr_uup_z", "flow_pit_corr_gld_z",
]

DEFAULT_FEATURES_PIT = BASELINE_FEATURES + PIT_FUND_FEATURES + PIT_FLOW_FEATURES


def load_data_pit(sector: str,
                   fstore_version: str = "v2",
                   universe_file: str = "universe_v2.parquet",
                   prices_file: str = "prices_v2.parquet") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Same shape as ga_formula.load_data, but merges PIT layers too."""
    universe = pd.read_parquet(CACHE / universe_file)
    if sector != "all":
        tickers = universe[universe["sector"] == sector]["ticker"].tolist()
    else:
        tickers = universe["ticker"].tolist()
    if not tickers:
        raise ValueError(f"No tickers in sector {sector!r}. Sectors: {universe['sector'].unique().tolist()}")

    fstore_dir = ROOT / "data" / "feature_store" / fstore_version

    fund = pd.read_parquet(fstore_dir / "fund_features.parquet")
    flow = pd.read_parquet(fstore_dir / "flow_features.parquet")
    factor = pd.read_parquet(fstore_dir / "factor_features.parquet")
    regime = pd.read_parquet(fstore_dir / "regime_features.parquet")

    # NEW layers:
    fund_pit = pd.read_parquet(fstore_dir / "fund_pit_features.parquet")
    flow_pit = pd.read_parquet(fstore_dir / "sector_flow_features.parquet")

    panel = (
        fund.merge(flow, on=["ticker", "date"], how="outer")
            .merge(factor, on=["ticker", "date"], how="outer")
            .merge(fund_pit.drop(columns=["fund_asof"], errors="ignore"),
                   on=["ticker", "date"], how="outer")
            .merge(flow_pit, on=["ticker", "date"], how="outer")
    )
    panel = panel[panel["ticker"].isin(tickers)].copy()
    panel = panel[(panel["date"] >= START) & (panel["date"] < END)].copy()
    panel = panel.merge(regime, on="date", how="left")

    prices = pd.read_parquet(CACHE / prices_file)
    prices = prices[prices["ticker"].isin(tickers)].copy()
    prices["fwd_ret_5d"] = prices.groupby("ticker")["close"].pct_change(5).shift(-5)
    prices["fwd_ret_1d"] = prices.groupby("ticker")["close"].pct_change(1).shift(-1)
    prices = prices[(prices["date"] >= START) & (prices["date"] < END)][
        ["ticker", "date", "close", "fwd_ret_1d", "fwd_ret_5d"]
    ]

    if "SPY" in universe["ticker"].values:
        spy = prices[prices["ticker"] == "SPY"][["date", "close"]].copy()
    else:
        spy = prices.groupby("date")["close"].mean().reset_index()
    spy["spy_ret_1d"] = spy["close"].pct_change(1)
    spy = spy[["date", "spy_ret_1d"]]

    panel = panel.merge(prices[["ticker", "date", "fwd_ret_1d", "fwd_ret_5d"]],
                        on=["ticker", "date"], how="inner")
    panel = panel.merge(spy, on="date", how="left")
    panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)
    return panel, prices, spy


def run_sector_pit(sector: str, pop: int, gens: int, seed: int,
                    oot_start: Optional[pd.Timestamp],
                    fstore_version: str,
                    universe_file: str,
                    prices_file: str,
                    out_tag: str) -> Dict:
    print(f"\n=== GA-PIT: sector={sector} pop={pop} gens={gens} fstore={fstore_version} oot_start={oot_start} ===", flush=True)
    panel, prices, spy = load_data_pit(sector,
                                         fstore_version=fstore_version,
                                         universe_file=universe_file,
                                         prices_file=prices_file)
    print(f"  panel rows={len(panel)} tickers={panel['ticker'].nunique()} dates={panel['date'].nunique()}", flush=True)

    if oot_start is not None:
        train_panel = panel[panel["date"] < oot_start].copy()
        oot_panel = panel[panel["date"] >= oot_start].copy()
        print(f"  TRAIN: rows={len(train_panel)} dates={train_panel['date'].nunique()}", flush=True)
        print(f"  OOT  : rows={len(oot_panel)} dates={oot_panel['date'].nunique()}", flush=True)
    else:
        train_panel, oot_panel = panel, pd.DataFrame()

    Xz_train, y_train, df_train, used_features = build_feature_matrix(train_panel, DEFAULT_FEATURES_PIT)
    print(f"  train shape={Xz_train.shape}  features_used={len(used_features)} / "
          f"requested={len(DEFAULT_FEATURES_PIT)}", flush=True)

    t0 = time.time()
    best_w, best_m, hist = ga_search(Xz_train, df_train, used_features, pop=pop, gens=gens, seed=seed)
    dt = time.time() - t0
    print(f"  GA done in {dt:.1f}s", flush=True)

    expr = format_expression(best_w, used_features)
    print(f"  formula: {expr}", flush=True)
    print(f"  TRAIN: sharpe={best_m['sharpe']:+.2f} calmar={best_m['calmar']:+.2f} "
          f"cagr={best_m['cagr']*100:+.1f}% maxdd={best_m['maxdd']*100:+.1f}% "
          f"green_S={best_m['green_sharpe']:+.2f} red_S={best_m['red_sharpe']:+.2f}", flush=True)

    oot_metrics = None
    if not oot_panel.empty:
        Xz_oot, _, df_oot, oot_feats = build_feature_matrix(oot_panel, DEFAULT_FEATURES_PIT)
        if oot_feats == used_features:
            oot_metrics = evaluate_formula(best_w, Xz_oot, df_oot)
            print(f"  OOT  : sharpe={oot_metrics['sharpe']:+.2f} calmar={oot_metrics['calmar']:+.2f} "
                  f"cagr={oot_metrics['cagr']*100:+.1f}% maxdd={oot_metrics['maxdd']*100:+.1f}% "
                  f"green_S={oot_metrics['green_sharpe']:+.2f} red_S={oot_metrics['red_sharpe']:+.2f}", flush=True)
        else:
            print(f"  WARN: OOT feature set differs ({len(oot_feats)} vs {len(used_features)})", flush=True)

    artifact = {
        "sector": sector,
        "features": used_features,
        "weights": best_w.tolist(),
        "expression": expr,
        "fitness_train": best_m,
        "fitness_oot": oot_metrics,
        "trained_on": {
            "start": str(train_panel["date"].min().date()),
            "end": str(train_panel["date"].max().date()),
            "n_tickers": int(train_panel["ticker"].nunique()),
            "n_days": int(train_panel["date"].nunique()),
            "n_rows": int(len(train_panel)),
        },
        "oot_window": ({
            "start": str(oot_panel["date"].min().date()),
            "end": str(oot_panel["date"].max().date()),
            "n_days": int(oot_panel["date"].nunique()),
        } if not oot_panel.empty else None),
        "deployable": bool(oot_metrics and oot_metrics["calmar"] >= HARD_CALMAR_FLOOR
                            and oot_metrics["regime_drift"] <= 0.5
                            and oot_metrics["sharpe"] > 0.7),
        "fstore_version": fstore_version,
        "feature_shelf": "v2_pit",
        "generated_at": pd.Timestamp.now().isoformat(),
    }
    out_path = OUT_DIR / f"formula_{out_tag}_{sector.replace(' ', '_')}.json"
    with open(out_path, "w") as f:
        json.dump(artifact, f, indent=2, default=str)
    print(f"  wrote {out_path}", flush=True)
    return artifact


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sector", default="Healthcare",
                    help="Sector name, 'all', or 'dead_sectors' to sweep the GA v2 failures.")
    ap.add_argument("--pop", type=int, default=80)
    ap.add_argument("--generations", type=int, default=40)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--fstore-version", default="v2")
    ap.add_argument("--universe-file", default="universe_v2.parquet")
    ap.add_argument("--prices-file", default="prices_v2.parquet")
    ap.add_argument("--out-tag", default="v2_pit")
    ap.add_argument("--oot-start", default=str(OOT_START.date()))
    args = ap.parse_args()
    oot_start = None if args.oot_start.lower() == "none" else pd.Timestamp(args.oot_start)

    if args.sector == "dead_sectors":
        sectors = DEAD_SECTORS
    elif args.sector == "all":
        u = pd.read_parquet(CACHE / args.universe_file)
        sectors = [s for s in u["sector"].unique() if s != "ETF"]
    else:
        sectors = [args.sector]

    results = []
    for s in sectors:
        try:
            art = run_sector_pit(s, pop=args.pop, gens=args.generations, seed=args.seed,
                                  oot_start=oot_start, fstore_version=args.fstore_version,
                                  universe_file=args.universe_file, prices_file=args.prices_file,
                                  out_tag=args.out_tag)
            results.append((s, art))
        except Exception as e:
            print(f"  ERR {s}: {e}", flush=True)

    # Summary
    print("\n=== SUMMARY (PIT-augmented GA) ===")
    print(f"{'Sector':25s} {'OOT_Sharpe':>10s} {'OOT_Calmar':>10s} {'OOT_CAGR':>10s} {'OOT_DD':>10s} {'Deploy':>8s}")
    print("-" * 80)
    sum_path = OUT_DIR / f"summary_{args.out_tag}.json"
    payload = []
    for s, art in results:
        m = art.get("fitness_oot") or {}
        sh = m.get("sharpe", float("nan"))
        ca = m.get("calmar", float("nan"))
        cg = (m.get("cagr") or 0) * 100
        dd = (m.get("maxdd") or 0) * 100
        dep = "YES" if art.get("deployable") else "no"
        print(f"{s:25s} {sh:>+10.2f} {ca:>+10.2f} {cg:>+9.1f}% {dd:>+9.1f}% {dep:>8s}")
        payload.append({"sector": s, "fitness_oot": m, "deployable": art.get("deployable")})
    with open(sum_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nWrote {sum_path}")


if __name__ == "__main__":
    main()
