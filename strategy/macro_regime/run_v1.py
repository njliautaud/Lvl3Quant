"""Main runner — macro_regime_rotation_v1.

Builds regime classifier, runs regime-conditioned rotation + baseline,
computes all gates, writes report.md + MLflow log.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(ROOT / "strategy"))
sys.path.insert(0, str(ROOT))

from macro_regime.regime_classifier import (
    build_feature_panel, classify_regime, regime_transitions,
)
from macro_regime.regime_rotation_backtest import (
    UNIVERSE, REGIME_TILTS, BASELINE_UNIVERSE,
    load_prices, compute_returns, run_strategy,
    perf_metrics, hc428_r1_check, day_concentration, perf_per_regime,
)


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / f"output/macro_regime/regime_rotation_v1_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    findings_dir = ROOT / "research/findings"
    findings_dir.mkdir(parents=True, exist_ok=True)

    print(f"[run_v1] output dir: {out_dir}")
    cache_dir = ROOT / "data/cache/regime_macro"
    print("[run_v1] building macro feature panel...")
    panel = build_feature_panel(start="2017-01-01", cache_dir=cache_dir)
    panel = panel.dropna(subset=["vix_close", "hy_oas_proxy", "dxy_close"]).copy()
    print(f"[run_v1] panel rows={len(panel)} dates={panel.index.min().date()}->{panel.index.max().date()}")

    print("[run_v1] classifying regime...")
    regime = classify_regime(panel)
    trans = regime_transitions(regime)
    reg_counts = regime.value_counts().to_dict()
    print(f"[run_v1] regime counts: {reg_counts}")
    print(f"[run_v1] n transitions: {len(trans)}")

    # Save regime timeline
    regime.to_frame("regime").to_parquet(out_dir / "regime_timeline.parquet")
    trans.to_parquet(out_dir / "regime_transitions.parquet")

    print("[run_v1] loading prices...")
    prices = load_prices()
    returns = compute_returns(prices)

    # SPY benchmark
    spy_ret = returns["SPY"]

    # === Regime-conditioned rotation ===
    print("[run_v1] running regime-conditioned rotation...")
    res = run_strategy(returns.drop(columns=["SPY"]), regime,
                       start="2018-01-01", end="2025-12-31",
                       rebalance_n=5, use_regime=True, prices=prices)
    book_r = res["daily_ret"]

    # === Baseline (unconditional momentum top-3 of 11 sectors) ===
    print("[run_v1] running unconditional baseline...")
    base = run_strategy(returns.drop(columns=["SPY"]), regime,
                        start="2018-01-01", end="2025-12-31",
                        rebalance_n=5, use_regime=False, prices=prices)
    base_r = base["daily_ret"]

    # SPY buy-and-hold
    spy_bh = spy_ret.loc[book_r.index]

    # Performance
    m_strat = perf_metrics(book_r)
    m_base = perf_metrics(base_r)
    m_spy = perf_metrics(spy_bh)

    hc428_strat = hc428_r1_check(book_r, spy_bh)
    hc428_base = hc428_r1_check(base_r, spy_bh)

    dc_strat = day_concentration(book_r)
    per_reg = perf_per_regime(book_r, regime)

    n_years = m_strat["years"]
    transitions_per_year = len(trans) / max(n_years, 1e-9)

    # Spot checks — regime calls
    spot_checks = {
        "covid_mar2020": dict(regime.loc["2020-03-01":"2020-04-15"].value_counts()),
        "bear_2022": dict(regime.loc["2022-01-01":"2022-12-31"].value_counts()),
        "mid_2023_onward": dict(regime.loc["2023-07-01":"2024-12-31"].value_counts()),
    }

    summary = {
        "config": {
            "universe": UNIVERSE,
            "regime_tilts": {k: v for k, v in REGIME_TILTS.items()},
            "backtest_window": ["2018-01-01", "2025-12-31"],
            "rebalance_n_days": 5,
            "anchor_capital": 20_000,
            "slippage_bps": 1.0,
            "commission_per_share": 0.005,
        },
        "regime_classifier": {
            "n_transitions": int(len(trans)),
            "transitions_per_year": float(transitions_per_year),
            "regime_counts": {str(k): int(v) for k, v in reg_counts.items()},
            "spot_checks": {k: {str(kk): int(vv) for kk, vv in v.items()}
                            for k, v in spot_checks.items()},
        },
        "strategy_perf": m_strat,
        "baseline_perf": m_base,
        "spy_bh_perf": m_spy,
        "hc428_r1_strategy": hc428_strat,
        "hc428_r1_baseline": hc428_base,
        "day_concentration_strategy": dc_strat,
        "per_regime_perf_strategy": per_reg,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    # Persist daily returns
    pd.DataFrame({
        "date": book_r.index,
        "strategy_ret": book_r.values,
        "baseline_ret": base_r.values,
        "spy_ret": spy_bh.values,
        "regime": regime.reindex(book_r.index, method="ffill").values,
    }).to_parquet(out_dir / "daily_returns.parquet", index=False)

    pd.DataFrame(res["holdings_log"]).to_parquet(out_dir / "holdings_log.parquet")

    # MLflow logging
    try:
        import mlflow
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
        mlflow.set_experiment("macro_regime_rotation_v1")
        with mlflow.start_run(run_name=f"regime_rotation_{ts}"):
            mlflow.log_params(summary["config"])
            mlflow.log_metrics({
                "cagr": m_strat["cagr"],
                "sharpe": m_strat["sharpe"],
                "sortino": m_strat["sortino"],
                "max_dd": m_strat["max_dd"],
                "calmar": m_strat["calmar"],
                "pf": m_strat["pf"],
                "wr": m_strat["wr"],
                "hc428_gap": hc428_strat["gap"],
                "hc428_pass": int(hc428_strat["pass_R1"]),
                "day_conc": dc_strat,
                "transitions_per_yr": transitions_per_year,
                "spy_sharpe": m_spy["sharpe"],
                "spy_cagr": m_spy["cagr"],
            })
            mlflow.log_artifacts(str(out_dir))
        with mlflow.start_run(run_name=f"baseline_unconditional_{ts}"):
            mlflow.log_params({"variant": "unconditional_baseline",
                               "universe": BASELINE_UNIVERSE})
            mlflow.log_metrics({
                "cagr": m_base["cagr"],
                "sharpe": m_base["sharpe"],
                "max_dd": m_base["max_dd"],
                "pf": m_base["pf"],
                "wr": m_base["wr"],
                "hc428_gap": hc428_base["gap"],
                "hc428_pass": int(hc428_base["pass_R1"]),
            })
        print("[run_v1] mlflow logged")
    except Exception as e:
        print(f"[run_v1] mlflow log failed: {e}")

    # === Report ===
    md = []
    md.append("# macro_regime_rotation_v1 — regime-conditioned sector rotation")
    md.append(f"_Run timestamp: {ts}_")
    md.append("")
    md.append("## TL;DR")
    md.append(f"- **Strategy CAGR**: {m_strat['cagr']*100:.1f}% | **Sharpe**: {m_strat['sharpe']:.2f} | "
              f"**Sortino**: {m_strat['sortino']:.2f} | **MaxDD**: {m_strat['max_dd']*100:.1f}% | "
              f"**Calmar**: {m_strat['calmar']:.2f} | **PF**: {m_strat['pf']:.2f} | "
              f"**WR**: {m_strat['wr']*100:.1f}%")
    md.append(f"- **SPY B&H**: CAGR {m_spy['cagr']*100:.1f}% | Sharpe {m_spy['sharpe']:.2f} | "
              f"MaxDD {m_spy['max_dd']*100:.1f}%")
    md.append(f"- **Unconditional baseline (no regime gating)**: CAGR {m_base['cagr']*100:.1f}% | "
              f"Sharpe {m_base['sharpe']:.2f} | MaxDD {m_base['max_dd']*100:.1f}%")
    md.append("")
    md.append("## HC #428 R1 — Regime-Agnostic Gate")
    md.append(f"- **Green-day Sharpe**: {hc428_strat['green']['sharpe']:.2f} (n={hc428_strat['green']['n']})")
    md.append(f"- **Red-day Sharpe**:   {hc428_strat['red']['sharpe']:.2f} (n={hc428_strat['red']['n']})")
    md.append(f"- **Flat-day Sharpe**:  {hc428_strat['flat']['sharpe']:.2f} (n={hc428_strat['flat']['n']})")
    md.append(f"- **Gap = |Sh_g - Sh_r| / max = {hc428_strat['gap']:.3f}** "
              f"(need ≤ 0.50)")
    md.append(f"- **HC #428 R1**: {'PASS' if hc428_strat['pass_R1'] else 'FAIL'}")
    md.append("")
    md.append(f"## Baseline HC #428 R1 (for comparison)")
    md.append(f"- Green Sharpe {hc428_base['green']['sharpe']:.2f} | Red Sharpe {hc428_base['red']['sharpe']:.2f} | "
              f"Gap {hc428_base['gap']:.3f} | "
              f"{'PASS' if hc428_base['pass_R1'] else 'FAIL'}")
    md.append("")
    md.append(f"## Day Concentration")
    md.append(f"- Strategy: {dc_strat:.4f} (need ≤ 0.70)")
    md.append("")
    md.append("## Regime Classifier Spot Checks")
    md.append(f"- **COVID Mar-Apr 2020**: {spot_checks['covid_mar2020']}")
    md.append(f"- **2022 bear**: {spot_checks['bear_2022']}")
    md.append(f"- **Mid 2023 → 2024**: {spot_checks['mid_2023_onward']}")
    md.append(f"- Regime transitions: **{len(trans)}** over {n_years:.1f}y → "
              f"**{transitions_per_year:.1f}/yr**")
    md.append("")
    md.append("## Per-Regime Performance (strategy)")
    md.append("| Regime | n_days | Sharpe (ann) | Mean daily | Cum return |")
    md.append("|---|---|---|---|---|")
    for r in ("EARLY", "MID", "LATE", "RECESSION"):
        m = per_reg[r]
        md.append(f"| {r} | {m['n']} | {m['sharpe']:.2f} | {m['mean_daily']*100:.3f}% | {m['cum']*100:.1f}% |")
    md.append("")
    md.append("## Regime Transition Timeline")
    md.append("| Start | End | Regime | Days |")
    md.append("|---|---|---|---|")
    for _, row in trans.tail(40).iterrows():
        md.append(f"| {pd.Timestamp(row['start']).date()} | {pd.Timestamp(row['end']).date()} | {row['regime']} | {row['days']} |")
    md.append("")
    md.append("## Verdict & Failure-Mode Diagnosis")
    pass_gates = hc428_strat["pass_R1"] and dc_strat <= 0.70
    md.append(f"- **HC #428 R1**: {'PASS' if hc428_strat['pass_R1'] else 'FAIL'} "
              f"(gap {hc428_strat['gap']:.3f})")
    md.append(f"- **Day-concentration ≤ 0.70**: {'PASS' if dc_strat <= 0.70 else 'FAIL'}")
    md.append(f"- **Sharpe vs SPY**: strategy {m_strat['sharpe']:.2f} vs SPY {m_spy['sharpe']:.2f} → "
              f"{'beats SPY' if m_strat['sharpe'] > m_spy['sharpe'] else 'loses to SPY'}")
    md.append(f"- **Strategy vs Baseline (regime gating value)**: "
              f"strat Sharpe {m_strat['sharpe']:.2f} vs unconditional {m_base['sharpe']:.2f}")
    md.append("")
    if not hc428_strat["pass_R1"]:
        md.append("### Failure mode")
        late_sh = per_reg.get("LATE", {}).get("sharpe", float("nan"))
        rec_sh = per_reg.get("RECESSION", {}).get("sharpe", float("nan"))
        mid_sh = per_reg.get("MID", {}).get("sharpe", float("nan"))
        # If RECESSION sharpe is negative or wildly different from MID, classifier OR allocation broken
        md.append(f"- Per-regime Sharpe spread: EARLY={per_reg['EARLY']['sharpe']:.2f}, "
                  f"MID={mid_sh:.2f}, LATE={late_sh:.2f}, RECESSION={rec_sh:.2f}")
        md.append("- If RECESSION Sharpe is negative AND classifier called COVID correctly → "
                  "**within-regime allocation wrong** (defensives still bled).")
        md.append("- If RECESSION Sharpe is OK but green/red gap is huge → "
                  "**classifier mis-timing transitions** (caught crisis too late or too early).")
        md.append("- If MID Sharpe dominates everything → strategy is ~MID-only and not "
                  "really regime-aware; the tilts are not differentiated enough.")
    md.append("")
    md.append("## Files")
    md.append(f"- Summary JSON: `{out_dir}/summary.json`")
    md.append(f"- Daily returns: `{out_dir}/daily_returns.parquet`")
    md.append(f"- Regime timeline: `{out_dir}/regime_timeline.parquet`")
    md.append(f"- Holdings log: `{out_dir}/holdings_log.parquet`")

    report_text = "\n".join(md)
    (out_dir / "report.md").write_text(report_text)
    (findings_dir / "macro_regime_rotation_v1.md").write_text(report_text)
    print(f"[run_v1] wrote report.md → {findings_dir}/macro_regime_rotation_v1.md")
    print(f"[run_v1] DONE — HC#428 R1: {'PASS' if hc428_strat['pass_R1'] else 'FAIL'} "
          f"(gap {hc428_strat['gap']:.3f})")
    print(f"[run_v1] strategy: Sharpe {m_strat['sharpe']:.2f} CAGR {m_strat['cagr']*100:.1f}% "
          f"MaxDD {m_strat['max_dd']*100:.1f}% vs SPY Sharpe {m_spy['sharpe']:.2f}")

    return summary


if __name__ == "__main__":
    main()
