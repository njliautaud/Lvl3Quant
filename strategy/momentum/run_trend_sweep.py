#!/usr/bin/env python3
"""
Momentum Strategy v2 — Comprehensive Trend-Filtered Sweep
Compares pure momentum vs trend-filtered vs vol-managed variants.

Outputs comparison table with regime gap analysis.
Saves results to /home/jupiter/Lvl3Quant/output/momentum/trend_filtered/
"""

import json
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from strategy.momentum.data import fetch_prices, fetch_spy, fetch_bond, LIQUID_100
from strategy.momentum.engine import (
    MomentumConfig,
    run_backtest,
    compute_metrics,
    print_report,
)

OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/momentum/trend_filtered")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

START = "2010-01-01"  # start early enough for 200d SMA warmup + 12m momentum warmup


def build_sweep_configs() -> list[tuple[str, MomentumConfig]]:
    """Build all config variants for the sweep."""
    configs = []

    # ---- 1. BASELINES (pure momentum, various top_n) ----
    for n in [10, 15, 20]:
        configs.append((
            f"pure_mom_top{n}",
            MomentumConfig(top_n=n, rebal_freq="monthly"),
        ))

    # ---- 2. TREND FILTER variants (200d SMA) ----
    for n in [10, 15, 20]:
        # 100% cash when below SMA
        configs.append((
            f"trend_cash100_top{n}",
            MomentumConfig(
                top_n=n, rebal_freq="monthly",
                trend_filter=True, trend_sma_days=200,
                trend_cash_frac=1.0,
            ),
        ))

    # 50% cash + 50% bonds (SHY) when below SMA, top 20
    configs.append((
        "trend_50shy_top20",
        MomentumConfig(
            top_n=20, rebal_freq="monthly",
            trend_filter=True, trend_sma_days=200,
            trend_cash_frac=0.5,
            trend_bond_ticker="SHY",
        ),
    ))

    # 100% to SHY when below SMA
    configs.append((
        "trend_shy100_top20",
        MomentumConfig(
            top_n=20, rebal_freq="monthly",
            trend_filter=True, trend_sma_days=200,
            trend_cash_frac=1.0,
            trend_bond_ticker="SHY",
        ),
    ))

    # 10-month SMA (~210 trading days)
    configs.append((
        "trend_10mo_top20",
        MomentumConfig(
            top_n=20, rebal_freq="monthly",
            trend_filter=True, trend_sma_days=210,
            trend_cash_frac=1.0,
        ),
    ))

    # ---- 3. TREND + VOL-MANAGED ----
    for n in [10, 15, 20]:
        configs.append((
            f"trend_vol_top{n}",
            MomentumConfig(
                top_n=n, rebal_freq="monthly",
                trend_filter=True, trend_sma_days=200,
                trend_cash_frac=1.0,
                vol_managed=True, vol_target=0.15,
            ),
        ))

    # ---- 4. SECTOR-NEUTRAL variants ----
    configs.append((
        "sector_neutral_2ps",
        MomentumConfig(
            top_n=20, rebal_freq="monthly",
            sector_neutral=True, top_per_sector=2,
        ),
    ))
    configs.append((
        "sector_neutral_trend",
        MomentumConfig(
            top_n=20, rebal_freq="monthly",
            sector_neutral=True, top_per_sector=2,
            trend_filter=True, trend_sma_days=200,
            trend_cash_frac=1.0,
        ),
    ))
    configs.append((
        "sector_neutral_trend_vol",
        MomentumConfig(
            top_n=20, rebal_freq="monthly",
            sector_neutral=True, top_per_sector=2,
            trend_filter=True, trend_sma_days=200,
            trend_cash_frac=1.0,
            vol_managed=True, vol_target=0.15,
        ),
    ))

    # ---- 5. INV-VOL WEIGHTING + TREND ----
    configs.append((
        "trend_ivol_top20",
        MomentumConfig(
            top_n=20, rebal_freq="monthly", weighting="inv_vol",
            trend_filter=True, trend_sma_days=200,
            trend_cash_frac=1.0,
        ),
    ))

    # ---- 6. LONG/SHORT IN DOWNTREND ----
    # When below SMA: keep 0% long + short bottom 10 at 50% weight
    configs.append((
        "trend_longshort_top20",
        MomentumConfig(
            top_n=20, rebal_freq="monthly",
            trend_filter=True, trend_sma_days=200,
            trend_cash_frac=1.0,  # 0% long
            short_in_downtrend=True, short_n=10, short_weight=0.5,
        ),
    ))
    # Keep 50% long + 30% short in downtrend
    configs.append((
        "trend_ls_5030_top20",
        MomentumConfig(
            top_n=20, rebal_freq="monthly",
            trend_filter=True, trend_sma_days=200,
            trend_cash_frac=0.5,  # keep 50% long
            short_in_downtrend=True, short_n=10, short_weight=0.3,
        ),
    ))
    # Full short only (no longs) in downtrend
    configs.append((
        "trend_short_only_top20",
        MomentumConfig(
            top_n=20, rebal_freq="monthly",
            trend_filter=True, trend_sma_days=200,
            trend_cash_frac=1.0,
            short_in_downtrend=True, short_n=15, short_weight=0.8,
        ),
    ))

    # ---- 7. DEFENSIVE ROTATION ----
    configs.append((
        "defensive_rotation_top20",
        MomentumConfig(
            top_n=20, rebal_freq="monthly",
            trend_filter=True, trend_sma_days=200,
            defensive_rotation=True,
        ),
    ))
    # Defensive + vol managed
    configs.append((
        "defensive_vol_top20",
        MomentumConfig(
            top_n=20, rebal_freq="monthly",
            trend_filter=True, trend_sma_days=200,
            defensive_rotation=True,
            vol_managed=True, vol_target=0.15,
        ),
    ))

    # ---- 8. COMBINED: sector neutral + long/short + vol ----
    configs.append((
        "sn_ls_vol_top20",
        MomentumConfig(
            top_n=20, rebal_freq="monthly",
            sector_neutral=True, top_per_sector=2,
            trend_filter=True, trend_sma_days=200,
            trend_cash_frac=1.0,
            short_in_downtrend=True, short_n=10, short_weight=0.4,
            vol_managed=True, vol_target=0.15,
        ),
    ))

    # ---- 9. AGGRESSIVE SHORT to achieve positive red-month Sharpe ----
    # Need the strategy to MAKE MONEY in red months.
    # Short momentum losers at full weight during downtrends.
    for sw in [1.0, 1.2, 1.5]:
        for sn in [15, 20]:
            tag = f"trend_aggshort_sw{int(sw*100)}_sn{sn}"
            configs.append((
                tag,
                MomentumConfig(
                    top_n=20, rebal_freq="monthly",
                    trend_filter=True, trend_sma_days=200,
                    trend_cash_frac=1.0,  # 0% long in downtrend
                    short_in_downtrend=True, short_n=sn, short_weight=sw,
                ),
            ))

    # ---- 10. ALWAYS long/short (market-neutral-ish) ----
    # Long top N + short bottom N at all times
    for sw in [0.3, 0.5, 0.7, 1.0]:
        tag = f"always_ls_sw{int(sw*100)}"
        configs.append((
            tag,
            MomentumConfig(
                top_n=20, rebal_freq="monthly",
                always_short=True, short_n=20, short_weight=sw,
            ),
        ))

    # Always LS + vol managed
    configs.append((
        "always_ls_50_vol",
        MomentumConfig(
            top_n=20, rebal_freq="monthly",
            always_short=True, short_n=20, short_weight=0.5,
            vol_managed=True, vol_target=0.15,
        ),
    ))

    # ---- 11. TREND + aggressive short + vol (kitchen sink) ----
    configs.append((
        "trend_aggshort_vol",
        MomentumConfig(
            top_n=20, rebal_freq="monthly",
            trend_filter=True, trend_sma_days=200,
            trend_cash_frac=1.0,
            short_in_downtrend=True, short_n=20, short_weight=1.0,
            vol_managed=True, vol_target=0.15,
        ),
    ))

    # ---- 12. FINE-GRAINED always LS search (looking for gap < 0.50) ----
    for sw in [0.75, 0.80, 0.85, 0.90, 0.95]:
        tag = f"always_ls_sw{int(sw*100)}"
        configs.append((
            tag,
            MomentumConfig(
                top_n=20, rebal_freq="monthly",
                always_short=True, short_n=20, short_weight=sw,
            ),
        ))

    # Always LS with vol management
    for sw in [0.80, 0.85, 0.90]:
        tag = f"always_ls_sw{int(sw*100)}_vol"
        configs.append((
            tag,
            MomentumConfig(
                top_n=20, rebal_freq="monthly",
                always_short=True, short_n=20, short_weight=sw,
                vol_managed=True, vol_target=0.15,
            ),
        ))

    # ---- 13. FINE-GRAINED trend + short sweep ----
    for sw in [1.3, 1.4, 1.6, 1.8, 2.0]:
        tag = f"trend_aggshort_sw{int(sw*100)}_sn20"
        configs.append((
            tag,
            MomentumConfig(
                top_n=20, rebal_freq="monthly",
                trend_filter=True, trend_sma_days=200,
                trend_cash_frac=1.0,
                short_in_downtrend=True, short_n=20, short_weight=sw,
            ),
        ))

    # ---- 14. BEST candidates + vol management ----
    # Trend + 150% short + vol
    configs.append((
        "trend_s150_vol",
        MomentumConfig(
            top_n=20, rebal_freq="monthly",
            trend_filter=True, trend_sma_days=200,
            trend_cash_frac=1.0,
            short_in_downtrend=True, short_n=15, short_weight=1.5,
            vol_managed=True, vol_target=0.12,
        ),
    ))
    # Trend + 200% short + vol
    configs.append((
        "trend_s200_vol",
        MomentumConfig(
            top_n=20, rebal_freq="monthly",
            trend_filter=True, trend_sma_days=200,
            trend_cash_frac=1.0,
            short_in_downtrend=True, short_n=20, short_weight=2.0,
            vol_managed=True, vol_target=0.12,
        ),
    ))

    return configs


def run_sweep():
    """Run full sweep, build comparison table, save results."""
    print("=" * 80)
    print("  MOMENTUM v2 COMPREHENSIVE SWEEP")
    print("  Trend Filter + Vol Management + Sector Neutral")
    print("=" * 80)

    # Fetch all data once
    print("\n[sweep] Fetching price data...")
    prices = fetch_prices(tickers=LIQUID_100, start=START)
    spy = fetch_spy(start=START)
    bond = fetch_bond(ticker="SHY", start=START)

    configs = build_sweep_configs()
    all_results = []

    for tag, cfg in configs:
        print(f"\n{'─'*60}")
        print(f"  >>> {tag}")
        print(f"{'─'*60}")
        try:
            # Determine what extra data the backtest needs
            spy_for_bt = spy if cfg.trend_filter else None
            bond_for_bt = bond if (cfg.trend_filter and cfg.trend_bond_ticker) else None

            result = run_backtest(prices, cfg, spy_prices=spy_for_bt, bond_prices=bond_for_bt)
            equity = result["equity_curve"]
            metrics = compute_metrics(equity, spy=spy, config=cfg)
            print_report(metrics, config=result["config"])

            # Save individual equity curve
            equity.to_csv(OUTPUT_DIR / f"{tag}_equity.csv")

            all_results.append({
                "tag": tag,
                "config": result["config"],
                "metrics": metrics,
                "start_date": result["start_date"],
                "end_date": result["end_date"],
            })
        except Exception as e:
            print(f"  ERROR in {tag}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not all_results:
        print("\nNo successful runs!")
        return

    # ─── COMPARISON TABLE ───
    print("\n\n" + "=" * 120)
    print("  COMPREHENSIVE COMPARISON TABLE")
    print("=" * 120)

    header = (f"  {'Tag':<30} {'CAGR%':>7} {'Sharpe':>7} {'Sortino':>8} "
              f"{'MaxDD%':>8} {'MoWR%':>7} {'S_grn':>7} {'S_red':>7} "
              f"{'Gap':>6} {'Pass':>5}")
    print(header)
    print("─" * 120)

    for r in all_results:
        m = r["metrics"]
        rg = m.get("regime", {})
        gap = rg.get("regime_gap", float("nan"))
        passes = rg.get("regime_pass", False)
        s_grn = rg.get("sharpe_green", float("nan"))
        s_red = rg.get("sharpe_red", float("nan"))

        pass_str = "PASS" if passes else "FAIL"
        print(f"  {r['tag']:<30} {m['cagr']:>7.2f} {m['sharpe']:>7.3f} {m['sortino']:>8.3f} "
              f"{m['max_drawdown_pct']:>8.2f} {m['monthly_win_rate']:>7.2f} "
              f"{s_grn:>7.3f} {s_red:>7.3f} {gap:>6.3f} {pass_str:>5}")

    print("=" * 120)

    # Count passes
    passing = [r for r in all_results if r["metrics"].get("regime", {}).get("regime_pass", False)]
    failing = [r for r in all_results if not r["metrics"].get("regime", {}).get("regime_pass", False)]
    print(f"\n  PASSING: {len(passing)}/{len(all_results)} configs")

    if passing:
        # Find best passing config by Sharpe
        best = max(passing, key=lambda r: r["metrics"]["sharpe"])
        m = best["metrics"]
        rg = m.get("regime", {})
        print(f"\n  BEST REGIME-PROOF CONFIG: {best['tag']}")
        print(f"    CAGR={m['cagr']:.2f}%, Sharpe={m['sharpe']:.3f}, Sortino={m['sortino']:.3f}, "
              f"MaxDD={m['max_drawdown_pct']:.2f}%, Regime Gap={rg.get('regime_gap', 0):.3f}")

    # Save full results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = OUTPUT_DIR / f"sweep_comparison_{timestamp}.json"
    summary_file.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\n  Full results saved to: {summary_file}")

    # Save comparison table as CSV for easy viewing
    rows = []
    for r in all_results:
        m = r["metrics"]
        rg = m.get("regime", {})
        rows.append({
            "config": r["tag"],
            "cagr_pct": m["cagr"],
            "sharpe": m["sharpe"],
            "sortino": m["sortino"],
            "max_dd_pct": m["max_drawdown_pct"],
            "monthly_wr": m["monthly_win_rate"],
            "sharpe_green": rg.get("sharpe_green", None),
            "sharpe_red": rg.get("sharpe_red", None),
            "regime_gap": rg.get("regime_gap", None),
            "regime_pass": rg.get("regime_pass", None),
        })
    table = pd.DataFrame(rows)
    table_file = OUTPUT_DIR / f"comparison_table_{timestamp}.csv"
    table.to_csv(table_file, index=False)
    print(f"  Comparison table saved to: {table_file}")

    return all_results


if __name__ == "__main__":
    run_sweep()
