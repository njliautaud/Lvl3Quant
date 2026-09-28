#!/usr/bin/env python3
"""
wheel_v3_combined_test.py — Test combined optimizations on the v3 engine.

Tests stacking: earnings filter + delta adjustment + DTE tuning.
Uses the validated v3_expand engine with native earnings integration.
"""
import sys
import time
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path("/home/jupiter/Lvl3Quant")
OUT_DIR = ROOT / "output" / "wheel_v3_combined"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "scripts"))
from wheel_universe_v3_expand import load_all_data, run_portfolio_v3
from wheel_earnings_filter import download_earnings_dates, build_earnings_lookup

logging.basicConfig(
    format='%(asctime)s [V3-COMB] %(levelname)s %(message)s',
    level=logging.INFO,
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger('V3-COMB')


def compute_metrics(equity_list, starting_cash=100_000):
    """Compute standard metrics from daily equity list."""
    eq = pd.DataFrame(equity_list)
    eq['ret'] = eq['equity'].pct_change()
    eq = eq.dropna(subset=['ret'])

    if len(eq) < 20:
        return None

    years = len(eq) / 252
    final = eq['equity'].iloc[-1]
    cagr = ((final / starting_cash) ** (1/years) - 1) * 100
    mu = eq['ret'].mean() * 252
    std = eq['ret'].std() * np.sqrt(252)
    sharpe = mu / std if std > 0 else 0
    downside = eq['ret'][eq['ret'] < 0].std() * np.sqrt(252)
    sortino = mu / downside if downside > 0 else 0
    cummax = eq['equity'].cummax()
    max_dd = (eq['equity'] / cummax - 1).min() * 100
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    wr = (eq['ret'] > 0).mean()
    pos = eq['ret'][eq['ret'] > 0].sum()
    neg = abs(eq['ret'][eq['ret'] < 0].sum())
    pf = pos / neg if neg > 0 else 999

    return {
        'cagr_pct': round(cagr, 1),
        'sharpe': round(sharpe, 2),
        'sortino': round(sortino, 2),
        'max_dd_pct': round(max_dd, 1),
        'calmar': round(calmar, 2),
        'daily_wr': round(wr, 3),
        'profit_factor': round(pf, 2),
        'final_equity': round(final, 2),
        'years': round(years, 2),
        'n_days': len(eq),
    }


def run_config(name, prices, spy_regime, sector_map, **overrides):
    """Run a single config and return metrics."""
    params = dict(
        starting_cash=100_000, margin_cap=0.40, per_name_pct=0.03,
        put_delta=0.30, dte_target=14, profit_take=0.65,
        bear_mode="liq_csp_only", max_assignments_5d=3,
        max_share_positions=5, loss_cut_pct=-0.15,
        earnings_lookup=None, earnings_buffer_days=2,
    )
    params.update(overrides)

    log.info(f"\n{'='*50}")
    log.info(f"CONFIG: {name}")
    log.info(f"  delta={params['put_delta']}, dte={params['dte_target']}, "
             f"pt={params['profit_take']}, margin={params['margin_cap']}, "
             f"earnings={'ON' if params['earnings_lookup'] is not None else 'OFF'}")
    log.info(f"{'='*50}")

    t0 = time.time()
    eq, trades = run_portfolio_v3(prices, spy_regime, sector_map, **params)
    elapsed = time.time() - t0

    metrics = compute_metrics(eq)
    if metrics is None:
        log.warning(f"  FAILED: insufficient data")
        return None
    metrics['n_trades'] = len(trades)
    metrics['elapsed_s'] = round(elapsed, 1)

    log.info(f"  CAGR: {metrics['cagr_pct']}% | Sharpe: {metrics['sharpe']} | "
             f"MaxDD: {metrics['max_dd_pct']}% | Calmar: {metrics['calmar']}")
    log.info(f"  Sortino: {metrics['sortino']} | WR: {metrics['daily_wr']} | "
             f"PF: {metrics['profit_factor']} | Trades: {metrics['n_trades']}")

    return metrics


def main():
    log.info("=" * 70)
    log.info("WHEEL V3 COMBINED OPTIMIZATION TEST")
    log.info("=" * 70)

    # Load data
    log.info("Loading price data...")
    prices, spy_regime, sector_map = load_all_data(start_date="2019-01-01")
    tickers = [t for t in prices['ticker'].unique() if t not in ('SPY', 'VIX', '^VIX')]
    log.info(f"Universe: {len(tickers)} tickers")

    # Load earnings dates
    log.info("Loading earnings dates...")
    earnings_df = download_earnings_dates(tickers)
    earnings_lookup = build_earnings_lookup(earnings_df)
    log.info(f"Earnings data: {len(earnings_lookup)} tickers with dates")

    results = {}

    # ---- Reference configs (already tested, rerun for consistent comparison) ----

    # 1. Baseline d30, no earnings
    results['d30_no_earn'] = run_config(
        'd30_no_earn', prices, spy_regime, sector_map,
        put_delta=0.30, earnings_lookup=None,
    )

    # 2. d30 + 2d earnings (prior best)
    results['d30_earn2d'] = run_config(
        'd30_earn2d', prices, spy_regime, sector_map,
        put_delta=0.30, earnings_lookup=earnings_lookup, earnings_buffer_days=2,
    )

    # 3. d35 no earnings (prior best IV-adaptive)
    results['d35_no_earn'] = run_config(
        'd35_no_earn', prices, spy_regime, sector_map,
        put_delta=0.35, earnings_lookup=None,
    )

    # ---- NEW combined configs ----

    # 4. d35 + 2d earnings (THE COMBO)
    results['d35_earn2d'] = run_config(
        'd35_earn2d', prices, spy_regime, sector_map,
        put_delta=0.35, earnings_lookup=earnings_lookup, earnings_buffer_days=2,
    )

    # 5. d35 + 2d earnings + shorter DTE (10 days)
    results['d35_earn2d_dte10'] = run_config(
        'd35_earn2d_dte10', prices, spy_regime, sector_map,
        put_delta=0.35, dte_target=10, earnings_lookup=earnings_lookup, earnings_buffer_days=2,
    )

    # 6. d35 + 2d earnings + longer DTE (21 days)
    results['d35_earn2d_dte21'] = run_config(
        'd35_earn2d_dte21', prices, spy_regime, sector_map,
        put_delta=0.35, dte_target=21, earnings_lookup=earnings_lookup, earnings_buffer_days=2,
    )

    # 7. d35 + 2d earnings + tighter profit take (50%)
    results['d35_earn2d_pt50'] = run_config(
        'd35_earn2d_pt50', prices, spy_regime, sector_map,
        put_delta=0.35, profit_take=0.50, earnings_lookup=earnings_lookup, earnings_buffer_days=2,
    )

    # 8. d35 + 2d earnings + wider profit take (80%)
    results['d35_earn2d_pt80'] = run_config(
        'd35_earn2d_pt80', prices, spy_regime, sector_map,
        put_delta=0.35, profit_take=0.80, earnings_lookup=earnings_lookup, earnings_buffer_days=2,
    )

    # 9. d35 + 2d earnings + higher per-name (5%)
    results['d35_earn2d_pn5'] = run_config(
        'd35_earn2d_pn5', prices, spy_regime, sector_map,
        put_delta=0.35, per_name_pct=0.05, earnings_lookup=earnings_lookup, earnings_buffer_days=2,
    )

    # 10. d35 + 2d earnings + tighter loss cut (-10%)
    results['d35_earn2d_lc10'] = run_config(
        'd35_earn2d_lc10', prices, spy_regime, sector_map,
        put_delta=0.35, loss_cut_pct=-0.10, earnings_lookup=earnings_lookup, earnings_buffer_days=2,
    )

    # ========== SUMMARY ==========
    log.info("\n" + "=" * 80)
    log.info("COMBINED OPTIMIZATION SUMMARY")
    log.info("=" * 80)
    log.info(f"{'Config':<25} {'CAGR':>6} {'Sharpe':>7} {'Sortino':>8} {'MaxDD':>7} {'Calmar':>7} {'PF':>5} {'Trades':>7}")
    log.info("-" * 80)

    # Sort by Sharpe
    sorted_results = sorted(results.items(), key=lambda x: x[1]['sharpe'] if x[1] else 0, reverse=True)
    for name, m in sorted_results:
        if m is None:
            continue
        log.info(f"{name:<25} {m['cagr_pct']:>5.1f}% {m['sharpe']:>7.2f} {m['sortino']:>8.2f} "
                 f"{m['max_dd_pct']:>6.1f}% {m['calmar']:>7.2f} {m['profit_factor']:>5.2f} {m['n_trades']:>7}")

    # Best config
    best_name = sorted_results[0][0]
    best = sorted_results[0][1]
    log.info(f"\nBEST CONFIG: {best_name}")
    log.info(f"  Sharpe {best['sharpe']} | CAGR {best['cagr_pct']}% | MaxDD {best['max_dd_pct']}% | "
             f"Calmar {best['calmar']} | PF {best['profit_factor']}")

    # Compare best vs baseline
    base = results['d30_no_earn']
    if base:
        log.info(f"\n  vs d30 baseline: Sharpe {best['sharpe'] - base['sharpe']:+.2f} "
                 f"({(best['sharpe']/base['sharpe']-1)*100:+.1f}%), "
                 f"CAGR {best['cagr_pct'] - base['cagr_pct']:+.1f}pp, "
                 f"MaxDD {best['max_dd_pct'] - base['max_dd_pct']:+.1f}pp")

    # Validate n_days consistency
    base_days = results['d30_no_earn']['n_days'] if results['d30_no_earn'] else None
    for name, m in results.items():
        if m and base_days and m['n_days'] != base_days:
            log.warning(f"WARNING: {name} has {m['n_days']} days vs baseline {base_days}")

    # Save
    out_file = OUT_DIR / "combined_test_results.json"
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    log.info(f"\nResults saved to {out_file}")

    return results


if __name__ == "__main__":
    main()
