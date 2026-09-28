#!/usr/bin/env python3
"""
wheel_v3_earnings_test.py — Test earnings avoidance using the CORRECT v3_expand engine.

Uses the native earnings_lookup parameter in run_portfolio_v3 to properly
skip CSP entries near earnings dates without corrupting the equity series.
"""
import sys
import time
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path("/home/jupiter/Lvl3Quant")
OUT_DIR = ROOT / "output" / "wheel_v3_earnings"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "scripts"))
from wheel_universe_v3_expand import load_all_data, run_portfolio_v3
from wheel_earnings_filter import download_earnings_dates, build_earnings_lookup

logging.basicConfig(
    format='%(asctime)s [V3-EARN] %(levelname)s %(message)s',
    level=logging.INFO,
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger('V3-EARN')


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


def main():
    log.info("=" * 70)
    log.info("WHEEL V3 EARNINGS AVOIDANCE TEST (native integration)")
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

    common_params = dict(
        starting_cash=100_000, margin_cap=0.40, per_name_pct=0.03,
        put_delta=0.30, dte_target=14, profit_take=0.65,
        bear_mode="liq_csp_only", max_assignments_5d=3,
        max_share_positions=5, loss_cut_pct=-0.15,
    )

    results = {}

    # ========== BASELINE (no earnings filter) ==========
    log.info("\n" + "=" * 50)
    log.info("TEST 1: BASELINE (no earnings filter)")
    log.info("=" * 50)

    t0 = time.time()
    eq_base, trades_base = run_portfolio_v3(
        prices, spy_regime, sector_map, **common_params,
        earnings_lookup=None,
    )
    elapsed = time.time() - t0

    metrics_base = compute_metrics(eq_base)
    metrics_base['n_trades'] = len(trades_base)
    metrics_base['elapsed_s'] = round(elapsed, 1)
    results['baseline'] = metrics_base

    log.info(f"  CAGR: {metrics_base['cagr_pct']}% | Sharpe: {metrics_base['sharpe']} | "
             f"MaxDD: {metrics_base['max_dd_pct']}% | Calmar: {metrics_base['calmar']}")
    log.info(f"  Sortino: {metrics_base['sortino']} | WR: {metrics_base['daily_wr']} | "
             f"PF: {metrics_base['profit_factor']} | Trades: {metrics_base['n_trades']}")

    # ========== EARNINGS FILTER (2-day buffer) ==========
    log.info("\n" + "=" * 50)
    log.info("TEST 2: EARNINGS FILTER (2-day buffer)")
    log.info("=" * 50)

    t0 = time.time()
    eq_filt2, trades_filt2 = run_portfolio_v3(
        prices, spy_regime, sector_map, **common_params,
        earnings_lookup=earnings_lookup, earnings_buffer_days=2,
    )
    elapsed = time.time() - t0

    metrics_filt2 = compute_metrics(eq_filt2)
    metrics_filt2['n_trades'] = len(trades_filt2)
    metrics_filt2['elapsed_s'] = round(elapsed, 1)
    results['earnings_filter_2d'] = metrics_filt2

    log.info(f"  CAGR: {metrics_filt2['cagr_pct']}% | Sharpe: {metrics_filt2['sharpe']} | "
             f"MaxDD: {metrics_filt2['max_dd_pct']}% | Calmar: {metrics_filt2['calmar']}")
    log.info(f"  Sortino: {metrics_filt2['sortino']} | WR: {metrics_filt2['daily_wr']} | "
             f"PF: {metrics_filt2['profit_factor']} | Trades: {metrics_filt2['n_trades']}")

    # ========== EARNINGS FILTER (5-day buffer) ==========
    log.info("\n" + "=" * 50)
    log.info("TEST 3: EARNINGS FILTER (5-day buffer)")
    log.info("=" * 50)

    t0 = time.time()
    eq_filt5, trades_filt5 = run_portfolio_v3(
        prices, spy_regime, sector_map, **common_params,
        earnings_lookup=earnings_lookup, earnings_buffer_days=5,
    )
    elapsed = time.time() - t0

    metrics_filt5 = compute_metrics(eq_filt5)
    metrics_filt5['n_trades'] = len(trades_filt5)
    metrics_filt5['elapsed_s'] = round(elapsed, 1)
    results['earnings_filter_5d'] = metrics_filt5

    log.info(f"  CAGR: {metrics_filt5['cagr_pct']}% | Sharpe: {metrics_filt5['sharpe']} | "
             f"MaxDD: {metrics_filt5['max_dd_pct']}% | Calmar: {metrics_filt5['calmar']}")
    log.info(f"  Sortino: {metrics_filt5['sortino']} | WR: {metrics_filt5['daily_wr']} | "
             f"PF: {metrics_filt5['profit_factor']} | Trades: {metrics_filt5['n_trades']}")

    # ========== SUMMARY ==========
    log.info("\n" + "=" * 70)
    log.info("COMPARISON SUMMARY")
    log.info("=" * 70)
    log.info(f"{'Config':<25} {'CAGR':>6} {'Sharpe':>7} {'Sortino':>8} {'MaxDD':>7} {'Calmar':>7} {'Trades':>7}")
    log.info("-" * 70)
    for name, m in results.items():
        log.info(f"{name:<25} {m['cagr_pct']:>5.1f}% {m['sharpe']:>7.2f} {m['sortino']:>8.2f} "
                 f"{m['max_dd_pct']:>6.1f}% {m['calmar']:>7.2f} {m['n_trades']:>7}")

    # Sharpe improvement
    base_s = results['baseline']['sharpe']
    for name in ['earnings_filter_2d', 'earnings_filter_5d']:
        if name in results:
            filt_s = results[name]['sharpe']
            log.info(f"\nSharpe improvement ({name} vs baseline): {filt_s - base_s:+.2f} "
                     f"({(filt_s/base_s-1)*100:+.1f}%)" if base_s > 0 else "")

    # Validate: n_days should match
    base_days = results['baseline']['n_days']
    for name, m in results.items():
        if m['n_days'] != base_days:
            log.warning(f"WARNING: {name} has {m['n_days']} days vs baseline {base_days} — data integrity issue!")

    # Save
    out_file = OUT_DIR / "v3_earnings_test_results.json"
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    log.info(f"\nResults saved to {out_file}")

    return results


if __name__ == "__main__":
    main()
