#!/usr/bin/env python3
"""
wheel_fast_analyses.py — Two high-value analyses using the proven fast engine:
1. Walk-forward out-of-sample validation (per-year + expanding window)
2. Earnings avoidance filter A/B test

Uses run_portfolio_v3 from wheel_universe_v3_expand directly (~55s per config)
instead of reimplementing the simulation loop.
"""
import sys
import time
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from copy import deepcopy

ROOT = Path("/home/jupiter/Lvl3Quant")
OUT_DIR = ROOT / "output" / "wheel_fast_analyses"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "scripts"))

logging.basicConfig(
    format='%(asctime)s [FAST] %(levelname)s %(message)s',
    level=logging.INFO,
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger('FAST')


def run_and_measure(prices_df, spy_regime, sector_map, label="", **kwargs):
    """Run portfolio backtest and compute all metrics. Returns dict."""
    from wheel_universe_v3_expand import run_portfolio_v3, compute_metrics, regime_analysis

    t0 = time.time()
    daily_eq, trades = run_portfolio_v3(prices_df, spy_regime, sector_map, **kwargs)
    elapsed = time.time() - t0

    metrics = compute_metrics(daily_eq, kwargs.get('starting_cash', 100_000))
    regime = regime_analysis(daily_eq, spy_regime)

    return {
        'label': label,
        'metrics': metrics,
        'regime': regime,
        'n_trades': len(trades),
        'elapsed_s': round(elapsed, 1),
    }


def analysis_1_per_year(prices, spy_regime, sector_map):
    """Per-year performance with fixed best params (d=0.30, DTE=14, margin=40%, PT=65%)."""
    log.info("=" * 70)
    log.info("ANALYSIS 1: PER-YEAR PERFORMANCE (fixed best params)")
    log.info("=" * 70)

    results = {}
    for year in range(2019, 2027):
        # Filter data to just this year
        year_prices = prices[prices['date'].dt.year == year].copy()
        year_regime = spy_regime[spy_regime['date'].dt.year == year].copy()

        if len(year_prices) < 100:  # Need enough data
            log.info(f"  {year}: insufficient data ({len(year_prices)} rows), skipping")
            continue

        log.info(f"  Running {year}...")
        r = run_and_measure(
            year_prices, year_regime, sector_map,
            label=f"year_{year}",
            starting_cash=100_000, put_delta=0.30, dte_target=14,
            margin_cap=0.40, per_name_pct=0.03, profit_take=0.65,
        )
        results[year] = r
        m = r['metrics']
        if m:
            log.info(f"  {year}: CAGR={m.get('cagr_pct', 'N/A')}% Sharpe={m.get('sharpe', 'N/A')} "
                     f"MaxDD={m.get('max_dd_pct', 'N/A')}% WR={m.get('daily_wr', 'N/A')} "
                     f"PF={m.get('profit_factor', 'N/A')} ({r['elapsed_s']}s)")

    # Summary
    sharpes = [r['metrics']['sharpe'] for r in results.values() if r['metrics']]
    cagrs = [r['metrics']['cagr_pct'] for r in results.values() if r['metrics']]
    positive = sum(1 for c in cagrs if c > 0)

    log.info(f"\n  Summary: {positive}/{len(cagrs)} positive years")
    log.info(f"  Avg Sharpe: {np.mean(sharpes):.2f}, Median: {np.median(sharpes):.2f}")
    log.info(f"  Avg CAGR: {np.mean(cagrs):.1f}%, Range: {min(cagrs):.1f}% to {max(cagrs):.1f}%")

    return results


def analysis_2_walkforward(prices, spy_regime, sector_map):
    """Walk-forward: expand training window, test on next year, compare param selection."""
    log.info("\n" + "=" * 70)
    log.info("ANALYSIS 2: WALK-FORWARD VALIDATION")
    log.info("Param grid: delta={0.25,0.30,0.35} x PT={0.50,0.65,0.80}")
    log.info("=" * 70)

    param_grid = [
        {'put_delta': d, 'profit_take': pt}
        for d in [0.25, 0.30, 0.35]
        for pt in [0.50, 0.65, 0.80]
    ]

    wf_results = []

    for test_year in range(2021, 2027):
        train_prices = prices[prices['date'].dt.year < test_year].copy()
        test_prices = prices[prices['date'].dt.year == test_year].copy()
        train_regime = spy_regime[spy_regime['date'].dt.year < test_year].copy()
        test_regime = spy_regime[spy_regime['date'].dt.year == test_year].copy()

        if len(train_prices) < 500 or len(test_prices) < 100:
            continue

        log.info(f"\n  --- Train 2019-{test_year-1}, Test {test_year} ---")

        # Find best params on training set
        best_sharpe = -999
        best_params = None
        for params in param_grid:
            r = run_and_measure(
                train_prices, train_regime, sector_map,
                starting_cash=100_000, margin_cap=0.40, per_name_pct=0.03,
                dte_target=14, **params,
            )
            s = r['metrics'].get('sharpe', -999) if r['metrics'] else -999
            if s > best_sharpe:
                best_sharpe = s
                best_params = params

        log.info(f"  Best train: delta={best_params['put_delta']}, PT={best_params['profit_take']}, "
                 f"Sharpe={best_sharpe:.2f}")

        # Test best params on OOS
        oos = run_and_measure(
            test_prices, test_regime, sector_map,
            starting_cash=100_000, margin_cap=0.40, per_name_pct=0.03,
            dte_target=14, **best_params,
        )

        # Also test fixed params for comparison
        fixed = run_and_measure(
            test_prices, test_regime, sector_map,
            starting_cash=100_000, margin_cap=0.40, per_name_pct=0.03,
            dte_target=14, put_delta=0.30, profit_take=0.65,
        )

        oos_sharpe = oos['metrics'].get('sharpe', 0) if oos['metrics'] else 0
        fixed_sharpe = fixed['metrics'].get('sharpe', 0) if fixed['metrics'] else 0
        oos_cagr = oos['metrics'].get('cagr_pct', 0) if oos['metrics'] else 0
        oos_dd = oos['metrics'].get('max_dd_pct', 0) if oos['metrics'] else 0

        wf_entry = {
            'test_year': test_year,
            'best_delta': best_params['put_delta'],
            'best_pt': best_params['profit_take'],
            'train_sharpe': round(best_sharpe, 2),
            'oos_sharpe': oos_sharpe,
            'oos_cagr': oos_cagr,
            'oos_max_dd': oos_dd,
            'fixed_sharpe': fixed_sharpe,
            'fixed_cagr': fixed['metrics'].get('cagr_pct', 0) if fixed['metrics'] else 0,
            'degradation': round(best_sharpe - oos_sharpe, 2),
        }
        wf_results.append(wf_entry)

        log.info(f"  OOS: Sharpe={oos_sharpe:.2f}, CAGR={oos_cagr:+.1f}%, DD={oos_dd:.1f}%")
        log.info(f"  Fixed: Sharpe={fixed_sharpe:.2f}")
        log.info(f"  Degradation: {wf_entry['degradation']:.2f}")

    # Summary
    if wf_results:
        avg_oos = np.mean([w['oos_sharpe'] for w in wf_results])
        avg_fixed = np.mean([w['fixed_sharpe'] for w in wf_results])
        avg_deg = np.mean([w['degradation'] for w in wf_results])
        log.info(f"\n  Walk-Forward Summary:")
        log.info(f"  Avg OOS Sharpe: {avg_oos:.2f}")
        log.info(f"  Avg Fixed Sharpe: {avg_fixed:.2f}")
        log.info(f"  Avg Degradation: {avg_deg:.2f}")
        log.info(f"  Verdict: {'Fixed params robust — no overfit' if avg_fixed >= avg_oos * 0.9 else 'Optimization adds value'}")

    return wf_results


def analysis_3_earnings_filter(prices, spy_regime, sector_map):
    """
    Earnings avoidance A/B test.
    Since we can't easily modify run_portfolio_v3's inner loop to check earnings,
    we test by REMOVING tickers from the universe when they have upcoming earnings.
    Approach: filter the prices DataFrame to exclude dates within earnings windows.
    """
    log.info("\n" + "=" * 70)
    log.info("ANALYSIS 3: EARNINGS AVOIDANCE (approximate)")
    log.info("=" * 70)

    # Load earnings dates
    import yfinance as yf
    cache_file = ROOT / "wheel_strategy_v1" / "data" / "cache" / "earnings_dates.parquet"

    if cache_file.exists():
        log.info("Loading cached earnings dates...")
        earn_df = pd.read_parquet(cache_file)
    else:
        log.info("No cached earnings data — skipping earnings analysis")
        return None

    # Build lookup
    earn_lookup = {}
    for ticker, grp in earn_df.groupby('ticker'):
        earn_lookup[ticker] = np.sort(grp['earnings_date'].values)

    tickers = prices['ticker'].unique()
    covered = sum(1 for t in tickers if t in earn_lookup)
    log.info(f"Earnings data: {covered}/{len(tickers)} tickers covered")

    # Strategy: for each date+ticker, if earnings within next 14 days, set price to NaN
    # This effectively makes run_portfolio_v3 skip that ticker on that date
    log.info("Building earnings-masked price data (2-day buffer)...")
    prices_masked = prices.copy()
    mask = np.zeros(len(prices_masked), dtype=bool)

    for i, row in enumerate(prices_masked.itertuples()):
        ticker = row.ticker
        date = row.date
        if ticker not in earn_lookup:
            continue
        dates = earn_lookup[ticker]
        window_start = np.datetime64(date) - np.timedelta64(2, 'D')
        window_end = np.datetime64(date) + np.timedelta64(16, 'D')  # DTE=14 + 2 buffer
        idx_start = np.searchsorted(dates, window_start, side='left')
        idx_end = np.searchsorted(dates, window_end, side='right')
        if idx_end > idx_start:
            mask[i] = True

    skipped = mask.sum()
    total = len(mask)
    log.info(f"Earnings mask: {skipped}/{total} rows ({skipped/total*100:.1f}%) would be filtered")
    log.info(f"Note: this is approximate — actual skip depends on whether a CSP would be opened that day")

    # Actually, a better approach: just remove the masked rows for tickers with earnings
    # This is approximate but gives directional signal
    prices_filtered = prices_masked[~mask].copy()
    log.info(f"Filtered data: {len(prices_filtered)} rows (from {len(prices_masked)})")

    # Run baseline
    log.info("\n  Running BASELINE (no filter)...")
    baseline = run_and_measure(
        prices, spy_regime, sector_map,
        label="baseline",
        starting_cash=100_000, put_delta=0.30, dte_target=14,
        margin_cap=0.40, per_name_pct=0.03, profit_take=0.65,
    )

    # Run with earnings filter
    log.info("  Running WITH earnings filter...")
    filtered = run_and_measure(
        prices_filtered, spy_regime, sector_map,
        label="earnings_filtered",
        starting_cash=100_000, put_delta=0.30, dte_target=14,
        margin_cap=0.40, per_name_pct=0.03, profit_take=0.65,
    )

    b = baseline['metrics']
    f = filtered['metrics']
    if b and f:
        log.info(f"\n  BASELINE:  CAGR={b['cagr_pct']}% Sharpe={b['sharpe']} MaxDD={b['max_dd_pct']}%")
        log.info(f"  FILTERED:  CAGR={f['cagr_pct']}% Sharpe={f['sharpe']} MaxDD={f['max_dd_pct']}%")
        sharpe_delta = f['sharpe'] - b['sharpe']
        dd_delta = f['max_dd_pct'] - b['max_dd_pct']
        log.info(f"  Delta: Sharpe {sharpe_delta:+.2f}, MaxDD {dd_delta:+.1f}%")
        if sharpe_delta > 0.05:
            log.info(f"  VERDICT: Earnings filter IMPROVES risk-adjusted returns")
        elif sharpe_delta > -0.05:
            log.info(f"  VERDICT: Earnings filter is NEUTRAL (within noise)")
        else:
            log.info(f"  VERDICT: Earnings filter HURTS (reduces premium opportunities more than risk)")

    return {'baseline': baseline, 'filtered': filtered, 'rows_skipped_pct': round(skipped/total*100, 1)}


def main():
    from wheel_universe_v3_expand import load_all_data

    log.info("=" * 70)
    log.info("WHEEL FAST ANALYSES — Walk-Forward + Earnings Filter")
    log.info("=" * 70)

    log.info("Loading data...")
    prices, spy_regime, sector_map = load_all_data(start_date="2019-01-01")

    # Ensure dates are Timestamps
    prices['date'] = pd.to_datetime(prices['date'])
    spy_regime['date'] = pd.to_datetime(spy_regime['date'])
    log.info(f"Loaded: {prices['ticker'].nunique()} tickers, {prices['date'].nunique()} dates")

    all_results = {}

    # Analysis 1: Per-year performance
    year_results = analysis_1_per_year(prices, spy_regime, sector_map)
    all_results['per_year'] = {str(k): v for k, v in year_results.items()}

    # Analysis 2: Walk-forward
    wf_results = analysis_2_walkforward(prices, spy_regime, sector_map)
    all_results['walk_forward'] = wf_results

    # Analysis 3: Earnings filter
    earn_results = analysis_3_earnings_filter(prices, spy_regime, sector_map)
    all_results['earnings'] = earn_results

    # Save
    out_file = OUT_DIR / "fast_analyses_results.json"
    with open(out_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    log.info(f"\nAll results saved to {out_file}")

    return all_results


if __name__ == "__main__":
    main()
