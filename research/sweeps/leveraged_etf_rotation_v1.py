#!/usr/bin/env python3
"""
Leveraged ETF Rotation Strategy v1
===================================
Tests whether momentum-based ETF rotation edge survives with 3x leveraged ETFs.

Variants:
1. Pure momentum — top 3 by 3mo return, no regime gate
2. Regime-gated — top 3 when SPY > 200d SMA, else safety
3. Dual momentum — top 3 must also have positive abs momentum, else safety
4. Conservative — top 2 leveraged + 1 safety always

Walk-forward: 252d train / 21d test, sliding window
"""

import json
import warnings
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings('ignore')

# Force unbuffered output
print = lambda *args, **kwargs: (sys.stdout.write(' '.join(str(a) for a in args) + kwargs.get('end', '\n')), sys.stdout.flush())

# ─── Configuration ───────────────────────────────────────────────────────────

LEVERAGED = ['TQQQ', 'UPRO', 'SOXL', 'TNA', 'LABU', 'FAS', 'CURE', 'ERX', 'DRN']
SAFETY = ['TMF', 'TLT', 'SHY', 'BIL', 'GLD']
BENCHMARKS = ['SPY', 'QQQ']

START_DATE = '2014-01-01'
END_DATE = '2026-07-23'

TRAIN_DAYS = 252
TEST_DAYS = 21
LOOKBACK_OPTIONS = [21, 63, 126]
N_HOLDINGS_OPTIONS = [2, 3, 4]
N_PERMUTATIONS = 200


def download_data():
    """Download daily close prices for all tickers."""
    all_tickers = list(set(LEVERAGED + SAFETY + BENCHMARKS))
    print(f"Downloading {len(all_tickers)} tickers...")

    prices = {}
    for ticker in all_tickers:
        try:
            t = yf.Ticker(ticker)
            hist = t.history(start=START_DATE, end=END_DATE, auto_adjust=True)
            if hist.index.tz is not None:
                hist.index = hist.index.tz_convert(None)
            if len(hist) > 100:
                prices[ticker] = hist['Close']
                print(f"  {ticker}: {len(hist)} days")
            else:
                print(f"  {ticker}: SKIP (only {len(hist)} days)")
        except Exception as e:
            print(f"  {ticker}: ERROR ({e})")

    df = pd.DataFrame(prices)
    df = df.dropna(how='all').ffill().bfill()
    print(f"\nPrice matrix: {df.shape[0]} days x {df.shape[1]} tickers")
    print(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")
    return df


def get_monthly_rebalance_indices(index):
    """Get indices (integer positions) of first trading day of each month."""
    months = index.to_period('M')
    rebal_idx = []
    seen = set()
    for i, m in enumerate(months):
        if m not in seen:
            seen.add(m)
            rebal_idx.append(i)
    return rebal_idx


def run_strategy_fast(returns_np, ticker_cols, rebal_indices, mom_matrix, mom_1m_matrix,
                      spy_above_sma, variant, lookback, n_hold,
                      lev_mask, safe_mask):
    """
    Vectorized strategy runner. Works on numpy arrays for speed.

    returns_np: (n_days, n_tickers) array of daily returns
    ticker_cols: list of ticker names
    rebal_indices: list of integer positions for rebalance dates
    mom_matrix: (n_days, n_tickers) momentum values
    mom_1m_matrix: (n_days, n_tickers) 1-month momentum
    spy_above_sma: (n_days,) bool array
    lev_mask: bool array marking leveraged tickers
    safe_mask: bool array marking safety tickers
    """
    n_days = returns_np.shape[0]
    n_tickers = returns_np.shape[1]

    # Weights array: (n_days, n_tickers)
    weights = np.zeros((n_days, n_tickers))

    for ri, idx in enumerate(rebal_indices):
        if idx >= n_days:
            break

        mom_row = mom_matrix[idx]
        mom_1m_row = mom_1m_matrix[idx]

        # Skip if all NaN
        if np.all(np.isnan(mom_row)):
            continue

        # Determine next rebalance
        next_idx = rebal_indices[ri + 1] if ri + 1 < len(rebal_indices) else n_days
        next_idx = min(next_idx, n_days)

        # Build weight vector for this period
        w = np.zeros(n_tickers)

        is_bull = spy_above_sma[idx] if not np.isnan(spy_above_sma[idx]) else True

        if variant == 'pure_momentum':
            lev_mom = np.where(lev_mask & ~np.isnan(mom_row), mom_row, -np.inf)
            top_n = np.argsort(lev_mom)[-n_hold:]
            top_n = top_n[lev_mom[top_n] > -np.inf]
            if len(top_n) > 0:
                w[top_n] = 1.0 / len(top_n)

        elif variant == 'regime_gated':
            if is_bull:
                lev_mom = np.where(lev_mask & ~np.isnan(mom_row), mom_row, -np.inf)
                top_n = np.argsort(lev_mom)[-n_hold:]
                top_n = top_n[lev_mom[top_n] > -np.inf]
                if len(top_n) > 0:
                    w[top_n] = 1.0 / len(top_n)
            else:
                safe_mom = np.where(safe_mask & ~np.isnan(mom_1m_row), mom_1m_row, -np.inf)
                best = np.argmax(safe_mom)
                if safe_mom[best] > -np.inf:
                    w[best] = 1.0

        elif variant == 'dual_momentum':
            if is_bull:
                lev_mom = np.where(lev_mask & ~np.isnan(mom_row), mom_row, -np.inf)
                top_indices = np.argsort(lev_mom)[-n_hold:]
                top_indices = top_indices[lev_mom[top_indices] > -np.inf]

                safe_slots = 0
                for ti in top_indices:
                    if mom_row[ti] > 0:
                        w[ti] = 1.0 / n_hold
                    else:
                        safe_slots += 1

                if safe_slots > 0:
                    safe_mom = np.where(safe_mask & ~np.isnan(mom_1m_row), mom_1m_row, -np.inf)
                    best = np.argmax(safe_mom)
                    if safe_mom[best] > -np.inf:
                        w[best] += safe_slots / n_hold
            else:
                safe_mom = np.where(safe_mask & ~np.isnan(mom_1m_row), mom_1m_row, -np.inf)
                best = np.argmax(safe_mom)
                if safe_mom[best] > -np.inf:
                    w[best] = 1.0

        elif variant == 'conservative':
            lev_mom = np.where(lev_mask & ~np.isnan(mom_row), mom_row, -np.inf)
            top_2 = np.argsort(lev_mom)[-2:]
            top_2 = top_2[lev_mom[top_2] > -np.inf]
            for ti in top_2:
                w[ti] = 1.0 / 3

            safe_mom = np.where(safe_mask & ~np.isnan(mom_1m_row), mom_1m_row, -np.inf)
            best = np.argmax(safe_mom)
            if safe_mom[best] > -np.inf:
                w[best] += 1.0 / 3

        weights[idx:next_idx] = w

    # Compute daily portfolio returns
    port_returns = np.nansum(weights * returns_np, axis=1)
    return port_returns


def prepare_strategy_inputs(prices):
    """Pre-compute all arrays needed by the fast strategy runner."""
    returns = prices.pct_change()
    returns_np = returns.values
    ticker_cols = list(prices.columns)

    rebal_indices = get_monthly_rebalance_indices(prices.index)

    # Pre-compute all momentum matrices
    mom_matrices = {}
    for lb in LOOKBACK_OPTIONS + [21]:  # include 1m for safety
        mom = (prices / prices.shift(lb) - 1).values
        mom_matrices[lb] = mom

    # SPY regime
    spy_idx = ticker_cols.index('SPY')
    spy_prices = prices['SPY'].values
    spy_sma200 = pd.Series(spy_prices).rolling(200).mean().values
    spy_above_sma = (spy_prices > spy_sma200).astype(float)
    spy_above_sma[np.isnan(spy_sma200)] = np.nan

    # Ticker masks
    lev_mask = np.array([t in LEVERAGED for t in ticker_cols])
    safe_mask = np.array([t in SAFETY for t in ticker_cols])

    return {
        'returns_np': returns_np,
        'ticker_cols': ticker_cols,
        'rebal_indices': rebal_indices,
        'mom_matrices': mom_matrices,
        'spy_above_sma': spy_above_sma,
        'lev_mask': lev_mask,
        'safe_mask': safe_mask,
        'index': prices.index,
    }


def run_variant(inputs, variant, lookback=63, n_hold=3):
    """Run a variant using pre-computed inputs."""
    port_ret = run_strategy_fast(
        inputs['returns_np'], inputs['ticker_cols'], inputs['rebal_indices'],
        inputs['mom_matrices'][lookback], inputs['mom_matrices'][21],
        inputs['spy_above_sma'], variant, lookback, n_hold,
        inputs['lev_mask'], inputs['safe_mask']
    )
    ret_series = pd.Series(port_ret, index=inputs['index'])
    # Trim leading zeros
    first_nz = (ret_series != 0).idxmax()
    return ret_series.loc[first_nz:]


def walk_forward(prices, inputs, variant):
    """Walk-forward optimization: 252d train / 21d test, sliding."""
    n_dates = len(prices)
    all_test_returns = []

    i = 0
    n_folds = 0
    while i + TRAIN_DAYS + TEST_DAYS <= n_dates:
        train_end = i + TRAIN_DAYS
        test_end = min(train_end + TEST_DAYS, n_dates)

        best_sharpe = -np.inf
        best_params = (63, 3)

        for lb in LOOKBACK_OPTIONS:
            for nh in N_HOLDINGS_OPTIONS:
                # Run on train slice
                train_ret = run_strategy_fast(
                    inputs['returns_np'][i:train_end], inputs['ticker_cols'],
                    [ri - i for ri in inputs['rebal_indices'] if i <= ri < train_end],
                    inputs['mom_matrices'][lb][i:train_end],
                    inputs['mom_matrices'][21][i:train_end],
                    inputs['spy_above_sma'][i:train_end],
                    variant, lb, nh, inputs['lev_mask'], inputs['safe_mask']
                )
                valid = train_ret[train_ret != 0]
                if len(valid) > 20 and np.std(valid) > 0:
                    sh = np.mean(valid) / np.std(valid) * np.sqrt(252)
                    if sh > best_sharpe:
                        best_sharpe = sh
                        best_params = (lb, nh)

        lb_best, nh_best = best_params

        # Apply on test
        test_ret = run_strategy_fast(
            inputs['returns_np'][i:test_end], inputs['ticker_cols'],
            [ri - i for ri in inputs['rebal_indices'] if i <= ri < test_end],
            inputs['mom_matrices'][lb_best][i:test_end],
            inputs['mom_matrices'][21][i:test_end],
            inputs['spy_above_sma'][i:test_end],
            variant, lb_best, nh_best, inputs['lev_mask'], inputs['safe_mask']
        )

        # Only keep test portion
        test_only = test_ret[train_end - i:test_end - i]
        if len(test_only) > 0:
            test_series = pd.Series(test_only, index=prices.index[train_end:test_end])
            all_test_returns.append(test_series)

        n_folds += 1
        i += TEST_DAYS

    print(f"  {n_folds} WF folds completed")

    if all_test_returns:
        combined = pd.concat(all_test_returns)
        combined = combined[~combined.index.duplicated(keep='first')]
        return combined.sort_index()
    return pd.Series(dtype=float)


def compute_metrics(returns, name="Strategy"):
    """Compute all required metrics."""
    returns = returns.dropna()
    if len(returns) < 20 or returns.std() == 0:
        return {'name': name, 'sharpe': 0, 'sortino': 0, 'cagr': 0, 'max_dd': 0,
                'calmar': 0, 'pf': 0, 'wr': 0, 'n_days': int(len(returns)),
                'annual_vol': 0, 'total_return': 0}

    ann_ret = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    downside = returns[returns < 0]
    downside_vol = downside.std() * np.sqrt(252) if len(downside) > 0 else 1e-10
    sortino = ann_ret / downside_vol

    cum = (1 + returns).cumprod()
    n_years = len(returns) / 252
    cagr = (cum.iloc[-1]) ** (1 / n_years) - 1 if n_years > 0 and cum.iloc[-1] > 0 else 0

    peak = cum.cummax()
    max_dd = ((cum - peak) / peak).min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    pf = gains / losses if losses > 0 else float('inf')
    wr = (returns > 0).sum() / len(returns)

    return {
        'name': name,
        'sharpe': float(round(sharpe, 3)),
        'sortino': float(round(sortino, 3)),
        'cagr': float(round(cagr * 100, 2)),
        'max_dd': float(round(max_dd * 100, 2)),
        'calmar': float(round(calmar, 3)),
        'pf': float(round(pf, 3)),
        'wr': float(round(wr * 100, 2)),
        'n_days': int(len(returns)),
        'annual_vol': float(round(ann_vol * 100, 2)),
        'total_return': float(round((cum.iloc[-1] - 1) * 100, 2)),
    }


def regime_stratification(returns, spy_returns):
    """Stratify by bull/bear/flat regime."""
    spy_20d = spy_returns.rolling(20).sum()
    common = returns.index.intersection(spy_20d.dropna().index)
    r = returns.loc[common]
    s = spy_20d.loc[common]

    results = {}
    for regime, mask in [('bull', s > 0.02), ('bear', s < -0.02), ('flat', (s >= -0.02) & (s <= 0.02))]:
        rets = r[mask]
        if len(rets) > 10 and rets.std() > 0:
            ann_ret = rets.mean() * 252
            ann_vol = rets.std() * np.sqrt(252)
            results[regime] = {
                'sharpe': float(round(ann_ret / ann_vol, 3)),
                'n_days': int(len(rets)),
                'mean_daily_ret_bps': float(round(rets.mean() * 10000, 2)),
            }
        else:
            results[regime] = {'sharpe': 0, 'n_days': int(len(rets)), 'mean_daily_ret_bps': 0}

    sh_bull = results.get('bull', {}).get('sharpe', 0)
    sh_bear = results.get('bear', {}).get('sharpe', 0)
    denom = max(abs(sh_bull), abs(sh_bear), 1e-10)
    r1_ratio = abs(sh_bull - sh_bear) / denom
    results['r1_ratio'] = float(round(r1_ratio, 3))
    results['r1_pass'] = bool(r1_ratio < 0.50)
    return results


def permutation_test_fast(inputs, actual_sharpe, n_perms=200):
    """Shuffle which leveraged ETFs are selected each month (vectorized)."""
    returns_np = inputs['returns_np']
    n_days = returns_np.shape[0]
    lev_indices = np.where(inputs['lev_mask'])[0]
    rebal_indices = inputs['rebal_indices']

    shuffled_sharpes = []

    for _ in range(n_perms):
        weights = np.zeros((n_days, returns_np.shape[1]))

        for ri, idx in enumerate(rebal_indices):
            if idx >= n_days:
                break
            next_idx = rebal_indices[ri + 1] if ri + 1 < len(rebal_indices) else n_days
            next_idx = min(next_idx, n_days)

            # Random 3 from leveraged
            chosen = np.random.choice(lev_indices, size=min(3, len(lev_indices)), replace=False)
            w = np.zeros(returns_np.shape[1])
            w[chosen] = 1.0 / len(chosen)
            weights[idx:next_idx] = w

        port_ret = np.nansum(weights * returns_np, axis=1)
        valid = port_ret[port_ret != 0]
        if len(valid) > 20 and np.std(valid) > 0:
            sh = np.mean(valid) / np.std(valid) * np.sqrt(252)
            shuffled_sharpes.append(sh)

    if not shuffled_sharpes:
        return 1.0

    return float(np.mean([s >= actual_sharpe for s in shuffled_sharpes]))


def sub_period_test(returns):
    """Split in half, both must be profitable."""
    mid = len(returns) // 2
    fh = returns.iloc[:mid]
    sh = returns.iloc[mid:]
    fh_cum = (1 + fh).prod() - 1
    sh_cum = (1 + sh).prod() - 1
    return {
        'first_half_return': float(round(fh_cum * 100, 2)),
        'second_half_return': float(round(sh_cum * 100, 2)),
        'both_profitable': bool(fh_cum > 0 and sh_cum > 0),
        'first_half_dates': f"{fh.index[0].date()} to {fh.index[-1].date()}" if len(fh) > 0 else "N/A",
        'second_half_dates': f"{sh.index[0].date()} to {sh.index[-1].date()}" if len(sh) > 0 else "N/A",
    }


def outlier_test(returns):
    """Remove top 5% months, must still be profitable."""
    monthly = (1 + returns).resample('ME').prod() - 1
    cutoff = monthly.quantile(0.95)
    filtered = monthly[monthly <= cutoff]
    cum = (1 + returns).cumprod()
    return {
        'total_return_with_outliers': float(round((cum.iloc[-1] - 1) * 100, 2)),
        'total_return_without_top5pct_months': float(round((filtered.prod()) * 100 - 100, 2)),
        'still_profitable': bool(filtered.prod() > 1.0),
        'months_removed': int(len(monthly) - len(filtered)),
    }


def main():
    print("=" * 70)
    print("LEVERAGED ETF ROTATION STRATEGY v1 — BACKTEST")
    print("=" * 70)
    print()

    prices = download_data()
    print()

    # Trim: need 200d SMA buffer
    prices = prices.iloc[200:]
    spy_returns = prices['SPY'].pct_change()

    # Pre-compute strategy inputs once
    print("Pre-computing strategy inputs...")
    inputs = prepare_strategy_inputs(prices)
    print(f"Ready. {len(inputs['rebal_indices'])} monthly rebalance dates.\n")

    variants = ['pure_momentum', 'regime_gated', 'dual_momentum', 'conservative']
    results = {}

    # ─── Walk-forward for each variant ───────────────────────────────────
    for variant in variants:
        print(f"\n{'─' * 60}")
        print(f"Walk-forward: {variant}")
        print(f"{'─' * 60}")

        wf_returns = walk_forward(prices, inputs, variant)

        if len(wf_returns) < 50:
            print(f"  WARNING: Only {len(wf_returns)} test days — skipping")
            continue

        print(f"  Test days: {len(wf_returns)} ({wf_returns.index[0].date()} to {wf_returns.index[-1].date()})")

        metrics = compute_metrics(wf_returns, variant)
        print(f"  Sharpe={metrics['sharpe']}, Sortino={metrics['sortino']}, CAGR={metrics['cagr']}%, MaxDD={metrics['max_dd']}%")

        regime = regime_stratification(wf_returns, spy_returns)
        print(f"  Regime R1={regime['r1_ratio']}, PASS={regime['r1_pass']}")

        subperiod = sub_period_test(wf_returns)
        print(f"  Sub-period: H1={subperiod['first_half_return']}%, H2={subperiod['second_half_return']}%, both_profitable={subperiod['both_profitable']}")

        outlier = outlier_test(wf_returns)
        print(f"  Outlier: w/o top 5%={outlier['total_return_without_top5pct_months']}%, still_profitable={outlier['still_profitable']}")

        print(f"  Permutation test (200 shuffles)...")
        p_value = permutation_test_fast(inputs, metrics['sharpe'], N_PERMUTATIONS)
        print(f"  Permutation p-value={p_value}")

        results[variant] = {
            'metrics': metrics,
            'regime': regime,
            'sub_period': subperiod,
            'outlier': outlier,
            'permutation_p_value': float(p_value),
        }

    # ─── Benchmarks ──────────────────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("BENCHMARKS (Buy & Hold)")
    print(f"{'─' * 60}")

    benchmarks = {}
    for ticker in ['SPY', 'QQQ', 'TQQQ']:
        if ticker in prices.columns:
            bm = compute_metrics(prices[ticker].pct_change().dropna(), f"{ticker} B&H")
            benchmarks[ticker] = bm
            print(f"  {ticker}: Sharpe={bm['sharpe']}, CAGR={bm['cagr']}%, MaxDD={bm['max_dd']}%, Sortino={bm['sortino']}")

    # ─── Simple backtest (fixed params) ──────────────────────────────────
    print(f"\n{'─' * 60}")
    print("SIMPLE BACKTEST (fixed: 63d lookback, top 3)")
    print(f"{'─' * 60}")

    simple_results = {}
    for variant in variants:
        rets = run_variant(inputs, variant, lookback=63, n_hold=3)
        if len(rets) > 50:
            m = compute_metrics(rets, f"{variant}_simple")
            simple_results[variant] = m
            print(f"  {variant}: Sharpe={m['sharpe']}, CAGR={m['cagr']}%, MaxDD={m['max_dd']}%")

    # ─── Summary ─────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("FINAL SUMMARY (Walk-Forward OOT)")
    print("=" * 70)

    print(f"\n{'Variant':<20} {'Sharpe':>7} {'Sortino':>8} {'CAGR%':>7} {'MaxDD%':>7} {'Calmar':>7} {'PF':>6} {'WR%':>6} {'R1':>5} {'Perm-p':>7}")
    print("-" * 90)

    for v in variants:
        if v in results:
            r = results[v]
            m = r['metrics']
            print(f"{v:<20} {m['sharpe']:>7.2f} {m['sortino']:>8.2f} {m['cagr']:>7.1f} "
                  f"{m['max_dd']:>7.1f} {m['calmar']:>7.2f} {m['pf']:>6.2f} {m['wr']:>6.1f} "
                  f"{'PASS' if r['regime']['r1_pass'] else 'FAIL':>5} {r['permutation_p_value']:>7.3f}")

    print(f"\n{'Benchmark':<20} {'Sharpe':>7} {'Sortino':>8} {'CAGR%':>7} {'MaxDD%':>7} {'Calmar':>7} {'PF':>6} {'WR%':>6}")
    print("-" * 75)
    for ticker, m in benchmarks.items():
        print(f"{ticker + ' B&H':<20} {m['sharpe']:>7.2f} {m['sortino']:>8.2f} {m['cagr']:>7.1f} "
              f"{m['max_dd']:>7.1f} {m['calmar']:>7.2f} {m['pf']:>6.2f} {m['wr']:>6.1f}")

    # ─── Gate Summary ────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("GATE RESULTS")
    print(f"{'=' * 70}")

    for v in variants:
        if v not in results:
            continue
        r = results[v]
        gates = {
            'R1_regime_agnostic': r['regime']['r1_pass'],
            'sub_period_both_profitable': r['sub_period']['both_profitable'],
            'outlier_still_profitable': r['outlier']['still_profitable'],
            'permutation_significant': r['permutation_p_value'] < 0.10,
            'sharpe_above_0.5': r['metrics']['sharpe'] > 0.5,
            'max_dd_above_neg60': r['metrics']['max_dd'] > -60,
        }
        all_pass = all(gates.values())
        print(f"\n{v}:")
        for gate, passed in gates.items():
            print(f"  {'PASS' if passed else 'FAIL'} -- {gate}")
        print(f"  >>> {'ALL GATES PASS' if all_pass else 'SOME GATES FAILED'}")

        results[v]['gates'] = {k: bool(val) for k, val in gates.items()}
        results[v]['all_gates_pass'] = bool(all_pass)

    # ─── Save ────────────────────────────────────────────────────────────
    output = {
        'strategy': 'Leveraged ETF Rotation v1',
        'run_date': datetime.now().isoformat(),
        'period': f"{prices.index[0].date()} to {prices.index[-1].date()}",
        'walk_forward': {'train_days': TRAIN_DAYS, 'test_days': TEST_DAYS, 'method': 'sliding'},
        'variants': results,
        'benchmarks': benchmarks,
        'simple_backtest': simple_results,
        'universe': {'leveraged': LEVERAGED, 'safety': SAFETY, 'benchmarks': BENCHMARKS},
    }

    output_path = Path('/home/jupiter/Lvl3Quant/research/findings/leveraged_etf_rotation_v1_results.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved.")
    print("DONE.")


if __name__ == '__main__':
    main()
