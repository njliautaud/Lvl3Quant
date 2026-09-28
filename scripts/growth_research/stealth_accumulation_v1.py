#!/usr/bin/env python3
"""
Stealth Accumulation Strategy v1
================================
Detects institutional stealth accumulation: sustained high volume with minimal price movement,
followed by breakouts. A well-documented market microstructure phenomenon.

Signal: N consecutive days where volume > threshold * 20d_avg AND |daily_return| < price_cap.
Entry: close of qualifying day. Hold: 5d or 10d.

Variants sweep: consecutive_days x vol_threshold x price_cap x hold_period x direction_filter
Validation: permutation test, regime gap, per-year consistency.
"""

import json
import os
import sys
import time
import warnings
from datetime import datetime, timedelta
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ── Configuration ──────────────────────────────────────────────────────────
OUTPUT_DIR = Path("/home/nick/Lvl3Quant/output/stealth_accumulation_v1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_PERMUTATIONS = 200
PERM_P_THRESHOLD = 0.05
REGIME_GAP_THRESHOLD = 0.50
YEAR_CONSISTENCY_THRESHOLD = 0.60
SMA_LOOKBACK = 50
VOL_AVG_LOOKBACK = 20
MIN_SIGNALS_PER_VARIANT = 30  # need enough trades to be meaningful

# Variant grid
CONSECUTIVE_DAYS = [3, 5, 7]
VOL_THRESHOLDS = [1.3, 1.5, 2.0]
PRICE_CAPS = [0.003, 0.005]  # 0.3%, 0.5%
HOLD_PERIODS = [5, 10]
DIRECTION_FILTERS = ['all', 'uptrend', 'downtrend']  # all=no filter

# Top ~200 S&P 500 stocks by market cap (hardcoded)
SP500_TOP200 = [
    'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META', 'BRK-B', 'TSLA', 'UNH', 'XOM',
    'JNJ', 'JPM', 'V', 'PG', 'MA', 'HD', 'CVX', 'MRK', 'LLY', 'ABBV',
    'PEP', 'KO', 'AVGO', 'COST', 'TMO', 'MCD', 'WMT', 'CSCO', 'ACN', 'ABT',
    'CRM', 'DHR', 'LIN', 'NEE', 'ADBE', 'TXN', 'AMD', 'PM', 'CMCSA', 'NKE',
    'RTX', 'NFLX', 'ORCL', 'HON', 'UNP', 'QCOM', 'LOW', 'UPS', 'COP', 'BA',
    'INTC', 'GS', 'BMY', 'CAT', 'AMGN', 'ELV', 'SBUX', 'ISRG', 'BLK', 'INTU',
    'MDLZ', 'DE', 'GILD', 'ADP', 'ADI', 'CI', 'SYK', 'REGN', 'TJX', 'BKNG',
    'MMC', 'CB', 'VRTX', 'PLD', 'TMUS', 'SCHW', 'MO', 'DUK', 'SO', 'EOG',
    'ZTS', 'CME', 'BDX', 'CL', 'ITW', 'SLB', 'USB', 'PNC', 'AON', 'WM',
    'TFC', 'ICE', 'BSX', 'LRCX', 'CSX', 'FDX', 'NOC', 'HUM', 'EQIX', 'APD',
    'GD', 'EMR', 'ATVI', 'ORLY', 'MPC', 'PSX', 'VLO', 'F', 'GM', 'SHW',
    'NSC', 'PXD', 'MCK', 'AJG', 'KLAC', 'SNPS', 'CDNS', 'KMB', 'ECL', 'D',
    'ROP', 'AFL', 'AEP', 'EW', 'PSA', 'TRV', 'AIG', 'MSCI', 'SRE', 'GIS',
    'CTVA', 'HCA', 'WELL', 'FIS', 'STZ', 'MNST', 'O', 'MAR', 'AZO', 'MCHP',
    'CTAS', 'DVN', 'ALL', 'PAYX', 'CMI', 'TEL', 'WEC', 'IQV', 'DXCM', 'PCAR',
    'KHC', 'IDXX', 'FTNT', 'NXPI', 'GWW', 'YUM', 'ED', 'OXY', 'HAL', 'BIIB',
    'DLTR', 'KDP', 'XEL', 'FAST', 'PPG', 'AWK', 'GEHC', 'ODFL', 'EA', 'ON',
    'FANG', 'WBD', 'GLW', 'DAL', 'ROST', 'CPRT', 'VRSK', 'HPQ', 'EXC', 'CSGP',
    'BKR', 'GPN', 'WTW', 'DHI', 'LEN', 'APTV', 'KR', 'ACGL', 'HSY', 'DG',
    'MTD', 'ANSS', 'CDW', 'EFX', 'KEYS', 'TSCO', 'WAB', 'IT', 'DOW', 'STT',
]

def download_data(tickers, start='2014-01-01'):
    """Download daily OHLCV for all tickers + SPY."""
    import yfinance as yf
    
    all_tickers = list(set(tickers + ['SPY']))
    print(f"Downloading {len(all_tickers)} tickers from {start}...")
    
    data = yf.download(all_tickers, start=start, auto_adjust=True, threads=True, progress=True)
    
    # yf.download returns multi-level columns: (Price, Ticker)
    # Extract close, volume, high, low
    result = {}
    for ticker in all_tickers:
        try:
            df = pd.DataFrame({
                'Open': data['Open'][ticker] if isinstance(data['Open'], pd.DataFrame) else data['Open'],
                'High': data['High'][ticker] if isinstance(data['High'], pd.DataFrame) else data['High'],
                'Low': data['Low'][ticker] if isinstance(data['Low'], pd.DataFrame) else data['Low'],
                'Close': data['Close'][ticker] if isinstance(data['Close'], pd.DataFrame) else data['Close'],
                'Volume': data['Volume'][ticker] if isinstance(data['Volume'], pd.DataFrame) else data['Volume'],
            }).dropna()
            if len(df) > 100:
                result[ticker] = df
        except Exception:
            continue
    
    print(f"Successfully loaded {len(result)} tickers ({len(result) - 1} stocks + SPY)")
    return result


def compute_spy_regime(spy_data):
    """Classify each day as green (SPY up) or red (SPY down)."""
    spy_ret = spy_data['Close'].pct_change()
    regime = pd.Series('green', index=spy_data.index)
    regime[spy_ret < 0] = 'red'
    regime[spy_ret == 0] = 'flat'
    return regime


def find_stealth_signals(df, consec_days, vol_thresh, price_cap, direction_filter, sma_lookback=50, vol_lookback=20):
    """
    Find stealth accumulation signals in a single stock.
    Returns array of indices where the qualifying streak ends (entry points).
    """
    close = df['Close'].values
    volume = df['Volume'].values
    n = len(close)
    
    if n < max(vol_lookback, sma_lookback) + consec_days + 1:
        return np.array([], dtype=int)
    
    # Daily absolute return
    abs_ret = np.abs(np.diff(close) / close[:-1])
    abs_ret = np.concatenate([[np.nan], abs_ret])
    
    # 20-day average volume (rolling)
    vol_avg = pd.Series(volume).rolling(vol_lookback).mean().values
    
    # Volume ratio
    vol_ratio = volume / np.where(vol_avg > 0, vol_avg, np.nan)
    
    # 50-day SMA for direction filter
    sma = pd.Series(close).rolling(sma_lookback).mean().values
    
    # Qualifying day: vol > threshold * avg AND |return| < cap
    qualifying = (vol_ratio >= vol_thresh) & (abs_ret < price_cap) & (~np.isnan(abs_ret)) & (~np.isnan(vol_ratio))
    
    # Find runs of N consecutive qualifying days
    signals = []
    streak = 0
    for i in range(n):
        if qualifying[i]:
            streak += 1
            if streak >= consec_days:
                # Check direction filter
                if direction_filter == 'all':
                    signals.append(i)
                elif direction_filter == 'uptrend' and not np.isnan(sma[i]) and close[i] > sma[i]:
                    signals.append(i)
                elif direction_filter == 'downtrend' and not np.isnan(sma[i]) and close[i] < sma[i]:
                    signals.append(i)
        else:
            streak = 0
    
    # De-duplicate: only keep the FIRST signal in each streak (avoid overlapping entries)
    if len(signals) == 0:
        return np.array([], dtype=int)
    
    signals = np.array(signals)
    # Remove signals within consec_days of each other (keep first)
    filtered = [signals[0]]
    for s in signals[1:]:
        if s - filtered[-1] >= consec_days:
            filtered.append(s)
    
    return np.array(filtered, dtype=int)


def compute_forward_returns(df, signal_indices, hold_days):
    """Compute forward returns for each signal."""
    close = df['Close'].values
    n = len(close)
    returns = []
    valid_indices = []
    dates = []
    
    for idx in signal_indices:
        exit_idx = idx + hold_days
        if exit_idx < n:
            ret = (close[exit_idx] - close[idx]) / close[idx]
            returns.append(ret)
            valid_indices.append(idx)
            dates.append(df.index[idx])
    
    return np.array(returns), valid_indices, dates


def compute_metrics(returns):
    """Compute strategy metrics from an array of trade returns."""
    if len(returns) < 5:
        return None
    
    mean_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1)
    
    wr = np.mean(returns > 0)
    
    # Annualized Sharpe (assume ~252/hold_days trades per year)
    sharpe = (mean_ret / std_ret) * np.sqrt(252) if std_ret > 0 else 0.0
    
    # Sortino
    downside = returns[returns < 0]
    downside_std = np.std(downside, ddof=1) if len(downside) > 1 else std_ret
    sortino = (mean_ret / downside_std) * np.sqrt(252) if downside_std > 0 else 0.0
    
    # Profit Factor
    gross_profit = np.sum(returns[returns > 0])
    gross_loss = np.abs(np.sum(returns[returns < 0]))
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    return {
        'n_trades': len(returns),
        'mean_return': float(mean_ret),
        'std_return': float(std_ret),
        'win_rate': float(wr),
        'sharpe': float(sharpe),
        'sortino': float(sortino),
        'profit_factor': float(pf),
        'total_return': float(np.sum(returns)),
        'max_drawdown': float(np.min(np.minimum.accumulate(np.cumsum(returns)) - np.cumsum(returns))),
    }


def permutation_test(returns, n_perms=200):
    """
    Permutation test: shuffle returns, compute mean. 
    p-value = fraction of shuffled means >= observed mean.
    """
    observed = np.mean(returns)
    rng = np.random.default_rng(42)
    count_ge = 0
    for _ in range(n_perms):
        shuffled = rng.choice(returns, size=len(returns), replace=True)
        # Actually: permutation test shuffles the sign (or timing), not bootstrap.
        # Classic approach: randomly flip signs to test if mean is significantly positive.
        signs = rng.choice([-1, 1], size=len(returns))
        perm_mean = np.mean(returns * signs)
        if perm_mean >= observed:
            count_ge += 1
    return count_ge / n_perms


def regime_test(returns, dates, spy_regime):
    """
    Split returns by SPY regime on entry day.
    Compute Sharpe for green vs red days.
    Returns regime gap ratio.
    """
    green_rets = []
    red_rets = []
    
    for ret, dt in zip(returns, dates):
        if dt in spy_regime.index:
            r = spy_regime.loc[dt]
            if r == 'green':
                green_rets.append(ret)
            elif r == 'red':
                red_rets.append(ret)
    
    green_rets = np.array(green_rets)
    red_rets = np.array(red_rets)
    
    def _sharpe(r):
        if len(r) < 3:
            return 0.0
        s = np.std(r, ddof=1)
        return (np.mean(r) / s) * np.sqrt(252) if s > 0 else 0.0
    
    sharpe_green = _sharpe(green_rets)
    sharpe_red = _sharpe(red_rets)
    
    max_abs = max(abs(sharpe_green), abs(sharpe_red))
    gap = abs(sharpe_green - sharpe_red) / max_abs if max_abs > 0 else 0.0
    
    return {
        'sharpe_green': float(sharpe_green),
        'sharpe_red': float(sharpe_red),
        'n_green': len(green_rets),
        'n_red': len(red_rets),
        'regime_gap': float(gap),
        'pass': gap < REGIME_GAP_THRESHOLD,
    }


def year_consistency_test(returns, dates):
    """Check if strategy is profitable in >60% of years."""
    yearly = {}
    for ret, dt in zip(returns, dates):
        y = dt.year
        if y not in yearly:
            yearly[y] = []
        yearly[y].append(ret)
    
    if len(yearly) < 3:
        return {'pass': False, 'reason': 'too_few_years', 'n_years': len(yearly)}
    
    profitable_years = sum(1 for rets in yearly.values() if np.sum(rets) > 0)
    ratio = profitable_years / len(yearly)
    
    per_year = {str(y): {'n_trades': len(r), 'total_return': float(np.sum(r)), 'mean_return': float(np.mean(r))}
                for y, r in sorted(yearly.items())}
    
    return {
        'profitable_years': profitable_years,
        'total_years': len(yearly),
        'consistency_ratio': float(ratio),
        'pass': ratio >= YEAR_CONSISTENCY_THRESHOLD,
        'per_year': per_year,
    }


def run_variant(data, spy_regime, consec_days, vol_thresh, price_cap, hold_days, direction_filter):
    """Run a single variant across all stocks."""
    all_returns = []
    all_dates = []
    per_stock = {}
    
    for ticker, df in data.items():
        if ticker == 'SPY':
            continue
        
        signals = find_stealth_signals(df, consec_days, vol_thresh, price_cap, direction_filter)
        if len(signals) == 0:
            continue
        
        rets, valid_idx, dates = compute_forward_returns(df, signals, hold_days)
        if len(rets) == 0:
            continue
        
        all_returns.extend(rets)
        all_dates.extend(dates)
        per_stock[ticker] = {
            'n_signals': len(rets),
            'mean_return': float(np.mean(rets)),
            'win_rate': float(np.mean(rets > 0)),
        }
    
    all_returns = np.array(all_returns)
    
    variant_name = f"d{consec_days}_v{vol_thresh}_p{price_cap}_h{hold_days}_{direction_filter}"
    
    if len(all_returns) < MIN_SIGNALS_PER_VARIANT:
        return {
            'variant': variant_name,
            'status': 'insufficient_signals',
            'n_signals': len(all_returns),
            'params': {
                'consec_days': consec_days,
                'vol_threshold': vol_thresh,
                'price_cap': price_cap,
                'hold_days': hold_days,
                'direction_filter': direction_filter,
            },
        }
    
    # Core metrics
    metrics = compute_metrics(all_returns)
    
    # Validation gates
    perm_p = permutation_test(all_returns, N_PERMUTATIONS)
    regime = regime_test(all_returns, all_dates, spy_regime)
    yearly = year_consistency_test(all_returns, all_dates)
    
    passes_all = (perm_p < PERM_P_THRESHOLD) and regime['pass'] and yearly['pass']
    
    return {
        'variant': variant_name,
        'status': 'PASS' if passes_all else 'FAIL',
        'params': {
            'consec_days': consec_days,
            'vol_threshold': vol_thresh,
            'price_cap': price_cap,
            'hold_days': hold_days,
            'direction_filter': direction_filter,
        },
        'metrics': metrics,
        'validation': {
            'permutation_p': float(perm_p),
            'permutation_pass': perm_p < PERM_P_THRESHOLD,
            'regime': regime,
            'year_consistency': yearly,
        },
        'n_stocks_with_signals': len(per_stock),
        'top_stocks': dict(sorted(per_stock.items(), key=lambda x: x[1]['n_signals'], reverse=True)[:10]),
    }


def main():
    t0 = time.time()
    print("=" * 70)
    print("STEALTH ACCUMULATION STRATEGY v1")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Download data
    data = download_data(SP500_TOP200)
    
    # SPY regime
    spy_regime = compute_spy_regime(data['SPY'])
    print(f"SPY regime: {(spy_regime == 'green').sum()} green, {(spy_regime == 'red').sum()} red days")
    
    # Build variant grid
    variants = list(product(CONSECUTIVE_DAYS, VOL_THRESHOLDS, PRICE_CAPS, HOLD_PERIODS, DIRECTION_FILTERS))
    print(f"\nRunning {len(variants)} variants across {len(data) - 1} stocks...")
    
    results = []
    for i, (cd, vt, pc, hp, df_filt) in enumerate(variants):
        label = f"d{cd}_v{vt}_p{pc}_h{hp}_{df_filt}"
        print(f"  [{i+1}/{len(variants)}] {label} ...", end=" ", flush=True)
        
        result = run_variant(data, spy_regime, cd, vt, pc, hp, df_filt)
        results.append(result)
        
        if result['status'] == 'insufficient_signals':
            print(f"SKIP ({result['n_signals']} signals)")
        else:
            m = result['metrics']
            print(f"{result['status']} | n={m['n_trades']} WR={m['win_rate']:.1%} "
                  f"Sharpe={m['sharpe']:.2f} Sortino={m['sortino']:.2f} PF={m['profit_factor']:.2f} "
                  f"perm_p={result['validation']['permutation_p']:.3f}")
    
    # Summary
    passing = [r for r in results if r['status'] == 'PASS']
    failing = [r for r in results if r['status'] == 'FAIL']
    skipped = [r for r in results if r['status'] == 'insufficient_signals']
    
    print("\n" + "=" * 70)
    print(f"RESULTS SUMMARY")
    print(f"  Total variants: {len(results)}")
    print(f"  PASS (all 3 gates): {len(passing)}")
    print(f"  FAIL: {len(failing)}")
    print(f"  Skipped (insufficient signals): {len(skipped)}")
    
    if passing:
        print("\n--- PASSING VARIANTS (sorted by Sharpe) ---")
        passing_sorted = sorted(passing, key=lambda x: x['metrics']['sharpe'], reverse=True)
        for r in passing_sorted:
            m = r['metrics']
            v = r['validation']
            print(f"  {r['variant']:40s} | n={m['n_trades']:4d} WR={m['win_rate']:.1%} "
                  f"Sharpe={m['sharpe']:.2f} Sortino={m['sortino']:.2f} PF={m['profit_factor']:.2f} "
                  f"perm_p={v['permutation_p']:.3f} regime_gap={v['regime']['regime_gap']:.2f} "
                  f"yr_consist={v['year_consistency']['consistency_ratio']:.0%}")
    
    if failing:
        print("\n--- TOP FAILING VARIANTS (by Sharpe, for reference) ---")
        failing_sorted = sorted(failing, key=lambda x: x['metrics']['sharpe'], reverse=True)
        for r in failing_sorted[:5]:
            m = r['metrics']
            v = r['validation']
            fail_reasons = []
            if v['permutation_p'] >= PERM_P_THRESHOLD:
                fail_reasons.append(f"perm_p={v['permutation_p']:.3f}")
            if not v['regime']['pass']:
                fail_reasons.append(f"regime_gap={v['regime']['regime_gap']:.2f}")
            if not v['year_consistency']['pass']:
                fail_reasons.append(f"yr_consist={v['year_consistency']['consistency_ratio']:.0%}")
            print(f"  {r['variant']:40s} | Sharpe={m['sharpe']:.2f} FAIL: {', '.join(fail_reasons)}")
    
    elapsed = time.time() - t0
    print(f"\nTotal runtime: {elapsed/60:.1f} minutes")
    
    # Save results
    output = {
        'metadata': {
            'strategy': 'stealth_accumulation_v1',
            'run_date': datetime.now().isoformat(),
            'runtime_minutes': round(elapsed / 60, 1),
            'n_stocks': len(data) - 1,
            'n_variants': len(results),
            'n_passing': len(passing),
            'n_failing': len(failing),
            'n_skipped': len(skipped),
            'validation_gates': {
                'permutation_test': f'{N_PERMUTATIONS} shuffles, p < {PERM_P_THRESHOLD}',
                'regime_gap': f'< {REGIME_GAP_THRESHOLD}',
                'year_consistency': f'> {YEAR_CONSISTENCY_THRESHOLD}',
            },
        },
        'passing_variants': sorted(passing, key=lambda x: x['metrics']['sharpe'], reverse=True),
        'failing_variants': sorted(failing, key=lambda x: x['metrics']['sharpe'], reverse=True),
        'skipped_variants': skipped,
    }
    
    results_path = OUTPUT_DIR / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")
    
    # Also save a compact summary CSV for quick review
    rows = []
    for r in results:
        if r['status'] == 'insufficient_signals':
            continue
        m = r['metrics']
        v = r['validation']
        rows.append({
            'variant': r['variant'],
            'status': r['status'],
            'n_trades': m['n_trades'],
            'mean_return': m['mean_return'],
            'win_rate': m['win_rate'],
            'sharpe': m['sharpe'],
            'sortino': m['sortino'],
            'profit_factor': m['profit_factor'],
            'perm_p': v['permutation_p'],
            'regime_gap': v['regime']['regime_gap'],
            'yr_consistency': v['year_consistency']['consistency_ratio'],
        })
    
    if rows:
        summary_df = pd.DataFrame(rows).sort_values('sharpe', ascending=False)
        summary_path = OUTPUT_DIR / 'summary.csv'
        summary_df.to_csv(summary_path, index=False)
        print(f"Summary CSV saved to {summary_path}")
    
    print("\nDONE.")


if __name__ == '__main__':
    main()
