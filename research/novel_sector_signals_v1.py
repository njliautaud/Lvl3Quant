#!/usr/bin/env python3
"""
Novel Sector Rotation Signal Research — Three Creative Ideas
============================================================
1. Seasonal Sector Rotation (month-of-year calendar effects)
2. International Leading Indicator (EFA/EEM lead US sectors)
3. Dollar Strength Sector Impact (UUP predicts sector rotation)

Each backtest:
- 2+ years of data, sliding windows, no look-ahead
- FIFO-realistic costs (0.03% round-trip for ETFs)
- Sharpe, win rate, profit factor
- Regime stratification (green vs red SPY days)
- 1000-iteration permutation test
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import json
import os

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
SECTOR_ETFS = ['XLF', 'XLE', 'XLU', 'XLK', 'XLY', 'XLP', 'XLRE', 'XLV', 'XLI', 'XLB', 'XLC']
BENCHMARK = 'SPY'
COST_RT_PCT = 0.03 / 100  # 0.03% round-trip cost for ETFs
HOLD_DAYS = 5
START_DATE = '2021-01-01'
END_DATE = '2026-08-15'
N_PERMUTATIONS = 1000
RESULTS_DIR = '/home/jupiter/Lvl3Quant/research/'

np.random.seed(42)

# ============================================================
# DATA DOWNLOAD
# ============================================================
def download_data():
    """Download all required data."""
    tickers = SECTOR_ETFS + [BENCHMARK, 'EFA', 'EEM', 'UUP']
    print(f"Downloading {len(tickers)} tickers...")
    data = yf.download(tickers, start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)

    # Handle multi-level columns
    if isinstance(data.columns, pd.MultiIndex):
        close = data['Close']
    else:
        close = data

    # Forward-fill missing data, then drop any remaining NaN rows
    close = close.ffill().dropna()
    print(f"Data: {close.index[0].strftime('%Y-%m-%d')} to {close.index[-1].strftime('%Y-%m-%d')}, {len(close)} trading days")
    return close


# ============================================================
# COMMON UTILITIES
# ============================================================
def compute_forward_returns(close, ticker, hold_days=HOLD_DAYS):
    """Compute forward returns for a ticker over hold_days."""
    return close[ticker].pct_change(hold_days).shift(-hold_days)


def compute_excess_returns(close, ticker, hold_days=HOLD_DAYS):
    """Compute excess returns vs SPY over hold_days."""
    sector_ret = close[ticker].pct_change(hold_days).shift(-hold_days)
    spy_ret = close[BENCHMARK].pct_change(hold_days).shift(-hold_days)
    return sector_ret - spy_ret


def compute_metrics(returns, cost_per_trade=COST_RT_PCT):
    """Compute Sharpe, win rate, profit factor from a series of trade returns."""
    # Apply costs
    net_returns = returns - cost_per_trade
    net_returns = net_returns.dropna()

    if len(net_returns) < 10:
        return {'sharpe': 0, 'win_rate': 0, 'profit_factor': 0, 'n_trades': 0,
                'avg_ret': 0, 'total_ret': 0}

    wins = net_returns[net_returns > 0]
    losses = net_returns[net_returns < 0]

    sharpe = net_returns.mean() / net_returns.std() * np.sqrt(252 / HOLD_DAYS) if net_returns.std() > 0 else 0
    win_rate = len(wins) / len(net_returns) if len(net_returns) > 0 else 0
    profit_factor = wins.sum() / abs(losses.sum()) if len(losses) > 0 and losses.sum() != 0 else 0

    return {
        'sharpe': round(sharpe, 3),
        'win_rate': round(win_rate, 4),
        'profit_factor': round(profit_factor, 3),
        'n_trades': len(net_returns),
        'avg_ret': round(net_returns.mean() * 100, 4),
        'total_ret': round(net_returns.sum() * 100, 2)
    }


def regime_stratify(returns, spy_close, cost_per_trade=COST_RT_PCT):
    """Split returns by green/red SPY days (using same-day SPY return)."""
    spy_daily = spy_close.pct_change()

    green_mask = spy_daily > 0
    red_mask = spy_daily <= 0

    green_returns = returns[green_mask].dropna()
    red_returns = returns[red_mask].dropna()

    green_metrics = compute_metrics(green_returns, cost_per_trade)
    red_metrics = compute_metrics(red_returns, cost_per_trade)

    # Regime gap check
    s_green = green_metrics['sharpe']
    s_red = red_metrics['sharpe']
    max_s = max(abs(s_green), abs(s_red))
    regime_gap = abs(s_green - s_red) / max_s if max_s > 0 else 0

    return {
        'green': green_metrics,
        'red': red_metrics,
        'regime_gap': round(regime_gap, 3)
    }


def permutation_test(signal_returns, all_returns, n_perms=N_PERMUTATIONS):
    """Permutation test: shuffle signal assignments, compute Sharpe distribution."""
    actual_sharpe = compute_metrics(signal_returns)['sharpe']

    n_trades = len(signal_returns)
    all_returns_clean = all_returns.dropna()

    if len(all_returns_clean) < n_trades or n_trades < 10:
        return {'actual_sharpe': actual_sharpe, 'p_value': 1.0, 'perm_mean': 0, 'perm_std': 0}

    perm_sharpes = []
    for _ in range(n_perms):
        random_idx = np.random.choice(len(all_returns_clean), size=n_trades, replace=False)
        random_returns = all_returns_clean.iloc[random_idx]
        perm_sharpe = compute_metrics(random_returns)['sharpe']
        perm_sharpes.append(perm_sharpe)

    perm_sharpes = np.array(perm_sharpes)
    p_value = np.mean(perm_sharpes >= actual_sharpe)

    return {
        'actual_sharpe': round(actual_sharpe, 3),
        'p_value': round(p_value, 4),
        'perm_mean': round(np.mean(perm_sharpes), 3),
        'perm_std': round(np.std(perm_sharpes), 3)
    }


# ============================================================
# STRATEGY 1: SEASONAL SECTOR ROTATION
# ============================================================
def test_seasonal_rotation(close):
    """
    Test if month-of-year predicts which sectors outperform SPY.

    Method:
    - For each sector, compute historical monthly excess returns vs SPY
    - Use a SLIDING 2-year lookback to compute average excess return per month
    - Signal: go long sector ETF when current month has historically positive excess return
    - Hold for 5 trading days
    """
    print("\n" + "="*70)
    print("STRATEGY 1: SEASONAL SECTOR ROTATION")
    print("="*70)

    all_trade_returns = []
    sector_results = {}

    for sector in SECTOR_ETFS:
        # Compute daily excess returns vs SPY
        sector_ret = close[sector].pct_change()
        spy_ret = close[BENCHMARK].pct_change()
        excess_daily = sector_ret - spy_ret

        # Forward 5-day sector returns (absolute, not excess — we trade the ETF)
        fwd_ret = close[sector].pct_change(HOLD_DAYS).shift(-HOLD_DAYS)

        months = close.index.to_series().dt.month

        trade_returns = []
        trade_dates = []

        # Sliding window: use past 2 years (504 trading days) to estimate monthly effect
        lookback = 504

        for i in range(lookback, len(close) - HOLD_DAYS):
            # Only trade on the 1st trading day of each month
            if i > 0 and months.iloc[i] == months.iloc[i-1]:
                continue

            current_month = months.iloc[i]

            # Look back 2 years to compute average excess return for this month
            hist_window = excess_daily.iloc[max(0, i-lookback):i]
            hist_months = months.iloc[max(0, i-lookback):i]

            month_mask = hist_months == current_month
            month_excess = hist_window[month_mask]

            if len(month_excess) < 15:  # Need at least 15 observations
                continue

            avg_excess = month_excess.mean()

            # Signal: go long if this month historically has positive excess return
            if avg_excess > 0:
                ret = fwd_ret.iloc[i]
                if not np.isnan(ret):
                    trade_returns.append(ret)
                    trade_dates.append(close.index[i])

        if len(trade_returns) > 0:
            trade_series = pd.Series(trade_returns, index=trade_dates)
            metrics = compute_metrics(trade_series)
            regime = regime_stratify(trade_series, close[BENCHMARK])

            # For permutation test, use all forward returns as the pool
            all_fwd = fwd_ret.dropna()
            perm = permutation_test(trade_series, all_fwd)

            sector_results[sector] = {
                'metrics': metrics,
                'regime': regime,
                'permutation': perm
            }
            all_trade_returns.extend(trade_returns)

            print(f"  {sector}: Sharpe={metrics['sharpe']:.3f}, WR={metrics['win_rate']:.1%}, "
                  f"PF={metrics['profit_factor']:.2f}, N={metrics['n_trades']}, "
                  f"p={perm['p_value']:.3f}, RegimeGap={regime['regime_gap']:.2f}")

    # Aggregate across all sectors
    if all_trade_returns:
        agg_series = pd.Series(all_trade_returns)
        agg_metrics = compute_metrics(agg_series)
        print(f"\n  AGGREGATE: Sharpe={agg_metrics['sharpe']:.3f}, WR={agg_metrics['win_rate']:.1%}, "
              f"PF={agg_metrics['profit_factor']:.2f}, N={agg_metrics['n_trades']}")

    return sector_results


# ============================================================
# STRATEGY 2: INTERNATIONAL LEADING INDICATOR
# ============================================================
def test_international_leading(close):
    """
    Test if EFA/EEM momentum leads US sector rotation by 1-3 days.

    Hypotheses:
    - When EEM (emerging) is strong → materials (XLB), energy (XLE), industrials (XLI) outperform
    - When EFA (developed) is weak → defensives (XLU, XLP, XLV) outperform
    - When both are strong → cyclicals (XLY, XLK, XLF) outperform

    Method:
    - Compute 5-day momentum of EFA and EEM
    - Map to sector predictions using predefined relationships
    - Go long predicted outperformers, hold 5 days
    """
    print("\n" + "="*70)
    print("STRATEGY 2: INTERNATIONAL LEADING INDICATOR")
    print("="*70)

    # Compute 5-day momentum for international ETFs
    efa_mom = close['EFA'].pct_change(5)
    eem_mom = close['EEM'].pct_change(5)

    # Define sector sensitivity to international signals
    # Positive = benefits from strong international, Negative = benefits from weak international
    eem_sensitivity = {
        'XLB': 1, 'XLE': 1, 'XLI': 1,  # Commodity/export exposed
        'XLK': 0.5, 'XLF': 0.5, 'XLY': 0.5,  # Moderate global exposure
        'XLC': 0, 'XLV': -0.5,  # Mixed
        'XLU': -1, 'XLP': -1, 'XLRE': -1  # Domestic/defensive
    }

    efa_sensitivity = {
        'XLK': 1, 'XLI': 1, 'XLF': 0.8,  # Developed market correlated
        'XLB': 0.5, 'XLE': 0.3, 'XLY': 0.5,
        'XLC': 0, 'XLV': -0.3,
        'XLU': -0.8, 'XLP': -0.8, 'XLRE': -0.5
    }

    all_trade_returns = []
    sector_results = {}

    for sector in SECTOR_ETFS:
        fwd_ret = close[sector].pct_change(HOLD_DAYS).shift(-HOLD_DAYS)

        trade_returns = []
        trade_dates = []

        eem_sens = eem_sensitivity.get(sector, 0)
        efa_sens = efa_sensitivity.get(sector, 0)

        # Use sliding 1-year lookback to adaptively scale signal
        lookback = 252

        for i in range(lookback, len(close) - HOLD_DAYS):
            # Skip weekends/holidays (only trade every 5 days to avoid overlapping trades)
            if i % 5 != 0:
                continue

            eem_val = eem_mom.iloc[i]
            efa_val = efa_mom.iloc[i]

            if np.isnan(eem_val) or np.isnan(efa_val):
                continue

            # Composite signal: weighted combination
            # Use z-score relative to lookback window
            eem_hist = eem_mom.iloc[i-lookback:i].dropna()
            efa_hist = efa_mom.iloc[i-lookback:i].dropna()

            if len(eem_hist) < 50 or len(efa_hist) < 50:
                continue

            eem_z = (eem_val - eem_hist.mean()) / eem_hist.std() if eem_hist.std() > 0 else 0
            efa_z = (efa_val - efa_hist.mean()) / efa_hist.std() if efa_hist.std() > 0 else 0

            # Composite score: how much this sector should benefit from current intl regime
            score = eem_sens * eem_z + efa_sens * efa_z

            # Go long if score > 0.5 (moderate conviction threshold)
            if score > 0.5:
                ret = fwd_ret.iloc[i]
                if not np.isnan(ret):
                    trade_returns.append(ret)
                    trade_dates.append(close.index[i])

        if len(trade_returns) > 0:
            trade_series = pd.Series(trade_returns, index=trade_dates)
            metrics = compute_metrics(trade_series)
            regime = regime_stratify(trade_series, close[BENCHMARK])

            all_fwd = fwd_ret.dropna()
            perm = permutation_test(trade_series, all_fwd)

            sector_results[sector] = {
                'metrics': metrics,
                'regime': regime,
                'permutation': perm
            }
            all_trade_returns.extend(trade_returns)

            print(f"  {sector}: Sharpe={metrics['sharpe']:.3f}, WR={metrics['win_rate']:.1%}, "
                  f"PF={metrics['profit_factor']:.2f}, N={metrics['n_trades']}, "
                  f"p={perm['p_value']:.3f}, RegimeGap={regime['regime_gap']:.2f}")
        else:
            print(f"  {sector}: No trades generated")

    if all_trade_returns:
        agg_series = pd.Series(all_trade_returns)
        agg_metrics = compute_metrics(agg_series)
        print(f"\n  AGGREGATE: Sharpe={agg_metrics['sharpe']:.3f}, WR={agg_metrics['win_rate']:.1%}, "
              f"PF={agg_metrics['profit_factor']:.2f}, N={agg_metrics['n_trades']}")

    return sector_results


# ============================================================
# STRATEGY 3: DOLLAR STRENGTH SECTOR IMPACT
# ============================================================
def test_dollar_strength(close):
    """
    Test if USD strength (via UUP) predicts sector rotation.

    Hypothesis:
    - Strong dollar hurts exporters/commodity sectors: XLI, XLB, XLE, XLK
    - Strong dollar helps domestics/importers: XLU, XLP, XLRE, XLV

    Method:
    - Compute 5-day UUP change (dollar strength proxy)
    - When dollar strengthens: go long defensives, avoid cyclicals
    - When dollar weakens: go long cyclicals, avoid defensives
    - Hold 5 days
    """
    print("\n" + "="*70)
    print("STRATEGY 3: DOLLAR STRENGTH SECTOR IMPACT")
    print("="*70)

    uup_mom = close['UUP'].pct_change(5)

    # Dollar sensitivity: negative = hurt by strong dollar, positive = helped by strong dollar
    dollar_sensitivity = {
        'XLE': -1.0,   # Energy: commodities priced in USD
        'XLB': -0.8,   # Materials: commodity exposed
        'XLI': -0.7,   # Industrials: export exposed
        'XLK': -0.5,   # Tech: significant foreign revenue
        'XLY': -0.3,   # Consumer discretionary: mixed
        'XLF': 0.0,    # Financials: mixed (rate differential)
        'XLC': -0.2,   # Communications: mixed
        'XLV': 0.3,    # Healthcare: some domestic, some global
        'XLP': 0.5,    # Consumer staples: domestic focus
        'XLU': 0.8,    # Utilities: pure domestic
        'XLRE': 0.7    # Real estate: domestic
    }

    all_trade_returns = []
    sector_results = {}

    for sector in SECTOR_ETFS:
        fwd_ret = close[sector].pct_change(HOLD_DAYS).shift(-HOLD_DAYS)

        trade_returns = []
        trade_dates = []

        sens = dollar_sensitivity.get(sector, 0)
        lookback = 252

        for i in range(lookback, len(close) - HOLD_DAYS):
            if i % 5 != 0:
                continue

            uup_val = uup_mom.iloc[i]
            if np.isnan(uup_val):
                continue

            # Z-score the dollar move
            uup_hist = uup_mom.iloc[i-lookback:i].dropna()
            if len(uup_hist) < 50:
                continue

            uup_z = (uup_val - uup_hist.mean()) / uup_hist.std() if uup_hist.std() > 0 else 0

            # Signal logic:
            # If dollar is strong (uup_z > 0) and sector is helped by strong dollar (sens > 0) → long
            # If dollar is weak (uup_z < 0) and sector is hurt by strong dollar (sens < 0) → long (weak $ helps them)
            # In both cases: signal = uup_z * sens
            signal = uup_z * sens

            # Go long when signal > 0.5
            if signal > 0.5:
                ret = fwd_ret.iloc[i]
                if not np.isnan(ret):
                    trade_returns.append(ret)
                    trade_dates.append(close.index[i])

        if len(trade_returns) > 0:
            trade_series = pd.Series(trade_returns, index=trade_dates)
            metrics = compute_metrics(trade_series)
            regime = regime_stratify(trade_series, close[BENCHMARK])

            all_fwd = fwd_ret.dropna()
            perm = permutation_test(trade_series, all_fwd)

            sector_results[sector] = {
                'metrics': metrics,
                'regime': regime,
                'permutation': perm
            }
            all_trade_returns.extend(trade_returns)

            print(f"  {sector}: Sharpe={metrics['sharpe']:.3f}, WR={metrics['win_rate']:.1%}, "
                  f"PF={metrics['profit_factor']:.2f}, N={metrics['n_trades']}, "
                  f"p={perm['p_value']:.3f}, RegimeGap={regime['regime_gap']:.2f}")
        else:
            print(f"  {sector}: No trades generated")

    if all_trade_returns:
        agg_series = pd.Series(all_trade_returns)
        agg_metrics = compute_metrics(agg_series)
        print(f"\n  AGGREGATE: Sharpe={agg_metrics['sharpe']:.3f}, WR={agg_metrics['win_rate']:.1%}, "
              f"PF={agg_metrics['profit_factor']:.2f}, N={agg_metrics['n_trades']}")

    return sector_results


# ============================================================
# MAIN
# ============================================================
def main():
    print("="*70)
    print("NOVEL SECTOR ROTATION SIGNAL RESEARCH")
    print(f"Period: {START_DATE} to {END_DATE}")
    print(f"Hold period: {HOLD_DAYS} days, Cost: {COST_RT_PCT*100:.3f}% RT")
    print(f"Permutation tests: {N_PERMUTATIONS} iterations")
    print("="*70)

    close = download_data()

    # Run all three strategies
    results = {}

    results['seasonal_rotation'] = test_seasonal_rotation(close)
    results['international_leading'] = test_international_leading(close)
    results['dollar_strength'] = test_dollar_strength(close)

    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "="*70)
    print("SUMMARY: PROMISING SIGNALS (Sharpe > 0.5, p < 0.05, regime_gap < 0.50)")
    print("="*70)

    promising = []
    for strategy_name, strategy_results in results.items():
        for sector, res in strategy_results.items():
            m = res['metrics']
            p = res['permutation']
            r = res['regime']

            if m['sharpe'] > 0.5 and p['p_value'] < 0.05 and r['regime_gap'] < 0.50:
                promising.append({
                    'strategy': strategy_name,
                    'sector': sector,
                    'sharpe': m['sharpe'],
                    'win_rate': m['win_rate'],
                    'profit_factor': m['profit_factor'],
                    'n_trades': m['n_trades'],
                    'p_value': p['p_value'],
                    'regime_gap': r['regime_gap']
                })
                print(f"  ** {strategy_name} / {sector}: Sharpe={m['sharpe']:.3f}, "
                      f"WR={m['win_rate']:.1%}, PF={m['profit_factor']:.2f}, "
                      f"p={p['p_value']:.3f}, RegimeGap={r['regime_gap']:.2f}")

    if not promising:
        print("  No signals passed all three filters.")

    # Also show near-misses (Sharpe > 0.3 OR p < 0.10)
    print("\n  NEAR-MISSES (Sharpe > 0.3 OR p < 0.10):")
    near_misses = []
    for strategy_name, strategy_results in results.items():
        for sector, res in strategy_results.items():
            m = res['metrics']
            p = res['permutation']
            r = res['regime']
            key = f"{strategy_name}/{sector}"

            if key not in [f"{x['strategy']}/{x['sector']}" for x in promising]:
                if m['sharpe'] > 0.3 or p['p_value'] < 0.10:
                    near_misses.append({
                        'strategy': strategy_name,
                        'sector': sector,
                        'sharpe': m['sharpe'],
                        'win_rate': m['win_rate'],
                        'profit_factor': m['profit_factor'],
                        'p_value': p['p_value'],
                        'regime_gap': r['regime_gap']
                    })
                    print(f"    {strategy_name} / {sector}: Sharpe={m['sharpe']:.3f}, "
                          f"WR={m['win_rate']:.1%}, PF={m['profit_factor']:.2f}, "
                          f"p={p['p_value']:.3f}, RegimeGap={r['regime_gap']:.2f}")

    if not near_misses:
        print("    None.")

    # Save results to JSON
    output = {
        'config': {
            'start_date': START_DATE,
            'end_date': END_DATE,
            'hold_days': HOLD_DAYS,
            'cost_rt_pct': COST_RT_PCT * 100,
            'n_permutations': N_PERMUTATIONS
        },
        'results': {},
        'promising': promising,
        'near_misses': near_misses
    }

    for strategy_name, strategy_results in results.items():
        output['results'][strategy_name] = {}
        for sector, res in strategy_results.items():
            output['results'][strategy_name][sector] = res

    output_path = os.path.join(RESULTS_DIR, 'novel_sector_signals_v1_results.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
