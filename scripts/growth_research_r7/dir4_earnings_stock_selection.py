#!/usr/bin/env python3
"""
R7 DIRECTION 4: Earnings-Driven Stock Selection (PEAD / Quality Factor)

Better approach than simple PEAD:
- Use earnings yield + momentum as quality signal
- Walk-forward factor model: rank stocks, go long top decile
- Monthly rebalance, hold 10-20 stocks

Since actual earnings surprise data is unreliable from yfinance,
we use fundamental-quality proxies that capture similar effects:
- Earnings yield (E/P from PE ratio)
- Revenue growth momentum (price as proxy)
- Quality: low vol + high momentum = proxy for consistent earners

HC #694: Commission-free (RH/IBKR)
HC #428: Regime-agnostic validation
HC #0: Sliding window walk-forward only
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
import json
import os
from datetime import datetime
from scipy.stats import spearmanr

OUT_DIR = '/home/jupiter/Lvl3Quant/output/growth_research_r7'
os.makedirs(OUT_DIR, exist_ok=True)

TRAIN_DAYS = 252
TEST_DAYS = 21
START = '2010-01-01'
END = '2026-07-14'

# Universe: liquid large caps we can actually trade on RH
UNIVERSE = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'BRK-B',
    'JPM', 'JNJ', 'V', 'UNH', 'HD', 'PG', 'MA', 'DIS', 'BAC', 'XOM',
    'ABBV', 'KO', 'PFE', 'PEP', 'TMO', 'COST', 'AVGO', 'MRK', 'CVX',
    'WMT', 'ABT', 'LLY', 'CSCO', 'MCD', 'ACN', 'DHR', 'TXN', 'NEE',
    'INTC', 'PM', 'UPS', 'RTX', 'LOW', 'NKE', 'LIN', 'ORCL', 'CRM',
    'AMD', 'QCOM', 'IBM', 'GS', 'CAT',
]


def calc_metrics(returns, name=''):
    rets = returns.dropna()
    if len(rets) < 20:
        return {'name': name, 'sharpe': 0, 'sortino': 0, 'cagr': 0,
                'max_dd': 0, 'win_rate': 0, 'pf': 0, 'n_days': len(rets)}
    total_ret = (1 + rets).prod() - 1
    n_years = len(rets) / 252
    cagr = (1 + total_ret) ** (1 / max(n_years, 0.01)) - 1
    sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
    downside = rets[rets < 0].std() * np.sqrt(252)
    sortino = rets.mean() * 252 / downside if downside > 0 else 0
    cum = (1 + rets).cumprod()
    max_dd = (cum / cum.cummax() - 1).min()
    wr = (rets > 0).mean()
    gp = rets[rets > 0].sum()
    gl = abs(rets[rets < 0].sum())
    pf = gp / gl if gl > 0 else float('inf')
    calmar = abs(cagr / max_dd) if max_dd != 0 else 0
    return {'name': name, 'sharpe': round(float(sharpe), 4),
            'sortino': round(float(sortino), 4),
            'cagr': round(float(cagr), 4), 'cagr_pct': round(float(cagr * 100), 2),
            'max_dd': round(float(max_dd), 4), 'max_dd_pct': round(float(max_dd * 100), 2),
            'win_rate': round(float(wr), 4), 'pf': round(float(pf), 3),
            'calmar': round(float(calmar), 3), 'n_days': int(len(rets))}


def classify_regime(spy_ret, threshold=0.003):
    regimes = pd.Series('flat', index=spy_ret.index)
    regimes[spy_ret > threshold] = 'green'
    regimes[spy_ret < -threshold] = 'red'
    return regimes


def regime_test(strat_rets, spy_rets, name=''):
    regimes = classify_regime(spy_rets.reindex(strat_rets.index))
    results = {}
    for r in ['green', 'red', 'flat']:
        mask = regimes == r
        if mask.sum() > 10:
            results[r] = calc_metrics(strat_rets[mask], f'{name} ({r})')
    if 'green' in results and 'red' in results:
        sg, sr = results['green']['sharpe'], results['red']['sharpe']
        results['regime_gap'] = round(abs(sg - sr) / max(abs(sg), abs(sr), 0.001), 3)
    return results


def run_direction4():
    print("=" * 80)
    print("DIRECTION 4: Earnings-Driven Stock Selection")
    print("=" * 80)

    # Download data
    print(f"Downloading {len(UNIVERSE)} stocks...")
    data = yf.download(UNIVERSE + ['SPY'], start=START, end=END, auto_adjust=True, progress=False)
    close = data['Close'] if isinstance(data.columns, pd.MultiIndex) else data
    close = close.ffill()
    rets = close.pct_change()

    # Filter to stocks with enough data
    available = []
    for t in UNIVERSE:
        if t in close.columns:
            valid = close[t].dropna()
            if len(valid) > TRAIN_DAYS + 200:
                available.append(t)
    print(f"Available stocks: {len(available)}")

    spy_rets = rets['SPY'].dropna()

    # ── Build Factor Signals ──
    # Since we can't get reliable earnings data from yfinance for historical walk-forward,
    # we use price-based quality proxies that capture similar effects

    def compute_signals(prices, returns, lookback_days):
        """Compute factor signals for stock ranking."""
        signals = {}

        for t in available:
            if t not in prices.columns:
                continue
            p = prices[t].dropna()
            r = returns[t].dropna()
            if len(p) < lookback_days:
                continue

            # 1. Momentum (12-1): 12-month return excluding last month (classic PEAD proxy)
            mom_12m = p.pct_change(252)
            mom_1m = p.pct_change(21)
            mom_12_1 = mom_12m - mom_1m  # Skip last month (mean reversion)

            # 2. Low volatility factor (quality proxy: consistent earners have lower vol)
            vol_60d = r.rolling(60).std() * np.sqrt(252)

            # 3. Earnings quality proxy: price-to-52w-high ratio
            # Stocks near 52w high tend to be ones with positive earnings surprises
            high_52w = p.rolling(252).max()
            nearness = p / high_52w.clip(lower=0.01)

            # 4. Acceleration: is momentum accelerating? (captures earnings surprise persistence)
            mom_3m = p.pct_change(63)
            mom_6m = p.pct_change(126)
            accel = mom_3m - (mom_6m / 2)  # Recent momentum > average momentum

            signals[t] = pd.DataFrame({
                'mom_12_1': mom_12_1,
                'vol_60d': vol_60d,
                'nearness_52w': nearness,
                'momentum_accel': accel,
            })

        return signals

    signals = compute_signals(close, rets, TRAIN_DAYS)

    # ── Walk-Forward Factor Model ──
    # Monthly rebalance: rank stocks by composite signal, go long top N
    TOP_N = 10
    rebal_freq = TEST_DAYS  # Monthly

    strategy_rets_dict = {}

    # Test different factor combos
    factor_configs = {
        'Momentum_12_1': ['mom_12_1'],
        'Quality_LowVol': ['vol_60d'],  # Inverted: want LOW vol
        'Composite_MomQual': ['mom_12_1', 'vol_60d', 'nearness_52w', 'momentum_accel'],
        'Accel_Only': ['momentum_accel'],
    }

    for config_name, factors in factor_configs.items():
        print(f"\n--- Factor: {config_name} ---")

        portfolio_rets = pd.Series(dtype=float)
        ic_list = []

        # Get common dates
        all_dates = close.index[TRAIN_DAYS:]

        for rebal_start in range(0, len(all_dates) - rebal_freq, rebal_freq):
            rebal_date = all_dates[rebal_start]
            hold_end = min(rebal_start + rebal_freq, len(all_dates))
            hold_dates = all_dates[rebal_start:hold_end]

            # Compute composite score for each stock
            scores = {}
            for t in available:
                if t not in signals:
                    continue
                sig = signals[t]
                if rebal_date not in sig.index:
                    continue

                score = 0
                valid = True
                for f in factors:
                    if f not in sig.columns:
                        valid = False
                        break
                    val = sig.loc[rebal_date, f]
                    if pd.isna(val):
                        valid = False
                        break
                    if f == 'vol_60d':
                        score -= val  # Invert: want LOW vol
                    else:
                        score += val
                if valid:
                    scores[t] = score

            if len(scores) < TOP_N:
                continue

            # Rank and select top N
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            longs = [t for t, s in ranked[:TOP_N]]

            # IC: correlation between signal score and forward return
            fwd_rets = {}
            for t, s in scores.items():
                fwd = rets[t].loc[hold_dates].sum() if t in rets.columns else np.nan
                if not pd.isna(fwd):
                    fwd_rets[t] = fwd

            if len(fwd_rets) > 10:
                sig_vals = [scores[t] for t in fwd_rets.keys()]
                ret_vals = [fwd_rets[t] for t in fwd_rets.keys()]
                ic_val, _ = spearmanr(sig_vals, ret_vals)
                ic_list.append(ic_val)

            # Equal weight portfolio of top N
            for dt in hold_dates:
                if dt in rets.index:
                    port_ret = rets.loc[dt, longs].mean()
                    if not pd.isna(port_ret):
                        portfolio_rets.loc[dt] = port_ret

        if len(portfolio_rets) > 100:
            avg_ic = np.nanmean(ic_list) if ic_list else 0
            print(f"  Walk-forward IC: {avg_ic:.4f} ({len(ic_list)} periods)")
            strategy_rets_dict[config_name] = {
                'returns': portfolio_rets,
                'ic': round(float(avg_ic), 4),
                'n_periods': len(ic_list),
            }

    # ── Combine best factor with base RP ──
    best_factor = None
    best_ic = -1
    for name, data_dict in strategy_rets_dict.items():
        if data_dict['ic'] > best_ic:
            best_ic = data_dict['ic']
            best_factor = name

    if best_factor:
        rp_tickers_list = ['SPY', 'TLT', 'GLD']
        rp_avail = [t for t in rp_tickers_list if t in rets.columns]
        if rp_avail:
            rp_3x = rets[rp_avail].mean(axis=1) * 3.0
            factor_ret = strategy_rets_dict[best_factor]['returns']
            combo = 0.70 * rp_3x + 0.30 * factor_ret
            combo = combo.dropna()
            strategy_rets_dict['70%_3xRP_+_30%_BestFactor'] = {
                'returns': combo,
                'ic': best_ic,
                'n_periods': 0,
            }

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    metrics_all = {}
    for name, data_dict in strategy_rets_dict.items():
        m = calc_metrics(data_dict['returns'], name)
        metrics_all[name] = m
        print(f"\n{name} (IC={data_dict['ic']:.4f}):")
        print(f"  CAGR: {m['cagr_pct']:.1f}%  |  Sharpe: {m['sharpe']:.3f}  |  Sortino: {m['sortino']:.3f}")
        print(f"  MaxDD: {m['max_dd_pct']:.1f}%  |  Calmar: {m['calmar']:.3f}  |  WR: {m['win_rate']:.3f}")

    # SPY benchmark
    m_spy = calc_metrics(spy_rets, 'SPY B&H')
    print(f"\nSPY B&H:")
    print(f"  CAGR: {m_spy['cagr_pct']:.1f}%  |  Sharpe: {m_spy['sharpe']:.3f}")

    # Regime tests
    print("\n--- Regime Tests ---")
    rt_results = {}
    for name, data_dict in strategy_rets_dict.items():
        ret_series = data_dict['returns']
        if len(ret_series) > 100:
            rt = regime_test(ret_series, spy_rets, name)
            rt_results[name] = rt
            for r in ['green', 'red', 'flat']:
                if r in rt:
                    print(f"  {name} {r}: Sharpe={rt[r]['sharpe']:.3f}")
            if 'regime_gap' in rt:
                print(f"  {name} regime gap: {rt['regime_gap']:.3f}")

    # Honest assessment
    assessment = []
    if best_ic > 0.05:
        assessment.append(f"Best factor ({best_factor}) has meaningful walk-forward IC ({best_ic:.4f}).")
    elif best_ic > 0.02:
        assessment.append(f"Best factor ({best_factor}) has weak but positive IC ({best_ic:.4f}).")
    else:
        assessment.append(f"Factor signals lack predictive edge walk-forward (best IC={best_ic:.4f}).")

    assessment.append("LIMITATION: Without actual earnings surprise data, we're using price-based proxies — this is momentum, not true PEAD.")
    assessment.append("LIMITATION: Long-only stock selection has high beta to SPY — regime gap will be large by construction.")
    assessment.append("LIMITATION: Survivorship bias — our universe is today's large caps, not the ones from 2010.")

    print(f"\nHONEST ASSESSMENT: {' | '.join(assessment)}")

    results = {
        'direction': 'D4_Earnings_Stock_Selection',
        'universe_size': len(available),
        'factor_ics': {k: v['ic'] for k, v in strategy_rets_dict.items()},
        'metrics': metrics_all,
        'regime_tests': rt_results,
        'best_factor': best_factor,
        'honest_assessment': ' | '.join(assessment),
        'timestamp': datetime.now().isoformat(),
    }

    with open(os.path.join(OUT_DIR, 'dir4_earnings_stock_selection.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {OUT_DIR}/dir4_earnings_stock_selection.json")
    return results


if __name__ == '__main__':
    run_direction4()
