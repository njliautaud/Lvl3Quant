#!/usr/bin/env python3
"""
Crypto Portfolio Enhancement Analysis
======================================
Tests whether adding crypto (BTC/ETH via ETFs) improves the vol-adjusted UPRO system.

Questions:
1. Does a small crypto allocation improve growth-phase returns?
2. Does crypto provide diversification (low correlation to UPRO)?
3. What's the optimal crypto weight for a growth portfolio?
4. Should crypto be vol-adjusted too, or always-on?
5. How does crypto behave during our vol regimes?

Tests:
1. US-only vol-adjusted (baseline)
2. 90% UPRO + 10% BTC proxy
3. 80% UPRO + 15% BTC + 5% ETH
4. Vol-adjusted crypto (reduce crypto when crypto vol high)
5. Momentum-switched (BTC when BTC trending, else UPRO-only)
6. Risk-parity UPRO + BTC
7. Crypto as safe-haven replacement (crypto instead of GLD in high-vol)

Note: IBIT/BITO only available since late 2021/2024. We use BTC-USD as proxy
for longer history and note the limitation.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import json
import os
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/jupiter/Lvl3Quant/output/growth_research/crypto_enhancement'
os.makedirs(OUTPUT_DIR, exist_ok=True)

INITIAL = 500
WEEKLY_DCA = 100

def download_data():
    # Core ETFs + crypto proxies
    tickers = ['SPY', 'UPRO', 'GLD', 'TLT',
               'BTC-USD', 'ETH-USD',  # Direct crypto prices
               'IBIT', 'BITO',  # BTC ETFs (shorter history)
               ]

    data = yf.download(tickers, start='2015-01-01', period='max',
                       auto_adjust=True, threads=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        closes = data['Close']
    else:
        closes = data
    if hasattr(closes.columns, 'droplevel'):
        try:
            closes.columns = closes.columns.droplevel(1)
        except:
            pass

    # Rename for convenience
    if 'BTC-USD' in closes.columns:
        closes = closes.rename(columns={'BTC-USD': 'BTC', 'ETH-USD': 'ETH'})

    closes = closes.dropna(how='all').dropna(subset=['SPY', 'UPRO'])
    print(f"  Data: {len(closes)} days, {closes.shape[1]} tickers")

    # Check crypto availability
    for t in ['BTC', 'ETH', 'IBIT', 'BITO']:
        if t in closes.columns:
            valid = closes[t].dropna()
            if len(valid) > 0:
                print(f"    {t}: {len(valid)} days ({valid.index[0].strftime('%Y-%m-%d')} to {valid.index[-1].strftime('%Y-%m-%d')})")

    return closes


def simulate_strategy(closes, get_alloc, name=""):
    """Simulate vol-adjusted strategy with optional crypto allocation."""
    spy = closes['SPY']
    spy_ret = spy.pct_change()
    vol_21d = spy_ret.rolling(21).std() * np.sqrt(252)
    spy_sma50 = spy.rolling(50).mean()
    returns = closes.pct_change().fillna(0)

    # Also compute crypto vol for vol-adjusted crypto strategies
    btc_vol_21d = None
    if 'BTC' in closes.columns:
        btc_ret = closes['BTC'].pct_change()
        btc_vol_21d = btc_ret.rolling(21).std() * np.sqrt(252)

    warmup = 63
    portfolio_val = float(INITIAL)
    holdings = {}
    cash = float(INITIAL)
    total_contributed = float(INITIAL)
    last_week = None
    last_regime = None

    daily_values = []
    regime_log = []

    for i in range(warmup, len(closes)):
        date = closes.index[i]

        # Weekly DCA
        week_key = (date.year, date.isocalendar()[1])
        if week_key != last_week:
            cash += WEEKLY_DCA
            total_contributed += WEEKLY_DCA
            last_week = week_key

        # Apply returns to holdings
        for ticker in list(holdings.keys()):
            if ticker in returns.columns:
                r = returns.loc[date, ticker]
                if not np.isnan(r):
                    holdings[ticker] *= (1 + r)

        # Determine vol regime
        vol = vol_21d.iloc[i] if not np.isnan(vol_21d.iloc[i]) else 0.15
        protection = spy.iloc[i] > spy_sma50.iloc[i] if not np.isnan(spy_sma50.iloc[i]) else True

        btc_vol = None
        if btc_vol_21d is not None and i < len(btc_vol_21d):
            btc_vol = btc_vol_21d.iloc[i] if not np.isnan(btc_vol_21d.iloc[i]) else 0.60

        # Determine target allocation
        if not protection:
            regime = 'CASH'
            target = {'SPY': 1.0}  # Cash proxy
        elif vol < 0.20:
            regime = 'UPRO'
            target = get_alloc(date, closes, i, vol, btc_vol, regime='low_vol')
        elif vol < 0.30:
            regime = 'SPY'
            target = get_alloc(date, closes, i, vol, btc_vol, regime='med_vol')
        else:
            regime = 'SAFE'
            target = get_alloc(date, closes, i, vol, btc_vol, regime='high_vol')

        # Rebalance on regime change
        if regime != last_regime and target:
            total_val = cash + sum(holdings.values())
            # Filter to available tickers
            valid_target = {t: w for t, w in target.items() if t in closes.columns and not np.isnan(closes[t].iloc[i])}
            if not valid_target:
                valid_target = {'SPY': 1.0}
            # Normalize weights
            total_w = sum(valid_target.values())
            if total_w > 0:
                valid_target = {t: w/total_w for t, w in valid_target.items()}

            holdings = {t: total_val * w for t, w in valid_target.items() if w > 0}
            cash = 0
            last_regime = regime
        elif cash > 50 and holdings:
            total_h = sum(holdings.values())
            if total_h > 0:
                for t in holdings:
                    holdings[t] += cash * (holdings[t] / total_h)
                cash = 0

        portfolio_val = cash + sum(holdings.values())
        daily_values.append(portfolio_val)
        regime_log.append(regime)

    return pd.Series(daily_values, index=closes.index[warmup:]), total_contributed, regime_log


def compute_metrics(portfolio, total_contributed):
    r = portfolio.pct_change().dropna()
    if len(r) < 63:
        return None
    years = len(r) / 252
    final = portfolio.iloc[-1]
    ann_ret = (1 + r).prod() ** (252 / len(r)) - 1
    ann_vol = r.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    neg = r[r < 0]
    downside_vol = neg.std() * np.sqrt(252) if len(neg) > 0 else ann_vol
    sortino = ann_ret / downside_vol if downside_vol > 0 else 0
    peak = portfolio.expanding().max()
    dd = (portfolio - peak) / peak
    max_dd = dd.min()
    cagr = (final / portfolio.iloc[0]) ** (1/years) - 1

    return {
        'final_value': float(final),
        'profit': float(final - total_contributed),
        'cagr': float(cagr * 100),
        'sharpe': float(sharpe),
        'sortino': float(sortino),
        'max_dd': float(max_dd * 100),
        'ann_vol': float(ann_vol * 100),
        'years': float(years),
    }


def main():
    print("="*70)
    print("CRYPTO PORTFOLIO ENHANCEMENT ANALYSIS")
    print("="*70)

    closes = download_data()

    # --- Correlation analysis ---
    print("\n  CORRELATION ANALYSIS (21d returns):")
    corr_tickers = ['SPY', 'UPRO', 'BTC', 'ETH', 'GLD', 'TLT']
    avail = [t for t in corr_tickers if t in closes.columns]
    ret_21d = closes[avail].pct_change(21).dropna()
    corr = ret_21d.corr()

    for t in avail:
        if t not in ['SPY', 'UPRO']:
            if 'UPRO' in corr.columns and t in corr.columns:
                print(f"    UPRO ↔ {t}: {corr.loc['UPRO', t]:.3f}")

    # Rolling correlation BTC/SPY
    if 'BTC' in closes.columns:
        btc_spy_corr = closes['BTC'].pct_change().rolling(252).corr(closes['SPY'].pct_change())
        recent_corr = btc_spy_corr.dropna().iloc[-252:] if len(btc_spy_corr.dropna()) >= 252 else btc_spy_corr.dropna()
        print(f"\n    BTC↔SPY rolling 1yr correlation:")
        print(f"      Mean (all time): {btc_spy_corr.dropna().mean():.3f}")
        print(f"      Mean (last year): {recent_corr.mean():.3f}")
        print(f"      Range: [{btc_spy_corr.dropna().min():.3f}, {btc_spy_corr.dropna().max():.3f}]")
        print(f"      % of time > 0.5: {(btc_spy_corr.dropna() > 0.5).mean()*100:.1f}%")

    # --- Standalone crypto performance ---
    print("\n  STANDALONE PERFORMANCE:")
    for t in avail:
        if t in closes.columns:
            p = closes[t].dropna()
            if len(p) > 252:
                total_ret = p.iloc[-1] / p.iloc[0] - 1
                years = len(p) / 252
                cagr = (1 + total_ret) ** (1/years) - 1
                vol = p.pct_change().std() * np.sqrt(252)
                sharpe = cagr / vol if vol > 0 else 0
                mdd = ((p - p.expanding().max()) / p.expanding().max()).min()
                print(f"    {t:<6s}: CAGR {cagr*100:>7.1f}%, Vol {vol*100:>5.1f}%, "
                      f"Sharpe {sharpe:>5.2f}, MaxDD {mdd*100:>6.1f}%")

    # --- Crypto during our vol regimes ---
    print("\n  CRYPTO BEHAVIOR BY VOL REGIME:")
    if 'BTC' in closes.columns:
        spy_ret = closes['SPY'].pct_change()
        vol_21d = spy_ret.rolling(21).std() * np.sqrt(252)
        btc_ret = closes['BTC'].pct_change()

        for regime_name, lo, hi in [('Low vol (<20%)', 0, 0.20),
                                      ('Med vol (20-30%)', 0.20, 0.30),
                                      ('High vol (>30%)', 0.30, 1.0)]:
            mask = (vol_21d >= lo) & (vol_21d < hi) & btc_ret.notna()
            if mask.sum() > 20:
                regime_btc = btc_ret[mask]
                ann_r = regime_btc.mean() * 252
                ann_v = regime_btc.std() * np.sqrt(252)
                sr = ann_r / ann_v if ann_v > 0 else 0
                print(f"    {regime_name:<22s}: BTC ann ret {ann_r*100:>7.1f}%, "
                      f"vol {ann_v*100:>5.1f}%, Sharpe {sr:>5.2f}, n={mask.sum()}")

    # --- Strategy variants ---
    strategies = {}

    # 1. US-only (baseline)
    def baseline(d, c, i, vol, btc_vol, regime='low_vol'):
        if regime == 'low_vol':
            return {'UPRO': 1.0}
        elif regime == 'med_vol':
            return {'SPY': 1.0}
        else:
            return {'GLD': 0.5, 'TLT': 0.5}
    strategies['1. US only (baseline)'] = baseline

    # 2. Small BTC allocation (5%)
    def upro_btc5(d, c, i, vol, btc_vol, regime='low_vol'):
        if regime == 'low_vol':
            return {'UPRO': 0.95, 'BTC': 0.05}
        elif regime == 'med_vol':
            return {'SPY': 0.95, 'BTC': 0.05}
        else:
            return {'GLD': 0.5, 'TLT': 0.5}
    strategies['2. 95/5 UPRO/BTC'] = upro_btc5

    # 3. 10% BTC
    def upro_btc10(d, c, i, vol, btc_vol, regime='low_vol'):
        if regime == 'low_vol':
            return {'UPRO': 0.90, 'BTC': 0.10}
        elif regime == 'med_vol':
            return {'SPY': 0.90, 'BTC': 0.10}
        else:
            return {'GLD': 0.5, 'TLT': 0.5}
    strategies['3. 90/10 UPRO/BTC'] = upro_btc10

    # 4. 20% BTC
    def upro_btc20(d, c, i, vol, btc_vol, regime='low_vol'):
        if regime == 'low_vol':
            return {'UPRO': 0.80, 'BTC': 0.20}
        elif regime == 'med_vol':
            return {'SPY': 0.80, 'BTC': 0.20}
        else:
            return {'GLD': 0.5, 'TLT': 0.5}
    strategies['4. 80/20 UPRO/BTC'] = upro_btc20

    # 5. BTC + ETH split
    def upro_btc_eth(d, c, i, vol, btc_vol, regime='low_vol'):
        if regime == 'low_vol':
            return {'UPRO': 0.85, 'BTC': 0.10, 'ETH': 0.05}
        elif regime == 'med_vol':
            return {'SPY': 0.85, 'BTC': 0.10, 'ETH': 0.05}
        else:
            return {'GLD': 0.5, 'TLT': 0.5}
    strategies['5. 85/10/5 UPRO/BTC/ETH'] = upro_btc_eth

    # 6. Vol-adjusted crypto (reduce crypto when BTC vol is high)
    def vol_adj_crypto(d, c, i, vol, btc_vol, regime='low_vol'):
        if regime != 'low_vol':
            if regime == 'med_vol':
                return {'SPY': 1.0}
            return {'GLD': 0.5, 'TLT': 0.5}

        # Scale crypto inversely with its own vol
        if btc_vol is not None and btc_vol > 0:
            # Target 10% at normal vol (60%), scale down when higher
            crypto_w = min(0.15, max(0.02, 0.10 * (0.60 / btc_vol)))
        else:
            crypto_w = 0.05
        return {'UPRO': 1.0 - crypto_w, 'BTC': crypto_w}
    strategies['6. Vol-adjusted BTC (inv-vol)'] = vol_adj_crypto

    # 7. BTC momentum (only hold BTC when trending up)
    def btc_momentum(d, c, i, vol, btc_vol, regime='low_vol'):
        if regime != 'low_vol':
            if regime == 'med_vol':
                return {'SPY': 1.0}
            return {'GLD': 0.5, 'TLT': 0.5}

        # BTC > 50d SMA = trending
        if 'BTC' in c.columns and i >= 50:
            btc_sma50 = c['BTC'].iloc[i-50:i].mean()
            btc_price = c['BTC'].iloc[i]
            if not np.isnan(btc_price) and not np.isnan(btc_sma50) and btc_price > btc_sma50:
                return {'UPRO': 0.85, 'BTC': 0.15}
        return {'UPRO': 1.0}
    strategies['7. BTC momentum (>SMA50)'] = btc_momentum

    # 8. Crypto as safe-haven replacement
    def crypto_safe_haven(d, c, i, vol, btc_vol, regime='low_vol'):
        if regime == 'low_vol':
            return {'UPRO': 1.0}
        elif regime == 'med_vol':
            return {'SPY': 0.90, 'BTC': 0.10}
        else:
            # Replace GLD with BTC in high-vol regime
            return {'BTC': 0.30, 'GLD': 0.35, 'TLT': 0.35}
    strategies['8. BTC replaces GLD (high vol)'] = crypto_safe_haven

    # 9. Risk parity UPRO + BTC (inverse vol weighting)
    def risk_parity_btc(d, c, i, vol, btc_vol, regime='low_vol'):
        if regime != 'low_vol':
            if regime == 'med_vol':
                return {'SPY': 1.0}
            return {'GLD': 0.5, 'TLT': 0.5}

        if btc_vol is not None and btc_vol > 0 and vol > 0:
            # UPRO vol ~ 3x SPY vol
            upro_vol = vol * 3
            # Inverse vol weights
            inv_upro = 1.0 / upro_vol
            inv_btc = 1.0 / btc_vol
            total_inv = inv_upro + inv_btc
            upro_w = inv_upro / total_inv
            btc_w = inv_btc / total_inv
            return {'UPRO': upro_w, 'BTC': btc_w}
        return {'UPRO': 0.90, 'BTC': 0.10}
    strategies['9. Risk parity UPRO+BTC'] = risk_parity_btc

    # 10. BTC only when both equity and BTC in uptrend
    def dual_momentum(d, c, i, vol, btc_vol, regime='low_vol'):
        if regime != 'low_vol':
            if regime == 'med_vol':
                return {'SPY': 1.0}
            return {'GLD': 0.5, 'TLT': 0.5}

        if 'BTC' in c.columns and i >= 63:
            btc_mom = c['BTC'].iloc[i] / c['BTC'].iloc[i-63] - 1
            spy_mom = c['SPY'].iloc[i] / c['SPY'].iloc[i-63] - 1
            if not np.isnan(btc_mom) and not np.isnan(spy_mom):
                if btc_mom > 0 and spy_mom > 0:
                    return {'UPRO': 0.85, 'BTC': 0.15}
        return {'UPRO': 1.0}
    strategies['10. Dual momentum (SPY+BTC up)'] = dual_momentum

    print(f"\nTesting {len(strategies)} strategies...")

    results = {}
    for name, alloc in strategies.items():
        print(f"  Running: {name}...", end=" ", flush=True)
        portfolio, total_cont, regimes = simulate_strategy(closes, alloc, name)
        m = compute_metrics(portfolio, total_cont)
        if m:
            results[name] = m
            print(f"${m['final_value']:,.0f} | Sharpe {m['sharpe']:.3f} | MaxDD {m['max_dd']:.1f}%")

    # --- Results table ---
    sorted_results = sorted(results.items(), key=lambda x: x[1]['final_value'], reverse=True)

    print("\n" + "="*70)
    print("RESULTS — RANKED BY FINAL VALUE")
    print("="*70)

    print(f"\n  {'Strategy':<35s} {'Final $':>10s} {'Sharpe':>7s} {'Sortino':>8s} {'MaxDD':>7s} {'CAGR':>7s} {'Vol':>6s}")
    print("  " + "-"*82)
    for name, m in sorted_results:
        print(f"  {name:<35s} ${m['final_value']:>9,.0f} {m['sharpe']:>7.3f} {m['sortino']:>8.3f} "
              f"{m['max_dd']:>6.1f}% {m['cagr']:>6.1f}% {m['ann_vol']:>5.1f}%")

    # --- vs baseline ---
    baseline_m = results.get('1. US only (baseline)', {})
    if baseline_m:
        print(f"\n  vs US-only baseline (${baseline_m['final_value']:,.0f}, Sharpe {baseline_m['sharpe']:.3f}):")
        for name, m in sorted_results:
            if '1. US only' in name:
                continue
            val_diff = m['final_value'] - baseline_m['final_value']
            sharpe_diff = m['sharpe'] - baseline_m['sharpe']
            dd_diff = m['max_dd'] - baseline_m['max_dd']
            val_pct = val_diff / baseline_m['final_value'] * 100
            print(f"    {name:<33s}: value {'+' if val_diff > 0 else ''}{val_diff:>+10,.0f} ({val_pct:>+5.1f}%), "
                  f"Sharpe {sharpe_diff:+.3f}, MaxDD {dd_diff:+.1f}pp")

    # --- Sharpe-ranked ---
    sharpe_sorted = sorted(results.items(), key=lambda x: x[1]['sharpe'], reverse=True)
    print(f"\n  RANKED BY SHARPE:")
    for rank, (name, m) in enumerate(sharpe_sorted, 1):
        print(f"    {rank}. {name:<33s}: Sharpe {m['sharpe']:.3f}")

    # --- Sub-period analysis for top strategies ---
    print("\n" + "="*70)
    print("SUB-PERIOD CONSISTENCY (top 3 by Sharpe)")
    print("="*70)

    top3 = [name for name, _ in sharpe_sorted[:3]]
    for name in top3:
        alloc = strategies[name]
        portfolio, _, _ = simulate_strategy(closes, alloc, name)
        n = len(portfolio)
        third = n // 3

        periods = {
            'Early': portfolio.iloc[:third],
            'Mid': portfolio.iloc[third:2*third],
            'Late': portfolio.iloc[2*third:]
        }

        print(f"\n  {name}:")
        for pname, p in periods.items():
            r = p.pct_change().dropna()
            if len(r) > 20:
                ann_r = (1+r).prod() ** (252/len(r)) - 1
                ann_v = r.std() * np.sqrt(252)
                sr = ann_r / ann_v if ann_v > 0 else 0
                print(f"    {pname:<6s}: Sharpe {sr:.3f}, Ann ret {ann_r*100:.1f}%, Vol {ann_v*100:.1f}%")

    # --- Key finding summary ---
    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)

    if baseline_m:
        # Check if any crypto strategy beats baseline on BOTH value and Sharpe
        winners = [(n, m) for n, m in results.items()
                   if '1. US only' not in n
                   and m['final_value'] > baseline_m['final_value']
                   and m['sharpe'] > baseline_m['sharpe']]

        losers_value = [(n, m) for n, m in results.items()
                        if '1. US only' not in n
                        and m['final_value'] < baseline_m['final_value']]

        if winners:
            print(f"\n  ✅ {len(winners)} strategies beat baseline on BOTH value AND Sharpe:")
            for n, m in winners:
                val_diff = m['final_value'] - baseline_m['final_value']
                print(f"     {n}: +${val_diff:,.0f}, Sharpe {m['sharpe']:.3f} vs {baseline_m['sharpe']:.3f}")
        else:
            print(f"\n  ❌ NO strategy beats baseline on both value AND Sharpe")

        if losers_value:
            worst_loss = min(losers_value, key=lambda x: x[1]['final_value'])
            print(f"\n  Worst crypto drag: {worst_loss[0]}: -${baseline_m['final_value'] - worst_loss[1]['final_value']:,.0f}")

    # Save
    output = {
        'run_date': pd.Timestamp.now().isoformat(),
        'results': results,
        'ranking_by_value': [n for n, _ in sorted_results],
        'ranking_by_sharpe': [n for n, _ in sharpe_sorted],
    }
    with open(os.path.join(OUTPUT_DIR, 'crypto_enhancement_results.json'), 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved.")

    print("\n" + "="*70)
    print("DONE")
    print("="*70)


if __name__ == '__main__':
    main()
