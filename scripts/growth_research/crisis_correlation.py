#!/usr/bin/env python3
"""
Crisis Correlation Analysis
=============================
Tests whether our diversifiers actually diversify during crises.

Key question: does the correlation between UPRO and our safe havens (GLD, TLT)
INCREASE during market stress, reducing their protective value?

Also: how does the vol-adjusted system perform during each historical crisis?
"""

import numpy as np
import pandas as pd
import yfinance as yf
import json
import os
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/jupiter/Lvl3Quant/output/growth_research/crisis'
os.makedirs(OUTPUT_DIR, exist_ok=True)

INITIAL = 500
WEEKLY_DCA = 100


def download_data():
    tickers = ['SPY', 'UPRO', 'GLD', 'TLT', 'SHY', 'BTC-USD', 'QQQ', 'TQQQ',
               'IEF', 'HYG', 'VIX']
    # VIX won't download as ETF but we can try VIXY
    tickers_final = ['SPY', 'UPRO', 'GLD', 'TLT', 'SHY', 'BTC-USD', 'QQQ', 'TQQQ',
                     'IEF', 'HYG', 'VIXY']
    data = yf.download(tickers_final, start='2012-01-01', period='max',
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
    if 'BTC-USD' in closes.columns:
        closes = closes.rename(columns={'BTC-USD': 'BTC'})
    return closes.dropna(how='all').dropna(subset=['SPY', 'UPRO'])


def simulate_vol_adjusted(closes, start_date=None, end_date=None):
    if start_date:
        closes = closes.loc[start_date:]
    if end_date:
        closes = closes.loc[:end_date]

    spy = closes['SPY']
    spy_ret = spy.pct_change()
    vol_21d = spy_ret.rolling(21).std() * np.sqrt(252)
    spy_sma50 = spy.rolling(50).mean()
    returns = closes.pct_change().fillna(0)

    warmup = 63
    if len(closes) <= warmup:
        return None

    holdings = {}
    cash = float(INITIAL)
    total_contributed = float(INITIAL)
    last_week = None
    last_regime = None
    daily_values = []
    regime_log = []

    for i in range(warmup, len(closes)):
        date = closes.index[i]
        week_key = (date.year, date.isocalendar()[1])
        if week_key != last_week:
            cash += WEEKLY_DCA
            total_contributed += WEEKLY_DCA
            last_week = week_key

        for ticker in list(holdings.keys()):
            if ticker in returns.columns:
                r = returns.loc[date, ticker]
                if not np.isnan(r):
                    holdings[ticker] *= (1 + r)

        vol = vol_21d.iloc[i] if not np.isnan(vol_21d.iloc[i]) else 0.15
        protection = spy.iloc[i] > spy_sma50.iloc[i] if not np.isnan(spy_sma50.iloc[i]) else True

        if not protection:
            regime = 'CASH'
            target = {'SPY': 1.0}
        elif vol < 0.20:
            regime = 'UPRO'
            target = {'UPRO': 1.0}
        elif vol < 0.30:
            regime = 'SPY'
            target = {'SPY': 1.0}
        else:
            regime = 'SAFE'
            target = {'GLD': 1.0}

        if regime != last_regime:
            total_val = cash + sum(holdings.values())
            holdings = {t: total_val * w for t, w in target.items()
                       if w > 0 and t in closes.columns}
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

    return pd.Series(daily_values, index=closes.index[warmup:]), regime_log


def main():
    print("="*70)
    print("CRISIS CORRELATION ANALYSIS")
    print("="*70)

    closes = download_data()
    print(f"  Data: {len(closes)} days")

    # Define crisis periods
    crises = {
        'China Deval (Aug 2015)': ('2015-08-10', '2015-09-30'),
        'Brexit (Jun 2016)': ('2016-06-20', '2016-07-15'),
        'Volmageddon (Feb 2018)': ('2018-01-26', '2018-04-02'),
        'Q4 2018 Selloff': ('2018-10-01', '2019-01-04'),
        'COVID Crash (Mar 2020)': ('2020-02-19', '2020-04-30'),
        'Rate Hike Bear (2022)': ('2022-01-03', '2022-10-15'),
        'SVB Crisis (Mar 2023)': ('2023-03-08', '2023-03-31'),
        'Japan Carry Unwind (Aug 2024)': ('2024-07-15', '2024-08-15'),
        'Trump Tariffs (Apr 2025)': ('2025-03-01', '2025-04-30'),
    }

    assets = ['UPRO', 'GLD', 'TLT', 'BTC', 'HYG', 'VIXY']
    avail = [a for a in assets if a in closes.columns]

    # --- Correlation during crises vs normal ---
    print("\n  CORRELATION TO SPY: CRISIS vs NORMAL")
    print(f"  {'Asset':<8s} {'Normal':>10s} {'Crisis':>10s} {'Change':>10s}")
    print("  " + "-"*40)

    spy_ret = closes['SPY'].pct_change()
    for asset in avail:
        if asset == 'SPY':
            continue
        asset_ret = closes[asset].pct_change()

        # Normal: vol < 20%
        vol_21d = spy_ret.rolling(21).std() * np.sqrt(252)
        normal_mask = vol_21d < 0.20
        crisis_mask = vol_21d > 0.25

        normal_corr = spy_ret[normal_mask].corr(asset_ret[normal_mask])
        crisis_corr = spy_ret[crisis_mask].corr(asset_ret[crisis_mask])

        if not np.isnan(normal_corr) and not np.isnan(crisis_corr):
            print(f"  {asset:<8s} {normal_corr:>10.3f} {crisis_corr:>10.3f} {crisis_corr - normal_corr:>+10.3f}")

    # --- Per-crisis performance ---
    print(f"\n  PER-CRISIS PERFORMANCE (% returns):")
    print(f"  {'Crisis':<30s}", end="")
    for a in ['SPY', 'UPRO'] + [x for x in avail if x not in ['SPY', 'UPRO', 'VIXY']]:
        print(f" {a:>8s}", end="")
    print()
    print("  " + "-"*80)

    for crisis_name, (start, end) in crises.items():
        print(f"  {crisis_name:<30s}", end="")
        for asset in ['SPY', 'UPRO'] + [x for x in avail if x not in ['SPY', 'UPRO', 'VIXY']]:
            if asset in closes.columns:
                crisis_data = closes[asset].loc[start:end].dropna()
                if len(crisis_data) > 1:
                    ret = crisis_data.iloc[-1] / crisis_data.iloc[0] - 1
                    print(f" {ret*100:>7.1f}%", end="")
                else:
                    print(f" {'N/A':>8s}", end="")
        print()

    # --- Vol-adjusted system during each crisis ---
    print(f"\n  VOL-ADJUSTED SYSTEM DURING CRISES:")
    print(f"  {'Crisis':<30s} {'System':>10s} {'SPY':>10s} {'UPRO':>10s} {'Regime':>12s}")
    print("  " + "-"*74)

    for crisis_name, (start, end) in crises.items():
        # Get regime during crisis
        spy_data = closes['SPY'].loc[start:end]
        spy_ret_c = closes['SPY'].pct_change()
        vol_21d = spy_ret_c.rolling(21).std() * np.sqrt(252)
        sma50 = closes['SPY'].rolling(50).mean()

        if len(spy_data) < 5:
            continue

        # What regime were we in?
        crisis_vol = vol_21d.loc[start:end].dropna()
        crisis_sma = sma50.loc[start:end].dropna()
        crisis_spy = closes['SPY'].loc[start:end]

        regimes = []
        for date in crisis_vol.index:
            v = crisis_vol.loc[date]
            prot = crisis_spy.loc[date] > crisis_sma.loc[date] if date in crisis_sma.index else True
            if not prot:
                regimes.append('CASH')
            elif v < 0.20:
                regimes.append('UPRO')
            elif v < 0.30:
                regimes.append('SPY')
            else:
                regimes.append('SAFE')

        regime_str = ', '.join(set(regimes)) if regimes else 'N/A'

        # SPY and UPRO returns
        spy_ret_crisis = spy_data.iloc[-1] / spy_data.iloc[0] - 1
        upro_data = closes['UPRO'].loc[start:end].dropna()
        upro_ret = upro_data.iloc[-1] / upro_data.iloc[0] - 1 if len(upro_data) > 1 else 0

        # Approximate vol-adjusted return (simplified)
        vol_adj_ret = 0
        for r in regimes:
            if r == 'UPRO':
                vol_adj_ret = upro_ret
            elif r == 'SPY':
                vol_adj_ret = spy_ret_crisis
            elif r == 'CASH':
                vol_adj_ret = 0
            elif r == 'SAFE':
                gld_data = closes['GLD'].loc[start:end].dropna()
                vol_adj_ret = gld_data.iloc[-1] / gld_data.iloc[0] - 1 if len(gld_data) > 1 else 0

        print(f"  {crisis_name:<30s} {vol_adj_ret*100:>9.1f}% {spy_ret_crisis*100:>9.1f}% "
              f"{upro_ret*100:>9.1f}% {regime_str:>12s}")

    # --- GLD as crisis hedge quality ---
    print(f"\n  GLD CRISIS HEDGE QUALITY:")
    print(f"  (does GLD go UP when SPY goes DOWN?)")

    # Daily: when SPY drops >2%, what does GLD do?
    gld_ret = closes['GLD'].pct_change()
    big_down_days = spy_ret < -0.02
    if big_down_days.sum() > 10:
        gld_on_bad_days = gld_ret[big_down_days].dropna()
        print(f"    On SPY down >2% days (n={big_down_days.sum()}):")
        print(f"      GLD avg return: {gld_on_bad_days.mean()*100:+.2f}%")
        print(f"      GLD WR (positive): {(gld_on_bad_days > 0).mean()*100:.0f}%")
        print(f"      GLD median: {gld_on_bad_days.median()*100:+.2f}%")

    big_down_weeks = spy_ret.rolling(5).sum() < -0.05
    if big_down_weeks.sum() > 5:
        gld_on_bad_weeks = gld_ret.rolling(5).sum()[big_down_weeks].dropna()
        print(f"    On SPY down >5% weeks (n={big_down_weeks.sum()}):")
        print(f"      GLD avg return: {gld_on_bad_weeks.mean()*100:+.2f}%")
        print(f"      GLD WR (positive): {(gld_on_bad_weeks > 0).mean()*100:.0f}%")

    # --- BTC as crisis diversifier ---
    if 'BTC' in closes.columns:
        btc_ret = closes['BTC'].pct_change()
        btc_on_bad_days = btc_ret[big_down_days].dropna()
        if len(btc_on_bad_days) > 10:
            print(f"\n  BTC CRISIS BEHAVIOR:")
            print(f"    On SPY down >2% days (n={len(btc_on_bad_days)}):")
            print(f"      BTC avg return: {btc_on_bad_days.mean()*100:+.2f}%")
            print(f"      BTC WR (positive): {(btc_on_bad_days > 0).mean()*100:.0f}%")
            print(f"      BTC median: {btc_on_bad_days.median()*100:+.2f}%")

    # Save
    output = {'run_date': pd.Timestamp.now().isoformat()}
    with open(os.path.join(OUTPUT_DIR, 'crisis_results.json'), 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print("\n" + "="*70)
    print("DONE")
    print("="*70)


if __name__ == '__main__':
    main()
