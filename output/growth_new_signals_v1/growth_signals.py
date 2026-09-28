#!/usr/bin/env python3
"""
New Predictive Growth Strategies — Walk-Forward Research
5 strategies tested: PEAD, Sector Rotation, Mean Reversion, Momentum+Crash Filter, Dispersion
Walk-forward sliding window (HC #0), commission-free (HC #694)
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
import lightgbm as lgb
from scipy.stats import spearmanr, percentileofscore
import json
import os
from datetime import datetime

OUT_DIR = '/home/jupiter/Lvl3Quant/output/growth_new_signals_v1'
os.makedirs(OUT_DIR, exist_ok=True)

TRAIN_DAYS = 504  # 2 years sliding
START = '2015-01-01'
END = '2026-07-11'

def calc_metrics(returns, name=''):
    rets = returns.dropna()
    if len(rets) < 20:
        return {'name': name, 'sharpe': 0, 'total_ret': 0, 'cagr': 0, 'max_dd': 0,
                'win_rate': 0, 'sortino': 0, 'pf': 0, 'n_days': len(rets), 'ic': 0}
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
    pf = gp / gl if gl > 0 else np.inf
    return {'name': name, 'sharpe': sharpe, 'total_ret': total_ret, 'cagr': cagr,
            'max_dd': max_dd, 'win_rate': wr, 'sortino': sortino, 'pf': pf, 'n_days': len(rets)}

###############################################################################
# STRATEGY 1: POST-EARNINGS ANNOUNCEMENT DRIFT (PEAD)
###############################################################################
def test_pead():
    """Test if earnings surprise predicts 20-day drift using sector ETFs as proxy."""
    print("\n" + "=" * 80)
    print("STRATEGY 1: POST-EARNINGS ANNOUNCEMENT DRIFT (PEAD)")
    print("=" * 80)

    # Use large-cap ETFs that show aggregate earnings momentum
    # QQQ heavy in tech earnings, XLF in financials, etc.
    tickers = ['SPY', 'QQQ', 'IWM']

    data = {}
    for t in tickers:
        df = yf.download(t, start=START, end=END, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        data[t] = df

    # Build earnings momentum proxy features
    spy = data['SPY'].copy()
    spy['ret'] = spy['Close'].pct_change()

    # SUE proxy: abnormal return around earnings season
    # Earnings seasons: Jan, Apr, Jul, Oct
    spy['month'] = spy.index.month
    spy['earnings_season'] = spy['month'].isin([1, 4, 7, 10]).astype(int)

    # Features for predicting next-20-day return
    spy['ret_5d'] = spy['ret'].rolling(5).sum()
    spy['ret_20d'] = spy['ret'].rolling(20).sum()
    spy['ret_60d'] = spy['ret'].rolling(60).sum()
    spy['vol_20d'] = spy['ret'].rolling(20).std()
    spy['vol_ratio'] = spy['ret'].rolling(5).std() / spy['ret'].rolling(20).std()

    # QQQ vs SPY relative strength (earnings momentum proxy)
    qqq = data['QQQ']
    spy['qqq_rel'] = (qqq['Close'].pct_change(20).reindex(spy.index) -
                       spy['Close'].pct_change(20))

    # IWM vs SPY (small cap earnings momentum)
    iwm = data['IWM']
    spy['iwm_rel'] = (iwm['Close'].pct_change(20).reindex(spy.index) -
                       spy['Close'].pct_change(20))

    # Volume surge during earnings
    spy['vol_surge'] = spy['Volume'] / spy['Volume'].rolling(20).mean()

    # Gap opens (earnings reaction proxy)
    spy['gap'] = spy['Open'] / spy['Close'].shift(1) - 1
    spy['gap_5d_avg'] = spy['gap'].rolling(5).mean()

    # Target: next 20-day return
    spy['target_20d'] = spy['Close'].pct_change(20).shift(-20)

    features = ['ret_5d', 'ret_20d', 'ret_60d', 'vol_20d', 'vol_ratio',
                'qqq_rel', 'iwm_rel', 'earnings_season', 'vol_surge', 'gap_5d_avg']

    df = spy.dropna(subset=features + ['target_20d']).copy()

    X = df[features].values
    y = df['target_20d'].values
    dates = df.index

    # Walk-forward
    preds = []
    for i in range(TRAIN_DAYS, len(X), 5):  # step by 5 to avoid overlapping targets
        ts = max(0, i - TRAIN_DAYS)
        train_ds = lgb.Dataset(X[ts:i], label=y[ts:i])
        model = lgb.train(
            {'objective': 'regression', 'metric': 'mae', 'num_leaves': 10,
             'learning_rate': 0.03, 'verbose': -1, 'seed': 42,
             'min_child_samples': 20, 'subsample': 0.8},
            train_ds, num_boost_round=100
        )
        if i < len(X):
            p = model.predict(X[i:i+1])[0]
            preds.append({'date': dates[i], 'pred': p, 'actual': y[i]})

    pred_df = pd.DataFrame(preds).set_index('date')
    ic = pred_df['pred'].corr(pred_df['actual'])
    rank_ic = pred_df['pred'].corr(pred_df['actual'], method='spearman')

    # Strategy: long SPY when pred > 0, else cash
    pred_daily = pred_df.reindex(spy.index, method='ffill')
    pred_daily['signal'] = (pred_daily['pred'] > 0).astype(int)
    pred_daily['strat_ret'] = spy['ret'] * pred_daily['signal']
    pred_daily = pred_daily.dropna(subset=['strat_ret'])

    m = calc_metrics(pred_daily['strat_ret'], 'PEAD_proxy')
    bh = calc_metrics(spy['ret'].reindex(pred_daily.index).dropna(), 'SPY_BH')

    print(f"  OOT IC (20d): {ic:.4f}")
    print(f"  OOT Rank IC: {rank_ic:.4f}")
    print(f"  Strategy: Sharpe={m['sharpe']:.2f}, CAGR={m['cagr']:.1%}, MaxDD={m['max_dd']:.1%}")
    print(f"  Benchmark: Sharpe={bh['sharpe']:.2f}, CAGR={bh['cagr']:.1%}, MaxDD={bh['max_dd']:.1%}")
    print(f"  PASS" if m['sharpe'] > bh['sharpe'] and ic > 0 else f"  FAIL")

    return {'name': 'PEAD_proxy', 'ic': ic, 'rank_ic': rank_ic,
            'strategy': m, 'benchmark': bh,
            'pass': m['sharpe'] > bh['sharpe'] and ic > 0}


###############################################################################
# STRATEGY 2: SECTOR ROTATION WITH MACRO SIGNALS
###############################################################################
def test_sector_rotation():
    print("\n" + "=" * 80)
    print("STRATEGY 2: SECTOR ROTATION WITH MACRO SIGNALS")
    print("=" * 80)

    sectors = ['XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLU', 'XLP', 'XLY', 'XLB', 'XLC']

    # Download sector data + macro proxies
    all_tickers = sectors + ['SPY', 'TLT', 'HYG', 'LQD', 'IEF', '^VIX']
    data = {}
    for t in all_tickers:
        try:
            df = yf.download(t, start=START, end=END, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            data[t] = df
            print(f"  {t}: {len(df)} rows")
        except:
            print(f"  {t}: FAILED")

    # Build common date index
    spy = data['SPY']
    common_idx = spy.index

    # Sector returns
    sector_rets = pd.DataFrame()
    for s in sectors:
        if s in data:
            sector_rets[s] = data[s]['Close'].reindex(common_idx).pct_change()

    # Macro features
    macro = pd.DataFrame(index=common_idx)

    # Yield curve proxy: TLT/IEF ratio (long vs intermediate duration)
    if 'TLT' in data and 'IEF' in data:
        macro['yield_slope'] = (data['TLT']['Close'].reindex(common_idx).pct_change(20) -
                                data['IEF']['Close'].reindex(common_idx).pct_change(20))

    # Credit spread proxy: HYG vs LQD
    if 'HYG' in data and 'LQD' in data:
        macro['credit_spread'] = (data['LQD']['Close'].reindex(common_idx).pct_change(20) -
                                   data['HYG']['Close'].reindex(common_idx).pct_change(20))

    # VIX features
    if '^VIX' in data:
        vix = data['^VIX']['Close'].reindex(common_idx).ffill()
        macro['vix'] = vix
        macro['vix_chg20'] = vix.pct_change(20)
        macro['vix_pctile'] = vix.rolling(252).apply(
            lambda x: percentileofscore(x, x.iloc[-1])/100 if len(x.dropna())>10 else np.nan, raw=False)

    # SPY momentum
    macro['spy_mom20'] = data['SPY']['Close'].reindex(common_idx).pct_change(20)
    macro['spy_mom60'] = data['SPY']['Close'].reindex(common_idx).pct_change(60)

    # For each sector, add sector-specific momentum
    for s in sectors:
        if s in sector_rets.columns:
            macro[f'{s}_mom20'] = sector_rets[s].rolling(20).sum()

    macro = macro.dropna()

    # Walk-forward: for each period, predict which sectors will outperform
    # Target: sector return rank over next 20 days
    all_ics = []
    strat_rets = []

    print(f"  Running walk-forward on {len(sectors)} sectors...")

    # Simple approach: predict each sector's 20d return, pick top 3
    for s in sectors:
        if s not in sector_rets.columns:
            continue

        fwd_ret = sector_rets[s].rolling(20).sum().shift(-20)

        features = [c for c in macro.columns if not c.startswith(s)][:10]  # top 10 macro features

        df_s = macro[features].copy()
        df_s['target'] = fwd_ret
        df_s = df_s.dropna()

        if len(df_s) < TRAIN_DAYS + 50:
            continue

        X = df_s[features].values
        y = df_s['target'].values
        dates = df_s.index

        preds = []
        for i in range(TRAIN_DAYS, len(X), 5):
            ts = max(0, i - TRAIN_DAYS)
            train_ds = lgb.Dataset(X[ts:i], label=y[ts:i])
            model = lgb.train(
                {'objective': 'regression', 'metric': 'mae', 'num_leaves': 8,
                 'learning_rate': 0.03, 'verbose': -1, 'seed': 42, 'min_child_samples': 20},
                train_ds, num_boost_round=80
            )
            if i < len(X):
                p = model.predict(X[i:i+1])[0]
                preds.append({'date': dates[i], 'pred': p, 'actual': y[i], 'sector': s})

        if preds:
            pdf = pd.DataFrame(preds)
            ic = pdf['pred'].corr(pdf['actual'])
            all_ics.append({'sector': s, 'ic': ic, 'n': len(pdf)})
            print(f"    {s}: IC={ic:.4f} (N={len(pdf)})")

    # Aggregate: equal-weight top predicted sectors each period
    # Simplified: long sectors with positive predicted return, else SPY
    avg_ic = np.mean([x['ic'] for x in all_ics]) if all_ics else 0

    # Strategy using top 3 sectors
    spy_ret = data['SPY']['Close'].reindex(common_idx).pct_change()
    # Equal weight all sectors as naive benchmark
    ew_ret = sector_rets.mean(axis=1).dropna()

    bh = calc_metrics(spy_ret.dropna(), 'SPY_BH')
    ew = calc_metrics(ew_ret, 'EW_sectors')

    print(f"\n  Average cross-sector IC: {avg_ic:.4f}")
    print(f"  SPY B&H: Sharpe={bh['sharpe']:.2f}")
    print(f"  EW Sectors: Sharpe={ew['sharpe']:.2f}")
    passed = avg_ic > 0
    print(f"  {'PASS' if passed else 'FAIL'} — avg IC {'>' if passed else '<='} 0")

    return {'name': 'Sector_Rotation', 'avg_ic': avg_ic, 'sector_ics': all_ics,
            'benchmark': bh, 'pass': passed}


###############################################################################
# STRATEGY 3: MEAN REVERSION ON HIGH-VOL STOCKS
###############################################################################
def test_mean_reversion():
    print("\n" + "=" * 80)
    print("STRATEGY 3: MEAN REVERSION ON HIGH-VOL STOCKS")
    print("=" * 80)

    # Use high-beta ETFs as proxies
    tickers = ['ARKK', 'TQQQ', 'SOXL', 'XBI', 'KWEB', 'IWM', 'QQQ', 'SPY']

    data = {}
    for t in tickers:
        try:
            df = yf.download(t, start='2018-01-01', end=END, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if len(df) > 100:
                data[t] = df
                print(f"  {t}: {len(df)} rows")
        except:
            pass

    all_results = []

    for ticker in data:
        df = data[ticker].copy()
        df['ret'] = df['Close'].pct_change()

        # Mean reversion features
        df['rsi_14'] = compute_rsi(df['Close'], 14)
        df['rsi_5'] = compute_rsi(df['Close'], 5)
        df['bb_pctb'] = compute_bb_pctb(df['Close'], 20, 2)
        df['ret_5d'] = df['ret'].rolling(5).sum()
        df['ret_10d'] = df['ret'].rolling(10).sum()
        df['vol_20d'] = df['ret'].rolling(20).std()
        df['vol_ratio'] = df['ret'].rolling(5).std() / df['ret'].rolling(20).std()
        df['dist_from_ma20'] = df['Close'] / df['Close'].rolling(20).mean() - 1
        df['dist_from_ma50'] = df['Close'] / df['Close'].rolling(50).mean() - 1

        # Target: next 5-day return
        df['target'] = df['Close'].pct_change(5).shift(-5)

        features = ['rsi_14', 'rsi_5', 'bb_pctb', 'ret_5d', 'ret_10d',
                    'vol_20d', 'vol_ratio', 'dist_from_ma20', 'dist_from_ma50']

        df = df.dropna(subset=features + ['target'])

        if len(df) < TRAIN_DAYS + 50:
            continue

        X = df[features].values
        y = df['target'].values
        dates = df.index

        preds = []
        for i in range(TRAIN_DAYS, len(X), 5):
            ts = max(0, i - TRAIN_DAYS)
            train_ds = lgb.Dataset(X[ts:i], label=y[ts:i])
            model = lgb.train(
                {'objective': 'regression', 'metric': 'mae', 'num_leaves': 10,
                 'learning_rate': 0.03, 'verbose': -1, 'seed': 42, 'min_child_samples': 15},
                train_ds, num_boost_round=100
            )
            if i < len(X):
                p = model.predict(X[i:i+1])[0]
                preds.append({'date': dates[i], 'pred': p, 'actual': y[i]})

        if preds:
            pdf = pd.DataFrame(preds)
            ic = pdf['pred'].corr(pdf['actual'])

            # Strategy: trade mean reversion signals
            pdf_indexed = pdf.set_index('date')
            daily_rets = df['ret'].reindex(pdf_indexed.index)
            signal = (pdf_indexed['pred'] > pdf_indexed['pred'].quantile(0.7)).astype(int)
            strat_ret = daily_rets * signal
            m = calc_metrics(strat_ret.dropna(), ticker)

            print(f"  {ticker}: IC={ic:.4f}, Sharpe={m['sharpe']:.2f}, "
                  f"WR={m['win_rate']:.1%}, N={len(pdf)}")
            all_results.append({'ticker': ticker, 'ic': ic, 'metrics': m})

    avg_ic = np.mean([r['ic'] for r in all_results]) if all_results else 0
    avg_sharpe = np.mean([r['metrics']['sharpe'] for r in all_results]) if all_results else 0
    passed = avg_ic > 0 and avg_sharpe > 0

    print(f"\n  Average IC: {avg_ic:.4f}, Average Sharpe: {avg_sharpe:.2f}")
    print(f"  {'PASS' if passed else 'FAIL'}")

    return {'name': 'Mean_Reversion', 'avg_ic': avg_ic, 'avg_sharpe': avg_sharpe,
            'results': all_results, 'pass': passed}


def compute_rsi(prices, period):
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_bb_pctb(prices, period, num_std):
    ma = prices.rolling(period).mean()
    std = prices.rolling(period).std()
    upper = ma + num_std * std
    lower = ma - num_std * std
    return (prices - lower) / (upper - lower)


###############################################################################
# STRATEGY 4: MOMENTUM WITH CRASH PROTECTION
###############################################################################
def test_momentum_crash():
    print("\n" + "=" * 80)
    print("STRATEGY 4: MOMENTUM WITH CRASH PROTECTION")
    print("=" * 80)

    # Momentum universe
    mom_tickers = ['QQQ', 'XLK', 'XLC', 'XLY', 'XLI', 'SMH', 'IGV', 'IWF']
    crash_tickers = ['^VIX', 'TLT', 'HYG', 'LQD', 'SPY']

    all_tickers = list(set(mom_tickers + crash_tickers))
    data = {}
    for t in all_tickers:
        try:
            df = yf.download(t, start=START, end=END, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            data[t] = df
        except:
            pass

    spy = data['SPY']
    common_idx = spy.index

    # Momentum signals for each ETF
    mom_rets = pd.DataFrame()
    for t in mom_tickers:
        if t in data:
            mom_rets[t] = data[t]['Close'].reindex(common_idx).pct_change()

    # 12-1 month momentum (skip most recent month)
    mom_12_1 = pd.DataFrame()
    for t in mom_tickers:
        if t in data:
            p = data[t]['Close'].reindex(common_idx)
            mom_12_1[t] = p.shift(21) / p.shift(252) - 1  # 12m ret, skip 1m

    # Crash indicators
    crash = pd.DataFrame(index=common_idx)
    if '^VIX' in data:
        vix = data['^VIX']['Close'].reindex(common_idx).ffill()
        crash['vix'] = vix
        crash['vix_above_25'] = (vix > 25).astype(int)
        crash['vix_spike'] = vix.pct_change(5) > 0.3
    if 'HYG' in data and 'LQD' in data:
        crash['credit_stress'] = (data['LQD']['Close'].reindex(common_idx).pct_change(20) -
                                   data['HYG']['Close'].reindex(common_idx).pct_change(20))
        crash['credit_alarm'] = crash['credit_stress'] > crash['credit_stress'].rolling(252).quantile(0.9)

    crash['spy_below_ma200'] = (data['SPY']['Close'].reindex(common_idx) <
                                 data['SPY']['Close'].reindex(common_idx).rolling(200).mean()).astype(int)

    # Strategy: long top-3 momentum, but go to cash if crash filter triggers
    crash_filter = (crash.get('vix_above_25', 0).astype(int) |
                   crash.get('credit_alarm', 0).astype(int) |
                   crash['spy_below_ma200'].astype(int))

    # Monthly rebalance
    strat_rets = []

    rebal_dates = common_idx[::21]  # monthly

    for i in range(1, len(rebal_dates)):
        dt = rebal_dates[i]
        prev_dt = rebal_dates[i-1]

        if dt not in mom_12_1.index:
            continue

        # Rank sectors by momentum
        scores = mom_12_1.loc[dt].dropna()
        if len(scores) < 3:
            continue

        top3 = scores.nlargest(3).index.tolist()

        # Get returns for this period
        period_mask = (common_idx > prev_dt) & (common_idx <= dt)
        period_rets = mom_rets.loc[period_mask, top3].mean(axis=1)

        # Apply crash filter
        crash_mask = crash_filter.reindex(period_rets.index).fillna(0).astype(bool)
        period_rets[crash_mask] = 0  # go to cash

        strat_rets.append(period_rets)

    if strat_rets:
        strat_all = pd.concat(strat_rets)

        # Also pure momentum (no crash filter)
        pure_mom_rets = []
        for i in range(1, len(rebal_dates)):
            dt = rebal_dates[i]
            prev_dt = rebal_dates[i-1]
            if dt not in mom_12_1.index:
                continue
            scores = mom_12_1.loc[dt].dropna()
            if len(scores) < 3:
                continue
            top3 = scores.nlargest(3).index.tolist()
            period_mask = (common_idx > prev_dt) & (common_idx <= dt)
            pr = mom_rets.loc[period_mask, top3].mean(axis=1)
            pure_mom_rets.append(pr)

        pure_all = pd.concat(pure_mom_rets) if pure_mom_rets else pd.Series()

        m_crash = calc_metrics(strat_all, 'Mom_CrashFilter')
        m_pure = calc_metrics(pure_all, 'Mom_Pure')
        spy_ret = data['SPY']['Close'].reindex(common_idx).pct_change()
        m_spy = calc_metrics(spy_ret.loc[strat_all.index].dropna(), 'SPY_BH')

        print(f"  Mom+CrashFilter: Sharpe={m_crash['sharpe']:.2f}, CAGR={m_crash['cagr']:.1%}, MaxDD={m_crash['max_dd']:.1%}")
        print(f"  Pure Momentum:   Sharpe={m_pure['sharpe']:.2f}, CAGR={m_pure['cagr']:.1%}, MaxDD={m_pure['max_dd']:.1%}")
        print(f"  SPY B&H:         Sharpe={m_spy['sharpe']:.2f}, CAGR={m_spy['cagr']:.1%}, MaxDD={m_spy['max_dd']:.1%}")

        # Does crash filter add value?
        filter_adds_value = m_crash['sharpe'] > m_pure['sharpe']
        beats_spy = m_crash['sharpe'] > m_spy['sharpe']
        passed = beats_spy

        print(f"  Crash filter adds value: {'YES' if filter_adds_value else 'NO'}")
        print(f"  Beats SPY: {'YES' if beats_spy else 'NO'}")
        print(f"  {'PASS' if passed else 'FAIL'}")

        return {'name': 'Momentum_CrashFilter', 'crash_filter': m_crash,
                'pure_momentum': m_pure, 'benchmark': m_spy,
                'filter_adds_value': filter_adds_value, 'pass': passed}

    print("  FAIL — insufficient data")
    return {'name': 'Momentum_CrashFilter', 'pass': False}


###############################################################################
# STRATEGY 5: DISPERSION TRADING PROXY
###############################################################################
def test_dispersion():
    print("\n" + "=" * 80)
    print("STRATEGY 5: DISPERSION TRADING PROXY")
    print("=" * 80)

    # Idea: when cross-sectional vol (dispersion) is high, stock-picking alpha rises
    # Predict dispersion, then go active (momentum) when high, passive when low

    sectors = ['XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLU', 'XLP', 'XLY', 'XLB', 'XLC']

    data = {}
    for t in sectors + ['SPY', '^VIX']:
        try:
            df = yf.download(t, start=START, end=END, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            data[t] = df
        except:
            pass

    spy = data['SPY']
    common_idx = spy.index

    sector_rets = pd.DataFrame()
    for s in sectors:
        if s in data:
            sector_rets[s] = data[s]['Close'].reindex(common_idx).pct_change()

    # Dispersion = cross-sectional standard deviation of sector returns
    dispersion_20d = sector_rets.rolling(20).apply(lambda x: x.std()).mean(axis=1)

    # Alternative: realized cross-sectional correlation
    # When correlation drops, dispersion rises
    rolling_corr = sector_rets.rolling(60).corr()
    # Average pairwise correlation
    avg_corr = []
    for dt in common_idx:
        if dt in rolling_corr.index.get_level_values(0):
            try:
                corr_matrix = rolling_corr.loc[dt]
                mask = np.ones(corr_matrix.shape, dtype=bool)
                np.fill_diagonal(mask, False)
                avg_c = corr_matrix.values[mask].mean()
                avg_corr.append(avg_c)
            except:
                avg_corr.append(np.nan)
        else:
            avg_corr.append(np.nan)

    disp_df = pd.DataFrame(index=common_idx)
    disp_df['dispersion'] = dispersion_20d
    disp_df['avg_corr'] = avg_corr
    disp_df['spy_ret'] = spy['Close'].pct_change()

    if '^VIX' in data:
        disp_df['vix'] = data['^VIX']['Close'].reindex(common_idx).ffill()

    # Features to predict dispersion
    disp_df['disp_lag5'] = disp_df['dispersion'].shift(5)
    disp_df['disp_lag20'] = disp_df['dispersion'].shift(20)
    disp_df['corr_lag5'] = disp_df['avg_corr'].shift(5)
    disp_df['vix_level'] = disp_df['vix'].shift(1)
    disp_df['spy_vol20'] = disp_df['spy_ret'].rolling(20).std().shift(1)

    # Target: is dispersion high next 20 days? (above median)
    disp_df['future_disp'] = disp_df['dispersion'].rolling(20).mean().shift(-20)
    disp_median = disp_df['dispersion'].rolling(252).median()
    disp_df['target'] = (disp_df['future_disp'] > disp_median).astype(int)

    features = ['disp_lag5', 'disp_lag20', 'corr_lag5', 'vix_level', 'spy_vol20']
    df = disp_df.dropna(subset=features + ['target']).copy()

    if len(df) < TRAIN_DAYS + 50:
        print("  FAIL — insufficient data")
        return {'name': 'Dispersion', 'pass': False}

    X = df[features].values
    y = df['target'].values
    dates = df.index

    preds = []
    for i in range(TRAIN_DAYS, len(X), 5):
        ts = max(0, i - TRAIN_DAYS)
        train_ds = lgb.Dataset(X[ts:i], label=y[ts:i])
        model = lgb.train(
            {'objective': 'binary', 'metric': 'auc', 'num_leaves': 8,
             'learning_rate': 0.03, 'verbose': -1, 'seed': 42, 'min_child_samples': 20},
            train_ds, num_boost_round=80
        )
        if i < len(X):
            p = model.predict(X[i:i+1])[0]
            preds.append({'date': dates[i], 'pred': p, 'actual': y[i]})

    pdf = pd.DataFrame(preds).set_index('date')

    # Accuracy of dispersion prediction
    acc = ((pdf['pred'] > 0.5).astype(int) == pdf['actual']).mean()
    ic = pdf['pred'].corr(pdf['actual'])

    # Strategy: when high dispersion predicted, go equal-weight momentum top 3 sectors
    # When low dispersion predicted, just hold SPY
    spy_daily = spy['Close'].pct_change().reindex(common_idx)

    # Simplified: use dispersion prediction as signal
    # High dispersion = more active management value
    print(f"  Dispersion prediction accuracy: {acc:.1%}")
    print(f"  IC: {ic:.4f}")

    passed = ic > 0 and acc > 0.52
    print(f"  {'PASS' if passed else 'FAIL'}")

    return {'name': 'Dispersion', 'accuracy': acc, 'ic': ic, 'pass': passed}


###############################################################################
# RUN ALL STRATEGIES
###############################################################################
if __name__ == '__main__':
    print("=" * 80)
    print("NEW GROWTH SIGNAL RESEARCH — Walk-Forward Testing")
    print("=" * 80)

    results = {}

    results['pead'] = test_pead()
    results['sector_rotation'] = test_sector_rotation()
    results['mean_reversion'] = test_mean_reversion()
    results['momentum_crash'] = test_momentum_crash()
    results['dispersion'] = test_dispersion()

    ###########################################################################
    # FINAL SUMMARY
    ###########################################################################
    print("\n" + "=" * 80)
    print("FINAL SUMMARY — PASS/FAIL")
    print("=" * 80)

    for key, r in results.items():
        status = "PASS" if r.get('pass', False) else "FAIL"
        name = r.get('name', key)
        print(f"  [{status}] {name}")
        if 'ic' in r:
            print(f"         IC: {r['ic']:.4f}")
        if 'avg_ic' in r:
            print(f"         Avg IC: {r['avg_ic']:.4f}")
        if 'strategy' in r and isinstance(r['strategy'], dict):
            print(f"         Sharpe: {r['strategy'].get('sharpe', 'N/A'):.2f}")

    # Save results
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, pd.Timestamp):
            return str(obj)
        else:
            return obj

    with open(os.path.join(OUT_DIR, 'results.json'), 'w') as f:
        json.dump(make_serializable(results), f, indent=2, default=str)

    print(f"\n  Results saved to {OUT_DIR}/results.json")
    print("\n" + "=" * 80)
    print("RESEARCH COMPLETE")
    print("=" * 80)
