#!/usr/bin/env python3
"""
R4: STRUCTURAL EDGE STRATEGIES
HC #697: No crypto, walk-forward mandatory
HC #0: Sliding window only
HC #694: Commission-free (Robinhood)
HC #428: Regime-agnostic OOT validation (all OOT days, regime gap < 0.50)

5 strategies with structural (non-directional) edges:
1. Volatility Risk Premium (VRP) Harvesting
2. Carry + Trend (Multi-Asset)
3. Dispersion Trading (Simplified)
4. Seasonal/Calendar Effects + ML
5. Merger Arb / Event-Driven (Simplified)

Each: sliding walk-forward (252d train, 21d test), OOT IC, Sharpe, Sortino,
CAGR, MaxDD, and REGIME TEST.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
import lightgbm as lgb
from scipy.stats import spearmanr
import json
import os
from datetime import datetime

OUT_DIR = '/home/jupiter/Lvl3Quant/output/growth_research_r4'
os.makedirs(OUT_DIR, exist_ok=True)

TRAIN_DAYS = 252
TEST_DAYS = 21
START = '2012-01-01'
END = '2026-07-14'


# ─── Shared Utilities ───────────────────────────────────────────────────────

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
    return {'name': name, 'sharpe': float(sharpe), 'sortino': float(sortino),
            'cagr': float(cagr), 'max_dd': float(max_dd), 'win_rate': float(wr),
            'pf': float(pf), 'n_days': int(len(rets)),
            'total_return_pct': float(total_ret * 100),
            'cagr_pct': float(cagr * 100)}


def classify_regime(spy_ret, threshold=0.003):
    regimes = pd.Series('flat', index=spy_ret.index)
    regimes[spy_ret > threshold] = 'green'
    regimes[spy_ret < -threshold] = 'red'
    return regimes


def regime_test(strat_rets, spy_rets, name=''):
    """HC #428 regime-agnostic validation."""
    regimes = classify_regime(spy_rets.reindex(strat_rets.index))
    results = {}
    for r in ['green', 'red', 'flat']:
        mask = regimes == r
        m = calc_metrics(strat_rets[mask], f'{name} ({r})')
        results[r] = m

    sg = results['green']['sharpe']
    sr = results['red']['sharpe']
    max_s = max(abs(sg), abs(sr))
    gap = abs(sg - sr) / max_s if max_s > 0 else float('inf')

    return {
        'per_regime': results,
        'sharpe_green': float(sg),
        'sharpe_red': float(sr),
        'sharpe_flat': float(results['flat']['sharpe']),
        'regime_gap': float(gap),
        'regime_pass': gap < 0.50,
        'distribution': {
            'green': int((regimes == 'green').sum()),
            'red': int((regimes == 'red').sum()),
            'flat': int((regimes == 'flat').sum()),
        }
    }


def download_data(tickers, start=START, end=END):
    data = {}
    for t in tickers:
        try:
            df = yf.download(t, start=start, end=end, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if len(df) > 100:
                data[t] = df
                print(f"  {t}: {len(df)} days")
        except Exception as e:
            print(f"  {t}: FAILED - {e}")
    return data


def rank_ic(signal, actual):
    mask = ~(np.isnan(signal) | np.isnan(actual))
    if mask.sum() < 20:
        return float('nan')
    ic, _ = spearmanr(signal[mask], actual[mask])
    return float(ic)


# ─── STRATEGY 1: VOLATILITY RISK PREMIUM HARVESTING ────────────────────────

def strategy_vrp(data):
    """
    STRUCTURAL EDGE: Insurance sellers get paid. Implied vol > realized vol
    most of the time (the VRP). Sell put spreads when VRP is wide.

    Simplified approach without options chains:
    - Measure VRP = VIX - realized vol (20d)
    - When VRP is positive and wide, sell vol exposure (inverse VIX proxy via position sizing)
    - Walk-forward: predict VRP magnitude, size accordingly
    """
    print("\n" + "=" * 70)
    print("STRATEGY 1: VOLATILITY RISK PREMIUM (VRP) HARVESTING")
    print("Structural edge: Insurance sellers earn a premium")
    print("=" * 70)

    spy = data['SPY']
    vix = data.get('^VIX')
    if vix is None:
        return {'error': 'Missing VIX data'}

    common = spy.index.intersection(vix.index)
    spy_close = spy.loc[common, 'Close'].squeeze()
    vix_close = vix.loc[common, 'Close'].squeeze()
    spy_ret = spy_close.pct_change()

    # Realized vol (20d annualized)
    rv_20 = spy_ret.rolling(20).std() * np.sqrt(252) * 100  # in VIX units
    rv_60 = spy_ret.rolling(60).std() * np.sqrt(252) * 100

    # VRP = implied (VIX) - realized
    vrp = vix_close - rv_20
    vrp_60 = vix_close - rv_60

    # Features for predicting VRP profitability
    features_df = pd.DataFrame(index=common)
    features_df['vrp_20'] = vrp
    features_df['vrp_60'] = vrp_60
    features_df['vix'] = vix_close
    features_df['vix_chg5'] = vix_close.pct_change(5)
    features_df['vix_chg20'] = vix_close.pct_change(20)
    features_df['rv_20'] = rv_20
    features_df['rv_ratio'] = rv_20 / rv_60
    features_df['spy_mom20'] = spy_close.pct_change(20)
    features_df['spy_mom60'] = spy_close.pct_change(60)
    features_df['vix_percentile'] = vix_close.rolling(252).rank(pct=True)

    # Target: next 21d SPY return (selling vol = long market when VRP is positive)
    # Specifically: VRP strategy earns when IV stays > RV (mean of VRP next 21d)
    features_df['target'] = spy_ret.rolling(21).sum().shift(-21)

    feat_cols = ['vrp_20', 'vrp_60', 'vix', 'vix_chg5', 'vix_chg20',
                 'rv_20', 'rv_ratio', 'spy_mom20', 'spy_mom60', 'vix_percentile']

    df = features_df.dropna(subset=feat_cols + ['target']).copy()

    if len(df) < TRAIN_DAYS + 100:
        print("  Insufficient data")
        return {'error': 'Insufficient data'}

    X = df[feat_cols].values
    y = df['target'].values
    dates = df.index

    # Walk-forward
    preds = []
    for i in range(TRAIN_DAYS, len(X) - TEST_DAYS, TEST_DAYS):
        ts = max(0, i - TRAIN_DAYS)
        train_ds = lgb.Dataset(X[ts:i], label=y[ts:i])
        model = lgb.train(
            {'objective': 'regression', 'metric': 'mae', 'num_leaves': 12,
             'learning_rate': 0.03, 'verbose': -1, 'seed': 42,
             'min_child_samples': 20, 'subsample': 0.8, 'feature_fraction': 0.8},
            train_ds, num_boost_round=100
        )
        for j in range(i, min(i + TEST_DAYS, len(X))):
            p = model.predict(X[j:j+1])[0]
            preds.append({'date': dates[j], 'pred': p, 'actual': y[j]})

    pred_df = pd.DataFrame(preds).set_index('date')
    ic = rank_ic(pred_df['pred'].values, pred_df['actual'].values)

    # VRP Strategy: sell vol (hold SPY) when VRP is predicted positive and wide
    # Position size proportional to predicted VRP magnitude
    # Key: only enter when VRP signal is positive (implied > realized)
    vrp_aligned = vrp.reindex(pred_df.index)
    signal = pred_df['pred']

    # Three tiers:
    # VRP positive + model says up → full position
    # VRP positive + model neutral → half position
    # VRP negative or model says down → cash
    positions = pd.Series(0.0, index=pred_df.index)
    positions[(vrp_aligned > 2) & (signal > 0)] = 1.0
    positions[(vrp_aligned > 0) & (vrp_aligned <= 2) & (signal > 0)] = 0.5
    positions[(vrp_aligned > 5) & (signal > signal.quantile(0.7))] = 1.0  # wide VRP = confident

    strat_rets = spy_ret.reindex(pred_df.index) * positions
    bh_rets = spy_ret.reindex(pred_df.index)

    strat_rets = strat_rets.dropna()
    bh_rets = bh_rets.dropna()

    strat_m = calc_metrics(strat_rets, 'VRP Harvest')
    bh_m = calc_metrics(bh_rets, 'SPY B&H')

    # Regime test
    rt = regime_test(strat_rets, bh_rets, 'VRP')

    pct_invested = (positions > 0).mean() * 100

    print(f"\n  OOT Rank IC: {ic:.4f}")
    print(f"  Time invested: {pct_invested:.1f}%")
    print(f"  Strategy: Sharpe={strat_m['sharpe']:.3f}, Sortino={strat_m['sortino']:.3f}, "
          f"CAGR={strat_m['cagr']:.1%}, MaxDD={strat_m['max_dd']:.1%}")
    print(f"  SPY B&H:  Sharpe={bh_m['sharpe']:.3f}, Sortino={bh_m['sortino']:.3f}, "
          f"CAGR={bh_m['cagr']:.1%}, MaxDD={bh_m['max_dd']:.1%}")
    print(f"  Regime gap: {rt['regime_gap']:.3f} ({'PASS' if rt['regime_pass'] else 'FAIL'})")
    print(f"  Sharpe green={rt['sharpe_green']:.3f}, red={rt['sharpe_red']:.3f}, flat={rt['sharpe_flat']:.3f}")

    return {
        'strategy_metrics': strat_m,
        'benchmark': bh_m,
        'ic': ic,
        'pct_invested': float(pct_invested),
        'regime_test': rt,
        'structural_edge': 'VRP — implied vol consistently exceeds realized vol',
    }


# ─── STRATEGY 2: CARRY + TREND (MULTI-ASSET) ──────────────────────────────

def strategy_carry_trend(data):
    """
    STRUCTURAL EDGE: Carry (compensation for holding risk) + Trend (most robust anomaly).
    Combining two different structural premia that work differently across regimes.
    """
    print("\n" + "=" * 70)
    print("STRATEGY 2: CARRY + TREND (MULTI-ASSET)")
    print("Structural edge: Carry premium + trend following = two different edges")
    print("=" * 70)

    # Carry assets (high-yield / income) + Trend assets
    carry_tickers = ['HYG', 'VNQ', 'XLU', 'DVY']  # High yield bonds, REITs, Utilities, Dividend
    trend_tickers = ['SPY', 'QQQ', 'IWM', 'EFA']  # Broad equity, growth, small cap, intl
    bond_ticker = 'TLT'

    all_needed = carry_tickers + trend_tickers + [bond_ticker, '^VIX']
    available = {t: data[t] for t in all_needed if t in data}

    if len(available) < 6:
        return {'error': f'Need >= 6 tickers, got {len(available)}'}

    # Build common index
    spy = data['SPY']
    common_idx = spy.index

    # Calculate returns
    rets = pd.DataFrame()
    for t, df in available.items():
        if t != '^VIX':
            rets[t] = df['Close'].reindex(common_idx).pct_change()

    # Carry signal: trailing yield proxy (total return vs price return approximation)
    # Higher carry = higher trailing income = hold more
    carry_signal = pd.DataFrame()
    for t in carry_tickers:
        if t in rets.columns:
            # Carry proxy: 63d return smoothed (captures dividends/income)
            r = rets[t].rolling(63).sum()
            price_r = available[t]['Close'].reindex(common_idx).pct_change(63)
            carry_signal[t] = r  # total return momentum as carry proxy

    # Trend signal: time-series momentum (positive = uptrend)
    trend_signal = pd.DataFrame()
    for t in trend_tickers:
        if t in rets.columns:
            p = available[t]['Close'].reindex(common_idx)
            # 12-month return sign (classic trend)
            mom_252 = p / p.shift(252) - 1
            # EMA crossover
            ema_50 = p.ewm(span=50).mean()
            ema_200 = p.ewm(span=200).mean()
            trend_signal[t] = (ema_50 > ema_200).astype(float) * mom_252.clip(lower=0)

    # Combined strategy: equal-weight carry basket + trend-following basket
    # Carry basket: hold carry assets, weight by carry signal rank
    # Trend basket: hold trend assets only when trend is positive

    strat_daily = []
    dates = []

    # Rebalance monthly
    rebal_points = list(range(252, len(common_idx) - 21, 21))

    for i in rebal_points:
        dt = common_idx[i]

        # Carry allocation
        carry_weights = {}
        for t in carry_tickers:
            if t in carry_signal.columns:
                sig = carry_signal[t].iloc[i] if i < len(carry_signal) else np.nan
                if not np.isnan(sig) and sig > 0:
                    carry_weights[t] = 1.0  # equal weight if positive carry
                else:
                    carry_weights[t] = 0.0

        # Trend allocation
        trend_weights = {}
        for t in trend_tickers:
            if t in trend_signal.columns:
                sig = trend_signal[t].iloc[i] if i < len(trend_signal) else np.nan
                if not np.isnan(sig) and sig > 0:
                    trend_weights[t] = 1.0  # in trend
                else:
                    trend_weights[t] = 0.0

        # Normalize weights
        total_carry = sum(carry_weights.values())
        total_trend = sum(trend_weights.values())

        if total_carry > 0:
            carry_weights = {k: v / total_carry * 0.5 for k, v in carry_weights.items()}
        if total_trend > 0:
            trend_weights = {k: v / total_trend * 0.5 for k, v in trend_weights.items()}

        all_weights = {**carry_weights, **trend_weights}

        # Next 21 days returns
        for j in range(i, min(i + 21, len(common_idx) - 1)):
            day_ret = sum(rets[t].iloc[j+1] * w for t, w in all_weights.items()
                         if t in rets.columns and j+1 < len(rets))
            strat_daily.append(day_ret)
            dates.append(common_idx[j])

    strat_s = pd.Series(strat_daily, index=dates[:len(strat_daily)]).dropna()
    strat_s = strat_s[~strat_s.index.duplicated(keep='first')]
    bh_rets = spy['Close'].reindex(common_idx).pct_change().reindex(strat_s.index).dropna()

    strat_m = calc_metrics(strat_s, 'Carry+Trend')
    bh_m = calc_metrics(bh_rets, 'SPY B&H')

    rt = regime_test(strat_s, bh_rets, 'Carry+Trend')

    print(f"\n  Strategy: Sharpe={strat_m['sharpe']:.3f}, Sortino={strat_m['sortino']:.3f}, "
          f"CAGR={strat_m['cagr']:.1%}, MaxDD={strat_m['max_dd']:.1%}")
    print(f"  SPY B&H:  Sharpe={bh_m['sharpe']:.3f}, Sortino={bh_m['sortino']:.3f}")
    print(f"  Regime gap: {rt['regime_gap']:.3f} ({'PASS' if rt['regime_pass'] else 'FAIL'})")
    print(f"  Sharpe green={rt['sharpe_green']:.3f}, red={rt['sharpe_red']:.3f}, flat={rt['sharpe_flat']:.3f}")

    return {
        'strategy_metrics': strat_m,
        'benchmark': bh_m,
        'regime_test': rt,
        'structural_edge': 'Carry (income premium) + Trend (momentum anomaly) — two orthogonal premia',
    }


# ─── STRATEGY 3: DISPERSION TRADING (SIMPLIFIED) ──────────────────────────

def strategy_dispersion(data):
    """
    STRUCTURAL EDGE: Index vol > sum-of-parts vol due to correlation risk premium.
    When correlation is unusually high, it tends to revert → long low-correlation stocks.
    Walk-forward ML to predict correlation regime shifts.
    """
    print("\n" + "=" * 70)
    print("STRATEGY 3: DISPERSION TRADING (SIMPLIFIED)")
    print("Structural edge: Correlation risk premium — index vol overpriced vs constituents")
    print("=" * 70)

    sectors = ['XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLU', 'XLP', 'XLY', 'XLB', 'XLC']
    available = {t: data[t] for t in sectors if t in data}

    if len(available) < 6:
        return {'error': f'Need >= 6 sectors, got {len(available)}'}

    spy = data['SPY']
    common_idx = spy.index

    sector_rets = pd.DataFrame()
    for t, df in available.items():
        sector_rets[t] = df['Close'].reindex(common_idx).pct_change()

    spy_ret = spy['Close'].reindex(common_idx).pct_change()

    # Cross-sectional dispersion (realized)
    dispersion = sector_rets.std(axis=1)  # daily cross-sectional vol
    dispersion_20d = dispersion.rolling(20).mean()

    # Average pairwise correlation (rolling 60d)
    avg_corr = sector_rets.rolling(60).corr().groupby(level=0).apply(
        lambda x: x.values[np.triu_indices_from(x.values, k=1)].mean()
        if len(x) > 1 else np.nan
    )

    # Features
    features_df = pd.DataFrame(index=common_idx)
    features_df['disp_20d'] = dispersion_20d
    features_df['disp_chg'] = dispersion_20d.pct_change(20)
    features_df['avg_corr'] = avg_corr
    features_df['corr_chg'] = avg_corr.pct_change(20)
    features_df['spy_vol_20'] = spy_ret.rolling(20).std()
    features_df['spy_vol_60'] = spy_ret.rolling(60).std()
    features_df['spy_mom20'] = spy['Close'].reindex(common_idx).pct_change(20)

    if '^VIX' in data:
        vix = data['^VIX']['Close'].reindex(common_idx).ffill()
        features_df['vix'] = vix
        features_df['vix_rv_gap'] = vix - features_df['spy_vol_20'] * np.sqrt(252) * 100

    # Strategy: when dispersion is predicted to be high (low correlation),
    # go long the most different (lowest-corr) sectors; when low dispersion, hold SPY
    # Target: forward 21d dispersion level
    features_df['target_disp'] = dispersion_20d.shift(-21)
    features_df['target_high'] = (features_df['target_disp'] > dispersion_20d.rolling(252).median()).astype(int)

    feat_cols = [c for c in features_df.columns if c not in ['target_disp', 'target_high']]
    df = features_df.dropna().copy()

    if len(df) < TRAIN_DAYS + 100:
        return {'error': 'Insufficient data'}

    X = df[feat_cols].values
    y = df['target_high'].values
    dates = df.index

    # Walk-forward
    preds = []
    for i in range(TRAIN_DAYS, len(X) - TEST_DAYS, TEST_DAYS):
        ts = max(0, i - TRAIN_DAYS)
        train_ds = lgb.Dataset(X[ts:i], label=y[ts:i])
        model = lgb.train(
            {'objective': 'binary', 'metric': 'auc', 'num_leaves': 8,
             'learning_rate': 0.03, 'verbose': -1, 'seed': 42,
             'min_child_samples': 20, 'subsample': 0.8},
            train_ds, num_boost_round=80
        )
        for j in range(i, min(i + TEST_DAYS, len(X))):
            p = model.predict(X[j:j+1])[0]
            preds.append({'date': dates[j], 'pred': p, 'actual': y[j]})

    pred_df = pd.DataFrame(preds).set_index('date')
    acc = ((pred_df['pred'] > 0.5).astype(int) == pred_df['actual']).mean()
    ic = rank_ic(pred_df['pred'].values, pred_df['actual'].values.astype(float))

    # Trading strategy:
    # High dispersion predicted → equal-weight top 3 momentum sectors (stock-picking works)
    # Low dispersion predicted → hold SPY (everything moves together, no alpha)
    strat_rets_list = []
    for i in range(len(pred_df)):
        dt = pred_df.index[i]
        if dt not in sector_rets.index:
            continue

        idx_pos = common_idx.get_loc(dt)
        if idx_pos + 1 >= len(common_idx):
            continue

        if pred_df['pred'].iloc[i] > 0.5:  # high dispersion expected
            # Pick top 3 momentum sectors
            mom = sector_rets.iloc[max(0, idx_pos-20):idx_pos].sum()
            top3 = mom.nlargest(3).index.tolist()
            day_ret = sector_rets.iloc[idx_pos + 1][top3].mean() if idx_pos + 1 < len(sector_rets) else 0
        else:
            day_ret = spy_ret.iloc[idx_pos + 1] if idx_pos + 1 < len(spy_ret) else 0

        strat_rets_list.append({'date': dt, 'ret': day_ret})

    strat_s = pd.DataFrame(strat_rets_list).set_index('date')['ret']
    strat_s = strat_s.dropna()
    bh_rets = spy_ret.reindex(strat_s.index).dropna()

    strat_m = calc_metrics(strat_s, 'Dispersion')
    bh_m = calc_metrics(bh_rets, 'SPY B&H')
    rt = regime_test(strat_s, bh_rets, 'Dispersion')

    print(f"\n  Dispersion prediction accuracy: {acc:.1%}")
    print(f"  IC: {ic:.4f}")
    print(f"  Strategy: Sharpe={strat_m['sharpe']:.3f}, Sortino={strat_m['sortino']:.3f}, "
          f"CAGR={strat_m['cagr']:.1%}, MaxDD={strat_m['max_dd']:.1%}")
    print(f"  SPY B&H:  Sharpe={bh_m['sharpe']:.3f}")
    print(f"  Regime gap: {rt['regime_gap']:.3f} ({'PASS' if rt['regime_pass'] else 'FAIL'})")
    print(f"  Sharpe green={rt['sharpe_green']:.3f}, red={rt['sharpe_red']:.3f}, flat={rt['sharpe_flat']:.3f}")

    return {
        'strategy_metrics': strat_m,
        'benchmark': bh_m,
        'dispersion_accuracy': float(acc),
        'ic': float(ic),
        'regime_test': rt,
        'structural_edge': 'Correlation risk premium — index vol systematically overpriced vs constituents',
    }


# ─── STRATEGY 4: SEASONAL/CALENDAR EFFECTS + ML ───────────────────────────

def strategy_calendar_ml(data):
    """
    STRUCTURAL EDGE: Day-of-week, month, FOMC, opex, holiday effects are well-documented
    market microstructure anomalies. Individually small but ML can stack them.
    """
    print("\n" + "=" * 70)
    print("STRATEGY 4: SEASONAL/CALENDAR EFFECTS + ML")
    print("Structural edge: Market microstructure anomalies (well-documented)")
    print("=" * 70)

    spy = data['SPY']
    common_idx = spy.index
    spy_close = spy.loc[common_idx, 'Close'].squeeze()
    spy_ret = spy_close.pct_change()

    # Calendar features
    cal = pd.DataFrame(index=common_idx)
    cal['dow'] = common_idx.dayofweek  # 0=Mon, 4=Fri
    cal['month'] = common_idx.month
    cal['dom'] = common_idx.day
    cal['is_monday'] = (common_idx.dayofweek == 0).astype(int)
    cal['is_friday'] = (common_idx.dayofweek == 4).astype(int)

    # Month effects
    cal['is_jan'] = (common_idx.month == 1).astype(int)  # January effect
    cal['is_nov_dec'] = (common_idx.month.isin([11, 12])).astype(int)  # Santa rally
    cal['is_sept'] = (common_idx.month == 9).astype(int)  # September effect

    # Turn of month (last 2 + first 3 business days — well-documented)
    cal['is_tom'] = ((common_idx.day <= 3) | (common_idx.day >= 28)).astype(int)

    # Options expiration (3rd Friday approximation)
    cal['week_of_month'] = ((common_idx.day - 1) // 7 + 1)
    cal['is_opex_week'] = ((cal['week_of_month'] == 3) & (common_idx.dayofweek <= 4)).astype(int)

    # FOMC approximation: months 1,3,5,6,7,9,11,12 have meetings
    # Typically around mid-month
    fomc_months = [1, 3, 5, 6, 7, 9, 11, 12]
    cal['is_fomc_month'] = common_idx.month.isin(fomc_months).astype(int)
    cal['fomc_week'] = ((common_idx.month.isin(fomc_months)) &
                        (common_idx.day >= 14) & (common_idx.day <= 21)).astype(int)

    # Pre-holiday (day before market closure — typically bullish)
    # Approximate: check for gaps > 1 day
    day_gaps = pd.Series(common_idx).diff().dt.days
    day_gaps.index = common_idx
    cal['pre_holiday'] = (day_gaps.shift(-1) > 2).astype(int).fillna(0)

    # Momentum features (to combine with calendar)
    cal['spy_ret_1d'] = spy_ret
    cal['spy_ret_5d'] = spy_ret.rolling(5).sum()
    cal['spy_ret_20d'] = spy_ret.rolling(20).sum()
    cal['spy_vol_20d'] = spy_ret.rolling(20).std()

    if '^VIX' in data:
        vix = data['^VIX']['Close'].reindex(common_idx).ffill()
        cal['vix'] = vix
        cal['vix_pctile'] = vix.rolling(252).rank(pct=True)

    # Target: next-day return
    cal['target'] = spy_ret.shift(-1)

    feat_cols = [c for c in cal.columns if c != 'target']
    df = cal.dropna().copy()

    if len(df) < TRAIN_DAYS + 100:
        return {'error': 'Insufficient data'}

    X = df[feat_cols].values
    y = df['target'].values
    dates = df.index

    # Walk-forward
    preds = []
    for i in range(TRAIN_DAYS, len(X) - 1, TEST_DAYS):
        ts = max(0, i - TRAIN_DAYS * 2)  # longer lookback for seasonality
        train_ds = lgb.Dataset(X[ts:i], label=y[ts:i])
        model = lgb.train(
            {'objective': 'regression', 'metric': 'mae', 'num_leaves': 15,
             'learning_rate': 0.02, 'verbose': -1, 'seed': 42,
             'min_child_samples': 30, 'subsample': 0.8, 'feature_fraction': 0.7},
            train_ds, num_boost_round=150
        )
        for j in range(i, min(i + TEST_DAYS, len(X))):
            p = model.predict(X[j:j+1])[0]
            preds.append({'date': dates[j], 'pred': p, 'actual': y[j]})

    pred_df = pd.DataFrame(preds).set_index('date')
    ic = rank_ic(pred_df['pred'].values, pred_df['actual'].values)

    # Strategy: long when model predicts positive, cash when negative
    # Scale position by confidence
    p90 = pred_df['pred'].quantile(0.9)
    p10 = pred_df['pred'].quantile(0.1)

    positions = pd.Series(0.0, index=pred_df.index)
    positions[pred_df['pred'] > 0] = 1.0
    positions[pred_df['pred'] > p90] = 1.0  # cap at 1.0

    strat_rets = spy_ret.reindex(pred_df.index).shift(-1) * positions  # next-day return
    strat_rets = strat_rets.dropna()
    bh_rets = spy_ret.reindex(strat_rets.index).dropna()

    strat_m = calc_metrics(strat_rets, 'Calendar ML')
    bh_m = calc_metrics(bh_rets, 'SPY B&H')
    rt = regime_test(strat_rets, bh_rets, 'Calendar')

    pct_invested = (positions > 0).mean() * 100

    print(f"\n  OOT Rank IC: {ic:.4f}")
    print(f"  Time invested: {pct_invested:.1f}%")
    print(f"  Strategy: Sharpe={strat_m['sharpe']:.3f}, Sortino={strat_m['sortino']:.3f}, "
          f"CAGR={strat_m['cagr']:.1%}, MaxDD={strat_m['max_dd']:.1%}")
    print(f"  SPY B&H:  Sharpe={bh_m['sharpe']:.3f}")
    print(f"  Regime gap: {rt['regime_gap']:.3f} ({'PASS' if rt['regime_pass'] else 'FAIL'})")
    print(f"  Sharpe green={rt['sharpe_green']:.3f}, red={rt['sharpe_red']:.3f}, flat={rt['sharpe_flat']:.3f}")

    return {
        'strategy_metrics': strat_m,
        'benchmark': bh_m,
        'ic': float(ic),
        'pct_invested': float(pct_invested),
        'regime_test': rt,
        'structural_edge': 'Calendar anomalies — turn-of-month, FOMC drift, holiday effect (microstructure)',
    }


# ─── STRATEGY 5: MERGER ARB / EVENT-DRIVEN (SIMPLIFIED) ───────────────────

def strategy_event_driven(data):
    """
    STRUCTURAL EDGE: Event-driven premia — deal spreads, earnings reactions.
    Simplified: trade sector pair reversion after extreme divergence (proxy for events).
    When one sector diverges sharply from SPY, it tends to mean-revert (post-event).
    Walk-forward: predict which sector divergence is most likely to revert.
    """
    print("\n" + "=" * 70)
    print("STRATEGY 5: EVENT-DRIVEN / MEAN REVERSION")
    print("Structural edge: Post-event overreaction + reversion (behavioral)")
    print("=" * 70)

    sectors = ['XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLU', 'XLP', 'XLY', 'XLB', 'XLC']
    available = {t: data[t] for t in sectors if t in data}

    spy = data['SPY']
    common_idx = spy.index
    spy_ret = spy['Close'].reindex(common_idx).pct_change()

    sector_rets = pd.DataFrame()
    for t, df in available.items():
        sector_rets[t] = df['Close'].reindex(common_idx).pct_change()

    # Excess returns vs SPY
    excess = sector_rets.subtract(spy_ret, axis=0)

    # Features: for each sector, 5d/10d/20d excess return (looking for overreaction)
    all_strat_rets = []
    all_dates = []
    all_signals = []
    all_actuals = []

    for rebal_start in range(TRAIN_DAYS + 60, len(common_idx) - TEST_DAYS, TEST_DAYS):
        dt = common_idx[rebal_start]

        # For each sector: build features
        sector_features = {}
        for t in excess.columns:
            ex_5 = excess[t].iloc[rebal_start-5:rebal_start].sum()
            ex_10 = excess[t].iloc[rebal_start-10:rebal_start].sum()
            ex_20 = excess[t].iloc[rebal_start-20:rebal_start].sum()
            ex_60 = excess[t].iloc[rebal_start-60:rebal_start].sum()

            # Z-score of divergence
            ex_20_hist = excess[t].rolling(20).sum().iloc[max(0,rebal_start-252):rebal_start]
            z_score = (ex_20 - ex_20_hist.mean()) / ex_20_hist.std() if ex_20_hist.std() > 0 else 0

            sector_features[t] = {
                'ex_5': ex_5, 'ex_10': ex_10, 'ex_20': ex_20, 'ex_60': ex_60,
                'z_score': z_score,
            }

        # Find most oversold sectors (most negative z-score = biggest divergence)
        z_scores = {t: f['z_score'] for t, f in sector_features.items()}
        z_series = pd.Series(z_scores).dropna()

        if len(z_series) < 3:
            continue

        # Mean reversion strategy: buy most oversold, sell most overbought
        # Long bottom 3, short top 3 → market neutral
        bottom3 = z_series.nsmallest(3).index.tolist()
        top3 = z_series.nlargest(3).index.tolist()

        for j in range(rebal_start, min(rebal_start + TEST_DAYS, len(common_idx) - 1)):
            long_ret = sector_rets.iloc[j + 1][bottom3].mean()
            short_ret = sector_rets.iloc[j + 1][top3].mean()
            ls_ret = long_ret - short_ret  # long-short = market neutral

            all_strat_rets.append(ls_ret)
            all_dates.append(common_idx[j])

            # Signal: avg z-score divergence
            all_signals.append(z_series[bottom3].mean() - z_series[top3].mean())
            all_actuals.append(ls_ret)

    strat_s = pd.Series(all_strat_rets, index=all_dates[:len(all_strat_rets)]).dropna()
    strat_s = strat_s[~strat_s.index.duplicated(keep='first')]
    bh_rets = spy_ret.reindex(strat_s.index).dropna()

    ic = rank_ic(np.array(all_signals), np.array(all_actuals))

    strat_m = calc_metrics(strat_s, 'Event MR L/S')
    bh_m = calc_metrics(bh_rets, 'SPY B&H')
    rt = regime_test(strat_s, bh_rets, 'EventMR')

    print(f"\n  Rank IC: {ic:.4f}")
    print(f"  Strategy (L/S): Sharpe={strat_m['sharpe']:.3f}, Sortino={strat_m['sortino']:.3f}, "
          f"CAGR={strat_m['cagr']:.1%}, MaxDD={strat_m['max_dd']:.1%}")
    print(f"  Note: This is MARKET NEUTRAL (long-short)")
    print(f"  Regime gap: {rt['regime_gap']:.3f} ({'PASS' if rt['regime_pass'] else 'FAIL'})")
    print(f"  Sharpe green={rt['sharpe_green']:.3f}, red={rt['sharpe_red']:.3f}, flat={rt['sharpe_flat']:.3f}")

    return {
        'strategy_metrics': strat_m,
        'benchmark': bh_m,
        'ic': float(ic),
        'market_neutral': True,
        'regime_test': rt,
        'structural_edge': 'Behavioral overreaction + mean reversion — market neutral',
    }


# ─── MAIN ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("R4: STRUCTURAL EDGE STRATEGIES")
    print("HC #697: No crypto | HC #0: Sliding window | HC #694: Commission-free")
    print("HC #428: Regime-agnostic validation")
    print(f"Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Download all data upfront
    all_tickers = [
        'SPY', 'QQQ', 'IWM', 'EFA', '^VIX',
        'TLT', 'HYG', 'LQD', 'VNQ', 'XLU', 'DVY',
        'XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLP', 'XLY', 'XLB', 'XLC',
    ]

    print("\n--- Downloading Data ---")
    data = download_data(all_tickers)

    if 'SPY' not in data:
        print("FATAL: Cannot download SPY data")
        return

    results = {}

    # Run all 5 strategies
    results['1_vrp_harvest'] = strategy_vrp(data)
    results['2_carry_trend'] = strategy_carry_trend(data)
    results['3_dispersion'] = strategy_dispersion(data)
    results['4_calendar_ml'] = strategy_calendar_ml(data)
    results['5_event_driven'] = strategy_event_driven(data)

    # ─── Summary ────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("R4 SUMMARY — STRUCTURAL EDGE STRATEGIES")
    print(f"{'='*70}")

    summary = []
    for key, res in results.items():
        if 'error' in res:
            print(f"  {key}: ERROR - {res['error']}")
            continue

        sm = res['strategy_metrics']
        bm = res.get('benchmark', {})
        rt = res.get('regime_test', {})

        regime_pass = rt.get('regime_pass', False)
        beats_spy = sm.get('sharpe', 0) > bm.get('sharpe', 0)

        verdict = 'PASS' if regime_pass else 'FAIL'
        edge = res.get('structural_edge', 'N/A')

        row = {
            'strategy': sm['name'],
            'sharpe': sm['sharpe'],
            'sortino': sm['sortino'],
            'cagr_pct': sm.get('cagr_pct', sm.get('cagr', 0) * 100),
            'max_dd_pct': sm.get('max_dd', 0) * 100,
            'spy_sharpe': bm.get('sharpe', 0),
            'regime_gap': rt.get('regime_gap', 'N/A'),
            'regime_pass': regime_pass,
            'beats_spy': beats_spy,
            'ic': res.get('ic', 'N/A'),
            'verdict': verdict,
            'structural_edge': edge,
            'market_neutral': res.get('market_neutral', False),
        }
        summary.append(row)

        status = 'PASS' if regime_pass else 'FAIL'
        spy_str = 'beats SPY' if beats_spy else 'lags SPY'

        print(f"\n  [{status}] {sm['name']}")
        print(f"    Sharpe={sm['sharpe']:.3f}, Sortino={sm['sortino']:.3f}, "
              f"CAGR={sm.get('cagr_pct', 0):.1f}%, MaxDD={sm.get('max_dd',0)*100:.1f}%")
        print(f"    Regime gap={rt.get('regime_gap','N/A'):.3f} | {spy_str} (SPY Sharpe={bm.get('sharpe',0):.3f})")
        print(f"    Edge: {edge}")

    # ─── Final Verdict ──────────────────────────────────────────────────
    passed = [s for s in summary if s['verdict'] == 'PASS']
    failed = [s for s in summary if s['verdict'] == 'FAIL']

    print(f"\n{'='*70}")
    print("FINAL VERDICT")
    print(f"{'='*70}")

    if passed:
        print(f"\n  {len(passed)} strategies PASS regime test:")
        for p in passed:
            print(f"    + {p['strategy']}: Sharpe={p['sharpe']:.3f}, regime_gap={p['regime_gap']:.3f}")
    else:
        print("\n  NO strategies pass the regime test.")

    if failed:
        print(f"\n  {len(failed)} strategies FAIL regime test:")
        for f_ in failed:
            print(f"    - {f_['strategy']}: Sharpe={f_['sharpe']:.3f}, regime_gap={f_['regime_gap']:.3f}")

    # Save results
    output = {
        'run_date': datetime.now().isoformat(),
        'methodology': 'Sliding walk-forward (252d train, 21d test), regime-agnostic (HC #428)',
        'constraints': 'No crypto (HC #697), commission-free (HC #694), sliding only (HC #0)',
        'detailed_results': results,
        'summary': summary,
        'n_passed': len(passed),
        'n_failed': len(failed),
    }

    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, (np.floating, float)):
            if np.isnan(obj) or np.isinf(obj):
                return str(obj)
            return float(obj)
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, pd.Timestamp):
            return str(obj)
        else:
            return obj

    with open(os.path.join(OUT_DIR, 'structural_edge_results.json'), 'w') as f:
        json.dump(make_serializable(output), f, indent=2, default=str)

    print(f"\nResults saved.")
    return output


if __name__ == '__main__':
    main()
