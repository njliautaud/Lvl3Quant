#!/usr/bin/env python3
"""
R5 TASK 2: UNCONVENTIONAL GROWTH ANGLES
HC #697: No crypto | HC #0: Sliding window only | HC #694: Commission-free
HC #428: Regime test (report honestly)

5 strategies:
1. Merger Arb / Event-Driven (simplified via spread proxy)
2. Seasonality + Calendar Effects (walk-forward predictive IC)
3. Volatility as Asset Class (long vol timing)
4. Gold Trend Following (momentum/breakout on GLD/GDX)
5. Tail-Hedged Growth (QQQ + systematic put spreads)

Each: sliding walk-forward, regime test, honest assessment.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import spearmanr, norm
import lightgbm as lgb
import json
import os
from datetime import datetime

OUT_DIR = '/home/jupiter/Lvl3Quant/output/growth_research_r5'
os.makedirs(OUT_DIR, exist_ok=True)

TRAIN_DAYS = 252
TEST_DAYS = 21
START = '2010-01-01'
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
    if max_dd < -1.0:
        max_dd = -1.0
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
        'sharpe_green': float(sg), 'sharpe_red': float(sr),
        'sharpe_flat': float(results['flat']['sharpe']),
        'regime_gap': float(gap), 'regime_pass': gap < 0.50,
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


def compute_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


# ─── STRATEGY 1: MERGER ARB / EVENT-DRIVEN (PROXY) ───────────────────────

def strategy_merger_arb(data):
    """
    Merger arb proxy: MNA ETF tracks merger arb returns.
    We test: can we predict MNA vs SPY relative returns?
    Signals: credit spreads (HYG-TLT), VIX level, market breadth.
    Walk-forward: predict next-month MNA alpha over SPY.
    """
    print("\n" + "=" * 70)
    print("STRATEGY 1: MERGER ARB / EVENT-DRIVEN")
    print("=" * 70)

    # MNA (IQ Merger Arbitrage ETF) as proxy
    needed = ['MNA', 'SPY', 'HYG', 'TLT']
    available = {t: data[t] for t in needed if t in data}

    if 'MNA' not in available:
        print("  MNA not available. Using alternative: long HYG short-term relative value.")
        # Fallback: HYG (high yield) tends to behave like merger arb (credit spread duration)
        if 'HYG' not in available or 'SPY' not in available:
            return {'error': 'Missing required data'}

        spy_close = available['SPY']['Close']
        hyg_close = available['HYG']['Close']
        common = spy_close.index.intersection(hyg_close.index)
        spy_ret = spy_close.reindex(common).pct_change()
        hyg_ret = hyg_close.reindex(common).pct_change()

        # Credit spread signal: HYG vs TLT relative return
        if 'TLT' in available:
            tlt_ret = available['TLT']['Close'].reindex(common).pct_change()
            spread_ret = hyg_ret - tlt_ret  # credit spread tightening = positive
        else:
            spread_ret = hyg_ret

        # Walk-forward: predict next 21d merger-arb-like returns
        # Features: trailing spread, vol, momentum
        strat_rets = []
        for i in range(TRAIN_DAYS, len(common) - TEST_DAYS, TEST_DAYS):
            train_end = i
            train_start = max(0, i - TRAIN_DAYS)

            # Features for training
            features_list = []
            targets_list = []
            for j in range(train_start + 63, train_end):
                f = {
                    'spread_21d': spread_ret.iloc[j-21:j].mean(),
                    'spread_vol_21d': spread_ret.iloc[j-21:j].std(),
                    'spy_mom_63': spy_ret.iloc[j-63:j].sum(),
                    'hyg_mom_21': hyg_ret.iloc[j-21:j].sum(),
                    'spread_zscore': (spread_ret.iloc[j-21:j].mean() - spread_ret.iloc[j-63:j].mean()) /
                                     max(spread_ret.iloc[j-63:j].std(), 1e-6),
                }
                # Target: next 21d merger arb return
                fwd = spread_ret.iloc[j:j+21].sum() if j + 21 <= train_end else np.nan
                if not np.isnan(fwd):
                    features_list.append(f)
                    targets_list.append(fwd)

            if len(features_list) < 30:
                continue

            X_train = pd.DataFrame(features_list)
            y_train = np.array(targets_list)

            # Predict for test period
            test_features = {
                'spread_21d': spread_ret.iloc[i-21:i].mean(),
                'spread_vol_21d': spread_ret.iloc[i-21:i].std(),
                'spy_mom_63': spy_ret.iloc[i-63:i].sum(),
                'hyg_mom_21': hyg_ret.iloc[i-21:i].sum(),
                'spread_zscore': (spread_ret.iloc[i-21:i].mean() - spread_ret.iloc[i-63:i].mean()) /
                                  max(spread_ret.iloc[i-63:i].std(), 1e-6),
            }
            X_test = pd.DataFrame([test_features])

            try:
                model = lgb.LGBMRegressor(n_estimators=50, max_depth=3, learning_rate=0.1,
                                          verbose=-1, n_jobs=1)
                model.fit(X_train, y_train)
                pred = model.predict(X_test)[0]
            except:
                pred = 0

            # Position: if predicted spread return > 0, go long merger arb (long HYG short TLT)
            # Scale position by confidence
            pos = np.clip(pred * 20, -1, 1)  # scale to [-1, 1]

            for j in range(i, min(i + TEST_DAYS, len(common) - 1)):
                daily_ret = spread_ret.iloc[j] * pos
                strat_rets.append({'date': common[j], 'ret': daily_ret})

        if not strat_rets:
            return {'error': 'No trades generated'}

        strat_s = pd.DataFrame(strat_rets).set_index('date')['ret']
        strat_s = strat_s[~strat_s.index.duplicated(keep='first')].dropna()

        m = calc_metrics(strat_s, 'Merger Arb Proxy')
        rt = regime_test(strat_s, spy_ret.reindex(strat_s.index).dropna(), 'MergerArb')

        print(f"\n  Merger Arb Proxy: Sharpe={m['sharpe']:.3f}, Sortino={m['sortino']:.3f}, "
              f"CAGR={m['cagr']:.1%}, MaxDD={m['max_dd']:.1%}")
        print(f"  Regime gap: {rt['regime_gap']:.3f} ({'PASS' if rt['regime_pass'] else 'FAIL'})")
        print(f"  Green Sharpe={rt['sharpe_green']:.3f}, Red Sharpe={rt['sharpe_red']:.3f}")

        return {'metrics': m, 'regime': rt, 'type': 'merger_arb_proxy'}

    # If MNA is available, use it directly
    spy_ret = available['SPY']['Close'].pct_change()
    mna_ret = available['MNA']['Close'].pct_change()
    common = spy_ret.index.intersection(mna_ret.index)
    spy_ret = spy_ret.reindex(common)
    mna_ret = mna_ret.reindex(common)

    m = calc_metrics(mna_ret, 'MNA (Merger Arb ETF)')
    rt = regime_test(mna_ret, spy_ret, 'MNA')

    print(f"\n  MNA ETF: Sharpe={m['sharpe']:.3f}, CAGR={m['cagr']:.1%}, "
          f"Regime gap={rt['regime_gap']:.3f}")

    return {'metrics': m, 'regime': rt, 'type': 'mna_etf'}


# ─── STRATEGY 2: SEASONALITY + CALENDAR EFFECTS ──────────────────────────

def strategy_seasonality(data):
    """
    Test calendar effects with walk-forward IC validation:
    - Turn-of-month (last 3 + first 3 trading days)
    - Sell-in-May (Nov-Apr vs May-Oct)
    - Holiday effect (day before holiday)
    - OPEX (options expiration week)
    - January effect

    Walk-forward: train LGBM on calendar features to predict next-day SPY return.
    """
    print("\n" + "=" * 70)
    print("STRATEGY 2: SEASONALITY + CALENDAR EFFECTS")
    print("=" * 70)

    if 'SPY' not in data:
        return {'error': 'No SPY data'}

    spy = data['SPY']
    spy_ret = spy['Close'].pct_change()
    idx = spy_ret.index

    # Build calendar features
    features = pd.DataFrame(index=idx)
    features['day_of_week'] = idx.dayofweek
    features['day_of_month'] = idx.day
    features['month'] = idx.month
    features['is_monday'] = (idx.dayofweek == 0).astype(int)
    features['is_friday'] = (idx.dayofweek == 4).astype(int)

    # Turn of month: last 3 and first 3 days
    features['turn_of_month'] = ((idx.day <= 3) | (idx.day >= 27)).astype(int)

    # Sell in May: May-Oct = 0, Nov-Apr = 1
    features['winter'] = (idx.month.isin([11, 12, 1, 2, 3, 4])).astype(int)

    # January effect
    features['january'] = (idx.month == 1).astype(int)

    # Options expiration: third Friday of month (approximate: day 15-21 and Friday)
    features['opex_week'] = ((idx.day >= 15) & (idx.day <= 21) & (idx.dayofweek <= 4)).astype(int)

    # Pre-holiday (day before market holiday = gap in trading days)
    trading_gaps = pd.Series(idx, index=idx).diff().dt.days
    features['pre_holiday'] = (trading_gaps.shift(-1) > 1).astype(int)  # tomorrow is a holiday

    # Momentum context (recent returns affect seasonal patterns)
    features['ret_5d'] = spy_ret.rolling(5).mean()
    features['ret_21d'] = spy_ret.rolling(21).mean()
    features['vol_21d'] = spy_ret.rolling(21).std()

    target = spy_ret.shift(-1)  # predict next-day return

    # Walk-forward with LGBM
    strat_rets = []
    ics = []

    for i in range(TRAIN_DAYS, len(idx) - TEST_DAYS, TEST_DAYS):
        train_start = max(0, i - TRAIN_DAYS)
        X_train = features.iloc[train_start:i].dropna()
        y_train = target.reindex(X_train.index).dropna()
        common_train = X_train.index.intersection(y_train.index)
        X_train = X_train.loc[common_train]
        y_train = y_train.loc[common_train]

        if len(X_train) < 50:
            continue

        # Test data
        test_end = min(i + TEST_DAYS, len(idx) - 1)
        X_test = features.iloc[i:test_end].dropna()
        y_test = target.reindex(X_test.index).dropna()
        common_test = X_test.index.intersection(y_test.index)
        X_test = X_test.loc[common_test]
        y_test = y_test.loc[common_test]

        if len(X_test) < 5:
            continue

        try:
            model = lgb.LGBMRegressor(n_estimators=50, max_depth=3, learning_rate=0.05,
                                       verbose=-1, n_jobs=1, min_child_samples=20)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
        except:
            continue

        # IC
        ic, _ = spearmanr(preds, y_test.values)
        if not np.isnan(ic):
            ics.append(ic)

        # Position: long if predicted positive, short if negative
        for j, (dt, pred) in enumerate(zip(X_test.index, preds)):
            pos = np.clip(pred * 500, -1, 1)  # scale
            actual = y_test.loc[dt] if dt in y_test.index else 0
            strat_rets.append({'date': dt, 'ret': pos * actual})

    if not strat_rets:
        return {'error': 'No trades generated'}

    strat_s = pd.DataFrame(strat_rets).set_index('date')['ret']
    strat_s = strat_s[~strat_s.index.duplicated(keep='first')].dropna()

    m = calc_metrics(strat_s, 'Seasonality + Calendar')
    rt = regime_test(strat_s, spy_ret.reindex(strat_s.index).dropna(), 'Seasonality')
    avg_ic = np.mean(ics) if ics else 0

    print(f"\n  Seasonality: Sharpe={m['sharpe']:.3f}, Sortino={m['sortino']:.3f}, "
          f"CAGR={m['cagr']:.1%}, MaxDD={m['max_dd']:.1%}")
    print(f"  Walk-forward IC: {avg_ic:.4f} (n={len(ics)})")
    print(f"  Regime gap: {rt['regime_gap']:.3f} ({'PASS' if rt['regime_pass'] else 'FAIL'})")

    # Also test simple overlay: only trade during favorable calendar windows
    print("\n  Simple overlay test: long SPY only during favorable periods...")
    favorable = features['turn_of_month'] | features['winter']
    simple_rets = spy_ret * favorable.astype(float)
    simple_rets = simple_rets.iloc[TRAIN_DAYS:]  # skip lookback
    m_simple = calc_metrics(simple_rets, 'Calendar Overlay (simple)')
    rt_simple = regime_test(simple_rets, spy_ret.reindex(simple_rets.index).dropna(), 'CalOverlay')

    print(f"  Calendar Overlay: Sharpe={m_simple['sharpe']:.3f}, CAGR={m_simple['cagr']:.1%}, "
          f"Regime gap={rt_simple['regime_gap']:.3f}")

    return {
        'ml_model': {'metrics': m, 'regime': rt, 'avg_ic': avg_ic, 'n_ics': len(ics)},
        'simple_overlay': {'metrics': m_simple, 'regime': rt_simple},
    }


# ─── STRATEGY 3: LONG VOLATILITY TIMING ──────────────────────────────────

def strategy_long_vol(data):
    """
    Long vol when cheap, time entries using VIX term structure.
    Instead of short vol (which fails in crashes), time LONG vol entries.
    Signals:
    - VIX level (< 15 = cheap vol)
    - VIX term structure (contango = calm, backwardation = panic)
    - Realized vs implied gap
    Walk-forward: predict next-month VIX spike probability.
    """
    print("\n" + "=" * 70)
    print("STRATEGY 3: LONG VOLATILITY TIMING")
    print("=" * 70)

    needed = ['SPY', 'VIXY']  # VIXY as VIX proxy (or VXX)
    available = {t: data[t] for t in needed if t in data}

    spy = data['SPY']
    spy_ret = spy['Close'].pct_change()
    spy_close = spy['Close']

    # Proxy VIX from SPY realized vol
    rv_21 = spy_ret.rolling(21).std() * np.sqrt(252) * 100  # annualized vol in VIX-like units
    rv_63 = spy_ret.rolling(63).std() * np.sqrt(252) * 100

    # VIX proxy: use rv_21 as base, add mean-reverting premium
    vix_proxy = rv_21 * 1.2  # implied vol is typically ~20% above realized

    # Vol spike target: did VIX spike > 5 points in next 21 days?
    vix_fwd_max = vix_proxy.rolling(21).max().shift(-21)
    vol_spike = (vix_fwd_max - vix_proxy > 5).astype(float)

    # Features for vol spike prediction
    features = pd.DataFrame(index=spy_ret.index)
    features['vix_level'] = vix_proxy
    features['vix_percentile'] = vix_proxy.rolling(252).apply(
        lambda x: (x.iloc[-1] <= x).mean() if len(x) > 0 else 0.5)
    features['rv_ratio'] = rv_21 / rv_63.clip(lower=1)  # term structure proxy
    features['spy_ret_5d'] = spy_ret.rolling(5).sum()
    features['spy_ret_21d'] = spy_ret.rolling(21).sum()
    features['spy_drawdown'] = spy_close / spy_close.rolling(252).max() - 1
    features['vol_of_vol'] = rv_21.rolling(21).std()

    # Walk-forward: predict vol spike, go long vol when spike predicted
    strat_rets = []
    ics = []

    for i in range(TRAIN_DAYS + 63, len(spy_ret.index) - TEST_DAYS, TEST_DAYS):
        train_start = max(0, i - TRAIN_DAYS)

        X_train = features.iloc[train_start:i].dropna()
        y_train = vol_spike.reindex(X_train.index).dropna()
        common_train = X_train.index.intersection(y_train.index)
        X_train = X_train.loc[common_train]
        y_train = y_train.loc[common_train]

        if len(X_train) < 50:
            continue

        test_end = min(i + TEST_DAYS, len(spy_ret.index) - 1)
        X_test = features.iloc[i:test_end].dropna()

        if len(X_test) < 3:
            continue

        try:
            model = lgb.LGBMClassifier(n_estimators=50, max_depth=3, learning_rate=0.05,
                                        verbose=-1, n_jobs=1, min_child_samples=20)
            model.fit(X_train, y_train)
            spike_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_test)
        except:
            continue

        # Position: if spike probability high, go long vol (short SPY as proxy)
        # Long vol ~ short market during vol expansion
        # But we want ASYMMETRIC: small position normally, big position when vol cheap + spike predicted
        for j, (dt, prob) in enumerate(zip(X_test.index, spike_prob)):
            vix_level = features['vix_level'].loc[dt] if dt in features.index else 20

            # Only take long vol when VIX is cheap (< 18) AND spike predicted
            if prob > 0.5 and vix_level < 18:
                pos = -0.5 * (prob - 0.5) * 2  # short SPY as vol proxy, scale by prob
            elif prob > 0.7:  # high prob even with normal VIX
                pos = -0.3 * (prob - 0.5) * 2
            else:
                pos = 0.0  # stay flat when no spike predicted

            actual = spy_ret.iloc[spy_ret.index.get_loc(dt)] if dt in spy_ret.index else 0
            strat_rets.append({'date': dt, 'ret': pos * actual})

    if not strat_rets:
        return {'error': 'No trades generated'}

    strat_s = pd.DataFrame(strat_rets).set_index('date')['ret']
    strat_s = strat_s[~strat_s.index.duplicated(keep='first')].dropna()

    m = calc_metrics(strat_s, 'Long Vol Timing')
    rt = regime_test(strat_s, spy_ret.reindex(strat_s.index).dropna(), 'LongVol')

    print(f"\n  Long Vol Timing: Sharpe={m['sharpe']:.3f}, Sortino={m['sortino']:.3f}, "
          f"CAGR={m['cagr']:.1%}, MaxDD={m['max_dd']:.1%}")
    print(f"  Regime gap: {rt['regime_gap']:.3f} ({'PASS' if rt['regime_pass'] else 'FAIL'})")
    print(f"  Green Sharpe={rt['sharpe_green']:.3f}, Red Sharpe={rt['sharpe_red']:.3f}")
    print(f"  Note: This strategy should make money in RED regimes (long vol = crisis alpha)")

    return {'metrics': m, 'regime': rt}


# ─── STRATEGY 4: GOLD TREND FOLLOWING ─────────────────────────────────────

def strategy_gold_trend(data):
    """
    Trend following on gold (GLD) and gold miners (GDX).
    Signals: momentum, MA crossover, breakout, RSI.
    Walk-forward each signal independently, then combine.
    Gold is partially regime-agnostic (crisis hedge + inflation hedge).
    """
    print("\n" + "=" * 70)
    print("STRATEGY 4: GOLD TREND FOLLOWING")
    print("=" * 70)

    gold_tickers = ['GLD', 'GDX']
    available = [t for t in gold_tickers if t in data]

    if not available:
        return {'error': 'No gold data available'}

    spy_ret = data['SPY']['Close'].pct_change()
    results = {}

    for ticker in available:
        print(f"\n  Testing {ticker}...")
        price = data[ticker]['Close']
        ret = price.pct_change()
        common = spy_ret.index.intersection(ret.index)
        ret = ret.reindex(common)

        # Build signals
        signals = pd.DataFrame(index=common)
        signals['mom_63'] = price.reindex(common).pct_change(63)  # 3-month momentum
        signals['mom_126'] = price.reindex(common).pct_change(126)  # 6-month momentum
        signals['mom_252'] = price.reindex(common).pct_change(252)  # 12-month momentum
        signals['ma_cross_50_200'] = (price.reindex(common).rolling(50).mean() /
                                       price.reindex(common).rolling(200).mean() - 1)
        signals['rsi_14'] = compute_rsi(price.reindex(common), 14)
        signals['breakout_63'] = (price.reindex(common) /
                                   price.reindex(common).rolling(63).max() - 1)  # distance from 63d high

        # Walk-forward: use LGBM to combine signals for next-21d return prediction
        target = ret.rolling(21).sum().shift(-21)

        strat_rets = []
        ics = []

        for i in range(TRAIN_DAYS, len(common) - TEST_DAYS, TEST_DAYS):
            train_start = max(0, i - TRAIN_DAYS)

            X_train = signals.iloc[train_start:i].dropna()
            y_train = target.reindex(X_train.index).dropna()
            common_train = X_train.index.intersection(y_train.index)
            X_train = X_train.loc[common_train]
            y_train = y_train.loc[common_train]

            if len(X_train) < 30:
                continue

            test_end = min(i + TEST_DAYS, len(common) - 1)
            X_test = signals.iloc[i:test_end].dropna()

            if len(X_test) < 3:
                continue

            try:
                model = lgb.LGBMRegressor(n_estimators=50, max_depth=3, learning_rate=0.1,
                                           verbose=-1, n_jobs=1)
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
            except:
                continue

            # IC
            y_test = target.reindex(X_test.index).dropna()
            if len(y_test) > 5:
                ic, _ = spearmanr(preds[:len(y_test)], y_test.values)
                if not np.isnan(ic):
                    ics.append(ic)

            # Position
            for j, (dt, pred) in enumerate(zip(X_test.index, preds)):
                pos = np.clip(pred * 10, -1, 1)
                actual = ret.loc[dt] if dt in ret.index else 0
                strat_rets.append({'date': dt, 'ret': pos * actual})

        if not strat_rets:
            results[ticker] = {'error': 'No trades'}
            continue

        strat_s = pd.DataFrame(strat_rets).set_index('date')['ret']
        strat_s = strat_s[~strat_s.index.duplicated(keep='first')].dropna()

        m = calc_metrics(strat_s, f'{ticker} Trend')
        rt = regime_test(strat_s, spy_ret.reindex(strat_s.index).dropna(), f'{ticker}Trend')
        avg_ic = np.mean(ics) if ics else 0

        results[ticker] = {'metrics': m, 'regime': rt, 'avg_ic': avg_ic}

        print(f"    {ticker}: Sharpe={m['sharpe']:.3f}, Sortino={m['sortino']:.3f}, "
              f"CAGR={m['cagr']:.1%}, MaxDD={m['max_dd']:.1%}")
        print(f"    Walk-forward IC: {avg_ic:.4f}")
        print(f"    Regime gap: {rt['regime_gap']:.3f} ({'PASS' if rt['regime_pass'] else 'FAIL'})")

    # Also test simple momentum: long GLD when 12-month mom > 0
    if 'GLD' in data:
        print("\n  Simple GLD momentum (long when 12m mom > 0)...")
        gld_price = data['GLD']['Close']
        gld_ret = gld_price.pct_change()
        gld_mom = gld_price.pct_change(252)
        simple_signal = (gld_mom > 0).astype(float)
        simple_rets = (gld_ret * simple_signal).dropna()
        simple_rets = simple_rets.iloc[252:]  # skip lookback

        m_simple = calc_metrics(simple_rets, 'GLD Simple Mom')
        rt_simple = regime_test(simple_rets, spy_ret.reindex(simple_rets.index).dropna(), 'GLD_SimpleMom')

        results['gld_simple_mom'] = {'metrics': m_simple, 'regime': rt_simple}
        print(f"    Simple Mom: Sharpe={m_simple['sharpe']:.3f}, CAGR={m_simple['cagr']:.1%}, "
              f"Regime gap={rt_simple['regime_gap']:.3f}")

    return results


# ─── STRATEGY 5: TAIL-HEDGED GROWTH ───────────────────────────────────────

def strategy_tail_hedged_growth(data):
    """
    Hold QQQ (or TQQQ) + systematic tail hedges.
    Hedge: buy OTM put spreads sized to cap portfolio DD at ~-20%.
    Walk-forward: can we predict when to increase/decrease hedge size?

    Simplified simulation:
    - Core: QQQ long position
    - Hedge: synthetic put protection (cost ~3-5% annual, pays off in crashes)
    - Dynamic: increase hedge when vol is rising
    """
    print("\n" + "=" * 70)
    print("STRATEGY 5: TAIL-HEDGED GROWTH (QQQ + Put Protection)")
    print("=" * 70)

    if 'QQQ' not in data or 'SPY' not in data:
        return {'error': 'Missing QQQ or SPY data'}

    qqq = data['QQQ']
    spy = data['SPY']
    qqq_ret = qqq['Close'].pct_change()
    spy_ret = spy['Close'].pct_change()
    qqq_close = qqq['Close']

    common = spy_ret.index.intersection(qqq_ret.index)
    qqq_ret = qqq_ret.reindex(common)
    spy_ret_common = spy_ret.reindex(common)

    # Realized vol
    rv_21 = qqq_ret.rolling(21).std() * np.sqrt(252)

    # Static tail hedge simulation
    # Cost of OTM put spread: ~0.015% daily (~3.8% annual)
    # Payoff: when QQQ drops > 5% in a month, hedge pays +partial offset
    HEDGE_COST_DAILY = 0.015 / 100  # daily cost of put protection
    HEDGE_STRIKE = -0.05  # put activates at -5% monthly

    def sim_hedged_returns(qqq_ret_series, hedge_ratio, dynamic=False, rv=None):
        """Simulate tail-hedged returns."""
        rets = []
        cum = 1.0
        monthly_ret = 0

        for i, (dt, r) in enumerate(qqq_ret_series.items()):
            # Core QQQ return
            port_ret = r

            # Hedge cost
            h = hedge_ratio
            if dynamic and rv is not None and dt in rv.index:
                v = rv.loc[dt]
                if not np.isnan(v):
                    # Increase hedge when vol is rising
                    if v > 0.25:
                        h = hedge_ratio * 1.5
                    elif v < 0.12:
                        h = hedge_ratio * 0.5

            port_ret -= HEDGE_COST_DAILY * h

            # Track monthly return for hedge payoff
            monthly_ret += r

            # Monthly reset (every 21 days approximately)
            if i % 21 == 20:
                if monthly_ret < HEDGE_STRIKE:
                    # Hedge pays off: recover portion of loss beyond strike
                    payoff = abs(monthly_ret - HEDGE_STRIKE) * h * 0.7  # 70% recovery
                    port_ret += payoff / 21  # spread payoff over month
                monthly_ret = 0

            rets.append({'date': dt, 'ret': port_ret})

        return pd.DataFrame(rets).set_index('date')['ret']

    # Test configurations
    configs = [
        ('QQQ Naked', 0, False),
        ('QQQ + Static Hedge (1x)', 1.0, False),
        ('QQQ + Static Hedge (0.5x)', 0.5, False),
        ('QQQ + Dynamic Hedge', 1.0, True),
    ]

    results = {}
    for name, hedge_ratio, dynamic in configs:
        if hedge_ratio == 0:
            s = qqq_ret.iloc[TRAIN_DAYS:]
        else:
            s = sim_hedged_returns(qqq_ret.iloc[TRAIN_DAYS:], hedge_ratio, dynamic, rv_21)

        m = calc_metrics(s, name)
        rt = regime_test(s, spy_ret_common.reindex(s.index).dropna(), name)
        results[name] = {'metrics': m, 'regime': rt}

        print(f"\n  {name}:")
        print(f"    Sharpe={m['sharpe']:.3f}, Sortino={m['sortino']:.3f}, "
              f"CAGR={m['cagr']:.1%}, MaxDD={m['max_dd']:.1%}")
        print(f"    Regime gap: {rt['regime_gap']:.3f}")

    # Test TQQQ (3x QQQ) with hedging
    print("\n  Testing TQQQ (3x QQQ simulated) + dynamic hedge...")
    # Simulate TQQQ with vol drag
    daily_var = qqq_ret.rolling(63).var()
    tqqq_drag = 0.5 * 3 * 2 * daily_var
    tqqq_ret = 3 * qqq_ret - tqqq_drag

    s_tqqq_hedged = sim_hedged_returns(tqqq_ret.iloc[TRAIN_DAYS:], 1.5, True, rv_21)
    m_tqqq = calc_metrics(s_tqqq_hedged, 'TQQQ + Dynamic Hedge')
    rt_tqqq = regime_test(s_tqqq_hedged, spy_ret_common.reindex(s_tqqq_hedged.index).dropna(), 'TQQQ_Hedged')

    results['TQQQ + Dynamic Hedge'] = {'metrics': m_tqqq, 'regime': rt_tqqq}

    print(f"    TQQQ Hedged: Sharpe={m_tqqq['sharpe']:.3f}, CAGR={m_tqqq['cagr']:.1%}, "
          f"MaxDD={m_tqqq['max_dd']:.1%}, Regime gap={rt_tqqq['regime_gap']:.3f}")

    return results


# ─── BONUS: COMBINED PORTFOLIO ────────────────────────────────────────────

def combined_portfolio(all_strategy_rets, spy_ret):
    """
    Combine the best strategies from each angle into a diversified growth portfolio.
    Equal risk contribution across strategies.
    """
    print("\n" + "=" * 70)
    print("COMBINED PORTFOLIO: BEST OF EACH STRATEGY")
    print("=" * 70)

    if not all_strategy_rets:
        print("  No strategy returns to combine.")
        return {}

    # Equal weight across strategies
    combined = pd.DataFrame(all_strategy_rets)
    common_idx = combined.dropna().index

    if len(common_idx) < 100:
        print(f"  Only {len(common_idx)} common days, skipping combined portfolio.")
        return {}

    # Equal weight
    equal_w = combined.loc[common_idx].mean(axis=1)

    # Risk parity weight
    vols = combined.loc[common_idx].rolling(63).std()
    inv_vols = 1 / vols.clip(lower=1e-6)
    rp_weights = inv_vols.div(inv_vols.sum(axis=1), axis=0)
    rp_w = (combined.loc[common_idx] * rp_weights).sum(axis=1)

    spy_r = spy_ret.reindex(common_idx).dropna()
    common_final = equal_w.index.intersection(spy_r.index)

    m_eq = calc_metrics(equal_w.loc[common_final], 'Combined (Equal Weight)')
    m_rp = calc_metrics(rp_w.loc[common_final], 'Combined (Risk Parity)')
    rt_eq = regime_test(equal_w.loc[common_final], spy_r.loc[common_final], 'Combined_EQ')
    rt_rp = regime_test(rp_w.loc[common_final], spy_r.loc[common_final], 'Combined_RP')

    print(f"\n  Equal Weight: Sharpe={m_eq['sharpe']:.3f}, CAGR={m_eq['cagr']:.1%}, "
          f"MaxDD={m_eq['max_dd']:.1%}, Regime gap={rt_eq['regime_gap']:.3f}")
    print(f"  Risk Parity:  Sharpe={m_rp['sharpe']:.3f}, CAGR={m_rp['cagr']:.1%}, "
          f"MaxDD={m_rp['max_dd']:.1%}, Regime gap={rt_rp['regime_gap']:.3f}")

    return {
        'equal_weight': {'metrics': m_eq, 'regime': rt_eq},
        'risk_parity': {'metrics': m_rp, 'regime': rt_rp},
    }


# ─── MAIN ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("R5 TASK 2: UNCONVENTIONAL GROWTH ANGLES")
    print(f"Start: {START} | End: {END}")
    print(f"Walk-forward: {TRAIN_DAYS}d train, {TEST_DAYS}d test (sliding)")
    print("=" * 70)

    # Download all needed data
    all_tickers = ['SPY', 'QQQ', 'GLD', 'GDX', 'TLT', 'HYG', 'MNA', 'VIXY',
                   'DBC', 'EFA', 'EEM', 'VNQ', 'TIP']
    print("\nDownloading data...")
    data = download_data(all_tickers)

    all_results = {}
    strategy_returns = {}

    # Strategy 1: Merger Arb
    result = strategy_merger_arb(data)
    all_results['merger_arb'] = result

    # Strategy 2: Seasonality
    result = strategy_seasonality(data)
    all_results['seasonality'] = result

    # Strategy 3: Long Vol Timing
    result = strategy_long_vol(data)
    all_results['long_vol'] = result

    # Strategy 4: Gold Trend Following
    result = strategy_gold_trend(data)
    all_results['gold_trend'] = result

    # Strategy 5: Tail-Hedged Growth
    result = strategy_tail_hedged_growth(data)
    all_results['tail_hedged'] = result

    # ─── FINAL SUMMARY ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FINAL SUMMARY — UNCONVENTIONAL GROWTH STRATEGIES")
    print("=" * 70)

    # Collect best from each
    summary = []
    for strat_name, result in all_results.items():
        if isinstance(result, dict):
            # Find the best sub-result
            if 'metrics' in result:
                summary.append({
                    'strategy': strat_name,
                    'sharpe': result['metrics']['sharpe'],
                    'cagr_pct': result['metrics'].get('cagr_pct', result['metrics']['cagr'] * 100),
                    'max_dd': result['metrics']['max_dd'],
                    'regime_gap': result.get('regime', {}).get('regime_gap', float('inf')),
                })
            elif 'error' not in result:
                # Nested results — find best
                for sub_name, sub_result in result.items():
                    if isinstance(sub_result, dict) and 'metrics' in sub_result:
                        summary.append({
                            'strategy': f'{strat_name}/{sub_name}',
                            'sharpe': sub_result['metrics']['sharpe'],
                            'cagr_pct': sub_result['metrics'].get('cagr_pct',
                                         sub_result['metrics']['cagr'] * 100),
                            'max_dd': sub_result['metrics']['max_dd'],
                            'regime_gap': sub_result.get('regime', {}).get('regime_gap', float('inf')),
                        })

    summary.sort(key=lambda x: x['sharpe'], reverse=True)

    print(f"\n  {'Strategy':<40} {'Sharpe':>7} {'CAGR%':>7} {'MaxDD%':>8} {'RegGap':>7}")
    print("  " + "-" * 70)
    for s in summary:
        print(f"  {s['strategy']:<40} {s['sharpe']:>7.3f} {s['cagr_pct']:>7.1f} "
              f"{s['max_dd']*100:>8.1f} {s['regime_gap']:>7.3f}")

    # Save results
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return str(obj)
        else:
            return obj

    out_file = os.path.join(OUT_DIR, 'task2_unconventional_growth_results.json')
    with open(out_file, 'w') as f:
        json.dump(make_serializable(all_results), f, indent=2, default=str)
    print(f"\n  Results saved to {out_file}")


if __name__ == '__main__':
    main()
