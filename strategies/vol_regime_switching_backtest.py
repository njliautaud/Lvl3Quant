#!/usr/bin/env python3
"""
Volatility Regime Switching Strategy — Backtest
=================================================
Trade based on transitions between realized volatility regimes rather than
just VIX levels. Identifies regime states using rolling realized vol and
generates signals when transitions occur.

Core thesis: Markets behave differently in low/mid/high vol regimes.
Transitions between regimes (especially high→low) create tradeable
opportunities. Stocks that lag the regime transition are mispriced.

Strategies:
  A) Vol compression breakout: low vol → expanding vol, trade direction
  B) Vol crush recovery: high vol → collapsing vol, buy the recovery
  C) Cross-sectional vol dispersion: buy low-vol stocks when dispersion high
  D) Vol term structure: short-term vol vs long-term, trade mean reversion
  E) Regime-conditional momentum: momentum only in correct vol regime
  F) Vol-of-vol: second-order vol changes as timing signal
  G) Relative vol ranking: buy stocks whose vol is declining fastest
  H) Vol breakout with trend confirmation

Multiple universes, sliding walk-forward, 5-gate + adversarial.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from scipy import stats
import json
import os
import time

# ─── CONFIG ──────────────────────────────────────────────────────────────────
UNIVERSES = {
    'quality_broad': [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'AVGO',
        'JPM', 'UNH', 'LLY', 'V', 'MA', 'ABBV', 'COST', 'HD',
        'PG', 'JNJ', 'MRK', 'PEP', 'KO', 'WMT', 'CRM', 'ORCL',
        'ACN', 'ADBE', 'AMD', 'QCOM', 'TXN', 'NEE', 'LOW'
    ],
    'sector_etf': [
        'XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLC', 'XLY', 'XLP',
        'XLB', 'XLRE', 'XLU', 'XBI', 'SOXX', 'XME', 'XOP',
        'KRE', 'IYT', 'XHB', 'ITB', 'JETS'
    ],
    'midcap_mix': [
        'ON', 'DECK', 'TRGP', 'WSM', 'FIX', 'EME', 'COHR',
        'FND', 'BURL', 'RBC', 'DUOL', 'WING', 'CACI', 'SKX',
        'SAIA', 'LNTH', 'CVLT', 'EXLS', 'HQY', 'WFRD'
    ],
}

START_DATE = '2020-01-01'
END_DATE = '2026-08-01'
TRAIN_MONTHS = 12
OOS_MONTHS = 1
N_PERMUTATIONS = 1000
FORWARD_HORIZONS = [5, 10, 21]
RANDOM_SEED = 42

# ─── DATA ────────────────────────────────────────────────────────────────────
def download_universe(universe_name):
    tickers = UNIVERSES[universe_name] + ['SPY', '^VIX']
    print(f"\nDownloading {universe_name}: {len(tickers)} tickers")
    all_data = {}
    for ticker in tickers:
        for attempt in range(3):
            try:
                df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(1)
                if len(df) > 200:
                    all_data[ticker] = df
                    break
                time.sleep(1)
            except:
                time.sleep(2)
        time.sleep(0.05)
    print(f"  Got {len(all_data)}/{len(tickers)} tickers")
    return all_data


# ─── VOL REGIME DETECTION ───────────────────────────────────────────────────
def classify_vol_regime(rv_series, lookback=60):
    """
    Classify into 3 regimes: low, mid, high vol based on rolling percentile.
    """
    rv_pct = rv_series.rolling(lookback).rank(pct=True)
    regime = pd.Series('mid', index=rv_series.index)
    regime[rv_pct < 0.33] = 'low'
    regime[rv_pct > 0.67] = 'high'
    return regime


def detect_regime_transitions(regime_series):
    """
    Detect transitions: returns the transition type or None.
    """
    transitions = pd.Series(None, index=regime_series.index, dtype=object)
    prev = regime_series.shift(1)
    transitions[(prev == 'high') & (regime_series == 'mid')] = 'high_to_mid'
    transitions[(prev == 'high') & (regime_series == 'low')] = 'high_to_low'
    transitions[(prev == 'mid') & (regime_series == 'low')] = 'mid_to_low'
    transitions[(prev == 'low') & (regime_series == 'mid')] = 'low_to_mid'
    transitions[(prev == 'low') & (regime_series == 'high')] = 'low_to_high'
    transitions[(prev == 'mid') & (regime_series == 'high')] = 'mid_to_high'
    return transitions


# ─── FEATURE ENGINEERING ────────────────────────────────────────────────────
def compute_vol_features(df):
    """Compute volatility regime features."""
    feat = pd.DataFrame(index=df.index)
    close = df['Close']
    log_ret = np.log(close / close.shift(1))

    # Realized vol at multiple horizons
    for w in [5, 10, 20, 40, 60]:
        feat[f'rv_{w}'] = log_ret.rolling(w).std() * np.sqrt(252)

    # Vol ratios (term structure)
    feat['vol_ts_5_20'] = feat['rv_5'] / feat['rv_20'].replace(0, np.nan)
    feat['vol_ts_10_40'] = feat['rv_10'] / feat['rv_40'].replace(0, np.nan)
    feat['vol_ts_20_60'] = feat['rv_20'] / feat['rv_60'].replace(0, np.nan)

    # Vol regime
    feat['vol_regime'] = classify_vol_regime(feat['rv_20'])
    feat['vol_transition'] = detect_regime_transitions(feat['vol_regime'])

    # Vol-of-vol
    feat['vov_20'] = feat['rv_5'].rolling(20).std()
    feat['vov_ratio'] = feat['vov_20'] / feat['vov_20'].rolling(60).mean().replace(0, np.nan)

    # Vol z-score
    feat['rv_zscore'] = (feat['rv_20'] - feat['rv_20'].rolling(60).mean()) / \
                         feat['rv_20'].rolling(60).std().replace(0, np.nan)

    # Vol percentile (rolling)
    feat['rv_pctile'] = feat['rv_20'].rolling(252).rank(pct=True)

    # Vol change rate
    feat['rv_change'] = feat['rv_20'].pct_change(5)

    # Price features
    feat['ret_5d'] = close.pct_change(5)
    feat['ret_10d'] = close.pct_change(10)
    feat['ret_20d'] = close.pct_change(20)
    feat['dist_sma20'] = (close - close.rolling(20).mean()) / close.rolling(20).mean()
    feat['dist_sma50'] = (close - close.rolling(50).mean()) / close.rolling(50).mean()

    # Bollinger width (vol proxy)
    bb_width = (close.rolling(20).std() * 2) / close.rolling(20).mean()
    feat['bb_width'] = bb_width
    feat['bb_width_pctile'] = bb_width.rolling(120).rank(pct=True)

    return feat


# ─── STRATEGY VARIANTS ──────────────────────────────────────────────────────
def generate_vol_signals(feat, df, variant, params):
    signals = pd.Series(0, index=feat.index)

    if variant == 'A_vol_compression_breakout':
        # Low vol → breakout: buy when vol starts expanding from low base
        buy = (feat['bb_width_pctile'] < params['width_pctile']) & \
              (feat['vol_ts_5_20'] > params['ts_thresh']) & \
              (feat['ret_5d'] > 0)
        sell = (feat['bb_width_pctile'] < params['width_pctile']) & \
               (feat['vol_ts_5_20'] > params['ts_thresh']) & \
               (feat['ret_5d'] < 0)
        signals[buy] = 1
        signals[sell] = -1

    elif variant == 'B_vol_crush_recovery':
        # High vol → collapsing: buy the recovery
        buy = (feat['vol_transition'].isin(['high_to_mid', 'high_to_low'])) | \
              ((feat['rv_change'] < params['rv_change_thresh']) & \
               (feat['rv_zscore'] > params['rv_z_thresh']))
        signals[buy] = 1

    elif variant == 'C_cross_vol_dispersion':
        # Not applicable for single-stock signals, skip
        pass

    elif variant == 'D_vol_term_structure_mr':
        # Vol term structure inversion → mean revert
        # Short-term vol >> long-term → expect vol compression + bounce
        buy = (feat['vol_ts_5_20'] > params['ts_high']) & \
              (feat['ret_5d'] < params['ret_thresh'])
        sell = (feat['vol_ts_5_20'] < params['ts_low']) & \
               (feat['ret_5d'] > abs(params['ret_thresh']))
        signals[buy] = 1
        signals[sell] = -1

    elif variant == 'E_regime_conditional_mom':
        # Momentum only works in low-vol regime
        buy = (feat['vol_regime'] == 'low') & \
              (feat['ret_20d'] > params['mom_thresh']) & \
              (feat['dist_sma50'] > 0)
        sell = (feat['vol_regime'] == 'low') & \
               (feat['ret_20d'] < -params['mom_thresh']) & \
               (feat['dist_sma50'] < 0)
        signals[buy] = 1
        signals[sell] = -1

    elif variant == 'F_vol_of_vol':
        # High vol-of-vol + negative returns → buy (panic peak)
        buy = (feat['vov_ratio'] > params['vov_thresh']) & \
              (feat['ret_5d'] < params['ret_thresh']) & \
              (feat['rv_zscore'] > params['rv_z_min'])
        signals[buy] = 1

    elif variant == 'G_relative_vol_decline':
        # Buy stocks whose vol is declining fastest (normalizing)
        buy = (feat['rv_change'] < params['rv_decline_thresh']) & \
              (feat['rv_pctile'] > params['rv_pctile_min']) & \
              (feat['dist_sma20'] < 0)
        signals[buy] = 1

    elif variant == 'H_vol_breakout_trend':
        # Vol expansion + trend confirmation
        buy = (feat['vol_ts_5_20'] > params['ts_thresh']) & \
              (feat['dist_sma20'] > params['trend_thresh']) & \
              (feat['bb_width_pctile'] > params['width_pctile'])
        sell = (feat['vol_ts_5_20'] > params['ts_thresh']) & \
               (feat['dist_sma20'] < -params['trend_thresh']) & \
               (feat['bb_width_pctile'] > params['width_pctile'])
        signals[buy] = 1
        signals[sell] = -1

    return signals


VARIANT_PARAMS = {
    'A_vol_compression_breakout': [
        {'width_pctile': 0.2, 'ts_thresh': 1.3},
        {'width_pctile': 0.15, 'ts_thresh': 1.5},
    ],
    'B_vol_crush_recovery': [
        {'rv_change_thresh': -0.15, 'rv_z_thresh': 1.0},
        {'rv_change_thresh': -0.20, 'rv_z_thresh': 1.5},
    ],
    'D_vol_term_structure_mr': [
        {'ts_high': 1.5, 'ts_low': 0.7, 'ret_thresh': -0.02},
        {'ts_high': 1.8, 'ts_low': 0.6, 'ret_thresh': -0.03},
    ],
    'E_regime_conditional_mom': [
        {'mom_thresh': 0.03},
        {'mom_thresh': 0.05},
    ],
    'F_vol_of_vol': [
        {'vov_thresh': 1.5, 'ret_thresh': -0.03, 'rv_z_min': 0.5},
        {'vov_thresh': 2.0, 'ret_thresh': -0.05, 'rv_z_min': 1.0},
    ],
    'G_relative_vol_decline': [
        {'rv_decline_thresh': -0.15, 'rv_pctile_min': 0.5},
        {'rv_decline_thresh': -0.20, 'rv_pctile_min': 0.6},
    ],
    'H_vol_breakout_trend': [
        {'ts_thresh': 1.3, 'trend_thresh': 0.02, 'width_pctile': 0.6},
        {'ts_thresh': 1.5, 'trend_thresh': 0.03, 'width_pctile': 0.7},
    ],
}


# ─── WALK-FORWARD ────────────────────────────────────────────────────────────
def walk_forward_backtest(all_data, universe_name, variant, params, horizon=10):
    spy_data = all_data.get('SPY')
    if spy_data is None:
        return []

    tickers = [t for t in UNIVERSES[universe_name] if t in all_data]
    if not tickers:
        return []

    dates = spy_data.index
    start = dates[0]
    end = dates[-1]

    windows = []
    current = start + pd.DateOffset(months=TRAIN_MONTHS)
    while current + pd.DateOffset(months=OOS_MONTHS) <= end:
        windows.append((current, current + pd.DateOffset(months=OOS_MONTHS)))
        current += pd.DateOffset(months=OOS_MONTHS)

    all_trades = []

    for oos_start, oos_end in windows:
        for ticker in tickers:
            df = all_data[ticker]
            feat = compute_vol_features(df)

            oos_mask = (df.index >= oos_start) & (df.index < oos_end)
            if oos_mask.sum() < 3:
                continue

            signals = generate_vol_signals(feat, df, variant, params)
            oos_signals = signals[oos_mask]

            for sig_date, sig_val in oos_signals.items():
                if sig_val == 0:
                    continue

                sig_idx = df.index.get_loc(sig_date)
                if sig_idx + horizon >= len(df):
                    continue

                entry_price = df['Close'].iloc[sig_idx]
                exit_price = df['Close'].iloc[sig_idx + horizon]

                if sig_val == 1:
                    ret = (exit_price - entry_price) / entry_price
                else:
                    ret = (entry_price - exit_price) / entry_price

                spy_idx = spy_data.index.get_indexer([sig_date], method='nearest')[0]
                spy_regime_ret = 0
                if spy_idx + horizon < len(spy_data):
                    spy_regime_ret = (spy_data['Close'].iloc[spy_idx + horizon] -
                                     spy_data['Close'].iloc[spy_idx]) / spy_data['Close'].iloc[spy_idx]

                all_trades.append({
                    'date': sig_date,
                    'ticker': ticker,
                    'direction': 'long' if sig_val == 1 else 'short',
                    'entry': float(entry_price),
                    'exit': float(exit_price),
                    'return': float(ret),
                    'spy_return': float(spy_regime_ret),
                    'spy_regime': 'green' if spy_regime_ret > 0.002 else ('red' if spy_regime_ret < -0.002 else 'flat'),
                })

    return all_trades


# ─── 5-GATE + ADVERSARIAL ───────────────────────────────────────────────────
def validate_5gate(trades, variant_name):
    if len(trades) < 5:
        return {'variant': variant_name, 'pass': False, 'reason': 'insufficient_trades', 'n_trades': len(trades)}

    df = pd.DataFrame(trades)
    returns = df['return'].values
    n = len(returns)

    mean_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1) if n > 1 else 999
    sharpe = (mean_ret / std_ret * np.sqrt(252)) if std_ret > 0 else 0

    downside = returns[returns < 0]
    downside_std = np.std(downside, ddof=1) if len(downside) > 1 else std_ret
    sortino = (mean_ret / downside_std * np.sqrt(252)) if downside_std > 0 else 0

    gross_profit = returns[returns > 0].sum() if (returns > 0).any() else 0
    gross_loss = abs(returns[returns < 0].sum()) if (returns < 0).any() else 0.001
    pf = gross_profit / gross_loss
    wr = (returns > 0).mean()

    df_sorted = df.sort_values('date')
    cum_ret = (1 + df_sorted['return']).cumprod()
    mdd = (cum_ret / cum_ret.expanding().max() - 1).min()

    gate5 = n >= 20
    gate1 = sharpe > 0.5

    np.random.seed(RANDOM_SEED)
    perm_sharpes = []
    for _ in range(N_PERMUTATIONS):
        pr = np.random.permutation(returns)
        ps = np.std(pr, ddof=1)
        perm_sharpes.append((np.mean(pr) / ps * np.sqrt(252)) if ps > 0 else 0)
    perm_p = np.mean(np.array(perm_sharpes) >= sharpe)
    gate2 = perm_p < 0.05

    green_trades = df[df['spy_regime'] == 'green']['return']
    red_trades = df[df['spy_regime'] == 'red']['return']

    if len(green_trades) > 2 and len(red_trades) > 2:
        gs = np.std(green_trades, ddof=1)
        rs = np.std(red_trades, ddof=1)
        green_sharpe = (np.mean(green_trades) / gs * np.sqrt(252)) if gs > 0 else 0
        red_sharpe = (np.mean(red_trades) / rs * np.sqrt(252)) if rs > 0 else 0
        regime_gap = abs(green_sharpe - red_sharpe) / max(abs(green_sharpe), abs(red_sharpe), 0.01)
    else:
        green_sharpe = red_sharpe = 0
        regime_gap = 999
    gate3 = regime_gap < 0.50
    gate4 = mdd > -0.50

    return {
        'variant': variant_name, 'n_trades': n,
        'sharpe': round(sharpe, 3), 'sortino': round(sortino, 3),
        'pf': round(pf, 3), 'wr': round(wr, 3), 'mdd': round(mdd, 3),
        'perm_p': round(perm_p, 4), 'regime_gap': round(regime_gap, 3),
        'green_sharpe': round(green_sharpe, 3), 'red_sharpe': round(red_sharpe, 3),
        'n_green': len(green_trades), 'n_red': len(red_trades),
        'gate1_sharpe': gate1, 'gate2_perm': gate2, 'gate3_regime': gate3,
        'gate4_mdd': gate4, 'gate5_trades': gate5,
        'pass': gate1 and gate2 and gate3 and gate4 and gate5,
    }


def adversarial_test(trades):
    df = pd.DataFrame(trades).sort_values('date')
    n = len(df)

    def qs(rets):
        if len(rets) < 5: return 0
        s = np.std(rets, ddof=1)
        return (np.mean(rets) / s * np.sqrt(252)) if s > 0 else 0

    mid = n // 2
    h1s = qs(df.iloc[:mid]['return'])
    h2s = qs(df.iloc[mid:]['return'])

    long_t = df[df['direction'] == 'long']['return']
    short_t = df[df['direction'] == 'short']['return']

    tc = df['ticker'].value_counts()
    top_conc = tc.iloc[0] / n if len(tc) > 0 else 1.0

    df['year'] = pd.to_datetime(df['date']).dt.year
    yearly = {str(y): round(qs(g['return']), 3) for y, g in df.groupby('year')}
    pyp = sum(1 for s in yearly.values() if s > 0) / max(len(yearly), 1)

    return {
        'variant': '',
        'first_half_sharpe': round(h1s, 3), 'second_half_sharpe': round(h2s, 3),
        'time_stable': h1s > 0 and h2s > 0,
        'long_sharpe': round(qs(long_t), 3) if len(long_t) > 5 else 'N/A',
        'short_sharpe': round(qs(short_t), 3) if len(short_t) > 5 else 'N/A',
        'long_count': len(long_t), 'short_count': len(short_t),
        'max_stock_concentration': round(top_conc, 3),
        'concentration_ok': top_conc < 0.30,
        'yearly_sharpe': yearly,
        'positive_year_pct': round(pyp, 2),
        'adversarial_pass': h1s > 0 and h2s > 0 and top_conc < 0.30 and pyp >= 0.5,
    }


# ─── MAIN ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 80)
    print("VOLATILITY REGIME SWITCHING — BACKTEST")
    print("=" * 80)

    all_results = []
    all_adversarial = []

    for universe_name in UNIVERSES:
        data = download_universe(universe_name)
        if len(data) < 5:
            continue

        for variant in VARIANT_PARAMS:
            for pi, params in enumerate(VARIANT_PARAMS[variant]):
                for horizon in FORWARD_HORIZONS:
                    label = f"{universe_name}|{variant}|p{pi}|h{horizon}"

                    trades = walk_forward_backtest(data, universe_name, variant, params, horizon)
                    result = validate_5gate(trades, label)
                    all_results.append(result)

                    if result['pass']:
                        print(f"  PASS: {label}")
                        print(f"    S={result['sharpe']:.3f} Sort={result['sortino']:.3f} "
                              f"PF={result['pf']:.2f} WR={result['wr']:.1%} "
                              f"MDD={result['mdd']:.1%} N={result['n_trades']} "
                              f"perm_p={result['perm_p']:.4f} RG={result['regime_gap']:.2f}")

                        if len(trades) > 20:
                            adv = adversarial_test(trades)
                            adv['variant'] = label
                            all_adversarial.append(adv)
                            if adv['adversarial_pass']:
                                print(f"    ADV-PASS: yr={adv['positive_year_pct']:.0%}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY — Vol Regime Switching")
    print("=" * 80)

    passing = [r for r in all_results if r['pass']]
    print(f"Total configs: {len(all_results)}")
    print(f"Passing 5-gate: {len(passing)}")

    if passing:
        for r in sorted(passing, key=lambda x: x['sharpe'], reverse=True)[:15]:
            print(f"  {r['variant']}")
            print(f"    S={r['sharpe']:.3f} Sort={r['sortino']:.3f} PF={r['pf']:.2f} "
                  f"WR={r['wr']:.1%} MDD={r['mdd']:.1%} N={r['n_trades']}")

    adv_pass = [a for a in all_adversarial if a['adversarial_pass']]
    print(f"Adversarial pass: {len(adv_pass)}")
    for a in adv_pass:
        print(f"  {a['variant']}: yr={a['positive_year_pct']:.0%} "
              f"1H={a['first_half_sharpe']:.2f} 2H={a['second_half_sharpe']:.2f}")

    output = {
        'strategy': 'vol_regime_switching',
        'run_date': datetime.now().isoformat(),
        'total_configs': len(all_results),
        'passing_5gate': len(passing),
        'adversarial_passed': len(adv_pass),
        'results': all_results,
        'adversarial': all_adversarial,
    }

    outpath = '/home/jupiter/Lvl3Quant/strategies/vol_regime_switching_results.json'
    with open(outpath, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved: {outpath}")

    return output


if __name__ == '__main__':
    main()
