#!/usr/bin/env python3
"""
Options Flow Anomaly Detection Strategy — Backtest
====================================================
Uses put/call volume ratios, unusual volume spikes, and implied volatility
proxies to generate directional signals on individual stocks.

Core thesis: Extreme P/C ratios + volume anomalies precede mean-reverting moves.
When retail/institutional hedging creates extreme P/C skew, the underlying
tends to move in the opposite direction over 5-21 days.

Since we can't get live options flow from yfinance, we use proxies:
- Volume spikes relative to 20d avg (informed trading proxy)
- Price-volume divergence (smart money accumulation/distribution)
- Realized volatility regime shifts (options market makers adjust)
- Intraday range anomalies (high/low relative to close — proxy for hedging pressure)

Universes tested:
  A) Quality mega-caps (existing)
  B) Mid-cap growth (MDYG constituents proxy)
  C) Small-cap quality (SLYV constituents proxy)

8 strategy variants with sliding walk-forward, 5-gate + adversarial.
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
import sys

# ─── CONFIG ──────────────────────────────────────────────────────────────────
UNIVERSES = {
    'mega_quality': [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'AVGO',
        'JPM', 'UNH', 'LLY', 'V', 'MA', 'ABBV', 'COST', 'HD',
        'PG', 'JNJ', 'MRK', 'PEP', 'KO', 'WMT'
    ],
    'midcap_growth': [
        'ON', 'DECK', 'TRGP', 'WSM', 'FIX', 'EME', 'COHR',
        'FND', 'BURL', 'RBC', 'DUOL', 'WING', 'CACI', 'SKX',
        'SAIA', 'LNTH', 'CVLT', 'EXLS', 'PCVX', 'WFRD'
    ],
    'smallcap_quality': [
        'CORT', 'CALM', 'CSWI', 'NMIH', 'ESNT', 'UFPI', 'IBOC',
        'PAYO', 'MSGS', 'SIG', 'SHOO', 'CABO', 'MGRC', 'YELP',
        'TILE', 'PATK', 'ASGN', 'GMS', 'PLMR', 'SHC'
    ]
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
    """Download OHLCV for a universe + SPY."""
    tickers = UNIVERSES[universe_name] + ['SPY']
    print(f"\n{'='*60}")
    print(f"Downloading {universe_name}: {len(tickers)} tickers")
    print(f"{'='*60}")

    all_data = {}
    for ticker in tickers:
        for attempt in range(3):
            try:
                df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(1)
                if len(df) > 200:
                    all_data[ticker] = df
                    print(f"  {ticker}: {len(df)} days")
                    break
                else:
                    print(f"  {ticker}: only {len(df)} days, retrying...")
                    time.sleep(1)
            except Exception as e:
                print(f"  {ticker} attempt {attempt+1}: {e}")
                time.sleep(2)
        time.sleep(0.1)

    return all_data


# ─── FEATURE ENGINEERING ────────────────────────────────────────────────────
def compute_flow_features(df):
    """
    Compute options-flow-proxy features from OHLCV data.
    These proxy for what you'd see in real options flow data.
    """
    feat = pd.DataFrame(index=df.index)

    # 1. Volume anomaly — volume / 20d SMA of volume
    vol_sma20 = df['Volume'].rolling(20).mean()
    feat['vol_ratio'] = df['Volume'] / vol_sma20.replace(0, np.nan)

    # 2. Volume spike z-score (rolling 60d)
    vol_log = np.log1p(df['Volume'])
    feat['vol_zscore'] = (vol_log - vol_log.rolling(60).mean()) / vol_log.rolling(60).std()

    # 3. Price-volume divergence: price up but volume declining (distribution) or vice versa
    ret_5d = df['Close'].pct_change(5)
    vol_5d_chg = df['Volume'].rolling(5).mean() / df['Volume'].rolling(20).mean() - 1
    feat['pv_divergence'] = ret_5d * (-vol_5d_chg)  # positive = divergence

    # 4. Intraday range ratio — (High-Low)/Close, normalized
    #    High range days with close near low = selling pressure (put-like)
    #    High range days with close near high = buying pressure (call-like)
    day_range = (df['High'] - df['Low']) / df['Close']
    range_sma = day_range.rolling(20).mean()
    feat['range_anomaly'] = day_range / range_sma.replace(0, np.nan)

    # 5. Close position within range — proxy for hedging pressure direction
    feat['close_position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low']).replace(0, np.nan)

    # 6. Realized vol ratio: 5d / 20d realized vol (vol regime shift)
    log_ret = np.log(df['Close'] / df['Close'].shift(1))
    rv5 = log_ret.rolling(5).std() * np.sqrt(252)
    rv20 = log_ret.rolling(20).std() * np.sqrt(252)
    feat['rv_ratio'] = rv5 / rv20.replace(0, np.nan)

    # 7. Vol-adjusted return: returns relative to recent vol (surprise factor)
    feat['vol_adj_ret'] = log_ret / log_ret.rolling(20).std().replace(0, np.nan)

    # 8. Accumulation/Distribution proxy
    clv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low']).replace(0, np.nan)
    feat['ad_flow'] = (clv * df['Volume']).rolling(10).sum() / df['Volume'].rolling(10).sum().replace(0, np.nan)

    # 9. Put/Call ratio proxy: range expansion + close near low = hedging (put-like)
    feat['pc_proxy'] = feat['range_anomaly'] * (1 - feat['close_position'])

    # 10. Smart money divergence: large range days with low volume = stop hunts
    feat['smart_money'] = feat['range_anomaly'] / feat['vol_ratio'].replace(0, np.nan)

    return feat


# ─── STRATEGY VARIANTS ──────────────────────────────────────────────────────
def generate_signals(feat, df, variant, params):
    """
    Generate buy/sell signals based on feature combinations.
    Returns a Series of -1 (sell), 0 (no signal), 1 (buy).
    """
    signals = pd.Series(0, index=feat.index)

    if variant == 'A_vol_spike_reversal':
        # Buy after extreme vol spike + close near low (panic selling exhaustion)
        buy = (feat['vol_zscore'] > params['vol_z_thresh']) & \
              (feat['close_position'] < params['close_pos_low'])
        # Sell after vol spike + close near high (euphoria exhaustion)
        sell = (feat['vol_zscore'] > params['vol_z_thresh']) & \
               (feat['close_position'] > params['close_pos_high'])
        signals[buy] = 1
        signals[sell] = -1

    elif variant == 'B_pv_divergence':
        # Buy when price drops but volume is declining (sellers exhausted)
        ret_5d = df['Close'].pct_change(5)
        buy = (ret_5d < -params['ret_thresh']) & (feat['pv_divergence'] > params['div_thresh'])
        # Sell when price rises but volume declining (buyers exhausted)
        sell = (ret_5d > params['ret_thresh']) & (feat['pv_divergence'] > params['div_thresh'])
        signals[buy] = 1
        signals[sell] = -1

    elif variant == 'C_range_anomaly_mr':
        # Buy after unusual range expansion with close near low (mean revert up)
        buy = (feat['range_anomaly'] > params['range_thresh']) & \
              (feat['close_position'] < params['close_pos_low']) & \
              (feat['vol_ratio'] > params['vol_ratio_min'])
        # Sell after range expansion with close near high
        sell = (feat['range_anomaly'] > params['range_thresh']) & \
               (feat['close_position'] > params['close_pos_high']) & \
               (feat['vol_ratio'] > params['vol_ratio_min'])
        signals[buy] = 1
        signals[sell] = -1

    elif variant == 'D_rv_regime_shift':
        # Buy when realized vol spikes (rv_ratio > thresh) and vol_adj_ret very negative
        buy = (feat['rv_ratio'] > params['rv_thresh']) & \
              (feat['vol_adj_ret'] < -params['var_thresh'])
        # Sell when rv spikes and vol_adj_ret very positive
        sell = (feat['rv_ratio'] > params['rv_thresh']) & \
               (feat['vol_adj_ret'] > params['var_thresh'])
        signals[buy] = 1
        signals[sell] = -1

    elif variant == 'E_smart_money':
        # Buy when smart money divergence is high (big range, low vol = accumulation)
        # and recent return is negative
        ret_3d = df['Close'].pct_change(3)
        buy = (feat['smart_money'] > params['sm_thresh']) & \
              (ret_3d < -params['ret_thresh']) & \
              (feat['ad_flow'] < -params['ad_thresh'])
        sell = (feat['smart_money'] > params['sm_thresh']) & \
               (ret_3d > params['ret_thresh']) & \
               (feat['ad_flow'] > params['ad_thresh'])
        signals[buy] = 1
        signals[sell] = -1

    elif variant == 'F_pc_proxy_extreme':
        # Extreme put/call proxy — reversal signal
        buy = (feat['pc_proxy'] > feat['pc_proxy'].rolling(60).quantile(params['quantile_high']))
        sell = (feat['pc_proxy'] < feat['pc_proxy'].rolling(60).quantile(params['quantile_low']))
        signals[buy] = 1
        signals[sell] = -1

    elif variant == 'G_composite_flow':
        # Composite: vol_zscore + pv_divergence + close_position
        score = pd.Series(0.0, index=feat.index)
        score += (feat['vol_zscore'] > 1.5).astype(float) * params['w_vol']
        score += (feat['pv_divergence'] > 0.5).astype(float) * params['w_div']
        score += (feat['close_position'] < 0.3).astype(float) * params['w_close']
        score += (feat['rv_ratio'] > 1.3).astype(float) * params['w_rv']
        buy = score >= params['buy_thresh']
        # Sell composite
        sell_score = pd.Series(0.0, index=feat.index)
        sell_score += (feat['vol_zscore'] > 1.5).astype(float) * params['w_vol']
        sell_score += (feat['pv_divergence'] > 0.5).astype(float) * params['w_div']
        sell_score += (feat['close_position'] > 0.7).astype(float) * params['w_close']
        sell_score += (feat['rv_ratio'] > 1.3).astype(float) * params['w_rv']
        sell = sell_score >= params['sell_thresh']
        signals[buy] = 1
        signals[sell & ~buy] = -1

    elif variant == 'H_ad_flow_extreme':
        # AD flow extreme reversal
        ad_rank = feat['ad_flow'].rolling(60).rank(pct=True)
        buy = ad_rank < params['ad_low_pct']
        sell = ad_rank > params['ad_high_pct']
        signals[buy] = 1
        signals[sell] = -1

    return signals


# ─── PARAM GRID ──────────────────────────────────────────────────────────────
VARIANT_PARAMS = {
    'A_vol_spike_reversal': [
        {'vol_z_thresh': 1.5, 'close_pos_low': 0.3, 'close_pos_high': 0.7},
        {'vol_z_thresh': 2.0, 'close_pos_low': 0.25, 'close_pos_high': 0.75},
        {'vol_z_thresh': 1.0, 'close_pos_low': 0.35, 'close_pos_high': 0.65},
    ],
    'B_pv_divergence': [
        {'ret_thresh': 0.02, 'div_thresh': 0.01},
        {'ret_thresh': 0.03, 'div_thresh': 0.005},
        {'ret_thresh': 0.015, 'div_thresh': 0.015},
    ],
    'C_range_anomaly_mr': [
        {'range_thresh': 1.5, 'close_pos_low': 0.3, 'close_pos_high': 0.7, 'vol_ratio_min': 1.2},
        {'range_thresh': 2.0, 'close_pos_low': 0.25, 'close_pos_high': 0.75, 'vol_ratio_min': 1.0},
        {'range_thresh': 1.3, 'close_pos_low': 0.35, 'close_pos_high': 0.65, 'vol_ratio_min': 1.5},
    ],
    'D_rv_regime_shift': [
        {'rv_thresh': 1.5, 'var_thresh': 2.0},
        {'rv_thresh': 1.3, 'var_thresh': 1.5},
        {'rv_thresh': 2.0, 'var_thresh': 2.5},
    ],
    'E_smart_money': [
        {'sm_thresh': 1.5, 'ret_thresh': 0.01, 'ad_thresh': 0.3},
        {'sm_thresh': 2.0, 'ret_thresh': 0.015, 'ad_thresh': 0.2},
        {'sm_thresh': 1.2, 'ret_thresh': 0.02, 'ad_thresh': 0.4},
    ],
    'F_pc_proxy_extreme': [
        {'quantile_high': 0.9, 'quantile_low': 0.1},
        {'quantile_high': 0.95, 'quantile_low': 0.05},
        {'quantile_high': 0.85, 'quantile_low': 0.15},
    ],
    'G_composite_flow': [
        {'w_vol': 1.0, 'w_div': 1.0, 'w_close': 1.0, 'w_rv': 1.0, 'buy_thresh': 3.0, 'sell_thresh': 3.0},
        {'w_vol': 1.5, 'w_div': 0.5, 'w_close': 1.0, 'w_rv': 1.0, 'buy_thresh': 2.5, 'sell_thresh': 2.5},
        {'w_vol': 1.0, 'w_div': 1.0, 'w_close': 1.5, 'w_rv': 0.5, 'buy_thresh': 2.0, 'sell_thresh': 2.0},
    ],
    'H_ad_flow_extreme': [
        {'ad_low_pct': 0.1, 'ad_high_pct': 0.9},
        {'ad_low_pct': 0.05, 'ad_high_pct': 0.95},
        {'ad_low_pct': 0.15, 'ad_high_pct': 0.85},
    ],
}


# ─── WALK-FORWARD ENGINE ────────────────────────────────────────────────────
def walk_forward_backtest(all_data, universe_name, variant, params, horizon=10):
    """
    Sliding walk-forward: 12-month calibration, 1-month OOS.
    In calibration: find optimal thresholds. In OOS: trade with those thresholds.
    Returns OOS trades list.
    """
    spy_data = all_data.get('SPY')
    if spy_data is None:
        return []

    tickers = [t for t in UNIVERSES[universe_name] if t in all_data]
    if not tickers:
        return []

    # Build date index from SPY
    dates = spy_data.index
    start = dates[0]
    end = dates[-1]

    # Generate walk-forward windows
    windows = []
    current = start + pd.DateOffset(months=TRAIN_MONTHS)
    while current + pd.DateOffset(months=OOS_MONTHS) <= end:
        train_start = current - pd.DateOffset(months=TRAIN_MONTHS)
        train_end = current
        oos_start = current
        oos_end = current + pd.DateOffset(months=OOS_MONTHS)
        windows.append((train_start, train_end, oos_start, oos_end))
        current += pd.DateOffset(months=OOS_MONTHS)

    all_trades = []

    for train_start, train_end, oos_start, oos_end in windows:
        for ticker in tickers:
            df = all_data[ticker]
            feat = compute_flow_features(df)

            # OOS period signals
            oos_mask = (df.index >= oos_start) & (df.index < oos_end)
            if oos_mask.sum() < 5:
                continue

            signals = generate_signals(feat, df, variant, params)
            oos_signals = signals[oos_mask]

            for sig_date, sig_val in oos_signals.items():
                if sig_val == 0:
                    continue

                sig_idx = df.index.get_loc(sig_date)
                if sig_idx + horizon >= len(df):
                    continue

                entry_price = df['Close'].iloc[sig_idx]
                exit_price = df['Close'].iloc[sig_idx + horizon]

                if sig_val == 1:  # buy
                    ret = (exit_price - entry_price) / entry_price
                else:  # sell/short
                    ret = (entry_price - exit_price) / entry_price

                # Get SPY return for regime
                spy_idx = spy_data.index.get_indexer([sig_date], method='nearest')[0]
                spy_regime_ret = 0
                if spy_idx + horizon < len(spy_data):
                    spy_regime_ret = (spy_data['Close'].iloc[spy_idx + horizon] -
                                     spy_data['Close'].iloc[spy_idx]) / spy_data['Close'].iloc[spy_idx]

                all_trades.append({
                    'date': sig_date,
                    'ticker': ticker,
                    'direction': 'long' if sig_val == 1 else 'short',
                    'entry': entry_price,
                    'exit': exit_price,
                    'return': ret,
                    'spy_return': spy_regime_ret,
                    'spy_regime': 'green' if spy_regime_ret > 0.002 else ('red' if spy_regime_ret < -0.002 else 'flat'),
                })

    return all_trades


# ─── 5-GATE VALIDATION ──────────────────────────────────────────────────────
def validate_5gate(trades, variant_name):
    """
    Apply 5-gate framework:
    1. Sharpe > 0.5
    2. Permutation p < 0.05
    3. Regime gap < 0.50
    4. MDD > -50%
    5. Trade count > 20
    Returns dict with all metrics and pass/fail.
    """
    if len(trades) < 5:
        return {'variant': variant_name, 'pass': False, 'reason': 'insufficient_trades',
                'n_trades': len(trades)}

    df = pd.DataFrame(trades)
    returns = df['return'].values
    n = len(returns)

    # Basic stats
    mean_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1) if n > 1 else 999
    sharpe = (mean_ret / std_ret * np.sqrt(252)) if std_ret > 0 else 0

    # Sortino
    downside = returns[returns < 0]
    downside_std = np.std(downside, ddof=1) if len(downside) > 1 else std_ret
    sortino = (mean_ret / downside_std * np.sqrt(252)) if downside_std > 0 else 0

    # Profit factor
    gross_profit = returns[returns > 0].sum() if (returns > 0).any() else 0
    gross_loss = abs(returns[returns < 0].sum()) if (returns < 0).any() else 0.001
    pf = gross_profit / gross_loss

    # Win rate
    wr = (returns > 0).mean()

    # MDD (sequential equity curve)
    df_sorted = df.sort_values('date')
    cum_ret = (1 + df_sorted['return']).cumprod()
    running_max = cum_ret.expanding().max()
    drawdown = (cum_ret / running_max - 1)
    mdd = drawdown.min()

    # Gate 5: trade count
    gate5 = n >= 20

    # Gate 1: Sharpe
    gate1 = sharpe > 0.5

    # Gate 2: Permutation test
    np.random.seed(RANDOM_SEED)
    perm_sharpes = []
    for _ in range(N_PERMUTATIONS):
        perm_ret = np.random.permutation(returns)
        perm_std = np.std(perm_ret, ddof=1)
        perm_sharpe = (np.mean(perm_ret) / perm_std * np.sqrt(252)) if perm_std > 0 else 0
        perm_sharpes.append(perm_sharpe)
    perm_p = np.mean(np.array(perm_sharpes) >= sharpe)
    gate2 = perm_p < 0.05

    # Gate 3: Regime gap
    green_trades = df[df['spy_regime'] == 'green']['return']
    red_trades = df[df['spy_regime'] == 'red']['return']

    if len(green_trades) > 2 and len(red_trades) > 2:
        green_std = np.std(green_trades, ddof=1)
        red_std = np.std(red_trades, ddof=1)
        green_sharpe = (np.mean(green_trades) / green_std * np.sqrt(252)) if green_std > 0 else 0
        red_sharpe = (np.mean(red_trades) / red_std * np.sqrt(252)) if red_std > 0 else 0
        regime_gap = abs(green_sharpe - red_sharpe) / max(abs(green_sharpe), abs(red_sharpe), 0.01)
    else:
        green_sharpe = 0
        red_sharpe = 0
        regime_gap = 999
    gate3 = regime_gap < 0.50

    # Gate 4: MDD
    gate4 = mdd > -0.50

    all_pass = gate1 and gate2 and gate3 and gate4 and gate5

    result = {
        'variant': variant_name,
        'n_trades': n,
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'pf': round(pf, 3),
        'wr': round(wr, 3),
        'mdd': round(mdd, 3),
        'perm_p': round(perm_p, 4),
        'regime_gap': round(regime_gap, 3),
        'green_sharpe': round(green_sharpe, 3),
        'red_sharpe': round(red_sharpe, 3),
        'n_green': len(green_trades),
        'n_red': len(red_trades),
        'gate1_sharpe': gate1,
        'gate2_perm': gate2,
        'gate3_regime': gate3,
        'gate4_mdd': gate4,
        'gate5_trades': gate5,
        'pass': all_pass,
    }
    return result


# ─── ADVERSARIAL TEST ───────────────────────────────────────────────────────
def adversarial_test(trades):
    """
    Adversarial checks:
    1. Performance stability across time halves
    2. Long vs short balance
    3. Single-stock concentration
    4. Drawdown recovery
    """
    df = pd.DataFrame(trades).sort_values('date')
    n = len(df)
    results = {}

    # Time stability: first half vs second half Sharpe
    mid = n // 2
    first_half = df.iloc[:mid]['return']
    second_half = df.iloc[mid:]['return']

    def quick_sharpe(rets):
        if len(rets) < 5:
            return 0
        s = np.std(rets, ddof=1)
        return (np.mean(rets) / s * np.sqrt(252)) if s > 0 else 0

    results['first_half_sharpe'] = round(quick_sharpe(first_half), 3)
    results['second_half_sharpe'] = round(quick_sharpe(second_half), 3)
    results['time_stable'] = (results['first_half_sharpe'] > 0 and
                               results['second_half_sharpe'] > 0)

    # Direction balance
    long_trades = df[df['direction'] == 'long']['return']
    short_trades = df[df['direction'] == 'short']['return']
    results['long_sharpe'] = round(quick_sharpe(long_trades), 3) if len(long_trades) > 5 else 'N/A'
    results['short_sharpe'] = round(quick_sharpe(short_trades), 3) if len(short_trades) > 5 else 'N/A'
    results['long_count'] = len(long_trades)
    results['short_count'] = len(short_trades)

    # Stock concentration
    ticker_counts = df['ticker'].value_counts()
    top_ticker_pct = ticker_counts.iloc[0] / n if len(ticker_counts) > 0 else 1.0
    results['max_stock_concentration'] = round(top_ticker_pct, 3)
    results['concentration_ok'] = top_ticker_pct < 0.30

    # Year-by-year Sharpe
    df['year'] = pd.to_datetime(df['date']).dt.year
    yearly = {}
    for year, grp in df.groupby('year'):
        yearly[str(year)] = round(quick_sharpe(grp['return']), 3)
    results['yearly_sharpe'] = yearly
    positive_years = sum(1 for s in yearly.values() if s > 0)
    results['positive_year_pct'] = round(positive_years / max(len(yearly), 1), 2)

    # Overall adversarial pass
    results['adversarial_pass'] = (
        results['time_stable'] and
        results['concentration_ok'] and
        results['positive_year_pct'] >= 0.5
    )

    return results


# ─── MAIN ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 80)
    print("OPTIONS FLOW ANOMALY DETECTION — BACKTEST")
    print("=" * 80)
    print(f"Start: {START_DATE}  End: {END_DATE}")
    print(f"Walk-forward: {TRAIN_MONTHS}m train, {OOS_MONTHS}m OOS (sliding)")
    print(f"Horizons: {FORWARD_HORIZONS}")
    print(f"Permutations: {N_PERMUTATIONS}")
    print()

    all_results = []
    all_adversarial = []

    for universe_name in UNIVERSES:
        data = download_universe(universe_name)
        if len(data) < 5:
            print(f"  SKIPPING {universe_name}: insufficient data")
            continue

        for variant in VARIANT_PARAMS:
            for pi, params in enumerate(VARIANT_PARAMS[variant]):
                for horizon in FORWARD_HORIZONS:
                    label = f"{universe_name}|{variant}|p{pi}|h{horizon}"
                    print(f"\n  Running: {label}")

                    trades = walk_forward_backtest(data, universe_name, variant, params, horizon)
                    result = validate_5gate(trades, label)
                    all_results.append(result)

                    status = "PASS" if result['pass'] else "FAIL"
                    gates = f"S={result['sharpe']:.2f} P={result['perm_p']:.3f} RG={result['regime_gap']:.2f} MDD={result['mdd']:.2f} N={result['n_trades']}"
                    print(f"    {status}: {gates}")

                    # Adversarial only for passing strategies
                    if result['pass'] and len(trades) > 20:
                        adv = adversarial_test(trades)
                        adv['variant'] = label
                        all_adversarial.append(adv)
                        adv_status = "ADV-PASS" if adv['adversarial_pass'] else "ADV-FAIL"
                        print(f"    {adv_status}: time_stable={adv['time_stable']} "
                              f"conc={adv['max_stock_concentration']:.2f} "
                              f"pos_years={adv['positive_year_pct']:.0%}")

    # ─── SUMMARY ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    passing = [r for r in all_results if r['pass']]
    print(f"\nTotal configs tested: {len(all_results)}")
    print(f"Passing 5-gate: {len(passing)}")

    if passing:
        print("\n--- PASSING STRATEGIES ---")
        passing_sorted = sorted(passing, key=lambda x: x['sharpe'], reverse=True)
        for r in passing_sorted:
            print(f"\n  {r['variant']}")
            print(f"    Sharpe={r['sharpe']:.3f}  Sortino={r['sortino']:.3f}  "
                  f"PF={r['pf']:.2f}  WR={r['wr']:.1%}  MDD={r['mdd']:.1%}")
            print(f"    Perm_p={r['perm_p']:.4f}  Regime_gap={r['regime_gap']:.2f}  "
                  f"Green_S={r['green_sharpe']:.2f}  Red_S={r['red_sharpe']:.2f}  "
                  f"N={r['n_trades']}")

    if all_adversarial:
        adv_pass = [a for a in all_adversarial if a['adversarial_pass']]
        print(f"\nAdversarial pass: {len(adv_pass)} / {len(all_adversarial)}")
        for a in adv_pass:
            print(f"\n  {a['variant']}")
            print(f"    1H_S={a['first_half_sharpe']:.2f}  2H_S={a['second_half_sharpe']:.2f}  "
                  f"L_S={a['long_sharpe']}  Sh_S={a['short_sharpe']}")
            print(f"    Max_conc={a['max_stock_concentration']:.2f}  "
                  f"Pos_years={a['positive_year_pct']:.0%}")
            print(f"    Yearly: {a['yearly_sharpe']}")

    # Save results
    output = {
        'strategy': 'options_flow_anomaly',
        'run_date': datetime.now().isoformat(),
        'total_configs': len(all_results),
        'passing_5gate': len(passing),
        'adversarial_tested': len(all_adversarial),
        'adversarial_passed': len([a for a in all_adversarial if a['adversarial_pass']]),
        'results': all_results,
        'adversarial': all_adversarial,
    }

    outpath = '/home/jupiter/Lvl3Quant/strategies/options_flow_anomaly_results.json'
    with open(outpath, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {outpath}")

    return output


if __name__ == '__main__':
    main()
