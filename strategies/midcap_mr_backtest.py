#!/usr/bin/env python3
"""
Mean Reversion in Non-Mega-Cap Universes — Backtest
=====================================================
Expands the validated Quality MR approach to different market cap tiers:
  - Mid-cap growth (high-growth mid-caps with quality screens)
  - Small-cap quality (profitable small-caps with low leverage)
  - Sector ETF MR (liquid ETFs with different dynamics than single stocks)

Thesis: MR works in mega-cap quality because of institutional mean-reversion
buying. Different cap tiers may have different MR dynamics — mid-caps may
overshoot more (less coverage), small-caps may have faster reversals
(less efficient), sector ETFs may have flow-driven dislocations.

10 variants × 3 universes × 3 horizons × 2 param sets = 180 configs.
Sliding walk-forward, 5-gate + adversarial.
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
    'midcap_growth': [
        # High-growth mid-caps ($5B-$20B) with strong earnings
        'ON', 'DECK', 'TRGP', 'WSM', 'FIX', 'EME', 'COHR',
        'FND', 'BURL', 'RBC', 'DUOL', 'WING', 'CACI', 'SKX',
        'SAIA', 'LNTH', 'CVLT', 'EXLS', 'HQY', 'WFRD',
        'ARMK', 'AZEK', 'COKE', 'DT', 'ENSG'
    ],
    'smallcap_quality': [
        # Profitable small-caps ($500M-$5B) with low debt, positive FCF
        'CORT', 'CALM', 'CSWI', 'NMIH', 'ESNT', 'UFPI',
        'SIG', 'SHOO', 'YELP', 'TILE', 'PATK', 'ASGN',
        'GMS', 'PLMR', 'KLIC', 'LQDT', 'PRGS', 'FOXF',
        'BOOT', 'VCEL', 'SPSC', 'ICFI', 'POWL', 'CPRX'
    ],
    'sector_etf': [
        # Liquid sector ETFs — flow-driven dislocations
        'XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLC', 'XLY', 'XLP',
        'XLB', 'XLRE', 'XLU', 'XBI', 'XHB', 'XME', 'XOP',
        'ARKK', 'KWEB', 'TAN', 'JETS', 'SOXX'
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
    """Download OHLCV for universe + SPY."""
    tickers = UNIVERSES[universe_name] + ['SPY']
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


# ─── FEATURE ENGINEERING ────────────────────────────────────────────────────
def compute_mr_features(df):
    """Compute mean-reversion features tailored for different cap tiers."""
    feat = pd.DataFrame(index=df.index)
    close = df['Close']
    volume = df['Volume']

    # Distance from moving averages
    for w in [10, 20, 50]:
        sma = close.rolling(w).mean()
        feat[f'dist_sma{w}'] = (close - sma) / sma

    # RSI variants
    for period in [5, 14]:
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        feat[f'rsi_{period}'] = 100 - (100 / (1 + rs))

    # Bollinger Band position
    for w in [20]:
        sma = close.rolling(w).mean()
        std = close.rolling(w).std()
        feat[f'bb_pos_{w}'] = (close - sma) / (2 * std).replace(0, np.nan)

    # Drawdown from 20d high
    feat['dd_20'] = close / close.rolling(20).max() - 1

    # Drawdown from 50d high
    feat['dd_50'] = close / close.rolling(50).max() - 1

    # Returns at multiple horizons
    for h in [1, 3, 5, 10, 20]:
        feat[f'ret_{h}d'] = close.pct_change(h)

    # Volume-weighted momentum
    vol_sma = volume.rolling(20).mean()
    vol_ratio = volume / vol_sma.replace(0, np.nan)
    feat['vol_ratio'] = vol_ratio

    # Realized volatility
    log_ret = np.log(close / close.shift(1))
    feat['rv_20'] = log_ret.rolling(20).std() * np.sqrt(252)
    feat['rv_5'] = log_ret.rolling(5).std() * np.sqrt(252)
    feat['rv_ratio'] = feat['rv_5'] / feat['rv_20'].replace(0, np.nan)

    # Z-score of price (rolling 60d)
    feat['price_z'] = (close - close.rolling(60).mean()) / close.rolling(60).std()

    # Mean reversion score composite
    feat['mr_score'] = (
        -feat['dist_sma20'].clip(-0.2, 0.2) * 2 +  # farther from SMA = stronger MR signal
        (50 - feat['rsi_14']) / 50 +  # lower RSI = stronger buy
        -feat['bb_pos_20'].clip(-2, 2) / 2 +  # lower BB position = stronger buy
        -feat['price_z'].clip(-3, 3) / 3  # lower z-score = stronger buy
    )

    return feat


# ─── STRATEGY VARIANTS ──────────────────────────────────────────────────────
def generate_mr_signals(feat, df, variant, params):
    """Generate MR signals. Returns Series of -1/0/1."""
    signals = pd.Series(0, index=feat.index)

    if variant == 'A_rsi_oversold':
        # Classic RSI oversold/overbought
        buy = feat['rsi_14'] < params['rsi_buy']
        sell = feat['rsi_14'] > params['rsi_sell']
        signals[buy] = 1
        signals[sell] = -1

    elif variant == 'B_bb_extreme':
        # Bollinger Band extreme
        buy = feat['bb_pos_20'] < params['bb_buy']
        sell = feat['bb_pos_20'] > params['bb_sell']
        signals[buy] = 1
        signals[sell] = -1

    elif variant == 'C_drawdown_snap':
        # Buy on drawdown from recent high
        buy = feat['dd_20'] < params['dd_thresh']
        signals[buy] = 1

    elif variant == 'D_zscore_revert':
        # Z-score mean reversion
        buy = feat['price_z'] < params['z_buy']
        sell = feat['price_z'] > params['z_sell']
        signals[buy] = 1
        signals[sell] = -1

    elif variant == 'E_vol_spike_mr':
        # Buy on vol spike + oversold
        buy = (feat['rv_ratio'] > params['rv_thresh']) & \
              (feat['rsi_14'] < params['rsi_thresh']) & \
              (feat['ret_5d'] < params['ret_thresh'])
        signals[buy] = 1

    elif variant == 'F_multi_factor_mr':
        # Composite MR score
        buy = feat['mr_score'] > params['score_buy']
        sell = feat['mr_score'] < params['score_sell']
        signals[buy] = 1
        signals[sell] = -1

    elif variant == 'G_sma_distance':
        # Distance from SMA
        buy = feat['dist_sma20'] < params['dist_buy']
        sell = feat['dist_sma20'] > params['dist_sell']
        signals[buy] = 1
        signals[sell] = -1

    elif variant == 'H_rsi5_fast_mr':
        # Fast RSI(5) for quicker MR
        buy = feat['rsi_5'] < params['rsi_buy']
        sell = feat['rsi_5'] > params['rsi_sell']
        signals[buy] = 1
        signals[sell] = -1

    elif variant == 'I_vol_adj_mr':
        # Volume-adjusted mean reversion: dip on high volume = capitulation
        buy = (feat['ret_5d'] < params['ret_thresh']) & \
              (feat['vol_ratio'] > params['vol_thresh']) & \
              (feat['bb_pos_20'] < params['bb_thresh'])
        signals[buy] = 1

    elif variant == 'J_deep_drawdown':
        # Deep drawdown from 50d high — quality bounce
        buy = feat['dd_50'] < params['dd_thresh']
        signals[buy] = 1

    return signals


VARIANT_PARAMS = {
    'A_rsi_oversold': [
        {'rsi_buy': 30, 'rsi_sell': 70},
        {'rsi_buy': 25, 'rsi_sell': 75},
    ],
    'B_bb_extreme': [
        {'bb_buy': -1.0, 'bb_sell': 1.0},
        {'bb_buy': -1.5, 'bb_sell': 1.5},
    ],
    'C_drawdown_snap': [
        {'dd_thresh': -0.08},
        {'dd_thresh': -0.12},
    ],
    'D_zscore_revert': [
        {'z_buy': -1.5, 'z_sell': 1.5},
        {'z_buy': -2.0, 'z_sell': 2.0},
    ],
    'E_vol_spike_mr': [
        {'rv_thresh': 1.3, 'rsi_thresh': 40, 'ret_thresh': -0.03},
        {'rv_thresh': 1.5, 'rsi_thresh': 35, 'ret_thresh': -0.05},
    ],
    'F_multi_factor_mr': [
        {'score_buy': 1.5, 'score_sell': -1.5},
        {'score_buy': 2.0, 'score_sell': -2.0},
    ],
    'G_sma_distance': [
        {'dist_buy': -0.05, 'dist_sell': 0.05},
        {'dist_buy': -0.08, 'dist_sell': 0.08},
    ],
    'H_rsi5_fast_mr': [
        {'rsi_buy': 20, 'rsi_sell': 80},
        {'rsi_buy': 15, 'rsi_sell': 85},
    ],
    'I_vol_adj_mr': [
        {'ret_thresh': -0.05, 'vol_thresh': 1.5, 'bb_thresh': -1.0},
        {'ret_thresh': -0.03, 'vol_thresh': 1.3, 'bb_thresh': -0.8},
    ],
    'J_deep_drawdown': [
        {'dd_thresh': -0.15},
        {'dd_thresh': -0.20},
    ],
}


# ─── WALK-FORWARD ────────────────────────────────────────────────────────────
def walk_forward_backtest(all_data, universe_name, variant, params, horizon=10):
    """Sliding walk-forward backtest."""
    spy_data = all_data.get('SPY')
    if spy_data is None:
        return []

    tickers = [t for t in UNIVERSES[universe_name] if t in all_data]
    if not tickers:
        return []

    dates = spy_data.index
    start = dates[0]
    end = dates[-1]

    # Walk-forward windows
    windows = []
    current = start + pd.DateOffset(months=TRAIN_MONTHS)
    while current + pd.DateOffset(months=OOS_MONTHS) <= end:
        oos_start = current
        oos_end = current + pd.DateOffset(months=OOS_MONTHS)
        windows.append((oos_start, oos_end))
        current += pd.DateOffset(months=OOS_MONTHS)

    all_trades = []

    for oos_start, oos_end in windows:
        for ticker in tickers:
            df = all_data[ticker]
            feat = compute_mr_features(df)

            oos_mask = (df.index >= oos_start) & (df.index < oos_end)
            if oos_mask.sum() < 3:
                continue

            signals = generate_mr_signals(feat, df, variant, params)
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


# ─── 5-GATE + ADVERSARIAL (reuse from options_flow) ─────────────────────────
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
    running_max = cum_ret.expanding().max()
    mdd = (cum_ret / running_max - 1).min()

    gate5 = n >= 20
    gate1 = sharpe > 0.5

    np.random.seed(RANDOM_SEED)
    perm_sharpes = []
    for _ in range(N_PERMUTATIONS):
        perm_ret = np.random.permutation(returns)
        perm_std = np.std(perm_ret, ddof=1)
        perm_sharpes.append((np.mean(perm_ret) / perm_std * np.sqrt(252)) if perm_std > 0 else 0)
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

    def quick_sharpe(rets):
        if len(rets) < 5: return 0
        s = np.std(rets, ddof=1)
        return (np.mean(rets) / s * np.sqrt(252)) if s > 0 else 0

    mid = n // 2
    first_half_s = quick_sharpe(df.iloc[:mid]['return'])
    second_half_s = quick_sharpe(df.iloc[mid:]['return'])
    time_stable = first_half_s > 0 and second_half_s > 0

    long_trades = df[df['direction'] == 'long']['return']
    short_trades = df[df['direction'] == 'short']['return']

    ticker_counts = df['ticker'].value_counts()
    top_conc = ticker_counts.iloc[0] / n if len(ticker_counts) > 0 else 1.0

    df['year'] = pd.to_datetime(df['date']).dt.year
    yearly = {str(y): round(quick_sharpe(g['return']), 3) for y, g in df.groupby('year')}
    pos_year_pct = sum(1 for s in yearly.values() if s > 0) / max(len(yearly), 1)

    return {
        'variant': df.iloc[0].get('variant', ''),
        'first_half_sharpe': round(first_half_s, 3),
        'second_half_sharpe': round(second_half_s, 3),
        'time_stable': time_stable,
        'long_sharpe': round(quick_sharpe(long_trades), 3) if len(long_trades) > 5 else 'N/A',
        'short_sharpe': round(quick_sharpe(short_trades), 3) if len(short_trades) > 5 else 'N/A',
        'long_count': len(long_trades),
        'short_count': len(short_trades),
        'max_stock_concentration': round(top_conc, 3),
        'concentration_ok': top_conc < 0.30,
        'yearly_sharpe': yearly,
        'positive_year_pct': round(pos_year_pct, 2),
        'adversarial_pass': time_stable and top_conc < 0.30 and pos_year_pct >= 0.5,
    }


# ─── MAIN ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 80)
    print("MEAN REVERSION — NON-MEGA-CAP UNIVERSES")
    print("=" * 80)
    print(f"Universes: {list(UNIVERSES.keys())}")
    print(f"Variants: {len(VARIANT_PARAMS)} × 2 param sets × 3 horizons × 3 universes")
    print()

    all_results = []
    all_adversarial = []

    for universe_name in UNIVERSES:
        data = download_universe(universe_name)
        if len(data) < 5:
            print(f"  SKIPPING {universe_name}")
            continue

        for variant in VARIANT_PARAMS:
            for pi, params in enumerate(VARIANT_PARAMS[variant]):
                for horizon in FORWARD_HORIZONS:
                    label = f"{universe_name}|{variant}|p{pi}|h{horizon}"

                    trades = walk_forward_backtest(data, universe_name, variant, params, horizon)
                    result = validate_5gate(trades, label)
                    all_results.append(result)

                    status = "PASS" if result['pass'] else "fail"
                    gates = f"S={result['sharpe']:.2f} p={result['perm_p']:.3f} RG={result['regime_gap']:.2f} MDD={result['mdd']:.2f} N={result['n_trades']}"
                    if result['pass']:
                        print(f"  PASS: {label}")
                        print(f"        {gates}")

                    if result['pass'] and len(trades) > 20:
                        adv = adversarial_test(trades)
                        adv['variant'] = label
                        all_adversarial.append(adv)
                        if adv['adversarial_pass']:
                            print(f"        ADV-PASS: yr={adv['positive_year_pct']:.0%} "
                                  f"conc={adv['max_stock_concentration']:.2f}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY — Mid-Cap & Small-Cap MR")
    print("=" * 80)

    passing = [r for r in all_results if r['pass']]
    print(f"Total configs: {len(all_results)}")
    print(f"Passing 5-gate: {len(passing)}")

    if passing:
        passing_sorted = sorted(passing, key=lambda x: x['sharpe'], reverse=True)[:15]
        print("\nTop passing strategies:")
        for r in passing_sorted:
            print(f"  {r['variant']}")
            print(f"    S={r['sharpe']:.3f} Sort={r['sortino']:.3f} PF={r['pf']:.2f} "
                  f"WR={r['wr']:.1%} MDD={r['mdd']:.1%} N={r['n_trades']} "
                  f"perm_p={r['perm_p']:.4f} RG={r['regime_gap']:.2f}")

    adv_pass = [a for a in all_adversarial if a['adversarial_pass']]
    if adv_pass:
        print(f"\nAdversarial pass: {len(adv_pass)}")
        for a in adv_pass:
            print(f"  {a['variant']}: yr={a['positive_year_pct']:.0%} "
                  f"1H={a['first_half_sharpe']:.2f} 2H={a['second_half_sharpe']:.2f}")

    output = {
        'strategy': 'midcap_smallcap_mr',
        'run_date': datetime.now().isoformat(),
        'total_configs': len(all_results),
        'passing_5gate': len(passing),
        'adversarial_passed': len(adv_pass),
        'results': all_results,
        'adversarial': all_adversarial,
    }

    outpath = '/home/jupiter/Lvl3Quant/strategies/midcap_mr_results.json'
    with open(outpath, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved: {outpath}")

    return output


if __name__ == '__main__':
    main()
