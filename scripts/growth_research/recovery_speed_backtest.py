#!/usr/bin/env python3
"""
Recovery Speed Backtest: Predict sector ETF recovery speed after dips
and optimize holding periods dynamically.

Classifies dips by recovery speed (V-shape vs slow grind) and tests
whether features at dip entry predict recovery speed. Then backtests
a dynamic exit strategy vs fixed baseline.
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from scipy import stats

warnings.filterwarnings('ignore')

# ─── CONFIG ────────────────────────────────────────────────────────
SECTOR_ETFS = ['XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLC', 'XLY', 'XLP', 'XLU', 'XLRE', 'XLB']
BENCHMARKS = ['SPY', '^VIX']
ALL_TICKERS = SECTOR_ETFS + ['SPY']
VIX_TICKER = '^VIX'
START = '2020-01-01'
END = datetime.now().strftime('%Y-%m-%d')
RSI_THRESHOLD = 35
COST_RT_PCT = 0.10 / 100  # 0.10% round-trip
RESULTS_PATH = '/home/jupiter/Lvl3Quant/scripts/growth_research/results/recovery_speed_results.json'


def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def download_data():
    print("Downloading data...")
    tickers = ALL_TICKERS + [VIX_TICKER]
    data = yf.download(tickers, start=START, end=END, auto_adjust=True, progress=False)

    close = data['Close'] if 'Close' in data.columns.get_level_values(0) else data
    volume = data['Volume'] if 'Volume' in data.columns.get_level_values(0) else None

    # Handle multi-level columns
    if isinstance(close.columns, pd.MultiIndex):
        close = data['Close']
        volume = data['Volume']

    # Rename VIX
    if '^VIX' in close.columns:
        close = close.rename(columns={'^VIX': 'VIX'})
    if volume is not None and '^VIX' in volume.columns:
        volume = volume.rename(columns={'^VIX': 'VIX'})

    print(f"  Data: {close.index[0].date()} to {close.index[-1].date()}, {len(close)} days")
    return close, volume


def classify_dips(close, volume):
    """Find all RSI<35 dips and classify recovery speed."""
    print("\n=== CLASSIFYING DIPS ===")
    dips = []

    for etf in SECTOR_ETFS:
        if etf not in close.columns:
            continue

        px = close[etf].dropna()
        vol = volume[etf].dropna() if volume is not None and etf in volume.columns else None
        rsi = compute_rsi(px)
        spy_px = close['SPY'].dropna()
        vix = close['VIX'].dropna() if 'VIX' in close.columns else None

        # Pre-compute indicators
        sma50_spy = spy_px.rolling(50).mean()
        sma200_etf = px.rolling(200).mean()
        high20 = px.rolling(20).max()
        vol_ma20 = vol.rolling(20).mean() if vol is not None else None

        # Cross-sectional dispersion (daily returns std across sectors)
        sector_rets = close[SECTOR_ETFS].pct_change()
        dispersion = sector_rets.std(axis=1)
        disp_median = dispersion.rolling(60).median()

        # Prior 20-day momentum
        mom20 = px.pct_change(20)

        # Find RSI < 35 entries (take first day of each dip episode)
        rsi_below = rsi < RSI_THRESHOLD
        # Mark start of each dip episode (transition from above to below)
        dip_starts = rsi_below & ~rsi_below.shift(1, fill_value=False)

        for date in dip_starts[dip_starts].index:
            idx = px.index.get_loc(date)
            if idx < 205:  # need 200-day SMA
                continue

            # Pre-dip level: close from 5 trading days before
            pre_dip_idx = max(0, idx - 5)
            pre_dip_level = px.iloc[pre_dip_idx]
            entry_price = px.iloc[idx]

            if entry_price >= pre_dip_level:
                continue  # not actually a dip

            dip_depth_pct = (pre_dip_level - entry_price) / pre_dip_level * 100

            # ── Features at entry ──
            # VIX level
            vix_val = vix.loc[:date].iloc[-1] if vix is not None and date in vix.index or len(vix.loc[:date]) > 0 else 20

            # VIX bin
            if vix_val < 15:
                vix_bin = '<15'
            elif vix_val < 20:
                vix_bin = '15-20'
            elif vix_val < 25:
                vix_bin = '20-25'
            elif vix_val < 35:
                vix_bin = '25-35'
            else:
                vix_bin = '>35'

            # SPY trend
            spy_above_50sma = spy_px.loc[:date].iloc[-1] > sma50_spy.loc[:date].iloc[-1] if not pd.isna(sma50_spy.loc[:date].iloc[-1]) else True

            # Sector above 200 SMA
            above_200sma = entry_price > sma200_etf.iloc[idx] if not pd.isna(sma200_etf.iloc[idx]) else True

            # Volume capitulation
            if vol_ma20 is not None and not pd.isna(vol_ma20.iloc[idx]):
                vol_ratio = vol.iloc[idx] / vol_ma20.iloc[idx] if vol_ma20.iloc[idx] > 0 else 1.0
                capitulation = vol_ratio > 1.5
            else:
                vol_ratio = 1.0
                capitulation = False

            # Cross-sectional dispersion
            disp_val = dispersion.loc[:date].iloc[-1] if date in dispersion.index or len(dispersion.loc[:date]) > 0 else 0
            disp_med = disp_median.loc[:date].iloc[-1] if len(disp_median.loc[:date]) > 0 and not pd.isna(disp_median.loc[:date].iloc[-1]) else disp_val
            high_dispersion = disp_val > disp_med if disp_med > 0 else False

            # Dip depth bin
            if dip_depth_pct < 3:
                continue  # too shallow
            elif dip_depth_pct < 5:
                depth_bin = '3-5%'
            elif dip_depth_pct < 8:
                depth_bin = '5-8%'
            else:
                depth_bin = '8%+'

            # Prior momentum
            prior_mom = mom20.iloc[idx] if not pd.isna(mom20.iloc[idx]) else 0
            mom_positive = prior_mom > 0

            # ── Recovery measurement ──
            max_forward = min(idx + 21, len(px))  # 20 days forward
            forward_prices = px.iloc[idx:max_forward]

            if len(forward_prices) < 2:
                continue

            # MFE within 1, 3, 5, 10, 20 days
            mfe = {}
            for horizon in [1, 3, 5, 10, 20]:
                end_idx = min(idx + horizon + 1, len(px))
                if end_idx > idx + 1:
                    fwd = px.iloc[idx+1:end_idx]
                    mfe[f'mfe_{horizon}d'] = ((fwd.max() - entry_price) / entry_price * 100) if len(fwd) > 0 else 0
                else:
                    mfe[f'mfe_{horizon}d'] = 0

            # Time to recover to pre-dip level
            recovery_time = None
            for j in range(idx + 1, min(idx + 61, len(px))):  # look up to 60 days
                if px.iloc[j] >= pre_dip_level:
                    recovery_time = j - idx
                    break

            # V-shape classification: recovered >80% of dip within 3 days
            dip_size = pre_dip_level - entry_price
            recovery_3d = 0
            if idx + 3 < len(px):
                max_3d = px.iloc[idx+1:idx+4].max()
                recovery_3d = (max_3d - entry_price) / dip_size if dip_size > 0 else 0

            v_shape = recovery_3d > 0.80
            slow_grind = recovery_time is not None and recovery_time > 10

            dip_record = {
                'etf': etf,
                'date': date,
                'entry_price': entry_price,
                'pre_dip_level': pre_dip_level,
                'dip_depth_pct': dip_depth_pct,
                'rsi': rsi.iloc[idx],
                # Features
                'vix': vix_val,
                'vix_bin': vix_bin,
                'spy_above_50sma': spy_above_50sma,
                'above_200sma': above_200sma,
                'vol_ratio': vol_ratio,
                'capitulation': capitulation,
                'high_dispersion': high_dispersion,
                'depth_bin': depth_bin,
                'prior_mom_positive': mom_positive,
                'prior_mom': prior_mom,
                # Recovery
                'recovery_time': recovery_time,
                'v_shape': v_shape,
                'slow_grind': slow_grind,
                'recovery_3d_pct': recovery_3d * 100,
                **mfe,
            }
            dips.append(dip_record)

    df = pd.DataFrame(dips)
    print(f"  Found {len(df)} dip episodes across {df['etf'].nunique()} sectors")
    print(f"  V-shape: {df['v_shape'].sum()} ({df['v_shape'].mean()*100:.1f}%)")
    print(f"  Slow grind: {df['slow_grind'].sum()} ({df['slow_grind'].mean()*100:.1f}%)")
    recovered = df['recovery_time'].notna()
    print(f"  Recovered within 60d: {recovered.sum()} ({recovered.mean()*100:.1f}%)")
    if recovered.any():
        print(f"  Median recovery time: {df.loc[recovered, 'recovery_time'].median():.0f} days")

    return df


def feature_analysis(dips_df):
    """Analyze which features predict fast vs slow recovery."""
    print("\n=== FEATURE ANALYSIS ===")

    features = {
        'vix_bin': {'type': 'categorical', 'values': ['<15', '15-20', '20-25', '25-35', '>35']},
        'spy_above_50sma': {'type': 'binary'},
        'above_200sma': {'type': 'binary'},
        'capitulation': {'type': 'binary'},
        'high_dispersion': {'type': 'binary'},
        'depth_bin': {'type': 'categorical', 'values': ['3-5%', '5-8%', '8%+']},
        'prior_mom_positive': {'type': 'binary'},
    }

    results = {}
    df = dips_df.copy()
    recovered = df['recovery_time'].notna()

    print(f"\n{'Feature':<25} {'Value':<15} {'N':>5} {'Med Recov':>10} {'V-shape%':>10} {'MFE_3d':>8}")
    print("-" * 80)

    for feat_name, feat_info in features.items():
        feat_results = {}

        if feat_info['type'] == 'binary':
            for val in [True, False]:
                mask = df[feat_name] == val
                n = mask.sum()
                if n < 5:
                    continue
                subset = df[mask]
                med_recov = subset.loc[subset['recovery_time'].notna(), 'recovery_time'].median() if subset['recovery_time'].notna().any() else np.nan
                v_pct = subset['v_shape'].mean() * 100
                mfe3 = subset['mfe_3d'].mean()
                label = str(val)
                print(f"  {feat_name:<23} {label:<15} {n:>5} {med_recov:>10.1f} {v_pct:>10.1f} {mfe3:>8.2f}")
                feat_results[label] = {
                    'n': int(n), 'median_recovery': float(med_recov) if not np.isnan(med_recov) else None,
                    'v_shape_pct': round(v_pct, 1), 'mfe_3d': round(mfe3, 2)
                }
        else:
            for val in feat_info['values']:
                mask = df[feat_name] == val
                n = mask.sum()
                if n < 3:
                    continue
                subset = df[mask]
                med_recov = subset.loc[subset['recovery_time'].notna(), 'recovery_time'].median() if subset['recovery_time'].notna().any() else np.nan
                v_pct = subset['v_shape'].mean() * 100
                mfe3 = subset['mfe_3d'].mean()
                print(f"  {feat_name:<23} {val:<15} {n:>5} {med_recov:>10.1f} {v_pct:>10.1f} {mfe3:>8.2f}")
                feat_results[val] = {
                    'n': int(n), 'median_recovery': float(med_recov) if not np.isnan(med_recov) else None,
                    'v_shape_pct': round(v_pct, 1), 'mfe_3d': round(mfe3, 2)
                }

        results[feat_name] = feat_results

        # Compute feature importance: correlation with v_shape
        if feat_info['type'] == 'binary':
            if df[feat_name].nunique() > 1:
                corr = df[feat_name].astype(float).corr(df['v_shape'].astype(float))
                results[feat_name]['_v_shape_corr'] = round(corr, 3)

    # Rank features by discriminative power (difference in V-shape rate)
    print("\n--- Feature Importance (V-shape prediction) ---")
    importance = []
    for feat_name, feat_info in features.items():
        if feat_info['type'] == 'binary':
            true_mask = df[feat_name] == True
            false_mask = df[feat_name] == False
            if true_mask.sum() >= 5 and false_mask.sum() >= 5:
                v_true = df.loc[true_mask, 'v_shape'].mean()
                v_false = df.loc[false_mask, 'v_shape'].mean()
                diff = abs(v_true - v_false)
                direction = 'True→fast' if v_true > v_false else 'False→fast'
                importance.append((feat_name, diff, direction, true_mask.sum(), false_mask.sum()))
        else:
            vals = [v for v in feat_info['values'] if (df[feat_name] == v).sum() >= 3]
            if len(vals) >= 2:
                v_rates = [df.loc[df[feat_name] == v, 'v_shape'].mean() for v in vals]
                diff = max(v_rates) - min(v_rates)
                best = vals[np.argmax(v_rates)]
                importance.append((feat_name, diff, f'{best}→fast', len(df), 0))

    importance.sort(key=lambda x: x[1], reverse=True)
    print(f"\n{'Feature':<25} {'V-shape Δ':>10} {'Direction':<20}")
    print("-" * 60)
    for feat, diff, direction, n1, n2 in importance:
        print(f"  {feat:<23} {diff*100:>9.1f}% {direction:<20}")

    results['_importance_ranking'] = [
        {'feature': f, 'v_shape_delta_pct': round(d*100, 1), 'direction': dr}
        for f, d, dr, _, _ in importance
    ]

    return results


def predict_recovery_speed(row):
    """Simple rule-based prediction of recovery speed using top features."""
    fast_signals = 0
    slow_signals = 0

    # VIX > 25 → tends to be V-shape (panic dips recover fast)
    if row['vix'] > 25:
        fast_signals += 2
    elif row['vix'] < 15:
        slow_signals += 1

    # Capitulation volume → fast recovery
    if row['capitulation']:
        fast_signals += 2

    # Above 200 SMA → trend intact, faster recovery
    if row['above_200sma']:
        fast_signals += 1
    else:
        slow_signals += 2

    # SPY uptrend → faster recovery
    if row['spy_above_50sma']:
        fast_signals += 1
    else:
        slow_signals += 1

    # High dispersion → rotation, slower recovery
    if row['high_dispersion']:
        slow_signals += 1

    # Deep dip → slower recovery typically
    if row['depth_bin'] == '8%+':
        slow_signals += 1
    elif row['depth_bin'] == '3-5%':
        fast_signals += 1

    # Prior positive momentum → faster recovery
    if row['prior_mom_positive']:
        fast_signals += 1
    else:
        slow_signals += 1

    score = fast_signals - slow_signals
    if score >= 3:
        return 'FAST'
    elif score <= -1:
        return 'SLOW'
    else:
        return 'DEFAULT'


def run_backtest(dips_df, close):
    """Backtest dynamic vs fixed exit strategies."""
    print("\n=== BACKTEST: DYNAMIC vs FIXED EXIT ===")

    # Strategy configs
    strategies = {
        'fixed_baseline': {
            'hold_days': 5,
            'tp_pct': 3.0,
            'sl_pct': -5.0,
        },
        'dynamic': None,  # determined per-trade
    }

    dynamic_rules = {
        'FAST': {'hold_days': 2, 'tp_pct': 2.0, 'sl_pct': -3.0},
        'SLOW': {'hold_days': 10, 'tp_pct': 5.0, 'sl_pct': -6.0},
        'DEFAULT': {'hold_days': 5, 'tp_pct': 3.0, 'sl_pct': -5.0},
    }

    results = {}

    for strat_name in ['fixed_baseline', 'dynamic']:
        trades = []

        for _, dip in dips_df.iterrows():
            etf = dip['etf']
            entry_date = dip['date']
            entry_price = dip['entry_price']

            if etf not in close.columns:
                continue

            px = close[etf].dropna()
            entry_idx = px.index.get_loc(entry_date) if entry_date in px.index else None
            if entry_idx is None:
                continue

            # Determine exit params
            if strat_name == 'dynamic':
                prediction = predict_recovery_speed(dip)
                params = dynamic_rules[prediction]
            else:
                params = strategies['fixed_baseline']
                prediction = 'FIXED'

            hold_days = params['hold_days']
            tp_pct = params['tp_pct']
            sl_pct = params['sl_pct']

            # Simulate trade
            exit_price = None
            exit_date = None
            exit_reason = None

            for d in range(1, hold_days + 1):
                if entry_idx + d >= len(px):
                    break

                day_price = px.iloc[entry_idx + d]
                day_ret = (day_price - entry_price) / entry_price * 100

                if day_ret >= tp_pct:
                    exit_price = entry_price * (1 + tp_pct / 100)
                    exit_date = px.index[entry_idx + d]
                    exit_reason = 'TP'
                    break
                elif day_ret <= sl_pct:
                    exit_price = entry_price * (1 + sl_pct / 100)
                    exit_date = px.index[entry_idx + d]
                    exit_reason = 'SL'
                    break

            if exit_price is None:
                # Exit at end of hold period
                end_idx = min(entry_idx + hold_days, len(px) - 1)
                exit_price = px.iloc[end_idx]
                exit_date = px.index[end_idx]
                exit_reason = 'HOLD'

            gross_ret = (exit_price - entry_price) / entry_price
            net_ret = gross_ret - COST_RT_PCT

            trades.append({
                'etf': etf,
                'entry_date': entry_date,
                'exit_date': exit_date,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'prediction': prediction,
                'hold_days': (exit_date - entry_date).days if exit_date is not None else hold_days,
                'gross_ret': gross_ret,
                'net_ret': net_ret,
                'exit_reason': exit_reason,
            })

        trades_df = pd.DataFrame(trades)
        if len(trades_df) == 0:
            continue

        # Compute metrics
        rets = trades_df['net_ret']
        n_trades = len(rets)
        avg_ret = rets.mean() * 100
        win_rate = (rets > 0).mean() * 100

        # Build equity curve for Sharpe/Sortino/DD
        equity = (1 + rets).cumprod()

        # Annualize: assume ~50 trades/year rough estimate
        avg_hold = trades_df['hold_days'].mean()
        trades_per_year = 252 / max(avg_hold, 1) * (n_trades / (len(close) / 252))

        # Use per-trade stats
        sharpe = rets.mean() / rets.std() * np.sqrt(min(trades_per_year, 252)) if rets.std() > 0 else 0
        downside = rets[rets < 0].std()
        sortino = rets.mean() / downside * np.sqrt(min(trades_per_year, 252)) if downside > 0 else 0

        # Profit factor
        gross_wins = rets[rets > 0].sum()
        gross_losses = abs(rets[rets < 0].sum())
        pf = gross_wins / gross_losses if gross_losses > 0 else float('inf')

        # Max drawdown
        peak = equity.expanding().max()
        dd = (equity - peak) / peak
        max_dd = dd.min() * 100

        # Exit reason breakdown
        exit_counts = trades_df['exit_reason'].value_counts().to_dict()

        # Regime analysis (bull = SPY above 50SMA at entry)
        trades_df['bull'] = trades_df['entry_date'].apply(
            lambda d: close['SPY'].loc[:d].iloc[-1] > close['SPY'].rolling(50).mean().loc[:d].iloc[-1]
            if d in close.index or len(close.loc[:d]) > 0 else True
        )

        bull_rets = trades_df.loc[trades_df['bull'], 'net_ret']
        bear_rets = trades_df.loc[~trades_df['bull'], 'net_ret']

        bull_sharpe = bull_rets.mean() / bull_rets.std() * np.sqrt(50) if len(bull_rets) > 5 and bull_rets.std() > 0 else 0
        bear_sharpe = bear_rets.mean() / bear_rets.std() * np.sqrt(50) if len(bear_rets) > 5 and bear_rets.std() > 0 else 0

        regime_gap = abs(bull_sharpe - bear_sharpe) / max(abs(bull_sharpe), abs(bear_sharpe), 0.01)

        results[strat_name] = {
            'n_trades': n_trades,
            'avg_ret_pct': round(avg_ret, 3),
            'win_rate': round(win_rate, 1),
            'sharpe': round(sharpe, 2),
            'sortino': round(sortino, 2),
            'profit_factor': round(pf, 2),
            'max_dd_pct': round(max_dd, 2),
            'avg_hold_days': round(avg_hold, 1),
            'exit_breakdown': {k: int(v) for k, v in exit_counts.items()},
            'bull_sharpe': round(bull_sharpe, 2),
            'bear_sharpe': round(bear_sharpe, 2),
            'regime_gap': round(regime_gap, 3),
        }

        print(f"\n  --- {strat_name.upper()} ---")
        print(f"  Trades: {n_trades}, Avg hold: {avg_hold:.1f}d")
        print(f"  Avg return: {avg_ret:.3f}%, WR: {win_rate:.1f}%")
        print(f"  Sharpe: {sharpe:.2f}, Sortino: {sortino:.2f}, PF: {pf:.2f}")
        print(f"  Max DD: {max_dd:.2f}%")
        print(f"  Exit breakdown: {exit_counts}")
        print(f"  Regime gap: {regime_gap:.3f} (bull Sharpe={bull_sharpe:.2f}, bear={bear_sharpe:.2f})")

        if strat_name == 'dynamic':
            pred_counts = trades_df['prediction'].value_counts()
            print(f"  Predictions: {pred_counts.to_dict()}")

            # Accuracy of prediction
            for pred in ['FAST', 'SLOW', 'DEFAULT']:
                mask = trades_df['prediction'] == pred
                if mask.sum() > 0:
                    pred_wr = (trades_df.loc[mask, 'net_ret'] > 0).mean() * 100
                    pred_avg = trades_df.loc[mask, 'net_ret'].mean() * 100
                    print(f"    {pred}: {mask.sum()} trades, WR={pred_wr:.1f}%, avg={pred_avg:.3f}%")

    return results


def permutation_test(dips_df, close, n_perms=1000):
    """Test if dynamic strategy improvement over baseline is statistically significant."""
    print("\n=== PERMUTATION TEST ===")

    # Get actual trade returns for both strategies
    def get_returns(df, strategy='fixed'):
        trades = []
        dynamic_rules = {
            'FAST': {'hold_days': 2, 'tp_pct': 2.0, 'sl_pct': -3.0},
            'SLOW': {'hold_days': 10, 'tp_pct': 5.0, 'sl_pct': -6.0},
            'DEFAULT': {'hold_days': 5, 'tp_pct': 3.0, 'sl_pct': -5.0},
        }

        for _, dip in df.iterrows():
            etf = dip['etf']
            if etf not in close.columns:
                continue
            px = close[etf].dropna()
            entry_date = dip['date']
            entry_price = dip['entry_price']
            entry_idx = px.index.get_loc(entry_date) if entry_date in px.index else None
            if entry_idx is None:
                continue

            if strategy == 'dynamic':
                pred = predict_recovery_speed(dip)
                params = dynamic_rules[pred]
            else:
                params = {'hold_days': 5, 'tp_pct': 3.0, 'sl_pct': -5.0}

            exit_price = None
            for d in range(1, params['hold_days'] + 1):
                if entry_idx + d >= len(px):
                    break
                day_ret = (px.iloc[entry_idx + d] - entry_price) / entry_price * 100
                if day_ret >= params['tp_pct']:
                    exit_price = entry_price * (1 + params['tp_pct'] / 100)
                    break
                elif day_ret <= params['sl_pct']:
                    exit_price = entry_price * (1 + params['sl_pct'] / 100)
                    break

            if exit_price is None:
                end_idx = min(entry_idx + params['hold_days'], len(px) - 1)
                exit_price = px.iloc[end_idx]

            trades.append((exit_price / entry_price - 1) - COST_RT_PCT)

        return np.array(trades)

    actual_dynamic = get_returns(dips_df, 'dynamic')
    actual_fixed = get_returns(dips_df, 'fixed')

    actual_diff = actual_dynamic.mean() - actual_fixed.mean()

    # Permutation: randomly assign dynamic/fixed labels
    combined = np.concatenate([actual_dynamic, actual_fixed])
    n = len(actual_dynamic)

    perm_diffs = []
    rng = np.random.RandomState(42)
    for _ in range(n_perms):
        perm = rng.permutation(combined)
        perm_diffs.append(perm[:n].mean() - perm[n:].mean())

    perm_diffs = np.array(perm_diffs)
    p_value = (np.abs(perm_diffs) >= np.abs(actual_diff)).mean()

    print(f"  Actual mean diff (dynamic - fixed): {actual_diff*100:.4f}%")
    print(f"  P-value (two-sided, {n_perms} perms): {p_value:.4f}")
    print(f"  Significant at p<0.05: {'YES' if p_value < 0.05 else 'NO'}")

    return {
        'actual_diff_pct': round(actual_diff * 100, 4),
        'p_value': round(p_value, 4),
        'significant': p_value < 0.05,
        'n_permutations': n_perms,
    }


def main():
    print("=" * 70)
    print("RECOVERY SPEED BACKTEST")
    print(f"Sector ETFs: {', '.join(SECTOR_ETFS)}")
    print(f"Period: {START} to {END}")
    print(f"Entry: RSI(14) < {RSI_THRESHOLD}, dip depth >= 3%")
    print(f"Cost: {COST_RT_PCT*100:.2f}% RT")
    print("=" * 70)

    close, volume = download_data()

    # 1. Classify dips
    dips_df = classify_dips(close, volume)

    if len(dips_df) < 20:
        print(f"\nWARNING: Only {len(dips_df)} dips found. Results may not be reliable.")

    # 2. Feature analysis
    feature_results = feature_analysis(dips_df)

    # 3. Backtest
    backtest_results = run_backtest(dips_df, close)

    # 4. Permutation test
    perm_results = permutation_test(dips_df, close)

    # 5. MFE analysis by recovery type
    print("\n=== MFE BY RECOVERY TYPE ===")
    for label, mask in [('V-shape', dips_df['v_shape']), ('Slow grind', dips_df['slow_grind'])]:
        subset = dips_df[mask]
        if len(subset) == 0:
            continue
        print(f"\n  {label} ({len(subset)} dips):")
        for h in [1, 3, 5, 10, 20]:
            col = f'mfe_{h}d'
            if col in subset.columns:
                print(f"    MFE {h:>2}d: median={subset[col].median():.2f}%, mean={subset[col].mean():.2f}%")

    # 6. Per-sector summary
    print("\n=== PER-SECTOR DIP SUMMARY ===")
    print(f"  {'ETF':<6} {'Dips':>5} {'V-shape%':>9} {'Med Recov':>10} {'MFE_5d':>8}")
    print("  " + "-" * 45)
    for etf in SECTOR_ETFS:
        subset = dips_df[dips_df['etf'] == etf]
        if len(subset) == 0:
            continue
        med_r = subset.loc[subset['recovery_time'].notna(), 'recovery_time'].median()
        med_r_str = f"{med_r:.0f}d" if not np.isnan(med_r) else "N/A"
        print(f"  {etf:<6} {len(subset):>5} {subset['v_shape'].mean()*100:>8.1f}% {med_r_str:>10} {subset['mfe_5d'].mean():>8.2f}%")

    # Compile and save results
    all_results = {
        'metadata': {
            'start': START,
            'end': END,
            'rsi_threshold': RSI_THRESHOLD,
            'cost_rt_pct': COST_RT_PCT * 100,
            'total_dips': len(dips_df),
            'run_date': datetime.now().isoformat(),
        },
        'dip_stats': {
            'total': len(dips_df),
            'v_shape_count': int(dips_df['v_shape'].sum()),
            'v_shape_pct': round(dips_df['v_shape'].mean() * 100, 1),
            'slow_grind_count': int(dips_df['slow_grind'].sum()),
            'slow_grind_pct': round(dips_df['slow_grind'].mean() * 100, 1),
            'median_recovery_days': round(float(dips_df.loc[dips_df['recovery_time'].notna(), 'recovery_time'].median()), 1),
        },
        'feature_importance': feature_results.get('_importance_ranking', []),
        'feature_analysis': {k: v for k, v in feature_results.items() if not k.startswith('_')},
        'backtest': backtest_results,
        'permutation_test': perm_results,
        'dynamic_exit_rules': {
            'FAST': {'hold_days': 2, 'tp_pct': 2.0, 'sl_pct': -3.0,
                     'conditions': 'VIX>25, capitulation volume, above 200SMA, SPY uptrend'},
            'SLOW': {'hold_days': 10, 'tp_pct': 5.0, 'sl_pct': -6.0,
                     'conditions': 'VIX<15, below 200SMA, no capitulation, SPY downtrend'},
            'DEFAULT': {'hold_days': 5, 'tp_pct': 3.0, 'sl_pct': -5.0,
                        'conditions': 'Mixed signals'},
        },
        'per_sector': {},
    }

    for etf in SECTOR_ETFS:
        subset = dips_df[dips_df['etf'] == etf]
        if len(subset) == 0:
            continue
        med_r = subset.loc[subset['recovery_time'].notna(), 'recovery_time'].median()
        all_results['per_sector'][etf] = {
            'dips': len(subset),
            'v_shape_pct': round(subset['v_shape'].mean() * 100, 1),
            'median_recovery': round(float(med_r), 1) if not np.isnan(med_r) else None,
            'avg_mfe_5d': round(subset['mfe_5d'].mean(), 2),
        }

    with open(RESULTS_PATH, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {RESULTS_PATH}")

    # Final summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if 'fixed_baseline' in backtest_results and 'dynamic' in backtest_results:
        fb = backtest_results['fixed_baseline']
        dy = backtest_results['dynamic']
        print(f"\n  {'Metric':<20} {'Fixed Baseline':>15} {'Dynamic':>15} {'Delta':>10}")
        print("  " + "-" * 65)
        for metric in ['avg_ret_pct', 'win_rate', 'sharpe', 'sortino', 'profit_factor', 'max_dd_pct']:
            v1 = fb[metric]
            v2 = dy[metric]
            delta = v2 - v1
            sign = '+' if delta > 0 else ''
            print(f"  {metric:<20} {v1:>15.2f} {v2:>15.2f} {sign}{delta:>9.2f}")

        print(f"\n  Regime gap (dynamic): {dy['regime_gap']:.3f} {'PASS' if dy['regime_gap'] < 0.50 else 'FAIL'}")
        print(f"  Permutation p-value: {perm_results['p_value']:.4f} {'PASS' if perm_results['significant'] else 'FAIL (not significant)'}")

    print("\nDone.")


if __name__ == '__main__':
    main()
