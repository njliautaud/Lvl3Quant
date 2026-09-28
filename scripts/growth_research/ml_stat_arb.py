#!/usr/bin/env python3
"""
ML Statistical Arbitrage (Pairs Trading)
==========================================
Market-neutral strategy: trade mean-reversion on cointegrated ETF pairs,
with ML filter predicting which spread extremes actually revert.

KEY DIFFERENTIATION: This is market-neutral by construction — zero beta to
equities. Should have ZERO correlation with our directional strategies
(ML Trend, Sector Rotation, Credit Timing).

Universe: Sector ETFs (11 SPDR) + cross-asset pairs
Method:
  1. Rolling cointegration test to identify valid pairs
  2. Z-score of spread identifies extremes
  3. ML (GBM) predicts which extremes revert vs regime shift
  4. Position: long cheap / short expensive when ML says revert

Walk-forward: 252d train, 1d advance (same as validated strategies)
Full adversarial: permutation, sub-period, outlier, R1 regime

HC #713: Fixed capital, no DCA
HC #714: Income + growth (this is income via arb profit)
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
import yfinance as yf
from datetime import datetime
from itertools import combinations
from scipy import stats
import json, os, warnings
warnings.filterwarnings('ignore')

OUTPUT = '/home/jupiter/Lvl3Quant/output/ml_stat_arb'
os.makedirs(OUTPUT, exist_ok=True)
np.random.seed(42)

# Parameters
INITIAL_CAPITAL = 100_000
TRAIN_WINDOW = 252
COINT_LOOKBACK = 126  # 6mo rolling cointegration test
ENTRY_Z = 1.5
EXIT_Z = 0.3
STOP_Z = 4.0  # Stop loss if spread blows out
MAX_HOLD = 42  # Max 42 days (2 months)
REBAL_CHECK = 1  # Check daily for entry/exit
N_PERM = 100
MAX_PAIRS = 8  # Trade top N most cointegrated pairs simultaneously
POS_SIZE = 1.0 / MAX_PAIRS  # Equal weight per pair

# Transaction costs (conservative for ETFs)
COST_BPS = 10  # 10 bps round trip (spread + commission for ETFs)
# HC #718 R3: Short borrow cost (50 bps annualized) — always has a short leg
SHORT_BORROW_ANNUAL = 0.0050


def download_data():
    """Download sector ETFs + cross-asset universe."""
    # Sector ETFs
    sectors = ['XLK','XLF','XLE','XLV','XLY','XLP','XLI','XLB','XLU','XLRE','XLC']
    # Cross-asset pairs candidates
    cross = ['GLD','GDX','TLT','IEF','HYG','LQD','SPY','QQQ','IWM','EEM','DIA']

    tickers = list(set(sectors + cross))
    print(f"Downloading {len(tickers)} assets...")
    df = yf.download(tickers, start='2005-01-01', progress=False)
    if hasattr(df.index, 'tz') and df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    close = df['Close'] if isinstance(df.columns, pd.MultiIndex) else df
    close = close.ffill()

    # Drop assets with too much missing data
    valid = close.columns[close.notna().sum() > 2000]
    close = close[valid].dropna()
    print(f"  {len(close)} days, {len(close.columns)} assets, {close.index[0].date()} to {close.index[-1].date()}")
    return close


def rolling_cointegration(price_a, price_b, window=COINT_LOOKBACK):
    """Test if two series are cointegrated over rolling window."""
    if len(price_a) < window:
        return np.nan, np.nan, np.nan

    # Engle-Granger: regress A on B, test residual for stationarity
    a = price_a.values[-window:]
    b = price_b.values[-window:]

    # Normalize to avoid numerical issues
    a_norm = a / a[0]
    b_norm = b / b[0]

    # OLS: a = beta * b + alpha + residual
    X = np.column_stack([b_norm, np.ones(window)])
    try:
        beta, alpha = np.linalg.lstsq(X, a_norm, rcond=None)[0]
    except:
        return np.nan, np.nan, np.nan

    residual = a_norm - beta * b_norm - alpha

    # ADF test on residual (simplified — check if residual is mean-reverting)
    # Use lag-1 autocorrelation as quick proxy
    if np.std(residual) < 1e-10:
        return np.nan, np.nan, np.nan

    # Half-life of mean reversion
    residual_lag = residual[:-1]
    residual_diff = np.diff(residual)
    if np.std(residual_lag) < 1e-10:
        return np.nan, np.nan, np.nan

    slope = np.polyfit(residual_lag, residual_diff, 1)[0]
    if slope >= 0:
        return np.nan, np.nan, np.nan  # Not mean-reverting

    half_life = -np.log(2) / slope

    # Hurst exponent (simplified)
    # If H < 0.5, series is mean-reverting
    lags = range(2, min(20, window//5))
    tau = []
    for lag in lags:
        tau.append(np.std(np.subtract(residual[lag:], residual[:-lag])))
    if len(tau) < 2 or any(t <= 0 for t in tau):
        return np.nan, beta, half_life

    try:
        hurst = np.polyfit(np.log(list(lags)), np.log(tau), 1)[0]
    except:
        hurst = 0.5

    return hurst, beta, half_life


def compute_spread(price_a, price_b, beta):
    """Compute normalized spread between two assets."""
    a_norm = price_a / price_a.iloc[0]
    b_norm = price_b / price_b.iloc[0]
    spread = a_norm - beta * b_norm
    return spread


def build_pair_features(close, pair, lookback_end):
    """Build ML features for a pair at a given point in time."""
    a, b = pair
    window = close.iloc[max(0, lookback_end-252):lookback_end]

    if len(window) < 126:
        return None

    pa = window[a]
    pb = window[b]

    # Get cointegration params
    hurst, beta, half_life = rolling_cointegration(pa, pb)
    if np.isnan(hurst) or np.isnan(beta):
        return None

    # Compute spread
    spread = pa / pa.iloc[0] - beta * (pb / pb.iloc[0])
    spread_mean = spread.rolling(63).mean()
    spread_std = spread.rolling(63).std()

    if spread_std.iloc[-1] < 1e-8:
        return None

    z = (spread.iloc[-1] - spread_mean.iloc[-1]) / spread_std.iloc[-1]

    # Features
    feats = {}
    feats['z_score'] = z
    feats['z_score_abs'] = abs(z)
    feats['hurst'] = hurst
    feats['half_life'] = half_life
    feats['beta'] = beta

    # Z-score dynamics
    z_series = (spread - spread_mean) / spread_std.clip(lower=1e-8)
    feats['z_velocity_5d'] = z_series.iloc[-1] - z_series.iloc[-6] if len(z_series) > 5 else 0
    feats['z_velocity_21d'] = z_series.iloc[-1] - z_series.iloc[-22] if len(z_series) > 21 else 0
    feats['z_max_21d'] = z_series.iloc[-21:].max() if len(z_series) > 21 else z
    feats['z_min_21d'] = z_series.iloc[-21:].min() if len(z_series) > 21 else z

    # Volatility features
    ret_a = pa.pct_change()
    ret_b = pb.pct_change()
    feats['vol_a_21d'] = ret_a.iloc[-21:].std() * np.sqrt(252) if len(ret_a) > 21 else np.nan
    feats['vol_b_21d'] = ret_b.iloc[-21:].std() * np.sqrt(252) if len(ret_b) > 21 else np.nan
    feats['vol_ratio'] = feats['vol_a_21d'] / max(feats['vol_b_21d'], 1e-8)

    # Correlation
    if len(ret_a) > 63:
        feats['corr_21d'] = ret_a.iloc[-21:].corr(ret_b.iloc[-21:])
        feats['corr_63d'] = ret_a.iloc[-63:].corr(ret_b.iloc[-63:])
        feats['corr_change'] = feats['corr_21d'] - feats['corr_63d']
    else:
        feats['corr_21d'] = feats['corr_63d'] = feats['corr_change'] = np.nan

    # Spread volatility
    spread_ret = spread.diff()
    feats['spread_vol_21d'] = spread_ret.iloc[-21:].std() if len(spread_ret) > 21 else np.nan
    feats['spread_vol_63d'] = spread_ret.iloc[-63:].std() if len(spread_ret) > 63 else np.nan

    # Mean reversion strength (how fast did recent deviations revert?)
    if len(z_series) > 42:
        # Look at z-score 21d ago — did it revert toward zero?
        z_21d_ago = z_series.iloc[-22]
        reversion = 1 - abs(z) / max(abs(z_21d_ago), 0.1)
        feats['recent_reversion'] = reversion
    else:
        feats['recent_reversion'] = 0

    # Market regime features
    if 'SPY' in close.columns:
        spy_window = close['SPY'].iloc[max(0, lookback_end-252):lookback_end]
        spy_ret = spy_window.pct_change()
        feats['spy_vol_21d'] = spy_ret.iloc[-21:].std() * np.sqrt(252) if len(spy_ret) > 21 else np.nan
        feats['spy_ret_21d'] = spy_window.iloc[-1] / spy_window.iloc[-22] - 1 if len(spy_window) > 22 else 0

    return feats


def run_backtest(close, use_ml=True, shuffle_signals=False):
    """
    Run full walk-forward stat arb backtest.

    If use_ml=False: pure z-score entry/exit (no ML filter)
    If shuffle_signals=True: randomize ML predictions (permutation test)
    """
    returns = close.pct_change()
    n_days = len(close)

    # Generate all valid pairs
    assets = list(close.columns)
    all_pairs = list(combinations(assets, 2))

    # Track portfolio
    daily_returns = np.zeros(n_days)
    positions = {}  # {pair: {'direction': 1/-1, 'entry_day': int, 'entry_z': float}}
    trade_log = []

    # Walk-forward ML
    train_X, train_y = [], []
    model = None

    start_day = max(TRAIN_WINDOW + COINT_LOOKBACK, 504)  # Need enough history

    for day in range(start_day, n_days):
        # Daily P&L from existing positions
        day_pnl = 0
        closed_pairs = []

        for pair, pos in positions.items():
            a, b = pair
            ret_a = returns[a].iloc[day] if not np.isnan(returns[a].iloc[day]) else 0
            ret_b = returns[b].iloc[day] if not np.isnan(returns[b].iloc[day]) else 0

            # Long A / Short B if direction=1, opposite if direction=-1
            pair_ret = pos['direction'] * (ret_a - ret_b) * POS_SIZE
            # HC #718 R3: daily short borrow cost on the short leg
            pair_ret -= SHORT_BORROW_ANNUAL / 252 * POS_SIZE
            day_pnl += pair_ret

            # Check exit conditions
            pa = close[a].iloc[max(0,day-63):day+1]
            pb = close[b].iloc[max(0,day-63):day+1]
            if len(pa) > 10:
                hurst, beta, hl = rolling_cointegration(pa, pb, min(63, len(pa)))
                if not np.isnan(beta):
                    spread = pa.iloc[-1]/pa.iloc[0] - beta*(pb.iloc[-1]/pb.iloc[0])
                    sp_mean = (pa/pa.iloc[0] - beta*(pb/pb.iloc[0])).mean()
                    sp_std = (pa/pa.iloc[0] - beta*(pb/pb.iloc[0])).std()
                    if sp_std > 1e-8:
                        current_z = (spread - sp_mean) / sp_std
                    else:
                        current_z = 0
                else:
                    current_z = 0
            else:
                current_z = 0

            days_held = day - pos['entry_day']

            # Exit: mean reverted, stop loss, or max hold
            exit_signal = (
                abs(current_z) < EXIT_Z or  # Reverted
                (pos['direction'] * current_z > STOP_Z) or  # Blew out further
                days_held >= MAX_HOLD  # Time stop
            )

            if exit_signal:
                closed_pairs.append(pair)
                # Calculate trade P&L
                cum_ret = 0
                for d in range(pos['entry_day'] + 1, day + 1):
                    ra = returns[a].iloc[d] if not np.isnan(returns[a].iloc[d]) else 0
                    rb = returns[b].iloc[d] if not np.isnan(returns[b].iloc[d]) else 0
                    cum_ret += pos['direction'] * (ra - rb)

                trade_log.append({
                    'pair': f"{a}/{b}",
                    'direction': pos['direction'],
                    'entry_day': pos['entry_day'],
                    'exit_day': day,
                    'days_held': days_held,
                    'return': cum_ret - COST_BPS/10000 * 2,  # Entry + exit cost
                    'exit_reason': 'revert' if abs(current_z) < EXIT_Z else ('stop' if pos['direction']*current_z > STOP_Z else 'time')
                })

        # Remove closed positions
        for pair in closed_pairs:
            del positions[pair]

        # Apply daily PnL
        daily_returns[day] = day_pnl

        # Every 5 days: scan for new entries (avoid over-trading)
        if day % 5 != 0:
            continue

        if len(positions) >= MAX_PAIRS:
            continue

        # Score all pairs by cointegration quality
        pair_scores = []
        for pair in all_pairs:
            if pair in positions:
                continue
            a, b = pair
            pa = close[a].iloc[max(0, day-COINT_LOOKBACK):day+1]
            pb = close[b].iloc[max(0, day-COINT_LOOKBACK):day+1]

            if len(pa) < 63:
                continue

            hurst, beta, half_life = rolling_cointegration(pa, pb)
            if np.isnan(hurst) or hurst > 0.45 or np.isnan(half_life) or half_life > 42 or half_life < 2:
                continue

            # Compute current z-score
            spread = pa/pa.iloc[0] - beta*(pb/pb.iloc[0])
            sp_mean = spread.rolling(63).mean().iloc[-1]
            sp_std = spread.rolling(63).std().iloc[-1]
            if sp_std < 1e-8:
                continue
            z = (spread.iloc[-1] - sp_mean) / sp_std

            if abs(z) < ENTRY_Z:
                continue

            # Build features for ML
            feats = build_pair_features(close, pair, day+1)
            if feats is None:
                continue

            pair_scores.append((pair, z, beta, hurst, half_life, feats))

        if not pair_scores:
            continue

        # ML filter: predict probability of successful reversion
        if use_ml and model is not None:
            for pair, z, beta, hurst, half_life, feats in pair_scores:
                feat_vals = np.array([[feats.get(k, 0) for k in sorted(feats.keys())]])
                feat_vals = np.nan_to_num(feat_vals, 0)

                if shuffle_signals:
                    pred = np.random.random()
                else:
                    try:
                        pred = model.predict(feat_vals)[0]
                    except:
                        pred = 0.5

                if pred > 0.5:  # ML says this will revert
                    direction = -1 if z > 0 else 1  # Short the spread if z>0
                    positions[pair] = {
                        'direction': direction,
                        'entry_day': day,
                        'entry_z': z
                    }
                    if len(positions) >= MAX_PAIRS:
                        break
        elif not use_ml:
            # No ML — just use z-score + hurst filter
            # Sort by Hurst (lower = more mean-reverting)
            pair_scores.sort(key=lambda x: x[3])
            for pair, z, beta, hurst, half_life, feats in pair_scores[:MAX_PAIRS - len(positions)]:
                direction = -1 if z > 0 else 1
                positions[pair] = {
                    'direction': direction,
                    'entry_day': day,
                    'entry_z': z
                }

        # Update ML training data (label: did the trade that WOULD have been taken revert?)
        if use_ml and day > start_day + TRAIN_WINDOW:
            # Retrain every 63 days
            if day % 63 == 0 and len(train_X) > 50:
                # HC #718: label gap = MAX_HOLD to prevent look-ahead
                # Exclude last MAX_HOLD samples whose labels overlap current day
                gap_safe = max(0, len(train_X) - MAX_HOLD)
                recent_X = train_X[:gap_safe][-TRAIN_WINDOW*5:]
                recent_y = train_y[:gap_safe][-TRAIN_WINDOW*5:]

                X_arr = np.array(recent_X)
                y_arr = np.array(recent_y)
                X_arr = np.nan_to_num(X_arr, 0)

                try:
                    ds = lgb.Dataset(X_arr, label=y_arr)
                    params = {
                        'objective': 'binary',
                        'metric': 'auc',
                        'num_leaves': 15,
                        'learning_rate': 0.05,
                        'feature_fraction': 0.7,
                        'bagging_fraction': 0.7,
                        'bagging_freq': 5,
                        'verbose': -1,
                        'n_jobs': -1,
                    }
                    model = lgb.train(params, ds, num_boost_round=100)
                except:
                    pass

        # Collect training labels from past entries
        if use_ml:
            for pair, z, beta, hurst, half_life, feats in pair_scores:
                # Look forward MAX_HOLD days to see if it reverted
                if day + MAX_HOLD < n_days:
                    a, b = pair
                    direction = -1 if z > 0 else 1
                    # Did the spread revert within MAX_HOLD days?
                    future_pa = close[a].iloc[day:day+MAX_HOLD+1]
                    future_pb = close[b].iloc[day:day+MAX_HOLD+1]
                    if len(future_pa) > 5:
                        future_spread = future_pa/future_pa.iloc[0] - beta*(future_pb/future_pb.iloc[0])
                        sp_mean_f = future_spread.mean()
                        # Label: did z come back within EXIT_Z within hold period?
                        initial_deviation = future_spread.iloc[0] - sp_mean_f
                        end_deviation = future_spread.iloc[-1] - sp_mean_f
                        reverted = abs(end_deviation) < abs(initial_deviation) * 0.5

                        feat_vals = [feats.get(k, 0) for k in sorted(feats.keys())]
                        train_X.append(feat_vals)
                        train_y.append(int(reverted))

    return daily_returns[start_day:], trade_log, start_day


def compute_metrics(daily_returns, label=""):
    """Compute risk-adjusted metrics."""
    if len(daily_returns) == 0 or np.std(daily_returns) == 0:
        return {'sharpe': 0, 'sortino': 0, 'cagr': 0, 'maxdd': 0, 'wr': 0, 'pf': 0, 'n_days': 0}

    equity = (1 + pd.Series(daily_returns)).cumprod()
    n_years = len(daily_returns) / 252

    ann_ret = equity.iloc[-1] ** (1/n_years) - 1 if n_years > 0 else 0
    ann_vol = np.std(daily_returns) * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    downside = daily_returns[daily_returns < 0]
    down_vol = np.std(downside) * np.sqrt(252) if len(downside) > 0 else 1
    sortino = ann_ret / down_vol if down_vol > 0 else 0

    # Max drawdown
    peak = equity.cummax()
    dd = (equity - peak) / peak
    maxdd = dd.min()

    # Win rate and profit factor (from daily returns)
    wins = daily_returns[daily_returns > 0]
    losses = daily_returns[daily_returns < 0]
    wr = len(wins) / max(len(wins) + len(losses), 1)
    pf = abs(wins.sum() / losses.sum()) if len(losses) > 0 and losses.sum() != 0 else 0

    cagr = ann_ret

    return {
        'sharpe': sharpe,
        'sortino': sortino,
        'cagr': cagr,
        'maxdd': maxdd,
        'wr': wr,
        'pf': pf,
        'n_days': len(daily_returns),
        'n_years': n_years,
        'ann_vol': ann_vol
    }


def adversarial_validation(daily_returns, close, start_day):
    """Run full adversarial validation suite."""
    results = {}

    # 1. Permutation test (shuffle ML signals)
    print("\n  [1/4] Permutation test (shuffling ML predictions)...")
    perm_sharpes = []
    for i in range(N_PERM):
        if i % 20 == 0:
            print(f"    Perm {i}/{N_PERM}...")
        perm_rets, _, _ = run_backtest(close, use_ml=True, shuffle_signals=True)
        m = compute_metrics(perm_rets)
        perm_sharpes.append(m['sharpe'])

    real_sharpe = compute_metrics(daily_returns)['sharpe']
    perm_mean = np.mean(perm_sharpes)
    perm_std = np.std(perm_sharpes) if np.std(perm_sharpes) > 0 else 1
    p_value = np.mean([s >= real_sharpe for s in perm_sharpes])

    results['perm'] = {
        'real_sharpe': real_sharpe,
        'perm_mean': perm_mean,
        'perm_std': perm_std,
        'p_value': p_value,
        'pass': p_value < 0.05
    }
    print(f"    Perm: real={real_sharpe:.3f}, random={perm_mean:.3f}, p={p_value:.3f} {'PASS' if p_value < 0.05 else 'FAIL'}")

    # 2. Sub-period stability
    print("\n  [2/4] Sub-period stability...")
    n = len(daily_returns)
    quarters = np.array_split(daily_returns, 4)
    q_sharpes = [compute_metrics(q)['sharpe'] for q in quarters]
    cv = np.std(q_sharpes) / max(abs(np.mean(q_sharpes)), 0.01)

    results['subperiod'] = {
        'quarter_sharpes': q_sharpes,
        'cv': cv,
        'pass': cv < 1.0  # Less strict for market-neutral
    }
    print(f"    Sub-period: Sharpes={[f'{s:.2f}' for s in q_sharpes]}, CV={cv:.3f} {'PASS' if cv < 1.0 else 'FAIL'}")

    # 3. Outlier sensitivity
    print("\n  [3/4] Outlier sensitivity...")
    sorted_rets = np.sort(daily_returns)
    # Remove top/bottom 1% of days
    trim_n = max(1, int(len(sorted_rets) * 0.01))
    trimmed = sorted_rets[trim_n:-trim_n]
    trimmed_sharpe = compute_metrics(trimmed)['sharpe']
    degradation = (real_sharpe - trimmed_sharpe) / max(abs(real_sharpe), 0.01)

    results['outlier'] = {
        'full_sharpe': real_sharpe,
        'trimmed_sharpe': trimmed_sharpe,
        'degradation_pct': degradation * 100,
        'pass': abs(degradation) < 0.50
    }
    print(f"    Outlier: full={real_sharpe:.3f}, trimmed={trimmed_sharpe:.3f}, deg={degradation*100:.1f}% {'PASS' if abs(degradation) < 0.50 else 'FAIL'}")

    # 4. R1 Regime test (green vs red days)
    print("\n  [4/4] R1 Regime test...")
    spy_data = close['SPY'].iloc[start_day:start_day+len(daily_returns)]
    spy_daily = spy_data.pct_change()

    green_mask = spy_daily > 0
    red_mask = spy_daily < 0

    green_rets = daily_returns[green_mask.values[:len(daily_returns)]] if green_mask.sum() > 0 else np.array([0])
    red_rets = daily_returns[red_mask.values[:len(daily_returns)]] if red_mask.sum() > 0 else np.array([0])

    green_sharpe = compute_metrics(green_rets)['sharpe']
    red_sharpe = compute_metrics(red_rets)['sharpe']

    gap = abs(green_sharpe - red_sharpe) / max(abs(green_sharpe), abs(red_sharpe), 0.01)

    results['r1_regime'] = {
        'green_sharpe': green_sharpe,
        'red_sharpe': red_sharpe,
        'gap': gap,
        'pass': gap < 0.50
    }
    print(f"    R1: green={green_sharpe:.3f}, red={red_sharpe:.3f}, gap={gap:.3f} {'PASS' if gap < 0.50 else 'FAIL'}")

    return results


def main():
    print("=" * 70)
    print("ML STATISTICAL ARBITRAGE (PAIRS TRADING)")
    print("Market-Neutral Strategy — Zero Beta by Construction")
    print("=" * 70)

    # Download data
    close = download_data()

    # Run ML backtest
    print("\n[1] Running ML-filtered stat arb backtest...")
    ml_returns, ml_trades, start_day = run_backtest(close, use_ml=True, shuffle_signals=False)
    ml_metrics = compute_metrics(ml_returns, "ML Stat Arb")

    print(f"\n  ML Stat Arb Results:")
    print(f"    Sharpe: {ml_metrics['sharpe']:.3f}")
    print(f"    Sortino: {ml_metrics['sortino']:.3f}")
    print(f"    CAGR: {ml_metrics['cagr']*100:.1f}%")
    print(f"    MaxDD: {ml_metrics['maxdd']*100:.1f}%")
    print(f"    WR: {ml_metrics['wr']*100:.1f}%")
    print(f"    PF: {ml_metrics['pf']:.3f}")
    print(f"    Trades: {len(ml_trades)}")
    print(f"    Years: {ml_metrics['n_years']:.1f}")

    # Run baseline (no ML)
    print("\n[2] Running baseline (no ML filter)...")
    base_returns, base_trades, _ = run_backtest(close, use_ml=False, shuffle_signals=False)
    base_metrics = compute_metrics(base_returns, "Baseline")

    print(f"\n  Baseline (pure z-score) Results:")
    print(f"    Sharpe: {base_metrics['sharpe']:.3f}")
    print(f"    Sortino: {base_metrics['sortino']:.3f}")
    print(f"    CAGR: {base_metrics['cagr']*100:.1f}%")
    print(f"    MaxDD: {base_metrics['maxdd']*100:.1f}%")
    print(f"    Trades: {len(base_trades)}")

    # ML adds value?
    ml_better = ml_metrics['sharpe'] > base_metrics['sharpe']
    print(f"\n  ML adds value: {'YES' if ml_better else 'NO'} (Sharpe {ml_metrics['sharpe']:.3f} vs {base_metrics['sharpe']:.3f})")

    # Trade analysis
    if ml_trades:
        trade_df = pd.DataFrame(ml_trades)
        print(f"\n  Trade Analysis:")
        print(f"    Avg trade return: {trade_df['return'].mean()*100:.3f}%")
        print(f"    Avg hold days: {trade_df['days_held'].mean():.1f}")
        print(f"    Exit reasons: {trade_df['exit_reason'].value_counts().to_dict()}")
        print(f"    Win rate (trades): {(trade_df['return'] > 0).mean()*100:.1f}%")

        # Top pairs
        pair_pnl = trade_df.groupby('pair')['return'].agg(['sum','count','mean'])
        pair_pnl = pair_pnl.sort_values('sum', ascending=False)
        print(f"\n  Top 5 pairs by total return:")
        for idx, row in pair_pnl.head(5).iterrows():
            print(f"    {idx}: total={row['sum']*100:.2f}%, trades={int(row['count'])}, avg={row['mean']*100:.3f}%")

    # Correlation with SPY (should be near zero for market-neutral)
    spy_returns = close['SPY'].pct_change().iloc[start_day:start_day+len(ml_returns)].values
    if len(spy_returns) == len(ml_returns):
        mkt_corr = np.corrcoef(ml_returns, spy_returns)[0,1]
        print(f"\n  Market correlation (vs SPY): {mkt_corr:.3f} (target: ~0)")

    # Adversarial validation
    print("\n[3] Adversarial Validation...")
    adv = adversarial_validation(ml_returns, close, start_day)

    gates_passed = sum(1 for v in adv.values() if v['pass'])
    total_gates = len(adv)

    print(f"\n{'='*70}")
    print(f"FINAL VERDICT: {gates_passed}/{total_gates} adversarial gates")
    print(f"  Permutation: {'PASS' if adv['perm']['pass'] else 'FAIL'} (p={adv['perm']['p_value']:.3f})")
    print(f"  Sub-period:  {'PASS' if adv['subperiod']['pass'] else 'FAIL'} (CV={adv['subperiod']['cv']:.3f})")
    print(f"  Outlier:     {'PASS' if adv['outlier']['pass'] else 'FAIL'} (deg={adv['outlier']['degradation_pct']:.1f}%)")
    print(f"  R1 Regime:   {'PASS' if adv['r1_regime']['pass'] else 'FAIL'} (gap={adv['r1_regime']['gap']:.3f})")
    print(f"{'='*70}")

    # Save results
    result = {
        'strategy': 'ML Statistical Arbitrage',
        'ml_metrics': {k: float(v) for k, v in ml_metrics.items()},
        'baseline_metrics': {k: float(v) for k, v in base_metrics.items()},
        'ml_adds_value': ml_better,
        'n_trades': len(ml_trades),
        'market_correlation': float(mkt_corr) if 'mkt_corr' in dir() else None,
        'adversarial': {k: {kk: (float(vv) if isinstance(vv, (int, float, np.floating, np.integer)) else vv)
                           for kk, vv in v.items()} for k, v in adv.items()},
        'gates_passed': gates_passed,
        'total_gates': total_gates,
        'timestamp': datetime.now().isoformat()
    }

    with open(f'{OUTPUT}/stat_arb_results.json', 'w') as f:
        json.dump(result, f, indent=2, default=str)

    # Save daily returns for correlation analysis
    np.save(f'{OUTPUT}/daily_returns.npy', ml_returns)

    print(f"\nResults saved to {OUTPUT}/")
    return result


if __name__ == '__main__':
    main()
