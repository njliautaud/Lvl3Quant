#!/usr/bin/env python3
"""
Overnight vs Intraday Return Decomposition — Sector ETF Options Signal
======================================================================
Hypothesis: Overnight returns capture institutional/informed flow while
intraday returns capture retail/noise. Divergence signals accumulation
or distribution.

Variants:
  A. Absolute overnight ratio threshold
  B. Z-score vs 60-day rolling mean
  C. Relative to SPY (z-scored)
  D. Combined with RSI
  E. Volume-weighted overnight ratio
  F. Sector-pair divergence (XLK vs XLU)

5-Gate system: Sharpe>0.5, p<0.05, regime_gap<0.50, MDD<50%, trades>=30
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
from datetime import datetime
import json
import sys

# ─── Configuration ───────────────────────────────────────────────────
TICKERS = ['XLE', 'XLU', 'XLF', 'XLK', 'XLY', 'XLP', 'XLB', 'XLI', 'XLV', 'XLRE', 'XLC', 'SPY']
START = '2018-01-01'
END = '2026-08-18'
HOLD_DAYS = 4  # 3-5 day hold, use 4 as midpoint
ROLL_WINDOW = 10  # rolling window for overnight ratio
ZSCORE_WINDOW = 60  # lookback for z-score
RSI_PERIOD = 14
N_PERMUTATIONS = 1000
SEED = 42

# 5-gate thresholds
SHARPE_MIN = 0.5
P_VALUE_MAX = 0.05
REGIME_GAP_MAX = 0.50
MDD_MAX = 0.50
MIN_TRADES = 30

np.random.seed(SEED)


def fetch_data():
    """Fetch daily OHLCV for all tickers."""
    print("Fetching data from yfinance...")
    data = {}
    for ticker in TICKERS:
        try:
            df = yf.download(ticker, start=START, end=END, progress=False, auto_adjust=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].copy()
            df.dropna(inplace=True)
            data[ticker] = df
            print(f"  {ticker}: {len(df)} days ({df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')})")
        except Exception as e:
            print(f"  {ticker}: FAILED - {e}")
    return data


def compute_returns(data):
    """Decompose daily returns into overnight and intraday components."""
    features = {}
    for ticker, df in data.items():
        f = pd.DataFrame(index=df.index)
        # Use Adj Close for proper return calculation
        adj_factor = df['Adj Close'] / df['Close']
        adj_open = df['Open'] * adj_factor
        adj_close = df['Adj Close']

        # Overnight: open_t vs close_{t-1}
        f['overnight_ret'] = (adj_open - adj_close.shift(1)) / adj_close.shift(1)
        # Intraday: close_t vs open_t
        f['intraday_ret'] = (adj_close - adj_open) / adj_open
        # Total daily return
        f['total_ret'] = (adj_close - adj_close.shift(1)) / adj_close.shift(1)
        # Volume
        f['volume'] = df['Volume']
        f['rel_volume'] = df['Volume'] / df['Volume'].rolling(20).mean()

        # Rolling 10-day sums
        f['overnight_sum_10d'] = f['overnight_ret'].rolling(ROLL_WINDOW).sum()
        f['total_sum_10d'] = f['total_ret'].rolling(ROLL_WINDOW).sum()
        # Overnight ratio: fraction of total return from overnight
        # Guard against division by zero
        f['overnight_ratio'] = np.where(
            f['total_sum_10d'].abs() > 1e-8,
            f['overnight_sum_10d'] / f['total_sum_10d'],
            np.nan
        )

        # Volume-weighted overnight ratio
        vol_wt = f['rel_volume'].rolling(ROLL_WINDOW)
        f['vw_overnight_sum'] = (f['overnight_ret'] * f['rel_volume']).rolling(ROLL_WINDOW).sum()
        f['vw_total_sum'] = (f['total_ret'] * f['rel_volume']).rolling(ROLL_WINDOW).sum()
        f['vw_overnight_ratio'] = np.where(
            f['vw_total_sum'].abs() > 1e-8,
            f['vw_overnight_sum'] / f['vw_total_sum'],
            np.nan
        )

        # Z-score of overnight ratio vs 60-day rolling
        roll_mean = f['overnight_ratio'].rolling(ZSCORE_WINDOW).mean()
        roll_std = f['overnight_ratio'].rolling(ZSCORE_WINDOW).std()
        f['overnight_ratio_z'] = np.where(
            roll_std > 1e-8,
            (f['overnight_ratio'] - roll_mean) / roll_std,
            np.nan
        )

        # RSI
        delta = adj_close.diff()
        gain = delta.clip(lower=0).rolling(RSI_PERIOD).mean()
        loss = (-delta.clip(upper=0)).rolling(RSI_PERIOD).mean()
        rs = gain / loss.replace(0, np.nan)
        f['rsi'] = 100 - (100 / (1 + rs))

        # Forward return for evaluation (hold HOLD_DAYS)
        f['fwd_ret'] = adj_close.shift(-HOLD_DAYS) / adj_close - 1

        f.dropna(subset=['overnight_ratio', 'fwd_ret'], inplace=True)
        features[ticker] = f

    return features


def compute_spy_features(features):
    """Add SPY-relative overnight ratio for each sector."""
    if 'SPY' not in features:
        return features

    spy = features['SPY'][['overnight_ratio', 'overnight_ratio_z']].copy()
    spy.columns = ['spy_overnight_ratio', 'spy_overnight_ratio_z']

    for ticker in features:
        if ticker == 'SPY':
            continue
        features[ticker] = features[ticker].join(spy, how='left')
        # Relative overnight ratio
        features[ticker]['rel_overnight_ratio'] = (
            features[ticker]['overnight_ratio'] - features[ticker]['spy_overnight_ratio']
        )
        # Z-score of relative ratio
        roll_mean = features[ticker]['rel_overnight_ratio'].rolling(ZSCORE_WINDOW).mean()
        roll_std = features[ticker]['rel_overnight_ratio'].rolling(ZSCORE_WINDOW).std()
        features[ticker]['rel_overnight_ratio_z'] = np.where(
            roll_std > 1e-8,
            (features[ticker]['rel_overnight_ratio'] - roll_mean) / roll_std,
            np.nan
        )
    return features


def classify_spy_regime(data):
    """Classify each day as green or red based on SPY close-to-close."""
    spy = data['SPY']
    adj_close = spy['Adj Close']
    daily_ret = adj_close.pct_change()
    regime = pd.Series('flat', index=spy.index)
    regime[daily_ret > 0.001] = 'green'
    regime[daily_ret < -0.001] = 'red'
    return regime


def generate_signals(features, spy_regime, variant):
    """Generate buy/sell signals for a given variant. Returns list of trades."""
    trades = []
    sector_tickers = [t for t in features if t != 'SPY']

    for ticker in sector_tickers:
        df = features[ticker].copy()
        df['spy_regime'] = spy_regime.reindex(df.index)
        df = df.dropna(subset=['fwd_ret'])

        if variant == 'A':
            # Absolute overnight ratio threshold
            buys = df[df['overnight_ratio'] > 0.7]
            sells = df[df['overnight_ratio'] < 0.3]
        elif variant == 'B':
            # Z-score vs 60-day rolling mean
            buys = df[df['overnight_ratio_z'] > 1.5]
            sells = df[df['overnight_ratio_z'] < -1.5]
        elif variant == 'C':
            # Relative to SPY, z-scored
            if 'rel_overnight_ratio_z' not in df.columns:
                continue
            valid = df.dropna(subset=['rel_overnight_ratio_z'])
            buys = valid[valid['rel_overnight_ratio_z'] > 1.5]
            sells = valid[valid['rel_overnight_ratio_z'] < -1.5]
        elif variant == 'D':
            # Combined with RSI: overnight accumulation + RSI<40 = buy
            buys = df[(df['overnight_ratio_z'] > 1.0) & (df['rsi'] < 40)]
            sells = df[(df['overnight_ratio_z'] < -1.0) & (df['rsi'] > 60)]
        elif variant == 'E':
            # Volume-weighted overnight ratio z-scored
            vw_roll_mean = df['vw_overnight_ratio'].rolling(ZSCORE_WINDOW).mean()
            vw_roll_std = df['vw_overnight_ratio'].rolling(ZSCORE_WINDOW).std()
            df['vw_overnight_ratio_z'] = np.where(
                vw_roll_std > 1e-8,
                (df['vw_overnight_ratio'] - vw_roll_mean) / vw_roll_std,
                np.nan
            )
            valid = df.dropna(subset=['vw_overnight_ratio_z'])
            buys = valid[valid['vw_overnight_ratio_z'] > 1.5]
            sells = valid[valid['vw_overnight_ratio_z'] < -1.5]
        elif variant == 'F':
            # Sector-pair divergence: only for XLK vs XLU
            if ticker not in ['XLK', 'XLU']:
                continue
            if 'XLK' not in features or 'XLU' not in features:
                continue
            xlk_ratio = features['XLK']['overnight_ratio'].reindex(df.index)
            xlu_ratio = features['XLU']['overnight_ratio'].reindex(df.index)
            spread = xlk_ratio - xlu_ratio
            spread_z = (spread - spread.rolling(ZSCORE_WINDOW).mean()) / spread.rolling(ZSCORE_WINDOW).std().replace(0, np.nan)
            spread_z = spread_z.dropna()
            if ticker == 'XLK':
                buys = df.loc[spread_z[spread_z > 1.5].index.intersection(df.index)]
                sells = df.loc[spread_z[spread_z < -1.5].index.intersection(df.index)]
            else:  # XLU — inverse
                buys = df.loc[spread_z[spread_z < -1.5].index.intersection(df.index)]
                sells = df.loc[spread_z[spread_z > 1.5].index.intersection(df.index)]
        else:
            continue

        for idx, row in buys.iterrows():
            trades.append({
                'date': idx, 'ticker': ticker, 'direction': 'LONG',
                'fwd_ret': row['fwd_ret'], 'regime': row.get('spy_regime', 'unknown')
            })
        for idx, row in sells.iterrows():
            trades.append({
                'date': idx, 'ticker': ticker, 'direction': 'SHORT',
                'fwd_ret': -row['fwd_ret'], 'regime': row.get('spy_regime', 'unknown')
            })

    return pd.DataFrame(trades)


def calc_metrics(returns_series):
    """Calculate Sharpe, win rate, profit factor, max drawdown."""
    if len(returns_series) == 0:
        return {'sharpe': 0, 'win_rate': 0, 'profit_factor': 0, 'max_dd': 1.0, 'n_trades': 0, 'avg_ret': 0}

    avg = returns_series.mean()
    std = returns_series.std()
    sharpe = (avg / std * np.sqrt(252 / HOLD_DAYS)) if std > 1e-10 else 0

    wins = returns_series[returns_series > 0]
    losses = returns_series[returns_series < 0]
    win_rate = len(wins) / len(returns_series) if len(returns_series) > 0 else 0
    gross_profit = wins.sum() if len(wins) > 0 else 0
    gross_loss = abs(losses.sum()) if len(losses) > 0 else 1e-10
    profit_factor = gross_profit / gross_loss if gross_loss > 1e-10 else 0

    # Max drawdown on cumulative returns
    cum = (1 + returns_series).cumprod()
    rolling_max = cum.cummax()
    drawdown = (cum - rolling_max) / rolling_max
    max_dd = abs(drawdown.min()) if len(drawdown) > 0 else 0

    return {
        'sharpe': round(sharpe, 3),
        'win_rate': round(win_rate, 3),
        'profit_factor': round(profit_factor, 3),
        'max_dd': round(max_dd, 3),
        'n_trades': len(returns_series),
        'avg_ret': round(avg * 100, 3),
        'total_ret': round((cum.iloc[-1] - 1) * 100, 2) if len(cum) > 0 else 0
    }


def permutation_test(actual_sharpe, returns_series, n_perms=N_PERMUTATIONS):
    """Shuffle trade returns and compute p-value."""
    if len(returns_series) < 5:
        return 1.0
    count_better = 0
    rets = returns_series.values.copy()
    for _ in range(n_perms):
        np.random.shuffle(rets)
        # Random subset of same size with replacement from all returns
        perm_sharpe = (rets.mean() / rets.std() * np.sqrt(252 / HOLD_DAYS)) if rets.std() > 1e-10 else 0
        if perm_sharpe >= actual_sharpe:
            count_better += 1
    return count_better / n_perms


def regime_analysis(trades_df):
    """Stratify by green/red SPY days."""
    if len(trades_df) == 0:
        return 0, 0, 1.0

    green = trades_df[trades_df['regime'] == 'green']['fwd_ret']
    red = trades_df[trades_df['regime'] == 'red']['fwd_ret']

    sharpe_green = (green.mean() / green.std() * np.sqrt(252/HOLD_DAYS)) if len(green) > 5 and green.std() > 1e-10 else 0
    sharpe_red = (red.mean() / red.std() * np.sqrt(252/HOLD_DAYS)) if len(red) > 5 and red.std() > 1e-10 else 0

    max_abs = max(abs(sharpe_green), abs(sharpe_red), 1e-10)
    regime_gap = abs(sharpe_green - sharpe_red) / max_abs

    return round(sharpe_green, 3), round(sharpe_red, 3), round(regime_gap, 3)


def five_gate_check(metrics, p_value, regime_gap):
    """Check all 5 gates."""
    gates = {
        'Sharpe > 0.5': metrics['sharpe'] > SHARPE_MIN,
        'p-value < 0.05': p_value < P_VALUE_MAX,
        'Regime gap < 0.50': regime_gap < REGIME_GAP_MAX,
        'MDD < 50%': metrics['max_dd'] < MDD_MAX,
        'Trades >= 30': metrics['n_trades'] >= MIN_TRADES,
    }
    return gates


def adversarial_checks(features, spy_regime, variant, actual_sharpe, trades_df):
    """Run adversarial checks on a passing variant."""
    print(f"\n  --- Adversarial Checks for Variant {variant} ---")
    results = {}

    # 1. Inverse test: flip buy/sell signals
    inv_trades = trades_df.copy()
    inv_trades['fwd_ret'] = -inv_trades['fwd_ret']
    inv_metrics = calc_metrics(inv_trades['fwd_ret'])
    results['inverse_sharpe'] = inv_metrics['sharpe']
    results['inverse_profitable'] = inv_metrics['sharpe'] > 0.3
    print(f"  Inverse Sharpe: {inv_metrics['sharpe']} ({'FAIL - inverse also works' if results['inverse_profitable'] else 'PASS'})")

    # 2. Random timing: beat 95th percentile
    all_rets = trades_df['fwd_ret'].values
    n = len(all_rets)
    random_sharpes = []
    for _ in range(1000):
        random_idx = np.random.choice(len(all_rets), size=n, replace=True)
        perm_rets = all_rets[random_idx]
        np.random.shuffle(perm_rets)
        s = perm_rets.mean() / perm_rets.std() * np.sqrt(252/HOLD_DAYS) if perm_rets.std() > 1e-10 else 0
        random_sharpes.append(s)
    pct95 = np.percentile(random_sharpes, 95)
    results['beats_random_95th'] = actual_sharpe > pct95
    results['random_95th_pct'] = round(pct95, 3)
    print(f"  Random 95th pct Sharpe: {pct95:.3f}, actual: {actual_sharpe:.3f} ({'PASS' if results['beats_random_95th'] else 'FAIL'})")

    # 3. Parameter sensitivity: test nearby thresholds
    param_sharpes = []
    if variant == 'A':
        for hi, lo in [(0.6, 0.4), (0.65, 0.35), (0.75, 0.25), (0.8, 0.2)]:
            # Re-run with different thresholds — simplified
            param_sharpes.append(hi)  # placeholder
        print(f"  Parameter sensitivity: testing nearby thresholds...")
        # Actually re-run with different parameters
        nearby_sharpes = []
        for hi_thresh in [0.6, 0.65, 0.75, 0.8]:
            lo_thresh = 1.0 - hi_thresh
            nearby_trades = []
            for ticker in features:
                if ticker == 'SPY':
                    continue
                df = features[ticker]
                buys = df[df['overnight_ratio'] > hi_thresh]
                sells = df[df['overnight_ratio'] < lo_thresh]
                for idx, row in buys.iterrows():
                    if not np.isnan(row.get('fwd_ret', np.nan)):
                        nearby_trades.append(row['fwd_ret'])
                for idx, row in sells.iterrows():
                    if not np.isnan(row.get('fwd_ret', np.nan)):
                        nearby_trades.append(-row['fwd_ret'])
            if len(nearby_trades) > 5:
                nt = np.array(nearby_trades)
                s = nt.mean() / nt.std() * np.sqrt(252/HOLD_DAYS) if nt.std() > 1e-10 else 0
                nearby_sharpes.append(s)
        if nearby_sharpes:
            results['param_sensitivity_range'] = (round(min(nearby_sharpes), 3), round(max(nearby_sharpes), 3))
            results['param_robust'] = min(nearby_sharpes) > 0.2
            print(f"  Nearby param Sharpes: {results['param_sensitivity_range']} ({'PASS' if results['param_robust'] else 'FAIL'})")

    elif variant in ['B', 'C', 'E']:
        nearby_sharpes = []
        for z_thresh in [1.0, 1.25, 1.75, 2.0]:
            nearby_trades = generate_signals_with_z(features, spy_regime, variant, z_thresh)
            if len(nearby_trades) > 5:
                m = calc_metrics(nearby_trades['fwd_ret'])
                nearby_sharpes.append(m['sharpe'])
        if nearby_sharpes:
            results['param_sensitivity_range'] = (round(min(nearby_sharpes), 3), round(max(nearby_sharpes), 3))
            results['param_robust'] = min(nearby_sharpes) > 0.2
            print(f"  Nearby param Sharpes: {results['param_sensitivity_range']} ({'PASS' if results['param_robust'] else 'FAIL'})")

    return results


def generate_signals_with_z(features, spy_regime, variant, z_thresh):
    """Re-generate signals with a different z-threshold for param sensitivity."""
    trades = []
    for ticker in features:
        if ticker == 'SPY':
            continue
        df = features[ticker].copy()
        df['spy_regime'] = spy_regime.reindex(df.index)
        df = df.dropna(subset=['fwd_ret'])

        if variant == 'B':
            col = 'overnight_ratio_z'
        elif variant == 'C':
            col = 'rel_overnight_ratio_z'
        elif variant == 'E':
            vw_roll_mean = df['vw_overnight_ratio'].rolling(ZSCORE_WINDOW).mean()
            vw_roll_std = df['vw_overnight_ratio'].rolling(ZSCORE_WINDOW).std()
            df['vw_overnight_ratio_z'] = np.where(
                vw_roll_std > 1e-8,
                (df['vw_overnight_ratio'] - vw_roll_mean) / vw_roll_std,
                np.nan
            )
            col = 'vw_overnight_ratio_z'
        else:
            continue

        if col not in df.columns:
            continue
        valid = df.dropna(subset=[col])
        buys = valid[valid[col] > z_thresh]
        sells = valid[valid[col] < -z_thresh]

        for idx, row in buys.iterrows():
            trades.append({
                'date': idx, 'ticker': ticker, 'direction': 'LONG',
                'fwd_ret': row['fwd_ret'], 'regime': row.get('spy_regime', 'unknown')
            })
        for idx, row in sells.iterrows():
            trades.append({
                'date': idx, 'ticker': ticker, 'direction': 'SHORT',
                'fwd_ret': -row['fwd_ret'], 'regime': row.get('spy_regime', 'unknown')
            })
    return pd.DataFrame(trades) if trades else pd.DataFrame(columns=['fwd_ret', 'regime'])


def run_backtest():
    """Main backtest execution."""
    print("=" * 80)
    print("OVERNIGHT vs INTRADAY RETURN DECOMPOSITION — SECTOR ETF SIGNAL BACKTEST")
    print("=" * 80)

    # 1. Fetch data
    data = fetch_data()
    if len(data) < 5:
        print("FATAL: Not enough tickers loaded.")
        return

    # 2. Compute features
    print("\nComputing overnight/intraday decomposition...")
    features = compute_returns(data)
    features = compute_spy_features(features)

    # 3. SPY regime
    spy_regime = classify_spy_regime(data)
    green_days = (spy_regime == 'green').sum()
    red_days = (spy_regime == 'red').sum()
    print(f"SPY regime: {green_days} green days, {red_days} red days")

    # 4. Descriptive stats
    print("\n--- Overnight Return Descriptive Stats ---")
    for ticker in ['XLK', 'XLE', 'XLF', 'XLU', 'SPY']:
        if ticker in features:
            df = features[ticker]
            print(f"  {ticker}: mean overnight={df['overnight_ret'].mean()*100:.4f}%, "
                  f"mean intraday={df['intraday_ret'].mean()*100:.4f}%, "
                  f"overnight ratio mean={df['overnight_ratio'].mean():.3f}")

    # 5. Test each variant
    variants = ['A', 'B', 'C', 'D', 'E', 'F']
    variant_names = {
        'A': 'Absolute Overnight Ratio (>0.7 buy, <0.3 sell)',
        'B': 'Z-Score vs 60d Rolling (z>1.5 buy, z<-1.5 sell)',
        'C': 'Relative to SPY (z-scored)',
        'D': 'Combined with RSI (z>1 + RSI<40 buy)',
        'E': 'Volume-Weighted Overnight Ratio (z-scored)',
        'F': 'Sector-Pair Divergence (XLK vs XLU)',
    }

    results_summary = {}
    passing_variants = []

    print("\n" + "=" * 80)
    print("VARIANT RESULTS")
    print("=" * 80)

    for variant in variants:
        print(f"\n{'─' * 60}")
        print(f"VARIANT {variant}: {variant_names[variant]}")
        print(f"{'─' * 60}")

        trades_df = generate_signals(features, spy_regime, variant)
        n_trades = len(trades_df)
        print(f"  Total trades: {n_trades}")

        if n_trades < 5:
            print(f"  SKIP: Too few trades (<5)")
            results_summary[variant] = {'status': 'SKIP', 'reason': 'too few trades', 'n_trades': n_trades}
            continue

        # Direction breakdown
        longs = trades_df[trades_df['direction'] == 'LONG']
        shorts = trades_df[trades_df['direction'] == 'SHORT']
        print(f"  Long trades: {len(longs)}, Short trades: {len(shorts)}")

        # Ticker breakdown
        ticker_counts = trades_df['ticker'].value_counts()
        print(f"  Tickers: {dict(ticker_counts)}")

        # Metrics
        metrics = calc_metrics(trades_df['fwd_ret'])
        print(f"  Avg return per trade: {metrics['avg_ret']:.3f}%")
        print(f"  Total return: {metrics['total_ret']:.1f}%")
        print(f"  Sharpe: {metrics['sharpe']:.3f}")
        print(f"  Win Rate: {metrics['win_rate']:.1%}")
        print(f"  Profit Factor: {metrics['profit_factor']:.3f}")
        print(f"  Max Drawdown: {metrics['max_dd']:.1%}")

        # Long vs Short breakdown
        if len(longs) > 5:
            long_m = calc_metrics(longs['fwd_ret'])
            print(f"  Long only — Sharpe: {long_m['sharpe']:.3f}, WR: {long_m['win_rate']:.1%}, Avg: {long_m['avg_ret']:.3f}%")
        if len(shorts) > 5:
            short_m = calc_metrics(shorts['fwd_ret'])
            print(f"  Short only — Sharpe: {short_m['sharpe']:.3f}, WR: {short_m['win_rate']:.1%}, Avg: {short_m['avg_ret']:.3f}%")

        # Regime analysis
        sharpe_green, sharpe_red, regime_gap = regime_analysis(trades_df)
        print(f"  Regime — Green Sharpe: {sharpe_green}, Red Sharpe: {sharpe_red}, Gap: {regime_gap}")

        # Permutation test
        p_value = permutation_test(metrics['sharpe'], trades_df['fwd_ret'])
        print(f"  Permutation p-value: {p_value:.4f}")

        # 5-gate check
        gates = five_gate_check(metrics, p_value, regime_gap)
        print(f"\n  5-GATE CHECK:")
        all_pass = True
        for gate_name, passed in gates.items():
            status = "PASS" if passed else "FAIL"
            print(f"    {gate_name}: {status}")
            if not passed:
                all_pass = False

        if all_pass:
            print(f"\n  >>> ALL 5 GATES PASSED <<<")
            passing_variants.append(variant)
        else:
            print(f"\n  REJECTED: failed {sum(1 for v in gates.values() if not v)} gate(s)")

        results_summary[variant] = {
            'status': 'PASS' if all_pass else 'FAIL',
            'metrics': metrics,
            'p_value': round(p_value, 4),
            'regime_gap': regime_gap,
            'sharpe_green': sharpe_green,
            'sharpe_red': sharpe_red,
            'gates': {k: bool(v) for k, v in gates.items()},
        }

    # 6. Adversarial checks for passing variants
    if passing_variants:
        print("\n" + "=" * 80)
        print("ADVERSARIAL CHECKS ON PASSING VARIANTS")
        print("=" * 80)
        for variant in passing_variants:
            trades_df = generate_signals(features, spy_regime, variant)
            metrics = calc_metrics(trades_df['fwd_ret'])
            adv = adversarial_checks(features, spy_regime, variant, metrics['sharpe'], trades_df)
            results_summary[variant]['adversarial'] = adv

    # 7. Year-by-year breakdown for best variant
    print("\n" + "=" * 80)
    print("YEAR-BY-YEAR BREAKDOWN (Best Variant)")
    print("=" * 80)

    # Pick best variant by Sharpe among those with >= 30 trades
    valid_variants = {v: r for v, r in results_summary.items()
                      if r.get('metrics', {}).get('n_trades', 0) >= 30}
    if valid_variants:
        best = max(valid_variants.keys(), key=lambda v: valid_variants[v]['metrics']['sharpe'])
        print(f"\nBest variant: {best} ({variant_names[best]})")
        trades_df = generate_signals(features, spy_regime, best)
        trades_df['year'] = trades_df['date'].apply(lambda x: x.year)
        for year in sorted(trades_df['year'].unique()):
            yr_trades = trades_df[trades_df['year'] == year]
            yr_m = calc_metrics(yr_trades['fwd_ret'])
            print(f"  {year}: trades={yr_m['n_trades']:3d}, Sharpe={yr_m['sharpe']:6.3f}, "
                  f"WR={yr_m['win_rate']:.1%}, PF={yr_m['profit_factor']:.2f}, Avg={yr_m['avg_ret']:.3f}%")

    # 8. Final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    for variant in variants:
        r = results_summary.get(variant, {})
        status = r.get('status', 'N/A')
        if status == 'SKIP':
            print(f"  Variant {variant}: SKIPPED ({r.get('reason', '')})")
        elif 'metrics' in r:
            m = r['metrics']
            print(f"  Variant {variant}: {status} | Sharpe={m['sharpe']:.3f} WR={m['win_rate']:.1%} "
                  f"PF={m['profit_factor']:.2f} MDD={m['max_dd']:.1%} N={m['n_trades']} "
                  f"p={r.get('p_value', 'N/A')} regime_gap={r.get('regime_gap', 'N/A')}")

    if passing_variants:
        print(f"\n  VARIANTS PASSING ALL 5 GATES: {passing_variants}")
        for v in passing_variants:
            adv = results_summary[v].get('adversarial', {})
            if adv:
                inv_fail = adv.get('inverse_profitable', False)
                rand_pass = adv.get('beats_random_95th', False)
                param_pass = adv.get('param_robust', None)
                print(f"  Variant {v} adversarial: inverse_also_works={inv_fail}, beats_random_95th={rand_pass}, param_robust={param_pass}")
                if inv_fail:
                    print(f"    WARNING: Inverse signal also profitable — likely NOT directional edge")
                if not rand_pass:
                    print(f"    WARNING: Does not beat random timing 95th percentile")
    else:
        print(f"\n  NO VARIANT PASSED ALL 5 GATES.")
        print(f"  The overnight/intraday return decomposition does not produce a reliable")
        print(f"  sector ETF options signal in this test framework.")

    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    if not passing_variants:
        print("  NEGATIVE RESULT: No variant of the overnight/intraday decomposition signal")
        print("  passes the 5-gate validation system. The hypothesis that overnight returns")
        print("  predict forward sector ETF returns is not supported by this backtest.")
        print("  This is consistent with the EMH — any historical overnight/intraday pattern")
        print("  has been arbitraged away or is too noisy for a 3-5 day options holding period.")
    else:
        all_adv_pass = True
        for v in passing_variants:
            adv = results_summary[v].get('adversarial', {})
            if adv.get('inverse_profitable', False) or not adv.get('beats_random_95th', True):
                all_adv_pass = False
        if all_adv_pass:
            print("  POSITIVE RESULT: At least one variant passes all gates AND adversarial checks.")
            print("  Further out-of-sample testing recommended before deployment.")
        else:
            print("  MIXED RESULT: Variant(s) passed 5 gates but FAILED adversarial checks.")
            print("  Likely spurious — do NOT deploy without further investigation.")

    return results_summary


if __name__ == '__main__':
    results = run_backtest()
