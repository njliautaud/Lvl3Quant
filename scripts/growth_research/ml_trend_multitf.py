#!/usr/bin/env python3
"""
Multi-Timeframe Trend Consensus with ML
=========================================
INSIGHT: Single MA crossover generates many whipsaws. Using THREE timeframes
(fast/medium/slow) and requiring consensus (2/3 or 3/3 agree) dramatically
reduces false signals. ML then filters the remaining consensus signals.

Strategy:
  1. Three timeframe trend signals per asset:
     - Fast:   10/30 MA crossover
     - Medium: 20/60 MA crossover
     - Slow:   50/200 MA crossover
  2. Consensus: only trade when 2/3 or 3/3 timeframes agree on direction
  3. GBM ML filter predicts which consensus signals lead to profitable continuations
  4. Risk parity sizing (each position targets same contribution to portfolio vol)

Assets: SPY, TLT, GLD, UUP, EEM, VNQ, HYG, XLE

Walk-forward: SLIDING 252d (HC #0). Fixed $100K, NO DCA (HC #713).
Full adversarial: permutation 100x, sub-period 4-block, outlier, R1 regime (HC #714).
"""

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier

warnings.filterwarnings('ignore')
np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
INITIAL_CAPITAL  = 100_000
TRAIN_WINDOW     = 252
N_PERMUTATIONS   = 100
REBAL_COST_BPS   = 10
ML_THRESHOLD     = 0.55
TARGET_VOL       = 0.10
CONSENSUS_MIN    = 2  # minimum timeframes agreeing (2 or 3 out of 3)

# Three timeframe MA pairs
TF_FAST   = (10, 30)
TF_MEDIUM = (20, 60)
TF_SLOW   = (50, 200)
TIMEFRAMES = {'fast': TF_FAST, 'medium': TF_MEDIUM, 'slow': TF_SLOW}

BASE   = Path("/home/jupiter/Lvl3Quant")
OUTPUT = BASE / "output" / "ml_trend_multitf"
OUTPUT.mkdir(parents=True, exist_ok=True)

UNIVERSE = {
    'SPY': 'US Equities',
    'TLT': 'Long Bonds',
    'GLD': 'Gold',
    'UUP': 'US Dollar',
    'EEM': 'Emerging Mkts',
    'VNQ': 'Real Estate',
    'HYG': 'High Yield',
    'XLE': 'Energy',
}


def download_data():
    print("=" * 80)
    print("STEP 1: DOWNLOADING DATA")
    print("=" * 80)

    all_tickers = list(UNIVERSE.keys()) + ['^VIX']

    cache = BASE / "data" / "cache" / "trend_multitf_data.parquet"
    if cache.exists() and (time.time() - cache.stat().st_mtime) < 3600:
        df = pd.read_parquet(cache)
        print(f"  Cached: {df.shape}, {df.index[0].date()} -> {df.index[-1].date()}")
        return df

    print("  Downloading...")
    raw = yf.download(all_tickers, start='2008-01-01', auto_adjust=True, progress=False)
    closes = raw['Close'] if isinstance(raw.columns, pd.MultiIndex) else raw
    if '^VIX' in closes.columns:
        closes = closes.rename(columns={'^VIX': 'VIX'})
    closes = closes.ffill().dropna(thresh=len(closes.columns) - 3)
    cache.parent.mkdir(parents=True, exist_ok=True)
    closes.to_parquet(cache)
    print(f"  Shape: {closes.shape}, {closes.index[0].date()} -> {closes.index[-1].date()}")
    return closes


def generate_multitf_signals(df):
    """Generate trend signals across THREE timeframes for each asset."""
    print("\n" + "=" * 80)
    print("STEP 2: MULTI-TIMEFRAME TREND SIGNALS")
    print("=" * 80)

    signals = {}
    for ticker in UNIVERSE:
        if ticker not in df.columns:
            continue
        price = df[ticker]

        tf_signals = {}
        for tf_name, (short, long) in TIMEFRAMES.items():
            ma_short = price.rolling(short).mean()
            ma_long = price.rolling(long).mean()
            trend = pd.Series(0.0, index=df.index)
            trend[ma_short > ma_long] = 1.0
            trend[ma_short < ma_long] = -1.0
            tf_signals[tf_name] = {
                'trend': trend,
                'ma_short': ma_short,
                'ma_long': ma_long,
            }

        # Consensus: count how many timeframes agree
        consensus = tf_signals['fast']['trend'] + tf_signals['medium']['trend'] + tf_signals['slow']['trend']
        # consensus_direction: +1 if majority up, -1 if majority down, 0 if split
        consensus_dir = pd.Series(0.0, index=df.index)
        consensus_dir[consensus >= CONSENSUS_MIN] = 1.0
        consensus_dir[consensus <= -CONSENSUS_MIN] = -1.0

        # Count agreement level (2 or 3)
        agreement_level = consensus.abs()  # 1, 2, or 3

        signals[ticker] = {
            'consensus_dir': consensus_dir,
            'agreement_level': agreement_level,
            'tf_signals': tf_signals,
            'price': price,
        }

        # Stats
        up = (consensus_dir == 1).mean() * 100
        dn = (consensus_dir == -1).mean() * 100
        full = (agreement_level == 3).mean() * 100
        print(f"  {ticker} ({UNIVERSE[ticker]:15s}): consensus up {up:.0f}%, down {dn:.0f}%, full agree {full:.0f}%")

    return signals


def build_ml_features(df, signals):
    """Build ML features incorporating multi-timeframe information."""
    print("\n" + "=" * 80)
    print("STEP 3: ML FEATURES (MULTI-TIMEFRAME)")
    print("=" * 80)

    # Need at least 200 days for slow MA
    min_start = 200 + 60

    rows = []
    for ticker, sig in signals.items():
        price = sig['price']
        consensus_dir = sig['consensus_dir']
        agreement_level = sig['agreement_level']
        tf_sigs = sig['tf_signals']
        ret = price.pct_change()

        for i in range(min_start, len(df)):
            idx = df.index[i]
            if consensus_dir.iloc[i] == 0:
                continue  # no consensus, skip

            feat = {}
            feat['ticker'] = ticker
            feat['date'] = idx
            feat['direction'] = consensus_dir.iloc[i]

            # Agreement level (2 or 3) - key feature
            feat['agreement_level'] = agreement_level.iloc[i]

            # Per-timeframe MA distances (normalized by price)
            for tf_name, tf_data in tf_sigs.items():
                ma_dist = (tf_data['ma_short'].iloc[i] - tf_data['ma_long'].iloc[i]) / (price.iloc[i] + 1e-8)
                feat[f'ma_dist_{tf_name}'] = ma_dist

            # Trend duration for each timeframe
            for tf_name, tf_data in tf_sigs.items():
                dur = 0
                trend = tf_data['trend']
                for j in range(i, max(i - 120, 0), -1):
                    if trend.iloc[j] == trend.iloc[i]:
                        dur += 1
                    else:
                        break
                feat[f'duration_{tf_name}'] = dur

            # Consensus duration (how long has consensus held)
            cons_dur = 0
            for j in range(i, max(i - 252, 0), -1):
                if consensus_dir.iloc[j] == consensus_dir.iloc[i]:
                    cons_dur += 1
                else:
                    break
            feat['consensus_duration'] = cons_dur

            # Momentum features
            feat['mom_5d'] = price.pct_change(5).iloc[i]
            feat['mom_20d'] = price.pct_change(20).iloc[i]
            feat['mom_60d'] = price.pct_change(60).iloc[i]

            # Volatility
            feat['vol_20d'] = ret.iloc[max(0,i-20):i].std() * np.sqrt(252)
            feat['vol_60d'] = ret.iloc[max(0,i-60):i].std() * np.sqrt(252)

            # Drawdown from peak
            peak = price.iloc[max(0,i-252):i+1].max()
            feat['drawdown'] = price.iloc[i] / peak - 1

            # Cross-asset consensus alignment
            same_dir = sum(1 for t2, s2 in signals.items()
                         if t2 != ticker and s2['consensus_dir'].iloc[i] == consensus_dir.iloc[i])
            feat['cross_align'] = same_dir / max(len(signals) - 1, 1)

            # Full agreement count across all assets
            full_agree_count = sum(1 for t2, s2 in signals.items()
                                  if s2['agreement_level'].iloc[i] == 3)
            feat['full_agree_assets'] = full_agree_count / len(signals)

            # VIX features
            if 'VIX' in df.columns:
                feat['vix'] = df['VIX'].iloc[i]
                feat['vix_ma20'] = df['VIX'].rolling(20).mean().iloc[i]
                feat['vix_percentile'] = (df['VIX'].iloc[max(0,i-252):i+1] <= df['VIX'].iloc[i]).mean()

            # Recent return skew and kurtosis
            if i > 20:
                feat['skew_20d'] = ret.iloc[i-20:i].skew()
                feat['kurt_20d'] = ret.iloc[i-20:i].kurt()

            # Timeframe divergence: are fast/slow disagreeing recently?
            fast_trend = tf_sigs['fast']['trend'].iloc[i]
            slow_trend = tf_sigs['slow']['trend'].iloc[i]
            feat['fast_slow_diverge'] = 1.0 if fast_trend != slow_trend else 0.0

            # Target: does trend continue profitably over next 20 days?
            if i + 20 < len(df):
                future_ret = (price.iloc[i + 20] / price.iloc[i] - 1) * consensus_dir.iloc[i]
                feat['target'] = 1 if future_ret > 0 else 0
                feat['future_ret'] = future_ret
            else:
                feat['target'] = np.nan
                feat['future_ret'] = np.nan

            rows.append(feat)

    features_df = pd.DataFrame(rows)
    print(f"  Total observations: {len(features_df)}")
    print(f"  Positive target rate: {features_df['target'].mean():.1%}")
    print(f"  Assets: {features_df['ticker'].nunique()}")
    print(f"  Full consensus (3/3) rate: {(features_df['agreement_level'] == 3).mean():.1%}")
    print(f"  Partial consensus (2/3) rate: {(features_df['agreement_level'] == 2).mean():.1%}")

    return features_df


def train_ml_filter(features_df):
    """Walk-forward sliding GBM to filter consensus signals."""
    print("\n" + "=" * 80)
    print("STEP 4: WALK-FORWARD ML FILTER (SLIDING 252d)")
    print("=" * 80)

    feature_cols = [c for c in features_df.columns
                   if c not in ['ticker', 'date', 'direction', 'target', 'future_ret']]

    features_df = features_df.sort_values('date').reset_index(drop=True)
    features_df['ml_prob'] = np.nan

    dates = sorted(features_df['date'].unique())
    print(f"  Unique dates: {len(dates)}")
    print(f"  Feature columns: {len(feature_cols)}")

    n_folds = 0
    batch_size = 5  # predict 5 days at a time for speed

    for i in range(TRAIN_WINDOW, len(dates), batch_size):
        batch_end = min(i + batch_size, len(dates))
        train_end_date = dates[i]
        train_start_date = dates[max(0, i - TRAIN_WINDOW)]

        train_mask = (features_df['date'] >= train_start_date) & (features_df['date'] < train_end_date)
        test_mask = (features_df['date'] >= dates[i]) & (features_df['date'] < dates[min(batch_end, len(dates)-1)])

        X_train = features_df.loc[train_mask, feature_cols].fillna(0)
        y_train = features_df.loc[train_mask, 'target']

        X_test = features_df.loc[test_mask, feature_cols].fillna(0)

        valid = ~y_train.isna()
        X_train = X_train[valid]
        y_train = y_train[valid]

        if len(X_train) < 50 or len(X_test) == 0:
            continue

        model = GradientBoostingClassifier(
            n_estimators=80,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        )
        model.fit(X_train, y_train)

        probs = model.predict_proba(X_test)[:, 1]
        features_df.loc[test_mask, 'ml_prob'] = probs
        n_folds += 1

        if n_folds % 100 == 0:
            print(f"    Fold {n_folds}, date: {dates[i].date()}")

    valid_preds = features_df.dropna(subset=['ml_prob', 'target'])
    if len(valid_preds) > 0:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(valid_preds['target'], valid_preds['ml_prob'])
        accuracy = ((valid_preds['ml_prob'] > 0.5) == valid_preds['target']).mean()
        print(f"\n  Walk-forward folds: {n_folds}")
        print(f"  OOS AUC: {auc:.3f}")
        print(f"  OOS Accuracy: {accuracy:.1%}")

        # Feature importance from last model
        if model:
            imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
            print(f"\n  Top 10 features:")
            for fname, fval in imp.head(10).items():
                print(f"    {fname:25s} {fval:.3f}")

    return features_df


def backtest_strategy(features_df, df, threshold=ML_THRESHOLD):
    """Backtest the ML-filtered multi-timeframe consensus strategy."""
    pred_df = features_df.dropna(subset=['ml_prob']).copy()
    dates = sorted(pred_df['date'].unique())

    portfolio_returns = []

    for date in dates:
        day_signals = pred_df[pred_df['date'] == date]
        high_conf = day_signals[day_signals['ml_prob'] > threshold]

        if len(high_conf) == 0:
            portfolio_returns.append({'date': date, 'return': 0.0, 'n_positions': 0})
            continue

        positions = []
        for _, row in high_conf.iterrows():
            ticker = row['ticker']
            direction = row['direction']
            vol = max(row.get('vol_20d', 0.2), 0.05)

            # Risk parity sizing
            weight = (TARGET_VOL / vol) / len(high_conf)
            weight = min(weight, 0.5)

            # Bonus weight for full consensus (3/3)
            if row.get('agreement_level', 2) == 3:
                weight *= 1.2

            if ticker in df.columns:
                ticker_dates = df.index
                date_loc = ticker_dates.get_loc(date) if date in ticker_dates else None
                if date_loc is not None and date_loc + 1 < len(ticker_dates):
                    next_ret = df[ticker].iloc[date_loc + 1] / df[ticker].iloc[date_loc] - 1
                    positions.append({
                        'ticker': ticker,
                        'direction': direction,
                        'weight': weight,
                        'return': next_ret * direction * weight,
                    })

        if positions:
            day_ret = sum(p['return'] for p in positions)
            turnover = sum(p['weight'] for p in positions)
            cost = turnover * REBAL_COST_BPS / 10000
            day_ret -= cost * 0.1

            portfolio_returns.append({
                'date': date,
                'return': day_ret,
                'n_positions': len(positions),
            })
        else:
            portfolio_returns.append({'date': date, 'return': 0.0, 'n_positions': 0})

    ret_df = pd.DataFrame(portfolio_returns).set_index('date')
    ret_series = ret_df['return']

    return ret_series, ret_df


def compute_metrics(returns, name="Strategy"):
    r = returns.dropna()
    if len(r) < 30:
        return {}

    mu = r.mean() * 252
    sigma = r.std() * np.sqrt(252)
    sharpe = mu / (sigma + 1e-8)

    downside = r[r < 0].std() * np.sqrt(252)
    sortino = mu / (downside + 1e-8)

    cumret = (1 + r).cumprod()
    total_return = cumret.iloc[-1] - 1
    years = len(r) / 252
    cagr = (cumret.iloc[-1]) ** (1 / years) - 1 if years > 0 else 0

    running_max = cumret.cummax()
    drawdown = cumret / running_max - 1
    max_dd = drawdown.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    gains = r[r > 0].sum()
    losses = abs(r[r < 0].sum())
    pf = gains / (losses + 1e-8)
    wr = (r > 0).mean()

    return {
        'name': name, 'sharpe': round(sharpe, 3), 'sortino': round(sortino, 3),
        'cagr': round(cagr * 100, 2), 'max_dd': round(max_dd * 100, 2),
        'calmar': round(calmar, 3), 'total_return': round(total_return * 100, 1),
        'profit_factor': round(pf, 3), 'win_rate': round(wr * 100, 1),
        'annual_vol': round(sigma * 100, 1), 'years': round(years, 1),
    }


def run_adversarial(ret_series, features_df, df):
    """Full adversarial validation: permutation, sub-period, outlier, R1 regime."""
    print("\n" + "=" * 80)
    print("ADVERSARIAL VALIDATION")
    print("=" * 80)

    results = {}
    real = compute_metrics(ret_series, "real")
    real_sharpe = real.get('sharpe', 0)

    # 1. PERMUTATION TEST (100 trials)
    print("\n  [1/4] Permutation test (100 trials)...")
    perm_sharpes = []
    for trial in range(N_PERMUTATIONS):
        perm_df = features_df.dropna(subset=['ml_prob']).copy()
        # Shuffle ml_prob across all observations (breaks signal-prediction link)
        perm_df['ml_prob'] = np.random.permutation(perm_df['ml_prob'].values)

        dates = sorted(perm_df['date'].unique())
        perm_rets = []

        for date in dates:
            day_signals = perm_df[perm_df['date'] == date]
            high_conf = day_signals[day_signals['ml_prob'] > ML_THRESHOLD]

            if len(high_conf) == 0:
                perm_rets.append(0.0)
                continue

            day_ret = 0.0
            for _, row in high_conf.iterrows():
                ticker = row['ticker']
                direction = row['direction']
                vol = max(row.get('vol_20d', 0.2), 0.05)
                weight = (TARGET_VOL / vol) / len(high_conf)
                weight = min(weight, 0.5)

                if ticker in df.columns:
                    ticker_dates = df.index
                    date_loc = ticker_dates.get_loc(date) if date in ticker_dates else None
                    if date_loc is not None and date_loc + 1 < len(ticker_dates):
                        next_ret = df[ticker].iloc[date_loc + 1] / df[ticker].iloc[date_loc] - 1
                        day_ret += next_ret * direction * weight

            perm_rets.append(day_ret)

        perm_ret_series = pd.Series(perm_rets, index=dates)
        m = compute_metrics(perm_ret_series, f"perm_{trial}")
        if m:
            perm_sharpes.append(m['sharpe'])

        if (trial + 1) % 20 == 0:
            print(f"    Trial {trial+1}/100...")

    perm_p = np.mean([s >= real_sharpe for s in perm_sharpes]) if perm_sharpes else 1.0
    perm_pass = perm_p < 0.05
    print(f"    Real Sharpe: {real_sharpe:.3f}")
    print(f"    Perm mean: {np.mean(perm_sharpes):.3f} +/- {np.std(perm_sharpes):.3f}")
    print(f"    p-value: {perm_p:.3f} -> {'PASS' if perm_pass else 'FAIL'}")
    results['permutation'] = {'p_value': round(perm_p, 4), 'pass': bool(perm_pass),
                              'real_sharpe': real_sharpe, 'perm_mean': round(float(np.mean(perm_sharpes)), 3)}

    # 2. SUB-PERIOD CONSISTENCY (4 blocks)
    print("\n  [2/4] Sub-period consistency (4 blocks)...")
    n = len(ret_series)
    block_size = n // 4
    block_sharpes = []
    block_details = []
    for b in range(4):
        start = b * block_size
        end = (b + 1) * block_size if b < 3 else n
        m = compute_metrics(ret_series.iloc[start:end], f"block_{b}")
        if m:
            block_sharpes.append(m['sharpe'])
            block_details.append(m)
            print(f"    Block {b+1}: Sharpe {m['sharpe']:.3f}, CAGR {m['cagr']:.1f}%, MaxDD {m['max_dd']:.1f}%")

    cv = np.std(block_sharpes) / abs(np.mean(block_sharpes)) if block_sharpes and np.mean(block_sharpes) != 0 else 999
    sub_pass = cv < 0.50
    print(f"    CV: {cv:.3f} -> {'PASS' if sub_pass else 'FAIL'}")
    results['sub_period'] = {'cv': round(cv, 3), 'pass': bool(sub_pass), 'block_sharpes': block_sharpes}

    # 3. OUTLIER ROBUSTNESS
    print("\n  [3/4] Outlier robustness (trim 5th/95th percentile)...")
    p95 = ret_series.quantile(0.95)
    p05 = ret_series.quantile(0.05)
    trimmed = ret_series[(ret_series > p05) & (ret_series < p95)]
    m_full = compute_metrics(ret_series)
    m_trim = compute_metrics(trimmed)
    if m_full.get('sharpe', 0) != 0:
        deg = (m_full['sharpe'] - m_trim['sharpe']) / abs(m_full['sharpe'])
    else:
        deg = 999
    outlier_pass = abs(deg) < 0.30
    print(f"    Full Sharpe: {m_full.get('sharpe', 0):.3f}, Trimmed Sharpe: {m_trim.get('sharpe', 0):.3f}")
    print(f"    Degradation: {deg:.1%} -> {'PASS' if outlier_pass else 'FAIL'}")
    results['outlier'] = {'degradation': round(deg, 3), 'pass': bool(outlier_pass),
                          'full_sharpe': m_full.get('sharpe', 0), 'trimmed_sharpe': m_trim.get('sharpe', 0)}

    # 4. R1 REGIME TEST (green/red days based on SPY close-to-close)
    print("\n  [4/4] R1 regime test (SPY 20d return > 0 = green)...")
    spy_ret = df['SPY'].pct_change()
    spy_20d = spy_ret.rolling(20).sum()

    green_mask = pd.Series(ret_series.index.map(lambda x: spy_20d.loc[x] > 0 if x in spy_20d.index else True).values, index=ret_series.index)
    m_green = compute_metrics(ret_series[green_mask], "green")
    m_red = compute_metrics(ret_series[~green_mask], "red")

    if m_green and m_red:
        s_g, s_r = m_green['sharpe'], m_red['sharpe']
        gap = abs(s_g - s_r) / max(abs(s_g), abs(s_r), 0.01)
        r1_pass = gap < 0.50
        print(f"    Green days Sharpe: {s_g:.3f} (n={green_mask.sum()})")
        print(f"    Red days Sharpe: {s_r:.3f} (n={(~green_mask).sum()})")
        print(f"    Gap ratio: {gap:.3f} -> {'PASS' if r1_pass else 'FAIL'}")
        results['r1_regime'] = {'green_sharpe': s_g, 'red_sharpe': s_r,
                                'gap': round(gap, 3), 'pass': bool(r1_pass)}
    else:
        r1_pass = False
        results['r1_regime'] = {'pass': False, 'note': 'insufficient data in one regime'}

    gates = sum([results.get(k, {}).get('pass', False)
                for k in ['permutation', 'sub_period', 'outlier', 'r1_regime']])
    results['summary'] = {'gates_passed': gates, 'total': 4,
                          'verdict': 'PASS' if gates >= 3 else 'FAIL'}
    print(f"\n  ADVERSARIAL SUMMARY: {gates}/4 gates passed -> {results['summary']['verdict']}")

    return results


def main():
    t0 = time.time()
    print("=" * 80)
    print("MULTI-TIMEFRAME TREND CONSENSUS WITH ML")
    print(f"  Timeframes: fast={TF_FAST}, medium={TF_MEDIUM}, slow={TF_SLOW}")
    print(f"  Consensus minimum: {CONSENSUS_MIN}/3 timeframes must agree")
    print(f"  ML threshold: {ML_THRESHOLD}")
    print(f"  Walk-forward: sliding {TRAIN_WINDOW}d")
    print("=" * 80)

    # Step 1: Data
    df = download_data()

    # Step 2: Multi-timeframe signals
    signals = generate_multitf_signals(df)

    # Step 3: ML features
    features_df = build_ml_features(df, signals)

    # Step 4: Walk-forward ML filter
    features_df = train_ml_filter(features_df)

    # Step 5: Backtest
    print("\n" + "=" * 80)
    print("STEP 5: BACKTEST")
    print("=" * 80)
    ret_series, ret_df = backtest_strategy(features_df, df)
    print(f"  Trading days: {len(ret_series)}")
    print(f"  Mean positions/day: {ret_df['n_positions'].mean():.1f}")
    print(f"  Days with positions: {(ret_df['n_positions'] > 0).sum()} ({(ret_df['n_positions'] > 0).mean():.1%})")

    # Compute metrics
    m = compute_metrics(ret_series, "ML Multi-TF Consensus")

    # Benchmarks
    idx = ret_series.index
    spy_ret = df['SPY'].pct_change().reindex(idx).fillna(0)
    bm_spy = compute_metrics(spy_ret, "SPY B&H")

    # Simple consensus (no ML filter)
    all_signals_df = features_df.copy()
    all_signals_df['ml_prob'] = 1.0
    simple_ret, _ = backtest_strategy(all_signals_df, df)
    bm_simple = compute_metrics(simple_ret, "Consensus (no ML)")

    # Full consensus only (3/3 agree, no ML)
    full_only_df = features_df[features_df['agreement_level'] == 3].copy()
    full_only_df['ml_prob'] = 1.0
    full_ret, _ = backtest_strategy(full_only_df, df)
    bm_full = compute_metrics(full_ret, "Full Consensus 3/3 (no ML)")

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"\n  {'Strategy':<30} {'Sharpe':>8} {'Sortino':>8} {'CAGR':>8} {'MaxDD':>8} {'Calmar':>8} {'WR':>6} {'PF':>6}")
    print(f"  {'-'*90}")
    for met in [m, bm_simple, bm_full, bm_spy]:
        if met:
            print(f"  {met['name']:<30} {met['sharpe']:>8.3f} {met['sortino']:>8.3f} "
                  f"{met['cagr']:>7.1f}% {met['max_dd']:>7.1f}% {met['calmar']:>8.3f} "
                  f"{met['win_rate']:>5.1f}% {met['profit_factor']:>5.2f}")

    # Adversarial validation
    adv = run_adversarial(ret_series, features_df, df)

    # Save results
    output = {
        'strategy': 'ML Multi-Timeframe Trend Consensus',
        'description': 'Three timeframes (fast 10/30, medium 20/60, slow 50/200) with 2/3 or 3/3 consensus + GBM ML filter',
        'metrics': {
            'portfolio': m,
            'spy_bh': bm_spy,
            'consensus_no_ml': bm_simple,
            'full_consensus_no_ml': bm_full,
        },
        'adversarial': adv,
        'parameters': {
            'tf_fast': list(TF_FAST),
            'tf_medium': list(TF_MEDIUM),
            'tf_slow': list(TF_SLOW),
            'consensus_min': CONSENSUS_MIN,
            'ml_threshold': ML_THRESHOLD,
            'target_vol': TARGET_VOL,
            'train_window': TRAIN_WINDOW,
            'universe': list(UNIVERSE.keys()),
            'initial_capital': INITIAL_CAPITAL,
        },
        'runtime_seconds': round(time.time() - t0, 1),
    }

    with open(OUTPUT / 'results.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)

    # Equity curve plot
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    cum = (1 + ret_series).cumprod() * INITIAL_CAPITAL
    spy_cum = (1 + spy_ret).cumprod() * INITIAL_CAPITAL
    simple_cum = (1 + simple_ret).cumprod() * INITIAL_CAPITAL

    axes[0].plot(cum.index, cum.values, label='ML Multi-TF Consensus', linewidth=2)
    axes[0].plot(simple_cum.index, simple_cum.values, label='Consensus (no ML)', alpha=0.7)
    axes[0].plot(spy_cum.index, spy_cum.values, label='SPY B&H', alpha=0.7)
    axes[0].set_title('Equity Curves ($100K, No DCA, No Leverage)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylabel('Portfolio Value ($)')

    dd = cum / cum.cummax() - 1
    axes[1].fill_between(dd.index, dd.values, 0, alpha=0.5, color='red')
    axes[1].set_title('Drawdown')
    axes[1].set_ylabel('Drawdown %')
    axes[1].grid(True, alpha=0.3)

    # Rolling Sharpe
    rolling_sharpe = ret_series.rolling(252).apply(lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0)
    axes[2].plot(rolling_sharpe.index, rolling_sharpe.values, label='Rolling 1Y Sharpe', color='purple')
    axes[2].axhline(y=0, color='gray', linestyle='--')
    axes[2].set_title('Rolling 1-Year Sharpe Ratio')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT / 'equity_curve.png', dpi=150)
    plt.close()

    elapsed = time.time() - t0
    print(f"\nCOMPLETE in {elapsed:.0f}s. Output saved to {OUTPUT}")
    print(f"\nFinal verdict: {adv['summary']['verdict']} ({adv['summary']['gates_passed']}/4 adversarial gates)")


if __name__ == '__main__':
    main()
