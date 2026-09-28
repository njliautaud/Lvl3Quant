#!/usr/bin/env python3
"""
ML-Enhanced Trend Following (CTA-Style)
=========================================
INSIGHT: Pure trend following works across asset classes over decades.
The problem is whipsaws (false breakouts) that kill returns. ML can
learn which breakouts are real vs noise from cross-asset context.

Strategy:
  1. Generate trend signals across 8 uncorrelated assets using dual-MA crossover
  2. Use ML (GBM) to filter signals: predict P(trend continuation | cross-asset features)
  3. Only take positions where ML confidence > threshold
  4. Risk parity sizing (each position targets same contribution to portfolio vol)

Assets: SPY, TLT, GLD, UUP, USO, EEM, VNQ, HYG
  - Spans equities, bonds, commodities, currencies, EM, real estate, credit

ML Features per signal:
  - Trend strength (MA distance from price)
  - Trend duration (days since crossover)
  - Cross-asset trend alignment (how many other assets trending same direction)
  - Volatility regime (VIX level, realized vol)
  - Recent drawdown, momentum, volume

Walk-forward: SLIDING 252d (HC #0). Fixed $100K, NO DCA (HC #713).
Full adversarial: permutation 100x, sub-period 4-block, outlier, R1 regime (HC #705).
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
REBAL_COST_BPS   = 10           # trend following has more turnover
MA_SHORT         = 20
MA_LONG          = 100
ML_THRESHOLD     = 0.55         # ML confidence threshold for taking position
TARGET_VOL       = 0.10         # per-position vol target

BASE   = Path("/home/jupiter/Lvl3Quant")
OUTPUT = BASE / "output" / "ml_trend_following"
OUTPUT.mkdir(parents=True, exist_ok=True)

# Universe: broadly uncorrelated assets
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

    cache = BASE / "data" / "cache" / "trend_following_data.parquet"
    if cache.exists() and (time.time() - cache.stat().st_mtime) < 3600:
        df = pd.read_parquet(cache)
        print(f"  Cached: {df.shape}, {df.index[0].date()} → {df.index[-1].date()}")
        return df

    print("  Downloading...")
    raw = yf.download(all_tickers, start='2008-01-01', auto_adjust=True, progress=False)
    closes = raw['Close'] if isinstance(raw.columns, pd.MultiIndex) else raw
    if '^VIX' in closes.columns:
        closes = closes.rename(columns={'^VIX': 'VIX'})
    closes = closes.ffill().dropna(thresh=len(closes.columns) - 3)
    cache.parent.mkdir(parents=True, exist_ok=True)
    closes.to_parquet(cache)
    print(f"  Shape: {closes.shape}, {closes.index[0].date()} → {closes.index[-1].date()}")
    return closes


def generate_trend_signals(df):
    """Generate trend signals for each asset using dual-MA crossover."""
    print("\n" + "=" * 80)
    print("STEP 2: TREND SIGNALS")
    print("=" * 80)

    signals = {}
    for ticker in UNIVERSE:
        if ticker not in df.columns:
            continue
        price = df[ticker]
        ma_short = price.rolling(MA_SHORT).mean()
        ma_long = price.rolling(MA_LONG).mean()

        # Signal: +1 when short MA > long MA (uptrend), -1 when below (downtrend)
        trend = pd.Series(0.0, index=df.index)
        trend[ma_short > ma_long] = 1.0
        trend[ma_short < ma_long] = -1.0

        signals[ticker] = {
            'trend': trend,
            'ma_short': ma_short,
            'ma_long': ma_long,
            'price': price,
        }

        # Stats
        up_pct = (trend == 1).mean() * 100
        dn_pct = (trend == -1).mean() * 100
        print(f"  {ticker} ({UNIVERSE[ticker]:15s}): up {up_pct:.0f}%, down {dn_pct:.0f}%")

    return signals


def build_ml_features(df, signals):
    """Build ML features for each asset-day to predict trend continuation."""
    print("\n" + "=" * 80)
    print("STEP 3: ML FEATURES")
    print("=" * 80)

    rows = []

    for ticker, sig in signals.items():
        price = sig['price']
        trend = sig['trend']
        ma_s = sig['ma_short']
        ma_l = sig['ma_long']
        ret = price.pct_change()

        for i in range(MA_LONG + 60, len(df)):
            idx = df.index[i]
            if trend.iloc[i] == 0:
                continue

            # Features
            feat = {}
            feat['ticker'] = ticker
            feat['date'] = idx
            feat['direction'] = trend.iloc[i]

            # Trend strength: distance between MAs as % of price
            feat['ma_dist'] = (ma_s.iloc[i] - ma_l.iloc[i]) / (price.iloc[i] + 1e-8)

            # Trend duration: consecutive days in same direction
            dur = 0
            for j in range(i, max(i - 252, 0), -1):
                if trend.iloc[j] == trend.iloc[i]:
                    dur += 1
                else:
                    break
            feat['trend_duration'] = dur

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

            # Cross-asset alignment: how many others trending same direction
            same_dir = sum(1 for t2, s2 in signals.items()
                         if t2 != ticker and s2['trend'].iloc[i] == trend.iloc[i])
            feat['cross_align'] = same_dir / max(len(signals) - 1, 1)

            # VIX features
            if 'VIX' in df.columns:
                feat['vix'] = df['VIX'].iloc[i]
                feat['vix_ma20'] = df['VIX'].rolling(20).mean().iloc[i]

            # Skew and kurtosis
            if i > 20:
                feat['skew_20d'] = ret.iloc[i-20:i].skew()
                feat['kurt_20d'] = ret.iloc[i-20:i].kurt()

            # Target: does trend continue profitably over next 20 days?
            if i + 20 < len(df):
                future_ret = (price.iloc[i + 20] / price.iloc[i] - 1) * trend.iloc[i]
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

    return features_df


def train_ml_filter(features_df):
    """Walk-forward GBM to filter trend signals."""
    print("\n" + "=" * 80)
    print("STEP 4: WALK-FORWARD ML FILTER")
    print("=" * 80)

    feature_cols = [c for c in features_df.columns
                   if c not in ['ticker', 'date', 'direction', 'target', 'future_ret']]

    # Sort by date
    features_df = features_df.sort_values('date').reset_index(drop=True)
    features_df['ml_prob'] = np.nan

    dates = sorted(features_df['date'].unique())

    n_folds = 0
    for i in range(TRAIN_WINDOW, len(dates)):
        train_end = dates[i]
        train_start = dates[max(0, i - TRAIN_WINDOW)]
        # HC #718: label gap = 20 days (target horizon) to prevent look-ahead
        train_end_safe = dates[max(0, i - 20)]

        train_mask = (features_df['date'] >= train_start) & (features_df['date'] < train_end_safe)
        test_mask = features_df['date'] == train_end

        X_train = features_df.loc[train_mask, feature_cols].fillna(0)
        y_train = features_df.loc[train_mask, 'target']

        X_test = features_df.loc[test_mask, feature_cols].fillna(0)

        # Remove NaN targets
        valid = ~y_train.isna()
        X_train = X_train[valid]
        y_train = y_train[valid]

        if len(X_train) < 50 or len(X_test) == 0:
            continue

        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        )
        model.fit(X_train, y_train)

        probs = model.predict_proba(X_test)[:, 1]
        features_df.loc[test_mask, 'ml_prob'] = probs
        n_folds += 1

    valid_preds = features_df.dropna(subset=['ml_prob', 'target'])
    if len(valid_preds) > 0:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(valid_preds['target'], valid_preds['ml_prob'])
        accuracy = ((valid_preds['ml_prob'] > 0.5) == valid_preds['target']).mean()
        print(f"  Walk-forward folds: {n_folds}")
        print(f"  OOS AUC: {auc:.3f}")
        print(f"  OOS Accuracy: {accuracy:.1%}")

    return features_df


def backtest_strategy(features_df, df):
    """Backtest the ML-filtered trend following strategy."""
    print("\n" + "=" * 80)
    print("STEP 5: BACKTEST")
    print("=" * 80)

    # Filter to dates with ML predictions
    pred_df = features_df.dropna(subset=['ml_prob']).copy()

    # Get unique dates
    dates = sorted(pred_df['date'].unique())

    portfolio_returns = []
    daily_positions = []
    # HC #718 R3: Transaction costs — 5 bps per leg on turnover
    COST_BPS_PER_LEG = 5
    prev_position_keys = set()  # track (ticker, direction) for turnover

    for date in dates:
        day_signals = pred_df[pred_df['date'] == date]

        # Filter by ML confidence
        high_conf = day_signals[day_signals['ml_prob'] > ML_THRESHOLD]

        if len(high_conf) == 0:
            # HC #718 R3: cost to exit all positions when going flat
            exit_cost = 0.0
            if prev_position_keys:
                exit_cost = COST_BPS_PER_LEG / 10000  # sell cost
            prev_position_keys = set()
            portfolio_returns.append({'date': date, 'return': -exit_cost, 'n_positions': 0})
            continue

        # Risk parity sizing: each position gets equal risk budget
        positions = []
        for _, row in high_conf.iterrows():
            ticker = row['ticker']
            direction = row['direction']
            vol = max(row['vol_20d'], 0.05)

            # Position size = target_vol / asset_vol / n_positions
            weight = (TARGET_VOL / vol) / len(high_conf)
            weight = min(weight, 0.5)  # cap at 50% per position

            # Get actual next-day return
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
            # HC #718 R3: transaction costs — 5 bps per leg on changed positions
            current_keys = set((p['ticker'], p['direction']) for p in positions)
            if prev_position_keys:
                changed = len(prev_position_keys.symmetric_difference(current_keys))
                total = max(len(prev_position_keys), len(current_keys))
                turnover_frac = changed / total if total > 0 else 0
                day_ret -= turnover_frac * 2 * COST_BPS_PER_LEG / 10000  # sell old + buy new
            else:
                day_ret -= COST_BPS_PER_LEG / 10000  # initial buy
            prev_position_keys = current_keys

            portfolio_returns.append({
                'date': date,
                'return': day_ret,
                'n_positions': len(positions),
            })
        else:
            prev_position_keys = set()
            portfolio_returns.append({'date': date, 'return': 0.0, 'n_positions': 0})

    ret_df = pd.DataFrame(portfolio_returns).set_index('date')
    ret_series = ret_df['return']

    print(f"  Trading days: {len(ret_series)}")
    print(f"  Mean positions/day: {ret_df['n_positions'].mean():.1f}")
    print(f"  Days with positions: {(ret_df['n_positions'] > 0).sum()} ({(ret_df['n_positions'] > 0).mean():.1%})")

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
        'cagr': round(cagr * 100, 1), 'max_dd': round(max_dd * 100, 1),
        'calmar': round(calmar, 3), 'total_return': round(total_return * 100, 1),
        'profit_factor': round(pf, 3), 'win_rate': round(wr * 100, 1),
        'annual_vol': round(sigma * 100, 1), 'years': round(years, 1),
    }


def run_adversarial(ret_series, features_df, df):
    """Full adversarial validation."""
    print("\n" + "=" * 80)
    print("ADVERSARIAL VALIDATION")
    print("=" * 80)

    results = {}
    real = compute_metrics(ret_series, "real")
    real_sharpe = real.get('sharpe', 0)

    # 1. PERMUTATION TEST
    print("\n  [1/4] Permutation test...")
    perm_sharpes = []
    for trial in range(N_PERMUTATIONS):
        # HC #718 R2: Shuffle ML probabilities WITHIN each date (cross-sectional permutation)
        # Tests whether ML's stock selection is better than random selection
        perm_df = features_df.copy()
        for dt in perm_df['date'].unique():
            mask = perm_df['date'] == dt
            perm_df.loc[mask, 'ml_prob'] = np.random.permutation(
                perm_df.loc[mask, 'ml_prob'].values)

        # Quick re-backtest with shuffled probs
        pred_df = perm_df.dropna(subset=['ml_prob']).copy()
        dates = sorted(pred_df['date'].unique())
        perm_rets = []

        for date in dates:
            day_signals = pred_df[pred_df['date'] == date]
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

    perm_p = np.mean([s >= real_sharpe for s in perm_sharpes]) if perm_sharpes else 1.0
    perm_pass = perm_p < 0.05
    print(f"    Real Sharpe: {real_sharpe:.3f}")
    print(f"    Perm mean: {np.mean(perm_sharpes):.3f} ± {np.std(perm_sharpes):.3f}")
    print(f"    p-value: {perm_p:.3f} → {'PASS' if perm_pass else 'FAIL'}")
    results['permutation'] = {'p_value': perm_p, 'pass': perm_pass}

    # 2. SUB-PERIOD
    print("\n  [2/4] Sub-period consistency...")
    n = len(ret_series)
    block_size = n // 4
    block_sharpes = []
    for b in range(4):
        start = b * block_size
        end = (b + 1) * block_size if b < 3 else n
        m = compute_metrics(ret_series.iloc[start:end], f"block_{b}")
        if m:
            block_sharpes.append(m['sharpe'])
            print(f"    Block {b+1}: Sharpe {m['sharpe']:.3f}")

    cv = np.std(block_sharpes) / abs(np.mean(block_sharpes)) if block_sharpes and np.mean(block_sharpes) != 0 else 999
    sub_pass = cv < 0.50
    print(f"    CV: {cv:.3f} → {'PASS' if sub_pass else 'FAIL'}")
    results['sub_period'] = {'cv': round(cv, 3), 'pass': sub_pass}

    # 3. OUTLIER
    print("\n  [3/4] Outlier robustness...")
    p95 = ret_series.quantile(0.95)
    p05 = ret_series.quantile(0.05)
    trimmed = ret_series[(ret_series > p05) & (ret_series < p95)]
    m_full = compute_metrics(ret_series)
    m_trim = compute_metrics(trimmed)
    deg = (m_full['sharpe'] - m_trim['sharpe']) / abs(m_full['sharpe']) if m_full.get('sharpe', 0) != 0 else 999
    outlier_pass = abs(deg) < 0.30
    print(f"    Degradation: {deg:.1%} → {'PASS' if outlier_pass else 'FAIL'}")
    results['outlier'] = {'degradation': round(deg, 3), 'pass': outlier_pass}

    # 4. R1 REGIME
    print("\n  [4/4] R1 regime test...")
    spy_ret = df['SPY'].pct_change()
    spy_20d = spy_ret.rolling(20).sum()

    green_mask = ret_series.index.map(lambda x: spy_20d.loc[x] > 0 if x in spy_20d.index else True)
    m_green = compute_metrics(ret_series[green_mask], "green")
    m_red = compute_metrics(ret_series[~green_mask], "red")

    if m_green and m_red:
        s_g, s_r = m_green['sharpe'], m_red['sharpe']
        gap = abs(s_g - s_r) / max(abs(s_g), abs(s_r), 0.01)
        r1_pass = gap < 0.50
        print(f"    Green: {s_g:.3f}, Red: {s_r:.3f}, Gap: {gap:.3f} → {'PASS' if r1_pass else 'FAIL'}")
        results['r1_regime'] = {'gap': round(gap, 3), 'pass': r1_pass}
    else:
        r1_pass = False
        results['r1_regime'] = {'pass': False}

    gates = sum([results.get(k, {}).get('pass', False) for k in ['permutation', 'sub_period', 'outlier', 'r1_regime']])
    results['summary'] = {'gates_passed': gates, 'total': 4, 'verdict': 'PASS' if gates >= 3 else 'FAIL'}
    print(f"\n  SUMMARY: {gates}/4 gates → {results['summary']['verdict']}")

    return results


def main():
    t0 = time.time()
    print("=" * 80)
    print("ML-ENHANCED TREND FOLLOWING (CTA-STYLE)")
    print("=" * 80)

    df = download_data()
    signals = generate_trend_signals(df)
    features_df = build_ml_features(df, signals)
    features_df = train_ml_filter(features_df)
    ret_series, ret_df = backtest_strategy(features_df, df)

    # Compute metrics
    m = compute_metrics(ret_series, "ML Trend Following")

    # Benchmarks
    idx = ret_series.index
    spy_ret = df['SPY'].pct_change().reindex(idx).fillna(0)
    bm_spy = compute_metrics(spy_ret, "SPY B&H")

    # Simple trend following (no ML filter)
    # Re-backtest without ML filter (all signals taken)
    all_signals_df = features_df.copy()
    all_signals_df['ml_prob'] = 1.0  # force all signals through
    simple_ret, _ = backtest_strategy(all_signals_df, df)
    bm_simple = compute_metrics(simple_ret, "Simple Trend (no ML)")

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"\n  {'Strategy':<25} {'Sharpe':>8} {'Sortino':>8} {'CAGR':>8} {'MaxDD':>8} {'Calmar':>8}")
    print(f"  {'-'*65}")
    for met in [m, bm_simple, bm_spy]:
        if met:
            print(f"  {met['name']:<25} {met['sharpe']:>8.3f} {met['sortino']:>8.3f} "
                  f"{met['cagr']:>7.1f}% {met['max_dd']:>7.1f}% {met['calmar']:>8.3f}")

    # Adversarial
    adv = run_adversarial(ret_series, features_df, df)

    # Save
    output = {
        'strategy': 'ML Trend Following',
        'metrics': {'portfolio': m, 'spy': bm_spy, 'simple_trend': bm_simple},
        'adversarial': adv,
        'parameters': {
            'ma_short': MA_SHORT, 'ma_long': MA_LONG,
            'ml_threshold': ML_THRESHOLD, 'target_vol': TARGET_VOL,
            'train_window': TRAIN_WINDOW, 'universe': list(UNIVERSE.keys()),
        },
        'runtime_seconds': round(time.time() - t0, 1),
    }

    with open(OUTPUT / 'results.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)

    # Save daily returns for combination analysis
    ret_series.to_csv(OUTPUT / 'daily_returns.csv', header=['return'])

    # Equity curve
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    cum = (1 + ret_series).cumprod() * INITIAL_CAPITAL
    spy_cum = (1 + spy_ret).cumprod() * INITIAL_CAPITAL
    simple_cum = (1 + simple_ret).cumprod() * INITIAL_CAPITAL

    axes[0].plot(cum.index, cum.values, label='ML Trend Following', linewidth=2)
    axes[0].plot(simple_cum.index, simple_cum.values, label='Simple Trend (no ML)', alpha=0.7)
    axes[0].plot(spy_cum.index, spy_cum.values, label='SPY B&H', alpha=0.7)
    axes[0].set_title('Equity Curves ($100K, No DCA)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    dd = cum / cum.cummax() - 1
    axes[1].fill_between(dd.index, dd.values, 0, alpha=0.5, color='red')
    axes[1].set_title('Drawdown')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT / 'equity_curve.png', dpi=150)
    plt.close()

    elapsed = time.time() - t0
    print(f"\nCOMPLETE in {elapsed:.0f}s. Output: {OUTPUT}")


if __name__ == '__main__':
    main()
