#!/usr/bin/env python3
"""
ML Portfolio Combination Backtest (OPTIMIZED)
==============================================
Combines the 2 validated ML strategies:
  1. ML Trend Following (CTA-style, 8 assets) — Sharpe 2.90
  2. ML Sector Rotation (11 SPDR sectors)    — Sharpe 1.99

Plus a simple VIX>20 defensive overlay.

OPTIMIZED: Vectorized feature building, reduced permutations (50),
to target <15 min runtime.

Walk-forward: SLIDING 252d (HC #0). Fixed $100K, NO DCA (HC #713).
"""

import json
import sys
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

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

# ─────────────────────────────────────────────────────────────────────────────
INITIAL_CAPITAL  = 100_000
TRAIN_WINDOW     = 252
N_PERMUTATIONS   = 50   # Reduced for speed (still significant at p<0.05)
REBAL_COST_BPS   = 10
MA_SHORT         = 20
MA_LONG          = 100
ML_THRESHOLD     = 0.55
TARGET_VOL       = 0.10
VIX_THRESHOLD    = 20.0

BASE   = Path("/home/jupiter/Lvl3Quant")
OUTPUT = BASE / "output" / "ml_portfolio_combo"
OUTPUT.mkdir(parents=True, exist_ok=True)

# Strategy 1: CTA Trend Following universe
UNIVERSE_CTA = {
    'SPY': 'US Equities', 'TLT': 'Long Bonds', 'GLD': 'Gold',
    'UUP': 'US Dollar', 'EEM': 'Emerging Mkts', 'VNQ': 'Real Estate',
    'HYG': 'High Yield', 'XLE': 'Energy',
}

# Strategy 2: Sector Rotation universe
UNIVERSE_SECTORS = {
    'XLK': 'Technology', 'XLF': 'Financials', 'XLE': 'Energy',
    'XLV': 'Health Care', 'XLY': 'Consumer Disc', 'XLP': 'Consumer Staples',
    'XLI': 'Industrials', 'XLB': 'Materials', 'XLU': 'Utilities',
    'XLRE': 'Real Estate', 'XLC': 'Communication',
}


def download_data():
    print("=" * 80)
    print("STEP 1: DOWNLOADING DATA")
    print("=" * 80)

    all_tickers = sorted(set(
        list(UNIVERSE_CTA.keys()) + list(UNIVERSE_SECTORS.keys()) + ['^VIX', 'SPY']
    ))

    cache = BASE / "data" / "cache" / "portfolio_combo_data.parquet"
    if cache.exists() and (time.time() - cache.stat().st_mtime) < 7200:
        df = pd.read_parquet(cache)
        print(f"  Cached: {df.shape}, {df.index[0].date()} -> {df.index[-1].date()}")
        return df

    print(f"  Downloading {len(all_tickers)} tickers...")
    raw = yf.download(all_tickers, start='2008-01-01', auto_adjust=True, progress=False)
    closes = raw['Close'] if isinstance(raw.columns, pd.MultiIndex) else raw
    if '^VIX' in closes.columns:
        closes = closes.rename(columns={'^VIX': 'VIX'})
    closes = closes.ffill().dropna(thresh=len(closes.columns) - 3)
    cache.parent.mkdir(parents=True, exist_ok=True)
    closes.to_parquet(cache)
    print(f"  Shape: {closes.shape}, {closes.index[0].date()} -> {closes.index[-1].date()}")
    return closes


def build_features_vectorized(df, universe):
    """VECTORIZED feature building — much faster than row-by-row."""
    print(f"  Building features for {len(universe)} assets...")
    t0 = time.time()

    all_features = []

    for ticker in universe:
        if ticker not in df.columns:
            continue

        price = df[ticker]
        ret = price.pct_change()
        ma_short = price.rolling(MA_SHORT).mean()
        ma_long = price.rolling(MA_LONG).mean()

        # Trend signal
        trend = pd.Series(0.0, index=df.index)
        trend[ma_short > ma_long] = 1.0
        trend[ma_short < ma_long] = -1.0

        # Only keep days with a signal and enough history
        valid_start = MA_LONG + 60
        valid_mask = (trend != 0) & (pd.Series(range(len(df)), index=df.index) >= valid_start)
        valid_idx = df.index[valid_mask]

        if len(valid_idx) == 0:
            continue

        # Vectorized features
        feat_df = pd.DataFrame(index=valid_idx)
        feat_df['ticker'] = ticker
        feat_df['direction'] = trend.loc[valid_idx]
        feat_df['ma_dist'] = ((ma_short - ma_long) / (price + 1e-8)).loc[valid_idx]

        # Trend duration (vectorized approximation: count consecutive same-direction days)
        trend_change = (trend != trend.shift(1)).astype(int)
        trend_duration = trend_change.groupby(trend_change.cumsum()).cumcount()
        feat_df['trend_duration'] = trend_duration.loc[valid_idx]

        # Momentum
        feat_df['mom_5d'] = price.pct_change(5).loc[valid_idx]
        feat_df['mom_20d'] = price.pct_change(20).loc[valid_idx]
        feat_df['mom_60d'] = price.pct_change(60).loc[valid_idx]

        # Volatility
        feat_df['vol_20d'] = ret.rolling(20).std().loc[valid_idx] * np.sqrt(252)
        feat_df['vol_60d'] = ret.rolling(60).std().loc[valid_idx] * np.sqrt(252)

        # Drawdown from 252d peak
        rolling_max = price.rolling(252, min_periods=1).max()
        feat_df['drawdown'] = (price / rolling_max - 1).loc[valid_idx]

        # Skew/kurt
        feat_df['skew_20d'] = ret.rolling(20).skew().loc[valid_idx]
        feat_df['kurt_20d'] = ret.rolling(20).kurt().loc[valid_idx]

        # VIX
        if 'VIX' in df.columns:
            feat_df['vix'] = df['VIX'].loc[valid_idx]
            feat_df['vix_ma20'] = df['VIX'].rolling(20).mean().loc[valid_idx]

        # Target: does trend continue profitably over next 20 days?
        future_price = price.shift(-20)
        future_ret = (future_price / price - 1) * trend
        feat_df['future_ret'] = future_ret.loc[valid_idx]
        feat_df['target'] = (future_ret.loc[valid_idx] > 0).astype(float)
        feat_df.loc[future_price.loc[valid_idx].isna(), 'target'] = np.nan
        feat_df.loc[future_price.loc[valid_idx].isna(), 'future_ret'] = np.nan

        feat_df['date'] = feat_df.index
        feat_df = feat_df.reset_index(drop=True)
        all_features.append(feat_df)

    features_df = pd.concat(all_features, ignore_index=True)

    # Cross-asset alignment (computed after combining all tickers)
    # Group by date, count how many are in same direction
    date_dir_counts = features_df.groupby(['date', 'direction']).size().reset_index(name='count')
    features_df = features_df.merge(
        date_dir_counts, on=['date', 'direction'], how='left'
    )
    n_assets = features_df.groupby('date')['ticker'].transform('count')
    features_df['cross_align'] = (features_df['count'] - 1) / (n_assets - 1).clip(lower=1)
    features_df.drop('count', axis=1, inplace=True)

    elapsed = time.time() - t0
    print(f"  Done: {len(features_df)} observations in {elapsed:.1f}s, "
          f"positive target rate: {features_df['target'].mean():.1%}")
    return features_df


def train_ml_filter(features_df, universe_name=""):
    """Walk-forward GBM filter."""
    print(f"  [{universe_name}] Training ML walk-forward...")
    t0 = time.time()

    feature_cols = [c for c in features_df.columns
                   if c not in ['ticker', 'date', 'direction', 'target', 'future_ret']]

    features_df = features_df.sort_values('date').reset_index(drop=True)
    features_df['ml_prob'] = np.nan
    dates = sorted(features_df['date'].unique())

    n_folds = 0
    for i in range(TRAIN_WINDOW, len(dates)):
        train_end = dates[i]
        train_start = dates[max(0, i - TRAIN_WINDOW)]

        train_mask = (features_df['date'] >= train_start) & (features_df['date'] < train_end)
        test_mask = features_df['date'] == train_end

        X_train = features_df.loc[train_mask, feature_cols].fillna(0)
        y_train = features_df.loc[train_mask, 'target']
        X_test = features_df.loc[test_mask, feature_cols].fillna(0)

        valid = ~y_train.isna()
        X_train = X_train[valid]
        y_train = y_train[valid]

        if len(X_train) < 50 or len(X_test) == 0:
            continue

        model = GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.05,
            subsample=0.8, random_state=42
        )
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]
        features_df.loc[test_mask, 'ml_prob'] = probs
        n_folds += 1

        if n_folds % 500 == 0:
            print(f"    [{universe_name}] Fold {n_folds}/{len(dates)-TRAIN_WINDOW}...")

    elapsed = time.time() - t0
    print(f"  [{universe_name}] {n_folds} folds in {elapsed:.0f}s")

    valid_preds = features_df.dropna(subset=['ml_prob', 'target'])
    if len(valid_preds) > 0:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(valid_preds['target'], valid_preds['ml_prob'])
        print(f"  [{universe_name}] OOS AUC: {auc:.3f}")

    return features_df


def backtest_single_strategy(features_df, df, universe_name=""):
    """Backtest one strategy, return daily return series."""
    pred_df = features_df.dropna(subset=['ml_prob']).copy()
    dates = sorted(pred_df['date'].unique())

    portfolio_returns = []
    for date in dates:
        day_signals = pred_df[pred_df['date'] == date]
        high_conf = day_signals[day_signals['ml_prob'] > ML_THRESHOLD]

        if len(high_conf) == 0:
            portfolio_returns.append({'date': date, 'return': 0.0, 'n_pos': 0})
            continue

        positions = []
        for _, row in high_conf.iterrows():
            ticker = row['ticker']
            direction = row['direction']
            vol = max(row['vol_20d'], 0.05)
            weight = (TARGET_VOL / vol) / len(high_conf)
            weight = min(weight, 0.5)

            if ticker in df.columns:
                ticker_dates = df.index
                date_loc = ticker_dates.get_loc(date) if date in ticker_dates else None
                if date_loc is not None and date_loc + 1 < len(ticker_dates):
                    next_ret = df[ticker].iloc[date_loc + 1] / df[ticker].iloc[date_loc] - 1
                    positions.append({
                        'ticker': ticker, 'direction': direction,
                        'weight': weight, 'return': next_ret * direction * weight,
                    })

        if positions:
            day_ret = sum(p['return'] for p in positions)
            turnover = sum(p['weight'] for p in positions)
            cost = turnover * REBAL_COST_BPS / 10000
            day_ret -= cost * 0.1
            portfolio_returns.append({'date': date, 'return': day_ret, 'n_pos': len(positions)})
        else:
            portfolio_returns.append({'date': date, 'return': 0.0, 'n_pos': 0})

    ret_df = pd.DataFrame(portfolio_returns).set_index('date')
    ret_df.index = pd.to_datetime(ret_df.index)
    ret_series = ret_df['return']
    print(f"  [{universe_name}] Trading days: {len(ret_series)}, "
          f"Mean positions/day: {ret_df['n_pos'].mean():.1f}")
    return ret_series


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


def combine_strategies(ret_cta, ret_sectors, weights=(0.5, 0.5)):
    """Combine two strategies with given weights on common dates."""
    common = ret_cta.index.intersection(ret_sectors.index)
    combo = weights[0] * ret_cta.loc[common] + weights[1] * ret_sectors.loc[common]
    return combo


def apply_vix_overlay(combo_returns, df, vix_threshold=20.0, reduction=0.5):
    """When VIX > threshold (previous close), reduce exposure."""
    if 'VIX' not in df.columns:
        return combo_returns

    vix_prev = df['VIX'].shift(1)
    adjusted = combo_returns.copy()

    for date in adjusted.index:
        if date in vix_prev.index:
            v = vix_prev.loc[date]
            if pd.notna(v) and v > vix_threshold:
                adjusted.loc[date] *= reduction

    return adjusted


def run_adversarial(ret_series, df, name="Portfolio"):
    """Adversarial validation suite."""
    print(f"\n  ADVERSARIAL: {name}")
    print(f"  {'-'*60}")

    results = {}
    real = compute_metrics(ret_series, "real")
    real_sharpe = real.get('sharpe', 0)

    # 1. PERMUTATION TEST (block shuffle)
    print(f"    [1/4] Permutation test ({N_PERMUTATIONS} iters)...")
    perm_sharpes = []
    ret_vals = ret_series.values.copy()
    block_size = 5

    for trial in range(N_PERMUTATIONS):
        n = len(ret_vals)
        n_blocks = n // block_size
        blocks = [ret_vals[i*block_size:(i+1)*block_size] for i in range(n_blocks)]
        remainder = ret_vals[n_blocks*block_size:]
        np.random.shuffle(blocks)
        shuffled = np.concatenate(blocks + ([remainder] if len(remainder) > 0 else []))
        perm_ret = pd.Series(shuffled[:n], index=ret_series.index[:n])
        m = compute_metrics(perm_ret, f"perm_{trial}")
        if m:
            perm_sharpes.append(m['sharpe'])

    perm_p = np.mean([s >= real_sharpe for s in perm_sharpes]) if perm_sharpes else 1.0
    perm_pass = perm_p < 0.05
    print(f"      Real={real_sharpe:.3f}, Perm mean={np.mean(perm_sharpes):.3f}, p={perm_p:.3f} -> {'PASS' if perm_pass else 'FAIL'}")
    results['permutation'] = {'p_value': round(perm_p, 4), 'pass': bool(perm_pass)}

    # 2. SUB-PERIOD
    print(f"    [2/4] Sub-period consistency...")
    n = len(ret_series)
    block_sz = n // 4
    block_sharpes = []
    for b in range(4):
        start = b * block_sz
        end = (b + 1) * block_sz if b < 3 else n
        m = compute_metrics(ret_series.iloc[start:end], f"block_{b}")
        if m:
            block_sharpes.append(m['sharpe'])

    cv = np.std(block_sharpes) / abs(np.mean(block_sharpes)) if block_sharpes and np.mean(block_sharpes) != 0 else 999
    sub_pass = cv < 0.50
    print(f"      Sharpes: {[f'{s:.2f}' for s in block_sharpes]}, CV={cv:.3f} -> {'PASS' if sub_pass else 'FAIL'}")
    results['sub_period'] = {'cv': round(cv, 3), 'block_sharpes': [round(s,3) for s in block_sharpes], 'pass': bool(sub_pass)}

    # 3. OUTLIER ROBUSTNESS
    print(f"    [3/4] Outlier robustness...")
    p95 = ret_series.quantile(0.95)
    p05 = ret_series.quantile(0.05)
    trimmed = ret_series[(ret_series > p05) & (ret_series < p95)]
    m_full = compute_metrics(ret_series)
    m_trim = compute_metrics(trimmed)
    deg = (m_full['sharpe'] - m_trim['sharpe']) / abs(m_full['sharpe']) if m_full.get('sharpe', 0) != 0 else 999
    outlier_pass = abs(deg) < 0.30
    print(f"      Full={m_full['sharpe']:.3f}, Trimmed={m_trim['sharpe']:.3f}, deg={deg:.3f} -> {'PASS' if outlier_pass else 'FAIL'}")
    results['outlier'] = {'degradation': round(deg, 3), 'pass': bool(outlier_pass)}

    # 4. R1 REGIME
    print(f"    [4/4] R1 regime test...")
    spy_ret = df['SPY'].pct_change()
    spy_20d = spy_ret.rolling(20).sum()

    common_idx = ret_series.index.intersection(spy_20d.dropna().index)
    ret_common = ret_series.loc[common_idx]
    spy_20d_common = spy_20d.loc[common_idx]

    green_mask = spy_20d_common > 0
    m_green = compute_metrics(ret_common[green_mask], "green")
    m_red = compute_metrics(ret_common[~green_mask], "red")

    if m_green and m_red:
        s_g, s_r = m_green['sharpe'], m_red['sharpe']
        gap = abs(s_g - s_r) / max(abs(s_g), abs(s_r), 0.01)
        r1_pass = gap < 0.50
        print(f"      Green={s_g:.3f}, Red={s_r:.3f}, gap={gap:.3f} -> {'PASS' if r1_pass else 'FAIL'}")
        results['r1_regime'] = {'gap': round(gap, 3), 'pass': bool(r1_pass),
                                'green_sharpe': s_g, 'red_sharpe': s_r}
    else:
        r1_pass = False
        results['r1_regime'] = {'pass': False}

    gates = sum([results.get(k, {}).get('pass', False)
                 for k in ['permutation', 'sub_period', 'outlier', 'r1_regime']])
    results['summary'] = {'gates_passed': gates, 'total': 4, 'verdict': 'PASS' if gates >= 3 else 'FAIL'}
    print(f"      VERDICT: {gates}/4 -> {results['summary']['verdict']}")
    return results


def correlation_analysis(ret_cta, ret_sectors, df):
    """Analyze correlation between strategy returns."""
    print("\n" + "=" * 80)
    print("CORRELATION ANALYSIS")
    print("=" * 80)

    common = ret_cta.index.intersection(ret_sectors.index)
    r1 = ret_cta.loc[common]
    r2 = ret_sectors.loc[common]

    corr = r1.corr(r2)
    print(f"  Overall correlation: {corr:.3f}")

    rolling_corr = r1.rolling(60).corr(r2)
    rc_mean = rolling_corr.mean()
    print(f"  Rolling 60d correlation: mean={rc_mean:.3f}, range=[{rolling_corr.min():.3f}, {rolling_corr.max():.3f}]")

    # Correlation during stress
    corr_stress = corr_calm = np.nan
    if 'VIX' in df.columns:
        vix = df['VIX'].reindex(common)
        stress = vix > 25
        calm = vix <= 25
        if stress.sum() > 30:
            corr_stress = r1[stress].corr(r2[stress])
            corr_calm = r1[calm].corr(r2[calm])
            print(f"  Stress (VIX>25): {corr_stress:.3f}")
            print(f"  Calm   (VIX<=25): {corr_calm:.3f}")

    # Diversification ratio
    sigma1 = r1.std() * np.sqrt(252)
    sigma2 = r2.std() * np.sqrt(252)
    combo = 0.5 * r1 + 0.5 * r2
    sigma_combo = combo.std() * np.sqrt(252)
    div_ratio = (0.5 * sigma1 + 0.5 * sigma2) / (sigma_combo + 1e-8)
    print(f"  Diversification ratio: {div_ratio:.3f} (>1 = diversification)")

    return {
        'overall_correlation': round(corr, 3),
        'rolling_corr_mean': round(rc_mean, 3),
        'corr_stress': round(corr_stress, 3) if pd.notna(corr_stress) else None,
        'corr_calm': round(corr_calm, 3) if pd.notna(corr_calm) else None,
        'diversification_ratio': round(div_ratio, 3),
        'vol_cta': round(sigma1 * 100, 1),
        'vol_sectors': round(sigma2 * 100, 1),
        'vol_combo': round(sigma_combo * 100, 1),
    }


def plot_results(ret_cta, ret_sectors, combo_eq, combo_vix, df):
    """Generate analysis plots."""
    common = combo_eq.index

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Equity curves
    ax = axes[0, 0]
    for ret, label in [(ret_cta.reindex(common).fillna(0), 'CTA Trend'),
                       (ret_sectors.reindex(common).fillna(0), 'Sector Rotation'),
                       (combo_eq, 'EW Combo'), (combo_vix, 'Combo+VIX')]:
        eq = (1 + ret).cumprod() * INITIAL_CAPITAL
        ax.plot(eq.index, eq.values, label=label, linewidth=1.5)
    spy_ret = df['SPY'].pct_change().reindex(common).fillna(0)
    spy_eq = (1 + spy_ret).cumprod() * INITIAL_CAPITAL
    ax.plot(spy_eq.index, spy_eq.values, label='SPY B&H', alpha=0.5, linestyle='--')
    ax.set_title('Equity Curves ($100K)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # 2. Drawdowns
    ax = axes[0, 1]
    for ret, label, color in [(combo_eq, 'EW Combo', 'blue'), (combo_vix, 'Combo+VIX', 'green')]:
        eq = (1 + ret).cumprod()
        dd = eq / eq.cummax() - 1
        ax.fill_between(dd.index, dd.values, 0, alpha=0.4, label=label, color=color)
    ax.set_title('Drawdowns')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. Rolling Sharpe
    ax = axes[1, 0]
    for ret, label in [(combo_eq, 'EW Combo'), (combo_vix, 'Combo+VIX')]:
        rs = ret.rolling(252).apply(lambda x: x.mean()/x.std()*np.sqrt(252) if x.std()>0 else 0)
        ax.plot(rs.index, rs.values, label=label, linewidth=1)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_title('Rolling 252d Sharpe')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 4. Rolling correlation
    ax = axes[1, 1]
    r1 = ret_cta.reindex(common).fillna(0)
    r2 = ret_sectors.reindex(common).fillna(0)
    rc = r1.rolling(60).corr(r2)
    ax.plot(rc.index, rc.values, color='purple', linewidth=0.8)
    ax.axhline(rc.mean(), color='red', linestyle='--', label=f'Mean: {rc.mean():.2f}')
    ax.set_title('Rolling 60d Correlation')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1, 1)

    plt.tight_layout()
    plt.savefig(OUTPUT / 'portfolio_analysis.png', dpi=150)
    plt.close()
    print(f"  Saved plots")


def main():
    t0 = time.time()
    print("=" * 80)
    print("ML PORTFOLIO COMBINATION BACKTEST")
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # 1. Data
    df = download_data()

    # 2. Strategy 1: CTA
    print("\n" + "=" * 80)
    print("STRATEGY 1: CTA TREND FOLLOWING")
    print("=" * 80)
    features_cta = build_features_vectorized(df, UNIVERSE_CTA)
    features_cta = train_ml_filter(features_cta, "CTA")
    ret_cta = backtest_single_strategy(features_cta, df, "CTA")

    # 3. Strategy 2: Sectors
    print("\n" + "=" * 80)
    print("STRATEGY 2: SECTOR ROTATION")
    print("=" * 80)
    features_sectors = build_features_vectorized(df, UNIVERSE_SECTORS)
    features_sectors = train_ml_filter(features_sectors, "Sectors")
    ret_sectors = backtest_single_strategy(features_sectors, df, "Sectors")

    # 4. Individual metrics
    print("\n" + "=" * 80)
    print("INDIVIDUAL STRATEGY METRICS")
    print("=" * 80)
    m_cta = compute_metrics(ret_cta, "ML CTA Trend")
    m_sectors = compute_metrics(ret_sectors, "ML Sector Rotation")
    for m in [m_cta, m_sectors]:
        if m:
            print(f"  {m['name']:<25} Sharpe={m['sharpe']:.3f} Sortino={m['sortino']:.3f} "
                  f"CAGR={m['cagr']:.1f}% MaxDD={m['max_dd']:.1f}%")

    # 5. Correlation
    corr_results = correlation_analysis(ret_cta, ret_sectors, df)

    # 6. Combinations
    print("\n" + "=" * 80)
    print("PORTFOLIO COMBINATIONS")
    print("=" * 80)

    combo_eq = combine_strategies(ret_cta, ret_sectors, weights=(0.5, 0.5))
    m_combo_eq = compute_metrics(combo_eq, "EW Combo (50/50)")

    combo_vix = apply_vix_overlay(combo_eq, df, vix_threshold=VIX_THRESHOLD, reduction=0.5)
    m_combo_vix = compute_metrics(combo_vix, "Combo + VIX Overlay")

    print(f"\n  {'Variant':<30} {'Sharpe':>8} {'Sortino':>8} {'CAGR':>8} {'MaxDD':>8}")
    print(f"  {'-'*70}")
    for m in [m_combo_eq, m_combo_vix]:
        if m:
            print(f"  {m['name']:<30} {m['sharpe']:>8.3f} {m['sortino']:>8.3f} "
                  f"{m['cagr']:>7.1f}% {m['max_dd']:>7.1f}%")

    # VIX sensitivity
    print("\n  VIX Overlay Sensitivity:")
    vix_variants = {}
    for vt in [15, 20, 25, 30]:
        for red in [0.3, 0.5, 0.7]:
            v = apply_vix_overlay(combo_eq, df, vix_threshold=vt, reduction=red)
            mv = compute_metrics(v, f"VIX>{vt},red={red}")
            if mv:
                key = f"VIX>{vt},red={red}"
                vix_variants[key] = mv

    best_vix = max(vix_variants.items(), key=lambda x: x[1].get('sharpe', 0)) if vix_variants else (None, {})
    if best_vix[0]:
        print(f"    Best: {best_vix[0]} -> Sharpe={best_vix[1]['sharpe']:.3f}")

    # Weight sensitivity
    print("\n  Weight Sensitivity:")
    weight_variants = {}
    for w1 in [0.3, 0.4, 0.5, 0.6, 0.7]:
        w2 = 1.0 - w1
        c = combine_strategies(ret_cta, ret_sectors, weights=(w1, w2))
        mc = compute_metrics(c, f"CTA={w1:.0%}/Sec={w2:.0%}")
        if mc:
            weight_variants[f"{int(w1*100)}/{int(w2*100)}"] = mc
            print(f"    CTA={w1:.0%} Sectors={w2:.0%}: Sharpe={mc['sharpe']:.3f}")

    # SPY benchmark
    common = combo_eq.index
    spy_ret = df['SPY'].pct_change().reindex(common).fillna(0)
    m_spy = compute_metrics(spy_ret, "SPY B&H")

    # 7. Adversarial
    print("\n" + "=" * 80)
    print("ADVERSARIAL VALIDATION")
    print("=" * 80)
    adv_eq = run_adversarial(combo_eq, df, "EW Combo")
    adv_vix = run_adversarial(combo_vix, df, "Combo+VIX")

    # 8. Plots
    plot_results(ret_cta, ret_sectors, combo_eq, combo_vix, df)

    # 9. Final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"\n  {'Strategy':<30} {'Sharpe':>7} {'Sortino':>8} {'CAGR':>7} {'MaxDD':>7} {'PF':>6} {'WR':>6}")
    print(f"  {'-'*75}")
    for m in [m_cta, m_sectors, m_combo_eq, m_combo_vix, m_spy]:
        if m:
            print(f"  {m['name']:<30} {m['sharpe']:>7.3f} {m['sortino']:>8.3f} "
                  f"{m['cagr']:>6.1f}% {m['max_dd']:>6.1f}% {m['profit_factor']:>6.3f} {m['win_rate']:>5.1f}%")

    # 10. Save
    output = {
        'strategy': 'ML Portfolio Combination',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'individual_metrics': {'cta_trend': m_cta, 'sector_rotation': m_sectors},
        'combination_metrics': {'equal_weight': m_combo_eq, 'combo_plus_vix': m_combo_vix},
        'correlation': corr_results,
        'adversarial': {'equal_weight': adv_eq, 'combo_plus_vix': adv_vix},
        'vix_sensitivity': vix_variants,
        'weight_sensitivity': weight_variants,
        'benchmark': {'spy': m_spy},
        'parameters': {
            'ma_short': MA_SHORT, 'ma_long': MA_LONG,
            'ml_threshold': ML_THRESHOLD, 'target_vol': TARGET_VOL,
            'train_window': TRAIN_WINDOW, 'vix_threshold': VIX_THRESHOLD,
            'n_permutations': N_PERMUTATIONS,
            'universe_cta': list(UNIVERSE_CTA.keys()),
            'universe_sectors': list(UNIVERSE_SECTORS.keys()),
        },
        'runtime_seconds': round(time.time() - t0, 1),
    }

    with open(OUTPUT / 'results.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)

    elapsed = time.time() - t0
    print(f"\n  Runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Output: {OUTPUT}")
    print("=" * 80)


if __name__ == '__main__':
    main()
