#!/usr/bin/env python3
"""
ML-Enhanced Cross-Sectional Momentum (Fama-French Style)
==========================================================
INSIGHT: Classic cross-sectional momentum (buy recent winners, sell recent losers)
is one of the most robust anomalies in finance. But momentum crashes (sharp reversals)
can wipe out years of gains. ML can learn which momentum signals will continue vs reverse
by incorporating volatility regimes, dispersion, and multi-horizon momentum features.

Strategy:
  1. Broad ETF universe (sectors, regions, bonds, commodities, real estate) — ~28 assets
  2. Base signal: 12-month return minus most recent 1-month (Jegadeesh-Titman)
  3. Rank assets cross-sectionally each month. Long top quintile, short bottom quintile (dollar-neutral)
  4. ML (LightGBM) predicts which top-quintile assets continue vs reverse
  5. Monthly rebalance with SLIDING walk-forward

Walk-forward: SLIDING 252d train, 21d advance (HC #0). Fixed $100K, NO DCA (HC #713).
Full adversarial: permutation 100x, sub-period 4-block, outlier, R1 regime (HC #705).
Cost: 10bps round-trip per rebalance.
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
import lightgbm as lgb

warnings.filterwarnings('ignore')
np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
INITIAL_CAPITAL  = 100_000
TRAIN_WINDOW     = 252       # ~1 year sliding
REBAL_PERIOD     = 21        # monthly rebalance
N_PERMUTATIONS   = 100
REBAL_COST_BPS   = 10        # 10bps round-trip
QUINTILE_FRAC    = 0.20      # top/bottom 20%

BASE   = Path("/home/jupiter/Lvl3Quant")
OUTPUT = BASE / "output" / "ml_cross_momentum"
OUTPUT.mkdir(parents=True, exist_ok=True)

# Universe: 28 liquid ETFs across asset classes
UNIVERSE = {
    # US Sectors
    'XLK': 'equity', 'XLF': 'equity', 'XLE': 'equity', 'XLV': 'equity',
    'XLI': 'equity', 'XLP': 'equity', 'XLU': 'equity', 'XLB': 'equity',
    'XLY': 'equity', 'XLRE': 'equity',
    # Broad US equity
    'SPY': 'equity', 'QQQ': 'equity', 'IWM': 'equity',
    # International
    'EEM': 'equity', 'EFA': 'equity', 'VWO': 'equity',
    # Bonds
    'TLT': 'bond', 'HYG': 'bond', 'LQD': 'bond',
    # Commodities
    'GLD': 'commodity', 'SLV': 'commodity', 'DBA': 'commodity', 'USO': 'commodity',
    # Real estate
    'VNQ': 'realestate',
}

ASSET_CLASS_MAP = UNIVERSE.copy()


def log(msg):
    print(msg)
    sys.stdout.flush()


def download_data():
    log("=" * 80)
    log("STEP 1: DOWNLOADING DATA")
    log("=" * 80)

    all_tickers = list(UNIVERSE.keys()) + ['^VIX', '^VIX9D']

    cache = BASE / "data" / "cache" / "cross_momentum_data.parquet"
    if cache.exists() and (time.time() - cache.stat().st_mtime) < 3600:
        df = pd.read_parquet(cache)
        log(f"  Cached: {df.shape}, {df.index[0].date()} -> {df.index[-1].date()}")
        return df

    log("  Downloading from yfinance...")
    raw = yf.download(all_tickers, start='2005-01-01', auto_adjust=True, progress=False)
    closes = raw['Close'] if isinstance(raw.columns, pd.MultiIndex) else raw
    if '^VIX' in closes.columns:
        closes = closes.rename(columns={'^VIX': 'VIX'})
    if '^VIX9D' in closes.columns:
        closes = closes.rename(columns={'^VIX9D': 'VIX9D'})
    closes = closes.ffill().dropna(thresh=len(closes.columns) - 5)
    cache.parent.mkdir(parents=True, exist_ok=True)
    closes.to_parquet(cache)
    log(f"  Shape: {closes.shape}, {closes.index[0].date()} -> {closes.index[-1].date()}")
    avail = [t for t in UNIVERSE if t in closes.columns]
    log(f"  Assets available: {len(avail)}/{len(UNIVERSE)}")
    return closes


def compute_momentum_features(df):
    """Compute cross-sectional momentum features for all assets."""
    log("\n" + "=" * 80)
    log("STEP 2: COMPUTING MOMENTUM FEATURES")
    log("=" * 80)

    tickers = [t for t in UNIVERSE if t in df.columns]
    returns = df[tickers].pct_change()

    # Get VIX
    vix = df['VIX'] if 'VIX' in df.columns else pd.Series(20.0, index=df.index)
    vix9d = df['VIX9D'] if 'VIX9D' in df.columns else None

    rows = []
    rebal_dates = df.index[::REBAL_PERIOD]  # monthly dates

    for i, date in enumerate(rebal_dates):
        date_loc = df.index.get_loc(date)
        if date_loc < 252:  # need 12 months of history
            continue

        # Cross-sectional features for each asset
        assets_data = []
        for ticker in tickers:
            price = df[ticker]

            # Skip if insufficient data
            if pd.isna(price.iloc[date_loc]) or date_loc < 252:
                continue

            # Momentum at multiple lookbacks
            ret_1m = price.iloc[date_loc] / price.iloc[max(0, date_loc - 21)] - 1
            ret_3m = price.iloc[date_loc] / price.iloc[max(0, date_loc - 63)] - 1
            ret_6m = price.iloc[date_loc] / price.iloc[max(0, date_loc - 126)] - 1
            ret_12m = price.iloc[date_loc] / price.iloc[max(0, date_loc - 252)] - 1

            # Classic Jegadeesh-Titman: 12m minus 1m (skip most recent month)
            jt_momentum = ret_12m - ret_1m

            # Momentum volatility (stability of rolling 1m returns over past 12m)
            monthly_rets = []
            for m in range(12):
                start_idx = max(0, date_loc - (m + 1) * 21)
                end_idx = max(0, date_loc - m * 21)
                if end_idx > start_idx and end_idx < len(price):
                    mr = price.iloc[end_idx] / price.iloc[start_idx] - 1
                    monthly_rets.append(mr)
            mom_vol = np.std(monthly_rets) if len(monthly_rets) > 2 else 0.0

            # Volume trend (if we had volume — approximate with return vol)
            ret_vol_20 = returns[ticker].iloc[max(0, date_loc-20):date_loc].std()
            ret_vol_60 = returns[ticker].iloc[max(0, date_loc-60):date_loc].std()
            vol_trend = ret_vol_20 / (ret_vol_60 + 1e-8) - 1

            # Recent max drawdown (past 63 days)
            recent_prices = price.iloc[max(0, date_loc-63):date_loc+1]
            running_max = recent_prices.cummax()
            dd = (recent_prices / running_max - 1)
            max_dd = dd.min() if len(dd) > 0 else 0.0

            # Relative strength vs SPY
            if 'SPY' in df.columns:
                spy_ret_6m = df['SPY'].iloc[date_loc] / df['SPY'].iloc[max(0, date_loc-126)] - 1
                rel_strength = ret_6m - spy_ret_6m
            else:
                rel_strength = 0.0

            # Asset class encoding
            ac = ASSET_CLASS_MAP.get(ticker, 'equity')
            ac_equity = 1 if ac == 'equity' else 0
            ac_bond = 1 if ac == 'bond' else 0
            ac_commodity = 1 if ac == 'commodity' else 0
            ac_realestate = 1 if ac == 'realestate' else 0

            assets_data.append({
                'date': date,
                'ticker': ticker,
                'jt_momentum': jt_momentum,
                'ret_1m': ret_1m,
                'ret_3m': ret_3m,
                'ret_6m': ret_6m,
                'ret_12m': ret_12m,
                'mom_vol': mom_vol,
                'vol_trend': vol_trend,
                'max_dd_63d': max_dd,
                'rel_strength_spy': rel_strength,
                'ret_vol_20d': ret_vol_20,
                'ac_equity': ac_equity,
                'ac_bond': ac_bond,
                'ac_commodity': ac_commodity,
                'ac_realestate': ac_realestate,
                'vix': vix.iloc[date_loc] if date_loc < len(vix) else 20.0,
            })

        if len(assets_data) < 5:
            continue

        day_df = pd.DataFrame(assets_data)

        # Cross-sectional dispersion (std of all momentum values this period)
        cs_dispersion = day_df['jt_momentum'].std()
        day_df['cs_dispersion'] = cs_dispersion

        # Cross-sectional rank (percentile rank of momentum)
        day_df['mom_rank'] = day_df['jt_momentum'].rank(pct=True)

        # VIX term structure (if available)
        if vix9d is not None and date_loc < len(vix9d):
            v = vix.iloc[date_loc]
            v9 = vix9d.iloc[date_loc]
            day_df['vix_term'] = (v / (v9 + 1e-8)) - 1 if not pd.isna(v9) else 0.0
        else:
            day_df['vix_term'] = 0.0

        # Forward return (target) — next rebalance period
        next_loc = min(date_loc + REBAL_PERIOD, len(df) - 1)
        for idx in range(len(day_df)):
            ticker = day_df.iloc[idx]['ticker']
            fwd = df[ticker].iloc[next_loc] / df[ticker].iloc[date_loc] - 1
            day_df.loc[day_df.index[idx], 'fwd_return'] = fwd

        rows.append(day_df)

    features_df = pd.concat(rows, ignore_index=True)
    log(f"  Total feature rows: {len(features_df)}")
    log(f"  Date range: {features_df['date'].min().date()} -> {features_df['date'].max().date()}")
    log(f"  Unique dates: {features_df['date'].nunique()}")
    return features_df


FEATURE_COLS = [
    'jt_momentum', 'ret_1m', 'ret_3m', 'ret_6m', 'ret_12m',
    'mom_vol', 'vol_trend', 'max_dd_63d', 'rel_strength_spy', 'ret_vol_20d',
    'ac_equity', 'ac_bond', 'ac_commodity', 'ac_realestate',
    'vix', 'cs_dispersion', 'mom_rank', 'vix_term',
]


def run_baseline(features_df, df):
    """Run pure momentum baseline (no ML) — long top quintile, short bottom quintile."""
    log("\n" + "=" * 80)
    log("STEP 3A: BASELINE (PURE MOMENTUM)")
    log("=" * 80)

    dates = sorted(features_df['date'].unique())
    portfolio_returns = []
    prev_positions = set()

    for date in dates:
        day = features_df[features_df['date'] == date].copy()
        if len(day) < 5:
            portfolio_returns.append({'date': date, 'return': 0.0, 'n_long': 0, 'n_short': 0})
            continue

        n_quintile = max(1, int(len(day) * QUINTILE_FRAC))

        # Sort by JT momentum
        day = day.sort_values('jt_momentum', ascending=False)
        longs = day.head(n_quintile)
        shorts = day.tail(n_quintile)

        n_positions = len(longs) + len(shorts)
        weight = 1.0 / n_positions if n_positions > 0 else 0

        day_ret = 0.0
        current_positions = set()

        for _, row in longs.iterrows():
            day_ret += row['fwd_return'] * weight
            current_positions.add(row['ticker'])

        for _, row in shorts.iterrows():
            day_ret -= row['fwd_return'] * weight  # short = negative exposure
            current_positions.add(row['ticker'])

        # Transaction costs on changed positions
        changed = current_positions.symmetric_difference(prev_positions)
        turnover_frac = len(changed) / max(len(current_positions), 1)
        cost = turnover_frac * weight * n_positions * REBAL_COST_BPS / 10000
        day_ret -= cost
        prev_positions = current_positions

        portfolio_returns.append({
            'date': date, 'return': day_ret,
            'n_long': len(longs), 'n_short': len(shorts),
        })

    ret_df = pd.DataFrame(portfolio_returns).set_index('date')
    ret_series = ret_df['return']
    log(f"  Rebalance periods: {len(ret_series)}")
    return ret_series, ret_df


def run_ml_strategy(features_df, df):
    """Walk-forward ML strategy: LightGBM predicts momentum continuation."""
    log("\n" + "=" * 80)
    log("STEP 3B: ML-ENHANCED MOMENTUM (WALK-FORWARD)")
    log("=" * 80)

    dates = sorted(features_df['date'].unique())
    # We need at least TRAIN_WINDOW // REBAL_PERIOD months of data for first train
    min_train_periods = TRAIN_WINDOW // REBAL_PERIOD  # ~12 months
    if len(dates) <= min_train_periods:
        log("  ERROR: not enough data for walk-forward")
        return pd.Series(dtype=float), pd.DataFrame()

    # Target: will this asset outperform cross-sectional median next period?
    features_df['target'] = (features_df['fwd_return'] > features_df.groupby('date')['fwd_return'].transform('median')).astype(int)

    portfolio_returns = []
    all_predictions = []
    prev_positions = set()

    lgb_params = {
        'objective': 'binary',
        'metric': 'auc',
        'n_estimators': 200,
        'max_depth': 4,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_samples': 10,
        'verbose': -1,
        'random_state': 42,
    }

    n_folds = 0
    for i in range(min_train_periods, len(dates)):
        test_date = dates[i]

        # Sliding window: train on last min_train_periods months
        train_start = max(0, i - min_train_periods)
        train_dates = dates[train_start:i]
        train_data = features_df[features_df['date'].isin(train_dates)]
        test_data = features_df[features_df['date'] == test_date]

        if len(train_data) < 50 or len(test_data) < 5:
            portfolio_returns.append({'date': test_date, 'return': 0.0, 'n_long': 0, 'n_short': 0})
            continue

        X_train = train_data[FEATURE_COLS].values
        y_train = train_data['target'].values
        X_test = test_data[FEATURE_COLS].values

        # Train LightGBM
        model = lgb.LGBMClassifier(**lgb_params)
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]

        test_data = test_data.copy()
        test_data['ml_prob'] = probs

        # ML-enhanced quintile selection
        n_quintile = max(1, int(len(test_data) * QUINTILE_FRAC))

        # Sort by JT momentum for base ranking
        test_data = test_data.sort_values('jt_momentum', ascending=False)
        top_mom = test_data.head(n_quintile)
        bot_mom = test_data.tail(n_quintile)

        # ML filter: from top momentum, keep those ML says will continue
        # From bottom momentum, keep those ML says will keep falling
        ml_longs = top_mom[top_mom['ml_prob'] > 0.5].copy()
        ml_shorts = bot_mom[bot_mom['ml_prob'] < 0.5].copy()

        # If ML filters out everything, fall back to pure momentum
        if len(ml_longs) == 0 and len(ml_shorts) == 0:
            ml_longs = top_mom
            ml_shorts = bot_mom

        n_positions = max(len(ml_longs) + len(ml_shorts), 1)
        weight = 1.0 / n_positions

        day_ret = 0.0
        current_positions = set()

        for _, row in ml_longs.iterrows():
            day_ret += row['fwd_return'] * weight
            current_positions.add(row['ticker'])

        for _, row in ml_shorts.iterrows():
            day_ret -= row['fwd_return'] * weight
            current_positions.add(row['ticker'])

        # Transaction costs
        changed = current_positions.symmetric_difference(prev_positions)
        turnover_frac = len(changed) / max(len(current_positions), 1)
        cost = turnover_frac * weight * n_positions * REBAL_COST_BPS / 10000
        day_ret -= cost
        prev_positions = current_positions

        portfolio_returns.append({
            'date': test_date, 'return': day_ret,
            'n_long': len(ml_longs), 'n_short': len(ml_shorts),
        })

        # Store predictions for adversarial
        for _, row in test_data.iterrows():
            all_predictions.append({
                'date': test_date,
                'ticker': row['ticker'],
                'ml_prob': row['ml_prob'],
                'jt_momentum': row['jt_momentum'],
                'fwd_return': row['fwd_return'],
            })

        n_folds += 1
        if n_folds % 20 == 0:
            log(f"  Fold {n_folds}: {test_date.date()}, positions: L={len(ml_longs)} S={len(ml_shorts)}")

    log(f"  Total WF folds: {n_folds}")
    ret_df = pd.DataFrame(portfolio_returns).set_index('date')
    ret_series = ret_df['return']
    pred_df = pd.DataFrame(all_predictions)

    # Feature importance from last model
    if n_folds > 0:
        importance = dict(zip(FEATURE_COLS, model.feature_importances_))
        log("\n  Feature importance (last model):")
        for feat, imp in sorted(importance.items(), key=lambda x: -x[1])[:10]:
            log(f"    {feat}: {imp}")

    return ret_series, ret_df, pred_df


def compute_metrics(returns, name="Strategy"):
    r = returns.dropna()
    if len(r) < 10:
        return {}

    mu = r.mean() * (252 / REBAL_PERIOD)  # annualize from monthly
    sigma = r.std() * np.sqrt(252 / REBAL_PERIOD)
    sharpe = mu / (sigma + 1e-8)

    downside = r[r < 0].std() * np.sqrt(252 / REBAL_PERIOD)
    sortino = mu / (downside + 1e-8)

    cumret = (1 + r).cumprod()
    total_return = cumret.iloc[-1] - 1
    years = len(r) * REBAL_PERIOD / 252
    cagr = (cumret.iloc[-1]) ** (1 / max(years, 0.1)) - 1

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


def run_adversarial(ret_series, pred_df, df):
    """Full adversarial validation suite."""
    log("\n" + "=" * 80)
    log("ADVERSARIAL VALIDATION")
    log("=" * 80)

    results = {}
    real = compute_metrics(ret_series, "ML Strategy")
    real_sharpe = real.get('sharpe', 0)

    # ── 1. PERMUTATION TEST ──
    log("\n  [1/4] Permutation test (100 shuffles)...")
    perm_sharpes = []
    dates = sorted(pred_df['date'].unique())

    for trial in range(N_PERMUTATIONS):
        # Shuffle ML probs → random quintile selection
        perm_pred = pred_df.copy()
        perm_pred['ml_prob'] = np.random.permutation(perm_pred['ml_prob'].values)

        perm_rets = []
        prev_pos = set()

        for date in dates:
            day = perm_pred[perm_pred['date'] == date].copy()
            if len(day) < 5:
                perm_rets.append(0.0)
                continue

            n_q = max(1, int(len(day) * QUINTILE_FRAC))
            day = day.sort_values('jt_momentum', ascending=False)
            top = day.head(n_q)
            bot = day.tail(n_q)

            longs = top[top['ml_prob'] > 0.5]
            shorts = bot[bot['ml_prob'] < 0.5]
            if len(longs) == 0 and len(shorts) == 0:
                longs = top
                shorts = bot

            n_pos = max(len(longs) + len(shorts), 1)
            w = 1.0 / n_pos
            day_ret = 0.0
            cur_pos = set()

            for _, row in longs.iterrows():
                day_ret += row['fwd_return'] * w
                cur_pos.add(row['ticker'])
            for _, row in shorts.iterrows():
                day_ret -= row['fwd_return'] * w
                cur_pos.add(row['ticker'])

            changed = cur_pos.symmetric_difference(prev_pos)
            cost = len(changed) / max(len(cur_pos), 1) * w * n_pos * REBAL_COST_BPS / 10000
            day_ret -= cost
            prev_pos = cur_pos
            perm_rets.append(day_ret)

        perm_ret_series = pd.Series(perm_rets, index=dates)
        m = compute_metrics(perm_ret_series, f"perm_{trial}")
        if m:
            perm_sharpes.append(m['sharpe'])

        if (trial + 1) % 20 == 0:
            log(f"    Perm {trial+1}/100 done, mean Sharpe so far: {np.mean(perm_sharpes):.3f}")
            sys.stdout.flush()

    perm_p = np.mean([s >= real_sharpe for s in perm_sharpes]) if perm_sharpes else 1.0
    perm_pass = perm_p < 0.05
    log(f"    Real Sharpe: {real_sharpe:.3f}")
    log(f"    Perm mean: {np.mean(perm_sharpes):.3f} +/- {np.std(perm_sharpes):.3f}")
    log(f"    p-value: {perm_p:.3f} -> {'PASS' if perm_pass else 'FAIL'}")
    results['permutation'] = {'p_value': round(perm_p, 4), 'pass': bool(perm_pass)}

    # ── 2. SUB-PERIOD STABILITY ──
    log("\n  [2/4] Sub-period stability (4 quarters)...")
    n = len(ret_series)
    block_size = n // 4
    block_sharpes = []
    for b in range(4):
        start = b * block_size
        end = (b + 1) * block_size if b < 3 else n
        m = compute_metrics(ret_series.iloc[start:end], f"block_{b}")
        if m:
            block_sharpes.append(m['sharpe'])
            log(f"    Block {b+1}: Sharpe {m['sharpe']:.3f}")

    if block_sharpes and abs(np.mean(block_sharpes)) > 0.01:
        cv = np.std(block_sharpes) / abs(np.mean(block_sharpes))
    else:
        cv = 999
    sub_pass = cv < 0.50
    log(f"    CV of Sharpe: {cv:.3f} -> {'PASS' if sub_pass else 'FAIL'}")
    results['sub_period'] = {'cv': round(cv, 3), 'block_sharpes': block_sharpes, 'pass': bool(sub_pass)}

    # ── 3. OUTLIER ROBUSTNESS ──
    log("\n  [3/4] Outlier robustness (trim 5%/95%)...")
    p95 = ret_series.quantile(0.95)
    p05 = ret_series.quantile(0.05)
    trimmed = ret_series[(ret_series > p05) & (ret_series < p95)]
    m_full = compute_metrics(ret_series, "full")
    m_trim = compute_metrics(trimmed, "trimmed")
    if m_full.get('sharpe', 0) != 0:
        deg = (m_full['sharpe'] - m_trim['sharpe']) / abs(m_full['sharpe'])
    else:
        deg = 999
    outlier_pass = abs(deg) < 0.30
    log(f"    Full Sharpe: {m_full.get('sharpe', 0):.3f}, Trimmed: {m_trim.get('sharpe', 0):.3f}")
    log(f"    Degradation: {deg:.1%} -> {'PASS' if outlier_pass else 'FAIL'}")
    results['outlier'] = {'degradation': round(deg, 3), 'pass': bool(outlier_pass)}

    # ── 4. R1 REGIME CHECK ──
    log("\n  [4/4] R1 regime test (green vs red days)...")
    if 'SPY' in df.columns:
        spy_ret = df['SPY'].pct_change()
        spy_20d = spy_ret.rolling(20).sum()

        common_dates = ret_series.index.intersection(spy_20d.dropna().index)
        if len(common_dates) > 20:
            aligned_rets = ret_series.loc[common_dates]
            aligned_spy = spy_20d.loc[common_dates]

            green_mask = aligned_spy > 0
            m_green = compute_metrics(aligned_rets[green_mask], "green")
            m_red = compute_metrics(aligned_rets[~green_mask], "red")

            if m_green and m_red:
                s_g, s_r = m_green['sharpe'], m_red['sharpe']
                gap = abs(s_g - s_r) / max(abs(s_g), abs(s_r), 0.01)
                r1_pass = gap < 0.50
                log(f"    Green Sharpe: {s_g:.3f}, Red Sharpe: {s_r:.3f}")
                log(f"    Gap: {gap:.3f} -> {'PASS' if r1_pass else 'FAIL'}")
                results['r1_regime'] = {'green_sharpe': s_g, 'red_sharpe': s_r, 'gap': round(gap, 3), 'pass': bool(r1_pass)}
            else:
                r1_pass = False
                results['r1_regime'] = {'pass': False, 'reason': 'insufficient data in one regime'}
        else:
            r1_pass = False
            results['r1_regime'] = {'pass': False, 'reason': 'insufficient common dates'}
    else:
        r1_pass = False
        results['r1_regime'] = {'pass': False, 'reason': 'no SPY data'}

    # ── SUMMARY ──
    gates = sum([
        results.get('permutation', {}).get('pass', False),
        results.get('sub_period', {}).get('pass', False),
        results.get('outlier', {}).get('pass', False),
        results.get('r1_regime', {}).get('pass', False),
    ])
    verdict = 'PASS' if gates >= 3 else 'FAIL'
    results['summary'] = {'gates_passed': gates, 'total': 4, 'verdict': verdict}
    log(f"\n  ADVERSARIAL SUMMARY: {gates}/4 gates -> {verdict}")

    return results


def make_plots(baseline_rets, ml_rets, output_dir):
    """Generate equity curve and comparison plots."""
    log("\n  Generating plots...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Equity curves
    ax = axes[0, 0]
    base_eq = INITIAL_CAPITAL * (1 + baseline_rets).cumprod()
    ml_eq = INITIAL_CAPITAL * (1 + ml_rets).cumprod()
    ax.plot(base_eq.index, base_eq.values, label='Baseline (Pure Mom)', alpha=0.8)
    ax.plot(ml_eq.index, ml_eq.values, label='ML-Enhanced', alpha=0.8)
    ax.set_title('Equity Curves')
    ax.legend()
    ax.set_ylabel('Portfolio Value ($)')
    ax.grid(True, alpha=0.3)

    # Drawdown
    ax = axes[0, 1]
    for rets, label in [(baseline_rets, 'Baseline'), (ml_rets, 'ML')]:
        cum = (1 + rets).cumprod()
        dd = cum / cum.cummax() - 1
        ax.fill_between(dd.index, dd.values, alpha=0.3, label=label)
    ax.set_title('Drawdowns')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Rolling Sharpe (12-month)
    ax = axes[1, 0]
    window = 12  # 12 monthly periods
    for rets, label in [(baseline_rets, 'Baseline'), (ml_rets, 'ML')]:
        roll_mean = rets.rolling(window).mean() * (252 / REBAL_PERIOD)
        roll_std = rets.rolling(window).std() * np.sqrt(252 / REBAL_PERIOD)
        roll_sharpe = roll_mean / (roll_std + 1e-8)
        ax.plot(roll_sharpe.index, roll_sharpe.values, label=label, alpha=0.8)
    ax.axhline(0, color='red', linestyle='--', alpha=0.5)
    ax.set_title('Rolling 12-Month Sharpe')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Monthly returns distribution
    ax = axes[1, 1]
    ax.hist(baseline_rets.values, bins=40, alpha=0.5, label='Baseline')
    ax.hist(ml_rets.values, bins=40, alpha=0.5, label='ML')
    ax.set_title('Return Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'cross_momentum_results.png', dpi=150)
    plt.close()
    log(f"  Saved plot to {output_dir / 'cross_momentum_results.png'}")


def main():
    t0 = time.time()
    log("=" * 80)
    log("ML CROSS-SECTIONAL MOMENTUM STRATEGY")
    log("Fama-French style with LightGBM enhancement")
    log(f"Universe: {len(UNIVERSE)} ETFs across sectors/regions/bonds/commodities/RE")
    log(f"Walk-forward: SLIDING {TRAIN_WINDOW}d train, {REBAL_PERIOD}d advance")
    log(f"Capital: ${INITIAL_CAPITAL:,} fixed (no DCA)")
    log("=" * 80)

    # Download data
    df = download_data()

    # Compute features
    features_df = compute_momentum_features(df)

    if len(features_df) < 100:
        log("ERROR: insufficient feature data")
        return

    # Run baseline
    base_rets, base_df = run_baseline(features_df, df)
    base_metrics = compute_metrics(base_rets, "Baseline (Pure Momentum)")

    log("\n  BASELINE METRICS:")
    for k, v in base_metrics.items():
        log(f"    {k}: {v}")

    # Run ML strategy
    ml_rets, ml_df, pred_df = run_ml_strategy(features_df, df)
    ml_metrics = compute_metrics(ml_rets, "ML-Enhanced Momentum")

    log("\n  ML STRATEGY METRICS:")
    for k, v in ml_metrics.items():
        log(f"    {k}: {v}")

    # Comparison
    log("\n" + "=" * 80)
    log("COMPARISON: BASELINE vs ML")
    log("=" * 80)
    log(f"  {'Metric':<20} {'Baseline':>12} {'ML':>12} {'Delta':>12}")
    log(f"  {'-'*56}")
    for key in ['sharpe', 'sortino', 'cagr', 'max_dd', 'profit_factor', 'win_rate', 'calmar']:
        b_val = base_metrics.get(key, 0)
        m_val = ml_metrics.get(key, 0)
        delta = m_val - b_val
        log(f"  {key:<20} {b_val:>12} {m_val:>12} {delta:>+12.3f}")

    # Adversarial validation on ML strategy
    if len(pred_df) > 0:
        adv_results = run_adversarial(ml_rets, pred_df, df)
    else:
        log("  WARNING: No predictions for adversarial validation")
        adv_results = {}

    # Generate plots
    if len(base_rets) > 10 and len(ml_rets) > 10:
        # Align dates for plotting
        common = base_rets.index.intersection(ml_rets.index)
        if len(common) > 10:
            make_plots(base_rets.loc[common], ml_rets.loc[common], OUTPUT)

    # Save results
    final_results = {
        'strategy': 'ML Cross-Sectional Momentum',
        'universe_size': len(UNIVERSE),
        'baseline_metrics': base_metrics,
        'ml_metrics': ml_metrics,
        'adversarial': adv_results,
        'config': {
            'train_window': TRAIN_WINDOW,
            'rebal_period': REBAL_PERIOD,
            'cost_bps': REBAL_COST_BPS,
            'capital': INITIAL_CAPITAL,
            'n_permutations': N_PERMUTATIONS,
        },
        'runtime_seconds': round(time.time() - t0, 1),
    }

    results_path = OUTPUT / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2, default=str)

    elapsed = time.time() - t0
    log(f"\n{'=' * 80}")
    log(f"DONE in {elapsed:.0f}s")
    log(f"Results saved to {OUTPUT}")
    log(f"{'=' * 80}")

    # Final verdict
    verdict = adv_results.get('summary', {}).get('verdict', 'N/A')
    ml_sharpe = ml_metrics.get('sharpe', 0)
    log(f"\n  VERDICT: ML Sharpe={ml_sharpe:.3f}, Adversarial={verdict}")
    if ml_sharpe > 1.0 and verdict == 'PASS':
        log("  >>> PROMISING — worth further investigation")
    elif ml_sharpe > 0.5:
        log("  >>> MARGINAL — needs improvement or different angle")
    else:
        log("  >>> WEAK — consider killing this approach")


if __name__ == '__main__':
    main()
