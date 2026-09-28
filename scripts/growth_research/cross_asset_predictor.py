"""
Cross-Asset Leading Indicator Predictive Model
================================================
Uses cross-asset signals (commodities, bonds, currencies, credit, vol)
to predict SPY/QQQ returns over 1-20 day horizons.

HC #0  : Sliding walk-forward (252d train, 63d test, slide 21d)
HC #428: Regime-agnostic validation (R1 gap < 0.50)
HC #432: MFE-within-horizon (daily holds — no concern here)
HC #705: Built-in permutation test, regime test, sub-period consistency

Models: LGBM (primary), optional MLP
Validation: Walk-forward sliding, permutation test, regime test
"""

import os
import sys
import json
import warnings
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Determine output dir cross-platform
if sys.platform == 'win32':
    BASE_DIR = Path(r"C:\Users\claude\Lvl3Quant")
else:
    BASE_DIR = Path("/home/jupiter/Lvl3Quant")

OUTPUT_DIR = BASE_DIR / "output" / "growth_research" / "cross_asset_predictor"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================
START_DATE = "2010-01-01"
END_DATE = "2026-07-16"

# Walk-forward params (SLIDING, HC #0)
TRAIN_DAYS = 252
TEST_DAYS = 63
SLIDE_DAYS = 21

# Target horizons
HORIZONS = [1, 5, 10, 20]

# Prediction targets
TARGETS = ["SPY", "QQQ"]

# Lookback windows for feature generation
LOOKBACKS = [5, 10, 20, 60]

# Permutation test
N_PERMUTATIONS = 100
PERM_P_THRESHOLD = 0.05

# R1 regime gap threshold
R1_GAP_THRESHOLD = 0.50

# LGBM hyperparameters
LGBM_PARAMS = {
    'objective': 'regression',
    'metric': 'mse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'n_estimators': 300,
    'early_stopping_rounds': 30,
    'min_child_samples': 20,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
}

# ETF tickers for features (HC #710: broad market signals)
FEATURE_TICKERS = {
    'GLD': 'Gold',
    'SLV': 'Silver',
    'COPX': 'Copper (miners proxy)',
    'USO': 'Oil',
    'UNG': 'Natural Gas',
    'DBA': 'Agriculture',
    'XLE': 'Energy sector',
    'UUP': 'Dollar',
    'TLT': 'Long-term bonds',
    'IEF': 'Intermediate bonds',
    'HYG': 'High-yield credit',
    'LQD': 'Investment-grade credit',
    'EEM': 'Emerging Markets',
    'IWM': 'Small Caps (Russell 2000)',
    'BTC-USD': 'Bitcoin',
}

# Tickers we need to download (features + targets)
# NOTE: ^VIX and ^VIX3M removed — yfinance hangs on Yahoo index tickers.
# VIX features replaced by VIXY ETF as proxy.
ALL_TICKERS = list(FEATURE_TICKERS.keys()) + ['SPY', 'QQQ', 'VIXY']


# ============================================================================
# DATA DOWNLOAD
# ============================================================================
def download_data():
    """Download all required data via yfinance bulk download."""
    import yfinance as yf

    print("=" * 70)
    print("DOWNLOADING DATA")
    print("=" * 70)

    cache_file = OUTPUT_DIR / "data_cache.pkl"
    # Use cache if less than 12 hours old
    if cache_file.exists():
        age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
        if age_hours < 12:
            print(f"  Using cached data ({age_hours:.1f}h old)")
            return pd.read_pickle(cache_file)

    all_data = {}

    # Bulk download all tickers (no index tickers — they hang)
    print(f"  Bulk downloading {len(ALL_TICKERS)} tickers...")
    try:
        raw = yf.download(ALL_TICKERS, start=START_DATE, end=END_DATE,
                         progress=False, auto_adjust=True, threads=True)
        if isinstance(raw.columns, pd.MultiIndex):
            closes = raw['Close']
        else:
            closes = raw[['Close']]
            closes.columns = ALL_TICKERS

        for ticker in closes.columns:
            s = closes[ticker].dropna()
            if len(s) > 100:
                all_data[ticker] = s
                print(f"    {ticker}: OK ({len(s)} rows)")
            else:
                print(f"    {ticker}: SKIP ({len(s)} rows)")
    except Exception as e:
        print(f"  Bulk download FAILED: {e}")
        return pd.DataFrame()

    # Build price DataFrame
    prices = pd.DataFrame(all_data)
    prices.index = pd.to_datetime(prices.index.tz_localize(None) if prices.index.tz else prices.index)
    prices = prices.sort_index()

    # Forward-fill missing dates (some ETFs don't trade on same days)
    prices = prices.ffill().dropna(how='all')

    print(f"\n  Final price matrix: {prices.shape[0]} days x {prices.shape[1]} tickers")
    print(f"  Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"  Available tickers: {list(prices.columns)}")

    prices.to_pickle(cache_file)
    return prices


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================
def compute_features(prices):
    """
    Build cross-asset feature matrix from price data.
    Features per asset: momentum, RSI, z-score, rate of change
    Plus: credit spread, yield curve slope, VIX features, BTC sentiment
    """
    print("\n" + "=" * 70)
    print("BUILDING FEATURES")
    print("=" * 70)

    features = pd.DataFrame(index=prices.index)
    returns = prices.pct_change()

    # Per-asset features
    feature_assets = [t for t in FEATURE_TICKERS.keys() if t in prices.columns]

    for ticker in feature_assets:
        px = prices[ticker]
        ret = returns[ticker]
        label = ticker.replace('-', '_').replace('^', '')

        for lb in LOOKBACKS:
            # Momentum (cumulative return over lookback)
            features[f'{label}_mom_{lb}d'] = px.pct_change(lb)

            # RSI
            features[f'{label}_rsi_{lb}d'] = _compute_rsi(px, lb)

            # Z-score of price relative to rolling mean
            roll_mean = px.rolling(lb).mean()
            roll_std = px.rolling(lb).std()
            features[f'{label}_zscore_{lb}d'] = (px - roll_mean) / (roll_std + 1e-10)

            # Volatility (realized vol over lookback)
            features[f'{label}_vol_{lb}d'] = ret.rolling(lb).std() * np.sqrt(252)

    # Credit spread: HYG vs LQD (lower ratio = wider spread = risk-off)
    if 'HYG' in prices.columns and 'LQD' in prices.columns:
        credit_ratio = prices['HYG'] / prices['LQD']
        for lb in LOOKBACKS:
            features[f'credit_spread_mom_{lb}d'] = credit_ratio.pct_change(lb)
            roll_mean = credit_ratio.rolling(lb).mean()
            roll_std = credit_ratio.rolling(lb).std()
            features[f'credit_spread_zscore_{lb}d'] = (credit_ratio - roll_mean) / (roll_std + 1e-10)

    # Yield curve slope: TLT vs IEF (steepening = risk-on)
    if 'TLT' in prices.columns and 'IEF' in prices.columns:
        curve_slope = prices['TLT'] / prices['IEF']
        for lb in LOOKBACKS:
            features[f'yield_curve_mom_{lb}d'] = curve_slope.pct_change(lb)
            roll_mean = curve_slope.rolling(lb).mean()
            roll_std = curve_slope.rolling(lb).std()
            features[f'yield_curve_zscore_{lb}d'] = (curve_slope - roll_mean) / (roll_std + 1e-10)

    # VIX proxy features (using VIXY ETF since yfinance hangs on ^VIX)
    if 'VIXY' in prices.columns:
        vixy = prices['VIXY']
        features['vixy_level'] = vixy
        features['vixy_log'] = np.log(vixy + 1)
        for lb in LOOKBACKS:
            features[f'vixy_zscore_{lb}d'] = (vixy - vixy.rolling(lb).mean()) / (vixy.rolling(lb).std() + 1e-10)
            features[f'vixy_change_{lb}d'] = vixy.pct_change(lb)

    # Cross-asset relative strength features
    # Gold vs Dollar (inverse relationship)
    if 'GLD' in prices.columns and 'UUP' in prices.columns:
        gld_uup = prices['GLD'] / prices['UUP']
        for lb in LOOKBACKS:
            features[f'gold_dollar_ratio_mom_{lb}d'] = gld_uup.pct_change(lb)

    # Oil vs Dollar
    if 'USO' in prices.columns and 'UUP' in prices.columns:
        oil_uup = prices['USO'] / prices['UUP']
        for lb in LOOKBACKS:
            features[f'oil_dollar_ratio_mom_{lb}d'] = oil_uup.pct_change(lb)

    # Gold/Silver ratio (risk indicator: rising = risk-off)
    if 'GLD' in prices.columns and 'SLV' in prices.columns:
        gld_slv = prices['GLD'] / prices['SLV']
        for lb in LOOKBACKS:
            features[f'gold_silver_ratio_mom_{lb}d'] = gld_slv.pct_change(lb)
            roll_mean = gld_slv.rolling(lb).mean()
            roll_std = gld_slv.rolling(lb).std()
            features[f'gold_silver_ratio_zscore_{lb}d'] = (gld_slv - roll_mean) / (roll_std + 1e-10)

    # Copper/Gold ratio (economic indicator: rising = growth, falling = recession)
    if 'COPX' in prices.columns and 'GLD' in prices.columns:
        copx_gld = prices['COPX'] / prices['GLD']
        for lb in LOOKBACKS:
            features[f'copper_gold_ratio_mom_{lb}d'] = copx_gld.pct_change(lb)

    # EM vs DM (risk appetite)
    if 'EEM' in prices.columns and 'SPY' in prices.columns:
        eem_spy = prices['EEM'] / prices['SPY']
        for lb in LOOKBACKS:
            features[f'em_dm_ratio_mom_{lb}d'] = eem_spy.pct_change(lb)

    # Small vs Large cap (risk appetite)
    if 'IWM' in prices.columns and 'SPY' in prices.columns:
        iwm_spy = prices['IWM'] / prices['SPY']
        for lb in LOOKBACKS:
            features[f'small_large_ratio_mom_{lb}d'] = iwm_spy.pct_change(lb)

    # SPY/QQQ own features (autoregressive — be careful of leakage)
    for target in TARGETS:
        if target in prices.columns:
            px = prices[target]
            ret = returns[target]
            for lb in LOOKBACKS:
                features[f'{target}_mom_{lb}d'] = px.pct_change(lb)
                features[f'{target}_vol_{lb}d'] = ret.rolling(lb).std() * np.sqrt(252)
                features[f'{target}_rsi_{lb}d'] = _compute_rsi(px, lb)

    # Drop any columns that are all NaN
    features = features.dropna(axis=1, how='all')

    # Count features
    n_feats = features.shape[1]
    print(f"  Total features: {n_feats}")
    print(f"  Feature groups:")
    for group in ['mom', 'rsi', 'zscore', 'vol', 'credit', 'yield', 'vix', 'ratio']:
        count = sum(1 for c in features.columns if group in c.lower())
        if count > 0:
            print(f"    {group}: {count}")

    return features


def _compute_rsi(prices, window):
    """Compute RSI indicator."""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))


def compute_targets(prices):
    """Compute forward returns for target assets."""
    targets = {}
    for target in TARGETS:
        if target not in prices.columns:
            continue
        for h in HORIZONS:
            # Forward return over h days (NO LEAKAGE — shift forward)
            fwd_ret = prices[target].pct_change(h).shift(-h)
            targets[f'{target}_fwd_{h}d'] = fwd_ret
    return pd.DataFrame(targets, index=prices.index)


# ============================================================================
# WALK-FORWARD VALIDATION (SLIDING, HC #0)
# ============================================================================
def walk_forward_lgbm(features, targets, target_col, prices):
    """
    Sliding walk-forward with LGBM.
    Train: 252d, Test: 63d, Slide: 21d.
    Returns concatenated OOT predictions.
    """
    import lightgbm as lgb

    # Align features and target
    valid_idx = features.dropna().index.intersection(targets[target_col].dropna().index)
    X = features.loc[valid_idx].copy()
    y = targets.loc[valid_idx, target_col].copy()

    # Sort by date
    X = X.sort_index()
    y = y.loc[X.index]

    n = len(X)
    all_preds = []
    all_actuals = []
    all_dates = []
    fold_results = []

    fold_idx = 0
    start = 0

    while start + TRAIN_DAYS + TEST_DAYS <= n:
        train_end = start + TRAIN_DAYS
        test_end = min(train_end + TEST_DAYS, n)

        X_train = X.iloc[start:train_end]
        y_train = y.iloc[start:train_end]
        X_test = X.iloc[train_end:test_end]
        y_test = y.iloc[train_end:test_end]

        # Drop NaN rows
        train_mask = X_train.notna().all(axis=1) & y_train.notna()
        test_mask = X_test.notna().all(axis=1) & y_test.notna()

        X_tr = X_train[train_mask]
        y_tr = y_train[train_mask]
        X_te = X_test[test_mask]
        y_te = y_test[test_mask]

        if len(X_tr) < 100 or len(X_te) < 10:
            start += SLIDE_DAYS
            continue

        # Train LGBM
        train_data = lgb.Dataset(X_tr, label=y_tr)
        val_data = lgb.Dataset(X_te.iloc[:len(X_te)//3], label=y_te.iloc[:len(y_te)//3])

        params = {k: v for k, v in LGBM_PARAMS.items() if k not in ['n_estimators', 'early_stopping_rounds']}
        model = lgb.train(
            params,
            train_data,
            num_boost_round=LGBM_PARAMS['n_estimators'],
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(LGBM_PARAMS['early_stopping_rounds'], verbose=False),
                       lgb.log_evaluation(period=0)],
        )

        preds = model.predict(X_te)

        all_preds.extend(preds)
        all_actuals.extend(y_te.values)
        all_dates.extend(X_te.index.tolist())

        # Per-fold IC
        if len(preds) > 5:
            ic = np.corrcoef(preds, y_te.values)[0, 1]
            fold_results.append({
                'fold': fold_idx,
                'train_start': X_tr.index[0].strftime('%Y-%m-%d'),
                'train_end': X_tr.index[-1].strftime('%Y-%m-%d'),
                'test_start': X_te.index[0].strftime('%Y-%m-%d'),
                'test_end': X_te.index[-1].strftime('%Y-%m-%d'),
                'ic': ic,
                'n_train': len(X_tr),
                'n_test': len(X_te),
            })

        fold_idx += 1
        start += SLIDE_DAYS

    # Get feature importance from last model
    feat_imp = dict(zip(X.columns, model.feature_importance(importance_type='gain')))

    return (np.array(all_preds), np.array(all_actuals),
            all_dates, fold_results, feat_imp, model)


# ============================================================================
# VALIDATION SUITE (HC #428, HC #705)
# ============================================================================
def run_permutation_test(features, targets, target_col, actual_ic, prices):
    """
    Permutation test: shuffle target N times, compute IC each time.
    p-value = fraction of shuffled ICs >= actual IC.
    """
    import lightgbm as lgb

    print(f"  Running permutation test ({N_PERMUTATIONS} shuffles)...")
    shuffled_ics = []

    valid_idx = features.dropna().index.intersection(targets[target_col].dropna().index)
    X = features.loc[valid_idx].copy()
    y = targets.loc[valid_idx, target_col].copy()
    X = X.sort_index()
    y = y.loc[X.index]

    n = len(X)

    for i in range(N_PERMUTATIONS):
        # Shuffle targets (break predictive relationship)
        y_shuffled = y.sample(frac=1.0, replace=False).values

        # Quick train/test on last fold only (speed)
        split = n - TEST_DAYS
        X_tr = X.iloc[:split]
        y_tr_s = y_shuffled[:split]
        X_te = X.iloc[split:]
        y_te_s = y_shuffled[split:]

        train_mask = X_tr.notna().all(axis=1)
        test_mask = X_te.notna().all(axis=1)
        X_tr_clean = X_tr[train_mask]
        y_tr_clean = y_tr_s[train_mask.values]
        X_te_clean = X_te[test_mask]
        y_te_clean = y_te_s[test_mask.values]

        if len(X_tr_clean) < 50 or len(X_te_clean) < 10:
            continue

        params = {k: v for k, v in LGBM_PARAMS.items()
                  if k not in ['n_estimators', 'early_stopping_rounds']}
        train_data = lgb.Dataset(X_tr_clean, label=y_tr_clean)
        model = lgb.train(params, train_data, num_boost_round=50,
                         callbacks=[lgb.log_evaluation(period=0)])
        preds = model.predict(X_te_clean)
        ic = np.corrcoef(preds, y_te_clean)[0, 1]
        shuffled_ics.append(ic)

        if (i + 1) % 25 == 0:
            print(f"    {i+1}/{N_PERMUTATIONS} done")

    shuffled_ics = np.array(shuffled_ics)
    p_value = np.mean(np.abs(shuffled_ics) >= np.abs(actual_ic))
    return p_value, shuffled_ics


def regime_test(preds, actuals, dates, prices):
    """
    R1 regime-agnostic test (HC #428).
    Split OOT days into green/red based on SPY close-to-close.
    Require |Sharpe_green - Sharpe_red| / max(|Sharpe_green|, |Sharpe_red|) < 0.50.
    """
    if 'SPY' not in prices.columns:
        return None, None, None, None

    spy_ret = prices['SPY'].pct_change()

    # Map each prediction date to regime
    green_mask = []
    red_mask = []

    for d in dates:
        d = pd.Timestamp(d)
        if d in spy_ret.index:
            r = spy_ret.loc[d]
            if r >= 0:
                green_mask.append(True)
                red_mask.append(False)
            else:
                green_mask.append(False)
                red_mask.append(True)
        else:
            green_mask.append(False)
            red_mask.append(False)

    green_mask = np.array(green_mask)
    red_mask = np.array(red_mask)

    # Strategy returns: go long when pred > 0, short when pred < 0
    strat_ret = np.sign(preds) * actuals

    # Compute Sharpe per regime
    def _sharpe(rets):
        if len(rets) < 5:
            return 0.0
        return np.mean(rets) / (np.std(rets) + 1e-10) * np.sqrt(252)

    sharpe_green = _sharpe(strat_ret[green_mask]) if green_mask.sum() > 5 else 0.0
    sharpe_red = _sharpe(strat_ret[red_mask]) if red_mask.sum() > 5 else 0.0

    max_sharpe = max(abs(sharpe_green), abs(sharpe_red))
    gap = abs(sharpe_green - sharpe_red) / (max_sharpe + 1e-10) if max_sharpe > 0 else 0.0

    return gap, sharpe_green, sharpe_red, gap < R1_GAP_THRESHOLD


def sub_period_consistency(preds, actuals, dates):
    """
    Split OOT into 3 sub-periods and check IC consistency.
    Warn if any sub-period has opposite sign.
    """
    n = len(preds)
    third = n // 3
    results = []
    for i, label in enumerate(['Early', 'Mid', 'Late']):
        s = i * third
        e = (i + 1) * third if i < 2 else n
        p = preds[s:e]
        a = actuals[s:e]
        ic = np.corrcoef(p, a)[0, 1] if len(p) > 5 else 0.0
        results.append({'period': label, 'ic': ic, 'n': len(p)})
    return results


def outlier_robustness(preds, actuals):
    """
    Remove top 5 most profitable trades and check if edge persists.
    """
    strat_ret = np.sign(preds) * actuals
    sorted_idx = np.argsort(strat_ret)[::-1]

    # Remove top 5
    mask = np.ones(len(strat_ret), dtype=bool)
    mask[sorted_idx[:5]] = False

    trimmed = strat_ret[mask]
    full_sharpe = np.mean(strat_ret) / (np.std(strat_ret) + 1e-10) * np.sqrt(252)
    trimmed_sharpe = np.mean(trimmed) / (np.std(trimmed) + 1e-10) * np.sqrt(252)

    return full_sharpe, trimmed_sharpe, trimmed_sharpe > 0


# ============================================================================
# PERFORMANCE METRICS
# ============================================================================
def compute_performance(preds, actuals, dates):
    """Compute risk-adjusted performance metrics for a long/short signal."""
    # Strategy: long when pred > 0, short when pred < 0
    positions = np.sign(preds)
    strat_ret = positions * actuals

    # Basic metrics
    total_ret = np.sum(strat_ret)
    n_days = len(strat_ret)
    ann_ret = np.mean(strat_ret) * 252

    # Sharpe
    sharpe = np.mean(strat_ret) / (np.std(strat_ret) + 1e-10) * np.sqrt(252)

    # Sortino
    downside = strat_ret[strat_ret < 0]
    sortino = np.mean(strat_ret) / (np.std(downside) + 1e-10) * np.sqrt(252) if len(downside) > 0 else 0.0

    # Win rate
    wins = np.sum(strat_ret > 0)
    losses = np.sum(strat_ret < 0)
    wr = wins / (wins + losses) if (wins + losses) > 0 else 0.0

    # Profit factor
    gross_profit = np.sum(strat_ret[strat_ret > 0])
    gross_loss = abs(np.sum(strat_ret[strat_ret < 0]))
    pf = gross_profit / (gross_loss + 1e-10)

    # Max drawdown (multiplicative equity curve — fixes v1 bug)
    equity = np.cumprod(1 + strat_ret)
    running_max = np.maximum.accumulate(equity)
    drawdown_pct = (equity - running_max) / running_max
    max_dd = np.min(drawdown_pct)  # Now in proper percentage terms

    # IC (information coefficient)
    ic = np.corrcoef(preds, actuals)[0, 1] if len(preds) > 5 else 0.0

    # Rank IC
    from scipy.stats import spearmanr
    rank_ic, _ = spearmanr(preds, actuals) if len(preds) > 5 else (0.0, 1.0)

    # CAGR
    equity_final = equity[-1] if len(equity) > 0 else 1.0
    years = n_days / 252
    cagr = (equity_final ** (1 / years) - 1) if years > 0 and equity_final > 0 else 0.0

    # Calmar
    calmar = cagr / abs(max_dd) if abs(max_dd) > 1e-6 else 0.0

    return {
        'n_days': n_days,
        'ann_return': round(ann_ret, 4),
        'cagr': round(cagr, 4),
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'calmar': round(calmar, 3),
        'win_rate': round(wr, 4),
        'profit_factor': round(pf, 3),
        'max_drawdown_pct': round(max_dd * 100, 2),
        'ic': round(ic, 4),
        'rank_ic': round(rank_ic, 4),
        'total_return': round(total_ret, 4),
        'long_pct': round(np.mean(positions > 0), 3),
        'short_pct': round(np.mean(positions < 0), 3),
    }


# ============================================================================
# MAIN
# ============================================================================
def main():
    t0 = time.time()

    print("=" * 70)
    print("CROSS-ASSET LEADING INDICATOR PREDICTIVE MODEL")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # 1. Download data
    prices = download_data()

    # 2. Build features
    features = compute_features(prices)

    # 3. Build targets
    targets = compute_targets(prices)

    # 4. Drop rows where features not yet available (lookback warmup)
    min_lookback = max(LOOKBACKS) + 5
    features = features.iloc[min_lookback:]
    targets = targets.loc[features.index]

    # Fill remaining NaN features with 0 (safe for tree models)
    features = features.fillna(0)

    print(f"\n  Usable data: {len(features)} days")
    print(f"  Date range: {features.index[0].date()} to {features.index[-1].date()}")

    # 5. Run walk-forward for each target/horizon combo
    all_results = {}

    for target_col in targets.columns:
        print(f"\n{'='*70}")
        print(f"TARGET: {target_col}")
        print(f"{'='*70}")

        target_valid = targets[target_col].dropna()
        if len(target_valid) < TRAIN_DAYS + TEST_DAYS + 100:
            print(f"  SKIP: insufficient data ({len(target_valid)} rows)")
            continue

        # Walk-forward
        print(f"\n  Walk-forward LGBM (train={TRAIN_DAYS}d, test={TEST_DAYS}d, slide={SLIDE_DAYS}d)...")
        preds, actuals, dates, fold_results, feat_imp, model = \
            walk_forward_lgbm(features, targets, target_col, prices)

        if len(preds) < 50:
            print(f"  SKIP: too few OOT predictions ({len(preds)})")
            continue

        # Performance
        perf = compute_performance(preds, actuals, dates)
        print(f"\n  CONCAT OOT RESULTS ({perf['n_days']} days):")
        print(f"    IC:           {perf['ic']:.4f}")
        print(f"    Rank IC:      {perf['rank_ic']:.4f}")
        print(f"    Sharpe:       {perf['sharpe']:.3f}")
        print(f"    Sortino:      {perf['sortino']:.3f}")
        print(f"    Win Rate:     {perf['win_rate']:.1%}")
        print(f"    Profit Factor:{perf['profit_factor']:.3f}")
        print(f"    CAGR:         {perf['cagr']:.2%}")
        print(f"    Calmar:       {perf['calmar']:.3f}")
        print(f"    Max DD:       {perf['max_drawdown_pct']:.2f}%")
        print(f"    Long/Short:   {perf['long_pct']:.0%} / {perf['short_pct']:.0%}")

        # Permutation test
        perm_p, shuffled_ics = run_permutation_test(
            features, targets, target_col, perf['ic'], prices)
        perm_pass = perm_p < PERM_P_THRESHOLD
        print(f"\n  PERMUTATION TEST: p={perm_p:.3f} {'PASS' if perm_pass else 'FAIL'}")
        if not perm_pass:
            print(f"  *** WARNING: Signal NOT statistically significant (p={perm_p:.3f} >= {PERM_P_THRESHOLD})")

        # Regime test (R1)
        r1_gap, sharpe_green, sharpe_red, r1_pass = regime_test(preds, actuals, dates, prices)
        if r1_gap is not None:
            print(f"\n  REGIME TEST (R1): gap={r1_gap:.3f} {'PASS' if r1_pass else 'FAIL'}")
            print(f"    Sharpe (green days): {sharpe_green:.3f}")
            print(f"    Sharpe (red days):   {sharpe_red:.3f}")
            if not r1_pass:
                print(f"  *** WARNING: Regime-dependent signal! Gap {r1_gap:.3f} >= {R1_GAP_THRESHOLD}")

        # Sub-period consistency
        sub_results = sub_period_consistency(preds, actuals, dates)
        print(f"\n  SUB-PERIOD CONSISTENCY:")
        sign_flips = 0
        for sr in sub_results:
            marker = ""
            if sr['ic'] * perf['ic'] < 0:
                marker = " *** SIGN FLIP"
                sign_flips += 1
            print(f"    {sr['period']:5s}: IC={sr['ic']:.4f} (n={sr['n']}){marker}")
        sub_pass = sign_flips == 0
        if not sub_pass:
            print(f"  *** WARNING: IC sign flip in {sign_flips} sub-period(s)")

        # Outlier robustness
        full_sharpe, trimmed_sharpe, outlier_pass = outlier_robustness(preds, actuals)
        print(f"\n  OUTLIER ROBUSTNESS:")
        print(f"    Full Sharpe:    {full_sharpe:.3f}")
        print(f"    Trimmed Sharpe: {trimmed_sharpe:.3f} (top 5 trades removed)")
        print(f"    {'PASS' if outlier_pass else 'FAIL'}: {'Edge persists' if outlier_pass else 'Edge driven by outliers'}")

        # Feature importance (top 20)
        sorted_feats = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:20]
        print(f"\n  TOP 20 FEATURES (by gain):")
        for fname, fimp in sorted_feats:
            print(f"    {fname:40s} {fimp:.1f}")

        # Overall verdict
        all_pass = perm_pass and (r1_pass if r1_pass is not None else True) and sub_pass and outlier_pass
        verdict = "PASS" if all_pass else "FAIL"

        if all_pass:
            print(f"\n  ========================================")
            print(f"  VERDICT: **PASS** — Signal appears genuine")
            print(f"  ========================================")
        else:
            failures = []
            if not perm_pass:
                failures.append(f"permutation (p={perm_p:.3f})")
            if r1_pass is not None and not r1_pass:
                failures.append(f"regime (gap={r1_gap:.3f})")
            if not sub_pass:
                failures.append(f"sub-period ({sign_flips} flips)")
            if not outlier_pass:
                failures.append("outlier-driven")
            print(f"\n  ========================================")
            print(f"  VERDICT: **FAIL** — {', '.join(failures)}")
            print(f"  ========================================")

        # Store results
        all_results[target_col] = {
            'performance': perf,
            'permutation_p': round(perm_p, 4),
            'permutation_pass': perm_pass,
            'r1_gap': round(r1_gap, 4) if r1_gap is not None else None,
            'r1_pass': r1_pass,
            'sub_period': sub_results,
            'sub_period_pass': sub_pass,
            'outlier_full_sharpe': round(full_sharpe, 3),
            'outlier_trimmed_sharpe': round(trimmed_sharpe, 3),
            'outlier_pass': outlier_pass,
            'verdict': verdict,
            'n_folds': len(fold_results),
            'fold_ics': [fr['ic'] for fr in fold_results],
            'top_features': [(f, round(v, 1)) for f, v in sorted_feats[:10]],
        }

    # ======================================================================
    # SUMMARY
    # ======================================================================
    elapsed = time.time() - t0

    print(f"\n\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"Runtime: {elapsed/60:.1f} minutes")
    print(f"Targets evaluated: {len(all_results)}")

    pass_count = sum(1 for r in all_results.values() if r['verdict'] == 'PASS')
    fail_count = sum(1 for r in all_results.values() if r['verdict'] == 'FAIL')

    print(f"\nResults by target:")
    print(f"{'Target':25s} {'IC':>8s} {'Sharpe':>8s} {'Sortino':>8s} {'WR':>8s} {'PF':>8s} {'Perm_p':>8s} {'R1_gap':>8s} {'Verdict':>8s}")
    print("-" * 105)
    for target_col, res in sorted(all_results.items()):
        p = res['performance']
        print(f"{target_col:25s} {p['ic']:8.4f} {p['sharpe']:8.3f} {p['sortino']:8.3f} "
              f"{p['win_rate']:8.1%} {p['profit_factor']:8.3f} {res['permutation_p']:8.3f} "
              f"{res['r1_gap'] if res['r1_gap'] is not None else 'N/A':>8} {res['verdict']:>8s}")

    if pass_count > 0:
        print(f"\n  {pass_count} target(s) PASSED all validation gates.")
    if fail_count > 0:
        print(f"  {fail_count} target(s) FAILED one or more gates.")
    if pass_count == 0:
        print(f"\n  *** ALL CONFIGS FAILED. No statistically significant cross-asset signal found. ***")
        print(f"  This is an HONEST result — no inflation.")

    # Save results
    summary = {
        'run_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'config': {
            'train_days': TRAIN_DAYS,
            'test_days': TEST_DAYS,
            'slide_days': SLIDE_DAYS,
            'horizons': HORIZONS,
            'lookbacks': LOOKBACKS,
            'n_permutations': N_PERMUTATIONS,
            'start_date': START_DATE,
            'end_date': END_DATE,
        },
        'results': {},
        'runtime_minutes': round(elapsed / 60, 1),
    }

    # Make results JSON serializable
    for k, v in all_results.items():
        res_clean = {}
        for kk, vv in v.items():
            if isinstance(vv, (np.floating, np.integer)):
                res_clean[kk] = float(vv)
            elif isinstance(vv, np.ndarray):
                res_clean[kk] = [float(x) for x in vv]
            elif isinstance(vv, list):
                res_clean[kk] = []
                for item in vv:
                    if isinstance(item, dict):
                        res_clean[kk].append({kkk: float(vvv) if isinstance(vvv, (np.floating, np.integer)) else vvv for kkk, vvv in item.items()})
                    elif isinstance(item, tuple):
                        res_clean[kk].append([str(item[0]), float(item[1])])
                    elif isinstance(item, (np.floating, np.integer)):
                        res_clean[kk].append(float(item))
                    else:
                        res_clean[kk].append(item)
            elif isinstance(vv, np.bool_):
                res_clean[kk] = bool(vv)
            else:
                res_clean[kk] = vv
        summary['results'][k] = res_clean

    summary_file = OUTPUT_DIR / "cross_asset_predictor_results.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Results saved to: {summary_file}")

    print(f"\n{'='*70}")
    print("DONE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
