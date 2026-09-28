#!/usr/bin/env python3
"""
ML Cross-Asset Lead-Lag Exploitation
======================================
CONCEPT: Some assets systematically lead equity moves by hours/days.
Copper & AUD lead equity, credit spreads lead equity vol, Treasury
moves lead equity rotation. This strategy exploits those lead-lag
relationships using LightGBM walk-forward on daily data.

Features: 1d/2d/5d lagged returns of 17 cross-asset tickers to predict
next-day SPY direction/magnitude.

Trade mapping:
  - Strong up signal  -> UPRO (3x bull)
  - Neutral signal    -> SPY
  - Strong down signal -> SHY (cash proxy)

Walk-forward: SLIDING 252d train, 21d test, 1-day gap (HC #0).
Fixed $100K, NO DCA.
Full adversarial: permutation 100x, sub-period 4-block, outlier trim,
R1 regime stratification (gap threshold 0.50).
Cost: 10bps round-trip per trade.
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
from scipy import stats

warnings.filterwarnings('ignore')
np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
INITIAL_CAPITAL   = 100_000
TRAIN_WINDOW      = 252        # ~1 year sliding
TEST_WINDOW       = 21         # ~1 month
LABEL_GAP         = 1          # 1-day gap between train end and test start
N_PERMUTATIONS    = 100
TRADE_COST_BPS    = 10         # 10bps per trade
OUTLIER_TRIM_PCT  = 5          # trim top/bottom 5%
REGIME_GAP_THRESH = 0.50       # R1 regime test threshold

BASE   = Path("/home/jupiter/Lvl3Quant")
OUTPUT = BASE / "output" / "ml_cross_asset_leadlag"
OUTPUT.mkdir(parents=True, exist_ok=True)

# Tickers for cross-asset features
FEATURE_TICKERS = [
    'SPY', 'QQQ', 'IWM',           # US equity
    'GLD', 'SLV', 'CPER',          # Metals/commodities
    'TLT', 'IEF', 'SHY',          # Treasuries
    'HYG', 'LQD',                  # Credit
    'UUP', 'FXA',                  # FX
    'USO', 'UNG',                  # Energy
    'BTC-USD',                     # Crypto
]
VIX_TICKER = '^VIX'

# Trading instruments
TRADE_TICKERS = ['SPY', 'UPRO', 'SHY']

LAG_DAYS = [1, 2, 5]


def log(msg):
    print(msg)
    sys.stdout.flush()


def download_data():
    log("=" * 80)
    log("STEP 1: DOWNLOADING DATA")
    log("=" * 80)

    all_tickers = list(set(FEATURE_TICKERS + TRADE_TICKERS + [VIX_TICKER]))
    log(f"Downloading {len(all_tickers)} tickers: {all_tickers}")

    data = {}
    for tk in all_tickers:
        try:
            df = yf.download(tk, start='2014-01-01', end='2026-07-20',
                             progress=False, auto_adjust=True)
            if len(df) > 100:
                # Handle MultiIndex columns from newer yfinance
                if isinstance(df.columns, pd.MultiIndex):
                    close = df['Close'][tk] if tk in df['Close'].columns else df['Close'].iloc[:, 0]
                else:
                    close = df['Close']
                data[tk] = close
                log(f"  {tk}: {len(df)} rows ({df.index[0].date()} to {df.index[-1].date()})")
            else:
                log(f"  {tk}: SKIPPED (only {len(df)} rows)")
        except Exception as e:
            log(f"  {tk}: FAILED ({e})")

    prices = pd.DataFrame(data)
    prices = prices.ffill().dropna(how='all')
    log(f"\nCombined price matrix: {prices.shape[0]} days x {prices.shape[1]} tickers")
    log(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}")

    # Drop rows where SPY is missing (need target)
    prices = prices.dropna(subset=['SPY'])
    return prices


def build_features(prices):
    log("\n" + "=" * 80)
    log("STEP 2: BUILDING LAGGED FEATURES")
    log("=" * 80)

    returns = prices.pct_change()
    features = pd.DataFrame(index=prices.index)

    # Lagged returns for each feature ticker
    available_feature_tickers = [t for t in FEATURE_TICKERS if t in returns.columns]
    vix_col = VIX_TICKER if VIX_TICKER in returns.columns else None

    for tk in available_feature_tickers:
        for lag in LAG_DAYS:
            # Return over lag days, then shift by 1 to avoid lookahead
            ret = returns[tk].rolling(lag).sum().shift(1)
            features[f'{tk}_ret{lag}d_lag1'] = ret

    # VIX level and changes (not returns - VIX is already a derivative)
    if vix_col:
        vix = prices[vix_col]
        features['VIX_level_lag1'] = vix.shift(1)
        features['VIX_chg1d_lag1'] = vix.diff(1).shift(1)
        features['VIX_chg5d_lag1'] = vix.diff(5).shift(1)
        # VIX percentile (rolling 63d)
        features['VIX_pctile_lag1'] = vix.rolling(63).apply(
            lambda x: stats.percentileofscore(x, x.iloc[-1]) / 100, raw=False
        ).shift(1)

    # Cross-asset ratios (known leading indicators)
    if 'CPER' in prices.columns and 'GLD' in prices.columns:
        copper_gold = (prices['CPER'] / prices['GLD']).pct_change(5).shift(1)
        features['copper_gold_5d_lag1'] = copper_gold

    if 'HYG' in prices.columns and 'LQD' in prices.columns:
        credit_spread_proxy = (prices['HYG'] / prices['LQD']).pct_change(5).shift(1)
        features['credit_spread_5d_lag1'] = credit_spread_proxy

    if 'TLT' in prices.columns and 'SHY' in prices.columns:
        yield_curve_proxy = (prices['TLT'] / prices['SHY']).pct_change(5).shift(1)
        features['yield_curve_5d_lag1'] = yield_curve_proxy

    if 'IWM' in prices.columns and 'SPY' in prices.columns:
        risk_appetite = (prices['IWM'] / prices['SPY']).pct_change(5).shift(1)
        features['risk_appetite_5d_lag1'] = risk_appetite

    # Volatility features
    features['SPY_vol5d_lag1'] = returns['SPY'].rolling(5).std().shift(1)
    features['SPY_vol21d_lag1'] = returns['SPY'].rolling(21).std().shift(1)

    # Target: next-day SPY return (no shift - this IS the forward return)
    target = returns['SPY'].shift(-1)  # tomorrow's return

    # Drop NaN rows
    valid = features.dropna().index.intersection(target.dropna().index)
    features = features.loc[valid]
    target = target.loc[valid]

    log(f"Feature matrix: {features.shape[0]} rows x {features.shape[1]} features")
    log(f"Sample features: {list(features.columns[:10])} ...")

    return features, target


def walk_forward_backtest(features, target, prices):
    log("\n" + "=" * 80)
    log("STEP 3: WALK-FORWARD BACKTEST (SLIDING 252d train, 21d test, 1d gap)")
    log("=" * 80)

    predictions = []
    actuals = []
    dates = []

    n = len(features)
    n_folds = 0
    i = TRAIN_WINDOW

    while i + LABEL_GAP + TEST_WINDOW <= n:
        # SLIDING window: train on [i-TRAIN_WINDOW : i], gap of 1 day, test on [i+gap : i+gap+TEST_WINDOW]
        train_start = i - TRAIN_WINDOW
        train_end = i
        test_start = i + LABEL_GAP
        test_end = min(test_start + TEST_WINDOW, n)

        X_train = features.iloc[train_start:train_end]
        y_train = target.iloc[train_start:train_end]
        X_test = features.iloc[test_start:test_end]
        y_test = target.iloc[test_start:test_end]

        # Drop any remaining NaN
        valid_train = X_train.dropna().index.intersection(y_train.dropna().index)
        X_train = X_train.loc[valid_train]
        y_train = y_train.loc[valid_train]

        if len(X_train) < 100 or len(X_test) == 0:
            i += TEST_WINDOW
            continue

        # Train LightGBM
        dtrain = lgb.Dataset(X_train, label=y_train, free_raw_data=False)

        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.7,
            'bagging_fraction': 0.7,
            'bagging_freq': 5,
            'min_child_samples': 20,
            'verbose': -1,
            'n_jobs': 2,  # Keep CPU moderate
            'seed': 42,
        }

        callbacks = [lgb.log_evaluation(period=0)]
        model = lgb.train(params, dtrain, num_boost_round=200, callbacks=callbacks)
        preds = model.predict(X_test)

        predictions.extend(preds)
        actuals.extend(y_test.values)
        dates.extend(y_test.index.tolist())

        n_folds += 1
        i += TEST_WINDOW

    log(f"Completed {n_folds} walk-forward folds")
    log(f"Total OOT predictions: {len(predictions)}")

    # Build results DataFrame
    results = pd.DataFrame({
        'date': dates,
        'prediction': predictions,
        'actual_spy_return': actuals,
    }).set_index('date').sort_index()

    # Feature importance from last model
    importance = pd.DataFrame({
        'feature': features.columns,
        'importance': model.feature_importance(importance_type='gain'),
    }).sort_values('importance', ascending=False)

    return results, importance


def apply_trading_logic(results, prices):
    log("\n" + "=" * 80)
    log("STEP 4: APPLYING TRADING LOGIC")
    log("=" * 80)

    # Prediction thresholds for position sizing
    pred_std = results['prediction'].std()
    strong_up_thresh = pred_std * 0.5
    strong_down_thresh = -pred_std * 0.5

    log(f"Prediction std: {pred_std:.6f}")
    log(f"Strong up threshold: {strong_up_thresh:.6f}")
    log(f"Strong down threshold: {strong_down_thresh:.6f}")

    # Get returns for trading instruments
    trade_returns = prices[['SPY']].pct_change()
    if 'UPRO' in prices.columns:
        trade_returns['UPRO'] = prices['UPRO'].pct_change()
    else:
        # Approximate UPRO as 3x SPY daily
        trade_returns['UPRO'] = trade_returns['SPY'] * 3
        log("  UPRO approximated as 3x SPY daily returns")

    if 'SHY' in prices.columns:
        trade_returns['SHY'] = prices['SHY'].pct_change()
    else:
        trade_returns['SHY'] = 0.0001  # ~2.5% annual risk-free approx

    # Assign positions
    results['position'] = 'SPY'  # default
    results.loc[results['prediction'] > strong_up_thresh, 'position'] = 'UPRO'
    results.loc[results['prediction'] < strong_down_thresh, 'position'] = 'SHY'

    # Calculate strategy returns
    strategy_returns = []
    trade_costs = []
    prev_position = None

    for idx, row in results.iterrows():
        pos = row['position']

        # Transaction cost if position changed
        cost = TRADE_COST_BPS / 10000 if pos != prev_position and prev_position is not None else 0.0
        prev_position = pos

        # Get actual return for the chosen instrument
        if idx in trade_returns.index:
            if pos == 'UPRO':
                ret = trade_returns.loc[idx, 'UPRO'] if 'UPRO' in trade_returns.columns else row['actual_spy_return'] * 3
            elif pos == 'SHY':
                ret = trade_returns.loc[idx, 'SHY'] if 'SHY' in trade_returns.columns else 0.0001
            else:
                ret = row['actual_spy_return']
        else:
            ret = row['actual_spy_return']

        strategy_returns.append(ret - cost)
        trade_costs.append(cost)

    results['strategy_return'] = strategy_returns
    results['trade_cost'] = trade_costs
    results['cumulative_strategy'] = (1 + results['strategy_return']).cumprod()
    results['cumulative_spy'] = (1 + results['actual_spy_return']).cumprod()

    # Position distribution
    pos_counts = results['position'].value_counts()
    log(f"\nPosition distribution:")
    for pos, count in pos_counts.items():
        log(f"  {pos}: {count} days ({count/len(results)*100:.1f}%)")

    n_trades = (results['trade_cost'] > 0).sum()
    total_cost = results['trade_cost'].sum()
    log(f"Total trades (position changes): {n_trades}")
    log(f"Total transaction costs: {total_cost*100:.2f}%")

    return results


def compute_metrics(returns_series, label="Strategy"):
    """Compute risk-adjusted performance metrics."""
    r = returns_series.dropna()
    n_days = len(r)
    if n_days < 10:
        return {}

    ann_factor = 252
    mean_daily = r.mean()
    std_daily = r.std()

    # Sharpe
    sharpe = (mean_daily / std_daily) * np.sqrt(ann_factor) if std_daily > 0 else 0

    # Sortino (downside deviation)
    downside = r[r < 0]
    downside_std = downside.std() if len(downside) > 0 else std_daily
    sortino = (mean_daily / downside_std) * np.sqrt(ann_factor) if downside_std > 0 else 0

    # CAGR
    total_return = (1 + r).prod()
    n_years = n_days / ann_factor
    cagr = total_return ** (1 / n_years) - 1 if n_years > 0 else 0

    # Max drawdown
    cum = (1 + r).cumprod()
    rolling_max = cum.expanding().max()
    drawdown = (cum - rolling_max) / rolling_max
    max_dd = drawdown.min()

    # Calmar
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    # Win rate
    wr = (r > 0).sum() / n_days

    # Profit factor
    gross_profit = r[r > 0].sum()
    gross_loss = abs(r[r < 0].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    return {
        'label': label,
        'n_days': n_days,
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'cagr': round(cagr * 100, 2),
        'max_dd': round(max_dd * 100, 2),
        'calmar': round(calmar, 3),
        'win_rate': round(wr * 100, 2),
        'profit_factor': round(pf, 3),
        'ann_vol': round(std_daily * np.sqrt(ann_factor) * 100, 2),
        'total_return': round((total_return - 1) * 100, 2),
    }


def permutation_test(features, target, results, prices):
    """Permutation test: shuffle signal-to-date mapping (not returns)."""
    log("\n" + "=" * 80)
    log("STEP 5: PERMUTATION TEST (100 shuffles of signal-to-date mapping)")
    log("=" * 80)

    actual_sharpe = compute_metrics(results['strategy_return'])['sharpe']
    log(f"Actual strategy Sharpe: {actual_sharpe:.3f}")

    perm_sharpes = []
    for i in range(N_PERMUTATIONS):
        # Shuffle the prediction-to-date mapping
        shuffled_preds = results['prediction'].values.copy()
        np.random.shuffle(shuffled_preds)

        # Re-apply trading logic with shuffled predictions
        pred_std = np.std(shuffled_preds)
        strong_up = pred_std * 0.5
        strong_down = -pred_std * 0.5

        perm_returns = []
        prev_pos = None
        for j, (idx, row) in enumerate(results.iterrows()):
            pred = shuffled_preds[j]
            if pred > strong_up:
                pos = 'UPRO'
            elif pred < strong_down:
                pos = 'SHY'
            else:
                pos = 'SPY'

            cost = TRADE_COST_BPS / 10000 if pos != prev_pos and prev_pos is not None else 0.0
            prev_pos = pos

            # Use actual returns for each instrument
            if pos == 'UPRO':
                ret = row['actual_spy_return'] * 3  # approx
            elif pos == 'SHY':
                ret = 0.0001
            else:
                ret = row['actual_spy_return']

            perm_returns.append(ret - cost)

        perm_r = pd.Series(perm_returns)
        perm_sharpe = (perm_r.mean() / perm_r.std()) * np.sqrt(252) if perm_r.std() > 0 else 0
        perm_sharpes.append(perm_sharpe)

        if (i + 1) % 25 == 0:
            log(f"  Completed {i+1}/{N_PERMUTATIONS} permutations")

    perm_sharpes = np.array(perm_sharpes)
    p_value = (perm_sharpes >= actual_sharpe).mean()
    log(f"\nPermutation test results:")
    log(f"  Actual Sharpe: {actual_sharpe:.3f}")
    log(f"  Permuted mean Sharpe: {perm_sharpes.mean():.3f} +/- {perm_sharpes.std():.3f}")
    log(f"  Permuted max Sharpe: {perm_sharpes.max():.3f}")
    log(f"  p-value: {p_value:.4f} ({'SIGNIFICANT' if p_value < 0.05 else 'NOT SIGNIFICANT'})")

    return {
        'actual_sharpe': actual_sharpe,
        'perm_mean_sharpe': round(perm_sharpes.mean(), 3),
        'perm_std_sharpe': round(perm_sharpes.std(), 3),
        'perm_max_sharpe': round(perm_sharpes.max(), 3),
        'p_value': round(p_value, 4),
        'significant': p_value < 0.05,
        'perm_sharpes': perm_sharpes.tolist(),
    }


def regime_test(results):
    """R1 regime-agnostic test: stratify by green/red SPY days."""
    log("\n" + "=" * 80)
    log("STEP 6: R1 REGIME TEST (Green/Red day stratification)")
    log("=" * 80)

    # Classify days by SPY close-to-close
    results['regime'] = np.where(results['actual_spy_return'] >= 0, 'green', 'red')

    green_days = results[results['regime'] == 'green']
    red_days = results[results['regime'] == 'red']

    green_metrics = compute_metrics(green_days['strategy_return'], 'Green Days')
    red_metrics = compute_metrics(red_days['strategy_return'], 'Red Days')

    log(f"\nGreen days ({len(green_days)}):")
    log(f"  Sharpe: {green_metrics.get('sharpe', 'N/A')}, WR: {green_metrics.get('win_rate', 'N/A')}%, PF: {green_metrics.get('profit_factor', 'N/A')}")

    log(f"Red days ({len(red_days)}):")
    log(f"  Sharpe: {red_metrics.get('sharpe', 'N/A')}, WR: {red_metrics.get('win_rate', 'N/A')}%, PF: {red_metrics.get('profit_factor', 'N/A')}")

    # Regime gap test
    sharpe_green = green_metrics.get('sharpe', 0)
    sharpe_red = red_metrics.get('sharpe', 0)
    max_sharpe = max(abs(sharpe_green), abs(sharpe_red))
    gap = abs(sharpe_green - sharpe_red) / max_sharpe if max_sharpe > 0 else 0

    passed = gap <= REGIME_GAP_THRESH
    log(f"\nRegime gap: |{sharpe_green:.3f} - {sharpe_red:.3f}| / max(|{sharpe_green:.3f}|, |{sharpe_red:.3f}|) = {gap:.3f}")
    log(f"Threshold: {REGIME_GAP_THRESH}")
    log(f"R1 REGIME TEST: {'PASSED' if passed else 'FAILED'}")

    return {
        'green_metrics': green_metrics,
        'red_metrics': red_metrics,
        'regime_gap': round(gap, 3),
        'threshold': REGIME_GAP_THRESH,
        'passed': passed,
        'n_green': len(green_days),
        'n_red': len(red_days),
    }


def subperiod_consistency(results):
    """Split into 4 blocks, check CV of Sharpe."""
    log("\n" + "=" * 80)
    log("STEP 7: SUB-PERIOD CONSISTENCY (4 blocks)")
    log("=" * 80)

    n = len(results)
    block_size = n // 4
    blocks = []

    for i in range(4):
        start = i * block_size
        end = start + block_size if i < 3 else n
        block = results.iloc[start:end]
        metrics = compute_metrics(block['strategy_return'], f'Block {i+1}')
        blocks.append(metrics)

        date_range = f"{block.index[0].date()} to {block.index[-1].date()}"
        log(f"  Block {i+1} ({date_range}): Sharpe={metrics.get('sharpe', 'N/A')}, "
            f"CAGR={metrics.get('cagr', 'N/A')}%, WR={metrics.get('win_rate', 'N/A')}%")

    sharpes = [b['sharpe'] for b in blocks if 'sharpe' in b]
    if len(sharpes) >= 2 and np.mean(sharpes) != 0:
        cv = np.std(sharpes) / abs(np.mean(sharpes))
    else:
        cv = float('inf')

    n_positive = sum(1 for s in sharpes if s > 0)
    log(f"\nSharpe CV across blocks: {cv:.3f}")
    log(f"Positive Sharpe blocks: {n_positive}/4")
    log(f"Consistency: {'GOOD' if cv < 1.0 and n_positive >= 3 else 'POOR'}")

    return {
        'blocks': blocks,
        'sharpe_cv': round(cv, 3),
        'n_positive_sharpe': n_positive,
        'consistent': cv < 1.0 and n_positive >= 3,
    }


def outlier_robustness(results):
    """Trim top/bottom 5% of returns and re-evaluate."""
    log("\n" + "=" * 80)
    log("STEP 8: OUTLIER ROBUSTNESS (trim top/bottom 5%)")
    log("=" * 80)

    r = results['strategy_return']
    lower = r.quantile(OUTLIER_TRIM_PCT / 100)
    upper = r.quantile(1 - OUTLIER_TRIM_PCT / 100)
    trimmed = r.clip(lower, upper)

    full_metrics = compute_metrics(r, 'Full')
    trimmed_metrics = compute_metrics(trimmed, 'Trimmed')

    log(f"Full:    Sharpe={full_metrics['sharpe']}, CAGR={full_metrics['cagr']}%")
    log(f"Trimmed: Sharpe={trimmed_metrics['sharpe']}, CAGR={trimmed_metrics['cagr']}%")

    sharpe_change = abs(full_metrics['sharpe'] - trimmed_metrics['sharpe'])
    robust = sharpe_change < 0.3
    log(f"Sharpe change after trimming: {sharpe_change:.3f} ({'ROBUST' if robust else 'NOT ROBUST'})")

    return {
        'full_metrics': full_metrics,
        'trimmed_metrics': trimmed_metrics,
        'sharpe_change': round(sharpe_change, 3),
        'robust': robust,
    }


def generate_plots(results, importance, perm_results):
    """Generate analysis plots."""
    log("\n" + "=" * 80)
    log("STEP 9: GENERATING PLOTS")
    log("=" * 80)

    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle('Cross-Asset Lead-Lag Strategy Analysis', fontsize=16, fontweight='bold')

    # 1. Cumulative returns
    ax = axes[0, 0]
    ax.plot(results.index, results['cumulative_strategy'], label='Strategy', linewidth=1.5)
    ax.plot(results.index, results['cumulative_spy'], label='SPY B&H', linewidth=1.0, alpha=0.7)
    ax.set_title('Cumulative Returns')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylabel('Growth of $1')

    # 2. Drawdown
    ax = axes[0, 1]
    cum = results['cumulative_strategy']
    rolling_max = cum.expanding().max()
    dd = (cum - rolling_max) / rolling_max
    ax.fill_between(results.index, dd, 0, alpha=0.5, color='red')
    ax.set_title('Strategy Drawdown')
    ax.set_ylabel('Drawdown %')
    ax.grid(True, alpha=0.3)

    # 3. Rolling Sharpe (63d)
    ax = axes[1, 0]
    rolling_sharpe = results['strategy_return'].rolling(63).apply(
        lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0
    )
    ax.plot(results.index, rolling_sharpe, linewidth=1.0)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.set_title('Rolling 63-day Sharpe')
    ax.grid(True, alpha=0.3)

    # 4. Feature importance (top 20)
    ax = axes[1, 1]
    top_imp = importance.head(20)
    ax.barh(range(len(top_imp)), top_imp['importance'].values)
    ax.set_yticks(range(len(top_imp)))
    ax.set_yticklabels(top_imp['feature'].values, fontsize=7)
    ax.set_title('Top 20 Feature Importance (Gain)')
    ax.invert_yaxis()

    # 5. Prediction distribution
    ax = axes[2, 0]
    ax.hist(results['prediction'], bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(x=results['prediction'].std() * 0.5, color='green', linestyle='--', label='Up thresh')
    ax.axvline(x=-results['prediction'].std() * 0.5, color='red', linestyle='--', label='Down thresh')
    ax.set_title('Prediction Distribution')
    ax.legend()

    # 6. Permutation test
    ax = axes[2, 1]
    ax.hist(perm_results['perm_sharpes'], bins=30, alpha=0.7, edgecolor='black', label='Permuted')
    ax.axvline(x=perm_results['actual_sharpe'], color='red', linewidth=2, label=f'Actual ({perm_results["actual_sharpe"]:.3f})')
    ax.set_title(f'Permutation Test (p={perm_results["p_value"]:.4f})')
    ax.legend()

    plt.tight_layout()
    plot_path = OUTPUT / 'analysis_plots.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    log(f"Saved plots to {plot_path}")


def main():
    t0 = time.time()
    log("=" * 80)
    log("ML CROSS-ASSET LEAD-LAG EXPLOITATION STRATEGY")
    log(f"Started: {pd.Timestamp.now()}")
    log("=" * 80)

    # Step 1: Download data
    prices = download_data()

    # Step 2: Build features
    features, target = build_features(prices)

    # Step 3: Walk-forward backtest
    results, importance = walk_forward_backtest(features, target, prices)

    if len(results) == 0:
        log("ERROR: No predictions generated. Exiting.")
        return

    # Step 4: Apply trading logic
    results = apply_trading_logic(results, prices)

    # Step 5: Compute overall metrics
    log("\n" + "=" * 80)
    log("OVERALL PERFORMANCE")
    log("=" * 80)

    strat_metrics = compute_metrics(results['strategy_return'], 'Strategy')
    spy_metrics = compute_metrics(results['actual_spy_return'], 'SPY B&H')

    log(f"\n{'Metric':<20} {'Strategy':>12} {'SPY B&H':>12}")
    log("-" * 44)
    for key in ['sharpe', 'sortino', 'cagr', 'max_dd', 'calmar', 'win_rate', 'profit_factor', 'ann_vol', 'total_return']:
        s_val = strat_metrics.get(key, 'N/A')
        b_val = spy_metrics.get(key, 'N/A')
        suffix = '%' if key in ['cagr', 'max_dd', 'win_rate', 'ann_vol', 'total_return'] else ''
        log(f"  {key:<18} {str(s_val)+suffix:>12} {str(b_val)+suffix:>12}")

    # Step 6: Permutation test
    perm_results = permutation_test(features, target, results, prices)

    # Step 7: Regime test
    regime_results = regime_test(results)

    # Step 8: Sub-period consistency
    subperiod_results = subperiod_consistency(results)

    # Step 9: Outlier robustness
    outlier_results = outlier_robustness(results)

    # Step 10: Generate plots
    generate_plots(results, importance, perm_results)

    # Step 11: Save all results
    log("\n" + "=" * 80)
    log("SAVING RESULTS")
    log("=" * 80)

    # Save predictions CSV
    results.to_csv(OUTPUT / 'predictions.csv')

    # Save feature importance
    importance.to_csv(OUTPUT / 'feature_importance.csv', index=False)

    # Compile full report
    report = {
        'strategy': 'ML Cross-Asset Lead-Lag Exploitation',
        'run_time': round(time.time() - t0, 1),
        'data_period': f"{results.index[0].date()} to {results.index[-1].date()}",
        'n_oot_days': len(results),
        'walk_forward': {
            'train_window': TRAIN_WINDOW,
            'test_window': TEST_WINDOW,
            'label_gap': LABEL_GAP,
            'window_type': 'SLIDING (never expanding)',
        },
        'strategy_metrics': strat_metrics,
        'benchmark_metrics': spy_metrics,
        'permutation_test': {k: v for k, v in perm_results.items() if k != 'perm_sharpes'},
        'regime_test': regime_results,
        'subperiod_consistency': subperiod_results,
        'outlier_robustness': outlier_results,
        'feature_importance_top10': importance.head(10).to_dict('records'),
        'position_distribution': results['position'].value_counts().to_dict(),
        'transaction_costs': {
            'cost_bps': TRADE_COST_BPS,
            'n_trades': int((results['trade_cost'] > 0).sum()),
            'total_cost_pct': round(results['trade_cost'].sum() * 100, 3),
        },
    }

    with open(OUTPUT / 'report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)

    log(f"\nResults saved to {OUTPUT}")

    # Final verdict
    log("\n" + "=" * 80)
    log("FINAL VERDICT")
    log("=" * 80)

    checks = {
        'Sharpe > 0': strat_metrics['sharpe'] > 0,
        'Beats SPY Sharpe': strat_metrics['sharpe'] > spy_metrics['sharpe'],
        'Permutation p < 0.05': perm_results['p_value'] < 0.05,
        'R1 Regime PASSED': regime_results['passed'],
        'Sub-period consistent': subperiod_results['consistent'],
        'Outlier robust': outlier_results['robust'],
        'MaxDD < -40%': strat_metrics['max_dd'] > -40,
    }

    for check, passed in checks.items():
        status = 'PASS' if passed else 'FAIL'
        log(f"  [{status}] {check}")

    n_pass = sum(checks.values())
    n_total = len(checks)
    log(f"\nPassed {n_pass}/{n_total} checks")

    if n_pass >= 5:
        log("VERDICT: PROMISING - Worth further investigation")
    elif n_pass >= 3:
        log("VERDICT: MARGINAL - Some edge detected but not robust")
    else:
        log("VERDICT: REJECT - No reliable edge found")

    log(f"\nTotal runtime: {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
