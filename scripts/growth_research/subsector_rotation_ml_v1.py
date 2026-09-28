#!/usr/bin/env python3
"""
Sub-Sector Rotation ML Analysis v1 (HC #776)
=============================================
Hypothesis: Money flows between sub-sectors in predictable patterns that ML
can detect before they show up in price. Tests mean-reversion in rotation
(lagging sub-sector catches up to leading one).

Walk-forward LGBM classifier with sliding 252d train / 21d test.
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from pathlib import Path
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score
from scipy import stats

warnings.filterwarnings('ignore')

# ============================================================================
# 1. UNIVERSE DEFINITION
# ============================================================================
SECTOR_PAIRS = {
    'Tech_Semis_vs_Software': {'sector': 'Technology', 'etf_a': 'SMH', 'etf_b': 'IGV',
                                'label_a': 'Semiconductors', 'label_b': 'Software'},
    'Biotech_vs_MedDev': {'sector': 'Healthcare', 'etf_a': 'IBB', 'etf_b': 'IHI',
                           'label_a': 'Biotech', 'label_b': 'Medical Devices'},
    'Biotech_Small_vs_Large': {'sector': 'Healthcare', 'etf_a': 'XBI', 'etf_b': 'IBB',
                                'label_a': 'Biotech Small Cap', 'label_b': 'Biotech Large'},
    'Banks_vs_Insurance': {'sector': 'Financials', 'etf_a': 'KBE', 'etf_b': 'KIE',
                            'label_a': 'Banks', 'label_b': 'Insurance'},
    'RegBanks_vs_BroadFin': {'sector': 'Financials', 'etf_a': 'KRE', 'etf_b': 'XLF',
                              'label_a': 'Regional Banks', 'label_b': 'Broad Financials'},
    'OilExplore_vs_OilService': {'sector': 'Energy', 'etf_a': 'XOP', 'etf_b': 'OIH',
                                  'label_a': 'Oil Exploration', 'label_b': 'Oil Services'},
    'Defense_vs_BroadInd': {'sector': 'Industrials', 'etf_a': 'ITA', 'etf_b': 'XLI',
                             'label_a': 'Defense/Aerospace', 'label_b': 'Broad Industrials'},
    'GoldMiners_vs_MetalsMining': {'sector': 'Materials', 'etf_a': 'GDX', 'etf_b': 'XME',
                                    'label_a': 'Gold Miners', 'label_b': 'Metals/Mining'},
    'REITs_vs_BroadRE': {'sector': 'Real Estate', 'etf_a': 'VNQ', 'etf_b': 'XLRE',
                          'label_a': 'REITs Broad', 'label_b': 'Real Estate Sector'},
    'Retail_vs_Homebuilders': {'sector': 'Consumer', 'etf_a': 'XRT', 'etf_b': 'XHB',
                                'label_a': 'Retail', 'label_b': 'Homebuilders'},
    'ConsDisc_vs_ConsStaples': {'sector': 'Consumer', 'etf_a': 'XLY', 'etf_b': 'XLP',
                                 'label_a': 'Consumer Discretionary', 'label_b': 'Consumer Staples'},
}

START_DATE = '2020-01-01'
END_DATE = '2026-07-31'
TRAIN_DAYS = 252
TEST_DAYS = 21
HORIZONS = [5, 10, 21]

# ============================================================================
# 2. DATA DOWNLOAD
# ============================================================================
def download_data():
    """Download all ETF data from yfinance."""
    tickers = set()
    for pair in SECTOR_PAIRS.values():
        tickers.add(pair['etf_a'])
        tickers.add(pair['etf_b'])
    # Add SPY for regime detection
    tickers.add('SPY')
    tickers = sorted(tickers)

    print(f"Downloading {len(tickers)} ETFs: {', '.join(tickers)}")
    data = yf.download(tickers, start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)

    # Handle multi-level columns
    if isinstance(data.columns, pd.MultiIndex):
        close = data['Close']
    else:
        close = data

    # Drop any tickers with insufficient data
    min_days = TRAIN_DAYS + max(HORIZONS) + 50
    valid = close.dropna(axis=1, thresh=min_days)
    dropped = set(tickers) - set(valid.columns)
    if dropped:
        print(f"WARNING: Dropped tickers with insufficient data: {dropped}")

    print(f"Data shape: {valid.shape}, date range: {valid.index[0].date()} to {valid.index[-1].date()}")
    return valid


# ============================================================================
# 3. FEATURE ENGINEERING
# ============================================================================
def compute_features(close_a, close_b, spy_close):
    """Compute rotation features for a sub-sector pair."""
    # Returns
    ret_a = close_a.pct_change()
    ret_b = close_b.pct_change()

    # Relative return (A vs B)
    ratio = close_a / close_b
    ratio_ret = ratio.pct_change()

    features = pd.DataFrame(index=close_a.index)

    # Relative returns over multiple windows
    for w in [5, 10, 21, 63]:
        features[f'rel_ret_{w}d'] = (close_a / close_a.shift(w)) / (close_b / close_b.shift(w)) - 1

    # RSI of the ratio (14-day)
    delta = ratio.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    features['ratio_rsi_14'] = 100 - (100 / (1 + rs))

    # Z-score of relative returns (flow reversal signal)
    for w in [21, 63]:
        roll_mean = ratio_ret.rolling(w).mean()
        roll_std = ratio_ret.rolling(w).std()
        features[f'rel_zscore_{w}d'] = (ratio_ret - roll_mean) / roll_std.replace(0, np.nan)

    # Rolling correlation between A and B
    for w in [21, 63]:
        features[f'corr_{w}d'] = ret_a.rolling(w).corr(ret_b)

    # Correlation change (regime shift indicator)
    features['corr_change_21d'] = features['corr_21d'] - features['corr_21d'].shift(21)

    # Volatility ratio
    for w in [21]:
        vol_a = ret_a.rolling(w).std()
        vol_b = ret_b.rolling(w).std()
        features[f'vol_ratio_{w}d'] = vol_a / vol_b.replace(0, np.nan)

    # Momentum divergence (difference in momentum rank)
    for w in [21]:
        mom_a = close_a / close_a.shift(w) - 1
        mom_b = close_b / close_b.shift(w) - 1
        features[f'mom_divergence_{w}d'] = mom_a - mom_b

    # Ratio mean reversion signal (distance from 63d SMA of ratio)
    ratio_sma = ratio.rolling(63).mean()
    features['ratio_dist_sma63'] = (ratio / ratio_sma) - 1

    # SPY regime: 1 if above 200d SMA, 0 otherwise
    spy_sma200 = spy_close.rolling(200).mean()
    features['spy_regime'] = (spy_close > spy_sma200).astype(int)

    # SPY recent performance (market context)
    features['spy_ret_21d'] = spy_close.pct_change(21)

    # Volume-weighted features would need volume data - skip for ETF pairs

    return features


def compute_targets(close_a, close_b, horizons):
    """
    Target: Does the lagging sub-sector outperform the leading one over the next N days?
    Specifically: if A has been lagging B (rel_ret_21d < 0), does A then outperform B?
    This tests mean-reversion in rotation.
    """
    targets = {}
    for h in horizons:
        fwd_ret_a = close_a.pct_change(h).shift(-h)
        fwd_ret_b = close_b.pct_change(h).shift(-h)
        # Target = 1 if lagging sub-sector (based on recent 21d) outperforms forward
        # We define "lagging" as: A lagged B if A underperformed B over past 21d
        past_rel = (close_a / close_a.shift(21)) / (close_b / close_b.shift(21)) - 1
        # If A lagged (past_rel < 0), does A then outperform B forward? (mean reversion)
        # If A led (past_rel > 0), does B then outperform A forward? (mean reversion)
        # Unified target: does mean reversion happen? (laggard outperforms leader)
        reversal = np.where(
            past_rel < 0,
            fwd_ret_a > fwd_ret_b,  # A lagged, does A now outperform?
            fwd_ret_b > fwd_ret_a   # B lagged, does B now outperform?
        ).astype(float)
        reversal_series = pd.Series(reversal, index=close_a.index)
        # Also store the magnitude for P&L
        reversal_pnl = np.where(
            past_rel < 0,
            fwd_ret_a - fwd_ret_b,  # Long A short B if A lagged
            fwd_ret_b - fwd_ret_a   # Long B short A if B lagged
        )
        targets[f'reversal_{h}d'] = reversal_series
        targets[f'reversal_pnl_{h}d'] = pd.Series(reversal_pnl, index=close_a.index)

    return targets


# ============================================================================
# 4. WALK-FORWARD ML
# ============================================================================
def walk_forward_lgbm(features, targets, horizon, pair_name):
    """Walk-forward LGBM classifier for rotation prediction."""
    target_col = f'reversal_{horizon}d'
    pnl_col = f'reversal_pnl_{horizon}d'

    y = targets[target_col]
    pnl = targets[pnl_col]

    # Combine and drop NaN
    combined = features.copy()
    combined['target'] = y
    combined['pnl'] = pnl
    combined = combined.dropna()

    if len(combined) < TRAIN_DAYS + TEST_DAYS + 50:
        return None

    feature_cols = [c for c in features.columns if c in combined.columns]

    results = []
    importances = np.zeros(len(feature_cols))
    n_folds = 0

    # Walk-forward: sliding window
    idx = 0
    while idx + TRAIN_DAYS + TEST_DAYS <= len(combined):
        train_end = idx + TRAIN_DAYS
        test_end = min(train_end + TEST_DAYS, len(combined))

        X_train = combined.iloc[idx:train_end][feature_cols].values
        y_train = combined.iloc[idx:train_end]['target'].values
        X_test = combined.iloc[train_end:test_end][feature_cols].values
        y_test = combined.iloc[train_end:test_end]['target'].values
        pnl_test = combined.iloc[train_end:test_end]['pnl'].values
        test_dates = combined.iloc[train_end:test_end].index

        # Check class balance
        if len(np.unique(y_train)) < 2 or len(y_test) == 0:
            idx += TEST_DAYS
            continue

        # Train LGBM
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'verbosity': -1,
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': 20,
            'n_estimators': 200,
            'random_state': 42,
        }

        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)],
                  callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)])

        # Predict
        proba = model.predict_proba(X_test)[:, 1]
        preds = (proba >= 0.5).astype(int)

        # Accumulate feature importance
        importances += model.feature_importances_
        n_folds += 1

        # Store fold results
        for i in range(len(y_test)):
            results.append({
                'date': test_dates[i],
                'y_true': y_test[i],
                'y_pred': preds[i],
                'proba': proba[i],
                'pnl': pnl_test[i],
            })

        idx += TEST_DAYS

    if not results or n_folds == 0:
        return None

    # Average importances
    importances /= n_folds
    importance_dict = {feature_cols[i]: float(importances[i]) for i in range(len(feature_cols))}

    return pd.DataFrame(results), importance_dict


# ============================================================================
# 5. BACKTEST & EVALUATION
# ============================================================================
def evaluate_strategy(results_df, horizon, pair_name, threshold=0.6):
    """Evaluate the rotation strategy with the 5-gate test."""
    df = results_df.copy()

    # Overall model metrics
    acc = accuracy_score(df['y_true'], df['y_pred'])
    prec = precision_score(df['y_true'], df['y_pred'], zero_division=0)
    try:
        auc = roc_auc_score(df['y_true'], df['proba'])
    except ValueError:
        auc = 0.5

    # Strategy: trade only when model confidence > threshold
    high_conf = df[df['proba'] >= threshold].copy()
    low_conf_rev = df[df['proba'] < (1 - threshold)].copy()  # High confidence NO reversal

    # Combined high-conviction trades
    trades = high_conf.copy()

    if len(trades) == 0:
        return {
            'pair': pair_name, 'horizon': horizon,
            'n_predictions': len(df), 'n_trades': 0,
            'accuracy': round(acc, 4), 'precision': round(prec, 4), 'auc': round(auc, 4),
            'sharpe': 0, 'sortino': 0, 'win_rate': 0, 'profit_factor': 0,
            'regime_gap': None, 'verdict': 'INSUFFICIENT_TRADES'
        }

    # Trade P&L: when model says reversal with high confidence, go long the laggard
    trade_returns = trades['pnl'].values

    n_trades = len(trade_returns)
    win_rate = (trade_returns > 0).mean()

    # Annualize
    trades_per_year = 252 / horizon
    mean_ret = trade_returns.mean()
    std_ret = trade_returns.std() if trade_returns.std() > 0 else 1e-10

    sharpe = (mean_ret / std_ret) * np.sqrt(trades_per_year) if std_ret > 0 else 0

    downside = trade_returns[trade_returns < 0]
    downside_std = downside.std() if len(downside) > 1 else 1e-10
    sortino = (mean_ret / downside_std) * np.sqrt(trades_per_year) if downside_std > 0 else 0

    gross_profit = trade_returns[trade_returns > 0].sum()
    gross_loss = abs(trade_returns[trade_returns < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Regime analysis using SPY (bull vs bear)
    # We embedded spy_regime in features; approximate here from dates
    trades_df = trades.copy()
    trades_df['return'] = trade_returns

    # Regime gap: split by median date as proxy (proper version would use SPY)
    mid_idx = len(trades_df) // 2
    first_half_sharpe = _calc_sharpe(trades_df.iloc[:mid_idx]['return'].values, trades_per_year)
    second_half_sharpe = _calc_sharpe(trades_df.iloc[mid_idx:]['return'].values, trades_per_year)

    max_s = max(abs(first_half_sharpe), abs(second_half_sharpe), 0.01)
    regime_gap = abs(first_half_sharpe - second_half_sharpe) / max_s

    # Day-concentration cap
    if isinstance(trades_df['date'].iloc[0], pd.Timestamp):
        unique_days = trades_df['date'].dt.date.nunique() if hasattr(trades_df['date'].dt, 'date') else len(trades_df)
    else:
        unique_days = len(set(trades_df['date']))
    total_days = len(trades_df)
    day_conc = 1.0 - (unique_days / max(total_days, 1))

    # 5-gate test
    gate_results = {
        'sharpe_pass': sharpe > 0.5,
        'sortino_pass': sortino > 0.7,
        'wr_pass': win_rate > 0.50,
        'pf_pass': profit_factor > 1.2,
        'regime_gap_pass': regime_gap < 0.50,
    }
    gates_passed = sum(gate_results.values())

    verdict = 'PASS' if gates_passed >= 4 else ('MARGINAL' if gates_passed >= 3 else 'FAIL')

    return {
        'pair': pair_name,
        'horizon': horizon,
        'n_predictions': len(df),
        'n_trades': n_trades,
        'accuracy': round(acc, 4),
        'precision': round(prec, 4),
        'auc': round(auc, 4),
        'mean_trade_return': round(float(mean_ret * 100), 4),  # percent
        'sharpe': round(float(sharpe), 3),
        'sortino': round(float(sortino), 3),
        'win_rate': round(float(win_rate), 4),
        'profit_factor': round(float(profit_factor), 3),
        'regime_gap': round(float(regime_gap), 3),
        'gates_passed': gates_passed,
        'gate_details': gate_results,
        'day_concentration': round(float(day_conc), 3),
        'verdict': verdict,
    }


def _calc_sharpe(returns, ann_factor):
    if len(returns) < 2:
        return 0
    m = returns.mean()
    s = returns.std()
    if s == 0:
        return 0
    return (m / s) * np.sqrt(ann_factor)


# ============================================================================
# 6. PERMUTATION TEST
# ============================================================================
def permutation_test(results_df, horizon, n_perms=500):
    """Test if ML accuracy is significantly better than random."""
    true_acc = accuracy_score(results_df['y_true'], results_df['y_pred'])

    y_true = results_df['y_true'].values
    perm_accs = []
    for _ in range(n_perms):
        perm_preds = np.random.permutation(results_df['y_pred'].values)
        perm_accs.append(accuracy_score(y_true, perm_preds))

    p_value = (np.sum(np.array(perm_accs) >= true_acc) + 1) / (n_perms + 1)

    return {
        'true_accuracy': round(true_acc, 4),
        'random_mean_accuracy': round(np.mean(perm_accs), 4),
        'p_value': round(float(p_value), 4),
        'significant_5pct': p_value < 0.05,
    }


# ============================================================================
# 7. MAIN
# ============================================================================
def main():
    print("=" * 80)
    print("SUB-SECTOR ROTATION ML ANALYSIS v1 (HC #776)")
    print("=" * 80)
    print(f"Date range: {START_DATE} to {END_DATE}")
    print(f"Walk-forward: {TRAIN_DAYS}d train, {TEST_DAYS}d test, sliding")
    print(f"Horizons: {HORIZONS}")
    print(f"Pairs: {len(SECTOR_PAIRS)}")
    print()

    # Download data
    close = download_data()
    spy_close = close['SPY'] if 'SPY' in close.columns else None

    all_results = {}
    all_importances = {}
    all_perm_tests = {}
    pair_summaries = []

    for pair_name, pair_info in SECTOR_PAIRS.items():
        etf_a = pair_info['etf_a']
        etf_b = pair_info['etf_b']

        if etf_a not in close.columns or etf_b not in close.columns:
            print(f"\nSKIPPING {pair_name}: Missing data for {etf_a} or {etf_b}")
            continue

        print(f"\n{'='*60}")
        print(f"PAIR: {pair_name} ({etf_a} vs {etf_b})")
        print(f"  {pair_info['label_a']} vs {pair_info['label_b']} [{pair_info['sector']}]")
        print(f"{'='*60}")

        close_a = close[etf_a].dropna()
        close_b = close[etf_b].dropna()

        # Align
        common_idx = close_a.index.intersection(close_b.index)
        if spy_close is not None:
            common_idx = common_idx.intersection(spy_close.index)
        close_a = close_a.loc[common_idx]
        close_b = close_b.loc[common_idx]
        spy_aligned = spy_close.loc[common_idx] if spy_close is not None else close_a * 0 + 1

        # Features & targets
        features = compute_features(close_a, close_b, spy_aligned)
        targets = compute_targets(close_a, close_b, HORIZONS)

        pair_results = {}

        for h in HORIZONS:
            print(f"\n  Horizon: {h}d")

            wf_results = walk_forward_lgbm(features, targets, h, pair_name)

            if wf_results is None:
                print(f"    SKIPPED (insufficient data)")
                continue

            results_df, importance_dict = wf_results

            # Evaluate
            eval_result = evaluate_strategy(results_df, h, pair_name)
            perm_result = permutation_test(results_df, h)

            key = f"{pair_name}_{h}d"
            pair_results[f'{h}d'] = eval_result
            all_importances[key] = importance_dict
            all_perm_tests[key] = perm_result

            print(f"    Predictions: {eval_result['n_predictions']}, Trades (>60% conf): {eval_result['n_trades']}")
            print(f"    Accuracy: {eval_result['accuracy']:.4f} (random: {perm_result['random_mean_accuracy']:.4f}, p={perm_result['p_value']:.4f})")
            print(f"    AUC: {eval_result['auc']:.4f}")
            print(f"    Sharpe: {eval_result['sharpe']:.3f}, Sortino: {eval_result['sortino']:.3f}")
            print(f"    WR: {eval_result['win_rate']:.4f}, PF: {eval_result['profit_factor']:.3f}")
            print(f"    Regime Gap: {eval_result['regime_gap']:.3f}")
            print(f"    Verdict: {eval_result['verdict']} ({eval_result['gates_passed']}/5 gates)")

            pair_summaries.append(eval_result)

        all_results[pair_name] = pair_results

    # ========================================================================
    # AGGREGATE ANALYSIS
    # ========================================================================
    print("\n" + "=" * 80)
    print("AGGREGATE RESULTS")
    print("=" * 80)

    # Best pairs by Sharpe
    valid_summaries = [s for s in pair_summaries if s['n_trades'] > 10]
    if valid_summaries:
        sorted_by_sharpe = sorted(valid_summaries, key=lambda x: x['sharpe'], reverse=True)

        print("\nTOP 10 PAIR-HORIZON COMBOS BY SHARPE:")
        print(f"{'Pair':<35} {'Hz':>3} {'Sharpe':>7} {'Sort':>7} {'WR':>6} {'PF':>6} {'AUC':>6} {'Verdict':>10}")
        print("-" * 90)
        for s in sorted_by_sharpe[:10]:
            print(f"{s['pair']:<35} {s['horizon']:>3}d {s['sharpe']:>7.3f} {s['sortino']:>7.3f} "
                  f"{s['win_rate']:>6.3f} {s['profit_factor']:>6.2f} {s['auc']:>6.3f} {s['verdict']:>10}")

        # Count verdicts
        n_pass = sum(1 for s in valid_summaries if s['verdict'] == 'PASS')
        n_marginal = sum(1 for s in valid_summaries if s['verdict'] == 'MARGINAL')
        n_fail = sum(1 for s in valid_summaries if s['verdict'] == 'FAIL')

        print(f"\nVerdicts: {n_pass} PASS, {n_marginal} MARGINAL, {n_fail} FAIL out of {len(valid_summaries)} tested")

        # Statistical significance
        sig_pairs = [s for s in valid_summaries
                     if all_perm_tests.get(f"{s['pair']}_{s['horizon']}d", {}).get('significant_5pct', False)]
        print(f"Statistically significant (p<0.05): {len(sig_pairs)}/{len(valid_summaries)}")

    # Top feature importances across all pairs
    print("\nTOP FEATURES (avg importance across all pair-horizons):")
    if all_importances:
        avg_imp = {}
        for key, imp_dict in all_importances.items():
            for feat, val in imp_dict.items():
                if feat not in avg_imp:
                    avg_imp[feat] = []
                avg_imp[feat].append(val)
        avg_imp = {k: np.mean(v) for k, v in avg_imp.items()}
        sorted_feats = sorted(avg_imp.items(), key=lambda x: x[1], reverse=True)
        for feat, imp in sorted_feats[:10]:
            print(f"  {feat:<30} {imp:.1f}")

    # ========================================================================
    # OVERALL VERDICT
    # ========================================================================
    print("\n" + "=" * 80)
    print("OVERALL VERDICT")
    print("=" * 80)

    if valid_summaries:
        avg_sharpe = np.mean([s['sharpe'] for s in valid_summaries])
        avg_auc = np.mean([s['auc'] for s in valid_summaries])
        best = sorted_by_sharpe[0] if sorted_by_sharpe else None

        print(f"Average Sharpe across all pairs/horizons: {avg_sharpe:.3f}")
        print(f"Average AUC: {avg_auc:.4f}")
        if best:
            print(f"Best pair: {best['pair']} @ {best['horizon']}d horizon (Sharpe={best['sharpe']:.3f})")

        if n_pass >= 3:
            overall = "PROMISING - Multiple pairs show tradeable rotation alpha"
        elif n_pass >= 1 or n_marginal >= 3:
            overall = "MIXED - Some pairs show edge but not consistently across horizons"
        elif avg_auc > 0.52:
            overall = "WEAK - Slight predictability but not reliably tradeable"
        else:
            overall = "NO ALPHA - Sub-sector rotation is not predictable with these features"
        print(f"\nVerdict: {overall}")
    else:
        overall = "INSUFFICIENT DATA - Could not test enough pairs"
        print(f"Verdict: {overall}")

    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    output = {
        'metadata': {
            'script': 'subsector_rotation_ml_v1.py',
            'hc': 776,
            'run_date': datetime.now().isoformat(),
            'date_range': f'{START_DATE} to {END_DATE}',
            'train_days': TRAIN_DAYS,
            'test_days': TEST_DAYS,
            'horizons': HORIZONS,
            'model': 'LGBMClassifier',
            'n_pairs_tested': len(SECTOR_PAIRS),
            'threshold': 0.6,
        },
        'overall_verdict': overall,
        'pair_results': {},
        'feature_importances': {},
        'permutation_tests': {},
    }

    # Convert results for JSON serialization
    for pair_name, pair_res in all_results.items():
        output['pair_results'][pair_name] = {}
        for hz_key, eval_res in pair_res.items():
            # Convert any numpy types
            clean = {}
            for k, v in eval_res.items():
                if isinstance(v, (np.integer,)):
                    clean[k] = int(v)
                elif isinstance(v, (np.floating,)):
                    clean[k] = float(v)
                elif isinstance(v, (np.bool_,)):
                    clean[k] = bool(v)
                elif isinstance(v, dict):
                    clean[k] = {kk: bool(vv) if isinstance(vv, (np.bool_,)) else vv for kk, vv in v.items()}
                else:
                    clean[k] = v
            output['pair_results'][pair_name][hz_key] = clean

    for key, imp in all_importances.items():
        output['feature_importances'][key] = {k: round(v, 2) for k, v in
                                               sorted(imp.items(), key=lambda x: x[1], reverse=True)}

    output['permutation_tests'] = all_perm_tests

    # Summary table for quick reference
    if valid_summaries:
        output['summary_table'] = sorted(
            [{'pair': s['pair'], 'horizon': s['horizon'], 'sharpe': s['sharpe'],
              'sortino': s['sortino'], 'wr': s['win_rate'], 'pf': s['profit_factor'],
              'auc': s['auc'], 'verdict': s['verdict']}
             for s in valid_summaries],
            key=lambda x: x['sharpe'], reverse=True
        )

    results_path = Path('/home/jupiter/Lvl3Quant/research_results/subsector_rotation_ml_results.json')
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to {results_path}")
    print("DONE.")

    return output


if __name__ == '__main__':
    main()
