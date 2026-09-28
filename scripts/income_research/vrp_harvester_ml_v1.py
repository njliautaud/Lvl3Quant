#!/usr/bin/env python3
"""
VRP Harvester with ML Timing v1
================================
Systematic Volatility Risk Premium harvesting with LGBM crash-timing overlay.

Core thesis: VRP (implied - realized vol) is persistently positive (~80% of days).
Shorting vol is systematically profitable but has catastrophic tail risk.
ML timing layer learns to flatten/reduce before vol spikes.

HC #428 compliant: regime-agnostic validation, walk-forward, permutation tests.
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

warnings.filterwarnings('ignore')

# Output directory
OUTPUT_DIR = Path("/home/nick/Lvl3Quant/output/vrp_harvester_ml_v1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

# =============================================================================
# 1. DATA ACQUISITION
# =============================================================================
def fetch_data():
    """Fetch VIX, SPY, VIX3M, HYG, IEF from yfinance."""
    log("Fetching market data from yfinance...")
    import yfinance as yf

    start = "2006-01-01"  # Extra buffer for feature computation
    end = datetime.now().strftime("%Y-%m-%d")

    tickers = {
        'VIX': '^VIX',
        'SPY': 'SPY',
        'VIX3M': '^VIX3M',  # 3-month VIX for term structure
        'HYG': 'HYG',       # High yield for credit spreads
        'IEF': 'IEF',       # Treasury for credit spreads
        'SVXY': 'SVXY',     # ProShares Short VIX Short-Term Futures (actual tradeable)
        'VIXY': 'VIXY',     # ProShares VIX Short-Term Futures (long vol)
    }

    data = {}
    for name, ticker in tickers.items():
        try:
            df = yf.download(ticker, start=start, end=end, progress=False)
            # Handle multi-index columns from newer yfinance
            if isinstance(df.columns, pd.MultiIndex):
                close = df[('Close', ticker)].copy()
            else:
                close = df['Close'].copy()
            close.name = name
            data[name] = close
            log(f"  {name}: {len(close)} days ({close.index[0].strftime('%Y-%m-%d')} to {close.index[-1].strftime('%Y-%m-%d')})")
        except Exception as e:
            log(f"  WARNING: Failed to fetch {name}: {e}")
            data[name] = None

    # Combine into single DataFrame
    combined = pd.DataFrame(data)
    combined = combined.dropna(subset=['VIX', 'SPY'])

    # Fill VIX3M if missing with VIX * 0.95 approximation
    if combined['VIX3M'].isna().sum() > len(combined) * 0.5:
        log("  VIX3M mostly missing, using VIX*0.92 approximation for term structure")
        combined['VIX3M'] = combined['VIX'] * 0.92
    else:
        combined['VIX3M'] = combined['VIX3M'].ffill()

    combined['HYG'] = combined['HYG'].ffill() if 'HYG' in combined.columns and combined['HYG'].notna().any() else 100.0
    combined['IEF'] = combined['IEF'].ffill() if 'IEF' in combined.columns and combined['IEF'].notna().any() else 100.0
    # SVXY and VIXY: keep NaN where not available (will use synthetic for those periods)
    for col in ['SVXY', 'VIXY']:
        if col in combined.columns:
            combined[col] = combined[col].ffill(limit=3)  # Only fill small gaps

    log(f"Combined dataset: {len(combined)} days")
    return combined


# =============================================================================
# 2. FEATURE ENGINEERING
# =============================================================================
def compute_features(df):
    """Compute VRP features for ML timing."""
    log("Computing features...")

    # Realized volatility (21-day, annualized)
    spy_ret = df['SPY'].pct_change()
    df['realized_vol_21d'] = spy_ret.rolling(21).std() * np.sqrt(252) * 100
    df['realized_vol_10d'] = spy_ret.rolling(10).std() * np.sqrt(252) * 100
    df['realized_vol_5d'] = spy_ret.rolling(5).std() * np.sqrt(252) * 100

    # Core VRP signal
    df['VRP'] = df['VIX'] - df['realized_vol_21d']
    df['VRP_10d'] = df['VIX'] - df['realized_vol_10d']

    # VRP statistics
    df['VRP_z'] = (df['VRP'] - df['VRP'].rolling(63).mean()) / df['VRP'].rolling(63).std()
    df['VRP_ma20'] = df['VRP'].rolling(20).mean()
    df['VRP_ma60'] = df['VRP'].rolling(60).mean()
    df['VRP_std20'] = df['VRP'].rolling(20).std()

    # VIX features
    df['VIX_z'] = (df['VIX'] - df['VIX'].rolling(63).mean()) / df['VIX'].rolling(63).std()
    df['VIX_ma20'] = df['VIX'].rolling(20).mean()
    df['VIX_change_5d'] = df['VIX'].pct_change(5)
    df['VIX_change_1d'] = df['VIX'].pct_change(1)

    # Term structure: VIX / VIX3M (contango < 1, backwardation > 1)
    df['term_structure'] = df['VIX'] / df['VIX3M']
    df['term_structure_ma5'] = df['term_structure'].rolling(5).mean()

    # Credit spreads (HYG-IEF total return differential)
    hyg_ret = df['HYG'].pct_change()
    ief_ret = df['IEF'].pct_change()
    df['credit_spread_ret'] = hyg_ret - ief_ret
    df['credit_spread_20d'] = df['credit_spread_ret'].rolling(20).sum()

    # Market momentum
    df['SPY_ret_1d'] = spy_ret
    df['SPY_ret_5d'] = df['SPY'].pct_change(5)
    df['SPY_ret_20d'] = df['SPY'].pct_change(20)
    df['SPY_sma20'] = df['SPY'] / df['SPY'].rolling(20).mean() - 1
    df['SPY_sma50'] = df['SPY'] / df['SPY'].rolling(50).mean() - 1
    df['SPY_sma200'] = df['SPY'] / df['SPY'].rolling(200).mean() - 1

    # Skew proxy: VIX relative to term structure
    df['skew_proxy'] = df['VIX'] * df['term_structure']

    # Realized vol acceleration
    df['vol_accel'] = df['realized_vol_5d'] - df['realized_vol_21d']

    # VRP regime (positive = harvest, negative = caution)
    df['vrp_positive'] = (df['VRP'] > 0).astype(int)
    df['vrp_positive_streak'] = df['vrp_positive'].groupby(
        (df['vrp_positive'] != df['vrp_positive'].shift()).cumsum()
    ).cumcount() + 1
    df.loc[df['vrp_positive'] == 0, 'vrp_positive_streak'] = 0

    log(f"  Computed {len([c for c in df.columns if c not in ['VIX','SPY','VIX3M','HYG','IEF']])} features")
    return df


# =============================================================================
# 3. CONSTRUCT SHORT-VOL RETURNS
# =============================================================================
def construct_short_vol_returns(df):
    """
    Construct short-vol returns using ACTUAL SVXY data where available,
    and a calibrated synthetic for earlier periods.

    Key insight: VIX spot != VIX futures. SVXY tracks short-term VIX futures,
    which have ~0.5x beta to VIX spot. Using raw -dVIX/VIX massively overstates
    both returns and drawdowns.

    Approach:
    1. Use actual SVXY returns where available (2011+)
    2. For pre-SVXY period: synthetic = -0.5 * dVIX/VIX + roll_yield
       (0.5x beta calibrated to match SVXY's actual behavior)
    """
    log("Constructing short-vol returns...")

    vix_ret = df['VIX'].pct_change()

    # Method 1: Actual SVXY returns (most accurate)
    has_svxy = 'SVXY' in df.columns and df['SVXY'].notna().sum() > 100
    if has_svxy:
        svxy_ret = df['SVXY'].pct_change()
        svxy_valid = svxy_ret.notna()
        log(f"  SVXY actual data: {svxy_valid.sum()} days")
    else:
        svxy_ret = pd.Series(np.nan, index=df.index)
        svxy_valid = pd.Series(False, index=df.index)
        log("  No SVXY data available, using full synthetic")

    # Method 2: Calibrated synthetic for periods without SVXY
    # VIX futures beta to spot is ~0.5 for front-month
    # Roll yield from contango (VIX3M > VIX)
    futures_beta = 0.50
    roll_yield = (df['VIX3M'] - df['VIX']) / df['VIX'] / 21  # Monthly roll
    roll_yield = roll_yield.clip(-0.005, 0.005)

    synthetic_ret = -futures_beta * vix_ret + roll_yield

    # Blend: use SVXY where available, synthetic otherwise
    short_vol_ret = synthetic_ret.copy()
    if has_svxy:
        short_vol_ret[svxy_valid] = svxy_ret[svxy_valid]

    # Cap extreme daily moves (margin/position limits in practice)
    short_vol_ret = short_vol_ret.clip(-0.20, 0.15)

    df['short_vol_ret'] = short_vol_ret

    # Also compute forward returns for target construction
    df['fwd_21d_short_vol_ret'] = df['short_vol_ret'].rolling(21).sum().shift(-21)
    df['fwd_21d_drawdown'] = df['short_vol_ret'].rolling(21).apply(
        lambda x: (1 + x).cumprod().min() - 1 if len(x) == 21 else np.nan
    ).shift(-21)

    log(f"  Mean daily short-vol return: {short_vol_ret.mean()*100:.4f}%")
    log(f"  Annualized short-vol return: {short_vol_ret.mean()*252*100:.1f}%")
    log(f"  Max daily loss: {short_vol_ret.min()*100:.1f}%")
    log(f"  Max daily gain: {short_vol_ret.max()*100:.1f}%")

    return df


# =============================================================================
# 4. ML TIMING MODEL (LGBM Walk-Forward)
# =============================================================================
def train_ml_timing(df):
    """
    Walk-forward LGBM to predict crash periods.
    Target: Will the short-vol strategy lose >5% in next 21 days?
    """
    log("Training ML timing model (walk-forward LGBM)...")
    import lightgbm as lgb

    feature_cols = [
        'VRP', 'VRP_10d', 'VRP_z', 'VRP_ma20', 'VRP_ma60', 'VRP_std20',
        'VIX', 'VIX_z', 'VIX_ma20', 'VIX_change_5d', 'VIX_change_1d',
        'term_structure', 'term_structure_ma5',
        'credit_spread_20d',
        'SPY_ret_5d', 'SPY_ret_20d', 'SPY_sma20', 'SPY_sma50', 'SPY_sma200',
        'realized_vol_21d', 'realized_vol_10d', 'realized_vol_5d',
        'vol_accel', 'skew_proxy',
        'vrp_positive_streak',
    ]

    # Target: next 21 days short-vol return < -5% (crash)
    df['crash_target'] = (df['fwd_21d_short_vol_ret'] < -0.05).astype(int)

    # Drop rows without target or features
    valid = df.dropna(subset=feature_cols + ['crash_target'])
    log(f"  Valid rows for ML: {len(valid)}")
    log(f"  Crash rate: {valid['crash_target'].mean()*100:.1f}%")

    # Walk-forward parameters
    train_days = 252
    test_days = 63
    step_days = 21

    predictions = pd.Series(index=valid.index, dtype=float)
    fold_metrics = []

    dates = valid.index
    start_idx = train_days
    fold = 0

    while start_idx + test_days <= len(dates):
        train_start = max(0, start_idx - train_days)
        train_end = start_idx
        test_end = min(start_idx + test_days, len(dates))

        train_idx = dates[train_start:train_end]
        test_idx = dates[start_idx:test_end]

        X_train = valid.loc[train_idx, feature_cols]
        y_train = valid.loc[train_idx, 'crash_target']
        X_test = valid.loc[test_idx, feature_cols]
        y_test = valid.loc[test_idx, 'crash_target']

        # Handle class imbalance
        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        scale = n_neg / max(n_pos, 1)

        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'learning_rate': 0.05,
            'num_leaves': 16,
            'max_depth': 4,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'scale_pos_weight': scale,
            'verbose': -1,
            'n_jobs': -1,
            'seed': 42,
        }

        train_data = lgb.Dataset(X_train, label=y_train)
        model = lgb.train(params, train_data, num_boost_round=200)

        preds = model.predict(X_test)
        predictions.loc[test_idx] = preds

        # Fold metrics
        from sklearn.metrics import roc_auc_score
        try:
            if y_test.nunique() > 1:
                auc = roc_auc_score(y_test, preds)
            else:
                auc = 0.5  # Can't compute AUC with single class
        except Exception:
            auc = 0.5

        fold_metrics.append({
            'fold': fold,
            'train_start': train_idx[0].strftime('%Y-%m-%d'),
            'test_start': test_idx[0].strftime('%Y-%m-%d'),
            'test_end': test_idx[-1].strftime('%Y-%m-%d'),
            'auc': auc,
            'crash_rate': y_test.mean(),
            'pred_mean': preds.mean(),
        })

        start_idx += step_days
        fold += 1

    # Feature importance (from last model)
    if fold == 0:
        log("  ERROR: No walk-forward folds completed. Check data.")
        importance = pd.DataFrame({'feature': feature_cols, 'importance': 0})
        return predictions, fold_metrics, importance, feature_cols

    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)

    log(f"  Completed {fold} walk-forward folds")
    log(f"  Mean AUC: {np.mean([f['auc'] for f in fold_metrics]):.3f}")
    log(f"\n  Top features:")
    for _, row in importance.head(8).iterrows():
        log(f"    {row['feature']}: {row['importance']:.0f}")

    return predictions, fold_metrics, importance, feature_cols


# =============================================================================
# 5. STRATEGY SIMULATION
# =============================================================================
def simulate_strategies(df, ml_predictions):
    """
    Simulate:
    1. Naive always-short-VIX
    2. VRP-only (short when VRP > 0)
    3. ML-timed VRP (VRP + crash prediction overlay)
    4. SPY buy-and-hold
    5. 60/40 portfolio
    """
    log("Simulating strategies...")

    # Align to prediction period
    valid_dates = ml_predictions.dropna().index
    sim = df.loc[valid_dates].copy()
    sim['ml_pred'] = ml_predictions.loc[valid_dates]

    spy_ret = sim['SPY'].pct_change()
    # Simple bond proxy: inverse of realized vol change (rough)
    bond_ret = sim['IEF'].pct_change() if 'IEF' in sim.columns else pd.Series(0.0001, index=sim.index)

    results = {}

    # 1. Naive always-short-VIX
    results['Naive Short Vol'] = sim['short_vol_ret'].copy()

    # 2. VRP-only (short when VRP > 0, flat otherwise)
    vrp_signal = (sim['VRP'] > 0).astype(float)
    results['VRP Only'] = sim['short_vol_ret'] * vrp_signal

    # 3. ML-timed VRP
    # Threshold: if crash prob > 0.4, reduce to 30%; if > 0.6, flatten
    ml_position = pd.Series(1.0, index=sim.index)
    ml_position[sim['ml_pred'] > 0.4] = 0.3
    ml_position[sim['ml_pred'] > 0.6] = 0.0
    ml_position[sim['VIX'] > 30] = 0.0  # Hard VIX cutoff
    ml_position[sim['VRP'] <= 0] = 0.0  # Still respect VRP signal
    results['ML-Timed VRP'] = sim['short_vol_ret'] * ml_position

    # 4. SPY buy-and-hold
    results['SPY B&H'] = spy_ret.loc[sim.index]

    # 5. 60/40
    results['60/40'] = 0.6 * spy_ret.loc[sim.index] + 0.4 * bond_ret.loc[sim.index]

    # Store position info for analysis
    sim['ml_position'] = ml_position
    sim['vrp_signal'] = vrp_signal

    log(f"  Simulation period: {sim.index[0].strftime('%Y-%m-%d')} to {sim.index[-1].strftime('%Y-%m-%d')}")
    log(f"  ML-timed: {(ml_position > 0).mean()*100:.1f}% of days in position")
    log(f"  VRP-only: {(vrp_signal > 0).mean()*100:.1f}% of days in position")

    return results, sim


# =============================================================================
# 6. PERFORMANCE METRICS
# =============================================================================
def compute_metrics(returns_dict, capital=100_000):
    """Compute comprehensive performance metrics."""
    log("Computing performance metrics...")

    metrics = {}
    for name, rets in returns_dict.items():
        rets = rets.dropna()
        if len(rets) == 0:
            continue

        # Basic stats
        total_ret = (1 + rets).prod() - 1
        n_years = len(rets) / 252
        cagr = (1 + total_ret) ** (1/n_years) - 1 if n_years > 0 else 0
        ann_vol = rets.std() * np.sqrt(252)

        # Sharpe
        sharpe = (rets.mean() * 252) / (rets.std() * np.sqrt(252)) if rets.std() > 0 else 0

        # Sortino
        downside = rets[rets < 0].std() * np.sqrt(252)
        sortino = (rets.mean() * 252) / downside if downside > 0 else 0

        # Max drawdown
        cum = (1 + rets).cumprod()
        rolling_max = cum.cummax()
        dd = (cum - rolling_max) / rolling_max
        max_dd = dd.min()

        # Calmar
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0

        # Win rate
        wr = (rets > 0).mean()

        # Profit factor
        gains = rets[rets > 0].sum()
        losses = abs(rets[rets < 0].sum())
        pf = gains / losses if losses > 0 else np.inf

        # Monthly income at given capital
        monthly_income = cagr * capital / 12

        # Skewness and kurtosis
        skew = rets.skew()
        kurt = rets.kurtosis()

        metrics[name] = {
            'CAGR': cagr,
            'Ann Vol': ann_vol,
            'Sharpe': sharpe,
            'Sortino': sortino,
            'Max DD': max_dd,
            'Calmar': calmar,
            'Win Rate': wr,
            'Profit Factor': pf,
            'Monthly Income ($100K)': monthly_income,
            'Skew': skew,
            'Kurtosis': kurt,
            'Total Return': total_ret,
            'N Days': len(rets),
            'N Years': n_years,
        }

    return metrics


def compute_yearly_returns(returns_dict):
    """Year-by-year returns for each strategy."""
    yearly = {}
    for name, rets in returns_dict.items():
        rets = rets.dropna()
        yearly[name] = rets.groupby(rets.index.year).apply(lambda x: (1+x).prod() - 1)
    return pd.DataFrame(yearly)


def compute_crisis_performance(returns_dict):
    """Performance during crisis periods."""
    crises = {
        'GFC (2008-09 to 2009-03)': ('2008-09-01', '2009-03-31'),
        'Flash Crash (2010-05)': ('2010-05-01', '2010-05-31'),
        'EU Crisis (2011-08 to 2011-10)': ('2011-08-01', '2011-10-31'),
        'Volmageddon (2018-02)': ('2018-02-01', '2018-02-28'),
        'COVID (2020-02 to 2020-04)': ('2020-02-15', '2020-04-15'),
        'Rate Shock (2022-01 to 2022-06)': ('2022-01-01', '2022-06-30'),
        'SVB (2023-03)': ('2023-03-01', '2023-03-31'),
    }

    crisis_perf = {}
    for crisis_name, (start, end) in crises.items():
        crisis_perf[crisis_name] = {}
        for strat_name, rets in returns_dict.items():
            rets = rets.dropna()
            mask = (rets.index >= start) & (rets.index <= end)
            if mask.sum() > 0:
                crisis_ret = (1 + rets[mask]).prod() - 1
                crisis_perf[crisis_name][strat_name] = crisis_ret

    return pd.DataFrame(crisis_perf).T


# =============================================================================
# 7. REGIME ANALYSIS (HC #428 R1)
# =============================================================================
def regime_analysis(returns_dict, df):
    """
    HC #428 R1: Regime-agnostic validation.
    Classify days as green/red/flat based on SPY close-to-close.
    Compute per-regime Sharpe for each strategy.
    """
    log("Running regime analysis (HC #428 R1)...")

    spy_ret = df['SPY'].pct_change()
    spy_ret_20d = df['SPY'].pct_change(20)

    # Classify regimes based on 20-day SPY returns (appropriate for monthly strategy)
    regimes = pd.Series('flat', index=spy_ret_20d.index)
    regimes[spy_ret_20d > 0.02] = 'green'    # >2% over 20 days = bull
    regimes[spy_ret_20d < -0.02] = 'red'     # <-2% over 20 days = bear

    regime_metrics = {}
    for strat_name, rets in returns_dict.items():
        rets = rets.dropna()
        common = rets.index.intersection(regimes.index)

        strat_regime = {}
        for regime in ['green', 'red', 'flat']:
            mask = regimes.loc[common] == regime
            r = rets.loc[common][mask]
            if len(r) > 20:
                sharpe = r.mean() * np.sqrt(252) / r.std() if r.std() > 0 else 0
                strat_regime[f'{regime}_sharpe'] = sharpe
                strat_regime[f'{regime}_mean'] = r.mean() * 252
                strat_regime[f'{regime}_n'] = len(r)
            else:
                strat_regime[f'{regime}_sharpe'] = np.nan
                strat_regime[f'{regime}_mean'] = np.nan
                strat_regime[f'{regime}_n'] = len(r)

        # Regime asymmetry check
        gs = strat_regime.get('green_sharpe', 0)
        rs = strat_regime.get('red_sharpe', 0)
        max_s = max(abs(gs), abs(rs))
        asymmetry = abs(gs - rs) / max_s if max_s > 0 else 0
        strat_regime['regime_asymmetry'] = asymmetry
        strat_regime['regime_pass'] = asymmetry <= 0.50

        regime_metrics[strat_name] = strat_regime

    return pd.DataFrame(regime_metrics).T


# =============================================================================
# 8. PERMUTATION TEST
# =============================================================================
def permutation_test(raw_returns, positions, n_perms=500):
    """
    Permutation test: shuffle the SIGNAL (positions) relative to underlying returns.
    This tests whether the ML timing adds value vs random timing.
    p-value = fraction of shuffled Sharpes >= observed.
    """
    log(f"Running permutation test ({n_perms} shuffles)...")

    # Align and drop NaN
    common = raw_returns.dropna().index.intersection(positions.dropna().index)
    raw = raw_returns.loc[common].values
    pos = positions.loc[common].values

    # Observed: actual strategy returns
    actual_rets = raw * pos
    observed_sharpe = actual_rets.mean() / actual_rets.std() * np.sqrt(252) if actual_rets.std() > 0 else 0

    rng = np.random.RandomState(42)
    perm_sharpes = np.zeros(n_perms)

    for i in range(n_perms):
        # Shuffle positions (break signal-return alignment)
        shuffled_pos = rng.permutation(pos)
        perm_rets = raw * shuffled_pos
        perm_sharpes[i] = perm_rets.mean() / perm_rets.std() * np.sqrt(252) if perm_rets.std() > 0 else 0

    p_value = (perm_sharpes >= observed_sharpe).mean()

    log(f"  Observed Sharpe: {observed_sharpe:.3f}")
    log(f"  Permutation p-value: {p_value:.4f}")
    log(f"  Perm Sharpe mean: {perm_sharpes.mean():.3f}, std: {perm_sharpes.std():.3f}")

    return {
        'observed_sharpe': float(observed_sharpe),
        'p_value': float(p_value),
        'perm_sharpe_mean': float(perm_sharpes.mean()),
        'perm_sharpe_std': float(perm_sharpes.std()),
        'perm_sharpe_95': float(np.percentile(perm_sharpes, 95)),
        'perm_sharpe_99': float(np.percentile(perm_sharpes, 99)),
    }


# =============================================================================
# 9. REPORT GENERATION
# =============================================================================
def generate_report(metrics, yearly, crisis, regime, perm_results, fold_metrics, importance):
    """Generate comprehensive text report."""

    lines = []
    lines.append("=" * 80)
    lines.append("VRP HARVESTER WITH ML TIMING v1 — RESEARCH REPORT")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 80)

    # Strategy comparison
    lines.append("\n" + "=" * 80)
    lines.append("STRATEGY COMPARISON")
    lines.append("=" * 80)

    header = f"{'Strategy':<20} {'CAGR':>8} {'Sharpe':>8} {'Sortino':>8} {'MaxDD':>8} {'WR':>6} {'PF':>6} {'Mo.Inc':>8}"
    lines.append(header)
    lines.append("-" * 80)

    for strat, m in metrics.items():
        line = f"{strat:<20} {m['CAGR']*100:>7.1f}% {m['Sharpe']:>8.2f} {m['Sortino']:>8.2f} {m['Max DD']*100:>7.1f}% {m['Win Rate']*100:>5.1f}% {m['Profit Factor']:>5.2f} ${m['Monthly Income ($100K)']:>7.0f}"
        lines.append(line)

    lines.append(f"\nNote: Monthly income assumes $100K capital")

    # Asymmetry analysis
    lines.append("\n" + "=" * 80)
    lines.append("ASYMMETRY ANALYSIS")
    lines.append("=" * 80)

    for strat, m in metrics.items():
        skew = m['Skew']
        kurt = m['Kurtosis']
        # Asymmetry ratio: upside vol / downside vol
        lines.append(f"{strat}: Skew={skew:.2f}, Kurtosis={kurt:.2f}")

    # Year-by-year
    lines.append("\n" + "=" * 80)
    lines.append("YEAR-BY-YEAR RETURNS (%)")
    lines.append("=" * 80)
    lines.append(yearly.map(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "N/A").to_string())

    # Crisis performance
    lines.append("\n" + "=" * 80)
    lines.append("CRISIS PERIOD RETURNS (%)")
    lines.append("=" * 80)
    lines.append(crisis.map(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "N/A").to_string())

    # Regime analysis
    lines.append("\n" + "=" * 80)
    lines.append("REGIME ANALYSIS (HC #428 R1)")
    lines.append("=" * 80)
    for strat in regime.index:
        row = regime.loc[strat]
        gs = row.get('green_sharpe', 'N/A')
        rs = row.get('red_sharpe', 'N/A')
        asym = row.get('regime_asymmetry', 'N/A')
        passed = row.get('regime_pass', False)
        gs_str = f"{gs:.2f}" if isinstance(gs, (int, float)) and not pd.isna(gs) else "N/A"
        rs_str = f"{rs:.2f}" if isinstance(rs, (int, float)) and not pd.isna(rs) else "N/A"
        asym_str = f"{asym:.2f}" if isinstance(asym, (int, float)) and not pd.isna(asym) else "N/A"
        status = "PASS" if passed else "FAIL"
        lines.append(f"  {strat}: Green Sharpe={gs_str}, Red Sharpe={rs_str}, Asymmetry={asym_str} [{status}]")

    # Permutation test
    lines.append("\n" + "=" * 80)
    lines.append("PERMUTATION TEST (ML-Timed VRP)")
    lines.append("=" * 80)
    lines.append(f"  Observed Sharpe: {perm_results['observed_sharpe']:.3f}")
    lines.append(f"  p-value: {perm_results['p_value']:.4f}")
    lines.append(f"  95th percentile null: {perm_results['perm_sharpe_95']:.3f}")
    lines.append(f"  99th percentile null: {perm_results['perm_sharpe_99']:.3f}")
    sig = "YES" if perm_results['p_value'] < 0.05 else "NO"
    lines.append(f"  Statistically significant (p<0.05): {sig}")

    # ML model performance
    lines.append("\n" + "=" * 80)
    lines.append("ML TIMING MODEL PERFORMANCE")
    lines.append("=" * 80)
    aucs = [f['auc'] for f in fold_metrics]
    lines.append(f"  Walk-forward folds: {len(fold_metrics)}")
    lines.append(f"  Mean AUC: {np.mean(aucs):.3f} +/- {np.std(aucs):.3f}")
    lines.append(f"  Min AUC: {np.min(aucs):.3f}, Max AUC: {np.max(aucs):.3f}")

    lines.append(f"\n  Top Features by Gain:")
    for _, row in importance.head(10).iterrows():
        lines.append(f"    {row['feature']}: {row['importance']:.0f}")

    # Conclusion
    lines.append("\n" + "=" * 80)
    lines.append("CONCLUSION")
    lines.append("=" * 80)

    ml_m = metrics.get('ML-Timed VRP', {})
    naive_m = metrics.get('Naive Short Vol', {})
    spy_m = metrics.get('SPY B&H', {})

    if ml_m and naive_m:
        dd_improvement = (ml_m['Max DD'] - naive_m['Max DD']) / abs(naive_m['Max DD']) * 100
        sharpe_improvement = (ml_m['Sharpe'] - naive_m['Sharpe']) / abs(naive_m['Sharpe']) * 100 if naive_m['Sharpe'] != 0 else 0
        lines.append(f"  ML timing vs Naive: Sharpe {sharpe_improvement:+.0f}%, MaxDD {dd_improvement:+.0f}%")

    if ml_m:
        lines.append(f"  ML-Timed VRP monthly income at $100K: ${ml_m['Monthly Income ($100K)']:.0f}")
        lines.append(f"  To hit $3K/mo target: ${3000 / max(ml_m['Monthly Income ($100K)'], 1) * 100000:.0f} capital needed")
        lines.append(f"  To hit $5K/mo target: ${5000 / max(ml_m['Monthly Income ($100K)'], 1) * 100000:.0f} capital needed")

    return "\n".join(lines)


# =============================================================================
# MAIN
# =============================================================================
def main():
    log("=" * 60)
    log("VRP HARVESTER WITH ML TIMING v1")
    log("=" * 60)

    # 1. Fetch data
    df = fetch_data()

    # 2. Feature engineering
    df = compute_features(df)

    # 3. Construct short-vol returns
    df = construct_short_vol_returns(df)

    # 4. Train ML timing model
    ml_predictions, fold_metrics, importance, feature_cols = train_ml_timing(df)

    # 5. Simulate strategies
    returns_dict, sim_df = simulate_strategies(df, ml_predictions)

    # 6. Performance metrics
    metrics = compute_metrics(returns_dict)

    # Print summary
    log("\n" + "=" * 60)
    log("STRATEGY COMPARISON SUMMARY")
    log("=" * 60)
    for strat, m in metrics.items():
        log(f"  {strat}: Sharpe={m['Sharpe']:.2f}, CAGR={m['CAGR']*100:.1f}%, MaxDD={m['Max DD']*100:.1f}%, "
            f"Sortino={m['Sortino']:.2f}, WR={m['Win Rate']*100:.1f}%, Mo.Inc=${m['Monthly Income ($100K)']:.0f}")

    # 7. Year-by-year
    yearly = compute_yearly_returns(returns_dict)

    # 8. Crisis performance
    crisis = compute_crisis_performance(returns_dict)

    # 9. Regime analysis (HC #428 R1)
    regime = regime_analysis(returns_dict, df)

    # 10. Permutation test on ML-timed strategy
    perm_results = permutation_test(sim_df['short_vol_ret'], sim_df['ml_position'])

    # 11. Generate report
    report = generate_report(metrics, yearly, crisis, regime, perm_results, fold_metrics, importance)

    # Save outputs
    log("\nSaving outputs...")

    # Report
    report_path = OUTPUT_DIR / "report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    log(f"  Report: {report_path}")

    # Metrics JSON
    metrics_serializable = {}
    for k, v in metrics.items():
        metrics_serializable[k] = {mk: float(mv) if isinstance(mv, (np.floating, float)) else mv for mk, mv in v.items()}

    with open(OUTPUT_DIR / "metrics.json", 'w') as f:
        json.dump(metrics_serializable, f, indent=2, default=str)

    # Fold metrics
    with open(OUTPUT_DIR / "fold_metrics.json", 'w') as f:
        json.dump(fold_metrics, f, indent=2, default=str)

    # Feature importance
    importance.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

    # Yearly returns
    yearly.to_csv(OUTPUT_DIR / "yearly_returns.csv")

    # Permutation results
    with open(OUTPUT_DIR / "permutation_test.json", 'w') as f:
        json.dump(perm_results, f, indent=2, default=str)

    # Regime results
    regime.to_csv(OUTPUT_DIR / "regime_analysis.csv")

    # Equity curves
    equity_curves = pd.DataFrame()
    for name, rets in returns_dict.items():
        equity_curves[name] = (1 + rets.dropna()).cumprod()
    equity_curves.to_csv(OUTPUT_DIR / "equity_curves.csv")

    # Print report to stdout
    print("\n" + report)

    log("\n" + "=" * 60)
    log("COMPLETE. All outputs saved.")
    log("=" * 60)


if __name__ == "__main__":
    main()
