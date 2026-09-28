#!/usr/bin/env python3
"""
Market-Neutral Sector Rotation — Lean Adversarial Audit v1
===========================================================
Audits the L3/S3 market-neutral sector rotation strategy (KB #285, entry 1206).

Strategy:
  - LGBM ranks 11 sector ETFs using 17 quality-momentum features
  - Long top-3, Short bottom-3, equal-weight, monthly rebalance
  - Sliding 500-day LGBM training window
  - $10K starting capital, zero commission
  - Reported: Sharpe 3.26, MDD -2.2%, WR 86.5%, Beta 0.01

5 Audit Tests:
  1. Permutation test (500x)   — shuffle rankings, p < 0.05 = PASS
  2. Sub-period stability      — >= 3/4 quarters Sharpe > 0 = PASS
  3. Regime analysis            — |bull-bear|/max < 0.50 = PASS
  4. Outlier removal            — drop top/bot 5%, Sharpe > 0.5 = PASS
  5. Random baseline (100x)    — actual > 2x random mean = PASS

VERDICT: VALIDATED if >= 4/5 pass.

Key optimization: LGBM trains walk-forward ONCE. Permutation/random tests
shuffle RANKINGS only (no retrain), making 500 perms take seconds.

Target runtime: < 5 minutes on 32-core machine.
"""

import warnings
warnings.filterwarnings('ignore')

import sys
import os
import time
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Flush-print wrapper
# ---------------------------------------------------------------------------
_builtin_print = print
def fprint(*args, **kwargs):
    _builtin_print(*args, **kwargs)
    sys.stdout.flush()

# ---------------------------------------------------------------------------
# MLflow setup
# ---------------------------------------------------------------------------
MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "market_neutral_lean_audit"
USE_MLFLOW = True

try:
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
except Exception as e:
    fprint(f"[WARN] MLflow unavailable: {e}. Continuing without tracking.")
    USE_MLFLOW = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SECTOR_ETFS = ['XLK', 'XLF', 'XLE', 'XLV', 'XLY', 'XLP', 'XLI', 'XLB', 'XLU', 'XLRE', 'XLC']
BENCHMARK = 'SPY'
ALL_TICKERS = SECTOR_ETFS + [BENCHMARK]

TRAIN_WINDOW = 500
REBAL_PERIOD = 21
STARTING_CAPITAL = 10000
N_LONG = 3
N_SHORT = 3

FEATURE_COLS = [
    'ret_5d', 'ret_10d', 'ret_21d', 'ret_63d', 'ret_126d', 'ret_252d',
    'vol_21d', 'vol_63d', 'sharpe_63d', 'maxdd_63d', 'pct_52w_high', 'mom_accel',
    'pct_pos_months_12m', 'sortino_63d', 'calmar_1y', 'up_capture',
    'trend_slope_63d',
]
assert len(FEATURE_COLS) == 17, f"Expected 17 features, got {len(FEATURE_COLS)}"

LGBM_PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'n_estimators': 100,
    'max_depth': 4,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_samples': 5,
    'verbose': -1,
    'n_jobs': -1,
    'random_state': 42,
}

# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------
def download_data(start='2005-01-01'):
    """Download all ETF data via yfinance."""
    import yfinance as yf
    end = datetime.now().strftime('%Y-%m-%d')
    fprint(f"[DATA] Downloading {len(ALL_TICKERS)} tickers from {start} to {end} ...")
    data = yf.download(ALL_TICKERS, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        close = data['Close']
    else:
        close = data[['Close']].copy()
    close = close.dropna(how='all')
    fprint(f"[DATA] {len(close)} trading days, {close.index[0].date()} to {close.index[-1].date()}")
    return close

# ---------------------------------------------------------------------------
# Feature computation — vectorized, no per-row loops
# ---------------------------------------------------------------------------
def compute_features(close):
    """Compute 17 quality-momentum features for each sector ETF. Returns long-form DataFrame."""
    fprint("[FEATURES] Computing 17 features for 11 sectors ...")
    t0 = time.time()
    spy_close = close[BENCHMARK]
    spy_ret = spy_close.pct_change()
    records = []

    for ticker in SECTOR_ETFS:
        c = close[ticker].dropna()
        ret = c.pct_change()

        # Return features
        ret_5d = c.pct_change(5)
        ret_10d = c.pct_change(10)
        ret_21d = c.pct_change(21)
        ret_63d = c.pct_change(63)
        ret_126d = c.pct_change(126)
        ret_252d = c.pct_change(252)

        # Volatility
        vol_21d = ret.rolling(21).std()
        vol_63d = ret.rolling(63).std()

        # Sharpe 63d
        mean_63d = ret.rolling(63).mean()
        sharpe_63d = mean_63d / vol_63d.replace(0, np.nan)

        # Max drawdown 63d
        def _maxdd(x):
            cum = (1 + x).cumprod()
            peak = cum.cummax()
            return (cum / peak - 1).min()
        maxdd_63d = ret.rolling(63).apply(_maxdd, raw=False)

        # Pct of 52-week high
        high_252 = c.rolling(252).max()
        pct_52w_high = c / high_252.replace(0, np.nan)

        # Momentum acceleration: ret_21d - ret_21d.shift(21)
        mom_accel = ret_21d - ret_21d.shift(21)

        # Pct positive months in trailing 12 months
        # Use 21-day non-overlapping blocks
        monthly_ret = c.pct_change(21)
        pct_pos_months_12m = monthly_ret.rolling(252).apply(
            lambda x: np.sum(x[::21] > 0) / max(len(x[::21]), 1), raw=True
        )

        # Sortino 63d
        downside = ret.copy()
        downside[downside > 0] = 0
        downside_std_63d = downside.rolling(63).std()
        sortino_63d = mean_63d / downside_std_63d.replace(0, np.nan)

        # Calmar 1y: ret_252d / |maxdd_252d|
        maxdd_252d = ret.rolling(252).apply(_maxdd, raw=False)
        calmar_1y = ret_252d / maxdd_252d.abs().replace(0, np.nan)

        # Up capture vs SPY (63d)
        spy_ret_aligned = spy_ret.reindex(ret.index)
        spy_up_mask = spy_ret_aligned > 0
        up_capture = pd.Series(np.nan, index=c.index)
        for i in range(63, len(c.index)):
            window_mask = spy_up_mask.iloc[i-63:i]
            if window_mask.sum() > 5:
                spy_up_days = spy_ret_aligned.iloc[i-63:i][window_mask]
                stock_up_days = ret.iloc[i-63:i][window_mask]
                spy_mean = spy_up_days.mean()
                if spy_mean != 0:
                    up_capture.iloc[i] = stock_up_days.mean() / spy_mean
                else:
                    up_capture.iloc[i] = 1.0

        # Trend slope 63d (linear regression slope of log prices)
        log_c = np.log(c.replace(0, np.nan))
        from scipy.stats import linregress
        def _trend_slope(x):
            if len(x) < 10 or np.any(np.isnan(x)):
                return np.nan
            slope, _, _, _, _ = linregress(np.arange(len(x)), x)
            return slope
        trend_slope_63d = log_c.rolling(63).apply(_trend_slope, raw=True)

        # Forward return (target)
        fwd_ret_21d = c.pct_change(21).shift(-21)

        df = pd.DataFrame({
            'date': c.index,
            'ticker': ticker,
            'ret_5d': ret_5d,
            'ret_10d': ret_10d,
            'ret_21d': ret_21d,
            'ret_63d': ret_63d,
            'ret_126d': ret_126d,
            'ret_252d': ret_252d,
            'vol_21d': vol_21d,
            'vol_63d': vol_63d,
            'sharpe_63d': sharpe_63d,
            'maxdd_63d': maxdd_63d,
            'pct_52w_high': pct_52w_high,
            'mom_accel': mom_accel,
            'pct_pos_months_12m': pct_pos_months_12m,
            'sortino_63d': sortino_63d,
            'calmar_1y': calmar_1y,
            'up_capture': up_capture.values,
            'trend_slope_63d': trend_slope_63d,
            'fwd_ret_21d': fwd_ret_21d,
            'close': c,
        })
        records.append(df.reset_index(drop=True))

    features_df = pd.concat(records, ignore_index=True)
    features_df = features_df.dropna(subset=FEATURE_COLS + ['fwd_ret_21d'])
    elapsed = time.time() - t0
    fprint(f"[FEATURES] {len(features_df)} rows x {len(FEATURE_COLS)} features ({elapsed:.1f}s)")
    return features_df

# ---------------------------------------------------------------------------
# Walk-forward LGBM ranking (runs ONCE)
# ---------------------------------------------------------------------------
def walk_forward_ranking(features_df, close):
    """
    Sliding 500-day LGBM walk-forward. Monthly rebalance.
    Returns rankings list and SPY returns aligned to rebalance dates.
    """
    fprint("[WF] Running walk-forward LGBM ranking ...")
    t0 = time.time()

    dates = sorted(features_df['date'].unique())
    all_close_dates = close.index.tolist()

    # Start from day TRAIN_WINDOW (need 500 days for training)
    valid_start_idx = TRAIN_WINDOW
    rebal_dates = []
    for i in range(valid_start_idx, len(all_close_dates), REBAL_PERIOD):
        d = all_close_dates[i]
        if d in set(dates):
            rebal_dates.append(d)

    fprint(f"[WF] {len(rebal_dates)} rebalance points: {rebal_dates[0].date()} to {rebal_dates[-1].date()}")

    rankings = []
    total = len(rebal_dates)

    for idx, rebal_date in enumerate(rebal_dates):
        # Print progress every 10 rebalances
        if idx % 10 == 0:
            fprint(f"  Rebalance {idx+1}/{total}...")

        # Training data: trailing TRAIN_WINDOW days strictly before rebal_date
        mask_before = features_df['date'] < rebal_date
        train_pool = features_df[mask_before].copy()
        train_dates_sorted = sorted(train_pool['date'].unique())
        if len(train_dates_sorted) < TRAIN_WINDOW:
            continue
        cutoff = train_dates_sorted[-TRAIN_WINDOW]
        train_pool = train_pool[train_pool['date'] >= cutoff]

        # Prediction data: features at rebal_date
        pred_pool = features_df[features_df['date'] == rebal_date]
        if len(pred_pool) < 6:  # need at least 6 sectors
            continue

        X_train = np.nan_to_num(train_pool[FEATURE_COLS].values, nan=0.0, posinf=0.0, neginf=0.0)
        y_train = np.nan_to_num(train_pool['fwd_ret_21d'].values, nan=0.0)
        X_pred = np.nan_to_num(pred_pool[FEATURE_COLS].values, nan=0.0, posinf=0.0, neginf=0.0)

        model = lgb.LGBMRegressor(**LGBM_PARAMS)
        model.fit(X_train, y_train)
        preds = model.predict(X_pred)

        ticker_scores = dict(zip(pred_pool['ticker'].values, preds))
        rankings.append((rebal_date, ticker_scores))

    elapsed = time.time() - t0
    fprint(f"[WF] Done: {len(rankings)} rankings in {elapsed:.1f}s")
    return rankings

# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------
def backtest_ls(rankings, close):
    """
    L3/S3 equal-weight backtest from pre-computed rankings.
    Monthly PnL = avg(long returns) - avg(short returns).
    """
    equity = float(STARTING_CAPITAL)
    equity_curve = []
    period_returns = []

    for i in range(len(rankings) - 1):
        rebal_date, scores = rankings[i]
        next_date = rankings[i + 1][0]

        sorted_sectors = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        longs = [t for t, _ in sorted_sectors[:N_LONG]]
        shorts = [t for t, _ in sorted_sectors[-N_SHORT:]]

        long_ret = 0.0
        short_ret = 0.0
        for t in longs:
            try:
                r = close[t].loc[next_date] / close[t].loc[rebal_date] - 1.0
                long_ret += r
            except (KeyError, TypeError):
                pass
        for t in shorts:
            try:
                r = close[t].loc[next_date] / close[t].loc[rebal_date] - 1.0
                short_ret += r
            except (KeyError, TypeError):
                pass

        # L/S return: avg long - avg short
        period_ret = long_ret / N_LONG - short_ret / N_SHORT
        equity *= (1.0 + period_ret)
        equity_curve.append({'date': next_date, 'equity': equity, 'return': period_ret})
        period_returns.append(period_ret)

    return pd.DataFrame(equity_curve), np.array(period_returns)

# ---------------------------------------------------------------------------
# Sharpe helper
# ---------------------------------------------------------------------------
def annualized_sharpe(returns, periods_per_year=None):
    if periods_per_year is None:
        periods_per_year = 252.0 / REBAL_PERIOD
    if len(returns) < 2 or np.std(returns) == 0:
        return 0.0
    return float(np.mean(returns) / np.std(returns) * np.sqrt(periods_per_year))

def max_drawdown(equity_curve):
    peak = equity_curve['equity'].cummax()
    dd = equity_curve['equity'] / peak - 1.0
    return float(dd.min())

# ===========================================================================
# 5 AUDIT TESTS
# ===========================================================================

def test_1_permutation(rankings, close, n_perms=500):
    """
    Permutation test: shuffle RANKINGS (not retrain), compute Sharpe.
    PASS if p < 0.05 (actual Sharpe exceeds 95% of random shuffles).
    """
    fprint("\n[Test 1] Permutation Test (500 shuffles) ...")
    t0 = time.time()

    # Actual Sharpe
    _, actual_returns = backtest_ls(rankings, close)
    actual_sharpe = annualized_sharpe(actual_returns)
    fprint(f"  Actual Sharpe: {actual_sharpe:.3f}")

    # Permutation Sharpes
    rng = np.random.RandomState(42)
    perm_sharpes = []
    tickers = SECTOR_ETFS.copy()

    for p in range(n_perms):
        if p % 100 == 0:
            fprint(f"  Permutation {p}/{n_perms}...")
        shuffled_rankings = []
        for rebal_date, scores in rankings:
            values = list(scores.values())
            rng.shuffle(values)
            shuffled = dict(zip(scores.keys(), values))
            shuffled_rankings.append((rebal_date, shuffled))
        _, perm_returns = backtest_ls(shuffled_rankings, close)
        perm_sharpes.append(annualized_sharpe(perm_returns))

    perm_sharpes = np.array(perm_sharpes)
    p_value = float(np.mean(perm_sharpes >= actual_sharpe))
    passed = p_value < 0.05
    elapsed = time.time() - t0

    fprint(f"  Perm Sharpe: mean={np.mean(perm_sharpes):.3f}, "
           f"std={np.std(perm_sharpes):.3f}, p95={np.percentile(perm_sharpes, 95):.3f}")
    fprint(f"  p-value: {p_value:.4f} — {'PASS' if passed else 'FAIL'} ({elapsed:.1f}s)")
    return {
        'test': 'Permutation Test',
        'passed': passed,
        'actual_sharpe': actual_sharpe,
        'p_value': p_value,
        'perm_mean': float(np.mean(perm_sharpes)),
        'perm_std': float(np.std(perm_sharpes)),
        'perm_p95': float(np.percentile(perm_sharpes, 95)),
    }


def test_2_subperiod_stability(rankings, close):
    """
    Split into 4 chronological quarters. PASS if >= 3/4 have Sharpe > 0.
    """
    fprint("\n[Test 2] Sub-Period Stability (4 quarters) ...")
    _, all_returns = backtest_ls(rankings, close)
    n = len(all_returns)
    quarter_size = n // 4

    quarter_sharpes = []
    positive_quarters = 0
    for q in range(4):
        start = q * quarter_size
        end = (q + 1) * quarter_size if q < 3 else n
        q_returns = all_returns[start:end]
        s = annualized_sharpe(q_returns)
        quarter_sharpes.append(s)
        if s > 0:
            positive_quarters += 1
        fprint(f"  Q{q+1}: Sharpe={s:.3f} ({len(q_returns)} periods)")

    passed = positive_quarters >= 3
    fprint(f"  {positive_quarters}/4 quarters positive — {'PASS' if passed else 'FAIL'}")
    return {
        'test': 'Sub-Period Stability',
        'passed': passed,
        'quarter_sharpes': quarter_sharpes,
        'positive_quarters': positive_quarters,
    }


def test_3_regime_analysis(rankings, close):
    """
    Regime analysis using SPY 63d return.
    Bull: > 5%, Bear: < -5%, Flat: else.
    PASS if |bull - bear| / max(|bull|, |bear|) < 0.50
    """
    fprint("\n[Test 3] Regime Analysis (Bull/Bear/Flat) ...")
    eq_curve, all_returns = backtest_ls(rankings, close)

    # Compute SPY 63d return at each rebalance date
    spy_ret_63d = close[BENCHMARK].pct_change(63)

    regime_returns = {'bull': [], 'bear': [], 'flat': []}
    for i in range(len(all_returns)):
        date = eq_curve.iloc[i]['date']
        try:
            spy_r = spy_ret_63d.loc[date]
        except KeyError:
            spy_r = 0.0

        if np.isnan(spy_r):
            spy_r = 0.0

        if spy_r > 0.05:
            regime_returns['bull'].append(all_returns[i])
        elif spy_r < -0.05:
            regime_returns['bear'].append(all_returns[i])
        else:
            regime_returns['flat'].append(all_returns[i])

    sharpe_bull = annualized_sharpe(np.array(regime_returns['bull'])) if len(regime_returns['bull']) > 2 else 0.0
    sharpe_bear = annualized_sharpe(np.array(regime_returns['bear'])) if len(regime_returns['bear']) > 2 else 0.0
    sharpe_flat = annualized_sharpe(np.array(regime_returns['flat'])) if len(regime_returns['flat']) > 2 else 0.0

    max_abs = max(abs(sharpe_bull), abs(sharpe_bear), 0.001)
    regime_gap = abs(sharpe_bull - sharpe_bear) / max_abs

    passed = regime_gap < 0.50
    fprint(f"  Bull: Sharpe={sharpe_bull:.3f} ({len(regime_returns['bull'])} periods)")
    fprint(f"  Bear: Sharpe={sharpe_bear:.3f} ({len(regime_returns['bear'])} periods)")
    fprint(f"  Flat: Sharpe={sharpe_flat:.3f} ({len(regime_returns['flat'])} periods)")
    fprint(f"  Regime gap: {regime_gap:.3f} — {'PASS' if passed else 'FAIL'}")
    return {
        'test': 'Regime Analysis',
        'passed': passed,
        'sharpe_bull': sharpe_bull,
        'sharpe_bear': sharpe_bear,
        'sharpe_flat': sharpe_flat,
        'regime_gap': regime_gap,
        'n_bull': len(regime_returns['bull']),
        'n_bear': len(regime_returns['bear']),
        'n_flat': len(regime_returns['flat']),
    }


def test_4_outlier_removal(rankings, close):
    """
    Remove top/bottom 5% of monthly returns. PASS if Sharpe > 0.5.
    """
    fprint("\n[Test 4] Outlier Removal (drop top/bottom 5%) ...")
    _, all_returns = backtest_ls(rankings, close)

    low = np.percentile(all_returns, 5)
    high = np.percentile(all_returns, 95)
    trimmed = all_returns[(all_returns >= low) & (all_returns <= high)]
    trimmed_sharpe = annualized_sharpe(trimmed)

    original_sharpe = annualized_sharpe(all_returns)
    passed = trimmed_sharpe > 0.5

    fprint(f"  Original: Sharpe={original_sharpe:.3f} ({len(all_returns)} periods)")
    fprint(f"  Trimmed:  Sharpe={trimmed_sharpe:.3f} ({len(trimmed)} periods)")
    fprint(f"  {'PASS' if passed else 'FAIL'}")
    return {
        'test': 'Outlier Removal',
        'passed': passed,
        'original_sharpe': original_sharpe,
        'trimmed_sharpe': trimmed_sharpe,
        'n_original': len(all_returns),
        'n_trimmed': len(trimmed),
    }


def test_5_random_baseline(rankings, close, n_random=100):
    """
    100 random rankings → average Sharpe. PASS if actual > 2x random mean.
    """
    fprint("\n[Test 5] Random Baseline (100 random rankings) ...")
    t0 = time.time()

    _, actual_returns = backtest_ls(rankings, close)
    actual_sharpe = annualized_sharpe(actual_returns)

    rng = np.random.RandomState(123)
    random_sharpes = []
    tickers = SECTOR_ETFS.copy()

    for r in range(n_random):
        random_rankings = []
        for rebal_date, scores in rankings:
            random_scores = {t: rng.randn() for t in scores.keys()}
            random_rankings.append((rebal_date, random_scores))
        _, rand_returns = backtest_ls(random_rankings, close)
        random_sharpes.append(annualized_sharpe(rand_returns))

    random_sharpes = np.array(random_sharpes)
    random_mean = float(np.mean(random_sharpes))
    threshold = 2.0 * abs(random_mean) if random_mean != 0 else 0.1
    passed = actual_sharpe > threshold
    elapsed = time.time() - t0

    fprint(f"  Actual Sharpe: {actual_sharpe:.3f}")
    fprint(f"  Random mean: {random_mean:.3f}, threshold (2x): {threshold:.3f}")
    fprint(f"  {'PASS' if passed else 'FAIL'} ({elapsed:.1f}s)")
    return {
        'test': 'Random Baseline',
        'passed': passed,
        'actual_sharpe': actual_sharpe,
        'random_mean': random_mean,
        'random_std': float(np.std(random_sharpes)),
        'threshold': threshold,
    }


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    fprint("=" * 70)
    fprint("MARKET-NEUTRAL SECTOR ROTATION — LEAN ADVERSARIAL AUDIT v1")
    fprint("=" * 70)
    fprint(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    fprint(f"Features: {len(FEATURE_COLS)}")
    fprint(f"Sectors: {len(SECTOR_ETFS)}")
    t_start = time.time()

    # --- Step 1: Download data ---
    close = download_data()

    # --- Step 2: Compute features ---
    features_df = compute_features(close)

    # --- Step 3: Walk-forward LGBM (train ONCE) ---
    rankings = walk_forward_ranking(features_df, close)

    # --- Step 4: Baseline backtest ---
    fprint("\n[BACKTEST] Running baseline L3/S3 backtest ...")
    eq_curve, period_returns = backtest_ls(rankings, close)
    sharpe = annualized_sharpe(period_returns)
    mdd = max_drawdown(eq_curve)
    win_rate = float(np.mean(period_returns > 0)) * 100
    total_ret = (eq_curve['equity'].iloc[-1] / STARTING_CAPITAL - 1) * 100

    # Sortino
    downside = period_returns[period_returns < 0]
    if len(downside) > 0:
        sortino = float(np.mean(period_returns) / np.std(downside) * np.sqrt(252.0 / REBAL_PERIOD))
    else:
        sortino = float('inf')

    # Profit factor
    gross_profit = period_returns[period_returns > 0].sum()
    gross_loss = abs(period_returns[period_returns < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Beta vs SPY
    spy_returns = []
    for i in range(len(rankings) - 1):
        rd = rankings[i][0]
        nd = rankings[i + 1][0]
        try:
            spy_r = close[BENCHMARK].loc[nd] / close[BENCHMARK].loc[rd] - 1.0
            spy_returns.append(spy_r)
        except (KeyError, TypeError):
            spy_returns.append(0.0)
    spy_returns = np.array(spy_returns)
    if np.var(spy_returns) > 0:
        beta = float(np.cov(period_returns, spy_returns)[0, 1] / np.var(spy_returns))
    else:
        beta = 0.0

    fprint(f"\n{'='*50}")
    fprint(f"  BASELINE RESULTS")
    fprint(f"{'='*50}")
    fprint(f"  Total Return:  {total_ret:.1f}%")
    fprint(f"  Sharpe:        {sharpe:.2f}")
    fprint(f"  Sortino:       {sortino:.2f}")
    fprint(f"  MDD:           {mdd*100:.1f}%")
    fprint(f"  Win Rate:      {win_rate:.1f}%")
    fprint(f"  Profit Factor: {profit_factor:.2f}")
    fprint(f"  Beta vs SPY:   {beta:.3f}")
    fprint(f"  Periods:       {len(period_returns)}")
    fprint(f"{'='*50}")

    # --- Step 5: Run 5 audit tests ---
    results = []
    results.append(test_1_permutation(rankings, close, n_perms=500))
    results.append(test_2_subperiod_stability(rankings, close))
    results.append(test_3_regime_analysis(rankings, close))
    results.append(test_4_outlier_removal(rankings, close))
    results.append(test_5_random_baseline(rankings, close, n_random=100))

    # --- Results table ---
    n_passed = sum(1 for r in results if r['passed'])
    verdict = "VALIDATED" if n_passed >= 4 else "REJECTED"

    fprint(f"\n{'='*70}")
    fprint(f"  AUDIT RESULTS TABLE")
    fprint(f"{'='*70}")
    fprint(f"  {'#':<4} {'Test':<30} {'Result':<10} {'Detail'}")
    fprint(f"  {'-'*4} {'-'*30} {'-'*10} {'-'*30}")
    for i, r in enumerate(results, 1):
        status = "PASS" if r['passed'] else "FAIL"
        detail = ""
        if r['test'] == 'Permutation Test':
            detail = f"p={r['p_value']:.4f}, actual={r['actual_sharpe']:.2f}"
        elif r['test'] == 'Sub-Period Stability':
            detail = f"{r['positive_quarters']}/4 quarters positive"
        elif r['test'] == 'Regime Analysis':
            detail = f"gap={r['regime_gap']:.3f}, bull={r['sharpe_bull']:.2f}, bear={r['sharpe_bear']:.2f}"
        elif r['test'] == 'Outlier Removal':
            detail = f"trimmed Sharpe={r['trimmed_sharpe']:.2f}"
        elif r['test'] == 'Random Baseline':
            detail = f"actual={r['actual_sharpe']:.2f} vs threshold={r['threshold']:.2f}"
        fprint(f"  {i:<4} {r['test']:<30} {status:<10} {detail}")

    fprint(f"\n  PASSED: {n_passed}/5")
    fprint(f"  VERDICT: {verdict}")
    fprint(f"{'='*70}")

    elapsed = time.time() - t_start
    fprint(f"\nTotal runtime: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # --- Save JSON ---
    # Auto-detect base path
    _base = Path("/home/nick/Lvl3Quant") if Path("/home/nick").exists() else Path("/home/jupiter/Lvl3Quant")
    output_dir = _base / "output" / "growth_research" / "market_neutral_lean_audit"
    output_dir.mkdir(parents=True, exist_ok=True)

    output = {
        'timestamp': datetime.now().isoformat(),
        'runtime_seconds': round(elapsed, 1),
        'strategy': {
            'name': 'Market-Neutral L3/S3 Sector Rotation',
            'n_features': len(FEATURE_COLS),
            'features': FEATURE_COLS,
            'train_window': TRAIN_WINDOW,
            'rebal_period': REBAL_PERIOD,
            'n_long': N_LONG,
            'n_short': N_SHORT,
        },
        'baseline': {
            'sharpe': round(sharpe, 3),
            'sortino': round(sortino, 3),
            'mdd_pct': round(mdd * 100, 2),
            'win_rate_pct': round(win_rate, 1),
            'profit_factor': round(profit_factor, 2),
            'beta': round(beta, 4),
            'total_return_pct': round(total_ret, 1),
            'n_periods': len(period_returns),
        },
        'tests': results,
        'n_passed': n_passed,
        'verdict': verdict,
    }

    json_path = output_dir / "lean_audit_results.json"
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    fprint(f"\nResults saved to {json_path}")

    # --- MLflow logging ---
    if USE_MLFLOW:
        try:
            with mlflow.start_run(run_name=f"lean_audit_{datetime.now().strftime('%Y%m%d_%H%M')}"):
                mlflow.log_param("n_features", len(FEATURE_COLS))
                mlflow.log_param("train_window", TRAIN_WINDOW)
                mlflow.log_param("rebal_period", REBAL_PERIOD)
                mlflow.log_param("n_long", N_LONG)
                mlflow.log_param("n_short", N_SHORT)
                mlflow.log_param("n_sectors", len(SECTOR_ETFS))
                mlflow.log_param("lgbm_n_estimators", LGBM_PARAMS['n_estimators'])
                mlflow.log_param("lgbm_max_depth", LGBM_PARAMS['max_depth'])

                mlflow.log_metric("sharpe", sharpe)
                mlflow.log_metric("sortino", sortino if sortino != float('inf') else 99.0)
                mlflow.log_metric("mdd_pct", mdd * 100)
                mlflow.log_metric("win_rate", win_rate)
                mlflow.log_metric("profit_factor", min(profit_factor, 99.0))
                mlflow.log_metric("beta", beta)
                mlflow.log_metric("total_return_pct", total_ret)

                mlflow.log_metric("perm_p_value", results[0]['p_value'])
                mlflow.log_metric("n_tests_passed", n_passed)
                mlflow.log_metric("runtime_seconds", elapsed)

                mlflow.log_artifact(str(json_path))
                mlflow.set_tag("verdict", verdict)
                fprint("[MLflow] Run logged successfully.")
        except Exception as e:
            fprint(f"[WARN] MLflow logging failed: {e}")

    fprint(f"\nDone. Verdict: {verdict}")
    return output


if __name__ == '__main__':
    main()
