#!/usr/bin/env python3
"""
Market-Neutral Equity Rotation — Fast Adversarial Audit v1
============================================================
Audits the L3/S3 market-neutral sector rotation strategy (KB #285 / entry 1206)
with 8 checks. The KEY insight: LGBM trains ONCE, then permutation/random tests
shuffle the RANKINGS (not retrain the model), making 500 permutations take
seconds instead of hours.

Strategy under audit:
  - LGBM ranks 11 sector ETFs using 17 features
  - Long top-3, Short bottom-3 (L3/S3), monthly rebalance
  - Sliding 500-day LGBM training window
  - Reported: Sharpe 3.26, MDD -2.2%, WR 86.5%, Beta 0.01

8 Audit Checks:
  1. Look-ahead features       — |corr(feature, future_ret)| > 0.5 = FAIL
  2. Label leakage             — |corr(train_label, feature)| > 0.5 = FAIL
  3. Walk-forward integrity    — Any train date >= rebal date = FAIL
  4. Permutation test (500x)   — Shuffle rankings, actual Sharpe > p95 = PASS
  5. Sub-period stability      — >=3/4 quarters with Sharpe > 0 = PASS
  6. Borrow cost stress        — Add 0.5% annual borrow, Sharpe > 0.5 = PASS
  7. Outlier removal           — Drop top 5% monthly returns, Sharpe > 0.5 = PASS
  8. Random ranking baseline   — Actual Sharpe > 2x random mean = PASS

Final: VALIDATED if >= 7/8 pass.

MLflow experiment: market_neutral_fast_audit
Target runtime: < 10 minutes on 32-core machine.
"""

import warnings
warnings.filterwarnings('ignore')

import sys
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# MLflow setup
# ---------------------------------------------------------------------------
MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "market_neutral_fast_audit"
USE_MLFLOW = True

try:
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
except Exception as e:
    print(f"[WARN] MLflow unavailable: {e}. Continuing without tracking.")
    USE_MLFLOW = False

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_builtin_print = print
def fprint(*args, **kwargs):
    _builtin_print(*args, **kwargs)
    sys.stdout.flush()

# ---------------------------------------------------------------------------
# Constants (match the original strategy exactly)
# ---------------------------------------------------------------------------
SECTOR_ETFS = ['XLB', 'XLC', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLRE', 'XLU', 'XLV', 'XLY']
BENCHMARK = 'SPY'
ALL_TICKERS = SECTOR_ETFS + [BENCHMARK]

TRAIN_WINDOW = 500       # trading days
REBAL_PERIOD = 21        # ~monthly rebalance
STARTING_CAPITAL = 10000
N_LONG = 3
N_SHORT = 3

FEATURE_COLS = [
    'ret_5d', 'ret_10d', 'ret_21d', 'ret_63d',
    'vol_21d', 'vol_ratio', 'rsi_14', 'macd', 'macd_signal', 'bb_pct',
    'obv_slope', 'atr_pct', 'sector_rel_strength',
    'skew_21d', 'kurt_21d', 'max_dd_21d', 'up_down_vol_ratio',
]

LGBM_PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'n_jobs': -1,
    'seed': 42,
}
LGBM_ROUNDS = 200

# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------
def download_data(start='2005-01-01', end=None):
    """Download all ETF data via yfinance."""
    import yfinance as yf
    fprint(f"Downloading {len(ALL_TICKERS)} tickers from {start} ...")
    if end is None:
        end = datetime.now().strftime('%Y-%m-%d')
    data = yf.download(ALL_TICKERS, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        close = data['Close']
        volume = data['Volume']
        high = data['High']
        low = data['Low']
    else:
        close = data[['Close']].copy()
        volume = data[['Volume']].copy()
        high = data[['High']].copy()
        low = data[['Low']].copy()
    fprint(f"  {len(close)} trading days, {close.index[0].date()} to {close.index[-1].date()}")
    return close, volume, high, low

# ---------------------------------------------------------------------------
# Feature computation (identical to original strategy)
# ---------------------------------------------------------------------------
def compute_features(close, volume, high, low):
    """Compute 17 features for each sector ETF on each date. Returns DataFrame."""
    spy_ret = close[BENCHMARK].pct_change()
    records = []

    for ticker in SECTOR_ETFS:
        c = close[ticker]
        v = volume[ticker]
        h = high[ticker]
        lo = low[ticker]
        ret = c.pct_change()

        ret_5d = c.pct_change(5)
        ret_10d = c.pct_change(10)
        ret_21d = c.pct_change(21)
        ret_63d = c.pct_change(63)

        vol_21d = ret.rolling(21).std()
        vol_5d = ret.rolling(5).std()
        vol_ratio = vol_5d / vol_21d

        delta = ret.copy()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi_14 = 100 - (100 / (1 + rs))

        ema12 = c.ewm(span=12).mean()
        ema26 = c.ewm(span=26).mean()
        macd = ema12 - ema26
        macd_signal = macd.ewm(span=9).mean()

        sma20 = c.rolling(20).mean()
        std20 = c.rolling(20).std()
        bb_pct = (c - sma20) / (2 * std20)

        obv = (np.sign(ret) * v).cumsum()
        obv_slope = obv.rolling(21).apply(
            lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) == 21 else np.nan,
            raw=True
        )
        obv_slope = obv_slope / v.rolling(21).mean()

        tr = pd.concat([h - lo, (h - c.shift(1)).abs(), (lo - c.shift(1)).abs()], axis=1).max(axis=1)
        atr14 = tr.rolling(14).mean()
        atr_pct = atr14 / c

        sector_rel_strength = ret_21d - spy_ret.rolling(21).apply(
            lambda x: (1 + x).prod() - 1 if len(x) == 21 else np.nan, raw=False
        )

        skew_21d = ret.rolling(21).skew()
        kurt_21d = ret.rolling(21).kurt()

        def max_dd_window(x):
            cum = (1 + x).cumprod()
            peak = cum.cummax()
            return (cum / peak - 1).min()

        max_dd_21d = ret.rolling(21).apply(max_dd_window, raw=False)

        up_vol = (ret.clip(lower=0) * v).rolling(21).sum()
        down_vol = ((-ret.clip(upper=0)) * v).rolling(21).sum()
        up_down_vol_ratio = up_vol / down_vol.replace(0, np.nan)

        fwd_ret_21d = c.pct_change(21).shift(-21)

        df = pd.DataFrame({
            'date': c.index,
            'ticker': ticker,
            'ret_5d': ret_5d.values,
            'ret_10d': ret_10d.values,
            'ret_21d': ret_21d.values,
            'ret_63d': ret_63d.values,
            'vol_21d': vol_21d.values,
            'vol_ratio': vol_ratio.values,
            'rsi_14': rsi_14.values,
            'macd': macd.values,
            'macd_signal': macd_signal.values,
            'bb_pct': bb_pct.values,
            'obv_slope': obv_slope.values,
            'atr_pct': atr_pct.values,
            'sector_rel_strength': sector_rel_strength.values,
            'skew_21d': skew_21d.values,
            'kurt_21d': kurt_21d.values,
            'max_dd_21d': max_dd_21d.values,
            'up_down_vol_ratio': up_down_vol_ratio.values,
            'fwd_ret_21d': fwd_ret_21d.values,
            'close': c.values,
        })
        records.append(df)

    features_df = pd.concat(records, ignore_index=True)
    features_df = features_df.dropna(subset=FEATURE_COLS + ['fwd_ret_21d'])
    fprint(f"  Feature matrix: {len(features_df)} rows x {len(FEATURE_COLS)} features")
    return features_df

# ---------------------------------------------------------------------------
# Walk-forward LGBM ranking (runs ONCE)
# ---------------------------------------------------------------------------
def walk_forward_ranking(features_df, close):
    """
    Walk-forward LGBM ranking with sliding 500-day window, monthly rebalance.
    Returns:
        rankings: list of (rebal_date, {ticker: predicted_score})
        train_meta: list of (rebal_date, train_start_date, train_end_date, X_train_features, y_train_labels)
            — used for leakage/integrity checks
    """
    dates = sorted(features_df['date'].unique())
    all_close_dates = close.index.tolist()

    valid_start_idx = TRAIN_WINDOW + 63 + 21
    rebal_dates = []
    for i in range(valid_start_idx, len(all_close_dates), REBAL_PERIOD):
        d = all_close_dates[i]
        if d in dates:
            rebal_dates.append(d)

    fprint(f"  {len(rebal_dates)} rebalance dates "
           f"({rebal_dates[0].date()} to {rebal_dates[-1].date()})")

    rankings = []
    train_meta = []  # for audit checks

    for rebal_date in rebal_dates:
        mask_train = features_df['date'] < rebal_date
        train_pool = features_df[mask_train].copy()

        train_dates_unique = sorted(train_pool['date'].unique())
        if len(train_dates_unique) < TRAIN_WINDOW:
            continue
        cutoff_date = train_dates_unique[-TRAIN_WINDOW]
        train_pool = train_pool[train_pool['date'] >= cutoff_date]

        pred_pool = features_df[features_df['date'] == rebal_date].copy()
        if len(pred_pool) < len(SECTOR_ETFS) * 0.5:
            continue

        X_train = train_pool[FEATURE_COLS].values
        y_train = train_pool['fwd_ret_21d'].values
        X_pred = pred_pool[FEATURE_COLS].values

        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        y_train = np.nan_to_num(y_train, nan=0.0)
        X_pred = np.nan_to_num(X_pred, nan=0.0, posinf=0.0, neginf=0.0)

        train_ds = lgb.Dataset(X_train, label=y_train)
        callbacks = [lgb.log_evaluation(period=-1)]
        model = lgb.train(LGBM_PARAMS, train_ds, num_boost_round=LGBM_ROUNDS, callbacks=callbacks)

        preds = model.predict(X_pred)
        ticker_scores = dict(zip(pred_pool['ticker'].values, preds))
        rankings.append((rebal_date, ticker_scores))

        # Save metadata for audit
        train_meta.append({
            'rebal_date': rebal_date,
            'train_start': train_pool['date'].min(),
            'train_end': train_pool['date'].max(),
            'X_train': X_train,
            'y_train': y_train,
        })

    fprint(f"  Generated {len(rankings)} rankings")
    return rankings, train_meta

# ---------------------------------------------------------------------------
# Backtest engine for L3/S3
# ---------------------------------------------------------------------------
def backtest_ls(rankings, close, n_long=N_LONG, n_short=N_SHORT,
                borrow_cost_annual=0.0):
    """
    Backtest L3/S3 equal-weight strategy from pre-computed rankings.
    Returns equity curve DataFrame and array of per-period returns.

    borrow_cost_annual: annualized borrow cost applied to short leg per period.
    """
    equity = STARTING_CAPITAL
    equity_curve = []
    period_returns = []
    weight_long = 1.0 / (n_long + n_short)   # equal weight each position
    weight_short = 1.0 / (n_long + n_short)

    for i in range(len(rankings) - 1):
        rebal_date, scores = rankings[i]
        next_date = rankings[i + 1][0]

        sorted_sectors = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        longs = [t for t, _ in sorted_sectors[:n_long]]
        shorts = [t for t, _ in sorted_sectors[-n_short:]]

        period_ret = 0.0
        for t in longs:
            try:
                r = close[t].loc[next_date] / close[t].loc[rebal_date] - 1.0
                period_ret += weight_long * r
            except (KeyError, TypeError):
                pass
        for t in shorts:
            try:
                r = close[t].loc[next_date] / close[t].loc[rebal_date] - 1.0
                period_ret -= weight_short * r  # short: profit when price drops
            except (KeyError, TypeError):
                pass

        # Apply borrow cost to short leg
        if borrow_cost_annual > 0 and n_short > 0:
            # Approximate holding period in years
            try:
                days_held = (next_date - rebal_date).days
            except Exception:
                days_held = REBAL_PERIOD * 365.25 / 252
            frac_year = days_held / 365.25
            borrow_drag = borrow_cost_annual * frac_year * (n_short * weight_short)
            period_ret -= borrow_drag

        equity *= (1 + period_ret)
        equity_curve.append({'date': next_date, 'equity': equity, 'return': period_ret})
        period_returns.append(period_ret)

    return pd.DataFrame(equity_curve), np.array(period_returns)

# ---------------------------------------------------------------------------
# Sharpe helper
# ---------------------------------------------------------------------------
def annualized_sharpe(returns, periods_per_year=None):
    """Annualized Sharpe from per-rebalance returns."""
    if periods_per_year is None:
        periods_per_year = 252.0 / REBAL_PERIOD
    if len(returns) < 2 or np.std(returns) == 0:
        return 0.0
    return float(np.mean(returns) / np.std(returns) * np.sqrt(periods_per_year))

# ===========================================================================
# 8 AUDIT CHECKS
# ===========================================================================

def check_1_lookahead(features_df):
    """
    Check 1 — Look-Ahead Features.
    For each feature, compute rank correlation with the FUTURE return
    (fwd_ret_21d) across the entire dataset. If |r| > 0.5 for any
    feature, FAIL — that feature may embed future information.
    """
    fprint("\n[Check 1] Look-Ahead Feature Correlation")
    from scipy.stats import spearmanr

    max_abs_r = 0.0
    worst_feat = None
    details = []

    for feat in FEATURE_COLS:
        valid = features_df[[feat, 'fwd_ret_21d']].dropna()
        if len(valid) < 50:
            continue
        r, p = spearmanr(valid[feat], valid['fwd_ret_21d'])
        details.append((feat, r))
        if abs(r) > max_abs_r:
            max_abs_r = abs(r)
            worst_feat = feat

    # Print top offenders
    details.sort(key=lambda x: abs(x[1]), reverse=True)
    for feat, r in details[:5]:
        fprint(f"    {feat:25s} r={r:+.4f}")

    passed = max_abs_r < 0.5
    fprint(f"  Worst: {worst_feat} |r|={max_abs_r:.4f}  "
           f"{'PASS' if passed else 'FAIL'} (threshold 0.50)")
    return passed, {'max_abs_r': round(max_abs_r, 4), 'worst_feature': worst_feat}


def check_2_label_leakage(train_meta):
    """
    Check 2 — Label Leakage.
    For a sample of training windows, compute correlation between each feature
    and the training label (fwd_ret_21d). If any |r| > 0.5 consistently, FAIL.
    """
    fprint("\n[Check 2] Label Leakage (feature-label correlation in training data)")
    from scipy.stats import spearmanr

    # Sample up to 20 windows to keep it fast
    sample_indices = np.linspace(0, len(train_meta) - 1, min(20, len(train_meta)), dtype=int)

    feat_max_r = {f: [] for f in FEATURE_COLS}

    for idx in sample_indices:
        meta = train_meta[idx]
        X = meta['X_train']
        y = meta['y_train']
        for j, feat in enumerate(FEATURE_COLS):
            r, _ = spearmanr(X[:, j], y)
            if np.isfinite(r):
                feat_max_r[feat].append(abs(r))

    # Report mean |r| per feature
    worst_feat = None
    worst_mean_r = 0.0
    details = []
    for feat in FEATURE_COLS:
        vals = feat_max_r[feat]
        if vals:
            mean_r = np.mean(vals)
            details.append((feat, mean_r))
            if mean_r > worst_mean_r:
                worst_mean_r = mean_r
                worst_feat = feat

    details.sort(key=lambda x: x[1], reverse=True)
    for feat, r in details[:5]:
        fprint(f"    {feat:25s} mean|r|={r:.4f}")

    passed = worst_mean_r < 0.5
    fprint(f"  Worst: {worst_feat} mean|r|={worst_mean_r:.4f}  "
           f"{'PASS' if passed else 'FAIL'} (threshold 0.50)")
    return passed, {'worst_mean_r': round(worst_mean_r, 4), 'worst_feature': worst_feat}


def check_3_walkforward_integrity(train_meta):
    """
    Check 3 — Walk-Forward Integrity.
    Verify that ALL training data dates are strictly before the rebalance date.
    Any overlap = FAIL.
    """
    fprint("\n[Check 3] Walk-Forward Integrity (no train/test overlap)")
    violations = 0
    total_checked = len(train_meta)

    for meta in train_meta:
        rebal = meta['rebal_date']
        train_end = meta['train_end']
        if train_end >= rebal:
            violations += 1

    passed = violations == 0
    fprint(f"  Checked {total_checked} rebalance windows, {violations} violations  "
           f"{'PASS' if passed else 'FAIL'}")
    return passed, {'violations': violations, 'windows_checked': total_checked}


def check_4_permutation_test(rankings, close, actual_sharpe, n_perms=500):
    """
    Check 4 — Permutation Test (500 shuffles).
    At each rebalance date, randomly shuffle which sectors are assigned
    which rank scores. Then recompute the L/S portfolio. This tests
    whether the LGBM ordering matters, without retraining.
    """
    fprint(f"\n[Check 4] Permutation Test ({n_perms} shuffles)")
    rng = np.random.RandomState(42)
    perm_sharpes = np.zeros(n_perms)

    for p in range(n_perms):
        shuffled_rankings = []
        for date, scores in rankings:
            tickers = list(scores.keys())
            vals = list(scores.values())
            rng.shuffle(vals)
            shuffled_rankings.append((date, dict(zip(tickers, vals))))

        _, perm_rets = backtest_ls(shuffled_rankings, close)
        perm_sharpes[p] = annualized_sharpe(perm_rets)

    percentile_95 = np.percentile(perm_sharpes, 95)
    percentile_99 = np.percentile(perm_sharpes, 99)
    perm_mean = np.mean(perm_sharpes)
    perm_std = np.std(perm_sharpes)
    p_value = float(np.mean(perm_sharpes >= actual_sharpe))

    passed = actual_sharpe > percentile_95
    fprint(f"  Actual Sharpe:  {actual_sharpe:.3f}")
    fprint(f"  Perm mean:      {perm_mean:.3f} +/- {perm_std:.3f}")
    fprint(f"  Perm 95th pctl: {percentile_95:.3f}")
    fprint(f"  Perm 99th pctl: {percentile_99:.3f}")
    fprint(f"  p-value:        {p_value:.4f}")
    fprint(f"  {'PASS' if passed else 'FAIL'} (actual > 95th percentile)")
    return passed, {
        'actual_sharpe': round(actual_sharpe, 4),
        'perm_mean': round(perm_mean, 4),
        'perm_std': round(perm_std, 4),
        'perm_p95': round(percentile_95, 4),
        'perm_p99': round(percentile_99, 4),
        'p_value': round(p_value, 4),
    }


def check_5_subperiod_stability(period_returns, rankings):
    """
    Check 5 — Sub-Period Stability.
    Split rebalance periods into 4 roughly equal quarters.
    Require >= 3/4 quarters with Sharpe > 0.
    """
    fprint("\n[Check 5] Sub-Period Stability (4 quarters)")
    n = len(period_returns)
    if n < 4:
        fprint("  Not enough periods for sub-period analysis. FAIL")
        return False, {'n_periods': n, 'positive_quarters': 0}

    quarter_size = n // 4
    positive_quarters = 0
    details = []

    for q in range(4):
        start = q * quarter_size
        end = (q + 1) * quarter_size if q < 3 else n
        q_rets = period_returns[start:end]
        q_sharpe = annualized_sharpe(q_rets)
        is_pos = q_sharpe > 0
        if is_pos:
            positive_quarters += 1

        # Get date range
        q_start_date = rankings[start + 1][0].date() if start + 1 < len(rankings) else "?"
        q_end_idx = min(end, len(rankings) - 1)
        q_end_date = rankings[q_end_idx][0].date() if q_end_idx < len(rankings) else "?"
        details.append((q + 1, q_sharpe, len(q_rets), q_start_date, q_end_date))
        fprint(f"    Q{q+1} ({q_start_date} to {q_end_date}): "
               f"Sharpe={q_sharpe:+.3f}, n={len(q_rets)}")

    passed = positive_quarters >= 3
    fprint(f"  {positive_quarters}/4 quarters Sharpe > 0  "
           f"{'PASS' if passed else 'FAIL'} (need >= 3)")
    return passed, {'positive_quarters': positive_quarters,
                    'quarter_sharpes': [d[1] for d in details]}


def check_6_borrow_cost(rankings, close):
    """
    Check 6 — Borrow Cost Stress Test.
    Add 0.5% annual borrow cost to short positions and verify
    Sharpe remains > 0.5.
    """
    fprint("\n[Check 6] Borrow Cost Stress (0.5% annual)")
    _, rets_with_borrow = backtest_ls(rankings, close, borrow_cost_annual=0.005)
    sharpe_with_borrow = annualized_sharpe(rets_with_borrow)

    passed = sharpe_with_borrow > 0.5
    fprint(f"  Sharpe with 0.5% borrow cost: {sharpe_with_borrow:.3f}  "
           f"{'PASS' if passed else 'FAIL'} (threshold 0.50)")
    return passed, {'sharpe_with_borrow': round(sharpe_with_borrow, 4)}


def check_7_outlier_removal(period_returns):
    """
    Check 7 — Outlier Removal.
    Remove top 5% of monthly returns (by magnitude). If Sharpe > 0.5,
    the strategy doesn't depend on a few lucky months.
    """
    fprint("\n[Check 7] Outlier Removal (drop top 5% returns)")
    n = len(period_returns)
    cutoff_idx = int(n * 0.95)

    # Sort by return value, remove top 5%
    sorted_indices = np.argsort(period_returns)
    keep_indices = sorted_indices[:cutoff_idx]
    trimmed = period_returns[keep_indices]

    sharpe_trimmed = annualized_sharpe(trimmed)
    n_removed = n - len(trimmed)

    passed = sharpe_trimmed > 0.5
    fprint(f"  Removed {n_removed}/{n} periods (top 5% returns)")
    fprint(f"  Sharpe after removal: {sharpe_trimmed:.3f}  "
           f"{'PASS' if passed else 'FAIL'} (threshold 0.50)")
    return passed, {'sharpe_trimmed': round(sharpe_trimmed, 4),
                    'n_removed': n_removed}


def check_8_random_baseline(rankings, close, n_trials=100):
    """
    Check 8 — Random Ranking Baseline (100 trials).
    At each rebalance, assign completely random rankings (not shuffled LGBM
    scores — fresh uniform random). If actual Sharpe > 2x random mean, PASS.
    """
    fprint(f"\n[Check 8] Random Ranking Baseline ({n_trials} trials)")
    rng = np.random.RandomState(123)
    random_sharpes = np.zeros(n_trials)

    for trial in range(n_trials):
        random_rankings = []
        for date, scores in rankings:
            tickers = list(scores.keys())
            # Completely random scores, independent of LGBM
            random_scores = rng.randn(len(tickers))
            random_rankings.append((date, dict(zip(tickers, random_scores))))

        _, rand_rets = backtest_ls(random_rankings, close)
        random_sharpes[trial] = annualized_sharpe(rand_rets)

    rand_mean = np.mean(random_sharpes)
    rand_std = np.std(random_sharpes)
    rand_median = np.median(random_sharpes)

    # Use the actual sharpe from the main run (passed in via the caller)
    # We return the random stats; the caller compares
    fprint(f"  Random mean Sharpe:   {rand_mean:.3f} +/- {rand_std:.3f}")
    fprint(f"  Random median Sharpe: {rand_median:.3f}")

    return rand_mean, rand_std, random_sharpes


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    t0 = time.time()
    fprint("=" * 70)
    fprint("  Market-Neutral Equity Rotation — Fast Adversarial Audit v1")
    fprint("=" * 70)

    # ── Phase 1: Download data + compute features ──
    fprint("\n[Phase 1] Data & Features")
    close, volume, high, low = download_data(start='2005-01-01')
    spy_close = close[BENCHMARK].dropna()
    features_df = compute_features(close, volume, high, low)

    # ── Phase 2: Run LGBM walk-forward ONCE ──
    fprint("\n[Phase 2] Walk-Forward LGBM Ranking (single pass)")
    t_wf = time.time()
    rankings, train_meta = walk_forward_ranking(features_df, close)
    wf_elapsed = time.time() - t_wf
    fprint(f"  Walk-forward completed in {wf_elapsed:.1f}s")

    if len(rankings) < 5:
        fprint("ERROR: Not enough rankings generated. Aborting.")
        return

    # ── Phase 3: Backtest actual strategy ──
    fprint("\n[Phase 3] Backtest Actual L3/S3 Strategy")
    equity_df, period_returns = backtest_ls(rankings, close)
    actual_sharpe = annualized_sharpe(period_returns)

    # Compute additional metrics
    periods_per_year = 252.0 / REBAL_PERIOD
    sortino_val = 0.0
    downside = period_returns[period_returns < 0]
    if len(downside) > 0 and np.std(downside) > 0:
        sortino_val = float(np.mean(period_returns) / np.std(downside) * np.sqrt(periods_per_year))

    gains = period_returns[period_returns > 0].sum()
    losses = abs(period_returns[period_returns < 0].sum())
    pf = gains / losses if losses > 0 else float('inf')

    wr = float(np.mean(period_returns > 0) * 100)

    cum_ret = np.cumprod(1 + period_returns)
    peak = np.maximum.accumulate(cum_ret)
    mdd = float(np.min(cum_ret / peak - 1) * 100)

    total_return = float((cum_ret[-1] - 1) * 100)

    # Beta vs SPY
    spy_period_rets = []
    for i in range(len(rankings) - 1):
        d0 = rankings[i][0]
        d1 = rankings[i + 1][0]
        try:
            s0 = spy_close.loc[d0]
            s1 = spy_close.loc[d1]
            spy_period_rets.append(float(s1 / s0 - 1))
        except (KeyError, TypeError):
            spy_period_rets.append(0.0)
    spy_arr = np.array(spy_period_rets)
    if len(spy_arr) > 2 and np.var(spy_arr) > 0:
        beta = float(np.cov(period_returns, spy_arr)[0, 1] / np.var(spy_arr))
    else:
        beta = 0.0

    fprint(f"  Sharpe:  {actual_sharpe:.3f}")
    fprint(f"  Sortino: {sortino_val:.3f}")
    fprint(f"  PF:      {pf:.2f}")
    fprint(f"  WR:      {wr:.1f}%")
    fprint(f"  MDD:     {mdd:.1f}%")
    fprint(f"  Return:  {total_return:.1f}%")
    fprint(f"  Beta:    {beta:.3f}")
    fprint(f"  Periods: {len(period_returns)}")

    # ── Phase 4: Run all 8 audit checks ──
    fprint("\n" + "=" * 70)
    fprint("  RUNNING 8 AUDIT CHECKS")
    fprint("=" * 70)

    results = {}

    # Check 1: Look-ahead
    p1, d1 = check_1_lookahead(features_df)
    results['check_1_lookahead'] = {'pass': p1, 'details': d1}

    # Check 2: Label leakage
    p2, d2 = check_2_label_leakage(train_meta)
    results['check_2_label_leakage'] = {'pass': p2, 'details': d2}

    # Check 3: Walk-forward integrity
    p3, d3 = check_3_walkforward_integrity(train_meta)
    results['check_3_wf_integrity'] = {'pass': p3, 'details': d3}

    # Check 4: Permutation test
    t_perm = time.time()
    p4, d4 = check_4_permutation_test(rankings, close, actual_sharpe, n_perms=500)
    perm_elapsed = time.time() - t_perm
    fprint(f"  (permutation test took {perm_elapsed:.1f}s)")
    results['check_4_permutation'] = {'pass': p4, 'details': d4}

    # Check 5: Sub-period stability
    p5, d5 = check_5_subperiod_stability(period_returns, rankings)
    results['check_5_subperiod'] = {'pass': p5, 'details': d5}

    # Check 6: Borrow cost
    p6, d6 = check_6_borrow_cost(rankings, close)
    results['check_6_borrow_cost'] = {'pass': p6, 'details': d6}

    # Check 7: Outlier removal
    p7, d7 = check_7_outlier_removal(period_returns)
    results['check_7_outlier'] = {'pass': p7, 'details': d7}

    # Check 8: Random baseline
    rand_mean, rand_std, random_sharpes = check_8_random_baseline(rankings, close, n_trials=100)
    threshold_8 = 2.0 * abs(rand_mean) if rand_mean != 0 else 0.5
    p8 = actual_sharpe > threshold_8
    fprint(f"  Actual Sharpe: {actual_sharpe:.3f} vs 2x random mean: {threshold_8:.3f}  "
           f"{'PASS' if p8 else 'FAIL'}")
    results['check_8_random_baseline'] = {
        'pass': p8,
        'details': {
            'actual_sharpe': round(actual_sharpe, 4),
            'random_mean': round(rand_mean, 4),
            'random_std': round(rand_std, 4),
            'threshold_2x': round(threshold_8, 4),
        }
    }

    # ── Phase 5: Final Verdict ──
    total_elapsed = time.time() - t0

    n_pass = sum(1 for v in results.values() if v['pass'])
    n_total = len(results)
    validated = n_pass >= 7

    fprint("\n" + "=" * 70)
    fprint("  AUDIT RESULTS")
    fprint("=" * 70)
    check_names = {
        'check_1_lookahead':      '1. Look-Ahead Features',
        'check_2_label_leakage':  '2. Label Leakage',
        'check_3_wf_integrity':   '3. Walk-Forward Integrity',
        'check_4_permutation':    '4. Permutation Test (500x)',
        'check_5_subperiod':      '5. Sub-Period Stability',
        'check_6_borrow_cost':    '6. Borrow Cost (0.5%)',
        'check_7_outlier':        '7. Outlier Removal',
        'check_8_random_baseline':'8. Random Baseline (100x)',
    }
    for key, name in check_names.items():
        status = "PASS" if results[key]['pass'] else "FAIL"
        fprint(f"  [{status}] {name}")

    fprint(f"\n  Score: {n_pass}/{n_total}")
    verdict = "VALIDATED" if validated else "REJECTED"
    fprint(f"  Verdict: {verdict} (need >= 7/8)")
    fprint(f"\n  Strategy metrics: Sharpe={actual_sharpe:.3f}, Sortino={sortino_val:.3f}, "
           f"PF={pf:.2f}, WR={wr:.1f}%, MDD={mdd:.1f}%, Beta={beta:.3f}")
    fprint(f"  Total runtime: {total_elapsed:.1f}s")

    # ── MLflow logging ──
    if USE_MLFLOW:
        try:
            run_name = f"fast_audit_L3S3_{datetime.now():%Y%m%d_%H%M}"
            with mlflow.start_run(run_name=run_name):
                mlflow.log_param("strategy", "L3/S3 market-neutral sector rotation")
                mlflow.log_param("n_sectors", len(SECTOR_ETFS))
                mlflow.log_param("train_window", TRAIN_WINDOW)
                mlflow.log_param("rebal_period", REBAL_PERIOD)
                mlflow.log_param("n_features", len(FEATURE_COLS))
                mlflow.log_param("n_rankings", len(rankings))
                mlflow.log_param("n_permutations", 500)
                mlflow.log_param("n_random_trials", 100)

                mlflow.log_metric("actual_sharpe", actual_sharpe)
                mlflow.log_metric("sortino", sortino_val)
                mlflow.log_metric("profit_factor", pf)
                mlflow.log_metric("win_rate", wr)
                mlflow.log_metric("max_drawdown_pct", mdd)
                mlflow.log_metric("total_return_pct", total_return)
                mlflow.log_metric("beta", beta)

                mlflow.log_metric("checks_passed", n_pass)
                mlflow.log_metric("checks_total", n_total)
                mlflow.log_metric("validated", 1 if validated else 0)

                for key in results:
                    mlflow.log_metric(f"{key}_pass", 1 if results[key]['pass'] else 0)

                # Log key detail metrics
                d4 = results['check_4_permutation']['details']
                mlflow.log_metric("perm_p_value", d4['p_value'])
                mlflow.log_metric("perm_p95", d4['perm_p95'])

                d6 = results['check_6_borrow_cost']['details']
                mlflow.log_metric("sharpe_with_borrow", d6['sharpe_with_borrow'])

                d7 = results['check_7_outlier']['details']
                mlflow.log_metric("sharpe_trimmed", d7['sharpe_trimmed'])

                d8 = results['check_8_random_baseline']['details']
                mlflow.log_metric("random_mean_sharpe", d8['random_mean'])

                mlflow.log_metric("wf_time_seconds", wf_elapsed)
                mlflow.log_metric("total_time_seconds", total_elapsed)

                fprint("  MLflow run logged successfully.")
        except Exception as e:
            fprint(f"  MLflow logging failed: {e}")

    fprint(f"\nDone. ({total_elapsed:.0f}s)")
    return verdict, results


if __name__ == '__main__':
    main()
