#!/usr/bin/env python3
"""
long_horizon_flow_v2_validation.py — Autocorrelation & robustness validation

CRITICAL CAVEAT: 3-day returns overlap → daily PnL series is autocorrelated → Sharpe is inflated.
This script:
1. Reconstructs the daily PnL series from the training output
2. Runs Ljung-Box test + ACF analysis on daily PnL
3. Computes Newey-West corrected Sharpe
4. Computes non-overlapping-period Sharpe (every 3rd day)
5. Measures position overlap %
6. Bootstrap CI on Sharpe
7. First-half vs second-half OOT stability
8. Logs everything to MLflow

Author: Claude (Head of Quant)
Date: 2026-06-23
"""
from __future__ import annotations

import json, logging, os, sys, time, warnings, re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
warnings.filterwarnings('ignore')

import mlflow

try:
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.tsa.stattools import acf
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools import add_constant
    from statsmodels.stats.sandwich_covariance import cov_hac
except ImportError:
    print("ERROR: statsmodels not installed"); sys.exit(1)

try:
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
except ImportError:
    print("ERROR: lightgbm/sklearn not installed"); sys.exit(1)

ROOT = Path("/home/nick/Lvl3Quant")
OUT_DIR = ROOT / "output" / "long_horizon_flow_v2"
VALIDATION_DIR = OUT_DIR / "validation"
VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = ROOT / "logs" / f"lh_flow_v2_validation_{time.strftime('%Y%m%d_%H%M%S')}.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stdout)])
log = logging.getLogger("lh_flow_v2_val")

ES_TICK_VALUE = 12.50
MARKET_COST_TICKS = 1.376
TICK_SIZE = 0.25

# Walk-forward params (must match training)
WF_TRAIN = 60
WF_TEST = 1
TARGET_COL = 'fwd_direction_3d'
HOLD_DAYS = 3
THRESHOLD = 0.55  # Best regime-passing threshold from results


def reconstruct_predictions(df, feat_cols):
    """Re-run walk-forward to get per-day predictions + trades."""
    log.info("Reconstructing walk-forward predictions...")
    dates = sorted(df['date'].unique())
    n_dates = len(dates)
    
    records = []
    
    for i in range(WF_TRAIN, n_dates):
        test_date = dates[i]
        train_dates = dates[i - WF_TRAIN:i]
        
        train_mask = df['date'].isin(train_dates)
        test_mask = df['date'] == test_date
        
        X_train = df.loc[train_mask, feat_cols].values
        y_train = df.loc[train_mask, TARGET_COL].values
        X_test = df.loc[test_mask, feat_cols].values
        
        valid_train = ~np.isnan(y_train)
        if valid_train.sum() < 20:
            continue
        X_train = X_train[valid_train]
        y_train = y_train[valid_train]
        
        # Replace NaN/inf in features
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
        
        params = {
            'objective': 'binary', 'metric': 'auc', 'verbosity': -1,
            'learning_rate': 0.05, 'num_leaves': 31, 'max_depth': 5,
            'min_child_samples': 10, 'subsample': 0.8, 'colsample_bytree': 0.8,
            'reg_alpha': 0.1, 'reg_lambda': 1.0, 'n_estimators': 200,
        }
        
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_train, y_train)],
                  callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(-1)])
        
        pred = model.predict_proba(X_test)[:, 1]
        
        # Get actual return
        test_row = df.loc[test_mask].iloc[0]
        fwd_ret = test_row.get('fwd_return_3d_ticks', np.nan)
        actual_dir = test_row.get(TARGET_COL, np.nan)
        
        if pd.isna(fwd_ret) or pd.isna(actual_dir):
            records.append({
                'date': test_date, 'pred': pred[0], 'fwd_return_3d_ticks': np.nan,
                'actual_dir': np.nan, 'direction': 0, 'pnl_ticks': 0, 'traded': False
            })
            continue
        
        # Trade decision
        p = pred[0]
        if p > THRESHOLD:
            direction = 1
        elif p < (1 - THRESHOLD):
            direction = -1
        else:
            direction = 0
        
        traded = direction != 0
        pnl_ticks = direction * fwd_ret - MARKET_COST_TICKS if traded else 0.0
        
        records.append({
            'date': test_date, 'pred': p, 'fwd_return_3d_ticks': fwd_ret,
            'actual_dir': actual_dir, 'direction': direction, 'pnl_ticks': pnl_ticks,
            'traded': traded,
        })
    
    trades_df = pd.DataFrame(records)
    log.info(f"  Reconstructed {len(trades_df)} OOT days, {trades_df['traded'].sum()} trades")
    return trades_df


def analyze_autocorrelation(daily_pnl):
    """Ljung-Box test + ACF on daily PnL series."""
    log.info("\n" + "=" * 70)
    log.info("AUTOCORRELATION ANALYSIS")
    log.info("=" * 70)
    
    results = {}
    
    # ACF values
    n_lags = min(15, len(daily_pnl) // 4)
    acf_vals, confint = acf(daily_pnl, nlags=n_lags, alpha=0.05)
    
    log.info("ACF of daily PnL:")
    for lag in range(1, min(8, n_lags + 1)):
        sig = "*" if abs(acf_vals[lag]) > 1.96 / np.sqrt(len(daily_pnl)) else ""
        log.info(f"  Lag {lag}: {acf_vals[lag]:+.4f} {sig}")
    
    results['acf_lag1'] = float(acf_vals[1])
    results['acf_lag2'] = float(acf_vals[2])
    results['acf_lag3'] = float(acf_vals[3]) if n_lags >= 3 else 0
    
    # Ljung-Box test (lags 1-5)
    lb_result = acorr_ljungbox(daily_pnl, lags=[1, 2, 3, 5], return_df=True)
    log.info("\nLjung-Box test:")
    for lag_val in lb_result.index:
        stat = lb_result.loc[lag_val, 'lb_stat']
        pval = lb_result.loc[lag_val, 'lb_pvalue']
        sig = "*** SIGNIFICANT" if pval < 0.05 else ""
        log.info(f"  Lag {lag_val}: stat={stat:.3f}, p={pval:.4f} {sig}")
    
    results['ljung_box_lag1_pval'] = float(lb_result.loc[lb_result.index[0], 'lb_pvalue'])
    results['ljung_box_lag3_pval'] = float(lb_result.loc[lb_result.index[2], 'lb_pvalue'])
    results['ljung_box_lag5_pval'] = float(lb_result.loc[lb_result.index[3], 'lb_pvalue'])
    
    # Durbin-Watson statistic
    diffs = np.diff(daily_pnl.values)
    dw = np.sum(diffs**2) / np.sum(daily_pnl.values**2)
    log.info(f"\nDurbin-Watson statistic: {dw:.4f} (2.0 = no AC, <2 = positive AC)")
    results['durbin_watson'] = float(dw)
    
    return results


def newey_west_sharpe(daily_pnl, max_lag=5):
    """Compute Sharpe with Newey-West corrected standard errors."""
    log.info("\n" + "=" * 70)
    log.info("NEWEY-WEST CORRECTED SHARPE")
    log.info("=" * 70)
    
    results = {}
    
    # Raw (naive) Sharpe
    raw_sharpe = daily_pnl.mean() / daily_pnl.std() * np.sqrt(252)
    log.info(f"Raw (naive) Sharpe: {raw_sharpe:.4f}")
    results['raw_sharpe'] = float(raw_sharpe)
    
    # Newey-West: regress PnL on constant, use HAC SEs
    y = daily_pnl.values
    X = add_constant(np.ones(len(y)))  # just intercept
    model = OLS(y, X[:, :1]).fit()  # OLS with constant only
    
    # HAC covariance (Newey-West)
    nw_cov = cov_hac(model, nlags=max_lag)
    nw_se = np.sqrt(nw_cov[0, 0])
    ols_se = model.bse[0]
    
    # Corrected Sharpe: scale the SE ratio
    se_inflation = nw_se / ols_se if ols_se > 0 else 1.0
    corrected_sharpe = raw_sharpe / se_inflation
    
    log.info(f"OLS SE of mean: {ols_se:.4f}")
    log.info(f"Newey-West SE (lag={max_lag}): {nw_se:.4f}")
    log.info(f"SE inflation factor: {se_inflation:.4f}")
    log.info(f"Newey-West corrected Sharpe: {corrected_sharpe:.4f}")
    log.info(f"Sharpe reduction: {(1 - corrected_sharpe/raw_sharpe)*100:.1f}%")
    
    results['nw_se'] = float(nw_se)
    results['ols_se'] = float(ols_se)
    results['se_inflation'] = float(se_inflation)
    results['nw_corrected_sharpe'] = float(corrected_sharpe)
    results['sharpe_reduction_pct'] = float((1 - corrected_sharpe/raw_sharpe)*100)
    
    # Also try different lag truncations
    for nl in [2, 3, 5, 7, 10]:
        if nl > len(daily_pnl) // 3:
            continue
        nw_c = cov_hac(model, nlags=nl)
        nw_s = np.sqrt(nw_c[0, 0])
        inf = nw_s / ols_se if ols_se > 0 else 1.0
        cs = raw_sharpe / inf
        log.info(f"  NW lag={nl}: SE_inflation={inf:.3f}, corrected_sharpe={cs:.3f}")
        results[f'nw_sharpe_lag{nl}'] = float(cs)
    
    return results


def non_overlapping_sharpe(trades_df):
    """Compute Sharpe using only non-overlapping 3-day periods."""
    log.info("\n" + "=" * 70)
    log.info("NON-OVERLAPPING PERIOD SHARPE")
    log.info("=" * 70)
    
    results = {}
    
    traded = trades_df[trades_df['traded']].copy()
    traded = traded.sort_values('date').reset_index(drop=True)
    
    if len(traded) == 0:
        log.info("No trades to analyze")
        return results
    
    # Method 1: Take every 3rd trade (non-overlapping holds)
    for stride in [2, 3, 4]:
        subset = traded.iloc[::stride]
        if len(subset) < 5:
            continue
        pnl = subset['pnl_ticks']
        sharpe = pnl.mean() / pnl.std() * np.sqrt(252 / stride) if pnl.std() > 0 else 0
        sortino_denom = pnl[pnl < 0].std()
        sortino = pnl.mean() / sortino_denom * np.sqrt(252 / stride) if sortino_denom > 0 else sharpe
        
        log.info(f"Every-{stride}-day sampling: N={len(subset)}, "
                 f"Sharpe={sharpe:.3f}, Sortino={sortino:.3f}, "
                 f"WR={((pnl > 0).mean()):.1%}, PF={(pnl[pnl>0].sum() / abs(pnl[pnl<0].sum()) if pnl[pnl<0].sum() != 0 else 0):.2f}")
        results[f'sharpe_stride{stride}'] = float(sharpe)
        results[f'sortino_stride{stride}'] = float(sortino)
        results[f'n_trades_stride{stride}'] = int(len(subset))
    
    # Method 2: Block the OOT period into non-overlapping 3-day blocks, take ONE trade per block
    dates = sorted(traded['date'].unique())
    date_to_idx = {d: i for i, d in enumerate(dates)}
    
    block_pnls = []
    block_start = 0
    while block_start < len(dates):
        block_end = min(block_start + HOLD_DAYS, len(dates))
        block_dates = dates[block_start:block_end]
        block_trades = traded[traded['date'].isin(block_dates)]
        if len(block_trades) > 0:
            # Take the FIRST trade in the block (most conservative)
            block_pnls.append(block_trades.iloc[0]['pnl_ticks'])
        block_start = block_end
    
    if len(block_pnls) > 3:
        bp = np.array(block_pnls)
        block_sharpe = bp.mean() / bp.std() * np.sqrt(252 / HOLD_DAYS) if bp.std() > 0 else 0
        log.info(f"\n3-day block Sharpe (non-overlapping): {block_sharpe:.3f} (N={len(bp)} blocks)")
        log.info(f"  Block mean PnL: {bp.mean():.1f} ticks, WR: {(bp > 0).mean():.1%}")
        results['block_sharpe_3d'] = float(block_sharpe)
        results['block_n'] = int(len(bp))
        results['block_mean_pnl'] = float(bp.mean())
        results['block_wr'] = float((bp > 0).mean())
    
    return results


def position_overlap_analysis(trades_df):
    """Analyze what % of days have overlapping positions."""
    log.info("\n" + "=" * 70)
    log.info("POSITION OVERLAP ANALYSIS")
    log.info("=" * 70)
    
    results = {}
    
    traded = trades_df[trades_df['traded']].copy().sort_values('date')
    all_dates = sorted(trades_df['date'].unique())
    
    if len(traded) == 0:
        return results
    
    # Each trade entered on date D is held until D+3 (3 trading days)
    # Track open positions per day
    open_positions = {}  # date -> list of (entry_date, direction)
    
    for _, row in traded.iterrows():
        entry_date = row['date']
        entry_idx = all_dates.index(entry_date) if entry_date in all_dates else -1
        if entry_idx < 0:
            continue
        
        for offset in range(HOLD_DAYS):
            hold_idx = entry_idx + offset
            if hold_idx < len(all_dates):
                d = all_dates[hold_idx]
                if d not in open_positions:
                    open_positions[d] = []
                open_positions[d].append((entry_date, row['direction']))
    
    # Count overlaps
    n_days_with_positions = len(open_positions)
    overlap_counts = [len(v) for v in open_positions.values()]
    n_overlap_days = sum(1 for c in overlap_counts if c > 1)
    max_overlap = max(overlap_counts) if overlap_counts else 0
    avg_overlap = np.mean(overlap_counts) if overlap_counts else 0
    
    # Direction conflicts (long + short simultaneously)
    n_conflict_days = 0
    for d, positions in open_positions.items():
        dirs = set(p[1] for p in positions)
        if 1 in dirs and -1 in dirs:
            n_conflict_days += 1
    
    log.info(f"Total OOT days: {len(all_dates)}")
    log.info(f"Days with open positions: {n_days_with_positions}")
    log.info(f"Days with overlapping positions (>1): {n_overlap_days} ({n_overlap_days/max(n_days_with_positions,1)*100:.1f}%)")
    log.info(f"Max simultaneous positions: {max_overlap}")
    log.info(f"Average positions per day: {avg_overlap:.2f}")
    log.info(f"Days with directional conflict: {n_conflict_days}")
    
    results['n_days_with_positions'] = n_days_with_positions
    results['n_overlap_days'] = n_overlap_days
    results['overlap_pct'] = float(n_overlap_days / max(n_days_with_positions, 1) * 100)
    results['max_simultaneous'] = max_overlap
    results['avg_positions_per_day'] = float(avg_overlap)
    results['n_conflict_days'] = n_conflict_days
    
    return results


def bootstrap_sharpe(daily_pnl, n_boot=10000):
    """Bootstrap confidence intervals on Sharpe ratio."""
    log.info("\n" + "=" * 70)
    log.info("BOOTSTRAP CONFIDENCE INTERVALS")
    log.info("=" * 70)
    
    results = {}
    n = len(daily_pnl)
    pnl_arr = daily_pnl.values
    
    np.random.seed(42)
    boot_sharpes = []
    
    for _ in range(n_boot):
        sample = np.random.choice(pnl_arr, size=n, replace=True)
        if sample.std() > 0:
            s = sample.mean() / sample.std() * np.sqrt(252)
        else:
            s = 0
        boot_sharpes.append(s)
    
    boot_sharpes = np.array(boot_sharpes)
    
    ci_95 = np.percentile(boot_sharpes, [2.5, 97.5])
    ci_90 = np.percentile(boot_sharpes, [5, 95])
    median_sharpe = np.median(boot_sharpes)
    prob_positive = (boot_sharpes > 0).mean()
    prob_above_1 = (boot_sharpes > 1).mean()
    prob_above_2 = (boot_sharpes > 2).mean()
    
    log.info(f"Bootstrap Sharpe (N={n_boot}):")
    log.info(f"  Median: {median_sharpe:.3f}")
    log.info(f"  95% CI: [{ci_95[0]:.3f}, {ci_95[1]:.3f}]")
    log.info(f"  90% CI: [{ci_90[0]:.3f}, {ci_90[1]:.3f}]")
    log.info(f"  P(Sharpe > 0): {prob_positive:.1%}")
    log.info(f"  P(Sharpe > 1): {prob_above_1:.1%}")
    log.info(f"  P(Sharpe > 2): {prob_above_2:.1%}")
    
    # Block bootstrap (preserve autocorrelation structure)
    block_size = HOLD_DAYS
    n_blocks = n // block_size
    boot_block_sharpes = []
    
    for _ in range(n_boot):
        blocks = np.random.randint(0, n - block_size + 1, size=n_blocks)
        sample = np.concatenate([pnl_arr[b:b+block_size] for b in blocks])
        if sample.std() > 0:
            s = sample.mean() / sample.std() * np.sqrt(252)
        else:
            s = 0
        boot_block_sharpes.append(s)
    
    boot_block_sharpes = np.array(boot_block_sharpes)
    block_ci_95 = np.percentile(boot_block_sharpes, [2.5, 97.5])
    
    log.info(f"\nBlock bootstrap (block_size={block_size}):")
    log.info(f"  Median: {np.median(boot_block_sharpes):.3f}")
    log.info(f"  95% CI: [{block_ci_95[0]:.3f}, {block_ci_95[1]:.3f}]")
    
    results['boot_median_sharpe'] = float(median_sharpe)
    results['boot_ci95_lo'] = float(ci_95[0])
    results['boot_ci95_hi'] = float(ci_95[1])
    results['boot_prob_positive'] = float(prob_positive)
    results['boot_prob_above_1'] = float(prob_above_1)
    results['boot_prob_above_2'] = float(prob_above_2)
    results['block_boot_median'] = float(np.median(boot_block_sharpes))
    results['block_boot_ci95_lo'] = float(block_ci_95[0])
    results['block_boot_ci95_hi'] = float(block_ci_95[1])
    
    return results


def oot_stability(trades_df):
    """First half vs second half of OOT period."""
    log.info("\n" + "=" * 70)
    log.info("OOT STABILITY (FIRST HALF vs SECOND HALF)")
    log.info("=" * 70)
    
    results = {}
    
    traded = trades_df[trades_df['traded']].copy().sort_values('date')
    if len(traded) < 10:
        log.info("Too few trades for stability analysis")
        return results
    
    mid = len(traded) // 2
    first_half = traded.iloc[:mid]
    second_half = traded.iloc[mid:]
    
    for label, subset in [('first_half', first_half), ('second_half', second_half)]:
        pnl = subset['pnl_ticks']
        sharpe = pnl.mean() / pnl.std() * np.sqrt(252) if pnl.std() > 0 else 0
        wr = (pnl > 0).mean()
        gp = pnl[pnl > 0].sum()
        gl = abs(pnl[pnl < 0].sum())
        pf = gp / gl if gl > 0 else 0
        
        dates = sorted(subset['date'].unique())
        log.info(f"{label}: {dates[0]} to {dates[-1]}")
        log.info(f"  N={len(subset)}, Sharpe={sharpe:.3f}, WR={wr:.1%}, PF={pf:.2f}, "
                 f"Avg PnL={pnl.mean():.1f}t")
        
        results[f'{label}_n'] = int(len(subset))
        results[f'{label}_sharpe'] = float(sharpe)
        results[f'{label}_wr'] = float(wr)
        results[f'{label}_pf'] = float(pf)
        results[f'{label}_avg_pnl'] = float(pnl.mean())
    
    # Sharpe stability ratio
    s1 = results.get('first_half_sharpe', 0)
    s2 = results.get('second_half_sharpe', 0)
    if max(abs(s1), abs(s2)) > 0:
        stability = min(abs(s1), abs(s2)) / max(abs(s1), abs(s2))
    else:
        stability = 0
    results['sharpe_stability_ratio'] = float(stability)
    log.info(f"\nSharpe stability ratio (min/max): {stability:.3f} (1.0 = perfectly stable)")
    
    # Monthly breakdown
    log.info("\nMonthly breakdown:")
    traded_copy = traded.copy()
    traded_copy['month'] = pd.to_datetime(traded_copy['date']).dt.to_period('M')
    for month, grp in traded_copy.groupby('month'):
        pnl = grp['pnl_ticks']
        sharpe = pnl.mean() / pnl.std() * np.sqrt(252) if pnl.std() > 0 and len(pnl) > 1 else 0
        log.info(f"  {month}: N={len(grp)}, PnL={pnl.sum():.0f}t, Sharpe={sharpe:.2f}, WR={(pnl>0).mean():.0%}")
    
    return results


def get_feature_cols(df):
    """Get feature columns."""
    exclude = {'date', 'open', 'close', 'high', 'low', 'session_vwap', 'trade_count'}
    exclude.update({c for c in df.columns if c.startswith('fwd_')})
    return [c for c in df.columns if c not in exclude and df[c].dtype in [np.float64, np.float32, np.int64, np.int32, float, int]]


def main():
    t0 = time.time()
    log.info("=" * 70)
    log.info("LONG HORIZON FLOW V2 — VALIDATION & AUTOCORRELATION CHECK")
    log.info("=" * 70)
    
    # Load data
    feats_path = OUT_DIR / "enhanced_daily_features.parquet"
    if not feats_path.exists():
        log.error(f"Features not found: {feats_path}")
        sys.exit(1)
    
    df = pd.read_parquet(feats_path)
    log.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    
    feat_cols = get_feature_cols(df)
    log.info(f"Feature columns: {len(feat_cols)}")
    
    # Set MLflow
    mlflow.set_tracking_uri("http://jupiter:5000")
    mlflow.set_experiment("long_horizon_flow_v2_validation")
    
    with mlflow.start_run(run_name=f"validation_{time.strftime('%Y%m%d_%H%M%S')}"):
        mlflow.log_params({
            'threshold': THRESHOLD,
            'hold_days': HOLD_DAYS,
            'wf_train': WF_TRAIN,
            'target': TARGET_COL,
            'n_features': len(feat_cols),
        })
        
        # Step 1: Reconstruct predictions
        trades_df = reconstruct_predictions(df, feat_cols)
        trades_df.to_parquet(VALIDATION_DIR / "trades_reconstructed.parquet", index=False)
        
        traded = trades_df[trades_df['traded']].copy()
        log.info(f"\nTraded: {len(traded)} / {len(trades_df)} days")
        log.info(f"Long: {(traded['direction']==1).sum()}, Short: {(traded['direction']==-1).sum()}")
        
        if len(traded) < 10:
            log.error("Too few trades for analysis")
            mlflow.log_metric("error", 1)
            return
        
        # Daily PnL series (including non-trade days as 0)
        daily_pnl = trades_df.set_index('date')['pnl_ticks'].fillna(0)
        # PnL only on trade days for some analyses
        trade_pnl = traded['pnl_ticks']
        
        # Basic stats
        total_pnl = trade_pnl.sum()
        n_trades = len(traded)
        wr = (trade_pnl > 0).mean()
        raw_sharpe = daily_pnl.mean() / daily_pnl.std() * np.sqrt(252) if daily_pnl.std() > 0 else 0
        
        log.info(f"\nBaseline: N={n_trades}, PnL={total_pnl:.0f}t (${total_pnl*ES_TICK_VALUE:.0f}), "
                 f"WR={wr:.1%}, Raw Sharpe={raw_sharpe:.3f}")
        
        mlflow.log_metrics({
            'n_trades': n_trades,
            'total_pnl_ticks': total_pnl,
            'win_rate': float(wr),
            'raw_sharpe': float(raw_sharpe),
        })
        
        # Step 2: Autocorrelation
        ac_results = analyze_autocorrelation(daily_pnl)
        for k, v in ac_results.items():
            mlflow.log_metric(f'ac_{k}', v)
        
        # Step 3: Newey-West corrected Sharpe
        nw_results = newey_west_sharpe(daily_pnl, max_lag=5)
        for k, v in nw_results.items():
            mlflow.log_metric(f'nw_{k}', v)
        
        # Step 4: Non-overlapping Sharpe
        no_results = non_overlapping_sharpe(trades_df)
        for k, v in no_results.items():
            mlflow.log_metric(f'no_{k}', v)
        
        # Step 5: Position overlap
        overlap_results = position_overlap_analysis(trades_df)
        for k, v in overlap_results.items():
            mlflow.log_metric(f'overlap_{k}', v)
        
        # Step 6: Bootstrap CI
        boot_results = bootstrap_sharpe(daily_pnl)
        for k, v in boot_results.items():
            mlflow.log_metric(f'boot_{k}', v)
        
        # Step 7: OOT stability
        stab_results = oot_stability(trades_df)
        for k, v in stab_results.items():
            mlflow.log_metric(f'stab_{k}', v)
        
        # FINAL VERDICT
        log.info("\n" + "=" * 70)
        log.info("FINAL VERDICT")
        log.info("=" * 70)
        
        nw_sharpe = nw_results.get('nw_corrected_sharpe', raw_sharpe)
        sharpe_reduction = nw_results.get('sharpe_reduction_pct', 0)
        ac_lag1 = ac_results.get('acf_lag1', 0)
        lb_pval = ac_results.get('ljung_box_lag3_pval', 1)
        overlap_pct = overlap_results.get('overlap_pct', 0)
        boot_lo = boot_results.get('block_boot_ci95_lo', 0)
        stability = stab_results.get('sharpe_stability_ratio', 0)
        
        log.info(f"Raw Sharpe: {raw_sharpe:.3f}")
        log.info(f"Newey-West corrected Sharpe: {nw_sharpe:.3f} ({sharpe_reduction:.0f}% reduction)")
        log.info(f"Block bootstrap 95% CI lower: {boot_lo:.3f}")
        log.info(f"ACF lag-1: {ac_lag1:.3f}")
        log.info(f"Ljung-Box lag-3 p-value: {lb_pval:.4f}")
        log.info(f"Position overlap: {overlap_pct:.0f}% of days")
        log.info(f"OOT stability ratio: {stability:.3f}")
        
        # Verdict
        issues = []
        if abs(ac_lag1) > 0.15:
            issues.append(f"High autocorrelation (lag-1={ac_lag1:.3f})")
        if lb_pval < 0.05:
            issues.append(f"Ljung-Box significant (p={lb_pval:.4f})")
        if sharpe_reduction > 30:
            issues.append(f"Large Sharpe correction ({sharpe_reduction:.0f}%)")
        if overlap_pct > 60:
            issues.append(f"Heavy position overlap ({overlap_pct:.0f}%)")
        if boot_lo < 0:
            issues.append(f"Bootstrap CI includes zero")
        if stability < 0.3:
            issues.append(f"Poor OOT stability ({stability:.3f})")
        
        if nw_sharpe > 2.0 and boot_lo > 0 and len(issues) <= 1:
            verdict = "STRONG — Strategy likely has real edge even after corrections"
        elif nw_sharpe > 1.0 and boot_lo > 0:
            verdict = "PROMISING — Edge exists but smaller than naive Sharpe suggests"
        elif nw_sharpe > 0.5:
            verdict = "MARGINAL — Corrected Sharpe borderline, proceed with caution"
        else:
            verdict = "WEAK — Most of the apparent edge may be from autocorrelation inflation"
        
        log.info(f"\nISSUES: {issues if issues else 'None'}")
        log.info(f"VERDICT: {verdict}")
        
        mlflow.log_metric('verdict_nw_sharpe', nw_sharpe)
        mlflow.log_metric('verdict_n_issues', len(issues))
        mlflow.set_tag('verdict', verdict)
        
        # Save full results
        all_results = {
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'elapsed_seconds': time.time() - t0,
            'baseline': {'n_trades': n_trades, 'total_pnl_ticks': total_pnl, 'wr': float(wr), 'raw_sharpe': float(raw_sharpe)},
            'autocorrelation': ac_results,
            'newey_west': nw_results,
            'non_overlapping': no_results,
            'overlap': overlap_results,
            'bootstrap': boot_results,
            'stability': stab_results,
            'verdict': verdict,
            'issues': issues,
        }
        
        results_path = VALIDATION_DIR / "validation_results.json"
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        mlflow.log_artifact(str(results_path))
        
        log.info(f"\nElapsed: {time.time() - t0:.0f}s")
        log.info(f"Results saved to {results_path}")


if __name__ == '__main__':
    main()
