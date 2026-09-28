"""
Regime Prediction Model — LGBM Walk-Forward
Predicts next-day SPY direction (green/red) for position sizing.

Walk-forward: 120d train, 5d purge, 1d OOS stride (SLIDING, not expanding — HC #0)
Logs to local MLflow (file-based).
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, f1_score
from datetime import datetime
import joblib

warnings.filterwarnings('ignore')

TRAIN_WINDOW = 120
PURGE_GAP = 5
PRICES_PATH = '/home/nick/Lvl3Quant/wheel_strategy_v1/data/cache/prices.parquet'
OUTPUT_DIR = '/home/nick/Lvl3Quant/wheel_strategy_v1/output/regime_predictor'
MLFLOW_LOCAL_DIR = '/home/nick/Lvl3Quant/wheel_strategy_v1/mlruns'

PARAM_GRID = [
    {'n_estimators': 200, 'max_depth': 4, 'learning_rate': 0.05, 'num_leaves': 15,
     'min_child_samples': 20, 'subsample': 0.8, 'colsample_bytree': 0.7, 'reg_alpha': 1.0, 'reg_lambda': 1.0},
    {'n_estimators': 300, 'max_depth': 5, 'learning_rate': 0.03, 'num_leaves': 20,
     'min_child_samples': 30, 'subsample': 0.7, 'colsample_bytree': 0.6, 'reg_alpha': 2.0, 'reg_lambda': 2.0},
    {'n_estimators': 150, 'max_depth': 3, 'learning_rate': 0.08, 'num_leaves': 10,
     'min_child_samples': 40, 'subsample': 0.9, 'colsample_bytree': 0.8, 'reg_alpha': 0.5, 'reg_lambda': 0.5},
    {'n_estimators': 400, 'max_depth': 6, 'learning_rate': 0.02, 'num_leaves': 25,
     'min_child_samples': 15, 'subsample': 0.75, 'colsample_bytree': 0.65, 'reg_alpha': 3.0, 'reg_lambda': 3.0},
    {'n_estimators': 250, 'max_depth': 4, 'learning_rate': 0.04, 'num_leaves': 12,
     'min_child_samples': 50, 'subsample': 0.85, 'colsample_bytree': 0.75, 'reg_alpha': 1.5, 'reg_lambda': 1.5},
]


def setup_mlflow():
    try:
        import mlflow
        mlflow.set_tracking_uri(f'file://{MLFLOW_LOCAL_DIR}')
        mlflow.set_experiment('regime_predictor_lgbm')
        print(f"MLflow: local file backend at {MLFLOW_LOCAL_DIR}")
        return mlflow
    except Exception as e:
        print(f"MLflow setup failed: {e}")
        return None


def download_spy_vix():
    import yfinance as yf
    print("Downloading SPY+VIX...")
    spy = yf.download('SPY', start='2014-01-01', end='2026-07-09', progress=False)
    vix = yf.download('^VIX', start='2014-01-01', end='2026-07-09', progress=False)
    for d in [spy, vix]:
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
    spy = spy[['Open','High','Low','Close','Volume']].copy()
    spy.columns = ['spy_open','spy_high','spy_low','spy_close','spy_volume']
    vix = vix[['Close']].copy()
    vix.columns = ['vix_close']
    for d in [spy, vix]:
        d.index = pd.to_datetime(d.index)
        if d.index.tz is not None:
            d.index = d.index.tz_localize(None)
    merged = spy.join(vix, how='left')
    merged['vix_close'] = merged['vix_close'].ffill()
    print(f"  SPY+VIX: {len(merged)} rows, {merged.index.min().date()} to {merged.index.max().date()}")
    return merged


def compute_breadth(prices_path):
    print("Computing breadth...")
    df = pd.read_parquet(prices_path)
    df['date'] = pd.to_datetime(df['date'])
    
    ret_pivot = df.pivot_table(index='date', columns='ticker', values='ret')
    close_pivot = df.pivot_table(index='date', columns='ticker', values='close')
    
    breadth = pd.DataFrame(index=ret_pivot.index)
    breadth['adv_ratio'] = (ret_pivot > 0).sum(axis=1) / ret_pivot.count(axis=1)
    breadth['adv_decline_diff'] = (ret_pivot > 0).sum(axis=1) - (ret_pivot < 0).sum(axis=1)
    breadth['mkt_ret_ew'] = ret_pivot.mean(axis=1)
    breadth['dispersion'] = ret_pivot.std(axis=1)
    
    ma20 = close_pivot.rolling(20).mean()
    breadth['pct_above_ma20'] = (close_pivot > ma20).sum(axis=1) / close_pivot.count(axis=1)
    ma50 = close_pivot.rolling(50).mean()
    breadth['pct_above_ma50'] = (close_pivot > ma50).sum(axis=1) / close_pivot.count(axis=1)
    
    breadth['adv_ratio_5d_ma'] = breadth['adv_ratio'].rolling(5).mean()
    breadth['adv_ratio_20d_ma'] = breadth['adv_ratio'].rolling(20).mean()
    breadth['ad_line'] = breadth['adv_decline_diff'].cumsum()
    breadth['ad_line_5d_chg'] = breadth['ad_line'].diff(5)
    breadth['ad_line_20d_chg'] = breadth['ad_line'].diff(20)
    
    # Dispersion momentum (faster than per-row vol quintile spread)
    breadth['dispersion_5d_ma'] = breadth['dispersion'].rolling(5).mean()
    breadth['dispersion_20d_ma'] = breadth['dispersion'].rolling(20).mean()
    breadth['disp_ratio'] = breadth['dispersion_5d_ma'] / breadth['dispersion_20d_ma'].replace(0, np.nan)
    
    print(f"  Breadth: {len(breadth)} rows, {breadth.shape[1]} features")
    return breadth


def engineer_features(spy_vix, breadth):
    print("Engineering features...")
    df = spy_vix.copy()

    for n in [1, 2, 5, 10, 20, 60]:
        df[f'spy_ret_{n}d'] = df['spy_close'].pct_change(n)

    for p in [5, 14, 20]:
        delta = df['spy_close'].diff()
        gain = delta.clip(lower=0).rolling(p).mean()
        loss = (-delta.clip(upper=0)).rolling(p).mean()
        rs = gain / loss.replace(0, np.nan)
        df[f'spy_rsi_{p}'] = 100 - (100 / (1 + rs))

    ma20 = df['spy_close'].rolling(20).mean()
    std20 = df['spy_close'].rolling(20).std()
    df['spy_bb_pos'] = (df['spy_close'] - ma20) / (2 * std20)

    ret1 = df['spy_ret_1d']
    for w in [5, 20, 60]:
        df[f'spy_rv_{w}'] = ret1.rolling(w).std() * np.sqrt(252)
    df['vol_ratio_5_20'] = df['spy_rv_5'] / df['spy_rv_20'].replace(0, np.nan)
    df['vol_ratio_5_60'] = df['spy_rv_5'] / df['spy_rv_60'].replace(0, np.nan)

    df['spy_range'] = (df['spy_high'] - df['spy_low']) / df['spy_close']
    df['spy_range_5d_ma'] = df['spy_range'].rolling(5).mean()
    df['spy_vol_ratio'] = df['spy_volume'] / df['spy_volume'].rolling(20).mean()

    df['vix_chg_1d'] = df['vix_close'].diff()
    df['vix_chg_5d'] = df['vix_close'].diff(5)
    df['vix_chg_20d'] = df['vix_close'].diff(20)
    df['vix_ma5'] = df['vix_close'].rolling(5).mean()
    df['vix_ma20'] = df['vix_close'].rolling(20).mean()
    df['vix_vs_ma20'] = df['vix_close'] / df['vix_ma20'].replace(0, np.nan)
    df['vix_pct_60d'] = df['vix_close'].rolling(60).rank(pct=True)
    df['vix_rv_spread'] = df['vix_close'] / 100 - df['spy_rv_20']

    df['dow'] = df.index.dayofweek
    df['month'] = df.index.month
    df['is_monday'] = (df['dow'] == 0).astype(int)
    df['is_friday'] = (df['dow'] == 4).astype(int)
    df['is_month_end'] = (df.index.day >= 25).astype(int)

    green = (ret1 > 0).astype(int)
    df['green_ratio_5d'] = green.rolling(5).mean()
    df['green_ratio_20d'] = green.rolling(20).mean()

    streak_vals = np.zeros(len(df))
    s, prev = 0, None
    for i in range(len(df)):
        v = ret1.iloc[i]
        if pd.isna(v):
            prev = None
            continue
        curr = 1 if v > 0 else -1
        s = s + curr if curr == prev else curr
        streak_vals[i] = s
        prev = curr
    df['streak'] = streak_vals

    # Merge breadth
    breadth.index = pd.to_datetime(breadth.index)
    if breadth.index.tz is not None:
        breadth.index = breadth.index.tz_localize(None)
    df = df.join(breadth, how='left')

    df['target'] = (ret1.shift(-1) > 0).astype(int)
    df['next_ret'] = ret1.shift(-1)

    exclude = {'spy_open','spy_high','spy_low','spy_close','spy_volume','vix_close','target','next_ret'}
    feature_cols = [c for c in df.columns if c not in exclude]
    df = df.dropna(subset=feature_cols + ['target'])

    print(f"  Dataset: {len(df)} rows, {len(feature_cols)} features")
    print(f"  Range: {df.index.min().date()} to {df.index.max().date()}")
    print(f"  Green ratio: {df['target'].mean():.3f}")
    return df, feature_cols


def walk_forward(df, feature_cols, params):
    dates = df.index.unique().sort_values()
    n = len(dates)
    start = TRAIN_WINDOW + PURGE_GAP
    total = n - start - 1
    print(f"  WF: {total} folds, train={TRAIN_WINDOW}d, purge={PURGE_GAP}d")

    results = []
    last_model = None

    for i in range(start, n - 1):
        t_start = dates[i - TRAIN_WINDOW - PURGE_GAP]
        t_end = dates[i - PURGE_GAP - 1]
        oos_date = dates[i]

        train_mask = (df.index >= t_start) & (df.index <= t_end)
        oos_mask = df.index == oos_date

        X_tr = df.loc[train_mask, feature_cols]
        y_tr = df.loc[train_mask, 'target']
        X_oos = df.loc[oos_mask, feature_cols]
        y_oos = df.loc[oos_mask, 'target']

        if len(X_tr) < 50 or len(X_oos) == 0:
            continue

        mdl = lgb.LGBMClassifier(objective='binary', verbosity=-1, random_state=42, **params)
        mdl.fit(X_tr, y_tr)

        prob = mdl.predict_proba(X_oos)[:, 1]
        results.append({
            'date': str(oos_date.date()),
            'actual': int(y_oos.values[0]),
            'pred': int(prob[0] >= 0.5),
            'prob_green': float(prob[0]),
            'next_ret': float(df.loc[oos_mask, 'next_ret'].values[0]),
        })
        last_model = mdl

        if len(results) % 500 == 0:
            acc_so_far = np.mean([r['actual'] == r['pred'] for r in results])
            print(f"    fold {len(results)}/{total}, running acc={acc_so_far:.3f}")

    return pd.DataFrame(results), last_model


def evaluate(oos):
    m = {}
    m['accuracy'] = accuracy_score(oos['actual'], oos['pred'])
    m['brier'] = brier_score_loss(oos['actual'], oos['prob_green'])
    m['logloss'] = log_loss(oos['actual'], oos['prob_green'])
    m['f1_macro'] = f1_score(oos['actual'], oos['pred'], average='macro')
    m['n_oos'] = len(oos)

    g, r = oos['actual'] == 1, oos['actual'] == 0
    m['green_acc'] = float((oos.loc[g, 'pred'] == 1).mean()) if g.sum() else 0
    m['red_acc'] = float((oos.loc[r, 'pred'] == 0).mean()) if r.sum() else 0
    m['n_green'], m['n_red'] = int(g.sum()), int(r.sum())

    pr = oos[oos['pred'] == 0]
    m['red_precision'] = float((pr['actual'] == 0).mean()) if len(pr) else 0
    m['red_pred_n'] = len(pr)
    m['avg_ret_pred_red'] = float(pr['next_ret'].mean()) if len(pr) else 0

    pg = oos[oos['pred'] == 1]
    m['green_precision'] = float((pg['actual'] == 1).mean()) if len(pg) else 0
    m['green_pred_n'] = len(pg)
    m['avg_ret_pred_green'] = float(pg['next_ret'].mean()) if len(pg) else 0

    if len(oos) > 20:
        hi = oos[oos['prob_green'] >= oos['prob_green'].quantile(0.8)]
        lo = oos[oos['prob_green'] <= oos['prob_green'].quantile(0.2)]
        m['hiconf_green_acc'] = float((hi['actual'] == 1).mean()) if len(hi) else 0
        m['hiconf_red_acc'] = float((lo['actual'] == 0).mean()) if len(lo) else 0

    sizing = oos['prob_green'].apply(lambda p: 1.0 if p > 0.6 else (0.5 if p > 0.4 else 0.0))
    sized_ret = sizing * oos['next_ret']
    m['sized_sharpe'] = float(sized_ret.mean() / sized_ret.std() * np.sqrt(252)) if sized_ret.std() > 0 else 0
    m['always_in_sharpe'] = float(oos['next_ret'].mean() / oos['next_ret'].std() * np.sqrt(252)) if oos['next_ret'].std() > 0 else 0
    m['sharpe_lift'] = m['sized_sharpe'] - m['always_in_sharpe']
    return m


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    mlf = setup_mlflow()

    print("=" * 60)
    print("REGIME PREDICTOR — LGBM Walk-Forward")
    print("=" * 60)

    spy_vix = download_spy_vix()
    breadth = compute_breadth(PRICES_PATH)
    df, fcols = engineer_features(spy_vix, breadth)

    with open(os.path.join(OUTPUT_DIR, 'feature_cols.json'), 'w') as f:
        json.dump(fcols, f, indent=2)

    print(f"\nSWEEP: {len(PARAM_GRID)} configs")
    print("=" * 60)

    best = {'metrics': None, 'params': None, 'oos': None, 'model': None}

    for idx, params in enumerate(PARAM_GRID):
        print(f"\n--- Config {idx}/{len(PARAM_GRID)-1} ---")
        
        run_ctx = None
        if mlf:
            try:
                run_ctx = mlf.start_run(run_name=f"regime_lgbm_cfg{idx}")
                run_ctx.__enter__()
                mlf.log_params(params)
                mlf.log_param('train_window', TRAIN_WINDOW)
                mlf.log_param('purge_gap', PURGE_GAP)
                mlf.log_param('n_features', len(fcols))
            except Exception as e:
                print(f"  MLflow params error: {e}")
                run_ctx = None

        oos, model = walk_forward(df, fcols, params)
        metrics = evaluate(oos)

        if mlf and run_ctx:
            try:
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        mlf.log_metric(k, v)
                if model:
                    imp = pd.Series(model.feature_importances_, index=fcols).sort_values(ascending=False)
                    imp.to_csv(os.path.join(OUTPUT_DIR, f'feat_imp_cfg{idx}.csv'))
                    mlf.log_artifact(os.path.join(OUTPUT_DIR, f'feat_imp_cfg{idx}.csv'))
            except Exception as e:
                print(f"  MLflow log error: {e}")
            try:
                run_ctx.__exit__(None, None, None)
            except:
                pass

        oos.to_csv(os.path.join(OUTPUT_DIR, f'oos_cfg{idx}.csv'), index=False)

        print(f"  Acc={metrics['accuracy']:.3f} RedPrec={metrics['red_precision']:.3f} "
              f"Brier={metrics['brier']:.4f} SizedSharpe={metrics['sized_sharpe']:.2f} "
              f"HiConfRed={metrics.get('hiconf_red_acc',0):.3f}")

        if best['metrics'] is None or metrics['red_precision'] > best['metrics']['red_precision']:
            best = {'metrics': metrics, 'params': params, 'oos': oos, 'model': model}

    m = best['metrics']
    print(f"\n{'='*60}")
    print("BEST MODEL")
    print(f"{'='*60}")
    print(f"Params: {best['params']}")
    print(f"OOS accuracy:        {m['accuracy']:.3f} ({m['n_oos']} days)")
    print(f"Green-day acc:       {m['green_acc']:.3f} ({m['n_green']} days)")
    print(f"Red-day acc:         {m['red_acc']:.3f} ({m['n_red']} days)")
    print(f"Red precision:       {m['red_precision']:.3f} ({m['red_pred_n']} preds)")
    print(f"Green precision:     {m['green_precision']:.3f} ({m['green_pred_n']} preds)")
    print(f"Brier:               {m['brier']:.4f}")
    print(f"Avg ret pred-red:    {m['avg_ret_pred_red']:.5f}")
    print(f"Avg ret pred-green:  {m['avg_ret_pred_green']:.5f}")
    print(f"Sized Sharpe:        {m['sized_sharpe']:.2f}")
    print(f"Always-in Sharpe:    {m['always_in_sharpe']:.2f}")
    print(f"Sharpe lift:         {m['sharpe_lift']:.2f}")
    for k in ['hiconf_red_acc', 'hiconf_green_acc']:
        if k in m:
            print(f"{k}: {m[k]:.3f}")

    if best['model']:
        joblib.dump(best['model'], os.path.join(OUTPUT_DIR, 'best_regime_model.pkl'))
    if best['oos'] is not None:
        best['oos'].to_csv(os.path.join(OUTPUT_DIR, 'best_oos.csv'), index=False)

    summary = {
        'best_params': best['params'],
        'metrics': {k: v for k, v in m.items() if isinstance(v, (int, float, str))},
        'train_window': TRAIN_WINDOW, 'purge_gap': PURGE_GAP,
        'timestamp': datetime.now().isoformat(),
    }
    with open(os.path.join(OUTPUT_DIR, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nDONE — all results saved")


if __name__ == '__main__':
    main()
