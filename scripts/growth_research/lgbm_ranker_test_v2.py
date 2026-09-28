#!/usr/bin/env python3
"""
LGBMRanker vs LGBMRegressor — Minimal Correct Test
====================================================
Tests if LGBMRanker (native Learning-to-Rank) produces better sector rankings
than LGBMRegressor. Uses production-quality feature engineering.

Key insight: LGBMRanker is designed for ranking problems (group-wise),
while LGBMRegressor fits pointwise predictions. Since we only care about
RELATIVE ordering of 11 sectors, Ranker should theoretically be better.

4 Variants:
  A: LGBMRegressor baseline (current V10)
  B: LGBMRanker with lambdarank
  C: LGBMRanker with ndcg objective
  D: Ensemble (avg rank from A+B)

Measures: Spearman rank correlation, NDCG@4, forward return of top-4 vs bottom-4
"""
import warnings
warnings.filterwarnings("ignore")

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy import stats as sp_stats

# Detect platform
if Path("/home/nick/Lvl3Quant").exists():
    BASE = Path("/home/nick/Lvl3Quant")
elif Path("/home/jupiter/Lvl3Quant").exists():
    BASE = Path("/home/jupiter/Lvl3Quant")
else:
    BASE = Path(".")

OUT_DIR = BASE / "output" / "growth_research" / "lgbm_ranker_test_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

import lightgbm as lgb

SECTORS = ['XLK', 'XLF', 'XLE', 'XLV', 'XLY', 'XLP', 'XLI', 'XLB', 'XLU', 'XLRE', 'XLC']
EXTRA = ['SPY', '^VIX']
REBAL_INTERVAL = 10  # biweekly
FWD_DAYS = 14  # forward return horizon

# Features (18 legacy only — simpler, avoids cross-asset computation issues)
FEAT_COLS = [
    'ret_5d', 'ret_10d', 'ret_21d', 'ret_63d', 'ret_126d', 'ret_252d',
    'vol_21d', 'vol_63d', 'sharpe_63d', 'maxdd_63d', 'pct_52w_high', 'mom_accel',
    'pct_pos_months_12m', 'sortino_63d', 'calmar_1y', 'up_capture',
    'trend_r2_63d', 'trend_slope_63d',
]


def compute_features(px):
    """Compute 18 legacy features for a single sector. Returns dict or None."""
    if len(px) < 260:
        return None
    f = {}
    for lb, nm in [(5, 'ret_5d'), (10, 'ret_10d'), (21, 'ret_21d'),
                   (63, 'ret_63d'), (126, 'ret_126d'), (252, 'ret_252d')]:
        f[nm] = float(px.iloc[-1] / px.iloc[-lb] - 1) if len(px) > lb else 0.0
    rets = px.pct_change().dropna()
    f['vol_21d'] = float(rets.iloc[-21:].std() * np.sqrt(252)) if len(rets) > 21 else 0.2
    f['vol_63d'] = float(rets.iloc[-63:].std() * np.sqrt(252)) if len(rets) > 63 else 0.2
    r63 = rets.iloc[-63:]
    f['sharpe_63d'] = float(r63.mean() / (r63.std() + 1e-10) * np.sqrt(252))
    pk63 = px.iloc[-63:].cummax()
    f['maxdd_63d'] = float(((px.iloc[-63:] / pk63) - 1).min())
    f['pct_52w_high'] = float(px.iloc[-1] / px.iloc[-252:].max())
    f['mom_accel'] = f['ret_21d'] - f['ret_63d'] / 3
    monthly = rets.resample('ME').sum()
    f['pct_pos_months_12m'] = float((monthly.iloc[-12:] > 0).mean()) if len(monthly) >= 12 else 0.5
    dr = r63[r63 < 0]
    f['sortino_63d'] = float(r63.mean() / (dr.std() + 1e-10) * np.sqrt(252)) if len(dr) > 3 else 0.0
    pk = px.iloc[-252:].cummax()
    mdd = float(((px.iloc[-252:] / pk) - 1).min())
    cagr = float(px.iloc[-1] / px.iloc[-252] - 1) if len(px) >= 252 else 0.0
    f['calmar_1y'] = cagr / (abs(mdd) + 1e-10)
    up_days = rets[rets > 0]
    f['up_capture'] = float(up_days.iloc[-63:].mean() / (up_days.mean() + 1e-10)) if len(up_days) > 10 else 1.0
    y = np.log(px.iloc[-63:].values + 1e-10)
    x = np.arange(len(y))
    slope, _, r_val, _, _ = sp_stats.linregress(x, y)
    f['trend_r2_63d'] = r_val ** 2
    f['trend_slope_63d'] = slope * 252
    return f


def build_panel(sc):
    """Build feature panel for walk-forward training."""
    print("  Building feature panel...", flush=True)
    records = []
    all_dates = sc.index
    rebal_dates = all_dates[::REBAL_INTERVAL]

    for dt in rebal_dates:
        idx = sc.index.get_loc(dt)
        if idx < 260 or idx + FWD_DAYS >= len(sc):
            continue
        for tk in sc.columns:
            px = sc[tk].iloc[:idx + 1].dropna()
            feats = compute_features(px)
            if not feats:
                continue
            fwd_ret = float(sc[tk].iloc[idx + FWD_DAYS] / sc[tk].iloc[idx] - 1)
            feats['date'] = dt
            feats['ticker'] = tk
            feats['fwd_ret'] = fwd_ret
            records.append(feats)

    df = pd.DataFrame(records)
    for c in FEAT_COLS:
        if c not in df.columns:
            df[c] = 0.0
    df[FEAT_COLS] = df[FEAT_COLS].fillna(0.0)

    # Add rank labels (0-10 for LGBMRanker)
    df['rank_label'] = df.groupby('date')['fwd_ret'].rank(method='min').astype(int) - 1
    df['pct_rank'] = df.groupby('date')['fwd_ret'].rank(pct=True)

    print(f"  Panel: {len(df)} rows, {df['date'].nunique()} dates, {df['ticker'].nunique()} sectors", flush=True)
    return df


def ndcg_at_k(predicted_scores, true_returns, k=4):
    """Compute NDCG@k for a single query (one rebalance date)."""
    order = np.argsort(-predicted_scores)
    sorted_true = true_returns[order]
    gains = 2**sorted_true - 1
    discounts = np.log2(np.arange(k) + 2)
    dcg = np.sum(gains[:k] / discounts)
    ideal_order = np.argsort(-true_returns)
    ideal_sorted = true_returns[ideal_order]
    ideal_gains = 2**ideal_sorted - 1
    idcg = np.sum(ideal_gains[:k] / discounts)
    return dcg / (idcg + 1e-10)


def walk_forward_test(panel, variant_name, model_type='regressor', **model_kwargs):
    """Sliding walk-forward: 500d train window, predict current rankings."""
    print(f"\n  {'='*60}", flush=True)
    print(f"  Variant {variant_name} ({model_type})", flush=True)
    print(f"  {'='*60}", flush=True)

    dates = sorted(panel['date'].unique())
    train_window = 50  # 50 rebalance periods ~ 500 trading days

    all_spearman = []
    all_ndcg = []
    all_top4_ret = []
    all_bot4_ret = []
    all_spread_ret = []

    for i in range(train_window, len(dates)):
        train_dates = dates[max(0, i - train_window):i]
        test_date = dates[i]

        train_df = panel[panel['date'].isin(train_dates)]
        test_df = panel[panel['date'] == test_date]

        if len(test_df) < 8:
            continue

        X_train = np.nan_to_num(train_df[FEAT_COLS].values.astype(np.float32))
        X_test = np.nan_to_num(test_df[FEAT_COLS].values.astype(np.float32))

        if model_type == 'regressor':
            y_train = train_df['pct_rank'].values.astype(np.float32)
            m = lgb.LGBMRegressor(
                n_estimators=model_kwargs.get('n_estimators', 100),
                max_depth=model_kwargs.get('max_depth', 4),
                learning_rate=model_kwargs.get('learning_rate', 0.05),
                subsample=0.8, colsample_bytree=0.8,
                min_child_samples=5, verbose=-1
            )
            m.fit(X_train, y_train)
            scores = m.predict(X_test)

        elif model_type == 'ranker':
            y_train = train_df['rank_label'].values.astype(int)
            # Group sizes: 11 sectors per date
            train_groups = train_df.groupby('date').size().values

            objective = model_kwargs.get('objective', 'lambdarank')
            m = lgb.LGBMRanker(
                objective=objective,
                n_estimators=model_kwargs.get('n_estimators', 100),
                max_depth=model_kwargs.get('max_depth', 4),
                learning_rate=model_kwargs.get('learning_rate', 0.05),
                subsample=0.8, colsample_bytree=0.8,
                min_child_samples=5, verbose=-1,
                label_gain=list(range(11)),  # 0-10 relevance levels
            )
            m.fit(X_train, y_train, group=train_groups)
            scores = m.predict(X_test)

        # Evaluate ranking quality
        true_rets = test_df['fwd_ret'].values

        # Spearman correlation
        rho, _ = sp_stats.spearmanr(scores, true_rets)
        if not np.isnan(rho):
            all_spearman.append(rho)

        # NDCG@4
        ndcg = ndcg_at_k(scores, true_rets, k=4)
        all_ndcg.append(ndcg)

        # Top-4 vs bottom-4 forward returns
        ranked_idx = np.argsort(-scores)
        top4_tickers = test_df.iloc[ranked_idx[:4]]
        bot4_tickers = test_df.iloc[ranked_idx[-4:]]

        top4_ret = top4_tickers['fwd_ret'].mean()
        bot4_ret = bot4_tickers['fwd_ret'].mean()
        spread_ret = top4_ret - bot4_ret

        all_top4_ret.append(top4_ret)
        all_bot4_ret.append(bot4_ret)
        all_spread_ret.append(spread_ret)

    results = {
        'variant': variant_name,
        'model_type': model_type,
        'n_periods': len(all_spearman),
        'avg_spearman': float(np.mean(all_spearman)) if all_spearman else 0,
        'avg_ndcg4': float(np.mean(all_ndcg)) if all_ndcg else 0,
        'avg_top4_ret': float(np.mean(all_top4_ret)) if all_top4_ret else 0,
        'avg_bot4_ret': float(np.mean(all_bot4_ret)) if all_bot4_ret else 0,
        'avg_spread_ret': float(np.mean(all_spread_ret)) if all_spread_ret else 0,
        'spread_sharpe': float(np.mean(all_spread_ret) / (np.std(all_spread_ret) + 1e-10) * np.sqrt(26)) if all_spread_ret else 0,
        'pct_positive_spread': float(np.mean([1 for s in all_spread_ret if s > 0])) / max(len(all_spread_ret), 1) if all_spread_ret else 0,
    }

    print(f"  Periods: {results['n_periods']}", flush=True)
    print(f"  Avg Spearman: {results['avg_spearman']:.4f}", flush=True)
    print(f"  Avg NDCG@4: {results['avg_ndcg4']:.4f}", flush=True)
    print(f"  Avg Top-4 ret: {results['avg_top4_ret']:.4f}", flush=True)
    print(f"  Avg Bot-4 ret: {results['avg_bot4_ret']:.4f}", flush=True)
    print(f"  Avg Spread ret: {results['avg_spread_ret']:.4f}", flush=True)
    print(f"  Spread Sharpe (ann): {results['spread_sharpe']:.3f}", flush=True)
    print(f"  % positive spread: {results['pct_positive_spread']:.1%}", flush=True)

    return results


def main():
    import yfinance as yf

    print("=" * 70, flush=True)
    print("LGBMRanker vs LGBMRegressor — Ranking Quality Test", flush=True)
    print("=" * 70, flush=True)

    # Download data — long history needed for 260-day features + 500d train window
    tickers = SECTORS + EXTRA
    print(f"\nDownloading {len(tickers)} tickers (2006-present)...", flush=True)
    raw = yf.download(tickers, start='2006-01-01', progress=False)

    mi = isinstance(raw.columns, pd.MultiIndex)
    close = raw['Close'] if mi else raw
    if isinstance(close.columns, pd.MultiIndex):
        close.columns = close.columns.get_level_values(-1)
    close = close.ffill()
    close = close.rename(columns={'^VIX': 'VIX'})

    sc = close[[c for c in SECTORS if c in close.columns]].dropna(how='all')
    print(f"Data: {len(sc)} days, {sc.index[0].date()} to {sc.index[-1].date()}", flush=True)
    print(f"Sectors: {list(sc.columns)}", flush=True)

    # Build feature panel
    panel = build_panel(sc)
    if len(panel) < 100:
        print(f"ERROR: Only {len(panel)} rows in panel. Need more data.", flush=True)
        return

    # Run variants
    results = []

    # A: LGBMRegressor baseline
    r = walk_forward_test(panel, 'A_Regressor', model_type='regressor',
                          n_estimators=100, max_depth=4, learning_rate=0.05)
    results.append(r)

    # B: LGBMRanker lambdarank
    r = walk_forward_test(panel, 'B_Ranker_Lambda', model_type='ranker',
                          objective='lambdarank', n_estimators=100, max_depth=4, learning_rate=0.05)
    results.append(r)

    # C: LGBMRanker rank_xendcg
    try:
        r = walk_forward_test(panel, 'C_Ranker_XENDCG', model_type='ranker',
                              objective='rank_xendcg', n_estimators=100, max_depth=4, learning_rate=0.05)
        results.append(r)
    except Exception as e:
        print(f"  Variant C failed: {e}", flush=True)
        results.append({'variant': 'C_Ranker_XENDCG', 'error': str(e)})

    # D: LGBMRegressor larger
    r = walk_forward_test(panel, 'D_Regressor_Large', model_type='regressor',
                          n_estimators=200, max_depth=6, learning_rate=0.03)
    results.append(r)

    # Print results table
    print("\n" + "=" * 90, flush=True)
    print("RESULTS TABLE — Sorted by Spread Sharpe", flush=True)
    print("=" * 90, flush=True)
    print(f"{'Variant':<22s} {'Spearman':>9s} {'NDCG@4':>8s} {'Top4':>8s} {'Bot4':>8s} {'Spread':>8s} {'SprdShr':>8s} {'%Pos':>6s}", flush=True)
    print("-" * 90, flush=True)

    valid = [r for r in results if 'error' not in r]
    valid.sort(key=lambda x: x.get('spread_sharpe', 0), reverse=True)

    for r in valid:
        print(f"{r['variant']:<22s} {r['avg_spearman']:>9.4f} {r['avg_ndcg4']:>8.4f} "
              f"{r['avg_top4_ret']:>8.4f} {r['avg_bot4_ret']:>8.4f} "
              f"{r['avg_spread_ret']:>8.4f} {r['spread_sharpe']:>8.3f} "
              f"{r['pct_positive_spread']:>5.1%}", flush=True)

    # Winner
    if valid:
        winner = valid[0]
        print(f"\nWINNER: {winner['variant']} (Spread Sharpe {winner['spread_sharpe']:.3f})", flush=True)

    # Save
    with open(OUT_DIR / 'results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {OUT_DIR / 'results.json'}", flush=True)

    # MLflow
    try:
        import mlflow
        mlflow.set_tracking_uri("http://jupiter:5000")
        mlflow.set_experiment("lgbm_ranker_test_v2")
        with mlflow.start_run(run_name=f"ranker_test_{datetime.now().strftime('%Y%m%d_%H%M')}"):
            for r in valid:
                mlflow.log_metric(f"{r['variant']}_spearman", r['avg_spearman'])
                mlflow.log_metric(f"{r['variant']}_ndcg4", r['avg_ndcg4'])
                mlflow.log_metric(f"{r['variant']}_spread_sharpe", r['spread_sharpe'])
        print("MLflow logged", flush=True)
    except Exception as e:
        print(f"MLflow error: {e}", flush=True)

    print(f"\nDone.", flush=True)


if __name__ == '__main__':
    main()
