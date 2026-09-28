#!/usr/bin/env python3
"""Quick analysis of midtrade_thesis_v1 results - precision analysis and winner confirmation."""
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import lightgbm as lgb

df = pd.read_parquet('/home/nick/Lvl3Quant/output/midtrade_thesis_v1/trade_tick_features.parquet')

print('=== Precision Analysis for Loser Detection ===')
print(f'Need precision > 88% to break even with asymmetric payoff')
print(f'Each correct loser cut saves ~3.5 ticks, each wrong winner cut costs ~26 ticks')
print()

for cp in [5, 10, 15, 30]:
    feat_cols = [c for c in df.columns if c.startswith(f'cp{cp}s_')]
    dates = sorted(df['date'].unique())
    preds = np.full(len(df), np.nan)

    for i, test_date in enumerate(dates):
        train_dates = dates[max(0, i-30):i]
        if len(train_dates) < 10:
            continue
        train_mask = df['date'].isin(train_dates)
        test_mask = df['date'] == test_date
        X_tr = np.nan_to_num(df.loc[train_mask, feat_cols].values)
        y_tr = df.loc[train_mask, 'winner'].values
        X_te = np.nan_to_num(df.loc[test_mask, feat_cols].values)

        n_pos = y_tr.sum()
        scale_pos = (len(y_tr) - n_pos) / max(n_pos, 1)

        model = lgb.LGBMClassifier(
            objective='binary', verbosity=-1, n_estimators=200,
            max_depth=4, learning_rate=0.05, num_leaves=16,
            min_child_samples=5, scale_pos_weight=scale_pos, random_state=42)
        model.fit(X_tr, y_tr)
        preds[df.index[test_mask]] = model.predict_proba(X_te)[:, 1]

    valid = ~np.isnan(preds)
    p = preds[valid]
    y = df.loc[valid, 'winner'].values
    d_dir = df.loc[valid, 'direction'].values
    et = df.loc[valid, 'exit_ticks'].values

    print(f'=== T+{cp}s (N={valid.sum()}) ===')

    for thresh in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
        cut = p < thresh
        if cut.sum() == 0:
            continue
        cut_losers = ((cut) & (y == 0)).sum()
        cut_winners = ((cut) & (y == 1)).sum()
        precision = cut_losers / max(cut.sum(), 1)

        long_cuts_correct = ((cut) & (y == 0) & (d_dir == 1)).sum()
        short_cuts_correct = ((cut) & (y == 0) & (d_dir == -1)).sum()
        savings = long_cuts_correct * 4.0 + short_cuts_correct * 3.0
        cost = cut_winners * 26.0
        net = savings - cost

        print(f'  thresh={thresh:.2f}: cuts={cut.sum():3d}, '
              f'prec={precision:.3f}, save={savings:.0f}t, cost={cost:.0f}t, net={net:+.0f}t')

print()
print('=== Winner Confirmation (keep only high P(win) trades) ===')
for cp in [10, 30]:
    feat_cols = [c for c in df.columns if c.startswith(f'cp{cp}s_')]
    dates = sorted(df['date'].unique())
    preds = np.full(len(df), np.nan)

    for i, test_date in enumerate(dates):
        train_dates = dates[max(0, i-30):i]
        if len(train_dates) < 10:
            continue
        train_mask = df['date'].isin(train_dates)
        test_mask = df['date'] == test_date
        X_tr = np.nan_to_num(df.loc[train_mask, feat_cols].values)
        y_tr = df.loc[train_mask, 'winner'].values
        X_te = np.nan_to_num(df.loc[test_mask, feat_cols].values)

        n_pos = y_tr.sum()
        scale_pos = (len(y_tr) - n_pos) / max(n_pos, 1)

        model = lgb.LGBMClassifier(
            objective='binary', verbosity=-1, n_estimators=200,
            max_depth=4, learning_rate=0.05, num_leaves=16,
            min_child_samples=5, scale_pos_weight=scale_pos, random_state=42)
        model.fit(X_tr, y_tr)
        preds[df.index[test_mask]] = model.predict_proba(X_te)[:, 1]

    valid = ~np.isnan(preds)
    p = preds[valid]
    y = df.loc[valid, 'winner'].values
    et = df.loc[valid, 'exit_ticks'].values
    d_valid = df[valid].copy()
    d_valid['pred'] = p

    baseline_pnl = et.sum()
    print(f'T+{cp}s baseline: {baseline_pnl:.1f}t, WR={y.mean():.3f}, N={valid.sum()}')
    for keep_thresh in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65]:
        keep = p >= keep_thresh
        if keep.sum() < 5:
            continue
        kept_pnl = et[keep].sum()
        kept_wr = y[keep].mean()
        kept_n = keep.sum()

        kept_df = d_valid[keep]
        day_pnl = kept_df.groupby('date')['exit_ticks'].sum()
        sharpe = day_pnl.mean() / day_pnl.std() * np.sqrt(252) if day_pnl.std() > 0 else np.nan
        pnl_per_trade = kept_pnl / kept_n

        print(f'  keep>{keep_thresh:.2f}: N={kept_n:3d}, PnL={kept_pnl:7.1f}t, '
              f'WR={kept_wr:.3f}, $/trade={pnl_per_trade:.1f}t, Sharpe={sharpe:.2f}')
