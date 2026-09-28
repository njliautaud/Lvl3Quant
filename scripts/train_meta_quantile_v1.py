#!/usr/bin/env python3
"""
Quantile Meta-Model v1 — TREE BRANCH from Production Meta-Model v1
====================================================================
Instead of MSE loss predicting mean 1s P&L, uses pinball/quantile loss
to predict the 10th, 50th, and 90th percentiles simultaneously.

WHY: The prediction interval width (p90 - p10) is a natural RISK measure.
Filtering for trades with narrow intervals AND high median should select
the safest high-edge trades. Structurally different from MSE (mean only).

ARCHITECTURE: Same deeper MLP 256->128->64->32 trunk, BUT with 3 output
heads (one per quantile) instead of 1.

LOSS: Pinball/quantile loss summed across tau = {0.10, 0.50, 0.90}.

DATA: Identical to production — 29 features, top 3% shorts, 1s P&L target.
"""

import json
import sys
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

if sys.platform == 'win32':
    _BASE = Path(r'C:\Users\claude\Lvl3Quant')
else:
    _BASE = Path('/home/jupiter/Lvl3Quant')

PRED_DIR = _BASE / 'output' / 'cnn_mamba_v2_bulk_oot_v2'
MBO_DIR = _BASE / 'data' / 'processed' / 'mbo_events_smart_v3'
OUT_DIR = _BASE / 'output' / 'meta_quantile_v1'
WEIGHTS_DIR = OUT_DIR / 'weights'
OUT_DIR.mkdir(parents=True, exist_ok=True)
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

COMMISSION_RT = 0.376
STOP_TICKS = 2.0
STOP_SLIPPAGE = 1.0
SHORT_PERCENTILE = 3
TRAIN_WINDOW = 10
BATCH_SIZE = 4096
LR = 1e-3
EPOCHS = 30
WEIGHT_DECAY = 1e-4

QUANTILES = [0.10, 0.50, 0.90]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class QuantileMetaMLP(nn.Module):
    """Deeper MLP trunk 256->128->64->32 with 3 quantile output heads."""
    def __init__(self, input_dim, dropout=0.2):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.GELU(), nn.Dropout(dropout),
        )
        self.head_p10 = nn.Linear(32, 1)  # 10th percentile
        self.head_p50 = nn.Linear(32, 1)  # median
        self.head_p90 = nn.Linear(32, 1)  # 90th percentile

    def forward(self, x):
        h = self.trunk(x)
        return self.head_p10(h).squeeze(-1), self.head_p50(h).squeeze(-1), self.head_p90(h).squeeze(-1)


def quantile_loss(pred, target, tau):
    """Pinball / quantile loss for a single quantile tau."""
    diff = target - pred
    return torch.mean(torch.max(tau * diff, (tau - 1) * diff))


def load_day(pred_file):
    """Load one day with full data — identical to production."""
    pred_data = np.load(pred_file, allow_pickle=True)
    date_str = str(pred_data['date'])
    preds = pred_data['predictions']
    n_windows = int(pred_data['n_windows'])
    window_size = int(pred_data['window_size'])
    stride = int(pred_data['stride'])

    mbo_file = MBO_DIR / f'{date_str}_mbo_events.npz'
    if not mbo_file.exists():
        return None

    mbo = np.load(mbo_file, allow_pickle=True)
    events = mbo['events']
    l1s = mbo['labels_1s']
    l5s = mbo['labels_5s'] if 'labels_5s' in mbo else None

    indices = np.arange(n_windows) * stride + (window_size - 1)
    max_idx = min(len(events), len(l1s)) - 1
    valid = indices <= max_idx
    indices = indices[valid]
    preds = preds[:len(indices)]

    feat = events[indices]
    label_1s = l1s[indices]
    label_5s = l5s[indices] if l5s is not None else None

    valid_mask = ~(np.isnan(label_1s) | np.any(np.isnan(feat), axis=1))
    if label_5s is not None:
        valid_mask &= ~np.isnan(label_5s)

    feat = feat[valid_mask]
    label_1s = label_1s[valid_mask]
    if label_5s is not None:
        label_5s = label_5s[valid_mask]
    preds = preds[valid_mask]

    if len(feat) == 0:
        return None

    # Select top 3% shorts
    pred_1s = preds[:, 0]
    threshold = np.percentile(pred_1s, SHORT_PERCENTILE)
    short_mask = pred_1s <= threshold

    feat_s = feat[short_mask]
    preds_s = preds[short_mask]
    label_1s_s = label_1s[short_mask]
    label_5s_s = label_5s[short_mask] if label_5s is not None else None

    # P&L target (1s horizon)
    short_pnl = -label_1s_s
    stop_hit = label_1s_s >= STOP_TICKS
    pnl_target = np.where(
        stop_hit,
        -(STOP_TICKS + STOP_SLIPPAGE + COMMISSION_RT),
        short_pnl - COMMISSION_RT,
    ).astype(np.float32)

    # Features: 25 MBO + 3 preds + rank = 29
    ranks = np.argsort(np.argsort(preds_s[:, 0])).astype(np.float32) / max(len(preds_s), 1)
    features = np.column_stack([feat_s, preds_s[:, 0], preds_s[:, 1], preds_s[:, 2], ranks]).astype(np.float32)

    return {
        'date': date_str,
        'features': features,
        'pnl_target': pnl_target,
        'label_1s': label_1s_s,
        'label_5s': label_5s_s,
        'n_total': int(valid_mask.sum()),
        'n_shorts': int(short_mask.sum()),
    }


def main():
    print(f"{'='*70}", flush=True)
    print(f"QUANTILE META-MODEL v1 TRAINING", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Architecture: 256->128->64->32 trunk + 3 quantile heads", flush=True)
    print(f"Quantiles: {QUANTILES}", flush=True)
    print(f"Loss: Pinball/quantile loss (sum across tau)", flush=True)
    print(f"Horizon: 1s (confirmed best)", flush=True)
    print(f"Signal: top {SHORT_PERCENTILE}% shorts", flush=True)
    print(f"Walk-forward: {TRAIN_WINDOW}-date train, 1-date OOT", flush=True)
    print(f"{'='*70}\n", flush=True)

    pred_files = sorted([f for f in PRED_DIR.glob('*_predictions.npz') if '_stale_' not in str(f)])
    print(f"Found {len(pred_files)} prediction files", flush=True)

    all_days = []
    for i, pf in enumerate(pred_files):
        if i % 10 == 0:
            print(f"  Loading day {i+1}/{len(pred_files)}...", flush=True)
        day = load_day(pf)
        if day is not None:
            all_days.append(day)
    print(f"Loaded {len(all_days)} days\n", flush=True)

    n_folds = len(all_days) - TRAIN_WINDOW
    print(f"Walk-forward folds: {n_folds}\n", flush=True)

    all_p10_list = []
    all_p50_list = []
    all_p90_list = []
    all_actuals_list = []
    fold_results = []

    for fold_idx in range(n_folds):
        train_days = all_days[fold_idx:fold_idx + TRAIN_WINDOW]
        test_day = all_days[fold_idx + TRAIN_WINDOW]

        train_X = np.concatenate([d['features'] for d in train_days])
        train_y = np.concatenate([d['pnl_target'] for d in train_days])
        test_X = test_day['features']
        test_y = test_day['pnl_target']

        if len(train_X) < 100 or len(test_X) < 10:
            print(f"  Fold {fold_idx}: SKIP (train={len(train_X)}, test={len(test_X)})", flush=True)
            continue

        # Normalize
        mean = train_X.mean(axis=0)
        std = train_X.std(axis=0) + 1e-8
        train_X_n = (train_X - mean) / std
        test_X_n = (test_X - mean) / std

        # Train
        model = QuantileMetaMLP(train_X_n.shape[1]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

        ds = TensorDataset(torch.from_numpy(train_X_n), torch.from_numpy(train_y))
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)

        best_loss = float('inf')
        model.train()
        for epoch in range(EPOCHS):
            epoch_loss = 0
            n_batches = 0
            for X_b, y_b in loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                optimizer.zero_grad()
                p10, p50, p90 = model(X_b)
                loss = (quantile_loss(p10, y_b, 0.10)
                        + quantile_loss(p50, y_b, 0.50)
                        + quantile_loss(p90, y_b, 0.90))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            scheduler.step()
            avg_loss = epoch_loss / n_batches
            if avg_loss < best_loss:
                best_loss = avg_loss

        # Save weights + normalizer for this fold
        fold_date = test_day['date']
        weight_file = WEIGHTS_DIR / f'fold_{fold_idx:03d}_{fold_date}.pt'
        torch.save({
            'model_state': model.state_dict(),
            'norm_mean': mean,
            'norm_std': std,
            'fold_idx': fold_idx,
            'test_date': fold_date,
            'train_dates': [d['date'] for d in train_days],
            'input_dim': train_X_n.shape[1],
            'best_loss': best_loss,
            'quantiles': QUANTILES,
        }, weight_file)

        # Evaluate
        model.eval()
        with torch.no_grad():
            t_X = torch.from_numpy(test_X_n).to(device)
            oot_p10, oot_p50, oot_p90 = model(t_X)
            oot_p10 = oot_p10.cpu().numpy()
            oot_p50 = oot_p50.cpu().numpy()
            oot_p90 = oot_p90.cpu().numpy()

        # Fold-level metrics using median as the primary prediction
        if np.std(oot_p50) > 1e-8:
            corr = float(np.corrcoef(oot_p50, test_y)[0, 1])
        else:
            corr = 0.0

        mean_pnl = float(test_y.mean())
        wr = float((test_y > 0).mean() * 100)
        wins = test_y[test_y > 0].sum() if (test_y > 0).any() else 0
        losses_val = abs(test_y[test_y < 0].sum()) if (test_y < 0).any() else 1
        pf = float(wins / losses_val) if losses_val > 0 else 999

        interval_width = oot_p90 - oot_p10
        mean_iw = float(interval_width.mean())

        fold_data = {
            'fold': fold_idx,
            'date': fold_date,
            'n_train': len(train_X),
            'n_test': len(test_X),
            'corr_p50': corr,
            'mean_pnl': mean_pnl,
            'wr': wr,
            'pf': pf,
            'mean_interval_width': mean_iw,
            'best_loss': best_loss,
        }
        fold_results.append(fold_data)

        all_p10_list.append(oot_p10)
        all_p50_list.append(oot_p50)
        all_p90_list.append(oot_p90)
        all_actuals_list.append(test_y)

        status = "+" if corr > 0 else "-"
        print(f"  Fold {fold_idx:2d} [{fold_date}]: corr_p50={corr:+.4f} pnl={mean_pnl:+.3f} "
              f"wr={wr:.1f}% pf={pf:.2f} iw={mean_iw:.3f} {status}", flush=True)

    if len(all_p50_list) == 0:
        print("ERROR: No folds completed. Check data paths.", flush=True)
        return

    # =========================================================================
    # CONCAT ANALYSIS
    # =========================================================================
    concat_p10 = np.concatenate(all_p10_list)
    concat_p50 = np.concatenate(all_p50_list)
    concat_p90 = np.concatenate(all_p90_list)
    concat_actuals = np.concatenate(all_actuals_list)
    concat_iw = concat_p90 - concat_p10

    concat_corr = float(np.corrcoef(concat_p50, concat_actuals)[0, 1])

    print(f"\n{'='*70}", flush=True)
    print(f"CONCAT RESULTS ({len(concat_p50)} trades across {len(fold_results)} folds)", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"Concat correlation (p50 vs actual): {concat_corr:+.4f}", flush=True)
    print(f"Positive folds: {sum(1 for f in fold_results if f['corr_p50'] > 0)}/{len(fold_results)}", flush=True)
    print(f"Mean fold corr: {np.mean([f['corr_p50'] for f in fold_results]):+.4f}", flush=True)

    # =========================================================================
    # QUANTILE CALIBRATION CHECK
    # =========================================================================
    print(f"\n{'='*70}", flush=True)
    print(f"QUANTILE CALIBRATION CHECK", flush=True)
    print(f"{'='*70}", flush=True)
    pct_below_p10 = float((concat_actuals < concat_p10).mean() * 100)
    pct_below_p50 = float((concat_actuals < concat_p50).mean() * 100)
    pct_below_p90 = float((concat_actuals < concat_p90).mean() * 100)
    print(f"  Actual below p10 prediction: {pct_below_p10:.1f}% (ideal: 10%)", flush=True)
    print(f"  Actual below p50 prediction: {pct_below_p50:.1f}% (ideal: 50%)", flush=True)
    print(f"  Actual below p90 prediction: {pct_below_p90:.1f}% (ideal: 90%)", flush=True)
    calib_ok = (5 < pct_below_p10 < 20) and (35 < pct_below_p50 < 65) and (80 < pct_below_p90 < 97)
    print(f"  Calibration: {'REASONABLE' if calib_ok else 'POOR — quantiles may not be well-calibrated'}", flush=True)

    print(f"\n  Interval width stats:", flush=True)
    print(f"    Mean: {concat_iw.mean():.3f}", flush=True)
    print(f"    Std:  {concat_iw.std():.3f}", flush=True)
    print(f"    p10:  {np.percentile(concat_iw, 10):.3f}", flush=True)
    print(f"    p50:  {np.percentile(concat_iw, 50):.3f}", flush=True)
    print(f"    p90:  {np.percentile(concat_iw, 90):.3f}", flush=True)

    # =========================================================================
    # FILTER ANALYSIS
    # =========================================================================
    baseline_pnl = concat_actuals.mean()
    baseline_gross = baseline_pnl + COMMISSION_RT

    def compute_filter_stats(mask, label):
        """Compute filter stats for a boolean mask."""
        sel = concat_actuals[mask]
        n = len(sel)
        if n == 0:
            return None
        mean_p = float(sel.mean())
        wr_val = float((sel > 0).mean() * 100)
        wins_v = sel[sel > 0].sum() if (sel > 0).any() else 0
        losses_v = abs(sel[sel < 0].sum()) if (sel < 0).any() else 1
        pf_val = float(wins_v / losses_v) if losses_v > 0 else 999
        gross_val = mean_p + COMMISSION_RT
        return {'label': label, 'n': n, 'mean_pnl': mean_p, 'gross': gross_val, 'wr': wr_val, 'pf': pf_val}

    def print_filter_table(title, rows):
        print(f"\n  {title}:", flush=True)
        print(f"  {'Filter':<35} {'N':>7} {'MeanPnL':>8} {'Gross':>7} {'WR%':>6} {'PF':>6}", flush=True)
        print(f"  {'-'*70}", flush=True)
        for r in rows:
            if r is not None:
                print(f"  {r['label']:<35} {r['n']:>7} {r['mean_pnl']:>+7.3f} {r['gross']:>+6.3f} "
                      f"{r['wr']:>5.1f} {r['pf']:>5.2f}", flush=True)

    # ---------- 1. Median filter (direct analog of MSE baseline) ----------
    median_rows = []
    median_rows.append(compute_filter_stats(np.ones(len(concat_actuals), dtype=bool), "ALL (baseline)"))
    for pct in [30, 20, 10, 5]:
        thresh = np.percentile(concat_p50, 100 - pct)
        mask = concat_p50 >= thresh
        median_rows.append(compute_filter_stats(mask, f"p50 top {pct}%"))
    print_filter_table("FILTER 1: MEDIAN (p50) — analog of MSE top-N", median_rows)

    # ---------- 2. Risk-adjusted filter: p50 / interval_width ----------
    # Avoid division by zero — clip interval_width to minimum
    safe_iw = np.clip(concat_iw, 0.01, None)
    reward_risk = concat_p50 / safe_iw
    rr_rows = []
    rr_rows.append(compute_filter_stats(np.ones(len(concat_actuals), dtype=bool), "ALL (baseline)"))
    for pct in [30, 20, 10, 5]:
        thresh = np.percentile(reward_risk, 100 - pct)
        mask = reward_risk >= thresh
        rr_rows.append(compute_filter_stats(mask, f"p50/iw top {pct}%"))
    print_filter_table("FILTER 2: RISK-ADJUSTED (p50 / interval_width)", rr_rows)

    # ---------- 3. Floor filter: p10 > threshold ----------
    floor_rows = []
    floor_rows.append(compute_filter_stats(np.ones(len(concat_actuals), dtype=bool), "ALL (baseline)"))
    for floor_t in [-0.5, 0.0, 0.25, 0.5]:
        mask = concat_p10 > floor_t
        floor_rows.append(compute_filter_stats(mask, f"p10 > {floor_t:+.2f}"))
    print_filter_table("FILTER 3: FLOOR (p10 > threshold)", floor_rows)

    # ---------- 4. Combined: p50 top 30% AND p10 > threshold ----------
    p50_top30_thresh = np.percentile(concat_p50, 70)
    combined_rows = []
    combined_rows.append(compute_filter_stats(np.ones(len(concat_actuals), dtype=bool), "ALL (baseline)"))
    for floor_t in [-0.5, 0.0, 0.25, 0.5]:
        mask = (concat_p50 >= p50_top30_thresh) & (concat_p10 > floor_t)
        combined_rows.append(compute_filter_stats(mask, f"p50 top30% AND p10>{floor_t:+.2f}"))
    print_filter_table("FILTER 4: COMBINED (p50 top 30% + floor)", combined_rows)

    # =========================================================================
    # DAILY VOLUME & RISK ESTIMATES (at best filter)
    # =========================================================================
    print(f"\n{'='*70}", flush=True)
    print(f"DAILY ESTIMATES", flush=True)
    print(f"{'='*70}", flush=True)
    n_days = len(fold_results)

    # Use median top 30% as primary comparison point
    p50_top30_mask = concat_p50 >= p50_top30_thresh
    top30_sel = concat_actuals[p50_top30_mask]
    if len(top30_sel) > 0:
        tpd = len(top30_sel) / n_days
        top30_mean = float(top30_sel.mean())
        top30_gross = top30_mean + COMMISSION_RT
        print(f"  p50 top 30%: {len(top30_sel)} trades, {tpd:.0f}/day, "
              f"gross={top30_gross:+.3f} ticks, ${tpd * top30_gross * 12.5:+,.0f}/day", flush=True)

    # Sharpe estimate
    fold_daily_pnl = []
    for f_idx, f in enumerate(fold_results):
        # Collect OOT actuals for this fold where p50 is in top 30%
        start = sum(len(all_actuals_list[j]) for j in range(f_idx))
        end = start + len(all_actuals_list[f_idx])
        fold_p50 = concat_p50[start:end]
        fold_act = concat_actuals[start:end]
        fold_sel = fold_act[fold_p50 >= p50_top30_thresh]
        fold_daily_pnl.append(float(fold_sel.sum()) if len(fold_sel) > 0 else 0.0)
    fold_daily_pnl = np.array(fold_daily_pnl)
    if fold_daily_pnl.std() > 0:
        daily_sharpe = fold_daily_pnl.mean() / fold_daily_pnl.std() * np.sqrt(252)
        sortino_denom = fold_daily_pnl[fold_daily_pnl < 0].std() if (fold_daily_pnl < 0).any() else 1
        sortino = fold_daily_pnl.mean() / sortino_denom * np.sqrt(252) if sortino_denom > 0 else 999
        green_days = (fold_daily_pnl > 0).sum()
        print(f"  Annualized Sharpe (daily, p50 top 30%): {daily_sharpe:.1f}", flush=True)
        print(f"  Annualized Sortino: {sortino:.1f}", flush=True)
        print(f"  Green days: {green_days}/{len(fold_daily_pnl)} ({green_days/len(fold_daily_pnl)*100:.0f}%)", flush=True)

    # =========================================================================
    # VERDICT
    # =========================================================================
    print(f"\n{'='*70}", flush=True)
    print(f"VERDICT", flush=True)
    print(f"{'='*70}", flush=True)

    baseline_gross_target = 1.02  # production top 30% gross

    # Check all filters for any that beat baseline with meaningful N
    best_filter = None
    best_gross = -999
    all_filter_rows = median_rows + rr_rows + floor_rows + combined_rows
    for r in all_filter_rows:
        if r is None:
            continue
        if r['label'] == "ALL (baseline)":
            continue
        if r['n'] >= 500 and r['gross'] > best_gross:
            best_gross = r['gross']
            best_filter = r

    if best_filter and best_gross > baseline_gross_target:
        print(f"  PASS — Best filter '{best_filter['label']}' achieves gross {best_gross:+.3f} ticks "
              f"(>{baseline_gross_target:+.3f} baseline) with N={best_filter['n']}", flush=True)
    elif best_filter:
        print(f"  FAIL — Best filter '{best_filter['label']}' achieves gross {best_gross:+.3f} ticks "
              f"(baseline target: {baseline_gross_target:+.3f}). "
              f"Quantile approach does not beat MSE production.", flush=True)
    else:
        print(f"  FAIL — No filter achieved meaningful N (>=500).", flush=True)

    print(f"  Calibration: {'REASONABLE' if calib_ok else 'POOR'}", flush=True)
    print(f"  (p10 coverage: {pct_below_p10:.1f}%, p50 coverage: {pct_below_p50:.1f}%, "
          f"p90 coverage: {pct_below_p90:.1f}%)", flush=True)
    print(f"{'='*70}", flush=True)

    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    filter_summary = {}
    for r in all_filter_rows:
        if r is not None:
            filter_summary[r['label']] = {k: v for k, v in r.items() if k != 'label'}

    results = {
        'config': {
            'architecture': '256_128_64_32_3heads',
            'loss': 'pinball_quantile',
            'quantiles': QUANTILES,
            'horizon': '1s',
            'signal_threshold': SHORT_PERCENTILE,
            'train_window': TRAIN_WINDOW,
            'epochs': EPOCHS,
            'lr': LR,
            'weight_decay': WEIGHT_DECAY,
            'batch_size': BATCH_SIZE,
        },
        'concat_corr_p50': concat_corr,
        'mean_fold_corr': float(np.mean([f['corr_p50'] for f in fold_results])),
        'pos_folds': sum(1 for f in fold_results if f['corr_p50'] > 0),
        'total_folds': len(fold_results),
        'total_trades': len(concat_actuals),
        'calibration': {
            'pct_below_p10': pct_below_p10,
            'pct_below_p50': pct_below_p50,
            'pct_below_p90': pct_below_p90,
            'reasonable': calib_ok,
        },
        'interval_width_stats': {
            'mean': float(concat_iw.mean()),
            'std': float(concat_iw.std()),
            'p10': float(np.percentile(concat_iw, 10)),
            'p50': float(np.percentile(concat_iw, 50)),
            'p90': float(np.percentile(concat_iw, 90)),
        },
        'filter_results': filter_summary,
        'per_fold': fold_results,
        'weights_saved': True,
        'weights_dir': str(WEIGHTS_DIR),
        'timestamp': datetime.now().isoformat(),
    }

    np.savez(OUT_DIR / 'concat_predictions.npz',
             p10=concat_p10, p50=concat_p50, p90=concat_p90,
             actuals=concat_actuals, interval_width=concat_iw)

    with open(OUT_DIR / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved weights ({len(fold_results)} folds), results.json, concat_predictions.npz", flush=True)
    print(f"Output: {OUT_DIR}", flush=True)


if __name__ == '__main__':
    main()
